#undef NDEBUG
#include "mme_mem_access_checker.h"
#include <zconf.h>
#include "mme_assert.h"
#include "print_utils.h"

namespace MmeCommon
{
void MmeMemAccessChecker::AULock::userBegin()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while ((m_pendingAdminNr > 0) || m_activeAdmin)
    {
        m_cond.wait(lock);
    }
    m_userNr++;
}

void MmeMemAccessChecker::AULock::userEnd()
{
    m_mutex.lock();
    bool notify = (--m_userNr == 0);
    m_mutex.unlock();
    if (notify)
    {
        m_cond.notify_all();
    }
}

void MmeMemAccessChecker::AULock::adminBegin()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_pendingAdminNr++;
    while ((m_userNr > 0) || m_activeAdmin)
    {
        m_cond.wait(lock);
    }
    m_pendingAdminNr--;
    m_activeAdmin = true;
}

void MmeMemAccessChecker::AULock::adminEnd()
{
    m_mutex.lock();
    m_activeAdmin = false;
    m_mutex.unlock();
    m_cond.notify_all();
}

std::pair<SegmentsSpace<unsigned> *, unsigned> MmeMemAccessChecker::getAccessSegSpace(
    const unsigned cmdIdx,
    uint64_t base,
    uint64_t size,
    const bool isWrite)
{
    // this method is called when the lock is locked in user mode.

    while (true)
    {
        // get pointer to the tracker and accessSegSpace
        auto trackerIt = m_cmd2Trackers.find(cmdIdx);
        if ((trackerIt->second.colors[0].valid ||trackerIt->second.colors[1].valid ) && trackerIt == m_cmd2Trackers.end())
        {
            MME_ASSERT(m_usedCmdIds.find(trackerIt->second.cmdIdx) == m_usedCmdIds.end(),
                       "MemAccessChecker: accessing memory of activation that was not opened yet\n");
            MME_ASSERT(0, "MemAccessChecker: accessing memory of activation that was already closed\n");
        }
        unsigned color = REMOTE;
        if (trackerIt->second.colors[LOCAL].valid)
        {
            const auto *expectedMap = isWrite ?
                                      &trackerIt->second.colors[LOCAL].expectedWriteAccess :
                                      &trackerIt->second.colors[LOCAL].allowedReadAccess;
            color = expectedMap->isSegmentCovered(base, size) ? LOCAL : REMOTE;
        }

        auto *segSpaceMap = isWrite ? &trackerIt->second.colors[color].writeAccess : &trackerIt->second.colors[color].readAccess;
        std::thread::id tid = std::this_thread::get_id();
        auto accessSegSpaceIt = segSpaceMap->find(tid);

        // check if this is the thread first access
        if (accessSegSpaceIt == segSpaceMap->end())
        {
            // switch to admin and add the seg space to the map
            m_auLock.userEnd();
            m_auLock.adminBegin();

            // make sure the tracker is still valid
            if (trackerIt != m_cmd2Trackers.find(cmdIdx))
            {
                MME_ASSERT(!(m_usedCmdIds.find(trackerIt->second.cmdIdx) == m_usedCmdIds.end()),
                           "MemAccessChecker: accessing memory of activation that was not opened yet\n");
                MME_ASSERT(0, "MemAccessChecker: accessing memory of activation that was already closed\n");
            }

            (*segSpaceMap)[tid] = SegmentsSpace<unsigned>();

            // switch back to user lock
            m_auLock.adminEnd();
            m_auLock.userBegin();
        }
        else
        {
            return std::pair<SegmentsSpace<unsigned> *, bool>(&accessSegSpaceIt->second, color);
        }
    }
}

template <int IS_WRITE>
void MmeMemAccessChecker::mergeAndNormThreads(
    CmdMemAccessTracker* tracker,
    const unsigned color,
    SegmentsSpace<unsigned> *segSpace)
{
    // quick, dirty and inefficient implementation
    const unsigned sigBase = tracker->colors[color].signalBase;
    auto *accessMap = IS_WRITE ? &tracker->colors[color].writeAccess : &tracker->colors[color].readAccess;
    for (auto & threadSegSpace : *accessMap)
    {
        threadSegSpace.second.mergeSegments();
        for (auto cit = threadSegSpace.second.cbegin();
             cit != threadSegSpace.second.cend();
             cit++)
        {
            if (cit->second.valid)
            {
                unsigned val = cit->second.seg > sigBase ? cit->second.seg - sigBase : 0;
                SegmentsSpace<unsigned>::const_iterator startIt;
                SegmentsSpace<unsigned>::const_iterator endIt;
                segSpace->getCoveredSegments(cit->first, cit->second.size, startIt, endIt);

                for (auto currIt = startIt; currIt != endIt; currIt++)
                {
                    if (IS_WRITE)
                    {
                        MME_ASSERT(!currIt->second.valid, "MemAccessChecker:  multiple writes to the same address\n");
                        segSpace->addSegment(currIt->first, currIt->second.size, val);
                    }
                    else if (!currIt->second.valid || (currIt->second.seg < val))
                    {
                        segSpace->addSegment(currIt->first, currIt->second.size, val);
                    }
                }
            }
        }
        segSpace->mergeSegments();
    }
}

void MmeMemAccessChecker::compareSegSpacesRead(
    SegmentsSpace<unsigned> *actual,
    const SegmentsSpace<unsigned> *expected,
    const SegmentsSpace<unsigned> *allowed)
{
    // yet again quick and very inefficient implementation (n^2logn)
    for (auto expectedIter = expected->cbegin(); expectedIter != expected->cend(); expectedIter++)
    {
        SegmentsSpace<unsigned>::const_iterator actualStartIter;
        SegmentsSpace<unsigned>::const_iterator actualEndIter;
        actual->getCoveredSegments(expectedIter->first, expectedIter->second.size, actualStartIter, actualEndIter);
        for (/* nothing */; actualStartIter != actualEndIter; actualStartIter++)
        {
            if (expectedIter->second.valid)
            {
                MME_ASSERT(actualStartIter->second.valid,
                           "FAILED READ: The expected iterator is valid, but the actual iterator is not.\n");
                MME_ASSERT(actualStartIter->second.seg <= expectedIter->second.seg,
                           "FAILED READ: The actual iterator segment is bigger than the expected iterator segment.\n");
            }
            else if (actualStartIter->second.valid)
            {
                MME_ASSERT(allowed->isSegmentCovered(actualStartIter->first, actualStartIter->second.size),
                           "FAILED READ: The addresses covered by the actual iterator aren't allowed.\n");
            }
        }
    }
}

void MmeMemAccessChecker::compareSegSpacesWrite(const SegmentsSpace<unsigned>* actualWriteSegment,
                                                const SegmentsSpace<unsigned>* expectedWriteSegment)
{
    auto actualWriteIt = actualWriteSegment->cbegin();
    auto expectedWriteIt = expectedWriteSegment->cbegin();
    for (; (actualWriteIt != actualWriteSegment->cend()) && (expectedWriteIt != expectedWriteSegment->cend());
         actualWriteIt++, expectedWriteIt++)
    {
        if (actualWriteIt->second.valid)
        {
            MME_ASSERT(actualWriteIt->first == expectedWriteIt->first,
                       "FAILED WRITE: didn't started at the same address.\n");
            MME_ASSERT(actualWriteIt->second.size == expectedWriteIt->second.size,
                       "FAILED WRITE: didn't write the same size\n");
        }
        MME_ASSERT(actualWriteIt->second.valid == expectedWriteIt->second.valid,
                   "FAILED WRITE: the actual and expected segments doesn't have the same validity.\n");
        MME_ASSERT(!actualWriteIt->second.valid || actualWriteIt->second.seg == expectedWriteIt->second.seg,
                   "FAILED WRITE: the actual segment is valid, but didn't match the expected segment\n");
    }
}

MmeMemAccessChecker::MmeMemAccessChecker()
{
    memset(m_signalsCtr, 0, sizeof(m_signalsCtr));
}

void MmeMemAccessChecker::testTracker(CmdMemAccessTracker* tracker, const unsigned color)
{
    SegmentsSpace<unsigned> readSegSpace;
    mergeAndNormThreads<false>(tracker, color, &readSegSpace);
    compareSegSpacesRead(&readSegSpace, &tracker->colors[color].expectedReadAccess, &tracker->colors[color].allowedReadAccess);

    SegmentsSpace<unsigned> writeSegSpace;
    mergeAndNormThreads<true>(tracker, color, &writeSegSpace);
    compareSegSpacesWrite(&writeSegSpace, &tracker->colors[color].expectedWriteAccess);
}

void MmeMemAccessChecker::write(const uint64_t base, const uint64_t size, const unsigned cmdIdx, const unsigned signal)
{
    if (size)
    {
        m_auLock.userBegin();
        auto segSpacePtrAndColorIndication = getAccessSegSpace(cmdIdx, base, size, true);
        std::vector<unsigned *> segs;
        segSpacePtrAndColorIndication.first->getSegments(base, size, segs);
        MME_ASSERT(segs.empty(), "MemAccessChecker: writing twice to the same address \n");
        const unsigned signalCtr = segmentCnt[segSpacePtrAndColorIndication.second];
        segSpacePtrAndColorIndication.first->addSegment(base, size, signal);
        m_auLock.userEnd();
    }
}

void MmeMemAccessChecker::read(const uint64_t base, const uint64_t size, const unsigned cmdIdx)
{
    if (size)
    {
        m_auLock.userBegin();
        auto segSpacePtrAndColorIndication = getAccessSegSpace(cmdIdx, base, size, false);
        const unsigned signalCtr = segmentCnt[segSpacePtrAndColorIndication.second];
        if (signalCtr > 0)
        {
            segSpacePtrAndColorIndication = getAccessSegSpace(cmdIdx, base, size, false);
        }
        segSpacePtrAndColorIndication.first->addSegment(base, size, signalCtr);
        m_auLock.userEnd();
    }
}

void MmeMemAccessChecker::signal(const unsigned colorSetIdx)
{
    MME_ASSERT(colorSetIdx < COLORS_NR, "MemAccessChecker: only supported colors of {0,1}\n");

    auto it = m_desc2Trackers[colorSetIdx].begin();
    while ((it != m_desc2Trackers[colorSetIdx].end()) &&
           (it->second->colors[colorSetIdx].signalBase + it->second->colors[colorSetIdx].maxRelSignal <= m_signalsCtr[colorSetIdx]))
    {
        if (it->second->colors[colorSetIdx].valid)
        {
            testTracker(it->second, colorSetIdx);
        }
        it->second->colors[colorSetIdx].valid = false;
        if (!it->second->colors[LOCAL].valid && !it->second->colors[REMOTE].valid)
        {
            m_cmd2Trackers.erase(it->second->cmdIdx);
            m_usedCmdIds.insert(it->second->cmdIdx);
        }
        it++;
    }
    segmentCnt[colorSetIdx] = 0;
    m_desc2Trackers[colorSetIdx].erase(m_desc2Trackers[colorSetIdx].begin(), it);
}

template<int ALLOW_OVERLAP, int IS_INPUT>
void MmeMemAccessChecker::addSubRoisToTracker(const std::list<OverlapRoi>* rois,
                                              CmdMemAccessTracker* tracker,
                                              const unsigned descColor)
{
    for (const auto & roi : *rois)
    {
        MME_ASSERT(!roi.isReduction, "MemAccessChecker: roi reduction is not supported yet.\n");

        const unsigned color = roi.isLocalSignal ? LOCAL : descColor;
        auto expectedSegSpace = IS_INPUT ? &tracker->colors[color].expectedReadAccess : &tracker->colors[color].expectedWriteAccess;

        for (const auto & subRoi : *roi.subRois)
        {
            tracker->colors[color].valid = true;
            tracker->colors[color].maxRelSignal = std::max(tracker->colors[color].maxRelSignal, subRoi.relSoIdx);
            for (const auto & range : subRoi.ranges)
            {
                SegmentsSpace<unsigned>::const_iterator startIt;
                SegmentsSpace<unsigned>::const_iterator endIt;
                expectedSegSpace->getCoveredSegments(roi.offset + range.start(), range.size(), startIt, endIt);

                if (!ALLOW_OVERLAP)
                {
                    auto tmpIt = startIt;
                }

                while (startIt != endIt)
                {
                    if (!startIt->second.valid || (startIt->second.seg < subRoi.relSoIdx))
                    {
                        expectedSegSpace->addSegment(startIt->first, startIt->second.size, subRoi.relSoIdx);
                    }
                    startIt++;
                }
            }
        }
        expectedSegSpace->mergeSegments();
    }
}

void MmeMemAccessChecker::addDesc(const OverlapDescriptor& desc, unsigned descColor /*= 0*/, bool ofNullDesc /*= false*/)
{

    auto cmd2TrackersSize = m_numOfDesc;
    m_numOfDesc++;
    unsigned cmdIdx = cmd2TrackersSize / 2;
    atomicColoredPrint(COLOR_CYAN,
                       "INFO: adding descriptor for engine %d of color %d into commandId %d\n",
                       desc.engineID,
                       descColor,
                       cmdIdx);
    auto & newTracker =  m_cmd2Trackers[cmdIdx];
    if (cmd2TrackersSize % 2 == 0)
    {
        initTracker(desc, cmdIdx, newTracker, ofNullDesc);
    }
    gatherAllSubRoisToOpenReads(desc, newTracker, descColor);
    addSubRoisToTracker<true, true>(&desc.inputRois, &newTracker, descColor);
    addSubRoisToTracker<false, false>(&desc.outputRois, &newTracker, descColor);

    for (unsigned color = 0; color < 2; color++)
    {
        // assign all sub rois to new tracker
        newTracker.colors[color].allowedReadAccess = openReads[color];
        unsigned nextSignal = newTracker.colors[color].signalBase + newTracker.colors[color].signalsNr;

        mergeAssignedSubRois(color, nextSignal);

        if (newTracker.colors[color].valid)
        {
            m_desc2Trackers[color].insert(std::pair<unsigned, CmdMemAccessTracker*>(
                newTracker.colors[color].signalBase + newTracker.colors[color].maxRelSignal, &newTracker));
        }
    }

    if (!newTracker.colors[LOCAL].valid && !newTracker.colors[REMOTE].valid)
    {
        // PATCH
        m_desc2Trackers[LOCAL].insert(std::pair<unsigned, CmdMemAccessTracker*>(
            newTracker.colors[LOCAL].signalBase + newTracker.colors[LOCAL].maxRelSignal, &newTracker));
    }
}

MmeMemAccessChecker::~MmeMemAccessChecker()
{
    MME_ASSERT((m_cmd2Trackers.empty() && m_desc2Trackers[REMOTE].empty() && m_desc2Trackers[LOCAL].empty()),
               "MemAccessChecker: finished the check but didn't checked all signals.\n");
}

// Calc number of signal per EU
unsigned MmeMemAccessChecker::calcSignalsNr(unsigned colorSetIdx) const
{
    MME_ASSERT(colorSetIdx < COLORS_NR, "MemAccessChecker: only supported colors of {0,1}\n");
    unsigned sigNr = 0;
    for (const auto& sig : m_desc2Trackers[colorSetIdx])
    {
        sigNr += sig.second->colors[colorSetIdx].signalsNr;
    }
    return sigNr;
}

void MmeMemAccessChecker::gatherAllSubRoisToOpenReads(
    const OverlapDescriptor& desc,
    std::map<unsigned int, CmdMemAccessTracker>::mapped_type& newTracker,
    unsigned descColor)
{
    for (const auto& inRoi : desc.inputRois)
    {
        unsigned color = inRoi.isLocalSignal ? LOCAL : descColor;
        for (const auto& subRoi : *inRoi.subRois)
        {
            unsigned signal = newTracker.colors[color].signalBase + subRoi.relSoIdx;
            for (const auto& range : subRoi.ranges)
            {
                uint64_t base = inRoi.offset + range.start();
                uint64_t size = range.size();
                openReads[color].addSegment(base, size, signal);
            }
        }
    }
}

void MmeMemAccessChecker::initTracker(const OverlapDescriptor& desc,
                                      unsigned int cmdIdx,
                                      std::map<unsigned int, CmdMemAccessTracker>::mapped_type& newTracker,
                                      bool ofNullDesc /*= false*/)
{
    newTracker.cmdIdx = cmdIdx;
    for (unsigned color = 0; color < 2; color++)
    {
        newTracker.colors[color].valid = false;
        newTracker.ofNullDesc = ofNullDesc;
        newTracker.colors[color].signalsNr = desc.numSignals;
        newTracker.colors[color].maxRelSignal = 0;
        newTracker.colors[color].signalBase = cmdIdx ? m_cmd2Trackers[cmdIdx - 1].colors[color].signalBase + m_cmd2Trackers[cmdIdx - 1].colors[color].signalsNr : 0;
    }
}

void MmeMemAccessChecker::mergeAssignedSubRois(unsigned int color, unsigned int nextSignal)
{
    for (auto it = openReads[color].cbegin(); it != openReads[color].cend(); it++)
    {
        if (it->second.valid && (it->second.seg < nextSignal))
        {
            openReads[color].erase(it->first, it->second.size);
        }
    }
    openReads[color].mergeSegments();
}

}  // namespace MmeCommon
