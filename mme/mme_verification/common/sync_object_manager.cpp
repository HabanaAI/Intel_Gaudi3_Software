#include "sync_object_manager.h"
#include "mme_assert.h"
#include "src/mme_common/mme_hal_reader.h"

namespace MmeCommon
{
SyncObjectManager::SyncObjectManager(uint64_t smBase, unsigned mmeNr, const MmeHalReader& mmeHal)
: m_smBase(smBase), m_mmeNr(mmeNr), m_mmeHal(mmeHal)
{
    m_maxSoIdx = m_mmeHal.getMaxSoIdx();
    m_minSoIdx = m_mmeHal.getMinSoIdx();

    // maxSoIdx has to be aligned to group size
    MME_ASSERT(m_maxSoIdx % c_so_group_size == 0, "max SO idx should be aligned to group size");

    // reserve the first SO as the groupSo
    m_groupSoIdx = m_minSoIdx;

    // align minSo for the test SOs
    m_minSoIdx++;
    unsigned offset = m_minSoIdx % c_so_group_size;
    if (offset)
    {
        m_minSoIdx += c_so_group_size - offset;
    }

    m_nextSoIdx = m_minSoIdx;
}

// this function assumes it is being called once per test, use twice carefully.
void SyncObjectManager::alignSoIdx(nlohmann::json& test, unsigned alignment)
{
    if (test["firstSoIdx"] != -1)
    {
        m_nextSoIdx = test["firstSoIdx"];
        MME_ASSERT(m_nextSoIdx % alignment == 0, "so idx should be aligned");
        MME_ASSERT(m_nextSoIdx >= m_minSoIdx, "so idx should be larger than min so idx");
        MME_ASSERT(m_nextSoIdx < m_maxSoIdx, "so idx should be smaller than max so idx");
    }

    unsigned offset = m_nextSoIdx % alignment;
    if (offset)
    {
        // advance soIdx to an aligned value
        m_nextSoIdx += alignment - offset;
        // make sure there are at least alignment amount of SoIdx left, otherwise wrap around.
        if (m_nextSoIdx > (m_maxSoIdx - alignment))
        {
            offset = m_minSoIdx % alignment;
            if (offset)
            {
                m_nextSoIdx = m_minSoIdx + alignment - offset;
            }
        }
    }
    test["firstSoIdx"] = m_nextSoIdx;
}

void SyncObjectManager::resetTestGroup(unsigned stream)
{
    m_usedSoIdxsInGroup.clear();
    //  actual first monitor will be used as the group monitor - the rest will be used for pole/zero monitors
    m_monitorIdx = uploadDmaQidFromStream(stream) + m_mmeHal.getFirstMonitor() + 1;
}

unsigned SyncObjectManager::getNewSoIdx()
{
    unsigned newIdx;
    MME_ASSERT(m_usedSoIdxsInGroup.size() < m_maxSoIdx - m_minSoIdx + 1,
              "group size is larger then the allowed number of sync objects");
    do
    {
        newIdx = m_nextSoIdx;
        m_nextSoIdx++;
        if (m_nextSoIdx == m_maxSoIdx)
        {
            // wrap around
            m_nextSoIdx = m_minSoIdx;
        }
    } while (isIdxUsed(newIdx));
    setUsedIdx(newIdx);
    return newIdx;
}

void SyncObjectManager::getSoIdx(std::pair<uint32_t, uint64_t>& so)
{
    so.first = getNewSoIdx();
    so.second = getSoAddress(so.first);
}

SyncInfo SyncObjectManager::createGroupSyncObj(unsigned groupSize)
{
    SyncInfo groupSI;
    groupSI.smBase = m_smBase;
    groupSI.outputSOIdx = m_groupSoIdx / c_so_group_size;
    groupSI.outputSOSel = 1 << (m_groupSoIdx % c_so_group_size);
    // each group will wait until all tests are finished, each test will signal twice (two poles)
    groupSI.outputSOTarget = groupSize * m_mmeNr;

    m_currentGroupInfo = groupSI;
    return groupSI;
}

void SyncObjectManager::addGroupMonitor(CPProgram& prog, unsigned stream)
{
    MME_ASSERT(m_currentGroupInfo.smBase != 0, "group synco info should be set before calling this method");
    const auto masterSoAddr = getSoAddress(m_groupSoIdx);
    unsigned monitorIdx = uploadDmaQidFromStream(stream) + m_mmeHal.getFirstMonitor();
    addMonitor(prog, getCurrentGroupSyncObj(), monitorIdx, masterSoAddr, 0);
}

void SyncObjectManager::addPoleMonitor(CPProgram& prog,
                                       unsigned int stream,
                                       unsigned int PoleIdx,
                                       unsigned int testIdInGroup,
                                       unsigned int poleSoIdx,
                                       unsigned int poleSoIdxSecondary,
                                       unsigned int poleSlaveSoIdx,
                                       unsigned int poleSlaveSoIdxSecondary,
                                       unsigned int groupSize,
                                       int soValue)
{
    unsigned outQId = uploadDmaQidFromStream(stream);

    // Build monitor of the pole (monitoring both poleSoIdx and poleSoIdxSecondary) and monitors to clean them
    SyncInfo poleSI;
    poleSI.smBase = m_smBase;
    poleSI.outputSOIdx = poleSoIdx / c_so_group_size;
    poleSI.outputSOSel = 1 << (poleSoIdx % c_so_group_size);
    if (poleSoIdxSecondary) poleSI.outputSOSel |= 1 << (poleSoIdxSecondary % c_so_group_size);
    if (poleSlaveSoIdx) poleSI.outputSOSel |= 1 << (poleSlaveSoIdx % c_so_group_size);
    if (poleSlaveSoIdxSecondary) poleSI.outputSOSel |= 1 << (poleSlaveSoIdxSecondary % c_so_group_size);
    poleSI.outputSOTarget = soValue;

    // Add the pole monitor on the given poleSO to increment the testSO
    const auto testSoAddr = getSoAddress(m_groupSoIdx);
    static const unsigned incSoWDATA = (1 << 31) | 1;
    addMonitor(prog, poleSI, m_monitorIdx++, testSoAddr, incSoWDATA);

    // add up to 2 more monitors to zero the SOs used after the test is done.
    SyncInfo zeroSI;
    zeroSI.smBase = m_smBase;
    zeroSI.outputSOIdx = m_groupSoIdx / c_so_group_size;
    zeroSI.outputSOSel = 1 << (m_groupSoIdx % c_so_group_size);
    // wait until all test finished before zeroing
    zeroSI.outputSOTarget = groupSize * m_mmeNr;
    uint64_t soAddr = getSoAddress(poleSoIdx);
    addMonitor(prog, zeroSI, m_monitorIdx++, soAddr, 0);

    if (poleSoIdxSecondary)
    {
        soAddr = getSoAddress(poleSoIdxSecondary);
        addMonitor(prog, zeroSI, m_monitorIdx++, soAddr, 0);
    }
    if (poleSlaveSoIdx)
    {
        soAddr = getSoAddress(poleSlaveSoIdx);
        addMonitor(prog, zeroSI, m_monitorIdx++, soAddr, 0);
    }
    if (poleSlaveSoIdxSecondary)
    {
        soAddr = getSoAddress(poleSlaveSoIdxSecondary);
        addMonitor(prog, zeroSI, m_monitorIdx++, soAddr, 0);
    }
}

bool SyncObjectManager::isIdxUsed(unsigned int idx)
{
    return (m_usedSoIdxsInGroup.count(idx) != 0);
}
}  // namespace MmeCommon
