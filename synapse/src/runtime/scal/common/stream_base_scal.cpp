#include "stream_base_scal.hpp"
#include "defs.h"
#include "scal_event.hpp"
#include "synapse_common_types.h"
#include "runtime/scal/common/entities/scal_stream_copy_interface.hpp"

using namespace std::chrono_literals;

QueueBaseScal::QueueBaseScal(const BasicQueueInfo& rBasicQueueInfo) : QueueBase(rBasicQueueInfo) {}

QueueBaseScalCommon::QueueBaseScalCommon(const BasicQueueInfo& rBasicQueueInfo, ScalStreamCopyInterface* scalStream)
: QueueBaseScal(rBasicQueueInfo), m_scalStream(scalStream)
{
}
/*
synStatus QueueBaseScalCommon::eventQuery(const ScalEvent& scalEvent, bool alwaysWaitForInterrupt)
{
    return m_scalStream->longSoQuery(scalEvent.longSo, alwaysWaitForInterrupt);
}
*/
synStatus QueueBaseScalCommon::eventQuery(const EventInterface& rEventInterface)
{
    const ScalEvent& rScalEvent = dynamic_cast<const ScalEvent&>(rEventInterface);
    // no need to copy event (for thread safety), already handled by the caller
    HB_ASSERT(!rScalEvent.isOnHclStream(), "Invalid source stream");

    return m_scalStream->longSoQuery(rScalEvent.longSo, false);
}

synStatus QueueBaseScalCommon::eventSynchronize(const EventInterface& rEventInterface)
{
    const ScalEvent& rScalEvent = dynamic_cast<const ScalEvent&>(rEventInterface);
    // no need to copy event (for thread safety), already handled by the caller
    HB_ASSERT(!rScalEvent.isOnHclStream(), "Invalid source stream");

    synStatus status = m_scalStream->longSoWait(rScalEvent.longSo, SCAL_FOREVER, __FUNCTION__);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: longSoWait timed out on stream {}", HLLOG_FUNC, m_scalStream->getName());
    }
    return status;
}

synStatus QueueBaseScalCommon::waitForLastLongSo(bool isUserReq)
{
    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    synStatus status = m_scalStream->longSoWaitForLast(isUserReq, SCAL_FOREVER, __FUNCTION__);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: longSoWaitForLast timed out on stream {}", HLLOG_FUNC, m_scalStream->getName());
    }
    return status;
}

/**
 * handles the scenario where the last cmd on the stream was "wait" and we need to align the counter.
 * @return status
 */
synStatus QueueBaseScalCommon::addCompletionAfterWait()
{
    synStatus status = synSuccess;
    // if prev cmd was wait, then lock the stream and add a pdma/barrier on the stream.
    // this way it will increment the counter.
    if (m_scalStream->prevCmdIsWait())
    {
        std::lock_guard<std::timed_mutex> lock(m_userOpLock);
        ScalLongSyncObject                longSo;
        status = m_scalStream->addBarrierOrEmptyPdma(longSo);
    }
    return status;
}

TdrRtn QueueBaseScalCommon::tdr(TdrType tdrType)
{
    return m_scalStream->tdr(tdrType);
}

/**
 * get stream status upon any kind of failure
 * @param logForUser if true, log minimized info, user-friendly.
 */
void QueueBaseScalCommon::dfaInfo(DfaReq dfaReq, uint64_t csSeq)
{
    std::set<ScalStreamCopyInterface*> scalStreams = dfaGetQueueScalStreams();

    switch (dfaReq)
    {
        case DfaReq::STREAM_INFO:
        case DfaReq::ALL_WORK:
        case DfaReq::ERR_WORK:
        case DfaReq::SCAL_STREAM:
        {
            std::unique_lock<std::timed_mutex> lock(m_userOpLock, std::defer_lock);

            bool gotLock = lock.try_lock_for(500ms);

            for (auto oneStream : scalStreams)
            {
                bool isCompute = (oneStream->isComputeStream());

                if (!gotLock)
                {
                    LOG_ERR(SYN_DEV_FAIL, "Couldn't get lock on stream {}, continue without", oneStream->getName());
                }

                if (dfaReq == DfaReq::SCAL_STREAM)
                {
                    oneStream->dfaDumpScalStream();
                    continue;
                }

                std::string infoStr;
                uint64_t    devLongSo;
                synStatus   status = oneStream->getStreamInfo(infoStr, devLongSo);
                if (status != synSuccess)
                {
                    LOG_ERR(SYN_DEV_FAIL, "failed to get scal stream info");
                }

                LOG_INFO(SYN_DEV_FAIL, "--- stream {} ---", oneStream->getName());

                if (dfaReq == DfaReq::ALL_WORK)
                {
                    LOG_INFO(SYN_DEV_FAIL,
                             "#stream {} longSo 0x{:x}",
                             oneStream->getName(),
                             devLongSo);  // for tools
                }

                TdrRtn tdrRtn = oneStream->tdr(TdrType::DFA);

                // for user, log only the first recipe of the stream which has tdr (=timeout detected).
                // for internal debug info, dump status of all streams (including all related recipes).
                if (dfaReq == DfaReq::STREAM_INFO)
                {
                    if (tdrRtn.failed)
                    {
                        if (isCompute)
                        {
                            dfaUniqStreamInfo(true, devLongSo, false, m_scalStream->getName());
                        }
                    }
                }
                else
                {
                    bool oldestOnly = (dfaReq == DfaReq::ERR_WORK);

                    oneStream->printCgTdrInfo(tdrRtn.failed);

                    LOG_DEBUG(SYN_DEV_FAIL, "Stream Description: {}", m_basicQueueInfo.getDescription());
                    LOG_DEBUG(SYN_DEV_FAIL, "{}", infoStr);
                    const bool dumpRecipe = oldestOnly;
                    if (isCompute)
                    {
                        dfaUniqStreamInfo(oldestOnly, devLongSo, dumpRecipe, m_scalStream->getName());
                    }
                }
            }
            break;
        }
        case DfaReq::PARSE_CSDC:
        {
            HB_ASSERT_DEBUG_ONLY(false, "Illegal request for scal dev");
            break;
        }
        default:
        {
            break;
        }
    }
}

void QueueBaseScalCommon::dfaUniqStreamInfo(bool               showOldestRecipeOnly,
                                            uint64_t           currentLongSo,
                                            bool               dumpRecipe,
                                            const std::string& callerMsg)
{
    LOG_INFO(SYN_DEV_FAIL,
             "Work information not supported for this stream. CurrentLongSo {:x} dumpRecipe {}",
             currentLongSo,
             dumpRecipe);
}
