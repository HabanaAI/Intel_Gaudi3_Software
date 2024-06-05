#include "queue_collective_network.hpp"
#include "log_manager.h"
#include "defs.h"
#include "physical_queues_manager.hpp"
#include "internal/hccl_internal.h"
#include "runtime/qman/common/qman_event.hpp"
#include "habana_global_conf.h"
#include "syn_singleton.hpp"
#include "event_triggered_logger.hpp"

QueueCollectiveNetwork::QueueCollectiveNetwork(const BasicQueueInfo&           rBasicQueueInfo,
                                               uint32_t                        physicalQueueOffset,
                                               synDeviceType                   deviceType,
                                               PhysicalQueuesManagerInterface* pPhysicalStreamsManager)
: QueueBaseQman(rBasicQueueInfo, physicalQueueOffset, deviceType, pPhysicalStreamsManager)
{
    HB_ASSERT(m_basicQueueInfo.queueType == INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK, "QueueComputeQman: illegal type");
}

QueueCollectiveNetwork::~QueueCollectiveNetwork() {}

synStatus QueueCollectiveNetwork::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = 0;
    return synSuccess;
}

synStatus QueueCollectiveNetwork::eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle)
{
    HB_ASSERT_PTR(streamHandle);

    LOG_TRACE(SYN_STREAM, "{}: Calling hcclFlushSubmissions", HLLOG_FUNC);

    /*We call a dedicated API that will block the user eventRecord API until the proactor will finish with the
        event. After the HCL_WaitOnLastEvent will return, RT will continue with the normal record activity.*/
    hcclResult_t hcclStatus = hcclFlushSubmissions(streamHandle);
    if (hcclStatus != hcclSuccess && hcclStatus != hcclInvalidArgument)
    {
        LOG_ERR(SYN_STREAM, "{}: Calling hcclFlushSubmissions failed. ", HLLOG_FUNC);
        return synFailHccl;
    }

    QmanEvent&      rEventHandle    = dynamic_cast<QmanEvent&>(rEventInterface);
    TrainingRetCode trainingRetCode = m_pPhysicalStreamsManager->signalEvent(m_basicQueueInfo, rEventHandle);
    if ((trainingRetCode != TRAINING_RET_CODE_SUCCESS) && (trainingRetCode != TRAINING_RET_CODE_NO_CHANGE))
    {
        LOG_ERR(SYN_STREAM, "{}: Operation failed on stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
        return synFailedToSubmitWorkload;
    }

#ifndef _POWER_PC_
    LOG_TRACE(SYN_STREAM,
              "{}: Found COLLECTIVE_NETWORK / SEND stream. "
              "Calling hcclEventRecord",
              HLLOG_FUNC);

    hcclStatus = hcclEventRecord(&(rEventHandle.handleRequest), streamHandle);
    if (hcclSuccess != hcclStatus && hcclStatus != hcclInvalidArgument)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Calling hcclEventRecord failed. With event {}",
                rEventHandle.handleRequest.event,
                HLLOG_FUNC);
        return synFail;
    }
#endif

    return synSuccess;
}

synStatus QueueCollectiveNetwork::eventWait(const EventInterface& rEventInterface,
                                            const unsigned int    flags,
                                            synStreamHandle       streamHandle)
{
    HB_ASSERT_PTR(streamHandle);

    LOG_TRACE(SYN_STREAM, "{}: Calling hcclFlushSubmissions", HLLOG_FUNC);

    /* We call a dedicated API that will block
        the user streamWait API until the proactor will finish with the preceding jobs. After the HCL_WaitOnLastEvent
       will return, RT will continue with the normal wait activity.*/
    hcclResult_t hcclStatus = hcclFlushSubmissions(streamHandle);
    if (hcclStatus != hcclSuccess && hcclStatus != hcclInvalidArgument)
    {
        LOG_ERR(SYN_STREAM, "{}: Calling hcclFlushSubmissions failed. ", HLLOG_FUNC);
        return synFailHccl;
    }

    return QueueBaseQman::eventWait(rEventInterface, flags, streamHandle);
}

synStatus QueueCollectiveNetwork::synchronize(synStreamHandle streamHandle, bool isUserRequest)
{
    HB_ASSERT_PTR(streamHandle);

    LOG_TRACE(SYN_STREAM, "{}: Calling hcclSynchronizeStream", HLLOG_FUNC);
    hcclResult_t hcclStatus = hcclSynchronizeStream(streamHandle);

    if (hcclStatus != hcclSuccess && hcclStatus != hcclInvalidArgument)
    {
        LOG_ERR(SYN_STREAM, "{}: Calling hcclSynchronizeStream failed. ", HLLOG_FUNC);
        return synFail;
    }

    InternalWaitHandlesVector streamWaitHandles;

    TrainingRetCode retCode = m_pPhysicalStreamsManager->getLastWaitHandles(m_basicQueueInfo, streamWaitHandles);
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM, "{}: Can not get wait-handle of {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
        return synFail;
    }

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    synStatus status =
        _SYN_SINGLETON_INTERNAL->waitAndReleaseStreamHandles(streamWaitHandles, SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT);

    ETL_ADD_LOG_T_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                        logId,
                        SYN_STREAM,
                        "{}: Synchronized {}",
                        __FUNCTION__,
                        m_basicQueueInfo.getDescription());

    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: Calling _waitAndReleaseStreamHandles failed", HLLOG_FUNC);
        return status;
    }

    return status;
}

void QueueCollectiveNetwork::dfaInfo(DfaReq dfaReq, uint64_t csSeq)
{
    switch (dfaReq)
    {
        case DfaReq::STREAM_INFO:
        {
            LOG_TRACE(SYN_DEV_FAIL, "stream {:x} {}", TO64(this), getBasicQueueInfo().getDescription());
            break;
        }
        case DfaReq::PARSE_CSDC:
        case DfaReq::ALL_WORK:
        case DfaReq::ERR_WORK:
        case DfaReq::SCAL_STREAM:
        {
            // Do nothing
            break;
        }
    }
}

synStatus QueueCollectiveNetwork::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                                  std::vector<tensor_info_t>& tensorInfoArray) const
{
    LOG_ERR(SYN_API, "Unsupported stream type for getLastTensorArray: {}", getBasicQueueInfo().queueType);
    return synFail;
}
