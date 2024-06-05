#include "submit_command_buffers.hpp"
#include "queues/basic_queue_info.hpp"
#include "profiler_api.hpp"
#include "syn_singleton.hpp"

// Todo change to work without SynSingleton (object will be supplied from Device)
synStatus SubmitCommandBuffers::submitCommandBuffers(const BasicQueueInfo& rBasicQueueInfo,
                                                     uint32_t              physicalQueueOffset,
                                                     CommandSubmission&    commandSubmission,
                                                     uint64_t*             csHandle,
                                                     uint32_t              queueOffset,
                                                     const StagedInfo*     pStagedInfo,
                                                     unsigned int          stageIdx)
{
    PROFILER_COLLECT_TIME()

    HB_ASSERT((csHandle != nullptr), "Got a nullptr csHandle");

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    // TODO - Remove all the calls to the protected Synapse methods
    LOG_DEBUG(SYN_STREAM, "{}: Calling submitCommandBuffers for stream {}", HLLOG_FUNC, rBasicQueueInfo.queueType);

    synStatus status = _SYN_SINGLETON_INTERNAL->submitCommandBuffers(commandSubmission,
                                                                     csHandle,
                                                                     nullptr,
                                                                     physicalQueueOffset,
                                                                     pStagedInfo,
                                                                     globalStatPointsEnum::LaunchSubmit);

    ETL_ADD_LOG_T_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                        logId,
                        SYN_STREAM,
                        "Submitted CS with handle {} stage {} queueOffset {} {}",
                        (uint64_t)*csHandle,
                        stageIdx,
                        queueOffset,
                        rBasicQueueInfo.getDescription());

    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to submit command-buffers. status {}", HLLOG_FUNC, status);
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
        return status;
    }

    PROFILER_MEASURE_TIME("sendCS");

    return synSuccess;
}
