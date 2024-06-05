#include "queue_base_qman.hpp"
#include "queues/event_interface.hpp"
#include "syn_singleton.hpp"
#include "physical_queues_manager.hpp"
#include "log_manager.h"
#include "runtime/qman/common/qman_event.hpp"
#include "synapse_runtime_logging.h"

QueueBaseQman::QueueBaseQman(const BasicQueueInfo&           rBasicQueueInfo,
                             uint32_t                        physicalQueueOffset,
                             synDeviceType                   deviceType,
                             PhysicalQueuesManagerInterface* pPhysicalStreamsManager)
: QueueBase(rBasicQueueInfo),
  m_physicalQueueOffset(physicalQueueOffset),
  m_deviceType(deviceType),
  m_pPhysicalStreamsManager(pPhysicalStreamsManager)
{
}

synStatus
QueueBaseQman::eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle)
{
    const QmanEvent& rQmanEvent = dynamic_cast<const QmanEvent&>(rEventInterface);

    TrainingRetCode trainingRetCode = m_pPhysicalStreamsManager->waitForSignal(m_basicQueueInfo, rQmanEvent, flags);
    if (trainingRetCode == TRAINING_RET_CODE_NO_CHANGE)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: No event-recording required for stream {}",
                  HLLOG_FUNC,
                  m_basicQueueInfo.getDescription());

        return synSuccess;
    }

    if (trainingRetCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to add wait-request on stream {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription());

        return synFailedToSubmitWorkload;
    }
    if (rQmanEvent.isInternalSignalingEvent())
    {
        LOG_TRACE(SYN_STREAM,
                  "Performed wait on external event for tensor {}, status {}, executionOrder {}",
                  rQmanEvent.getTensorName(),
                  trainingRetCode,
                  rQmanEvent.getSequenceOffset());
    }
    return synSuccess;
}

synStatus QueueBaseQman::query()
{
    InternalWaitHandlesVector streamWaitHandles;

    TrainingRetCode retCode = m_pPhysicalStreamsManager->getLastWaitHandles(m_basicQueueInfo, streamWaitHandles);
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM, "{}: Can not get wait-handle of {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
        return synFail;
    }

    return _SYN_SINGLETON_INTERNAL->waitAndReleaseStreamHandles(streamWaitHandles,
                                                                SYNAPSE_WAIT_FOR_QUERY_TIMEOUT,
                                                                true);
}

synStatus QueueBaseQman::performStreamsSynchronization(QueueInterface& rPrecedingStreamInterface, bool isUser)
{
    const BasicQueueInfo& rPrecedingBasicQueueInfo = rPrecedingStreamInterface.getBasicQueueInfo();

    if (m_pPhysicalStreamsManager->isStreamSynchronizationRequired(m_basicQueueInfo, rPrecedingBasicQueueInfo))
    {
        TrainingRetCode trainingRetCode =
            m_pPhysicalStreamsManager->performStreamsSynchronization(m_basicQueueInfo,
                                                                     rPrecedingBasicQueueInfo,
                                                                     isUser);
        if (trainingRetCode != TRAINING_RET_CODE_SUCCESS)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to perform stream-synchronization via training-manager on device",
                    HLLOG_FUNC);

            return synFailedToSubmitWorkload;
        }
    }

    return synSuccess;
}
