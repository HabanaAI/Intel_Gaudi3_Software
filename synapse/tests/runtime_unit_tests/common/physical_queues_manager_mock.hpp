#pragma once

#include "runtime/qman/common/physical_queues_manager_interface.hpp"
#include "runtime/common/queues/basic_queue_info.hpp"

class PhysicalQueuesManagerMock : public PhysicalQueuesManagerInterface
{
public:
    virtual ~PhysicalQueuesManagerMock() = default;

    virtual TrainingRetCode createStream(BasicQueueInfo& streamHandle, const uint32_t flags) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode destroyStream(const BasicQueueInfo& streamHandle) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode signalEvent(const BasicQueueInfo& streamHandle, QmanEvent& eventHandle) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode
    waitForSignal(const BasicQueueInfo& streamHandle, const QmanEvent& eventHandle, const unsigned int flags) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode performStreamsSynchronization(const BasicQueueInfo& currentStreamHandle,
                                                          const BasicQueueInfo& pPreviousStreamHandle,
                                                          bool                  isUser) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode updateStreamPostExecution(const BasicQueueInfo&     rBasicQueueInfo,
                                                      const InternalWaitHandle& waitHandle,
                                                      const std::string&        desc) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode getPhysicalQueueIds(const BasicQueueInfo& streamHandle,
                                                PhysicalQueuesId&     physicalQueuesId) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual spQueueInfo getStreamInfo(const BasicQueueInfo& streamHandle) override { return streamHandle.pQueueInfo; }

    virtual bool isStreamSynchronizationRequired(const BasicQueueInfo& currentStreamHandle,
                                                 const BasicQueueInfo& pPreviousStreamHandle) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode getLastWaitHandles(InternalWaitHandlesVector& lastWaitHandles) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode getLastWaitHandles(const BasicQueueInfo&      streamHandle,
                                               InternalWaitHandlesVector& lastWaitHandles) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual uint32_t getStreamPhysicalOffset(const BasicQueueInfo& streamHandle) override { return 0; }

    virtual TrainingRetCode reserveSignalObjects(const BasicQueueInfo& streamHandle,
                                                 size_t                numOfTensors,
                                                 reserve_sig_handle*   sigHandle) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual TrainingRetCode unreserveSignalObjects(const BasicQueueInfo& streamHandle,
                                                   reserve_sig_handle*   sigHandle) override
    {
        return TRAINING_RET_CODE_SUCCESS;
    }

    virtual uint64_t getStreamId(const BasicQueueInfo& streamHandle) const override { return 0; }
};
