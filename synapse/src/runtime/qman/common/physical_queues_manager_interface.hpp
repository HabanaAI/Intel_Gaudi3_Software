#pragma once

#include "runtime/common/common_types.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include <memory>

class QueueInfo;
struct reserve_sig_handle;
typedef std::shared_ptr<QueueInfo> spQueueInfo;
struct BasicQueueInfo;
struct QmanEvent;

class PhysicalQueuesManagerInterface
{
public:
    virtual ~PhysicalQueuesManagerInterface() = default;

    virtual TrainingRetCode createStream(BasicQueueInfo& streamHandle, const uint32_t flags) = 0;

    virtual TrainingRetCode destroyStream(const BasicQueueInfo& streamHandle) = 0;

    virtual TrainingRetCode signalEvent(const BasicQueueInfo& streamHandle, QmanEvent& eventHandle) = 0;

    virtual TrainingRetCode
    waitForSignal(const BasicQueueInfo& streamHandle, const QmanEvent& eventHandle, const unsigned int flags) = 0;

    virtual TrainingRetCode performStreamsSynchronization(const BasicQueueInfo& currentStreamHandle,
                                                          const BasicQueueInfo& pPreviousStreamHandle,
                                                          bool                  isUser) = 0;

    virtual TrainingRetCode updateStreamPostExecution(const BasicQueueInfo&     rBasicQueueInfo,
                                                      const InternalWaitHandle& waitHandle,
                                                      const std::string&        desc) = 0;

    virtual TrainingRetCode getPhysicalQueueIds(const BasicQueueInfo& streamHandle,
                                                PhysicalQueuesId&     physicalQueuesId) = 0;

    virtual spQueueInfo getStreamInfo(const BasicQueueInfo& streamHandle) = 0;

    virtual bool isStreamSynchronizationRequired(const BasicQueueInfo& currentStreamHandle,
                                                 const BasicQueueInfo& pPreviousStreamHandle) = 0;

    virtual TrainingRetCode getLastWaitHandles(InternalWaitHandlesVector& lastWaitHandles) = 0;

    virtual TrainingRetCode getLastWaitHandles(const BasicQueueInfo&      streamHandle,
                                               InternalWaitHandlesVector& lastWaitHandles) = 0;

    virtual uint32_t getStreamPhysicalOffset(const BasicQueueInfo& streamHandle) = 0;

    virtual TrainingRetCode
    reserveSignalObjects(const BasicQueueInfo& streamHandle, size_t numOfTensors, reserve_sig_handle* sigHandle) = 0;

    virtual TrainingRetCode unreserveSignalObjects(const BasicQueueInfo& streamHandle,
                                                   reserve_sig_handle*   sigHandle) = 0;

    virtual uint64_t getStreamId(const BasicQueueInfo& streamHandle) const = 0;
};
