#pragma once

#include "synapse_api_types.h"

#include "define_synapse_common.hpp"

#include "queue_creator.hpp"
#include "log_manager.h"

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

#include "physical_queues_manager_interface.hpp"
#include "runtime/common/queues/event_with_mapped_tensor.hpp"

// Eventually should inherit from DeviceCommon
class PhysicalQueuesManager : public PhysicalQueuesManagerInterface
{
public:
    PhysicalQueuesManager(common::StreamCreator* pStreamCreator);

    ~PhysicalQueuesManager() override;

    TrainingRetCode createStream(BasicQueueInfo& rBasicQueueInfo, const uint32_t flags) override;

    TrainingRetCode destroyStream(const BasicQueueInfo& rBasicQueueInfo) override;

    TrainingRetCode signalEvent(const BasicQueueInfo& streamHandle, QmanEvent& eventHandle) override;

    TrainingRetCode
    waitForSignal(const BasicQueueInfo& streamHandle, const QmanEvent& eventHandle, const unsigned int flags) override;

    TrainingRetCode performStreamsSynchronization(const BasicQueueInfo& currentStreamHandle,
                                                  const BasicQueueInfo& pPreviousStreamHandle,
                                                  bool                  isUser) override;

    TrainingRetCode updateStreamPostExecution(const BasicQueueInfo&     rBasicQueueInfo,
                                              const InternalWaitHandle& waitHandle,
                                              const std::string&        desc) override;

    TrainingRetCode getPhysicalQueueIds(const BasicQueueInfo& rBasicQueueInfo,
                                        PhysicalQueuesId&     physicalQueuesId) override;

    spQueueInfo getStreamInfo(const BasicQueueInfo& rBasicQueueInfo) override;

    bool isStreamSynchronizationRequired(const BasicQueueInfo& currentStreamHandle,
                                         const BasicQueueInfo& pPreviousStreamHandle) override;

    TrainingRetCode getLastWaitHandles(InternalWaitHandlesVector& lastWaitHandles) override;

    TrainingRetCode getLastWaitHandles(const BasicQueueInfo&      streamHandle,
                                       InternalWaitHandlesVector& lastWaitHandles) override;

    uint32_t getStreamPhysicalOffset(const BasicQueueInfo& streamHandle) override;

    TrainingRetCode reserveSignalObjects(const BasicQueueInfo& streamHandle,
                                         size_t                numOfTensors,
                                         reserve_sig_handle*   sigHandle) override;

    TrainingRetCode unreserveSignalObjects(const BasicQueueInfo& streamHandle, reserve_sig_handle* sigHandle) override;

    uint64_t getStreamId(const BasicQueueInfo& rBasicQueueInfo) const override;

    static TrainingRetCode debugGetPhysicalStreamOffset(TrainingQueue logicalQueue, uint32_t& physicalQueueOffset)
    {
        return _getPhysicalStreamOffset(logicalQueue, physicalQueueOffset);
    }

    void logStreamsSyncHistory(synapse::LogManager::LogType logType);

private:
    // Single stream-type info. Hence streamtype
    struct SingleStreamTypeInfo
    {
        std::vector<BasicQueueInfo>  streamInfoDB;
        uint32_t                     numOfElements;
        uint32_t                     nextInstanceIndex;
        bool                         valid;  // maybe replace with streamInfoDB.size()?
    };

    TrainingRetCode _allocateAllStreamTypesElements();

    TrainingRetCode _destroyAllStreamTypesElements();

    static TrainingRetCode _getPhysicalStreamOffset(TrainingQueue logicalQueue, uint32_t& physicalQueueOffset);

    TrainingRetCode
    _selectTrainingQueue(const internalStreamType& queueType, TrainingQueue& type, uint32_t elementIndex);

    TrainingRetCode _updateStreamTypeAllocation(internalStreamType queueType, bool isAlloc);

    TrainingRetCode _destroyStream(const BasicQueueInfo& streamHandle);

    TrainingRetCode
    _signalEvent(uint64_t& signalSequence, spQueueInfo pQueueInfo, bool isUser, bool collectTime = false);

    TrainingRetCode _waitForSignal(uint64_t&   waitSequenceId,
                                   uint64_t    signalSeqId,
                                   spQueueInfo pQueueInfo,
                                   bool        isUser,
                                   bool        isCollectiveStream,
                                   uint32_t    reductionStreamQueueId,
                                   uint64_t    sequenceOffset = SEQUENCE_OFFSET_NOT_USED);

    bool _performRegularWaitRequest(uint64_t& waitSequenceId,
                                    uint64_t  signalSequenceId,
                                    uint32_t  waitQueueId,
                                    bool      isUserRequest,
                                    uint32_t  sequenceOffset);

    bool _performCollectiveWaitRequest(uint64_t& waitSequenceId,
                                       uint64_t  signalSequenceId,
                                       uint32_t  waitQueueId,
                                       uint32_t  reductionStreamQueueId,
                                       bool      isUserRequest,
                                       uint32_t  sequenceOffset);

    common::StreamCreator*                                     m_pStreamCreator;
    uint32_t                                                   m_nextStreamIndex;
    std::map<uint32_t, std::shared_ptr<QueueInfo>>             m_streamsInfo;
    std::array<SingleStreamTypeInfo, INTERNAL_STREAM_TYPE_NUM> m_allStreamTypesInfo;
    std::mutex m_streamMutex;
};
