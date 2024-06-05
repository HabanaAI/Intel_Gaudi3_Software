#include "physical_queues_manager.hpp"

#include "defs.h"
#include "event_triggered_logger.hpp"
#include "generate_packet.hpp"
#include "graph_compiler/sync/sync_types.h"
#include "habana_global_conf_runtime.h"
#include "habana_global_conf.h"
#include "queue_info.hpp"
#include "runtime/common/osal/osal.hpp"
#include "runtime/common/queues/basic_queue_info.hpp"
#include "runtime/qman/common/command_submission_builder.hpp"
#include "runtime/qman/common/command_submission.hpp"
#include "runtime/qman/common/qman_event.hpp"
#include "synapse_api.h"
#include "synapse_runtime_logging.h"
#include "types_exception.h"
#include "types.h"

// Used for Collective, which is required only for Gaudi, so for the meantime...
#include "runtime/qman/gaudi/master_qmans_definition.hpp"

#include "drm/habanalabs_accel.h"
#include "hlthunk.h"  // mosher - We should not call hlthunk directly, but via OSAL
#include "syn_event_dispatcher.hpp"

#include <mutex>
#include <sstream>
#include <hlthunk.h>

#define VERIFY_IS_NULL_POINTER(pointer, name)                                                                          \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_WARN(SYN_STREAM, "{}: got null pointer for {} ", HLLOG_FUNC, name);                                        \
        return TRAINING_RET_CODE_FAIL;                                                                                 \
    }

static const std::array<uint32_t, INTERNAL_STREAM_TYPE_NUM> s_streamsAllocationRestriction = {
    4,  // [INTERNAL_STREAM_TYPE_DMA_UP]
    1,  // [INTERNAL_STREAM_TYPE_DMA_UP_PROFILER]
    1,  // [INTERNAL_STREAM_TYPE_DEV_TO_DEV]
    4,  // [INTERNAL_STREAM_TYPE_DMA_DOWN_USER]
    1,  // [INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE]
    3,  // [INTERNAL_STREAM_TYPE_COMPUTE]
    4   // [INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK]
};

PhysicalQueuesManager::PhysicalQueuesManager(common::StreamCreator* pStreamCreator)
: m_pStreamCreator(pStreamCreator), m_nextStreamIndex(0)
{
    TrainingRetCode retCode = _allocateAllStreamTypesElements();
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        throw SynapseException("PhysicalQueuesManager: Failed to allocate all stream-types elements");
    }
}

PhysicalQueuesManager::~PhysicalQueuesManager()
{
    TrainingRetCode retCode = _destroyAllStreamTypesElements();
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_CRITICAL(SYN_STREAM, "Failed to destroy all stream-info elements");
    }
    // Log last operations to help debug in cases it is needed
}

TrainingRetCode PhysicalQueuesManager::createStream(BasicQueueInfo& rBasicQueueInfo, const uint32_t flags)
{
    std::lock_guard<std::mutex> guard(m_streamMutex);

    if (m_nextStreamIndex == std::numeric_limits<uint32_t>::max())
    {
        LOG_ERR(SYN_STREAM, "{}: All stream-IDs are used", HLLOG_FUNC);
        return TRAINING_RET_CODE_FAIL;
    }

    TrainingRetCode          retCode   = TRAINING_RET_CODE_SUCCESS;
    const internalStreamType queueType = rBasicQueueInfo.queueType;

    auto& streamtypeInfo = m_allStreamTypesInfo[queueType];

    if (!streamtypeInfo.valid)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to find queueType {} in DB", HLLOG_FUNC, queueType);
        return TRAINING_RET_CODE_FAIL;
    }

    retCode = _updateStreamTypeAllocation(queueType, true);
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        return retCode;
    }

    uint32_t& nextInstanceIndex = streamtypeInfo.nextInstanceIndex;
    HB_ASSERT(nextInstanceIndex >= 0 && nextInstanceIndex < streamtypeInfo.streamInfoDB.size(),
              "nextInstanceIndex out of range");
    BasicQueueInfo& basicQueueInfo = streamtypeInfo.streamInfoDB[nextInstanceIndex];

    rBasicQueueInfo.queueIndex      = basicQueueInfo.queueIndex;
    rBasicQueueInfo.logicalType     = basicQueueInfo.logicalType;
    rBasicQueueInfo.userQueueIndex  = m_nextStreamIndex;
    rBasicQueueInfo.pQueueInfo      = basicQueueInfo.pQueueInfo;

    nextInstanceIndex = (nextInstanceIndex + 1) % streamtypeInfo.streamInfoDB.size();

    LOG_TRACE(SYN_STREAM, "{}: stream-handle ({}) requested", HLLOG_FUNC, rBasicQueueInfo.getDescription());

    m_nextStreamIndex++;

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::destroyStream(const BasicQueueInfo& rBasicQueueInfo)
{
    TrainingRetCode retCode = TRAINING_RET_CODE_SUCCESS;

    std::lock_guard<std::mutex> guard(m_streamMutex);

    const internalStreamType queueType = rBasicQueueInfo.queueType;

    auto& streamTypeInfo = m_allStreamTypesInfo[queueType];

    if (!streamTypeInfo.valid)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to find queueType {} in DB", HLLOG_FUNC, queueType);
        return TRAINING_RET_CODE_FAIL;
    }

    retCode = _updateStreamTypeAllocation(queueType, false);
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to destroy stream {}", HLLOG_FUNC, rBasicQueueInfo.getDescription());

        return TRAINING_RET_CODE_FAIL;
    }

    return retCode;
}

TrainingRetCode PhysicalQueuesManager::signalEvent(const BasicQueueInfo& streamHandle, QmanEvent& eventHandle)
{
    spQueueInfo pQueueInfo = getStreamInfo(streamHandle);

    uint64_t        signalSequence = 0;
    TrainingRetCode retCode        = _signalEvent(signalSequence, pQueueInfo, true, eventHandle.getCollectTime());

    if (retCode == TRAINING_RET_CODE_SUCCESS)
    {
        eventHandle.setSignalSeqId(pQueueInfo->getQueueId(), signalSequence);
        eventHandle.clearTensorMapping();

        LOG_TRACE(SYN_PROGRESS,
                  GAUDI_PROGRESS_FMT,
                  streamHandle.queueType,
                  pQueueInfo->getQueueId(),
                  pQueueInfo->getPhysicalQueueOffset(),
                  pQueueInfo->getPhysicalQueuesId(),
                  signalSequence,
                  HLLOG_FUNC,
                  __LINE__);
    }

    return retCode;
}

TrainingRetCode PhysicalQueuesManager::waitForSignal(const BasicQueueInfo& streamHandle,
                                                     const QmanEvent&      eventHandle,
                                                     const unsigned int    flags)
{
    spQueueInfo pQueueInfo       = getStreamInfo(streamHandle);
    uint64_t    signalSequenceId = eventHandle.getSeqIdsExcludeStreamId(pQueueInfo->getQueueId());

    if (signalSequenceId == QmanEvent::INVALID_SEQ_ID)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: No synchronization required on a stream {} (with event {})",
                  HLLOG_FUNC,
                  streamHandle.getDescription(),
                  eventHandle.getHandle());

        return TRAINING_RET_CODE_NO_CHANGE;
    }

    TrainingQueue waitingQueue           = streamHandle.logicalType;
    bool          isCollectiveStream     = false;  // Supported for single stream-flavor only
    uint32_t      reductionStreamQueueId = 0;

    if ((waitingQueue >= TRAINING_QUEUE_COLLECTIVE_0) && (waitingQueue <= TRAINING_QUEUE_COLLECTIVE_2))
    {
        isCollectiveStream     = true;
        reductionStreamQueueId = gaudi::QmansDefinition::getInstance()->getCollectiveReductionEngineId();
    }

    uint64_t waitSequenceId  = 0;
    TrainingRetCode retCode         = _waitForSignal(waitSequenceId,
                                             signalSequenceId,
                                             pQueueInfo,
                                             true,
                                             isCollectiveStream,
                                             reductionStreamQueueId,
                                             eventHandle.getSequenceOffset());
    if (retCode == TRAINING_RET_CODE_FAIL)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to wait for event on training queue {}", HLLOG_FUNC, waitingQueue);
    }
    else
    {
        LOG_TRACE(SYN_PROGRESS,
                  GAUDI_PROGRESS_FMT_WAIT,
                  streamHandle.queueType,
                  pQueueInfo->getQueueId(),
                  pQueueInfo->getPhysicalQueueOffset(),
                  pQueueInfo->getPhysicalQueuesId(),
                  (waitSequenceId != std::numeric_limits<uint64_t>::max()) ? std::to_string(waitSequenceId) : "",
                  "(" + std::to_string(signalSequenceId) + ")",
                  HLLOG_FUNC,
                  __LINE__);
    }


    return retCode;
}

TrainingRetCode PhysicalQueuesManager::performStreamsSynchronization(const BasicQueueInfo& currentStreamHandle,
                                                                     const BasicQueueInfo& pPreviousStreamHandle,
                                                                     bool                  isUser)
{
    TrainingRetCode        retCode             = TRAINING_RET_CODE_SUCCESS;
    const BasicQueueInfo&  signalingStreamInfo = pPreviousStreamHandle;
    const BasicQueueInfo&  waitingStreamInfo   = currentStreamHandle;

    spQueueInfo pSignalingStreamInfo = getStreamInfo(signalingStreamInfo);
    spQueueInfo pWaitStreamInfo      = getStreamInfo(waitingStreamInfo);

    // Inner-signaling stream-sync will only be performed in case of:
    // (1) DMA-Synapse(Recipe-Cache) -> Enqueue:
    // We may perform a redundant signal call, but it requires multiple compute-streams usage, so we will IGNORE this
    //
    // (2) DMA-Synapse(User's workspace) <-> Enqueue:
    // We prefer to take the KISS approach and always add signal

    LOG_DEBUG(SYN_STREAM, "{}", __FUNCTION__);

    uint64_t signalSequence = 0;
    retCode = _signalEvent(signalSequence, pSignalingStreamInfo, isUser);

    if (retCode == TRAINING_RET_CODE_FAIL)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to signal", HLLOG_FUNC);
        return retCode;
    }

    LOG_TRACE(SYN_PROGRESS,
              GAUDI_PROGRESS_FMT,
              signalingStreamInfo.queueType,
              pSignalingStreamInfo->getQueueId(),
              pSignalingStreamInfo->getPhysicalQueueOffset(),
              pSignalingStreamInfo->getPhysicalQueuesId(),
              signalSequence,
              HLLOG_FUNC,
              __LINE__);

    bool     isCollectiveStream     = false;  // Supported for single stream-flavor only
    uint32_t reductionStreamQueueId = 0;
    if ((currentStreamHandle.logicalType >= TRAINING_QUEUE_COLLECTIVE_0) &&
        (currentStreamHandle.logicalType <= TRAINING_QUEUE_COLLECTIVE_2))
    {
        isCollectiveStream     = true;
        reductionStreamQueueId = gaudi::QmansDefinition::getInstance()->getCollectiveReductionEngineId();
    }

    uint64_t waitSequenceId  = 0;
    retCode = _waitForSignal(waitSequenceId, signalSequence, pWaitStreamInfo,
                             isUser, isCollectiveStream, reductionStreamQueueId);
    if (retCode == TRAINING_RET_CODE_FAIL)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to wait for signal {} on {} training queue",
                HLLOG_FUNC,
                signalSequence,
                waitingStreamInfo.logicalType);
        return TRAINING_RET_CODE_FAIL;
    }
    else
    {
        LOG_TRACE(SYN_PROGRESS,
                  GAUDI_PROGRESS_FMT_WAIT,
                  waitingStreamInfo.queueType,
                  pWaitStreamInfo->getQueueId(),
                  pWaitStreamInfo->getPhysicalQueueOffset(),
                  pWaitStreamInfo->getPhysicalQueuesId(),
                  (waitSequenceId != std::numeric_limits<uint64_t>::max()) ? std::to_string(waitSequenceId) : "",
                  "(" + std::to_string(signalSequence) + ")",
                  HLLOG_FUNC,
                  __LINE__);
    }

    return retCode;
}

TrainingRetCode PhysicalQueuesManager::updateStreamPostExecution(const BasicQueueInfo&     rBasicQueueInfo,
                                                                 const InternalWaitHandle& waitHandle,
                                                                 const std::string&        desc)
{
    spQueueInfo pQueueInfo = getStreamInfo(rBasicQueueInfo);
    return pQueueInfo->updateStatus(waitHandle, desc);
}

TrainingRetCode PhysicalQueuesManager::getPhysicalQueueIds(const BasicQueueInfo& streamHandle,
                                                           PhysicalQueuesId&     physicalQueuesId)
{
    spQueueInfo pQueueInfo = getStreamInfo(streamHandle);
    physicalQueuesId       = pQueueInfo->getPhysicalQueuesId();

    return TRAINING_RET_CODE_SUCCESS;
}

spQueueInfo PhysicalQueuesManager::getStreamInfo(const BasicQueueInfo& rBasicQueueInfo)
{
    return rBasicQueueInfo.pQueueInfo;
}

uint64_t PhysicalQueuesManager::getStreamId(const BasicQueueInfo& rBasicQueueInfo) const
{
    return rBasicQueueInfo.pQueueInfo->getQueueId();
}

bool PhysicalQueuesManager::isStreamSynchronizationRequired(const BasicQueueInfo& currentStreamHandle,
                                                            const BasicQueueInfo& pPreviousStreamHandle)
{
    return (pPreviousStreamHandle.handle != currentStreamHandle.handle);
}

TrainingRetCode PhysicalQueuesManager::getLastWaitHandles(InternalWaitHandlesVector& lastWaitHandles)
{
    LOG_TRACE(SYN_STREAM, "{}", HLLOG_FUNC);

    std::lock_guard<std::mutex> guard(m_streamMutex);
    if (m_streamsInfo.size() == 0)
    {
        return TRAINING_RET_CODE_NO_CHANGE;
    }

    for (auto& currStream : m_streamsInfo)
    {
        currStream.second->getLastWaitHandles(lastWaitHandles);
    }

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::getLastWaitHandles(const BasicQueueInfo&      streamHandle,
                                                          InternalWaitHandlesVector& lastWaitHandles)
{
    spQueueInfo pQueueInfo = getStreamInfo(streamHandle);
    return pQueueInfo->getLastWaitHandles(lastWaitHandles);
}

uint32_t PhysicalQueuesManager::getStreamPhysicalOffset(const BasicQueueInfo& streamHandle)
{
    spQueueInfo pQueueInfo = getStreamInfo(streamHandle);
    return pQueueInfo->getPhysicalQueueOffset();
}

// Should be called during CTR, and hence no mutex is used
TrainingRetCode PhysicalQueuesManager::_allocateAllStreamTypesElements()
{
    TrainingRetCode retCode             = TRAINING_RET_CODE_SUCCESS;
    TrainingQueue   commandLogicalQueue = TRAINING_QUEUE_NUM;

    HB_ASSERT(s_streamsAllocationRestriction.size() == INTERNAL_STREAM_TYPE_NUM,
              "bad size for s_streamsAllocationRestriction");

    for (uint32_t streamTypeVal = 0; streamTypeVal < INTERNAL_STREAM_TYPE_NUM; streamTypeVal++)
    {
        const internalStreamType queueType   = (internalStreamType)streamTypeVal;
        const uint32_t           maxElements = m_pStreamCreator->getStreamsElementsRestriction(queueType);

        if (maxElements == 0)
        {
            m_allStreamTypesInfo[streamTypeVal].valid = false;
            continue;
        }

        m_allStreamTypesInfo[streamTypeVal].valid = true;

        uint32_t              physicalQueueOffset     = 0;
        SingleStreamTypeInfo& requestedStreamTypeInfo = m_allStreamTypesInfo[queueType];
        requestedStreamTypeInfo.numOfElements         = 0;
        requestedStreamTypeInfo.nextInstanceIndex     = 0;

        for (uint32_t i = 0; i < maxElements; i++)
        {
            retCode = _selectTrainingQueue(queueType, commandLogicalQueue, i);
            if (retCode != TRAINING_RET_CODE_SUCCESS)
            {
                LOG_WARN(SYN_STREAM, "{}: Cant select logical-queue from stream type {}", HLLOG_FUNC, queueType);
                return retCode;
            }

            retCode = _getPhysicalStreamOffset(commandLogicalQueue, physicalQueueOffset);
            if (retCode != TRAINING_RET_CODE_SUCCESS)
            {
                LOG_ERR(SYN_STREAM, "{}: Failed to retrieve physicalQueueOffset {}", HLLOG_FUNC, physicalQueueOffset);
                return retCode;
            }

            PhysicalQueuesId physicalQueuesId;
            bool             status =
                m_pStreamCreator->getPhysicalQueueId(physicalQueuesId, commandLogicalQueue, physicalQueueOffset);
            if (!status)
            {
                LOG_ERR(SYN_STREAM,
                        "{}: Failed to getPhysicalQueueId for training-queue {}",
                        HLLOG_FUNC,
                        commandLogicalQueue);
                return TRAINING_RET_CODE_FAIL;
            }

            std::shared_ptr<QueueInfo> pQueueInfo =
                std::make_shared<QueueInfo>(m_nextStreamIndex, physicalQueueOffset, physicalQueuesId, 0);
            ;
            m_streamsInfo.emplace(m_nextStreamIndex, pQueueInfo);

            BasicQueueInfo basicQueueInfo;
            basicQueueInfo.queueIndex      = m_nextStreamIndex;
            basicQueueInfo.queueType       = queueType;
            basicQueueInfo.logicalType     = commandLogicalQueue;
            basicQueueInfo.userQueueIndex  = 0;
            basicQueueInfo.pQueueInfo      = pQueueInfo;
            m_nextStreamIndex++;

            requestedStreamTypeInfo.streamInfoDB.push_back(basicQueueInfo);
        }
    }

    return TRAINING_RET_CODE_SUCCESS;
}

// Should be called during DTR, and hence no mutex is used
TrainingRetCode PhysicalQueuesManager::_destroyAllStreamTypesElements()
{
    TrainingRetCode retCode = TRAINING_RET_CODE_SUCCESS;

    for (uint32_t streamTypeVal = 0; streamTypeVal < INTERNAL_STREAM_TYPE_NUM; streamTypeVal++)
    {
        internalStreamType queueType = (internalStreamType)streamTypeVal;

        auto requestedStreamTypeInfo = m_allStreamTypesInfo[queueType];

        std::vector<BasicQueueInfo>& streamInfoDB = requestedStreamTypeInfo.streamInfoDB;
        for (auto streamInfoElement : streamInfoDB)
        {
            TrainingRetCode elementRetCode = _destroyStream(streamInfoElement);
            if (elementRetCode != TRAINING_RET_CODE_SUCCESS)
            {
                LOG_ERR(SYN_STREAM,
                        "{}: Failed to destroy stream-info element {}",
                        HLLOG_FUNC,
                        streamInfoElement.getDescription());
                retCode = elementRetCode;
            }
        }
        requestedStreamTypeInfo.valid = false;
    }

    m_streamsInfo.clear();

    return retCode;
}

TrainingRetCode PhysicalQueuesManager::_getPhysicalStreamOffset(TrainingQueue logicalQueue,
                                                                uint32_t&     physicalQueueOffset)
{
    // mosher - Physical stream-offset should not be related to the stream-type

    switch (logicalQueue)
    {
        // todo - check if case is used
        case TRAINING_QUEUE_DEV_TO_DEV_SYNAPSE:
            physicalQueueOffset = 3;  // master in Master Slave Definition Table for GRECO -> DDMA0 (stream 3)
            break;

        case TRAINING_QUEUE_DMA_UP:
        case TRAINING_QUEUE_DMA_DOWN_USER:
            physicalQueueOffset = 0;
            break;

        case TRAINING_QUEUE_COMPUTE_0:
            physicalQueueOffset = 0;
            break;

        case TRAINING_QUEUE_DMA_DOWN_SYNAPSE:
            physicalQueueOffset = 1;
            break;

        case TRAINING_QUEUE_COMPUTE_1:
            physicalQueueOffset = 1;
            break;

        case TRAINING_QUEUE_COLLECTIVE_0:
            physicalQueueOffset = 1;
            break;

        case TRAINING_QUEUE_COLLECTIVE_1:
            physicalQueueOffset = 2;
            break;

        case TRAINING_QUEUE_COLLECTIVE_2:
            physicalQueueOffset = 3;
            break;

        case TRAINING_QUEUE_NUM:
            HB_ASSERT(false, "Unexpected logical-queue type");
            return TRAINING_RET_CODE_FAIL;
    }

    LOG_DEBUG(SYN_STREAM, "{}: logicalQueue {} physicalQueueOffset {}", HLLOG_FUNC, logicalQueue, physicalQueueOffset);

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::_selectTrainingQueue(const internalStreamType& queueType,
                                                            TrainingQueue&            type,
                                                            uint32_t                  elementIndex)
{
    return m_pStreamCreator->selectTrainingQueue(queueType, type, elementIndex);
}

TrainingRetCode PhysicalQueuesManager::reserveSignalObjects(const BasicQueueInfo& streamHandle,
                                                            size_t                numOfTensors,
                                                            reserve_sig_handle*   sigHandle)
{
    spQueueInfo pQueueInfo = getStreamInfo(streamHandle);
    HB_ASSERT(pQueueInfo != nullptr, "pQueueInfo is nullptr");

    const PhysicalQueuesId& mainCbsPhysicalQueuesIds = pQueueInfo->getPhysicalQueuesId();

    uint32_t queueID = mainCbsPhysicalQueuesIds;
    int      fd      = OSAL::getInstance().getFd();
    if (fd < 0)
    {
        LOG_ERR(SYN_STREAM, "{}: failed to get file descriptor for device", HLLOG_FUNC);
        return TRAINING_RET_CODE_FAIL;
    }
    struct hlthunk_sig_res_in sigResIn;
    sigResIn.count = numOfTensors;
    struct hlthunk_sig_res_out sigResOut;
    sigResIn.queue_index = queueID;
    int ret              = hlthunk_reserve_encaps_signals(fd, &sigResIn, &sigResOut);
    if (ret)
    {
        LOG_ERR(SYN_STREAM, "{}: failed to reserve encaps signals", HLLOG_FUNC);
        return TRAINING_RET_CODE_FAIL;
    }

    *sigHandle = sigResOut.handle;
    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::unreserveSignalObjects(const BasicQueueInfo& streamHandle,
                                                              reserve_sig_handle*   sigHandle)
{
    spQueueInfo pQueueInfo = getStreamInfo(streamHandle);
    HB_ASSERT(pQueueInfo != nullptr, "pQueueInfo is nullptr");
    int fd = OSAL::getInstance().getFd();
    if (fd < 0)
    {
        LOG_ERR(SYN_STREAM, "{}: failed to get device file descriptor {}", HLLOG_FUNC, fd);
        return TRAINING_RET_CODE_FAIL;
    }
    uint32_t status = 0;
    if (hlthunk_unreserve_encaps_signals(fd, sigHandle, &status))
    {
        LOG_ERR(SYN_STREAM, "{}: unable to unreserve encaps signals - status {}", HLLOG_FUNC, status);
        return TRAINING_RET_CODE_FAIL;
    }
    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode
PhysicalQueuesManager::_signalEvent(uint64_t& signalSequence, spQueueInfo pQueueInfo, bool isUser, bool collectTime)
{
    HB_ASSERT(pQueueInfo != nullptr, "pQueueInfo is nullptr");

    const PhysicalQueuesId& mainCbsPhysicalQueuesIds = pQueueInfo->getPhysicalQueuesId();

    uint32_t queueID = mainCbsPhysicalQueuesIds;

    struct hlthunk_signal_in  signalIn;
    struct hlthunk_signal_out signalOut;
    int                       rc;

    memset(&signalIn, 0, sizeof(signalIn));
    memset(&signalOut, 0, sizeof(signalOut));

    signalIn.num_chunks_restore = 0;

    signalIn.queue_index = queueID;

    signalIn.flags |= (collectTime ? HL_CS_FLAGS_TIMESTAMP : 0);

    int fd = OSAL::getInstance().getFd();

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    STAT_GLBL_START(signalSubmission);
    rc = hlthunk_signal_submission(fd, &signalIn, &signalOut);

    if (isUser)
    {
        STAT_GLBL_COLLECT_TIME(signalSubmission, globalStatPointsEnum::synEventRecordUserSubmit);
    }
    else
    {
        STAT_GLBL_COLLECT_TIME(signalSubmission, globalStatPointsEnum::EventRecordLaunchSubmit);
    }

    if (rc != 0)
    {
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
        LOG_ERR(SYN_STREAM, "{}: Failed to signal {}", HLLOG_FUNC, queueID);
        return TRAINING_RET_CODE_FAIL;
    }

    signalSequence = signalOut.seq;

    ETL_ADD_LOG_DEBUG(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                      logId,
                      SYN_STREAM,
                      "Submitted Signal request on queue {}, with handle = {}",
                      queueID,
                      signalOut.seq);

    pQueueInfo->operationCompletion(true, signalSequence, false, false, "_signalEvent");

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::_waitForSignal(uint64_t&   waitSequenceId,
                                                      uint64_t    signalSeqId,
                                                      spQueueInfo pQueueInfo,
                                                      bool        isUser,
                                                      bool        isCollectiveStream,
                                                      uint32_t    reductionStreamQueueId,
                                                      uint64_t    sequenceOffset)
{
    HB_ASSERT(pQueueInfo != nullptr, "pQueueInfo is nullptr");

    const PhysicalQueuesId& mainCbsPhysicalQueuesIds = pQueueInfo->getPhysicalQueuesId();

    uint32_t waitQueueID = mainCbsPhysicalQueuesIds;

    if (signalSeqId == QmanEvent::INVALID_SEQ_ID)
    {
        return TRAINING_RET_CODE_NO_CHANGE;
    }

    bool operationStatus = true;

    if (!isCollectiveStream)
    {  // we need sequence offset only non-collective streams
        operationStatus = _performRegularWaitRequest(waitSequenceId, signalSeqId, waitQueueID, isUser, sequenceOffset);
    }
    else
    {
        operationStatus = _performCollectiveWaitRequest(waitSequenceId,
                                                        signalSeqId,
                                                        waitQueueID,
                                                        reductionStreamQueueId,
                                                        isUser,
                                                        sequenceOffset);
    }

    LOG_DEBUG(
        SYN_STREAM,
        "{}: wait-for-signal waitSequenceId 0x{:x} signalSeqId 0x{:x} waitQueueID {} isUser {}, sequenceOffset {}",
        __FUNCTION__,
        waitSequenceId,
        signalSeqId,
        waitQueueID,
        isUser,
        sequenceOffset);

    if (!operationStatus)
    {
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
        LOG_ERR(SYN_STREAM,
                "{}: Failed to wait-for-signal 0x{:x} on queue 0x{:x}",
                HLLOG_FUNC,
                signalSeqId,
                waitQueueID);
        return TRAINING_RET_CODE_FAIL;
    }

    pQueueInfo->operationCompletion(false, waitSequenceId, isUser, false, "_waitForSignal");

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::_updateStreamTypeAllocation(internalStreamType queueType, bool isAlloc)
{
    auto& requestedStreamTypeInfo = m_allStreamTypesInfo[queueType];

    if (!requestedStreamTypeInfo.valid)
    {
        return TRAINING_RET_CODE_INVALID_REQUEST;
    }

    uint32_t& numOfElements = requestedStreamTypeInfo.numOfElements;

    if (isAlloc)
    {
        uint32_t maxStreams = s_streamsAllocationRestriction[queueType];
        if (numOfElements >= maxStreams)
        {
            LOG_WARN(SYN_STREAM, "{}: All streams of type {} taken", HLLOG_FUNC, queueType);
            return TRAINING_RET_CODE_FULLY_USED;
        }
        numOfElements++;
    }
    else
    {
        if (numOfElements == 0)
        {
            LOG_WARN(SYN_STREAM, "{}: Trying to free unused stream slot {}", HLLOG_FUNC, queueType);
            return TRAINING_RET_CODE_INVALID_REQUEST;
        }

        numOfElements--;
    }

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode PhysicalQueuesManager::_destroyStream(const BasicQueueInfo& streamHandle)
{
    auto stream = m_streamsInfo.find(streamHandle.queueIndex);
    if (stream == m_streamsInfo.end())
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid stream-handle ({}) requested", HLLOG_FUNC, streamHandle.getDescription());
        return TRAINING_RET_CODE_INVALID_REQUEST;
    }

    TrainingRetCode retCode = TRAINING_RET_CODE_INVALID_REQUEST;
    if (m_streamsInfo.erase(streamHandle.queueIndex) == 1)
    {
        retCode = TRAINING_RET_CODE_SUCCESS;
    }

    // The m_allStreamTypesInfo DB will be cleared outside this method call

    return retCode;
}

bool PhysicalQueuesManager::_performRegularWaitRequest(uint64_t& waitSequenceId,
                                                       uint64_t  signalSequenceId,
                                                       uint32_t  waitQueueId,
                                                       bool      isUserRequest,
                                                       uint32_t  sequenceOffset)
{
    struct hlthunk_wait_in              wait_in;
    struct hlthunk_wait_out             wait_out;
    struct hlthunk_wait_for_signal_data wait_for_signal;

    memset(&wait_in, 0, sizeof(wait_in));
    memset(&wait_out, 0, sizeof(wait_out));
    memset(&wait_for_signal, 0, sizeof(wait_for_signal));

    wait_for_signal.queue_index          = waitQueueId;
    wait_for_signal.signal_seq_arr       = &signalSequenceId;
    wait_for_signal.signal_seq_nr        = 1;
    wait_for_signal.encaps_signal_offset = 0;

    wait_in.hlthunk_wait_for_signal = (uint64_t*)&wait_for_signal;
    wait_in.num_wait_for_signal     = 1;

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    int fd = OSAL::getInstance().getFd();
    STAT_GLBL_START(waitSignal);
    int rc = 0;
    if (SEQUENCE_OFFSET_NOT_USED != sequenceOffset)
    {
        sequenceOffset += 1;  // promoting by 1 as this is actually sequenceIndex
        wait_for_signal.encaps_signal_offset = sequenceOffset;
        wait_in.flags |= HL_CS_FLAGS_ENCAP_SIGNALS;
        wait_for_signal.encaps_signal_seq = signalSequenceId;
        rc                                = hlthunk_wait_for_reserved_encaps_signals(fd, &wait_in, &wait_out);
    }
    else
    {
        rc = hlthunk_wait_for_signal(fd, &wait_in, &wait_out);
    }

    if (isUserRequest)
    {
        STAT_GLBL_COLLECT_TIME(waitSignal, globalStatPointsEnum::StreamWaitEventUserSubmit);
    }
    else
    {
        STAT_GLBL_COLLECT_TIME(waitSignal, globalStatPointsEnum::StreamWaitEventLaunchSubmit);
    }

    waitSequenceId = wait_out.seq;

    ETL_ADD_LOG_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                      logId,
                      SYN_STREAM,
                      "Wait for signal {} on queue {} with handle {}, sequenceOffset {}",
                      signalSequenceId,
                      waitQueueId,
                      waitSequenceId,
                      (int)sequenceOffset);

    return (rc == 0);
}

bool PhysicalQueuesManager::_performCollectiveWaitRequest(uint64_t& waitSequenceId,
                                                          uint64_t  signalSequenceId,
                                                          uint32_t  waitQueueId,
                                                          uint32_t  reductionStreamQueueId,
                                                          bool      isUserRequest,
                                                          uint32_t  sequenceOffset)
{
    struct hlthunk_wait_in              wait_in;
    struct hlthunk_wait_out             wait_out;
    struct hlthunk_wait_for_signal_data wait_for_signal;

    memset(&wait_in, 0, sizeof(wait_in));
    memset(&wait_out, 0, sizeof(wait_out));
    memset(&wait_for_signal, 0, sizeof(wait_for_signal));

    wait_for_signal.queue_index          = waitQueueId;
    wait_for_signal.signal_seq_arr       = &signalSequenceId;
    wait_for_signal.signal_seq_nr        = 1;
    wait_for_signal.collective_engine_id = reductionStreamQueueId;
    wait_for_signal.encaps_signal_offset = 0;

    wait_in.hlthunk_wait_for_signal = (uint64_t*)&wait_for_signal;
    wait_in.num_wait_for_signal     = 1;

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    int fd = OSAL::getInstance().getFd();
    STAT_GLBL_START(waitSignalCollective);
    int rc = 0;
    if (SEQUENCE_OFFSET_NOT_USED != sequenceOffset)
    {
        sequenceOffset += 1;  // promoting by 1 as this is actually sequenceIndex
        wait_for_signal.encaps_signal_offset = sequenceOffset;
        wait_in.flags |= HL_CS_FLAGS_ENCAP_SIGNALS;
        wait_for_signal.encaps_signal_seq = signalSequenceId;
        rc = hlthunk_wait_for_reserved_encaps_collective_signals(fd, &wait_in, &wait_out);
    }
    else
    {
        rc = hlthunk_wait_for_collective_signal(fd, &wait_in, &wait_out);
    }

    if (isUserRequest)
    {
        STAT_GLBL_COLLECT_TIME(waitSignalCollective, globalStatPointsEnum::StreamWaitEventUserSubmit);
    }
    else
    {
        STAT_GLBL_COLLECT_TIME(waitSignalCollective, globalStatPointsEnum::StreamWaitEventLaunchSubmit);
    }

    waitSequenceId = wait_out.seq;

    ETL_ADD_LOG_DEBUG(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                      logId,
                      SYN_STREAM,
                      "Wait for signal {} on queue {} with handle {}, sequenceOffset {}",
                      signalSequenceId,
                      waitQueueId,
                      waitSequenceId,
                      (int)sequenceOffset);

    return (rc == 0);
}
