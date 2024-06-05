#include "dfa_defines.hpp"
#include "scal_stream_copy_scheduler_mode.hpp"
#include "syn_logging.h"
#include "defs.h"
#include "habana_global_conf_runtime.h"
#include "synapse_common_types.h"

#include "device_info_interface.hpp"
#include "scal_completion_group.hpp"
#include "scal_memory_pool.hpp"

#include "log_manager.h"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "runtime/scal/common/infra/scal_utils.hpp"

#define LOG_SCALSTRM_TRACE(msg, ...)                                                                                   \
    LOG_TRACE(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_DEBUG(msg, ...)                                                                                   \
    LOG_DEBUG(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_INFO(msg, ...)                                                                                    \
    LOG_INFO(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_ERR(msg, ...)                                                                                     \
    LOG_ERR(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_CRITICAL(msg, ...)                                                                                \
    LOG_CRITICAL(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)

using namespace common;

// We assume enum PdmaDirCtx match the enum in FW. The compile time asserts are to verify it
static_assert((uint32_t)PdmaDirCtx::UP == (uint32_t)g2fw::sched_arc_pdma_dir_t::PDMA_DIR_DEV2HOST);
static_assert((uint32_t)PdmaDirCtx::DOWN == (uint32_t)g2fw::sched_arc_pdma_dir_t::PDMA_DIR_HOST2DEV);
static_assert((uint32_t)PdmaDirCtx::DEV2DEV == (uint32_t)g2fw::sched_arc_pdma_dir_t::PDMA_DIR_DEV2DEV);

static_assert((uint32_t)PdmaDirCtx::UP == (uint32_t)g3fw::sched_arc_pdma_dir_t::PDMA_DIR_DEV2HOST);
static_assert((uint32_t)PdmaDirCtx::DOWN == (uint32_t)g3fw::sched_arc_pdma_dir_t::PDMA_DIR_HOST2DEV);
static_assert((uint32_t)PdmaDirCtx::DEV2DEV == (uint32_t)g3fw::sched_arc_pdma_dir_t::PDMA_DIR_DEV2DEV);

static_assert((uint8_t)PdmaDirCtx::NUM <= 4);  // we assume no more than 2 bits
static_assert(INTERNAL_STREAM_TYPE_NUM <= 8);  // we assume no more than 3 bits

ScalStreamCopySchedulerMode::ScalStreamCopySchedulerMode(const ScalStreamCtorInfoBase* pScalStreamCtorInfo)
: ScalStreamCopyBase(pScalStreamCtorInfo),
  m_maxBatchesInChunk(0),
  m_scalMonitor(pScalStreamCtorInfo->syncMonitorId,
                *pScalStreamCtorInfo->deviceInfoInterface,
                pScalStreamCtorInfo->devHndl),
  m_streamCyclicBuffer(m_name)
{
    const common::DeviceInfoInterface& deviceInfoInterface = *pScalStreamCtorInfo->deviceInfoInterface;
    m_qmanEngineGroupsAmount                 = deviceInfoInterface.getQmanEngineGroupTypeCount();
    m_schedPdmaCommandsTransferMaxParamCount = deviceInfoInterface.getSchedPdmaCommandsTransferMaxParamCount();
    m_schedPdmaCommandsTransferMaxCopySize   = deviceInfoInterface.getSchedPdmaCommandsTransferMaxCopySize();
}

ScalStreamCopySchedulerMode::~ScalStreamCopySchedulerMode()
{
}

/*
 ***************************************************************************************************
 *   @brief addLbwWrite() adds a packet, which writes data to an LBW-address
 *
 *   @param  dst_addr     - LBW address to write to
 *   @param  data         - data to be writen
 *   @param  block_stream -
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCopySchedulerMode::addLbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool send, bool isInSyncMgr)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return addPacket<LbwWritePkt<T>>(send, (uint32_t)dst_addr, data, block_stream);
        },
        m_gxPackets);
}

bool ScalStreamCopySchedulerMode::setStreamPriority(uint32_t priority)
{
    ScalRtn rc = scal_stream_set_priority(m_streamHndl, priority);
    LOG_SCALSTRM_INFO("set stream priority {} rc {} m_streamHndl {:x}", priority, rc, TO64(m_streamHndl));
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to set stream priority rc {}", rc);
        return false;
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @brief addPdmaCommandsTransfer() sends batched Pdma request (Pdma Commands)
 *
 *   @param  operation (UP/DOWN), dst, src, size, send (do submit or just put on queue)
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCopySchedulerMode::addPdmaBatchTransfer(ResourceStreamType  resourceType,
                                                            const void* const&& params,
                                                            uint32_t            params_count,
                                                            bool                send,
                                                            uint8_t             apiId,
                                                            bool                bMemset,
                                                            uint32_t            payload,
                                                            uint32_t            pay_addr,
                                                            uint32_t            completionGroupIndex)
{
    if ((completionGroupIndex != MAX_COMP_SYNC_GROUP_COUNT) && (hasPayload(pay_addr)))
    {
        LOG_SCALSTRM_ERR("Both Payload-address and CS-Index are set");
        return synInvalidArgument;
    }

    const PdmaParams         pdmaParams = getPdmaParams(resourceType, m_qmanEngineGroupsAmount);
    const bool               watchdog   = m_pScalCompletionGroup->getCgInfo().tdr_enabled;
    const ResourceStreamType realType   = isComputeStream() ? ResourceStreamType::COMPUTE : resourceType;
    const uint8_t            ctxId      = getContextId(pdmaParams.dir, realType, getStreamIndex());

    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return addPacketCommon<BatchedPdmaTransferPkt<T>>(send,
                                                              BatchedPdmaTransferPkt<T>::getSize(params_count),
                                                              std::forward<const void* const>(params),
                                                              params_count,
                                                              pdmaParams.engGrp,
                                                              pdmaParams.workloadType,
                                                              ctxId,
                                                              apiId,
                                                              payload,
                                                              pay_addr,
                                                              watchdog,
                                                              bMemset,
                                                              completionGroupIndex);
        },
        m_gxPackets);
}

/*
 ***************************************************************************************************
 *   @brief addFenceWait() - sends a fence decrement command
 *
 *   @param  target - the value that will be decremented from the fence value
 *   @param  fenceId
 *   @param  send
 *   @param  isGlobal
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCopySchedulerMode::addFenceWait(uint32_t target, FenceIdType fenceId, bool send, bool isGlobal)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            if constexpr(!std::is_same_v<T, G2Packets>)
            {
                if (isGlobal)
                {
                    return addPacket<FenceWaitPkt<T>>(send, fenceId, target);
                }
                else
                {
                    return addPacket<AcpFenceWaitPkt<T>>(send, fenceId, target);
                }
            }
            else
            {
                return addPacket<FenceWaitPkt<T>>(send, fenceId, target);
            }
        },
        m_gxPackets);
}

/*
 ***************************************************************************************************
 *   @brief sendFenceInc() - sends a fence increment command
 *
 *   @param  fenceId
 *   @param  send
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCopySchedulerMode::addFenceInc(FenceIdType fenceId, bool send)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return addPacket<FenceIncImmediatePkt<T>>(send, fenceId);
        },
        m_gxPackets);
}

/*
 ***************************************************************************************************
 *   @brief initInternal() init the stream.
 *   get stream handle
 *   set priority
 *   allocate memory for cyclic buffer
 *   assign the memory for the buffer
 *   get stream info
 *
 *   @param  None
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCopySchedulerMode::initInternal()
{
    if (!retrievStreamHandle())
    {
        return synFail;
    }

    const uint32_t priority = SCAL_LOW_PRIORITY_STREAM;
    if (!setStreamPriority(priority))
    {
        return synFail;
    }

    if (!allocateAndSetCommandsBuffer())
    {
        return synFail;
    }

    if (!retrievStreamInfo())
    {
        return synFail;
    }

    scal_control_core_info_t streamCoreInfo;
    ScalRtn rc = scal_control_core_get_info(m_schedulerHandle, &streamCoreInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to get streamCoreInfo rc {}", rc);
        return synFail;
    }

    if (!retrievCommandsBufferInfo())
    {
        return synFail;
    }

    m_streamCyclicBuffer.init(m_pScalCompletionGroup,
                              getCommandBufferHostAddress(),
                              TO64(m_streamHndl),
                              m_cmdAlign,
                              m_submitAlign,
                              m_gxPackets);

    rc = initStaticMonitor();
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("failed in initStaticMonitor rc {}", rc);
        return synFail;
    }

    // calculate m_maxBatchesInChunk (how many pdmas we can do before sending signal_to_cg)
    const uint32_t maxPdmaBatchSize = std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return BatchedPdmaTransferPkt<T>::getSize(m_schedPdmaCommandsTransferMaxParamCount);
        },
        m_gxPackets);  // currently 0x5C

    // How many in a chunk
    uint16_t pdmaInAlignChunk      = m_cmdAlign / maxPdmaBatchSize;   // How many in 0x100, currently 2
    uint32_t ccbChunkSize          = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024 / GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
    uint16_t alignChunksInCcbChunk = ccbChunkSize / m_cmdAlign;  // currently 0x4000 / 0x100 = 0x40
    m_maxBatchesInChunk =
        (alignChunksInCcbChunk - 1) * pdmaInAlignChunk;  // Currently, 0x7E. The -1 is just to be on the safe side

    LOG_SCALSTRM_INFO("Need to add signal_to_cg every {:x} pdma-s. pdmaInAlignChunk {:x}, alignChunksInCcbChunk {:x}",
                      m_maxBatchesInChunk,
                      pdmaInAlignChunk,
                      alignChunksInCcbChunk);

    return synSuccess;
}

std::string ScalStreamCopySchedulerMode::getSchedulerInfo(unsigned& schedulerIdx) const
{
    scal_control_core_info_t streamCoreInfo {};
    int rc = scal_control_core_get_info(m_schedulerHandle, &streamCoreInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEV_FAIL, "Could not get streamCoreInfo rc {}", rc);
    }

    schedulerIdx  = streamCoreInfo.idx;
    return streamCoreInfo.name ? streamCoreInfo.name : "N/A";
}

ScalStreamCopySchedulerMode::PdmaParams ScalStreamCopySchedulerMode::getPdmaParams(ResourceStreamType operationType,
                                                 uint8_t            qmanEngineGroupsAmount)
{
    PdmaParams pdmaParams = {.engGrp = qmanEngineGroupsAmount, .workloadType = 0, .dir = PdmaDirCtx::UP};

    if ((operationType < ResourceStreamType::PDMA_FIRST) ||
        (operationType > ResourceStreamType::PDMA_LAST) ||
        (m_devStreamInfo.resourcesInfo[(uint8_t)operationType].engineGrpArr.numEngineGroups == 0))
    {
        HB_ASSERT(false, "Invalid operationType {}", operationType);
        return pdmaParams;
    }

    // Cluster
    pdmaParams.engGrp = m_devStreamInfo.resourcesInfo[(uint8_t)operationType].engineGrpArr.eng[0];

    // Workload-type
    if (operationType == ResourceStreamType::SYNAPSE_DMA_DOWN)
    {
        pdmaParams.workloadType = 1;
    }

    pdmaParams.dir = getDir(operationType);

    return pdmaParams;
}

synStatus ScalStreamCopySchedulerMode::memcopy(ResourceStreamType           resourceType,
                                               const internalMemcopyParams& memcpyParams,
                                               bool                         isUserRequest,
                                               bool                         send,
                                               uint8_t                      apiId,
                                               ScalLongSyncObject&          longSo,
                                               uint64_t                     overrideMemsetVal,
                                               MemcopySyncInfo&             memcopySyncInfo)
{
    LOG_TRACE_T(SYN_STREAM,
                "{}: resourceType={} isUserRequest={} send={} overrideMemsetVal={}, "
                "pdmaSyncMechanism={} consecutiveWaitCommands={}",
                HLLOG_FUNC,
                resourceType,
                isUserRequest,
                send,
                overrideMemsetVal,
                memcopySyncInfo.m_pdmaSyncMechanism,
                m_consecutiveWaitCommands);

    if (memcpyParams.size() == 0)
    {
        LOG_DEBUG(SYN_STREAM, "{}: No memcopy params given", HLLOG_FUNC);
        return synSuccess;
    }

    synStatus status              = synSuccess;
    uint32_t  cgIndex             = MAX_COMP_SYNC_GROUP_COUNT;
    bool      sendUnfence         = false;
    bool      sendLastPdmaCommand = send;

    // calculate how many PDMA batches can fit into one chunk
    // done by dividing size of chunk (in bytes) by size of the biggest batch

    switch (memcopySyncInfo.m_pdmaSyncMechanism)
    {
        case PDMA_TX_SYNC_MECH_LONG_SO:
            cgIndex = m_pScalCompletionGroup->getIndexInScheduler();
            break;

        case PDMA_TX_SYNC_FENCE_ONLY:
            sendUnfence = true;
            break;

        default:
            break;
    }

    bool     isDmaDownSynapse = (resourceType == ResourceStreamType::SYNAPSE_DMA_DOWN);
    size_t   pdmaIndex        = 0;
    size_t   batchCounter     = 0;
    unsigned batchSize        = 0;

    internalMemcopyParamEntry curMemcpyParam   = memcpyParams[0];
    bool                      curMemsetMode    = curMemcpyParam.src == 0;
    const uint64_t            maxCopySize      = m_schedPdmaCommandsTransferMaxCopySize;
    // parameters array
    internalMemcopyParamEntry pdmaBatchParams[m_schedPdmaCommandsTransferMaxParamCount];

    while (pdmaIndex < memcpyParams.size())
    {
        // check for invalid input (memset with INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE)
        if (isDmaDownSynapse && curMemsetMode)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: src == 0 (i = {}). memset mode is not supported for {}",
                    HLLOG_FUNC,
                    pdmaIndex,
                    resourceType);
            return synInvalidArgument;
        }

        uint32_t itemSize = std::min(maxCopySize, curMemcpyParam.size);

        // if in memset mode, use a given value as the src
        const uint64_t src         = curMemsetMode ? overrideMemsetVal : curMemcpyParam.src;
        pdmaBatchParams[batchSize] = {.src = src, .dst = curMemcpyParam.dst, .size = itemSize};

        curMemcpyParam.size -= itemSize;
        curMemcpyParam.dst += itemSize;
        if (curMemcpyParam.src)
        {
            curMemcpyParam.src += itemSize;
        }
        batchSize++;
        bool shouldAddBatch = batchSize == m_schedPdmaCommandsTransferMaxParamCount;
        if (curMemcpyParam.size == 0)
        {
            pdmaIndex++;
            if (pdmaIndex < memcpyParams.size())
            {
                curMemcpyParam = memcpyParams[pdmaIndex];
                // all pdmas in a batch should have the same memset value
                // if the current command has a different memset value, send the current collected batch
                if (curMemsetMode != (curMemcpyParam.src == 0))
                {
                    shouldAddBatch = true;
                }
            }
            else
            {
                shouldAddBatch = true;
            }
        }

        if (shouldAddBatch)
        {
            // check if the chunk reached limit, or the current command is the last
            const bool lastBatchInChunk     = ((batchCounter + 1) % m_maxBatchesInChunk == 0) && (batchCounter > 0);
            const bool lastMemcpyItem       = pdmaIndex >= memcpyParams.size();
            const bool useLastBatchSendConf = lastBatchInChunk || lastMemcpyItem;

            // last batch in chunk and last command should be sent
            if (useLastBatchSendConf)
            {
                if (lastBatchInChunk &&
                    !lastMemcpyItem)  // if (lastMemcpyItem), no need to add, it is added after the loop
                {
                    doneChunkOfCommands(isUserRequest, longSo);
                    LOG_TRACE(SYN_PROGRESS, "{:20} : {:>8x} : {:>8x} : {}/{}",
                             m_name,
                             longSo.m_index,
                             longSo.m_targetValue,
                             HLLOG_FUNC,
                             __LINE__);
                }
            }

            if (lastMemcpyItem)
            {
                status = memcopyImpl(resourceType,
                                     pdmaBatchParams,
                                     batchSize,
                                     sendLastPdmaCommand,
                                     apiId,
                                     curMemsetMode,
                                     sendUnfence,
                                     cgIndex,
                                     memcopySyncInfo);
            }
            else if (lastBatchInChunk)
            {
                status = memcopyImpl(resourceType,
                                     pdmaBatchParams,
                                     batchSize,
                                     sendLastPdmaCommand,
                                     apiId,
                                     curMemsetMode,
                                     false /* sendUnfence */,
                                     m_pScalCompletionGroup->getIndexInScheduler() /* completionGroupIndex */,
                                     memcopySyncInfo);
            }
            else
            {
                status = memcopyImpl(resourceType,
                                     pdmaBatchParams,
                                     batchSize,
                                     false /* send */,
                                     apiId,
                                     curMemsetMode,
                                     false /* sendUnfence */,
                                     MAX_COMP_SYNC_GROUP_COUNT /* completionGroupIndex */,
                                     memcopySyncInfo);
            }

            if (status != synSuccess)
            {
                return status;
            }

            batchSize = 0;
            batchCounter++;
        }
        curMemsetMode = curMemcpyParam.src == 0;
    }

    if (memcopySyncInfo.m_pdmaSyncMechanism == PDMA_TX_SYNC_MECH_LONG_SO)
    {
        doneChunkOfCommands(isUserRequest, longSo);
        LOG_TRACE(SYN_PROGRESS, SCAL_PROGRESS_FMT,
                 m_name,
                 longSo.m_index,
                 longSo.m_targetValue,
                 HLLOG_FUNC,
                 __LINE__);
    }

    return synSuccess;
}

/**
 * add a zero-sized pdma
 * @return status
 */
synStatus ScalStreamCopySchedulerMode::addEmptyPdmaPacket()
{
    synStatus status = std::visit(
        [this](auto pkts)
        {
            using T = decltype(pkts);

            typename T::sched_arc_pdma_commands_params_t pdmaCommandsParams[1] = {{}};

            return addPdmaBatchTransfer(getResourceType(),
                                        (const void*)pdmaCommandsParams,
                                        1,
                                        true,
                                        0,
                                        true,
                                        0,
                                        0,
                                        m_pScalCompletionGroup->getIndexInScheduler());
        },
        m_gxPackets);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: addPdmaBatchTransfer returned status {}", HLLOG_FUNC, status);
    }

    return synSuccess;
}
bool ScalStreamCopySchedulerMode::retrievStreamInfo()
{
    // get info on stream (need alignment (in submit), align)
    scal_stream_info_t streamInfo;  // get scheduler info
    ScalRtn rc = scal_stream_get_info(m_streamHndl, &streamInfo);
    LOG_SCALSTRM_INFO("get streamInfo for {:x} {} index {} rc {} command_alignment {:x}",
                      TO64(m_streamHndl),
                      streamInfo.name,
                      streamInfo.index,
                      rc,
                      streamInfo.command_alignment);
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to get streamInfo rc {}", rc);
        return false;
    }

    m_cmdAlign        = streamInfo.command_alignment;
    m_submitAlign     = streamInfo.submission_alignment;
    m_schedulerHandle = streamInfo.scheduler_handle;

    uint32_t ccbSize      = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
    uint32_t ccbChunkSize = ccbSize / GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();

    HB_ASSERT(ccbSize % m_cmdAlign == 0,
              "hostCyclicBufferSize not alligned to commands"); // static assert
    HB_ASSERT(ccbChunkSize % m_cmdAlign == 0,
              "cyclicChunkSize not alligned to commands");  // static assert

    return true;
}

/**
 * add a barrier, or zero-size pdma-cmd (depending on stream type).
 * used when the last added command was 'wait'
 * *** NOTICE: before calling, the 'm_userOpLock' mutex should be locked ***
 * @return status
 */
synStatus ScalStreamCopySchedulerMode::addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo)
{
    synStatus status = addEmptyPdma(rLongSo);
    if (status != synSuccess)
    {
        LOG_SCALSTRM_ERR("failed to add an empty memcopy");
    }

    return status;
}