#include "scal_stream_copy_direct_mode.hpp"

#include "dfa_defines.hpp"
#include "defs.h"
#include "scal_memory_pool.hpp"
#include "synapse_common_types.h"
#include "habana_global_conf_runtime.h"

#include "log_manager.h"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "runtime/scal/gaudi3/direct_mode_packets/pqm_packets.hpp"

#include "spdlog/fmt/bundled/core.h"
#include "utils.h"
// specs
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"  // block_sob_objs
#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"
#include "gaudi3/asic_reg_structs/pdma_ctx_a_regs.h"

#define LOG_SCALSTRM_TRACE(msg, ...)                                                                                   \
    LOG_TRACE(SYN_DM_STREAM, "stream {}:" msg, getName(), ##__VA_ARGS__)
#define LOG_SCALSTRM_DEBUG(msg, ...)                                                                                   \
    LOG_DEBUG(SYN_DM_STREAM, "stream {}:" msg, getName(), ##__VA_ARGS__)
#define LOG_SCALSTRM_INFO(msg, ...)                                                                                    \
    LOG_INFO(SYN_DM_STREAM, "stream {}:" msg, getName(), ##__VA_ARGS__)
#define LOG_SCALSTRM_ERR(msg, ...)                                                                                     \
    LOG_ERR(SYN_DM_STREAM, "stream {}:" msg, getName(), ##__VA_ARGS__)
#define LOG_SCALSTRM_CRITICAL(msg, ...)                                                                                \
    LOG_CRITICAL(SYN_DM_STREAM, "stream {}:" msg, getName(), ##__VA_ARGS__)

using namespace common;

ScalStreamCopyDirectMode::ScalStreamCopyDirectMode(const ScalStreamCtorInfoBase* pScalStreamCtorInfo)
: ScalStreamCopyBase(pScalStreamCtorInfo),
  m_maxLinPdmasInChunk(0),
  m_scalMonitor(pScalStreamCtorInfo->syncMonitorId,
                *pScalStreamCtorInfo->deviceInfoInterface,
                pScalStreamCtorInfo->devHndl),
  m_streamCyclicBuffer(getName()),
  m_apiId(std::numeric_limits<uint8_t>::max())
{
    LOG_INFO(SYN_DM_STREAM,
                  "scal stream {} syncMonitorId {} streamIdx {}",
                  getName(),
                  pScalStreamCtorInfo->syncMonitorId,
                  getStreamIndex());
}

ScalStreamCopyDirectMode::~ScalStreamCopyDirectMode()
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
synStatus ScalStreamCopyDirectMode::addLbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool send, bool isInSyncMgr)
{
    if (block_stream)
    {
        LOG_SCALSTRM_ERR("block-stream mode is not supported");
        return synFail;
    }

    if (isInSyncMgr)
    {
        return addPqmMsgShort(data, dst_addr, send);
    }

    return addPqmMsgLong(data, dst_addr, send);
}

// pqm packets functions
synStatus ScalStreamCopyDirectMode::addPqmMsgLong(uint32_t value, uint64_t address, bool send)
{
    return addPacket<pqm::MsgLong>(send, value, address);
}

synStatus ScalStreamCopyDirectMode::addPqmMsgShort(uint32_t value, uint64_t address, bool send)
{
    uint64_t smId = m_scalMonitor.getSyncMgrId();

    if (m_pqmMsgShortBaseAddrs.count(smId))
    {
        uint64_t smBaseAddr            = m_pqmMsgShortBaseAddrs[smId].first;
        unsigned spdmaMsgBaseAddrIndex = m_pqmMsgShortBaseAddrs[smId].second;

        return addPacket<pqm::MsgShort>(send, value, spdmaMsgBaseAddrIndex, address - smBaseAddr);
    }

    LOG_SCALSTRM_ERR("Failed to add a PQM msgShort, requested address [{:x}] is not in the range of any configured PQM PDMA base addresses",
                     address);
    return synFail;
}

synStatus ScalStreamCopyDirectMode::addPqmFence(uint8_t fenceId, uint32_t decVal, uint32_t targetVal, bool send)
{
    return addPacket<pqm::Fence>(send, fenceId, decVal, targetVal);
}

synStatus ScalStreamCopyDirectMode::addPqmNopCmd(bool send)
{
    return addPacket<pqm::Nop>(send);
}

synStatus ScalStreamCopyDirectMode::addPqmLinPdmaCmd(uint64_t           src,
                                                     uint64_t           dst,
                                                     uint32_t           size,
                                                     bool               bMemset,
                                                     PdmaDir            direction,
                                                     LinPdmaBarrierMode barrierMode,
                                                     uint64_t           barrierAddress,
                                                     uint32_t           barrierData,
                                                     uint32_t           fenceDecVal,
                                                     uint32_t           fenceTarget,
                                                     uint32_t           fenceId,
                                                     bool               send,
                                                     uint8_t            apiId)
{
    synStatus status = synFail;
    const scal_completion_group_infoV2_t& rcgInfo = m_pScalCompletionGroup->getCgInfo();
    // if barrierMode is INTERNAL, longSO is increased
    if(rcgInfo.tdr_enabled && (barrierMode == LinPdmaBarrierMode::INTERNAL))
    {
        // get tdr sob device address and create a pqm msgLong to inc it before starting the lindma
        // scal cfg (Scal_Gaudi3::parseTdrCompletionQueues) ensures that tdr sos is in the same sm as cq.monitorsPool->smIndex
        // which is stored in cgInfo.sm_base_addr

        uint64_t watch_dog_sob_addr = rcgInfo.sm_base_addr + varoffsetof(gaudi3::block_sob_objs, sob_obj_0[rcgInfo.tdr_sos]);
        uint32_t value = 1 | (1UL << 31);// inc by 1
        status = addPqmMsgLong(value, watch_dog_sob_addr, false);
        if (status != synSuccess)
        {
            return status;
        }
    }

    if (GCFG_ENABLE_PROFILER.value())
    {
        if (apiId != m_apiId)
        {
            const ResourceStreamType resourceType = getResourceType();
            const enum PdmaDirCtx    dir          = getDir(resourceType);
            const ResourceStreamType realType     = isComputeStream() ? ResourceStreamType::COMPUTE : resourceType;
            const uint8_t            ctxId        = getContextId(dir, realType, getStreamIndex());

            const uint8_t  fragNum = 0x0;
            const uint32_t value =
                (uint32_t)ctxId + (((uint32_t)apiId & 0b11111) << 8) + (((uint32_t)fragNum & 0b111) << 13);

            const uint32_t regOffset =
                varoffsetof(gaudi3::block_pdma_ch_a, ctx) + varoffsetof(gaudi3::block_pdma_ctx_a, idx);
            LOG_SCALSTRM_INFO("ChWreg32 regOffset {:X} value {:X}", regOffset, value);

            status = addPacketCommon<pqm::ChWreg32>(false, pqm::ChWreg32::getSize(), regOffset, value);
            if (status != synSuccess)
            {
                return status;
            }
            m_apiId = apiId;
        }
    }

    status = addPacketCommon<pqm::LinPdma>(send, pqm::LinPdma::getSize(barrierMode != LinPdmaBarrierMode::DISABLED),
                                         src, dst, size, bMemset, direction, barrierMode, barrierAddress, barrierData,
                                         fenceDecVal, fenceTarget, fenceId);
    return status;
}

synStatus ScalStreamCopyDirectMode::addFenceWait(uint32_t target, FenceIdType fenceId, bool send, bool isGlobal)
{
    return addPacket<pqm::Fence>(send, fenceId, target, target);
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
synStatus ScalStreamCopyDirectMode::initInternal()
{
    if (!retrievStreamHandle())
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

    if (!retrievCommandsBufferInfo())
    {
        return synFail;
    }
    const uint16_t submissionQueueAlignment = 8;
    if (!checkSubmissionQueueAlignment(submissionQueueAlignment))
    {
        LOG_SCALSTRM_ERR("Submission-queue has invalid alignment");
        return synFail;
    }

    m_streamCyclicBuffer.init(m_pScalCompletionGroup,
                              getCommandBufferHostAddress(),
                              TO64(m_streamHndl),
                              m_cmdAlign);

    if (!initPqmMsgShortBaseAddrs())
    {
        return synFail;
    }

    ScalRtn rc = initStaticMonitor();
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("failed in initStaticMonitor rc {}", rc);
        return synFail;
    }

    // calculate how many Lin-Pdmas we can submit before signaling
    uint32_t maxLinPdmaSize = pqm::LinPdma::getSize(true);

    if (m_pScalCompletionGroup->getCgInfo().tdr_enabled)
    {
        maxLinPdmaSize += pqm::MsgLong::getSize();
    }

    if (GCFG_ENABLE_PROFILER.value())
    {
        maxLinPdmaSize += pqm::ChWreg32::getSize();
    }

    // How many in a chunk
    uint16_t pdmasInAlignChunk            = m_cmdAlign / maxLinPdmaSize;
    uint32_t ccbChunkSize                 = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024 / GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
    uint16_t cmdAlignChunksInCyclicBuffer = ccbChunkSize / m_cmdAlign;
    m_maxLinPdmasInChunk = cmdAlignChunksInCyclicBuffer * pdmasInAlignChunk;

    LOG_SCALSTRM_INFO("Need to add signal_to_cg every {:x} pdma-s. pdmaInAlignChunk {:x}, cmdAlignChunksInCyclicBuffer {:x}",
                      m_maxLinPdmasInChunk,
                      pdmasInAlignChunk,
                      cmdAlignChunksInCyclicBuffer);

    return synSuccess;
}

synStatus ScalStreamCopyDirectMode::memcopy(ResourceStreamType           resourceType,
                                            const internalMemcopyParams& memcpyParams,
                                            bool                         isUserRequest,
                                            bool                         send,
                                            uint8_t                      apiId,
                                            ScalLongSyncObject&          longSo,
                                            uint64_t                     overrideMemsetVal,
                                            MemcopySyncInfo&             memcopySyncInfo)
{
    LOG_TRACE_T(SYN_DM_STREAM,
                "{}: resourceType={} isUserRequest={} send={} overrideMemsetVal={},"
                " pdmaSyncMechanism={} consecutiveWaitCommands={}",
                HLLOG_FUNC,
                resourceType,
                isUserRequest,
                send,
                overrideMemsetVal,
                memcopySyncInfo.m_pdmaSyncMechanism,
                m_consecutiveWaitCommands);

    if (memcpyParams.size() == 0)
    {
        LOG_DEBUG(SYN_DM_STREAM, "{}: No memcopy params given", HLLOG_FUNC);
        return synSuccess;
    }

    bool sendUnfence         = false;
    bool sendLastPdmaCommand = send;

    synStatus          status      = synSuccess;
    LinPdmaBarrierMode barrierMode = LinPdmaBarrierMode::DISABLED;

    switch (memcopySyncInfo.m_pdmaSyncMechanism)
    {
        case PDMA_TX_SYNC_MECH_LONG_SO:
            barrierMode = LinPdmaBarrierMode::INTERNAL;
            break;

        case PDMA_TX_SYNC_FENCE_ONLY:
            barrierMode = LinPdmaBarrierMode::EXTERNAL;
            sendUnfence = true;
            break;

        default:
            break;
    }

    uint64_t cqLongSoAddress  = getCqLongSoAddress();
    uint32_t cqLongSoIncValue = getCqLongSoValue();

    PdmaDir direction =
        ((resourceType == ResourceStreamType::SYNAPSE_DMA_UP) ||
         (resourceType == ResourceStreamType::USER_DMA_UP)) ?
        PdmaDir::DEVICE_TO_HOST : PdmaDir::HOST_TO_DEVICE;

    bool isDmaDownSynapse = (resourceType == ResourceStreamType::SYNAPSE_DMA_DOWN);

    unsigned                  memcpyParamIndex = 0;
    unsigned                  linPdmaCounter   = 0;
    internalMemcopyParamEntry curMemcpyParam   = memcpyParams[memcpyParamIndex];
    bool                      curMemsetMode    = (curMemcpyParam.src == 0);

    const uint64_t maxCopySize = std::numeric_limits<uint32_t>::max();
    while (memcpyParamIndex < memcpyParams.size())
    {
        // check for invalid input (memset with INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE)
        if (isDmaDownSynapse && curMemsetMode)
        {
            LOG_ERR(SYN_DM_STREAM,
                    "{}: src == 0 (i = {}). memset mode is not supported for {}",
                    HLLOG_FUNC,
                    memcpyParamIndex,
                    resourceType);
            return synInvalidArgument;
        }

        uint32_t itemSize = std::min(maxCopySize, curMemcpyParam.size);

        // if in memset mode, use a given value as the src
        uint64_t src = curMemsetMode ? overrideMemsetVal : curMemcpyParam.src;
        uint64_t dst = curMemcpyParam.dst;

        curMemcpyParam.size -= itemSize;
        curMemcpyParam.dst += itemSize;
        if (curMemcpyParam.src)
        {
            curMemcpyParam.src += itemSize;
        }

        if (curMemcpyParam.size == 0)
        {
            memcpyParamIndex++;
            if (memcpyParamIndex < memcpyParams.size())
            {
                curMemcpyParam = memcpyParams[memcpyParamIndex];
                curMemsetMode  = curMemcpyParam.src == 0;
            }
        }

        {
            linPdmaCounter++;

            // check if the chunk reached limit, or the current command is the last
            const bool lastPdmaInChunk      = ((linPdmaCounter % m_maxLinPdmasInChunk) == 0);
            const bool lastMemcpyItem       = (memcpyParamIndex >= memcpyParams.size());
            const bool useLastBatchSendConf = lastPdmaInChunk || lastMemcpyItem;

            // last Lin-Pdma in chunk and last command should be sent
            if (useLastBatchSendConf)
            {
                // if (lastMemcpyItem), no need to add, it is added after the loop
                if (lastPdmaInChunk && !lastMemcpyItem)
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
                status = addPqmLinPdmaCmd(src,
                                          dst,
                                          itemSize,
                                          curMemsetMode,
                                          direction,
                                          barrierMode,
                                          sendUnfence ? memcopySyncInfo.m_workCompletionAddress : cqLongSoAddress,
                                          sendUnfence ? memcopySyncInfo.m_workCompletionValue : cqLongSoIncValue,
                                          0 /* fenceDecVal */,
                                          0 /* fenceTarget */,
                                          INVALID_FENCE_ID /* fenceId - Disabled */,
                                          sendLastPdmaCommand /* send */,
                                          apiId);
            }
            else if (lastPdmaInChunk)
            {
                status = addPqmLinPdmaCmd(src,
                                          dst,
                                          itemSize,
                                          curMemsetMode,
                                          direction,
                                          LinPdmaBarrierMode::INTERNAL,
                                          cqLongSoAddress,
                                          cqLongSoIncValue,
                                          0 /* fenceDecVal */,
                                          0 /* fenceTarget */,
                                          INVALID_FENCE_ID /* fenceId - Disabled */,
                                          sendLastPdmaCommand /* send */,
                                          apiId);
            }
            else
            {
                status = addPqmLinPdmaCmd(src,
                                          dst,
                                          itemSize,
                                          curMemsetMode,
                                          direction,
                                          LinPdmaBarrierMode::DISABLED,
                                          cqLongSoAddress,
                                          cqLongSoIncValue,
                                          0 /* fenceDecVal */,
                                          0 /* fenceTarget */,
                                          INVALID_FENCE_ID /* fenceId - Disabled */,
                                          false /* send */,
                                          apiId);
            }

            if (status != synSuccess)
            {
                return status;
            }
        }
    }

    switch (memcopySyncInfo.m_pdmaSyncMechanism)
    {
        case PDMA_TX_SYNC_MECH_LONG_SO:
            doneChunkOfCommands(isUserRequest, longSo);
            LOG_TRACE(SYN_PROGRESS, SCAL_PROGRESS_FMT,
                     m_name,
                     longSo.m_index,
                     longSo.m_targetValue,
                     HLLOG_FUNC,
                     __LINE__);
            break;

        default:
            break;
    }

    return synSuccess;
}

/**
 * add a zero-sized pdma
 * @return status
 */
synStatus ScalStreamCopyDirectMode::addEmptyPdmaPacket()
{
    uint64_t barrierAddress = getCqLongSoAddress();
    uint32_t barrierData    = getCqLongSoValue();

    // we don't care about the stream type here - it can be any dma type
    synStatus status = addPqmLinPdmaCmd(0 /* src */,
                                        0 /* dst */,
                                        0 /* size */,
                                        true /* bMemset */,
                                        PdmaDir::HOST_TO_DEVICE,
                                        LinPdmaBarrierMode::INTERNAL,
                                        barrierAddress,
                                        barrierData,
                                        0 /* fenceDecVal - NA */,
                                        0 /* fenceTarget - NA */,
                                        4 /* fenceId - Disabled */,
                                        true /* send */,
                                        0);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DM_STREAM, "{}: addPqmLinPdmaCmd returned status {}", HLLOG_FUNC, status);
    }

    return synSuccess;
}

bool ScalStreamCopyDirectMode::retrievStreamInfo()
{
    // get info on stream (need alignment (in submit), align)
    scal_stream_info_t streamInfo;  // get scheduler info
    ScalRtn rc = scal_stream_get_info(m_streamHndl, &streamInfo);
    LOG_SCALSTRM_INFO("get streamInfo for {:x} {} index {} rc {}",
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

    uint32_t ccbSize      = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
    uint32_t ccbChunkSize = ccbSize / GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
    m_cmdAlign            = ccbChunkSize;

    HB_ASSERT(ccbSize % m_cmdAlign == 0,
              "hostCyclicBufferSize not alligned to commands"); // static assert
    HB_ASSERT(ccbChunkSize % m_cmdAlign == 0,
              "cyclicChunkSize not alligned to commands");  // static assert

    return true;
}

bool ScalStreamCopyDirectMode::initPqmMsgShortBaseAddrs()
{
    const scal_sm_base_addr_tuple_t* smBaseAddrsDb;
    unsigned numAddrs;

    ScalRtn rc = scal_get_used_sm_base_addrs(getDeviceHandle(), &numAddrs, &smBaseAddrsDb);
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to get SM base addr list rc {}", rc);
        return false;
    }

    LOG_SCALSTRM_INFO("get SM base addr list for {} rc {}",
                      m_name,
                      rc);

    for (unsigned i = 0; i < numAddrs; i++)
    {
        unsigned smId              = smBaseAddrsDb[i].smId;
        uint64_t smBaseAddr        = smBaseAddrsDb[i].smBaseAddr;
        unsigned spdmaMsgBaseIndex = smBaseAddrsDb[i].spdmaMsgBaseIndex;

        m_pqmMsgShortBaseAddrs[smId] = {smBaseAddr, spdmaMsgBaseIndex};
    }

    return true;
}

uint64_t ScalStreamCopyDirectMode::getCqLongSoAddress()
{
    return m_pScalCompletionGroup->getLongSoAddress();
}

uint32_t ScalStreamCopyDirectMode::getCqLongSoValue()
{
    union sync_object_update
    {
        struct
        {
            uint32_t sync_value :16;
            uint32_t reserved1  :8;
            uint32_t long_mode  :1;
            uint32_t reserved2  :5;
            uint32_t te         :1;
            uint32_t mode       :1;
        } so_update;
        uint32_t raw;
    };
    sync_object_update syncObjUpdate;
    syncObjUpdate.raw                  = 0;
    syncObjUpdate.so_update.long_mode  = 1; // Long-SO mode
    syncObjUpdate.so_update.mode       = 1; // Increment
    syncObjUpdate.so_update.sync_value = 1; // By 1

    return syncObjUpdate.raw;
}

/**
 * add a barrier, or zero-size pdma-cmd (depending on stream type).
 * used when the last added command was 'wait'
 * *** NOTICE: before calling, the 'm_userOpLock' mutex should be locked ***
 * @return status
 */
synStatus ScalStreamCopyDirectMode::addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo)
{
    synStatus status = addEmptyPdma(rLongSo);
    if (status != synSuccess)
    {
        LOG_SCALSTRM_ERR("failed to add an empty memcopy");
    }

    return status;
}
