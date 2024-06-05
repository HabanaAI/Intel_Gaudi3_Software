#include "scal_stream_base.hpp"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "dfa_defines.hpp"
#include "scal_memory_pool.hpp"
#include "log_manager.h"
#include "synapse_profiler_api.hpp"

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

ScalStreamBase::ScalStreamBase(const ScalStreamCtorInfoBase* pScalStreamCtorInfo)
: m_streamHndl(nullptr),
  m_cmdAlign(0),
  m_submitAlign(0),
  m_consecutiveWaitCommands(0),
  m_submitStatPoint(pScalStreamCtorInfo->resourceType == ResourceStreamType::COMPUTE ?
                        globalStatPointsEnum::scalSubmitCompute : globalStatPointsEnum::scalSubmit),
  m_prevCmdIsWait(false),
  m_pScalCompletionGroup(pScalStreamCtorInfo->pScalCompletionGroup),
  m_name(pScalStreamCtorInfo->name),
  m_devStreamInfo(*pScalStreamCtorInfo->devStreamInfo),
  m_devHndl(pScalStreamCtorInfo->devHndl),
  m_streamIdx(pScalStreamCtorInfo->streamIdx),
  m_mpHostShared(pScalStreamCtorInfo->mpHostShared),
  m_ctrlBuffHndl(nullptr),
  m_cmdBufferInfo {},
  m_fenceId(pScalStreamCtorInfo->fenceId),
  m_fenceIdForCompute(pScalStreamCtorInfo->fenceIdForCompute),
  m_resourceType(pScalStreamCtorInfo->resourceType)
{
    LOG_INFO(SYN_STREAM,
                  "scal stream {} syncMonitorId {} fenceId {} streamIdx {}",
                  m_name,
                  pScalStreamCtorInfo->syncMonitorId,
                  getFenceId(),
                  getStreamIndex());

    switch (pScalStreamCtorInfo->devType)
    {
        case synDeviceGaudi2:
            m_gxPackets = G2Packets();
            break;

        case synDeviceGaudi3:
            m_gxPackets = G3Packets();
            break;

        default:
            LOG_ERR_T(SYN_STREAM, "unsupported dev {}", pScalStreamCtorInfo->devType);
            HB_ASSERT(0, "bad device type");
    }
}

ScalStreamBase::~ScalStreamBase()
{
    releaseResources();
}

/*
 ***************************************************************************************************
 *   @brief init() init the stream. If fails, release all resources.
 *
 *   @param  None
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamBase::init()
{
    synStatus status = initInternal();
    if (status != synSuccess)
    {
        releaseResources();
        return status;
    }

    return status;
}

/*
 ***************************************************************************************************
 *   @brief releaseResources() release stream resources (before deleting it)
 *
 *   @param  None
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamBase::releaseResources()
{
    LOG_SCALSTRM_INFO("release all for {:x}", TO64(this));

    if (m_ctrlBuffHndl)
    {
        ScalRtn rc = scal_free_buffer(m_ctrlBuffHndl);
        LOG_SCALSTRM_INFO("release m_ctrlBuffHndl {:x} rc {}", TO64(m_ctrlBuffHndl), rc);
        m_ctrlBuffHndl = nullptr;
        if (rc != SCAL_SUCCESS)
        {
            LOG_SCALSTRM_CRITICAL("fail to release rc {}", rc);
            return synFail;
        }
    }

    return synSuccess;
}

void ScalStreamBase::longSoRecord(bool isUserReq, ScalLongSyncObject& rLongSo) const
{
    m_pScalCompletionGroup->longSoRecord(isUserReq, rLongSo);
}

synStatus ScalStreamBase::eventRecord(bool isUserReq, ScalEvent& scalEvent) const
{
    return m_pScalCompletionGroup->eventRecord(isUserReq, scalEvent);
}

synStatus ScalStreamBase::longSoQuery(const ScalLongSyncObject& rLongSo, bool alwaysWaitForInterrupt) const
{
    return m_pScalCompletionGroup->longSoWait(rLongSo, 0, alwaysWaitForInterrupt);
}

synStatus ScalStreamBase::longSoWaitForLast(bool isUserReq, uint64_t timeoutMicroSec, const char* caller) const
{
    LOG_TRACE(SYN_PROGRESS, "{:20} : {:>8x} : {:>8x} : {}/{}",
             m_name,
             m_pScalCompletionGroup->getLastTarget(isUserReq).m_index,
             m_pScalCompletionGroup->getLastTarget(isUserReq).m_targetValue,
             caller,
             __LINE__);

    return m_pScalCompletionGroup->longSoWaitForLast(isUserReq, timeoutMicroSec);
}

synStatus ScalStreamBase::longSoWaitOnDevice(const ScalLongSyncObject& rLongSo, bool isUserReq)
{
    MonitorAddressesType addr;
    MonitorValuesType    value;
    uint8_t              numRegs = 0;
    synStatus            status;
    const uint32_t       fenceTarget = 1;

    // we are currently adding a 'wait' cmd
    if (isUserReq)
    {
        m_prevCmdIsWait = true;
    }

    // arm the monitor for specific target value
    getMonitor()->getArmRegsForLongSO(rLongSo, m_fenceId, numRegs, addr, value);

    // write the config + payload + arm data
    for (unsigned i = 0; i < numRegs; i++)
    {
        status = addLbwWrite(addr[i], value[i], false, false, true);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "Failed to LbWrite for event {}", status);
            return status;
        }
    }

    // if isUserReq, we have to increment PI (send = true)
    // for launch flow, no need to increment PI (we assume caller will handle it)
    status = addStreamFenceWait(fenceTarget, isUserReq, false);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "sendMemFence failed with status {}", status);
        return status;
    }

    LOG_TRACE(SYN_PROGRESS, SCAL_PROGRESS_FMT,
             m_name,
             rLongSo.m_index,
             rLongSo.m_targetValue,
             HLLOG_FUNC,
             __LINE__);

    LOG_INFO(SYN_STREAM,
             "{}: stream {} longSo (index {}, value {:#x}) fence {}, isUserReq {} targetLongSo {:#x}",
             HLLOG_FUNC,
             m_name,
             rLongSo.m_index,
             rLongSo.m_targetValue,
             fenceTarget,
             isUserReq,
             getCompletionTarget());

    m_consecutiveWaitCommands++;

    if (getStreamCyclicBuffer()->isFirstJobInChunk() && isUserReq)
    {
        ScalLongSyncObject longSo;
        status = addBarrierOrEmptyPdma(longSo);
        if (status != synSuccess)
        {
            return status;
        }
    }

    return synSuccess;
}

ScalLongSyncObject ScalStreamBase::getIncrementedLongSo(bool isUserReq, uint64_t targetOffset /* = 1*/)
{
    ScalLongSyncObject longSo = m_pScalCompletionGroup->getIncrementedLongSo(isUserReq, targetOffset);

    // a command which isn't wait was submitted to the stream. Verify it is not 0 and user-req (only when it is
    // user request we know for sure that both user-req and m_completionTarget were incremented)
    if ((targetOffset > 0) && isUserReq)
    {
        m_prevCmdIsWait = false;
        m_consecutiveWaitCommands = 0;
    }
    return longSo;
}

ScalLongSyncObject ScalStreamBase::getTargetLongSo(uint64_t targetOffset) const
{
    return m_pScalCompletionGroup->getTargetLongSo(targetOffset);
}
/*
 ***************************************************************************************************
 *   @brief longSoWait() wait for a target value with a timeout
 *
 *   @param  target          - the value to wait for
     @param  timeoutMicroSec - timeout
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus
ScalStreamBase::longSoWait(const ScalLongSyncObject& rLongSo, uint64_t timeoutMicroSec, const char* caller) const
{
    LOG_TRACE(SYN_PROGRESS, "{:20} : {:>8x} : {:>8x} : {}/{}",
             m_name,
             rLongSo.m_index,
             rLongSo.m_targetValue,
             caller,
             __LINE__);

    return m_pScalCompletionGroup->longSoWait(rLongSo, timeoutMicroSec);
}

synStatus ScalStreamBase::addStreamFenceWait(uint32_t target, bool isUserReq, bool isInternalComputeSync)
{
    FenceIdType fenceId;
    if (isInternalComputeSync)
    {
        fenceId = getFenceIdForCompute();
    }
    else
    {
        fenceId = getFenceId();
    }
    return addFenceWait(target, fenceId, isUserReq, false /* isGlobal */);
}

void ScalStreamBase::dfaDumpScalStream()
{
    scal_stream_info_t streamInfo;  // get scheduler info
    int rc = scal_stream_get_info(m_streamHndl, &streamInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEV_FAIL, "Could not get streamInfo rc {}", rc);
    }

    scal_buffer_info_t scalBuffInfo;
    rc = scal_buffer_get_info(streamInfo.control_core_buffer, &scalBuffInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEV_FAIL, "Could not get scalBuffInfo rc {}", rc);
    }

    unsigned    schedulerIdx  = 0;
    std::string schedulerName = getSchedulerInfo(schedulerIdx);

    uint32_t ccbSize = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
    std::string out =
        fmt::format("\n#ccb stream-name {} scheduler-name {} scheduler-idx {} stream-idx {} CCB info: {} size {:x}\n",
                    m_name,
                    schedulerName,
                    schedulerIdx,
                    streamInfo.index,
                    getStreamCyclicBuffer()->dfaInfo(),
                    ccbSize);

    out += fmt::format("#registers {:x} {:x}\n", scalBuffInfo.device_address, ccbSize);

    for (int i = 0; i < ccbSize / sizeof(uint32_t); i += REGS_PER_LINE)
    {
        out += fmt::format("{:016x}: ", i * sizeof(uint32_t) + scalBuffInfo.device_address);

        for (int j = 0; j < REGS_PER_LINE; j++)
        {
            out += fmt::format("{:08x} ", static_cast<uint32_t*>(m_cmdBufferInfo.host_address)[i + j]);
        }

        if (i != (ccbSize - REGS_PER_LINE))
        {
            out += "\n";
        }
    }

    LOG_INFO(SYN_DEV_FAIL, "{}", out);
}

TdrRtn ScalStreamBase::tdr(TdrType tdrType)
{
    return m_pScalCompletionGroup->tdr(tdrType);
}

synStatus ScalStreamBase::getStreamInfo(std::string& info, uint64_t& devLongSo)
{
    scal_completion_group_infoV2_t cgInfo;

    synStatus status = m_pScalCompletionGroup->getCurrentCgInfo(cgInfo);
    if (status != synSuccess)
    {
        return synFail;
    }

    info = fmt::format(FMT_COMPILE("stream name: {},  CCB info: {},   fenceId: 0x{:x},   longSoIdx: 0x{:x},   "
                       "(longSo, sent2eng, in_ccb): ({:#x}, {:#x}, {:#x}), timeout {}"),
                       m_name,
                       getStreamCyclicBuffer()->dfaInfo(),
                       getFenceId(),
                       cgInfo.long_so_index,
                       cgInfo.current_value,
                       cgInfo.tdr_value,
                       m_pScalCompletionGroup->getCompletionTarget(),
                       cgInfo.timeoutUs);

    devLongSo = cgInfo.current_value;

    return synSuccess;
}

bool ScalStreamBase::getStaticMonitorPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData) const
{
    return getMonitor()->getPayloadInfo(payloadAddress, payloadData, getFenceIdForCompute());
}

synStatus ScalStreamBase::initStaticMonitor()
{
    // we assume all cgs have the same dcore and scheduler id
    const scal_completion_group_infoV2_t cgInfo = m_pScalCompletionGroup->getCgInfo();

    scal_stream_info_t streamInfo;  // get scheduler info
    int rc = scal_stream_get_info(m_streamHndl, &streamInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_DEV_FAIL, "Could not get streamInfo rc {}", rc);

        return synFail;
    }

    synStatus status = getMonitor()->init(cgInfo, &streamInfo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "Failed to init scal monitor");
        return synFail;
    }

    MonitorAddressesType addr;
    MonitorValuesType    value;
    uint8_t              numRegs = 0;

    if (!getMonitor()->getConfigRegsForLongSO(getFenceId(), numRegs, addr, value))
    {
        LOG_ERR(SYN_STREAM, "Failed to get configuration for scal monitor");
        return synFail;
    }

    for (unsigned i = 0; i < numRegs; i++)
    {
        bool send = i == numRegs - 1;

        // LOG_ERR(SYN_STREAM, "Pre addLbwWrite addr 0x{:x} value 0x{:x}", (uint32_t)addr[i], value[i]);
        status = addLbwWrite(addr[i], value[i], false, send, true);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "Failed to LbWrite for monitor, {}", status);
            return synFail;
        }
    }

    return status;
}

bool ScalStreamBase::retrievStreamHandle()
{
    ScalRtn rc = scal_get_stream_handle_by_name(getDeviceHandle(), m_name.c_str(), &m_streamHndl);
    LOG_SCALSTRM_INFO("get stream handle rc {} m_streamHndl {:x} name {}", rc, TO64(m_streamHndl), m_name);
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to get stream handle rc {}", rc);
        return false;
    }

    return true;
}

bool ScalStreamBase::allocateAndSetCommandsBuffer()
{
    // Allocate buffer on host shared for stream commands
    uint32_t ccbSize = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
    ScalRtn rc = m_mpHostShared.allocateDeviceMemory(ccbSize, m_ctrlBuffHndl);
    LOG_SCALSTRM_INFO("get ctrl buffer size {:x} from host_shared hbm_shared rc {} m_ctrlBuffHndl {:x}",
                      ccbSize,
                      rc,
                      TO64(m_ctrlBuffHndl));
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to get pool handle for hbm_shared rc {}", rc);
        return false;
    }

    // bind stream and ctrl handle
    rc = scal_stream_set_commands_buffer(m_streamHndl, m_ctrlBuffHndl);  // bind stream and ctrl handle
    LOG_SCALSTRM_INFO("bind stream {:x} and ctrlBuff {:x} rc {}", TO64(m_streamHndl), TO64(m_ctrlBuffHndl), rc);
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to set stream priority rc {}", rc);
        return false;
    }

    return true;
}

bool ScalStreamBase::retrievCommandsBufferInfo()
{
    ScalRtn rc = scal_buffer_get_info(m_ctrlBuffHndl, &m_cmdBufferInfo);  // info on the device/host addr and more
    LOG_SCALSTRM_INFO("get m_ctrlBuffHndl for {:x} rc {} host addr {:x}",
                      TO64(m_ctrlBuffHndl),
                      rc,
                      TO64(m_cmdBufferInfo.host_address));
    if (rc != SCAL_SUCCESS)
    {
        LOG_SCALSTRM_ERR("fail to get m_cmdBufferInfo rc {}", rc);
        return false;
    }

    return true;
}

/**
 * add a zero-sized pdma
 * @return status
 */
synStatus ScalStreamBase::addEmptyPdma(ScalLongSyncObject& rLongSo)
{
    synStatus status = addEmptyPdmaPacket();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: addEmptyPdmaPacket returned status {}", HLLOG_FUNC, status);
        return status;
    }

    status = doneChunkOfCommands(true, rLongSo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: doneChunkOfCommands returned status {}", HLLOG_FUNC, status);
        return status;
    }

    LOG_TRACE(SYN_PROGRESS, SCAL_PROGRESS_FMT,
             m_name,
             rLongSo.m_index,
             rLongSo.m_targetValue,
             HLLOG_FUNC,
             __LINE__);

    return synSuccess;
}

synStatus ScalStreamBase::doneChunkOfCommands(bool isUserReq, ScalLongSyncObject& rLongSo)
{
    rLongSo = getIncrementedLongSo(isUserReq);
    getStreamCyclicBuffer()->doneChunkOfCommands(rLongSo);

    return synSuccess;
}

void ScalStreamBase::printCgTdrInfo(bool tdr) const
{
    const CgTdrInfo* pTdrInfo = nullptr;
    if (!m_pScalCompletionGroup->getCgTdrInfo(pTdrInfo) || (pTdrInfo == nullptr))
    {
        LOG_ERR(SYN_DEV_FAIL, "stream {}: Failed to get CG TDR info", m_name);
        return;
    }

    uint64_t sinceArm = TimeTools::timeFromUs(pTdrInfo->armTime);

    LOG_DEBUG(SYN_DEV_FAIL,
              "Stream: {}, TDR status: {},   armed: {},   since armed: {}us",
              m_name,
              tdr ?  "* TDR triggered *" : "TDR not triggered",
              pTdrInfo->armed,
              sinceArm);
}

enum PdmaDirCtx ScalStreamCopyBase::getDir(ResourceStreamType resourceType)
{
    enum PdmaDirCtx dir;
    if ((resourceType == ResourceStreamType::SYNAPSE_DMA_DOWN) || (resourceType == ResourceStreamType::USER_DMA_DOWN))
    {
        dir = PdmaDirCtx::DOWN;
    }
    else if ((resourceType == ResourceStreamType::SYNAPSE_DMA_UP) || (resourceType == ResourceStreamType::USER_DMA_UP))
    {
        dir = PdmaDirCtx::UP;
    }
    else
    {
        dir = PdmaDirCtx::DEV2DEV;
    }

    return dir;
}

internalStreamType ScalStreamCopyBase::getInternalStreamType(ResourceStreamType resourceType)
{
    internalStreamType type;

    switch (resourceType)
    {
        case ResourceStreamType::USER_DMA_UP:
        {
            type = INTERNAL_STREAM_TYPE_DMA_UP;
            break;
        }
        case ResourceStreamType::USER_DMA_DOWN:
        {
            type = INTERNAL_STREAM_TYPE_DMA_DOWN_USER;
            break;
        }
        case ResourceStreamType::USER_DEV_TO_DEV:
        {
            type = INTERNAL_STREAM_TYPE_DEV_TO_DEV;
            break;
        }
        case ResourceStreamType::SYNAPSE_DMA_UP:
        {
            type = INTERNAL_STREAM_TYPE_DMA_UP;
            break;
        }
        case ResourceStreamType::SYNAPSE_DMA_DOWN:
        {
            type = INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE;
            break;
        }
        case ResourceStreamType::SYNAPSE_DEV_TO_DEV:
        {
            type = INTERNAL_STREAM_TYPE_DEV_TO_DEV;
            break;
        }
        case ResourceStreamType::COMPUTE:
        {
            type = INTERNAL_STREAM_TYPE_COMPUTE;
            break;
        }
        default:
        {
            type = INTERNAL_STREAM_TYPE_NUM;
        }
    }

    return type;
}

uint8_t ScalStreamCopyBase::getContextId(enum PdmaDirCtx dir, ResourceStreamType resourceType, uint32_t index)
{
    const internalStreamType type = getInternalStreamType(resourceType);
    return (((((uint8_t)dir) & ContextEncoding::DIR_MASK) << ContextEncoding::DIR_OFFSET) |
            ((((uint8_t)type) & ContextEncoding::TYPE_MASK) << ContextEncoding::TYPE_OFFSET) |
            ((((uint8_t)index) & ContextEncoding::STREAM_MASK) << ContextEncoding::STREAM_OFFSET));
}