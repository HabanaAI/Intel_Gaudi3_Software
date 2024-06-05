#include "device_scal.hpp"

#include <cstdint>
#include <iostream>
#include <memory>

#include "types_exception.h"
#include "defenders.h"
#include "habana_global_conf_runtime.h"
#include "runtime/common/osal/osal.hpp"
#include "profiler_api.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "internal/hccl_internal.h"

#include "runtime/common/recipe/recipe_utils.hpp"
#include "runtime/common/streams/stream.hpp"
#include "runtime/common/streams/stream_job.hpp"

#include "runtime/scal/common/stream_compute_scal.hpp"
#include "runtime/scal/common/stream_copy_scal.hpp"
#include "runtime/scal/common/stream_collective_network_scal.hpp"
#include "runtime/scal/common/stream_wait_for_event_scal.hpp"

#include "runtime/scal/common/entities/scal_completion_group.hpp"
#include "runtime/scal/common/entities/scal_memory_pool.hpp"

#include "syn_event_dispatcher.hpp"

#include "graph_compiler/types.h"
#include "synapse_common_types.h"

#include "runtime/scal/common/coeff_table_configuration_manager.hpp"

#include "runtime/scal/common/entities/scal_stream_compute_interface.hpp"
#include "global_statistics.hpp"

#include <unistd.h>

using namespace common;

//#define DISABLE_SYNC_ON_DEV


static uint32_t get_affinity()
{
    const char * envVarValue = getenv("GAUDI3_SINGLE_DIE_CHIP");
    if (envVarValue)
    {
        return 2;
    }
    return 3;
}
const uint32_t DeviceScal::MAX_AFFINITY_QUEUES_NUM  = get_affinity();

const AffinityCountersArray DeviceScal::s_maxAffinitiesDefault = {MAX_AFFINITY_QUEUES_NUM /* D2H */,
                                                                  MAX_AFFINITY_QUEUES_NUM /* H2D */,
                                                                  MAX_AFFINITY_QUEUES_NUM /* D2D */,
                                                                  MAX_AFFINITY_QUEUES_NUM /* COMPUTE */,
                                                                  MAX_AFFINITY_QUEUES_NUM /* HCL */};
const AffinityCountersArray DeviceScal::s_maxAffinitiesHCLDisable = {MAX_AFFINITY_QUEUES_NUM /* D2H */,
                                                                     MAX_AFFINITY_QUEUES_NUM /* H2D */,
                                                                     MAX_AFFINITY_QUEUES_NUM /* D2D */,
                                                                     MAX_AFFINITY_QUEUES_NUM /* COMPUTE */,
                                                                     0 /* HCL */};

DeviceScal::DeviceScal(synDeviceType deviceType, const DeviceConstructInfo& deviceConstructInfo)
: DeviceCommon(deviceType,
               new DevMemoryAllocScal(&m_scalDev,
                                      deviceType,
                                      deviceConstructInfo.deviceInfo.dramSize,
                                      deviceConstructInfo.deviceInfo.dramBaseAddress),
               deviceConstructInfo,
               true,
               GCFG_INIT_HCCL_ON_ACQUIRE.value() ? s_maxAffinitiesDefault : s_maxAffinitiesHCLDisable),
  m_scalDev(deviceType),
  m_collectiveStreamNum(0),
  m_hclInit(false)
{
    if ((deviceType != synDeviceGaudi2) && (deviceType != synDeviceGaudi3))
    {
        throw SynapseStatusException(fmt::format("{}: Invalid device-type: {}", __FUNCTION__, deviceType), synFail);
    }

}

DeviceScal::~DeviceScal()
{
}

/*
 ***************************************************************************************************
 *   @brief acquire() is called when we acquire a new device
 *
 *   @param  numSyncObj - number of events
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus DeviceScal::_acquire(const uint16_t numSyncObj, common::CoeffTableConf& rCoeffTableConf)
{
    m_scalEventsPool = new ScalEventsPool(m_osalInfo.fd);

    std::string scalCfgFile(GCFG_SCAL_CONFIG_FILE_PATH.value());

    bool EdmaV3Commands = false;
    auto ret = hl_gcfg::getGcfgItemValue("HCL_USE_EDMA_COMMAND_V3");
    if (!ret.has_error() && (ret.value().compare("1") == 0))
    {
        EdmaV3Commands = true;
    }

    if (m_devType == synDeviceType::synDeviceGaudi2 && EdmaV3Commands)
    {
        scalCfgFile = std::string(":/gaudi2/default_edma_v3.json");
    }

    // scalCfgFile empty filename is ok - it means a default one that is inside scal library

    LOG_TRACE_T(SYN_DEVICE, "{} numSyncObj {}", HLLOG_FUNC, numSyncObj);

    synStatus status = m_scalDev.acquire(m_osalInfo.fd, scalCfgFile);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Device acquire failed with rc {}", status);
        return status;
    }

    status = m_hclApiWrapper.provideScal(m_scalDev.getScalHandle());

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE,
                "Device acquire failed on call to HCL layer with scal handle, with rc {}",
                status);
        return status;
    }

    m_devSpecificInfo = m_scalDev.getDevSpecificInfo();

    status = preAllocateMemoryForComputeStreams();
    if (status != synSuccess)
    {
        return status;
    }

    status = startEventFdThread();
    if (status != synSuccess)
    {
        return status;
    }

    if ((GCFG_INIT_HCCL_ON_ACQUIRE.value()) && (hcclInitDevice(0) != hcclSuccess))
    {
        LOG_ERR(SYN_DEVICE, "{}: Failed to initialize HCCL device for device", HLLOG_FUNC);
        return synFail;
    }
    m_hclInit = true;

    status = rCoeffTableConf.submitCoeffTableConfiguration();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Device acquire failed on Coeff table configuration with rc {}", status);
        releasePreAllocatedMem();
        return status;
    }

    status = addStreamAffinities(m_maxAffinities, false);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: addAffinitis failed with status {} ", HLLOG_FUNC, status);
        return status;
    }

    synapse::LogManager::instance().clearLogContext();

    return synSuccess;
}

void DeviceScal::bgWork()
{
    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex, std::defer_lock);
    //  The shared lock will be taken unless a WR lock was taken (acquire, release, create/destroy stream). In these
    //  cases we prefer to skip the bgWork and avoid deadlock on release which took the unique lock and wait on join to
    //  the EFD thread.
    const bool       isLocked = lock.try_lock();
    if (!isLocked)
    {
        // The device is busy/releasing at the moment. Skipping background work.
        return;
    }
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    // run until triggered, notifies the failure and not does not run again
    if (!m_dfa.getStatus().hasError(DfaErrorCode::tdrFailed))
    {
        TdrRtn rtn = streamsTdr(TdrType::CHECK);
        if (rtn.failed)
        {
            DfaExtraInfo dfaExtraInfo = {
                .extraInfo = DfaExtraInfo::DfaExtraInfoMsg { .msg = rtn.msg }};

            notifyDeviceFailure(DfaErrorCode::tdrFailed, dfaExtraInfo);
        }
    }

    // run until triggered, notifies the failure and not does not run again
    if (!m_dfa.getStatus().hasError(DfaErrorCode::scalTdrFailed))
    {
        constexpr int msgSize = 1000;
        char msgBuff[msgSize] {};
        bool failed = scal_bg_workV2(m_scalDev.getScalHandle(), nullptr, msgBuff, msgSize);
        if (failed)
        {
            std::string msgStr(msgBuff);
            DfaExtraInfo dfaExtraInfo = {
                .extraInfo = DfaExtraInfo::DfaExtraInfoMsg { .msg = msgStr }};

            notifyDeviceFailure(DfaErrorCode::scalTdrFailed, dfaExtraInfo);
        }
    }
}

void DeviceScal::debugCheckWorkStatus()
{
    std::shared_lock lock(m_mutex, std::defer_lock);
    // The shared lock will be taken unless a WR lock was taken (acquire, release, create/destroy stream).
    // In these case, we prefer to skip the debug-background-work and avoid deadlock on release
    // which took the unique lock and wait on join to the EFD thread.
    const bool       isLocked = lock.try_lock();
    if (!isLocked)
    {
        // The device is busy/releasing at the moment. Skipping debug background work.
        return;
    }

    scal_debug_background_work(m_scalDev.getScalHandle());
}

synStatus DeviceScal::preAllocateMemoryForComputeStreams()
{
    synStatus status = synSuccess;
    PreAllocatedStreamMemory glblRefBuffer;
    PreAllocatedStreamMemory sharedRefBuffer;

    glblRefBuffer.Size   = getHbmGlblSize();
    sharedRefBuffer.Size = hbmSharedSize;

    // Allocate global device memory for stream
    status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)->allocateDeviceMemory(glblRefBuffer.Size * ScalDev::MaxNumOfComputeStreams, glblRefBuffer.BufHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Device failed pre allocating global memory for compute streams, status: {}",
        status);
        return status;
    }
    status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)
                    ->getDeviceMemoryAddress(glblRefBuffer.BufHandle, glblRefBuffer.AddrCore, glblRefBuffer.AddrDev);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE,
                "Device failed in getDeviceMemoryAddress when pre allocating global memory for compute streams, status: {}",
                status);
        return status;
    }

    uint64_t allocAddrStart = glblRefBuffer.AddrDev;
    uint64_t allocAddrEnd   = glblRefBuffer.AddrDev + (glblRefBuffer.Size * ScalDev::MaxNumOfComputeStreams);
    // If the allocation is crossing the previous 4GB address space, allocate again with padding to avoid memory fragmentation
    if (allocAddrStart >> 32 != allocAddrEnd >> 32)
    {
        LOG_INFO(SYN_DEVICE,
                 "Device pre allocating global memory for compute streams is not aligned, moving to next 4GB alignment, allocation start {:x} allocation end {:x}",
                 allocAddrStart,
                 allocAddrEnd);

        status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)->releaseDeviceMemory(glblRefBuffer.BufHandle);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "Device failed releaseDeviceMemory() of pre allocated global memory for compute streams ");
            return status;
        }
        glblRefBuffer.BufHandle = nullptr;

        uint64_t alignmentPad = UINT64_HIGH_PART(allocAddrEnd) - allocAddrStart;
        status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)->allocateDeviceMemory(alignmentPad + glblRefBuffer.Size * ScalDev::MaxNumOfComputeStreams, glblRefBuffer.BufHandle);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "Device failed pre allocating global memory for compute streams, status: {}",
            status);
            return status;
        }
        status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)
                        ->getDeviceMemoryAddress(glblRefBuffer.BufHandle, glblRefBuffer.AddrCore, glblRefBuffer.AddrDev);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "Device failed in getDeviceMemoryAddress when pre allocating global memory for compute streams, status: {}",
                    status);
            return status;
        }
        glblRefBuffer.AddrDev += alignmentPad;
    }

    // Allocate arc shared memory for stream
    status =
        m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_HBM_SHARED)->allocateDeviceMemory(sharedRefBuffer.Size * ScalDev::MaxNumOfComputeStreams, sharedRefBuffer.BufHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Device failed pre allocating shared memory for compute streams, status: {}",
        status);
        return status;
    }
    status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_HBM_SHARED)
                    ->getDeviceMemoryAddress(sharedRefBuffer.BufHandle, sharedRefBuffer.AddrCore, sharedRefBuffer.AddrDev);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE,
                "Device failed in getDeviceMemoryAddress when pre allocating shared memory for compute streams, status: {}",
                status);
        return status;
    }

    for (unsigned idx = 0; idx < ScalDev::MaxNumOfComputeStreams; idx++)
    {
        PreAllocatedStreamMemory* buf;
        buf             = &m_preAllocatedMem[idx].global;
        buf->Size       = glblRefBuffer.Size;
        buf->BufHandle  = glblRefBuffer.BufHandle;
        buf->AddrCore   = glblRefBuffer.AddrCore + (glblRefBuffer.Size * idx);
        buf->AddrDev    = glblRefBuffer.AddrDev  + (glblRefBuffer.Size * idx);

        buf             = &m_preAllocatedMem[idx].shared;
        buf->Size       = sharedRefBuffer.Size;
        buf->BufHandle  = sharedRefBuffer.BufHandle;
        buf->AddrCore   = sharedRefBuffer.AddrCore + (sharedRefBuffer.Size * idx);
        buf->AddrDev    = sharedRefBuffer.AddrDev  + (sharedRefBuffer.Size * idx);
    }
    return status;
}

synStatus DeviceScal::releasePreAllocatedMem()
{
    synStatus status = synSuccess;

    PreAllocatedStreamMemory* buf = &m_preAllocatedMem[0].global;
    if (buf->BufHandle)
    {
        status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)->releaseDeviceMemory(buf->BufHandle);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "Device failed releaseDeviceMemory() of pre allocated global memory for compute streams ");
            return status;
        }
        buf->BufHandle = nullptr;
    }

    buf = &m_preAllocatedMem[0].shared;
    if (buf->BufHandle)
    {
        status = m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_HBM_SHARED)->releaseDeviceMemory(buf->BufHandle);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "Device failed releaseDeviceMemory() of pre allocated shared memory for compute streams ");
            return status;
        }
        buf->BufHandle = nullptr;
    }
    return status;
}

/**
 * dump status for each stream
 * @param logForUser if true, then show only short info relevant for the user
 */
void DeviceScal::dfaInfo(DfaReq dfaReq, uint64_t csSeq)
{
    LOG_DEBUG(SYN_DEV_FAIL, "Number of queues: {}", m_queueInterfaces.size());

    for (QueueInterface* pQueueInterface : m_queueInterfaces)
    {
        pQueueInterface->dfaInfo(dfaReq, csSeq);
    }

    m_streamsContainer.dump(synapse::LogManager::LogType::SYN_DEV_FAIL);
}

TdrRtn DeviceScal::streamsTdr(TdrType tdrType)
{
    TdrRtn tdrRtn {};

    for (auto& rpQueueInterface : m_queueInterfaces)
    {
        if ((rpQueueInterface != nullptr) &&
            (rpQueueInterface->getBasicQueueInfo().queueType == INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK))
        {
            continue;
        }

        TdrRtn temp = static_cast<QueueBaseScalCommon*>(rpQueueInterface)->tdr(tdrType);

        tdrRtn.failed |= temp.failed;

        if (temp.failed)
        {
            if (!tdrRtn.msg.empty())
            {
                tdrRtn.msg += ", ";
            }
            tdrRtn.msg += temp.msg;
        }
    }

    return tdrRtn;
}

bool DeviceScal::isDirectModeUserDownloadStream() const
{
    internalStreamType internalType;
    synStatus          status = getInternalStreamTypes(QUEUE_TYPE_COPY_HOST_TO_DEVICE, &internalType);
    return ((status == synSuccess) && m_scalDev.isDireceModeStream(internalType));
}

synStatus DeviceScal::getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                      synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const
{
    CHECK_POINTER(SYN_DEVICE, streamHandle, "streamHandle", synInvalidArgument);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    return streamSptr->getDynamicShapesTensorInfoArray(recipeHandle, tensorInfoArray);
}

uint64_t DeviceScal::getHbmGlblSize()
{
    return GCFG_HBM_GLOBAL_MEM_SIZE_MEGAS.value() * 1024 * 1024;
}

uint64_t DeviceScal::getHbmGlblMaxRecipeSize()
{
    return QueueComputeScal::getHbmGlblMaxRecipeSize(getHbmGlblSize());
}

uint64_t DeviceScal::getHbmSharedMaxRecipeSize()
{
    return QueueComputeScal::getHbmSharedMaxRecipeSize(hbmSharedSize);
}

static void tdrLogFunc(int logLevel, const char* msg)
{
    SYN_LOG_TYPE(SYN_DEVICE, logLevel, "{}", msg);
}

synStatus DeviceScal::release(std::atomic<bool>& rDeviceBeingReleased)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    LOG_TRACE_T(SYN_DEVICE, "{}", HLLOG_FUNC);

    synStatus status = releaseAllStreams();
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: releaseStreams returned synStatus {}.", HLLOG_FUNC, status);
        return status;
    }

    if (GCFG_INIT_HCCL_ON_ACQUIRE.value() && m_hclInit)
    {
        if (hcclDestroyDevice(0) != hcclSuccess)
        {
            LOG_ERR(SYN_API, "{}: Failed to destroy HCCL device", HLLOG_FUNC);
            return synFail;
        }
        m_hclInit = false;
    }

    rDeviceBeingReleased = true;

    status = stopEventFdThread();
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: stopEventFdThread returned synStatus {}.", HLLOG_FUNC, status);
        return status;
    }

    LOG_INFO_T(SYN_DEVICE, "scal tdr status when releasing device is below");

    scal_handle_t scalHandle = m_scalDev.getScalHandle();

    if (!scalHandle)
    {
        LOG_INFO_T(SYN_DEVICE, "scal handle is nullptr, probably releasing device after failing to acquire");
    }
    else
    {
        scal_bg_work(scalHandle, tdrLogFunc);  // log tdr status before closing the stream
    }

    status = releasePreAllocatedMem();
    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: releasePreAllocatedMem returned synStatus {}.", HLLOG_FUNC, status);
        return status;
    }

    delete m_scalEventsPool;

    status = OSAL::getInstance().releaseAcquiredDeviceBuffers();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: releaseAcquiredDeviceBuffers returned synStatus {}", HLLOG_FUNC, status);
        return status;
    }

    status = m_scalDev.release();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: m_scalDev.release() returned synStatus {}", HLLOG_FUNC, status);
        return status;
    }

    status = OSAL::getInstance().releaseAcquiredDevice();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: releaseAcquiredDevice returned synStatus {}", HLLOG_FUNC, status);
        return status;
    }

    return synSuccess;
}

synStatus DeviceScal::allocateMemory(uint64_t           size,
                                     uint32_t           flags,
                                     void**             buffer,
                                     bool               isUserRequest,
                                     uint64_t           reqVAAddress,
                                     const std::string& mappingDesc,
                                     uint64_t*          deviceVA)
{
    return m_devMemoryAlloc->allocateMemory(size, flags, buffer, isUserRequest, reqVAAddress, mappingDesc, deviceVA);
}

synStatus DeviceScal::getDramMemInfo(uint64_t& free, uint64_t& total) const
{
    PoolMemoryStatus poolMemoryStatus;
    synStatus        status = m_scalDev.getMemoryPoolStatus(ScalDev::MEMORY_POOL_GLOBAL, poolMemoryStatus);

    if (status != synSuccess)
    {
        return status;
    }

    free  = poolMemoryStatus.free;
    total = poolMemoryStatus.total;

    return status;
}

/*
 ***************************************************************************************************
 *   @brief createStream() creates a new stream (copy, compute, etc.)
 *          Called when the user requests a new stream. It looks for a free scalStream (and completion queue),
 *          creates a handle to return to the user and save the new stream in a DB (used if we need to
 *          destroy all streams).
 *
 *   @param  queueType - requested stream type
 *   @param  pStreamHandle - a handle given by the user to be filled by us
 *   @return status
 *
 *   NOTE!!!: the function assumes m_mutex is locked !!!
 ***************************************************************************************************
 */
synStatus
DeviceScal::createStreamQueue(QueueType queueType, uint32_t flags, bool isReduced, QueueInterface*& rpQueueInterface)
{
    // flags & isReduced are not used

    internalStreamType internalType;
    synStatus          ret = getInternalStreamTypes(queueType, &internalType);
    if (ret != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: getInternalStreamTypes returned synStatus {}", HLLOG_FUNC, ret);
        return ret;
    }

    QueueInterface* pQueueInterace = nullptr;
    BasicQueueInfo  basicQueueInfo {{0, 0}, INTERNAL_STREAM_TYPE_NUM, TRAINING_QUEUE_NUM, 0, nullptr};
    basicQueueInfo.queueType = internalType;

    std::string name {"name not set"};

    {
        unsigned                        streamIdx = 0;
        ScalStreamBaseInterface*        pScalStream       = nullptr;
        const ComputeCompoundResources* pComputeResources = nullptr;

        if (internalType == INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK)
        {
            pScalStream = nullptr;
            streamIdx   = 0;
            name        = std::string("collective_") + std::to_string(m_collectiveStreamNum);

            if (m_collectiveStreamNum == MAX_AFFINITY_QUEUES_NUM)
            {
                LOG_ERR(SYN_DEVICE,
                        "{}: m_collectiveStreamNum reached max ({})",
                        HLLOG_FUNC,
                        MAX_AFFINITY_QUEUES_NUM);
                return synAllResourcesTaken;
            }
            m_collectiveStreamNum++;
        }
        else
        {
            if (internalType == INTERNAL_STREAM_TYPE_COMPUTE)
            {
                // get scal stream
                pScalStream = m_scalDev.getFreeComputeResources(pComputeResources);
                if ((pScalStream       == nullptr) ||
                    (pComputeResources == nullptr))
                {
                    return synFail;
                }
                streamIdx = pComputeResources->m_streamIndex;
            }
            else
            {
                StreamAndIndex streamInfo;

                // get scal stream
                if (!m_scalDev.getFreeStream(internalType, streamInfo))
                {
                    return synFail;
                }

                streamIdx   = streamInfo.idx;
                pScalStream = streamInfo.pStream;
            }

            if (pScalStream == nullptr)
            {
                LOG_ERR(SYN_DEVICE, "{}: can't get a free stream - all resources are taken.", HLLOG_FUNC);
                return synAllResourcesTaken;
            }

            name = pScalStream->getName();
        }

        // create the stream
        ret = createStreamScal(pQueueInterace, basicQueueInfo, internalType, pScalStream, streamIdx, pComputeResources);
        if (ret != synSuccess)
        {
            if (internalType == INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK)
            {
                m_collectiveStreamNum--;
            }
            else
            {
                if (internalType == INTERNAL_STREAM_TYPE_COMPUTE)
                {
                    m_scalDev.releaseComputeResources(pScalStream);
                }
                else
                {
                    m_scalDev.releaseStream(pScalStream);
                }
            }
            LOG_ERR(SYN_DEVICE, "{}: createStreamScal returned synStatus {}", HLLOG_FUNC, ret);
            return ret;
        }

        rpQueueInterface = pQueueInterace;
        m_queueInterfaces.push_back(rpQueueInterface);
    }

    LOG_INFO(SYN_DEVICE,
             "{}, new streamHanlde {:x} for type {} stream {:x} pScalStream name {}",
             HLLOG_FUNC,
             TO64(rpQueueInterface),
             queueType,
             TO64(rpQueueInterface),
             name);

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief destroyStreamQueue() destroys the given streamHandle assuming the stream container is locked
 *          Called when the user requests to destroy a stream.
 *          It first verify no work on the stream then it returns the scal stream & completion queue
 *          back to free, removes it from the stream DB and deletes it.
 *
 *   @param  pQueueInterface - queue to be deleted
 *   @return status
 *
 *   NOTE!!!: the function assumes m_mutex is locked !!!
 *
 ***************************************************************************************************
 */
synStatus DeviceScal::destroyStreamQueue(QueueInterface* pQueueInterface)
{
    if (pQueueInterface == nullptr)
    {
        LOG_DEBUG(SYN_API, "pQueueInterface 0x{:x} had already been destroyed", (uint64_t)pQueueInterface);
        return synSuccess;
    }

    if (pQueueInterface->getBasicQueueInfo().queueType == INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK)
    {
        if (m_collectiveStreamNum == 0)
        {
            LOG_ERR(SYN_DEVICE,
                    "{}: destroyStreamQueue failed since there is no collective stream to destroy {}",
                    HLLOG_FUNC,
                    pQueueInterface->getBasicQueueInfo().getDescription());
            return synInvalidArgument;
        }

        synStatus status = pQueueInterface->synchronize(nullptr, false);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: synchronize failed with status {}", HLLOG_FUNC, status);
            return synFail;
        }

        status = pQueueInterface->destroyHclStream();
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: destroyHclStream failed with status {}", HLLOG_FUNC, status);
            return synFail;
        }

        m_collectiveStreamNum--;
    }
    else
    {
        // WA - prevent handle access after deletion
        // it's not a correct protection - SlotMap must be used instead
        QueueBaseScalCommon* pStreamBaseScalCommon = dynamic_cast<QueueBaseScalCommon*>(pQueueInterface);
        CHECK_POINTER(SYN_DEVICE, pStreamBaseScalCommon, "pStreamBaseScalCommon", synInvalidArgument);

        ScalStreamBaseInterface* pScalStream = pStreamBaseScalCommon->getScalStream();
        CHECK_POINTER(SYN_DEVICE, pScalStream, "pScalStream", synInvalidArgument);

        // We don't check there are no open requests before removing the scal stream,
        // we need to do it here (not easy to do the sync inside the scalStream. for now)
        synStatus status = pStreamBaseScalCommon->waitForLastLongSo(false);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: waitForLastLongSo failed with status {}", HLLOG_FUNC, status);
            return status;
        }

        LOG_INFO_T(SYN_STREAM, "tdr - before closing stream");
        pScalStream->tdr(TdrType::CLOSE_STREAM);

        status = m_scalDev.releaseStream(pScalStream);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "{}: releaseStream failed with status {}", HLLOG_FUNC, status);
        }
    }

    removeFromStreamHandlesL(pQueueInterface);

    delete pQueueInterface;

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief releaseAllStreams() destroys all user streams
 *
 *   @param  None
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus DeviceScal::releaseAllStreams()
{
    notifyAllRecipeRemoval();

    synStatus status = removeAllStreamAffinities();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: removeAllStreamAffinities failed with status {} ", HLLOG_FUNC, status);
        return status;
    }

    while (!m_queueInterfaces.empty())
    {
        QueueInterface* pQueueInterface = m_queueInterfaces.back();

        status = destroyStreamQueue(pQueueInterface);

        if (status == synDeviceReset)
        {
            notifyHlthunkFailure(DfaErrorCode::deviceReset);
        }
        else if (status != synSuccess)
        { // Basically, should never get here
            break;
        }
    }

    return status;
}

void DeviceScal::removeFromStreamHandlesL(QueueInterface* pQueueInterface)
{
    // This is called from DeviceCommon::createStream that holds the m_mutex

    auto it = find(m_queueInterfaces.begin(), m_queueInterfaces.end(), pQueueInterface);
    if (it == m_queueInterfaces.end())
    {
        LOG_ERR(SYN_DEVICE, "stream handle {:x} not found during destroy", TO64(pQueueInterface));
    }
    else
    {
        m_queueInterfaces.erase(it);
    }
}

synStatus DeviceScal::createEvent(synEventHandle* pEventHandle, const unsigned int flags)
{
    CHECK_POINTER(SYN_DEVICE, pEventHandle, "pEventHandle", synInvalidArgument);

    std::pair<synEventHandle, SlotMapItemSptr<ScalEvent>> newEventHandle =
        m_scalEventsPool->getNewEvent(flags);
    if (newEventHandle.second == nullptr)
    {
        *pEventHandle = nullptr;
        return synAllResourcesTaken;
    }

    *pEventHandle = (synEventHandle)newEventHandle.first;
    LOG_INFO(SYN_STREAM, "Created new event {:x}", TO64(*pEventHandle));
    return synSuccess;
}

synStatus DeviceScal::destroyEvent(synEventHandle eventHandle)
{
    synStatus ret = m_scalEventsPool->destroyEvent(eventHandle);
    if (ret != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "failed to destroy scalEvent {:x}", (SMHandle)eventHandle);
    }
    return ret;
}

EventSptr DeviceScal::getEventSptr(synEventHandle eventHandle)
{
    return m_scalEventsPool->getEventSptr(eventHandle);
}

synStatus DeviceScal::eventQuery(const EventInterface* pEventInterface)
{
    // Todo extend EventInterface to include the recorded stream and avoid casting
    const ScalEvent* pScalEvent = dynamic_cast<const ScalEvent*>(pEventInterface);
    CHECK_POINTER(SYN_DEVICE, pScalEvent, "pScalEvent", synInvalidArgument);
    // defend against overriding the event while we're inside eventQuery
    // by creating a private copy of it
    ScalEvent scalEvent = *pScalEvent;
    QueueBaseScal* pStream   = scalEvent.pStreamIfScal;
    CHECK_POINTER(SYN_DEVICE, pStream, "pStream", synInvalidArgument);

    LOG_INFO(SYN_STREAM, "{}: {}", HLLOG_FUNC, scalEvent.toString());

    synStatus status = pStream->eventQuery(scalEvent);
    pScalEvent->setWaitMode(EventInterface::WaitMode::waited);
    return status;
}

synStatus DeviceScal::synchronizeEvent(const EventInterface* pEventInterface)
{
#ifdef DISABLE_SYNC_ON_DEV
    return synSuccess;
#endif

    const ScalEvent* pScalEvent = dynamic_cast<const ScalEvent*>(pEventInterface);
    CHECK_POINTER(SYN_DEVICE, pScalEvent, "pScalEvent", synInvalidArgument);
    // defend against overriding the event while we're inside synchronizeEvent
    // by creating a private copy of it
    ScalEvent scalEvent = *pScalEvent;

    QueueBaseScal* pStream = scalEvent.pStreamIfScal;
    CHECK_POINTER(SYN_DEVICE, pStream, "pStream", synInvalidArgument);

    LOG_INFO(SYN_STREAM, "{}: {}", HLLOG_FUNC, scalEvent.toString());

    const synStatus status = pStream->eventSynchronize(scalEvent);
    if (status == synDeviceReset)
    {
        notifyHlthunkFailure(DfaErrorCode::eventSyncFailed);
    }
    pScalEvent->setWaitMode(EventInterface::WaitMode::waited);
    return status;
}

synStatus DeviceScal::getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const
{
    totalStreamMappedMemorySize = 0;

    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    for (auto pQueueInterface : m_queueInterfaces)
    {
        uint64_t size;

        if (pQueueInterface->getBasicQueueInfo().queueType == INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK)
        {
            ;  // not supported
        }
        else
        {
            synStatus status = pQueueInterface->getMappedMemorySize(size);
            totalStreamMappedMemorySize += size;

            if (status != synSuccess)
            {
                return synFail;
            }
        }
    }

    return synSuccess;
}

void DeviceScal::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    for (auto& pQueueInterface : m_queueInterfaces)
    {
        if (pQueueInterface->getBasicQueueInfo().queueType == INTERNAL_STREAM_TYPE_COMPUTE)
        {
            static_cast<QueueComputeScal*>(pQueueInterface)->notifyRecipeRemoval(rRecipeHandle);
        }
    }
}

void DeviceScal::notifyAllRecipeRemoval()
{
    for (auto& pQueueInterface : m_queueInterfaces)
    {
        if (pQueueInterface->getBasicQueueInfo().queueType == INTERNAL_STREAM_TYPE_COMPUTE)
        {
            static_cast<QueueComputeScal*>(pQueueInterface)->notifyAllRecipeRemoval();
        }
    }
}

synStatus DeviceScal::getDeviceInfo(synDeviceInfo& rDeviceInfo) const
{
    rDeviceInfo = m_osalInfo;
    uint64_t free;
    return getDramMemInfo(free, rDeviceInfo.dramSize);
}

synStatus DeviceScal::getDeviceInfo(synDeviceInfoV2& rDeviceInfo) const
{
    // Always returns synSuccess
    DeviceCommon::getDeviceInfo(rDeviceInfo);

    uint64_t free;
    return getDramMemInfo(free, rDeviceInfo.dramSize);
}

void DeviceScal::checkDevFailure(uint64_t csSeqTimeout, DfaStatus dfaStatus, ChkDevFailOpt option, bool isSimulator)
{
    if (option == ChkDevFailOpt::MAIN)
    {
        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::QUEUES_DETAILS));

        scal_timeouts_t timeouts;
        int             rtn = scal_get_timeouts(m_scalDev.getScalHandle(), &timeouts);

        if (rtn != 0)
        {
            LOG_ERR(SYN_DEV_FAIL, "Couldn't get timeouts from scal with rt n{}", rtn);
        }
        else
        {
            LOG_TRACE(SYN_DEV_FAIL, "engines timeout: {} us   no-progress-timeout: {} us", timeouts.timeoutUs, timeouts.timeoutNoProgressUs);
        }

        LOG_TRACE(SYN_DEV_FAIL, "Total launches until now {}", QueueComputeUtils::getCurrnetSeqId());

        dfaInfo(DfaReq::STREAM_INFO);  // dfaInfo

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::NO_PROGRESS_TDR));
        scal_bg_work(m_scalDev.getScalHandle(), dfaLogFunc);

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::OLDEST));
        dfaInfo(DfaReq::ERR_WORK);

        /********************/
        /* dump debug info  */
        /********************/
        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::Q_STATUS));
        dfaInfo(DfaReq::ALL_WORK);

        LOG_DEBUG(SYN_DEV_FAIL, "Logging History to file {}", DFA_API_FILE);

        hl_logger::logAllLazyLogs(hl_logger::getLogger(synapse::LogManager::LogType::DFA_API_INFO));

        m_hclApiWrapper.checkHclFailure(dfaStatus, nullptr, hcl::HclPublicStreams::DfaLogPhase::Main); // 0: main hcl info

        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR(DfaMsg::ARC_HEARTBEAT));
        checkArcsHeartBeat(isSimulator);
    }
    else
    {
        LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("Arc Scheduler cyclic buffers (CCBs)"));
        dfaInfo(DfaReq::SCAL_STREAM);
        m_hclApiWrapper.checkHclFailure(dfaStatus, nullptr, hcl::HclPublicStreams::DfaLogPhase::Ccb); // 1: ccb info
    }
}

/*
 ***************************************************************************************************
 *   @brief launch() synLaunch
 *          verify input params, build a structure with all the input params (LaunchInf)
 *          and calls the compute stream to do the launch
 *
 *   @param  streamIf stream, vector of src/dst/size, direction, isUserReq
 *   @return UP/DOWN
 *
 ***************************************************************************************************
 */
synStatus DeviceScal::launch(Stream*                       pStream,
                             const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsInfoAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             EventWithMappedTensorDB&      events,
                             uint32_t                      flags)
{
    synStatus status;
    bool      isKernelPrintf = RecipeUtils::isKernelPrintf(*pRecipeHandle);
    if (isKernelPrintf)
    {
        status = clearKernelsPrintfWs(workspaceAddress, pRecipeHandle->deviceAgnosticRecipeHandle.m_workspaceSize);
        if (status != synSuccess)
        {
            return status;
        }
    }

    uint64_t                   assertAsyncMappedAddress = (uint64_t)getAssertAsyncMappedAddress();
    std::unique_ptr<StreamJob> job                      = std::make_unique<ComputeJob>(launchTensorsInfo,
                                                                  launchTensorsInfoAmount,
                                                                  workspaceAddress,
                                                                  pRecipeHandle,
                                                                  assertAsyncMappedAddress,
                                                                  flags,
                                                                  events,
                                                                  generateApiId());
    status                                              = m_streamsContainer.addJob(pStream, job);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Failed to launch status {}", HLLOG_FUNC, status);
        return status;
    }

    if (isKernelPrintf)
    {
        kernelsPrintf(*pRecipeHandle, workspaceAddress, nullptr);
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief clearKernelsPrintfWs() Optionally clears (if the user has a dev-dev stream)
 *          the part of the workspace that is later used to save
 *          the kernelsPrintf. This helps with debug.
 *
 *   @param  launchInfo
 *   @return UP/DOWN
 *
 ***************************************************************************************************
 */
synStatus DeviceScal::clearKernelsPrintfWs(uint64_t workspaceAddress, uint64_t workspaceSize)
{
    const uint64_t fillVal = 0xAAAAAAAAAAAAAAAA;

    QueueInterface* streamCopy = getAnyStream(INTERNAL_STREAM_TYPE_DEV_TO_DEV);
    // fill the buffer with some value (for debug), if we have the stream
    if (streamCopy)
    {
        internalMemcopyParams memcpyParams {{.src = 0, .dst = workspaceAddress, .size = workspaceSize}};

        synStatus status = streamCopy->memcopy(memcpyParams,
                                               MEMCOPY_DRAM_TO_DRAM,
                                               true /* isUserRequest */,
                                               nullptr,
                                               fillVal /* overrideMemsetVal */,
                                               false /* inspectCopiedContent */,
                                               nullptr /* pRecipeProgramBuffer */,
                                               generateApiId());
        if (status != synSuccess)
        {
            return status;
        }

        status = streamCopy->synchronize(nullptr, true /* isUserRequest */);
        if (status != synSuccess)
        {
            return status;
        }
    }
    return synSuccess;  // even if we don't find a DEV-DEV stream, we can still
                        // continue (clearing the ws is for debug only)
}

/*
 ***************************************************************************************************
 *   @brief kernelsPrintf() Prints the kernelsPrintf to screen/log from the workspace
 *          If a buffer is given, prints to buffer and log. If not, prints to screen
 *
 *   @param  recipe, workspace addr, buffer
 *   @return UP/DOWN
 *
 ***************************************************************************************************
 */
synStatus DeviceScal::kernelsPrintf(const InternalRecipeHandle& rInternalRecipeHandle, uint64_t wsAddr, void* hostBuff)
{
    uint8_t*      userBuff = (uint8_t*)hostBuff;
    const uint8_t fillVal = 0x55;

    char*                   buff = nullptr;
    std::unique_ptr<char[]> localBuff;
    uint64_t                singleBuffSize = GCFG_TPC_PRINTF_TENSOR_SIZE.value();

    // use user buffer or create a new one
    if (userBuff == nullptr)
    {
        localBuff = std::unique_ptr<char[]>(new char[singleBuffSize]);
        buff      = localBuff.get();
    }
    else
    {
        buff = reinterpret_cast<char*>(userBuff);
    }

    // map the buffer
    std::string mappingDesc("Kernel Printf Log");
    synStatus   status = mapBufferToDevice(singleBuffSize, buff, true /* isUserRequest */, 0, mappingDesc);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API,
                "{}: Can not map to device host-memory; buff 0x{:x} size 0x{:x}",
                HLLOG_FUNC,
                TO64(buff),
                singleBuffSize);
        return status;
    }

    memset(buff, fillVal, singleBuffSize);  // just for easier debug

    QueueInterface* streamUpload = getAnyStream(INTERNAL_STREAM_TYPE_DMA_UP);  // just use any stream
    uint32_t printAddrNr = rInternalRecipeHandle.basicRecipeHandle.recipe->debug_profiler_info.printf_addr_nr;
    std::vector<std::string> finalPrints;

    internalMemcopyParams memcpyParam {1};  // one entry
    for (uint32_t iPrint = 0; iPrint < printAddrNr; iPrint++)
    {
        memcpyParam[0].size = singleBuffSize;
        memcpyParam[0].src =
            wsAddr + RecipeUtils::getKernelPrintOffsetInWs(&rInternalRecipeHandle) + (iPrint * singleBuffSize);
        memcpyParam[0].dst  = (uint64_t)buff;

        status = streamUpload->memcopy(memcpyParam,
                                       MEMCOPY_DRAM_TO_HOST,
                                       true /* isUserRequest */,
                                       nullptr,
                                       0 /* overrideMemsetVal */,
                                       false /* inspectCopiedContent */,
                                       nullptr /* pRecipeProgramBuffer */,
                                       generateApiId());
        if (status != synSuccess)
        {
            LOG_ERR_T(SYN_TPC_PRINT, "failed memcopy");
            // continue, to help debug
        }

        streamUpload->synchronize(nullptr, true /* isUserRequest */);

        parsePrintf((uint32_t*)buff, finalPrints);
    }

    // dump the output to screen or user buffer
    for (const std::string& printf : finalPrints)
    {
        // for Gaudi/Gaudi2/Gaudi3, print to screen,
        // unless a buffer was given from user (testing of this feature)
        if (userBuff == nullptr)
        {
            std::cout << printf;
        }
        else
        {
            LOG_DEBUG(SYN_TPC_PRINT, "Kernel print: {}", printf);
        }
    }

    unmapBufferFromDevice(buff, true, nullptr);

    return status;
}

/*
 ***************************************************************************************************
 *   @brief getAnyStream() get the first found stream from the requested tyep
 *
 *   @param  requested type
 *   @return stream / null (if not found)
 *
 ***************************************************************************************************
 */
QueueInterface* DeviceScal::getAnyStream(internalStreamType requestedType)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    for (QueueInterface* pQueueInterface : m_queueInterfaces)
    {
        if (pQueueInterface->getBasicQueueInfo().queueType == requestedType)
        {
            return pQueueInterface;
        }
    }
    return nullptr;
}

synStatus DeviceScal::createStreamScal(QueueInterface*&                rpQueueInterface,
                                       const BasicQueueInfo&           rBasicQueueInfo,
                                       internalStreamType              internalType,
                                       ScalStreamBaseInterface*        pScalStream,
                                       unsigned                        streamIdx,
                                       const ComputeCompoundResources* pComputeResources)
{
    rpQueueInterface = nullptr;

    switch (internalType)
    {
        case INTERNAL_STREAM_TYPE_DMA_UP:
        case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
        case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
        {
            ScalStreamCopyInterface* pScalStreamCopy = dynamic_cast<ScalStreamCopyInterface*>(pScalStream);
            HB_ASSERT(pScalStreamCopy != nullptr, "DeviceScal invalid call detected internalType {}", internalType);
            rpQueueInterface = new QueueCopyScal(rBasicQueueInfo, pScalStreamCopy, *m_devMemoryAlloc.get());
            break;
        }
        case INTERNAL_STREAM_TYPE_COMPUTE:
        {
            ScalStreamComputeInterface* pScalStreamCompute = dynamic_cast<ScalStreamComputeInterface*>(pScalStream);
            HB_ASSERT(pScalStreamCompute != nullptr, "DeviceScal invalid call detected internalType {}", internalType);
            QueueComputeScal* pQueueCompute = new QueueComputeScal(rBasicQueueInfo,
                                                                   pScalStreamCompute,
                                                                   pComputeResources,
                                                                   m_devType,
                                                                   m_devSpecificInfo,
                                                                   *m_devMemoryAlloc.get());

            synStatus status = pQueueCompute->init(m_preAllocatedMem[streamIdx]);
            if (status != synSuccess)
            {
                delete pQueueCompute;
                break;
            }
            rpQueueInterface = pQueueCompute;
            break;
        }
        case INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK:
        {
            rpQueueInterface = new QueueCollectiveNetworkScal(rBasicQueueInfo, m_hclApiWrapper);

            synStatus status = rpQueueInterface->createHclStream();
            if (status != synSuccess)
            {
                delete rpQueueInterface;
                return status;
            }

            break;
        }
        case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
        case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
        case INTERNAL_STREAM_TYPE_NUM:
        {
            LOG_ERR_T(SYN_DEVICE, "Unknown internalType {}", internalType);
            break;
        }
    }

    if (rpQueueInterface == nullptr)
    {
        LOG_ERR_T(SYN_DEVICE, "Couldn't create new stream internalType {}", internalType);
        return synFail;
    }
    return synSuccess;
}

synStatus DeviceScal::getClusterInfo(scal_cluster_info_t& clusterInfo, char* clusterName)
{
    return m_scalDev.getClusterInfo(clusterInfo, clusterName);
}


synStatus DeviceScal::eventRecord(EventInterface* pEventInterface, synStreamHandle streamHandle)
{
    CHECK_POINTER(SYN_DEVICE, pEventInterface, "pEventInterface", synInvalidArgument);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    ScalEvent* pScalEvent = dynamic_cast<ScalEvent*>(pEventInterface);
    LOG_INFO(SYN_STREAM, "{}: {}", HLLOG_FUNC, pScalEvent->toString());
    std::lock_guard guard(*pScalEvent);
    std::unique_ptr<StreamJob> job = std::make_unique<EventRecordJob>(*pEventInterface, nullptr);
    if (pScalEvent->getWaitMode() == EventInterface::WaitMode::not_yet)
    {
        // if changed, change also stream_event_wait_cyclic_buffer_quarter_signal test
        if (GCFG_ENABLE_CHECK_EVENT_REUSE.value())
        {
            LOG_ERR(SYN_STREAM, "{}: using an event that was recorded but not waited upon {}", __FUNCTION__, pScalEvent->toString());
            return synResourceBadUsage;
        }
        else
        {
            LOG_WARN(SYN_STREAM, "{}: using an event that was recorded but not waited upon {}", __FUNCTION__, pScalEvent->toString());
        }
    }
    pScalEvent->setWaitMode(EventInterface::WaitMode::not_yet);
    return m_streamsContainer.addJob(streamSptr.get(), job);
}

synStatus DeviceScal::streamGenericWaitEvent(synStreamHandle       streamHandle,
                                             const EventInterface& rEventInterface,
                                             const unsigned int    flags)
{
    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    ScalEvent&                 rEvent = (ScalEvent&)rEventInterface;
    std::unique_ptr<StreamJob> job    = std::make_unique<WaitForEventJobScal>(rEvent, flags);
    synStatus status = m_streamsContainer.addJob(streamSptr.get(), job);
    rEvent.setWaitMode(EventInterface::WaitMode::waited);
    return status;
}

/*
 ***************************************************************************************************
 *   @brief checkArcsHeartBeat() - logs all arcs registers. In addition, checks the heartbeat of the arcs
 *                          By reading the values twice
 *
 *   @param - None
 *   @return - Nonce
 *
 ***************************************************************************************************
 */
void DeviceScal::checkArcsHeartBeat(bool isSimulator)
{
    const common::DeviceInfoInterface* deviceInfoInterface = m_scalDev.getDeviceInfoInterface();

    if (!deviceInfoInterface) // Should not happen
    {
        LOG_ERR(SYN_DEV_FAIL, "Could not get cpu count (deviceInfoInterface)");
        return;
    }

    uint16_t                           maxNumArcCpus       = deviceInfoInterface->getNumArcCpus();

    std::vector<uint32_t>           prevHeartBeat(maxNumArcCpus);
    std::vector<uint32_t>           currHeartBeat(maxNumArcCpus);
    std::vector<scal_core_handle_t> coreHandle(   maxNumArcCpus);
    std::vector<const char*>        arcNames(     maxNumArcCpus);

    for (int arcId = 0; arcId < maxNumArcCpus; arcId++)
    {
        scal_get_core_handle_by_id(m_scalDev.getScalHandle(), arcId, &coreHandle[arcId]);
    }

    // get heartbeat
    for (int arcId = 0; arcId < maxNumArcCpus; arcId++)
    {
        auto arcCpuHandle = coreHandle[arcId];
        if (arcCpuHandle == nullptr) continue;

        scal_control_core_debug_info_t coreDebugInfo;
        scal_control_core_get_debug_info(arcCpuHandle, nullptr, 0, &coreDebugInfo);
        prevHeartBeat[arcId] = coreDebugInfo.heartBeat;

        scal_control_core_info_t coreInfo;
        scal_control_core_get_info(arcCpuHandle, &coreInfo);
        arcNames[arcId] = coreInfo.name;
    }

    // if this is a simulator, sleep for 5 seconds to let the heart-beats increment
    if (isSimulator)
    {
        sleep(5);
    }

    // get heartbeat again and all registers, log the registers
    for (int arcId = 0; arcId < maxNumArcCpus; arcId++)
    {
        auto arcCpuHandle = coreHandle[arcId];
        if (arcCpuHandle == nullptr)
        {
            LOG_ERR(SYN_DEV_FAIL, "arc {} isn't configured. Might NOT be an error but can be a binned engine.", arcId);
            continue;
        }

        scal_control_core_debug_info_t coreDebugInfo;
        scal_control_core_get_debug_info(arcCpuHandle, nullptr, 0, &coreDebugInfo);
        currHeartBeat[arcId] = coreDebugInfo.heartBeat;
    }

    // formatting as a table with cells width of 10
    // formatting guide: https://fmt.dev/latest/syntax.html
    int cellWidth = 10;

    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("Logging engines heartbeat values"));
    LOG_TRACE(SYN_DEV_FAIL, "has {} cpu-s", maxNumArcCpus);
    LOG_TRACE(SYN_DEV_FAIL,
              "{1:^{0}} | {2:^{0}} | {3:^{0}} | {4:^{0}} | {5:{0}}",
              cellWidth,
              "ID",
              "previous",
              "current",
              "equals",
              "name");
    LOG_TRACE(SYN_DEV_FAIL, "{1:-^{0}} + {1:-^{0}} + {1:-^{0}} + {1:-^{0}} + {1:-^{0}}", cellWidth, "");

    // log the heartbeat values (from both reads)
    for (int arcId = 0; arcId < maxNumArcCpus; arcId++)
    {
        if (coreHandle[arcId] == nullptr) continue;

        std::string mark;

        if (currHeartBeat[arcId] == prevHeartBeat[arcId])
        {
            mark = "***";
        }
        LOG_TRACE(SYN_DEV_FAIL,
                  "{1:{0}} | {2:{0}} | {3:{0}} | {4:^{0}} | {5:{0}}",
                  cellWidth,
                  arcId,
                  prevHeartBeat[arcId],
                  currHeartBeat[arcId],
                  mark,
                  arcNames[arcId]);
    }
}

synStatus DeviceScal::getTdrIrqMonitorArmRegAddr(volatile uint32_t*& tdrIrqMonitorArmRegAddr)
{
    synStatus status = synSuccess;
    tdrIrqMonitorArmRegAddr = m_scalDev.getTdrIrqMonitorArmRegAddr();
    if (tdrIrqMonitorArmRegAddr == nullptr)
    {
        status = synUninitialized;
    }

    return status;
}

void DeviceScal::getDeviceHbmVirtualAddresses(uint64_t& hbmBaseAddr, uint64_t& hbmEndAddr)
{
    PoolMemoryStatus poolMemoryStatus;
    m_scalDev.getMemoryPool(ScalDev::MEMORY_POOL_GLOBAL)->getMemoryStatus(poolMemoryStatus);
    hbmBaseAddr = poolMemoryStatus.devBaseAddr;
    hbmEndAddr  = poolMemoryStatus.devBaseAddr + poolMemoryStatus.totalSize;
}
