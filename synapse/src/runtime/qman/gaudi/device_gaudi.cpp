#include "device_gaudi.hpp"

#include "defenders.h"
#include "define_synapse_common.hpp"
#include "gaudi/asic_reg/gaudi_blocks.h"
#include "gaudi/gaudi.h"
#include "habana_global_conf.h"
#include "habana_global_conf_runtime.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "profiler_api.hpp"
#include "recipe/recipe_utils.hpp"
#include "runtime/common/common_types.hpp"
#include "runtime/common/osal/osal.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/streams/stream.hpp"
#include "runtime/common/streams/stream_job.hpp"
#include "runtime/qman/common/coeff_table_configuration_manager.hpp"
#include "runtime/qman/common/command_buffer.hpp"
#include "runtime/qman/common/command_buffer_packet_generator.hpp"
#include "runtime/qman/common/command_submission.hpp"
#include "runtime/qman/common/command_submission_builder.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/data_chunk/data_chunk.hpp"
#include "runtime/qman/common/master_qmans_definition_interface.hpp"
#include "runtime/qman/common/queue_compute_qman.hpp"
#include "runtime/qman/common/stream_wait_for_event_qman.hpp"
#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "syn_singleton.hpp"
#include "synapse_common_types.h"
#include "types_exception.h"

#include <iostream>

#define GENERATE_ARBITRATION_COMMAND(isRelease, arbPriority)                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        status =                                                                                                       \
            pCmdBuffPktGenerator->generateArbitrationCommand(pWriteBuffer, arbCommandSize, isRelease, arbPriority);    \
        if (unlikely(status != synSuccess))                                                                            \
        {                                                                                                              \
            LOG_ERR(SYN_API, "Can not generate arbitration-set command");                                              \
            status = synInvalidArgument;                                                                               \
            break;                                                                                                     \
        }                                                                                                              \
                                                                                                                       \
        pWriteBuffer += arbCommandSize;                                                                                \
    } while (0)

#define GENERATE_LINDMA_COMMAND(engBarrierRequired)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        status = pCmdBuffPktGenerator->generateLinDmaPacket(pWriteBuffer,                                              \
                                                            commandSize,                                               \
                                                            currentSourceAddress,                                      \
                                                            currentDestinationAddress,                                 \
                                                            currentLinDmaSize,                                         \
                                                            direction,                                                 \
                                                            linDmaContextId,                                           \
                                                            isMemset,                                                  \
                                                            engBarrierRequired);                                       \
        if (unlikely(status != synSuccess))                                                                            \
        {                                                                                                              \
            LOG_ERR(SYN_API, "Can not generate Lin-DMA command");                                                      \
            status = synInvalidArgument;                                                                               \
            break;                                                                                                     \
        }                                                                                                              \
                                                                                                                       \
        pWriteBuffer += sizeOfLinDmaCommand;                                                                           \
    } while (0)

#define GENERTAE_WRAPPED_LINDMA_COMMAND(isArbitrationRequired, arbitrationPriority, engBarrierStatus)                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (isArbitrationRequired)                                                                                     \
        {                                                                                                              \
            GENERATE_ARBITRATION_COMMAND(false, generic::ARB_PRIORITY_NORMAL);                                         \
            GENERATE_LINDMA_COMMAND(engBarrierStatus);                                                                 \
            GENERATE_ARBITRATION_COMMAND(true, arbitrationPriority);                                                   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            GENERATE_LINDMA_COMMAND(engBarrierStatus);                                                                 \
        }                                                                                                              \
    } while (0)

const uint32_t        DeviceGaudi::INVALID_PHYSICAL_QUEUE_ID = std::numeric_limits<int32_t>::max();
std::atomic<uint32_t> DeviceGaudi::s_linDmaContextId {0};

const AffinityCountersArray DeviceGaudi::s_maxAffinitiesDefault           = {1, 1, 1, 2, 1};
const AffinityCountersArray DeviceGaudi::s_maxAffinitiesHCLDisable        = {1, 1, 1, 2, 0};
const AffinityCountersArray DeviceGaudi::s_allocationAffinitiesDefault    = {1, 1, 1, 1, 1};
const AffinityCountersArray DeviceGaudi::s_allocationAffinitiesHCLDisable = {1, 1, 1, 1, 0};

const std::vector<uint64_t> DeviceGaudi::s_tpcAddrVector {mmKERNEL_TPC0_CFG_BASE,
                                                          mmKERNEL_TPC1_CFG_BASE,
                                                          mmKERNEL_TPC2_CFG_BASE,
                                                          mmKERNEL_TPC3_CFG_BASE,
                                                          mmKERNEL_TPC4_CFG_BASE,
                                                          mmKERNEL_TPC5_CFG_BASE,
                                                          mmKERNEL_TPC6_CFG_BASE,
                                                          mmKERNEL_TPC7_CFG_BASE};

// Create the Slaves list for compute stream, excluding masterQmanId
static void _createSlavesListForComputeStream(generic::CommonQmansIdDB& slaveQmansId,
                                              uint32_t                  masterQmanId,
                                              generic::CommonQmansIdDB& disabledQmansId)
{
    using namespace gaudi;

    HB_ASSERT((masterQmanId < GAUDI_ENGINE_ID_SIZE), "Invalid qman-id");

    slaveQmansId.clear();
    // MME
    if (GAUDI_ENGINE_ID_MME_0 != masterQmanId)
    {
        slaveQmansId.push_back(GAUDI_ENGINE_ID_MME_0);
    }

    if (GAUDI_ENGINE_ID_MME_2 != masterQmanId)
    {
        slaveQmansId.push_back(GAUDI_ENGINE_ID_MME_2);
    }
    // --- MME1 is a Slave-MME
    // --- MME3 is a Slave-MME
    // Inner DMA
    uint32_t dmaEngineId = GAUDI_ENGINE_ID_DMA_2;
    while (dmaEngineId <= GAUDI_ENGINE_ID_DMA_7)
    {
        if (dmaEngineId != masterQmanId)
        {
            if (dmaEngineId != GAUDI_ENGINE_ID_DMA_5)
            {
                slaveQmansId.push_back(dmaEngineId);
            }
        }
        dmaEngineId++;
    }
    // TPC
    uint32_t    tpcEngineId    = GAUDI_ENGINE_ID_TPC_0;
    uint32_t    tpcEnginesMask = GCFG_TPC_ENGINES_ENABLED_MASK.value();
    const auto& halReader      = GaudiHalReader::instance(synDeviceGaudi);
    auto        numTpcEngines  = halReader->getNumTpcEngines();
    for (uint32_t i = 0; i < numTpcEngines; i++, tpcEngineId++)
    {
        if (tpcEngineId == masterQmanId)
        {
            continue;
        }

        uint32_t currentTpcEngineMask = tpcEnginesMask & (1 << i);
        if (currentTpcEngineMask != 0)
        {
            slaveQmansId.push_back(tpcEngineId);
        }
        else
        {
            disabledQmansId.push_back(tpcEngineId);
        }
    }
}

DeviceGaudi::DeviceGaudi(const DeviceConstructInfo& deviceConstructInfo)
: DeviceCommon(synDeviceGaudi,
               new DevMemoryAllocCommon(synDeviceGaudi,
                                        deviceConstructInfo.deviceInfo.dramSize,
                                        deviceConstructInfo.deviceInfo.dramBaseAddress),
               deviceConstructInfo,
               false,
               GCFG_INIT_HCCL_ON_ACQUIRE.value() ? s_maxAffinitiesDefault : s_maxAffinitiesHCLDisable),
  m_deviceRecipeAddressesGenerator(m_devType, *m_devMemoryAlloc),
  m_multiCsQuerier(m_osalInfo.fd),
  m_workCompletionManager(&m_multiCsQuerier),
  m_deviceMapper(*m_devMemoryAlloc),
  m_pQmansDefinition(gaudi::QmansDefinition::getInstance()),
  m_pCmdBuffPktGenerator(gaudi::CommandBufferPktGenerator::getInstance()),
  m_allocationAffinities(GCFG_INIT_HCCL_ON_ACQUIRE.value() ? s_allocationAffinitiesDefault
                                                           : s_allocationAffinitiesHCLDisable),
  m_deviceStreams(m_devType,
                  getAmountOfEnginesInComputeArbGroupAquire(),
                  *m_devMemoryAlloc,
                  m_deviceRecipeAddressesGenerator,
                  m_workCompletionManager),
  m_pDeviceDownloader(nullptr)
{
}

synStatus DeviceGaudi::acquire(const uint16_t numSyncObj)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    LOG_INFO_T(SYN_DEVICE, "{} devType 0x{:x}", HLLOG_FUNC, m_devType);

    synStatus status = deviceAcquireConfig();
    if (status != synSuccess)
    {
        return status;
    }
    if (GCFG_GAUDI_DEMO.value())
    {
        HB_ASSERT(false, "GAUDI_DEMO is no longer supported");
    }

    m_workCompletionManager.start();

    status = allocateResources(numSyncObj);
    if (status != synSuccess)
    {
        return status;
    }

    status = startEventFdThread();
    if (status != synSuccess)
    {
        return status;
    }

    if (submitPredicateDefaultConfiguration() != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not submit predicate configuration to device", HLLOG_FUNC);
        return synFail;
    }

    std::unique_ptr<CoeffTableConfManager> coeffTableconfManager = createCoeffTableConfManager(this);
    if (coeffTableconfManager->submitCoeffTableConfiguration(m_devType) != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not submit coeff table configuration to device", HLLOG_FUNC);
        return synFail;
    }

    lock.unlock();

    if (GCFG_INIT_HCCL_ON_ACQUIRE.value())
    {
        // In a different manner from Gaudi2/3 we unlocked the mutex and call hcclInitDevice at the end of acquire phase
        // since it might create internal streams
        if (hcclInitDeviceGaudi(0) != hcclSuccess)
        {
            LOG_ERR(SYN_API, "{}: Failed to initialize HCCL device for device", HLLOG_FUNC);
            return synFail;
        }
    }

    synapse::LogManager::instance().clearLogContext();

    return status;
}

synStatus DeviceGaudi::deviceAcquireConfig()
{
    // Basically could be done on the same CS as the above, but... ["it is the begining of the world"]
    if (submitArbitratorsDefaultConfigurationForGaudi() != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Can not reset device sync-objects", HLLOG_FUNC);
        return synFail;
    }
    return synSuccess;
}

synStatus DeviceGaudi::release(std::atomic<bool>& rDeviceBeingReleased)
{
    LOG_TRACE_T(SYN_DEVICE, "{}", HLLOG_FUNC);

    if (GCFG_INIT_HCCL_ON_ACQUIRE.value() && ((m_devType == synDeviceGaudi)))
    {
        if (hcclDestroyDevice(0) != hcclSuccess)
        {
            LOG_ERR(SYN_API, "{}: Failed to destroy HCCL device", HLLOG_FUNC);
            return synFail;
        }
    }
    rDeviceBeingReleased = true;

    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    synStatus totalStatus = synSuccess;
    synStatus status;

    stopWorkCompletionManager();

    if ((status = releaseResources()) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: releaseResources failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }

    if ((status = stopEventFdThread()) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: stopEventFdThread failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }

    if ((status = m_devMemoryAlloc->destroyHostAllocations(false)) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: destroyProgramsAllocations failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }
    if ((status = m_devMemoryAlloc->destroyHostAllocations(true)) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: destroyUserAllocations failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }

    dumpCsStatistics();

    uint32_t numOfDestroyedElem = 0;
    if ((status = CommandBufferMap::GetInstance()->Clear(numOfDestroyedElem)) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: Clear() failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }

    if ((status = OSAL::getInstance().releaseAcquiredDeviceBuffers()) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: releaseAcquiredDeviceBuffers failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }

    if ((status = OSAL::getInstance().releaseAcquiredDevice()) != synSuccess)
    {
        LOG_ERR_T(SYN_DEVICE, "{}: releaseAcquiredDevice failed with synStatus {}.", HLLOG_FUNC, status);
        totalStatus = synFail;
    }

    return totalStatus;
}

synStatus DeviceGaudi::getDramMemInfo(uint64_t& free, uint64_t& total) const
{
    return m_devMemoryAlloc->getDramMemInfo(free, total);
}

synStatus DeviceGaudi::getDeviceInfo(synDeviceInfo& rDeviceInfo) const
{
    rDeviceInfo = m_osalInfo;
    return synSuccess;
}

/***************************************************************************************************/
/*                                                                                                 */
/*                                         Stream operations                                       */
/*                                                                                                 */
/***************************************************************************************************/
synStatus
DeviceGaudi::createStreamQueue(QueueType queueType, uint32_t flags, bool isReduced, QueueInterface*& rpQueueInterface)
{
    internalStreamType internalStreamType;
    synStatus          status = getInternalStreamTypes(queueType, &internalStreamType);
    if (status != synSuccess)
    {
        return status;
    }

    return m_deviceStreams.createStream(internalStreamType,
                                        flags,
                                        isReduced,
                                        *m_pDeviceRecipeDownloaderContainer,
                                        rpQueueInterface);
}

synStatus DeviceGaudi::destroyStreamQueue(QueueInterface* pQueueInterface)
{
    return m_deviceStreams.destroyStream(pQueueInterface);
}

// Note: the method assumes that m_mutex was already taken on DeviceQman::release()
synStatus DeviceGaudi::deallocateRecipesAddresses()
{
    while (!m_deviceAddressesToReleaseOnStreamDestroy.empty())
    {
        const uint32_t flags   = synMemFlags::synMemDevice;
        void*          pBuffer = reinterpret_cast<void*>(m_deviceAddressesToReleaseOnStreamDestroy.front());

        if (deallocateMemory(pBuffer, flags, true) != synSuccess)
        {
            LOG_ERR(SYN_DEVICE, "Can not deallocateMemory for tensorAddr {:p}", pBuffer);
            return synFail;
        }
        m_deviceAddressesToReleaseOnStreamDestroy.pop();
    }
    return synSuccess;
}

synStatus DeviceGaudi::synchronizeEvent(const EventInterface* pEventInterface)
{
    const QmanEvent* pQmanEvent = dynamic_cast<const QmanEvent*>(pEventInterface);
    CHECK_POINTER(SYN_STREAM, pQmanEvent, "pQmanEvent", synInvalidArgument);

    const QmanEvent& rQmanEvent = *pQmanEvent;

    LOG_INFO(SYN_STREAM, "{}: {}", HLLOG_FUNC, rQmanEvent.toString());

    if (rQmanEvent.isInternalSignalingEvent())
    {
        LOG_ERR(SYN_DEVICE,
                "{}: event synchronize when mapped to external tensor not supported, sequence offset {}, mapped tensor "
                "id {} ",
                HLLOG_FUNC,
                rQmanEvent.getSequenceOffset(),
                rQmanEvent.getTensorIdx());
        return synUnsupported;
    }

    return rQmanEvent.synchronizeEvent();
}

synStatus DeviceGaudi::eventQuery(const EventInterface* pEventHandle)
{
    const QmanEvent* pInternalEventHandle = dynamic_cast<const QmanEvent*>(pEventHandle);
    CHECK_POINTER(SYN_STREAM, pInternalEventHandle, "pInternalEventHandle", synInvalidArgument);

    LOG_INFO(SYN_STREAM, "{}: {}", HLLOG_FUNC, pInternalEventHandle->toString());

    return pInternalEventHandle->eventHandleQuery();
}

synStatus DeviceGaudi::createEvent(synEventHandle* pEventHandle, const unsigned int flags)
{
    return m_deviceStreams.createEvent(pEventHandle, flags);
}

synStatus DeviceGaudi::destroyEvent(synEventHandle eventHandle)
{
    return m_deviceStreams.destroyEvent(eventHandle);
}

EventSptr DeviceGaudi::getEventSptr(synEventHandle eventHandle)
{
    return m_deviceStreams.getEventSptr(eventHandle);
}

TrainingRetCode DeviceGaudi::validateEventHandle(const EventInterface* pEventHandle)
{
    const QmanEvent* pInternalEventHandle = dynamic_cast<const QmanEvent*>(pEventHandle);
    CHECK_POINTER(SYN_DEVICE, pInternalEventHandle, "eventHandle", TRAINING_RET_CODE_INVALID_REQUEST);

    return m_deviceStreams.validateEventHandle(pInternalEventHandle);
}

synStatus DeviceGaudi::getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const
{
    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    return m_deviceStreams.getDeviceTotalStreamMappedMemory(totalStreamMappedMemorySize);
}

void DeviceGaudi::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::shared_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    m_deviceStreams.notifyRecipeRemoval(rRecipeHandle);

    if (m_pDeviceRecipeDownloaderContainer)
    {
        m_pDeviceRecipeDownloaderContainer->removeDeviceRecipeInfo(rRecipeHandle);
    }
}

void DeviceGaudi::checkDevFailure(uint64_t csSeqTimeout, DfaStatus dfaStatus, ChkDevFailOpt option, bool isSimulator)
{
    if (option == ChkDevFailOpt::CCB) return;  // CCB is not supported on qman

    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("Streams details"));
    LOG_TRACE(SYN_DEV_FAIL, "Total launches until now {}", QueueComputeUtils::getCurrnetSeqId());

    dfaInfo(DfaReq::STREAM_INFO, 0);

    const char* title = (csSeqTimeout == 0) ? "No cs from LKD, Oldest work in each stream" : "Showing cs given by LKD";
    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("{}", title));
    dfaInfo(DfaReq::ERR_WORK, csSeqTimeout);

    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("All cs-s"));
    dfaInfo(DfaReq::ALL_WORK, 0);

    m_deviceStreams.logStreamsSyncHistory();

    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("Log CsDc"));
    dfaInfo(DfaReq::PARSE_CSDC, csSeqTimeout);

    LOG_TRACE(SYN_DEV_FAIL, TITLE_STR("Logging HCL DFA Info"));
    hcclDFA(dfaStatus, dfaLogFunc);
}

void DeviceGaudi::notifyAllRecipeRemoval()
{
    m_deviceStreams.notifyAllRecipeRemoval();

    if (m_pDeviceRecipeDownloaderContainer)
    {
        m_pDeviceRecipeDownloaderContainer->removeAllDeviceRecipeInfo();
    }
}

void DeviceGaudi::addAddrToReleaseOnStreamDestroy(uint64_t addr)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    m_deviceAddressesToReleaseOnStreamDestroy.push(addr);
}

synStatus DeviceGaudi::allocateResources(const uint16_t numSyncObj)
{
    try
    {
        // this does a new, might throw
        m_deviceStreams.init(numSyncObj);
    }
    catch (const SynapseException& exc)
    {
        LOG_ERR(SYN_DEVICE, "Cant create Training-Signal-Manager due to {}", exc.what());
        return synFail;
    }

    synStatus status = allocateDevMem();

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Device allocateDevMem failed with status {}", status);
        return status;
    }

    status = allocateSharedStreams();

    if (status != synSuccess)
    {
        // Todo SW-71415 Device QMA does not rolled back allocation on failures
        LOG_ERR(SYN_DEVICE, "Device allocateSharedStreams failed with status {}", status);
        return status;
    }

    return synSuccess;
}

synStatus DeviceGaudi::allocateDevMem()
{
    synStatus status = m_devMemoryAlloc->allocate();

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "m_devMemoryAlloc allocate failed with status {}", status);
        return status;
    }

    status = m_deviceRecipeAddressesGenerator.allocate();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "m_deviceRecipeAddressesGenerator allocate failed with status {}", status);
        return status;
    }

    return synSuccess;
}

synStatus DeviceGaudi::allocateSharedStreams()
{
    synStatus status = m_deviceStreams.createDownSynStream();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to create synapse stream(s) {}", status);
        return synFail;
    }

    m_pDeviceDownloader.reset(new DeviceDownloader(*m_deviceStreams.getDmaDownStream()));

    m_pDeviceRecipeDownloaderContainer.reset(new DeviceRecipeDownloaderContainer(m_devType,
                                                                                 m_deviceRecipeAddressesGenerator,
                                                                                 *m_pQmansDefinition,
                                                                                 *m_pCmdBuffPktGenerator,
                                                                                 *m_pDeviceDownloader,
                                                                                 m_deviceMapper));

    status = addStreamAffinities(m_allocationAffinities, false);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: addStreamAffinities failed with status {} ", HLLOG_FUNC, status);
        return status;
    }

    return synSuccess;
}

synStatus DeviceGaudi::releaseAllStreams()
{
    notifyAllRecipeRemoval();

    synStatus status = removeAllStreamAffinities();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: removeAllStreamAffinities failed with status {} ", HLLOG_FUNC, status);
        return status;
    }

    m_deviceStreams.destroyUserStreams();

    if (m_pDeviceRecipeDownloaderContainer)
    {
        m_pDeviceRecipeDownloaderContainer.reset();
    }

    if (m_pDeviceDownloader)
    {
        m_pDeviceDownloader.reset();
    }

    return m_deviceStreams.destroyDownSynStream();
}

synStatus DeviceGaudi::getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress)
{
    return m_deviceRecipeAddressesGenerator.getCacheDeviceAddressRange(baseAddress, lastAddress);
}

synStatus DeviceGaudi::allocateRecipeMemory(const synRecipeHandle      recipeHandle,
                                            const synLaunchTensorInfo* launchTensorsInfo,
                                            const uint32_t             numberTensors,
                                            DeviceRecipeMemory&        recipeMem,
                                            bool&                      isMemoryAddedToReleaseList)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    const uint32_t flags = synMemFlags::synMemDevice;

    auto mapIter = m_recipeToMemoryMap.find(recipeHandle);
    if (mapIter != m_recipeToMemoryMap.end())
    {
        recipeMem                  = mapIter->second;
        isMemoryAddedToReleaseList = true;
        return synSuccess;
    }

    synStatus status = _SYN_SINGLETON_->getTopologyWorkspaceSize(&recipeMem.m_workspace.m_memSize, recipeHandle);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "recipeHandle 0x{:x} getTopologyWorkspaceSize failed", (uint64_t)recipeHandle);
        return synFail;
    }

    if (recipeMem.m_workspace.m_memSize > 0)
    {
        status = _SYN_SINGLETON_->allocateDeviceMemory(0,
                                                       recipeMem.m_workspace.m_memSize,
                                                       flags,
                                                       (void**)&recipeMem.m_workspace.m_memAddr);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "recipeHandle 0x{:x} workspaceSize 0x{:x} allocateMemory failed",
                    (uint64_t)recipeHandle,
                    recipeMem.m_workspace.m_memSize);
            return synFail;
        }
    }
    else
    {
        recipeMem.m_workspace.m_memAddr = (uint64_t) nullptr;
    }

    recipeMem.m_tensors.resize(numberTensors);

    for (uint32_t index = 0; index < numberTensors; index++)
    {
        if (launchTensorsInfo[index].tensorName == nullptr)
        {
            LOG_ERR(SYN_DEVICE, "{}: tensor-name is null", HLLOG_FUNC);
            return synFail;
        }

        size_t tensorIndex;
        for (tensorIndex = 0; tensorIndex < recipeHandle->basicRecipeHandle.recipe->persist_tensors_nr; tensorIndex++)
        {
            const char* currentTensorName = (*recipeHandle).basicRecipeHandle.recipe->tensors[tensorIndex].name;

            // match the tensor name the user provided  with the tensor name in the recipe
            if (strcmp(currentTensorName, launchTensorsInfo[index].tensorName) == 0)
            {
                break;
            }
        }

        if (tensorIndex == recipeHandle->basicRecipeHandle.recipe->persist_tensors_nr)
        {
            LOG_ERR(SYN_DEVICE, "{}: Can not find tensor-name {}", HLLOG_FUNC, launchTensorsInfo[index].tensorName);
            return synFail;
        }

        recipeMem.m_tensors[index].m_memSize = recipeHandle->basicRecipeHandle.recipe->tensors[tensorIndex].size;
        if (recipeMem.m_tensors[index].m_memSize == 0)
        {
            LOG_ERR(SYN_DEVICE, "{}: zero size tensor {}", HLLOG_FUNC, launchTensorsInfo[index].tensorName);
            return synFail;
        }

        status = _SYN_SINGLETON_->allocateDeviceMemory(0,
                                                       recipeMem.m_tensors[index].m_memSize,
                                                       flags,
                                                       (void**)&recipeMem.m_tensors[index].m_memAddr);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "recipeHandle 0x{:x} tensorSize 0x{:x} allocateMemory failed",
                    (uint64_t)recipeHandle,
                    recipeMem.m_tensors[index].m_memSize);
            return synFail;
        }
    }
    isMemoryAddedToReleaseList        = false;
    m_recipeToMemoryMap[recipeHandle] = recipeMem;

    return synSuccess;
}

synStatus DeviceGaudi::releaseRecipeMemory(const DeviceRecipeMemory& recipeMem, const synRecipeHandle recipeHandle)
{
    STAT_GLBL_START(deviceMutexDuration);
    std::unique_lock lock(m_mutex);
    STAT_GLBL_COLLECT_TIME(deviceMutexDuration, globalStatPointsEnum::deviceMutexDuration);

    const uint32_t flags = synMemFlags::synMemDevice;

    auto mapIter = m_recipeToMemoryMap.find(recipeHandle);
    if (mapIter == m_recipeToMemoryMap.end())
    {
        return synSuccess;
    }

    for (const auto& tensor : recipeMem.m_tensors)
    {
        if (tensor.m_memSize > 0)
        {
            synStatus status =
                _SYN_SINGLETON_->deallocateDeviceMemory(0, reinterpret_cast<void*>(tensor.m_memAddr), flags);

            if (status != synSuccess)
            {
                LOG_ERR(SYN_DEVICE,
                        "tensorSize 0x{:x} tensorAddr 0x{:x} deallocateMemory failed",
                        tensor.m_memSize,
                        tensor.m_memAddr);
                return synFail;
            }
        }
    }

    if (recipeMem.m_workspace.m_memSize > 0)
    {
        synStatus status =
            _SYN_SINGLETON_->deallocateDeviceMemory(0, reinterpret_cast<void*>(recipeMem.m_workspace.m_memAddr), flags);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_DEVICE,
                    "workspaceSize 0x{:x} workspaceAddr 0x{:x} deallocateMemory failed",
                    recipeMem.m_workspace.m_memSize,
                    recipeMem.m_workspace.m_memAddr);
            return synFail;
        }
    }

    m_recipeToMemoryMap.erase(mapIter);
    return synSuccess;
}

/***************************************************************************************************/
/*                                                                                                 */
/*                                      Stream operations End                                      */
/*                                                                                                 */
/***************************************************************************************************/

synStatus DeviceGaudi::releaseDevMem()
{
    synStatus status = m_deviceRecipeAddressesGenerator.release();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "m_deviceRecipeAddressesGenerator release failed with status {}", status);
        return status;
    }

    status = m_devMemoryAlloc->release();

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "m_devMemoryAlloc release failed with status {}", status);
        return status;
    }

    return status;
}

void DeviceGaudi::stopWorkCompletionManager()
{
    m_deviceStreams.finalizeStreams();
    m_workCompletionManager.stop();
}

synStatus DeviceGaudi::releaseResources()
{
    synStatus status = synSuccess;

    if (releaseAllStreams() != synSuccess)
    {
        status = synFail;
    }

    if (releaseDevMem() != synSuccess)
    {
        status = synFail;
    }

    m_deviceStreams.finalize();

    return status;
}

synStatus DeviceGaudi::launch(Stream*                       pStream,
                             const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsInfoAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             EventWithMappedTensorDB&      events,
                             uint32_t                      flags)
{
    if (RecipeUtils::isSfg(*pRecipeHandle))
    {
        LOG_ERR(SYN_DEVICE, "SFG is not supported on Gaudi device, recipe {:x}", TO64(pRecipeHandle));
        // TODO: uncomment this return value once BE fix SFG use for Gaudi
        // return synFail;
    }

    uint64_t                   assertAsyncMappedAddress = (uint64_t)getAssertAsyncMappedAddress();
    std::unique_ptr<StreamJob> job                      = std::make_unique<ComputeJob>(launchTensorsInfo,
                                                                  launchTensorsInfoAmount,
                                                                  workspaceAddress,
                                                                  pRecipeHandle,
                                                                  assertAsyncMappedAddress,
                                                                  flags,
                                                                  events,
                                                                  0);
    synStatus                  status                   = m_streamsContainer.addJob(pStream, job);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "{}: Failed to launch status {}", HLLOG_FUNC, status);
        return status;
    }

    if (RecipeUtils::isKernelPrintf(*pRecipeHandle))
    {
        kernelsPrintfAfterLaunch(*pRecipeHandle, workspaceAddress);
    }

    return synSuccess;
}

void DeviceGaudi::kernelsPrintfAfterLaunch(InternalRecipeHandle& rInternalRecipeHandle, uint64_t workspaceAddress)
{
    synchronizeAllStreams();
    kernelsPrintf(rInternalRecipeHandle, workspaceAddress, nullptr);
}

synStatus DeviceGaudi::submitLinDmaCommand(const internalMemcopyParams& rMemcpyParams,
                                           internalDmaDir               direction,
                                           bool                         isArbitrationRequired,
                                           PhysicalQueuesId             physicalQueueId,
                                           InternalWaitHandle*          waitHandle,
                                           DataChunksDB&                rDataChunks,
                                           CommandSubmissionDataChunks* pCsDataChunks,
                                           bool                         isUserRequest,
                                           bool                         isMemset,
                                           bool                         isInspectCopiedContent,
                                           uint64_t                     maxLinDmaBufferSize,
                                           uint64_t                     arbCommandSize,
                                           uint64_t                     sizeOfLinDmaCommand,
                                           uint64_t                     sizeOfWrappedLinDmaCommand,
                                           uint64_t                     sizeOfSingleCommandBuffer)
{
    const uint64_t offset = 0;
    VERIFY_IS_NULL_POINTER(SYN_API, pCsDataChunks, "CS Data-Chunks");

    DataChunksDB::iterator dataChunkEndIter = rDataChunks.end();
    DataChunksDB::iterator dataChunkIter    = rDataChunks.begin();
    DataChunk*             pDataChunk       = (*dataChunkIter);

    HB_ASSERT((pDataChunk != nullptr), "Got a nullptr Data-Chunk for Lin-DMA operation");

    char* pWriteBuffer = (char*)pDataChunk->getChunkBuffer();

    uint64_t singleDcSize = pDataChunk->getChunkSize();
    HB_ASSERT((singleDcSize % sizeOfWrappedLinDmaCommand) == 0,
              "DC size is not a multiple of wrapped packet size",
              "DC size ({}) is not a multiple of wrapped packet size ({})",
              singleDcSize,
              sizeOfWrappedLinDmaCommand);

    CommandSubmission* pLinDmaCS = nullptr;
    CommandSubmission  linDmaCS;
    if (waitHandle != nullptr)
    {
        pLinDmaCS = new CommandSubmission(m_devType == synDeviceGaudi);
    }
    else
    {
        pLinDmaCS = &linDmaCS;
    }

    uint64_t recipeCacheBaseAddress          = 0;
    uint64_t recipeCacheLastAddress          = 0;
    bool     isRecipeCacheValidationRequired = g_recipeCacheOverrunValidation && isUserRequest;
    if (isRecipeCacheValidationRequired)
    {
        if (!m_deviceRecipeAddressesGenerator.isRecipeCacheValid())
        {
            isRecipeCacheValidationRequired = false;
        }
        else
        {
            m_deviceRecipeAddressesGenerator.getCacheDeviceAddressRange(recipeCacheBaseAddress, recipeCacheLastAddress);
        }
    }

    synStatus                           status                = synSuccess;
    const synDmaDir                     dir                   = getDir(direction);
    generic::CommandBufferPktGenerator* pCmdBuffPktGenerator  = _getCommandBufferGeneratorInstance(m_devType);
    bool                                csDeletionRequired    = (waitHandle != nullptr);
    bool                                synCbDeletionRequired = false;

    STAT_GLBL_START(streamCopySubmit);

    std::unique_lock<std::mutex> guard(pLinDmaCS->getMutex());

    STAT_GLBL_START(streamCopySubmitPrepare);

    uint32_t engBarrierStatus         = 1;
    uint64_t totalWrappedPacketsCount = 0;

    if ((pCsDataChunks != nullptr) && isInspectCopiedContent)
    {
        pCsDataChunks->hostRangeMappingReserveSize(rMemcpyParams.size());
    }

    for (internalMemcopyParams::const_iterator entry = rMemcpyParams.begin();
         (status == synSuccess) && entry != rMemcpyParams.end();
         ++entry)
    {
        uint64_t srcAddress = entry->src;
        uint64_t dstAddress = entry->dst;
        uint64_t size       = entry->size;

        if (unlikely(size == 0))
        {
            LOG_TRACE(SYN_API, "{}: Got element with 0 as size", HLLOG_FUNC);
            continue;
        }

        eMappingStatus mappingStatus = HATVA_MAPPING_STATUS_FOUND;

        if (isRecipeCacheValidationRequired)
        {
            if (!_checkRecipeCacheOverlap(srcAddress,
                                          dstAddress,
                                          recipeCacheBaseAddress,
                                          recipeCacheLastAddress,
                                          size,
                                          dir,
                                          isMemset))
            {
                status = synInvalidArgument;
                break;
            }
        }

        if (dir == HOST_TO_DRAM)
        {
            if (!isMemset)
            {
                mappingStatus = getDeviceVirtualAddress(isUserRequest, (void*)entry->src, size, &srcAddress);
                if (mappingStatus != HATVA_MAPPING_STATUS_FOUND)
                {
                    LOG_ERR(SYN_API,
                            "Can not find a source VA for host-address 0x{:x} status {}, isUserRequest {}",
                            entry->src,
                            mappingStatus,
                            isUserRequest);
                    status = synMappingNotFound;
                    break;
                }

                if ((!isUserRequest) && (pCsDataChunks != nullptr) && isInspectCopiedContent)
                {
                    pCsDataChunks->hostRangeMappingAddEntry(entry->src, srcAddress, size);
                }
            }

            dstAddress += offset;
        }

        if (dir == DRAM_TO_HOST)
        {
            if (isMemset)
            {
                status = synInvalidArgument;
                break;
            }

            mappingStatus = getDeviceVirtualAddress(isUserRequest, (void*)entry->dst, size, &dstAddress);
            if (mappingStatus != HATVA_MAPPING_STATUS_FOUND)
            {
                LOG_ERR(SYN_API, "Can not find a destination VA for host-address 0x{:x}", entry->dst);
                status = synMappingNotFound;
                break;
            }

            srcAddress += offset;
        }

        LOG_TRACE(SYN_API,
                  "{}: LDMA 0x{:x} -> 0x{:x} (Direction {} size 0x{:x})",
                  HLLOG_FUNC,
                  srcAddress,
                  dstAddress,
                  direction,
                  size);

        uint32_t numberOfLinDmaPackets = CEIL(size, maxLinDmaBufferSize);
        uint64_t commandSize           = sizeOfWrappedLinDmaCommand * numberOfLinDmaPackets;

        uint64_t currentSourceAddress      = srcAddress;
        uint64_t currentDestinationAddress = dstAddress;

        uint64_t currentLinDmaSize     = maxLinDmaBufferSize;
        uint32_t lastLinDmaPacketIndex = numberOfLinDmaPackets - 1;
        uint64_t srcAddressUpdateSize  = !isMemset ? currentLinDmaSize : 0;

        generic::eArbitrationPriority arbitrationPriority = generic::ARB_PRIORITY_HIGH;
        if (isUserRequest)
        {
            arbitrationPriority = generic::ARB_PRIORITY_NORMAL;
        }

        // generate unique context_id for each api call related to lin dma
        uint32_t linDmaContextId = ++s_linDmaContextId;

        uint32_t entryLinDmaPktIdx           = 0;
        uint32_t currentDcLinDmaPacketsLimit = std::numeric_limits<int32_t>::max();
        // handle all packets in current memcpyParams entry except the last
        while (entryLinDmaPktIdx < lastLinDmaPacketIndex)
        {
            currentDcLinDmaPacketsLimit = pDataChunk->getFreeSize() / sizeOfWrappedLinDmaCommand;

            // calculate how many packets can fit into what is left from current buffer
            uint32_t leftoverPacketsNum            = (lastLinDmaPacketIndex - entryLinDmaPktIdx);
            uint32_t currentIterationLinDmaPackets = std::min(currentDcLinDmaPacketsLimit, leftoverPacketsNum);

            for (uint32_t bufferLinDmaPktIdx = 0; bufferLinDmaPktIdx < currentIterationLinDmaPackets;
                 bufferLinDmaPktIdx++, entryLinDmaPktIdx++, totalWrappedPacketsCount++)
            {
                GENERTAE_WRAPPED_LINDMA_COMMAND(isArbitrationRequired, arbitrationPriority, engBarrierStatus);

                currentSourceAddress += srcAddressUpdateSize;
                currentDestinationAddress += currentLinDmaSize;
                engBarrierStatus = 0;
            }

            if (unlikely(status != synSuccess))
            {
                break;
            }

            if (unlikely(!(pDataChunk->updateUsedSize(sizeOfWrappedLinDmaCommand * currentIterationLinDmaPackets))))
            {
                LOG_ERR(SYN_API, "Failed to update data-chunk with new used size");
                status = synCbFull;
                break;
            }

            if (pDataChunk->getFreeSize() < sizeOfWrappedLinDmaCommand)
            {
                pDataChunk = *(++dataChunkIter);
                if (dataChunkIter != dataChunkEndIter)
                {
                    pWriteBuffer = (char*)pDataChunk->getNextChunkAddress();
                }
            }
        }

        {  // last packet in each memcpy entry
            currentLinDmaSize = size % maxLinDmaBufferSize;

            if (currentLinDmaSize == 0)
            {
                currentLinDmaSize = maxLinDmaBufferSize;
            }

            GENERTAE_WRAPPED_LINDMA_COMMAND(isArbitrationRequired, arbitrationPriority, engBarrierStatus);
            engBarrierStatus = 0;

            if (unlikely(!(pDataChunk->updateUsedSize(sizeOfWrappedLinDmaCommand))))
            {
                LOG_ERR(SYN_API, "Failed to update data-chunk with new used size");
                status = synCbFull;
                break;
            }

            if (pDataChunk->getFreeSize() < sizeOfWrappedLinDmaCommand)
            {
                pDataChunk = *(++dataChunkIter);
                if (dataChunkIter != dataChunkEndIter)
                {
                    pWriteBuffer = (char*)pDataChunk->getNextChunkAddress();
                }
            }
        }

        if (unlikely(status != synSuccess))
        {
            break;
        }
    }

    STAT_GLBL_COLLECT_TIME(streamCopySubmitPrepare, globalStatPointsEnum::streamCopySubmitPrepare);
    STAT_GLBL_START(streamCopySubmitExecute);

    while (status == synSuccess)
    {
        status = _getPhysicalQueueId(physicalQueueId, m_devType, (dir == HOST_TO_DRAM));

        ePrimeQueueEntryType pqEntryType = PQ_ENTRY_TYPE_INTERNAL_EXECUTION;
        if (m_devType == synDeviceGaudi)
        {
            pqEntryType = PQ_ENTRY_TYPE_EXTERNAL_EXECUTION;
        }

        for (dataChunkIter = rDataChunks.begin(); dataChunkIter != dataChunkEndIter; dataChunkIter++)
        {
            pDataChunk = (*dataChunkIter);

            pLinDmaCS->addPrimeQueueEntry(pqEntryType,
                                          physicalQueueId,
                                          pDataChunk->getUsedSize(),
                                          pDataChunk->getHandle());
        }

        if (status == synSuccess)
        {
            uint64_t localHandle;
            // default queue offset index is zero
            const uint32_t physicalQueueOffset = 0;

            ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

            pCsDataChunks->addProgramBlobsDataChunks(rDataChunks);

            if ((g_validateDataChunksUsage) && (!isUserRequest))
            {
                LOG_ERR(SYN_API, "Non-User's Copy-operation Data-Chunks usage:");
                printDataChunksIds("  Data Chunks", rDataChunks);
            }

            if (g_preSubmissionCsInspection && isInspectCopiedContent)
            {
                checkForCsUndefinedOpcode(pCsDataChunks, pLinDmaCS, 0);
            }

            globalStatPointsEnum point = isUserRequest ? globalStatPointsEnum::memcpyAsyncUserSubmit
                                                       : globalStatPointsEnum::memcpyAsyncLaunchSubmit;
            pLinDmaCS->setCalledFrom("submitLinDmaCommand");

            if (submitCommandBuffers(*pLinDmaCS, &localHandle, nullptr, physicalQueueOffset, nullptr, point) !=
                synSuccess)
            {
                pCsDataChunks->clearCommandsBufferDataChunks();
                status = synInvalidArgument;
                break;
            }

            ETL_ADD_LOG_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                              logId,
                              SYN_API,
                              "Submitted CS with handle {} queueId {}. Memcopy params:",
                              localHandle,
                              physicalQueueId);

            if (g_printLdmaMemcpyParams)
            {
                for (internalMemcopyParams::const_iterator entry = rMemcpyParams.begin();
                     (status == synSuccess) && entry != rMemcpyParams.end();
                     ++entry)
                {
                    ETL_ADD_ETL_ONLY_LOG_BASIC(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                                               "src 0x{:x} dest 0x{:x} size 0x{:x} [operation's log-ID {}]",
                                               entry->src,
                                               entry->dst,
                                               entry->size,
                                               logId);
                }
            }

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
            dataChunksReady(dataChunks);
#endif
            pCsDataChunks->addWaitForEventHandle(localHandle);

            bool operationStatus = pCsDataChunks->setCommandSubmissionInstance(pLinDmaCS);
            // It is the CSDC responsibility to delete his CS
            csDeletionRequired = false;

            UNUSED(operationStatus);  // for release
            HB_ASSERT(operationStatus, "Failed to add Command-Submission instance of Lin-DMA operation, to CS-DC");

            waitHandle->handle    = localHandle;
            synCbDeletionRequired = false;
        }

        break;  // do once
    }

    STAT_GLBL_COLLECT_TIME(streamCopySubmitExecute, globalStatPointsEnum::streamCopySubmitExecute);

    guard.unlock();

    STAT_GLBL_COLLECT_TIME(streamCopySubmit, globalStatPointsEnum::streamCopySubmit);

    if (synCbDeletionRequired)
    {
        if (CommandSubmissionBuilder::getInstance()->destroyCmdSubmissionSynCBs(linDmaCS))
        {
            LOG_CRITICAL(SYN_API, "{}: Can not destroy command-submission", HLLOG_FUNC);
            // Does not affect the returned status
        }
    }

    if (csDeletionRequired)
    {
        delete pLinDmaCS;
        pLinDmaCS = nullptr;
    }

    return status;
}

synStatus
DeviceGaudi::kernelsPrintf(const InternalRecipeHandle& rInternalRecipeHandle, uint64_t workspaceAddr, void* hostBuff)
{
    synStatus               status   = synSuccess;
    char*                   buff     = nullptr;
    uint64_t                deviceVA = 0;
    uint64_t                handle   = std::numeric_limits<uint64_t>::max();
    std::unique_ptr<char[]> localBuff;
    std::vector<uint64_t>   kernelPrintAddr;

    CommandSubmission tmp_cs;
    do
    {
        if (hostBuff == nullptr)
        {
            localBuff = std::unique_ptr<char[]>(new char[GCFG_TPC_PRINTF_TENSOR_SIZE.value()]);
            buff      = localBuff.get();
        }
        else
        {
            buff = reinterpret_cast<char*>(hostBuff);
        }

        // We can malloc / map upon where buffer is allocated, but for simplicity...
        std::string mappingDesc("Kernel Printf Log");
        status = mapBufferToDevice(GCFG_TPC_PRINTF_TENSOR_SIZE.value(), buff, true, 0, mappingDesc);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API,
                    "{}: Can not map to device host-memory; buff 0x{:x} size {}",
                    HLLOG_FUNC,
                    TO64(buff),
                    GCFG_TPC_PRINTF_TENSOR_SIZE.value());
            return status;
        }

        eMappingStatus mappingStatus =
            getDeviceVirtualAddress(true, buff, GCFG_TPC_PRINTF_TENSOR_SIZE.value(), &deviceVA);
        if (mappingStatus != HATVA_MAPPING_STATUS_FOUND)
        {
            LOG_ERR(SYN_API,
                    "{}: Can not get device virtual-address for host-memory; buff {:p} size {}",
                    HLLOG_FUNC,
                    buff,
                    GCFG_TPC_PRINTF_TENSOR_SIZE.value());
            status = synMappingNotFound;
            break;
        }

        memset(buff, 0, GCFG_TPC_PRINTF_TENSOR_SIZE.value());

        const internalDmaDir dmaDir = MEMCOPY_DRAM_TO_HOST;

        tmp_cs.setNumExecuteExternalQueue(1);
        tmp_cs.setNumExecuteInternalQueue(0);
        tmp_cs.setExecuteExternalQueueCb(new synCommandBuffer[1]);
        tmp_cs.setCalledFrom("kernelsPrintf");

        const uint32_t queueId = m_pQmansDefinition->getStreamsMasterQueueIdForMemcopyFromDevice();
        status = createSynCommandBuffer(tmp_cs.getExecuteExternalQueueCb(), queueId, CommandBuffer::c_defaultCbSize);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "Can not create command buffer");
            status = synFail;
            break;
        }

        std::vector<std::string> finalPrints;
        uint32_t                 printAddrNum;
        uint64_t*                printAddrArr;
        uint64_t                 programDataAddress = 0;

        const recipe_t& rRecipe = *rInternalRecipeHandle.basicRecipeHandle.recipe;
        printAddrNum            = rRecipe.debug_profiler_info.printf_addr_nr;
        printAddrArr            = rRecipe.debug_profiler_info.printf_addr;
        auto sectionIdx         = rRecipe.debug_profiler_info.printf_section_idx;
        if (sectionIdx != MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            status = synUnsupported;
            break;
        }
        if (printAddrNum > 0)
        {
            status = m_deviceRecipeAddressesGenerator.getProgramDataAddress(rInternalRecipeHandle, programDataAddress);
            if (status != synSuccess)
            {
                break;
            }
        }

        uint32_t linDmaContextId = ++s_linDmaContextId;
        for (uint32_t iPrint = 0; iPrint < printAddrNum; iPrint++)
        {
            char*    pPacket    = nullptr;
            uint64_t packetSize = 0;

            m_pCmdBuffPktGenerator->generateLinDmaPacket(pPacket,
                                                         packetSize,
                                                         programDataAddress + maskOutMemoryID(printAddrArr[iPrint]),
                                                         deviceVA,
                                                         GCFG_TPC_PRINTF_TENSOR_SIZE.value(),
                                                         dmaDir,
                                                         linDmaContextId);

            status = CommandSubmissionBuilder::getInstance()->setBufferOnCb(*tmp_cs.getExecuteExternalQueueCb(),
                                                                            packetSize,
                                                                            pPacket);
            delete[] pPacket;
            if (status != synSuccess)
            {
                LOG_ERR(SYN_API, "Can not set buffer on command buffer");
                status = synFail;
                break;
            }

            const uint32_t queueOffset = 0;

            if (submitCommandBuffers(tmp_cs, &handle, nullptr, queueOffset, nullptr, globalStatPointsEnum::colLast) !=
                synSuccess)
            {
                LOG_ERR(SYN_API, "Can not submit command submission");
                status = synFail;
                break;
            }

            LOG_TRACE(SYN_API, "{}: Submitted CS with handle {}", HLLOG_FUNC, handle);

            if (waitAndReleaseCS(handle, SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT) != synSuccess)
            {
                LOG_ERR(SYN_API, "Can not wait for event");
                status = synFail;
                break;
            }

            parsePrintf((uint32_t*)buff, finalPrints);
        }

        // Print even in case failing on some (will assist debuging)
        for (const std::string& printf : finalPrints)
        {
            // print to screen, unless a buffer was given from user (testing of this feature)
            if (hostBuff == nullptr)
            {
                std::cout << printf;
            }
            else
            {
                LOG_DEBUG(SYN_TPC_PRINT, "Kernel print: {}", printf);
            }
        }
    } while (0);

    if (*tmp_cs.getExecuteExternalQueueCb() != nullptr)
    {
        status = CommandSubmissionBuilder::getInstance()->destroySynCommandBuffer(*tmp_cs.getExecuteExternalQueueCb());
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API,
                    "Failed to destroy buffer of command-buffer 0x{:x}",
                    (uint64_t)tmp_cs.getExecuteExternalQueueCb());
        }
    }
    unmapBufferFromDevice(buff, true, nullptr);

    return status;
}

synStatus DeviceGaudi::_getPhysicalQueueId(uint32_t& queueId, synDeviceType deviceType, bool isCopyFromHost)
{
    uint32_t physicalQueueId = 0;
    // TODO: check if can be removed completely

    switch (deviceType)
    {
        case synDeviceGaudi:
            if (queueId == INVALID_PHYSICAL_QUEUE_ID)
            {
                LOG_ERR(SYN_API, "{}: User must supply specific queue-id for the lin-dma operation", HLLOG_FUNC);
                return synInvalidArgument;
            }
            break;
        default:
            LOG_ERR(SYN_API, "{}: Invalid device-type ({})", HLLOG_FUNC, deviceType);
            return synInvalidArgument;
    }

    if (queueId == INVALID_PHYSICAL_QUEUE_ID)
    {
        queueId = physicalQueueId;
    }
    return synSuccess;
}

synStatus DeviceGaudi::submitCommandBuffers(CommandSubmission&   commandSubmission,
                                            uint64_t*            csHandle,
                                            uint64_t*            mappedBuff,
                                            uint32_t             queueOffset,
                                            const StagedInfo*    pStagedInfo,
                                            globalStatPointsEnum point)
{
    return commandSubmission.submitCommandBuffers(csHandle, nullptr, queueOffset, pStagedInfo, point);
}

void DeviceGaudi::dumpCsStatistics() const
{
    hl_info_cs_counters info;
    memset(&info, 0xFF, sizeof(struct hl_info_cs_counters));
    const int ret = hlthunk_get_cs_counters_info(m_osalInfo.fd, &info);
    if (ret != 0)
    {
        LOG_ERR(SYN_DEVICE, "hlthunk_get_cs_counters_info returned error {}", ret);
        return;
    }

    LOG_INFO(SYN_DEVICE, "total_out_of_mem_drop_cnt       {}", info.total_out_of_mem_drop_cnt);
    LOG_INFO(SYN_DEVICE, "ctx_out_of_mem_drop_cnt         {}", info.ctx_out_of_mem_drop_cnt);
    LOG_INFO(SYN_DEVICE, "total_parsing_drop_cnt          {}", info.total_parsing_drop_cnt);
    LOG_INFO(SYN_DEVICE, "ctx_parsing_drop_cnt            {}", info.ctx_parsing_drop_cnt);
    LOG_INFO(SYN_DEVICE, "total_queue_full_drop_cnt       {}", info.total_queue_full_drop_cnt);
    LOG_INFO(SYN_DEVICE, "ctx_queue_full_drop_cnt         {}", info.ctx_queue_full_drop_cnt);
    LOG_INFO(SYN_DEVICE, "total_device_in_reset_drop_cnt  {}", info.total_device_in_reset_drop_cnt);
    LOG_INFO(SYN_DEVICE, "ctx_device_in_reset_drop_cnt    {}", info.ctx_device_in_reset_drop_cnt);
    LOG_INFO(SYN_DEVICE, "total_max_cs_in_flight_drop_cnt {}", info.total_max_cs_in_flight_drop_cnt);
    LOG_INFO(SYN_DEVICE, "ctx_max_cs_in_flight_drop_cnt   {}", info.ctx_max_cs_in_flight_drop_cnt);
    LOG_INFO(SYN_DEVICE, "total_validation_drop_cnt       {}", info.total_validation_drop_cnt);
    LOG_INFO(SYN_DEVICE, "ctx_validation_drop_cnt         {}", info.ctx_validation_drop_cnt);

    STAT_GLBL_COLLECT(info.ctx_queue_full_drop_cnt, ctx_queue_full_drop_cnt);
}

synDmaDir DeviceGaudi::getDir(internalDmaDir direction)
{
    synDmaDir dir = DIRECTION_ENUM_MAX;  // invalid value
    switch (direction)
    {
        case MEMCOPY_HOST_TO_DRAM:
        case MEMCOPY_HOST_TO_SRAM:
            dir = HOST_TO_DRAM;
            break;
        case MEMCOPY_DRAM_TO_HOST:
        case MEMCOPY_SRAM_TO_HOST:
            dir = DRAM_TO_HOST;
            break;
        case MEMCOPY_DRAM_TO_DRAM:
            dir = DRAM_TO_DRAM;
            break;
        default:
            // We do not want to return or to print an error message, as it MIGHT work
            // But we do want to log it
            LOG_INFO(SYN_API, "*** Unhandled direction {} ***", direction);
            break;
    }
    return dir;
}

synStatus DeviceGaudi::submitArbitratorsDefaultConfigurationForGaudi()
{
    char*    pPackets    = nullptr;
    uint64_t packetsSize = 0;

    std::string description("arbitration default configuration");

    synStatus status = generateArbitratorsDefaultConfigPackets(pPackets, packetsSize, m_devType);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not generate packets for {}", HLLOG_FUNC, description);
        return status;
    }

    return submitTrainingConfigurationCS(m_devType,
                                         pPackets,
                                         packetsSize,
                                         description,
                                         m_pQmansDefinition->getAcquireDeviceDefaultQman());
}

synStatus
DeviceGaudi::generateArbitratorsDefaultConfigPackets(char*& pPackets, uint64_t& packetsSize, synDeviceType deviceType)
{
    using namespace gaudi;

    if (deviceType != synDeviceGaudi)
    {
        LOG_ERR(SYN_API, "{}: Invalid device-type ({})", HLLOG_FUNC, deviceType);
        return synFail;
    }

    if (pPackets != nullptr)
    {
        LOG_ERR(SYN_API, "{}: pPackets param must be nullptr", HLLOG_FUNC);
        return synFail;
    }

    packetsSize = 0;

    // Disabled arbitrators DB
    generic::CommonQmansIdDB disabledQmansId;  // Currently NA, but for completeness...

    generic::masterSlaveArbitrationInfoDB masterSlaveArbInfoDb;
    generic::MasterSlabeArbitrationInfo   singleMasterSlaveArbInfo;

    generic::masterSlavesArbitration& singleMasterSlaveArb = singleMasterSlaveArbInfo.masterSlaveArbQmans;

    uint64_t ArbitratorMasterId       = m_pQmansDefinition->getArbitratorMasterQueueIdForCompute();
    singleMasterSlaveArb.masterQmanId = m_pCmdBuffPktGenerator->getQmanId(ArbitratorMasterId);
    // Create the Slaves list for the Compute Stream, excluding masterQmanId
    _createSlavesListForComputeStream(singleMasterSlaveArb.slaveQmansId,
                                      singleMasterSlaveArb.masterQmanId,
                                      disabledQmansId);
    singleMasterSlaveArbInfo.isArbByPriority = false;
    masterSlaveArbInfoDb.push_back(singleMasterSlaveArbInfo);

    unsigned nicEnabledBitMask = 0;

    synStatus status = OSAL::getInstance().getAvailableNicElementsMask(nicEnabledBitMask);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "Failed to get NICs MAC Address information");
        return status;
    }

    uint64_t memcopyMasterQmanStreamId;

    memcopyMasterQmanStreamId = m_pQmansDefinition->getArbitratorMasterQueueIdForMemcopyToDevice();
    // In the new design, we should put all nics into the disabled DB
    //
    // The external queue of the nics is DMA_1 (NICs.1 -> DMA_1_1, NICs.2 -> DMA_1_2, NICs.3 -> DMA_1_3).
    // No arbitration for the nics.
    const auto& halReader     = GaudiHalReader::instance(synDeviceGaudi);
    auto        numNicEngines = halReader->getNumNicEngines();
    for (uint8_t port = 0; port < numNicEngines; port++)
    {
        unsigned nicId = GAUDI_ENGINE_ID_NIC_0 + port;
        if (nicEnabledBitMask & (1 << port))
        {
            disabledQmansId.push_back(nicId);
        }
    }

    // DMA5 is now used exclusively by HCL
    // DMA_5_n (n in [0-3]) is the master, and the there are no slaves
    singleMasterSlaveArb.masterQmanId =
        m_pCmdBuffPktGenerator->getQmanId(m_pQmansDefinition->getArbitratorMasterQueueIdForCollective());
    singleMasterSlaveArb.slaveQmansId.clear();
    singleMasterSlaveArbInfo.isArbByPriority = false;
    masterSlaveArbInfoDb.push_back(singleMasterSlaveArbInfo);

    singleMasterSlaveArb.masterQmanId = m_pCmdBuffPktGenerator->getQmanId(memcopyMasterQmanStreamId);
    singleMasterSlaveArb.slaveQmansId.clear();
    singleMasterSlaveArbInfo.isArbByPriority = true;
    masterSlaveArbInfoDb.push_back(singleMasterSlaveArbInfo);

    status = generateArbitratorsDefaultConfigPacketsCommon(pPackets,
                                                           packetsSize,
                                                           m_pCmdBuffPktGenerator,
                                                           masterSlaveArbInfoDb,
                                                           m_pQmansDefinition->getEnginesWithArbitrator());
    return status;
}

synStatus
DeviceGaudi::generateArbitratorsDefaultConfigPacketsCommon(char*&                                 pPackets,
                                                           uint64_t&                              packetsSize,
                                                           generic::CommandBufferPktGenerator*    pCmdBuffPktGenerator,
                                                           generic::masterSlaveArbitrationInfoDB& masterSlaveArbInfoDb,
                                                           const std::deque<uint32_t>* pEnginesWithArbitrator)
{
    synStatus status = synSuccess;

    if ((pEnginesWithArbitrator == nullptr) || (pEnginesWithArbitrator->size() == 0))
    {
        return synSuccess;
    }

    // Get packets-sizes for configuring QMANs' arbitrators different states
    uint64_t arbitratorDisabledCommandSize = 0;
    uint64_t basicMasterCommandSize        = 0;
    uint64_t masterSingleSlaveCommandSize  = 0;
    uint64_t slaveCommandSize              = 0;
    pCmdBuffPktGenerator->getArbitratorDisableConfigCommandSize(arbitratorDisabledCommandSize);
    pCmdBuffPktGenerator->getMasterArbitratorBasicConfigCommandSize(basicMasterCommandSize);
    pCmdBuffPktGenerator->getMasterSingleSlaveArbitratorConfigCommandSize(masterSingleSlaveCommandSize);
    pCmdBuffPktGenerator->getSlaveArbitratorConfigCommandSize(slaveCommandSize);

    uint32_t numOfBasicMasterCommands = 0;
    uint32_t numOfMasterSlaveCommands = 0;
    uint32_t numOfSlaveCommands       = 0;
    for (auto singleMasterSlaveArbInfoItr : masterSlaveArbInfoDb)
    {
        uint32_t numOfSlaves = singleMasterSlaveArbInfoItr.masterSlaveArbQmans.slaveQmansId.size();
        if (numOfSlaves != 0)
        {
            numOfMasterSlaveCommands += numOfSlaves;
            numOfSlaveCommands += numOfSlaves;
            numOfBasicMasterCommands++;
        }
        else
        {
            numOfBasicMasterCommands++;
        }
    }

    const std::deque<uint32_t>& enginesWithArbitrator = *pEnginesWithArbitrator;
    uint32_t                    numOfEngines          = enginesWithArbitrator.size();

    // For simplicity, we will first disable all of them
    packetsSize += numOfEngines * arbitratorDisabledCommandSize;

    packetsSize += numOfBasicMasterCommands * basicMasterCommandSize;
    packetsSize += numOfMasterSlaveCommands * masterSingleSlaveCommandSize;
    packetsSize += numOfSlaveCommands * slaveCommandSize;

    // Allocate buffer for packets
    pPackets = new char[packetsSize];

    char* pCurrPackets = pPackets;

    do  // once
    {
        // Create packets for disabled QMANs' arbitrators
        for (auto engineId : enginesWithArbitrator)
        {
            status = pCmdBuffPktGenerator->generateArbitratorDisableConfigCommand(pCurrPackets, engineId);
            if (status != synSuccess)
            {
                break;
            }

            pCurrPackets += arbitratorDisabledCommandSize;
        }
        if (status != synSuccess)
        {
            break;
        }

        // Create packets for master/slave QMANs' arbitrators
        for (auto singleMasterSlaveArbInfoItr : masterSlaveArbInfoDb)
        {
            generic::masterSlavesArbitration& masterSlaveArbQmans = singleMasterSlaveArbInfoItr.masterSlaveArbQmans;
            bool                              isArbByPriority     = singleMasterSlaveArbInfoItr.isArbByPriority;
            status = pCmdBuffPktGenerator->generateMasterArbitratorConfigCommand(pCurrPackets,
                                                                                 masterSlaveArbQmans,
                                                                                 isArbByPriority);
            if (status != synSuccess)
            {
                break;
            }
            pCurrPackets += basicMasterCommandSize;
            pCurrPackets += masterSlaveArbQmans.slaveQmansId.size() * masterSingleSlaveCommandSize;

            uint32_t numOfSlaves = masterSlaveArbQmans.slaveQmansId.size();
            if (numOfSlaves != 0)
            {
                uint32_t slaveIndex = 0;
                for (auto singleSlaveArbItr : masterSlaveArbQmans.slaveQmansId)
                {
                    status =
                        pCmdBuffPktGenerator->generateSlaveArbitratorConfigCommand(pCurrPackets,
                                                                                   slaveIndex,
                                                                                   singleSlaveArbItr,
                                                                                   masterSlaveArbQmans.masterQmanId);
                    if (status != synSuccess)
                    {
                        break;
                    }
                    slaveIndex++;
                    pCurrPackets += slaveCommandSize;
                }
            }
            if (status != synSuccess)
            {
                break;
            }
        }
    } while (0);  // Do once

    if (status != synSuccess)
    {
        delete[] pPackets;
        return synFail;
    }

    return synSuccess;
}

uint32_t DeviceGaudi::getAmountOfEnginesInComputeArbGroupAquire()
{
    // Disabled arbitrators DB
    generic::CommonQmansIdDB disabledQmansId;  // Currently NA, but for completeness...

    generic::masterSlaveArbitrationInfoDB masterSlaveArbInfoDb;
    generic::MasterSlabeArbitrationInfo   singleMasterSlaveArbInfo;

    generic::masterSlavesArbitration& singleMasterSlaveArb = singleMasterSlaveArbInfo.masterSlaveArbQmans;

    uint64_t ArbitratorMasterId       = gaudi::QmansDefinition::getInstance()->getArbitratorMasterQueueIdForCompute();
    singleMasterSlaveArb.masterQmanId = gaudi::CommandBufferPktGenerator::getInstance()->getQmanId(ArbitratorMasterId);
    // Create the Slaves list for the Compute Stream, excluding masterQmanId
    _createSlavesListForComputeStream(singleMasterSlaveArb.slaveQmansId,
                                      singleMasterSlaveArb.masterQmanId,
                                      disabledQmansId);
    singleMasterSlaveArbInfo.isArbByPriority = false;
    masterSlaveArbInfoDb.push_back(singleMasterSlaveArbInfo);

    return (singleMasterSlaveArb.slaveQmansId.size() + 1);
}

synStatus DeviceGaudi::submitPredicateDefaultConfiguration()
{
    char*                    pPackets    = nullptr;
    uint64_t                 packetsSize = 0;
    synStatus                status      = synSuccess;
    std::shared_ptr<uint8_t> table;
    uint64_t                 devPredicateTableAddr = 0;
    const unsigned           numPhysicalStreams    = 4;

    uint32_t arbMasterQmanId    = 0;
    uint32_t streamMasterQmanId = 0;

    bool isConfigOnInternal         = false;
    bool isSyncWithExternalRequired = false;

    arbMasterQmanId    = m_pQmansDefinition->getArbitratorMasterQueueIdForCompute();
    streamMasterQmanId = m_pQmansDefinition->getStreamMasterQueueIdForCompute();

    isConfigOnInternal         = true;
    isSyncWithExternalRequired = true;

    std::string description("predicates configuration");

    // generate predicate table
    status = generatePredicateDataOnHostWithMapping(numPhysicalStreams, table);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not generate predicate table", HLLOG_FUNC);
        return status;
    }

    do
    {
        std::string mappingDesc("Predicates' Data (Device)");
        status = allocateMemory((numPhysicalStreams * DEVICE_CACHE_LINE_SIZE),
                                0,
                                (void**)&devPredicateTableAddr,
                                false /* isUserRequest */,
                                0,
                                mappingDesc);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can not generate device memory", HLLOG_FUNC);
            break;
        }

        internalMemcopyParams memcpyParams {{.src  = (uint64_t)table.get(),
                                             .dst  = devPredicateTableAddr,
                                             .size = (uint64_t)(numPhysicalStreams * DEVICE_CACHE_LINE_SIZE)}};
        QueueInterface*       pStream = getDmaDownStream();

        if (pStream == nullptr)
        {
            LOG_ERR(SYN_API, "{}: Can not copy predicate table to device memory", HLLOG_FUNC);
            status = synFail;
            break;
        }

        status = pStream->memcopy(memcpyParams,
                                  MEMCOPY_HOST_TO_DRAM,
                                  false /* isUserRequest */,
                                  nullptr,
                                  0 /* overrideMemsetVal */,
                                  false /* inspectCopiedContent */,
                                  nullptr /* pRecipeProgramBuffer */,
                                  0);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can not copy predicate table to device memory", HLLOG_FUNC);
            break;
        }

        status = pStream->synchronize(nullptr, false /* isUserRequest */);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can not synchronize stream", HLLOG_FUNC);
            break;
        }

        uint64_t predicateDevAddr = devPredicateTableAddr;
        for (int offset = 0; offset < numPhysicalStreams; offset++)
        {
            status = generatePredicateConfigurationPackets(pPackets,
                                                           packetsSize,
                                                           predicateDevAddr,
                                                           isSyncWithExternalRequired);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_API, "{}: Can not generate packets for {}", HLLOG_FUNC, description);
                break;
            }

            status = submitTrainingConfigurationCS(synDeviceGaudi,
                                                   pPackets,
                                                   packetsSize,
                                                   description,
                                                   arbMasterQmanId + offset,
                                                   isConfigOnInternal,
                                                   isSyncWithExternalRequired,
                                                   streamMasterQmanId);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_API, "{}: Can not submit CS", HLLOG_FUNC);
                break;
            }

            predicateDevAddr += DEVICE_CACHE_LINE_SIZE;
        }
    } while (0);

    if (unmapBufferFromDevice((uint32_t*)table.get(), false, nullptr) != synSuccess)
    {
        LOG_ERR(SYN_API, "{}: Can not unmap host predicate table", HLLOG_FUNC);
    }

    if (devPredicateTableAddr != 0)
    {
        if (deallocateMemory((void*)devPredicateTableAddr, 0, false) != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can not deallocate device predicate table", HLLOG_FUNC);
        }
    }

    return status;
}

synStatus DeviceGaudi::generatePredicateDataOnHostWithMapping(unsigned numPreds, std::shared_ptr<uint8_t>& table)
{
    // Allocate host memory
    uint64_t                 predTableSizeInBytes = numPreds * DEVICE_CACHE_LINE_SIZE;
    std::shared_ptr<uint8_t> hostTable(new uint8_t[predTableSizeInBytes], [](uint8_t* p) { delete[] p; });

    // Fill the predicate table in host memory
    // Example:
    //   The following is a table of 8 predicates. Note that predicate 0 is reserved and shouldn't be used.
    //   Each digit represents a 32bit value, total of 128 bytes per line.
    //   Currently we only support single predicate per line.
    //
    //   31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
    //   -----------------------------------------------------------------------------------------------
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0
    //    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
    //
    std::string mappingDesc("Predicates' Data (Host)");
    synStatus   status = mapBufferToDevice(predTableSizeInBytes, (uint32_t*)hostTable.get(), false, 0, mappingDesc);
    if (status != synSuccess)
    {
        return status;
    }

    memset(hostTable.get(), 0, predTableSizeInBytes);
    uint32_t* predicateEntry = (uint32_t*)hostTable.get();
    for (unsigned i = 0, j = 1; i < numPreds; ++i, ++j)
    {
        predicateEntry[j] = 1;
        predicateEntry += (DEVICE_CACHE_LINE_SIZE / sizeof(uint32_t));  // move to next line
    }

    table = hostTable;
    return synSuccess;
}

synStatus DeviceGaudi::generatePredicateConfigurationPackets(char*&    pPackets,
                                                             uint64_t& packetsSize,
                                                             uint64_t  predAddr,
                                                             bool      isSyncWithExternalRequired)
{
    uint32_t syncId = 0;

    if (pPackets != nullptr)
    {
        LOG_ERR(SYN_API, "{}: pPackets param must be nullptr", HLLOG_FUNC);
        return synFail;
    }

    uint64_t signalPacketsSize = 0;
    uint64_t loadPacketsSize   = 0;

    if (isSyncWithExternalRequired)
    {
        signalPacketsSize = m_pCmdBuffPktGenerator->getSignalCommandSize();
    }

    m_pCmdBuffPktGenerator->getLoadAndExecCommandSize(loadPacketsSize);
    packetsSize = loadPacketsSize + signalPacketsSize;
    pPackets    = new char[packetsSize];

    synStatus status = m_pCmdBuffPktGenerator->generateLoadPredicateCommand(pPackets, predAddr);
    if (status != synSuccess)
    {
        delete[] pPackets;
        return synFail;
    }

    if (isSyncWithExternalRequired)
    {
        char* pTmpPackets = pPackets + loadPacketsSize;
        status =
            m_pCmdBuffPktGenerator->generateSignalCommand(pTmpPackets, signalPacketsSize, syncId, 1, 0, ALL_BARRIERS);
    }

    if (status != synSuccess)
    {
        delete[] pPackets;
        return synFail;
    }

    return synSuccess;
}

void DeviceGaudi::dfaInfo(DfaReq dfaReq, uint64_t csSeq)
{
    std::deque<QueueInterface*>& queueInterfaces = m_deviceStreams.getQueueInterfaces();
    LOG_DEBUG(SYN_DEV_FAIL, "Number of queues: {}", queueInterfaces.size());

    for (auto pQueueInterface : queueInterfaces)
    {
        pQueueInterface->dfaInfo(dfaReq, csSeq);
    }

    m_streamsContainer.dump(synapse::LogManager::LogType::SYN_DEV_FAIL);
}

synStatus DeviceGaudi::getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
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

synStatus DeviceGaudi::eventRecord(EventInterface* pEventInterface, synStreamHandle streamHandle)
{
    CHECK_POINTER(SYN_DEVICE, pEventInterface, "pEventInterface", synInvalidArgument);

    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    QmanEvent* pInternalEventHandle = dynamic_cast<QmanEvent*>(pEventInterface);
    CHECK_POINTER(SYN_STREAM, pInternalEventHandle, "pInternalEventHandle", synInvalidArgument);

    QmanEvent& internalEventHandle = *pInternalEventHandle;

    LOG_TRACE(SYN_STREAM, "{}: Recording over event {}", HLLOG_FUNC, pInternalEventHandle->toString());

    // Ensure that this event will not be overriden by others
    QmanEvent::lock lock(&internalEventHandle, false);

    TrainingRetCode trainingRetCode = validateEventHandle(&internalEventHandle);
    if (trainingRetCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Event validation failed on internalEventHandle 0x{:x}",
                HLLOG_FUNC,
                internalEventHandle.getHandle());
        return synFail;
    }

    internalEventHandle.clearDb();

    std::unique_ptr<StreamJob> job = std::make_unique<EventRecordJob>(*pEventInterface, streamHandle);
    return m_streamsContainer.addJob(streamSptr.get(), job);
}

synStatus DeviceGaudi::streamGenericWaitEvent(synStreamHandle       streamHandle,
                                              const EventInterface& rEventInterface,
                                              const unsigned int    flags)
{
    auto streamSptr = loadAndValidateStream(streamHandle, __FUNCTION__);
    if (streamSptr == nullptr)
    {
        return synInvalidArgument;
    }

    QmanEvent&                 rEvent = (QmanEvent&)rEventInterface;
    std::unique_ptr<StreamJob> job    = std::make_unique<WaitForEventJobQman>(rEvent, flags, streamHandle);
    return m_streamsContainer.addJob(streamSptr.get(), job);
}

synStatus DeviceGaudi::submitTrainingConfigurationCS(synDeviceType      deviceType,
                                                     char*&             pPackets,
                                                     uint64_t           packetsSize,
                                                     const std::string& operationDescription,
                                                     uint32_t           queueId,
                                                     bool               isConfigOnInternal,
                                                     bool               isSyncWithExternalRequired,
                                                     uint32_t           waitQmanId)
{
    synStatus status         = synSuccess;
    bool      isSynCbCreated = false;

    CommandSubmission* pCommandSubmission = nullptr;
    // For config-on-external OR for sync-with-external (on this case the config will be on internal)
    synCommandBuffer* pCommandBuffer = nullptr;

    do
    {
        uint16_t numOfConfigurationCbs = (isConfigOnInternal && !isSyncWithExternalRequired) ? 0 : 1;

        pCommandSubmission = new CommandSubmission(deviceType == synDeviceGaudi);

        if (isConfigOnInternal)
        {
            // Perform MMU-mapping
            uint64_t    mmuMappedAddress = 0;
            std::string mappingDesc("Training-Configuration");

            status = mapBufferToDevice(packetsSize, pPackets, false, 0, mappingDesc);
            if (status == synSuccess)
            {
                eMappingStatus mappingStatus = getDeviceVirtualAddress(false, pPackets, packetsSize, &mmuMappedAddress);
                if ((mappingStatus != HATVA_MAPPING_STATUS_FOUND) && !mmuMappedAddress)
                {
                    status = (mappingStatus != HATVA_MAPPING_STATUS_FOUND) ? synMappingNotFound : synFail;
                }
            }
            if (status != synSuccess)
            {
                LOG_ERR(SYN_API, "{}: Failed to map Config buffer (with packets-size of {})", HLLOG_FUNC, packetsSize);
                unmapBufferFromDevice(pPackets, false, nullptr);
                delete pCommandSubmission;
                return synFail;
            }

            // Add PQ entry
            pCommandSubmission->addPrimeQueueEntry(PQ_ENTRY_TYPE_INTERNAL_EXECUTION,
                                                   queueId,
                                                   packetsSize,
                                                   mmuMappedAddress);

            if (isSyncWithExternalRequired)
            {
                // using new[] as the builder performs delete[]
                const unsigned singleCb = 1;
                pCommandBuffer          = new synCommandBuffer[singleCb];
                status                  = _syncWithInnerQueueOperation(deviceType, &pCommandBuffer, waitQmanId);
            }
        }
        else
        {
            // using new[] as the builder performs delete[]
            const unsigned singleCb = 1;
            pCommandBuffer          = new synCommandBuffer[singleCb];
            status                  = DeviceCommon::createAndAddHangCommandBuffer(pCommandBuffer,
                                                                 0,
                                                                 pPackets,
                                                                 packetsSize,
                                                                 queueId,
                                                                 isConfigOnInternal);
        }

        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "{}: Can not create CB for {}", HLLOG_FUNC, operationDescription);
            if (isConfigOnInternal)
            {
                unmapBufferFromDevice(pPackets, false, nullptr);
            }

            delete[] pCommandBuffer;
            delete pCommandSubmission;

            break;
        }
        isSynCbCreated = true;

        uint64_t handle = 0;

        pCommandSubmission->setNumExecuteExternalQueue(numOfConfigurationCbs);
        pCommandSubmission->setExecuteExternalQueueCb(pCommandBuffer);

        _mapCB(pCommandSubmission);
        // default queue offset index is zero
        const uint32_t physicalQueueIndex = 0;

        ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

        pCommandSubmission->setCalledFrom("submitTrainingConfigurationCS");
        status = _SYN_SINGLETON_INTERNAL->submitCommandBuffers(*pCommandSubmission,
                                                               &handle,
                                                               nullptr,
                                                               physicalQueueIndex,
                                                               nullptr,
                                                               globalStatPointsEnum::colLast);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "Can not submit execute command-submission for {}", operationDescription);
            break;
        }

        ETL_ADD_LOG_TRACE(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                          logId,
                          SYN_API,
                          "Submitted {} CS with handle {}",
                          operationDescription,
                          handle);

        status = waitAndReleaseCS(handle, SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_API, "Can not wait for event for {}", operationDescription);
            break;
        }
    } while (0);  // Do once

    if (isConfigOnInternal)
    {
        unmapBufferFromDevice(pPackets, false, nullptr);
    }

    delete[] pPackets;
    pPackets = nullptr;

    if (isSynCbCreated)
    {
        if (CommandSubmissionBuilder::getInstance()->destroyCmdSubmissionSynCBs(*pCommandSubmission))
        {
            LOG_CRITICAL(SYN_API, "{}: Can not unmap packets for {}", HLLOG_FUNC, operationDescription);
        }

        delete pCommandSubmission;
    }

    return status;
}

synStatus DeviceGaudi::_syncWithInnerQueueOperation(synDeviceType      deviceType,
                                                    synCommandBuffer** ppCommandBuffers,
                                                    uint32_t           waitQmanId)
{
    // Addig a NOP packet is a hack since we can't submit just the setup without execution.
    generic::CommandBufferPktGenerator* pCmdBuffPktGenerator = nullptr;

    uint32_t syncId    = 0;
    uint32_t monitorId = 0;
    if (deviceType == synDeviceGaudi)
    {
        pCmdBuffPktGenerator = gaudi::CommandBufferPktGenerator::getInstance();
        syncId               = 0;
        monitorId            = 0;
    }
    else
    {
        LOG_ERR(SYN_API, "{}: Invalid device-type ({})", HLLOG_FUNC, deviceType);
        return synFail;
    }

    synStatus status = synFail;
    do
    {
        BREAK_AND_SET_FAIL_STATUS_IF_NULL_POINTER(SYN_API,
                                                  pCmdBuffPktGenerator,
                                                  "Command-Buffer's packet-generator",
                                                  status);

        uint64_t waitCommandSize          = pCmdBuffPktGenerator->getWaitCommandSize();
        uint64_t signalCommandSize        = pCmdBuffPktGenerator->getSignalCommandSize();
        uint64_t commandSize              = waitCommandSize + signalCommandSize;
        char*    pCommand                 = new char[commandSize];
        char*    pTmpCommand              = pCommand;
        uint32_t signalAndWaitCmdBarriers = REGISTER_BARRIER | MESSAGE_BARRIER;

        pCmdBuffPktGenerator->generateWaitCommand(pTmpCommand,
                                                  waitCommandSize,
                                                  waitQmanId,
                                                  monitorId,
                                                  syncId,
                                                  1,
                                                  0,
                                                  signalAndWaitCmdBarriers);
        pTmpCommand += waitCommandSize;

        status = pCmdBuffPktGenerator
                     ->generateSignalCommand(pTmpCommand, signalCommandSize, 0, 0, 0, signalAndWaitCmdBarriers);

        if (status != synSuccess)
        {
            delete[] pCommand;
            return synFail;
        }

        status = DeviceCommon::createAndAddHangCommandBuffer(*ppCommandBuffers, 0, pCommand, commandSize, waitQmanId);

        delete[] pCommand;
    } while (0);  // Do once
    return status;
}

void DeviceGaudi::_mapCB(CommandSubmission* cb)
{
    synCommandBuffer* cbExt   = cb->getExecuteExternalQueueCb();
    uint32_t          numOfCb = cb->getNumExecuteExternalQueue();

    for (uint32_t cbIndex = 0; cbIndex < numOfCb; cbIndex++)
    {
        CommandBuffer* pCB = reinterpret_cast<CommandBuffer*>(cbExt[cbIndex]);
        pCB->MapBuffer();
    }
}

void DeviceGaudi::getDeviceHbmVirtualAddresses(uint64_t& hbmBaseAddr, uint64_t& hbmEndAddr)
{
    hbmBaseAddr = m_osalInfo.dramBaseAddress;
    hbmEndAddr  = m_osalInfo.dramBaseAddress + m_osalInfo.dramSize;
}