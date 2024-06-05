#include "queue_copy_qman.hpp"

#include "defenders.h"
#include "habana_global_conf_runtime.h"
#include "memory_manager.hpp"
#include "physical_queues_manager.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/data_chunk/data_chunk.hpp"
#include "runtime/qman/common/queue_info.hpp"
#include "runtime/qman/common/recipe_program_buffer.hpp"
#include "runtime/qman/common/wcm/work_completion_manager.hpp"
#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"
#include "syn_singleton.hpp"
#include "types_exception.h"

static const uint64_t HUGE_COPY_REQUEST_TOTAL_BUFFER_SIZE = (uint64_t)5 * 1024 * 1024 * 1024;  // 5GB
static const uint32_t MAX_COMMAND_BUFFER_SIZE             = HL_MAX_CB_SIZE;
static const uint64_t MAX_LIN_DMA_BUFFER_SIZE             = 16 * 1024 * sizeof(uint8_t);  // 16KB
// For huge copy requests (H2D), a larger chunk's size will be used
static const uint64_t MAX_LIN_DMA_BUFFER_SIZE_HUGE_COPY_REQ = 64 * 1024 * sizeof(uint8_t);  // 64KB
// Due to packet_lin_dma tsize field's limitation
static const uint64_t MAX_LIN_DMA_BUFFER_SIZE_HW_LIMITATION = std::numeric_limits<uint32_t>::max();
static const uint64_t MAX_AMOUNT_OF_DATA_CHUNKS             = 100;

QueueCopyQman::QueueCopyQman(const BasicQueueInfo&           rBasicQueueInfo,
                             uint32_t                        physicalQueueOffset,
                             synDeviceType                   deviceType,
                             PhysicalQueuesManagerInterface* pPhysicalStreamsManager,
                             WorkCompletionManagerInterface& rWorkCompletionManager,
                             DevMemoryAllocInterface&        rDevMemAlloc)
: QueueBaseQmanWcm(rBasicQueueInfo, physicalQueueOffset, deviceType, pPhysicalStreamsManager, rWorkCompletionManager),
  m_pMemoryManager(nullptr),
  m_poolMemoryManager(nullptr),
  m_pAllocator(nullptr),
  m_isArbitrationDmaNeeded(false),
  m_csDcMappingDbSize(0),
  m_maxCommandSize(0)
{
    internalStreamType queueType = m_basicQueueInfo.queueType;

    uint16_t numOfArbitrationPackets = 2;

    generic::CommandBufferPktGenerator* generator = _getPacketGenerator(m_deviceType);
    switch (m_deviceType)
    {
        case synDeviceGaudi:
            m_csDcMappingDbSize = (size_t)gaudi_queue_id::GAUDI_QUEUE_ID_SIZE;
            break;
        case synDeviceGaudi2:
            m_csDcMappingDbSize     = (size_t)gaudi2_queue_id::GAUDI2_QUEUE_ID_SIZE;
            numOfArbitrationPackets = 0;  // Not yet supported
            break;
        default:
            HB_ASSERT(false, "unsupported device type");
    }

    bool isDataChunkCreationFailure = true;
    try
    {
        const uint64_t linDmaPacketSize      = generator->getLinDmaPacketSize();
        uint64_t       singleBufferChunkSize = linDmaPacketSize;

        switch (queueType)
        {
            case INTERNAL_STREAM_TYPE_DMA_UP:
            case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
            case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
            {
                if (deviceType == synDeviceGaudi)
                {
                    m_pAllocator.reset(new DataChunksAllocatorCommandBuffer("DMA-Up", MAX_AMOUNT_OF_DATA_CHUNKS));
                }
                else
                {
                    m_pMemoryManager = createMemoryManager(rDevMemAlloc);
                    m_pAllocator.reset(
                        new DataChunksAllocatorMmuBuffer("DMA-Up", m_pMemoryManager.get(), MAX_AMOUNT_OF_DATA_CHUNKS));
                }

                m_isArbitrationDmaNeeded = false;
                // DC-Cache 1 - single DMA packet

                uint64_t maximalCacheAmount     = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();
                uint64_t minimalFreeCacheAmount = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();
                uint64_t maximalFreeCacheAmount = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();

                m_pAllocator->addDataChunksCache(singleBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                // DC-Cache 2 - 10 KB (due to multiple-memcopy API)
                singleBufferChunkSize = (10 * 1024 / linDmaPacketSize) * linDmaPacketSize;

                maximalCacheAmount     = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();
                minimalFreeCacheAmount = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();
                maximalFreeCacheAmount = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();

                m_pAllocator->addDataChunksCache(singleBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                // DC-Cache 3 - Full CB
                singleBufferChunkSize = (HL_MAX_CB_SIZE / linDmaPacketSize) * linDmaPacketSize;

                maximalCacheAmount     = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_LARGE.value();
                minimalFreeCacheAmount = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_LARGE.value();
                maximalFreeCacheAmount = GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_LARGE.value();

                m_pAllocator->addDataChunksCache(singleBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                m_maxCommandSize = singleBufferChunkSize * std::min(MAX_AMOUNT_OF_DATA_CHUNKS, maximalCacheAmount);
                break;
            }
            case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
            {
                uint64_t poolSize = GCFG_POOL_MAPPING_SIZE_IN_STREAM_COPY.value();
                if (poolSize > 0 && (deviceType == synDeviceGaudi))
                {
                    bool useWait        = GCFG_ENABLE_POOL_MAPPING_WAIT_IN_STREAM_COPY.value();
                    m_poolMemoryManager = createPoolMemoryManager(rDevMemAlloc, poolSize, useWait);
                }
                else
                    m_pMemoryManager = createMemoryManager(rDevMemAlloc);
                if (deviceType == synDeviceGaudi)
                {
                    m_pAllocator.reset(new DataChunksAllocatorCommandBuffer("DMA-Synapse", MAX_AMOUNT_OF_DATA_CHUNKS));
                }
                else
                {
                    m_pAllocator.reset(new DataChunksAllocatorMmuBuffer("DMA-Synapse",
                                                                        m_pMemoryManager.get(),
                                                                        MAX_AMOUNT_OF_DATA_CHUNKS));
                }

                m_isArbitrationDmaNeeded = generator->isDmaDownArbitrationRequired();

                if (m_isArbitrationDmaNeeded)
                {
                    singleBufferChunkSize += (numOfArbitrationPackets * generator->getArbitrationCommandSize());
                }

                uint64_t maximalCacheAmount = GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();
                uint64_t minimalFreeCacheAmount =
                    GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();
                uint64_t maximalFreeCacheAmount =
                    GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();

                m_pAllocator->addDataChunksCache(singleBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                // DC-Cache 2 - 10 KB
                uint64_t multiBufferChunkSize = (10 * 1024 / singleBufferChunkSize) * singleBufferChunkSize;

                maximalCacheAmount     = GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();
                minimalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();
                maximalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();

                m_pAllocator->addDataChunksCache(multiBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                m_maxCommandSize = multiBufferChunkSize * std::min(MAX_AMOUNT_OF_DATA_CHUNKS, maximalCacheAmount);
                break;
            }
            case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
            {
                if (deviceType == synDeviceGaudi)
                {
                    m_pAllocator.reset(new DataChunksAllocatorCommandBuffer("DMA-User", MAX_AMOUNT_OF_DATA_CHUNKS));
                }
                else
                {
                    m_pMemoryManager = createMemoryManager(rDevMemAlloc);
                    m_pAllocator.reset(new DataChunksAllocatorMmuBuffer("DMA-User",
                                                                        m_pMemoryManager.get(),
                                                                        MAX_AMOUNT_OF_DATA_CHUNKS));
                }

                // DC-Cache 1 - 20 LDMA commands (several commands)
                uint64_t numOfLinDmaCommands = 20;

                m_isArbitrationDmaNeeded = generator->isDmaDownArbitrationRequired();

                if (m_isArbitrationDmaNeeded)
                {
                    singleBufferChunkSize += (numOfArbitrationPackets * generator->getArbitrationCommandSize());
                }
                uint64_t multiBufferChunkSize = singleBufferChunkSize * numOfLinDmaCommands;

                uint64_t maximalCacheAmount     = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();
                uint64_t minimalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();
                uint64_t maximalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_SMALL.value();

                m_pAllocator->addDataChunksCache(multiBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                // DC-Cache 2 - 10 KB
                multiBufferChunkSize = (10 * 1024 / singleBufferChunkSize) * singleBufferChunkSize;

                maximalCacheAmount     = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();
                minimalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();
                maximalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM.value();

                m_pAllocator->addDataChunksCache(multiBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                // DC-Cache 3 - Full CB
                multiBufferChunkSize = (HL_MAX_CB_SIZE / singleBufferChunkSize) * singleBufferChunkSize;

                maximalCacheAmount     = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_LARGE.value();
                minimalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_LARGE.value();
                maximalFreeCacheAmount = GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_LARGE.value();

                m_pAllocator->addDataChunksCache(multiBufferChunkSize,
                                                 minimalFreeCacheAmount,
                                                 maximalFreeCacheAmount,
                                                 maximalCacheAmount);

                m_maxCommandSize = multiBufferChunkSize * std::min(MAX_AMOUNT_OF_DATA_CHUNKS, maximalCacheAmount);
                break;
            }
            default:
            {
                throw SynapseException("QueueBase: illegal type");
            }
        }
    }
    catch (const SynapseException& err)
    {
        if (!isDataChunkCreationFailure)
        {
            throw;
        }

        LOG_ERR(SYN_STREAM,
                "{}: Failed to create Data-Chunks cache on stream {} due to {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                err.what());

        throw SynapseException("QueueBase: Failed to create Data-Chunks cache");
    }
}

QueueCopyQman::~QueueCopyQman()
{
    STAT_GLBL_START(streamDbMutexDuration);
    std::unique_lock<std::mutex> mutex(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);
    _clearSharedDBs();
}

void QueueCopyQman::_clearSharedDBs()
{
    if (m_pAllocator != nullptr)
    {
        m_pAllocator->updateCache();
    }

    uint32_t csdcDbSize = m_csDataChunksDb.size();
    if (csdcDbSize != 0)
    {
        LOG_CRITICAL(SYN_STREAM, "free all CS Data-Chunk, should not happen unless device reset!");
        while (csdcDbSize != 0)
        {
            auto csdc = m_csDataChunksDb.front();
            delete csdc;
            m_csDataChunksDb.pop_front();
            csdcDbSize--;
        }
    }
}

synStatus QueueCopyQman::memcopy(internalMemcopyParams& memcpyParams,
                                 const internalDmaDir   direction,
                                 bool                   isUserRequest,
                                 QueueInterface*        pPreviousStream,
                                 const uint64_t         overrideMemsetVal,
                                 bool                   inspectCopiedContent,
                                 SpRecipeProgramBuffer* pRecipeProgramBuffer,
                                 uint8_t                apiId)
{
    STAT_GLBL_START(streamCopyMemCopyDuration);

    if (memcpyParams.empty())
    {
        LOG_DEBUG(SYN_STREAM, "{}: Got empty memCopyParams", HLLOG_FUNC);
        return synSuccess;
    }

    uint64_t    mappingSize        = 0;
    uint64_t    mappingHostAddress = 0;
    std::string mappingDesc("");

    if (pRecipeProgramBuffer != nullptr)
    {
        SpRecipeProgramBuffer spRecipeProgramBuffer(*pRecipeProgramBuffer);

        bool shouldMap = spRecipeProgramBuffer->shouldMap();
        if (shouldMap)
        {
            if (direction != MEMCOPY_HOST_TO_DRAM)
            {
                LOG_ERR(SYN_STREAM, "{}: shouldMap requested while direction = {}", HLLOG_FUNC, direction);
                return synFailedToSubmitWorkload;
            }

            mappingSize = spRecipeProgramBuffer->getSize();
        }

        mappingHostAddress = (uint64_t)spRecipeProgramBuffer->getBuffer();
        mappingDesc        = spRecipeProgramBuffer->getMappingDescription();

        // There might be multiple memcopy-requests, but only a single buffer is mapped,
        // and according to the RecipeProgramBuffer information
        //
        // Here, we will validate only the first parameter
        if (memcpyParams[0].src != mappingHostAddress)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Mismatech between memcpy.src = {} and recipe's program-buffer {}",
                    HLLOG_FUNC,
                    memcpyParams[0].src,
                    mappingHostAddress);
            return synFailedToSubmitWorkload;
        }
    }

    synStatus status = isValidOperation(direction, memcpyParams);
    if (status != synSuccess)
    {
        return status;
    }

    uint64_t totalSize = 0;
    for (const auto& element : memcpyParams)
    {
        totalSize += element.size;
    }
    if (totalSize == 0)
    {
        LOG_DEBUG(SYN_STREAM, "{}: Got 0 total size", HLLOG_FUNC);
        return synSuccess;
    }

    bool isMemset = false;
    if (memcpyParams.front().src == 0)
    {
        // use LinDma to perform memset
        // src value is overridden by value to memset
        if (memcpyParams.size() != 1)
        {
            return synInvalidArgument;
        }

        isMemset                 = true;
        memcpyParams.front().src = overrideMemsetVal;
    }

    generic::CommandBufferPktGenerator* pCmdBuffPktGenerator  = _getPacketGenerator(m_deviceType);
    const bool                          isArbitrationRequired = isArbitrationDmaNeeded();
    uint64_t maxLinDmaBufferSize, arbCommandSize, sizeOfLinDmaCommand, sizeOfWrappedLinDmaCommand,
        sizeOfSingleCommandBuffer, totalCommandSize;
    status = getLinDmaParams(pCmdBuffPktGenerator,
                             memcpyParams,
                             isArbitrationRequired,
                             isUserRequest,
                             m_maxCommandSize,
                             maxLinDmaBufferSize,
                             arbCommandSize,
                             sizeOfLinDmaCommand,
                             sizeOfWrappedLinDmaCommand,
                             sizeOfSingleCommandBuffer,
                             totalCommandSize);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: {} getLinDmaParams failed with status {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);
        return status;
    }

    // memory map the source if needed
    if (pRecipeProgramBuffer != nullptr)
    {
        (*pRecipeProgramBuffer)->setMappedAddr(0);
    }
    if (mappingSize != 0)
    {
        uint64_t mmuAddr = 0;
        if (m_poolMemoryManager)
        {
            uint64_t  newHostAddress   = 0;
            uint64_t* p_newHostAddress = nullptr;
            unsigned  num_entries =
                memcpyParams.size();  // if > 1 we cannot use the pool as some requests (in Retina) are just not found
            if (num_entries == 1)
            {
                p_newHostAddress = &newHostAddress;
            }
            status = m_poolMemoryManager->mapBufferEx(mmuAddr,
                                                      p_newHostAddress,
                                                      (void*)mappingHostAddress,
                                                      mappingSize,
                                                      mappingDesc);
            if (status == synSuccess && newHostAddress)
            {
                memcpyParams.front().src = newHostAddress;
                (*pRecipeProgramBuffer)->setMappedAddr(mmuAddr);
                memcpy((void*)newHostAddress, (void*)mappingHostAddress, mappingSize);
            }
        }
        else  // no pool mode
        {
            status = m_pMemoryManager->mapBuffer(mmuAddr, (void*)mappingHostAddress, mappingSize, false, mappingDesc);
        }
        HB_ASSERT(status == synSuccess, "{} error in mapBuffer {}", __FUNCTION__, status);
    }

    // acquire CS-DS data chunks
    DataChunksDB dataChunks;
    do
    {
        status = retrieveCsDc(totalCommandSize, dataChunks);

        if (status == synOutOfResources)
        {
            m_pAllocator->dumpStat();
            LOG_PERIODIC_BY_LEVEL(SYN_STREAM,
                                  SPDLOG_LEVEL_WARN,
                                  std::chrono::milliseconds(1000),
                                  10,
                                   "{} There are no available Data Chunks, wait for WCM to release resources totalCommandSize {}",
                                  m_basicQueueInfo.getDescription(),
                                  totalCommandSize,
                                  strerror(errno));
            _waitForWCM(false /* waitForAllCsdcs*/);
        }
    } while (status == synOutOfResources);

    HB_ASSERT(status == synSuccess, "{} Invalid status {}", __FUNCTION__, status);

    CommandSubmissionDataChunks* pCsDataChunks =
        new CommandSubmissionDataChunks(CS_DC_TYPE_MEMCOPY, m_deviceType, m_csDcMappingDbSize);

    if (pRecipeProgramBuffer != nullptr)
    {
        pCsDataChunks->setRecipeProgramBuffer(*pRecipeProgramBuffer);
    }

    status = memCpyAsync(pPreviousStream,
                         memcpyParams,
                         direction,
                         dataChunks,
                         pCsDataChunks,
                         isUserRequest,
                         overrideMemsetVal,
                         inspectCopiedContent,
                         isMemset,
                         isArbitrationRequired,
                         maxLinDmaBufferSize,
                         arbCommandSize,
                         sizeOfLinDmaCommand,
                         sizeOfWrappedLinDmaCommand,
                         sizeOfSingleCommandBuffer);

    if (status != synSuccess)
    {
        clearCsDcBuffers(*pCsDataChunks);
        delete pCsDataChunks;
        releaseCsDc(dataChunks);
        return status;
    }

    // store meta data on the cs for debug purpose
    std::unique_lock<std::mutex> mutex(m_DBMutex);
    m_csDescriptionDB[pCsDataChunks] = {memcpyParams, direction};

    STAT_GLBL_COLLECT_TIME(streamCopyMemCopyDuration, globalStatPointsEnum::streamCopyMemCopyDuration);

    return status;
}

void QueueCopyQman::clearCsDcBuffers(CommandSubmissionDataChunks& csDataChunks) const
{
    const SpRecipeProgramBuffer spRecipeProgramBuffer = csDataChunks.getRecipeProgramBuffer();
    if (spRecipeProgramBuffer != nullptr)
    {
        if (spRecipeProgramBuffer->shouldMap())
        {
            LOG_DEBUG(SYN_STREAM,
                      "Clear CS {} {:#x}",
                      spRecipeProgramBuffer->getMappingDescription(),
                      (uint64_t)spRecipeProgramBuffer->getBuffer());
            if (m_poolMemoryManager)
            {
                m_poolMemoryManager->unmapBufferEx(spRecipeProgramBuffer->getMappedAddr(),
                                                   spRecipeProgramBuffer->getBuffer(),
                                                   spRecipeProgramBuffer->getSize());
            }
            else
            {
                m_pMemoryManager->unmapBuffer((void*)spRecipeProgramBuffer->getBuffer(), false);
            }
        }
    }
}

synStatus QueueCopyQman::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = 0;
    if (m_pAllocator != nullptr)
    {
        mappedMemorySize += m_pAllocator->getMappedBufferSize();
    }

    return synSuccess;
}

bool QueueCopyQman::isRecipeHasInflightCsdc(InternalRecipeHandle* pRecipeHandle)
{
    return false;
}

synStatus QueueCopyQman::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                         std::vector<tensor_info_t>& tensorInfoArray) const
{
    LOG_ERR(SYN_API, "Unsupported stream type for getLastTensorArray: {}", getBasicQueueInfo().queueType);
    return synFail;
}

void QueueCopyQman::notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed)
{
    for (uint64_t csHandle : rCsHandles)
    {
        _notifyCsCompleted(csHandle, csFailed);
        _wcmReleaseThreadIfNeeded();
    }

    std::unique_lock<std::mutex> condMutex(m_condVarMutex);
    m_cvFlag = true;
    m_condVar.notify_all();
}

synStatus QueueCopyQman::waitForRecipeCsdcs(uint64_t recipeId)
{
    while (true)
    {
        unsigned i = 0;

        std::unique_lock<std::mutex> dbMutex(m_DBMutex);
        bool                         foundRecipeInFlight = false;
        {
            for (const auto pCsDataChunks : m_csDataChunksDb)
            {
                const SpRecipeProgramBuffer spRecipeProgramBuffer = pCsDataChunks->getRecipeProgramBuffer();
                if (spRecipeProgramBuffer != nullptr)
                {
                    if (spRecipeProgramBuffer->getRecipeId() == recipeId)
                    {
                        foundRecipeInFlight = true;
                        break;
                    }
                }
            }
        }

        // no matching recipe found
        if (!foundRecipeInFlight)
        {
            break;
        }

        // waits for recipe removal
        LOG_TRACE(SYN_STREAM, "{}: recipe 0x{:x} iter {}", HLLOG_FUNC, recipeId, ++i);

        std::unique_lock<std::mutex> condMutex(m_condVarMutex);
        dbMutex.unlock();
        m_cvFlag = false;

        int  csQuerytimeoutSec = 10;
        bool waitSuccess =
            m_condVar.wait_for(condMutex, std::chrono::seconds(csQuerytimeoutSec), [&] { return m_cvFlag; });
        if (!waitSuccess)
        {
            LOG_ERR(SYN_STREAM, "{}: 0x{:x} waiting done failure", HLLOG_FUNC, csQuerytimeoutSec);
            return synFail;
        }
    }

    return synSuccess;
}

void QueueCopyQman::_notifyCsCompleted(uint64_t waitForEventHandle, bool csFailed)
{
    STAT_GLBL_START(wcmObserverDbMutexDuration);
    std::unique_lock<std::mutex> mutex(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(wcmObserverDbMutexDuration, globalStatPointsEnum::wcmObserverDbMutexDuration);

    std::deque<CommandSubmissionDataChunks*>::const_iterator iter;
    CommandSubmissionDataChunks*                             pCsDataChunks;
    for (iter = m_csDataChunksDb.begin(); iter != m_csDataChunksDb.end(); ++iter)
    {
        pCsDataChunks = *iter;
        if (pCsDataChunks->containsHandle(waitForEventHandle))
        {
            break;
        }
    }

    if (iter == m_csDataChunksDb.end())
    {
        LOG_CRITICAL(SYN_STREAM,
                     "{}: Can not find waitForEventHandle {} m_csDataChunksDb size {}",
                     HLLOG_FUNC,
                     waitForEventHandle,
                     m_csDataChunksDb.size());

        for (const auto& pCsDcs : m_csDataChunksDb)
        {
            LOG_CRITICAL(SYN_STREAM, "pCsDcs {:#x}", TO64(pCsDcs));
            const std::string csList = pCsDcs->dfaGetCsList();
            if (!csList.empty())
            {
                LOG_CRITICAL(SYN_STREAM, "cs: {}", csList);
            }
        }

        return;
    }

    if (iter != m_csDataChunksDb.begin())
    {
        LOG_DEBUG(SYN_STREAM, "{}: unordered CS completion detected {}", HLLOG_FUNC, waitForEventHandle);
    }

    const commandsDataChunksDB& rCommandsDataChunks = pCsDataChunks->getCommandsBufferDataChunks();

    if (!m_pAllocator->releaseDataChunksThreadSafe(rCommandsDataChunks))
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to free CS Data-Chunks", HLLOG_FUNC);
        return;
    }

    STAT_GLBL_COLLECT(rCommandsDataChunks.size(), csdcStreamCopyRelease);

    clearCsDcBuffers(*pCsDataChunks);

    LOG_DEBUG(SYN_STREAM,
              "{} release CS {:#x} waitForEventHandle {:#x} csFailed {}",
              m_basicQueueInfo.getDescription(),
              TO64(pCsDataChunks),
              waitForEventHandle,
              csFailed);

    delete pCsDataChunks;
    m_csDataChunksDb.erase(iter);
    m_csDescriptionDB.erase(pCsDataChunks);
}

synStatus QueueCopyQman::isValidOperation(internalDmaDir direction, const internalMemcopyParams& rMemcpyParams)
{
    if (rMemcpyParams.empty())
    {
        LOG_WARN(SYN_STREAM, "{}: Got empty memcpy parameters", HLLOG_FUNC);
        return synInvalidArgument;
    }

    bool isCopyFromHost = ((direction == MEMCOPY_HOST_TO_DRAM) || (direction == MEMCOPY_HOST_TO_SRAM));

    bool isCopyToHost = ((direction == MEMCOPY_DRAM_TO_HOST) || (direction == MEMCOPY_SRAM_TO_HOST));

    bool isValidOperation = true;
    if (isCopyFromHost)
    {
        if ((m_basicQueueInfo.queueType != INTERNAL_STREAM_TYPE_DMA_DOWN_USER) &&
            (m_basicQueueInfo.queueType != INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE))
        {
            isValidOperation = false;
        }
    }
    else
    {
        if (!isCopyToHost)
        {
            if (m_basicQueueInfo.queueType != INTERNAL_STREAM_TYPE_DEV_TO_DEV)
            {
                isValidOperation = false;
            }
        }
        else
        {
            if ((m_basicQueueInfo.queueType != INTERNAL_STREAM_TYPE_DMA_UP) &&
                (m_basicQueueInfo.queueType != INTERNAL_STREAM_TYPE_DMA_UP_PROFILER))
            {
                isValidOperation = false;
            }
        }
    }

    if (!isValidOperation)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Invalid operation (direction {} isCopyFromHost {} isCopyToHost {}) for stream's {}",
                HLLOG_FUNC,
                direction,
                isCopyFromHost,
                isCopyToHost,
                m_basicQueueInfo.getDescription());
        return synInvalidArgument;
    }

    return synSuccess;
}

synStatus QueueCopyQman::memCpyAsync(QueueInterface*              pPreviousStream,
                                     const internalMemcopyParams& rMemcpyParams,
                                     const internalDmaDir         direction,
                                     DataChunksDB&                rDataChunks,
                                     CommandSubmissionDataChunks* pCsDataChunks,
                                     bool                         isUserRequest,
                                     const uint64_t               overrideMemsetVal,
                                     bool                         isInspectCopiedContent,
                                     bool                         isMemset,
                                     bool                         isArbitrationRequired,
                                     uint64_t                     maxLinDmaBufferSize,
                                     uint64_t                     arbCommandSize,
                                     uint64_t                     sizeOfLinDmaCommand,
                                     uint64_t                     sizeOfWrappedLinDmaCommand,
                                     uint64_t                     sizeOfSingleCommandBuffer)
{
    STAT_GLBL_START(streamCopyMutexDuration);
    std::unique_lock<std::mutex> mutex(m_mutex);
    STAT_GLBL_COLLECT_TIME(streamCopyMutexDuration, globalStatPointsEnum::streamCopyMutexDuration);

    STAT_GLBL_START(streamCopyOperationDuration);

    if (pPreviousStream != nullptr && !isUserRequest)
    {
        synStatus status = performStreamsSynchronization(*pPreviousStream, isUserRequest);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Can not perform inner-stream's signaling on stream {}",
                    HLLOG_FUNC,
                    m_basicQueueInfo.getDescription());

            return synFail;
        }
    }

    PhysicalQueuesId physicalQueuesId;
    TrainingRetCode  trainingRetCode =
        m_pPhysicalStreamsManager->getPhysicalQueueIds(m_basicQueueInfo, physicalQueuesId);
    if (trainingRetCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Can not get valid physical-queues ID for stream {} (trainingRetCode {})",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                trainingRetCode);

        return synFail;
    }

    LOG_TRACE(SYN_STREAM, "{}: linDMA queue: {}", HLLOG_FUNC, physicalQueuesId);

    InternalWaitHandle internalHandle;
    synStatus          status = _SYN_SINGLETON_INTERNAL->submitLinDmaCommand(rMemcpyParams,
                                                                    direction,
                                                                    isArbitrationRequired,
                                                                    physicalQueuesId,
                                                                    &internalHandle,
                                                                    rDataChunks,
                                                                    pCsDataChunks,
                                                                    isUserRequest,
                                                                    isMemset,
                                                                    isInspectCopiedContent,
                                                                    maxLinDmaBufferSize,
                                                                    arbCommandSize,
                                                                    sizeOfLinDmaCommand,
                                                                    sizeOfWrappedLinDmaCommand,
                                                                    sizeOfSingleCommandBuffer);

    if (status == synSuccess)
    {
        spQueueInfo pQueueInfo = m_pPhysicalStreamsManager->getStreamInfo(m_basicQueueInfo);
        LOG_TRACE(SYN_PROGRESS,
                  GAUDI_PROGRESS_FMT,
                  m_basicQueueInfo.queueType,
                  pQueueInfo->getQueueId(),
                  pQueueInfo->getPhysicalQueueOffset(),
                  pQueueInfo->getPhysicalQueuesId(),
                  internalHandle.handle,
                  HLLOG_FUNC,
                  __LINE__);

        m_pPhysicalStreamsManager->updateStreamPostExecution(
            m_basicQueueInfo,
            internalHandle,
            isUserRequest
                ? (direction == internalDmaDir::MEMCOPY_HOST_TO_DRAM ? "memCopyH2D" : "memCopyD2H")
                : (direction == internalDmaDir::MEMCOPY_HOST_TO_DRAM ? "internalmMemcpyH2D" : "internalMemcpyD2H"));
        if (trainingRetCode != TRAINING_RET_CODE_SUCCESS)
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Can not update stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
            return synFail;
        }

        const commandsDataChunksDB& rCommandsDataChunks = pCsDataChunks->getCommandsBufferDataChunks();
        STAT_GLBL_COLLECT(rCommandsDataChunks.size(), csdcStreamCopyAllocate);

        LOG_DEBUG(SYN_STREAM,
                  "{}: {} allocate {} CS DC",
                  HLLOG_FUNC,
                  m_basicQueueInfo.getDescription(),
                  rCommandsDataChunks.size());

        _addCsdcToDb(pCsDataChunks);
    }

    STAT_GLBL_COLLECT_TIME(streamCopyOperationDuration, globalStatPointsEnum::streamCopyOperationDuration);
    return status;
}

generic::CommandBufferPktGenerator* QueueCopyQman::_getPacketGenerator(synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            return gaudi::CommandBufferPktGenerator::getInstance();
        }
        default:
        {
        }
    }

    HB_ASSERT(false, "unsupported device type");
    return nullptr;
}

synStatus QueueCopyQman::getLinDmaParams(generic::CommandBufferPktGenerator* pCmdBuffPktGenerator,
                                         const internalMemcopyParams&        rMemcpyParams,
                                         bool                                isArbitrationRequired,
                                         bool                                isUserRequest,
                                         uint64_t                            maxCommandSize,
                                         uint64_t&                           rMaxLinDmaBufferSize,
                                         uint64_t&                           rArbCommandSize,
                                         uint64_t&                           rSizeOfLinDmaCommand,
                                         uint64_t&                           rSizeOfWrappedLinDmaCommand,
                                         uint64_t&                           rSizeOfSingleCommandBuffer,
                                         uint64_t&                           rTotalCommandSize)
{
    const uint8_t numOfArbitrationPointsPerLinDma = isArbitrationRequired ? 2 : 0;
    const bool    isLimitLinDmaBufferSize         = isArbitrationRequired && isUserRequest;

    rArbCommandSize      = pCmdBuffPktGenerator->getArbitrationCommandSize();
    rSizeOfLinDmaCommand = pCmdBuffPktGenerator->getLinDmaPacketSize();

    rSizeOfWrappedLinDmaCommand = rArbCommandSize * numOfArbitrationPointsPerLinDma + rSizeOfLinDmaCommand;

    const uint16_t numOfLinDmaCommandsInSingleCB = MAX_COMMAND_BUFFER_SIZE / rSizeOfWrappedLinDmaCommand;
    rSizeOfSingleCommandBuffer                   = numOfLinDmaCommandsInSingleCB * rSizeOfWrappedLinDmaCommand;

    uint64_t  totalWrappedPacketsNum = 0;
    synStatus status                 = _getTotalCommandSize(totalWrappedPacketsNum,
                                            rMaxLinDmaBufferSize,
                                            rTotalCommandSize,
                                            rMemcpyParams,
                                            maxCommandSize,
                                            rSizeOfWrappedLinDmaCommand,
                                            isLimitLinDmaBufferSize);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: maxCommandSize {} totalCommandSize {} is too big",
                HLLOG_FUNC,
                maxCommandSize,
                rTotalCommandSize);
    }

    LOG_TRACE(SYN_STREAM,
              "rMaxLinDmaBufferSize {} rArbCommandSize {} rSizeOfLinDmaCommand {}"
              " rSizeOfWrappedLinDmaCommand {} rSizeOfSingleCommandBuffer {} rTotalCommandSize {}",
              rMaxLinDmaBufferSize,
              rArbCommandSize,
              rSizeOfLinDmaCommand,
              rSizeOfWrappedLinDmaCommand,
              rSizeOfSingleCommandBuffer,
              rTotalCommandSize);
    return status;
}

synStatus QueueCopyQman::retrieveCsDc(uint64_t totalCommandSize, DataChunksDB& rDataChunks)
{
    STAT_GLBL_START(streamCopyRetrieveCsDc);

    bool retVal = m_pAllocator->acquireDataChunksThreadSafe(rDataChunks, totalCommandSize);

    if (!retVal)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to acquire data-chunk totalCommandSize {}", HLLOG_FUNC, totalCommandSize);
        return synOutOfResources;
    }

    if (g_validateDataChunksUsage)
    {
        LOG_ERR(SYN_STREAM, "Copy-stream's Data-Chunks usage:");
        printDataChunksIds("", rDataChunks);
    }

    HB_ASSERT(!(rDataChunks.empty()), "Got empty DC DB for Lin-DMA operation");

    STAT_GLBL_COLLECT_TIME(streamCopyRetrieveCsDc, globalStatPointsEnum::streamCopyRetrieveCsDc);
    return synSuccess;
}

synStatus QueueCopyQman::releaseCsDc(const DataChunksDB& rDataChunks)
{
    bool retVal = m_pAllocator->releaseDataChunksThreadSafe(rDataChunks);

    if (!retVal)
    {
        LOG_CRITICAL(SYN_STREAM, "{}: Failed to release data-chunks", HLLOG_FUNC);
    }

    return synSuccess;
}

void QueueCopyQman::_dfaLogCsDcInfo(CommandSubmissionDataChunks* csPtr, int logLevel, bool errorCsOnly)
{
    std::string csDetails = "";
    auto        descIter  = m_csDescriptionDB.find(csPtr);
    if (descIter != m_csDescriptionDB.end())
    {
        csCopyMetaData desc = descIter->second;
        csDetails           = fmt::format("copy direction {}, ", desc.direction);
        for (auto copy : desc.params)
        {
            csDetails += fmt::format("copy src: 0x{:x}, dst 0x{:x}, size 0x{:x}\n", copy.src, copy.dst, copy.size);
        }
    }
    else
    {
        csDetails = "can't find data about this cs";
    }
    SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, "{}", csDetails);
}

synStatus QueueCopyQman::_getTotalCommandSize(uint64_t&                    totalWrappedPacketsNum,
                                              uint64_t&                    rMaxLinDmaBufferSize,
                                              uint64_t&                    rTotalCommandSize,
                                              const internalMemcopyParams& rMemcpyParams,
                                              uint64_t                     maxCommandSize,
                                              uint64_t                     sizeOfWrappedLinDmaCommand,
                                              const bool                   isLimitLinDmaBufferSize)
{
    rMaxLinDmaBufferSize = isLimitLinDmaBufferSize ? MAX_LIN_DMA_BUFFER_SIZE : MAX_LIN_DMA_BUFFER_SIZE_HW_LIMITATION;

    uint64_t totalSize = 0;
    if (isLimitLinDmaBufferSize)
    {
        for (auto element : rMemcpyParams)
        {
            if (element.size > 0)
            {
                totalSize += element.size;
            }
        }
    }

    if (totalSize >= HUGE_COPY_REQUEST_TOTAL_BUFFER_SIZE)
    {
        rMaxLinDmaBufferSize = MAX_LIN_DMA_BUFFER_SIZE_HUGE_COPY_REQ;
    }

    totalWrappedPacketsNum = 0;
    for (auto element : rMemcpyParams)
    {
        if (element.size > 0)
        {
            totalWrappedPacketsNum += CEIL(element.size, rMaxLinDmaBufferSize);
        }
    }

    if (totalWrappedPacketsNum == 0)
    {
        LOG_ERR(SYN_STREAM, "{}: Got 0 total size", HLLOG_FUNC);

        return synInvalidArgument;
    }

    rTotalCommandSize = sizeOfWrappedLinDmaCommand * totalWrappedPacketsNum;

    return (maxCommandSize >= rTotalCommandSize) ? synSuccess : synInvalidArgument;
}