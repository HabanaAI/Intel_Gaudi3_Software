#include "data_chunks_cache.hpp"

#include "data_chunk.hpp"
#include "data_chunks_statistics_manager.hpp"

#include "defenders.h"
#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "host_buffers_mapper.hpp"
#include "runtime/common/osal/osal.hpp"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "types_exception.h"
#include "types.h"
#include "utils.h"

#include <sys/mman.h>

#include <cstring>

DataChunksCache::DataChunksCache(uint64_t singleChunkSize,
                                 uint64_t minimalCacheAmount,
                                 uint64_t maximalFreeCacheAmount,
                                 uint64_t maximalCacheAmount,
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                 uint32_t statisticsId,
#endif
                                 eMappingType mappingType /* = MAPPING_TYPE_MMU */)
:
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
  m_statisticsId(statisticsId),
#endif
  m_nextChunkId(0),
  m_singleChunkSize(singleChunkSize),
  m_maximalFreeCacheAmount(maximalFreeCacheAmount),
  m_maximalCacheAmount(maximalCacheAmount),
  m_minimalCacheAmount(minimalCacheAmount)
{
    if (singleChunkSize == 0)
    {
        throw SynapseException("Zero single-chunk-size is prohibited");
    }

    if (maximalFreeCacheAmount < minimalCacheAmount)
    {
        throw SynapseException("DataChunksCache: Minimal threshold is larger than maximal free cache thresold");
    }

    if (maximalCacheAmount < maximalFreeCacheAmount)
    {
        throw SynapseException("DataChunksCache: Maximal free cache thresold is larger than Maximal database size");
    }
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    LOG_ERR(SYN_DATA_CHUNK, "Statistics ID {} mapping-type {}", statisticsId, mappingType);
#endif
}

DataChunksCache::~DataChunksCache()
{
    for (auto usedDataChunkItr : m_usedDataChunks)
    {
        _deleteDataChunk(usedDataChunkItr.second);
    }
    m_usedDataChunks.clear();

    LOG_TRACE(SYN_DATA_CHUNK, "{}: finished to clean used DCs", HLLOG_FUNC);

    for (auto* freeDataChunkItr : m_freeDataChunks)
    {
        _deleteDataChunk(freeDataChunkItr);
    }
    m_freeDataChunks.clear();
}

uint64_t DataChunksCache::acquireDataChunks(DataChunksDB& dataChunks, uint64_t dataSize)
{
    if (dataSize == 0)
    {
        LOG_DEBUG(SYN_DATA_CHUNK, "{}: No chunks created when dataSize is zero", HLLOG_FUNC);
        return 0;
    }

    uint64_t numOfRequestedChunks = ((dataSize - 1) / m_singleChunkSize) + 1;

    return acquireDataChunksAmount(dataChunks, numOfRequestedChunks);
}

uint64_t DataChunksCache::acquireDataChunksAmount(DataChunksDB& dataChunks, uint64_t chunksAmount)
{
    if (!_allocateDataChunks(chunksAmount, false))
    {
        LOG_DEBUG(SYN_DATA_CHUNK, "{}: Cant allocate {} chunks", HLLOG_FUNC, chunksAmount);
        return 0;
    }

    for (uint64_t i = 0; i < chunksAmount; i++)
    {
        DataChunk* pDataChunk = nullptr;

        _acquireSingleDataChunk(pDataChunk);
        dataChunks.push_back(pDataChunk);
    }

    return chunksAmount;
}

bool DataChunksCache::acquireSingleDataChunk(DataChunk*& pDataChunk)
{
    const uint64_t singleDataChunk = 1;

    if (!_allocateDataChunks(singleDataChunk, false))
    {
        LOG_DEBUG(SYN_DATA_CHUNK, "{}: Cant allocate single chunk", HLLOG_FUNC);
        return false;
    }

    _acquireSingleDataChunk(pDataChunk);

    return true;
}

uint64_t DataChunksCache::releaseDataChunks(const DataChunksDB& dataChunks, bool isBreakUponFailure)
{
    uint64_t releasedChunksAmount = 0;

    for (auto singleDataChunk : dataChunks)
    {
        if (!_freeDataChunk(singleDataChunk->getChunkId()))
        {
            if (isBreakUponFailure)
            {
                break;
            }
            LOG_CRITICAL(SYN_DATA_CHUNK,
                         "{}: Failed to release Data-Chunk {}",
                         HLLOG_FUNC,
                         singleDataChunk->getChunkId());
        }
        releasedChunksAmount++;
    }

    return releasedChunksAmount;
}

bool DataChunksCache::releaseSingleDataChunk(uint64_t dataChunkId)
{
    bool status = true;

    if (!_freeDataChunk(dataChunkId))
    {
        status = false;
        LOG_CRITICAL(SYN_DATA_CHUNK, "{}: Failed to release Data-Chunk {}", HLLOG_FUNC, dataChunkId);
    }

    return status;
}

void DataChunksCache::updateCache()
{
    uint64_t freeCacheSize = m_freeDataChunks.size();

    if (freeCacheSize > m_maximalFreeCacheAmount)
    {
        uint64_t numOfElementsToDelete = freeCacheSize - m_maximalFreeCacheAmount;
        for (uint64_t i = 0; i < numOfElementsToDelete; i++)
        {
            DataChunk* pDataChunk = m_freeDataChunks.front();
            _deleteDataChunk(pDataChunk);
            m_freeDataChunks.pop_front();
        }
    }
}

uint64_t DataChunksCache::getAmountOfAvailableElements() const
{
    return m_freeDataChunks.size();
}

bool DataChunksCache::_allocateDataChunks(uint64_t numFreeChunksRequested, bool isMinimalCacheAllocation)
{
    uint64_t numOfFreeChunksAllocated = m_freeDataChunks.size();

    if (numOfFreeChunksAllocated >= numFreeChunksRequested)
    {
        m_statAlloc += numFreeChunksRequested;
        return true;
    }

    uint64_t numOfUsedChunksAllocated = m_usedDataChunks.size();
    uint64_t additionalChunksRequired = numFreeChunksRequested - numOfFreeChunksAllocated;

    if (numOfUsedChunksAllocated + numFreeChunksRequested > m_maximalCacheAmount)
    {
        LOG_DEBUG(SYN_DATA_CHUNK,
                  "Can not allocate {} chunks as already {} chunks are allocated",
                  numFreeChunksRequested,
                  (numOfUsedChunksAllocated + numOfFreeChunksAllocated));
        return false;
    }

    DataChunksDB chunks;
    bool         status = _createDataChunks(chunks, additionalChunksRequired, isMinimalCacheAllocation);

    m_freeDataChunks.insert(m_freeDataChunks.end(),
                            std::make_move_iterator(chunks.begin()),
                            std::make_move_iterator(chunks.end()));

    m_statCreated += numFreeChunksRequested;
    return status;
}

void DataChunksCache::_acquireSingleDataChunk(DataChunk*& pDataChunk)
{
    HB_ASSERT((m_freeDataChunks.size() != 0), "No more free data-chunks in DB");

    pDataChunk             = m_freeDataChunks.front();
    const uint64_t chunkId = m_nextChunkId;
    pDataChunk->initialize(chunkId);
    m_usedDataChunks[m_nextChunkId] = pDataChunk;
    m_nextChunkId++;
    m_freeDataChunks.pop_front();
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    DataChunksStatisticsManager::getInstance().dataChunkAcquired(m_statisticsId, _getSingleChunkSize());
#endif

    LOG_TRACE(SYN_DATA_CHUNK,
              "DataChunksCache 0x{:x} Chunk-ID {} had been acquired from used-chunks' DB",
              TO64(this),
              chunkId);
}

bool DataChunksCache::_freeDataChunk(uint64_t chunkId)
{
    m_statReleased++;
    auto usedDataChunkItr = m_usedDataChunks.find(chunkId);
    if (usedDataChunkItr == m_usedDataChunks.end())
    {
        LOG_ERR(SYN_DATA_CHUNK, "{}: Chunk-ID {} was not found in used-chunks' DB", HLLOG_FUNC, chunkId);
        return false;
    }

    DataChunk* pDataChunk = usedDataChunkItr->second;

    m_freeDataChunks.push_front(pDataChunk);
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    uint64_t releasedSpace = pDataChunk->getUsedSize();
#endif
    pDataChunk->finalize();

    m_usedDataChunks.erase(chunkId);

    LOG_TRACE(SYN_DATA_CHUNK,
              "DataChunksCache 0x{:x} Chunk-ID {} had been free from used-chunks' DB",
              TO64(this),
              chunkId);

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    DataChunksStatisticsManager::getInstance().dataChunkReleased(m_statisticsId, releasedSpace, _getSingleChunkSize());
#endif
    return true;
}

void DataChunksCache::_deleteDataChunk(DataChunk* pDataChunk)
{
    m_statDestruct++;
    delete pDataChunk;
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    DataChunksStatisticsManager::getInstance().dataChunkDestroyed(m_statisticsId, _getSingleChunkSize());
#endif
}

DataChunksCacheMmuBuffer::DataChunksCacheMmuBuffer(HostBuffersMapper* pHostBuffersMapper,
                                                   uint64_t           singleChunkSize,
                                                   uint64_t           minimalCacheAmount,
                                                   uint64_t           maximalFreeCacheAmount,
                                                   uint64_t           maximalCacheAmount
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                   ,
                                                   uint32_t statisticsId
#endif
                                                   ,
                                                   bool writeProtect            = false,
                                                   bool canaryProtectDataChunks = false)
: DataChunksCache(singleChunkSize,
                  minimalCacheAmount,
                  maximalFreeCacheAmount,
                  maximalCacheAmount,
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                  statisticsId,
#endif
                  MAPPING_TYPE_MMU),
  m_pHostBuffersMapper(pHostBuffersMapper),
  m_pMinimalCacheChunk(nullptr),
  m_pMinimalCacheHostVirtualAddress(0),
  m_pageSize((unsigned)OSAL::getInstance().getPageSize()),
  m_numOfCanaryProtectionHeaders(0)
{
    if (pHostBuffersMapper == nullptr)
    {
        throw SynapseException("DataChunksCache: Constructed with nullptr as host-buffers-mapper parameter");
    }
    if (canaryProtectDataChunks)
    {
        m_numOfCanaryProtectionHeaders =
            (getMinimalChunksAmount() - 1) / GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER.value() + 2;
    }
    const bool isMinimalCacheAllocation = minimalCacheAmount > 0;

    if (isMinimalCacheAllocation)
    {
        if (maximalFreeCacheAmount != maximalCacheAmount)
        {
            // The breathing is no operational while using minimalCacheAmount at the moment.
            // In order to operate it, one has to ensure that during the breathing only non minimalCacheAmount
            // allocations are released.
            throw SynapseException(
                "DataChunksCache: Minimal allocation is allowed only when minimalCacheAmount == maximalCacheAmount");
        }
        uint64_t bufferSize = getSingleDataChunkSize() * getMinimalChunksAmount();
        if (m_numOfCanaryProtectionHeaders > 0)
        {
            bufferSize += m_pageSize * m_numOfCanaryProtectionHeaders;
            LOG_DEBUG(SYN_DATA_CHUNK,
                      "{}: Protecting DC buffer - added {} to buffer with size {} = {}",
                      HLLOG_FUNC,
                      m_pageSize * m_numOfCanaryProtectionHeaders,
                      bufferSize,
                      (float)((float)m_pageSize * m_numOfCanaryProtectionHeaders / (float)bufferSize));
        }
        DataChunkMmuBufferMapper::AllocateAndMap(bufferSize,
                                                 m_pHostBuffersMapper,
                                                 m_pMinimalCacheChunk,
                                                 m_pMinimalCacheHostVirtualAddress,
                                                 writeProtect);
        if (m_numOfCanaryProtectionHeaders > 0)
        {
            _setBufferCanaryProtection(PROT_NONE);
        }
    }

    // Allocating at the "specific" level, to ensure that we first verify all parameters
    bool status = _allocateDataChunks(minimalCacheAmount, isMinimalCacheAllocation);
    UNUSED(status);  // For release
    HB_ASSERT(status, "Failed to allocate data-chunk cache");
}

DataChunksCacheMmuBuffer::~DataChunksCacheMmuBuffer()
{
    if (m_pMinimalCacheChunk)
    {
        try
        {
            uint64_t bufferSize = getSingleDataChunkSize() * getMinimalChunksAmount();
            if (m_numOfCanaryProtectionHeaders > 0)
            {
                _setBufferCanaryProtection(PROT_READ | PROT_WRITE);
                LOG_DEBUG(SYN_DATA_CHUNK,
                          "{}: unProtected DC buffer - added {} to buffer with size {} = {}",
                          HLLOG_FUNC,
                          (m_numOfCanaryProtectionHeaders)*m_pageSize,
                          bufferSize,
                          (float)((float)(m_numOfCanaryProtectionHeaders)*m_pageSize) / (float)bufferSize);
                bufferSize += (m_numOfCanaryProtectionHeaders)*m_pageSize;
            }
            DataChunkMmuBufferMapper::UnmapAndDeallocate(bufferSize, m_pMinimalCacheChunk, m_pHostBuffersMapper);
        }
        catch (const std::exception& err)
        {
            LOG_WARN(SYN_DATA_CHUNK,
                     "{}: Failed to unmap data-chunk 0x{:x} due to {}",
                     HLLOG_FUNC,
                     (uint64_t)m_pMinimalCacheChunk,
                     err.what());
            // *May* result by a memory-leak, although not expected, as we don't try to delete the buffer,
            // as that may lead to double-deletion, which is worse...
        }
    }
}

void DataChunksCacheMmuBuffer::_setBufferCanaryProtection(uint32_t prot)
{
    for (size_t i = 0; i < m_numOfCanaryProtectionHeaders - 1; i++)  // calling mprotect on first (n -1)/k + 1 headers
    {
        void* address = (void*)((uint64_t)m_pMinimalCacheChunk +
                                i * GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER.value() * getSingleDataChunkSize() +
                                i * m_pageSize);

        int ret = mprotect(address, m_pageSize, prot);
        if (ret != 0)
        {
            LOG_ERR(SYN_DATA_CHUNK,
                    "{}: unable to mprotect address {:#x} with flag {} data chunk {} return value {} errno {} ({})",
                    HLLOG_FUNC,
                    (uint64_t)address,
                    prot,
                    i,
                    ret,
                    errno,
                    strerror(errno));
        }
    }
    int ret = mprotect((void*)((uint64_t)m_pMinimalCacheChunk + getSingleDataChunkSize() * getMinimalChunksAmount() +
                               (m_numOfCanaryProtectionHeaders - 1) * m_pageSize),
                       m_pageSize,
                       prot);  // footer
    if (ret != 0)
    {
        LOG_ERR(SYN_DATA_CHUNK,
                "{}: unable to mprotect with flag {} data chunk footer return value {} errno {} ({})",
                HLLOG_FUNC,
                prot,
                ret,
                errno,
                strerror(errno));
    }
}

bool DataChunksCacheMmuBuffer::_createDataChunks(DataChunksDB& freeDataChunks,
                                                 uint64_t      additionalChunksRequired,
                                                 bool          isMinimalCacheAllocation)
{
    for (uint64_t i = 0; i < additionalChunksRequired; i++)
    {
        DataChunk* pNewDataChunk = nullptr;
        try
        {
            DataChunkMmuBuffer* pDataChunk = nullptr;
            if (isMinimalCacheAllocation)
            {
                uint64_t offset = (i * _getSingleChunkSize());
                if (m_numOfCanaryProtectionHeaders > 0)
                {
                    offset += (i / GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER.value() + 1) * m_pageSize;
                }
                uint8_t* pChunk              = ((uint8_t*)m_pMinimalCacheChunk) + offset;
                uint8_t* pHostVirtualAddress = ((uint8_t*)m_pMinimalCacheHostVirtualAddress) + offset;

                pDataChunk = new DataChunkMmuBuffer(_getSingleChunkSize(),
                                                    (void*)pChunk,
                                                    (uint64_t)pHostVirtualAddress
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                    ,
                                                    m_statisticsId
#endif
                );
            }
            else
            {
                pDataChunk = new DataChunkMmuBufferMapper(_getSingleChunkSize(),
                                                          m_pHostBuffersMapper
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                          ,
                                                          m_statisticsId
#endif
                );
            }
            HB_ASSERT((pDataChunk != nullptr), "Failed to allocate DataChunk");

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
            DataChunksStatisticsManager::getInstance().dataChunkCreated(m_statisticsId, _getSingleChunkSize());
#endif
            pNewDataChunk = pDataChunk;
        }
        catch (const SynapseException& err)
        {
            LOG_DEBUG(SYN_DATA_CHUNK, "Cant allocate DataChunk due to {}", err.what());
            return false;
        }

        if (pNewDataChunk == nullptr)
        {
            LOG_DEBUG(SYN_DATA_CHUNK, "Cant to allocate DataChunkMmuBufferMapper");
            return false;
        }

        freeDataChunks.push_front(pNewDataChunk);
    }

    return true;
}

DataChunksCacheCommandBuffer::DataChunksCacheCommandBuffer(uint64_t singleChunkSize,
                                                           uint64_t minimalCacheAmount,
                                                           uint64_t maximalFreeCacheAmount,
                                                           uint64_t maximalCacheAmount
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                           ,
                                                           uint32_t statisticsId
#endif
                                                           ,
                                                           bool writeProtect)
: DataChunksCache(singleChunkSize,
                  minimalCacheAmount,
                  maximalFreeCacheAmount,
                  maximalCacheAmount,
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                  statisticsId,
#endif
                  MAPPING_TYPE_COMMAND_BUFFER_HANDLE)
{
    // Allocating at the "specific" level, to ensure that we first verify all parameters
    /* We are not required to use minimalCacheAmount allocation in DataChunksCacheCommandBuffer for the time being*/
    bool status = _allocateDataChunks(minimalCacheAmount, false);
    UNUSED(status);  // For release
    HB_ASSERT(status, "Failed to allocate data-chunk cache");
}

DataChunksCacheCommandBuffer::~DataChunksCacheCommandBuffer() {}

bool DataChunksCacheCommandBuffer::_createDataChunks(DataChunksDB& freeDataChunks,
                                                     uint64_t      additionalChunksRequired,
                                                     bool          isMinimalCacheAllocation)
{
    for (uint64_t i = 0; i < additionalChunksRequired; i++)
    {
        DataChunk* pNewDataChunk = nullptr;
        try
        {
            DataChunkCommandBuffer* pDataChunk = new DataChunkCommandBuffer(_getSingleChunkSize()
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                                                ,
                                                                            m_statisticsId
#endif
            );
            HB_ASSERT((pDataChunk != nullptr), "Failed to allocate DataChunk");

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
            DataChunksStatisticsManager::getInstance().dataChunkCreated(m_statisticsId, _getSingleChunkSize());
#endif
            pNewDataChunk = pDataChunk;
        }
        catch (const SynapseException& err)
        {
            LOG_DEBUG(SYN_DATA_CHUNK, "Cant allocate DataChunk due to {}", err.what());
            return false;
        }

        if (pNewDataChunk == nullptr)
        {
            LOG_DEBUG(SYN_DATA_CHUNK, "Cant to allocate DataChunkCommandBuffer");
            return false;
        }

        freeDataChunks.push_front(pNewDataChunk);
    }

    return true;
}

void DataChunksCache::dumpStat()
{
    logStat();

    if (m_statAlloc)
    {
        STAT_GLBL_COLLECT(m_statAlloc, dataChunkCacheAlloc);
        m_statAlloc = 0;
    }
    if (m_statCreated)
    {
        STAT_GLBL_COLLECT(m_statCreated, dataChunkCacheCreated);
        m_statCreated = 0;
    }
    if (m_statReleased)
    {
        STAT_GLBL_COLLECT(m_statReleased, dataChunkCacheReleased);
        m_statReleased = 0;
    }
    if (m_statDestruct)
    {
        STAT_GLBL_COLLECT(m_statDestruct, dataChunkCacheDestruct);
        m_statDestruct = 0;
    }

    STAT_GLBL_COLLECT(m_usedDataChunks.size(), usedDataChunks);
    STAT_GLBL_COLLECT(m_freeDataChunks.size(), freeDataChunks);
}

void DataChunksCache::logStat() const
{
    LOG_INFO(SYN_DATA_CHUNK, "singleChunkSize {} minimalCacheAmount {}", m_singleChunkSize, m_minimalCacheAmount);
    LOG_INFO(SYN_DATA_CHUNK, "statAlloc {} statCreated {}", m_statAlloc, m_statCreated);
    LOG_INFO(SYN_DATA_CHUNK, "statReleased {} statDestruct {}", m_statReleased, m_statDestruct);
    LOG_INFO(SYN_DATA_CHUNK, "usedDataChunks {} freeDataChunks {}", m_usedDataChunks.size(), m_freeDataChunks.size());
}
