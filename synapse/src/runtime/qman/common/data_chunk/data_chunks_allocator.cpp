#include "data_chunks_allocator.hpp"

#include "data_chunk.hpp"
#include "defenders.h"
#include "host_buffers_mapper.hpp"
#include "profiler_api.hpp"
#include "synapse_runtime_logging.h"
#include "types_exception.h"
#include "types.h"

#define VERIFY_THREAD_SAFENESS()                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (m_isThreadSafe)                                                                                            \
        {                                                                                                              \
            LOG_ERR(SYN_DATA_CHUNK, "{}: Unsupported call on a thread-safe mode", HLLOG_FUNC);                         \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

DataChunksAllocator::DataChunksAllocator(std::string                   description,
                                         uint16_t                      maximalAmountOfDataChunks,
                                         DataChunksCache::eMappingType mappingType,
                                         bool                          isThreadSafe)
: m_description(std::move(description)),
  m_mappingType(mappingType),
  m_isThreadSafe(isThreadSafe),
  m_maximalAmountOfDataChunks(maximalAmountOfDataChunks)
{
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    m_statisticsId = DataChunksStatisticsManager::getInstance().addAllocator();
    LOG_ERR(SYN_DATA_CHUNK, "Setting statistics ID {} for {}", m_statisticsId, description);
#endif
}

// -- Non thread-safe methods (start) -- //
//
bool DataChunksAllocator::acquireDataChunks(DataChunksDB& dataChunks, uint64_t dataSize)
{
    VERIFY_THREAD_SAFENESS();
    return _acquireDataChunks(dataChunks, dataSize);
}

bool DataChunksAllocator::acquireDataChunksAmount(DataChunksDB& dataChunks, uint64_t chunkSize, uint64_t chunkAmount)
{
    VERIFY_THREAD_SAFENESS();
    return _acquireDataChunksAmount(dataChunks, chunkSize, chunkAmount);
}

bool DataChunksAllocator::releaseDataChunks(const DataChunksDB& dataChunks)
{
    VERIFY_THREAD_SAFENESS();
    return _releaseDataChunks(dataChunks);
}

bool DataChunksAllocator::updateCache()
{
    VERIFY_THREAD_SAFENESS();
    _updateCache();

    return true;
}

uint64_t DataChunksAllocator::getAmountOfAvailableElements(uint64_t chunkSize)
{
    VERIFY_THREAD_SAFENESS();
    return _getAmountOfAvailableElements(chunkSize);
}
//
// -- Non thread-safe methods (end)   -- //

// -- Thread-safe methods (start)     -- //
//
bool DataChunksAllocator::acquireDataChunksThreadSafe(DataChunksDB& dataChunks, uint64_t dataSize)
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    return _acquireDataChunks(dataChunks, dataSize);
}

bool DataChunksAllocator::acquireDataChunksAmountThreadSafe(DataChunksDB& dataChunks,
                                                            uint64_t      chunkSize,
                                                            uint64_t      chunkAmount)
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    return _acquireDataChunksAmount(dataChunks, chunkSize, chunkAmount);
}

bool DataChunksAllocator::releaseDataChunksThreadSafe(const DataChunksDB& dataChunks)
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    return _releaseDataChunks(dataChunks);
}

bool DataChunksAllocator::updateCacheThreadSafe()
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    _updateCache();

    return true;
}

uint64_t DataChunksAllocator::getAmountOfAvailableElementsThreadSafe(uint64_t chunkSize) const
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    return _getAmountOfAvailableElements(chunkSize);
}
bool DataChunksAllocator::cleanupCsDataChunkThreadSafe(uint32_t&           totalAvailableDataChunks,
                                                       const DataChunksDB& dataChunks)
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    return cleanupCsDataChunk(totalAvailableDataChunks, dataChunks);
}

void DataChunksAllocator::dumpStat()
{
    std::unique_lock<std::mutex> mutex(m_mutex);
    for (auto& it : m_dataChunksCacheDB)
    {
        it.second->dumpStat();
    }
}

bool DataChunksAllocator::cleanupCsDataChunk(uint32_t& totalAvailableDataChunks, const DataChunksDB& dataChunks)
{
    if (!releaseDataChunks(dataChunks))
    {
        LOG_CRITICAL(SYN_DATA_CHUNK, "{}: Failed to release data-chunks", HLLOG_FUNC);
        return false;
    }
    totalAvailableDataChunks += dataChunks.size();

    return true;
}

//
// -- Thread-safe methods (end)       -- //

bool DataChunksAllocator::_acquireDataChunks(DataChunksDB& dataChunks, uint64_t dataSize)
{
    if (m_dataChunksCacheDB.empty())
    {
        LOG_ERR(SYN_DATA_CHUNK, "{}: Failed to allocate size 0x{:x} as DB is empty", HLLOG_FUNC, dataSize);
        return false;
    }

    uint64_t fragmentationSize        = 0;
    uint64_t requiredDataChunksAmount = 0;

    struct FragmentatioInfo
    {
        DataChunksCacheSptr spDataChunkCache;
        uint64_t            requiredDataChunksAmount;
    };
    typedef std::deque<FragmentatioInfo> FragmentatioInfoDeque;
    // Key   - fragmentationSize;
    // value - requiredDataChunksAmount ;
    typedef std::map<uint64_t, FragmentatioInfoDeque> DataChunkFragmentationInfo;

    DataChunkFragmentationInfo fragmentationInfoDB;
    FragmentatioInfo           currentFragmentationInfo = {0, 0};

    for (const auto& currentDataChunkCache : m_dataChunksCacheDB)
    {
        DataChunksCacheSptr spDataChunkCache = currentDataChunkCache.second;

        bool fragmentationStatus =
            _calculateFragmentation(dataSize, spDataChunkCache, fragmentationSize, requiredDataChunksAmount);

        if (fragmentationStatus)
        {
            currentFragmentationInfo.spDataChunkCache         = spDataChunkCache;
            currentFragmentationInfo.requiredDataChunksAmount = requiredDataChunksAmount;
            fragmentationInfoDB[fragmentationSize].push_back(currentFragmentationInfo);
        }
    }

    for (const auto& singleFramentaitonInfoDequeIter : fragmentationInfoDB)
    {
        const FragmentatioInfoDeque& singleFragmentationInfoDeque = singleFramentaitonInfoDequeIter.second;

        for (const auto& singleFragmentationInfo : singleFragmentationInfoDeque)
        {
            uint64_t chunksAmount = singleFragmentationInfo.requiredDataChunksAmount;

            bool status = _tryToAcquireDataChunks(dataChunks, singleFragmentationInfo.spDataChunkCache, chunksAmount);
            if (status)
            {
                uint64_t allocatedInnerFragmentation = singleFramentaitonInfoDequeIter.first;
                uint64_t chunkSize = singleFragmentationInfo.spDataChunkCache->getSingleDataChunkSize();

                if (allocatedInnerFragmentation > 1000)
                {
                    LOG_DEBUG(SYN_DATA_CHUNK,
                              "Allocated {} Data-Chunks with {} Chunk-Size and {} fragmentation (mapping-type {})",
                              chunksAmount,
                              chunkSize,
                              allocatedInnerFragmentation,
                              m_mappingType);
                }

                return true;
            }
        }
    }

    LOG_DEBUG(SYN_DATA_CHUNK, "Failed to allocate {} bytes of {} mapping-type", dataSize, m_mappingType);
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    DataChunksStatisticsManager::getInstance().printStatistics();
#endif
    return false;
}

uint64_t DataChunksAllocator::_getAmountOfAvailableElements(uint64_t chunkSize) const
{
    auto requestedChunkSizeIter = m_dataChunksCacheDB.find(chunkSize);
    auto requestedChunkSizeEnd  = m_dataChunksCacheDB.end();

    if (requestedChunkSizeIter == requestedChunkSizeEnd)
    {
        LOG_ERR(SYN_DATA_CHUNK, "{}: DC-Cache with requested Chunk-Size {} was not found", HLLOG_FUNC, chunkSize);
        return false;
    }
    const DataChunksCacheSptr& spRequestedDataChunkCache = (*requestedChunkSizeIter).second;
    return spRequestedDataChunkCache->getAmountOfAvailableElements();
}

bool DataChunksAllocator::_acquireDataChunksAmount(DataChunksDB& dataChunks, uint64_t chunkSize, uint64_t chunkAmount)
{
    auto requestedChunkSizeIter = m_dataChunksCacheDB.find(chunkSize);
    auto requestedChunkSizeEnd  = m_dataChunksCacheDB.end();

    if (requestedChunkSizeIter == requestedChunkSizeEnd)
    {
        LOG_ERR(SYN_DATA_CHUNK, "{}: DC-Cache with requested Chunk-Size {} was not found", HLLOG_FUNC, chunkSize);
        return false;
    }

    DataChunksCacheSptr& spRequestedDataChunkCache = (*requestedChunkSizeIter).second;
    return spRequestedDataChunkCache->acquireDataChunksAmount(dataChunks, chunkAmount);
}

bool DataChunksAllocator::_releaseDataChunks(const DataChunksDB& dataChunks)
{
    if (dataChunks.empty())
    {
        return true;
    }

    uint64_t numOfDataChunksReleased = 0;

    for (auto singleDataChunk : dataChunks)
    {
        auto dataChunkCacheIter = m_dataChunksCacheDB.find(singleDataChunk->getChunkSize());
        if (dataChunkCacheIter == m_dataChunksCacheDB.end())
        {
            LOG_ERR(SYN_DATA_CHUNK,
                    "{}: Failed to find DC-Cache for Data-Chunk {}",
                    HLLOG_FUNC,
                    singleDataChunk->getChunkId());

            continue;
        }

        bool status = dataChunkCacheIter->second->releaseSingleDataChunk(singleDataChunk->getChunkId());
        if (!status)
        {
            LOG_ERR(SYN_DATA_CHUNK, "{}: Failed to release Data-Chunk {}", HLLOG_FUNC, singleDataChunk->getChunkId());

            continue;
        }
        else
        {
            numOfDataChunksReleased++;
        }
    }

    if (dataChunks.size() != numOfDataChunksReleased)
    {
        LOG_ERR(SYN_DATA_CHUNK, "{}: Failed to clear all ({} left) Data-Chunks", HLLOG_FUNC, dataChunks.size());
        return false;
    }

    return true;
}

void DataChunksAllocator::_updateCache()
{
    for (auto dataChunkCache : m_dataChunksCacheDB)
    {
        HB_ASSERT((dataChunkCache.second != nullptr), "nullptr Data-Chunk-Cache was found in DC-Allocator map");
        dataChunkCache.second->updateCache();
    }
}

bool DataChunksAllocator::_calculateFragmentation(uint64_t            dataSize,
                                                  DataChunksCacheSptr spDataChunkCache,
                                                  uint64_t&           fragmentationSize,
                                                  uint64_t&           requiredDataChunksAmount)
{
    CHECK_POINTER(SYN_DATA_CHUNK, spDataChunkCache, "Data-chunk cache", false);

    uint64_t sizeOfSingleDataChunk       = spDataChunkCache->getSingleDataChunkSize();
    uint64_t availableAmountOfDataChunks = spDataChunkCache->getMaxAvailableChunksAmount();

    requiredDataChunksAmount = ((dataSize - 1) / sizeOfSingleDataChunk) + 1;

    if ((requiredDataChunksAmount > availableAmountOfDataChunks) ||
        (requiredDataChunksAmount > m_maximalAmountOfDataChunks))
    {
        LOG_DEBUG(SYN_DATA_CHUNK,
                  "{}: Stream {} Can not allocate - dataSize {} required {} available {} maximal-amount {}",
                  HLLOG_FUNC,
                  m_description,
                  dataSize,
                  requiredDataChunksAmount,
                  availableAmountOfDataChunks,
                  m_maximalAmountOfDataChunks);

        return false;
    }

    fragmentationSize = requiredDataChunksAmount * sizeOfSingleDataChunk - dataSize;
    return true;
}

bool DataChunksAllocator::_tryToAcquireDataChunks(DataChunksDB&       dataChunks,
                                                  DataChunksCacheSptr spDataChunkCache,
                                                  uint64_t            requiredDataChunksAmount)
{
    if (spDataChunkCache->acquireDataChunksAmount(dataChunks, requiredDataChunksAmount) == 0)
    {
        LOG_DEBUG(SYN_DATA_CHUNK,
                  "Failed to acquire {} Data-Chunks with chunk-size {}",
                  requiredDataChunksAmount,
                  spDataChunkCache->getSingleDataChunkSize());
        spDataChunkCache->updateCache();
        return false;
    }

    return true;
}

DataChunksAllocatorMmuBuffer::DataChunksAllocatorMmuBuffer(std::string        description,
                                                           HostBuffersMapper* pHostBuffersMapper,
                                                           uint16_t           maximalAmountOfDataChunks,
                                                           bool               isThreadSafe)
: DataChunksAllocator(std::move(description),
                      maximalAmountOfDataChunks,
                      DataChunksCache::MAPPING_TYPE_MMU,
                      isThreadSafe),
  m_pHostBuffersMapper(pHostBuffersMapper)
{
    if (pHostBuffersMapper == nullptr)
    {
        throw SynapseException(
            "DataChunksAllocatorMmuBuffer: Constructed with nullptr as host-buffers-mapper parameter");
    }
}

bool DataChunksAllocatorMmuBuffer::addDataChunksCache(uint64_t singleChunkSize,
                                                      uint64_t minimalCacheAmount,
                                                      uint64_t maximalFreeCacheAmount,
                                                      uint64_t maximalCacheAmount,
                                                      bool     writeProtect,
                                                      bool     canaryProtectDataChunks)
{
    auto requestedChunkSizeIter = m_dataChunksCacheDB.find(singleChunkSize);
    auto requestedChunkSizeEnd  = m_dataChunksCacheDB.end();

    if (requestedChunkSizeIter != requestedChunkSizeEnd)
    {
        LOG_WARN(SYN_DATA_CHUNK, "{}: Requested chunk-size 0x{:x} had already been added", HLLOG_FUNC, singleChunkSize);
        return false;
    }

    m_dataChunksCacheDB[singleChunkSize] = std::make_shared<DataChunksCacheMmuBuffer>(m_pHostBuffersMapper,
                                                                                      singleChunkSize,
                                                                                      minimalCacheAmount,
                                                                                      maximalFreeCacheAmount,
                                                                                      maximalCacheAmount
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                                                      ,
                                                                                      m_statisticsId
#endif
                                                                                      ,
                                                                                      writeProtect,
                                                                                      canaryProtectDataChunks);

    return true;
}

uint64_t DataChunksAllocatorMmuBuffer::getMappedBufferSize() const
{
    return m_pHostBuffersMapper->getMappedSize();
}

DataChunksAllocatorCommandBuffer::DataChunksAllocatorCommandBuffer(std::string description,
                                                                   uint16_t    maximalAmountOfDataChunks,
                                                                   bool        isThreadSafe)
: DataChunksAllocator(std::move(description),
                      maximalAmountOfDataChunks,
                      DataChunksCache::MAPPING_TYPE_COMMAND_BUFFER_HANDLE,
                      isThreadSafe)
{
}

DataChunksAllocatorCommandBuffer::~DataChunksAllocatorCommandBuffer() {}

bool DataChunksAllocatorCommandBuffer::addDataChunksCache(uint64_t singleChunkSize,
                                                          uint64_t minimalCacheAmount,
                                                          uint64_t maximalFreeCacheAmount,
                                                          uint64_t maximalCacheAmount,
                                                          bool     writeProtect,
                                                          bool     canaryProtectDataChunks)
{
    auto requestedChunkSizeIter = m_dataChunksCacheDB.find(singleChunkSize);
    auto requestedChunkSizeEnd  = m_dataChunksCacheDB.end();

    if (requestedChunkSizeIter != requestedChunkSizeEnd)
    {
        LOG_WARN(SYN_DATA_CHUNK, "{}: Requested chunk-size 0x{:x} had already been added", HLLOG_FUNC, singleChunkSize);
        return false;
    }

    m_dataChunksCacheDB[singleChunkSize] = std::make_shared<DataChunksCacheCommandBuffer>(singleChunkSize,
                                                                                          minimalCacheAmount,
                                                                                          maximalFreeCacheAmount,
                                                                                          maximalCacheAmount
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                                                          ,
                                                                                          m_statisticsId
#endif
                                                                                          ,
                                                                                          writeProtect);

    return true;
}

uint64_t DataChunksAllocatorCommandBuffer::getMappedBufferSize() const
{
    uint64_t totalSize = 0;
    for (auto& it : m_dataChunksCacheDB)
    {
        totalSize += it.second->getAllocatedDataChunkSize();
    }
    return totalSize;
}
