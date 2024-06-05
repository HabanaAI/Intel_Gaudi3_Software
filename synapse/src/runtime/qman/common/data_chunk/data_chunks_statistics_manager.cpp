#include "data_chunks_statistics_manager.hpp"

#ifdef ENABLE_DATA_CHUNKS_STATISTICS

#include "synapse_runtime_logging.h"

const uint32_t DataChunksStatisticsManager::m_managerAllocatorId = 0;

DataChunksStatisticsManager::DataChunksStatisticsManager() : m_nextAllocatorId(m_managerAllocatorId + 1) {}

uint32_t DataChunksStatisticsManager::addAllocator()
{
    uint32_t allocatorId = m_nextAllocatorId++;
    HB_ASSERT((allocatorId != m_managerAllocatorId), "Allocator's ID can not be Manager's one");
    return allocatorId;
}

void DataChunksStatisticsManager::dataChunkCreated(uint32_t allocatorId, uint64_t chunkSize)
{
    std::unique_lock<std::recursive_mutex> mutex(m_mutex);

    SingleAllocatorStatistics& aggregatorAllocatorStatistics = m_aggregatedStatistics;
    SingleCacheStatistics&     aggregatorCacheStatistics     = aggregatorAllocatorStatistics[chunkSize];
    aggregatorCacheStatistics.amountOfCreatedDataChunks++;

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    SingleAllocatorStatistics& allocatorStatistics = m_allAllocatorsStatistics[allocatorId];
    SingleCacheStatistics&     cacheStatistics     = allocatorStatistics[chunkSize];
    cacheStatistics.amountOfCreatedDataChunks++;
#endif
}

void DataChunksStatisticsManager::dataChunkDestroyed(uint32_t allocatorId, uint64_t chunkSize)
{
    std::unique_lock<std::recursive_mutex> mutex(m_mutex);

    SingleAllocatorStatistics& aggregatorAllocatorStatistics = m_aggregatedStatistics;
    SingleCacheStatistics&     aggregatorCacheStatistics     = aggregatorAllocatorStatistics[chunkSize];
    aggregatorCacheStatistics.amountOfCreatedDataChunks--;

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    SingleAllocatorStatistics& allocatorStatistics = m_allAllocatorsStatistics[allocatorId];
    SingleCacheStatistics&     cacheStatistics     = allocatorStatistics[chunkSize];
    cacheStatistics.amountOfCreatedDataChunks--;
#endif
}

void DataChunksStatisticsManager::dataChunkAcquired(uint32_t allocatorId, uint64_t chunkSize)
{
    std::unique_lock<std::recursive_mutex> mutex(m_mutex);

    SingleAllocatorStatistics& aggregatorAllocatorStatistics = m_aggregatedStatistics;
    SingleCacheStatistics&     aggregatorCacheStatistics     = aggregatorAllocatorStatistics[chunkSize];
    aggregatorCacheStatistics.amountOfAcquiredDataChunks++;

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    SingleAllocatorStatistics& allocatorStatistics = m_allAllocatorsStatistics[allocatorId];
    SingleCacheStatistics&     cacheStatistics     = allocatorStatistics[chunkSize];
    cacheStatistics.amountOfAcquiredDataChunks++;
#endif
}

void DataChunksStatisticsManager::dataChunkReleased(uint32_t allocatorId, uint64_t releasedSpace, uint64_t chunkSize)
{
    std::unique_lock<std::recursive_mutex> mutex(m_mutex);

    SingleAllocatorStatistics& aggregatorAllocatorStatistics = m_aggregatedStatistics;
    SingleCacheStatistics&     aggregatorCacheStatistics     = aggregatorAllocatorStatistics[chunkSize];
    aggregatorCacheStatistics.amountOfAcquiredDataChunks--;
    aggregatorCacheStatistics.totalUsedSpace -= releasedSpace;

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    SingleAllocatorStatistics& allocatorStatistics = m_allAllocatorsStatistics[allocatorId];
    SingleCacheStatistics&     cacheStatistics     = allocatorStatistics[chunkSize];
    cacheStatistics.amountOfAcquiredDataChunks--;
    cacheStatistics.totalUsedSpace -= releasedSpace;
#endif
}

void DataChunksStatisticsManager::dataChunkReady(uint32_t allocatorId,
                                                 uint64_t usedSpace,
                                                 uint64_t chunkSize,
                                                 uint64_t chunksAmount,
                                                 bool     isLast)
{
    std::unique_lock<std::recursive_mutex> mutex(m_mutex);

    if (chunksAmount == 0)
    {
        return;
    }

    SingleAllocatorStatistics& aggregatorAllocatorStatistics = m_aggregatedStatistics;
    SingleCacheStatistics&     aggregatorCacheStatistics     = aggregatorAllocatorStatistics[chunkSize];
    aggregatorCacheStatistics.totalUsedSpace += usedSpace;

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    SingleAllocatorStatistics& allocatorStatistics = m_allAllocatorsStatistics[allocatorId];
    SingleCacheStatistics&     cacheStatistics     = allocatorStatistics[chunkSize];
    cacheStatistics.totalUsedSpace += usedSpace;
#endif

#ifdef ENABLE_PRINT_UPON_DC_CHUNLS_READY
    uint64_t acquiredSize    = chunksAmount * chunkSize;
    uint64_t wastedSize      = acquiredSize - usedSpace;
    double   wastePercentage = ((double)wastedSize / (double)acquiredSize) * 100;
    LOG_ERR(SYN_DATA_CHUNK,
            "Local statistics ({}): Acquired {} Wasted {} (Waste-percentage {}) over {} chunks",
            allocatorId,
            acquiredSize,
            wastedSize,
            wastePercentage,
            chunksAmount);
    if (isLast)
    {
        printStatistics(false);
    }
#endif
}

void DataChunksStatisticsManager::printStatistics(bool isUserRequest)
{
    std::unique_lock<std::recursive_mutex> mutex(m_mutex);

    if ((!isUserRequest) && (!_nonUserExplicitPrintoutCriteria()))
    {
        return;
    }

    LOG_ERR(SYN_DATA_CHUNK, "** Data-Chunks statistics printouts **");
    _printStatistics(m_managerAllocatorId, m_aggregatedStatistics);

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    for (auto allocatorStatsIter : m_allAllocatorsStatistics)
    {
        uint32_t allocatorId = allocatorStatsIter.first;

        SingleAllocatorStatistics& allocatorStatistics = allocatorStatsIter.second;

        _printStatistics(allocatorId, allocatorStatistics);
    }
#endif
}

void DataChunksStatisticsManager::_printStatistics(uint32_t                         allocatorId,
                                                   const SingleAllocatorStatistics& allocatorStatistics)
{
    uint64_t totalAcquiredSize = 0;
    uint64_t totalChunksAmount = 0;
    uint64_t totalUsedSize     = 0;

    LOG_ERR(SYN_DATA_CHUNK, "Data-Chunk statistics for allocator-ID {}:", allocatorId);

    for (auto singleChunkSizeStatsIter : allocatorStatistics)
    {
        uint64_t                     chunkSize       = singleChunkSizeStatsIter.first;
        const SingleCacheStatistics& cacheStatistics = singleChunkSizeStatsIter.second;

        uint64_t acquiredAmount  = cacheStatistics.amountOfAcquiredDataChunks;
        uint64_t acquiredSize    = acquiredAmount * chunkSize;
        uint64_t wastedSize      = acquiredSize - cacheStatistics.totalUsedSpace;
        double   wastePercentage = 0;

        if (acquiredSize != 0)
        {
            wastePercentage = ((double)wastedSize / (double)acquiredSize) * 100;
        }

        LOG_ERR(SYN_DATA_CHUNK,
                "Chunk-size {} - Created {} Acquired {} (size {}) Wasted {} waste-percentage {}",
                chunkSize,
                cacheStatistics.amountOfCreatedDataChunks,
                acquiredAmount,
                acquiredSize,
                wastedSize,
                wastePercentage);

        totalAcquiredSize += acquiredSize;
        totalChunksAmount += acquiredAmount;
        totalUsedSize += cacheStatistics.totalUsedSpace;
    }

    uint64_t wastedSize      = totalAcquiredSize - totalUsedSize;
    double   wastePercentage = 0;
    if (totalAcquiredSize != 0)
    {
        wastePercentage = ((double)wastedSize / (double)totalAcquiredSize) * 100;
    }
    LOG_ERR(SYN_DATA_CHUNK,
            "Total statistics: Acquired-Size {} Wasted {} waste-percentage {} over {} chunks",
            totalAcquiredSize,
            wastedSize,
            wastePercentage,
            totalChunksAmount);
}

bool DataChunksStatisticsManager::_nonUserExplicitPrintoutCriteria()
{
    static uint32_t nonUserCounter         = 0;
    const uint32_t  nonUserCounterCriteria = 100;

    nonUserCounter++;
    if (nonUserCounter != nonUserCounterCriteria)
    {
        return false;
    }
    else
    {
        nonUserCounter = 0;
    }

    return true;
}

#endif
