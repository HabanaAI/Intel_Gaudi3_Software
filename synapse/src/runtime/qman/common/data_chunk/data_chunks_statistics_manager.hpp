#pragma once

// #define ENABLE_DATA_CHUNKS_STATISTICS

#ifdef ENABLE_DATA_CHUNKS_STATISTICS

// When disabled, only aggregated statistics will be gathered
#define ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
#define ENABLE_PRINT_UPON_DC_CHUNLS_READY

// For printing stats not upon failure (or any specific user trigger - printStatistics request),
// One should update the _nonUserExplicitPrintoutCriteria method

#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <cstdint>

class DataChunksStatisticsManager
{
public:
    static DataChunksStatisticsManager& getInstance()
    {
        static DataChunksStatisticsManager instance;
        return instance;
    }

    uint32_t addAllocator();

    void dataChunkCreated(uint32_t allocatorId, uint64_t chunkSize);

    void dataChunkDestroyed(uint32_t allocatorId, uint64_t chunkSize);

    void dataChunkAcquired(uint32_t allocatorId, uint64_t chunkSize);

    void dataChunkReleased(uint32_t allocatorId, uint64_t releasedSpace, uint64_t chunkSize);

    void dataChunkReady(uint32_t allocatorId,
                        uint64_t usedSpace,
                        uint64_t chunkSize,
                        uint64_t chunksAmount,
                        bool     isLast = true);

    void printStatistics(bool isUserRequest = true);

private:
    struct SingleCacheStatistics
    {
        SingleCacheStatistics()
        {
            amountOfCreatedDataChunks  = 0;
            amountOfAcquiredDataChunks = 0;
            totalUsedSpace             = 0;
        }

        uint64_t amountOfCreatedDataChunks = 0;
        // An acquired DC is also an allocated one
        uint64_t amountOfAcquiredDataChunks = 0;

        uint64_t totalUsedSpace = 0;
    };

    // Key is Cache' Chunk-Size
    typedef std::map<uint64_t, SingleCacheStatistics> SingleAllocatorStatistics;

    DataChunksStatisticsManager();

    void _printStatistics(uint32_t allocatorId, const SingleAllocatorStatistics& allocatorStatistics);

    bool _nonUserExplicitPrintoutCriteria();

#ifdef ENABLE_DATA_CHUNKS_ALLOCATORS_STATISTICS
    std::map<uint32_t, SingleAllocatorStatistics> m_allAllocatorsStatistics;
#endif

    SingleAllocatorStatistics m_aggregatedStatistics;

    mutable std::recursive_mutex m_mutex;

    std::atomic<uint32_t> m_nextAllocatorId;

    static const uint32_t m_managerAllocatorId;
};

#endif