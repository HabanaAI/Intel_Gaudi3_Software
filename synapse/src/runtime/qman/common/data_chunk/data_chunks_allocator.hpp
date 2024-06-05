#pragma once

#include "data_chunks_cache.hpp"
#include "data_chunks_statistics_manager.hpp"

#include <map>
#include <memory>
#include <mutex>

// A data chunks allocator is a SW component which holds multiple data-chunks-cache
// Upon a request to allocate some size of memory (data-chunk), it will select the best fit
// from the DC-Cache it holds
//
// Best fit - is currently defined as follows:
//      1) Find the minimal amount of DCs
//      2) Select proper cache to have minimal framentation
//
// Basically it means that the Allocator will have some chunk-size ordered cache elements.
// Let's say A(1) > A(2) > A(3) > ...
//
// Then, when it will allocate (reqSize), it will find the minimal cache that is bigger than requested size
// Meaning - A(i) > reqSize > A(i+1) => A(i). Hence...
// Selecting A(i), means a single DC with a fragmentation of [B(reqSize, i) = A(i) - reqSize]
// Selecting A(i+1), means at least two DCs [as B(reqSize, i) might be larger than A(i+1)], and a smaller fragmentation
//
// We will use the following rule of thumb:
//      Define selection in order of reducing fragmentation to minimum,
//      but have a hard-limit of no more than m_maximalAmountOfDataChunks of DCs (if possible)

class DataChunksAllocator
{
public:
    DataChunksAllocator(std::string                   description,
                        uint16_t                      maximalAmountOfDataChunks,
                        DataChunksCache::eMappingType mappingType  = DataChunksCache::MAPPING_TYPE_MMU,
                        bool                          isThreadSafe = false);

    virtual ~DataChunksAllocator() = default;

    // Upon requirments of the feature, we will only support *adding* allocators
    // They will be removed only upon DTR
    virtual bool addDataChunksCache(uint64_t singleChunkSize,
                                    uint64_t minimalCacheAmount,
                                    uint64_t maximalFreeCacheAmount,
                                    uint64_t maximalCacheAmount,
                                    bool     writeProtect            = false,
                                    bool     canaryProtectDataChunks = false) = 0;

    // -- Non thread-safe methods (start) -- //
    //
    bool acquireDataChunks(DataChunksDB& dataChunks, uint64_t dataSize);

    // Acquiring a specific amount of DC upon a given requested chunk-size (not even larger)
    bool acquireDataChunksAmount(DataChunksDB& dataChunks, uint64_t chunkSize, uint64_t chunkAmount);

    bool releaseDataChunks(const DataChunksDB& dataChunks);

    virtual uint64_t getMappedBufferSize() const = 0;

    // Delete DC in case DC-Cache is above max-free-threshold
    bool updateCache();

    uint64_t getAmountOfAvailableElements(uint64_t chunkSize);
    //
    // -- Non thread-safe methods (end)   -- //

    // -- Thread-safe methods (start)     -- //
    //
    bool acquireDataChunksThreadSafe(DataChunksDB& dataChunks, uint64_t dataSize);

    // Acquiring a specific amount of DC upon a given requested chunk-size (not even larger)
    bool acquireDataChunksAmountThreadSafe(DataChunksDB& dataChunks, uint64_t chunkSize, uint64_t chunkAmount);

    bool releaseDataChunksThreadSafe(const DataChunksDB& dataChunks);

    // Delete DC in case DC-Cache is above max-free-threshold
    bool updateCacheThreadSafe();
    //

    uint64_t getAmountOfAvailableElementsThreadSafe(uint64_t chunkSize) const;

    bool cleanupCsDataChunk(uint32_t& totalAvailableDataChunks, const DataChunksDB& dataChunks);

    bool cleanupCsDataChunkThreadSafe(uint32_t& totalAvailableDataChunks, const DataChunksDB& dataChunks);

    void dumpStat();

    // -- Thread-safe methods (end)       -- //

protected:
    // We want the map below to be ordered from biggest to samllest
    struct ChunkSizeComparator
    {
        bool operator()(uint64_t chunkSize1, uint64_t chunkSize2) const { return (chunkSize1 > chunkSize2); }
    };

    typedef std::shared_ptr<DataChunksCache>                             DataChunksCacheSptr;
    typedef std::map<uint64_t, DataChunksCacheSptr, ChunkSizeComparator> DataChunksCacheMap;
    typedef DataChunksCacheMap::iterator                                 DataChunksCacheMapIterator;
    typedef DataChunksCacheMap::reverse_iterator                         DataChunksCacheMapReverseIterator;

    bool _acquireDataChunks(DataChunksDB& dataChunks, uint64_t dataSize);

    // Acquiring a specific amount of DC upon a given requested chunk-size (not even larger)
    bool _acquireDataChunksAmount(DataChunksDB& dataChunks, uint64_t chunkSize, uint64_t chunkAmount);

    bool _releaseDataChunks(const DataChunksDB& dataChunks);

    // Delete DC in case DC-Cache is above max-free-threshold
    void _updateCache();

    // Calculate inner-fragmentation
    bool _calculateFragmentation(uint64_t            dataSize,
                                 DataChunksCacheSptr spDataChunkCache,
                                 uint64_t&           fragmentationSize,
                                 uint64_t&           requiredDataChunksAmount);

    bool _tryToAcquireDataChunks(DataChunksDB&       dataChunks,
                                 DataChunksCacheSptr spDataChunkCache,
                                 uint64_t            requiredDataChunksAmount);

    uint64_t _getAmountOfAvailableElements(uint64_t chunkSize) const;

    std::string m_description;

    // Ordered map of chunk-size to DC-Cache
    DataChunksCacheMap m_dataChunksCacheDB;

    DataChunksCache::eMappingType m_mappingType;

    bool m_isThreadSafe;

    mutable std::mutex m_mutex;

    const uint16_t m_maximalAmountOfDataChunks;

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    uint32_t m_statisticsId;
#endif
};

class DataChunksAllocatorMmuBuffer : public DataChunksAllocator
{
public:
    DataChunksAllocatorMmuBuffer(std::string        description,
                                 HostBuffersMapper* pHostBuffersMapper,
                                 uint16_t           maximalAmountOfDataChunks,
                                 bool               isThreadSafe = false);

    virtual ~DataChunksAllocatorMmuBuffer() = default;

    // Upon requirments of the feature, we will only support *adding* allocators
    // They will be removed only upon DTR
    virtual bool addDataChunksCache(uint64_t singleChunkSize,
                                    uint64_t minimalCacheAmount,
                                    uint64_t maximalFreeCacheAmount,
                                    uint64_t maximalCacheAmount,
                                    bool     writeProtect            = false,
                                    bool     canaryProtectDataChunks = false) override;

    virtual uint64_t getMappedBufferSize() const override;

private:
    HostBuffersMapper* m_pHostBuffersMapper;
};

class DataChunksAllocatorCommandBuffer : public DataChunksAllocator
{
public:
    DataChunksAllocatorCommandBuffer(std::string description,
                                     uint16_t    maximalAmountOfDataChunks,
                                     bool        isThreadSafe = false);

    virtual ~DataChunksAllocatorCommandBuffer();

    // Upon requirments of the feature, we will only support *adding* allocators
    // They will be removed only upon DTR
    virtual bool addDataChunksCache(uint64_t singleChunkSize,
                                    uint64_t minimalCacheAmount,
                                    uint64_t maximalFreeCacheAmount,
                                    uint64_t maximalCacheAmount,
                                    bool     writeProtect            = false,
                                    bool     canaryProtectDataChunks = false) override;

    virtual uint64_t getMappedBufferSize() const override;
};