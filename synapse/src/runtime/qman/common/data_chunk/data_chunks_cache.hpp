#pragma once

#include "settable.h"

#include <map>
#include <cstdint>
#include "runtime/qman/common/qman_types.hpp"

class HostBuffersMapper;

// The chunk-id should be is an ID which is unique inside of a given cache
//
// No thread-safeness is guaranteed on the following classes
// DataChunk is a container for host-buffer (mapped)
// DataChunksCache is a cache (use & free DB) managing DataChunks acquire & release
// The class expects that the order of release will be aligned with its acquire order
// Meaning - acquire DC(0), DC(1) -> release DC(1), DC(0)
class DataChunksCache
{
public:
    enum eMappingType
    {
        MAPPING_TYPE_MMU,
        MAPPING_TYPE_COMMAND_BUFFER_HANDLE,
    };

    DataChunksCache(uint64_t singleChunkSize,
                    uint64_t minimalCacheAmount,
                    uint64_t maximalFreeCacheAmount,
                    uint64_t maximalCacheAmount,
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                    uint32_t statisticsId,
#endif
                    eMappingType mappingType = MAPPING_TYPE_MMU);

    virtual ~DataChunksCache();

    // Both acquireDataChunks methods returns amount of DC allocated
    uint64_t acquireDataChunks(DataChunksDB& dataChunks, uint64_t dataSize);

    virtual uint64_t acquireDataChunksAmount(DataChunksDB& dataChunks, uint64_t chunksAmount);

    bool acquireSingleDataChunk(DataChunk*& pDataChunk);

    uint64_t releaseDataChunks(const DataChunksDB& dataChunks, bool isBreakUponFailure = false);

    bool releaseSingleDataChunk(uint64_t dataChunkId);

    // Delete DC in case DC-Cache is above max-free-threshold
    void updateCache();

    uint64_t getSingleDataChunkSize() const { return m_singleChunkSize; };

    uint64_t getAllocatedDataChunkSize() const { return m_usedDataChunks.size() * m_singleChunkSize; }

    uint64_t getMaxAvailableChunksAmount() { return m_maximalCacheAmount - m_usedDataChunks.size(); }

    uint64_t getMinimalChunksAmount() { return m_minimalCacheAmount; }

    void dumpStat();

    void logStat() const;

    uint64_t getAmountOfAvailableElements() const;

protected:
    bool _allocateDataChunks(uint64_t numFreeChunksRequested, bool isMinimalCacheAllocation);

    void _acquireSingleDataChunk(DataChunk*& pDataChunk);

    bool _freeDataChunk(uint64_t chunkId);

    void _deleteDataChunk(DataChunk* pDataChunk);

    inline uint64_t _getSingleChunkSize() { return m_singleChunkSize; }

    virtual bool _createDataChunks(DataChunksDB& freeDataChunks,
                                   uint64_t      additionalChunksRequired,
                                   bool          isMinimalCacheAllocation) = 0;

protected:
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    uint32_t m_statisticsId;
#endif

private:
    std::map<uint64_t, DataChunk*> m_usedDataChunks;
    DataChunksDB                   m_freeDataChunks;
    uint64_t                       m_nextChunkId;
    const uint64_t                 m_singleChunkSize;
    uint64_t                       m_maximalFreeCacheAmount;
    uint64_t                       m_maximalCacheAmount;
    const uint64_t                 m_minimalCacheAmount;

    uint64_t m_statAlloc {};
    uint64_t m_statCreated {};
    uint64_t m_statReleased {};
    uint64_t m_statDestruct {};
};

class DataChunksCacheMmuBuffer : public DataChunksCache
{
public:
    DataChunksCacheMmuBuffer(HostBuffersMapper* pHostBuffersMapper,
                             uint64_t           singleChunkSize,
                             uint64_t           minimalCacheAmount,
                             uint64_t           maximalFreeCacheAmount,
                             uint64_t           maximalCacheAmount
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                             ,
                             uint32_t statisticsId
#endif
                             ,
                             bool writeProtect,
                             bool canaryProtectDataChunks);

    virtual ~DataChunksCacheMmuBuffer();

protected:
    virtual bool _createDataChunks(DataChunksDB& freeDataChunks,
                                   uint64_t      additionalChunksRequired,
                                   bool          isMinimalCacheAllocation) override;

    virtual void _setBufferCanaryProtection(uint32_t prot);

private:
    HostBuffersMapper* m_pHostBuffersMapper;
    void*              m_pMinimalCacheChunk;
    uint64_t           m_pMinimalCacheHostVirtualAddress;
    uint32_t           m_pageSize;
    uint64_t           m_numOfCanaryProtectionHeaders;
};

class DataChunksCacheCommandBuffer : public DataChunksCache
{
public:
    DataChunksCacheCommandBuffer(uint64_t singleChunkSize,
                                 uint64_t minimalCacheAmount,
                                 uint64_t maximalFreeCacheAmount,
                                 uint64_t maximalCacheAmount
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                 ,
                                 uint32_t statisticsId
#endif
                                 ,
                                 bool writeProtect = false);

    virtual ~DataChunksCacheCommandBuffer();

protected:
    virtual bool _createDataChunks(DataChunksDB& freeDataChunks,
                                   uint64_t      additionalChunksRequired,
                                   bool          isMinimalCacheAllocation) override;
};
