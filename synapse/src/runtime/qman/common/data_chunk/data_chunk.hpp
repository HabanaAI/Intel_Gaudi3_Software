#pragma once

#include "data_chunks_statistics_manager.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "settable.h"

#include <cstdint>
#include <string>

class CommandBuffer;
class HostBuffersMapper;

void printDataChunksIds(const std::string& title, const DataChunksDB& dataChunks);

// DataChunk is a container for host-buffer that may have an handle to it
// Handle types - MMU-mapping handle, Command-Buffer handle or other

class DataChunk
{
public:
    DataChunk(uint64_t chunkSize
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
              ,
              uint32_t statisticsId
#endif
    );

    virtual ~DataChunk();

    bool initialize(uint64_t chunkId);

    void finalize();

    bool updateReferenceCounter(bool isIncrement);

    bool fillChunkData(void* pData, uint64_t dataSize);

    // Assumption - Sum of used-size and data-size will always be smaller than uint64_t max-value
    // Returns amount of data copied - data may be split over multiple DataChunks elements
    virtual bool fillChunkData(uint64_t& copiedSize, void* pData, uint64_t dataSize) = 0;

    virtual void resetUsedChunkArea() = 0;

    virtual bool updateUsedSize(uint64_t additionalUsedSize) = 0;

    virtual const void* getChunkBuffer() const = 0;

    // TODO - no reason to pass the buffer as a non-const buffer
    virtual void* getChunkBuffer() = 0;

    virtual uint64_t getHandle() = 0;

    virtual uint64_t getUsedSize() const = 0;

    virtual void* getNextChunkAddress() = 0;

    uint64_t getChunkId() const { return m_chunkId; }

    uint64_t getChunkSize() const { return m_chunkSize; }

    uint64_t getFreeSize() const { return m_chunkSize - getUsedSize(); }

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    uint32_t getStatisticsId() const { return m_statisticsId; }
#endif

protected:
    bool           m_isInitialized;
    uint64_t       m_chunkId;
    const uint64_t m_chunkSize;
    uint64_t       m_referenceCounter;

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
    uint32_t m_statisticsId;
#endif

    // The buffrer-size that the CP fetches on each call
    // This also defined the size of each packet to be a multiple of that size.
    // Hence, that a buffer must be a multiple of that size
    static const uint64_t m_cpFetchSize;
};

class DataChunkMmuBuffer : public DataChunk
{
public:
    DataChunkMmuBuffer(uint64_t chunkSize,
                       void*    pChunk,
                       uint64_t handle
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                       ,
                       uint32_t statisticsId
#endif
                       ,
                       bool write_protect = 0);

    virtual ~DataChunkMmuBuffer() = default;

    virtual bool fillChunkData(uint64_t& copiedSize, void* pData, uint64_t dataSize) override;

    virtual void resetUsedChunkArea() override;

    virtual bool updateUsedSize(uint64_t additionalUsedSize) override;

    virtual const void* getChunkBuffer() const override;

    // TODO - no reason to pass the buffer as a non-const buffer
    virtual void* getChunkBuffer() override;

    virtual uint64_t getHandle() override;

    virtual uint64_t getUsedSize() const override;

    virtual void* getNextChunkAddress() override;

    void dump() const;

protected:
    void*    m_pChunk;
    uint64_t m_handle;
    uint64_t m_usedSize;
};

class DataChunkMmuBufferMapper : public DataChunkMmuBuffer
{
public:
    DataChunkMmuBufferMapper(uint64_t           chunkSize,
                             HostBuffersMapper* pHostBufferMapper
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                             ,
                             uint32_t statisticsId
#endif
    );

    virtual ~DataChunkMmuBufferMapper();

    static void AllocateAndMap(uint64_t           chunkSize,
                               HostBuffersMapper* pHostBufferMapper,
                               void*&             pChunk,
                               uint64_t&          hostVirtualAddress,
                               bool               writeProtect = false);

    static void UnmapAndDeallocate(uint64_t chunkSize, void* pChunk, HostBuffersMapper* pHostBufferMapper);

private:
    HostBuffersMapper* m_pHostBuffersMapper;
};

class DataChunkCommandBuffer : public DataChunk
{
public:
    DataChunkCommandBuffer(uint64_t chunkSize
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                           ,
                           uint32_t statisticsId
#endif
    );

    virtual ~DataChunkCommandBuffer();

    virtual bool fillChunkData(uint64_t& copiedSize, void* pData, uint64_t dataSize) override;

    virtual void resetUsedChunkArea() override;

    virtual bool updateUsedSize(uint64_t additionalUsedSize) override;

    virtual const void* getChunkBuffer() const override;

    // TODO - no reason to pass the buffer as a non-const buffer
    virtual void* getChunkBuffer() override;

    virtual uint64_t getHandle() override;

    virtual uint64_t getUsedSize() const override;

    virtual void* getNextChunkAddress() override;

private:
    CommandBuffer* m_pCommandBuffer;
};

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
void dataChunksReady(DataChunksDB dataChunks, bool isLast = true);
#endif