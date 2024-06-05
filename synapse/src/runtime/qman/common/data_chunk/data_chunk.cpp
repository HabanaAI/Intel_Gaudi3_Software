#include "data_chunk.hpp"

#include "defenders.h"
#include "host_buffers_mapper.hpp"
#include "memory_allocator_utils.hpp"
#include "runtime/qman/common/command_buffer.hpp"
#include "synapse_runtime_logging.h"
#include "types_exception.h"
#include "types.h"

const uint64_t DataChunk::m_cpFetchSize = 8;

void printDataChunksIds(const std::string& title, const DataChunksDB& dataChunks)
{
    LOG_ERR(SYN_API, "{} Data-Chunks:", title);

    for (auto dataChunkIter : dataChunks)
    {
        LOG_ERR(SYN_API,
                "0x{:x} Chunk-ID {} Mapped-Address [ 0x{:x}, 0x{:x} )",
                (uint64_t)dataChunkIter,
                dataChunkIter->getChunkId(),
                dataChunkIter->getHandle(),
                dataChunkIter->getHandle() + dataChunkIter->getChunkSize());
    }
}

DataChunk::DataChunk(uint64_t chunkSize
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                     ,
                     uint32_t statisticsId
#endif
                     )
: m_isInitialized(false),
  m_chunkSize(chunkSize),
  m_referenceCounter(0)
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
  ,
  m_statisticsId(statisticsId)
#endif
{
    if ((chunkSize == 0) || ((chunkSize % m_cpFetchSize) != 0))
    {
        throw SynapseException("DataChunks initialized with invalid chunk-size");
    }
}

DataChunk::~DataChunk() {}

bool DataChunk::initialize(uint64_t chunkId)
{
    if (m_isInitialized)
    {
        // No reason to validate that it is not initialized with the same chunk-id

        LOG_ERR(SYN_DATA_CHUNK, "{}: Already initialized (with chunk-ID {})", HLLOG_FUNC, m_chunkId);
        return false;
    }

    m_chunkId       = chunkId;
    m_isInitialized = true;

    return true;
}

void DataChunk::finalize()
{
    m_isInitialized = false;
    resetUsedChunkArea();
}

bool DataChunk::updateReferenceCounter(bool isIncrement)
{
    if (isIncrement)
    {
        m_referenceCounter++;
    }
    else
    {
        if (m_referenceCounter == 0)
        {
            LOG_ERR(SYN_DATA_CHUNK, "Can not decrement (zero) reference-counter for Chunk-ID {}", m_chunkId);
            return false;
        }

        m_referenceCounter--;
    }

    return true;
}

bool DataChunk::fillChunkData(void* pData, uint64_t dataSize)
{
    uint64_t copiedSize = 0;

    bool status = fillChunkData(copiedSize, pData, dataSize);
    return ((status) && (copiedSize == dataSize));
}

DataChunkMmuBuffer::DataChunkMmuBuffer(uint64_t chunkSize,
                                       void*    pChunk,
                                       uint64_t handle
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                       ,
                                       uint32_t statisticsId
#endif
                                       ,
                                       bool writeProtect)
: DataChunk(chunkSize
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
            ,
            statisticsId
#endif
            ),
  m_pChunk(pChunk),
  m_handle(handle),
  m_usedSize(0)
{
}

bool DataChunkMmuBuffer::fillChunkData(uint64_t& copiedSize, void* pData, uint64_t dataSize)
{
    CHECK_POINTER(SYN_DATA_CHUNK, pData, "Data", false);

    copiedSize = dataSize;

    // Assumption - sum is smaller than uint64_t max-value
    if (dataSize + m_usedSize > m_chunkSize)
    {
        copiedSize = getFreeSize();
    }

    std::memcpy((uint8_t*)m_pChunk + m_usedSize, pData, copiedSize);
    m_usedSize += copiedSize;

    return true;
}

void DataChunkMmuBuffer::resetUsedChunkArea()
{
    m_usedSize = 0;
}

bool DataChunkMmuBuffer::updateUsedSize(uint64_t additionalUsedSize)
{
    m_usedSize += additionalUsedSize;

    return true;
}

const void* DataChunkMmuBuffer::getChunkBuffer() const
{
    return m_pChunk;
}

// TODO - no reason to pass the buffer as a non-const buffer
void* DataChunkMmuBuffer::getChunkBuffer()
{
    return m_pChunk;
}

uint64_t DataChunkMmuBuffer::getHandle()
{
    return m_handle;
}

uint64_t DataChunkMmuBuffer::getUsedSize() const
{
    return m_usedSize;
}

void* DataChunkMmuBuffer::getNextChunkAddress()
{
    return ((uint8_t*)m_pChunk) + getUsedSize();
}

DataChunkMmuBufferMapper::DataChunkMmuBufferMapper(uint64_t           chunkSize,
                                                   HostBuffersMapper* pHostBufferMapper
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                                   ,
                                                   uint32_t statisticsId
#endif
                                                   )
: DataChunkMmuBuffer(chunkSize,
                     nullptr,
                     0
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                     ,
                     statisticsId
#endif
                     ),
  m_pHostBuffersMapper(pHostBufferMapper)
{
    AllocateAndMap(m_chunkSize, m_pHostBuffersMapper, m_pChunk, m_handle);
}

DataChunkMmuBufferMapper::~DataChunkMmuBufferMapper()
{
    try
    {
        UnmapAndDeallocate(m_chunkSize, m_pChunk, m_pHostBuffersMapper);
    }
    catch (const std::exception& err)
    {
        LOG_WARN(SYN_DATA_CHUNK,
                 "{}: Failed to unmap data-chunk 0x{:x} due to {}",
                 HLLOG_FUNC,
                 (uint64_t)m_pChunk,
                 err.what());
        // *May* result by a memory-leak, although not expected, as we don't try to delete the buffer,
        // as that may lead to double-deletion, which is worse...
    }
}

void DataChunkMmuBufferMapper::AllocateAndMap(uint64_t           chunkSize,
                                              HostBuffersMapper* pHostBufferMapper,
                                              void*&             pChunk,
                                              uint64_t&          hostVirtualAddress,
                                              bool               writeProtect)
{
    int prot = PROT_READ;
    if (!writeProtect) prot |= PROT_WRITE;

    pChunk = MemoryAllocatorUtils::alloc_memory_to_be_mapped_to_device(chunkSize, nullptr, prot);

    std::string     mappingDesc("Data-Chunk");
    const synStatus status = pHostBufferMapper->mapBuffer(hostVirtualAddress, pChunk, chunkSize, false, mappingDesc);
    if (status != synSuccess)
    {
        throw SynapseException("Failed to map data-chunk");
    }
}

void DataChunkMmuBufferMapper::UnmapAndDeallocate(uint64_t           chunkSize,
                                                  void*              pChunk,
                                                  HostBuffersMapper* pHostBufferMapper)
{
    const synStatus status = pHostBufferMapper->unmapBuffer(pChunk, false);
    if (status != synSuccess)
    {
        throw SynapseException("Failed to unmap data-chunk of Chunk-ID");
    }

    MemoryAllocatorUtils::free_memory(pChunk, chunkSize);
}

DataChunkCommandBuffer::DataChunkCommandBuffer(uint64_t chunkSize
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
                                               ,
                                               uint32_t statisticsId
#endif
                                               )
: DataChunk(chunkSize
#ifdef ENABLE_DATA_CHUNKS_STATISTICS
            ,
            statisticsId
#endif
  )
{
    synStatus status = CommandBufferMap::GetInstance()->AddCommandBuffer(chunkSize, &m_pCommandBuffer);
    if (status != synSuccess)
    {
        throw SynapseException("Failed to create DataChunk's command buffer");
    }
}

DataChunkCommandBuffer::~DataChunkCommandBuffer()
{
    synStatus status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pCommandBuffer);
    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_DATA_CHUNK, "{}: Failed to remove DataChunk's command buffer", HLLOG_FUNC);
    }
}

bool DataChunkCommandBuffer::fillChunkData(uint64_t& copiedSize, void* pData, uint64_t dataSize)
{
    CHECK_POINTER(SYN_DATA_CHUNK, pData, "Data", false);

    copiedSize = dataSize;

    // Assumption - sum is smaller than uint64_t max-value
    if (dataSize + getUsedSize() > m_chunkSize)
    {
        copiedSize = getFreeSize();
    }

    if (m_pCommandBuffer->SetBufferToCB(pData, copiedSize) != synSuccess)
    {
        LOG_ERR(SYN_DATA_CHUNK, "{}: Failed to fill command buffer with {} of data", HLLOG_FUNC, copiedSize);
        return false;
    }

    return true;
}

void DataChunkCommandBuffer::resetUsedChunkArea()
{
    m_pCommandBuffer->ClearCB();
}

bool DataChunkCommandBuffer::updateUsedSize(uint64_t additionalUsedSize)
{
    synStatus status = m_pCommandBuffer->UpdateOccupiedSize(additionalUsedSize);

    return (status == synSuccess);
}

const void* DataChunkCommandBuffer::getChunkBuffer() const
{
    return m_pCommandBuffer->GetBufferMap();
}

// TODO - no reason to pass the buffer as a non-const buffer
void* DataChunkCommandBuffer::getChunkBuffer()
{
    return m_pCommandBuffer->GetBufferMap();
}

uint64_t DataChunkCommandBuffer::getHandle()
{
    return m_pCommandBuffer->GetCbHandle();
}

uint64_t DataChunkCommandBuffer::getUsedSize() const
{
    return m_pCommandBuffer->GetOccupiedSize();
}

void* DataChunkCommandBuffer::getNextChunkAddress()
{
    return ((uint8_t*)getChunkBuffer()) + getUsedSize();
}

#ifdef ENABLE_DATA_CHUNKS_STATISTICS
void dataChunksReady(DataChunksDB dataChunks, bool isLast /* = true */)
{
    uint64_t   totalUsedSize = 0;
    auto       dataChunkIter = dataChunks.begin();
    DataChunk* dataChunk     = *dataChunkIter;
    uint32_t   statisticsId  = dataChunk->getStatisticsId();
    uint64_t   chunkSize     = dataChunk->getChunkSize();

    for (auto singleDataChunk : dataChunks)
    {
        totalUsedSize += singleDataChunk->getUsedSize();
    }

    DataChunksStatisticsManager::getInstance().dataChunkReady(statisticsId,
                                                              totalUsedSize,
                                                              chunkSize,
                                                              dataChunks.size(),
                                                              isLast);
}
#endif

void DataChunkMmuBuffer::dump() const
{
    for (unsigned iter = 0; iter < m_usedSize / sizeof(uint64_t); iter++)
    {
        uint64_t* addr = &(((uint64_t*)m_pChunk)[iter]);
        uint64_t  val  = ((uint64_t*)m_pChunk)[iter];
        LOG_DEBUG(SYN_DATA_CHUNK,
                  "m_handle {:x} m_usedSize {} iter {} addr {:x} val {:x}",
                  m_handle,
                  m_usedSize,
                  iter,
                  TO64(addr),
                  val);
    }
}