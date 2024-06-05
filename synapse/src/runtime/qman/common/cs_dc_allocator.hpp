#pragma once

#include <array>
#include "synapse_common_types.h"
#include "recipe_package_types.hpp"
#include "runtime/qman/common/data_chunk/data_chunks_allocator.hpp"

class CommandSubmissionDataChunks;
class MemoryManager;

#define dcAmountAvailableCpDma   dcAmountsAvailable[CS_DC_ALLOCATOR_CP_DMA]
#define dcAmountAvailableCommand dcAmountsAvailable[CS_DC_ALLOCATOR_COMMAND]
#define dcAmountRequiredDma      dcAmountsRequired[CS_DC_ALLOCATOR_CP_DMA]
#define dcAmountRequiredCommand  dcAmountsRequired[CS_DC_ALLOCATOR_COMMAND]

enum eCsDcAllocatorTypes
{
    CS_DC_ALLOCATOR_CP_DMA,
    CS_DC_ALLOCATOR_COMMAND,
    CS_DC_ALLOCATOR_MAX
};

enum eCsDataChunkStatus
{
    CS_DATA_CHUNKS_STATUS_FAILURE,
    CS_DATA_CHUNKS_STATUS_NOT_COMPLETED,
    CS_DATA_CHUNKS_STATUS_COMPLETED
};

typedef std::array<DataChunksAllocatorMmuBuffer, CS_DC_ALLOCATOR_MAX> DataChunksAllocators;
typedef std::array<uint64_t, CS_DC_ALLOCATOR_MAX>             DataChunksAmounts;
typedef std::array<uint64_t, CS_DC_ALLOCATOR_MAX>             DataChunksSizes;
typedef std::array<DataChunksDB, CS_DC_ALLOCATOR_MAX>         DataChunksDBs;

class CsDcAllocator
{
public:
    CsDcAllocator(MemoryManager& rMemoryManager, bool isReduced);

    virtual ~CsDcAllocator();

    synStatus initAllocators();

    eCsDataChunkStatus tryToAcquireDataChunkAmounts(DataChunksAmounts&       dcAmountsAvailable,
                                                    DataChunksDBs&           dcDbs,
                                                    const DataChunksAmounts& dcAmountsRequired,
                                                    bool                     isForceAcquire);

    bool releaseDataChunks(CommandSubmissionDataChunks* pCsDataChunks);

    bool cleanupCsDataChunk(CommandSubmissionDataChunks* pCsDataChunks, DataChunksAmounts& dcAmountsAvailable);

    void updateCache();

    inline uint64_t getDcSizeCpDma() const { return m_dcSizes[CS_DC_ALLOCATOR_CP_DMA]; }

    inline uint64_t getDcSizeCommand() const { return m_dcSizes[CS_DC_ALLOCATOR_COMMAND]; }

    DataChunksAmounts getDataChunksAmounts() const;

private:
    MemoryManager& m_rMemoryManager;

    const DataChunksSizes m_dcSizes;

    const DataChunksAmounts m_dcAmounts;

    DataChunksAllocators m_allocators;

    static bool canaryProtectionAlreadyUsed;
};
