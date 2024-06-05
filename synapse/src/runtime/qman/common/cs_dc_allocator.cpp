#include "cs_dc_allocator.hpp"

#include "defs.h"
#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/memory_manager.hpp"
#include "types_exception.h"
#include "types.h"

bool CsDcAllocator::canaryProtectionAlreadyUsed         = false;

const uint16_t MAXIMAL_AMOUNT_OF_DATA_CHUNKS = 100;

CsDcAllocator::CsDcAllocator(MemoryManager& rMemoryManager, bool isReduced)
: m_rMemoryManager(rMemoryManager),
  m_dcSizes {GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP.value() * 1024,
             GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP.value() * 1024},
  m_dcAmounts {isReduced ? GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.value() /
                               GCFG_STREAM_COMPUTE_REDUCE_FACTOR.value()
                         : GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.value(),
               isReduced ? GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP.value() /
                               GCFG_STREAM_COMPUTE_REDUCE_FACTOR.value()
                         : GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP.value()},
  m_allocators {
      DataChunksAllocatorMmuBuffer("CS_DC_ALLOCATOR_CP_DMA", &m_rMemoryManager, MAXIMAL_AMOUNT_OF_DATA_CHUNKS),
      DataChunksAllocatorMmuBuffer("CS_DC_ALLOCATOR_COMMAND", &m_rMemoryManager, MAXIMAL_AMOUNT_OF_DATA_CHUNKS)}
{
    LOG_INFO(SYN_STREAM,
             "CsDcAllocator {:#x}, created m_dcSizes {} {} m_dcAmounts {} {} m_allocators {} m_allocators {}",
             TO64(this),
             m_dcSizes[CS_DC_ALLOCATOR_CP_DMA],
             m_dcSizes[CS_DC_ALLOCATOR_COMMAND],
             m_dcAmounts[CS_DC_ALLOCATOR_CP_DMA],
             m_dcAmounts[CS_DC_ALLOCATOR_COMMAND],
             TO64(&m_allocators[CS_DC_ALLOCATOR_CP_DMA]),
             TO64(&m_allocators[CS_DC_ALLOCATOR_COMMAND]));
}

CsDcAllocator::~CsDcAllocator()
{
}

synStatus CsDcAllocator::initAllocators()
{
    bool canaryProtectDataChunks = false;
    if (GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER.value())
    {
        if (!canaryProtectionAlreadyUsed)
        {
            LOG_DEBUG(SYN_DATA_CHUNK, "Protecting stream compute data chunks buffer");
            canaryProtectDataChunks = true;
        }
        canaryProtectionAlreadyUsed = true;
    }

    try
    {
        m_allocators[CS_DC_ALLOCATOR_CP_DMA].addDataChunksCache(m_dcSizes[CS_DC_ALLOCATOR_CP_DMA],
                                                                m_dcAmounts[CS_DC_ALLOCATOR_CP_DMA],
                                                                m_dcAmounts[CS_DC_ALLOCATOR_CP_DMA],
                                                                m_dcAmounts[CS_DC_ALLOCATOR_CP_DMA],
                                                                false,
                                                                canaryProtectDataChunks);

        m_allocators[CS_DC_ALLOCATOR_COMMAND].addDataChunksCache(m_dcSizes[CS_DC_ALLOCATOR_COMMAND],
                                                                 m_dcAmounts[CS_DC_ALLOCATOR_COMMAND],
                                                                 m_dcAmounts[CS_DC_ALLOCATOR_COMMAND],
                                                                 m_dcAmounts[CS_DC_ALLOCATOR_COMMAND],
                                                                 false,
                                                                 canaryProtectDataChunks);
    }
    catch (const SynapseException& err)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to create Data-Chunks cache on stream due to {}", HLLOG_FUNC, err.what());
        return synFail;
    }

    return synSuccess;
}

bool CsDcAllocator::releaseDataChunks(CommandSubmissionDataChunks* pCsDataChunks)
{
    HB_ASSERT_PTR(pCsDataChunks);

    const commandsDataChunksDB& commandsDataChunks = pCsDataChunks->getCommandsBufferDataChunks();

    //  Compute uses both CP-DMA data chunks and commands data-chunks
    if ((!m_allocators[CS_DC_ALLOCATOR_COMMAND].releaseDataChunksThreadSafe(commandsDataChunks)))
    {
        LOG_CRITICAL(SYN_STREAM,
                     "{}: Failed to release commands data-chunks of handle {:x}",
                     HLLOG_FUNC,
                     TO64(pCsDataChunks));
        return false;
    }

    const cpDmaDataChunksDB& cpDmaDataChunks = pCsDataChunks->getCpDmaDataChunks();
    if (!m_allocators[CS_DC_ALLOCATOR_CP_DMA].releaseDataChunksThreadSafe(cpDmaDataChunks))
    {
        LOG_CRITICAL(SYN_STREAM,
                     "{}: Failed to release CP-DMA data-chunks of handle {:x}",
                     HLLOG_FUNC,
                     TO64(pCsDataChunks));
        return false;
    }

    return true;
}

void CsDcAllocator::updateCache()
{
    for (size_t allocIter = 0; allocIter < CS_DC_ALLOCATOR_MAX; allocIter++)
    {
        m_allocators[allocIter].updateCacheThreadSafe();
    }
}

eCsDataChunkStatus CsDcAllocator::tryToAcquireDataChunkAmounts(DataChunksAmounts&       dcAmountsAvailable,
                                                               DataChunksDBs&           dcDbs,
                                                               const DataChunksAmounts& dcAmountsRequired,
                                                               bool                     isForceAcquire)
{
    const uint64_t totalRequiredDataChunkAmounts = dcAmountRequiredDma + dcAmountRequiredCommand;

    const bool isAllocRequired =
        ((dcAmountAvailableCpDma < dcAmountRequiredDma) || (dcAmountAvailableCommand < dcAmountRequiredCommand));

    // In case not enough free items are available, try to acquire new elements
    if (isAllocRequired)
    {
        if (!isForceAcquire)
        {
            return CS_DATA_CHUNKS_STATUS_NOT_COMPLETED;
        }
        else
        {
            bool operationStatus = m_allocators[CS_DC_ALLOCATOR_CP_DMA].acquireDataChunksAmountThreadSafe(
                dcDbs[CS_DC_ALLOCATOR_CP_DMA],
                m_dcSizes[CS_DC_ALLOCATOR_CP_DMA],
                dcAmountRequiredDma);
            if (!operationStatus)
            {
                dcAmountAvailableCpDma = m_allocators[CS_DC_ALLOCATOR_CP_DMA].getAmountOfAvailableElementsThreadSafe(
                    m_dcSizes[CS_DC_ALLOCATOR_CP_DMA]);
                LOG_ERR(SYN_STREAM, "{}: Failed to acquire data-chunks (CP-DMA)", HLLOG_FUNC);
                return CS_DATA_CHUNKS_STATUS_NOT_COMPLETED;
            }

            operationStatus = m_allocators[CS_DC_ALLOCATOR_COMMAND].acquireDataChunksAmountThreadSafe(
                dcDbs[CS_DC_ALLOCATOR_COMMAND],
                m_dcSizes[CS_DC_ALLOCATOR_COMMAND],
                dcAmountRequiredCommand);
            if (!operationStatus)
            {
                dcAmountAvailableCommand = m_allocators[CS_DC_ALLOCATOR_COMMAND].getAmountOfAvailableElementsThreadSafe(
                    m_dcSizes[CS_DC_ALLOCATOR_COMMAND]);
                LOG_ERR(SYN_STREAM, "{}: Failed to acquire data-chunks (CP-DMA)", HLLOG_FUNC);
                return CS_DATA_CHUNKS_STATUS_NOT_COMPLETED;
            }

            return CS_DATA_CHUNKS_STATUS_COMPLETED;
        }
    }

    if (dcAmountRequiredDma > 0)
    {
        bool operationStatus =
            m_allocators[CS_DC_ALLOCATOR_CP_DMA].acquireDataChunksAmountThreadSafe(dcDbs[CS_DC_ALLOCATOR_CP_DMA],
                                                                                   m_dcSizes[CS_DC_ALLOCATOR_CP_DMA],
                                                                                   dcAmountRequiredDma);
        if (!operationStatus)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to acquire data-chunks CP DMA", HLLOG_FUNC);
            return CS_DATA_CHUNKS_STATUS_FAILURE;
        }
    }

    if (dcAmountRequiredCommand > 0)
    {
        bool operationStatus =
            m_allocators[CS_DC_ALLOCATOR_COMMAND].acquireDataChunksAmountThreadSafe(dcDbs[CS_DC_ALLOCATOR_COMMAND],
                                                                                    m_dcSizes[CS_DC_ALLOCATOR_COMMAND],
                                                                                    dcAmountRequiredCommand);
        if (!operationStatus)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to acquire data-chunks command", HLLOG_FUNC);

            if (dcAmountRequiredDma > 0)
            {
                m_allocators[CS_DC_ALLOCATOR_CP_DMA].releaseDataChunksThreadSafe(dcDbs[CS_DC_ALLOCATOR_CP_DMA]);
            }

            return CS_DATA_CHUNKS_STATUS_FAILURE;
        }
    }

    STAT_GLBL_COLLECT(totalRequiredDataChunkAmounts, csdcStreamComputeAllocate);

    return CS_DATA_CHUNKS_STATUS_COMPLETED;
}

bool CsDcAllocator::cleanupCsDataChunk(CommandSubmissionDataChunks* pCsDataChunks,
                                       DataChunksAmounts&           dcAmountsAvailable)
{
    HB_ASSERT_PTR(pCsDataChunks);

    if (!m_allocators[CS_DC_ALLOCATOR_CP_DMA].cleanupCsDataChunkThreadSafe((uint32_t&)dcAmountAvailableCpDma,
                                                                           pCsDataChunks->getCpDmaDataChunks()))
    {
        return false;
    }

    if (!m_allocators[CS_DC_ALLOCATOR_COMMAND].cleanupCsDataChunkThreadSafe(
            (uint32_t&)dcAmountAvailableCommand,
            pCsDataChunks->getCommandsBufferDataChunks()))
    {
        return false;
    }

    return true;
}

DataChunksAmounts CsDcAllocator::getDataChunksAmounts() const
{
    return {
        m_allocators[CS_DC_ALLOCATOR_CP_DMA].getAmountOfAvailableElementsThreadSafe(m_dcSizes[CS_DC_ALLOCATOR_CP_DMA]),
        m_allocators[CS_DC_ALLOCATOR_COMMAND].getAmountOfAvailableElementsThreadSafe(
            m_dcSizes[CS_DC_ALLOCATOR_COMMAND])};
}