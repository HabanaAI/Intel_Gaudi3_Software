#include "scal_stream.hpp"

#include "log_manager.h"

// engines-arc
#include "gaudi2_arc_sched_packets.h"       // SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT

synStatus ScalStreamCopyGaudi2::memcopyImpl(ResourceStreamType               resourceType,
                                            const internalMemcopyParamEntry* memcpyParams,
                                            uint32_t                         params_count,
                                            bool                             send,
                                            uint8_t                          apiId,
                                            bool                             memsetMode,
                                            bool                             sendUnfence,
                                            uint32_t                         completionGroupIndex,
                                            MemcopySyncInfo&                 memcopySyncInfo)
{
    g2fw::sched_arc_pdma_commands_params_t pdmaCommandsParams[SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT];

    uint32_t batchSize = params_count;

    for (size_t j = 0; j < params_count; ++j)
    {
        auto& pdmaCommandsParam = pdmaCommandsParams[j];
        auto& memcpyParam       = memcpyParams[j];

        // if in memset mode, use a given value as the src
        pdmaCommandsParam.src_addr      = memcpyParam.src;
        pdmaCommandsParam.dst_addr      = memcpyParam.dst;
        pdmaCommandsParam.transfer_size = memcpyParam.size;
    }

    uint32_t payload     = 0;
    uint32_t payloadAddr = 0;

    if (sendUnfence)
    {
        payload     = memcopySyncInfo.m_workCompletionValue;
        payloadAddr = memcopySyncInfo.m_workCompletionAddress;
    }

    return ScalStreamCopySchedulerMode::addPdmaBatchTransfer(resourceType,
                                                             (const void* const)pdmaCommandsParams,
                                                             batchSize,
                                                             send,
                                                             apiId,
                                                             memsetMode,
                                                             payload,
                                                             payloadAddr,
                                                             completionGroupIndex);
}

synStatus ScalStreamComputeGaudi2::memcopyImpl(ResourceStreamType               resourceType,
                                               const internalMemcopyParamEntry* memcpyParams,
                                               uint32_t                         params_count,
                                               bool                             send,
                                               uint8_t                          apiId,
                                               bool                             memsetMode,
                                               bool                             sendUnfence,
                                               uint32_t                         completionGroupIndex,
                                               MemcopySyncInfo&                 memcopySyncInfo)
{
    g2fw::sched_arc_pdma_commands_params_t pdmaCommandsParams[SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT];

    uint32_t batchSize = params_count;

    for (size_t j = 0; j < params_count; ++j)
    {
        auto& pdmaCommandsParam = pdmaCommandsParams[j];
        auto& memcpyParam       = memcpyParams[j];

        // if in memset mode, use a given value as the src
        pdmaCommandsParam.src_addr      = memcpyParam.src;
        pdmaCommandsParam.dst_addr      = memcpyParam.dst;
        pdmaCommandsParam.transfer_size = memcpyParam.size;
    }

    uint32_t payload     = 0;
    uint32_t payloadAddr = 0;

    if (sendUnfence)
    {
        payload     = memcopySyncInfo.m_workCompletionValue;
        payloadAddr = memcopySyncInfo.m_workCompletionAddress;
    }

    return ScalStreamCopySchedulerMode::addPdmaBatchTransfer(resourceType,
                                                             (const void* const)pdmaCommandsParams,
                                                             batchSize,
                                                             send,
                                                             apiId,
                                                             memsetMode,
                                                             payload,
                                                             payloadAddr,
                                                             completionGroupIndex);
}
