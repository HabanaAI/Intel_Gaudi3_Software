#include "master_qmans_definition.hpp"
#include "habana_global_conf.h"
#include "defs.h"
#include "gaudi/asic_reg/gaudi_blocks.h"
#include "platform/gaudi/graph_compiler/hal_conventions.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <string_view>

#define USE_TPC_FOR_COMPUTE_ARB_MASTER

using namespace gaudi;

std::deque<uint32_t> gaudiEnginesWithArbitrator = {
    GAUDI_ENGINE_ID_DMA_0,
    GAUDI_ENGINE_ID_DMA_1,
    GAUDI_ENGINE_ID_DMA_2,
    GAUDI_ENGINE_ID_DMA_3,
    GAUDI_ENGINE_ID_DMA_4,
    GAUDI_ENGINE_ID_DMA_5,
    GAUDI_ENGINE_ID_DMA_6,
    GAUDI_ENGINE_ID_DMA_7,
    GAUDI_ENGINE_ID_MME_0,
    // GAUDI_ENGINE_ID_MME_1,
    GAUDI_ENGINE_ID_MME_2,
    // GAUDI_ENGINE_ID_MME_3,
    GAUDI_ENGINE_ID_TPC_0,
    GAUDI_ENGINE_ID_TPC_1,
    GAUDI_ENGINE_ID_TPC_2,
    GAUDI_ENGINE_ID_TPC_3,
    GAUDI_ENGINE_ID_TPC_4,
    GAUDI_ENGINE_ID_TPC_5,
    GAUDI_ENGINE_ID_TPC_6,
    GAUDI_ENGINE_ID_TPC_7,
    GAUDI_ENGINE_ID_NIC_0,
    GAUDI_ENGINE_ID_NIC_1,
    GAUDI_ENGINE_ID_NIC_2,
    GAUDI_ENGINE_ID_NIC_3,
    GAUDI_ENGINE_ID_NIC_4,
    GAUDI_ENGINE_ID_NIC_5,
    GAUDI_ENGINE_ID_NIC_6,
    GAUDI_ENGINE_ID_NIC_7,
    GAUDI_ENGINE_ID_NIC_8,
    GAUDI_ENGINE_ID_NIC_9,
};

static const gaudi_queue_id computeSlaves[] = {GAUDI_QUEUE_ID_DMA_2_0,
                                               GAUDI_QUEUE_ID_DMA_3_0,
                                               GAUDI_QUEUE_ID_DMA_4_0,
                                               GAUDI_QUEUE_ID_DMA_6_0,
                                               GAUDI_QUEUE_ID_DMA_7_0,
                                               GAUDI_QUEUE_ID_MME_0_0,
                                               GAUDI_QUEUE_ID_MME_1_0,
                                               GAUDI_QUEUE_ID_TPC_0_0,
                                               GAUDI_QUEUE_ID_TPC_1_0,
                                               GAUDI_QUEUE_ID_TPC_2_0,
                                               GAUDI_QUEUE_ID_TPC_3_0,
                                               GAUDI_QUEUE_ID_TPC_4_0,
                                               GAUDI_QUEUE_ID_TPC_5_0,
                                               GAUDI_QUEUE_ID_TPC_6_0,
                                               GAUDI_QUEUE_ID_TPC_7_0};

const uint64_t     GAUDI_QUEUE_ID_FAKE_FOR_WORK_COMPLETION = 999;
constexpr uint64_t GAUDI_COMPLETION_QUEUE_ID               = GAUDI_QUEUE_ID_FAKE_FOR_WORK_COMPLETION;

QmansDefinition* QmansDefinition::m_pInstance = nullptr;

uint64_t QmansDefinition::getComputeStreamsMasterQmanBaseAddr() const
{
#if defined(USE_TPC_FOR_COMPUTE_ARB_MASTER)
    return mmTPC0_QM_BASE;
#else
    return mmMME2_QM_BASE;
#endif
}

bool QmansDefinition::isStreamMasterQueueIdForCompute(uint64_t id) const
{
    uint64_t streamMasterId;

    streamMasterId = getStreamMasterQueueIdForCompute();

    return (id == streamMasterId);
}

uint64_t QmansDefinition::getArbitratorMasterQueueIdForCompute() const
{
#if defined(USE_TPC_FOR_COMPUTE_ARB_MASTER)
    return GAUDI_QUEUE_ID_TPC_0_0;
#else
    return GAUDI_QUEUE_ID_MME_0_0;
#endif
}

uint64_t QmansDefinition::getStreamsMasterQueueIdForMemcopyToDevice() const
{
    return GAUDI_QUEUE_ID_DMA_0_0;
}

uint64_t QmansDefinition::getArbitratorMasterQueueIdForMemcopyToDevice() const
{
    return GAUDI_QUEUE_ID_DMA_0_0;
}

// SynapseMemcopyToDevice  -  Synapse DMA Down
uint64_t QmansDefinition::getStreamsMasterQueueIdForSynapseMemcopyToDevice() const
{
    return GAUDI_QUEUE_ID_DMA_0_1;
}

uint64_t QmansDefinition::getWorkCompletionQueueId() const
{
    // This function defines what logical queue GC is using to send
    // the completion job, and RT is responsible to move it to the getArbitratorMasterQueueIdForCompute
    //   (In static_information_processor.cpp  _extractJobsInformation() )
    // It should use a queue that is otherwised unused (e.g. no actual jobs are sent to be executed on this queue)
    return GAUDI_COMPLETION_QUEUE_ID;
}

bool QmansDefinition::isWorkCompletionQueueId(uint64_t id) const
{
    if (id == getWorkCompletionQueueId())
    {
        return true;
    }
    return false;
}

uint64_t QmansDefinition::getStreamsMasterQueueIdForMemcopyFromDevice() const
{
    return GAUDI_QUEUE_ID_DMA_1_0;
}

uint64_t QmansDefinition::getStreamsMasterQueueIdForCollective() const
{
    return GAUDI_QUEUE_ID_DMA_1_0;
}

uint64_t QmansDefinition::getArbitratorMasterQueueIdForCollective() const
{
    return GAUDI_QUEUE_ID_DMA_5_0;
}

bool QmansDefinition::isExternalQueueId(uint64_t id) const
{
    if ((id < GAUDI_QUEUE_ID_DMA_1_3) || ((id >= GAUDI_QUEUE_ID_DMA_5_0) && (id <= GAUDI_QUEUE_ID_DMA_5_3)))
        return true;
    return false;
}

bool QmansDefinition::isComputeArbSlaveQueueId(uint64_t id) const
{
    return std::find(std::begin(computeSlaves), std::end(computeSlaves), id) != std::end(computeSlaves);
}

uint64_t QmansDefinition::getCollectiveReductionEngineId() const
{
    return GAUDI_ENGINE_ID_DMA_5;
}

uint64_t QmansDefinition::getStreamMasterQueueIdForCompute() const
{
    return GAUDI_QUEUE_ID_DMA_0_2;
}

bool QmansDefinition::isArbMasterForComputeAndNewGaudiSyncScheme(uint64_t id) const
{
    if (id == getArbitratorMasterQueueIdForCompute())
    {
        return true;
    }
    return false;
};

uint64_t QmansDefinition::getComputeInferenceStreamMasterQueueId() const
{
    HB_ASSERT(false, "device does not support this method");
    return 0;
}

uint64_t QmansDefinition::getComputeInferenceArbitrationMasterQueueId() const
{
    HB_ASSERT(false, "device does not support this method");
    return 0;
}

uint64_t QmansDefinition::getComputeInferenceMasterQmanBaseAddr() const
{
    HB_ASSERT(false, "device does not support this method");
    return 0;
}

bool QmansDefinition::isNonInternalCommandsDcQueueId(uint64_t id) const
{
    return (isExternalQueueId(id) || isWorkCompletionQueueId(id));
}

QmanDefinitionInterface* QmansDefinition::getInstance()
{
    if (m_pInstance == nullptr)
    {
        m_pInstance = new QmansDefinition();
    }

    return m_pInstance;
}

uint32_t QmansDefinition::getAcquireDeviceDefaultQman() const
{
    return GAUDI_QUEUE_ID_DMA_0_0;
}

std::string_view QmansDefinition::getQmanIdName(uint32_t id) const
{
    return gaudi::getQmanIdName(id);
}

uint64_t QmansDefinition::getArbitrationMaster(internalStreamType streamId) const
{
    HB_ASSERT(false, "Not supported");
    return GAUDI_QUEUE_ID_SIZE;
}

const uint32_t* QmansDefinition::getArbitrationSlaves(internalStreamType streamId) const
{
    HB_ASSERT(false, "Not supported");
    return nullptr;
}

uint32_t QmansDefinition::getEndStreamArrayIndicator() const
{
    HB_ASSERT(false, "Not supported");
    return 0;
}

const std::deque<uint32_t>* QmansDefinition::getEnginesWithArbitrator() const
{
    return &gaudiEnginesWithArbitrator;
}

uint32_t QmansDefinition::getFirstTpcEngineId() const
{
    return GAUDI_ENGINE_ID_TPC_0;
}

uint32_t QmansDefinition::getFirstNicEngineId() const
{
    return GAUDI_ENGINE_ID_NIC_0;
}

bool QmansDefinition::isTpcEngineId(uint32_t engineId, uint32_t& engineIndex, bool& isDisabled) const
{
    HB_ASSERT(false, "Not supported");
    return false;
}

bool QmansDefinition::isNicEngineId(uint32_t engineId, uint32_t& engineIndex) const
{
    HB_ASSERT(false, "Not supported");
    return false;
}

bool QmansDefinition::isRotatorEngineId(uint32_t engineId, uint32_t& engineIndex) const
{
    HB_ASSERT(false, "Not supported");
    return false;
}

bool QmansDefinition::isEdmaEngineId(uint32_t engineId, uint32_t& engineIndex) const
{
    HB_ASSERT(false, "Not supported");
    return false;
}
