#pragma once

#include "define_synapse_common.hpp"
#include "synapse_common_types.h"

#include <cstdint>
#include <deque>
#include <string_view>

class QmanDefinitionInterface
{
public:
    virtual ~QmanDefinitionInterface() = default;

    virtual uint64_t getComputeStreamsMasterQmanBaseAddr() const = 0;

    // work completion
    virtual uint64_t getWorkCompletionQueueId() const = 0;

    virtual bool isWorkCompletionQueueId(uint64_t id) const = 0;

    // Description: Will return the first stream on the relavant QMAN
    // Future:      Will return the first stream on a QMAN for the given relevant to operation

    // Compute
    virtual uint64_t getStreamMasterQueueIdForCompute() const = 0;

    virtual bool isStreamMasterQueueIdForCompute(uint64_t id) const = 0;

    virtual uint64_t getArbitratorMasterQueueIdForCompute() const = 0;

    // MemcopyToDevice  -  User DMA Down
    virtual uint64_t getStreamsMasterQueueIdForMemcopyToDevice() const = 0;

    virtual uint64_t getArbitratorMasterQueueIdForMemcopyToDevice() const = 0;

    // SynapseMemcopyToDevice  -  Synapse DMA Down
    virtual uint64_t getStreamsMasterQueueIdForSynapseMemcopyToDevice() const = 0;

    // MemcopyFromDevice  -  DMA Up
    virtual uint64_t getStreamsMasterQueueIdForMemcopyFromDevice() const = 0;

    // Collective
    virtual uint64_t getStreamsMasterQueueIdForCollective() const = 0;

    virtual uint64_t getArbitratorMasterQueueIdForCollective() const = 0;

    virtual uint64_t getCollectiveReductionEngineId() const = 0;

    virtual bool isArbMasterForComputeAndNewGaudiSyncScheme(uint64_t id) const = 0;

    // To be used by RT
    virtual uint64_t getComputeInferenceStreamMasterQueueId() const = 0;

    // To be used by GC
    virtual uint64_t getComputeInferenceArbitrationMasterQueueId() const = 0;

    virtual uint64_t getComputeInferenceMasterQmanBaseAddr() const = 0;

    virtual bool isNonInternalCommandsDcQueueId(uint64_t id) const = 0;

    virtual std::string_view getQmanIdName(uint32_t id) const = 0;

    virtual uint64_t getArbitrationMaster(internalStreamType streamId) const = 0;

    virtual const uint32_t* getArbitrationSlaves(internalStreamType streamId) const = 0;

    virtual uint32_t getEndStreamArrayIndicator() const = 0;

    // Used for disabling all of the engines, prior of configuring them
    virtual const std::deque<uint32_t>* getEnginesWithArbitrator() const = 0;

    virtual uint32_t getAcquireDeviceDefaultQman() const = 0;

    virtual uint32_t getFirstTpcEngineId() const = 0;

    virtual uint32_t getFirstNicEngineId() const = 0;

    virtual bool isTpcEngineId(uint32_t engineId, uint32_t& engineIndex, bool& isDisabled) const = 0;

    virtual bool isNicEngineId(uint32_t engineId, uint32_t& engineIndex) const = 0;

    virtual bool isRotatorEngineId(uint32_t engineId, uint32_t& engineIndex) const = 0;

    virtual bool isEdmaEngineId(uint32_t engineId, uint32_t& engineIndex) const = 0;

    // Only Gaudi-1 has External QMANs
    virtual bool isExternalQueueId(uint64_t id) const { return false; };

    // Implement upon need
    virtual bool isComputeArbSlaveQueueId(uint64_t id) const { return false; };
};

QmanDefinitionInterface* getQmansDefinition(synDeviceType deviceType);