#pragma once

#include "drm/habanalabs_accel.h"
#include "runtime/qman/common/master_qmans_definition_interface.hpp"

#include <cstdint>
#include <deque>
#include <string_view>

namespace gaudi
{
class QmansDefinition : public QmanDefinitionInterface
{
public:
    virtual ~QmansDefinition() = default;

    uint64_t getComputeStreamsMasterQmanBaseAddr() const override;

    // work completion
    virtual uint64_t getWorkCompletionQueueId() const override;

    virtual bool isWorkCompletionQueueId(uint64_t id) const override;

    // Description: Will return the first stream on the relavant QMAN
    // Future:      Will return the first stream on a QMAN for the given relevant to operation

    // Compute
    virtual uint64_t getStreamMasterQueueIdForCompute() const override;

    virtual bool isStreamMasterQueueIdForCompute(uint64_t id) const override;

    virtual uint64_t getArbitratorMasterQueueIdForCompute() const override;

    // MemcopyToDevice  -  User DMA Down
    virtual uint64_t getStreamsMasterQueueIdForMemcopyToDevice() const override;

    virtual uint64_t getArbitratorMasterQueueIdForMemcopyToDevice() const override;

    // SynapseMemcopyToDevice  -  Synapse DMA Down
    virtual uint64_t getStreamsMasterQueueIdForSynapseMemcopyToDevice() const override;

    // MemcopyFromDevice  -  DMA Up
    virtual uint64_t getStreamsMasterQueueIdForMemcopyFromDevice() const override;

    // Collective
    virtual uint64_t getStreamsMasterQueueIdForCollective() const override;

    virtual uint64_t getArbitratorMasterQueueIdForCollective() const override;

    virtual uint64_t getCollectiveReductionEngineId() const override;

    virtual bool isArbMasterForComputeAndNewGaudiSyncScheme(uint64_t id) const override;

    // To be used by RT
    virtual uint64_t getComputeInferenceStreamMasterQueueId() const override;

    // To be used by GC
    virtual uint64_t getComputeInferenceArbitrationMasterQueueId() const override;

    virtual uint64_t getComputeInferenceMasterQmanBaseAddr() const override;

    virtual bool isNonInternalCommandsDcQueueId(uint64_t id) const override;

    virtual uint32_t getAcquireDeviceDefaultQman() const override;

    virtual std::string_view getQmanIdName(uint32_t id) const override;

    virtual uint64_t getArbitrationMaster(internalStreamType streamId) const override;

    virtual const uint32_t* getArbitrationSlaves(internalStreamType streamId) const override;

    virtual uint32_t getEndStreamArrayIndicator() const override;

    virtual const std::deque<uint32_t>* getEnginesWithArbitrator() const override;

    virtual uint32_t getFirstTpcEngineId() const override;

    virtual uint32_t getFirstNicEngineId() const override;

    virtual bool isTpcEngineId(uint32_t engineId, uint32_t& engineIndex, bool& isDisabled) const override;

    virtual bool isNicEngineId(uint32_t engineId, uint32_t& engineIndex) const override;

    virtual bool isRotatorEngineId(uint32_t engineId, uint32_t& engineIndex) const override;

    virtual bool isEdmaEngineId(uint32_t engineId, uint32_t& engineIndex) const override;

    virtual bool isExternalQueueId(uint64_t id) const override;

    virtual bool isComputeArbSlaveQueueId(uint64_t id) const override;

    static QmanDefinitionInterface* getInstance();

private:
    QmansDefinition() = default;

    static QmansDefinition* m_pInstance;
};
}  // namespace gaudi
