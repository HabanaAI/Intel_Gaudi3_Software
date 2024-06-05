#pragma once

#include "gaudi3_types.h"
#include "hal_conventions.h"
#include "graph_compiler/command_queue.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include <utility>

struct SyncObject;
class ResetSobs;

namespace gaudi3
{
class MmeQueue : public ::MmeQueue<gaudi3::MmeDesc>
{
public:
    MmeQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return DEVICE_MME_LOGICAL_QUEUE; }
    static uint32_t      getCommitRegVal(bool isTranspose);
    static mme_op_type_t getOpType(bool isTranspose);

protected:
    QueueCommandPtr getExeCmd(pNode n, const DescriptorWrapper<MmeDesc>& descWrap, bool enableSignal = true) override;
    virtual void forceStaticConfig() override;
    virtual DescriptorShadow& getDescShadow(const NodePtr& n) override;
    virtual void pushAdditionalDynamicCmds4sobReset(const pNode& node, unsigned pipeLevel) override;
    virtual void pushAdditionalDynamicCmds4mcidRollover(const pNode& node, unsigned pipeLevel) override;

    DescriptorShadow m_descriptorShadowGemm; // for gemm descriptor
    DescriptorShadow m_descriptorShadowDma;  // for dma (transpose pipe) descriptor

    ResetSobs*    m_prevResetCmd    = nullptr;
    unsigned      m_prevResetId     = 0;
    McidRollover* m_prevRolloverCmd = nullptr;
    unsigned      m_prevRolloverId  = 0;
};

class TpcQueue : public ::TpcQueue<gaudi3::TpcDesc>
{
public:
    TpcQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return DEVICE_TPC_LOGICAL_QUEUE; }

protected:
    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(pNode n, const DescriptorWrapper<TpcDesc>& desc) override;
    QueueCommandPtr getExeCmd(pNode n, const DescriptorWrapper<TpcDesc>& descWrap, bool enableSignal = true) override;
};

class RotatorQueue : public ::RotatorQueue<gaudi3::RotatorDesc>
{
public:
    RotatorQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return DEVICE_ROT_LOGICAL_QUEUE; }

protected:
    virtual QueueCommandPtr getExeCmd(NodePtr n, const DescriptorWrapper<RotatorDesc>& descWrap, bool enableSignal = true) override;
    virtual void setDescriptorSignaling(gaudi3::RotatorDesc& desc, const std::shared_ptr<SyncObject>& sync) override;
    virtual void forceStaticConfig() override;
    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(NodePtr n, const DescriptorWrapper<RotatorDesc>& descWrapper) override;
};

} // namespace gaudi3
