#pragma once

#include "gaudi2_types.h"
#include "hal_conventions.h"
#include "graph_compiler/command_queue.h"


struct SyncObject;

namespace gaudi2
{

class CompletionQueue : public ::CommandQueue
{
public:
    CompletionQueue();
    unsigned GetLogicalQueue() const override { return DEVICE_COMPLETION_LOGICAL_QUEUE; }
};

class MmeQueue : public ::MmeQueue<gaudi2::MmeDesc>
{
public:
    MmeQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return DEVICE_MME_LOGICAL_QUEUE; }
    virtual void setDescriptorSignaling(gaudi2::MmeDesc& desc, const std::shared_ptr<SyncObject>& sync) override;
    static uint32_t getCommitRegVal();
private:
    QueueCommandPtr getExeCmd(pNode n, const DescriptorWrapper<MmeDesc>& descWrap, bool enableSignal = true) override;
    virtual void forceStaticConfig() override;
    virtual void createNullDescRegsList() override;
};

class TpcQueue : public ::TpcQueue<gaudi2::TpcDesc>
{
public:
    TpcQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return DEVICE_TPC_LOGICAL_QUEUE; }

protected:
    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(pNode n, const DescriptorWrapper<TpcDesc>& desc) override;
    QueueCommandPtr getExeCmd(pNode n, const DescriptorWrapper<TpcDesc>& descWrap, bool enableSignal = true) override;
    virtual void createNullDescRegsList() override;
    virtual void validateAllAddressPatchpointsDropped(const NodePtr& n) const override;
};

class RotatorQueue : public ::RotatorQueue<gaudi2::RotatorDesc>
{
public:
    RotatorQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return DEVICE_ROT_LOGICAL_QUEUE; }
protected:
    virtual QueueCommandPtr getExeCmd(NodePtr n, const DescriptorWrapper<RotatorDesc>& descWrap, bool enableSignal = true) override;
    virtual void setDescriptorSignaling(gaudi2::RotatorDesc& desc, const std::shared_ptr<SyncObject>& sync) override;
    virtual void forceStaticConfig() override;
    virtual void createNullDescRegsList() override;
    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(NodePtr n, const DescriptorWrapper<RotatorDesc>& descWrapper) override;
};

class DmaDescQueue : public ::DmaDescQueue<gaudi2::DmaDesc>
{
public:
    DmaDescQueue(unsigned logicalQueue, unsigned engineId, unsigned engineIndex, bool sendSyncEvents);
    unsigned GetLogicalQueue() const override { return m_logicalQueue; }

    static gaudi2::dma_core_ctx::reg_commit getCommitRegVal(const DMANode& node, const DmaDesc& desc, bool enableSignal);
    static uint32_t                         calcDescriptorSignaling();

protected:
    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(NodePtr n, const DescriptorWrapper<DmaDesc>& descWrapper) override;
    virtual QueueCommandPtr getExeCmd(NodePtr n, const DescriptorWrapper<DmaDesc>& descWrap, bool enableSignal = true) override;

    virtual void setDescriptorSignaling(gaudi2::DmaDesc& desc, const std::shared_ptr<SyncObject>& sync) override;
    virtual void forceStaticConfig() override;
    virtual void createNullDescRegsList() override;
    virtual void updateQueueStateAfterPush(NodePtr n) override;

private:
    unsigned    m_logicalQueue;
    DMA_OP_TYPE m_lastNodeOpType;
};

} // namespace gaudi2
