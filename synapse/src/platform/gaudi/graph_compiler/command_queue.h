#pragma once

#include "gaudi_types.h"
#include "graph_compiler/command_queue.h"


struct SyncObject;

namespace gaudi
{

class CompletionQueue : public ::CommandQueue
{
public:
    CompletionQueue();
    unsigned GetLogicalQueue() const override {return DEVICE_COMPLETION_LOGICAL_QUEUE;}
};

class MmeQueue : public ::MmeQueue<gaudi::MmeDesc>
{
public:
    MmeQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents);

    unsigned GetLogicalQueue() const override {return DEVICE_MME_LOGICAL_QUEUE;}
    virtual void setDescriptorSignaling(gaudi::MmeDesc& desc, const std::shared_ptr<SyncObject>& sync) override;

protected:
    virtual void validateAllAddressPatchpointsDropped(const NodePtr& n) const override {} // no validation for gaudi1

private:
    virtual bool isSignalingFromQman() override { return true; }

};

class TpcQueue : public ::TpcQueue<gaudi::TpcDesc>
{
public:
    TpcQueue(unsigned engineId, unsigned engineIndex, bool sendSyncEvents, bool graphHasDynamicity);

    unsigned                         GetLogicalQueue() const override { return DEVICE_TPC_LOGICAL_QUEUE; }
    virtual std::vector<DescSection> getPredicatedSections(pNode n, const gaudi::TpcDesc& desc) const override;
    virtual std::vector<DescSection> getUnpredicatedSections(pNode n, const gaudi::TpcDesc& desc) const override;

protected:
    bool predicateSectionEnabled(pNode n) const;
    bool separateSizeSectionEnabled(pNode n) const;
    void setDescriptorSignaling(gaudi::TpcDesc& desc, const std::shared_ptr<SyncObject>& sync) override;
    virtual void validateAllAddressPatchpointsDropped(const NodePtr& n) const override {} // no validation for gaudi1

private:
    bool m_graphHasDynamicity;

    virtual bool isSignalingFromQman() override { return true; }
};

class DmaDescQueue : public ::DmaDescQueue<gaudi::DmaDesc>
{
public:
    DmaDescQueue(unsigned logicalQueue, unsigned engineId, unsigned engineIndex, bool sendSyncEvents);

    unsigned GetLogicalQueue() const override {return m_logicalQueue;}

protected:
    virtual DescriptorShadow::AllRegistersProperties registersPropertiesForDesc(pNode n, const DescriptorWrapper<DmaDesc>& desc) override;
    virtual QueueCommandPtr getExeCmd(pNode n, const DescriptorWrapper<DmaDesc>& descWrap, bool enableSignal = true) override;
    virtual void            updateQueueStateAfterPush(pNode n) override;
    virtual bool            allowNoDescUpdates(pNode n) override;
    virtual bool            canUseLinDmaPacket(pNode n);
    virtual void            updateLinDMALoadedDescSections(const DmaDesc& desc);
    virtual void            setDescriptorSignaling(gaudi::DmaDesc& desc, const std::shared_ptr<SyncObject>& sync) override;
    virtual void            validateAllAddressPatchpointsDropped(const NodePtr& n) const override {} // no validation for gaudi1

private:
    QueueCommandPtr getLinDmaCmd(pNode n, const DmaDesc& desc, bool enableSignal);

    unsigned    m_logicalQueue;
    DMA_OP_TYPE m_lastNodeOpType;

    virtual bool isSignalingFromQman() override { return true; }
};

} // namespace gaudi
