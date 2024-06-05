#pragma once

#include "gaudi3/gaudi3_arc_eng_packets.h"
#include "gaudi3/gaudi3_arc_common_packets.h"
#include "eng_arc_command.h"
#include "cache_types.h"

class Gaudi3StaticCpDmaEngArcCommand : public StaticCpDmaEngArcCommand
{
public:
    Gaudi3StaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                   EngArcBufferAddrBase srcAddrBaseId,
                                   uint64_t             dataSize,
                                   bool                 yield,
                                   unsigned             engId);
    virtual ~Gaudi3StaticCpDmaEngArcCommand() = default;
    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setEngId(unsigned engId);
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_static_desc_v2_t m_binary;
};

class Gaudi3NopEngArcCommand : public NopEngArcCommand
{
public:
    Gaudi3NopEngArcCommand(bool switchCQ = false, bool yield = false, unsigned padding = 0);
    virtual ~Gaudi3NopEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setPadding(unsigned padding) { m_binary.padding = padding; }
    void             setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_nop_t m_binary;
};

class Gaudi3DynamicWorkDistEngArcCommand : public DynamicWorkDistEngArcCommand
{
public:
    Gaudi3DynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion = 1);
    virtual ~Gaudi3DynamicWorkDistEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_wd_fence_and_exec_t m_binary;
};

class Gaudi3ScheduleDmaEngArcCommand : public ScheduleDmaEngArcCommand
{
public:
    Gaudi3ScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                   EngArcBufferAddrBase srcAddrBaseId,
                                   uint64_t             dstOffset,
                                   uint64_t             dataSize,
                                   bool                 isDmaWithDcoreLocality,
                                   bool                 yield);
    virtual ~Gaudi3ScheduleDmaEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_sched_dma_t m_binary;
};

class Gaudi3ListSizeEngArcCommand : public ListSizeEngArcCommand
{
public:
    Gaudi3ListSizeEngArcCommand();
    virtual ~Gaudi3ListSizeEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setListSize(unsigned listSize);
    void             setTopologyStart();
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_list_size_t m_binary;
};

class Gaudi3SignalOutEngArcCommand : public SignalOutEngArcCommand
{
public:
    Gaudi3SignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield);
    virtual ~Gaudi3SignalOutEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }
    virtual void     setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }

protected:
    eng_arc_cmd_sfg_t m_binary;
};

class Gaudi3ResetSobsArcCommand : public ResetSobsArcCommand
{
public:
    Gaudi3ResetSobsArcCommand(unsigned target, unsigned targetXps, unsigned totalNumEngs, bool switchBit, bool yield);
    virtual ~Gaudi3ResetSobsArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }
    virtual void     setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }

protected:
    eng_arc_cmd_reset_soset_t m_binary;
};

class Gaudi3McidRolloverArcCommand : public EngArcCommand
{
public:
    Gaudi3McidRolloverArcCommand(unsigned target, unsigned targetXps, bool switchBit, bool yield);
    virtual ~Gaudi3McidRolloverArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }
    virtual void     setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }

protected:
    eng_arc_cmd_mcid_rollover_t m_binary;
};

//==============================================================
//  CME Commands hereafter
//==============================================================

class Gaudi3CmeNopEngArcCommand : public NopEngArcCommand
{
public:
    Gaudi3CmeNopEngArcCommand(unsigned padding = 0);
    virtual ~Gaudi3CmeNopEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setPadding(unsigned padding) { m_binary.padding = padding; }
    void             setSwitchCQ(bool switchCQ) override { /* not supported in CME*/ }
    virtual void     setYield(bool y) override { /* not supported in CME */ }

protected:
    cme_arc_cmd_nop_t m_binary;
};

class Gaudi3CmeDegradeArcCommand : public EngArcCommand
{
public:
    Gaudi3CmeDegradeArcCommand(const DependencyMap& deps, PhysicalMcid mcid, bool useDiscardBase = false);
    virtual ~Gaudi3CmeDegradeArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setSwitchCQ(bool switchCQ) override { /* not supported in CME*/ }
    virtual void     setYield(bool y) override { /* not supported in CME */ }

protected:
    cme_arc_cmd_degrade_cls_t m_binary;
};

class Gaudi3CmeDiscardArcCommand : public EngArcCommand
{
public:
    Gaudi3CmeDiscardArcCommand(const DependencyMap& deps, PhysicalMcid mcid);
    virtual ~Gaudi3CmeDiscardArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setSwitchCQ(bool switchCQ) override { /* not supported in CME*/ }
    virtual void     setYield(bool y) override { /* not supported in CME */ }

protected:
    cme_arc_cmd_discard_cls_t m_binary;
};

class Gaudi3CmeMcidRolloverArcCommand : public EngArcCommand
{
public:
    Gaudi3CmeMcidRolloverArcCommand(bool incMme = false, bool incRot = false);
    virtual ~Gaudi3CmeMcidRolloverArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setSwitchCQ(bool switchCQ) override { /* not supported in CME*/ }
    virtual void     setYield(bool y) override { /* not supported in CME */ }

protected:
    cme_arc_cmd_mcid_rollover_t m_binary;
};

class Gaudi3CmeResetSobsArcCommand : public EngArcCommand
{
public:
    Gaudi3CmeResetSobsArcCommand(unsigned totalNumEngines);
    virtual ~Gaudi3CmeResetSobsArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setSwitchCQ(bool switchCQ) override { /* not supported in CME*/ }
    virtual void     setYield(bool y) override { /* not supported in CME */ }

protected:
    cme_arc_cmd_reset_soset_t m_binary;
};
