#pragma once

#include "gaudi2_arc_eng_packets.h"
#include "gaudi2_arc_common_packets.h"
#include "eng_arc_command.h"

class Gaudi2StaticCpDmaEngArcCommand : public StaticCpDmaEngArcCommand
{
public:
    Gaudi2StaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                   EngArcBufferAddrBase srcAddrBaseId,
                                   uint64_t             dataSize,
                                   bool                 yield,
                                   unsigned             engId);
    virtual ~Gaudi2StaticCpDmaEngArcCommand() = default;
    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setEngId(unsigned engId);
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_static_desc_v2_t m_binary;
};

class Gaudi2NopEngArcCommand : public NopEngArcCommand
{
public:
    Gaudi2NopEngArcCommand(bool switchCQ = false, bool yield = false, unsigned padding = 0);
    virtual ~Gaudi2NopEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setPadding(unsigned padding) { m_binary.padding = padding; }
    virtual void     setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_nop_t m_binary;
};

class Gaudi2DynamicWorkDistEngArcCommand : public DynamicWorkDistEngArcCommand
{
public:
    Gaudi2DynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion = 1);
    virtual ~Gaudi2DynamicWorkDistEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_wd_fence_and_exec_t m_binary;
};

class Gaudi2ScheduleDmaEngArcCommand : public ScheduleDmaEngArcCommand
{
public:
    Gaudi2ScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                   EngArcBufferAddrBase srcAddrBaseId,
                                   uint64_t             dstOffset,
                                   uint64_t             dataSize,
                                   bool                 yield);
    virtual ~Gaudi2ScheduleDmaEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_sched_dma_t m_binary;
};

class Gaudi2ListSizeEngArcCommand : public ListSizeEngArcCommand
{
public:
    Gaudi2ListSizeEngArcCommand();
    virtual ~Gaudi2ListSizeEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    void             setListSize(unsigned listSize);
    void             setTopologyStart();
    virtual void     setYield(bool y) override { m_binary.yield = y; }

protected:
    eng_arc_cmd_list_size_t m_binary;
};

class Gaudi2SignalOutEngArcCommand : public SignalOutEngArcCommand
{
public:
    Gaudi2SignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield);
    virtual ~Gaudi2SignalOutEngArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }
    virtual void     setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }

protected:
    eng_arc_cmd_sfg_t m_binary;
};

class Gaudi2ResetSobsArcCommand : public ResetSobsArcCommand
{
public:
    Gaudi2ResetSobsArcCommand(unsigned target, unsigned totalNumEngs, bool switchBit, bool yield);
    virtual ~Gaudi2ResetSobsArcCommand() = default;

    virtual void     print() const override;
    virtual unsigned sizeInBytes() const override;
    virtual uint64_t serialize(void* dst) const override;
    virtual void     setYield(bool y) override { m_binary.yield = y; }
    virtual void     setSwitchCQ(bool switchCQ) override { m_binary.switch_cq = switchCQ; }

protected:
    eng_arc_cmd_reset_soset_t m_binary;
};
