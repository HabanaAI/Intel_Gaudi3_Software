#pragma once

#include "graph_compiler/eng_arc_hooks.h"
#include "hal_reader/gaudi3/hal_reader.h"

namespace gaudi3
{
class EngArcHooks : public ::EngArcHooks
{
public:
    static EngArcHooks& instance();

    virtual ~EngArcHooks();

    virtual std::shared_ptr<StaticCpDmaEngArcCommand> getStaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                                                                  EngArcBufferAddrBase srcAddrBaseId,
                                                                                  uint64_t             dataSize,
                                                                                  bool                 yield,
                                                                                  unsigned engId) const override;

    virtual std::shared_ptr<NopEngArcCommand>
    getNopEngArcCommand(bool switchCQ = false, bool yield = false, unsigned padding = 0) const override;

    virtual std::shared_ptr<DynamicWorkDistEngArcCommand>
    getDynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion = 1) const override;

    virtual std::shared_ptr<ScheduleDmaEngArcCommand>
    getScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                EngArcBufferAddrBase srcAddrBaseId,
                                uint64_t             dstOffset,
                                uint64_t             dataSize,
                                bool                 isDmaWithDcoreLocality,
                                bool                 yield) const override;

    virtual std::shared_ptr<ListSizeEngArcCommand> getListSizeEngArcCommand() const override;

    virtual std::shared_ptr<SignalOutEngArcCommand>
    getSignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield) const override;

    virtual std::shared_ptr<ResetSobsArcCommand> getResetSobsArcCommand(unsigned target,
                                                                        unsigned targetXps,
                                                                        unsigned totalNumEngs,
                                                                        bool     switchBit,
                                                                        bool     yield) const override;

    virtual std::shared_ptr<EngArcCommand> getMcidRolloverArcCommand(unsigned target,
                                                                     unsigned targetXps,
                                                                     bool     switchBit,
                                                                     bool     yield) const override;
    virtual unsigned getTpcWdCtxtSize() const override;

    virtual unsigned getMmeWdCtxtSize() const override;

    virtual unsigned getEdmaWdCtxtSize() const override;

    virtual unsigned getRotWdCtxtSize() const override;

    virtual unsigned getWdCtxtCount() const override;

    virtual unsigned getCpuIdAll() const override;

    virtual unsigned getStaticEcbChunkSize() const override;

    virtual unsigned getDynamicEcbChunkSize() const override;

    virtual bool
    getQueueIdInfo(HabanaDeviceType& type, unsigned& deviceID, unsigned& streamID, unsigned queueId) const override;

    virtual unsigned engIdx2cpuId(unsigned idx, HabanaDeviceType type) const override;

private:
    EngArcHooks();
};

//==============================================================
//  CME Hooks hereafter
//==============================================================

class EngArcHooksForCme : public ::EngArcHooks
{
public:
    static EngArcHooksForCme& instance();

    virtual ~EngArcHooksForCme();

    virtual std::shared_ptr<StaticCpDmaEngArcCommand> getStaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                                                                  EngArcBufferAddrBase srcAddrBaseId,
                                                                                  uint64_t             dataSize,
                                                                                  bool                 yield,
                                                                                  unsigned engId) const override;

    virtual std::shared_ptr<NopEngArcCommand>
    getNopEngArcCommand(bool switchCQ = false, bool yield = false, unsigned padding = 0) const override;

    virtual std::shared_ptr<DynamicWorkDistEngArcCommand>
    getDynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion = 1) const override;

    virtual std::shared_ptr<ScheduleDmaEngArcCommand>
    getScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                EngArcBufferAddrBase srcAddrBaseId,
                                uint64_t             dstOffset,
                                uint64_t             dataSize,
                                bool                 isDmaWithDcoreLocality,
                                bool                 yield) const override;

    virtual std::shared_ptr<ListSizeEngArcCommand> getListSizeEngArcCommand() const override;

    virtual std::shared_ptr<SignalOutEngArcCommand>
    getSignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield) const override;

    virtual std::shared_ptr<ResetSobsArcCommand> getResetSobsArcCommand(unsigned target,
                                                                        unsigned targetXps,
                                                                        unsigned totalNumEngs,
                                                                        bool     switchBit,
                                                                        bool     yield) const override;

    virtual std::shared_ptr<EngArcCommand> getMcidRolloverArcCommand(unsigned target,
                                                                     unsigned targetXps,
                                                                     bool     switchBit,
                                                                     bool     yield) const override;
    virtual unsigned getTpcWdCtxtSize() const override;

    virtual unsigned getMmeWdCtxtSize() const override;

    virtual unsigned getEdmaWdCtxtSize() const override;

    virtual unsigned getRotWdCtxtSize() const override;

    virtual unsigned getWdCtxtCount() const override;

    virtual unsigned getCpuIdAll() const override;

    virtual unsigned getStaticEcbChunkSize() const override;

    virtual unsigned getDynamicEcbChunkSize() const override;

    virtual bool
    getQueueIdInfo(HabanaDeviceType& type, unsigned& deviceID, unsigned& streamID, unsigned queueId) const override;

    virtual unsigned engIdx2cpuId(unsigned idx, HabanaDeviceType type) const override;

private:
    EngArcHooksForCme();
};

}  // namespace gaudi3
