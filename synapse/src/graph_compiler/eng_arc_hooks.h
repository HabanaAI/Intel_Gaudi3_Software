#pragma once

#include "types.h"
#include "utils.h"
#include "habana_device_types.h"
#include "eng_arc_command.h"

class EngArcHooks
{
public:
    virtual ~EngArcHooks() {};

    virtual std::shared_ptr<StaticCpDmaEngArcCommand> getStaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                                                                  EngArcBufferAddrBase srcAddrBaseId,
                                                                                  uint64_t             dataSize,
                                                                                  bool                 yield,
                                                                                  unsigned             engId) const = 0;

    virtual std::shared_ptr<NopEngArcCommand>
    getNopEngArcCommand(bool switchCQ = false, bool yield = false, unsigned padding = 0) const = 0;

    virtual std::shared_ptr<DynamicWorkDistEngArcCommand>
    getDynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion = 1) const = 0;

    virtual std::shared_ptr<ScheduleDmaEngArcCommand>
    getScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                EngArcBufferAddrBase srcAddrBaseId,
                                uint64_t             dstOffset,
                                uint64_t             dataSize,
                                bool                 isDmaWithDcoreLocality,
                                bool                 yield) const = 0;

    virtual std::shared_ptr<ListSizeEngArcCommand> getListSizeEngArcCommand() const = 0;

    virtual std::shared_ptr<SignalOutEngArcCommand>
    getSignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield) const = 0;

    virtual std::shared_ptr<ResetSobsArcCommand> getResetSobsArcCommand(unsigned target,
                                                                        unsigned targetXps,  // used by gaudi3
                                                                        unsigned totalNumEngs,
                                                                        bool     switchBit,
                                                                        bool     yield) const = 0;

    virtual std::shared_ptr<EngArcCommand> getMcidRolloverArcCommand(unsigned target,
                                                                     unsigned targetXps,
                                                                     bool     switchBit,
                                                                     bool     yield) const = 0;
    virtual unsigned getTpcWdCtxtSize() const = 0;

    virtual unsigned getMmeWdCtxtSize() const = 0;

    virtual unsigned getEdmaWdCtxtSize() const = 0;

    virtual unsigned getRotWdCtxtSize() const = 0;

    virtual unsigned getWdCtxtCount() const = 0;

    virtual unsigned getCpuIdAll() const = 0;

    virtual unsigned getStaticEcbChunkSize() const = 0;

    virtual unsigned getDynamicEcbChunkSize() const = 0;

    virtual bool
    getQueueIdInfo(HabanaDeviceType& type, unsigned& deviceID, unsigned& streamID, unsigned queueId) const = 0;

    virtual unsigned engIdx2cpuId(unsigned idx, HabanaDeviceType type) const = 0;
};
