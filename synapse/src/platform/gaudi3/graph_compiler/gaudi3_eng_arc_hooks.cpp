#include <defs.h>
#include "gaudi3_eng_arc_hooks.h"
#include "gaudi3_eng_arc_command.h"
#include "gaudi3/gaudi3_packets.h"
#include "hal_conventions.h"

namespace gaudi3
{
EngArcHooks& EngArcHooks::instance()
{
    // Singleton implementation - static variable is created only once.
    thread_local static EngArcHooks onlyOneInstance;
    return onlyOneInstance;
}

EngArcHooks::EngArcHooks() {}

EngArcHooks::~EngArcHooks() {}

std::shared_ptr<StaticCpDmaEngArcCommand> EngArcHooks::getStaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                                                                   EngArcBufferAddrBase srcAddrBaseId,
                                                                                   uint64_t             dataSize,
                                                                                   bool                 yield,
                                                                                   unsigned             engId) const
{
    return std::make_shared<Gaudi3StaticCpDmaEngArcCommand>(srcOffset, srcAddrBaseId, dataSize, yield, engId);
}

std::shared_ptr<NopEngArcCommand> EngArcHooks::getNopEngArcCommand(bool switchCQ, bool yield, unsigned padding) const
{
    return std::make_shared<Gaudi3NopEngArcCommand>(switchCQ, yield, padding);
}

std::shared_ptr<DynamicWorkDistEngArcCommand>
EngArcHooks::getDynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion) const
{
    return std::make_shared<Gaudi3DynamicWorkDistEngArcCommand>(wdCtxSlot, yield, numDmaCompletion);
}

std::shared_ptr<ScheduleDmaEngArcCommand>
EngArcHooks::getScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                         EngArcBufferAddrBase srcAddrBaseId,
                                         uint64_t             dstOffset,
                                         uint64_t             dataSize,
                                         bool                 isDmaWithDcoreLocality,
                                         bool                 yield) const
{
    return std::make_shared<Gaudi3ScheduleDmaEngArcCommand>(srcOffset,
                                                            srcAddrBaseId,
                                                            dstOffset,
                                                            dataSize,
                                                            isDmaWithDcoreLocality,
                                                            yield);
}

std::shared_ptr<ListSizeEngArcCommand> EngArcHooks::getListSizeEngArcCommand() const
{
    return std::make_shared<Gaudi3ListSizeEngArcCommand>();
}

std::shared_ptr<SignalOutEngArcCommand>
EngArcHooks::getSignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield) const
{
    return std::make_shared<Gaudi3SignalOutEngArcCommand>(sigValue, switchBit, yield);
}

std::shared_ptr<ResetSobsArcCommand> EngArcHooks::getResetSobsArcCommand(unsigned target,
                                                                         unsigned targetXps,
                                                                         unsigned totalNumEngs,
                                                                         bool     switchBit,
                                                                         bool     yield) const
{
    return std::make_shared<Gaudi3ResetSobsArcCommand>(target, targetXps, totalNumEngs, switchBit, yield);
}

std::shared_ptr<EngArcCommand> EngArcHooks::getMcidRolloverArcCommand(unsigned target,
                                                                      unsigned targetXps,
                                                                      bool     switchBit,
                                                                      bool     yield) const
{
    return std::make_shared<Gaudi3McidRolloverArcCommand>(target, targetXps, switchBit, yield);
}

unsigned EngArcHooks::getTpcWdCtxtSize() const
{
    return sizeof(tpc_wd_ctxt_t);
}

unsigned EngArcHooks::getMmeWdCtxtSize() const
{
    return sizeof(mme_wd_ctxt_t);
}

unsigned EngArcHooks::getEdmaWdCtxtSize() const
{
    return sizeof(edma_wd_ctxt_t);
}

unsigned EngArcHooks::getRotWdCtxtSize() const
{
    return sizeof(rot_wd_ctxt_t);
}

unsigned EngArcHooks::getWdCtxtCount() const
{
    return WD_CTXT_COUNT;
}

unsigned EngArcHooks::getCpuIdAll() const
{
    return CPU_ID_ALL;
}

unsigned EngArcHooks::getStaticEcbChunkSize() const
{
    return STATIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

unsigned EngArcHooks::getDynamicEcbChunkSize() const
{
    return DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

bool EngArcHooks::getQueueIdInfo(HabanaDeviceType& type, unsigned& deviceID, unsigned& streamID, unsigned queueId) const
{
    bool retStatus = true;
    streamID       = 0;

    // TPC
    if (queueId >= GAUDI3_HDCORE0_ENGINE_ID_TPC_0 && queueId <= GAUDI3_HDCORE7_ENGINE_ID_TPC_8)
    {
        type     = DEVICE_TPC;
        deviceID = 0;  // TPC deviceID in gc is always 0 for gaudi3
    }
    // MME
    else if (queueId >= GAUDI3_HDCORE0_ENGINE_ID_MME_0 && queueId <= GAUDI3_HDCORE7_ENGINE_ID_MME_0)
    {
        type     = DEVICE_MME;
        deviceID = queueId - GAUDI3_HDCORE0_ENGINE_ID_MME_0;
    }
    // ROTATOR
    else if (queueId >= GAUDI3_HDCORE1_ENGINE_ID_ROT_0 && queueId <= GAUDI3_HDCORE6_ENGINE_ID_ROT_1)
    {
        type     = DEVICE_ROTATOR;
        deviceID = queueId - GAUDI3_HDCORE1_ENGINE_ID_ROT_0;
    }
    else
    {
        HB_ASSERT(0, "Got unexpected queue ID");
        retStatus = false;
    }

    return retStatus;
}

unsigned EngArcHooks::engIdx2cpuId(unsigned idx, HabanaDeviceType type) const
{
    HalReaderPtr halReader = Gaudi3HalReader::instance();
    switch (type)
    {
        case DEVICE_TPC:
            HB_ASSERT(idx < halReader->getNumTpcEngines(), "unexpected tpc engine index");
            return idx;
        case DEVICE_MME:
            HB_ASSERT(idx < halReader->getNumMmeEngines(), "unexpected mme engine index");
            return idx;
        case DEVICE_ROTATOR:
            HB_ASSERT(idx < halReader->getNumRotatorEngines(), "unexpected rotator engine index");
            return idx;
        default:
            break;
    }
    HB_ASSERT(0, "Don't know to compose ARC cpu ID for engine index {} and device type {}", idx, type);
    return 0xFF;
}


//==============================================================
//  CME Hooks hereafter
//==============================================================

EngArcHooksForCme& EngArcHooksForCme::instance()
{
    // Singleton implementation - static variable is created only once.
    thread_local static EngArcHooksForCme onlyOneInstance;
    return onlyOneInstance;
}

EngArcHooksForCme::EngArcHooksForCme() {}

EngArcHooksForCme::~EngArcHooksForCme() {}

std::shared_ptr<NopEngArcCommand> EngArcHooksForCme::getNopEngArcCommand(bool switchCQ, bool yield, unsigned padding) const
{
    return std::make_shared<Gaudi3CmeNopEngArcCommand>(padding);
}

unsigned EngArcHooksForCme::getDynamicEcbChunkSize() const
{
    return DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
}

//------------------------------------------------------------
// All the rest are not supported for CME
//------------------------------------------------------------

std::shared_ptr<StaticCpDmaEngArcCommand> EngArcHooksForCme::getStaticCpDmaEngArcCommand(uint64_t             srcOffset,
                                                                                         EngArcBufferAddrBase srcAddrBaseId,
                                                                                         uint64_t             dataSize,
                                                                                         bool                 yield,
                                                                                         unsigned             engId) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

std::shared_ptr<DynamicWorkDistEngArcCommand>
EngArcHooksForCme::getDynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

std::shared_ptr<ScheduleDmaEngArcCommand>
EngArcHooksForCme::getScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                               EngArcBufferAddrBase srcAddrBaseId,
                                               uint64_t             dstOffset,
                                               uint64_t             dataSize,
                                               bool                 isDmaWithDcoreLocality,
                                               bool                 yield) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

std::shared_ptr<ListSizeEngArcCommand> EngArcHooksForCme::getListSizeEngArcCommand() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

std::shared_ptr<SignalOutEngArcCommand>
EngArcHooksForCme::getSignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

std::shared_ptr<ResetSobsArcCommand> EngArcHooksForCme::getResetSobsArcCommand(unsigned target,
                                                                               unsigned targetXps,
                                                                               unsigned totalNumEngs,
                                                                               bool     switchBit,
                                                                               bool     yield) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

std::shared_ptr<EngArcCommand> EngArcHooksForCme::getMcidRolloverArcCommand(unsigned target,
                                                                            unsigned targetXps,
                                                                            bool     switchBit,
                                                                            bool     yield) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return nullptr;
}

unsigned EngArcHooksForCme::getTpcWdCtxtSize() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

unsigned EngArcHooksForCme::getMmeWdCtxtSize() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

unsigned EngArcHooksForCme::getEdmaWdCtxtSize() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

unsigned EngArcHooksForCme::getRotWdCtxtSize() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

unsigned EngArcHooksForCme::getWdCtxtCount() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

unsigned EngArcHooksForCme::getCpuIdAll() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

unsigned EngArcHooksForCme::getStaticEcbChunkSize() const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

bool EngArcHooksForCme::getQueueIdInfo(HabanaDeviceType& type, unsigned& deviceID, unsigned& streamID, unsigned queueId) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return false;
}

unsigned EngArcHooksForCme::engIdx2cpuId(unsigned idx, HabanaDeviceType type) const
{
    HB_ASSERT(0, "command is not supported by CME");
    return 0;
}

}  // namespace gaudi3
