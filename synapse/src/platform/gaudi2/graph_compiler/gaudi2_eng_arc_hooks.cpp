#include "gaudi2_eng_arc_hooks.h"

#include "gaudi2_eng_arc_command.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "platform/gaudi2/utils.hpp"

namespace gaudi2
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
    return std::make_shared<Gaudi2StaticCpDmaEngArcCommand>(srcOffset, srcAddrBaseId, dataSize, yield, engId);
}

std::shared_ptr<NopEngArcCommand> EngArcHooks::getNopEngArcCommand(bool switchCQ, bool yield, unsigned padding) const
{
    return std::make_shared<Gaudi2NopEngArcCommand>(switchCQ, yield, padding);
}

std::shared_ptr<DynamicWorkDistEngArcCommand>
EngArcHooks::getDynamicWorkDistEngArcCommand(unsigned wdCtxSlot, bool yield, unsigned numDmaCompletion) const
{
    return std::make_shared<Gaudi2DynamicWorkDistEngArcCommand>(wdCtxSlot, yield, numDmaCompletion);
}

std::shared_ptr<ScheduleDmaEngArcCommand>
EngArcHooks::getScheduleDmaEngArcCommand(uint64_t             srcOffset,
                                         EngArcBufferAddrBase srcAddrBaseId,
                                         uint64_t             dstOffset,
                                         uint64_t             dataSize,
                                         bool                 isDmaWithDcoreLovality, // unused in gaudi2
                                         bool                 yield) const
{
    return std::make_shared<Gaudi2ScheduleDmaEngArcCommand>(srcOffset, srcAddrBaseId, dstOffset, dataSize, yield);
}

std::shared_ptr<ListSizeEngArcCommand> EngArcHooks::getListSizeEngArcCommand() const
{
    return std::make_shared<Gaudi2ListSizeEngArcCommand>();
}

std::shared_ptr<SignalOutEngArcCommand>
EngArcHooks::getSignalOutEngArcCommand(unsigned sigValue, bool switchBit, bool yield) const
{
    return std::make_shared<Gaudi2SignalOutEngArcCommand>(sigValue, switchBit, yield);
}

std::shared_ptr<ResetSobsArcCommand> EngArcHooks::getResetSobsArcCommand(unsigned target,
                                                                         unsigned targetXps,  // unused in gaudi2
                                                                         unsigned totalNumEngs,
                                                                         bool     switchBit,
                                                                         bool     yield) const
{
    return std::make_shared<Gaudi2ResetSobsArcCommand>(target, totalNumEngs, switchBit, yield);
}

std::shared_ptr<EngArcCommand> EngArcHooks::getMcidRolloverArcCommand(unsigned target,
                                                                      unsigned targetXps,
                                                                      bool     switchBit,
                                                                      bool     yield) const
{
    HB_ASSERT(0, "getMcidRolloverArcCommand is not expected to be used by Gaudi2");
    return nullptr;
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
    return gaudi2::getQueueIdInfo(type, deviceID, streamID, (gaudi2_queue_id)queueId);
}

unsigned EngArcHooks::engIdx2cpuId(unsigned idx, HabanaDeviceType type) const
{
    HalReaderPtr halReader = Gaudi2HalReader::instance();
    switch (type)
    {
        case DEVICE_TPC:
            HB_ASSERT(idx < halReader->getNumTpcEngines(), "unexpected tpc engine index");
            return idx;
        case DEVICE_EDMA:
            HB_ASSERT(idx < halReader->getNumInternalDmaEngines(), "unexpected edma engine index");
            return idx;
        case DEVICE_MME:
            HB_ASSERT(idx < halReader->getNumMmeEnginesWithSlaves(), "unexpected mme engine index");
            return idx == 0 ? 0 : 1;
        case DEVICE_ROTATOR:
            HB_ASSERT(idx < halReader->getNumRotatorEngines(), "unexpected rotator engine index");
            return idx;
        default:
            break;
    }
    HB_ASSERT(0, "Don't know to compose ARC cpu ID for engine index {} and device type {}", idx, type);
    return 0xFF;
}

}  // namespace gaudi2
