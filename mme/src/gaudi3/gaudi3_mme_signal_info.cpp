#include "include/gaudi3/gaudi3_mme_signal_info.h"

namespace gaudi3
{
void Gaudi3SignalingInfo::setSlaveSyncObjectAddr(unsigned colorSet, uint32_t addr, Mme::Desc* desc) const
{
    MME_ASSERT(colorSet < 2, "mme sync only have 2 colors");
    if (colorSet == 0)
    {
        desc->syncObject.slaveSo0Addr = addr;
    }
    else if (colorSet == 1)
    {
        desc->syncObject.slaveSo1Addr = addr;
    }
}

unsigned Gaudi3SignalingInfo::getAguOutLoopMask(unsigned outputPortIdx, const Mme::Desc* desc) const
{
    bool isDMA = desc->header.dmaMode;
    if (isDMA)
    {
        return desc->brains.aguOutDma.loopMask;
    }
    else
    {
        return desc->brains.aguOut.loopMask;
    }
}

void Gaudi3SignalingInfo::handleNonStoreDescOnSignal(Mme::Desc* desc) const
{
    MME_ASSERT(getStoreEn(0, desc) == false, "descriptor should be non-store");
    bool isDMA = desc->header.dmaMode;
    if (isDMA)
    {
        desc->brains.aguOutDma.masterEn = 1;
        // We have to enable slave because master waits for its signal
        desc->brains.aguOutDma.slaveEn = desc->brains.aguA.slaveEn || desc->brains.aguB.slaveEn;
    }
    else
    {
        desc->brains.aguOut.masterEn = 1;
        desc->brains.aguOut.slaveEn = desc->brains.aguA.slaveEn || desc->brains.aguB.slaveEn;
    }
}

}  // namespace gaudi3
