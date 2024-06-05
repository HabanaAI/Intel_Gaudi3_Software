#include "include/gaudi2/gaudi2_mme_signal_info.h"

namespace Gaudi2
{
void Gaudi2SignalingInfo::setSlaveSyncObjectAddr(unsigned colorSet, uint32_t addr, Mme::Desc* desc) const
{
    MME_ASSERT(colorSet < 2, "mme sync only have 2 colors");
    if (colorSet == 0)
    {
        desc->slaveSyncObject0Addr = addr;
    }
    else if (colorSet == 1)
    {
        desc->slaveSyncObject1Addr = addr;
    }
}

unsigned Gaudi2SignalingInfo::getAguOutLoopMask(unsigned outputPortIdx, const Mme::Desc* desc) const
{
    MME_ASSERT(outputPortIdx < 2, "mme only have 2 output ports");
    if (outputPortIdx == 0)
    {
        return desc->brains.aguOut0.loopMask;
    }
    else
    {
        return desc->brains.aguOut1.loopMask;
    }
}

void Gaudi2SignalingInfo::handleNonStoreDescOnSignal(Mme::Desc* desc) const
{
    MME_ASSERT(getStoreEn(0, desc) == false, "descriptor should be non-store");
    desc->brains.aguOut0.masterEn = 1;
    // We have to enable slave because master waits for its signal
    desc->brains.aguOut0.slaveEn = desc->brains.aguA.slaveEn || desc->brains.aguB.slaveEn;
    // Output port1 behaves like port0 when enabled
    desc->brains.aguOut1.masterEn = desc->brains.aguOut0.masterEn;
    desc->brains.aguOut1.slaveEn = desc->brains.aguOut0.slaveEn;
}

}  // namespace Gaudi2
