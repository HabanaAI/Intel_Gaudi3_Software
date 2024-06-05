#include "gaudi3_sbreuse.h"
#include "gaudi3/mme.h"
#include "include/mme_common/mme_common_enum.h"

using namespace MmeCommon;

namespace gaudi3
{
void Gaudi3SBReuse::setDescSbRepeatSteps(unsigned repeatDenseSteps, unsigned repeatSpatialSteps, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->sbRepeat.repeatAMinus1 = (repeatDenseSteps > 0) ? repeatDenseSteps - 1 : 0;
    desc->sbRepeat.repeatBMinus1 = (repeatSpatialSteps > 0) ? repeatSpatialSteps - 1 : 0;
}

void Gaudi3SBReuse::setDescSbRepeatMask(EMmeLoopMask repeatDenseMask, EMmeLoopMask repeatSpatialMask, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->sbRepeat.repeatAMask = repeatDenseMask;
    desc->sbRepeat.repeatBMask = repeatSpatialMask;
}

void Gaudi3SBReuse::setDescBrainsAgu(uint8_t denseBitMaskLoops, uint8_t spatialBitMaskLoops, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->brains.aguA.loopMask = denseBitMaskLoops;
    desc->brains.aguB.loopMask = spatialBitMaskLoops;
}

void Gaudi3SBReuse::setDescAccums(unsigned rollAccums,
                                  bool isAccumOutputEnable,
                                  bool isStoreOutputEnable,
                                  void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->header.rollAccums = rollAccums;
    desc->header.accumEn = isAccumOutputEnable;
    desc->header.storeEn0 = isStoreOutputEnable;
}

}  // namespace gaudi3