#include "mme_assert.h"
#include "include/mme_common/mme_signal_info.h"

namespace MmeCommon
{
template<typename Desc>
void MmeDescriptorSignalingInfo<Desc>::addSignalInfo(const EMmeSignalingMode signalingMode,
                                                     const bool slaveSignaling,
                                                     const bool isLast,
                                                     const bool squashIORois,
                                                     const unsigned signalAmount,
                                                     Desc* desc)
{
    // we save the signal mode in color0  - and patch color 1 if needed later.
    resetSignal(0 /*=colorSet*/, desc);
    resetSignal(1 /*=colorSet*/, desc);

    initializeSlaveSignaling(desc, slaveSignaling);
    initializeSyncObjectConfig(desc);

    if (e_mme_signaling_none == signalingMode)
    {
        // nothing to do
    }
    else if ((e_mme_signaling_desc == signalingMode) || (e_mme_signaling_once == signalingMode && isLast))
    {
        // signal on outer loop , signaling on each descriptor or on last one in once mode.
        setSignal(0 /*=colorSet*/, e_mme_outer_loop, desc);
        // handle non store descriptors.
        if (!getStoreEn(0, desc))
        {
            handleNonStoreDescOnSignal(desc);
        }
    }
    else if (e_mme_signaling_desc_with_store == signalingMode)
    {
        // signaling on outer loop , only when storeEn == 1
        setSignal(0 /*=colorSet*/, e_mme_outer_loop, desc, getStoreEn(0, desc));
    }
    else if (e_mme_signaling_output == signalingMode)
    {
        setSignal(0 /*=colorSet*/,
                  getAguOutLoopMask(0 /*=outputPortIdx*/, desc),
                  desc,
                  getStoreEn(0 /*=outputIdx*/, desc));
    }
    else if (e_mme_signaling_partial == signalingMode)
    {
        setSignal(0 /*=colorSet*/, getAguOutLoopMask(0 /*=outputPortIdx*/, desc), desc);
    }
    else if (e_mme_signaling_chunk == signalingMode)
    {
        // mask all non storing loops and the first storing loop.
        // dont signal if this descriptor is a partial desc.
        unsigned mask = (getAguOutLoopMask(0 /*=outputPortIdx*/, desc) << 1) | 1;
        setSignal(0 /*=colorSet*/, mask, desc, getStoreEn(0, desc));
    }
    else if (e_mme_signaling_amount == signalingMode)
    {
        // signal approximately on the given amount.
        // dont signal if this descriptor is a partial desc.

        MME_ASSERT(signalAmount > 0, "must have signal amount larger than 0");
        setSignal(0 /*colorSet*/, generateSignalAmountMask(signalAmount, desc), desc, getStoreEn(0, desc));
    }
    if ((getStoreEn(0, desc) || getStoreEn(1, desc)) && !getSignalEn(0, desc))
    {
        MME_ASSERT(squashIORois, "a storing descriptor must signal unless ROIs are squashed");
    }
}

/* this function scans the MME loops from the outer most to the ineer most,
 * trying to mask as many loops as possible that will allow us to signal at least signalAmount times.
 *
 * when signaling once we simply mask all the loops (Mme::e_mme_outer_loop)
 * otherwise gradually add unmasked loops until the MME will signal enough times.
 *
 * this function assumes that EU accumulation is done at most using loops conv0 up to conv3,
 * right now we never accumulate using the spatial loop (we use up to conv2 in FWD/DEDX).
 */
template<typename Desc>
unsigned MmeDescriptorSignalingInfo<Desc>::generateSignalAmountMask(const unsigned signalAmount, const Desc* desc)
{
    unsigned accumulatedSignals = 1;
    if (accumulatedSignals >= signalAmount)
    {
        return e_mme_outer_loop;
    }

    accumulatedSignals *= getOuterLoopSize(desc);
    if (accumulatedSignals >= signalAmount)
    {
        return e_mme_tetris_loop;
    }
    accumulatedSignals *= getTetrisLoopSize(desc);
    if (accumulatedSignals >= signalAmount)
    {
        return e_mme_conv_loop_3;
    }

    for (unsigned dim = MME_MAX_CONV_DIMS - 1; dim >= 0; dim--)
    {
        accumulatedSignals *= getConvLoopSize(dim, desc);
        unsigned signalMask = getLoopMaskfromLoop(dim);
        if (accumulatedSignals >= signalAmount)
        {
            return signalMask;
        }
        if (signalMask == getAguOutLoopMask(0 /*=outputPortIdx*/, desc))
        {
            // reached the non storing loops cant use them to signal
            break;
        }
    }

    MME_ASSERT(0, "couldnt generate enough signal using all the loops.");
    return e_mme_gemm_loop;
}

template<typename Desc>
unsigned MmeDescriptorSignalingInfo<Desc>::countSignals(const Desc* desc)
{
    if (!getSignalEn(0 /*=colorSet*/, desc) && !getSignalEn(1 /*=colorSet*/, desc))
    {
        return 0;
    }

    if (getSignalEn(0 /*=colorSet*/, desc) && getSignalEn(1 /*=colorSet*/, desc))
    {
        MME_ASSERT(getSignalMask(0, desc) == getSignalMask(1, desc), "signal mask should be the same for both colors");
    }

    unsigned ret = 1;
    unsigned signalMask = getSignalEn(0, desc) ? getSignalMask(0, desc) : getSignalMask(1, desc);

    for (unsigned dim = 0; dim < MME_MAX_CONV_DIMS; dim++)
    {
        if ((signalMask & (1 << dim)) == 0)
        {
            ret *= getConvLoopSize(dim, desc);
        }
    }
    if ((signalMask & getLoopFromLoopMask(e_mme_tetris_loop)) == 0)
    {
        ret *= getTetrisLoopSize(desc);
    }

    if ((signalMask & getLoopFromLoopMask(e_mme_outer_loop)) == 0)
    {
        ret *= getOuterLoopSize(desc);
    }

    return ret;
}

template<typename Desc>
void MmeDescriptorSignalingInfo<Desc>::patchSignalColoring(Desc& desc,
                                                           const bool addr0isSram,
                                                           const bool addr1isSram,
                                                           bool useSameColorSet)
{
    if (useSameColorSet)
    {
        //  use color 0 for all output (reduces runtime patching complexity).
        setColorSet(0 /*=outputIdx*/, 0 /*=colorSet*/, &desc);
        setColorSet(1 /*=outputIdx*/, 0 /*=colorSet*/, &desc);
    }
    else
    {
        setColorSet(0 /*=outputIdx*/, !addr0isSram /*=colorSet*/, &desc);
        setColorSet(1 /*=outputIdx*/, !addr1isSram /*=colorSet*/, &desc);
    }
}

template<typename Desc>
void MmeDescriptorSignalingInfo<Desc>::patchSyncObject(Desc* desc,
                                                       const uint32_t addr0,
                                                       const uint32_t addr1,
                                                       const uint32_t slaveAddr0,
                                                       const uint32_t slaveAddr1)
{
    // signalEn was stored at signalEn0, now determine which signal to actually send.
    bool signalEn = getSignalEn(0, desc);
    unsigned signalMask = getSignalMask(0, desc);
    resetSignal(0, desc);

    setSyncObject(desc, getColorSet(0, desc), signalMask, addr0, slaveAddr0, signalEn);

    if (addr1)
    {
        setSyncObject(desc, getColorSet(1, desc), signalMask, addr1, slaveAddr1, signalEn);
    }
}

template<typename Desc>
void CommonSignalingInfo<Desc>::resetSignal(unsigned colorSet, Desc* desc) const
{
    MME_ASSERT(colorSet < 2, "mme sync only have 2 colors");
    if (colorSet == 0)
    {
        desc->syncObject.signalMask0 = c_outer_loop_idx;
        desc->syncObject.signalEn0 = 0;
    }
    else if (colorSet == 1)
    {
        desc->syncObject.signalMask1 = c_outer_loop_idx;
        desc->syncObject.signalEn1 = 0;
    }
}

template<typename Desc>
void CommonSignalingInfo<Desc>::setSignal(unsigned colorSet, unsigned mask, Desc* desc, bool signalEnable) const
{
    MME_ASSERT(colorSet < 2, "mme sync only have 2 colors");
    if (colorSet == 0)
    {
        desc->syncObject.signalMask0 = mask;
        desc->syncObject.signalEn0 = signalEnable;
    }
    else if (colorSet == 1)
    {
        desc->syncObject.signalMask1 = mask;
        desc->syncObject.signalEn1 = signalEnable;
    }
}

template<typename Desc>
bool CommonSignalingInfo<Desc>::getSignalEn(unsigned colorSet, const Desc* desc) const
{
    MME_ASSERT(colorSet < 2, "mme sync only have 2 colors");
    if (colorSet == 0)
    {
        return desc->syncObject.signalEn0;
    }
    else
    {
        return desc->syncObject.signalEn1;
    }
}

template<typename Desc>
MmeCommon::EMmeLoopMask CommonSignalingInfo<Desc>::getSignalMask(unsigned colorSet, const Desc* desc) const
{
    MME_ASSERT(colorSet < 2, "mme sync only have 2 colors");
    if (colorSet == 0)
    {
        return (MmeCommon::EMmeLoopMask) desc->syncObject.signalMask0;
    }
    else
    {
        return (MmeCommon::EMmeLoopMask) desc->syncObject.signalMask1;
    }
}

template<typename Desc>
void CommonSignalingInfo<Desc>::initializeSlaveSignaling(Desc* desc, const bool slaveSignaling) const
{
    auto& syncObject = desc->syncObject;
    syncObject.masterWaitForSlaveFence =
        (!slaveSignaling && (desc->brains.aguA.slaveEn || desc->brains.aguB.slaveEn)) ? 1 : 0;
    syncObject.slaveSendFence2Master = desc->syncObject.masterWaitForSlaveFence;
    syncObject.slaveSignalEn = slaveSignaling;
}

template<typename Desc>
void CommonSignalingInfo<Desc>::initializeSyncObjectConfig(Desc* desc) const
{
    auto& syncObject = desc->syncObject;
    syncObject.so0Addr = 0;
    syncObject.so0Val.soValue = 1;
    syncObject.so0Val.soOp = 1;
    syncObject.so0Val.soPerfEn = 0;
    syncObject.so1Addr = 0;
    syncObject.so1Val.soValue = 1;
    syncObject.so1Val.soOp = 1;
    syncObject.so1Val.soPerfEn = 0;
    setSlaveSyncObjectAddr(0, 0, desc);
    setSlaveSyncObjectAddr(1, 0, desc);
}

template<typename Desc>
bool CommonSignalingInfo<Desc>::getStoreEn(unsigned outputIdx, const Desc* desc) const
{
    MME_ASSERT(outputIdx < 2, "mme only have 2 output targets");
    if (outputIdx == 0)
    {
        return desc->header.storeEn0;
    }
    else
    {
        return desc->header.storeEn1;
    }
}

template<typename Desc>
unsigned CommonSignalingInfo<Desc>::getOuterLoopSize(const Desc* desc) const
{
    return desc->outerLoop.sizeMinus1 + 1;
}

template<typename Desc>
unsigned CommonSignalingInfo<Desc>::getTetrisLoopSize(const Desc* desc) const
{
    return desc->numIterationsMinus1 + 1;
}

template<typename Desc>
unsigned CommonSignalingInfo<Desc>::getConvLoopSize(unsigned loopIdx, const Desc* desc) const
{
    MME_ASSERT(loopIdx < MME_MAX_CONV_DIMS, "loop idx cannot be greater than MAX_CONV_DIMS(=4)");
    return desc->conv.kernelSizeMinus1.dim[loopIdx] + 1;
}

template<typename Desc>
void CommonSignalingInfo<Desc>::setColorSet(unsigned outputIdx, unsigned colorSet, Desc* desc) const
{
    MME_ASSERT(outputIdx < 2, "mme only have 2 output targets");
    MME_ASSERT(colorSet < 2, "mme sync only has 2 colors");
    if (outputIdx == 0)
    {
        desc->header.storeColorSet0 = colorSet;
    }
    else if (outputIdx == 1)
    {
        desc->header.storeColorSet1 = colorSet;
    }
}

template<typename Desc>
uint32_t CommonSignalingInfo<Desc>::getColorSet(unsigned outputIdx, Desc* desc) const
{
    MME_ASSERT(outputIdx < 2, "mme only have 2 output targets");
    if (outputIdx == 0)
    {
        return desc->header.storeColorSet0;
    }
    else
    {
        return desc->header.storeColorSet1;
    }
}

template<typename Desc>
void CommonSignalingInfo<Desc>::setSyncObject(Desc* desc,
                                              unsigned colorSet,
                                              unsigned signalMask,
                                              uint32_t masterAddr,
                                              uint32_t slaveAddr,
                                              bool signalEnable) const
{
    MME_ASSERT(masterAddr != 0, "SO address cannot be 0");
    MME_ASSERT(colorSet < 2, "mme sync only has 2 colors");
    if (colorSet == 0)
    {
        desc->syncObject.so0Addr = masterAddr;
        setSignal(0, signalMask, desc, signalEnable);
    }
    else
    {
        desc->syncObject.so1Addr = masterAddr;
        setSignal(1, signalMask, desc, signalEnable);
    }
    // Handle slave SO. It's optional.
    if (desc->syncObject.slaveSignalEn)
    {
        setSlaveSyncObject(desc, colorSet, slaveAddr);
    }
}

template<typename Desc>
void CommonSignalingInfo<Desc>::setSlaveSyncObject(Desc* desc, unsigned colorSet, uint32_t slaveAddr) const
{
    MME_ASSERT(slaveAddr != 0, "slave SO address cannot be 0");
    MME_ASSERT(colorSet < 2, "mme sync only has 2 colors");
    if (colorSet == 0)
    {
        desc->syncObject.slave0UseSlaveSOAddr = true;
    }
    else
    {
        desc->syncObject.slave1UseSlaveSOAddr = true;
    }
    setSlaveSyncObjectAddr(colorSet, slaveAddr, desc);
}

}  // namespace MmeCommon

#include "gaudi2/mme.h"
#include "gaudi3/mme.h"
// instantiate for all chips
template class MmeCommon::MmeDescriptorSignalingInfo<Gaudi2::Mme::Desc>;
template class MmeCommon::CommonSignalingInfo<Gaudi2::Mme::Desc>;
template class MmeCommon::MmeDescriptorSignalingInfo<gaudi3::Mme::Desc>;
template class MmeCommon::CommonSignalingInfo<gaudi3::Mme::Desc>;
