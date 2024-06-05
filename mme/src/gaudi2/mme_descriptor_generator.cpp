#include "include/gaudi2/mme_descriptor_generator.h"

#include <limits>
#include <sstream>
#include <string>

#include "gaudi2_mme_hal_reader.h"
#include "include/gaudi2/gaudi2_utils.h"
#include "gaudi2/mme.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe_generator.h"
#include "agu_iterator.h"
#include "general_utils.h"
#include "gaudi2_linear_ranges.h"
#include "gaudi2_agu_config.h"
#include "gaudi2_sbreuse.h"
#include "src/mme_common/mme_hal_factory.h"
#include "mme_assert.h"
#include "include/mme_common/conv_sub_problems.h"
#define FMT_HEADER_ONLY
#include "spdlog/fmt/bundled/format.h"
using namespace MmeCommon;

namespace Gaudi2
{

enum aguCtrl
{
    e_mme_sb_sel_a0 = 0b000,
    e_mme_sb_sel_a1 = 0b001,
    e_mme_sb_sel_a2 = 0b010,
    e_mme_sb_sel_a3 = 0b011,
    e_mme_sb_sel_b0 = 0b100,
    e_mme_sb_sel_b1 = 0b101,
    e_mme_sb_sel_b2 = 0b110,
    e_mme_sb_sel_b3 = 0b111,
};

void MmeDescriptorGenerator::createHardcodedDesc(Mme::Desc* desc, Mme::EMmeCore coreType)
{
    memset(desc, 0, sizeof(Mme::Desc));
    // TODO: baseaddr

    // Set header
    desc->header.transA = 1;
    desc->header.transB = 0;

    desc->header.advanceA = 1;
    desc->header.advanceC = 1;

    desc->header.aguReadsA = 0b10001;  // SB0 && SB4
    desc->header.aguReadsB = 0b01100;  // SB2 && SB3

    desc->header.dataTypeIn = Mme::EMmeDataType::e_mme_dt_bf16;
    desc->header.dataTypeOut = Mme::EMmeDataType::e_mme_dt_bf16;

    desc->header.swapBaseAndOffsetA = 1;
    desc->header.swapBaseAndOffsetB = 0;
    desc->header.swapBaseAndOffsetOut = 1;

    desc->header.storeEn0 = 1;
    desc->header.doubleAccums = 1;

    desc->header.partialHeightLoopA = getLoopFromLoopMask(e_mme_tetris_loop);
    desc->header.partialHeightLoopB = getLoopFromLoopMask(e_mme_conv_loop_3);  // walk pattern - skf

    // Routing
    for (auto& type : {Mme::MME_CORE_MASTER, Mme::MME_CORE_SLAVE})
    {
        desc->ctrl.eus[type].sb0En = 1;
        desc->ctrl.eus[type].sb2En = 1;
        desc->ctrl.eus[type].sb3En = 1;
        desc->ctrl.eus[type].sb4En = 1;
        desc->ctrl.eus[type].in0En = 1;
        desc->ctrl.eus[type].in1En = 1;
        desc->ctrl.eus[type].sb0OutEn = 0;
        desc->ctrl.eus[type].sb2OutEn = 1;
        desc->ctrl.eus[type].sb3OutEn = 1;
    }

    desc->ctrl.eus[Mme::MME_CORE_MASTER].sb0Sel = e_mme_sb_sel_a0;  // SB0->a0
    desc->ctrl.eus[Mme::MME_CORE_MASTER].sb2Sel = e_mme_sb_sel_b0;  // SB2->b0
    desc->ctrl.eus[Mme::MME_CORE_MASTER].sb3Sel = e_mme_sb_sel_b1;  // SB3->b1
    desc->ctrl.eus[Mme::MME_CORE_MASTER].sb4Sel = e_mme_sb_sel_a1;  // SB4->a1
    desc->ctrl.eus[Mme::MME_CORE_MASTER].in0Sel = e_mme_sb_sel_b2;  // In0 -> b2
    desc->ctrl.eus[Mme::MME_CORE_MASTER].in1Sel = e_mme_sb_sel_b3;  // In1 -> b3

    desc->ctrl.eus[Mme::MME_CORE_SLAVE].sb0Sel = e_mme_sb_sel_a0;  // SB0->a0
    desc->ctrl.eus[Mme::MME_CORE_SLAVE].sb2Sel = e_mme_sb_sel_b2;  // SB2->b2
    desc->ctrl.eus[Mme::MME_CORE_SLAVE].sb3Sel = e_mme_sb_sel_b3;  // SB3->b3
    desc->ctrl.eus[Mme::MME_CORE_SLAVE].sb4Sel = e_mme_sb_sel_a1;  // SB4->a1
    desc->ctrl.eus[Mme::MME_CORE_SLAVE].in0Sel = e_mme_sb_sel_b0;  // In0 -> b0
    desc->ctrl.eus[Mme::MME_CORE_SLAVE].in1Sel = e_mme_sb_sel_b1;  // In1 -> b1

    // Brains
    // Brain A
    desc->brains.aguA.masterEn = 1;
    desc->brains.aguA.slaveEn = 1;
    desc->brains.aguA.loopMask = 0;
    // Brain B
    desc->brains.aguB.masterEn = 1;
    desc->brains.aguB.slaveEn = 1;
    desc->brains.aguB.loopMask = 0;
    // Brain EU
    desc->brains.eu.masterEn = 1;
    desc->brains.eu.slaveEn = 1;
    desc->brains.eu.loopMask = 0;
    // Brain ap
    desc->brains.ap.masterEn = 1;
    desc->brains.ap.slaveEn = 1;
    desc->brains.ap.loopMask = e_mme_conv_loop_2;
    // Brain aguOut
    desc->brains.aguOut0.masterEn = 1;
    desc->brains.aguOut0.slaveEn = 1;
    desc->brains.aguOut0.loopMask = e_mme_conv_loop_2;
    desc->brains.aguOut1.masterEn = 1;
    desc->brains.aguOut1.slaveEn = 1;
    desc->brains.aguOut1.loopMask = desc->brains.aguOut0.loopMask;
    desc->brains.roundingMode = Mme::EMmeRoundingMode::e_mme_rm_rn;
    desc->brains.shuffleA = Mme::e_mme_shuffle_2ports;

    const unsigned dimK = 256;
    const unsigned dimC = 256;
    const unsigned dimW = 512;
    const unsigned aPortsNr = 8;
    const unsigned bPortsNr = 4;
    const unsigned cSpatialPortsNr = 4;

    // Tensor Desc A
    desc->tensorA.validElements[0] = dimC * 1;  // stride = 1
    desc->tensorA.validElements[1] = dimW * desc->tensorA.validElements[0];

    desc->tensorA.loopStride[0] = 0;
    desc->tensorA.loopStride[1] = dimC;
    desc->tensorA.loopStride[2] = dimC * dimW;
    desc->tensorA.loopStride[3] = desc->tensorA.loopStride[2];
    desc->tensorA.loopStride[4] = 0;

    desc->tensorA.roiSize[0] = desc->tensorA.validElements[0];
    desc->tensorA.roiSize[1] = desc->tensorA.validElements[1];

    desc->tensorA.spatialStrides[0] = aPortsNr * dimC;
    desc->tensorA.spatialStrides[1] = dimC * dimW;

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorA.roiSize[dim] = desc->tensorA.roiSize[dim - 1];
        desc->tensorA.spatialStrides[dim] = desc->tensorA.spatialStrides[dim - 1];
    }

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        desc->tensorA.validElements[dim] = desc->tensorA.validElements[dim - 1];
    }

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorA.startOffset[dim] = 0;
    }

    // Tensor Desc B
    desc->tensorB.validElements[0] = dimK * 1;  // stride = 1
    desc->tensorB.validElements[1] = dimC * desc->tensorB.validElements[0];

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        desc->tensorB.validElements[dim] = desc->tensorB.validElements[dim - 1];
    }

    desc->tensorB.loopStride[0] = dimK;
    desc->tensorB.loopStride[1] = 0;
    desc->tensorB.loopStride[2] = dimC * dimK;
    desc->tensorB.loopStride[3] = desc->tensorB.loopStride[2];
    desc->tensorB.loopStride[4] = desc->tensorB.loopStride[2];

    desc->tensorB.spatialStrides[0] = dimK;
    desc->tensorB.spatialStrides[1] = 1;
    desc->tensorB.spatialStrides[2] = 1;
    desc->tensorB.spatialStrides[3] = 1;

    desc->tensorB.roiSize[0] = desc->tensorB.validElements[0];
    desc->tensorB.roiSize[1] = desc->tensorB.validElements[1];
    desc->tensorB.roiSize[2] = 1;
    desc->tensorB.roiSize[3] = 1;

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorB.startOffset[dim] = 0;
    }

    // Tensor Desc Cout
    desc->tensorCOut.validElements[0] = dimK;
    desc->tensorCOut.validElements[1] = dimW * dimK;
    desc->tensorCOut.validElements[2] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.validElements[3] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.validElements[4] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.loopStride[0] = dimK;
    desc->tensorCOut.loopStride[1] = 0;
    desc->tensorCOut.loopStride[2] = 0;
    desc->tensorCOut.loopStride[3] = 0;
    desc->tensorCOut.loopStride[3] = 0;
    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorCOut.roiSize[dim] = desc->tensorCOut.validElements[dim];
    }

    desc->tensorCOut.spatialStrides[0] = cSpatialPortsNr * dimK;
    desc->tensorCOut.spatialStrides[1] = dimK * dimW;
    desc->tensorCOut.spatialStrides[2] = dimK * dimW;
    desc->tensorCOut.spatialStrides[3] = dimK * dimW;

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorCOut.startOffset[dim] = 0;
    }

    // Sync obj
    desc->syncObject.signalMask0 = EMmeLoopMask::e_mme_outer_loop;
    desc->syncObject.signalEn0 = 1;
    desc->syncObject.signalMask1 = EMmeLoopMask::e_mme_outer_loop;
    desc->syncObject.masterWaitForSlaveFence = 1;
    desc->syncObject.slaveSendFence2Master = 1;
    desc->syncObject.so0Val.soValue = 1;
    desc->syncObject.so0Val.soOp = 1;
    desc->syncObject.so1Val.soValue = 1;

    // AGU
    unsigned aCoreOffset = (coreType == Mme::MME_CORE_SW) ? 0 : dimC;
    unsigned cCoreOffset = (coreType == Mme::MME_CORE_SW) ? 0 : dimK;
    // aguA

    desc->aguIn[0][Mme::MME_CORE_MASTER].roiBaseOffset[1] = 0 + aCoreOffset;

    desc->aguIn[0][Mme::MME_CORE_SLAVE].roiBaseOffset[1] = 2 * dimC + aCoreOffset;
    desc->aguIn[0][Mme::MME_CORE_SLAVE].roiBaseOffset[2] = 0;
    desc->aguIn[0][Mme::MME_CORE_SLAVE].roiBaseOffset[3] = 0;
    desc->aguIn[0][Mme::MME_CORE_SLAVE].roiBaseOffset[4] = 0;

    desc->aguIn[4][Mme::MME_CORE_MASTER].roiBaseOffset[1] = 4 * dimC + aCoreOffset;
    desc->aguIn[4][Mme::MME_CORE_MASTER].roiBaseOffset[2] = 0;
    desc->aguIn[4][Mme::MME_CORE_MASTER].roiBaseOffset[3] = 0;
    desc->aguIn[4][Mme::MME_CORE_MASTER].roiBaseOffset[4] = 0;

    desc->aguIn[4][Mme::MME_CORE_SLAVE].roiBaseOffset[1] = 6 * dimC + aCoreOffset;
    desc->aguIn[4][Mme::MME_CORE_SLAVE].roiBaseOffset[2] = 0;
    desc->aguIn[4][Mme::MME_CORE_SLAVE].roiBaseOffset[3] = 0;
    desc->aguIn[4][Mme::MME_CORE_SLAVE].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1A = (dimW / aPortsNr) - 1;

    // aguB
    desc->aguIn[2][Mme::MME_CORE_MASTER].roiBaseOffset[0] = 0;

    desc->aguIn[3][Mme::MME_CORE_MASTER].roiBaseOffset[0] = dimK / bPortsNr;
    desc->aguIn[3][Mme::MME_CORE_MASTER].roiBaseOffset[1] = 0;
    desc->aguIn[3][Mme::MME_CORE_MASTER].roiBaseOffset[2] = 0;
    desc->aguIn[3][Mme::MME_CORE_MASTER].roiBaseOffset[3] = 0;
    desc->aguIn[3][Mme::MME_CORE_MASTER].roiBaseOffset[4] = 0;

    desc->aguIn[2][Mme::MME_CORE_SLAVE].roiBaseOffset[0] = 2 * dimK / bPortsNr;
    desc->aguIn[2][Mme::MME_CORE_SLAVE].roiBaseOffset[1] = 0;
    desc->aguIn[2][Mme::MME_CORE_SLAVE].roiBaseOffset[2] = 0;
    desc->aguIn[2][Mme::MME_CORE_SLAVE].roiBaseOffset[3] = 0;
    desc->aguIn[2][Mme::MME_CORE_SLAVE].roiBaseOffset[4] = 0;

    desc->aguIn[3][Mme::MME_CORE_SLAVE].roiBaseOffset[0] = 3 * dimK / bPortsNr;
    desc->aguIn[3][Mme::MME_CORE_SLAVE].roiBaseOffset[1] = 0;
    desc->aguIn[3][Mme::MME_CORE_SLAVE].roiBaseOffset[2] = 0;
    desc->aguIn[3][Mme::MME_CORE_SLAVE].roiBaseOffset[3] = 0;
    desc->aguIn[3][Mme::MME_CORE_SLAVE].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1B = dimC - 1;

    // aguOut

    desc->aguOut[0][Mme::MME_CORE_MASTER].roiBaseOffset[0] = 0;
    desc->aguOut[0][Mme::MME_CORE_MASTER].roiBaseOffset[1] = 0 + cCoreOffset;
    desc->aguOut[0][Mme::MME_CORE_MASTER].roiBaseOffset[2] = 0;
    desc->aguOut[0][Mme::MME_CORE_MASTER].roiBaseOffset[3] = 0;
    desc->aguOut[0][Mme::MME_CORE_MASTER].roiBaseOffset[4] = 0;

    desc->aguOut[1][Mme::MME_CORE_MASTER].roiBaseOffset[0] = dimK / 2;
    desc->aguOut[1][Mme::MME_CORE_MASTER].roiBaseOffset[1] = 0 + cCoreOffset;
    desc->aguOut[1][Mme::MME_CORE_MASTER].roiBaseOffset[2] = 0;
    desc->aguOut[1][Mme::MME_CORE_MASTER].roiBaseOffset[3] = 0;
    desc->aguOut[1][Mme::MME_CORE_MASTER].roiBaseOffset[4] = 0;

    desc->aguOut[0][Mme::MME_CORE_SLAVE].roiBaseOffset[0] = 0;
    desc->aguOut[0][Mme::MME_CORE_SLAVE].roiBaseOffset[1] = 2 * dimK + cCoreOffset;
    desc->aguOut[0][Mme::MME_CORE_SLAVE].roiBaseOffset[2] = 0;
    desc->aguOut[0][Mme::MME_CORE_SLAVE].roiBaseOffset[3] = 0;
    desc->aguOut[0][Mme::MME_CORE_SLAVE].roiBaseOffset[4] = 0;

    desc->aguOut[1][Mme::MME_CORE_SLAVE].roiBaseOffset[0] = dimK / 2;
    desc->aguOut[1][Mme::MME_CORE_SLAVE].roiBaseOffset[1] = 2 * dimK + cCoreOffset;
    desc->aguOut[1][Mme::MME_CORE_SLAVE].roiBaseOffset[2] = 0;
    desc->aguOut[1][Mme::MME_CORE_SLAVE].roiBaseOffset[3] = 0;
    desc->aguOut[1][Mme::MME_CORE_SLAVE].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1Cout = (dimW / cSpatialPortsNr) - 1;

    desc->conv.associatedDims[0].dimA = 1;  // DIM_W
    desc->conv.associatedDims[0].dimB = 2;  // DIM_S
    desc->conv.associatedDims[0].dimOut = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[1].dimA = 2;  // DIM_H
    desc->conv.associatedDims[1].dimB = 3;  // DIM_R
    desc->conv.associatedDims[1].dimOut = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimA = 3;  // DIM_D/DIM_B
    desc->conv.associatedDims[2].dimB = 4;  // DIM_Q
    desc->conv.associatedDims[2].dimOut = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimB = 0;  // DIM_K
    desc->conv.associatedDims[3].dimOut = 0;  // DIM_K

    // outer loop
    desc->outerLoop.associatedDims.dimA = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimB = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimOut = Mme::c_mme_max_tensor_dims;  // skf pattern

    // Rate Limiter
    desc->rateLimiter.aguA = 4;
    desc->rateLimiter.aguB = 4;
    desc->rateLimiter.aguOut = 4;

    // TODO: pcu - not needed
    // TODO: slaveSyncObject0Addr - not needed
    // TODO: powerLoop - not needed.
}

// Calc number of ACC need to bypass at end of descriptor in order to return to first one
unsigned MmeDescriptorGenerator::getRollAccumsVal(const unsigned accumNrSp,
                                                  const unsigned accumNrDense,
                                                  const bool isBGemm8x,
                                                  const bool incInLast) const
{
    if (getRecipe().isLastPartial() || getRecipe().isPartialToMemory())
    {
        return incInLast ? 1 : 0;
    }
    else
    {
        MME_ASSERT((accumNrSp == 1) || (accumNrDense == 1), "spatial or dense size of accum's need to be 1");
        const unsigned accumNr = (accumNrSp != 1) ? accumNrSp : accumNrDense;
        if (isBGemm8x)
        {
            MME_ASSERT(accumNr <= Mme::c_mme_2x_accums_nr, "calculated accum number is above total num of accums");
            // Batch gemm 8x uses half ACC per gemm
            return (Mme::c_mme_2x_accums_nr - accumNr);
        }
        else
        {
            MME_ASSERT(accumNr <= Mme::c_mme_accums_nr, "calculated accum number is above total num of accums");
            // Multiplying by 2 as half ACC is the default unit. In this case we use 1 ACC ie two units.
            return 2 * (Mme::c_mme_accums_nr - accumNr);
        }
    }
}

void MmeDescriptorGenerator::getReuseAttr(const EMmePattern pattern, const EMmeOpType op, ReuseAttr* reuseAttr) const
{
    MME_ASSERT_PTR(reuseAttr);

    const MmeRecipe& recipe = getRecipe();
    const auto& convView = recipe.curNonSpatial();
    const auto& fcdView = recipe.curFcd();
    const auto& spView = recipe.curSp();
    const bool isDedw = isDedwOperation(op);

    unsigned filterStepsNr = isDedw ? (convView.sizes[2] * convView.sizes[3] * convView.sizes[4]) : 1;
    unsigned aSpatialSize = isDedw ? convView.sizes[1] : spView.viewSize;
    reuseAttr->aPartialHeightStepsNr = div_round_up(aSpatialSize, getGeoAttr().getGeometryHeight());
    reuseAttr->denseStepsNr = div_round_up(fcdView.viewSize, getGeoAttr().getGeometryWidth());

    EMmeLoopMask denseLoopMask;
    if ((op == e_mme_fwd) && ((reuseAttr->aPartialHeightStepsNr == 1) || (reuseAttr->denseStepsNr == 1) ||
                              (pattern == e_mme_z_reduction_skf)))
    {
        denseLoopMask = pattern2LoopMask(e_mme_z_reduction_skf, EMmeLoopDim::dim_k);
    }
    else
    {
        denseLoopMask = pattern2LoopMask(pattern, EMmeLoopDim::dim_k);
    }
    reuseAttr->denseLoopSelector = getLoopFromLoopMask(denseLoopMask);

    const bool dontUseSpatialLoop = getParams().isDedwOrGemm();
    const EMmeLoopDim dim = dontUseSpatialLoop ? EMmeLoopDim::dim_c : EMmeLoopDim::dim_s;
    EMmeLoopMask spatialLoopMask = pattern2LoopMask(getParams().strategy.pattern, dim);
    reuseAttr->aPartialHeightLoopSelector = getLoopFromLoopMask(spatialLoopMask);

    unsigned filterLoopSelector = 0;
    EMmeLoopMask filterLoopMask = e_mme_gemm_loop;
    if (isDedw)  // can reuse on filter only in dedw
    {
        filterLoopMask = pattern2LoopMask(getParams().strategy.pattern, EMmeLoopDim::dim_f);
        EMmeLoopMask curFilterLoopMask = filterLoopMask;
        for (unsigned convDim = 0; convDim < Mme::c_mme_max_conv_dims - 1; ++convDim)
        {
            MME_ASSERT(curFilterLoopMask <= e_mme_outer_loop, "loop mask cannot be above outer_loop");

            filterLoopSelector |= getLoopFromLoopMask(curFilterLoopMask);
            do
            {
                curFilterLoopMask = (EMmeLoopMask)((((unsigned) curFilterLoopMask) << 1) + 1);
            } while (curFilterLoopMask == e_mme_tetris_loop);
        }
    }

    switch (pattern)
    {
        case e_mme_sp_reduction_ckf:
            if (!isDedw || (reuseAttr->denseStepsNr == 1))
            {
                reuseAttr->spatialLoopSelector = reuseAttr->aPartialHeightLoopSelector | filterLoopSelector;
                reuseAttr->spatialStepsNr = filterStepsNr * reuseAttr->aPartialHeightStepsNr;
            }
            else
            {
                reuseAttr->spatialLoopSelector = filterLoopSelector;
                reuseAttr->spatialStepsNr = filterStepsNr;
                spatialLoopMask = filterLoopMask;
            }
            break;
        case e_mme_sp_reduction_kcf:
        case e_mme_sp_reduction_kfc:
        case e_mme_sp_reduction_cfk:
        case e_mme_sp_reduction_fck:
            // C and F are adjacent, can reuse on both of them.
            // in cfk/fck spatialLoopSelector will be used only if k==1 (no movement in k)
            reuseAttr->spatialLoopSelector = reuseAttr->aPartialHeightLoopSelector | filterLoopSelector;
            reuseAttr->spatialStepsNr = filterStepsNr * reuseAttr->aPartialHeightStepsNr;
            break;
        case e_mme_sp_reduction_fkc:
            if (reuseAttr->denseStepsNr == 1)
            {
                reuseAttr->spatialLoopSelector = reuseAttr->aPartialHeightLoopSelector | filterLoopSelector;
                reuseAttr->spatialStepsNr = filterStepsNr * reuseAttr->aPartialHeightStepsNr;
            }
            else
            {
                reuseAttr->spatialLoopSelector = reuseAttr->aPartialHeightLoopSelector;
                reuseAttr->spatialStepsNr = reuseAttr->aPartialHeightStepsNr;
            }
            break;
        case e_mme_z_reduction_skf:
        case e_mme_z_reduction_ksf:
            // no movement on filter dimensions, a reuse is simply the spatial size.
            reuseAttr->spatialLoopSelector = reuseAttr->aPartialHeightLoopSelector;
            reuseAttr->spatialStepsNr = reuseAttr->aPartialHeightStepsNr;
            break;
        default:
            MME_ASSERT(0, "invalid pattern");
    }

    unsigned spRem = aSpatialSize % getGeoAttr().getGeometryHeight() ? aSpatialSize % getGeoAttr().getGeometryHeight()
                                                                     : getGeoAttr().getGeometryHeight();
    unsigned denseRem = fcdView.viewSize % getGeoAttr().getGeometryWidth()
                            ? fcdView.viewSize % getGeoAttr().getGeometryWidth()
                            : getGeoAttr().getGeometryWidth();

    reuseAttr->lastAPartialHeightStepSize =
        isTransposed(getParams().opType, e_mme_in_a)
            ? round_to_multiple(spRem, getGeoAttr().getInterleavedSpatialPortsNr(MmeCommon::e_mme_op_a))
            : spRem;

    reuseAttr->lastDenseStepSize = (denseRem >= getGeoAttr().getEuWidth()) ? getGeoAttr().getEuWidth() : denseRem;

    // Setup loop mask of original reuse operand
    reuseAttr->accumDimLoopMask = (isDedw || getParams().isGemmOperation()) ? e_mme_gemm_loop : e_mme_conv_loop_2;
    // Setup 2D reuse loop mask
    if (getRecipe().reuseType() == e_mme_2d_reuse_ab)
    {
        reuseAttr->reuse2dLoopMask = spatialLoopMask;
    }
    else if (getRecipe().reuseType() == e_mme_2d_reuse_ba)
    {
        reuseAttr->reuse2dLoopMask = denseLoopMask;
    }
}

void MmeDescriptorGenerator::setFieldsFromParams(const ReuseAttr& reuseAttr, Mme::Desc* desc)
{
    const MmeRecipe& recipe = getRecipe();
    const auto& batchView = recipe.curNonSpatial();
    const bool bgemmBit = getGeoAttr().getBgemmBit();
    const bool doubleAccumsBit = getGeoAttr().getDoubleAccumsBit();
    bool extraRollInLast = false;
    if (getGeoAttr().supportsConcurrency())
    {
        unsigned batchLoops[c_batchDimNr] = {batchView.sizes[GEMM_DIM_B1],
                                             batchView.sizes[GEMM_DIM_B2],
                                             batchView.sizes[GEMM_DIM_B3]};
        unsigned concurrentIdx = getGeoAttr().getConcurrentDim() - GEMM_DIM_B1;  //  move idx to zero based
        batchLoops[concurrentIdx] = div_round_up(batchLoops[concurrentIdx], getGeoAttr().getGeometryConcurrency());
        const unsigned bgemmAccumsPerDcore = multiplyElements(&batchLoops[0], &batchLoops[c_batchDimNr - 1]);
        extraRollInLast = bgemmBit && (bgemmAccumsPerDcore % 2 == 1);
    }
    const unsigned rollAccums =
        getRollAccumsVal(reuseAttr.spatialStepsNr, reuseAttr.denseStepsNr, bgemmBit, extraRollInLast);

    desc->header.lowerA = recipe.lowering;
    desc->header.lowerB = 0;
    desc->header.accumEn = recipe.isAccumEn();
    desc->header.rollAccums = rollAccums;
    desc->header.doubleAccums = (bgemmBit || doubleAccumsBit) ? 0 : 1;  // polarity inverted
    desc->header.storeEn0 = recipe.isStoreEn();
    // storeEn1 is set in the patching phase

    desc->header.dataTypeIn = ConvertDataTypeToGaudi2(recipe.getOperand(e_mme_op_a).elementType);
    desc->header.dataTypeOut = ConvertDataTypeToGaudi2(getOutputDataType(recipe.getOperand(e_mme_op_c).elementType));
    setDescFp8Bias(desc);

    // storeColorSet0/1: to be set in patching
    desc->header.hx2 = getGeoAttr().getHx2Bit();

    // todo AlonG: replace the convert functions by access to the param fields
    bool isInputTypeFp8 = isTypeFp8(ConvertDataTypeFromGaudi2((Mme::EMmeDataType) desc->header.dataTypeIn));
    bool isOutputTypeFp8 = isTypeFp8(ConvertDataTypeFromGaudi2((Mme::EMmeDataType) desc->header.dataTypeOut));

    // set related parts of brains
    desc->brains.decEn = (isInputTypeFp8 && bgemmBit) ? 1 : 0;
    desc->brains.bgemm = bgemmBit ? 1 : 0;
    // TODO: add separate support for two types of clipping - in Execution unit and Activation Pipe
    desc->brains.clipFpEu = getOriginalParams().controls.clippingEn ? 1 : 0;
    desc->brains.clipFpAp = getOriginalParams().controls.clippingEn ? 1 : 0;
    desc->brains.sbACacheEn = getOriginalParams().controls.sbCacheEn ? 1 : 0;
    desc->brains.sbBCacheEn = getOriginalParams().controls.sbCacheEn ? 1 : 0;

    desc->brains.roundingMode = getOriginalParams().controls.conversionRoundingMode;
    desc->brains.reluEn = getOriginalParams().controls.reluEn;
    desc->brains.noRollup = false;
}

void MmeDescriptorGenerator::setEngineBrains(const ReuseAttr& reuseAttr, Mme::Desc* desc)
{
    // Brain A
    desc->brains.aguA.masterEn = 1;
    desc->brains.aguA.slaveEn = 1;
    desc->brains.aguA.loopMask = getRecipe().reuseA() ? reuseAttr.denseLoopSelector : 0;
    // Brain B
    desc->brains.aguB.masterEn = 1;
    desc->brains.aguB.slaveEn = 1;
    desc->brains.aguB.loopMask = getRecipe().reuseB() ? reuseAttr.spatialLoopSelector : 0;
    // Brain EU
    desc->brains.eu.masterEn = 1;
    desc->brains.eu.slaveEn = 1;
    desc->brains.eu.loopMask = getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0 : e_mme_gemm_loop;
    // Brain ap
    desc->brains.ap.masterEn = !desc->brains.noRollup;
    desc->brains.ap.slaveEn = desc->brains.ap.masterEn;
    desc->brains.ap.loopMask =
        getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0 : reuseAttr.accumDimLoopMask;
    // Brain aguOut0
    desc->brains.aguOut0.masterEn = desc->header.storeEn0;
    desc->brains.aguOut0.slaveEn = desc->brains.aguOut0.masterEn;
    desc->brains.aguOut0.loopMask =
        getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0 : reuseAttr.accumDimLoopMask;
    // Brain aguOut1
    desc->brains.aguOut1.masterEn = desc->brains.aguOut0.masterEn;
    desc->brains.aguOut1.slaveEn = desc->brains.aguOut0.slaveEn;
    desc->brains.aguOut1.loopMask =
        getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0 : reuseAttr.accumDimLoopMask;

    desc->brains.reserved = 0;
}

void MmeDescriptorGenerator::setSimpleFields(Mme::Desc* desc)
{
    ReuseAttr reuseAttr;
    getReuseAttr(getOriginalParams().strategy.pattern, getOriginalParams().opType, &reuseAttr);

    setFieldsFromParams(reuseAttr, desc);
    setEngineBrains(reuseAttr, desc);

    // same as old code
    Gaudi2SBReuse sbReuse(getParams(), getGeoAttr(), getRecipe());
    sbReuse.configDescSBReuse(desc);
    commonDescriptorConfigPost(*desc);
}

void MmeDescriptorGenerator::buildDescNew(unsigned mmeIDx, Mme::Desc* desc)
{
    setSimpleFields(desc);
    Gaudi2AguConfig aguConfig(getParams(), getGeoAttr(), MmeHalReader::getInstance(), mmeIDx, getRecipe());
    if (!ConvSubProblemContainer::isOutOfBounds(getParams()))
    {
        MME_ASSERT(getCurrentSubProblem() != nullptr,
                   "should have a defined sub-problem for the current descriptor generation");
        const OffsetArray& descAddrOffset = getCurrentSubProblem()->addressOffset;
        aguConfig.setDescOffset(descAddrOffset);
    }
    aguConfig.config(desc);
}

void MmeDescriptorGenerator::mmeGenerateActivations()
{
    MME_ASSERT(m_originalParams.has_value(), "LayerParams have not been initialized");
    MME_ASSERT(!m_convSubProblems.empty(), "should have at least one sub-problem");
    if (!getActivationsFromCache())
    {
        bool isGemmOperation = getOriginalParams().isGemmOperation();
        m_activations.reserve(calculateActivationsCount());
        for (unsigned subProblemIdx = 0; subProblemIdx < m_convSubProblems.size(); subProblemIdx++)
        {
            m_convSubProblems.current = &m_convSubProblems[subProblemIdx];
            auto& recipeIterator = getRecipe().getIterator();
            for (const auto& iters : recipeIterator)
            {
                recipeIterator.setCurIterVals(iters);
                m_activations.emplace_back(Mme::MME_CORE_MASTERS_NR);
                MmeActivation& activation = m_activations.back();
                activation.isGemm = isGemmOperation;
                activation.isMask = getRecipe().isMaskActivation();
                // Initialize the operand roles - by default role is the primary tensor
                activation.operandRoles[e_mme_op_a] =
                    getRecipe().isMaskActivation() ? MmeCommon::TensorRoles::AUX_ROLE_MASKED_BGEMM_A : MmeCommon::INPUT_TENSOR_A;
                activation.operandRoles[e_mme_op_b] =
                    getRecipe().isMaskActivation() ? MmeCommon::TensorRoles::AUX_ROLE_MASKED_BGEMM_B : MmeCommon::INPUT_TENSOR_B;
                activation.operandRoles[e_mme_op_c] = MmeCommon::OUTPUT_TENSOR_C;

                m_commonGeoAttr->setPrimaryTensors(!activation.isMask);

                for (unsigned descIdx = 0; descIdx < activation.descriptors.size(); descIdx++)
                {
                    buildDescNew(descIdx, &activation.getDesc(descIdx));
                }
                getRoiCalculator(getRecipe())->setSkipRoiCalc(getGeoAttr(), activation);
                activation.numSignals = m_signalingInfo.countSignals(&activation.getDesc(0));
                MME_ASSERT(activation.numSignals == m_signalingInfo.countSignals(&activation.getDesc(1)),
                        "num of signals should be equal between north and south descriptor");

                const MmeRecipe& recipe = getRecipe();
                activation.spView = recipe.curSp();
                activation.fcdView = recipe.curFcd();
                activation.nonSpatialView = recipe.curNonSpatial();

                activation.numTetrises = countTetrises(&activation.getDesc(0));
                activation.numRollups = activation.getDesc(0).brains.noRollup ? 0 : activation.numTetrises;
                getMMEBrain().addNumOfRollUpsForCurActivation(activation.numTetrises);
            }
        }
        // Verify last activation was reached
        const bool isLastActivation = getRecipe().getIterator().isLast();
        MME_ASSERT(isLastActivation, "should be on last activation");
        addParamsActivationsToCache();
    }
    m_convSubProblems.current = nullptr;

    RoiCalculator::countSkipSignals(getMmeActivations());
    configurePerfEvents(getMmeActivations());
}

void MmeDescriptorGenerator::createRoiCalculator()
{
    if (getOriginalParams().isGemmOperation())
    {
        m_roiCalculator = std::make_shared<BGemmRoiCalculator>(getRecipe(), getOriginalParams());
    }
    else
    {
        m_roiCalculator = std::make_shared<ConvRoiCalculator>(getRecipe(), getOriginalParams());
    }
}

unsigned MmeDescriptorGenerator::countTetrises(const Mme::Desc* desc)
{
    unsigned ret = 1;
    uint16_t mask = desc->brains.aguOut0.loopMask;

    for (int d = 0; d < Mme::c_mme_max_conv_dims; d++)
    {
        if ((mask & (1 << d)) == 0)
        {
            ret *= desc->conv.kernelSizeMinus1.dim[d] + 1;
        }
    }

    if ((mask & ((Mme::e_mme_tetris_loop + 1) >> 1)) == 0)
    {
        ret *= desc->numIterationsMinus1 + 1;
    }

    if ((mask & ((Mme::e_mme_outer_loop + 1) >> 1)) == 0)
    {
        ret *= desc->outerLoop.sizeMinus1 + 1;
    }

    return ret;
}

/*
 * disabling the SB cache is a small optimization to reduce HBM accesses.
 *
 * unlike the rest of the chip, the HBMs cache line size is 64B.
 * because of that every time we go out to the mesh with a 128B request it is split
 * at the HBM to two 64B accesses.
 * in some cases we only need 64B out of the whole 128B access.
 * unless we would immediately require the next 64B in memory, we should avoid generating 128B access.
 *
 * the SB cache line size is 128B, so each access generated by it will always be 128B.
 * by turning it off the MME will be able to generate 64B accesses.
 *
 * thus, the SB cache should be turned off whenever the access size is at most 64B
 * and the following 64B will not be needed (stride > 128B)
 */
void MmeDescriptorGenerator::setSBCacheDisable(const EMmeOperand operand,
                                               const bool isSram,
                                               const bool addressAligned,
                                               Mme::Desc& desc)
{
    if (isSram) return;

    bool enableCache = true;
    EMmeInternalOperand internal_op = mmeOpToInternalOp(operand, getOriginalParams().opType);
    unsigned dtSize = getElementSize(getOriginalParams().getOperand(operand).elementType);
    unsigned fcd = dtSize;
    unsigned stride = dtSize;

    switch (internal_op)
    {
        case e_mme_op_a:
            stride *= desc.tensorA.spatialStrides[0];
            fcd *= desc.tensorA.validElements[0];
            break;
        case e_mme_op_b:
            stride *= desc.tensorB.spatialStrides[0];
            fcd *= desc.tensorB.validElements[0];
            break;
        default:
            MME_ASSERT(0, "invalid operand");
    }

    if (fcd <= (Mme::c_cl_size / 2) && stride >= Mme::c_cl_size)
    {
        enableCache = false;
    }

    switch (internal_op)
    {
        case e_mme_op_a:
            // if A is unaligned the cache could prevent extra mesh accesses
            desc.brains.sbACacheEn = addressAligned ? enableCache : true;
            break;
        case e_mme_op_b:
            desc.brains.sbBCacheEn = enableCache;
            break;
        default:
            MME_ASSERT(0, "invalid operand");
    }
}

void MmeDescriptorGenerator::mmePatchSyncObjects(const uint32_t mmeIdx,
                                                 const uint32_t addr0,
                                                 const uint32_t addr1,
                                                 const uint32_t slaveAddr0,
                                                 const uint32_t slaveAddr1)
{
    for (auto& act : getMmeActivations())
    {
        m_signalingInfo.patchSyncObject(&act.getDesc(mmeIdx), addr0, addr1, slaveAddr0, slaveAddr1);
    }
}

void MmeDescriptorGenerator::mmePatchSyncObject(Mme::Desc& desc,
                                                const uint32_t addr0,
                                                const uint32_t addr1,
                                                const uint32_t slaveAddr0,
                                                const uint32_t slaveAddr1)
{
    Gaudi2SignalingInfo signalingInfo;
    signalingInfo.patchSyncObject(&desc, addr0, addr1, slaveAddr0, slaveAddr1);
}

void MmeDescriptorGenerator::patchSignalColoring(MmeActivation& activation, const bool addr0isSram, const bool addr1isSram)
{
    for (auto& desc : activation.descriptors)
    {
        m_signalingInfo.patchSignalColoring(desc,
                                            addr0isSram,
                                            addr1isSram,
                                            getOriginalParams().controls.useSameColorSet);
    }
}

#define UNSET_DESC_FIELD(M, F) memset(M + offsetof(Mme::Desc, F), 0, sizeof(Mme::Desc::F) * sizeof(bool))
#define SET_DESC_FIELD(M, F)   memset(M + offsetof(Mme::Desc, F), 1, sizeof(Mme::Desc::F) * sizeof(bool))

void MmeDescriptorGenerator::mmeGetDescValidMask(const Mme::Desc& desc,
                                                 bool* mask,
                                                 bool* aguOut1FromAguOut0_DW0,
                                                 bool* aguOut1FromAguOut0_DW1_4)
{
    for (unsigned i = 0; i < sizeof(Mme::Desc); i++)
    {
        mask[i] = true;
    }

    bool storeEn = desc.header.storeEn0 || desc.header.storeEn1;
    bool signalEn = desc.syncObject.signalEn0 || desc.syncObject.signalEn1;

    // aguOut
    if (!desc.brains.aguOut0.masterEn || (!storeEn && !signalEn))
    {
        UNSET_DESC_FIELD(mask, aguOut[0][0]);
    }
    if (!desc.brains.aguOut0.slaveEn || (!storeEn && !signalEn))
    {
        UNSET_DESC_FIELD(mask, aguOut[0][1]);
    }
    if (!desc.brains.aguOut1.masterEn || (!storeEn && !signalEn))
    {
        UNSET_DESC_FIELD(mask, aguOut[1][0]);
    }
    if (!desc.brains.aguOut1.slaveEn || (!storeEn && !signalEn))
    {
        UNSET_DESC_FIELD(mask, aguOut[1][1]);
    }
    if (!desc.header.storeEn0)
    {
        UNSET_DESC_FIELD(mask, baseAddrCOut0);
    }
    if (!desc.header.storeEn1)
    {
        UNSET_DESC_FIELD(mask, baseAddrCOut1);
    }
    if ((!desc.brains.aguOut0.masterEn && !desc.brains.aguOut0.slaveEn && !desc.brains.aguOut1.masterEn &&
         !desc.brains.aguOut1.slaveEn) ||
        !storeEn)
    {
        UNSET_DESC_FIELD(mask, tensorCOut);
        // we need C roisize0 to configure the EU
        SET_DESC_FIELD(mask, tensorCOut.roiSize[0]);  // make sure this is still correct
    }

    // disabled slave
    if (!desc.brains.eu.slaveEn)
    {
        UNSET_DESC_FIELD(mask, aguOut[0][1]);
        UNSET_DESC_FIELD(mask, aguOut[1][1]);
        UNSET_DESC_FIELD(mask, aguIn[0][1]);
        UNSET_DESC_FIELD(mask, aguIn[1][1]);
        UNSET_DESC_FIELD(mask, aguIn[2][1]);
        UNSET_DESC_FIELD(mask, aguIn[3][1]);
        UNSET_DESC_FIELD(mask, aguIn[4][1]);
    }

    // non f8
    if (desc.header.dataTypeIn != Mme::EMmeDataType::e_mme_dt_fp8_152 &&
        desc.header.dataTypeIn != Mme::EMmeDataType::e_mme_dt_fp8_143 &&
        desc.header.dataTypeOut != Mme::EMmeDataType::e_mme_dt_fp8_152 &&
        desc.header.dataTypeOut != Mme::EMmeDataType::e_mme_dt_fp8_143)
    {
        UNSET_DESC_FIELD(mask, fp8Bias);
    }

    // signaling
    if (!desc.syncObject.signalEn0)
    {
        UNSET_DESC_FIELD(mask, syncObject.so0Addr);
        UNSET_DESC_FIELD(mask, syncObject.so0Val);
    }
    if (!desc.syncObject.signalEn1)
    {
        UNSET_DESC_FIELD(mask, syncObject.so1Addr);
        UNSET_DESC_FIELD(mask, syncObject.so1Val);
    }
    if (!desc.syncObject.signalEn0 || !desc.syncObject.slaveSignalEn || !desc.syncObject.slave0UseSlaveSOAddr)
    {
        UNSET_DESC_FIELD(mask, slaveSyncObject0Addr);
    }
    if (!desc.syncObject.signalEn1 || !desc.syncObject.slaveSignalEn || !desc.syncObject.slave1UseSlaveSOAddr)
    {
        UNSET_DESC_FIELD(mask, slaveSyncObject1Addr);
    }
}

pMmeDescriptorGenerator MmeDescriptorGenerator::createMmeDescGenerator()
{
    return std::make_unique<MmeDescriptorGenerator>();
}

EMmeDataType MmeDescriptorGenerator::getOutputDataType(EMmeDataType recipeOutputDataType) const
{
    // fp32 flavors cannot be output types as they are different modes in the eu.
    if (recipeOutputDataType == EMmeDataType::e_type_fp32_ieee || recipeOutputDataType == EMmeDataType::e_type_tf32)
    {
        return EMmeDataType::e_type_fp32;
    }
    else
    {
        return recipeOutputDataType;
    }
}

void MmeDescriptorGenerator::setDescFp8Bias(Mme::Desc* desc) const
{
    MME_ASSERT(desc->header.dataTypeIn != 0 && desc->header.dataTypeOut != 0, "data types should be defined");
    if (desc->header.dataTypeIn == Mme::EMmeDataType::e_mme_dt_fp8_143)
    {
        MME_ASSERT((getParams().controls.fp8BiasIn == EXPONENT_BIAS_FP8_143_3 ||
                   getParams().controls.fp8BiasIn == EXPONENT_BIAS_FP8_143_7 ||
                   getParams().controls.fp8BiasIn == EXPONENT_BIAS_FP8_143_11 ||
                   getParams().controls.fp8BiasIn == EXPONENT_BIAS_FP8_143_15),
                  "user defined bias of input should be of values 3/7/11/15 only");
        MME_ASSERT((getParams().controls.fp8BiasIn2 == EXPONENT_BIAS_FP8_143_3 ||
                    getParams().controls.fp8BiasIn2 == EXPONENT_BIAS_FP8_143_7 ||
                    getParams().controls.fp8BiasIn2 == EXPONENT_BIAS_FP8_143_11 ||
                    getParams().controls.fp8BiasIn2 == EXPONENT_BIAS_FP8_143_15),
                   "user defined bias of input should be of values 3/7/11/15 only");
    }

    if (desc->header.dataTypeIn == Mme::EMmeDataType::e_mme_dt_fp8_152)
    {
        MME_ASSERT((getParams().controls.fp8BiasIn == EXPONENT_BIAS_FP8_152_15),
                  "user defined bias of input should be 15");
        MME_ASSERT((getParams().controls.fp8BiasIn2 == EXPONENT_BIAS_FP8_152_15),
                   "user defined bias of input should be 15");
    }

    if (desc->header.dataTypeOut == Mme::EMmeDataType::e_mme_dt_fp8_143)
    {
        MME_ASSERT((getParams().controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_3 ||
                   getParams().controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_7 ||
                   getParams().controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_11 ||
                   getParams().controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_15),
                  "user defined bias of output should be of values 3/7/11/15 only");
    }
    if (desc->header.dataTypeOut == Mme::EMmeDataType::e_mme_dt_fp8_152)
    {
        if (getParams().memoryCfg.reductionEn())
        {
            MME_ASSERT((getParams().controls.fp8BiasOut == EXPONENT_BIAS_FP8_152_15),
                       "reduction requires output bias to be 15");
        }
        else
        {
            MME_ASSERT((getParams().controls.fp8BiasOut >= EXPONENT_BIAS_FP8_152_MIN_VALUE &&
                       getParams().controls.fp8BiasOut <= EXPONENT_BIAS_FP8_152_MAX_VALUE),
                      "user defined bias of output should be of values 1-30 only");
        }
    }
    // handle input

    desc->fp8Bias.a = getParams().controls.fp8BiasIn;
    desc->fp8Bias.b = getParams().controls.fp8BiasIn2;

    // handle output

    desc->fp8Bias.out = getParams().controls.fp8BiasOut;
}

void MmeDescriptorGenerator::setSinglePerfEvent(EMmeTraceEngine engine, unsigned startEndMask, Mme::Desc& desc)
{
    Mme::MmePerfEvt* traceEngine;
    unsigned operand;
    switch (engine)
    {
        default:
            MME_ASSERT(0, "tracing for engine not supported");
        case e_mme_trace_input:
            traceEngine = &desc.perfEvtIn;
            operand = e_mme_operand_0;  // trace should be sent when any port starts, use 0 for simplicity.
            break;
        case e_mme_trace_output:
            traceEngine = &desc.perfEvtOut;
            operand = e_mme_operand_0 | e_mme_operand_1;  // trace should be sent when both output ports finish
            break;
    }
    traceEngine->value = getOriginalParams().tracing.ctxId;
    traceEngine->incMask = 0;
    traceEngine->rst = 1;
    traceEngine->loopMask = e_mme_outer_loop;
    traceEngine->startEndMask = startEndMask;
    traceEngine->operand = operand;
    traceEngine->slaveSendsPerfEvent = 1;
}

bool MmeDescriptorGenerator::validateParams(const MmeLayerParams& params, std::string& errorMsg)
{
    const MmeTensorView& output = params.isDedxOperation()        ? params.x
                                  : (params.isDedwOperation()) ? params.w
                                                                  : params.y;
    const MmeTensorView& operandA = params.isDedxOperation() ? params.y : params.x;
    const MmeTensorView& operandB = params.isDedxOperation()        ? params.w
                                    : (params.isDedwOperation()) ? params.y
                                                                    : params.w;
    if (params.strategy.mmeLimit != 1 && params.strategy.mmeLimit != 2)
    {
        errorMsg = "mmes can be limited to either 1 or 2";
        return false;
    }

    if (operandA.elementType != operandB.elementType)
    {
        errorMsg = "input element types should match";
        return false;
    }

    if (params.strategy.maskedBgemm)
    {
        if (params.x.elementType != params.xAux.elementType)
        {
            errorMsg = "input element types should match";
            return false;
        }
        if (params.w.elementType != params.wAux.elementType)
        {
            errorMsg = "input element types should match";
            return false;
        }
        int nonCommonDimA = isTransposed(params.opType, e_mme_in_a);
        int nonCommonDimB = isTransposed(params.opType, e_mme_in_b);
        if (params.x.sizes[nonCommonDimA] != params.xAux.sizes[nonCommonDimA])
        {
            errorMsg = "masked SP size doesnt match gemm SP size";
            return false;
        }
        if (params.w.sizes[nonCommonDimB] != params.wAux.sizes[nonCommonDimB])
        {
            errorMsg = "masked FCD size doesnt match gemm FCD size";
            return false;
        }
        for (int dim = 0; dim < MAX_DIMENSION; dim++)
        {
            if (params.y.sizes[dim] != params.yAux.sizes[dim])
            {
                errorMsg = "masked output size doesnt match gemm output size";
                return false;
            }
        }
    }

    if (params.isDeterministicCdConcurrency())
    {
        // Output tensor cannot be strided
        MME_ASSERT(!params.getOperand(e_mme_op_c).isStrided(), "Deterministic cd concurrency requires non-strided output tensor");
        // The packing factor must be a multiplication of some upper dims of the output tensor
        unsigned packingFactor = params.strategy.packingFactor;
        unsigned validPackingFactorValues = 1;  // 1 is valid
        bool packingFactorIsValid = false;
        for (int i=MME_MAX_TENSOR_DIMS-1; i >= 0; i--)
        {
            if (packingFactor == validPackingFactorValues)
            {
                packingFactorIsValid = true;
                break;
            }
            validPackingFactorValues *= params.getOperand(e_mme_op_c).sizes[i];
        }
        MME_ASSERT(packingFactorIsValid, "Packing factor for the deterministic cd concurrency is invalid");
    }

    bool isOutputDtypeFp8 = (output.elementType == e_type_fp8_143) || (output.elementType == e_type_fp8_152);
    if (isOutputDtypeFp8 && ((params.memoryCfg.reductionOp != e_mme_reduction_none) || params.controls.atomicAdd))
    {
        errorMsg = "reduction is not supported for fp8 output data type";
        return false;
    }

    // rounding
    if (!isOutputDtypeFp8 && params.controls.conversionRoundingMode == StochasticRoundingAndNearest)
    {
        errorMsg = "RSN rounding can only be used with fp8";
        return false;
    }
    if (params.controls.roundingMode != RoundToNearest || params.controls.accRoundingMode != RoundToNearest)
    {
        errorMsg = "gaudi2 only supports round nearest in EU";
        return false;
    }

    // input bias
    if (operandA.elementType == e_type_fp8_143 && params.controls.fp8BiasIn != EXPONENT_BIAS_FP8_143_3 &&
        params.controls.fp8BiasIn != EXPONENT_BIAS_FP8_143_7 && params.controls.fp8BiasIn != EXPONENT_BIAS_FP8_143_11 &&
        params.controls.fp8BiasIn != EXPONENT_BIAS_FP8_143_15)
    {
        errorMsg = "user defined bias of input should be of values 3/7/11/15 only";
        return false;
    }
    if (operandB.elementType == e_type_fp8_143 && params.controls.fp8BiasIn2 != EXPONENT_BIAS_FP8_143_3 &&
        params.controls.fp8BiasIn2 != EXPONENT_BIAS_FP8_143_7 &&
        params.controls.fp8BiasIn2 != EXPONENT_BIAS_FP8_143_11 &&
        params.controls.fp8BiasIn2 != EXPONENT_BIAS_FP8_143_15)
    {
        errorMsg = "user defined bias of input should be of values 3/7/11/15 only";
        return false;
    }

    else if (operandA.elementType == e_type_fp8_152 && params.controls.fp8BiasIn != EXPONENT_BIAS_FP8_152_15)
    {
        errorMsg = "user defined bias of input should be 15";
        return false;
    }
    else if (operandB.elementType == e_type_fp8_152 && params.controls.fp8BiasIn2 != EXPONENT_BIAS_FP8_152_15)
    {
        errorMsg = "user defined bias of input should be 15";
        return false;
    }
    // output bias
    if (output.elementType == e_type_fp8_143 && !(params.controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_3 ||
                                                  params.controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_7 ||
                                                  params.controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_11 ||
                                                  params.controls.fp8BiasOut == EXPONENT_BIAS_FP8_143_15))
    {
        errorMsg = "user defined bias of output should be of values 3/7/11/15 only";
        return false;
    }
    if (output.elementType == e_type_fp8_152 && !(params.controls.fp8BiasOut >= EXPONENT_BIAS_FP8_152_MIN_VALUE &&
                                                  params.controls.fp8BiasOut <= EXPONENT_BIAS_FP8_152_MAX_VALUE))
    {
        errorMsg = "user defined bias of output should be of values 1-30 only";
        return false;
    }

    if (params.strategy.isDeterministic && params.opType == MmeCommon::e_mme_deterministic_dedw)
    {
        MME_ASSERT(params.strategy.cdConcurrencyEn == MmeCommon::TurnedOn &&
                    params.strategy.reductionLevel > 1,
                    "Deterministic dedw is set only when cd concurrency is active");
    }
    else if (params.controls.atomicAdd || (params.strategy.cdConcurrencyEn == TurnedOn))
    {
        // CD Concurrency non-deterministic implies writes with Reduction Add
        const_cast<EMmeReductionOp&>(params.memoryCfg.reductionOp) = EMmeReductionOp::e_mme_reduction_add;
        const_cast<EMmeReductionRm&>(params.memoryCfg.reductionRm) = EMmeReductionRm::e_mme_reduction_round_down;
    }
    if (params.controls.signalingMode == e_mme_signaling_once && !params.controls.squashIORois)
    {
        errorMsg = "when signaling only once, ROIs have to be squashed";
        return false;
    }
    if (params.controls.signalingMode == e_mme_signaling_partial)
    {
        errorMsg = "signal partial is not yet supported";
        return false;
    }
    if (params.strategy.unrollEn && (params.strategy.batchConcurrencyEn == TurnedOn))
    {
        errorMsg = "these are competing optimizations, only one can be selected";
        return false;
    }
    if (!validateParamOfGemmOp(params, errorMsg))
    {
        return false;
    }
    if (!validateParamOfReductionAddOp(params, errorMsg))
    {
        return false;
    }
    if (params.opType == e_mme_transposed_dedx && ((params.conv.stride[0] / params.strategy.packingFactor) != 1 ||
                                                   params.conv.stride[1] != 1 || params.conv.stride[2] != 1))
    {
        errorMsg = "transposed_dedx doesn't support subProblems";
        return false;
    }
    // all checks passed.
    return true;
}

void MmeDescriptorGenerator::setDescRateLimiters(Mme::Desc* desc)
{
    desc->rateLimiter.aguA = 4;
    desc->rateLimiter.aguB = 4;
    desc->rateLimiter.aguOut = 4;
    desc->rateLimiter.eu = 0;
}

void MmeDescriptorGenerator::setPmuSaturation(Mme::Desc& desc)
{
    // TODO
    // do something with: getOriginalParams().controls.pmuSaturationVal
}

void MmeDescriptorGenerator::commonDescriptorConfigPost(Mme::Desc& desc)
{
    m_signalingInfo.addSignalInfo(getOriginalParams().controls.signalingMode,
                                  getOriginalParams().controls.slaveSignaling,
                                  getRecipe().getIterator().isLast() && isLastSubProblem(),
                                  getOriginalParams().controls.squashIORois,
                                  getRecipe().signalAmount,
                                  &desc);

    configureMemoryDirectives(desc, getRecipe());
    setDescRateLimiters(&desc);
    setPmuSaturation(desc);
}

std::string MmeDescriptorGenerator::getVerboseRecipeSummaryStr() const
{
    std::string vrbSummaryStr;

    // TODO need to generalize and expand logic to handle subproblems
    // Partials
    const unsigned partials = getRecipe().getPartialsNr();
    if (partials > 1)
    {
        const std::string partialsStr = ", partials=" + std::to_string(partials);
        vrbSummaryStr += partialsStr;
    }

    // Repeats
    unsigned repeats = 0;
    for (auto& activation : m_activations)
    {
        const auto& desc = activation.descriptors.back();
        if (desc.sbRepeat.repeatAMinus1 != 0)
        {
            repeats += desc.sbRepeat.repeatAMinus1 + 1;
        }
        if (desc.sbRepeat.repeatBMinus1 != 0)
        {
            repeats += desc.sbRepeat.repeatBMinus1 + 1;
        }
    }
    if (repeats > 1)
    {
        const std::string repeatsStr = ", SBRepeats=" + std::to_string(repeats);
        vrbSummaryStr += repeatsStr;
    }

    // Activations
    const unsigned activations = getMmeActivationNr();
    if (activations > 1)
    {
        const std::string activationsStr = ", activations=" + std::to_string(activations);
        vrbSummaryStr += activationsStr;
    }

    return vrbSummaryStr;
}

std::vector<std::string> MmeDescriptorGenerator::getRecipeDebugInfo(bool verbose) const
{
    std::vector<std::string> debugInfo;
    std::vector<unsigned> summaryStrIdx;

    if (m_convSubProblems.empty())
    {
        debugInfo = getRecipe().getRecipeDebugInfo(verbose);
        if (!debugInfo.empty())
        {
            debugInfo[0] = "MME Recipe: " + debugInfo[0];
            if (verbose)
            {
                debugInfo[0] += getVerboseRecipeSummaryStr();
            }

            if (verbose)
            {
                for (auto& debugStr : getBrainDebugInfo(getOriginalParams()))
                {
                    debugInfo.push_back(debugStr);
                }
            }
        }
        if (verbose)
        {
            summaryStrIdx.push_back(0);
        }
    }
    else
    {
        for (unsigned i = 0; i < m_convSubProblems.size(); i++)
        {
            if (verbose)
            {
                summaryStrIdx.push_back(debugInfo.size());
            }
            auto subDebugInfo = m_convSubProblems[i].recipe.getRecipeDebugInfo(verbose);
            MME_ASSERT(!subDebugInfo.empty(), "Invalid recipe debug info");
            const std::string idxStr = (m_convSubProblems.size() == 1) ? "" : (" " + std::to_string(i));
            subDebugInfo[0] = "MME Recipe" + idxStr + ": " + subDebugInfo[0];
            for (auto& debugStr : subDebugInfo)
            {
                debugInfo.push_back(debugStr);
            }
            for (auto& debugStr : getBrainDebugInfo(m_convSubProblems[i].params))
            {
                debugInfo.push_back(debugStr);
            }
        }
    }

    if (verbose)
    {
        for (unsigned idx : summaryStrIdx)
        {
            debugInfo[idx] += getVerboseRecipeSummaryStr();
        }
    }

    if (verbose)
    {
        debugInfo.push_back(getRecurringMisalignmentDebugInfo());
    }

    return debugInfo;
}

std::string MmeDescriptorGenerator::getRecurringMisalignmentDebugInfo() const
{
    std::string recurringMisalignmentDebugStr = RecurringMisalignmentOptimization::getDebugInfo(m_convSubProblems,
                                                                                                getGeoAttr(),
                                                                                                getMmeHal(e_mme_Gaudi2),
                                                                                                getOriginalParams());
    return recurringMisalignmentDebugStr;
}
}  // namespace Gaudi2
