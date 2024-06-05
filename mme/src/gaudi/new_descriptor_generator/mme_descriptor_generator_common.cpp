#include "include/gaudi/new_descriptor_generator/mme_common.h"
#include "include/mme_common/mme_common_enum.h"
#include "src/gaudi/gaudi_linear_ranges.h"
#include <gaudi/mme_descriptor_generator.h>
#include "mme_assert.h"
#include "utils.h"

#define set_bf_to_all_ones(bf)  {(bf)=1; (bf)=-(bf);}

//#define MME_STACK_TRACE
#ifdef MME_STACK_TRACE
static unsigned __indent__ = 0;
#define TRACE_ENTER for (unsigned i=0; i<__indent__; i++) printf("  "); printf("Enter (%05u) - %s\n", __LINE__, __FUNCTION__); __indent__++
#define TRACE_EXIT __indent__--; for (unsigned i=0; i<__indent__; i++) printf("  ");printf("Exit  (%05u) - %s\n",  __LINE__, __FUNCTION__)
#else
#define TRACE_ENTER
#define TRACE_EXIT
#define TRACE_EXIT
#endif

using namespace MmeCommon;

namespace gaudi
{
associatedDimPattern getConfigPattern(MmeCommon::EMmePattern pattern)
{
    switch (pattern)
    {
        case MmeCommon::EMmePattern::e_mme_z_reduction_skf:
            return associatedDimPattern::pattern_z_skf;
        case MmeCommon::EMmePattern::e_mme_z_reduction_ksf:
            return associatedDimPattern::pattern_z_ksf;
        case MmeCommon::EMmePattern::e_mme_sp_reduction_kfc:
            return associatedDimPattern::pattern_sp_kfc;
        case MmeCommon::EMmePattern::e_mme_sp_reduction_kcf:
            return associatedDimPattern::pattern_sp_kcf;
        case MmeCommon::EMmePattern::e_mme_sp_reduction_fkc:
            return associatedDimPattern::pattern_sp_fkc;
        case MmeCommon::EMmePattern::e_mme_sp_reduction_fck:
            return associatedDimPattern::pattern_sp_fck;
        case MmeCommon::EMmePattern::e_mme_sp_reduction_cfk:
            return associatedDimPattern::pattern_sp_cfk;
        case MmeCommon::EMmePattern::e_mme_sp_reduction_ckf:
            return associatedDimPattern::pattern_sp_ckf;
        default:
            MME_ASSERT(0, "invalid walking pattern");
    }
    return associatedDimPattern::pattern_z_skf;
}

void setLoopDim(Mme::Desc* desc, EMmePattern pattern, unsigned loop, EPhysOperand operand, unsigned dim)
{
    associatedDimPattern configPattern = getConfigPattern(pattern);
    unsigned loopIdx = std::min(loop, (unsigned) LOOP_FILTER);
    Mme::MmeAssociatedDims* assocDim;
    unsigned descLoopIdx = (((uint8_t*) &configPattern)[loopIdx]) + (loop - loopIdx);
    if (descLoopIdx < Mme::c_mme_max_conv_dims)
    {
        assocDim = &desc->conv.associatedDims[descLoopIdx];
    }
    else
    {
        assocDim = &desc->outerLoop.associatedDims;
    }

    switch (operand)
    {
        case OP_S:
            assocDim->dimS = dim;
            break;
        case OP_L:
            assocDim->dimL = dim;
            break;
        case OP_O:
            assocDim->dimO = dim;
            break;
        default:
            MME_ASSERT(0, "invalid operand");
    }
}

void setLoopSize(Mme::Desc* desc, EMmePattern pattern, unsigned loop, unsigned sizeMinus1)
{
    associatedDimPattern configPattern = getConfigPattern(pattern);
    if (loop == LOOP_SPATIAL)
    {
        desc->numIterationsMinus1 = sizeMinus1;
    }
    else
    {
        unsigned loopIdx = std::min(loop, (unsigned) LOOP_FILTER);
        unsigned descLoopIdx = (((uint8_t*) &configPattern)[loopIdx]) + (loop - loopIdx);

        if (sizeMinus1 > std::numeric_limits<uint8_t>::max())
        {
            MME_ASSERT(0, "MME desc convolution loop overflow");
        }

        if (descLoopIdx < Mme::c_mme_max_conv_dims)
        {
            desc->conv.kernelSizeMinus1.dim[descLoopIdx] = sizeMinus1;
        }
        else
        {
            desc->outerLoop.sizeMinus1 = sizeMinus1;
        }
    }
}

unsigned getLoopSize(const Mme::Desc* desc, const EMmePattern pattern, const unsigned loop)
{
    associatedDimPattern configPattern = getConfigPattern(pattern);
    if (loop == LOOP_SPATIAL)
    {
        return desc->numIterationsMinus1;
    }
    else
    {
        unsigned loopIdx = std::min(loop, (unsigned) LOOP_FILTER);
        unsigned descLoopIdx = (((uint8_t*) &configPattern)[loopIdx]) + (loop - loopIdx);
        if (descLoopIdx < Mme::c_mme_max_conv_dims)
        {
            return desc->conv.kernelSizeMinus1.dim[descLoopIdx];
        }
        else
        {
            return desc->outerLoop.sizeMinus1;
        }
    }
}

unsigned getLoopMask(const EMmePattern pattern, const unsigned loop)
{
    associatedDimPattern configPattern = getConfigPattern(pattern);
    unsigned loopIdx = std::min(loop, (unsigned) LOOP_FILTER);
    MME_ASSERT(loopIdx != 0xff, "loop overflow");
    unsigned descLoopIdx = (((uint8_t*) &configPattern)[loopIdx]) + (loop - loopIdx);
    unsigned result;
    if (loop == LOOP_SPATIAL)
    {
        result = (1 << Mme::c_mme_max_conv_dims);
    }
    else if (descLoopIdx < Mme::c_mme_max_conv_dims)
    {
        result = (1 << descLoopIdx);
    }
    else
    {
        result = (2 << Mme::c_mme_max_conv_dims);
    }

    return result;
}

uint8_t getLoopFromLoopMask(EMmeLoopMask mask)
{
    return (mask + 1) >> 1;
}

Mme::EMmeDataType ConvertToGaudiDataType(EMmeDataType dt)
{
    switch (dt)
    {
        case e_type_bf16:
            return Mme::EMmeDataType::e_mme_dt_bf;
        case e_type_fp32:
            return Mme::EMmeDataType::e_mme_dt_sp;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return Mme::EMmeDataType::e_mme_dt_bf;
}

unsigned countSignals(const Mme::Desc *desc)
{
    TRACE_ENTER;
    unsigned ret;

    if (!desc->header.signalEn)
    {
        ret = 0;
    }
    else if (desc->header.accStoreIncDisable)
    {
        ret = 0;
    }
    else
    {
        ret = 1;

        for (int d = 0; d < Mme::c_mme_max_conv_dims; d++)
        {
            if ((desc->header.signalMask & (1 << d)) == 0)
            {
                ret *= desc->conv.kernelSizeMinus1.dim[d] + 1;
            }
        }

        if ((desc->header.signalMask & (1 << Mme::c_mme_max_conv_dims)) == 0)
        {
            ret *= desc->numIterationsMinus1 + 1;
        }

        if ((desc->header.signalMask & (1 << (Mme::c_mme_max_conv_dims + 1))) == 0)
        {
            ret *= desc->outerLoop.sizeMinus1 + 1;
        }
    }

    TRACE_EXIT;
    return ret;
}

void aguRanges2Rois(const unsigned signalBase, std::vector<AguRanges>* ranges, OverlapRoi& roi)
{
    auto& subRois = *roi.subRois;
    for (unsigned i=0; i<ranges->size(); i++)
    {
        OverlapSubRoi * currSubRoi = nullptr;
        AguRanges::const_iterator cbegin;
        AguRanges::const_iterator cend;
        (*ranges)[i].getCoveredSegments(0, UINT64_MAX, cbegin, cend);
        uint64_t prevStart = 0;
        uint64_t prevEnd = 0;
        for (auto it = cbegin; it != cend; ++it)
        {
            if (it->second.valid)
            {
                if (!currSubRoi)
                {
                    subRois.push_back(OverlapSubRoi());
                    currSubRoi = &subRois.back();
                    currSubRoi->relSoIdx = signalBase + i;
                }

                if (prevEnd != it->first)
                {
                    if (prevEnd != prevStart)
                    {
                        currSubRoi->ranges.push_back(DataRange<uint64_t>(prevStart, prevEnd));
                    }

                    prevStart = it->first;
                    prevEnd = it->first + it->second.size;
                }
                else
                {
                    prevEnd += it->second.size;

                }
            }
        }

        if (prevEnd != prevStart)
        {
            currSubRoi->ranges.push_back(DataRange<uint64_t>(prevStart, prevEnd));
        }
    }
}

void MmeDescriptorGenerator::params2OperandsViews(MmeTensorView* a, MmeTensorView* b, MmeTensorView* c) const
{
    const MmeLayerParams params = getOriginalParams();
    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_atbt:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
            *a = params.x;
            *b = params.w;
            *c = params.y;
            break;
        case e_mme_dedx:
            *a = params.y;
            *b = params.w;
            *c = params.x;
            break;
        case e_mme_dedw:
            *a = params.x;
            *b = params.y;
            *c = params.w;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }
}

// Raster walk refers to GEMM calculations order (of the output).
// Assume FCD (relative to output) has multiple views {fcdV1, fcdV2, ..., fcdVn}.
// We call the pattern raster when GEMM1 consumes fcdV1, GEMM2 consumes fcdV2,.... GEMMn consumes fcdVn.
// Otherwise we call it non-raster.
// (K) associated to FCD (relative to output).
// (C)(S) associated to SP (spatial, relative to output). (C) Used for DEDW. (S) used for FWD and DEDX.
// (F) associated to the three filters.
//
// Example 1 - CFK:
// (K) First scan FCD subview of filter 1 then move to second filter (F). Before moving to (C)
//
// Example 2 - CKF:
// (F) First scan first GEMM of filter 1, then move to first GEMM of filter 2...
// After scanning all filters, go back to first one and move to second GEMM at FCD direction (K)
// When complete scanning all GEMMS at first rows of all filters, move to second raw (C) and perform previous process.
//
// Example 3 - KCF:
// (F) First scan first GEMM of filter 1, then move to first GEMM of filter 2... (same as previous example).
// After scanning all filters, go back to first one and move to second GEMM at SP direction (C)
// When complete scanning all GEMMS at first columns of all filters, move to second column (K) and repeat as before.
bool MmeDescriptorGenerator::isPatternRaster(const EMmePattern pattern) const
{
    switch (pattern)
    {
        case e_mme_z_reduction_skf:
        case e_mme_sp_reduction_fck:
        case e_mme_sp_reduction_cfk:
            return true;
        case e_mme_sp_reduction_ckf:
        case e_mme_sp_reduction_kfc:
        case e_mme_sp_reduction_fkc:
        case e_mme_sp_reduction_kcf:
        case e_mme_z_reduction_ksf:
            return false;
        default:
            MME_ASSERT(0, "invalid walking pattern");
            return false;
    }
}

//============== Set the performance signals =======================
EPhysOperand mmeOperand2PhysOperand(const EMmeOperand userOperand, const EMmeOpType opType, const bool transposed)
{
    switch ( opType )
    {
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_abt:
        case e_mme_atb:
        case e_mme_atbt:
            if (userOperand == EMmeOperand::e_mme_op_y) return OP_O;
            else if (userOperand == EMmeOperand::e_mme_op_x)
                return (transposed ? OP_L : OP_S);
            else return (transposed ? OP_S : OP_L);
            break;
        case e_mme_dedx:
            if (userOperand == EMmeOperand::e_mme_op_x) return OP_O;
            else if (userOperand == EMmeOperand::e_mme_op_y)
                return (transposed ? OP_L : OP_S);
            else return (transposed ? OP_S : OP_L);
            break;
        case e_mme_dedw:
            if (userOperand == EMmeOperand::e_mme_op_w) return OP_O;
            else if (userOperand == EMmeOperand::e_mme_op_x)
                return (transposed ? OP_L : OP_S);
            else return (transposed ? OP_S : OP_L);
            break;
        default:
            return (transposed ? OP_S : OP_L);
    }
}

void setOpStartAndEndEvents(const MmeLayerParams* params,
                            std::list<MmeActivation>& activations,
                            const EMmeOperand userOperand)
{
    TRACE_ENTER;
    Mme::MmePerfEvt evt;
    evt.incMask = 1;
    set_bf_to_all_ones(evt.loopMask);
    evt.rst = 1;
    evt.value = params->tracing.ctxId;
    for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
    {
        evt.startEndMask = (activations.size() == 1) ? 0x0 : 0x2;
        Mme::Desc *desc;
        desc = &(activations.front().getDesc(masterIdx));
        EPhysOperand operand = mmeOperand2PhysOperand(userOperand, params->opType, desc->header.transO);
        if (operand == OP_S)
        {
            desc->perfEvtS.dw = evt.dw;
        }
        else if (operand == OP_L)
        {
            desc->perfEvtL[Mme::e_mme_local].dw = evt.dw;
            desc->perfEvtL[Mme::e_mme_remote].dw = evt.dw;
        }
        else
        {
            desc->perfEvtO[Mme::e_mme_remote].dw = evt.dw;
            desc->perfEvtO[Mme::e_mme_local].dw = evt.dw;
        }
        if (activations.size() != 1)
        {
            desc = &(activations.back().getDesc(masterIdx));
            evt.startEndMask = 1; // mask start
            if (operand == OP_S)
            {
                desc->perfEvtS.dw = evt.dw;
            }
            else if (operand == OP_L)
            {
                desc->perfEvtL[Mme::e_mme_local].dw = evt.dw;
                desc->perfEvtL[Mme::e_mme_remote].dw = evt.dw;
            }
            else
            {
                desc->perfEvtO[Mme::e_mme_remote].dw = evt.dw;
                desc->perfEvtO[Mme::e_mme_local].dw = evt.dw;
            }
        }
    }
    TRACE_EXIT;
}

void spPosToCoord(const unsigned spPos, const unsigned* spSizes, unsigned* spCoord)
{
    unsigned rem = spPos;
    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        spCoord[dim] = rem % spSizes[dim];
        rem /= spSizes[dim];
    }

    if (rem)
    {
        spCoord[Mme::c_mme_max_tensor_dims - 2] = spSizes[Mme::c_mme_max_tensor_dims - 2];
    }
}

const MmeRoi MmeDescriptorGenerator::getRoi()
{
    MmeRoi roi;
    auto recipe = getRecipe();

    roi.denseBase = recipe.curFcd().viewBase;
    roi.spBase = recipe.curSp().viewBase;
    roi.spSize = recipe.curSp().viewSize;

    const MmeLayerParams params = getOriginalParams();
    auto operand = (params.opType == MmeCommon::e_mme_dedw) ? MmeCommon::e_mme_op_b : MmeCommon::e_mme_op_c;
    MmeCommon::SizeArray roiSize = recipe.getRoiSizes(operand);
    memcpy(roi.size, &roiSize, sizeof(roi.size));

    return roi;
}

unsigned spCoordToPos(const unsigned* spCoord, const unsigned* spSizes)
{
    unsigned ret = 0;
    unsigned factor = 1;
    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        ret += spCoord[dim] * factor;
        factor *= spSizes[dim];
    }

    return ret;
}

void padRoi(const unsigned spatialAGUs, const MmeRoi* roiIn, MmeRoi* roiOut)
{
    *roiOut = *roiIn;
    if (roiOut->size[1] < spatialAGUs)
    {
        unsigned spBaseCoord[Mme::c_mme_max_tensor_dims - 1];
        unsigned spEndCoord[Mme::c_mme_max_tensor_dims - 1];
        spPosToCoord(roiIn->spBase, &roiIn->size[1], spBaseCoord);
        spPosToCoord(roiIn->spBase + roiIn->spSize, &roiIn->size[1], spEndCoord);

        roiOut->size[1] = spatialAGUs;
        roiOut->spBase = spCoordToPos(spBaseCoord, &roiOut->size[1]);
        roiOut->spSize = spCoordToPos(spEndCoord, &roiOut->size[1]) - roiOut->spBase;
    }
}

unsigned MmeDescriptorGenerator::countTetrises(const Mme::Desc& desc)
{
    unsigned ret = 1;
    uint16_t mask = desc.header.accumMask;

    for (int d = 0; d < Mme::c_mme_max_conv_dims; d++)
    {
        if ((mask & (1 << d)) == 0)
        {
            ret *= desc.conv.kernelSizeMinus1.dim[d] + 1;
        }
    }

    if ((mask & ((Mme::e_mme_tetris_loop + 1) >> 1)) == 0)
    {
        ret *= desc.numIterationsMinus1 + 1;
    }

    if ((mask & ((Mme::e_mme_outer_loop + 1) >> 1)) == 0)
    {
        ret *= desc.outerLoop.sizeMinus1 + 1;
    }

    return ret;
}

void MmeDescriptorGenerator::activation2Roi(const ExecuteParams& params)
{
    TRACE_ENTER;
    auto roiCalc = getRoiCalculator();
    MmeActivation *firstOutputAct = nullptr;
    bool convOp = ((params.opType == e_mme_fwd) || (params.opType == e_mme_dedx) || (params.opType == e_mme_dedw));
    for (unsigned actIdx = 0; actIdx < m_activations.size(); actIdx++)
    {
        auto it = std::next(m_activations.begin(), actIdx);
        MmeActivation& act = *it;
        const Mme::Desc& desc = act.getDesc(0);
        if (act.getDesc(0).header.storeEn)
        {
            MME_ASSERT(act.getDesc(1).header.storeEn, "both north & south descs should have storeEn.");
            if (!firstOutputAct)
            {
                firstOutputAct = &act;
            }
        }
        if (!(convOp && (params.controls->squashIORois) && (actIdx != (m_activations.size() - 1))))
        {
            MmeActivation& inputRoiAct = convOp && params.controls->squashIORois ? m_activations.front() : act;
            MmeActivation& outputRoiAct = convOp && params.controls->squashIORois ? *firstOutputAct : act;
            if (params.controls->squashIORois ||
                (!desc.sw.swMemsetFwd && !desc.sw.swMemsetDedx && !desc.sw.swMemsetDedw))
            {
                generateRoi(inputRoiAct, e_mme_op_a, params.opType, roiCalc, params.controls->squashIORois);
                generateRoi(inputRoiAct, e_mme_op_b, params.opType, roiCalc, params.controls->squashIORois);
            }
            generateRoi(outputRoiAct, e_mme_op_c, params.opType, roiCalc, params.controls->squashIORois);
        }
    }
    TRACE_EXIT;
}

void generateRoi(MmeActivation& act,
                 EMmeInternalOperand internalOperand,
                 EMmeOpType opType,
                 std::shared_ptr<CommonRoiCalculator<Mme::Desc>> roiCalc,
                 bool squashIORois)
{
    uint32_t* addrHigh;
    uint32_t* addrLow;
    ptrToInt addr;
    EMmeOperand operand;
    operand = internalOpToMmeOp(internalOperand, opType);
    getTensorAddressFields(operand, act.getDesc(0), &addrHigh, &addrLow, act.isGemm);
    addr.u32[0] = *addrLow;
    addr.u32[1] = *addrHigh;
    roiCalc->createRoi(addr.u64, act, operand, false, squashIORois);
}

//#####################################################

std::unique_ptr<MmeDescriptorGenerator> MmeDescriptorGenerator::createMmeDescGenerator(const MmeLayerParams& params)
{
    if (params.isGemmOperation() || isDedwAsBgemm(params) || isZeroCD(params))
    {
        return std::make_unique<MmeBgemmDescriptorGenerator>(params);
    }
    return MmeConvDescriptorGenerator::createMmeConvDescGenerator(params);
}

unsigned getDataTypeShiftAmount(MmeCommon::EMmeDataType dt)
{
    switch (dt)
    {
        case MmeCommon::e_type_fp32:
            return 1;
        case MmeCommon::e_type_bf16:
            return 0;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return 0;
}

} // namespace gaudi
