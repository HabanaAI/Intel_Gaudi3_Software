#include "gaudi2/mme_descriptor_generator.h"
#include "gaudi3/mme_descriptor_generator.h"
#include "gaudi/mme_descriptor_generator.h"
#include "common_linear_ranges.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe.h"
#include <memory>
#include <optional>

using namespace MmeCommon;

template<typename Desc>
void CommonRoiCalculator<Desc>::createRoi(uint64_t addr,
                                          MmeActivation<Desc>& activation,
                                          const EMmeOperand operand,
                                          bool isSram,
                                          bool squashIORois) const
{
    OverlapRoi& roi = operand2Roi(activation, operand);
    roi.isSram = isSram;
    roi.offset = addr;
    roi.isLocalSignal = isSram;
    roi.isReduction = false;
    const MmeLayerParams& params =
        activation.paramsIdx.has_value() ? m_paramsVec->at(activation.paramsIdx.value()) : *m_params;
    EMmeInternalOperand internalOp = mmeOpToInternalOp(operand, params.opType);
    if (squashIORois)
    {
        //  Rois are squashed. no need to calculate what part of the tensor is accessed at each signal.
        //  Simply return the whole size of the tensor.
        if (params.getOperand(operand).isStrided())
        {
            addWholeStridedTensor(operand, activation, roi);
        }
        else
        {
            addWholeTensor(operand, activation, roi);
        }
    }
    else if (params.isGemmOperation() && params.controls.signalingMode == MmeCommon::e_mme_signaling_desc_with_store)
    {
        //  in Bgemm the input and output pattern are simple and can be inferred directly from the recipe splits.
        if (internalOp != MmeCommon::e_mme_op_c || isStoreEn(activation))
        //  skip if output is not stored (assumes that if first MME doesnt store none will)
        {
            addSplitTensor(operand, activation, roi);
        }
    }
    else
    {
        addSimulatedTensor(operand, activation, roi);
    }
}

template<typename Desc>
OverlapRoi& CommonRoiCalculator<Desc>::operand2Roi(MmeActivation<Desc>& activation, const EMmeOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_x:
            return activation.roiX;
            break;
        case e_mme_op_w:
            return activation.roiW;
            break;
        case e_mme_op_y:
            return activation.roiY;
            break;
        case e_mme_op_o:
            return activation.roiO;
            break;
    }
}

//  this funciton generate roi segments out of non contigoues tensor accesses according to sizes and bases.
template<typename Desc>
void CommonRoiCalculator<Desc>::addTensorSegments(MmeCommon::EMmeOperand operand,
                                                  MmeActivation<Desc>& act,
                                                  OverlapRoi& roi,
                                                  SizeArray sizes,
                                                  SizeArray bases) const
{
    unsigned stridedDim;
    MmeTensorView view = m_params->getOperand(operand, !act.isMask);
    uint64_t contiguousSize = sizes[0];
    for (stridedDim = 1; stridedDim < MME_MAX_TENSOR_DIMS; stridedDim++)
    {
        //  if the accumulated size is smaller then this dims stride there is a gap and the tensor is not contiguous
        if (contiguousSize != view.strides[stridedDim])
        {
            break;
        }
        //  if the previous dimension had a base offset then there is a gap and the tensor is not contiguous
        if (bases[stridedDim - 1] != 0)
        {
            break;
        }
        contiguousSize *= sizes[stridedDim];
    }

    OverlapSubRoi subRoi;
    subRoi.relSoIdx = 0;
    bool lastDimDone = false;
    SizeArray loopIdx = bases;
    unsigned dtSize = getElementSize(view.elementType);
    uint64_t start;
    uint64_t end;

    while (!lastDimDone)
    {
        //  set new start offset
        start = 0;
        for (unsigned dim = stridedDim - 1; dim < MME_MAX_TENSOR_DIMS; dim++)
        {
            start += (uint64_t) loopIdx[dim] * (uint64_t) view.strides[dim];
        }

        //  save a single contiguously accessed block
        end = start + contiguousSize;
        DataRange<uint64_t> dt(start * dtSize, end * dtSize);
#ifndef NDEBUG
        //  check the new range doesnt overlap with any previous range
        for (auto& range : subRoi.ranges)
        {
            //  this probably should be dropped for released versions as it might take a long time.
            MME_ASSERT_DEBUG_ONLY(!range.isOverlap(dt), "new ranges overlaps with previous range");
        }
#endif
        subRoi.ranges.push_back(dt);
        //  protection in case we get to this function while working on a full tensor, remove later.
        if (stridedDim == MME_MAX_TENSOR_DIMS)
        {
            lastDimDone = true;
            break;
        }
        //  advance dimensions
        for (unsigned dim = stridedDim; dim < MME_MAX_TENSOR_DIMS; dim++)
        {
            loopIdx[dim]++;
            if (loopIdx[dim] == sizes[dim] + bases[dim])
            {
                //  dimension finished, reset and advance the next dim in the next loop iteration
                loopIdx[dim] = bases[dim];
                //  if it was the last dim we are done
                if (dim == MME_MAX_TENSOR_DIMS - 1)
                {
                    lastDimDone = true;
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }
    roi.subRois->push_back(subRoi);
}

template<typename Desc>
void CommonRoiCalculator<Desc>::addSplitTensor(EMmeOperand operand, MmeActivation<Desc>& act, OverlapRoi& roi) const
{
    auto& strides = m_params->getOperand(operand, !act.isMask).strides;
    auto& spView = act.spView;
    auto& fcdView = act.fcdView;
    auto& nonSpView = act.nonSpatialView;

    const unsigned cdA = isTransposed(m_params->opType, MmeCommon::e_mme_in_a) ? 0 : 1;
    const unsigned cdB = isTransposed(m_params->opType, MmeCommon::e_mme_in_b) ? 0 : 1;
    const unsigned nonCdA = cdA == 1 ? 0 : 1;
    const unsigned nonCdB = cdB == 1 ? 0 : 1;
    SizeArray sizes = {0};
    SizeArray bases = {0};

    const SizeArray* sizesForRestDims = &nonSpView.sizes;
    const SizeArray* basesForRestDims = &nonSpView.bases;

    switch (operand)
    {
        case MmeCommon::e_mme_op_x:
            sizes[cdA] = nonSpView.sizes[cdB];
            sizes[nonCdA] = spView.viewOrigSize;
            bases[cdA] = nonSpView.bases[cdB];
            bases[nonCdA] = spView.viewBase;
            if (m_params->isGemmOperation() && (m_params->getOperand(e_mme_op_x, !act.isMask).sizes[2] == 1) &&
                (m_params->getOperand(e_mme_op_y).sizes[2] != 1))
            {
                auto& orgTensor = m_params->getOperand(e_mme_op_x, !act.isMask);
                sizesForRestDims = &orgTensor.sizes;
                basesForRestDims = &orgTensor.bases;
            }
            break;
        case MmeCommon::e_mme_op_w:
            sizes[nonCdB] = fcdView.viewOrigSize;
            sizes[cdB] = nonSpView.sizes[cdB];
            bases[nonCdB] = fcdView.viewBase;
            bases[cdB] = nonSpView.bases[cdB];
            if (m_params->isGemmOperation() && (m_params->getOperand(e_mme_op_w, !act.isMask).sizes[2] == 1) &&
                (m_params->getOperand(e_mme_op_y).sizes[2] != 1))
            {
                auto& orgTensor = m_params->getOperand(e_mme_op_w, !act.isMask);
                sizesForRestDims = &orgTensor.sizes;
                basesForRestDims = &orgTensor.bases;
            }
            break;
        case MmeCommon::e_mme_op_o:
        case MmeCommon::e_mme_op_y:
            sizes[0] = fcdView.viewOrigSize;
            sizes[1] = spView.viewOrigSize;
            bases[0] = fcdView.viewBase;
            bases[1] = spView.viewBase;
            break;
    }
    sizes[2] = (*sizesForRestDims)[2];
    sizes[3] = (*sizesForRestDims)[3];
    sizes[4] = (*sizesForRestDims)[4];
    bases[2] = (*basesForRestDims)[2];
    bases[3] = (*basesForRestDims)[3];
    bases[4] = (*basesForRestDims)[4];

    addTensorSegments(operand, act, roi, sizes, bases);
}

template<typename Desc>
const SkipData& CommonRoiCalculator<Desc>::getSkipDataForOperand(const EMmeOperand operand,
                                                                 const MmeActivation<Desc>& act,
                                                                 const EMmeOpType opType)
{
    switch (operand)
    {
        case e_mme_op_x:
            if (opType != e_mme_dedx && opType != e_mme_transposed_dedx)
            {
                return act.skipDataA;
            }
            else
            {
                return act.skipDataC;
            }
        case e_mme_op_w:
            if (!isDedwOperation(opType))
            {
                return act.skipDataB;
            }
            else
            {
                return act.skipDataC;
            }
        case e_mme_op_y:
            if (opType == e_mme_fwd || opType == e_mme_ab || opType == e_mme_atb || opType == e_mme_abt ||
                opType == e_mme_atbt || opType == e_mme_reductionAdd)
            {
                return act.skipDataC;
            }
            else
            {
                if (opType == e_mme_dedx || opType == e_mme_transposed_dedx)
                {
                    return act.skipDataA;
                }
                else
                {
                    MME_ASSERT(isDedwOperation(opType), "invalid operation");
                    return act.skipDataB;
                }
            }
        case e_mme_op_o:
            return act.skipDataC;
        default:
            MME_ASSERT(0, "invalid operand");
            return act.skipDataA;
    }
}

//  when a tensor is strided it is represented as a seires of contiguous blocks in memory
//  this funcion determines the size of a single contiguous block and the start offset of each block
//  and registers them in the overlap.
template<typename Desc>
void CommonRoiCalculator<Desc>::addWholeStridedTensor(EMmeOperand operand,
                                                      MmeActivation<Desc>& act,
                                                      OverlapRoi& roi) const
{
    const MmeCommon::MmeLayerParams* params =
        act.paramsIdx.has_value() ? &m_paramsVec->at(act.paramsIdx.value()) : m_params;
    MmeTensorView view = params->getOperand(operand, !act.isMask);
    addTensorSegments(operand, act, roi, view.sizes, view.bases);
}

template<typename Desc>
void CommonRoiCalculator<Desc>::addWholeTensor(EMmeOperand operand, MmeActivation<Desc>& act, OverlapRoi& roi) const
{
    uint64_t start = 0;
    uint64_t size =
        getOperandSizeInBytes(operand,
                              !act.isMask,
                              act.paramsIdx.has_value() ? m_paramsVec->at(act.paramsIdx.value()) : *m_params);
    uint64_t end = start + size;
    OverlapSubRoi subRoi;
    DataRange<uint64_t> dt(start, end);
    subRoi.ranges.push_back(dt);
    subRoi.relSoIdx = 0;
    roi.subRois->push_back(subRoi);
}

// use the original tensors from the params to get the full size
template<typename Desc>
uint64_t CommonRoiCalculator<Desc>::getOperandSizeInBytes(EMmeOperand operand,
                                                          bool primaryTensor,
                                                          const MmeLayerParams& params) const
{
    const MmeTensorView& view = params.getOperand(operand, primaryTensor);
    return (uint64_t) view.strides[MAX_DIMENSION - 1] * (uint64_t) view.sizes[MAX_DIMENSION - 1] *
           (uint64_t) getElementSize(view.elementType);
}

/*
 * simulate the tensor walk using the chip specific agu implementation
 * this should be used only for complex walks.
 * due to extremely long run time this should be avoided at all cost
 */
template<typename Desc>
void CommonRoiCalculator<Desc>::addSimulatedTensor(const EMmeOperand operand,
                                                   MmeActivation<Desc>& act,
                                                   OverlapRoi& roi) const
{
    MME_ASSERT(0, "not implemented yet for device");
}

template<typename Desc>
bool CommonRoiCalculator<Desc>::isStoreEn(MmeActivation<Desc>& act) const
{
    MME_ASSERT(0, "not implemented yet for device");
    return false;
}
namespace MmeCommon
{
// instantiate for all chips
template class CommonRoiCalculator<Gaudi2::Mme::Desc>;
template class CommonRoiCalculator<gaudi3::Mme::Desc>;
template class CommonRoiCalculator<Mme::gaudi::_Desc>;
} // namespace MmeCommon
