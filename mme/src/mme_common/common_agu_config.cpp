#include "common_agu_config.h"
#include "general_utils.h"
#include "mme_assert.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/conv_sub_problems.h"
namespace MmeCommon
{
unsigned CommonAguConfig::fixForTeAcceleration(EMmeInternalOperand operand, unsigned size, bool isSpatial)
{
    //  TE acceleration on input makes the MME work on less row in each activation so
    //  when accelerating we need to jump less rows ahead to account for it
    //  this is relevant only for input since the output doesnt make spatial loop strides.
    if (m_recipe.acceleratedOperand == operand)
    {
        if (isSpatial)
            size >>= m_recipe.teAcceleration;
        else
            size <<= m_recipe.teAcceleration;
    }
    return size;
}
unsigned CommonAguConfig::getPaddedCommonDim(const unsigned originalCommonDim, const EMmeDataType dataType)
{
    unsigned dtAlignment = m_mmeHal.getNumElementsForCommonDimAlignment(dataType, m_params.opType);
    unsigned coreInterleavedReadersA =
        isTransposed(m_params.opType, e_mme_in_a) ? 1 : m_geoAttr.getCoreSpatialEuPort(e_mme_op_a);
    unsigned coreInterleavedReadersB =
        isTransposed(m_params.opType, e_mme_in_b) ? 1 : m_geoAttr.getCoreSpatialEuPort(e_mme_op_b);
    unsigned coreInterleavedReaders = std::max(coreInterleavedReadersA, coreInterleavedReadersB);
    //  every reader must be aligned by itself, a single reader requirements are the max between the number
    //  of interleaving ports it has and the DT requirements
    unsigned coreAlignment = std::max(coreInterleavedReaders, dtAlignment);
    //  need to make sure all concurrent cores are aligned to the single core requirements,
    unsigned alignedSize = coreAlignment * m_geoAttr.getGeometryCdConcurrency();
    return ((originalCommonDim + alignedSize - 1) / alignedSize) * alignedSize;
}

void CommonAguConfig::config(void* descPtr)
{
    //  configure global tensor parameters
    switch (m_params.opType)
    {
        case e_mme_gemm_transpose:
            configureGemmTranspose();
            setAssociatedDimsDma(descPtr);
            break;
        case e_mme_memcpy:
            configureDma();
            setAssociatedDimsDma(descPtr);
            break;
        case e_mme_trans:
            configureTranspose();
            setAssociatedDimsDma(descPtr);
            break;
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            configureBgemm();
            setAssociatedDimsBgemmDedw(descPtr);
            break;
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            configureDedw();
            setAssociatedDimsBgemmDedw(descPtr);
            break;
        case e_mme_fwd:
            configureFwd();
            setAssociatedDimsFwdDedx(descPtr);
            break;
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            configureDedx();
            setAssociatedDimsFwdDedx(descPtr);
            break;
        default:
            MME_ASSERT(0, "operation not supported");
    }

    for (auto operand : m_geoAttr.getOperands())
    {
        //  configure port offsets
        setPortOffsets(operand);
        //  multiply strides according to number of readers
        multiplyStrides(operand);
        //  set final HW values for all fields that need to be multiplied by the stride
        finalizeSizes(operand);
    }

    //  write the HW descriptor
    configureDescriptor(descPtr);
}

void CommonAguConfig::configureDma()
{
    for (auto operand : m_geoAttr.getOperands())
    {
        TensorAttr& tensor = getTensor(operand);
        const MmeTensorView& tensorView = m_recipe.getOperand(operand);
        const SizeArray& roiSizes = m_recipe.getRoiSizes(operand);

        tensor.spatialStrides[0] = 1;
        tensor.baseOffset[0] = 0;
        tensor.startOffset[0] = 0;
        tensor.roiSize[0] = roiSizes[0];
        tensor.validElements[0] = tensorView.sizes[0];
        tensor.loopStride[0] = 0;  // no fcd steps

        for (unsigned dim = 1; dim < MME_MAX_TENSOR_DIMS; ++dim)
        {
            tensor.spatialStrides[dim] = 1;
            tensor.baseOffset[dim] = 0;
            tensor.startOffset[dim] = 0;

            tensor.roiSize[dim] = roiSizes[dim];
            tensor.validElements[dim] = tensorView.sizes[dim];
            tensor.loopStride[dim] = 1;
        }

        // in memcpy we dont use a loop for the fcd steps, so we are always in the last FCD step.
        tensor.lastFcdStep = getSinglePortFcdSize(operand);
        tensor.lastSpatialStep = calcLastStepSize(operand, true, m_params.spSize);
    }
}

void CommonAguConfig::configureTranspose()
{
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();
    //  pad CD to TE constrains, this needs to move to the recipe
    unsigned paddedFcdA = getPaddedCommonDim(tensorViewA.sizes[0], m_params.x.elementType);
    unsigned paddedSpC = getPaddedCommonDim(tensorViewC.sizes[1], m_params.y.elementType);
    // a bit unintuitive but if the output is accelerated than the fcdView given by the recipe is accelerated
    // and it needs to be deaccelerated to get the original sliced input SP size
    unsigned originalSpA = fixForTeAcceleration(e_mme_op_c, fcdView.viewSize);  // unaccelerated A spatial size

    unsigned effectiveAMmeWidthA = fixForTeAcceleration(e_mme_op_a, m_geoAttr.getGeometryWidth());
    unsigned effectiveAMmeHeightA = fixForTeAcceleration(e_mme_op_a, m_geoAttr.getGeometryHeight(), false);
    unsigned effectiveAMmeWidthC = fixForTeAcceleration(e_mme_op_c, m_geoAttr.getGeometryWidth(), false);
    unsigned effectiveAMmeHeightC = fixForTeAcceleration(e_mme_op_c, m_geoAttr.getGeometryHeight());

    unsigned singlePortFcdA = fixForTeAcceleration(e_mme_op_a, m_geoAttr.getPortSize(e_mme_op_a), false);
    unsigned singlePortFcdC = fixForTeAcceleration(e_mme_op_c, m_geoAttr.getPortSize(e_mme_op_c));

    // input
    tensorA.spatialStrides[0] = 1;
    tensorA.roiSize[0] = paddedFcdA;
    tensorA.validElements[0] = tensorViewA.sizes[0];
    tensorA.loopStride[0] = effectiveAMmeHeightA;
    tensorA.startOffset[0] = fixForTeAcceleration(e_mme_op_c, spView.viewBase, false);

    tensorA.spatialStrides[1] = 1;
    tensorA.roiSize[1] = tensorViewA.sizes[1];
    tensorA.validElements[1] = tensorViewA.sizes[1];
    //  though we advance spatially the amount of row A needs to advance is according to the total width of the output.
    tensorA.loopStride[1] = effectiveAMmeWidthA;
    tensorA.baseOffset[1] = fcdView.viewBase >> m_recipe.teAcceleration;

    tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, false, originalSpA % m_geoAttr.getGeometryWidth());
    tensorA.lastFcdStep = std::min(paddedFcdA, singlePortFcdA);

    //  output
    tensorC.spatialStrides[0] = 1;
    tensorC.roiSize[0] = tensorViewC.sizes[0];
    tensorC.validElements[0] = tensorViewC.sizes[0];
    tensorC.loopStride[0] = effectiveAMmeWidthC;
    tensorC.baseOffset[0] = fcdView.viewBase;

    tensorC.spatialStrides[1] = 1;
    tensorC.roiSize[1] = paddedSpC;
    tensorC.validElements[1] = tensorViewC.sizes[1];
    tensorC.loopStride[1] = effectiveAMmeHeightC;
    tensorC.startOffset[1] = spView.viewBase;

    tensorC.lastSpatialStep = std::min(paddedSpC, singlePortFcdC);
    tensorC.lastFcdStep = calcLastStepSize(e_mme_op_c, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());

    //  configure upper dims
    for (auto operand : {e_mme_op_a, e_mme_op_c})
    {
        TensorAttr& tensor = getTensor(operand);
        const MmeTensorView& tensorView = m_params.getOperand(operand);
        const auto& batchView = m_recipe.curNonSpatial();
        for (unsigned dim = 2; dim < 5; dim++)
        {
            tensor.spatialStrides[dim] = 1;
            tensor.roiSize[dim] = tensorView.sizes[dim];
            tensor.validElements[dim] = tensorView.sizes[dim];
            tensor.baseOffset[dim] = batchView.bases[dim];
            tensor.loopStride[dim] = 1;
        }
    }
}

void CommonAguConfig::configureUnitMatrix()
{
    tensorB.baseOffset[DIM_K] = 0;
    tensorB.startOffset[DIM_K] = 0;
    tensorB.validElements[DIM_K] = m_geoAttr.getMmeWidth();
    tensorB.roiSize[DIM_K] = m_geoAttr.getMmeWidth();  // fill the EU width with zeroes
    tensorB.spatialStrides[DIM_K] = 0;
    tensorB.loopStride[DIM_K] = 0;

    tensorB.baseOffset[WEIGHT_DIM_C] = 0;
    tensorB.startOffset[WEIGHT_DIM_C] = 0;
    tensorB.validElements[WEIGHT_DIM_C] = 1; // only a single valid element use to create the unit matrix
    tensorB.roiSize[WEIGHT_DIM_C] = m_geoAttr.getMmeWidth();
    tensorB.spatialStrides[WEIGHT_DIM_C] = 1;
    tensorB.loopStride[WEIGHT_DIM_C] = -1 * m_geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b);

    tensorB.lastSpatialStep = 1;
    // last FCD step can be reduced for A/B to slightly reduce power
    tensorB.lastFcdStep = m_geoAttr.getPortSize(e_mme_op_b);

    // configure the higher dims, should be all 1s
    configureBatchLoops(e_mme_op_b);
}

void CommonAguConfig::configureGemmTranspose()
{
    configureBgemmNonTransposedA();
    configureUnitMatrix();
    configureBgemmOutput();
}

void CommonAguConfig::configureBgemm()
{
    if (m_geoAttr.isTransposed(e_mme_op_a))
    {
        configureBgemmTransposedA();
    }
    else
    {
        configureBgemmNonTransposedA();
    }
    if (m_geoAttr.isTransposed(e_mme_op_b))
    {
        configureBgemmTransposedB();
    }
    else
    {
        configureBgemmNonTransposedB();
    }

    configureBgemmOutput();
}
void CommonAguConfig::configureNonConvDimsNonTransposedB(unsigned paddedCommonDim)
{
    const auto& convView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();

    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);

    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);

    tensorB.roiSize[DIM_K] = bRoiSizes[DIM_K];
    tensorB.validElements[DIM_K] = tensorViewB.sizes[DIM_K];
    tensorB.loopStride[DIM_K] = m_geoAttr.getGeometryWidth();
    tensorB.spatialStrides[DIM_K] = 1;  // FCD dim doenst have a spatial stride
    tensorB.startOffset[DIM_K] = fcdView.viewBase;
    tensorB.baseOffset[DIM_K] = 0;

    tensorB.spatialStrides[WEIGHT_DIM_C] = 1;
    tensorB.roiSize[WEIGHT_DIM_C] = paddedCommonDim;
    tensorB.validElements[WEIGHT_DIM_C] = isMemsetDesc() ? 0 : tensorViewB.sizes[WEIGHT_DIM_C];
    tensorB.loopStride[WEIGHT_DIM_C] = 0;
    tensorB.startOffset[WEIGHT_DIM_C] = 0;
    tensorB.baseOffset[WEIGHT_DIM_C] = convView.bases[WEIGHT_DIM_C];

    tensorB.lastSpatialStep = calcLastStepSize(e_mme_op_b, true, paddedCommonDim);
    tensorB.lastFcdStep = m_geoAttr.getPortSize(e_mme_op_b);
}

void CommonAguConfig::configureNonConvDimsTransposedB(unsigned paddedCommonDim)
{
    const auto& convView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();

    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);

    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);

    tensorB.roiSize[DIM_K] = paddedCommonDim;
    tensorB.validElements[DIM_K] = isMemsetDesc() ? 0 : tensorViewB.sizes[DIM_K];
    tensorB.loopStride[DIM_K] = 0;
    tensorB.spatialStrides[DIM_K] = 1;  // FCD dim doenst have a spatial stride
    tensorB.baseOffset[DIM_K] = convView.bases[DIM_K];
    tensorB.startOffset[DIM_K] = 0;

    tensorB.spatialStrides[WEIGHT_DIM_C] = 1;
    tensorB.roiSize[WEIGHT_DIM_C] = bRoiSizes[WEIGHT_DIM_C];
    tensorB.validElements[WEIGHT_DIM_C] = tensorViewB.sizes[WEIGHT_DIM_C];
    tensorB.loopStride[WEIGHT_DIM_C] = m_geoAttr.getGeometryWidth();
    tensorB.baseOffset[WEIGHT_DIM_C] = 0;
    tensorB.startOffset[WEIGHT_DIM_C] = fcdView.viewBase;

    tensorB.lastSpatialStep = calcLastStepSize(e_mme_op_b, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());
    tensorB.lastFcdStep = paddedCommonDim;
}
void CommonAguConfig::configureDedx()
{
    const auto& convView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();

    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);

    unsigned originalCommonDim = convView.sizes[m_geoAttr.isTransposed(e_mme_op_b) ? DIM_K : WEIGHT_DIM_C];
    // when stride is larger then the kernel - there are void pixels in dX -
    // pixels where the gradient is 0.
    // memset void pixels optimization sets the void pixel value to 0, otherwise they are skipped.
    unsigned paddedCommonDim = getPaddedCommonDim(isMemsetDesc() ? 1 : originalCommonDim, m_params.w.elementType);

    // tensor A - DY - BHWxK - non - conv dims
    tensorA.roiSize[DIM_K] = paddedCommonDim;
    tensorA.validElements[DIM_K] = isMemsetDesc() ? 0 : tensorViewA.sizes[DIM_K];
    tensorA.loopStride[DIM_K] = 0;
    tensorA.spatialStrides[DIM_K] = 0;
    tensorA.baseOffset[DIM_K] = convView.bases[m_geoAttr.isTransposed(e_mme_op_b) ? DIM_K : WEIGHT_DIM_C];
    tensorA.startOffset[DIM_K] = 0;

    // tensor B non-conv dims
    if (m_geoAttr.isTransposed(e_mme_op_b))
    {
        configureNonConvDimsTransposedB(paddedCommonDim);
    }
    else
    {
        configureNonConvDimsNonTransposedB(paddedCommonDim);
    }

    // tensor C - non-conv dims
    tensorC.roiSize[DIM_C] = cRoiSizes[DIM_C];
    tensorC.validElements[DIM_C] = tensorViewC.sizes[DIM_C];
    tensorC.loopStride[DIM_C] = m_geoAttr.getGeometryWidth();
    tensorC.spatialStrides[DIM_C] = 0;
    tensorC.baseOffset[DIM_C] = 0;
    tensorC.startOffset[DIM_C] = fcdView.viewBase;

    //  set conv dimensions
    SizeArray spPos = m_recipe.calcSpPos(spView.viewBase);
    for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; convDim++)
    {
        const unsigned actDim = convDim + 1;
        const unsigned weightDim = convDim + 2;

        // tensor A - startOffset & RoiBaseOffset are swapped for FWD,DEDX.
        tensorA.spatialStrides[actDim] = m_params.conv.stride[convDim];
        unsigned convSize = (m_recipe.lowering && isFilterDimReversed((MmeDimsIndex) weightDim))
                                ? m_params.getOperand(e_mme_op_b).sizes[weightDim]
                                : convView.sizes[weightDim];
        tensorA.baseOffset[actDim] = -(convSize - 1 + convView.bases[weightDim]);
        tensorA.baseOffset[actDim] *= m_params.conv.dilation[convDim];
        tensorA.baseOffset[actDim] += m_params.conv.padding[convDim];
        tensorA.roiSize[actDim] = cRoiSizes[actDim] * m_params.conv.stride[convDim];
        tensorA.validElements[actDim] = tensorViewA.sizes[actDim];
        tensorA.loopStride[actDim] = m_params.conv.dilation[convDim];
        tensorA.startOffset[actDim] = spPos[convDim] * m_params.conv.stride[convDim];

        tensorC.spatialStrides[actDim] = 1;
        tensorC.baseOffset[actDim] = 0;
        tensorC.roiSize[actDim] = cRoiSizes[actDim];
        tensorC.validElements[actDim] = tensorViewC.sizes[actDim];
        tensorC.loopStride[actDim] = 1;
        tensorC.startOffset[actDim] = spPos[convDim];

        tensorB.spatialStrides[weightDim] = 0;
        tensorB.baseOffset[weightDim] = 0;
        tensorB.roiSize[weightDim] = bRoiSizes[weightDim];
        tensorB.validElements[weightDim] = tensorViewB.sizes[weightDim];
        if (isFilterDimReversed((MmeDimsIndex) weightDim))
        {
            tensorB.loopStride[weightDim] = 1;
            tensorB.startOffset[weightDim] = convView.bases[weightDim];
        }
        else
        {
            tensorB.loopStride[weightDim] = -1;
            // we calculates the weights backwards on the conv dims.
            tensorB.startOffset[weightDim] = (convView.bases[weightDim] + convView.sizes[weightDim] - 1);
        }
    }
    // tensor A - Batch dim
    tensorA.spatialStrides[DIM_B] = 1;
    tensorA.baseOffset[DIM_B] = 0;
    tensorA.startOffset[DIM_B] = spPos[DIM_B - 1];
    tensorA.roiSize[DIM_B] = aRoiSizes[DIM_B];
    tensorA.validElements[DIM_B] = tensorViewA.sizes[DIM_B];
    tensorA.loopStride[DIM_B] = 0;

    // tensor C - Batch dim
    tensorC.spatialStrides[DIM_B] = 1;
    tensorC.baseOffset[DIM_B] = 0;
    tensorC.startOffset[DIM_B] = spPos[DIM_B - 1];
    tensorC.roiSize[DIM_B] = cRoiSizes[DIM_B];
    tensorC.validElements[DIM_B] = tensorViewC.sizes[DIM_B];
    tensorC.loopStride[DIM_B] = 1;

    tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, true, spView.viewSize % m_geoAttr.getGeometryHeight());
    tensorA.lastFcdStep = paddedCommonDim;

    tensorC.lastSpatialStep = calcLastStepSize(e_mme_op_c, true, spView.viewSize % m_geoAttr.getGeometryHeight());
    tensorC.lastFcdStep = calcLastStepSize(e_mme_op_c, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());
}

bool CommonAguConfig::isFilterDimReversed(MmeDimsIndex dim) const
{
    return (m_params.opType == e_mme_transposed_dedx && dim == DIM_S);
}
void CommonAguConfig::configureFwd()
{
    const auto& convView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();

    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);

    bool isLastPartialInFCD = (convView.sizes[WEIGHT_DIM_C] + convView.bases[WEIGHT_DIM_C]) == tensorViewA.sizes[DIM_C];
    unsigned originalCommonDim = convView.sizes[WEIGHT_DIM_C];
    unsigned paddedCommonDim = getPaddedCommonDim(originalCommonDim, m_params.w.elementType);
    if (!isLastPartialInFCD)
    {
        //  partials in the middle of the FCD cant be aligned, because we cant pad in the middle of the FCD.
        //  so their size has to be already aligned.
        MME_ASSERT(paddedCommonDim == originalCommonDim, "subView size isnt aligned to interleaving ports");
    }

    // tensor A - non - conv dims
    tensorA.roiSize[DIM_C] = aRoiSizes[DIM_C];
    tensorA.validElements[DIM_C] = tensorViewA.sizes[DIM_C];
    tensorA.loopStride[DIM_C] = 0;
    tensorA.spatialStrides[DIM_C] = 1;
    tensorA.startOffset[DIM_C] = 0;
    tensorA.baseOffset[DIM_C] = convView.bases[WEIGHT_DIM_C];

    // tensor B non-conv dims
    configureNonConvDimsNonTransposedB(paddedCommonDim);

    // tensor C - non-conv dims
    tensorC.roiSize[DIM_K] = cRoiSizes[DIM_K];
    tensorC.validElements[DIM_K] = tensorViewC.sizes[DIM_K];
    tensorC.loopStride[DIM_K] = m_geoAttr.getGeometryWidth();
    tensorC.startOffset[DIM_K] = fcdView.viewBase;
    tensorC.baseOffset[DIM_K] = 0;

    //  set conv dimensions
    SizeArray spPos = m_recipe.calcSpPos(spView.viewBase);
    for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; convDim++)
    {
        const unsigned actDim = convDim + 1;
        const unsigned weightDim = convDim + 2;

        // tensor A - startOffset & RoiBaseOffset are swapped for FWD,DEDX.
        tensorA.spatialStrides[actDim] = m_params.conv.stride[convDim];
        tensorA.startOffset[actDim] = spPos[actDim - 1] * m_params.conv.stride[convDim];
        tensorA.roiSize[actDim] = cRoiSizes[actDim] * m_params.conv.stride[convDim];
        tensorA.validElements[actDim] = tensorViewA.sizes[actDim];
        tensorA.loopStride[actDim] = m_params.conv.dilation[convDim];
        tensorA.baseOffset[actDim] =
            ((m_params.conv.dilation[convDim] * convView.bases[weightDim]) - m_params.conv.padding[convDim]);

        tensorC.spatialStrides[actDim] = 1;
        tensorC.startOffset[actDim] = spPos[actDim - 1];
        tensorC.baseOffset[actDim] = 0;
        tensorC.roiSize[actDim] = cRoiSizes[actDim];
        tensorC.validElements[actDim] = tensorViewC.sizes[actDim];
        tensorC.loopStride[actDim] = 0;

        tensorB.spatialStrides[weightDim] = 1;
        tensorB.startOffset[weightDim] = 0;
        tensorB.roiSize[weightDim] = bRoiSizes[weightDim];
        tensorB.validElements[weightDim] = tensorViewB.sizes[weightDim];
        tensorB.loopStride[weightDim] = 1;
        tensorB.baseOffset[weightDim] = convView.bases[weightDim];
    }

    // tensor A - Batch dim
    tensorA.spatialStrides[DIM_B] = 1;
    tensorA.startOffset[DIM_B] = spPos[DIM_B - 1];
    tensorA.baseOffset[DIM_B] = 0;
    tensorA.roiSize[DIM_B] = aRoiSizes[DIM_B];
    tensorA.validElements[DIM_B] = tensorViewA.sizes[DIM_B];
    tensorA.loopStride[DIM_B] = 0;

    // tensor C - Batch dim
    tensorC.spatialStrides[DIM_B] = 1;
    tensorC.startOffset[DIM_B] = spPos[DIM_B - 1];
    tensorC.baseOffset[DIM_B] = 0;
    tensorC.roiSize[DIM_B] = cRoiSizes[DIM_B];
    tensorC.validElements[DIM_B] = tensorViewC.sizes[DIM_B];
    tensorC.loopStride[DIM_B] = 0;

    tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, true, spView.viewSize % m_geoAttr.getGeometryHeight());
    tensorA.lastFcdStep = paddedCommonDim;

    tensorC.lastSpatialStep = calcLastStepSize(e_mme_op_c, true, spView.viewSize % m_geoAttr.getGeometryHeight());
    tensorC.lastFcdStep = calcLastStepSize(e_mme_op_c, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());
}

void CommonAguConfig::configureDedw()
{
    const auto& convView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();

    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);

    // tensor A - non - conv dims
    tensorA.roiSize[DIM_C] = aRoiSizes[DIM_C];
    tensorA.validElements[DIM_C] = tensorViewA.sizes[DIM_C];
    tensorA.loopStride[DIM_C] = m_geoAttr.getGeometryHeight();
    tensorA.startOffset[DIM_C] = convView.bases[1];

    // tensor B non-conv dims
    tensorB.roiSize[DIM_K] = bRoiSizes[DIM_K];
    tensorB.validElements[DIM_K] = tensorViewB.sizes[DIM_K];
    tensorB.loopStride[DIM_K] = m_geoAttr.getGeometryWidth();
    tensorB.startOffset[DIM_K] = fcdView.viewBase;

    // tensor C - non-conv dims
    tensorC.roiSize[DIM_K] = cRoiSizes[DIM_K];
    tensorC.validElements[DIM_K] = tensorViewC.sizes[DIM_K];
    tensorC.loopStride[DIM_K] = m_geoAttr.getGeometryWidth();
    tensorC.startOffset[DIM_K] = fcdView.viewBase;

    //  Roi Size probably needs needs to be padded to a multiple of the EU.
    tensorC.roiSize[WEIGHT_DIM_C] = std::max(cRoiSizes[WEIGHT_DIM_C], m_geoAttr.getGeometryHeight());
    tensorC.validElements[WEIGHT_DIM_C] = tensorViewC.sizes[WEIGHT_DIM_C];
    tensorC.loopStride[WEIGHT_DIM_C] = m_geoAttr.getGeometryHeight();
    tensorC.spatialStrides[WEIGHT_DIM_C] = 1;
    tensorC.startOffset[WEIGHT_DIM_C] = convView.bases[1];

    //  set conv dimensions
    SizeArray spPos = m_recipe.calcSpPos(spView.viewBase);
    for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; convDim++)
    {
        const unsigned actDim = convDim + 1;
        const unsigned weightDim = convDim + 2;
        const unsigned batchConcurrenty =
            m_geoAttr.getConcurrentDim() == weightDim ? m_geoAttr.getGeometryConcurrency() : 1;
        tensorA.spatialStrides[actDim] = m_params.conv.stride[convDim];
        tensorA.baseOffset[actDim] = m_params.conv.dilation[convDim] * convView.bases[weightDim] * batchConcurrenty -
                                     m_params.conv.padding[convDim];
        tensorA.startOffset[actDim] = spPos[actDim - 1] * m_params.conv.stride[convDim];
        tensorA.roiSize[actDim] = bRoiSizes[actDim] * m_params.conv.stride[convDim];
        tensorA.validElements[actDim] = tensorViewA.sizes[actDim];
        tensorA.loopStride[actDim] = m_params.conv.dilation[convDim];

        tensorB.spatialStrides[actDim] = 1;
        tensorB.baseOffset[actDim] = 0;
        tensorB.startOffset[actDim] = spPos[actDim - 1];
        tensorB.roiSize[actDim] = bRoiSizes[actDim];
        tensorB.validElements[actDim] = tensorViewB.sizes[actDim];
        tensorB.loopStride[actDim] = 0;

        tensorC.spatialStrides[weightDim] = 1;
        tensorC.baseOffset[weightDim] = 0;
        tensorC.startOffset[weightDim] = convView.bases[weightDim] * batchConcurrenty;
        tensorC.roiSize[weightDim] = cRoiSizes[weightDim];
        tensorC.validElements[weightDim] = tensorViewC.sizes[weightDim];
        tensorC.loopStride[weightDim] = 1;
    }

    //  set Batch dimension
    tensorA.spatialStrides[DIM_B] = 1;
    tensorA.baseOffset[DIM_B] = 0;
    tensorA.startOffset[DIM_B] = spPos[DIM_B - 1];
    tensorA.validElements[DIM_B] = tensorViewA.sizes[DIM_B];
    tensorA.loopStride[DIM_B] = 0;

    tensorB.spatialStrides[DIM_B] = 1;
    tensorB.baseOffset[DIM_B] = 0;
    tensorB.startOffset[DIM_B] = spPos[DIM_B - 1];
    tensorB.validElements[DIM_B] = tensorViewB.sizes[DIM_B];
    tensorB.loopStride[DIM_B] = 0;

    // Check if common dim is misaligned relative to reduction tree. If so pad the overall spatial size
    const unsigned spatialSizeCD = getPaddedCommonDim(spView.viewSize, tensorViewB.elementType);
    tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, true, spatialSizeCD);
    tensorB.lastSpatialStep = calcLastStepSize(e_mme_op_b, true, spatialSizeCD);
    if (m_geoAttr.getMmeConcurrency() > 1)
    {
        // configure the MME to read all relevant row in the EU.
        // in dedw batch concurrency multiple batches are calculated in the same EU, so the effective EU
        // height is larger than what geoAttr presents.
        tensorC.lastSpatialStep = 128;
    }
    else
    {
        tensorC.lastSpatialStep = calcLastStepSize(e_mme_op_c, true, convView.sizes[1] % m_geoAttr.getGeometryHeight());
    }
    // last FCD step can be reduced for A/B to slightly reduce power
    tensorA.lastFcdStep = m_geoAttr.getPortSize(e_mme_op_a);
    tensorB.lastFcdStep = m_geoAttr.getPortSize(e_mme_op_b);
    tensorC.lastFcdStep = calcLastStepSize(e_mme_op_c, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());
}

void CommonAguConfig::configureBatchLoops(EMmeInternalOperand operand)
{
    TensorAttr& tensor = getTensor(operand);
    const MmeTensorView& tensorView = m_recipe.getOperand(operand);
    const auto& batchView = m_recipe.curNonSpatial();

    for (unsigned batchDim = 2; batchDim < 2 + c_batchDimNr; ++batchDim)
    {
        bool isBroadcast = m_geoAttr.isOperandBroadcasted(operand, (MmeDimsIndex) batchDim);
        tensor.spatialStrides[batchDim] = 1;  // not used but configured for completeness.
        tensor.baseOffset[batchDim] = 0;
        tensor.startOffset[batchDim] = isBroadcast ? 0 : batchView.bases[batchDim];
        tensor.roiSize[batchDim] = m_recipe.getRoiSizes(operand)[batchDim];
        tensor.validElements[batchDim] = tensorView.sizes[batchDim];
        tensor.loopStride[batchDim] = isBroadcast ? 0 : 1;
    }
}

void CommonAguConfig::configureBgemmTransposedA()
{
    const MmeTensorView& tensorView = m_recipe.getOperand(e_mme_op_a);
    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const auto& batchView = m_recipe.curNonSpatial();
    const auto& spView = m_recipe.curSp();

    const unsigned commonDimIdx = m_geoAttr.isTransposed(e_mme_op_b) ? 0 : 1;
    unsigned originalCommonDim = batchView.sizes[commonDimIdx];
    unsigned paddedCommonDim = getPaddedCommonDim(originalCommonDim, m_params.w.elementType);
    if (!m_recipe.isLastPartial())
    {
        if (!m_recipe.maskedBgemm)
        {
            //  middle partials can not be padded so their size has to be already aligned
            MME_ASSERT(originalCommonDim == paddedCommonDim, "subView size is not aligned");
        }
    }

    tensorA.spatialStrides[DIM_C] = 1;  // FCD dim doenst have a spatial stride
    tensorA.baseOffset[DIM_C] = batchView.bases[commonDimIdx];
    tensorA.startOffset[DIM_C] = 0;
    tensorA.roiSize[DIM_C] = paddedCommonDim;
    tensorA.validElements[DIM_C] = tensorView.sizes[DIM_C];
    tensorA.loopStride[DIM_C] = 0;

    tensorA.spatialStrides[DIM_W] = 1;
    tensorA.baseOffset[DIM_W] = 0;
    tensorA.startOffset[DIM_W] = spView.viewBase;
    tensorA.roiSize[DIM_W] = aRoiSizes[DIM_W];
    tensorA.validElements[DIM_W] = tensorView.sizes[DIM_W];
    tensorA.loopStride[DIM_W] = m_geoAttr.getGeometryHeight();

    tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, true, spView.viewSize % m_geoAttr.getGeometryHeight());
    tensorA.lastFcdStep = paddedCommonDim;

    configureBatchLoops(e_mme_op_a);
}

void CommonAguConfig::configureBgemmTransposedB()
{
    const MmeTensorView& tensorView = m_recipe.getOperand(e_mme_op_b);
    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);
    const auto& batchView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();

    unsigned originalCommonDim = batchView.sizes[DIM_C];
    unsigned paddedCommonDim = getPaddedCommonDim(originalCommonDim, m_params.w.elementType);
    if (!m_recipe.isLastPartial())
    {
        if (!m_recipe.maskedBgemm)
        {
            //  middle partials can not be padded so their size has to be already aligned
            MME_ASSERT(originalCommonDim == paddedCommonDim, "subView size is not aligned");
        }
    }

    tensorB.spatialStrides[DIM_C] = 1;  // FCD dim doenst have a spatial stride
    tensorB.baseOffset[DIM_C] = batchView.bases[DIM_C];
    tensorB.startOffset[DIM_C] = 0;
    tensorB.roiSize[DIM_C] = paddedCommonDim;
    tensorB.validElements[DIM_C] = tensorView.sizes[DIM_C];
    tensorB.loopStride[DIM_C] = 0;

    tensorB.spatialStrides[DIM_W] = 1;  //  B operand doesnt interleave ports.
    tensorB.baseOffset[DIM_W] = 0;
    tensorB.startOffset[DIM_W] = fcdView.viewBase;
    tensorB.roiSize[DIM_W] = bRoiSizes[DIM_W];
    tensorB.validElements[DIM_W] = tensorView.sizes[DIM_W];
    tensorB.loopStride[DIM_W] = m_geoAttr.getGeometryWidth();

    tensorB.lastSpatialStep = calcLastStepSize(e_mme_op_b, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());
    tensorB.lastFcdStep = paddedCommonDim;

    configureBatchLoops(e_mme_op_b);
}
void CommonAguConfig::configureBgemmNonTransposedA()
{
    const MmeTensorView& tensorView = m_recipe.getOperand(e_mme_op_a);
    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const auto& batchView = m_recipe.curNonSpatial();
    const auto& spView = m_recipe.curSp();

    const unsigned commonDimIdx = m_geoAttr.isTransposed(e_mme_op_b) ? 0 : 1;
    unsigned originalCommonDim = batchView.sizes[commonDimIdx];
    unsigned paddedCommonDim = getPaddedCommonDim(originalCommonDim, m_params.w.elementType);
    if (!m_recipe.isLastPartial())
    {
        if (!m_recipe.maskedBgemm)
        {
            //  middle partials can not be padded so their size has to be already aligned
            MME_ASSERT(originalCommonDim == paddedCommonDim, "subView size is not aligned");
        }
    }

    tensorA.baseOffset[DIM_C] = 0;
    tensorA.startOffset[DIM_C] = spView.viewBase;
    tensorA.roiSize[DIM_C] = aRoiSizes[DIM_C];
    tensorA.validElements[DIM_C] = tensorView.sizes[DIM_C];
    tensorA.spatialStrides[DIM_C] = 1;  // FCD dim doenst have a spatial stride
    tensorA.loopStride[DIM_C] = m_geoAttr.getGeometryHeight();

    tensorA.baseOffset[DIM_W] = batchView.bases[commonDimIdx];
    tensorA.startOffset[DIM_W] = 0;
    tensorA.roiSize[DIM_W] = paddedCommonDim;
    tensorA.validElements[DIM_W] = tensorView.sizes[DIM_W];
    tensorA.spatialStrides[DIM_W] = 1;
    tensorA.loopStride[DIM_W] = 0;  // unused

    if (!m_params.isGemmDmaOperation())
    {
        MME_ASSERT(paddedCommonDim % m_geoAttr.getMmeSpatialPorts(e_mme_op_a) == 0,
                   "spatial dim must be aligned to spatial port num in bgemm");
        tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, true, paddedCommonDim);
    }
    else
    {
        // in the case of unit matrix transpose spatial movements in the input translate to FCD movements in the output
        // using the MME loops. therefor we need to initialize the height loop stride to the geometry width.
        // for the same reason we also want to limit the CD (input A spatial dim) to a single MME width.
        // and set the startOffset of the first spatial dim to be the FCD view offset
        tensorA.loopStride[DIM_W] = m_geoAttr.getGeometryWidth();
        tensorA.startOffset[DIM_W] = m_recipe.curFcd().viewBase;
        tensorA.lastSpatialStep = calcLastStepSize(e_mme_op_a, true, m_geoAttr.getMmeWidth());
    }
    // last FCD step can be reduced for A/B to slightly reduce power
    tensorA.lastFcdStep = m_geoAttr.getPortSize(e_mme_op_a);

    configureBatchLoops(e_mme_op_a);
}

void CommonAguConfig::configureBgemmNonTransposedB()
{
    const MmeTensorView& tensorView = m_recipe.getOperand(e_mme_op_b);
    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);
    const auto& batchView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();

    unsigned originalCommonDim = batchView.sizes[WEIGHT_DIM_C];
    unsigned paddedCommonDim = getPaddedCommonDim(originalCommonDim, m_params.w.elementType);
    if (!m_recipe.isLastPartial())
    {
        if (!m_recipe.maskedBgemm)
        {
            //  middle partials can not be padded so their size has to be already aligned
            MME_ASSERT(originalCommonDim == paddedCommonDim, "subView size is not aligned");
        }
    }

    tensorB.baseOffset[DIM_K] = 0;
    tensorB.startOffset[DIM_K] = fcdView.viewBase;
    tensorB.roiSize[DIM_K] = bRoiSizes[DIM_K];
    tensorB.validElements[DIM_K] = tensorView.sizes[DIM_K];
    tensorB.spatialStrides[DIM_K] = 1;  // FCD dim doenst have a spatial stride
    tensorB.loopStride[DIM_K] = m_geoAttr.getGeometryWidth();

    tensorB.baseOffset[WEIGHT_DIM_C] = batchView.bases[WEIGHT_DIM_C];
    tensorB.startOffset[WEIGHT_DIM_C] = 0;
    tensorB.roiSize[WEIGHT_DIM_C] = paddedCommonDim;
    tensorB.validElements[WEIGHT_DIM_C] = tensorView.sizes[WEIGHT_DIM_C];
    tensorB.spatialStrides[WEIGHT_DIM_C] = 1;
    tensorB.loopStride[WEIGHT_DIM_C] = 0;

    MME_ASSERT(paddedCommonDim % m_geoAttr.getMmeSpatialPorts(e_mme_op_b) == 0,
               "spatial dim must be aligned to spatial port num in bgemm");
    tensorB.lastSpatialStep = calcLastStepSize(e_mme_op_b, true, paddedCommonDim);
    // last FCD step can be reduced for A/B to slightly reduce power
    tensorB.lastFcdStep = m_geoAttr.getPortSize(e_mme_op_b);

    configureBatchLoops(e_mme_op_b);
}
void CommonAguConfig::configureBgemmOutput()
{
    const MmeTensorView& tensorView = m_recipe.getOperand(e_mme_op_c);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);
    const auto& batchView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();

    tensorC.baseOffset[DIM_K] = 0;
    tensorC.startOffset[DIM_K] = fcdView.viewBase;
    tensorC.roiSize[DIM_K] = cRoiSizes[DIM_K];
    tensorC.validElements[DIM_K] = tensorView.sizes[DIM_K];
    tensorC.spatialStrides[DIM_K] = 1;  // FCD dim doenst have a spatial stride
    tensorC.loopStride[DIM_K] = m_geoAttr.getGeometryWidth();

    tensorC.baseOffset[DIM_W] = 0;
    tensorC.startOffset[DIM_W] = spView.viewBase;
    tensorC.roiSize[DIM_W] = cRoiSizes[DIM_W];
    tensorC.validElements[DIM_W] = tensorView.sizes[DIM_W];
    tensorC.spatialStrides[DIM_W] = 1;
    tensorC.loopStride[DIM_W] = m_geoAttr.getGeometryHeight();

    tensorC.lastSpatialStep = calcLastStepSize(e_mme_op_c, true, spView.viewSize % m_geoAttr.getGeometryHeight());
    tensorC.lastFcdStep = calcLastStepSize(e_mme_op_c, false, fcdView.viewSize % m_geoAttr.getGeometryWidth());

    configureBatchLoops(e_mme_op_c);
}

unsigned CommonAguConfig::calcLastStepSize(EMmeInternalOperand operand, bool isSpatial, unsigned lastStepSize)
{
    unsigned geoSize = isSpatial ? m_geoAttr.getGeometryHeight() : m_geoAttr.getGeometryWidth();  // what about CD?
    lastStepSize = lastStepSize == 0 ? geoSize : lastStepSize;
    if (isSpatial && m_geoAttr.isSpatiallyInterleavedAcrossCores(operand))  // TODO: [SW-78758] change to AcrossMmes
    {
        //  interleaving ports share the same spatial step
        return div_round_up(lastStepSize, m_geoAttr.getInterleavedSpatialPortsNr(operand));
    }
    else
    {
        //  when ports are not interleaved only one MME will have hit the partial last step
        //  other ports will either has a full step or empty step
        GeometryGrid mmeGrid = m_geoAttr.mmeIdxToGrid(m_mmeIdx);
        unsigned mmeIdx = isSpatial ? mmeGrid.spatial : mmeGrid.fcd;
        unsigned mmeSize = isSpatial ? m_geoAttr.getMmeHeight() : m_geoAttr.getMmeWidth();

        unsigned curMmeOffset = mmeIdx * mmeSize;
        curMmeOffset = fixForTeAcceleration(operand, curMmeOffset);

        if (curMmeOffset >= lastStepSize)
        {
            //  current MME is out of bound in the last step
            return 1;
        }
        else
        {
            //  full step size differs between operands and whether they are transposed
            //  for FCD walk the full step is port size for inputs and EU width for output.
            //  for spatial walk it is TE height for input and EuHeight/InterleavedSpatialPortsNr for output
            //  (this is non-interleaved across cores case, so InterleavedSpatialPortsNr is inside the EU)
            unsigned fullStepSize;
            if (operand == e_mme_op_c)
            {
                if (m_params.isNativeDmaOperation()) fullStepSize = m_geoAttr.getPortSize(operand);
                else
                    fullStepSize = isSpatial ? m_geoAttr.getEuHeight() / m_geoAttr.getInterleavedSpatialPortsNr(operand)
                                             : m_geoAttr.getEuWidth();
            }
            else
            {
                fullStepSize = m_geoAttr.getEuFacingPortSize(operand);
                fullStepSize = fixForTeAcceleration(operand, fullStepSize);
            }

            unsigned coreSize = isSpatial ? m_geoAttr.getEuHeight() : m_geoAttr.getEuWidth();
            coreSize = fixForTeAcceleration(operand, coreSize);
            if (lastStepSize - curMmeOffset >= coreSize)
            {
                // not interleaved across cores, and the last step is the same for all cores, so since the first core
                // is full then the last step has to be a full step for all cores.
                return fullStepSize;
            }
            else
            // last step fits inside a single core
            {
                if (isSpatial && m_geoAttr.isSpatiallyInterleavedInsideCore(operand))
                {
                    return div_round_up(lastStepSize - curMmeOffset, m_geoAttr.getInterleavedSpatialPortsNr(operand));
                }

                //  full step size is usually smaller than core size because several ports will use the same size
                //  so we only limit the last step size if its smaller than a single ports full step size.
                return std::min(lastStepSize - curMmeOffset, fullStepSize);
            }
        }
    }
}

//  update tensorAttr with the offset of the current MME. offset will be the same for all ports.
//  offset given in element of the appropriate dimension - needs to be multiplied by the appropriate stride.
void CommonAguConfig::setMmeOffset(EMmeInternalOperand operand)
{
    const GeometryGrid mmeGrid = m_geoAttr.mmeIdxToGrid(m_mmeIdx);
    TensorAttr& tensor = getTensor(operand);

    //  spatial Offset
    if (operand == e_mme_op_a || operand == e_mme_op_c)
    {
        if (operand == e_mme_op_a && (!m_geoAttr.isTransposed(operand) || m_params.opType == e_mme_trans))
        {
            //  if A is non transposed (or in the case of native transpose) then the MME spatial offset applies for its FCD
            tensor.baseOffset[GEMM_DIM_W] += mmeGrid.spatial * m_geoAttr.getMmeHeight();
        }
        else
        {
            //  if the spatial dimension is interleaved each MMEs offset is a single row.
            //  otherwise the offset is MME height rows
            if (m_geoAttr.isSpatiallyInterleavedAcrossMmes(operand))
            {
                unsigned portSpOffset = mmeGrid.spatial;
                if (operand == e_mme_op_a)
                {
                    // when configuring spatial offset we want to set the current port to start reading X pixels ahead
                    // of the base port. in fwd/dedx the gemm is constructed using im2col, because of that advancing X
                    // pixels ahead means we actually want to read the input pixel that is located X*convStride pixels
                    // ahead.
                    portSpOffset *= m_params.conv.stride[0];
                }
                tensor.startOffset[GEMM_DIM_H] += portSpOffset;
            }
            else
            {
                unsigned offset = mmeGrid.spatial * m_geoAttr.getMmeHeight();
                offset = fixForTeAcceleration(operand, offset);
                tensor.baseOffset[GEMM_DIM_H] += offset;
            }
        }
    }

    //  fcd Offset
    //  in dma mode a behaves like a b port
    if (operand == e_mme_op_b || operand == e_mme_op_c || (operand == e_mme_op_a && m_params.isDmaOperation()))
    {
        unsigned dim;
        unsigned offset = mmeGrid.fcd * m_geoAttr.getMmeWidth();

        if (m_params.opType == e_mme_memcpy)
        {
            // unlike any other op, memcpy doesnt have a constant MME size, as the amount of elements each port handles
            // is determined by the current activation FCD size.
            // the offset will be calculated using the number of ports instead.
            offset = mmeGrid.fcd * m_geoAttr.getMmeFcdPorts(operand) * getSinglePortFcdSize(operand);
        }

        if ((operand != e_mme_op_c && m_geoAttr.isTransposed(operand)) ||
            (operand == e_mme_op_a && m_params.isGemmDmaOperation()))
        {
            //  fix offset if it is spatial
            offset = fixForTeAcceleration(operand, offset);
            //  if b (or a in DMA mode) is transposed then the MME FCD offset is applied to its first spatial dimension
            dim = GEMM_DIM_H;
        }
        else
        {
            dim = GEMM_DIM_W;
        }

        tensor.baseOffset[dim] += offset;
    }

    // batch offset
    if (m_geoAttr.supportsConcurrency())
    {
        //  advance the MME ahead by its index and by the amount of batches each MME handles.
        unsigned batchIdx = mmeGrid.batch * m_geoAttr.getMmeConcurrency();
        MmeDimsIndex concurrentDim = m_geoAttr.getConcurrentDim();
        if (!m_geoAttr.isOperandBroadcasted(operand, concurrentDim))
        {
            if (operand == e_mme_op_a && m_params.isDedwOperation())
            {
                //  in DEDW batch movements affect the input differently since the gemm is created using im2col.
                //  when advancing X filter ahead we actually need to advance the input x*dilation pixels ahead.
                unsigned spDim = concurrentDim - DIM_W;  // translate from filter dim to spatial dim
                unsigned convDim = concurrentDim - DIM_S;  // translate W filter dim to actual filter index
                tensor.baseOffset[spDim] += batchIdx * m_params.conv.dilation[convDim];
            }
            else
            {
                tensor.baseOffset[concurrentDim] += batchIdx;
            }
        }
    }

    // cd Offset
    if (operand == e_mme_op_a || operand == e_mme_op_b)
    {
        //  if the spatial dimension is interleaved each MMEs offset is a single row.
        unsigned spDim = m_geoAttr.getSpInterleavingDim(operand);
        if (m_geoAttr.isSpatiallyInterleavedAcrossMmes(operand))
        {
            unsigned portSpOffset = mmeGrid.cd;
            if (operand == e_mme_op_a && m_params.isDedwOperation())
            {
                // when configuring spatial offset we want to set the current port to start reading X pixels ahead
                // of the base port. in fwd the gemm is constructed using im2col, because of that advancing X pixels
                // ahead means we actually want to read the input pixel that is located X*convStride pixels ahead.
                portSpOffset *= m_params.conv.stride[spDim - 1];
            }
            tensor.startOffset[spDim] += portSpOffset;
        }
    }
}

void CommonAguConfig::setCoreOffset(EMmeInternalOperand operand, unsigned coreIdx, PortAttr& coreBasePort)
{
    GeometryGrid effectiveCoreGrid = m_geoAttr.coreIdxToEffectiveGrid(operand, coreIdx);

    // fcd offset
    coreBasePort.portOffset[GEMM_DIM_W] +=
        effectiveCoreGrid.fcd * m_geoAttr.getCoreFcdPorts(operand) * m_geoAttr.getPortSize(operand);

    // spatial offset
    unsigned spDim = m_geoAttr.getSpInterleavingDim(operand);
    unsigned coreSpatialOffset = effectiveCoreGrid.spatial * getTensor(operand).spatialStrides[spDim];
    if (m_geoAttr.isSpatiallyInterleavedAcrossCores(operand))
    {
        // when spatially interleaved, the offset is 1 row for each interleaved port. if interleaved across MMEs that
        // would mean the number of interleaved MMEs multiplied by the core's spatial index, otherwise it's just the
        // core's spatial index.
        if (m_geoAttr.isSpatiallyInterleavedAcrossMmes(operand))
        {
            coreSpatialOffset *= m_geoAttr.getSpatialMmeNr(operand);
        }
    }
    else
    {
        // no interleaving between cores, so each core is EuHeight rows ahead for output,
        // or (coreSpatialPorts * TeHeight) rows for input
        if (operand == e_mme_op_c)
        {
            coreSpatialOffset *= m_geoAttr.getEuHeight();
        }
        else
        {
            coreSpatialOffset *= m_geoAttr.getTeHeight() * m_geoAttr.getCoreSpatialPorts(operand);
        }
    }

    coreSpatialOffset = fixForTeAcceleration(operand, coreSpatialOffset);
    coreBasePort.portOffset[spDim] += coreSpatialOffset;

    // batch offset between cores is simply the batch offset between ports of the same core multiplied by the total
    // number of batch ports in previous cores
    unsigned coreBatchOffset = effectiveCoreGrid.batch * m_geoAttr.getCoreConcurrency();
    setPortBatchOffset(operand, coreBatchOffset, coreBasePort);
}

inline PortAttr CommonAguConfig::getBasePort(EMmeInternalOperand operand)
{
    TensorAttr& tensor = getTensor(operand);
    PortAttr basePort;
    const bool isPortStartOffset = m_geoAttr.isPortStartOffset(operand);

    for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        if (isPortStartOffset)
        {
            basePort.portOffset[dim] = tensor.startOffset[dim];
        }
        else
        {
            basePort.portOffset[dim] = tensor.baseOffset[dim];
        }
    }

    //  since startOffset doesnt have an FCD field this offset has to be folded into the
    //  offset of the ports.
    if (isPortStartOffset)
    {
        basePort.portOffset[0] += tensor.baseOffset[0];
    }
    else
    {
        basePort.portOffset[0] += tensor.startOffset[0];
    }
    return basePort;
}
//  this is a utility function to calculate the current port spatial offset
//  each port behaves differently spatially, so this logic was exported to its own function.
void CommonAguConfig::setPortSpatialOffset(EMmeInternalOperand operand, unsigned spatialIdx, PortAttr& curPort)
{
    //  the ports begins at spatialIdx steps ahead of the base port.
    unsigned portSpOffset = spatialIdx * getTensor(operand).spatialStrides[DIM_W];
    if (m_geoAttr.isSpatiallyInterleavedInsideCore(operand))
    {
        if (m_geoAttr.isSpatiallyInterleavedAcrossCores(operand))
        {
            portSpOffset *= m_geoAttr.getEffectiveCoreGrid(operand).spatial;
            if (m_geoAttr.isSpatiallyInterleavedAcrossMmes(operand))
            {
                portSpOffset *= m_geoAttr.getSpatialMmeNr(operand);
            }
        }
        //  when spatially interleaving inside core but not across cores, each port is 1 spatial line ahead of the
        //  previous port. so its spatial index is the spatial offset.
    }
    else
    {
        //  non interleaved, each port is EuHeight rows ahead for output, or TE Height rows ahead for input.
        if (operand == e_mme_op_c)
        {
            if (m_params.opType != e_mme_trans)
            {
                portSpOffset *= m_geoAttr.getEuHeight();
            }
            else
            {
                // in this case we want ot make sure that there is enough work to fully utilize the input BW
                portSpOffset *= m_geoAttr.getPortSize(e_mme_op_a);
            }

        }
        else
        {
            portSpOffset *= m_geoAttr.getTeHeight();
        }
    }

    MmeDimsIndex interleavedDim = m_geoAttr.getSpInterleavingDim(operand);
    portSpOffset = fixForTeAcceleration(operand, portSpOffset);
    int overallOffset = curPort.portOffset[interleavedDim] + portSpOffset;
    int maxOffset = getTensor(operand).roiSize[interleavedDim];
    if (overallOffset < maxOffset || m_geoAttr.getLastSpatialDim(operand) == FIRST_SP_DIM ||
        !m_geoAttr.isPortStartOffset(operand))
    {
        //  set the port offset to the first spatial dim, it is eiterh
        //  1. fits inside ROI
        //  2. there is only a single spatial dim so we cant wrap around (padding will be read)
        //  3. the port is not startOffset - meaning we are not setitng a spatial offset that should be wrapped around
        //     but moving the entire ROI instead.
        curPort.portOffset[interleavedDim] = overallOffset;
    }
    else
    {
        // overall offset spills over roi - wrap around first sp dim and check all following dims are still in roi.
        curPort.portOffset[interleavedDim] = overallOffset - maxOffset;
        for (int spDim = interleavedDim + 1; spDim <= m_geoAttr.getLastSpatialDim(operand); spDim++)
        {
            curPort.portOffset[spDim] += getTensor(operand).spatialStrides[spDim];
            if (curPort.portOffset[spDim] == getTensor(operand).roiSize[spDim])
            {
                if (spDim < m_geoAttr.getLastSpatialDim(operand))
                {
                    //  keep the offset at the last spatial dim to make sure the offset is out of the valid elements
                    //  and only padding is read.
                    curPort.portOffset[spDim] = 0;
                }
            }
            else
            {
                MME_ASSERT(curPort.portOffset[spDim] < (int) getTensor(operand).roiSize[spDim],
                           "expected offset to be inside ROI");
                break;
            }
        }
    }
}

void CommonAguConfig::setPortBatchOffset(EMmeInternalOperand operand, unsigned batchIdx, PortAttr& curPort)
{
    //  not batch dimension for fwd or dedx
    if (!m_geoAttr.supportsConcurrency()) return;
    // if the operand is broadcasted dont apply an offset to it.
    MmeDimsIndex dim = m_geoAttr.getConcurrentDim();
    if (m_geoAttr.isOperandBroadcasted(operand, dim)) return;
    if (m_params.isDedwOperation() && operand == e_mme_op_a)
    {
        //  for each batch advancements the port needs to advance dilation steps ahead
        unsigned convDim = dim - DIM_S;
        batchIdx *= m_params.conv.dilation[convDim];
        //  for a the batch advancment is a appropriate spatial dim advancement
        dim = (MmeDimsIndex)(dim - 1);
    }

    curPort.portOffset[dim] += batchIdx;
}

void CommonAguConfig::invalidateLogicalPort(PortAttr& corePort)
{
    corePort.portOffset[MAX_DIMENSION - 1] = -10;
}

void CommonAguConfig::unitMatrixOffsets(PortsComplex& ports)
{
    unsigned coreSpatialPorts = m_geoAttr.getCoreSpatialPorts(e_mme_op_b);
    ports.resize(m_geoAttr.getCoresPerMmeNr());
    for (unsigned coreIdx = 0; coreIdx < m_geoAttr.getCoresPerMmeNr(); coreIdx++)
    {
        ports.at(coreIdx).resize(1);
        ports.at(coreIdx).at(0).resize(1);
        ports.at(coreIdx).at(0).at(0).resize(1);
        ports.at(coreIdx).at(0).at(0).at(0).resize(coreSpatialPorts);
        for (unsigned sp = 0; sp < coreSpatialPorts; sp++)
        {
            PortAttr& curPort = ports.at(coreIdx).at(0).at(0).at(0).at(sp);
            curPort.portOffset[1] -= coreIdx * coreSpatialPorts + sp;
        }
    }
}

//  configure the portOffset of the current MME
void CommonAguConfig::setPortOffsets(EMmeInternalOperand operand)
{
    setMmeOffset(operand);
    PortAttr basePort = getBasePort(operand);
    PortsComplex& ports = getPortComplex(operand);

    if (operand == e_mme_op_b && m_params.opType == e_mme_gemm_transpose)
    {
        unitMatrixOffsets(ports);
        return;
    }

    unsigned coreFcdPorts = m_geoAttr.getCoreFcdPorts(operand);
    unsigned coreSpatialPorts = m_geoAttr.getCoreSpatialPorts(operand);
    unsigned coreBatchPorts = m_geoAttr.getCoreBatchPorts(operand);
    unsigned coreCdPorts = m_geoAttr.getCoreCdPorts(operand);

    ports.resize(m_geoAttr.getCoresPerMmeNr());
    for (unsigned coreIdx = 0; coreIdx < m_geoAttr.getCoresPerMmeNr(); coreIdx++)
    {
        PortAttr coreBasePort = basePort;
        setCoreOffset(operand, coreIdx, coreBasePort);

        ports.at(coreIdx).resize(coreCdPorts);
        for (unsigned cd = 0; cd < coreCdPorts; cd++)
        {
            ports.at(coreIdx).at(cd).resize(coreBatchPorts);
            for (unsigned batch = 0; batch < coreBatchPorts; batch++)
            {
                ports.at(coreIdx).at(cd).at(batch).resize(coreFcdPorts);
                for (unsigned fcd = 0; fcd < coreFcdPorts; fcd++)
                {
                    ports.at(coreIdx).at(cd).at(batch).at(fcd).resize(coreSpatialPorts);
                    for (unsigned sp = 0; sp < coreSpatialPorts; sp++)
                    {
                        PortAttr& curPort = ports.at(coreIdx).at(cd).at(batch).at(fcd).at(sp);

                        if (m_geoAttr.isPortValid(operand, coreIdx, cd, batch, fcd, sp))
                        {
                            curPort = coreBasePort;

                            //  FCD adjacent ports have portSize elements between them
                            curPort.portOffset[GEMM_DIM_W] += fcd * getSinglePortFcdSize(operand);
                            setPortSpatialOffset(operand, sp, curPort);
                            setPortBatchOffset(operand, batch, curPort);
                        }
                        else
                        {
                            invalidateLogicalPort(curPort);
                        }
                    }
                }
            }
        }
    }
}

void CommonAguConfig::multiplyStrides(EMmeInternalOperand operand)
{
    TensorAttr& tensor = getTensor(operand);

    //  increase spatial stride over the interleaved spatial dimension
    tensor.spatialStrides[m_geoAttr.getSpInterleavingDim(operand)] *= m_geoAttr.getInterleavedSpatialPortsNr(operand);

    //  increase loop stride over the concurrent batch dimension
    if (m_geoAttr.supportsConcurrency())
    {
        unsigned concurrentDim = m_geoAttr.getConcurrentDim();
        if (operand == e_mme_op_a && m_params.isDedwOperation())
        {
            //  in dedw batch movements moves A spatially
            concurrentDim--;
        }
        tensor.loopStride[concurrentDim] *= m_geoAttr.getGeometryConcurrency();
    }
}

void CommonAguConfig::finalizeSizes(EMmeInternalOperand operand)
{
    if (operand == e_mme_op_b && m_params.opType == e_mme_gemm_transpose) return;

    const MmeTensorView& tensorView = m_recipe.getOperand(operand);
    TensorAttr& tensor = getTensor(operand);
    PortsComplex& ports = getPortComplex(operand);

    for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        unsigned stride = tensorView.strides[dim];
        tensor.spatialStrides[dim] *= stride;
        tensor.roiSize[dim] *= stride;
        tensor.validElements[dim] *= stride;
        tensor.baseOffset[dim] *= stride;
        tensor.startOffset[dim] *= stride;
        tensor.loopStride[dim] *= stride;
    }

    fixDescOffsets(operand, getTensor(operand));

    for (auto& Ports4D : ports)
    {
        for (auto& Ports3D : Ports4D)
        {
            for (auto& Ports2D : Ports3D)
            {
                for (auto& Ports1D : Ports2D)
                {
                    for (auto& singlePort : Ports1D)
                    {
                        for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
                        {
                            singlePort.portOffset[dim] *= tensorView.strides[dim];
                        }
                    }
                }
            }
        }
    }
}

unsigned CommonAguConfig::getSinglePortFcdSize(EMmeInternalOperand operand)
{
    if (m_params.opType == e_mme_memcpy)
    {
        unsigned singlePortFcdSteps = div_round_up(m_recipe.curFcd().viewSize,
                                                   m_geoAttr.getChipFcdPorts(operand) * m_geoAttr.getPortSize(operand));
        unsigned portFcdSize = singlePortFcdSteps * m_geoAttr.getPortSize(operand);
        return portFcdSize;
    }
    else
    {
        return m_geoAttr.getPortSize(operand);
    }
}

void CommonAguConfig::fixDescOffsets(EMmeInternalOperand operand, TensorAttr& tensorAttr)
{
    // In some cases we may break the mme desc generation into multiple sub-problems, such
    // that some sub-problems do not start at the same origin as the original task.
    // For example, in dedx we break the task into sub-problems according to the strides,
    // and each sub-problem starts at a different offset in the input tensors.
    // In fwd, when we break the task into sub-problem for better handling recurrent misalignments,
    // Each sub-problem has its offsets for both input and output tensors.
    if (operand == mmeOpToInternalOp(e_mme_op_x, m_params.opType))
    {
        if (m_geoAttr.isPortStartOffset(operand))
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                tensorAttr.baseOffset[dim] += m_descAddrOffset.xOffset[dim];
            }
        }
        else
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                tensorAttr.startOffset[dim] += m_descAddrOffset.xOffset[dim];
            }
        }
    }
    else if (operand == mmeOpToInternalOp(e_mme_op_w, m_params.opType))
    {
        if (m_geoAttr.isPortStartOffset(operand))
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                tensorAttr.baseOffset[dim] += m_descAddrOffset.wOffset[dim];
            }
        }
        else
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                tensorAttr.startOffset[dim] += m_descAddrOffset.wOffset[dim];
            }
        }
    }
    else if (operand == mmeOpToInternalOp(e_mme_op_y, m_params.opType))
    {
        if (m_geoAttr.isPortStartOffset(operand))
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                tensorAttr.baseOffset[dim] += m_descAddrOffset.yOffset[dim];
            }
        }
        else
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                tensorAttr.startOffset[dim] += m_descAddrOffset.yOffset[dim];
            }
        }
    }
    else
    {
        MME_ASSERT(0, "invalid operand");
    }
}

PortsComplex& CommonAguConfig::getPortComplex(EMmeInternalOperand operand)
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_a:
            return portsA;
        case e_mme_op_b:
            return portsB;
        case e_mme_op_c:
            return portsC;
    }
}
TensorAttr& CommonAguConfig::getTensor(EMmeInternalOperand operand)
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_a:
            return tensorA;
        case e_mme_op_b:
            return tensorB;
        case e_mme_op_c:
            return tensorC;
    }
}

void CommonAguConfig::setAssociatedDimsDma(void* descPtr)
{
    //    MME_ASSERT(m_params.strategy.pattern == e_mme_sp_reduction_fck,
    //               "dma operations assiciated dims logic only implemented for fck pattern");
    const auto& batchView = m_recipe.curNonSpatial();
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();
    EMmeLoopMask loopMask = e_mme_conv_loop_0;

    // add gemm transpose virtual CD loop
    if (m_params.opType == e_mme_gemm_transpose)
    {
        // we are utilizing the first batch loop for the gemm_transpose operation,
        // so it becomes necessary to shift the batch sizes of the remaining batch dimensions.
        // As a result, currently we cannot transpose a tensor with 5 dims
        MME_ASSERT(batchView.sizes[GEMM_DIM_B3] == 1, "expected the last batch dim to be trivial");
        unsigned cdSteps = div_round_up(m_geoAttr.getMmeWidth(), m_geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b));
        setAssociatedDimAndSize(loopMask, cdSteps, MME_MAX_TENSOR_DIMS, 1, MME_MAX_TENSOR_DIMS, descPtr);
        //  advance loop mask
        loopMask = (EMmeLoopMask) ((((unsigned) loopMask) << 1) + 1);
    }

    //  set output FCD steps
    unsigned denseStepsNr = div_round_up(fcdView.viewSize, m_geoAttr.getGeometryWidth());
    EMmeLoopMask fcdMask = loopMask;
    if (m_geoAttr.isTransposed(e_mme_op_a) || m_params.isGemmDmaOperation())
    {
        setAssociatedDimAndSize(fcdMask, denseStepsNr, 1, MME_MAX_TENSOR_DIMS, 0, descPtr);
    }
    else
    {
        setAssociatedDimAndSize(fcdMask, 1, 0, MME_MAX_TENSOR_DIMS, 0, descPtr);
    }
    loopMask = (EMmeLoopMask) ((((unsigned) loopMask) << 1) + 1);

    // Set output Height steps
    unsigned spatialStepsNr;
    if (m_params.opType == e_mme_memcpy)
        // native DMA operations cover the entire SP dim using the gemm loop, so not sp steps for it.
        spatialStepsNr = 1;
    else
        spatialStepsNr = div_round_up(fixForTeAcceleration(e_mme_op_c, spView.viewSize, false), m_geoAttr.getGeometryHeight());
    EMmeLoopMask spMask = loopMask;
    setAssociatedDimAndSize(spMask, spatialStepsNr, 0, MME_MAX_TENSOR_DIMS, 1, descPtr);
    loopMask = (EMmeLoopMask) ((((unsigned) loopMask) << 1) + 1);

    unsigned maxBatch = m_geoAttr.getBatchDimsNr();
    unsigned batchDim = 0;
    // set the remaining upper loops, will either be used for "batch" movement or masked
    do
    {
        if (loopMask == e_mme_tetris_loop)
        {
            setSpatialLoopSize(1, descPtr);
        }
        else if (batchDim < maxBatch)
        {
            // the batch dimensions start at index 2
            unsigned batchDimIdx = batchDim + 2;
            //  in transpose mode we use the loops, in memcpy we dont.
            unsigned loopSize = m_params.opType == e_mme_memcpy ? 1 : batchView.sizes[batchDimIdx];
            setAssociatedDimAndSize(loopMask, loopSize, batchDimIdx, MME_MAX_TENSOR_DIMS, batchDimIdx, descPtr);
            batchDim++;
        }
        else
        {
            setAssociatedDimAndSize(e_mme_outer_loop,
                                    1,
                                    MME_MAX_TENSOR_DIMS,
                                    MME_MAX_TENSOR_DIMS,
                                    MME_MAX_TENSOR_DIMS,
                                    descPtr);
        }
        loopMask = (EMmeLoopMask)((((unsigned) loopMask) << 1) + 1);
    } while (loopMask <= e_mme_outer_loop);

    // set partial loop masks
    if (m_params.isGemmDmaOperation())
    {
        setPartialHeightLoopMaskA(getLoopFromLoopMask(spMask), descPtr);
        setPartialHeightLoopMaskB(getLoopFromLoopMask(fcdMask), descPtr);
    }
    else
    {
        // in native transpose mode, counterintuitively we set the partial height using the fcd mask..
        // this is because of the transpose action, the output width is the inputs height.
        setPartialHeightLoopMaskA(getLoopFromLoopMask(fcdMask), descPtr);
        // needs to always be set to last step, use spatial loop as it isnt used
        setPartialHeightLoopMaskB(getLoopFromLoopMask(e_mme_tetris_loop), descPtr);
    }
}

void CommonAguConfig::setAssociatedDimsBgemmDedw(void* descPtr)
{
    const auto& batchView = m_recipe.curNonSpatial();
    unsigned batchLoops[c_batchDimNr] = {batchView.sizes[GEMM_DIM_B1],
                                         batchView.sizes[GEMM_DIM_B2],
                                         batchView.sizes[GEMM_DIM_B3]};
    if (m_params.isGemmOperation())
    {
        unsigned concurrentIdx = m_geoAttr.getConcurrentDim() - GEMM_DIM_B1;  //  move idx to zero based
        batchLoops[concurrentIdx] = div_round_up(batchLoops[concurrentIdx], m_geoAttr.getGeometryConcurrency());
    }
    unsigned maxBatch = m_geoAttr.getBatchDimsNr();
    // Set batch loops
    EMmeLoopMask batchMask = pattern2LoopMask(m_params.strategy.pattern, EMmeLoopDim::dim_b);
    for (unsigned batchDim = 0; batchDim < maxBatch; ++batchDim)
    {
        MME_ASSERT(batchMask <= e_mme_outer_loop, "invalid filter mask");
        unsigned dimA, dimB, dimC;
        if (m_params.isDedwOperation())
        {
            dimA = batchDim + 1,
            dimB = MAX_DIMENSION;  //  no movement in Y.
            dimC = batchDim + 2;
        }
        else if (m_params.opType == e_mme_gemm_transpose)
        {
            if (batchDim == 0)
            {
                dimA = MAX_DIMENSION;
                dimB = batchDim;
                dimC = MAX_DIMENSION;
            }
            else
            {
                unsigned batchDimIdx = batchDim + 1;
                dimA = batchDimIdx;
                dimB = batchDimIdx;
                dimC = batchDimIdx;
            }
        }
        else
        {
            // bgemm
            MME_ASSERT(m_params.isGemmOperation(), "expected gemm operation");
            unsigned batchDimIdx = batchDim + 2;
            bool aBroadcast = m_recipe.getOperand(e_mme_op_a).sizes[batchDimIdx] == 1;
            bool bBroadcast = m_recipe.getOperand(e_mme_op_b).sizes[batchDimIdx] == 1;
            dimA = aBroadcast ? MAX_DIMENSION : batchDimIdx;
            dimB = bBroadcast ? MAX_DIMENSION : batchDimIdx;
            dimC = batchDimIdx;
            // in the following workaround when A is transposed data from two batches are interleaved in the same EU.
            // a virtual dimensions will be created to handle that, reserve a loop for it.
            if (m_geoAttr.isMmeConcurrencyRoutingWorkAround() && m_geoAttr.isTransposed(e_mme_op_a)) dimC++;
        }

        setAssociatedDimAndSize(batchMask, batchLoops[batchDim], dimA, dimB, dimC, descPtr);
        do
        {
            batchMask = (EMmeLoopMask)((((unsigned) batchMask) << 1) + 1);
        } while (batchMask == e_mme_tetris_loop);
    }

    //  calculate number of step in each dimension
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();
    const auto& convView = m_recipe.curNonSpatial();
    unsigned aSpatialSize = (m_params.isDedwOperation()) ? convView.sizes[1] : spView.viewSize;
    unsigned denseStepsNr = div_round_up(fcdView.viewSize, m_geoAttr.getGeometryWidth());
    unsigned spatialStepsNr = div_round_up(aSpatialSize, m_geoAttr.getGeometryHeight());

    // Set k
    const EMmeLoopMask kMask = pattern2LoopMask(m_params.strategy.pattern, EMmeLoopDim::dim_k);
    MME_ASSERT(kMask != e_mme_tetris_loop, "kMask cannot be tetris loop");
    setAssociatedDimAndSize(kMask,
                            denseStepsNr,
                            MME_MAX_TENSOR_DIMS,
                            m_geoAttr.isTransposed(e_mme_op_b) ? 1 : 0,
                            0,
                            descPtr);

    // Set C
    const EMmeLoopMask cMask = pattern2LoopMask(m_params.strategy.pattern, EMmeLoopDim::dim_c);
    MME_ASSERT(cMask != e_mme_tetris_loop, "cMask cannot be tertris loop");
    setAssociatedDimAndSize(cMask,
                            spatialStepsNr,
                            m_geoAttr.isTransposed(e_mme_op_a) ? 1 : 0,
                            MME_MAX_TENSOR_DIMS,
                            1,
                            descPtr);

    // Set S - no S in bgemm
    setSpatialLoopSize(1, descPtr);

    // set partial loop masks
    setPartialHeightLoopMaskA(getLoopFromLoopMask(cMask), descPtr);
    setPartialHeightLoopMaskB(getLoopFromLoopMask(kMask), descPtr);
}

void CommonAguConfig::setAssociatedDimsFwdDedx(void* descPtr)
{
    const auto& convView = m_recipe.curNonSpatial();
    unsigned weightSizes[MME_MAX_CONV_DIMS - 1] = {convView.sizes[DIM_S], convView.sizes[DIM_R], convView.sizes[DIM_Q]};

    // Set filter
    EMmeLoopMask currentFilterMask = pattern2LoopMask(m_params.strategy.pattern, EMmeLoopDim::dim_f);
    for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; convDim++)
    {
        const unsigned ACT_DIM = convDim + 1;
        const unsigned WEIGHT_DIM = convDim + 2;
        // conv loops
        setAssociatedDimAndSize(currentFilterMask,
                                isMemsetDesc() ? 1 : weightSizes[convDim],
                                isMemsetDesc() ? MME_MAX_TENSOR_DIMS : ACT_DIM,
                                isMemsetDesc() ? MME_MAX_TENSOR_DIMS : WEIGHT_DIM,
                                MME_MAX_TENSOR_DIMS,
                                descPtr);
        currentFilterMask = (EMmeLoopMask)((((unsigned) currentFilterMask) << 1) + 1);
    }
    // set Non Accumulated Loops

    //  calculate number of step in each dimension
    const auto& fcdView = m_recipe.curFcd();
    const auto& spView = m_recipe.curSp();
    unsigned denseStepsNr = div_round_up(fcdView.viewSize, m_geoAttr.getGeometryWidth());
    unsigned spatialStepsNr = div_round_up(spView.viewSize, m_geoAttr.getGeometryHeight());

    setPartialHeightLoopMaskA(getLoopFromLoopMask(EMmeLoopMask::e_mme_tetris_loop), descPtr);
    setSpatialLoopSize(spatialStepsNr, descPtr);

    EMmeLoopMask denseLoopMask = pattern2LoopMask(m_params.strategy.pattern, EMmeLoopDim::dim_k);

    setAssociatedDimAndSize(denseLoopMask,
                            denseStepsNr,
                            MME_MAX_TENSOR_DIMS,
                            m_geoAttr.isTransposed(e_mme_op_b) ? WEIGHT_DIM_C : DIM_K,
                            DIM_K,
                            descPtr);

    setAssociatedDimAndSize((denseLoopMask == e_mme_conv_loop_3) ? e_mme_outer_loop : e_mme_conv_loop_3,
                            1,
                            MME_MAX_TENSOR_DIMS,
                            MME_MAX_TENSOR_DIMS,
                            MME_MAX_TENSOR_DIMS,
                            descPtr);

    setPartialHeightLoopMaskB(getLoopFromLoopMask(denseLoopMask), descPtr);
}

}  // namespace MmeCommon
