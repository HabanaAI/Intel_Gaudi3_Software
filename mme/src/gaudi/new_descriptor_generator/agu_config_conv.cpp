#include "include/gaudi/new_descriptor_generator/agu_config.h"
#include "include/mme_assert.h"
#include "include/gaudi/new_descriptor_generator/mme_common.h"
#include "include/mme_common/mme_common_enum.h"
#include "src/gaudi/gaudi_geo_attr.h"

using namespace MmeCommon;

namespace gaudi
{
// The updated padding is derived by subtracting the filter base from the appropriate padding value
static std::array<int, MME_MAX_CONV_DIMS - 1>
updatePaddingForDedw(std::array<int, MME_MAX_CONV_DIMS - 1> origPadding,
                     SizeArray convViewBases,
                     const std::array<unsigned, MME_MAX_CONV_DIMS - 1>& dilation)
{
    std::array<int, MME_MAX_CONV_DIMS - 1> updatedPadding = {0};
    for (unsigned d = 0; d < updatedPadding.size(); d++)
    {
        updatedPadding[d] = origPadding[d] - convViewBases[d + 2] * dilation[d];
    }
    return updatedPadding;
}

void AguConvFwdDedxConfig::setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                            Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    // In gaudi2, we set the valid elements to the whole tensor (e.g. tensorViewA.sizes[0]).
    // In gaudi, we use the specific sub-tensor we work on per recipe iteration.
    auto partialViews = getRecipe().getNonSpatialSubviews();
    // We match the legacy code that updates only tensor A upon lowering. Because of that, tensor B
    // and the subviews do not reflect the actual sizes after lowering.
    // Therefore, in case there are no partials, we need to set tensorS.validElements[0] from A
    // tensor.
    if (partialViews.size() == 1)  // no partials
    {
        localDesc.tensorS.validElements[0] = tensorViewA.sizes[0];
        localDesc.tensorS.roiSize[0] = localDesc.tensorS.validElements[0];
    }
    else
    {
        switch (params.opType)
        {
            case e_mme_fwd:
                localDesc.tensorS.validElements[0] = tensorViewA.sizes[DIM_C];
                localDesc.tensorS.roiSize[0] = getRecipe().curNonSpatial().sizes[1];
                break;
            case e_mme_dedx:
                localDesc.tensorS.validElements[0] = getRecipe().curNonSpatial().sizes[0];
                localDesc.tensorS.roiSize[0] = localDesc.tensorS.validElements[0];
                break;
            default:
                MME_ASSERT(0, "invalid operation type");
        }
    }

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguS[j].roiBaseOffset[0] = getRecipe().curNonSpatial().bases[1];
    }
    localDesc.tensorS.loopStride[0] = 0;
}

void AguConvFwdDedxConfig::setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                           Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();
    auto partialViews = getRecipe().getNonSpatialSubviews();
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);

    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        // tensor A: spatial
        localDesc.tensorS.validElements[ioDim] = tensorViewA.sizes[ioDim] * tensorViewA.strides[ioDim];
        // todo AlonG: drop paddedRoi once recipe is fixed for the case that output width is below num A ports
        localDesc.tensorS.roiSize[ioDim] = params.conv.stride[dim] * cRoiSizes[ioDim] * tensorViewA.strides[ioDim];
        localDesc.tensorS.loopStride[ioDim] = params.conv.dilation[dim] * tensorViewA.strides[ioDim];
        localDesc.tensorS.spatialStrides[ioDim - 1] =
            params.conv.stride[dim] * tensorViewA.strides[ioDim] * (dim ? 1 : gp->m_totalAports);

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            if (params.opType == e_mme_dedx)
            {
                aguS[j].roiBaseOffset[ioDim] =
                    -(getRecipe().curNonSpatial().sizes[wDim] + getRecipe().curNonSpatial().bases[wDim] - 1);
                aguS[j].roiBaseOffset[ioDim] *= params.conv.dilation[dim];
                aguS[j].roiBaseOffset[ioDim] += params.conv.padding[dim];
            }
            else
            {
                aguS[j].roiBaseOffset[ioDim] = getRecipe().curNonSpatial().bases[wDim];
                aguS[j].roiBaseOffset[ioDim] *= params.conv.dilation[dim];
                aguS[j].roiBaseOffset[ioDim] -= params.conv.padding[dim];
            }
            aguS[j].roiBaseOffset[ioDim] += (params.conv.stride[dim] * tensorViewC.bases[ioDim]);
            aguS[j].roiBaseOffset[ioDim] -= tensorViewA.bases[ioDim];
            aguS[j].roiBaseOffset[ioDim] *= tensorViewA.strides[ioDim];
            //  add subproblem offset
            if (params.opType == e_mme_dedx)
            {
                aguS[j].roiBaseOffset[ioDim] += m_convSubProblem->addressOffset.yOffset[ioDim];
            }

            aguS[j].startOffset[ioDim - 1] =
                getSpPos(j % gp->m_totalAports)[dim] * params.conv.stride[dim] * tensorViewA.strides[ioDim];
        }
    }

    for (int dim = c_padded_conv_dim + 1; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        // tensor A: outer spatial loops
        localDesc.tensorS.validElements[dim] =
            (tensorViewA.sizes[dim] - tensorViewA.bases[dim]) * tensorViewA.strides[dim];
        localDesc.tensorS.spatialStrides[dim - 1] = tensorViewA.strides[dim];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[dim] = 0;
            aguS[j].startOffset[dim - 1] = getSpPos(j % gp->m_totalAports)[dim - 1] * tensorViewA.strides[dim];
        }
    }
}

//======================= Tensor B ========================================
void AguConvFwdDedxConfig::setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                            Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();

    const auto& fcdView = getRecipe().curFcd();
    auto partialViews = getRecipe().getNonSpatialSubviews();
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);

    // tensor B: K (dim 0), C (dim 1)
    // In gaudi2, the valid elements cover the full tensor, so we should use tensorViewB.sizes[1]
    // In gaudi, we set to the actual sub-tensor
    // todo [SW-51592] change gaudi new desc roi to match gaudi2
    // todo AlonG: change getRoi().size[0] to getRoi().size[1] and analyze the effect

    unsigned roiLWidth = 0;
    if (partialViews.size() == 1)  // no partials
    {
        localDesc.tensorL.validElements[0] = tensorViewB.sizes[0];
        roiLWidth = localDesc.header.transL ? tensorViewB.sizes[0] : (getRoi().size[0] % gp->m_subMatrixWidth);
        localDesc.tensorL.roiSize[0] = roiLWidth ? roiLWidth : gp->m_subMatrixWidth;

        localDesc.tensorL.validElements[1] = tensorViewB.sizes[1] * tensorViewB.strides[1];
        localDesc.tensorL.roiSize[1] =
            tensorViewB.strides[1] *
            (localDesc.header.transL ? std::min(gp->m_subMatrixWidth, getRoi().size[0]) : tensorViewB.sizes[1]);
    }
    else
    {
        switch (params.opType)
        {
            // todo [SW-51592] change gaudi new desc roi to match gaudi2
            case e_mme_fwd:  // B is non-transposed
                localDesc.tensorL.validElements[0] = tensorViewB.sizes[0];
                roiLWidth = (fcdView.viewSize % gp->m_subMatrixWidth);
                localDesc.tensorL.roiSize[0] = roiLWidth ? roiLWidth : gp->m_subMatrixWidth;
                localDesc.tensorL.validElements[1] = getRecipe().curNonSpatial().sizes[1] * tensorViewB.strides[1];
                localDesc.tensorL.roiSize[1] = localDesc.tensorL.validElements[1];
                break;
            case e_mme_dedx:  // B is transposed
                localDesc.tensorL.validElements[0] = getRecipe().curNonSpatial().sizes[0];
                localDesc.tensorL.roiSize[0] = getRecipe().curNonSpatial().sizes[0];
                localDesc.tensorL.validElements[1] = tensorViewB.sizes[1] * tensorViewB.strides[1];
                // todo AlonG: this is likely wrong.
                localDesc.tensorL.roiSize[1] =
                    std::min(gp->m_subMatrixWidth, getRecipe().curFcd().viewSize) * tensorViewB.strides[1];
                break;
            default:
                MME_ASSERT(0, "invalid operation type");
        }
    }

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguL[j].roiBaseOffset[0] =
            localDesc.header.transL ? 0 : (((j % gp->m_totalBports) * gp->m_subMatrixWidth) + getRoi().denseBase);
        // TODO: align partial ROI of A and B using the base
        // aguL[j].roiBaseOffset[0] -= tensorViewB.bases[0];
    }

    localDesc.tensorL.spatialStrides[0] = tensorViewB.strides[1];
    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguL[j].roiBaseOffset[1] =
            localDesc.header.transL ? (((j % gp->m_totalBports) * gp->m_subMatrixWidth) + getRoi().denseBase) : 0;
        // TODO: align partial ROI of A and B using the base
        // aguL[j].roiBaseOffset[1] -= tensorViewB.bases[1];
        aguL[j].roiBaseOffset[1] *= tensorViewB.strides[1];
    }

    if (params.opType == e_mme_dedx)
    {
        localDesc.tensorL.loopStride[1] = gp->m_matrixWidth * tensorViewB.strides[1];
        setLoopDim(&localDesc, params.strategy.pattern, LOOP_K, OP_L, 1);
    }
    else
    {
        localDesc.tensorL.loopStride[0] = gp->m_matrixWidth;
        setLoopDim(&localDesc, params.strategy.pattern, LOOP_K, OP_L, 0);
    }
}

void AguConvFwdDedxConfig::setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                           Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    auto gp = getGeoAttr();

    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        localDesc.tensorL.validElements[wDim] = getRecipe().curNonSpatial().sizes[wDim] * tensorViewB.strides[wDim];
        if (wDim != Mme::c_mme_max_tensor_dims - 1)
        {
            localDesc.tensorL.roiSize[wDim] = tensorViewB.strides[wDim];
        }
        localDesc.tensorL.loopStride[wDim] =
            (params.opType == e_mme_dedx) ? -tensorViewB.strides[wDim] : tensorViewB.strides[wDim];
        localDesc.tensorL.spatialStrides[wDim - 1] = tensorViewB.strides[wDim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[wDim] =
                (params.opType == e_mme_dedx) ? (getRecipe().curNonSpatial().sizes[wDim] - 1) : 0;
            aguL[j].roiBaseOffset[wDim] *= tensorViewB.strides[wDim];
            aguL[j].startOffset[wDim - 1] = 0;
        }
    }
}

//====================== Tensor C ========================================
void AguConvFwdDedxConfig::setDescFirstOfAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                                 Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();
    const auto& fcdView = getRecipe().curFcd();
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    // tensor C: (dim 0)
    localDesc.tensorO.roiSize[0] = fcdView.viewSize % gp->m_subMatrixWidth;
    localDesc.tensorO.validElements[0] = tensorViewC.sizes[0];
    localDesc.tensorO.roiSize[0] = localDesc.tensorO.roiSize[0] ? localDesc.tensorO.roiSize[0] : gp->m_subMatrixWidth;
    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguO[j].roiBaseOffset[0] =
            ((j % gp->m_totalBports) * gp->m_subMatrixWidth) + getRoi().denseBase - tensorViewC.bases[0];
    }
    localDesc.tensorO.loopStride[0] = gp->m_matrixWidth;
}

void AguConvFwdDedxConfig::setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                              Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);
    auto gp = getGeoAttr();

    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        // tensor C: spatial loops
        localDesc.tensorO.validElements[ioDim] = tensorViewC.sizes[ioDim] * tensorViewC.strides[ioDim];
        localDesc.tensorO.roiSize[ioDim] = cRoiSizes[ioDim] * tensorViewC.strides[ioDim];
        localDesc.tensorO.loopStride[ioDim] = 0;
        localDesc.tensorO.spatialStrides[ioDim - 1] = tensorViewC.strides[ioDim] * (dim ? 1 : gp->m_totalAports);

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[ioDim] = m_convSubProblem->addressOffset.xOffset[ioDim];
            aguO[j].startOffset[dim] = getSpPos(j / getGeoAttr()->m_totalBports)[dim] * tensorViewC.strides[ioDim];
        }
        setLoopDim(&localDesc, params.strategy.pattern, LOOP_FILTER + dim, OP_O, ioDim);
    }

    for (int dim = c_padded_conv_dim + 1; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        // tensor C: spatial loops
        localDesc.tensorO.validElements[dim] =
            (tensorViewC.sizes[dim] - tensorViewC.bases[dim]) * tensorViewC.strides[dim];
        if (dim != Mme::c_mme_max_tensor_dims - 1)
        {
            localDesc.tensorO.roiSize[dim] = cRoiSizes[dim] * tensorViewC.strides[dim];
        }
        localDesc.tensorO.spatialStrides[dim - 1] = tensorViewC.strides[dim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[dim] = m_convSubProblem->addressOffset.xOffset[dim];
            aguO[j].startOffset[dim - 1] = getSpPos(j / gp->m_totalBports)[dim - 1] * tensorViewC.strides[dim];
        }
    }
}

//===================================================================
//============= Dedw Agu Config =====================================
void AguConvDedwConfig::setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& localDesc)
{
    auto gp = getGeoAttr();

    const SingleDimSubView& fcdView = getRecipe().curFcd();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const MultiDimSubView& convView = getRecipe().curNonSpatial();

    localDesc.tensorS.validElements[0] = convView.sizes[1];
    localDesc.tensorS.roiSize[0] = std::min(convView.sizes[1], gp->m_subMatrixHeight);

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguS[j].roiBaseOffset[0] = (j % gp->m_totalAports) * gp->m_subMatrixHeight;
    }
    localDesc.tensorS.loopStride[0] = gp->m_matrixHeight;
}
void AguConvDedwConfig::setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();
    const MultiDimSubView& convView = getRecipe().curNonSpatial();
    std::array<int, MME_MAX_CONV_DIMS - 1> currentPadding =
        updatePaddingForDedw(params.conv.padding, convView.bases, params.conv.dilation);
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);

    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        // tensor A: spatial
        localDesc.tensorS.validElements[ioDim] = tensorViewA.sizes[ioDim] * tensorViewA.strides[ioDim];
        localDesc.tensorS.roiSize[ioDim] =
            params.conv.stride[dim] * params.getOperand(e_mme_op_b).sizes[ioDim] * tensorViewA.strides[ioDim];
        localDesc.tensorS.loopStride[ioDim] = params.conv.dilation[dim] * tensorViewA.strides[ioDim];
        localDesc.tensorS.spatialStrides[ioDim - 1] = params.conv.stride[dim] * tensorViewA.strides[ioDim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[ioDim] = -currentPadding[dim] * tensorViewA.strides[ioDim];
            aguS[j].startOffset[ioDim - 1] = getSpPosB(ioDim) * params.conv.stride[dim] * tensorViewA.strides[ioDim];
        }

        setLoopDim(&localDesc, params.strategy.pattern, LOOP_FILTER + dim, OP_S, ioDim);
    }

    for (int dim = c_padded_conv_dim + 1; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        // tensor A: outer spatial loops
        localDesc.tensorS.validElements[dim] = tensorViewA.sizes[dim] * tensorViewA.strides[dim];
        if (dim != Mme::c_mme_max_tensor_dims - 1)
        {
            localDesc.tensorS.roiSize[dim] = tensorViewA.sizes[dim] * tensorViewA.strides[dim];
        }
        localDesc.tensorS.spatialStrides[dim - 1] = tensorViewA.strides[dim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[dim] = 0;
            aguS[j].startOffset[dim - 1] = getSpPosB(dim) * tensorViewA.strides[dim];
        }
    }
}
void AguConvDedwConfig::setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& localDesc)
{
    auto gp = getGeoAttr();
    const SingleDimSubView& fcdView = getRecipe().curFcd();

    localDesc.tensorL.validElements[0] = fcdView.viewSize;
    localDesc.tensorL.roiSize[0] = gp->m_subMatrixWidth;

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguL[j].roiBaseOffset[0] = (j % gp->m_totalBports) * gp->m_subMatrixWidth;
    }
    localDesc.tensorL.loopStride[0] = gp->m_matrixWidth;
}

void AguConvDedwConfig::setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        // tensor B: spatial
        localDesc.tensorL.validElements[ioDim] =
            params.getOperand(e_mme_op_b).sizes[ioDim] * params.getOperand(e_mme_op_b).strides[ioDim];
        localDesc.tensorL.roiSize[ioDim] =
            params.getOperand(e_mme_op_b).sizes[ioDim] * params.getOperand(e_mme_op_b).strides[ioDim];
        localDesc.tensorL.loopStride[ioDim] = 0;
        localDesc.tensorL.spatialStrides[ioDim - 1] = params.getOperand(e_mme_op_b).strides[ioDim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[ioDim] = 0;
            aguL[j].startOffset[ioDim - 1] = getSpPosB(ioDim) * params.getOperand(e_mme_op_b).strides[ioDim];
        }

        setLoopDim(&localDesc, params.strategy.pattern, LOOP_FILTER + dim, OP_L, ioDim);
    }

    for (int dim = c_padded_conv_dim + 1; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        // tensor B: outer spatial loops
        localDesc.tensorL.validElements[dim] =
            params.getOperand(e_mme_op_b).sizes[dim] * params.getOperand(e_mme_op_b).strides[dim];
        if (dim != Mme::c_mme_max_tensor_dims - 1)
        {
            localDesc.tensorL.roiSize[dim] =
                params.getOperand(e_mme_op_b).sizes[dim] * params.getOperand(e_mme_op_b).strides[dim];
        }
        localDesc.tensorL.spatialStrides[dim - 1] = params.getOperand(e_mme_op_b).strides[dim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[dim] = 0;
            aguL[j].startOffset[dim - 1] = getSpPosB(dim) * params.getOperand(e_mme_op_b).strides[dim];
        }
    }
}

void AguConvDedwConfig::setDescFirstOfAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                              Mme::Desc& localDesc)
{
    const SingleDimSubView& fcdView = getRecipe().curFcd();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const MultiDimSubView& convView = getRecipe().curNonSpatial();
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    auto gp = getGeoAttr();

    localDesc.tensorO.validElements[0] = fcdView.viewSize;
    localDesc.tensorO.roiSize[0] = gp->m_subMatrixWidth;

    localDesc.tensorO.validElements[1] = convView.sizes[1] * tensorViewC.strides[1];
    localDesc.tensorO.roiSize[1] = std::min(convView.sizes[1], gp->m_subMatrixHeight) * tensorViewC.strides[1];
    localDesc.tensorO.spatialStrides[0] = tensorViewC.strides[1];

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguO[j].roiBaseOffset[0] = (j % gp->m_totalBports) * gp->m_subMatrixWidth;
        aguO[j].roiBaseOffset[1] = (j / gp->m_totalBports) * gp->m_subMatrixHeight * tensorViewC.strides[1];
    }
    localDesc.tensorO.loopStride[0] = gp->m_matrixWidth;
    localDesc.tensorO.loopStride[1] = gp->m_matrixHeight * tensorViewC.strides[1];
}

void AguConvDedwConfig::setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                           Mme::Desc& localDesc)
{
    const SingleDimSubView& fcdView = getRecipe().curFcd();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const MultiDimSubView& convView = getRecipe().curNonSpatial();
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        // tensor C: kernel loops
        localDesc.tensorO.validElements[wDim] = convView.sizes[wDim] * tensorViewC.strides[wDim];
        if (wDim != Mme::c_mme_max_tensor_dims - 1)
        {
            localDesc.tensorO.roiSize[wDim] = tensorViewC.strides[wDim];
        }
        localDesc.tensorO.loopStride[wDim] = tensorViewC.strides[wDim];
        localDesc.tensorO.spatialStrides[wDim - 1] = tensorViewC.strides[wDim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[wDim] = 0;
            aguO[j].startOffset[wDim - 1] = 0;
        }
    }
}

//==========================================================================
//============= Main Agu setting code ======================================
//==========================================================================

static void transposeDesc(Mme::Desc* desc)
{
    desc->header.transO = !desc->header.transO;

    std::swap(desc->baseAddrHighS, desc->baseAddrHighL);
    std::swap(desc->baseAddrLowS, desc->baseAddrLowL);
    swap_bf(desc->header.transS, desc->header.transL);
    swap_bf(desc->header.advanceS, desc->header.advanceL);
    swap_bf(desc->header.lowerS, desc->header.lowerL);
    for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        swap_bf(desc->conv.associatedDims[i].dimS, desc->conv.associatedDims[i].dimL);
    }
    swap_bf(desc->outerLoop.associatedDims.dimS, desc->outerLoop.associatedDims.dimL);
    std::swap(desc->tensorS, desc->tensorL);
    std::swap(desc->paddingValueS, desc->paddingValueL);
}

void AguConvConfig::setNumIterationsMinus1(Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();

    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_dedx:
            localDesc.numIterationsMinus1 =
                ((getRecipe().curSp().viewSize + gp->m_matrixHeight - 1) / gp->m_matrixHeight) - 1;
            break;
        case e_mme_dedw:
            localDesc.numIterationsMinus1 = 0;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }
}

void AguConvFwdDedxConfig::setLoopDimsAndSizes(Mme::Desc& desc)
{
    const MmeLayerParams& params = getParams();
    EMmePattern pattern = params.getPattern();
    auto gp = m_geoAttrSPtr;

    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;

        // Tensor S
        setLoopDim(&desc, pattern, LOOP_FILTER + dim, OP_S, ioDim);

        // Tensor L
        setLoopDim(&desc, pattern, LOOP_FILTER + dim, OP_L, wDim);
        setLoopSize(&desc, pattern, LOOP_FILTER + dim, getRecipe().curNonSpatial().sizes[wDim] - 1);

        // Tensor O
        setLoopDim(&desc, pattern, LOOP_FILTER + dim, OP_O, ioDim);
    }

    if (params.opType == e_mme_dedx)
    {
        setLoopDim(&desc, pattern, LOOP_K, OP_L, 1);
    }
    else
    {
        setLoopDim(&desc, pattern, LOOP_K, OP_L, 0);
    }
    setLoopDim(&desc, pattern, LOOP_K, OP_O, 0);

    unsigned fcdIdx = getRecipe().getIterator().fcdIdx();
    unsigned denseLoopSize =
        ((getRecipe().getFcdSubviews()[fcdIdx].viewSize + gp->m_matrixWidth - 1) / gp->m_matrixWidth) - 1;
    setLoopSize(&desc, pattern, LOOP_K, denseLoopSize);
}

void AguConvDedwConfig::setLoopDimsAndSizes(Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    EMmePattern pattern = params.getPattern();
    auto gp = getGeoAttr();
    auto& fcdView = getRecipe().curFcd();
    auto& convView = getRecipe().curNonSpatial();

    // Set the filter loops
    for (int dim = 0; dim < c_padded_conv_dim; dim++)
    {
        int ioDim = dim + 1;
        int wDim = dim + 2;
        setLoopDim(&localDesc, pattern, LOOP_FILTER + dim, OP_S, ioDim);
        setLoopDim(&localDesc, pattern, LOOP_FILTER + dim, OP_L, ioDim);
        setLoopDim(&localDesc, pattern, LOOP_FILTER + dim, OP_O, wDim);
        setLoopSize(&localDesc, pattern, LOOP_FILTER + dim, convView.sizes[wDim] - 1);
    }

    unsigned loopSizeK = div_round_up(fcdView.viewSize, gp->m_matrixWidth) - 1;
    setLoopSize(&localDesc, pattern, LOOP_K, loopSizeK);
    setLoopDim(&localDesc, pattern, LOOP_K, OP_S, Mme::c_mme_max_tensor_dims);
    setLoopDim(&localDesc, pattern, LOOP_K, OP_L, 0);
    setLoopDim(&localDesc, pattern, LOOP_K, OP_O, 0);

    unsigned loopSizeC = div_round_up(convView.sizes[1], gp->m_matrixHeight) - 1;
    setLoopSize(&localDesc, pattern, LOOP_C, loopSizeC);
    setLoopDim(&localDesc, pattern, LOOP_C, OP_S, 0);
    setLoopDim(&localDesc, pattern, LOOP_C, OP_L, Mme::c_mme_max_tensor_dims);
    setLoopDim(&localDesc, pattern, LOOP_C, OP_O, 1);
}

void AguConvFwdDedxConfig::setSpatialSize(Mme::Desc& localDesc, bool isMemsetDesc)
{
    const MmeLayerParams& params = getParams();
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    auto gp = getGeoAttr();

    int spElementsPerAguA = (m_recipe.curSp().viewSize + gp->m_totalAports - 1) / gp->m_totalAports;
    int spatialSizeAC =
        spElementsPerAguA % gp->m_subMatrixHeight ? spElementsPerAguA % gp->m_subMatrixHeight : gp->m_subMatrixHeight;
    int spatialSizeB;
    auto partialViews = getRecipe().getNonSpatialSubviews();
    if (partialViews.size() == 1)  // no partials
    {
        spatialSizeB = tensorViewB.sizes[1];
    }
    else
    {
        spatialSizeB = (params.opType == e_mme_dedx) ? getRecipe().getRoiSizes(e_mme_op_c)[0]
                                                     : getRecipe().curNonSpatial().sizes[1];
    }

    if (params.opType == e_mme_dedx)
    {
        spatialSizeB %= gp->m_subMatrixWidth;
        spatialSizeB = (spatialSizeB != 0) ? spatialSizeB : gp->m_subMatrixWidth;
    }

    localDesc.tensorS.spatialSizeMinus1 = spatialSizeAC - 1;
    localDesc.tensorO.spatialSizeMinus1 = spatialSizeAC - 1;
    localDesc.tensorL.spatialSizeMinus1 = isMemsetDesc ? 0 : spatialSizeB - 1;
}

void AguConvDedwConfig::setSpatialSize(Mme::Desc& localDesc, bool isMemsetDesc)
{
    auto gp = getGeoAttr();
    const MultiDimSubView& convView = getRecipe().curNonSpatial();

    localDesc.tensorS.spatialSizeMinus1 = m_recipe.curSp().viewSize - 1;
    localDesc.tensorL.spatialSizeMinus1 = m_recipe.curSp().viewSize - 1;
    localDesc.tensorO.spatialSizeMinus1 = std::min(convView.sizes[1], gp->m_subMatrixHeight) - 1;
}
void AguConvFwdDedxConfig::setLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                    std::array<unsigned, c_operand_max_agu>& loopS,
                                    std::array<unsigned, c_operand_max_agu>& loopL,
                                    std::array<unsigned, c_operand_max_agu>& loopO)
{
    const MmeLayerParams& params = getParams();
    auto fcdView = getRecipe().curFcd();
    auto partialViews = getRecipe().getNonSpatialSubviews();
    auto gp = getGeoAttr();

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        loopS[j] = getLoopMask(params.strategy.pattern, LOOP_SPATIAL);
        unsigned rem = fcdView.viewSize % gp->m_matrixWidth;
        rem = rem ? rem : gp->m_matrixWidth;
        loopL[j] = ((rem + getRoi().denseBase - aguO[j].roiBaseOffset[0]) < gp->m_subMatrixWidth)
                       ? getLoopMask(params.strategy.pattern, LOOP_K)
                       : 0;
        loopO[j] = ((rem + getRoi().denseBase - aguO[j].roiBaseOffset[0]) < gp->m_subMatrixWidth)
                       ? getLoopMask(params.strategy.pattern, LOOP_K)
                       : 0;
    }
}

void AguConvDedwConfig::setLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                 std::array<unsigned, c_operand_max_agu>& loopS,
                                 std::array<unsigned, c_operand_max_agu>& loopL,
                                 std::array<unsigned, c_operand_max_agu>& loopO)
{
    const MmeLayerParams& params = getParams();

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        loopS[j] = loopL[j] = loopO[j] = 0;
    }
    loopS[0] = getLoopMask(params.strategy.pattern, LOOP_C);
    loopL[0] = getLoopMask(params.strategy.pattern, LOOP_K);
}

AguConfig::AguConfig(const MmeCommon::MmeRecipe& recipe,
                     const MmeCommon::MmeLayerParams& params,
                     const gaudiReuseAttr& reuseAttr,
                     GeoAttr::UseDataTypeForAttributeCalculation useDataType) :
    m_params(params),
    m_recipe(recipe),
    m_geoAttrSPtr(std::make_shared<GeoAttr>(params, gaudi::MmeHalReader::getInstance(), useDataType)),
    m_newGeoAttrSPtr(std::make_shared<GaudiGeoAttr>(params)),
    m_reuseAttr(reuseAttr)
{}

void AguConfig::setPartialHeightLoops(Mme::Desc* descPtr,
                                      const std::array<unsigned, c_operand_max_agu>& loopS,
                                      const std::array<unsigned, c_operand_max_agu>& loopL,
                                      const std::array<unsigned, c_operand_max_agu>& loopO,
                                      unsigned descIdx)
{
    descPtr[descIdx].header.partialHeightLoopS = loopS[descIdx];
    for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
    {
        if (aguIdx == Mme::e_mme_local)
        {
            descPtr[descIdx].header.partialHeightLoopLLocal = loopL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            descPtr[descIdx].header.partialHeightLoopOLocal = loopO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
        }
        else
        {
            descPtr[descIdx].header.partialHeightLoopLRemote = loopL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            descPtr[descIdx].header.partialHeightLoopORemote = loopO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
        }
    }
}

void AguConvDedwConfig::setPartialHeightLoops(Mme::Desc* descPtr,
                                              const std::array<unsigned, c_operand_max_agu>& loopS,
                                              const std::array<unsigned, c_operand_max_agu>& loopL,
                                              const std::array<unsigned, c_operand_max_agu>& loopO,
                                              unsigned descIdx)
{
    descPtr[descIdx].header.partialHeightLoopS = loopS[0];
    descPtr[descIdx].header.partialHeightLoopLLocal = loopS[0];
    descPtr[descIdx].header.partialHeightLoopLRemote = loopS[0];
    descPtr[descIdx].header.partialHeightLoopLLocal = loopL[0];
    descPtr[descIdx].header.partialHeightLoopLRemote = loopL[0];
}

void AguConvConfig::setDescAgus(Mme::Desc* descPtr,
                                const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                unsigned descIdx)
{
    descPtr[descIdx].aguS = aguS[descIdx];

    for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
    {
        descPtr[descIdx].aguL[aguIdx] = aguL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
        descPtr[descIdx].aguO[aguIdx] = aguO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
    }
}

void AguConvConfig::setSpPos()
{
    for (unsigned i = 0; i < m_geoAttrSPtr->m_totalAports; i++)
    {
        auto roi = getRoi();
        spPosToCoord(roi.spBase + i, &roi.size[1], m_spPos[i]);
    }
}
void AguConvConfig::setSpPosB()
{
    const MmeLayerParams& params = getParams();
    auto spView = getRecipe().curSp();
    for (unsigned i = 0; i < m_geoAttrSPtr->m_totalAports; i++)
    {
        spPosToCoord(spView.viewBase, &params.getOperand(e_mme_op_b).sizes[1], &m_spPosB[1]);
    }
}
static void resetAssociatedDims(Mme::Desc& localDesc)
{
    // Initialize associatedDims fields
    for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        localDesc.conv.associatedDims[i].dimS = Mme::c_mme_max_tensor_dims;
        localDesc.conv.associatedDims[i].dimL = Mme::c_mme_max_tensor_dims;
        localDesc.conv.associatedDims[i].dimO = Mme::c_mme_max_tensor_dims;
    }
    localDesc.outerLoop.associatedDims.dimS = Mme::c_mme_max_tensor_dims;
    localDesc.outerLoop.associatedDims.dimL = Mme::c_mme_max_tensor_dims;
    localDesc.outerLoop.associatedDims.dimO = Mme::c_mme_max_tensor_dims;
}

void AguConvConfig::setAgu(DescGroup& descGroup)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();

    setSpPos();
    setSpPosB();  // Relevant only to fwd & dedx

    // Use localDesc to set all the fields that are common among all descriptors
    Mme::Desc localDesc = descGroup.desc[0];

    // ====== First of AGU =================================
    std::array<Mme::MmeAguCoreDesc, c_operand_max_agu> aguS {0}, aguL {0}, aguO {0};
    std::array<unsigned, c_operand_max_agu> loopS;
    std::array<unsigned, c_operand_max_agu> loopL;
    std::array<unsigned, c_operand_max_agu> loopO;

    resetAssociatedDims(localDesc);
    setNumIterationsMinus1(localDesc);

    setDescFirstAguA(aguS, localDesc);
    setDescRestAguA(aguS, localDesc);
    setDescFirstAguB(aguL, localDesc);
    setDescRestAguB(aguL, localDesc);
    setDescFirstOfAguCout(aguO, localDesc);
    setDescRestAguCout(aguO, localDesc);

    setLoopDimsAndSizes(localDesc);
    setSpatialSize(localDesc, false);
    setLoops(aguO, loopS, loopL, loopO);

    if (gp->isTransO())
    {
        transposeDesc(&localDesc);
        std::swap(aguS, aguL);
        std::swap(gp->m_totalAports, gp->m_totalBports);
        std::swap(loopS, loopL);
    }

    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        descGroup.desc[descIdx] = localDesc;
        setDescAgus(descGroup.desc, aguS, aguL, aguO, descIdx);
        setPartialHeightLoops(descGroup.desc, loopS, loopL, loopO, descIdx);
    }
}

//===========================================================================
//========= AGU Memset config ===============================================
// todo: this setting is done to match the legacy code
// todo [SW-51591] Check and remove redundant settings
void AguMemsetConfig::resetTensorAgus(Mme::Desc& localDesc)
{
    const MmeLayerParams& params = getParams();
    MME_ASSERT(params.opType == e_mme_dedx, "Agu Memset supports dedx only");

    for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
    {
        localDesc.tensorS.loopStride[i] = 0;
        localDesc.tensorL.loopStride[i] = 0;
        localDesc.tensorO.loopStride[i] = 0;
        localDesc.tensorS.validElements[i] = 0;
        localDesc.tensorL.validElements[i] = 0;
        localDesc.tensorO.validElements[i] = 1;

        if (i < Mme::c_mme_max_tensor_dims - 1)
        {
            localDesc.tensorS.roiSize[i] = 1;
            localDesc.tensorL.roiSize[i] = 1;
            localDesc.tensorO.roiSize[i] = 1;
        }
        if (i > 0)
        {
            localDesc.tensorS.spatialStrides[i - 1] = 0;
            localDesc.tensorL.spatialStrides[i - 1] = 0;
            localDesc.tensorO.spatialStrides[i - 1] = 1;
        }
    }
}

void AguMemsetConfig::setTensorLRoiSize(Mme::Desc& localDesc)
{
    auto gp = getGeoAttr();

    // tensor B: K (dim 0)
    localDesc.tensorL.roiSize[0] = (getRoi().size[0] % gp->m_subMatrixWidth);
    localDesc.tensorL.roiSize[0] = localDesc.tensorL.roiSize[0] ? localDesc.tensorL.roiSize[0] : gp->m_subMatrixWidth;
}

void AguMemsetConfig::setAgu(DescGroup& descGroup)
{
    const MmeLayerParams& params = getParams();
    auto gp = getGeoAttr();

    setSpPos();

    // Use localDesc to set all the fields that are common among all descriptors
    Mme::Desc localDesc = descGroup.desc[0];

    // ====== First of AGU =================================
    // set default values
    std::array<Mme::MmeAguCoreDesc, c_operand_max_agu> aguS {0};
    std::array<Mme::MmeAguCoreDesc, c_operand_max_agu> aguL {0};
    std::array<Mme::MmeAguCoreDesc, c_operand_max_agu> aguO {0};

    resetTensorAgus(localDesc);
    resetAssociatedDims(localDesc);
    setNumIterationsMinus1(localDesc);
    setSpatialSize(localDesc, true);
    setDescFirstOfAguCout(aguO, localDesc);
    setDescRestAguCout(aguO, localDesc);
    setTensorLRoiSize(localDesc);

    std::array<unsigned, c_operand_max_agu> loopS;
    std::array<unsigned, c_operand_max_agu> loopL;
    std::array<unsigned, c_operand_max_agu> loopO;

    setLoops(aguO, loopS, loopL, loopO);

    // todo Alon: This code is duplication from Conv. Move to a common function.

    localDesc.tensorO.loopStride[0] = gp->m_matrixWidth;
    setLoopDim(&localDesc, params.strategy.pattern, LOOP_K, OP_O, 0);
    unsigned denseLoopSize = ((getRoi().size[0] + gp->m_matrixWidth - 1) / gp->m_matrixWidth) - 1;
    setLoopSize(&localDesc, params.strategy.pattern, LOOP_K, denseLoopSize);

    if (gp->isTransO())
    {
        transposeDesc(&localDesc);
        std::swap(aguS, aguL);
        std::swap(gp->m_totalAports, gp->m_totalBports);
        std::swap(loopS, loopL);
    }

    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        descGroup.desc[descIdx] = localDesc;
        descGroup.desc[descIdx].aguS = aguS[descIdx];
        descGroup.desc[descIdx].header.partialHeightLoopS = loopS[descIdx];
        localDesc.header.partialHeightLoopS = getLoopMask(params.strategy.pattern, LOOP_SPATIAL);

        for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
        {
            descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            if (aguIdx == Mme::e_mme_local)
            {
                descGroup.desc[descIdx].header.partialHeightLoopLLocal =
                    loopL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].header.partialHeightLoopOLocal =
                    loopO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            }
            else
            {
                descGroup.desc[descIdx].header.partialHeightLoopLRemote =
                    loopL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].header.partialHeightLoopORemote =
                    loopO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            }
        }
    }
}

}  // namespace gaudi
