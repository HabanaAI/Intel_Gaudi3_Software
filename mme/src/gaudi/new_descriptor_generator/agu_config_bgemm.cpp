#include "include/gaudi/new_descriptor_generator/agu_config.h"
#include "include/mme_assert.h"
#include "include/gaudi/new_descriptor_generator/mme_common.h"
#include "src/gaudi/gaudi_geo_attr.h"

using namespace MmeCommon;

namespace gaudi
{
void AguBGemmConfig::setDescFirstAguATranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                               Mme::Desc* desc)
{
    // width (in conv this is the common dim)
    const MultiDimSubView& batchView = getRecipe().curNonSpatial();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);

    const unsigned firstBatchViewIdx = desc->header.transL ? 0 : 1;
    unsigned commonDim = batchView.sizes[firstBatchViewIdx];

    //------ width --------------------------
    // roiSize is the amount of data we push to the mme
    desc->tensorS.roiSize[DIM_C] = commonDim;
    // valid elements is the actual data we read from memory. The difference beyond roiSize will be read as padding
    desc->tensorS.validElements[DIM_C] = std::max(tensorViewA.sizes[0], commonDim);
    desc->tensorS.loopStride[DIM_C] = 0;  // the common dim is handled by the gemm loop

    //------ height (in conv it is width) --------------
    desc->tensorS.spatialStrides[DIM_W - 1] = tensorViewA.strides[1];
    desc->tensorS.roiSize[DIM_W] = aRoiSizes[DIM_W] * tensorViewA.strides[1];
    desc->tensorS.validElements[DIM_W] = tensorViewA.sizes[1] * tensorViewA.strides[1];
    desc->tensorS.loopStride[DIM_W] = m_geoAttrSPtr->m_geoTotalElemHeight * tensorViewA.strides[1];

    for (int j = 0; j < c_operand_max_agu; j++)
    {
        aguS[j].roiBaseOffset[DIM_C] = batchView.bases[WEIGHT_DIM_C];
        aguS[j].startOffset[DIM_W - 1] = 0;  // we do not split on the spatial dim
        aguS[j].roiBaseOffset[DIM_W] =
            spSubview.viewBase * tensorViewA.strides[1];  // in rows (because A is transposed)
    }

    if (m_reuseAttr.aPartialHeightStepsNr > 1 || m_geoAttrSPtr->isTransO())
        // due to HW limitation we cant use partial height, pad to port size
        desc->tensorS.spatialSizeMinus1 = m_geoAttrSPtr->m_portElemHeight - 1;
    else
    {
        desc->tensorS.spatialSizeMinus1 = m_geoAttrSPtr->m_isAInterleaved
                                              ?
                                              // interleave: num steps is the height divided by num ports
                                              m_reuseAttr.lastAPartialHeightStepSize / m_geoAttrSPtr->m_totalAports - 1
                                              :
                                              // No interleave: Use last step
                                              m_reuseAttr.lastAPartialHeightStepSize - 1;
    }
}

void AguBGemmConfig::setDescFirstAguANonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                                  Mme::Desc* desc)
{
    const MultiDimSubView& batchView = getRecipe().curNonSpatial();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);

    const unsigned firstBatchViewIdx = desc->header.transL ? 0 : 1;
    unsigned commonDim = batchView.sizes[firstBatchViewIdx];

    desc->tensorS.roiSize[0] = std::min(tensorViewA.sizes[0], m_geoAttrSPtr->m_portElemHeight);
    desc->tensorS.validElements[0] = tensorViewA.sizes[0];
    desc->tensorS.loopStride[0] = m_geoAttrSPtr->m_geoTotalElemHeight;

    desc->tensorS.spatialStrides[1 - 1] = tensorViewA.strides[1];
    desc->tensorS.roiSize[1] = commonDim * tensorViewA.strides[1];
    desc->tensorS.validElements[1] = getZeroCD() ? 0 : commonDim * tensorViewA.strides[1];
    desc->tensorS.loopStride[1] = 0;

    for (int core = 0; core < Mme::e_mme_local_and_remote; core++)
    {
        aguS[core].roiBaseOffset[0] = spSubview.viewBase;
        aguS[core].startOffset[1 - 1] = 0;
        aguS[core].roiBaseOffset[1] = batchView.bases[1] * tensorViewA.strides[1];
    }

    desc->tensorS.spatialSizeMinus1 = commonDim - 1;
}

void AguBGemmConfig::setDescRestAguATranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                              bool isSouth,
                                              Mme::Desc* desc)
{
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    unsigned verticalOffset = m_geoAttrSPtr->m_subMatrixWidth * tensorViewA.strides[1];  // 64 rows

    if (m_geoAttrSPtr->m_isAInterleaved)
    {
        // When we read A inteleaved, spatial stride is 2 rows, and north is set to start at row 1
        desc->tensorS.spatialStrides[DIM_W - 1] *= 2;
        // Set the offset of north's agu to start at row 1
        aguS[Mme::e_mme_remote].roiBaseOffset[DIM_W] += (isSouth ? 0 : tensorViewA.strides[1]);
    }
    else
    {
        switch (m_geoAttrSPtr->m_totalAports)
        {
            case 1:
                aguS[Mme::e_mme_local].roiBaseOffset[DIM_W] = aguS[Mme::e_mme_remote].roiBaseOffset[DIM_W] = 0;
                break;
            case 2:
                aguS[Mme::e_mme_local].roiBaseOffset[DIM_W] = 0;
                aguS[Mme::e_mme_remote].roiBaseOffset[DIM_W] = verticalOffset;
                break;
            case 4:
                aguS[Mme::e_mme_local].roiBaseOffset[DIM_W] = isSouth ? 0 : 2 * verticalOffset;
                aguS[Mme::e_mme_remote].roiBaseOffset[DIM_W] =
                    aguS[Mme::e_mme_local].roiBaseOffset[DIM_W] + verticalOffset;
                break;
        }
    }
}

void AguBGemmConfig::setDescRestAguANonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                                 bool isSouth,
                                                 Mme::Desc* desc)
{
    unsigned horizontalOffset = m_geoAttrSPtr->m_subMatrixWidth;
    // If we have only 1 port, all offsets are zero.
    // If we have 2 ports (1x), north reads the lower part.
    // In 2x, north reads the next matrix, so the offset is zero
    switch (m_geoAttrSPtr->m_totalAports)
    {
        case 1:
            aguS[Mme::e_mme_local].roiBaseOffset[0] = aguS[Mme::e_mme_remote].roiBaseOffset[0] = 0;
            break;
        case 2:
            aguS[Mme::e_mme_local].roiBaseOffset[0] = 0;
            aguS[Mme::e_mme_remote].roiBaseOffset[0] = horizontalOffset;
            break;
        case 4:
            aguS[Mme::e_mme_local].roiBaseOffset[0] = isSouth ? 0 : 2 * horizontalOffset;
            aguS[Mme::e_mme_remote].roiBaseOffset[0] = aguS[Mme::e_mme_local].roiBaseOffset[0] + horizontalOffset;
            break;
    }
}
//=============== Tensor B =============================
void AguBGemmConfig::setDescFirstAguBNonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                                  Mme::Desc* desc)
{
    const MultiDimSubView& batchView = getRecipe().curNonSpatial();
    const SingleDimSubView& fcdView = getRecipe().curFcd();
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);

    const unsigned firstBatchViewIdx = desc->header.transL ? 0 : 1;
    unsigned commonDim = batchView.sizes[firstBatchViewIdx];
    //------ width --------------------------
    desc->tensorL.roiSize[DIM_K] = std::min(m_geoAttrSPtr->m_inputPortElemWidth, tensorViewB.sizes[DIM_K]);

    if (m_geoAttrSPtr->isTransO())
    {
        desc->tensorL.roiSize[DIM_K] = m_geoAttrSPtr->m_portElemHeight;
    }
    desc->tensorL.validElements[DIM_K] = tensorViewB.sizes[DIM_K];
    desc->tensorL.loopStride[DIM_K] = m_geoAttrSPtr->m_geoTotalElemWidth;
    aguL[Mme::e_mme_local].roiBaseOffset[DIM_K] = fcdView.viewBase;

    //------ height --------------------------
    desc->tensorL.spatialStrides[WEIGHT_DIM_C - 1] = tensorViewB.strides[WEIGHT_DIM_C];
    desc->tensorL.roiSize[WEIGHT_DIM_C] = commonDim * tensorViewB.strides[WEIGHT_DIM_C];
    desc->tensorL.validElements[WEIGHT_DIM_C] = getZeroCD() ? 0 : commonDim * tensorViewB.strides[WEIGHT_DIM_C];
    desc->tensorL.loopStride[WEIGHT_DIM_C] = 0;

    aguL[Mme::e_mme_local].startOffset[WEIGHT_DIM_C - 1] = 0;
    aguL[Mme::e_mme_local].roiBaseOffset[WEIGHT_DIM_C] =
        batchView.bases[WEIGHT_DIM_C] * tensorViewB.strides[WEIGHT_DIM_C];

    desc->tensorL.spatialSizeMinus1 = commonDim - 1;  // Number of rows to read
}

void AguBGemmConfig::setDescFirstAguBTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                               Mme::Desc* desc)
{
    const MultiDimSubView& batchView = getRecipe().curNonSpatial();
    const SingleDimSubView& fcdSubview = getRecipe().curFcd();
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    bool transO = m_geoAttrSPtr->isTransO();
    const unsigned firstBatchViewIdx = desc->header.transL ? 0 : 1;
    unsigned commonDim = batchView.sizes[firstBatchViewIdx];

    desc->tensorL.roiSize[0] = commonDim;
    desc->tensorL.validElements[0] = commonDim;
    desc->tensorL.loopStride[0] = 0;
    aguL[Mme::e_mme_local].roiBaseOffset[0] = batchView.bases[1];

    desc->tensorL.spatialStrides[1 - 1] = tensorViewB.strides[1];
    desc->tensorL.roiSize[1] = fcdSubview.viewSize * tensorViewB.strides[1];
    desc->tensorL.validElements[1] = fcdSubview.viewSize * tensorViewB.strides[1];
    desc->tensorL.loopStride[1] = m_geoAttrSPtr->m_geoTotalElemWidth * tensorViewB.strides[WEIGHT_DIM_C];
    aguL[Mme::e_mme_local].startOffset[1 - 1] = 0;
    aguL[Mme::e_mme_local].roiBaseOffset[1] = fcdSubview.viewBase * tensorViewB.strides[1];

    // When transO is set, B will be mapped to A to which due to hw limitation we need to pad
    // spatialSizeMinus1 to the port width. So the setting is dependent on transO...
    desc->tensorL.spatialSizeMinus1 =
        transO ? m_geoAttrSPtr->m_inputPortElemWidth - 1
               : std::min(getRecipe().getFcdSubviews()[0].viewSize, m_geoAttrSPtr->m_inputPortElemWidth) - 1;
}

void AguBGemmConfig::setDescRestAguBNonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                                 bool isSouth,
                                                 Mme::Desc* desc)
{
    unsigned horizontalOffset = m_geoAttrSPtr->m_subMatrixWidth;

    memcpy(&aguL[Mme::e_mme_remote], &aguL[Mme::e_mme_local], sizeof(aguL[Mme::e_mme_remote]));
    switch (m_geoAttrSPtr->m_totalBports)
    {
        case 1:  // Only one port, offsets are all zero
            aguL[Mme::e_mme_local].roiBaseOffset[DIM_K] = 0;
            aguL[Mme::e_mme_remote].roiBaseOffset[DIM_K] = 0;
            break;
        case 2:  // Two ports. Masters at zero, slaves in offset
            aguL[Mme::e_mme_local].roiBaseOffset[DIM_K] = 0;
            aguL[Mme::e_mme_remote].roiBaseOffset[DIM_K] = horizontalOffset;
            break;
        case 4:  // Four ports.
            aguL[Mme::e_mme_local].roiBaseOffset[DIM_K] = isSouth ? 0 : 2 * horizontalOffset;
            aguL[Mme::e_mme_remote].roiBaseOffset[DIM_K] =
                aguL[Mme::e_mme_local].roiBaseOffset[DIM_K] + horizontalOffset;
            break;
        default:
            MME_ASSERT(0, "Invalid number of ports");
    }
}

void AguBGemmConfig::setDescRestAguBTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                              bool isSouth,
                                              Mme::Desc* desc)
{
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);

    if (m_geoAttrSPtr->m_totalBports > 1)
    {
        memcpy(&aguL[Mme::e_mme_remote], &aguL[Mme::e_mme_local], sizeof(aguL[Mme::e_mme_remote]));

        // When we have 4 ports, then north master starts 128 rows ahead
        if (!isSouth && m_geoAttrSPtr->m_totalBports > 2)
        {
            MME_ASSERT(m_geoAttrSPtr->m_totalBports == 4, "expected 4 B ports");
            aguL[Mme::e_mme_local].roiBaseOffset[DIM_W] = 2 * m_geoAttrSPtr->m_subMatrixWidth * tensorViewB.strides[1];
        }
        aguL[Mme::e_mme_remote].roiBaseOffset[DIM_W] =
            aguL[Mme::e_mme_local].roiBaseOffset[DIM_W] + m_geoAttrSPtr->m_subMatrixWidth * tensorViewB.strides[1];
    }
}
//===================== C Tensor =============================
void AguBGemmConfig::setDescFirstOfAguCoutAndBatchLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                                        std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                                        std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                                        Mme::Desc* desc)
{
    const MultiDimSubView& batchView = getRecipe().curNonSpatial();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const SingleDimSubView& fcdView = getRecipe().curFcd();
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    const MmeCommon::SizeArray& aRoiSizes = m_recipe.getRoiSizes(e_mme_op_a);
    const MmeCommon::SizeArray& bRoiSizes = m_recipe.getRoiSizes(e_mme_op_b);
    const MmeCommon::SizeArray& cRoiSizes = m_recipe.getRoiSizes(e_mme_op_c);

    bool transO = m_geoAttrSPtr->isTransO();

    if (!transO)
    {
        //------ width --------------------------
        desc->tensorO.roiSize[DIM_K] = std::min(m_geoAttrSPtr->m_inputPortElemWidth, tensorViewC.sizes[DIM_K]);

        desc->tensorO.validElements[DIM_K] = tensorViewC.sizes[DIM_K];
        desc->tensorO.loopStride[DIM_K] = m_geoAttrSPtr->m_geoTotalElemWidth;
        aguO[Mme::e_mme_local].roiBaseOffset[DIM_K] = fcdView.viewBase;

        //------ height --------------------------
        desc->tensorO.roiSize[DIM_W] = cRoiSizes[DIM_W] * tensorViewC.strides[1];
        desc->tensorO.validElements[DIM_W] = tensorViewC.sizes[DIM_W] * tensorViewC.strides[1];
        desc->tensorO.loopStride[DIM_W] = m_geoAttrSPtr->m_geoTotalElemHeight * tensorViewC.strides[1];
        desc->tensorO.spatialStrides[DIM_W - 1] = tensorViewC.strides[1];
        aguO[Mme::e_mme_local].roiBaseOffset[1] = spSubview.viewBase * tensorViewC.strides[1];
    }
    else
    {
        //------ Height --------------------------
        desc->tensorO.roiSize[DIM_W] = cRoiSizes[DIM_W] * tensorViewC.strides[DIM_W];
        desc->tensorO.validElements[DIM_W] = tensorViewC.sizes[DIM_W] * tensorViewC.strides[DIM_W];
        desc->tensorO.loopStride[DIM_W] = m_geoAttrSPtr->m_geoTotalElemHeight * tensorViewC.strides[DIM_W];
        aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = fcdView.viewBase * tensorViewC.strides[1];

        //------ Width --------------------------
        desc->tensorO.roiSize[DIM_K] = m_geoAttrSPtr->m_inputPortElemWidth;
        desc->tensorO.validElements[DIM_K] = tensorViewC.sizes[DIM_K];
        desc->tensorO.loopStride[DIM_K] = std::min(tensorViewC.sizes[DIM_K], m_geoAttrSPtr->m_geoTotalElemWidth);
        desc->tensorO.spatialStrides[0] = tensorViewC.strides[1];

        aguO[Mme::e_mme_local].roiBaseOffset[0] = spSubview.viewBase;
    }

    //------- Batch dims -----------------------
    constexpr unsigned maxBatchDim = c_batchDimNr + 2;
    for (unsigned batchDim = 2; batchDim < maxBatchDim; ++batchDim)
    {
        // --------------------- Tensor S -------------------------

        desc->tensorS.spatialStrides[batchDim - 1] = tensorViewA.strides[batchDim];

        if (batchDim <= 3)
        {
            if (batchDim == m_newGeoAttrSPtr->getConcurrentDim())
            {
                unsigned alignedBatchSize =
                    ((tensorViewA.sizes[batchDim] + m_geoAttrSPtr->m_concurrentLevel - 1) / m_geoAttrSPtr->m_concurrentLevel) *
                    m_geoAttrSPtr->m_concurrentLevel;
                desc->tensorS.roiSize[batchDim] = tensorViewA.strides[batchDim] * alignedBatchSize;
            }
            else
            {
                desc->tensorS.roiSize[batchDim] = tensorViewA.strides[batchDim] * tensorViewA.sizes[batchDim];
            }
        }

        desc->tensorS.validElements[batchDim] = tensorViewA.strides[batchDim] * tensorViewA.sizes[batchDim];
        // In case of opA's broadcast there will be no movement in the batch dim
        bool broadcastA =
            (tensorViewA.sizes[batchDim] == 1) && (tensorViewA.sizes[batchDim] != tensorViewC.sizes[batchDim]);
        desc->tensorS.loopStride[batchDim] = broadcastA ? 0 : tensorViewA.strides[batchDim];
        desc->aguS.roiBaseOffset[batchDim] = 0;

        // --------------------- Tensor L -------------------------
        desc->tensorL.spatialStrides[batchDim - 1] = tensorViewB.strides[batchDim];
        if (batchDim <= 3)
        {
            if (batchDim == m_newGeoAttrSPtr->getConcurrentDim())
            {
                unsigned alignedBatchSize =
                    ((tensorViewB.sizes[batchDim] + m_geoAttrSPtr->m_concurrentLevel - 1) / m_geoAttrSPtr->m_concurrentLevel) *
                    m_geoAttrSPtr->m_concurrentLevel;
                desc->tensorL.roiSize[batchDim] = tensorViewB.strides[batchDim] * alignedBatchSize;
            }
            else
            {
                desc->tensorL.roiSize[batchDim] = tensorViewB.strides[batchDim] * tensorViewB.sizes[batchDim];
            }
        }

        desc->tensorL.validElements[batchDim] = tensorViewB.strides[batchDim] * tensorViewB.sizes[batchDim];
        // In case of opB's broadcast there will be no movement in the batch dim
        bool broadcastB =
            (tensorViewB.sizes[batchDim] == 1) && (tensorViewB.sizes[batchDim] != tensorViewC.sizes[batchDim]);
        desc->tensorL.loopStride[batchDim] = broadcastB ? 0 : tensorViewB.strides[batchDim];
        desc->aguL[Mme::e_mme_local].roiBaseOffset[batchDim] = 0;

        // --------------------- Tensor O -------------------------
        desc->tensorO.spatialStrides[batchDim - 1] = tensorViewC.strides[batchDim];

        if (batchDim <= 3)
        {
            if (batchDim == m_newGeoAttrSPtr->getConcurrentDim())
            {
                unsigned alignedBatchSize =
                    ((tensorViewC.sizes[batchDim] + m_geoAttrSPtr->m_concurrentLevel - 1) / m_geoAttrSPtr->m_concurrentLevel) *
                    m_geoAttrSPtr->m_concurrentLevel;
                desc->tensorO.roiSize[batchDim] = tensorViewC.strides[batchDim] * alignedBatchSize;
            }
            else
            {
                desc->tensorO.roiSize[batchDim] = tensorViewC.strides[batchDim] * tensorViewC.sizes[batchDim];
            }
        }

        desc->tensorO.validElements[batchDim] = tensorViewC.strides[batchDim] * tensorViewC.sizes[batchDim];
        desc->tensorO.loopStride[batchDim] = tensorViewC.strides[batchDim];

        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].startOffset[batchDim - 1] = tensorViewA.strides[batchDim] * batchView.bases[batchDim];
            aguS[j].roiBaseOffset[batchDim] = 0;

            aguL[Mme::e_mme_local].startOffset[batchDim - 1] =
                tensorViewB.strides[batchDim] * batchView.bases[batchDim];

            aguL[Mme::e_mme_local].roiBaseOffset[batchDim] = 0;

            aguO[Mme::e_mme_local].startOffset[batchDim - 1] = 0;
            aguO[Mme::e_mme_remote].startOffset[batchDim - 1] = 0;
            aguO[Mme::e_mme_local].roiBaseOffset[batchDim] = 0;
        }
    }
}

void AguBGemmConfig::setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                        bool isSouth,
                                        Mme::Desc* desc)
{
    const MmeTensorView& tensorViewC = m_recipe.getOperand(e_mme_op_c);

    if (m_geoAttrSPtr->m_totalBports > 1)
    {
        memcpy(&aguO[Mme::e_mme_remote], &aguO[Mme::e_mme_local], sizeof(aguO[Mme::e_mme_remote]));
    }

    bool transO = m_geoAttrSPtr->isTransO();
    unsigned horizointalOffset = m_geoAttrSPtr->m_subMatrixWidth * tensorViewC.strides[DIM_K];
    unsigned verticalOffset = m_geoAttrSPtr->m_subMatrixHeight * tensorViewC.strides[1];

    switch (m_geoAttrSPtr->m_totalBports)
    {
        case 1:  // Only one port. Horizontal offset is zero.
            aguO[Mme::e_mme_local].roiBaseOffset[DIM_K] = 0;
            aguO[Mme::e_mme_remote].roiBaseOffset[DIM_K] = 0;
            break;
        case 2:  // Two ports. Masters at offset zero, slaves are to the right of masters.
            aguO[Mme::e_mme_local].roiBaseOffset[DIM_K] = 0;
            aguO[Mme::e_mme_remote].roiBaseOffset[DIM_K] = horizointalOffset;
            break;
        case 4:  // Four ports. Order: S-Master, S-Slave, N-Master, N-Slave
            aguO[Mme::e_mme_local].roiBaseOffset[DIM_K] = isSouth ? 0 : 2 * horizointalOffset;
            aguO[Mme::e_mme_remote].roiBaseOffset[DIM_K] =
                aguO[Mme::e_mme_local].roiBaseOffset[DIM_K] + horizointalOffset;
            break;
        default:
            MME_ASSERT(0, "Invalid number of ports");
    }

    if (desc->header.transS)
    {
        if (m_geoAttrSPtr->m_isAInterleaved)  // 2 ports. South vertical offset is 0, north vertical offset is 1 row. Stride
                                         // is doubled
        {
            aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] += (isSouth ? 0 : tensorViewC.strides[1]);
            aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] += (isSouth ? 0 : tensorViewC.strides[1]);
            desc->tensorO.spatialStrides[DIM_W-1] *= 2;
        }
        else
            switch (m_geoAttrSPtr->m_totalAports)
            {
                case 1:  // Only 1 port. Offsets are zero
                    aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = 0;
                    aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] = 0;
                    break;
                case 2:  // 2 ports. Masters are aligned at offset zero. Slave is below master.
                    aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = 0;
                    aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] = verticalOffset;
                    break;
                case 4:  // 4 ports. South master is below North slave
                    aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = isSouth ? 0 : 2 * verticalOffset;
                    aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] =
                        aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] + verticalOffset;
                    break;
                default:
                    MME_ASSERT(0, "Invalid number of ports");
            }

        if (transO)
        {
            desc->tensorO.spatialSizeMinus1 = m_geoAttrSPtr->m_cPortElemWidth - 1;
        }
        else
        {
            if (m_reuseAttr.aPartialHeightStepsNr > 1)
            {
                // due to HW limitation we cant use partial height, pad to port size
                desc->tensorO.spatialSizeMinus1 = m_geoAttrSPtr->m_portElemHeight - 1;
            }
            else
            {
                desc->tensorO.spatialSizeMinus1 =
                    m_geoAttrSPtr->m_isAInterleaved
                        ?
                        // When we read interleaved, num steps is the height divided by num ports
                        (m_reuseAttr.lastAPartialHeightStepSize / m_geoAttrSPtr->m_totalAports) - 1
                        : m_reuseAttr.lastAPartialHeightStepSize - 1;  // Otherwise, use the last step
            }
        }
    }
    else
    {
        // If we have more than 1 port, we have 2 cases:
        // - With 2 ports, north local and slave are aligned and below south
        // - With 4 ports, north is below south, and slave is below master
        switch (m_geoAttrSPtr->m_totalAports)
        {
            case 1:
                aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = 0;
                aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] = 0;
                break;
            case 2:
                if (transO)
                {
                    aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = 0;
                    aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] = verticalOffset;
                }
                else
                {
                    aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] += (isSouth ? 0 : verticalOffset);
                    aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] += (isSouth ? 0 : verticalOffset);
                }
                break;
            case 4:
                aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] = (isSouth ? 0 : 2 * verticalOffset);  // north below south
                aguO[Mme::e_mme_remote].roiBaseOffset[DIM_W] =
                    aguO[Mme::e_mme_local].roiBaseOffset[DIM_W] + verticalOffset;  // remote always below local
                break;
            default:
                MME_ASSERT(0, "Invalid number of ports");
        }

        desc->tensorO.spatialSizeMinus1 =
            std::min(getRecipe().getSpSubviews()[0].viewSize, m_geoAttrSPtr->m_portElemHeight) - 1;
    }
}

void AguBGemmConfig::doubleWorkForBatchMode2xw(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                               std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                               std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                               bool isSouth,
                                               Mme::Desc* desc)
{
    // For all tensors, we set the roiBaseOffset to point on the next batch (loopStride[m_newGeoAttrSPtr.getConcurrentDim()])
    // and then we double the loopStride[m_newGeoAttrSPtr.getConcurrentDim()]

    aguS[Mme::e_mme_local].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()] += (isSouth ? 0 : desc->tensorS.loopStride[m_newGeoAttrSPtr->getConcurrentDim()]);
    aguS[Mme::e_mme_remote].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()] += (isSouth ? 0 : desc->tensorS.loopStride[m_newGeoAttrSPtr->getConcurrentDim()]);
    aguL[Mme::e_mme_local].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()] = (isSouth ? 0 : desc->tensorL.loopStride[m_newGeoAttrSPtr->getConcurrentDim()]);
    aguL[Mme::e_mme_remote].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()] = aguL[Mme::e_mme_local].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()];
    aguO[Mme::e_mme_local].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()] = (isSouth ? 0 : desc->tensorO.loopStride[m_newGeoAttrSPtr->getConcurrentDim()]);
    aguO[Mme::e_mme_remote].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()] = aguO[Mme::e_mme_local].roiBaseOffset[m_newGeoAttrSPtr->getConcurrentDim()];

    desc->tensorS.loopStride[m_newGeoAttrSPtr->getConcurrentDim()] *= m_geoAttrSPtr->m_concurrentLevel;
    desc->tensorL.loopStride[m_newGeoAttrSPtr->getConcurrentDim()] *= m_geoAttrSPtr->m_concurrentLevel;
    desc->tensorO.loopStride[m_newGeoAttrSPtr->getConcurrentDim()] *= m_geoAttrSPtr->m_concurrentLevel;
}

void AguBGemmConfig::doubleWorkForBatchMode(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                            std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                            std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                            bool isSouth,
                                            Mme::Desc* desc)
{
    switch (m_geoAttrSPtr->m_batchMode)
    {
        case MmeBatchMode::mme_batch_none:
            return;
        case MmeBatchMode::mme_batch_2xw:
        case MmeBatchMode::mme_batch_2xh:
            doubleWorkForBatchMode2xw(aguS, aguL, aguO, isSouth, desc);
            return;
        default:
            MME_ASSERT(0, "invalid batch mode");
    }
}

static void setAssociatedDimAndSize(EMmeLoopMask mask,
                                    unsigned size,
                                    unsigned dimA,
                                    unsigned dimB,
                                    unsigned dimOut,
                                    Mme::Desc* desc)
{
    MME_ASSERT(size - 1 <= std::numeric_limits<uint8_t>::max(), "");

    Mme::MmeAssociatedDims* assocDim {};
    switch (mask)
    {
        case e_mme_conv_loop_0:
            desc->conv.kernelSizeMinus1.dim[0] = size - 1;
            assocDim = &desc->conv.associatedDims[0];
            break;
        case e_mme_conv_loop_1:
            desc->conv.kernelSizeMinus1.dim[1] = size - 1;
            assocDim = &desc->conv.associatedDims[1];
            break;
        case e_mme_conv_loop_2:
            assocDim = &desc->conv.associatedDims[2];
            desc->conv.kernelSizeMinus1.dim[2] = size - 1;
            break;
        case e_mme_conv_loop_3:
            desc->conv.kernelSizeMinus1.dim[3] = size - 1;
            assocDim = &desc->conv.associatedDims[3];
            break;
        case e_mme_outer_loop:
            desc->outerLoop.sizeMinus1 = size - 1;
            assocDim = &desc->outerLoop.associatedDims;
            break;
        default:
            MME_ASSERT(0, "unsupported EMmeLoopMask");
    }
    assocDim->dimS = dimA;
    assocDim->dimL = dimB;
    assocDim->dimO = dimOut;
    assocDim->reserved = 0;
}

void AguBGemmConfig::setAssociatedDims(Mme::Desc& desc)
{
    const MmeLayerParams& params = getParams();
    const MmeTensorView& tensorViewA = m_recipe.getOperand(e_mme_op_a);
    const MmeTensorView& tensorViewB = m_recipe.getOperand(e_mme_op_b);

    const MultiDimSubView& batchView = getRecipe().curNonSpatial();
    // If every pole or every core processes a different gemm, then number of loops per pole/core is smaller
    unsigned firstBatchDimLoopsNr = div_round_up(batchView.sizes[2], m_geoAttrSPtr->m_concurrentLevel);
    unsigned batchSizes[c_batchDimNr] = {firstBatchDimLoopsNr, batchView.sizes[3], batchView.sizes[4]};

    // Set filter/batch
    EMmeLoopMask currentFilterMask = pattern2LoopMask(params.getPattern(), EMmeLoopDim::dim_b);
    for (unsigned batchDim = 0; batchDim < c_batchDimNr; ++batchDim)
    {
        unsigned int aSize = tensorViewA.sizes[batchDim + 2];
        unsigned int bSize = tensorViewB.sizes[batchDim + 2];
        MME_ASSERT(currentFilterMask <= e_mme_outer_loop, "invalid filter mask");
        // In case of broadcast - we disable the loop on that operand
        setAssociatedDimAndSize(currentFilterMask,
                                batchSizes[batchDim],
                                (aSize == 1) ? Mme::c_mme_max_tensor_dims : batchDim + 2,
                                (bSize == 1) ? Mme::c_mme_max_tensor_dims : batchDim + 2,
                                batchDim + 2,
                                &desc);
        do
        {  // continue to next loop. If the tetris loop is within the filter loops, skip it.
            currentFilterMask = (EMmeLoopMask)((((unsigned) currentFilterMask) << 1) + 1);
        } while (currentFilterMask == e_mme_tetris_loop);
    }

    EMmeLoopMask kMask = pattern2LoopMask(params.getPattern(), EMmeLoopDim::dim_k);
    MME_ASSERT(kMask != e_mme_tetris_loop, "kMask loop overflow");

    // Set C
    EMmeLoopMask cMask = pattern2LoopMask(params.getPattern(), EMmeLoopDim::dim_c);
    MME_ASSERT(cMask != e_mme_tetris_loop, "cMask loop overflow");

    // Set k & c dims according to the horizontal and vertical movements
    // Horizontal movement moves L in dim 0 and vertical moves S in dim 1 before trans come to play
    unsigned dimA_h = Mme::c_mme_max_tensor_dims;  // don't care
    unsigned dimB_h = desc.header.transL ? 1 : 0;
    unsigned dimO_h = 0;
    unsigned dimA_v = desc.header.transS ? 1 : 0;  // because transS=1 means no transpose...
    unsigned dimB_v = Mme::c_mme_max_tensor_dims;
    unsigned dimO_v = 1;
    // But when transO is set, we need to swap the vertical and horizontal movements
    if (m_geoAttrSPtr->isTransO())
    {
        std::swap(cMask, kMask);
    }
    setAssociatedDimAndSize(kMask, m_reuseAttr.denseStepsNr, dimA_h, dimB_h, dimO_h, &desc);
    setAssociatedDimAndSize(cMask, m_reuseAttr.aPartialHeightStepsNr, dimA_v, dimB_v, dimO_v, &desc);

    // Set S - no S in bgemm
    desc.numIterationsMinus1 = 0;
}

static void transposeDescBgemm(Mme::Desc* desc)
{
    std::swap(desc->baseAddrHighS, desc->baseAddrHighL);
    std::swap(desc->baseAddrLowS, desc->baseAddrLowL);
    swap_bf(desc->header.transS, desc->header.transL);
    swap_bf(desc->header.lowerS, desc->header.lowerL);
    for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        swap_bf(desc->conv.associatedDims[i].dimS, desc->conv.associatedDims[i].dimL);
    }
    swap_bf(desc->outerLoop.associatedDims.dimS, desc->outerLoop.associatedDims.dimL);
    std::swap(desc->tensorS, desc->tensorL);
    std::swap(desc->paddingValueS, desc->paddingValueL);
}
void AguBGemmConfig::setDescAgusBgemm(Mme::Desc* descPtr,
                                      const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                      const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                      const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                      unsigned descIdx)
{
    descPtr[descIdx].aguS = aguS[descIdx];

    for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
    {
        descPtr[descIdx].aguL[aguIdx] = aguL[aguIdx];
        descPtr[descIdx].aguO[aguIdx] = aguO[aguIdx];
    }
}
void AguBGemmConfig::setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& localDesc)
{
    if (localDesc.header.transS)
    {
        setDescFirstAguATranspose(aguS, &localDesc);
    }
    else
    {
        setDescFirstAguANonTranspose(aguS, &localDesc);
    }
}
void AguBGemmConfig::setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& localDesc)
{
    if (localDesc.header.transL)
    {
        setDescFirstAguBTranspose(aguL, &localDesc);
    }
    else
    {
        setDescFirstAguBNonTranspose(aguL, &localDesc);
    }
}
void AguBGemmConfig::setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                     bool isSouth,
                                     Mme::Desc& localDesc)
{
    if (localDesc.header.transS)
    {
        setDescRestAguATranspose(aguS, isSouth, &localDesc);
    }
    else
    {
        setDescRestAguANonTranspose(aguS, isSouth, &localDesc);
    }
}
void AguBGemmConfig::setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                     bool isSouth,
                                     Mme::Desc& localDesc)
{
    if (localDesc.header.transL)
    {
        setDescRestAguBTranspose(aguL, isSouth, &localDesc);
    }
    else
    {
        setDescRestAguBNonTranspose(aguL, isSouth, &localDesc);
    }
}

void AguBGemmConfig::setAgu(DescGroup& descGroup)
{
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        std::array<Mme::MmeAguCoreDesc, c_operand_max_agu> aguS {0}, aguL {0}, aguO {0};
        std::array<unsigned, c_operand_max_agu> loopS;
        std::array<unsigned, c_operand_max_agu> loopL;
        std::array<unsigned, c_operand_max_agu> loopO;

        Mme::Desc localDesc = descGroup.desc[descIdx];

        setAssociatedDims(localDesc);
        setNumIterationsMinus1(localDesc);

        // ====== First of AGU (AKA first worker) =================================
        setDescFirstAguA(aguS, localDesc);
        setDescFirstAguB(aguL, localDesc);
        setDescFirstOfAguCoutAndBatchLoops(aguS, aguL, aguO, &localDesc);

        // =========== Rest of the AGUs (AKA other workers) =========================
        setDescRestAguA(aguS, descIdx == 0, localDesc);
        setDescRestAguB(aguL, descIdx == 0, localDesc);
        setDescRestAguCout(aguO, descIdx == 0, &localDesc);

        // ============= Double the number of gemms if possible ===============
        doubleWorkForBatchMode(aguS, aguL, aguO, descIdx == 0, &localDesc);

        if (m_geoAttrSPtr->isTransO())
        {
            transposeDescBgemm(&localDesc);
            std::swap(aguS, aguL);
            std::swap(loopS, loopL);
        }

        descGroup.desc[descIdx] = localDesc;
        setDescAgusBgemm(descGroup.desc, aguS, aguL, aguO, descIdx);
        setPartialHeightLoops(descGroup.desc, loopS, loopL, loopO, descIdx);
    }
}

}  // namespace gaudi
