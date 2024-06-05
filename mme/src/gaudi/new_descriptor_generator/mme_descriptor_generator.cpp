#include "include/gaudi/mme_descriptor_generator.h"
#include "include/gaudi/new_descriptor_generator/agu_config.h"
#include "include/gaudi/new_descriptor_generator/mme_common.h"
#include "include/utils/gaudi_desc_dumper.h"
#include "src/gaudi/gaudi_geo_attr.h"
#include "src/gaudi/gaudi_linear_ranges.h"
#include "src/mme_common/mme_hal_factory.h"
#include "src/mme_common/mme_params_dumper.h"

#define FMT_HEADER_ONLY
#include "spdlog/fmt/bundled/format.h"

#define set_bf_to_all_ones(bf)                                                                                         \
    {                                                                                                                  \
        (bf) = 1;                                                                                                      \
        (bf) = -(bf);                                                                                                  \
    }

//#define MME_STACK_TRACE
#ifdef MME_STACK_TRACE
static unsigned __indent__ = 0;
#define TRACE_ENTER                                                                                                    \
    for (unsigned i = 0; i < __indent__; i++)                                                                          \
        printf("  ");                                                                                                  \
    printf("Enter (%05u) - %s\n", __LINE__, __FUNCTION__);                                                             \
    __indent__++
#define TRACE_EXIT                                                                                                     \
    __indent__--;                                                                                                      \
    for (unsigned i = 0; i < __indent__; i++)                                                                          \
        printf("  ");                                                                                                  \
    printf("Exit  (%05u) - %s\n", __LINE__, __FUNCTION__)
#else
#define TRACE_ENTER
#define TRACE_EXIT
#define TRACE_EXIT
#endif

using namespace MmeCommon;
static const unsigned c_operand_max_agu = 4;
static const unsigned c_padded_conv_dim = Mme::c_mme_max_conv_dims - 1;

namespace gaudi
{
//=================== Create the geo params =============================================
static void
getPortElementSizes(unsigned& portElemHeight, unsigned& cPortElemWidth, unsigned cl_size, unsigned shiftAmount)
{
    portElemHeight = (cl_size / 2) >> shiftAmount;  // 64 for bfloat, 32 for float32
    cPortElemWidth = (cl_size / 2) >> shiftAmount;  // 64 for bfloat, 32 for float32
}

MmeBatchMode getBatchGemmBatchMode(EMmeOpType opType,
                                   EMmeGeometry geometry,
                                   unsigned outputHeight,
                                   unsigned outputWidth,
                                   unsigned cPortElemWidth)
{
    // 2x modes are currently not supported in conv operations
    if ((opType == e_mme_fwd) || (opType == e_mme_dedx) || (opType == e_mme_dedw))
    {
        return mme_batch_none;
    }

    if (geometry == e_mme_geometry_2wx2h)
    {
        if (outputHeight <= cPortElemWidth)
        {
            return mme_batch_2xw;
        }
        if (outputWidth <= cPortElemWidth)
        {
            return mme_batch_2xh;
        }
    }
    return mme_batch_none;
}
// todo: useDataType is added to match the legacy code for memset desc
// todo [SW-51591] Check and remove redundant settings
GeoAttr::GeoAttr(MmeLayerParams params,
                 const MmeCommon::MmeHalReader& mmeHalReader,
                 UseDataTypeForAttributeCalculation useDataType)
: m_mmeHalReader(mmeHalReader)
{
    EMmeGeometry geometry = params.getGeometry();
    EMmeDataType inDataType = params.getOperand(e_mme_op_a).elementType;
    EMmeDataType outDataType = params.getOperand(e_mme_op_c).elementType;
    MME_ASSERT(inDataType == e_type_fp32 || inDataType == e_type_bf16, "invalid data type");
    MME_ASSERT(outDataType == e_type_fp32 || outDataType == e_type_bf16, "invalid data type");

    unsigned outputWidth = params.y.sizes[0];
    unsigned outputHeight = params.y.sizes[1];
    unsigned portElemHeight;
    unsigned cPortElemWidth;
    unsigned shiftAmount = getDataTypeShiftAmount(
        useDataType == UseDataTypeForAttributeCalculation::InputDataType ? inDataType : outDataType);
    const MmeCommon::MmeHalReader& mmeHal = gaudi::MmeHalReader::getInstance();
    unsigned cl_size = mmeHal.getClSize();
    getPortElementSizes(m_portElemHeight, m_cPortElemWidth, cl_size, shiftAmount);

    m_inputPortElemWidth = (cl_size / 2) >> shiftAmount;  // 64 for bfloat, 32 for float32
    m_euElemWidth = (cl_size / 2) >> shiftAmount;  // 64 for bfloat, 32 for float32
    m_euElemHeight = (cl_size / 2) >> shiftAmount;  // 64 for bfloat, 32 for float32
    m_subMatrixHeight = Mme::c_mme_matrix_size >> shiftAmount;
    m_subMatrixWidth = Mme::c_mme_matrix_size >> shiftAmount;

    m_concurrentLevel = 1;  // Number of cores/poles that concurrently execute horizontally

    // Set the batch mode
    m_batchMode = getBatchGemmBatchMode(params.opType, geometry, outputHeight, outputWidth, m_cPortElemWidth);
    m_transO = false;

    int numApoles = 0, numBpoles = 0, numAportsPerPole = 0, numBportsPerPole = 0;
    switch (geometry)
    {
        case e_mme_geometry_2wx2h:
            // Config is 2x2: 2 pairs of 2x
            numApoles = 2;
            numBpoles = 1;
            // The below holds also for 2xh because then we swap between A and B
            numAportsPerPole = 1;
            numBportsPerPole = 2;

            // Define the batch mode
            if (m_batchMode == mme_batch_2xw)
            {
                m_concurrentLevel = 2;
                numApoles = 1;  // 1 pole, each of 1 port
                numBpoles = 1;  // 1 poles, each of 2 ports
                numAportsPerPole = 1;
                numBportsPerPole = 2;
            }
            else if (m_batchMode == mme_batch_2xh)
            {
                m_concurrentLevel = 2;
                numApoles = 1;
                numBpoles = 1;
                numAportsPerPole = 2;
                numBportsPerPole = 1;
                m_transO = true;
            }
            break;
        case e_mme_geometry_4wx1h:
            numApoles = 1;
            numBpoles = 2;
            numAportsPerPole = 1;
            numBportsPerPole = 2;
            break;
        case e_mme_geometry_1wx4h:
            numApoles = 2;
            numBpoles = 1;
            numAportsPerPole = 2;
            numBportsPerPole = 1;
            m_transO = true;
            break;
        default:
            MME_ASSERT(0, "invalid geometry");
            break;  // Initialize fields to silent compile time warning
    }

    m_totalAports = numApoles * numAportsPerPole;  // Number of ports with which we read A
    m_totalBports = numBpoles * numBportsPerPole;

    m_matrixHeight = m_totalAports * m_subMatrixHeight;
    m_matrixWidth = m_totalBports * m_subMatrixWidth;

    m_geoElemHeightPerPair = numAportsPerPole * m_portElemHeight;
    m_geoElemWidthPerPair = numBportsPerPole * m_cPortElemWidth;

    m_geoTotalElemHeight = m_totalAports * m_portElemHeight;
    m_geoTotalElemWidth = m_totalBports * m_cPortElemWidth;

    m_isAInterleaved = (m_totalAports == 2) && !m_transO;
}

//==================== Set of header fields ======================================
void MmeBgemmDescriptorGenerator::setDescHeader(Mme::Desc& desc)
{
    const MmeLayerParams params = getParams();

    switch (params.opType)
    {
        case e_mme_ab:
            desc.header.transS = 1;  // When set, the shared operand is transposed.
            desc.header.transL = 0;  // When set, the local operand is transposed.
            break;
        case e_mme_abt:
            desc.header.transS = 1;
            desc.header.transL = 1;
            break;
        case e_mme_atb:
            desc.header.transS = 0;
            desc.header.transL = 0;
            break;
        case e_mme_atbt:
            desc.header.transS = 0;
            desc.header.transL = 1;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }
    desc.header.transO = m_geoParamsSPtr->isTransO();  // When set, the output operand is transposed.

    // For start, we support only native walk patterns of w->h->b and h->w->b. In both,
    // all operands are advanced on the outer conv loop
    desc.header.advanceS = 1;  // Advance the shared operand in the outer conv loop.
    desc.header.advanceL = 0;  // Advance the local operand in the outer conv loop.
    desc.header.advanceO = 1;  // Advance the output operand in the outer conv loop.

    // No lowering in bgemm
    desc.header.lowerL = 0;  // lower the local operand.
    desc.header.lowerS = 0;  // lower the shared operand.

    // In bgemm there is no accumulation (except the gemm loop which is not counted)
    desc.header.accumMask = 0x0;  // EMmeLoopMask value. Bit mask of loops to accumulate in the GRF.
                                  // "1" means the loop is accumulated.
                                  // The field is 4 bits wide (and not 5) since the tetris loop
                                  // is always outer to the conv and is never accumulated.
    // Because there is no reuse / partials, we always promote the accumulator
    desc.header.accStoreIncDisable = 0;  // Avoid incrementing the store acc counter.
    desc.header.roundingMode = (int) params.controls.roundingMode | (int) params.controls.conversionRoundingMode;

    desc.header.dataTypeIn =
        ConvertToGaudiDataType(params.x.elementType);  // The data type of the input operands. (EMmeDataType)
    desc.header.dataTypeOut =
        ConvertToGaudiDataType(params.y.elementType);  // The data type of the output operands. (EMmeDataType)

    desc.header.accum = 0;  // Accumulate output in the accumulator.
    desc.header.storeEn = 1;  // Store the output if we are at the last partial slice

    desc.header.rollAccums = 0;  // The number of accumolator inc to do after the last rollup.

    desc.header.reluEn = params.controls.reluEn;  // Enable RELU.

    desc.header.fpEn = 1;  // reserved.
    desc.header.euBEn = 1;
    desc.sw.dw = 0;
    if (getZeroCD())
    {
        // If the original operation was conv, we need to set the appropriate memset flag
        EMmeOpType originalOpType = getOriginalParams().opType;
        switch (originalOpType)
        {
            case MmeCommon::e_mme_fwd:
                desc.sw.swMemsetFwd = 1;
                break;
            case MmeCommon::e_mme_dedw:
                desc.sw.swMemsetDedw = 1;
                break;
            case MmeCommon::e_mme_dedx:
                desc.sw.swMemsetDedx = 1;
                break;
            default:
                break;
        }
    }
}

void MmeFwdDedxDescriptorGenerator::setDescHeader(Mme::Desc& desc)
{
    const MmeLayerParams params = getParams();
    bool isMemsetDesc = m_convSubProblems.current->isMemsetDesc();

    MmeRoi roi = getRoi();
    MmeRoi paddedRoi;
    padRoi(m_geoParamsSPtr->m_totalAports, &roi, &paddedRoi);

    desc.header.fpEn = 1;
    desc.header.euBEn = 1;
    desc.header.transS = 1;
    // todo: this setting is done to match the legacy code for memset desc, likely redundant
    // todo [SW-51591] Check and remove redundant settings
    desc.header.transL = isMemsetDesc ? 0 : (params.opType == e_mme_dedx);
    desc.header.lowerS = 0;
    desc.header.lowerL = 0;
    desc.header.transO = 0;
    desc.header.accumMask = (1 << c_padded_conv_dim) - 1;
    desc.header.storeEn = 1;
    // todo: this setting is done to match the legacy code for memset desc, likely redundant
    // todo [SW-51591] Check and remove redundant settings
    desc.header.advanceS = isMemsetDesc ? 0 : 1;
    desc.header.advanceL = 0;
    desc.header.advanceO = 1;
    desc.header.accStoreIncDisable = 0;
    // for memset desc we set the input data type according to output to match the legacy code.
    desc.header.dataTypeIn = isMemsetDesc ? ConvertToGaudiDataType(params.getOperand(e_mme_op_c).elementType)
                                          : ConvertToGaudiDataType(params.getOperand(e_mme_op_a).elementType);
    desc.header.dataTypeOut = ConvertToGaudiDataType(params.getOperand(e_mme_op_c).elementType);
    desc.header.reluEn = params.controls.reluEn;
    desc.header.accum = 0;
    desc.header.rollAccums = 0;
    desc.header.roundingMode = isMemsetDesc ? MmeCommon::RoundingMode::RoundToZero : params.controls.roundingMode;
    desc.header.signalEn = 0;
    desc.header.signalMask = 0;
    desc.syncObject.addrHigh = 0;
    desc.syncObject.addrLow[0] = 0;
    desc.syncObject.addrLow[1] = 0;
    desc.syncObject.operation = 1;
    desc.syncObject.value = 1;

    desc.sw.swMemsetFwd = isMemsetDesc && (params.opType == e_mme_fwd);
    desc.sw.swMemsetDedx = isMemsetDesc && (params.opType == e_mme_dedx);
    desc.sw.swMemsetDedw = isMemsetDesc && (params.opType == e_mme_dedw);
}

void MmeDedwDescriptorGenerator::setDescHeader(Mme::Desc& desc)
{
    const MmeLayerParams params = getParams();

    desc.header.euBEn = 1;
    desc.header.fpEn = 1;
    desc.header.transO = 0;
    desc.header.transS = 0;
    desc.header.transL = 0;
    desc.header.lowerS = 0;
    desc.header.lowerL = 0;
    desc.header.accumMask = 0;
    desc.header.storeEn = 1;
    desc.header.advanceS = 0;
    desc.header.advanceL = 0;
    desc.header.advanceO = 0;
    desc.header.accStoreIncDisable = 0;
    desc.header.dataTypeIn = ConvertToGaudiDataType(params.getOperand(e_mme_op_a).elementType);
    desc.header.dataTypeOut = ConvertToGaudiDataType(params.getOperand(e_mme_op_c).elementType);
    desc.header.reluEn = 0;
    desc.header.accum = 0;
    desc.header.rollAccums = 0;
    desc.header.roundingMode = (int) params.controls.roundingMode | (int) params.controls.conversionRoundingMode;
    desc.header.signalEn = 0;
    desc.header.signalMask = 0;

    desc.syncObject.addrHigh = 0;
    desc.syncObject.addrLow[0] = 0;
    desc.syncObject.addrLow[1] = 0;
    desc.syncObject.operation = 1;
    desc.syncObject.value = 1;

    desc.sw.dw = 0;

    desc.outerLoop.sizeMinus1 = 0;
    desc.conv.kernelSizeMinus1.dw = 0;
}

//========================================================
void MmeBgemmDescriptorGenerator::setHeaderLoopSelectors(DescGroup& descGroup)
{
    const MmeLayerParams params = getParams();

    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        switch (params.getPattern())
        {
            case MmeCommon::e_mme_sp_reduction_ckf:
                descGroup.desc[descIdx].header.partialHeightLoopS =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_outer_loop);  // vertical A done by Outer
                descGroup.desc[descIdx].header.partialHeightLoopLLocal =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_conv_loop_3);  // horizontal B done by Conv3
                break;
            case MmeCommon::e_mme_sp_reduction_kcf:
                descGroup.desc[descIdx].header.partialHeightLoopS =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_conv_loop_3);  // vertical A done by Conv3
                descGroup.desc[descIdx].header.partialHeightLoopLLocal =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_outer_loop);  // horizontal B done by Outer
                break;
            case MmeCommon::e_mme_sp_reduction_fkc:
                descGroup.desc[descIdx].header.partialHeightLoopS =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_conv_loop_0);  // vertical A done by Conv0
                descGroup.desc[descIdx].header.partialHeightLoopLLocal =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_conv_loop_1);  // horizontal B done by Conv1
                break;
            case MmeCommon::e_mme_sp_reduction_fck:
                descGroup.desc[descIdx].header.partialHeightLoopS =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_conv_loop_1);  // vertical A done by Conv1
                descGroup.desc[descIdx].header.partialHeightLoopLLocal =
                    getLoopFromLoopMask(EMmeLoopMask::e_mme_conv_loop_0);  // horizontal B done by Conv0
                break;
            default:
                MME_ASSERT(0, "invalid walking pattern");
        }

        descGroup.desc[descIdx].header.partialHeightLoopOLocal =
            m_geoParamsSPtr->isTransO() ? descGroup.desc[descIdx].header.partialHeightLoopS
                                   : descGroup.desc[descIdx].header.partialHeightLoopLLocal;
        descGroup.desc[descIdx].header.partialHeightLoopORemote =
            descGroup.desc[descIdx].header.partialHeightLoopOLocal;
        descGroup.desc[descIdx].header.partialHeightLoopLRemote =
            descGroup.desc[descIdx].header.partialHeightLoopLLocal;
    }
}

void MmeConvDescriptorGenerator::setHeaderLoopSelectors(DescGroup& descGroup)
{
    // This is left empty on purpose.
    // Setting the Loop selectors is done in conv while setting the agu, as opposed to bgemm
    // todo AlonG: align bgemm and conv
}

void MmeDescriptorGenerator::setAddressOffsets(Mme::Desc& desc)
{
    const auto params = getParams();
    auto recipe = getRecipe();

    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_dedw:
            break;
        default:  // all other ops
            return;
    }

    const MultiDimSubView& convSubview = getRecipe().curNonSpatial();
    const SingleDimSubView fcdDimSubview = getRecipe().curFcd();
    const SingleDimSubView& spSubview = getRecipe().curSp();

    auto& convSubViews = getRecipe().getNonSpatialSubviews();
    unsigned convIdx = getRecipe().getIterator().nonSpatialIdx();
    unsigned convSize = convSubViews.size();
    unsigned curConvBase = convSubViews[convIdx].bases[1];
    unsigned curConvBaseReversed = convSubViews[convSize - convIdx - 1].bases[1];

    auto aTensor = params.getOperand(e_mme_op_a);
    auto bTensor = params.getOperand(e_mme_op_b);
    auto cTensor = params.getOperand(e_mme_op_c);

    unsigned aBaseOffset = 0, bBaseOffset = 0, cBaseOffset = 0;
    switch (params.opType)
    {
        case e_mme_fwd:
            for (unsigned d = 1; d < convSubview.sizes.size(); d++)
            {
                bBaseOffset += convSubview.bases[d] * bTensor.strides[d];
            }
            break;
        case e_mme_dedx:
            aBaseOffset = convSubview.bases[0] * aTensor.strides[0];
            for (unsigned d = 0; d < convSubview.sizes.size(); d++)
            {
                bBaseOffset += convSubview.bases[d] * bTensor.strides[d];
            }
            break;
        case e_mme_dedw:
            aBaseOffset = convSubview.bases[1];
            bBaseOffset = fcdDimSubview.viewBase;
            // The output offset is derived from all filter dimensions and the fcd movement
            for (unsigned d = 0; d < convSubview.sizes.size(); d++)
            {
                unsigned convDim = d + 1;
                cBaseOffset += convSubview.bases[convDim] * params.getOperand(e_mme_op_c).strides[convDim];
            }
            cBaseOffset += fcdDimSubview.viewBase;
            break;
        default:
            MME_ASSERT(0, "Invalid opType");
    }

    unsigned aDataSize = (aTensor.elementType == e_type_bf16) ? sizeof(uint16_t) : sizeof(float);
    unsigned bDataSize = (bTensor.elementType == e_type_bf16) ? sizeof(uint16_t) : sizeof(float);
    unsigned cDataSize = (cTensor.elementType == e_type_bf16) ? sizeof(uint16_t) : sizeof(float);

    OffsetArray& descAddrOffset = m_convSubProblems.current->addressOffset;
    union
    {
        uint64_t addr;
        uint32_t addr_u32[2];
    } aBase, bBase, cBase;
    aBase.addr = aBaseOffset;
    bBase.addr = bBaseOffset;
    cBase.addr = cBaseOffset;

    for (int dim = 0; dim < m_convSubProblems.current->addressOffset.xOffset.size(); dim++)
    {
        switch (params.opType)
        {
            case e_mme_fwd:
            case e_mme_dedw:
                aBase.addr += m_convSubProblems.current->addressOffset.xOffset[dim];
                bBase.addr += m_convSubProblems.current->addressOffset.wOffset[dim];
                cBase.addr += m_convSubProblems.current->addressOffset.yOffset[dim];
                break;
            case e_mme_dedx:
                // todo: this setting is done to match the legacy code for memset desc, likely redundant
                // todo [SW-51591] Check and remove redundant settings
                if (!m_convSubProblems.current->isMemsetDesc())
                {
                    /*
                     * X offset was folded into roiBaseOffset instead of base address. later the rest will be folded as
                     * well.
                    cBase.addr += m_convSubProblems.current->addressOffset.xOffset[dim];
                    aBase.addr += m_convSubProblems.current->addressOffset.yOffset[dim];
                     */
                    bBase.addr += m_convSubProblems.current->addressOffset.wOffset[dim];
                }
                break;
            default:
                MME_ASSERT(0, "Invalid op type");
        }
    }

    aBase.addr *= aDataSize;
    bBase.addr *= bDataSize;
    cBase.addr *= cDataSize;

    desc.baseAddrLowS = aBase.addr_u32[0];
    desc.baseAddrLowL = bBase.addr_u32[0];
    desc.baseAddrLowO = cBase.addr_u32[0];
    desc.baseAddrHighS = aBase.addr_u32[1];
    desc.baseAddrHighL = bBase.addr_u32[1];
    desc.baseAddrHighO = cBase.addr_u32[1];
}

void MmeDescriptorGenerator::handlePartials(DescGroup& descGroup)
{
    const MmeLayerParams params = getParams();
    const MmeRecipe recipe = getRecipe();
    // If no partials, return
    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_dedx:
            if (recipe.getNonSpatialSubviews().size() == 1) return;
            break;
        case e_mme_dedw:
            if (recipe.getSpSubviews().size() == 1) return;
            break;
        default:  // Other ops do not have partials
            return;
    }

    const MultiDimSubView& convSubview = recipe.curNonSpatial();
    unsigned convIdx = recipe.getIterator().nonSpatialIdx();
    const unsigned spIdx = recipe.getIterator().spIdx();
    unsigned accumsNr;

    Mme::Desc* desc0 = &(descGroup.desc[0]);
    unsigned loopKSize = getLoopSize(desc0, params.getPattern(), LOOP_K) + 1;

    if (params.opType == e_mme_dedw)
    {
        unsigned loopCSize = getLoopSize(desc0, params.getPattern(), LOOP_C) + 1;
        unsigned filterLoopSize = 1;
        for (unsigned d = 0; d < c_padded_conv_dim; d++)
        {
            filterLoopSize *= (getLoopSize(desc0, params.getPattern(), LOOP_FILTER + d) + 1);
        }
        accumsNr = loopKSize * loopCSize * filterLoopSize;
    }
    else
    {
        unsigned tetrisNr = desc0->numIterationsMinus1 + 1;
        accumsNr = loopKSize * tetrisNr;
    }

    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        Mme::Desc& desc = descGroup.desc[descIdx];

        bool firstPartial = false, lastPartial = false;
        switch (params.opType)
        {
            case e_mme_fwd:
            case e_mme_dedx:
                firstPartial = (convIdx == 0);
                lastPartial = (convIdx == recipe.getNonSpatialSubviews().size() - 1);
                break;
            case e_mme_dedw:
                firstPartial = (spIdx == 0);
                lastPartial = (spIdx == recipe.getSpSubviews().size() - 1);
                break;
            default:
                MME_ASSERT(0, "Unsupported op type");
        }
        // In the first partial, accum should be set to 0
        desc.header.accum = firstPartial ? 0 : (accumsNr == 1) ? 0 : 1;

        // If the last partial, some fields should be set differently
        if (lastPartial)
        {
            desc.header.storeEn = 1;
            desc.header.accStoreIncDisable = 0;
            desc.header.rollAccums = 0;
            desc.header.reluEn = params.controls.reluEn && (params.opType == e_mme_fwd);
        }
        else
        {
            desc.header.storeEn = 0;
            desc.header.accStoreIncDisable = (accumsNr == 1) ? 1 : 0;
            desc.header.rollAccums = (accumsNr == 1) ? 0 : Mme::c_mme_accums_nr - accumsNr;
            desc.header.reluEn = false;
        }
    }
}

//======================= Generic desc functions (not per-op) ==============================
void MmeDescriptorGenerator::initSignalingInfo(Mme::Desc& desc)
{
    desc.syncObject.addrHigh = 0;
    desc.syncObject.addrLow[0] = 0;
    desc.syncObject.addrLow[1] = 0;
    desc.syncObject.operation = 1;
    desc.syncObject.value = 1;
}
void MmeDescriptorGenerator::resetMetaDataFields(Mme::Desc& desc)
{
    desc.metaData.aguS = 0;
    desc.metaData.aguL[Mme::e_mme_local] = 0;
    desc.metaData.aguL[Mme::e_mme_remote] = 0;
    desc.metaData.aguO[Mme::e_mme_local] = 0;
    desc.metaData.aguO[Mme::e_mme_remote] = 0;
}
void MmeDescriptorGenerator::setUserDataFields(Mme::Desc& desc)
{
    const MmeLayerParams params = getParams();
    desc.axiUserData.dw = getUserDataVal(params.controls.atomicAdd,
                                         params.getOperand(e_mme_op_c).elementType,
                                         params.controls.roundingMode);

    desc.pcu.rlSaturation = 16 * 4096;
}

void MmeDescriptorGenerator::resetDescPerfFields(Mme::Desc& desc)
{
    desc.perfEvtS.dw = 0;
    set_bf_to_all_ones(desc.perfEvtS.startEndMask);
    desc.perfEvtL[Mme::e_mme_local].dw = 0;
    set_bf_to_all_ones(desc.perfEvtL[Mme::e_mme_local].startEndMask);
    desc.perfEvtL[Mme::e_mme_remote].dw = 0;
    set_bf_to_all_ones(desc.perfEvtL[Mme::e_mme_remote].startEndMask);
    desc.perfEvtO[Mme::e_mme_local].dw = 0;
    set_bf_to_all_ones(desc.perfEvtO[Mme::e_mme_local].startEndMask);
    desc.perfEvtO[Mme::e_mme_remote].dw = 0;
    set_bf_to_all_ones(desc.perfEvtO[Mme::e_mme_remote].startEndMask);
}

void MmeDescriptorGenerator::buildDescTensorPointers(Mme::Desc& desc)
{
    // Default values, will be patched later
    desc.baseAddrHighS = 0;
    desc.baseAddrLowS = 0;
    desc.baseAddrHighL = 0;
    desc.baseAddrLowL = 0;
    desc.baseAddrHighO = 0;
    desc.baseAddrLowO = 0;
}

void MmeDescriptorGenerator::setSignals(DescGroup& descGroup, bool isLast)
{
    const MmeLayerParams params = getParams();
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        Mme::Desc& desc = descGroup.desc[descIdx];

        if ((params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_none) ||
            (params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_once))
        {
            desc.header.signalEn = 0;
        }
        else if (params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_desc)
        {
            desc.header.signalEn = 1;
            set_bf_to_all_ones(desc.header.signalMask);
        }
        else if (params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_desc_with_store)
        {
            desc.header.signalEn = desc.header.storeEn;  // todo Alon: verify it is initialized
            set_bf_to_all_ones(desc.header.signalMask);
        }
        else if (params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_output)
        {
            desc.header.signalEn = desc.header.storeEn;
            desc.header.signalMask = desc.header.accumMask;
        }
        else if (params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_partial)
        {
            desc.header.signalEn = 1;
            desc.header.signalMask = desc.header.accumMask;
        }
        else
        {
            MME_ASSERT(0, "invalid signaling mode");
        }
    }

    // In case of signal once, apply signalig to the last desc group
    if (params.controls.signalingMode == EMmeSignalingMode::e_mme_signaling_once && isLast)
    {
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            Mme::Desc& desc = descGroup.desc[descIdx];
            desc.header.signalEn = 1;
            set_bf_to_all_ones(desc.header.signalMask);
        }
    }
}

void MmeDescriptorGenerator::setReuseAttr(const GeoAttr& geoParams,
                                          const EMmeOpType op,
                                          const bool reuseA,
                                          const bool reuseB)
{
    const MultiDimSubView& convSubview = getRecipe().curNonSpatial();
    const SingleDimSubView fcdDimSubview = getRecipe().curFcd();
    const SingleDimSubView& spSubview = getRecipe().curSp();
    const MmeLayerParams params = getParams();
    EMmePattern pattern = params.getPattern();

    unsigned filterStepsNr = 1;
    unsigned aSpatialSize = spSubview.viewSize;
    m_reuseAttr.aPartialHeightStepsNr = div_round_up(aSpatialSize, geoParams.m_geoTotalElemHeight);
    m_reuseAttr.denseStepsNr = div_round_up(fcdDimSubview.viewSize, geoParams.m_geoTotalElemWidth);

    // When the output tensor is transposed, the walks are reversed and so the step numbers
    if (m_geoParamsSPtr->isTransO() && !params.isGemmOperation())
    {
        std::swap(m_reuseAttr.aPartialHeightStepsNr, m_reuseAttr.denseStepsNr);
    }
    m_reuseAttr.denseLoopSelector = getLoopFromLoopMask(pattern2LoopMask(pattern, EMmeLoopDim::dim_k));
    m_reuseAttr.aPartialHeightLoopSelector = getLoopFromLoopMask(pattern2LoopMask(
        params.getPattern(),
        (params.opType == e_mme_dedw || params.isGemmOperation()) ? EMmeLoopDim::dim_c : EMmeLoopDim::dim_s));
    unsigned filterLoopSelector = 0;

    switch (pattern)
    {
        case e_mme_sp_reduction_ckf:
            m_reuseAttr.spatialLoopSelector = filterLoopSelector;
            m_reuseAttr.spatialStepsNr = filterStepsNr;
            break;
        case e_mme_sp_reduction_kcf:
        case e_mme_sp_reduction_kfc:
        case e_mme_sp_reduction_cfk:
        case e_mme_sp_reduction_fck:
            // C and F are adjacent, can reuse on both of them.
            m_reuseAttr.spatialStepsNr = filterStepsNr * m_reuseAttr.aPartialHeightStepsNr;
            break;
        case e_mme_sp_reduction_fkc:
            m_reuseAttr.spatialStepsNr = m_reuseAttr.aPartialHeightStepsNr;
            break;
        case e_mme_z_reduction_skf:
        case e_mme_z_reduction_ksf:
            // no movement on filter dimensions, a reuse is simply the spatial size.
            m_reuseAttr.spatialStepsNr = m_reuseAttr.aPartialHeightStepsNr;
            break;
        default:
            MME_ASSERT(0, "invalid walking pattern");
    }

    bool actualReuseA = (!m_geoParamsSPtr->isTransO() && reuseA) || (m_geoParamsSPtr->isTransO() && reuseB);
    bool actualReuseB = (!m_geoParamsSPtr->isTransO() && reuseB) || (m_geoParamsSPtr->isTransO() && reuseA);

    MME_ASSERT(!actualReuseA || (m_reuseAttr.denseStepsNr <= Mme::c_mme_max_sb_reuse),
               "dense steps larger then max reuse size");
    MME_ASSERT(!actualReuseB || (m_reuseAttr.spatialStepsNr <= Mme::c_mme_max_sb_reuse),
               "spatial steps larger then max reuse size");

    unsigned spRem = aSpatialSize % geoParams.m_geoTotalElemHeight ? aSpatialSize % geoParams.m_geoTotalElemHeight
                                                                   : geoParams.m_geoTotalElemHeight;
    unsigned denseRem = fcdDimSubview.viewSize % geoParams.m_geoTotalElemWidth
                            ? fcdDimSubview.viewSize % geoParams.m_geoTotalElemWidth
                            : geoParams.m_geoTotalElemWidth;

    // Pad the last spatial step to the number of ports (not needed if transO is set)
    m_reuseAttr.lastAPartialHeightStepSize =
        m_geoParamsSPtr->isTransO() ? spRem : (spRem + geoParams.m_totalAports - 1) & ~(geoParams.m_totalAports - 1);
    m_reuseAttr.lastDenseStepSize =
        denseRem %
        geoParams.m_euElemWidth;  // (denseRem >= geoParams.m_euElemWidth) ? geoParams.m_euElemWidth : denseRem;

    m_reuseAttr.accumDim =
        (op == e_mme_dedw || params.isGemmOperation()) ? Mme::e_mme_gemm_loop : Mme::e_mme_conv_loop_2;
}

static void initSbRepeatFields(Mme::Desc& desc)
{
    desc.sbRepeat.teEnS = 1;  // suspention buffer reuse
    desc.sbRepeat.teEnL = 1;
    desc.sbRepeat.loadS = 1;
    desc.sbRepeat.loadL = 1;
    desc.sbRepeat.aguSLoopMask = 0;
    desc.sbRepeat.aguLLoopMask = 0;
    desc.sbRepeat.repeatSMinus1 = 0;
    desc.sbRepeat.repeatLMinus1 = 0;
}
void MmeBgemmDescriptorGenerator::setDescSBReuse(DescGroup& descGroup)
{
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        Mme::Desc& desc = descGroup.desc[descIdx];
        // Just Initialization
        initSbRepeatFields(desc);
    }
}

void MmeFwdDedxDescriptorGenerator::setDescSBReuse(DescGroup& descGroup)
{
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        Mme::Desc& desc = descGroup.desc[descIdx];
        // Just Initialization
        initSbRepeatFields(desc);
    }

    const MmeLayerParams params = getParams();
    if (!params.strategy.sbReuse)
    {
        return;
    }
    auto pattern = params.strategy.pattern;
    Mme::Desc& desc0 = descGroup.desc[0];

    unsigned numFcdSteps = getLoopSize(&desc0, params.strategy.pattern, LOOP_K);
    unsigned numSpatialSteps = getLoopSize(&desc0, params.strategy.pattern, LOOP_SPATIAL);
    bool reuseA = ((params.strategy.pattern == e_mme_z_reduction_skf) && (numFcdSteps > 0));
    bool reuseB = ((params.strategy.pattern == e_mme_z_reduction_ksf) && (numSpatialSteps > 0));
    bool reuse = (reuseA || reuseB) && (m_recipe.reuseA() || m_recipe.reuseB());

    if (reuse)
    {
        unsigned mask = 0;
        bool sharedOperandResue = reuseB ? false : true;

        // todo AlonG: verify that the SB Reuse setting works with all geometries
        // todo [SW-51662] Gaudi new descriptors: check SB Reuse in all geometries
        if (reuseA)
        {
            mask = getLoopMask(pattern, LOOP_K);
        }
        else if (reuseB)
        {
            mask = getLoopMask(pattern, LOOP_SPATIAL);
        }

        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            Mme::Desc& desc = descGroup.desc[descIdx];

            if (sharedOperandResue != m_geoParamsSPtr->isTransO())
            {
                // Reuse is in the dense direction
                desc.sbRepeat.repeatSMinus1 = m_reuseAttr.denseStepsNr - 1;
                desc.sbRepeat.aguSLoopMask = mask;
            }
            else
            {
                // Reuse is in the spatial direction
                desc.sbRepeat.repeatLMinus1 = m_reuseAttr.spatialStepsNr - 1;
                desc.sbRepeat.aguLLoopMask = mask;
            }
        }
    }
}
void MmeDedwDescriptorGenerator::setDescSBReuse(DescGroup& descGroup)
{
    const MmeLayerParams params = getOriginalParams();
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        Mme::Desc& desc = descGroup.desc[descIdx];
        // Just Initialization
        initSbRepeatFields(desc);
    }

    if (!params.strategy.sbReuse)
    {
        return;
    }

    unsigned mask;
    unsigned reuse;
    unsigned filter = 1;
    bool sharedOperandResue;
    auto pattern = params.strategy.pattern;
    auto desc0 = &descGroup.desc[0];

    unsigned filterMask = 0;
    for (unsigned d = 0; d < c_padded_conv_dim; d++)
    {
        unsigned f = getLoopSize(desc0, pattern, LOOP_FILTER + d);
        if (f)
        {
            filter *= f + 1;
            filterMask |= getLoopMask(pattern, LOOP_FILTER + d);
        }
    }
    filter--;

    switch (pattern)
    {
        case e_mme_sp_reduction_kfc:
            mask = getLoopMask(pattern, LOOP_C) | filterMask;
            reuse = ((getLoopSize(desc0, pattern, LOOP_C) + 1) * (filter + 1)) - 1;
            sharedOperandResue = false;
            break;
        case e_mme_sp_reduction_fkc:
            mask = getLoopMask(pattern, LOOP_C);
            reuse = getLoopSize(desc0, pattern, LOOP_C);
            sharedOperandResue = false;
            break;
        case e_mme_sp_reduction_fck:
            mask = getLoopMask(pattern, LOOP_K);
            reuse = getLoopSize(desc0, pattern, LOOP_K);
            sharedOperandResue = true;
            break;
        case e_mme_sp_reduction_cfk:
            mask = getLoopMask(pattern, LOOP_K);
            reuse = getLoopSize(desc0, pattern, LOOP_K);
            sharedOperandResue = true;
            break;
        case e_mme_sp_reduction_kcf:
            mask = getLoopMask(pattern, LOOP_C) | filterMask;
            reuse = ((getLoopSize(desc0, pattern, LOOP_C) + 1) * (filter + 1)) - 1;
            sharedOperandResue = false;
            break;
        case e_mme_sp_reduction_ckf:
            mask = filterMask;
            reuse = filter;
            sharedOperandResue = false;
            break;
        default:
            MME_ASSERT(0, "invalid pattern");
            mask = 0;
            reuse = 0;
            sharedOperandResue = false;
    }

    if (reuse)
    {
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            Mme::Desc& desc = descGroup.desc[descIdx];

            if (sharedOperandResue != m_geoParamsSPtr->isTransO())
            {
                // Reuse is in the dense direction
                desc.sbRepeat.repeatSMinus1 = reuse;
                desc.sbRepeat.aguSLoopMask = mask;
            }
            else
            {
                // Reuse is in the spatial direction
                desc.sbRepeat.repeatLMinus1 = reuse;
                desc.sbRepeat.aguLLoopMask = mask;
            }
        }
    }
}

void MmeDescriptorGenerator::setDescPerfFields()
{
    const MmeLayerParams params = getOriginalParams();
    switch (params.tracing.traceModeX)
    {
        case e_mme_trace_mode_none:
            break;
        case e_mme_trace_mode_layer_act:
            setOpStartAndEndEvents(&params, m_activations, e_mme_op_x);
            break;
        default:
            MME_ASSERT(0, "invalid trace mode");
    }
    switch (params.tracing.traceModeY)
    {
        case e_mme_trace_mode_none:
            break;
        case e_mme_trace_mode_layer_act:
            setOpStartAndEndEvents(&params, m_activations, e_mme_op_y);
            break;
        default:
            MME_ASSERT(0, "invalid trace mode");
    }
    switch (params.tracing.traceModeW)
    {
        case e_mme_trace_mode_none:
            break;
        case e_mme_trace_mode_layer_act:
            setOpStartAndEndEvents(&params, m_activations, e_mme_op_w);
            break;
        default:
            MME_ASSERT(0, "invalid trace mode");
    }
}

void MmeDescriptorGenerator::setDescRateLimiters(Mme::Desc& desc)
{
    desc.rateLimiter.aguL = 4;
    desc.rateLimiter.aguS = 4;
    desc.rateLimiter.aguO = 2;
}

void MmeDescriptorGenerator::setActivation(MmeActivation& activation, DescGroup& descGroup)
{
    auto params = getOriginalParams();
    activation.getDesc(0) = descGroup.desc[Mme::MME_CORE_NORTH_MASTER];
    activation.getDesc(1) = descGroup.desc[Mme::MME_CORE_SOUTH_MASTER];
    activation.numSignals = countSignals(descGroup.desc);
    activation.numTetrises = countTetrises(activation.getDesc(0));
    activation.numRollups = activation.numTetrises;
    activation.roiX.isSram = false;
    activation.roiW.isSram = false;
    activation.roiY.isSram = false;
    activation.roiX.isReduction = false;
    activation.roiW.isReduction = false;
    activation.roiY.isReduction = false;
    activation.isGemm = !params.isConvOperation();
    const MmeRecipe& recipe = getRecipe();
    activation.spView = recipe.curSp();
    activation.fcdView = recipe.curFcd();
    activation.nonSpatialView = recipe.curNonSpatial();
}

void MmeConvDescriptorGenerator::setPaddingValues(Mme::Desc& desc)
{
    const MmeLayerParams params = getParams();
    // todo: this setting is done to match the legacy code for memset desc, likely redundant
    // todo [SW-51591] Check and remove redundant settings
    if (m_convSubProblems.current->isMemsetDesc())
    {
        desc.paddingValueS = desc.paddingValueL = 0;
    }
    else
    {
        if (params.opType == e_mme_fwd)
        {
            MmeDW_New pv;
            pv.f32 = params.conv.paddingValue;
            if (params.getOperand(e_mme_op_a).elementType == EMmeDataType::e_type_bf16)
            {
                pv.u16[0] = pv.u16[1];
            }
            desc.paddingValueS = pv.u32;
        }
        else
        {
            desc.paddingValueS = 0;
        }

        desc.paddingValueL = 0;
    }
}

void MmeBgemmDescriptorGenerator::setPaddingValues(Mme::Desc& desc)
{
    desc.paddingValueS = 0;
    desc.paddingValueL = 0;
}

void MmeFwdDedxDescriptorGenerator::setDescLowering(DescGroup& descGroup)
{
    const MmeLayerParams params = getParams();
    bool lowerA = getRecipe().lowering;
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        if (m_geoParamsSPtr->isTransO())
        {
            descGroup.desc[descIdx].header.lowerL = lowerA ? 1 : 0;
            descGroup.desc[descIdx].sbRepeat.aguLLoopMask |=
                lowerA ? getLoopMask(params.strategy.pattern, LOOP_FILTER0) : 0;
        }
        else
        {
            descGroup.desc[descIdx].header.lowerS = lowerA ? 1 : 0;
            // todo AlonG: are these two lines redundant?
            descGroup.desc[descIdx].sbRepeat.aguSLoopMask |=
                lowerA ? getLoopMask(params.strategy.pattern, LOOP_FILTER0) : 0;
        }
    }
}

bool MmeDedwDescriptorGenerator::enableUnroll()
{
    const MmeLayerParams params = getParams();
    MmeRecipe recipe = getRecipe();
    auto gp = getGeoAttr();
    unsigned filter = recipe.getOperand(e_mme_op_b).sizes[2];
    for (unsigned i = 3; i < Mme::c_mme_max_tensor_dims; i++)
    {
        filter = std::max(filter, recipe.getOperand(e_mme_op_b).sizes[i]);
    }

    // todo AlonG: change back! unrollEn should be set from outside
    bool unroll = ((params.strategy.unrollEn) && (params.strategy.geometry == e_mme_geometry_4wx1h) &&
                   (recipe.getOperand(e_mme_op_c).sizes[0] <= gp->m_subMatrixWidth) && (filter > 1));
    return unroll;
}

void MmeDedwDescriptorGenerator::setDescLowering(DescGroup& descGroup)
{
    const MmeLayerParams params = getParams();
    // Do not apply lowering in some cases, primarily when SB Reuse is set
    if (params.strategy.sbReuse && (params.strategy.pattern != EMmePattern::e_mme_sp_reduction_kcf) &&
        (params.strategy.pattern != EMmePattern::e_mme_sp_reduction_ckf) &&
        (params.strategy.pattern != EMmePattern::e_mme_sp_reduction_cfk))
    {
        return;
    }

    bool lowerA = getRecipe().lowering;
    if (lowerA)
    {
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            if (m_geoParamsSPtr->isTransO())
            {
                descGroup.desc[descIdx].header.lowerL = lowerA;
            }
            else
            {
                descGroup.desc[descIdx].header.lowerS = lowerA;
            }
        }
    }
}

MmeCommon::MmeLayerParams MmeDescriptorGenerator::makeParamsForZeroCD(const MmeCommon::MmeLayerParams& params)
{
    MmeLayerParams newParams = params;
    setZeroCD(true);

    // In case of Zero CD, all we care is for setting the output to zero. For that, we can
    // create a whole new params that will perform the memset efficiently.
    // We do that by setting the opType to atb, and fixing the input operands to match the output.
    switch (newParams.opType)
    {
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_fwd:
            break;
        case e_mme_dedx:
            std::swap(newParams.x, newParams.y);  // set y to be of the same size of the output
            break;
        case e_mme_dedw:
            std::swap(newParams.w, newParams.y);  // set y to be of the same size of the output
            break;
        default:
            MME_ASSERT(0, "Invalid op type");
    }
    newParams.opType = e_mme_atb;
    newParams.strategy.pattern = e_mme_sp_reduction_fck;

    // Extract the dims of the output, and set them in the input tensors in atb order
    unsigned outputWidth = newParams.y.sizes[0];
    unsigned outputHeight = newParams.y.sizes[1];
    unsigned batchDim0 = newParams.y.sizes[2];
    unsigned batchDim1 = newParams.y.sizes[3];
    unsigned batchDim2 = newParams.y.sizes[4];

    newParams.x.sizes = {outputHeight, 1, batchDim0, batchDim1, batchDim2};
    newParams.w.sizes = {outputWidth, 1, batchDim0, batchDim1, batchDim2};

    // Update the strides to match the new dims
    newParams.x.strides[0] = 1;
    newParams.w.strides[0] = 1;
    for (int i = 1; i < Mme::c_mme_max_tensor_dims; i++)
    {
        newParams.x.strides[i] = newParams.x.strides[i - 1] * newParams.x.sizes[i - 1];
        newParams.w.strides[i] = newParams.w.strides[i - 1] * newParams.w.sizes[i - 1];
    }

    return newParams;
}

void MmeDedwDescriptorGenerator::fixUnrollDesc(DescGroup& descGroup)
{
    if (m_dedwUnroll.getUnrollFactor() == 1)  // no dedw unroll
    {
        return;  // No need to fix
    }

    auto new_params = getParams();
    auto originalParams = getOriginalParams();
    auto pattern = new_params.getPattern();
    Mme::Desc* desc0 = &(descGroup.desc[0]);

    const unsigned tensorDim = m_dedwUnroll.getUnrollDim();
    const unsigned weightDim = tensorDim + 1;
    const unsigned convDim = tensorDim - 1;
    const unsigned currUnrollFactor = m_dedwUnroll.getUnrollFactor();

    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        Mme::Desc* currDesc = &(desc0[descIdx]);

        currDesc->tensorS.loopStride[tensorDim] = 0;
        currDesc->tensorL.loopStride[tensorDim] = currDesc->tensorL.spatialStrides[tensorDim - 1] * -currUnrollFactor;
        currDesc->tensorO.loopStride[weightDim] = currDesc->tensorO.spatialStrides[weightDim - 1] * currUnrollFactor;

        setLoopDim(currDesc, pattern, LOOP_FILTER + convDim, OP_L, tensorDim);
        setLoopDim(currDesc, pattern, LOOP_FILTER + convDim, OP_O, weightDim);
        setLoopDim(currDesc, pattern, LOOP_FILTER + convDim, OP_S, Mme::c_mme_max_tensor_dims);

        unsigned loopSize = getLoopSize(currDesc, pattern, LOOP_FILTER + convDim) / currUnrollFactor;
        setLoopSize(currDesc, pattern, LOOP_FILTER + convDim, loopSize);

        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_remote].startOffset[tensorDim - 1], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_local].startOffset[tensorDim - 1], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_remote].startOffset[weightDim - 1], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_local].startOffset[weightDim - 1], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_remote].roiBaseOffset[tensorDim], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_local].roiBaseOffset[tensorDim], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_remote].roiBaseOffset[weightDim], "expected roiBaseOffset = 0");
        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_local].roiBaseOffset[weightDim], "expected roiBaseOffset = 0");

        currDesc->aguL[Mme::e_mme_local].roiBaseOffset[0] = desc0->aguL[Mme::e_mme_local].roiBaseOffset[0];
        currDesc->aguO[Mme::e_mme_local].roiBaseOffset[0] = desc0->aguO[Mme::e_mme_local].roiBaseOffset[0];

        const unsigned subProblemIdx = m_convSubProblems.current->key;
        int RDMinusP = (originalParams.conv.dilation[convDim] * subProblemIdx) - originalParams.conv.padding[convDim];
        currDesc->aguL[Mme::e_mme_local].roiBaseOffset[tensorDim] =
            currDesc->tensorL.spatialStrides[tensorDim - 1] *
            -(div_round_down(RDMinusP, (int) originalParams.conv.stride[convDim]) + descIdx);

        currDesc->aguO[Mme::e_mme_local].roiBaseOffset[weightDim] =
            currDesc->tensorO.spatialStrides[weightDim - 1] * descIdx;

        currDesc->aguL[Mme::e_mme_remote].roiBaseOffset[0] = desc0->aguL[Mme::e_mme_local].roiBaseOffset[0];
        currDesc->aguO[Mme::e_mme_remote].roiBaseOffset[0] = desc0->aguO[Mme::e_mme_local].roiBaseOffset[0];

        currDesc->aguL[Mme::e_mme_remote].roiBaseOffset[tensorDim] =
            currDesc->aguL[Mme::e_mme_local].roiBaseOffset[tensorDim] -
            (Mme::MME_MASTERS_NR * currDesc->tensorL.spatialStrides[tensorDim - 1]);

        currDesc->aguO[Mme::e_mme_remote].roiBaseOffset[weightDim] =
            currDesc->aguO[Mme::e_mme_local].roiBaseOffset[weightDim] +
            (Mme::MME_MASTERS_NR * currDesc->tensorO.spatialStrides[weightDim - 1]);

        unsigned wDataSize = (new_params.w.elementType == e_type_bf16) ? 2 : 4;
        union
        {
            uint32_t baseAddressUint32[2];
            uint64_t baseAddress;
        } outputBaseAddress;
        outputBaseAddress.baseAddressUint32[0] = currDesc->baseAddrLowO;
        outputBaseAddress.baseAddressUint32[1] = currDesc->baseAddrHighO;
        outputBaseAddress.baseAddress += subProblemIdx * originalParams.w.strides[weightDim] * wDataSize;

        currDesc->baseAddrLowO = outputBaseAddress.baseAddressUint32[0];
        currDesc->baseAddrHighO = outputBaseAddress.baseAddressUint32[1];
    }
}

void MmeBgemmDescriptorGenerator::fixDedwAsBgemm(DescGroup& descGroup)
{
    if (!isDedwAsBgemm())
    {
        return;
    }

    // Fixing the descriptors include:
    // 1. Map the output of the second descriptor onto the first
    // 2. In case of odd CD, reduce the validElement for the CD of the second desc by 1

    Mme::Desc& desc0 = descGroup.desc[0];
    Mme::Desc& desc1 = descGroup.desc[1];

    // 1. Map the output of the second descriptor onto the first
    MME_ASSERT(desc0.aguO[Mme::e_mme_local].roiBaseOffset[1] == 0, "Expected all offsets of desc0 to be 0");

    for (int i = 0; i < Mme::e_mme_local_and_remote; i++)
    {
        for (int j = 0; j < Mme::c_mme_max_tensor_dims; j++)
        {
            // Sanity check...
            MME_ASSERT(desc0.aguO[i].roiBaseOffset[j] == 0 || j == 0, "Expected all offsets of desc0 to be 0");
            desc1.aguO[i].roiBaseOffset[j] = desc0.aguO[i].roiBaseOffset[j];
        }
    }

    // 2. In case of odd CD, reduce the validElement for the CD of the second desc by 1
    // A is transposed, so the CD is at dim 1
    if (isDedwAsBgemmOddCD())
    {
        const MultiDimSubView& batchView = getRecipe().curNonSpatial();

        desc1.tensorS.validElements[1] -= getRecipe().getOperand(e_mme_op_a).strides[1];
        desc1.tensorL.validElements[1] -= getRecipe().getOperand(e_mme_op_b).strides[1];
    }
}
void MmeDescriptorGenerator::buildDesc(DescGroup& descGroup)
{
    Mme::Desc localDesc;

    setReuseAttr(*m_geoParamsSPtr, getParams().opType, getRecipe().reuseA(), getRecipe().reuseB());

    // todo AlonG: Remove these resets and verify functionality
    memset(&localDesc, 0, sizeof(Mme::Desc));  // to be on the safe side. Todo: remove
    resetDescPerfFields(localDesc);

    setDescHeader(localDesc);

    initSignalingInfo(localDesc);
    buildDescTensorPointers(localDesc);
    setDescRateLimiters(localDesc);
    resetMetaDataFields(localDesc);
    setUserDataFields(localDesc);
    setPaddingValues(localDesc);
    setAddressOffsets(localDesc);

    // Now, set the desc group
    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
    {
        descGroup.desc[descIdx] = localDesc;
    }

    setAguConfig(descGroup);
    // todo: add support for reuse in bgemm
    setDescSBReuse(descGroup);
    handlePartials(descGroup);
    setDescLowering(descGroup);
    setHeaderLoopSelectors(descGroup);

    fixUnrollDesc(descGroup);
}

void MmeDescriptorGenerator::handleSignalOverflow(DescGroup& descGroup)
{
    unsigned signalsNum = countSignals(descGroup.desc);
    if (signalsNum > gaudi::MmeHalReader::getInstance().getSyncObjectMaxValue())
    {
        auto newParams = getParams();
        newParams.controls.signalingMode = MmeCommon::e_mme_signaling_desc;
        newParams.controls.squashIORois = true;
        setCurParams(newParams);
        setSignals(descGroup, false);  // false- no need to fill isLast in signaling desc
        m_signalOverflow = true;
    }
}

const MmeRecipe& MmeDescriptorGenerator::getRecipe() const
{
    if (m_convSubProblems.current != nullptr)
    {
        return m_convSubProblems.current->recipe;
    }
    return m_recipe;
}

void MmeConvDescriptorGenerator::mmeGenerateActivations()
{
    m_activations.clear();

    MME_ASSERT(!m_convSubProblems.empty(), "should have at least one sub-problem");

    for (unsigned subProblemIdx = 0; subProblemIdx < m_convSubProblems.size(); subProblemIdx++)
    {
        DescGroup descGroup;
        m_convSubProblems.current = &m_convSubProblems[subProblemIdx];
        auto recipe = getRecipe();
        setCurParams(m_convSubProblems.current->params);
        auto params = getParams();
        auto& recipeIterator = getRecipe().getIterator();
        for (auto iters : recipeIterator)
        {
            recipeIterator.setCurIterVals(iters);
            m_activations.emplace_back(gaudi::MmeHalReader::getInstance().getMmeNr());
            MmeActivation& act = m_activations.back();
            buildDesc(descGroup);
            setSignals(descGroup, getRecipe().getIterator().isLast() && m_convSubProblems.isLast());
            handleSignalOverflow(descGroup);
            setActivation(act, descGroup);
        }
    }
    const bool isLastActivation = getRecipe().getIterator().isLast();
    MME_ASSERT(isLastActivation, "should be on last activation");
    setDescPerfFields();

    // todo AlonG: get rid of ExecuteParams in next gerrit. Currently the function activation2Roi
    // is used also by the old conv flow
    MmeLayerParams params = getOriginalParams();
    ExecuteParams execParams(getOriginalParams().opType);
    execParams.controls = &params.controls;
    activation2Roi(execParams);
}

void MmeBgemmDescriptorGenerator::mmeGenerateActivations()
{
    DescGroup descGroup;
    auto& recipeIterator = getRecipe().getIterator();

    for (auto iters : recipeIterator)
    {
        recipeIterator.setCurIterVals(iters);
        m_activations.emplace_back(gaudi::MmeHalReader::getInstance().getMmeNr());
        MmeActivation& act = m_activations.back();
        buildDesc(descGroup);
        setSignals(descGroup, getRecipe().getIterator().isLast());

        // patch the descriptors in case of dedwAsBgemm
        fixDedwAsBgemm(descGroup);
        handleSignalOverflow(descGroup);
        setActivation(act, descGroup);
    }
    setDescPerfFields();

    // todo AlonG: get rid of ExecuteParams in next gerrit. Currently the function activation2Roi
    // is used also by the old conv flow
    MmeLayerParams params = getParams();
    ExecuteParams execParams(getOriginalParams().opType);
    execParams.controls = &params.controls;
    activation2Roi(execParams);
}

//=========================================================================================
//===== Constructor functions =============================================================
//=========================================================================================

//================== dedx recipe generation ======================
bool MmeConvDescriptorGenerator::extractGcdFromConvParams(std::array<unsigned, MME_MAX_CONV_DIMS - 1>* stride,
                                                          std::array<unsigned, MME_MAX_CONV_DIMS - 1>* dilation,
                                                          std::array<unsigned, MME_MAX_CONV_DIMS - 1>* commonDivs) const
{
    bool hasGcd = false;
    // divide both stride and dilation by their gcd, and keep that gcd in a separate list to be used later
    for (unsigned i = 0; i < MME_MAX_CONV_DIMS - 1; ++i)
    {
        unsigned gcd = std::__gcd((*stride)[i], (*dilation)[i]);
        if (gcd != 0 && gcd != 1)
        {
            hasGcd = true;
            (*dilation)[i] /= gcd;
            (*stride)[i] /= gcd;
            (*commonDivs)[i] = gcd;
        }
    }
    return hasGcd;
}

unsigned MmeConvDescriptorGenerator::getTotalDedxNumOfDesc() const
{
    unsigned numOfSubProblems = multiplyElements(getParams().conv.stride.begin(), getParams().conv.stride.end());
    if (getOriginalParams().strategy.packingFactor > 1)
    {
        //  currently packing is supported only for convolutions without strides.
        //  packing increases the conv stride by packing factor on the first dimension.
        //  after packing we expect the conv strides to be [packingFactor, 1, 1]
        //  thus we expect numOfSubProblems to be equal to the packingFactor.
        MME_ASSERT(numOfSubProblems == getOriginalParams().strategy.packingFactor,
                   "conv stride is not yet supported with dedx packing");
        return 1;
    }
    return numOfSubProblems;
}

unsigned MmeConvDescriptorGenerator::calcTotalNumOfSubProblems(const CommonGeoAttr& geoAttr) const
{
    switch (getParams().opType)
    {
        case e_mme_dedx:
            return getTotalDedxNumOfDesc();
        case e_mme_dedw:
            return getTotalDedwNumOfDesc();
        case e_mme_fwd:
            return MmeCommon::RecurringMisalignmentOptimization::getNumSubProblems(getParams(),
                                                                                   geoAttr,
                                                                                   getMmeHal(e_mme_Gaudi));
        default:
            return 1;
    }
}

void MmeConvDescriptorGenerator::makeParamsForDedxSubProblem(unsigned numOfSubProblems, unsigned subProblemIdx)
{
    MmeLayerParams originalParams = getOriginalParams();
    MmeLayerParams& convParamsForSubProblem = m_convSubProblems.current->params;
    OffsetArray& descAddrOffset = m_convSubProblems.current->addressOffset;

    if (originalParams.strategy.packingFactor > 1)
    {
        // adjust padding to packing factor.
        convParamsForSubProblem.conv.padding[0] += originalParams.strategy.packingFactor - 1;
        return;
    }
    // Modify pipeline level
    convParamsForSubProblem.strategy.pipelineLevel =
        div_round_up(convParamsForSubProblem.strategy.pipelineLevel, numOfSubProblems);

    // group offset
    unsigned K0 = subProblemIdx;

    // for each group the formula to get the input location from output (Dy location from DX)
    // M = N + KD - (K0D-P)/S
    // S' = 1 , D' = D , P' = (K0D-P)/S
    unsigned remX = 0, remW = 0;
    std::fill(convParamsForSubProblem.conv.stride.begin(), convParamsForSubProblem.conv.stride.end(), 1);

    // if gcd(dilation[i], strides[i]) > 1 for some 0 <= i <= 2, transfer it to the tensors stride
    auto dilation = originalParams.conv.dilation;
    auto strides = originalParams.conv.stride;
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> commonDivs = {1, 1, 1};
    bool hasGcd = extractGcdFromConvParams(&strides, &dilation, &commonDivs);
    if (hasGcd)
    {
        for (unsigned convDim = 0; convDim < MME_MAX_CONV_DIMS - 1; ++convDim)
        {
            if (commonDivs[convDim] != 1)
            {
                convParamsForSubProblem.conv.dilation[convDim] = dilation[convDim];

                unsigned tensorDim = convDim + 1;
                convParamsForSubProblem.x.strides[tensorDim] *= commonDivs[convDim];
                convParamsForSubProblem.x.sizes[tensorDim] /= commonDivs[convDim];
            }
        }
    }

    // each descriptor takes the relevant weights, and is written strided to the output dx.
    unsigned rem = K0;
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> offsetWithinGcd;
    for (int convDim = Mme::c_mme_max_conv_dims - 2; convDim >= 0; --convDim)
    {
        unsigned weightDim = convDim + 2;
        offsetWithinGcd[convDim] = rem % commonDivs[convDim];
        convParamsForSubProblem.w.bases[weightDim] = (rem / commonDivs[convDim]) % strides[convDim];
        descAddrOffset.wOffset[weightDim] =
            convParamsForSubProblem.w.bases[weightDim] * convParamsForSubProblem.w.strides[weightDim];
        rem /= originalParams.conv.stride[convDim];
    }

    for (unsigned convDim = 0; convDim < Mme::c_mme_max_conv_dims - 1; convDim++)
    {
        unsigned tensorDim = convDim + 1;
        unsigned weightDim = convDim + 2;

        int padding = div_round_down(originalParams.conv.padding[convDim], commonDivs[convDim]);
        int paddingRem = 0;
        if (hasGcd)
        {
            // fix padding and offset in case of stride & dilation gcd
            paddingRem = originalParams.conv.padding[convDim] % commonDivs[convDim];

            if (offsetWithinGcd[convDim] < paddingRem)
            {
                if (paddingRem != 0)
                {
                    padding++;
                }
                paddingRem = (commonDivs[convDim] - originalParams.conv.padding[convDim]) % commonDivs[convDim];
            }
            else
            {
                paddingRem = -paddingRem;
            }
        }

        int K0DMinusP = (convParamsForSubProblem.w.bases[weightDim] * dilation[convDim]) - padding;
        int newPadding = -div_round_down(K0DMinusP, strides[convDim]);
        convParamsForSubProblem.conv.padding[convDim] = newPadding;
        convParamsForSubProblem.x.bases[tensorDim] = mod_neg(K0DMinusP, strides[convDim]);
        descAddrOffset.xOffset[tensorDim] =
            convParamsForSubProblem.x.bases[tensorDim] * convParamsForSubProblem.x.strides[tensorDim];
        // fix xOffset to match the original padding, and the gcd part of the offset which is not present in x.bases
        descAddrOffset.xOffset[tensorDim] +=
            (paddingRem + offsetWithinGcd[convDim]) * originalParams.x.strides[tensorDim];
        if (convParamsForSubProblem.x.bases[tensorDim] > 0 && originalParams.conv.stride[convDim] == 2 &&
            ((originalParams.x.sizes[tensorDim] % originalParams.conv.stride[convDim]) != 0) &&
            originalParams.strategy.dedxDynamicPadding)
        {
            //  expand the subproblem by a single row to align both descriptors for dynamic patching
            descAddrOffset.xOffset[tensorDim] = -originalParams.x.strides[tensorDim];
            descAddrOffset.yOffset[tensorDim] = -originalParams.y.strides[tensorDim];
        }
        // write each desc to output (X) strided
        remX =
            (convParamsForSubProblem.x.bases[tensorDim] * commonDivs[convDim]) + paddingRem + offsetWithinGcd[convDim];
        convParamsForSubProblem.x.strides[tensorDim] *= strides[convDim];
        convParamsForSubProblem.x.sizes[tensorDim] /= strides[convDim];
        convParamsForSubProblem.x.bases[tensorDim] /= strides[convDim];
        // read each desc from w strided
        remW = convParamsForSubProblem.w.bases[weightDim] % strides[convDim];
        convParamsForSubProblem.w.strides[weightDim] *= strides[convDim];
        convParamsForSubProblem.w.sizes[weightDim] /= strides[convDim];
        convParamsForSubProblem.w.bases[weightDim] /= strides[convDim];
        // add the remainder
        if (remX < (originalParams.x.sizes[tensorDim] % originalParams.conv.stride[convDim]))
        {
            convParamsForSubProblem.x.sizes[tensorDim]++;
        }
        if (remW < (originalParams.w.sizes[weightDim] % strides[convDim]))
        {
            convParamsForSubProblem.w.sizes[weightDim]++;
        }
    }

    if (hasGcd)
    {
        // if gcd(dilation[i], strides[i]) > 1 for some 0 <= i <= 2, some subproblems are empty and should be
        // replaced with memset
        for (int convDim = Mme::c_mme_max_conv_dims - 2; convDim >= 0; --convDim)
        {
            if (offsetWithinGcd[convDim] != 0)
            {
                unsigned weightDim = convDim + 2;
                convParamsForSubProblem.w.sizes[weightDim] = 0;  // this will turn into memset desc later
            }
        }
    }
}

//================== dedw recipe generation ======================
unsigned MmeConvDescriptorGenerator::getTotalDedwNumOfDesc() const
{
    if (m_dedwUnroll.getUnrollFactor() > 1)
    {
        MME_ASSERT(m_dedwUnroll.getUnrollDim() >= 1 && m_dedwUnroll.getUnrollDim() <= 3,
                   "expected unroll dim to be between 1-3 for unroll factor > 1");
        return getOriginalParams().conv.stride[m_dedwUnroll.getUnrollDim() - 1];
    }
    return 1;  // no unroll
}

void MmeConvDescriptorGenerator::makeParamsForDedwSubProblem(unsigned numOfSubProblems, unsigned subProblemIdx)
{
    if (m_dedwUnroll.getUnrollFactor() == 1)  // if no unroll, return
    {
        return;
    }

    const MmeLayerParams originalParams = getOriginalParams();
    MmeLayerParams& new_params = m_convSubProblems.current->params;
    OffsetArray& descAddrOffset = m_convSubProblems.current->addressOffset;

    unsigned weightDim = m_dedwUnroll.getUnrollDim() + 1;
    unsigned tensorDim = m_dedwUnroll.getUnrollDim();
    unsigned convDim = m_dedwUnroll.getUnrollDim() - 1;

    // Modify pipeline level
    new_params.strategy.pipelineLevel = div_round_up(new_params.strategy.pipelineLevel, numOfSubProblems);

    new_params.w.strides[weightDim] *= originalParams.conv.stride[convDim];
    new_params.w.sizes[weightDim] /= originalParams.conv.stride[convDim];
    if (subProblemIdx < (originalParams.w.sizes[weightDim] % originalParams.conv.stride[convDim]))
    {
        new_params.w.sizes[weightDim]++;
    }

    int RDMinusP = (originalParams.conv.dilation[convDim] * subProblemIdx) - originalParams.conv.padding[convDim];
    new_params.conv.padding[convDim] = -mod_neg(RDMinusP, originalParams.conv.stride[convDim]);

    setCurParams(new_params);
}

//==================== memset descriptor handling ======================
bool MmeConvDescriptorGenerator::shouldAddMemsetDesc(const MmeLayerParams& newParams) const
{
    const bool shouldAddMemset = hasZeros(newParams.w.sizes);
    return shouldAddMemset;
}
//==================== Identify and patch params for zero CD ============
bool MmeDescriptorGenerator::isZeroCD(const MmeLayerParams& params)
{
    unsigned cdDimInA = isTransposed(params.opType, e_mme_in_a) ? 0 : 1;
    unsigned cdDimInB = isTransposed(params.opType, e_mme_in_b) ? 0 : 1;
    unsigned cdA = params.getOperand(e_mme_op_a).sizes[cdDimInA];
    unsigned cdB = params.getOperand(e_mme_op_b).sizes[cdDimInB];
    MME_ASSERT(((cdA == 0) && (cdB == 0)) || ((cdA != 0) && (cdB != 0)),
               "Mismatch in CD values between the two input tensors");

    return (params.getOperand(e_mme_op_b).sizes[cdDimInB] == 0);
}
bool MmeConvDescriptorGenerator::skipRecipeGeneration(const MmeLayerParams& params)
{
    const unsigned spSize = multiplyElements(params.x.sizes.begin() + 1, params.x.sizes.end());
    return spSize == 0;
}

std::unique_ptr<MmeDescriptorGenerator>
MmeConvDescriptorGenerator::createMmeConvDescGenerator(const MmeCommon::MmeLayerParams& params)
{
    if (params.opType == MmeCommon::EMmeOpType::e_mme_fwd || params.opType == MmeCommon::EMmeOpType::e_mme_dedx)
    {
        return std::make_unique<MmeFwdDedxDescriptorGenerator>(params);
    }
    return std::make_unique<MmeDedwDescriptorGenerator>(params);
}

void MmeConvDescriptorGenerator::mmeCalcLayerRecipe()
{
    MmeLayerParams params = getParams();

    const bool isDedx = (params.opType == e_mme_dedx);
    const bool isDedw = (params.opType == e_mme_dedw);
    const unsigned numOfSubProblems = calcTotalNumOfSubProblems(*m_commonGeoAttrSPtr);
    for (unsigned currSubProblemIdx = 0; currSubProblemIdx < numOfSubProblems; currSubProblemIdx++)
    {
        m_convSubProblems.push(params);
        if (isDedx)
        {
            makeParamsForDedxSubProblem(numOfSubProblems, currSubProblemIdx);
            auto& newParams = m_convSubProblems.current->params;
            bool isMemsetDesc = shouldAddMemsetDesc(newParams);
            if (isMemsetDesc)
            {
                newParams.strategy.pipelineLevel = 1;
                newParams.strategy.sbReuse = false;
                m_convSubProblems.current->setMemsetDesc(true);
            }
            if (skipRecipeGeneration(newParams))
            {
                m_convSubProblems.pop();
                continue;
            }
        }
        if (isDedw)
        {
            makeParamsForDedwSubProblem(numOfSubProblems, currSubProblemIdx);
        }

        setCurParams(m_convSubProblems.current->params);
        RecipeGenerator recipeGen(e_mme_conv_recipe,
                                  getParams(),
                                  gaudi::MmeHalReader::getInstance(),
                                  *m_commonGeoAttrSPtr);
        m_convSubProblems.current->recipe = recipeGen.generateRecipe();
        m_recipe = m_convSubProblems.current->recipe;
    }
}

void MmeBgemmDescriptorGenerator::mmeCalcLayerRecipe()
{
    MmeLayerParams params = getParams();
    RecipeGenerator recipeGen(e_mme_bgemm_recipe, params, gaudi::MmeHalReader::getInstance(), *m_commonGeoAttrSPtr, true);
    m_recipe = recipeGen.generateRecipe();
}

bool MmeDescriptorGenerator::validateParams(const MmeLayerParams& params, std::string& errorMsg)
{
    // run some basic checks
    auto operandAType = params.getOperand(MmeCommon::e_mme_op_a).elementType;
    auto operandBType = params.getOperand(MmeCommon::e_mme_op_b).elementType;
    if (operandAType != operandBType)
    {
        errorMsg = "operand A and B element types should match";
        return false;
    }
    bool sizeCheck = true;
    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_dedw:
            // TODO: fix for conv + padding\strides\dilation.
            if ((params.w.sizes[0] != params.y.sizes[0]))
            {
                errorMsg += "w.sizes[0] != y.sizes[0] ";
                sizeCheck = false;
            }
            if (params.w.sizes[1] != params.x.sizes[0])
            {
                errorMsg += "w.sizes[1] != x.sizes[0] ";
                sizeCheck = false;
            }
            break;
        case e_mme_ab:  // X[c,n] x W[m,c] = Y[m,n]
            if (params.y.sizes[0] != params.w.sizes[0])
            {
                errorMsg += "y.sizes[0] != w.sizes[0] ";
                sizeCheck = false;
            }
            if (params.x.sizes[0] != params.w.sizes[1])
            {
                errorMsg += "x.sizes[0] != w.sizes[1] ";
                sizeCheck = false;
            }
            if (params.y.sizes[1] != params.x.sizes[1])
            {
                errorMsg += "y.sizes[1] != x.sizes[1] ";
                sizeCheck = false;
            }
            break;
        case e_mme_abt:
            if (params.y.sizes[0] != params.w.sizes[1])
            {
                errorMsg += "y.sizes[0] != w.sizes[1] ";
                sizeCheck = false;
            }
            if (params.x.sizes[0] != params.w.sizes[0])
            {
                errorMsg += "x.sizes[0] != w.sizes[0] ";
                sizeCheck = false;
            }
            if (params.y.sizes[1] != params.x.sizes[1])
            {
                errorMsg += "y.sizes[1] != x.sizes[1] ";
                sizeCheck = false;
            }
            break;
        case e_mme_atb:
            if (params.y.sizes[0] != params.w.sizes[0])
            {
                errorMsg += "y.sizes[0] != w.sizes[0] ";
                sizeCheck = false;
            }
            if (params.x.sizes[1] != params.w.sizes[1])
            {
                errorMsg += "x.sizes[1] != w.sizes[1] ";
                sizeCheck = false;
            }
            if (params.y.sizes[1] != params.x.sizes[0])
            {
                errorMsg += "y.sizes[1] != x.sizes[1] ";
                sizeCheck = false;
            }
            break;
        case e_mme_atbt:
            if (params.y.sizes[0] != params.w.sizes[1])
            {
                errorMsg += "y.sizes[0] != w.sizes[1] ";
                sizeCheck = false;
            }
            if (params.x.sizes[1] != params.w.sizes[0])
            {
                errorMsg += "x.sizes[1] != w.sizes[0] ";
                sizeCheck = false;
            }
            if (params.y.sizes[1] != params.x.sizes[0])
            {
                errorMsg += "y.sizes[1] != x.sizes[0] ";
                sizeCheck = false;
            }
            break;
        default:
            errorMsg = "invalid operation";
            return false;
    }
    if (!sizeCheck)
    {
        errorMsg = "input\\output sizes are invalid";
        return false;
    }

    bool walkPatternCheck = true;
    // Check the validity of the walking pattern
    switch (params.opType)
    {
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
            switch (params.getPattern())
            {
                case e_mme_sp_reduction_ckf:
                case e_mme_sp_reduction_kcf:
                case e_mme_sp_reduction_fck:
                case e_mme_sp_reduction_fkc:
                    break;  // we currently supports only these patterns
                default:
                    walkPatternCheck = false;
            }
            break;
        case e_mme_dedx:
        case e_mme_fwd:
            switch (params.getPattern())
            {
                case e_mme_z_reduction_ksf:
                case e_mme_z_reduction_skf:
                    break;
                default:
                    walkPatternCheck = false;
            }
            break;
        case e_mme_dedw:
            switch (params.getPattern())
            {
                case e_mme_sp_reduction_kfc:
                case e_mme_sp_reduction_fkc:
                case e_mme_sp_reduction_fck:
                case e_mme_sp_reduction_cfk:
                case e_mme_sp_reduction_kcf:
                case e_mme_sp_reduction_ckf:
                    break;
                default:
                    walkPatternCheck = false;
            }
            break;
        default:
        {
            errorMsg = "invalid operation";
            return false;
        }
    }
    if (!walkPatternCheck)
    {
        errorMsg = "Walking pattern is not compatible with operation";
        return false;
    }

    const auto& a = params.getOperand(e_mme_op_a);
    const auto& b = params.getOperand(e_mme_op_b);
    const auto& c = params.getOperand(e_mme_op_c);

    if (params.isGemmOperation())
    {
        for (int batchDim = 2; batchDim < MmeCommon::c_batchDimNr + 2; batchDim++)
        {
            // Batches of A/B can be 1 (broadcast), otherwise they must be equal to batches of C (classic bgemm)
            if (!(((a.sizes[batchDim] == 1 || a.sizes[batchDim] == c.sizes[batchDim]) &&
                   ((b.sizes[batchDim] == 1 || b.sizes[batchDim] == c.sizes[batchDim])))))
            {
                errorMsg = "batch dim - " + std::to_string(batchDim) + " does not match between input and output.";
                return false;
            }
        }
    }

    return true;
}
void MmeConvDescriptorGenerator::setAguConfig(DescGroup& descGroup)
{
    const MmeLayerParams params = getOriginalParams();
    auto recipe = getRecipe();
    if (m_convSubProblems.current->isMemsetDesc())
    {
        AguMemsetConfig AguMemsetConfig(recipe, m_curParams, m_reuseAttr, m_convSubProblems.current, getRoi());
        AguMemsetConfig.setAgu(descGroup);
    }
    else
    {
        switch (params.opType)
        {
            case e_mme_fwd:
            case e_mme_dedx:
            {
                AguConvFwdDedxConfig aguConfig(recipe, m_curParams, m_reuseAttr, m_convSubProblems.current, getRoi());
                aguConfig.setAgu(descGroup);
            }
            break;
            case e_mme_dedw:
            {
                AguConvDedwConfig aguConfig(recipe, m_curParams, m_reuseAttr, getRoi());
                aguConfig.setAgu(descGroup);
            }
            break;
            default:
                MME_ASSERT(0, "Unsupported operation type");
        }
    }
}

bool MmeDescriptorGenerator::isDedwAsBgemm(MmeCommon::MmeLayerParams params)
{
    if (params.opType != e_mme_dedw)
    {
        return false;
    }
    // Verify that the optimization is enabled.
    // For its enabling, a memset must precede and the output tensor must reside in sram
    if (!params.strategy.dedwAsBgemmEn)
    {
        return false;
    }
    if (params.x.elementType == MmeCommon::e_type_bf16 && ((params.w.sizes[0] > 128) || (params.w.sizes[1] > 64)))
    {
        return false;
    }
    if (params.x.elementType == MmeCommon::e_type_fp32 && ((params.w.sizes[0] > 64) || (params.w.sizes[1] > 32)))
    {
        return false;
    }
    // Verify that all weight dims are 1
    for (unsigned d = 0; d < Mme::c_mme_max_conv_dims; d++)
    {
        unsigned kDim = d + 2;
        if (params.w.sizes[kDim] != 1)
        {
            return false;
        }
    }
    // Verify that there is no internal padding within any of the tensors
    if (params.x.isStrided() || params.w.isStrided() || params.y.isStrided())
    {
        return false;
    }
    // Currently only 2w2h geometry is supported
    if (params.strategy.geometry != e_mme_geometry_2wx2h)
    {
        return false;
    }
    // Sanity checks
    if ((params.strategy.packingFactor != 1))
    {
        return false;
    }

    return true;
}

MmeTensorView createTensorView(EMmeDataType dataType, const SizeArray& sizes)
{
    MmeTensorView view;
    view.elementType = dataType;
    view.sizes = sizes;
    view.strides[0] = 1;
    for (unsigned dim = 1; dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        view.strides[dim] = view.strides[dim - 1] * view.sizes[dim - 1];
    }
    return view;
}

MmeLayerParams MmeDescriptorGenerator::makeParamsForDedwAsBgemm(const MmeLayerParams& params)
{
    MME_ASSERT(MmeDescriptorGenerator::isDedwAsBgemm(params), "This function should be called when dedwAsBgemm is set");

    // Change param fields in order to implement the specific dedw as batch-gemm of two halfs
    // This includes:
    // - Change the operation to atb
    // - Change the input sizes from {C,W,H,D,B} to {C,W*H*D*B/2,2,1,1} (A is transposed)
    // - Change the output sizes from {K,W,H,D,B} to {K,W*H*D*B/2,2,1,1}
    // - Change the output sizes to {K,C,2,1,1}
    // This way we perform the bgemm on the two batch dims concurrently.
    // Throughout the desc gen, using the isDedwAsBgemm flag, we will map the two outputs to
    // the same tensor using Acc.

    setDedwAsBgemm(true);
    MmeLayerParams newParams = params;
    newParams.opType = e_mme_atb;

    SizeArray inSizes = params.x.sizes;
    SizeArray outSizes = params.w.sizes;
    unsigned K = outSizes[0];
    unsigned C = outSizes[1];
    unsigned CD = inSizes[1] * inSizes[2] * inSizes[3] * inSizes[4];
    setDedwAsBgemmOddCD(CD % 2);

    unsigned newCD = div_round_up(CD, 2);  // In case CD is odd, we will patch the 2nd desc validElements
    SizeArray newXSizes = {C, newCD, 2, 1, 1};
    SizeArray newWSizes = {K, newCD, 2, 1, 1};
    SizeArray newYSizes = {K, C, 2, 1, 1};
    newParams.x = createTensorView(params.x.elementType, newXSizes);
    newParams.w = createTensorView(params.y.elementType, newWSizes);
    newParams.y = createTensorView(params.w.elementType, newYSizes);

    // The op is executed in a single step. So we don't care for the pattern. We will
    // force it to a pattern that is supported by bgemm.
    newParams.strategy.pattern = MmeCommon::e_mme_sp_reduction_fck;

    // Set the Reduction fields in the params
    newParams.controls.atomicAdd = 1;

    return newParams;
}

void MmeBgemmDescriptorGenerator::setAguConfig(DescGroup& descGroup)
{
    AguBGemmConfig aguConfig(getRecipe(), m_curParams, m_reuseAttr, getZeroCD());
    aguConfig.setAgu(descGroup);
}

MmeLayerParams MmeDescriptorGenerator::processParams(const MmeLayerParams& params)
{
    // Set the default values
    m_zeroCD = false;
    m_dedwAsBgemm = false;
    m_dedwAsBgemmOddCD = false;

    // The function updates params when specific conditions are met
    if (isDedwAsBgemm(params))
    {
        return makeParamsForDedwAsBgemm(params);
    }
    if (isZeroCD(params))
    {
        return makeParamsForZeroCD(params);
    }

    return params;
}

MmeBgemmDescriptorGenerator::MmeBgemmDescriptorGenerator(const MmeLayerParams& params) : MmeDescriptorGenerator(params)
{
    std::string errorMsg = "";
    if (!validateParams(params, errorMsg))
    {
        MME_ASSERT(0, fmt::format("BatchGemm parameters are not valid - {}", errorMsg).c_str());
    }
    mmeCalcLayerRecipe();
}

MmeDescriptorGenerator::MmeDescriptorGenerator(const MmeCommon::MmeLayerParams& params)
: m_originalParams(params),
  m_curParams(processParams(params)),
  m_geoParamsSPtr(std::make_shared<GeoAttr>(getParams(), gaudi::MmeHalReader::getInstance())),
  m_commonGeoAttrSPtr(std::make_shared<GaudiGeoAttr>(getParams())),
  m_convSubProblems(e_mme_Gaudi),
  m_roiCalculator(createRoiCalculator())
{
    if (isZeroCD(params))
    {
        m_commonGeoAttrSPtr->setATranspose(false);
        m_commonGeoAttrSPtr->setBTranspose(false);
    }

    auto* dumpMmeParamsEnvPtr = getenv("DUMP_MME_PARAMS_TO_CFG_FILE");
    if (dumpMmeParamsEnvPtr != nullptr)
    {
        MmeCommon::MmeParamsDumper(params).dumpMmeParamsForGaudiCfg(dumpMmeParamsEnvPtr);
    }
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
        const auto& desc = activation.getDesc(0);

        if (desc.sbRepeat.repeatSMinus1 != 0)
        {
            repeats += desc.sbRepeat.aguSLoopMask + 1;
        }
        if (desc.sbRepeat.repeatLMinus1 != 0)
        {
            repeats += desc.sbRepeat.repeatLMinus1 + 1;
        }
    }
    if (repeats > 1)
    {
        const std::string repeatsStr = ", SBRepeats=" + std::to_string(repeats);
        vrbSummaryStr += repeatsStr;
    }

    // Activations
    const unsigned activations = m_activations.size();
    if (activations > 1)
    {
        const std::string activationsStr = ", activations=" + std::to_string(activations);
        vrbSummaryStr += activationsStr;
    }

    return vrbSummaryStr;
}

std::vector<std::string> MmeDescriptorGenerator::getRecipeDebugInfo(bool verbose) const
{
    auto debugInfo = getRecipe().getRecipeDebugInfo(verbose);
    if (!debugInfo.empty())
    {
        debugInfo[0] = "MME Recipe: " + debugInfo[0];
        if (verbose)
        {
            debugInfo[0] += getVerboseRecipeSummaryStr();
        }
    }
    return debugInfo;
}

const std::shared_ptr<CommonRoiCalculator<Mme::Desc>>& MmeDescriptorGenerator::getRoiCalculator() const
{
    m_roiCalculator->resetRecipe(getRecipe());
    return m_roiCalculator;
}

std::shared_ptr<CommonRoiCalculator<Mme::Desc>> MmeDescriptorGenerator::createRoiCalculator() const
{
    const MmeCommon::MmeLayerParams& origParams = getOriginalParams();
    return std::make_shared<RoiCalculator>(getRecipe(), getOriginalParams());
}

std::vector<std::vector<std::string>> MmeDescriptorGenerator::dumpDescriptors(bool dumpAsBinary) const
{
    GaudiDescriptorDumper descDumper;
    std::vector<std::vector<std::string>> dbgDump = {};
    unsigned actCount = 0;
    for (const auto& act : m_activations)
    {
        std::vector<std::string> actDump;
        descDumper.setFileNamePrefix("act_" + std::to_string(actCount) + "_");
        const auto& descSouth = act.getDesc(0);
        const auto& descNorth = act.getDesc(1);
        descDumper << "\nActivation " + std::to_string(actCount) + " Descriptor " + "South" + ": \n";
        std::string descDumpSouth = descDumper.dump(descSouth, 0, dumpAsBinary, false);
        if (!dumpAsBinary)
        {
            actDump.push_back(descDumpSouth);
        }
        descDumper.clear();
        descDumper << "\nActivation " + std::to_string(actCount) + " Descriptor " + "North" + ": \n";
        std::string descDumpNorth = descDumper.dump(descNorth, 1, dumpAsBinary, false);
        if (!dumpAsBinary)
        {
            actDump.push_back(descDumpNorth);
        }
        actCount++;
        if (!dumpAsBinary)
        {
            dbgDump.push_back(actDump);
        }
    }
    return dbgDump;
}

}  // namespace gaudi
