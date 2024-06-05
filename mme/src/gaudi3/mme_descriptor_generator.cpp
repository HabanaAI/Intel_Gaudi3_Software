#include <cstring>
#include <optional>
#include "general_utils.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/gaudi3/mme_descriptor_generator.h"
#include "mme_assert.h"
#include "gaudi3_mme_hal_reader.h"
#include "gaudi3_agu_config.h"
#include "include/mme_common/mme_descriptor_generator_base.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include "include/mme_common/recipe_generator.h"
#include "gaudi3_sbreuse.h"
#include "src/mme_common/mme_hal_factory.h"
#define FMT_HEADER_ONLY
#include "spdlog/fmt/bundled/format.h"

using namespace MmeCommon;

namespace gaudi3
{
unsigned MmeDescriptorGenerator::countSignals(const Mme::Desc* desc)
{
    Gaudi3SignalingInfo info;
    return info.countSignals(desc);
}

bool MmeDescriptorGenerator::setParams(const MmeCommon::MmeLayerParams& newParams)
{
    std::string errorMsg = "";
    if (!validateParams(newParams, errorMsg))
    {
        MME_ASSERT(0, fmt::format("MME params are not valid - {}", errorMsg).c_str());
        return false;
    }

    if (!MmeCommon::MmeDescriptorGenerator<Mme::Desc>::setParams(newParams))
    {
        return false;
    }
    return true;
}

pMmeDescriptorGenerator MmeDescriptorGenerator::createMmeDescGenerator(bool isDmaOperation,
                                                                       unsigned numOfTotalMmes)
{
    if (isDmaOperation)
    {
        return std::make_unique<MmeDmaDescriptorGenerator>(numOfTotalMmes);
    }
    else
    {
        return std::make_unique<MmeConvDescriptorGenerator>(numOfTotalMmes);
    }
}

void MmeDescriptorGenerator::mmeGenerateActivations()
{
    MME_ASSERT(m_originalParams.has_value(), "LayerParams have not been initialized");
    MME_ASSERT(!m_convSubProblems.empty(), "should have at least one sub-problem");
    unsigned initialNumOfActivations = getMmeActivations().size();
    if (!getActivationsFromCache())
    {
        getMmeActivations().reserve(initialNumOfActivations + calculateActivationsCount());
        for (unsigned subProblemIdx = 0; subProblemIdx < m_convSubProblems.size(); subProblemIdx++)
        {
            m_convSubProblems.current = &m_convSubProblems[subProblemIdx];
            auto& recipeIterator = getRecipe().getIterator();
            for (const auto& iters : recipeIterator)
            {
                recipeIterator.setCurIterVals(iters);
                MmeActivation& currentActivation = getMmeActivations().emplace_back(getGeoAttr().getMmeNr());
                for (int i = 0; i < getGeoAttr().getMmeNr(); i++)
                {
                    buildDesc(i, &currentActivation.getDesc(i));
                }
                currentActivation.numSignals = m_signalingInfo.countSignals(&currentActivation.getDesc(0));
                for (unsigned i = 1; i < getGeoAttr().getMmeNr(); i++)
                {
                    MME_ASSERT(currentActivation.numSignals ==
                                   m_signalingInfo.countSignals(&currentActivation.getDesc(i)),
                               "num of signals should be equal between all descriptors");
                }
                const MmeRecipe& recipe = getRecipe();
                currentActivation.spView = recipe.curSp();
                currentActivation.fcdView = recipe.curFcd();
                currentActivation.nonSpatialView = recipe.curNonSpatial();
            }
            // Verify last activation was reached
            const bool isLastActivation = getRecipe().getIterator().isLast();
            MME_ASSERT(isLastActivation, "should be on the last activation");
        }
        if (initialNumOfActivations == 0)
        {
            addParamsActivationsToCache();
        }
    }
    if (isPerforated())
    {
        m_numOfActivationsPerDcore.push_back(getMmeActivations().size() - initialNumOfActivations);
    }
    else
    {
        configurePerfEvents(getMmeActivations());
    }
    m_convSubProblems.current = nullptr;
}

void MmeDescriptorGenerator::setZeroActivationsForDcore()
{
    m_numOfActivationsPerDcore.push_back(0);
}

void MmeDescriptorGenerator::mmeGenerateNullDescs()
{
    unsigned maxNumSignals = getMaxNumSignals();
    unsigned maxActivations = *std::max_element(m_numOfActivationsPerDcore.begin(), m_numOfActivationsPerDcore.end());
    for (unsigned dcoreIdx = 0; dcoreIdx < getMmeHal(MmeCommon::e_mme_Gaudi3).getDcoreNr(); dcoreIdx++)
    {
        unsigned numOfActsAdded = 0;
        // in case of locality of one dcore, we would like to pad other dcores with nullActivations
        bool emptyDcore = (dcoreIdx >= m_numOfActivationsPerDcore.size());
        unsigned partialActIdx = (emptyDcore) ? 0 : m_numOfActivationsPerDcore[dcoreIdx];
        unsigned currDcoreSignalsToAdd = emptyDcore ? maxNumSignals : maxNumSignals - calcDcoreNumSignals(dcoreIdx);
        // advance to the first activation of the current dcore
        unsigned numOfActToAdvance =
            std::accumulate(m_numOfActivationsPerDcore.begin(), m_numOfActivationsPerDcore.begin() + dcoreIdx, 0);

        for (unsigned actIdx = partialActIdx; actIdx < maxActivations; actIdx++)
        {
            getMmeActivations().insert(getMmeActivations().begin() + numOfActToAdvance, getGeoAttr().getMmeNr());
            MmeActivation& currentActivation = getMmeActivations().at(numOfActToAdvance);
            for (int i = 0; i < getGeoAttr().getMmeNr(); i++)
            {
                buildEmptyJobDesc(&currentActivation.getDesc(i), actIdx == (maxActivations - 1), currDcoreSignalsToAdd);
            }
            currentActivation.isGemm = true;
            currentActivation.isMask = false;
            currentActivation.numTetrises = 0;
            currentActivation.numRollups = 0;
            currentActivation.numSignals = m_signalingInfo.countSignals(&currentActivation.getDesc(0));
            for (unsigned i = 1; i < getGeoAttr().getMmeNr(); i++)
            {
                MME_ASSERT(currentActivation.numSignals == m_signalingInfo.countSignals(&currentActivation.getDesc(i)),
                           "num of signals should be equal between all descriptors");
            }
            numOfActsAdded++;
            numOfActToAdvance++;
        }
        if (emptyDcore)
        {
            m_numOfActivationsPerDcore.push_back((numOfActsAdded));
        }
        else
        {
            m_numOfActivationsPerDcore[dcoreIdx] += numOfActsAdded;
        }
    }
    for (unsigned i = 1; i < getMmeHal(MmeCommon::e_mme_Gaudi3).getDcoreNr(); i++)
    {
        MME_ASSERT(m_numOfActivationsPerDcore[i] == m_numOfActivationsPerDcore[0],
                   "num of activations should be equal for all dcores");
    }
}

Mme::EMmeDataType ConvertDataTypeToMME(EMmeDataType dt, bool isInput)
{
    switch (dt)
    {
        default:
            MME_ASSERT(0, "invalid data type");
        case e_type_fp16:
        case e_type_ufp16:
            return Mme::EMmeDataType::e_mme_dt_fp16;
        case e_type_bf16:
            return Mme::EMmeDataType::e_mme_dt_bf16;
        case e_type_fp8_143:
        case e_type_fp8_152:
            return Mme::EMmeDataType::e_mme_dt_fp8;
        case e_type_fp32:
            return Mme::EMmeDataType::e_mme_dt_fp32;
        case e_type_tf32:
            return isInput ? Mme::EMmeDataType::e_mme_dt_tf32 : Mme::EMmeDataType::e_mme_dt_fp32;
    }
}

//  a quick fix to set all the fields that dont have any logic in them
//  this will be span out to a class later
//  also need to filter out fields not use by dma desc
void MmeDmaDescriptorGenerator::setSimpleFields(Mme::Desc* desc)
{
    desc->header.dmaMode = 1;
    desc->header.dualGemm = 0;
    desc->header.dataTypeIn = ConvertDataTypeToMME(getParams().x.elementType, true);
    desc->header.dataTypeOut = ConvertDataTypeToMME(getParams().y.elementType, false);
    desc->header.storeEn0 = 1;
    desc->header.doubleAccums = 1;
    desc->header.lowerA = 0;
    desc->header.lowerB = 0;
    desc->header.noRollup = 1;
    desc->header.teBypassA = 0;
    desc->header.teBypassB = 0;
    desc->header.sbACacheEn = getOriginalParams().controls.sbCacheEn ? 1 : 0;
    desc->header.wbCacheEn = 1;

    // Brains
    // Brain A
    desc->brains.aguA.masterEn = 1;
    desc->brains.aguA.slaveEn = 1;
    desc->brains.aguA.loopMask = 0;
    // Brain B
    desc->brains.aguB.masterEn = 0;
    desc->brains.aguB.slaveEn = 0;
    desc->brains.aguB.loopMask = 0;
    // Brain EU
    desc->brains.eu.masterEn = 1;
    desc->brains.eu.slaveEn = 1;
    desc->brains.eu.loopMask = 0;
    // Brain ap
    desc->brains.ap.masterEn = 0;
    desc->brains.ap.slaveEn = 0;
    desc->brains.ap.loopMask = 0;
    // Brain aguOut
    desc->brains.aguOut.masterEn = 0;
    desc->brains.aguOut.slaveEn = 0;
    desc->brains.aguOut.loopMask = 0;
    // Brain aguOutDma
    desc->brains.aguOutDma.masterEn = 1;
    desc->brains.aguOutDma.slaveEn = 1;
    desc->brains.aguOutDma.loopMask = 0;

    // Rate Limiter - better to set all field to avoid desc diffs
    desc->rateLimiter.aguA = 4;
    desc->rateLimiter.aguB = 4;
    desc->rateLimiter.aguOut = 4;
    desc->rateLimiter.eu = 4;

    //  set workload ID
    desc->wkldID = getParams().tracing.ctxId;

    // transpose TE acceleration
    if (getRecipe().teAcceleration)
    {
        EMmeInternalOperand acceleratedOperand = getRecipe().acceleratedOperand;
        unsigned direction = acceleratedOperand == MmeCommon::e_mme_op_a ? 0 : 1;
        desc->header.teAccelA = direction << 2 | getRecipe().teAcceleration;
    }
}
void MmeConvDescriptorGenerator::setSimpleFields(Mme::Desc* desc)
{
    desc->header.dmaMode = 0;
    desc->header.dualGemm = getOriginalParams().strategy.dualGemm;
    desc->header.dataTypeIn = ConvertDataTypeToMME(getParams().getOperand(MmeCommon::e_mme_op_b).elementType, true);
    desc->header.dataTypeOut = ConvertDataTypeToMME(getParams().getOperand(MmeCommon::e_mme_op_c).elementType, false);
    desc->header.storeEn0 = 1;
    desc->header.accumEn = 0;

    desc->header.roundingMode = getOriginalParams().controls.conversionRoundingMode;
    desc->numerics.accRoundingMode = getOriginalParams().controls.accRoundingMode;

    desc->header.bgemm = getGeoAttr().getBgemmBit();
    desc->header.doubleAccums = !getGeoAttr().getDoubleAccumsBit();  // polarity is inverted
    desc->header.opANonShared = getGeoAttr().getNonShareABit();
    desc->header.sbACacheEn = getOriginalParams().controls.sbCacheEn ? 1 : 0;
    desc->header.sbBCacheEn = getOriginalParams().controls.sbCacheEn ? 1 : 0;
    desc->header.lowerA = getRecipe().lowering;
    desc->header.lowerB = getOriginalParams().isGemmDmaOperation() ? 1 : 0; // gemm transpose uses the lowering mechanism to create the unit matrix
    desc->header.reluEn = getOriginalParams().controls.reluEn;
    desc->header.noRollup = 0;
    desc->header.teBypassA = 0;
    desc->header.teBypassB = 0;
    desc->header.wbCacheEn = 1;

    // Brains
    // Brain A
    desc->brains.aguA.masterEn = 1;
    desc->brains.aguA.slaveEn = 1;
    // Brain B
    desc->brains.aguB.masterEn = 1;
    desc->brains.aguB.slaveEn = 1;
    // Brain EU
    desc->brains.eu.masterEn = 1;
    desc->brains.eu.slaveEn = 1;
    desc->brains.eu.loopMask = getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0 : e_mme_gemm_loop;
    // Brain ap
    desc->brains.ap.masterEn = 1;
    desc->brains.ap.slaveEn = 1;
    desc->brains.ap.loopMask = getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0
                               : getOriginalParams().isFwdOrDedx()      ? e_mme_conv_loop_2
                                                                        : e_mme_gemm_loop;
    // Brain aguOut
    desc->brains.aguOut.masterEn = 1;
    desc->brains.aguOut.slaveEn = 1;
    desc->brains.aguOut.loopMask = getOriginalParams().isGemmDmaOperation() ? e_mme_conv_loop_0
                                   : getOriginalParams().isFwdOrDedx()      ? e_mme_conv_loop_2
                                                                            : e_mme_gemm_loop;
    // Brain aguOutDma
    desc->brains.aguOutDma.masterEn = 0;
    desc->brains.aguOutDma.slaveEn = 0;
    desc->brains.aguOutDma.loopMask = 0;

    // Rate Limiter
    desc->rateLimiter.aguA = 4;
    desc->rateLimiter.aguB = 4;
    desc->rateLimiter.aguOut = 4;
    desc->rateLimiter.eu = 4;

    //  bias
    desc->numerics.biasA = getParams().controls.fp8BiasIn;
    desc->numerics.biasB = getParams().controls.fp8BiasIn2;
    desc->numerics.biasOut = getParams().controls.fp8BiasOut;

    //  data type flavors
    desc->numerics.fp8FlavorA = getParams().getOperand(e_mme_op_a).elementType == e_type_fp8_152 ? 1 : 0;
    desc->numerics.fp8FlavorB = getParams().getOperand(e_mme_op_b).elementType == e_type_fp8_152 ? 1 : 0;
    desc->numerics.fp8FlavorOut = getParams().getOperand(e_mme_op_c).elementType == e_type_fp8_152 ? 1 : 0;
    desc->numerics.fp16FlavorA = getParams().getOperand(e_mme_op_a).elementType == e_type_ufp16 ? 1 : 0;
    desc->numerics.fp16FlavorB = getParams().getOperand(e_mme_op_b).elementType == e_type_ufp16 ? 1 : 0;
    desc->numerics.fp16FlavorOut = getParams().getOperand(e_mme_op_c).elementType == e_type_ufp16 ? 1 : 0;

    //  numeric flavors
    desc->header.ftz = getParams().controls.flushDenormals;
    desc->header.sftzFp32ToFp8 = getParams().controls.stochasticFlush;
    desc->header.clipFpEu = getOriginalParams().controls.clippingEn;
    desc->header.clipFpAp = getOriginalParams().controls.clippingEn;
    desc->numerics.infNanModeA = getOriginalParams().controls.infNanModeA;
    desc->numerics.infNanModeB = getOriginalParams().controls.infNanModeB;
    desc->numerics.infNanModeOut = getOriginalParams().controls.infNanModeOut;
    // waiting for register for clipInf
}

uint64_t getBaseOffsetInBytes(SizeArray offsets, SizeArray strides, EMmeDataType dtype)
{
    uint64_t totalOffset = 0;
    for (int dim = 0; dim < MAX_DIMENSION; dim++)
    {
        totalOffset += offsets[dim] * strides[dim];
    }
    return totalOffset * getElementSize(dtype);
}

void setDcoreBaseOffset(const MmeLayerParams& params, Mme::Desc* desc)
{
    const MmeTensorView& opA = params.getOperand(e_mme_op_a);
    const MmeTensorView& opB = params.getOperand(e_mme_op_b);
    const MmeTensorView& opC = params.getOperand(e_mme_op_c);
    desc->baseAddrA.addr += getBaseOffsetInBytes(opA.dcoreBases, opA.strides, opA.elementType);
    desc->baseAddrB.addr += getBaseOffsetInBytes(opB.dcoreBases, opB.strides, opB.elementType);
    desc->baseAddrCOut0.addr += getBaseOffsetInBytes(opC.dcoreBases, opC.strides, opC.elementType);
    desc->baseAddrCOut1.addr = desc->baseAddrCOut0.addr;
}

//  TODO fill this function with the rest of the flow - perhaps move it to common.
void MmeDescriptorGenerator::buildDesc(unsigned mmeIDx, Mme::Desc* desc)
{
    setSimpleFields(desc);
    //  TODO consider holding aguConfig as a member variable
    Gaudi3AguConfig aguConfig(getParams(), getGeoAttr(), gaudi3::MmeHalReader::getInstance(), mmeIDx, getRecipe());
    if (!ConvSubProblemContainer::isOutOfBounds(getParams()))
    {
        MME_ASSERT(getCurrentSubProblem() != nullptr,
                   "should have a defined sub-problem for dedx descriptor generation");
        const OffsetArray& descAddrOffset = getCurrentSubProblem()->addressOffset;
        aguConfig.setDescOffset(descAddrOffset);
    }
    aguConfig.config(desc);

    Gaudi3SBReuse sbReuse(getParams(), getGeoAttr(), getRecipe());
    sbReuse.configDescSBReuse(desc);
    m_signalingInfo.addSignalInfo(getOriginalParams().controls.signalingMode,
                                  getOriginalParams().controls.slaveSignaling,
                                  getRecipe().getIterator().isLast() && isLastSubProblem(),
                                  getOriginalParams().controls.squashIORois,
                                  getRecipe().signalAmount,
                                  desc);
    configureMemoryDirectives(*desc, getRecipe());
    setDcoreBaseOffset(getOriginalParams(), desc);
}

// Build a descriptor with minimal ROI and zero valid elements.
void MmeDescriptorGenerator::buildEmptyJobDesc(Mme::Desc* desc, bool isLast, unsigned signalsToAdd)
{
    memset(desc, 0, sizeof(Mme::Desc));

    // Header
    desc->header.transA = 1;
    desc->header.sbTransA = 1;

    desc->header.advanceA = 1;
    desc->header.advanceC = 1;
    desc->header.shuffleA = Mme::e_mme_shuffle_2ports;
    desc->header.dataTypeIn = Mme::EMmeDataType::e_mme_dt_bf16;
    desc->header.dataTypeOut = Mme::EMmeDataType::e_mme_dt_bf16;

    desc->header.storeEn0 = 1;
    desc->header.rollAccums = 1;

    desc->header.partialHeightLoopA = getLoopFromLoopMask(e_mme_conv_loop_0);  // walk pattern - fkc
    desc->header.partialHeightLoopB = getLoopFromLoopMask(e_mme_conv_loop_1);  // walk pattern - fkc

    // Brain A
    desc->brains.aguA.masterEn = 1;
    desc->brains.aguA.slaveEn = 1;
    // Brain B
    desc->brains.aguB.masterEn = 1;
    desc->brains.aguB.slaveEn = 1;
    // Brain EU
    desc->brains.eu.masterEn = 1;
    desc->brains.eu.slaveEn = 1;
    // Brain ap
    desc->brains.ap.masterEn = 1;
    desc->brains.ap.slaveEn = 1;
    // Brain aguOut
    desc->brains.aguOut.masterEn = 1;
    desc->brains.aguOut.slaveEn = 1;

    // Tensor Desc A
    desc->tensorA.loopStride[0] = 0;
    desc->tensorA.loopStride[1] = 512;
    desc->tensorA.loopStride[2] = 1;
    desc->tensorA.loopStride[3] = 1;
    desc->tensorA.loopStride[4] = 1;

    desc->tensorA.spatialStrides[0] = 16;
    desc->tensorA.spatialStrides[1] = 1;
    desc->tensorA.spatialStrides[2] = 1;
    desc->tensorA.spatialStrides[3] = 1;

    desc->tensorA.roiSize[0] = 8;
    desc->tensorA.roiSize[1] = 16;
    desc->tensorA.roiSize[2] = 1;
    desc->tensorA.roiSize[3] = 1;

    // Tensor Desc B
    desc->tensorB.loopStride[0] = 512;
    desc->tensorB.loopStride[1] = 0;
    desc->tensorB.loopStride[2] = 1;
    desc->tensorB.loopStride[3] = 1;
    desc->tensorB.loopStride[4] = 1;

    desc->tensorB.spatialStrides[0] = 4;
    desc->tensorB.spatialStrides[1] = 1;
    desc->tensorB.spatialStrides[2] = 1;
    desc->tensorB.spatialStrides[3] = 1;

    desc->tensorB.roiSize[0] = 256;
    desc->tensorB.roiSize[1] = 8;
    desc->tensorB.roiSize[2] = 1;
    desc->tensorB.roiSize[3] = 1;

    // Tensor Desc Cout
    desc->tensorCOut.loopStride[0] = 512;
    desc->tensorCOut.loopStride[1] = 512;
    desc->tensorCOut.loopStride[2] = 1;
    desc->tensorCOut.loopStride[3] = 1;
    desc->tensorCOut.loopStride[4] = 1;

    desc->tensorCOut.spatialStrides[0] = 8;
    desc->tensorCOut.spatialStrides[1] = 1;
    desc->tensorCOut.spatialStrides[2] = 1;
    desc->tensorCOut.spatialStrides[3] = 1;

    desc->tensorCOut.roiSize[0] = 1;
    desc->tensorCOut.roiSize[1] = 16;
    desc->tensorCOut.roiSize[2] = 1;
    desc->tensorCOut.roiSize[3] = 1;

    // aguA
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[1] = 0;

    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[1] = 8;

    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[1] = 4;

    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[1] = 12;

    desc->spatialSizeMinus1A = 0;

    // aguB
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[1] = 0;

    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[1] = 1;

    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[1] = 2;

    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[1] = 3;

    desc->spatialSizeMinus1B = 1;

    // aguOut
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[0] = 0;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[1] = 0;

    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[0] = 0;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[1] = 4;

    desc->spatialSizeMinus1Cout = 1;

    desc->conv.associatedDims[0].dimA = 0;  // DIM_C
    desc->conv.associatedDims[0].dimB = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[0].dimOut = 1;  // DIM_C
    desc->conv.associatedDims[1].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[1].dimB = 0;  // DIM_K
    desc->conv.associatedDims[1].dimOut = 0;  // DIM_K
    desc->conv.associatedDims[2].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimB = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimOut = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimB = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimOut = Mme::c_mme_max_tensor_dims;  // dont care

    // outer loop
    desc->outerLoop.associatedDims.dimA = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimB = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimOut = Mme::c_mme_max_tensor_dims;  // skf pattern

    // Rate Limiter
    desc->rateLimiter.aguA = 4;
    desc->rateLimiter.aguB = 4;
    desc->rateLimiter.aguOut = 4;

    m_signalingInfo.addSignalInfo(signalsToAdd ? getOriginalParams().controls.signalingMode : e_mme_signaling_none,
                                  false,
                                  isLast,
                                  true,  // not relevant
                                  isLast ? signalsToAdd : 0,
                                  desc);
}

void MmeDescriptorGenerator::patchDualGemm(MmeCommon::MmeTensorView& x,
                                           MmeCommon::MmeTensorView& w,
                                           MmeCommon::MmeTensorView& y,
                                           const uint64_t addrX,
                                           const uint64_t addrW,
                                           const uint64_t addrY,
                                           const uint64_t addrO,
                                           const bool YisSram,
                                           const bool OisSram,
                                           unsigned gemmIdx)
{
    for (auto& activation : getMmeActivations())
    {
        patchSignalColoring(activation, YisSram, OisSram);
    }

    bool transA = m_commonGeoAttr->isTransposed(MmeCommon::e_mme_op_a);
    bool transB = m_commonGeoAttr->isTransposed(MmeCommon::e_mme_op_b);
    int cdA = transA ? GEMM_DIM_W : GEMM_DIM_H;
    int cdB = transB ? GEMM_DIM_W : GEMM_DIM_H;
    Mme::EMmeCore core = gemmIdx == 0 ? Mme::MME_MASTER : Mme::MME_SLAVE;
    for (auto& act : getMmeActivations())
    {
        for (int mmeIdx = 0; mmeIdx < getGeoAttr().getMmeNr(); mmeIdx++)
        {
            auto& desc = act.getDesc(mmeIdx);
            // reset offsets
            memset(desc.tensorA.startOffset, 0, sizeof(desc.tensorA.startOffset));
            memset(desc.tensorB.startOffset, 0, sizeof(desc.tensorB.startOffset));
            memset(desc.tensorCOut.startOffset, 0, sizeof(desc.tensorCOut.startOffset));
            memset(&desc.aguIn[core], 0, sizeof(desc.aguIn[core]));
            memset(&desc.aguOut[core], 0, sizeof(desc.aguOut[core]));

            unsigned alignedSize = 1;
            if (!transA)
            {
                alignedSize = std::max(alignedSize, getGeoAttr().getInterleavedSpatialPortsNr(e_mme_op_a));
            }
            if (!transB)
            {
                alignedSize = std::max(alignedSize, getGeoAttr().getInterleavedSpatialPortsNr(e_mme_op_b));
            }
            alignedSize = std::max(alignedSize,
                                   getMmeHal(MmeCommon::e_mme_Gaudi3)
                                       .getNumElementsForCommonDimAlignment(w.elementType, getParams().opType));
            unsigned paddedCommonDim = round_to_multiple(x.sizes[cdA], alignedSize);

            unsigned aFCD = x.sizes[0];
            unsigned aSP = x.sizes[1];
            unsigned paddedSpA = round_to_multiple(aSP, getGeoAttr().getInterleavedSpatialPortsNr(e_mme_op_a));
            unsigned bFCD = w.sizes[0];
            unsigned bSP = w.sizes[1];
            unsigned paddedSpB = round_to_multiple(bSP, getGeoAttr().getInterleavedSpatialPortsNr(e_mme_op_b));
            unsigned cFCD = y.sizes[0];
            unsigned cSP = y.sizes[1];
            //  tensor sizes and strides
            desc.tensorA.dualGemm.roiSize[core][0] = transA ? paddedCommonDim : 128;
            desc.tensorA.dualGemm.roiSize[core][1] = transA ? paddedSpA * aFCD : paddedCommonDim * aFCD;
            desc.tensorB.dualGemm.roiSize[core][0] = transB ? paddedCommonDim : 128;
            desc.tensorB.dualGemm.roiSize[core][1] = transB ? paddedSpB * bFCD : paddedCommonDim * bFCD;
            desc.tensorCOut.dualGemm.roiSize[core][0] = cFCD;
            desc.tensorCOut.dualGemm.roiSize[core][1] = cSP * cFCD;
            desc.tensorA.dualGemm.validElements[core][0] = aFCD;
            desc.tensorA.dualGemm.validElements[core][1] = aSP * aFCD;
            desc.tensorB.dualGemm.validElements[core][0] = bFCD;
            desc.tensorB.dualGemm.validElements[core][1] = bSP * bFCD;
            desc.tensorCOut.dualGemm.validElements[core][0] = cFCD;
            desc.tensorCOut.dualGemm.validElements[core][1] = cSP * cFCD;
            desc.tensorA.dualGemm.spatialStrides[core] = aFCD;
            desc.tensorB.dualGemm.spatialStrides[core] = bFCD;
            desc.tensorCOut.dualGemm.spatialStrides[core] = cFCD;
            //  spatialSize
            aSP = transA ? paddedSpA : paddedCommonDim;
            unsigned aSpSize = aSP / 2;  // 2 interleaving ports
            bSP = transB ? paddedSpB : paddedCommonDim;
            unsigned bSpSize = transB ? std::min(bSP, (unsigned) 64) : (bSP / 2);
            if (core == Mme::MME_MASTER)
            {
                //  master uses original spatial size and base address fields
                desc.spatialSizeMinus1A = aSpSize - 1;
                desc.spatialSizeMinus1B = bSpSize - 1;
                desc.spatialSizeMinus1Cout = cSP - 1;
                // patch base address
                desc.baseAddrA.addr = addrX;
                desc.baseAddrB.addr = addrW;
                desc.baseAddrCOut0.addr = addrY;
                desc.baseAddrCOut1.addr = addrO;
                if (addrO) desc.header.storeEn1 = desc.header.storeEn0;
            }
            else
            {
                // slave uses the new spatial size fields
                desc.tensorA.dualGemm.spatialSizeMinus1Gemm1 = aSpSize - 1;
                desc.tensorB.dualGemm.spatialSizeMinus1Gemm1 = bSpSize - 1;
                desc.tensorCOut.dualGemm.spatialSizeMinus1Gemm1 = cSP - 1;
                // slave also needs the base address patched
                desc.tensorA.dualGemm.baseAddrGemm1.addr = addrX;
                desc.tensorB.dualGemm.baseAddrGemm1.addr = addrW;
                desc.tensorCOut.dualGemm.baseAddrGemm1.addr = addrY;
                desc.tensorCOut.dualGemm.baseAddrGemm1Dup.addr = addrO;
            }

            //  advance second port offset
            desc.aguIn[core][1].roiBaseOffset[DIM_W] = aFCD;  // seocnd A port
            desc.tensorA.dualGemm.spatialStrides[core] *= 2;
            if (!transB)
            {  // B interleaved only if it is non transposed
                desc.aguIn[core][3].roiBaseOffset[DIM_W] = bFCD;  // seocnd B port
                desc.tensorB.dualGemm.spatialStrides[core] *= 2;
            }
            else
            {
                desc.aguIn[core][3].roiBaseOffset[DIM_W] = bFCD * 64;  // advance 64 columns ahead
            }
        }
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
    Gaudi3SignalingInfo signalingInfo;
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
    if ((!desc.brains.aguOut.masterEn && !desc.brains.aguOutDma.masterEn) || (!storeEn && !signalEn))
    {
        UNSET_DESC_FIELD(mask, aguOut[0]);
    }
    if ((!desc.brains.aguOut.slaveEn && !desc.brains.aguOutDma.slaveEn) || (!storeEn && !signalEn))
    {
        UNSET_DESC_FIELD(mask, aguOut[1]);
    }
    if (!desc.header.storeEn0)
    {
        UNSET_DESC_FIELD(mask, baseAddrCOut0);
    }
    if (!desc.header.storeEn1)
    {
        UNSET_DESC_FIELD(mask, baseAddrCOut1);
    }
    if ((!desc.brains.aguOut.masterEn && !desc.brains.aguOut.slaveEn && !desc.brains.aguOutDma.masterEn &&
         !desc.brains.aguOutDma.slaveEn) ||
        !storeEn)
    {
        UNSET_DESC_FIELD(mask, tensorCOut);
        // we need C roisize0 to configure the EU
        SET_DESC_FIELD(mask, tensorCOut.roiSize[0]);
    }

    // disabled slave
    if (!desc.brains.eu.slaveEn)
    {
        UNSET_DESC_FIELD(mask, aguOut[1]);
        UNSET_DESC_FIELD(mask, aguIn[0][1]);
        UNSET_DESC_FIELD(mask, aguIn[1][1]);
        UNSET_DESC_FIELD(mask, aguIn[2][1]);
        UNSET_DESC_FIELD(mask, aguIn[3][1]);
        UNSET_DESC_FIELD(mask, aguIn[4][1]);
    }

    // non fp8/fp16
    if (desc.header.dataTypeIn != Mme::EMmeDataType::e_mme_dt_fp8 &&
        desc.header.dataTypeIn != Mme::EMmeDataType::e_mme_dt_fp16 &&
        desc.header.dataTypeOut != Mme::EMmeDataType::e_mme_dt_fp8 &&
        desc.header.dataTypeOut != Mme::EMmeDataType::e_mme_dt_fp16)
    {
        UNSET_DESC_FIELD(mask, numerics);
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
        UNSET_DESC_FIELD(mask, syncObject.slaveSo0Addr);
    }
    if (!desc.syncObject.signalEn1 || !desc.syncObject.slaveSignalEn || !desc.syncObject.slave1UseSlaveSOAddr)
    {
        UNSET_DESC_FIELD(mask, syncObject.slaveSo1Addr);
    }
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
            operand = e_mme_operand_0;  // dont care
            break;
        case e_mme_trace_eu:
            traceEngine = &desc.perfEvtEU;
            operand = e_mme_operand_0;  // dont care
            break;
    }
    traceEngine->value = getOriginalParams().tracing.ctxId;
    traceEngine->incEn = 0;
    traceEngine->rst = 1;
    traceEngine->loopMask = e_mme_outer_loop;
    traceEngine->startEndEn = startEndMask;
    traceEngine->operand = operand;
    traceEngine->slaveSendsPerfEvent = 1;
}

bool MmeDescriptorGenerator::validateParams(const MmeLayerParams& params, std::string& errorMsg)
{
    const MmeTensorView& output = params.getOperand(e_mme_op_c);
    const MmeTensorView& operandA = params.getOperand(e_mme_op_a);
    const MmeTensorView& operandB = params.getOperand(e_mme_op_b);

    if (!isPowerOf2(params.strategy.mmeLimit) || params.strategy.mmeLimit > 8)
    {
        errorMsg = "mmes can be limited to either 1, 2, 4 or 8";
        return false;
    }

    if (params.strategy.maskedBgemm)
    {
        errorMsg = "maskedBgemm is not yet supported";
        return false;
    }

    if (params.isNativeDmaOperation())
    {
        if (operandA.elementType != output.elementType)
        {
            errorMsg = "dma operations dont support type conversion";
            return false;
        }

        bool fieldsEqual = true;
        if (params.opType == MmeCommon::e_mme_memcpy)
        {
            if (params.strategy.geometry != MmeCommon::e_mme_geometry_4xw &&
                params.strategy.geometry != MmeCommon::e_mme_geometry_2xw)
            {
                errorMsg = "4xw/2xw geometries are not yet supported in memcpy operation";
                return false;
            }
            fieldsEqual &= operandA.sizes[0] == output.sizes[0];
            fieldsEqual &= operandA.sizes[1] == output.sizes[1];
            fieldsEqual &= operandA.sizes[2] == output.sizes[2];
            fieldsEqual &= operandA.sizes[3] == output.sizes[3];
            fieldsEqual &= operandA.sizes[4] == output.sizes[4];
        }
        else if (params.opType == MmeCommon::e_mme_trans)
        {
            fieldsEqual &= validateTranspose(operandA.sizes, output.sizes);
        }
        errorMsg = "invalid input/output sizes";
        return fieldsEqual;
    }

    if (operandA.elementType != operandB.elementType)
    {
        if ((!isTypeFp8(operandA.elementType) || !isTypeFp8(operandB.elementType)) &&
            (!isTypeFp16(operandA.elementType) || !isTypeFp16(operandB.elementType)))
        {
            errorMsg = "input element types should match";
            return false;
        }
    }

    bool isOutputDtypeFp8 = (output.elementType == e_type_fp8_143) || (output.elementType == e_type_fp8_152);
    if (isOutputDtypeFp8 && ((params.memoryCfg.reductionOp != e_mme_reduction_none) || params.controls.atomicAdd))
    {
        errorMsg = "reduction is not supported for fp8 output data type";
        return false;
    }

    // rounding
    if (!isTypeFp8(output.elementType) && params.controls.conversionRoundingMode == StochasticRoundingAndNearest)
    {
        errorMsg = "RSN rounding can only be used with fp8";
        return false;
    }
    if (0 && params.controls.roundingMode != RoundToZero)  //  check disabled until we have per chip config parser.
    {
        errorMsg = "gaudi3 only supports round nearest in EU";
        return false;
    }

    // input bias
    if ((operandA.elementType == e_type_ufp16 && params.controls.fp8BiasIn != EXPONENT_BIAS_UFP16_31) ||
        (operandB.elementType == e_type_ufp16 && params.controls.fp8BiasIn2 != EXPONENT_BIAS_UFP16_31))
    {
        errorMsg = "ufp16 supports only a single bias";
        return false;
    }

    // CD Concurrency currently implies writes with Reduction Add
    if (params.controls.atomicAdd || (params.strategy.cdConcurrencyEn == TurnedOn))
    {
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

MmeConvDescriptorGenerator::MmeConvDescriptorGenerator(unsigned totalNumOfMmeUnits)
: MmeDescriptorGenerator(totalNumOfMmeUnits)
{
}

MmeDmaDescriptorGenerator::MmeDmaDescriptorGenerator(unsigned totalNumOfMmeUnits)
: MmeDescriptorGenerator(totalNumOfMmeUnits)
{
}
std::vector<std::string> MmeDescriptorGenerator::getRecipeDebugInfo(bool verbose) const
{
    std::vector<std::string> debugInfo;
    std::vector<unsigned> summaryStrIdx;

    if (m_convSubProblems.empty())
    {
        debugInfo = MmeDescriptorGenerator::getRecipeDebugInfo(verbose);
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

// in dcore split (perforation) - we generate all activations for all dcores in a long list of 2-core activations. <D0 activations> -> <D1 activations> -> ...
// this method we re-arrange that long list to regular 8-core activations .
// on monolithic nodes (non-perforated) this method doesn't do anything.
void MmeDescriptorGenerator::reorderDcoreActivations(const MmeLayerParams& fullNodeParams)
{
    if (!isPerforated()) //not-perforated.
    {
        MME_ASSERT(!m_activations.empty(), "workload is not perforated, but no full activations generated");
    }
    else
    {
        MME_ASSERT(m_activations.empty(), "workload is perforated, but full activations was generated");

        const unsigned numOfDcores = getTotalNumOfMmeUnits() / getGeoAttr().getMmeNr();
        unsigned firstDcoreNumSignals = 0;

        // TODO [SW-136623]: need to allow cases with different number of activations for each dcore
        for (unsigned dcoreIdx = 1; dcoreIdx < numOfDcores; dcoreIdx++)
        {
            MME_ASSERT(m_numOfActivationsPerDcore[dcoreIdx] == m_numOfActivationsPerDcore[0],
                       "All dcores must have the same #activation");
        }
        // initialize the activations array
        m_activations.insert(m_activations.end(), m_numOfActivationsPerDcore[0], {getTotalNumOfMmeUnits()});
        ActivationVec::iterator monolithicActivationIt = m_activations.begin();

        // re-organize the descriptors in activations that contain all the dcores together.
        // for dcore 0 - also copy all activation meta-data which is const across all dcores
        for (unsigned dcoreIdx = 0; dcoreIdx < numOfDcores; dcoreIdx++)
        {
            ActivationVec::iterator dcoreActivationIt = m_dcoreActivations.begin();
            monolithicActivationIt = m_activations.begin();
            unsigned dcoreNumSignals = 0;
            // advance to the first activation of the current dcore
            unsigned numOfActToAdvance =
                std::accumulate(m_numOfActivationsPerDcore.begin(), m_numOfActivationsPerDcore.begin() + dcoreIdx, 0);
            std::advance(dcoreActivationIt, numOfActToAdvance);
            // add the current dcore activations to the full activations vector
            const unsigned numOfActivationForCurrentDcore = m_numOfActivationsPerDcore[dcoreIdx];
            for (unsigned actIdx = 0; actIdx < numOfActivationForCurrentDcore; actIdx++)
            {
                for (unsigned mmeIdx = 0; mmeIdx < getGeoAttr().getMmeNr(); mmeIdx++)
                {
                    const unsigned descOffset = dcoreIdx * getGeoAttr().getMmeNr() + mmeIdx;
                    auto& desc = dcoreActivationIt->descriptors[mmeIdx];
                    monolithicActivationIt->descriptors[descOffset] = desc;
                }
                if (dcoreIdx == 0)
                {
                    monolithicActivationIt->numSignals = dcoreActivationIt->numSignals;
                    monolithicActivationIt->spView = {};
                    monolithicActivationIt->fcdView = {};
                    monolithicActivationIt->nonSpatialView = {};
                    monolithicActivationIt->isGemm = dcoreActivationIt->isGemm;
                    monolithicActivationIt->isMask = dcoreActivationIt->isMask;
                    monolithicActivationIt->numTetrises = dcoreActivationIt->numTetrises;
                    monolithicActivationIt->numRollups = dcoreActivationIt->numRollups;
                    firstDcoreNumSignals += dcoreActivationIt->numSignals;
                }
                else
                {
                    dcoreNumSignals += dcoreActivationIt->numSignals;
                }

                dcoreActivationIt++;
                monolithicActivationIt++;
            }
            MME_ASSERT(dcoreIdx ? dcoreNumSignals == firstDcoreNumSignals : true,
                       "num of signals should be the same between dcores");
        }
        // basically after reordering the activations - we are the same as non-perforated node.
        setPerforated(false);
        // reset original node params
        setParams(fullNodeParams);
    }
    // configuring the perf events here to allow correct first\last detection.
    configurePerfEvents(getMmeActivations());
}

unsigned MmeDescriptorGenerator::calcDcoreNumSignals(const unsigned targetDcoreIdx)
{
    MME_ASSERT(targetDcoreIdx <= getTotalNumOfMmeUnits() / getGeoAttr().getMmeNr(),
               "can't request signals of non existent dcore");
    unsigned totalNumSignals = 0;

    ActivationVec::iterator dcoreActivationIt = m_dcoreActivations.begin();
    // advance to the first activation of the current dcore
    unsigned numOfActToAdvance =
        std::accumulate(m_numOfActivationsPerDcore.begin(), m_numOfActivationsPerDcore.begin() + targetDcoreIdx, 0);
    std::advance(dcoreActivationIt, numOfActToAdvance);

    for (unsigned actIdx = 0; actIdx < m_numOfActivationsPerDcore[targetDcoreIdx]; actIdx++)
    {
        totalNumSignals += dcoreActivationIt->numSignals;
        dcoreActivationIt++;
    }
    return totalNumSignals;
}

unsigned MmeDescriptorGenerator::getMaxNumSignals()
{
    unsigned maxNumSignals = 0;
    for (unsigned dcoreIdx = 0; dcoreIdx < m_numOfActivationsPerDcore.size(); dcoreIdx++)
    {
        unsigned dcoreSignals = calcDcoreNumSignals(dcoreIdx);
        maxNumSignals = (dcoreSignals > maxNumSignals) ? dcoreSignals : maxNumSignals;
    }
    return maxNumSignals;
}

std::string MmeDescriptorGenerator::getRecurringMisalignmentDebugInfo() const
{
    std::string recurringMisalignmentDebugStr = RecurringMisalignmentOptimization::getDebugInfo(m_convSubProblems,
                                                                                                getGeoAttr(),
                                                                                                getMmeHal(e_mme_Gaudi3),
                                                                                                getOriginalParams());
    return recurringMisalignmentDebugStr;
}

void MmeDescriptorGenerator::patchMcids(uint16_t mcidA, uint16_t mcidB, uint16_t mcidC)
{
    patchMcids(mcidA, mcidB, mcidC, std::nullopt, std::nullopt);
}

void MmeDescriptorGenerator::patchMcids(uint16_t mcidA,
                                        uint16_t mcidB,
                                        uint16_t mcidC,
                                        std::optional<uint16_t> mcidAuxScratchpad,
                                        std::optional<uint16_t> mcidAuxReductionAdd)
{
    uint16_t mcidInA;
    uint16_t mcidInB;
    uint16_t mcidOut;

    for (auto& activation : getMmeActivations())
    {
        if (activation.operandRoles[MmeCommon::OUTPUT_TENSOR_C] == MmeCommon::AUX_TENSOR_SCRATCHPAD)
        {
            // CD parallel - compute activation (writing to scaratchpad)
            MME_ASSERT(mcidAuxScratchpad.has_value(), "no mcid for scratchpad aux tensor");
            mcidInA = mcidA;
            mcidInB = mcidB;
            mcidOut = mcidAuxScratchpad.value();
        }
        else if (activation.operandRoles[MmeCommon::INPUT_TENSOR_A] == MmeCommon::AUX_TENSOR_REDUCTION &&
                 activation.operandRoles[MmeCommon::INPUT_TENSOR_B] == MmeCommon::AUX_TENSOR_SCRATCHPAD)
        {
            // CD parallel - reductionAdd activation
            MME_ASSERT(mcidAuxScratchpad.has_value() && mcidAuxReductionAdd.has_value(),
                       "no mcid for cd parallel aux tensors");
            mcidInA = mcidAuxReductionAdd.value();
            mcidInB = mcidAuxScratchpad.value();
            mcidOut = mcidC;
        }
        else
        {
            // general case - no CD parallel
            mcidInA = mcidA;
            mcidInB = mcidB;
            mcidOut = mcidC;
        }

        for (auto& desc : activation.descriptors)
        {
            desc.axiUserDataA.mcid = mcidInA;
            desc.axiUserDataB.mcid = mcidInB;
            desc.axiUserDataCout.mcid = mcidOut;
        }
    }
}

}  // namespace gaudi3
