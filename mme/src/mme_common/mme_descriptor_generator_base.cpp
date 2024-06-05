#include "include/mme_common/mme_descriptor_generator_base.h"
#include "mme_common/mme_descriptor_cache_utils.h"
#include "mme_hal_factory.h"
#include "memory_config.h"
#include "gaudi2/mme_descriptor_generator.h"
#include "gaudi3/mme_descriptor_generator.h"
#include "common_linear_ranges.h"
#include "mme_params_dumper.h"
#include "mme_geo_factory.h"
#include <memory>
#include <numeric>
#include <optional>
#include <string>

namespace MmeCommon
{
//  MmeDescriptorGeneratorBase
pMmeDescriptorGeneratorBase MmeDescriptorGeneratorBase::createMmeDescGenerator(ChipType chipType,
                                                                               bool isDmaOperation,
                                                                               const unsigned numOfTotalMmes)
{
    switch (chipType)
    {
        case e_mme_Gaudi2:
            return Gaudi2::MmeDescriptorGenerator::createMmeDescGenerator();
        case e_mme_Gaudi3:
            return gaudi3::MmeDescriptorGenerator::createMmeDescGenerator(isDmaOperation, numOfTotalMmes);
        default:
            MME_ASSERT(0, "chip not supported yet");
    }

    return nullptr;
}
std::vector<std::string> MmeDescriptorGeneratorBase::getBrainDebugInfo(const MmeLayerParams& curParams) const
{
    std::vector<std::string> debugInfo;
    MmeBrain brain(getChipType(), getMMEBrain().getOperationModes());

    std::string optimizationInfo = brain.getGeometryDebugInfo(getGeoAttr());
    debugInfo.push_back(optimizationInfo);

    PerfAttr perfAttr;
    brain.getPerfAttr(curParams, perfAttr);
    std::string perfInfo = brain.getPerfDebugInfo(perfAttr);
    debugInfo.push_back(perfInfo);

    return debugInfo;
}

unsigned MmeDescriptorGeneratorBase::getMmeNr() const
{
    return getGeoAttr().getMmeNr();
}

MmeCommon::EMmeGeometry MmeDescriptorGeneratorBase::getGeometry() const
{
    return getGeoAttr().getGeometry();
}

unsigned MmeDescriptorGeneratorBase::getGeometryWidth() const
{
    return getGeoAttr().getGeometryWidth();
}

unsigned MmeDescriptorGeneratorBase::getGeometryHeight() const
{
    return getGeoAttr().getGeometryHeight();
}

unsigned MmeDescriptorGeneratorBase::getEffectiveBatchConcurrency() const
{
    return getGeoAttr().getEffectiveBatchConcurrency();
}

unsigned MmeDescriptorGeneratorBase::getGeometryCdConcurrency() const
{
    return getGeoAttr().getGeometryCdConcurrency();
}

MmeDimsIndex MmeDescriptorGeneratorBase::getSpInterleavingDim(EMmeInternalOperand operand) const
{
    return getGeoAttr().getSpInterleavingDim(operand);
}

bool MmeDescriptorGeneratorBase::isAsymPortConfigMode() const
{
    return getGeoAttr().isAsymPortConfigMode();
}

template<typename Desc>
inline constexpr ChipType descToChipType()
{
    static_assert(std::is_same_v<gaudi3::Mme::Desc, Desc> ||
                  std::is_same_v<Gaudi2::Mme::Desc, Desc>,
                  "chip not supported yet");
    return std::is_same_v<gaudi3::Mme::Desc, Desc> ?  e_mme_Gaudi3 : e_mme_Gaudi2;
}

template<typename Desc>
ChipType MmeDescriptorGenerator<Desc>::getChipType() const
{
    return descToChipType<Desc>();
}

// MmeDescriptorGenerator<Desc>
template<typename Desc>
MmeDescriptorGenerator<Desc>::MmeDescriptorGenerator()
: m_mmeBrain(MmeDescriptorGenerator<Desc>::getChipType()),
  m_convSubProblems(MmeDescriptorGenerator<Desc>::getChipType()),
  m_paramsVec(std::make_shared<std::vector<MmeLayerParams>>())
{
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::createRecipes(const MmeLayerParams& params)
{
    for (unsigned subProblemIdx = 0; subProblemIdx < m_convSubProblems.size(); subProblemIdx++)
    {
        RecipeGenerator recipeGen(params.isGemmOperation() ? e_mme_bgemm_recipe : e_mme_conv_recipe,
                                  m_convSubProblems[subProblemIdx].params,
                                  getMmeHal(getChipType()),
                                  getGeoAttr());
        m_convSubProblems[subProblemIdx].recipe = recipeGen.generateRecipe();
    }

    // also create a false recipe from the original params - needed for debug prints
    auto paramsForDebug = params;
    paramsForDebug.strategy.recurringMisalignmentOptEn = false; // disable recurring misalignment optimization
                                                                // in the false recipe
    RecipeGenerator recipeGen(paramsForDebug.isGemmOperation() ? e_mme_bgemm_recipe : e_mme_conv_recipe,
                              paramsForDebug,
                              getMmeHal(getChipType()),
                              getGeoAttr());
    m_recipe = recipeGen.generateRecipe();
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::patchDebugWkldID(const unsigned wkldID, Desc& desc)
{
    desc.wkldID = wkldID;
}

template<typename Desc>
inline constexpr bool supportDualGemm()
{
    return std::is_same<gaudi3::Mme::Desc, Desc>::value;
}

template<typename Desc>
inline constexpr bool supportCache()
{
    return std::is_same<gaudi3::Mme::Desc, Desc>::value;
}

template<typename Desc>
inline constexpr bool supportEuTracing()
{
    return std::is_same<gaudi3::Mme::Desc, Desc>::value;
}

// This method should be called post patching so storeEn0/storeEn1 have valid values.
// Outputs addresses are returned if storEn is set, otherwise the descriptor will not write so no patch point is needed.
template<typename Desc>
uint64_t*
MmeDescriptorGenerator<Desc>::mmeGetTensorAddressFields(const EMmeOperand operand, Desc& desc, bool secondGemm) const
{
    if constexpr (!supportDualGemm<Desc>())
    {
        MME_ASSERT(secondGemm == 0, "chip does not support dualGemm mode");
    }

    uint64_t* addr = nullptr;
    const EMmeOpType op = getOriginalParams().opType;

    switch (operand)
    {
        case e_mme_op_x:
            if (op == e_mme_dedx || op == e_mme_transposed_dedx)
            {
                if (desc.header.storeEn0)
                {
                    if constexpr (supportDualGemm<Desc>())
                        addr = secondGemm ? &desc.tensorCOut.dualGemm.baseAddrGemm1.addr : &desc.baseAddrCOut0.addr;
                    else
                        addr = &desc.baseAddrCOut0.addr;
                }
            }
            else
            {
                if constexpr (supportDualGemm<Desc>())
                    addr = secondGemm ? &desc.tensorA.dualGemm.baseAddrGemm1.addr : &desc.baseAddrA.addr;
                else
                    addr = &desc.baseAddrA.addr;
            }
            break;

        case e_mme_op_w:
            if (isDedwOperation(op))
            {
                if (desc.header.storeEn0)
                {
                    if constexpr (supportDualGemm<Desc>())
                        addr = secondGemm ? &desc.tensorCOut.dualGemm.baseAddrGemm1.addr : &desc.baseAddrCOut0.addr;
                    else
                        addr = &desc.baseAddrCOut0.addr;
                }
            }
            else
            {
                if constexpr (supportDualGemm<Desc>())
                    addr = secondGemm ? &desc.tensorB.dualGemm.baseAddrGemm1.addr : &desc.baseAddrB.addr;
                else
                    addr = &desc.baseAddrB.addr;
            }
            break;

        case e_mme_op_y:
            switch (op)
            {
                case e_mme_memcpy:
                case e_mme_trans:
                case e_mme_gemm_transpose:
                case e_mme_fwd:
                case e_mme_ab:
                case e_mme_atb:
                case e_mme_abt:
                case e_mme_atbt:
                case e_mme_reductionAdd:
                    if (desc.header.storeEn0)
                    {
                        if constexpr (supportDualGemm<Desc>())
                            addr = secondGemm ? &desc.tensorCOut.dualGemm.baseAddrGemm1.addr : &desc.baseAddrCOut0.addr;
                        else
                            addr = &desc.baseAddrCOut0.addr;
                    }
                    break;
                case e_mme_dedx:
                case e_mme_transposed_dedx:
                    if constexpr (supportDualGemm<Desc>())
                        addr = secondGemm ? &desc.tensorA.dualGemm.baseAddrGemm1.addr : &desc.baseAddrA.addr;
                    else
                        addr = &desc.baseAddrA.addr;
                    break;
                case e_mme_dedw:
                case e_mme_deterministic_dedw:
                    if constexpr (supportDualGemm<Desc>())
                        addr = secondGemm ? &desc.tensorB.dualGemm.baseAddrGemm1.addr : &desc.baseAddrB.addr;
                    else
                        addr = &desc.baseAddrB.addr;
                    break;
                default:
                    MME_ASSERT(0, "invalid operation");
                    break;
            }
            break;

        case e_mme_op_o:
            if (desc.header.storeEn1)
            {
                if constexpr (supportDualGemm<Desc>())
                    addr = secondGemm ? &desc.tensorCOut.dualGemm.baseAddrGemm1Dup.addr : &desc.baseAddrCOut1.addr;
                else
                    addr = &desc.baseAddrCOut1.addr;
            }
            break;

        default:
            MME_ASSERT(0, "invalid tensor operand");
    }

    return addr;
}

template<typename Desc>
template<typename TensorDesc>
TensorDesc* MmeDescriptorGenerator<Desc>::mmeGetTensorDescriptor(const EMmeOperand operand, Desc& desc) const
{
    TensorDesc* tensorDesc = nullptr;
    const EMmeOpType op = getOriginalParams().opType;

    switch (operand)
    {
        case e_mme_op_x:
            if (op == e_mme_dedx || op == e_mme_transposed_dedx)
            {
                tensorDesc = &desc.tensorCOut;
            }
            else
            {
                tensorDesc = &desc.tensorA;
            }
            break;

        case e_mme_op_w:
            if (isDedwOperation(op))
            {
                tensorDesc = &desc.tensorCOut;
            }
            else
            {
                tensorDesc = &desc.tensorB;
            }
            break;

        case e_mme_op_y:
            switch (op)
            {
                case e_mme_memcpy:
                case e_mme_trans:
                case e_mme_gemm_transpose:
                case e_mme_fwd:
                case e_mme_ab:
                case e_mme_atb:
                case e_mme_abt:
                case e_mme_atbt:
                case e_mme_reductionAdd:
                    tensorDesc = &desc.tensorCOut;
                    break;
                case e_mme_dedx:
                case e_mme_transposed_dedx:
                    tensorDesc = &desc.tensorA;
                    break;
                case e_mme_dedw:
                case e_mme_deterministic_dedw:
                    tensorDesc = &desc.tensorB;
                    break;
                default:
                    MME_ASSERT(0, "invalid operation");
                    break;
            }
            break;

        case e_mme_op_o:
            tensorDesc = nullptr;
            break;

        default:
            MME_ASSERT(0, "invalid tensor operand");
    }

    return tensorDesc;
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::patchContextId(uint64_t contextId)
{
    for (auto& activation : getMmeActivations())
    {
        for (auto& desc : activation.descriptors)
        {
            desc.perfEvtIn.value = contextId;
            desc.perfEvtOut.value = contextId;
        }
    }
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::mmeIncrementDataTensorViews(const MmeCommon::EMmeOperand operand,
                                                               const uint64_t offset)
{
    for (auto& activation : getMmeActivations())
    {
        for (auto& desc : activation.descriptors)
        {
            uint64_t* addr = mmeGetTensorAddressFields(operand, desc);
            if (addr) (*addr) += offset;

            if constexpr (supportDualGemm<Desc>())
            {
                if (desc.header.dualGemm)
                {
                    uint64_t* secondGemmAddr = mmeGetTensorAddressFields(operand, desc, true);
                    if (secondGemmAddr) (*secondGemmAddr) += offset;
                }
            }
        }
    }
}

template<typename Desc>
uint64_t& MmeDescriptorGenerator<Desc>::getInputAddrPtr(Desc& desc, const EMmeOperand operand, bool secondGemm) const
{
    if constexpr (!supportDualGemm<Desc>())
    {
        MME_ASSERT(secondGemm == 0, "chip does not support dualGemm mode");
    }

    const EMmeOpType opType = getOriginalParams().opType;
    switch (operand)
    {
        case e_mme_op_x:
            MME_ASSERT(opType != e_mme_dedx && opType != e_mme_transposed_dedx, "tensor X is not input for DEDX");
            if constexpr (supportDualGemm<Desc>())
                return secondGemm ? desc.tensorA.dualGemm.baseAddrGemm1.addr : desc.baseAddrA.addr;
            else
                return desc.baseAddrA.addr;
        case e_mme_op_w:
            MME_ASSERT(!isDedwOperation(opType), "tensor W is not input for DEDW");
            if constexpr (supportDualGemm<Desc>())
                return secondGemm ? desc.tensorB.dualGemm.baseAddrGemm1.addr : desc.baseAddrB.addr;
            else
                return desc.baseAddrB.addr;
        case e_mme_op_y:
            MME_ASSERT((isDedwOperation(opType) || opType == e_mme_dedx || opType == e_mme_transposed_dedx),
                       "tensor Y is not input for fwd \\ gemm ops \\ dma");
            if (isDedwOperation(opType))
            {
                if constexpr (supportDualGemm<Desc>())
                    return secondGemm ? desc.tensorB.dualGemm.baseAddrGemm1.addr : desc.baseAddrB.addr;
                else
                    return desc.baseAddrB.addr;
            }
            else
            {
                if constexpr (supportDualGemm<Desc>())
                    return secondGemm ? desc.tensorA.dualGemm.baseAddrGemm1.addr : desc.baseAddrA.addr;
                else
                    return desc.baseAddrA.addr;
            }
        default:
            MME_ASSERT(0, "invalid operand");
            return desc.baseAddrA.addr;
    }
}

template<typename Desc>
uint64_t& MmeDescriptorGenerator<Desc>::getOutputAddrPtr(Desc& desc,
                                                         const EMmeOperand operand,
                                                         bool isPrimary,
                                                         bool secondGemm) const
{
    if constexpr (!supportDualGemm<Desc>())
    {
        MME_ASSERT(secondGemm == 0, "chip does not support dualGemm mode");
    }

    switch (getOriginalParams().opType)
    {
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_atbt:
        case e_mme_abt:
        case e_mme_reductionAdd:
        {
            MME_ASSERT(operand == (isPrimary ? e_mme_op_y : e_mme_op_o), "output operand in FWD is y or o");
            break;
        }
        case e_mme_dedx:
        case e_mme_transposed_dedx:
        {
            MME_ASSERT(operand == (isPrimary ? e_mme_op_x : e_mme_op_o), "output operand in DEDX is x or o");
            break;
        }
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
        {
            MME_ASSERT(operand == (isPrimary ? e_mme_op_w : e_mme_op_o), "output operand in DEDW is w or o");
            break;
        }
        default:
            MME_ASSERT(0, "invalid operation");
            break;
    }

    if constexpr (supportDualGemm<Desc>())
    {
        if (isPrimary)
        {
            return secondGemm ? desc.tensorCOut.dualGemm.baseAddrGemm1.addr : desc.baseAddrCOut0.addr;
        }
        else
        {
            return secondGemm ? desc.tensorCOut.dualGemm.baseAddrGemm1Dup.addr : desc.baseAddrCOut1.addr;
        }
    }
    else
    {
        return isPrimary ? desc.baseAddrCOut0.addr : desc.baseAddrCOut1.addr;
    }
}

template<typename Desc>
bool MmeDescriptorGenerator<Desc>::canSquashRois()
{
    if (getOriginalParams().controls.squashIORois) return true;

    if (getOriginalParams().controls.signalingMode == MmeCommon::e_mme_signaling_desc_with_store &&
        !getOriginalParams().strategy.maskedBgemm)
    {
        unsigned storingDescs = 0;
        for (auto& activation : getMmeActivations())
        {
            if (activation.getDesc(0).header.storeEn0) storingDescs++;
        }
        //  if theres a single descriptor and signaling mode is desc it is effectively signal once mode
        if (storingDescs == 1) return true;
    }

    return false;
}

template<typename Desc>
const std::shared_ptr<CommonRoiCalculator<Desc>>& MmeDescriptorGenerator<Desc>::getRoiCalculator() const
{
    return m_roiCalculator;
}

template<typename Desc>
const std::shared_ptr<CommonRoiCalculator<Desc>>& MmeDescriptorGenerator<Desc>::getRoiCalculator(const MmeCommon::MmeRecipe& recipe) const
{
    m_roiCalculator->resetRecipe(recipe);
    return m_roiCalculator;
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::createRoiCalculator()
{
    if (!m_roiCalculator)
    {
        m_roiCalculator = std::make_shared<CommonRoiCalculator<Desc>>(getRecipe(), getOriginalParams(), m_paramsVec);
    }
    else
    {
        m_roiCalculator->resetRecipe(getRecipe());
        m_roiCalculator->resetParams(getOriginalParams());
        m_roiCalculator->resetParamsVec(m_paramsVec);
    }
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::patchOutputView(MmeActivation<Desc>& activation,
                                                   const EMmeOperand operand,
                                                   const uint64_t addr,
                                                   const bool isSram,
                                                   const bool isPrimary,
                                                   const bool squashRois,
                                                   const bool calcRoi)
{
    for (auto& desc : activation.descriptors)
    {
        uint64_t& baseAddr = getOutputAddrPtr(desc, operand, isPrimary, false);
        // When there is no store we need to set different address so that when the descriptor that does store is
        // processed by Synapse code gen patch logic, it hits different address so patch history optimization will
        // not be activated to ignore it, but a new patch point is created.
        if (desc.header.storeEn0 == 0)
        {
            baseAddr = -1ULL;  // 0xFF...F is a value that is not expected to be actual base address of any store desc
        }
        else
        {
            baseAddr += addr;
        }
    }

    if (!squashRois && calcRoi)
    {
        calculateLinearRanges(activation, getRoiCalculator(), operand, false /* isInput */, isSram, squashRois);
    }
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::calculateLinearRanges(MmeActivation<Desc>& activation,
                                                         const std::shared_ptr<CommonRoiCalculator<Desc>>& roiCalc,
                                                         const EMmeOperand operand,
                                                         const bool isInput,
                                                         const bool isSram,
                                                         const bool squashRoi)
{
    uint64_t& addrPtr = isInput ? getInputAddrPtr(activation.getDesc(0), operand)
                                : getOutputAddrPtr(activation.getDesc(0), operand, operand!=e_mme_op_o);
    roiCalc->createRoi(addrPtr, activation, operand, isSram, squashRoi);
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::patchInputTensor(MmeActivation<Desc>& activation,
                                                    const EMmeOperand operand,
                                                    const std::optional<TensorMemoryMetaData>& tensorMemoryData,
                                                    bool calcRoi)
{
    MME_ASSERT(tensorMemoryData != std::nullopt, "Input memory data must be set");
    uint64_t addr    = tensorMemoryData->addr;
    bool     isSram  = tensorMemoryData->isSram;
    auto     roiCalc = getRoiCalculator();

    bool addrIsAligned = (addr % getMmeHal(getChipType()).getMemoryClSize() == 0);
    bool squashRois = canSquashRois();

    for (auto& desc : activation.descriptors)
    {
        setSBCacheDisable(operand, isSram, addrIsAligned, desc);
        uint64_t& baseAddr = getInputAddrPtr(desc, operand);
        baseAddr += addr;
    }

    if (!squashRois && calcRoi)
    {
        calculateLinearRanges(activation, getRoiCalculator(), operand, true, isSram, squashRois);
    }
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::patchOutputTensor(MmeActivation<Desc>& activation,
                                                     const EMmeOperand operand,
                                                     const std::optional<TensorMemoryMetaData>& primaryOutputTensorMemoryData,
                                                     bool oOperandUsed,
                                                     const std::optional<TensorMemoryMetaData>& secondaryOutputTensorMemoryData,
                                                     const bool calcRoi)
{
    MME_ASSERT(primaryOutputTensorMemoryData != std::nullopt, "Output memory data must be se");
    MME_ASSERT(!oOperandUsed || secondaryOutputTensorMemoryData != std::nullopt, "o output is used but its memory data is not set");

    uint64_t addr0   = primaryOutputTensorMemoryData->addr;
    bool     isSram0 = primaryOutputTensorMemoryData->isSram;
    uint64_t addr1   = secondaryOutputTensorMemoryData->addr;
    bool     isSram1 = secondaryOutputTensorMemoryData->isSram;

    bool squashRois = canSquashRois();
    patchSignalColoring(activation, isSram0, isSram1);
    auto roiCalc = getRoiCalculator();

    patchOutputView(activation, operand, addr0, isSram0, true, squashRois, calcRoi);
    if (oOperandUsed)
    {
        // turn on secondary output if we got an address for it.
        for (auto& desc : activation.descriptors)
        {
            desc.header.storeEn1 = desc.header.storeEn0;
        }
        patchOutputView(activation, e_mme_op_o, addr1, isSram1, false, squashRois, calcRoi);
    }
}

static void getTensorRoles(OperandRoles operandRoles,
                           TensorRoles& aTensorRole,
                           TensorRoles& bTensorRole,
                           TensorRoles& cTensorRole,
                           TensorRoles& oTensorRole)
{
    aTensorRole = operandRoles[e_mme_op_a];
    bTensorRole = operandRoles[e_mme_op_b];
    cTensorRole = operandRoles[e_mme_op_c];
    oTensorRole = MmeCommon::OUTPUT_TENSOR_O;  // When o is used it is always the primary tensor
}

template<typename Desc>
TensorRoles MmeDescriptorGenerator<Desc>::getAuxTensorRole(MmeAuxTensorIdx idx) const
{
    switch (idx)
    {
        case MASKED_BGEMM_A:
            return TensorRoles::AUX_ROLE_MASKED_BGEMM_A;
        case MASKED_BGEMM_B:
            return TensorRoles::AUX_ROLE_MASKED_BGEMM_B;
        case CD_SCRATCHPAD:
            return TensorRoles::AUX_ROLE_CD_SCRATCHPAD;
        case CD_REDUCTION:
            return TensorRoles::AUX_ROLE_CD_REDUCTION;
        default:
            MME_ASSERT(0, "Invalid aux tensor index");
    }
    return TensorRoles::AUX_ROLE_MASKED_BGEMM_A;
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::patchMmeDescriptors(MmePatchMetaData patchMetaData,
                                                       bool calcRoi)
{
    bool squashRois = canSquashRois();
    auto roiCalc = getRoiCalculator();
    const auto& params = getOriginalParams();

    EMmeOperand operandA = params.getExternalOperand(e_mme_op_a);
    EMmeOperand operandB = params.getExternalOperand(e_mme_op_b);
    EMmeOperand operandC = params.getExternalOperand(e_mme_op_c);

    for (auto& activation : getMmeActivations())
    {
        TensorRoles aTensorRole, bTensorRole, cTensorRole, oTensorRole;
        getTensorRoles(activation.operandRoles, aTensorRole, bTensorRole, cTensorRole, oTensorRole);

        patchInputTensor(activation, operandA, patchMetaData.tensorMetaData[aTensorRole], calcRoi);
        if (patchMetaData.bOperandUsed)
        {
            patchInputTensor(activation, operandB, patchMetaData.tensorMetaData[bTensorRole], calcRoi);
        }
        patchOutputTensor(activation,
                          operandC,
                          patchMetaData.tensorMetaData[cTensorRole],
                          patchMetaData.oOperandUsed,
                          patchMetaData.tensorMetaData[oTensorRole],
                          calcRoi);
    }

    if (squashRois && calcRoi)
    {
        MME_ASSERT(!getOriginalParams().strategy.maskedBgemm, "maskedBgemm doesnt support ROI squashing");
        // the first activation is enough since the next activations have a physical dependency on this one.
        auto& firstAct = getMmeActivations().front();
        TensorRoles aTensorRole, bTensorRole, cTensorRole, oTensorRole;
        getTensorRoles(firstAct.operandRoles, aTensorRole, bTensorRole, cTensorRole, oTensorRole);

        bool isAInSram = patchMetaData.tensorMetaData[aTensorRole]->isSram;
        bool isBInSram = patchMetaData.tensorMetaData[bTensorRole].value_or(TensorMemoryMetaData{}).isSram;
        bool isCInSram = patchMetaData.tensorMetaData[cTensorRole]->isSram;
        bool isOInSram = patchMetaData.tensorMetaData[oTensorRole].value_or(TensorMemoryMetaData{}).isSram;

        calculateLinearRanges(firstAct, roiCalc, operandA, true  /* isInput */, isAInSram, squashRois);
        calculateLinearRanges(firstAct, roiCalc, operandB, true  /* isInput */, isBInSram, squashRois);
        calculateLinearRanges(firstAct, roiCalc, operandC, false /* isInput */, isCInSram, squashRois);
        if (patchMetaData.oOperandUsed)
        {
            calculateLinearRanges(firstAct, roiCalc, e_mme_op_o, false /* isInput */, isOInSram, squashRois);
        }

        auto& lastAct = getMmeActivations().back();
        // In case of cd perforation - last activation is reductionAdd
        if (lastAct.operandRoles[0] == AUX_ROLE_CD_REDUCTION)
        {
            TensorRoles aTensorRole, bTensorRole, cTensorRole, oTensorRole;
            getTensorRoles(lastAct.operandRoles, aTensorRole, bTensorRole, cTensorRole, oTensorRole);

            bool isAInSram = patchMetaData.tensorMetaData[aTensorRole]->isSram;
            bool isBInSram = patchMetaData.tensorMetaData[bTensorRole].value_or(TensorMemoryMetaData {}).isSram;
            bool isCInSram = patchMetaData.tensorMetaData[cTensorRole]->isSram;
            bool isOInSram = patchMetaData.tensorMetaData[oTensorRole].value_or(TensorMemoryMetaData {}).isSram;

            calculateLinearRanges(lastAct, roiCalc, operandA, true /* isInput */, isAInSram, squashRois);
            calculateLinearRanges(lastAct, roiCalc, operandB, true /* isInput */, isBInSram, squashRois);
            calculateLinearRanges(lastAct, roiCalc, operandC, false /* isInput */, isCInSram, squashRois);
        }
    }
}

template<typename Desc>
std::vector<std::vector<std::string>> MmeDescriptorGenerator<Desc>::dumpDescriptors(bool dumpAsBinary) const
{
    std::shared_ptr<MmeDescriptorDumper<Desc>> descDumper = getDescriptorDumper();
    std::vector<std::vector<std::string>> dbgDump = {};
    unsigned actCount = 0;
    for (const auto& act : getMmeActivations())
    {
        std::vector<std::string> actDump;
        auto actCountStr = std::to_string(actCount);
        descDumper->setFileNamePrefix("act_" + actCountStr + "_");
        unsigned descCount = 0;
        for (const auto& desc : act.descriptors)
        {
            descDumper->clear();
            *descDumper << "\nActivation " + actCountStr + " Descriptor " + std::to_string(descCount) + ": \n";
            std::string descDump = descDumper->dump(desc, descCount, dumpAsBinary, false);
            if (!dumpAsBinary)
            {
                actDump.push_back(descDump);
            }
            descCount++;
        }
        actCount++;
        if (!dumpAsBinary)
        {
            dbgDump.push_back(actDump);
        }
    }
    return dbgDump;
}

EMmeReductionOp getReductionOp(const MmeCommon::MmeLayerParams& params, const MmeCommon::MmeRecipe& recipe)
{
    EMmeReductionOp op = params.memoryCfg.reductionOp;
    if (recipe.isReductionEn())
    {
        // make sure that we dont override GC decision.
        MME_ASSERT(op == e_mme_reduction_add || op == e_mme_reduction_none, "overriding GC reduction decision");
        op = e_mme_reduction_add;
    }
    return op;
}

EMmeReductionRm getReductionRm(const MmeCommon::MmeLayerParams& params, const MmeCommon::MmeRecipe& recipe)
{
    EMmeReductionRm rm = params.memoryCfg.reductionRm;
    if (recipe.isReductionEn())
    {
        // make sure that we dont override GC decision.
        MME_ASSERT(rm == e_mme_reduction_round_down || rm == e_mme_reduction_round_nr,
                   "overriding GC reduction decision");
        rm = e_mme_reduction_round_down;
    }
    return rm;
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::configureMemoryDirectives(Desc& desc, const MmeCommon::MmeRecipe& recipe) const
{
    // configure reduction - AXIUserData
    const auto& params = getOriginalParams();
    const auto& output = params.getOperand(e_mme_op_c);
    auto configureMemory = MmeMemoryConfigMgr::getMmeMemoryConfigMgr(getChipType());
    configureMemory->setReductionParams(getReductionOp(params, recipe),
                                        getReductionRm(params, recipe),
                                        output.elementType,
                                        params.controls.clippingEn);
    configureMemory->setReductionUserBits(desc);

    // configure cache - not needed for H6 and below.
    if constexpr (!supportCache<Desc>()) return;
    configureMemory->setCacheParams(params.memoryCfg.cacheDirective,
                                    params.memoryCfg.clss,
                                    params.memoryCfg.qos,
                                    params.memoryCfg.mcId);
    configureMemory->setCacheUserBits(desc);
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::configurePerfEvents(ActivationVec<Desc>& activations)
{
    // early exit in case tracing is not enabled
    if (e_mme_trace_mode_none == getOriginalParams().tracing.traceMode) return;
    int actIdx = 0;
    for (auto& activation : activations)
    {
        bool first = actIdx == 0;
        bool last = actIdx == activations.size() - 1;
        bool partialDesc = (activation.numSignals == 0);
        for (auto& desc : activation.descriptors)
        {
            setDescPerfEvent(first, last, partialDesc, desc);
        }
        actIdx++;
    }
}

template<typename Desc>
void MmeDescriptorGenerator<Desc>::setDescPerfEvent(const bool first,
                                                    const bool last,
                                                    const bool partialDesc,
                                                    Desc& desc)
{
    constexpr static uint8_t startMask = 0b01;  // set event at start of operation
    constexpr static uint8_t endMask = 0b10;  // set event at end of operation

    desc.perfEvtIn.dw = 0;
    desc.perfEvtOut.dw = 0;
    auto traceMode = getOriginalParams().tracing.traceMode;
    if (partialDesc && traceMode == MmeCommon::e_mme_trace_mode_desc)
    {
        traceMode = MmeCommon::e_mme_trace_mode_layer_act;
    }

    switch (traceMode)
    {
        default:
            MME_ASSERT(0, "invalid trace mode");
        case e_mme_trace_mode_none:
            break;
        case e_mme_trace_mode_layer_act:
            if (first)
            {
                setSinglePerfEvent(e_mme_trace_input, startMask, desc);
            }
            if (last)
            {
                setSinglePerfEvent(e_mme_trace_output, endMask, desc);
            }
            break;
        case e_mme_trace_mode_desc:
            setSinglePerfEvent(e_mme_trace_input, startMask, desc);
            setSinglePerfEvent(e_mme_trace_output, endMask, desc);
            break;
        case e_mme_trace_mode_advanced:
            setSinglePerfEvent(e_mme_trace_input, startMask | endMask, desc);
            if constexpr (supportEuTracing<Desc>())
            {
                setSinglePerfEvent(e_mme_trace_eu, startMask | endMask, desc);
            }
            setSinglePerfEvent(e_mme_trace_output, startMask | endMask, desc);
            break;
    }
}

template<typename Desc>
bool MmeDescriptorGenerator<Desc>::validateParamOfGemmOp(const MmeCommon::MmeLayerParams& params, std::string& errorMsg)
{
    const auto& a = params.getOperand(e_mme_op_a);
    const auto& b = params.getOperand(e_mme_op_b);
    const auto& c = params.getOperand(e_mme_op_c);
    if (params.isGemmOperation() && !params.isGemmDmaOperation())
    {
        for (int batchDim = 2; batchDim < MmeCommon::c_batchDimNr + 2; batchDim++)
        {
            // Batches of A/B can be 1 (broadcast), otherwise they must be equal to batches of C (classic bgemm)
            if (!(((a.sizes[batchDim] == 1 || a.sizes[batchDim] == c.sizes[batchDim]) &&
                   ((b.sizes[batchDim] == 1 || b.sizes[batchDim] == c.sizes[batchDim])))))
            {
                errorMsg = "batch sizes are invalid";
                return false;
            }
        }
    }

    if (params.isGemmDmaOperation())
    {
        if (a.sizes[MAX_DIMENSION - 1] != 1 || c.sizes[MAX_DIMENSION - 1] != 1)
        {
            errorMsg = "gemm transpose doesnt have enough loops to support 5 dim tensors";
            return false;
        }
    }
    return true;
}

template<typename Desc>
bool MmeDescriptorGenerator<Desc>::validateParamOfReductionAddOp(const MmeCommon::MmeLayerParams& params, std::string& errorMsg)
{
    if (params.opType == e_mme_reductionAdd)
    {
        unsigned packingFactor = params.getPackingFactor();
        unsigned reductionLevel = params.getReductionLevel();
        if (params.y.isStrided() || params.y.isStrided())
        {
            errorMsg = "In reductionAdd, tensors cannot be strided";
            return false;
        }

        if (params.x.sizes[1] != packingFactor)
        {
            errorMsg = "In reductionAdd, packing factor is inconsistent with x tensor sizes";
            return false;
        }
        if (params.x.sizes[0] != packingFactor * reductionLevel)
        {
            errorMsg = "In reductionAdd, packing factor and reduction level are inconsistent with x tensor sizes";
            return false;
        }

        for (int i=2; i < MME_MAX_TENSOR_DIMS; i++)
        {
            if (params.x.sizes[i] != 1 || params.w.sizes[i] != 1 || params.y.sizes[i] != 1 )
            {
                errorMsg = "In reduction add, all batch dims are expected to be 1";
                return false;
            }
        }
    }
    return true;
}

template<typename Desc>
bool MmeDescriptorGenerator<Desc>::setParams(const MmeLayerParams& newParams)
{
    std::string errorMsg;
    if (!validateParams(newParams, errorMsg))
    {
        MME_ASSERT(0, errorMsg.c_str());
        return false;
    }

    m_commonGeoAttr = MmeCommon::getGeoAttr(getChipType(), newParams);
    m_originalParams.emplace(newParams);
    m_convSubProblems.reset(newParams);
    createRecipes(newParams);
    createRoiCalculator();

    static const bool DUMP_MME_PARAMS_TO_JSON_FILE = getenv("DUMP_MME_PARAMS_TO_JSON_FILE") != nullptr;
    if (DUMP_MME_PARAMS_TO_JSON_FILE)
    {
        MmeParamsDumper(newParams).dumpMmeParamsJson();
    }
    return true;
}

template<typename Desc>
unsigned int MmeDescriptorGenerator<Desc>::addParams(const MmeLayerParams& newParams)
{
    m_paramsVec->emplace_back(newParams);
    return m_paramsVec->size() - 1;
}

template<typename Desc>
size_t MmeDescriptorGenerator<Desc>::calculateActivationsCount() const
{
    if (getCurrentSubProblem() == nullptr) return m_recipe.getIterator().size();
    return std::accumulate(m_convSubProblems.begin(),
                           m_convSubProblems.end(),
                           0,
                           [](size_t activationsNr, const ConvSubProblem& subProblem) {
                               return activationsNr + subProblem.recipe.getIterator().size();
                           });
}

#ifndef DSC_CACHE
#define DSC_CACHE DescriptorsCache<MmeCommon::MmeLayerParams, MmeActivation<Desc>>::getCacheInstance()
#endif

template<typename Desc>
std::shared_ptr<const std::vector<MmeActivation<Desc>>>
MmeDescriptorGenerator<Desc>::getSharedOwnershipForCachedActivations(const MmeLayerParams& originalParams)
{
    return DSC_CACHE.get(originalParams);
}

template<typename Desc>
bool MmeDescriptorGenerator<Desc>::getActivationsFromCache()
{
    if (!isPerforated() && getOriginalParams().isUsingDescCache())
    {
        auto cachedActivationVecPtr = DSC_CACHE.get(getOriginalParamsAndHash());
        if (cachedActivationVecPtr != nullptr)
        {
            m_activations.reserve(cachedActivationVecPtr->size());
            m_activations.insert(m_activations.end(), cachedActivationVecPtr->begin(), cachedActivationVecPtr->end());
            MmeBrain& mmeBrain = getMMEBrain();
            for (auto& activation : m_activations)
            {
                if constexpr (descToChipType<Desc>() == e_mme_Gaudi2)
                {
                    mmeBrain.addNumOfRollUpsForCurActivation(activation.numTetrises);
                }
                m_commonGeoAttr->setPrimaryTensors(!activation.isMask);
            }
            return true;
        }
    }
    return false;
}

template<typename Desc>
bool MmeDescriptorGenerator<Desc>::addParamsActivationsToCache()
{
    if (getOriginalParams().isUsingDescCache())
    {
        DSC_CACHE.add(getOriginalParamsAndHash(), m_activations);
        return true;
    }
    return false;
}

template<typename Desc>
std::vector<std::string> MmeDescriptorGenerator<Desc>::getDescCacheDebugInfo() const
{
    return DSC_CACHE.getDebugInfo();
}

// instantiate for all chips
template class MmeDescriptorGenerator<Gaudi2::Mme::Desc>;
template Gaudi2::Mme::MmeTensorDesc*
MmeDescriptorGenerator<Gaudi2::Mme::Desc>::mmeGetTensorDescriptor<Gaudi2::Mme::MmeTensorDesc>(const EMmeOperand,
                                                                                              Gaudi2::Mme::Desc&) const;
template class MmeDescriptorGenerator<gaudi3::Mme::Desc>;
template gaudi3::Mme::MmeTensorDesc*
MmeDescriptorGenerator<gaudi3::Mme::Desc>::mmeGetTensorDescriptor<gaudi3::Mme::MmeTensorDesc>(const EMmeOperand,
                                                                                              gaudi3::Mme::Desc&) const;

}  // namespace MmeCommon
