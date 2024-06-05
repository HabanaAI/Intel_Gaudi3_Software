#include <unordered_map>
#include "mme/mme_logger.h"
#include "mme/mme_utils.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "include/utils/gaudi_desc_dumper.h"

using namespace MmeCommon;
using namespace mme_utils;

std::string MmeLogger::createTensorString(const MmeCommon::MmeTensorView& tensor)
{
    return fmt::format("Element Type : {}, Size [{}], Base [{}], Dcore Base [{}], Strides [{}]",
                       toString(tensor.elementType),
                       ::toString(tensor.sizes, ','),
                       ::toString(tensor.bases, ','),
                       ::toString(tensor.dcoreBases, ','),
                       ::toString(tensor.strides, ','));
}

void MmeLogger::printMmeParams(const MmeCommon::MmeLayerParams& params)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(MME_STACK))
    {
        return;
    }

    LOG_TRACE(MME_STACK, "MMeLayerParams:");

    LOG_TRACE(MME_STACK, "       Op Type: {}", toString(params.opType));
    LOG_TRACE(MME_STACK,
              "       ConvParams: strides [{}], dilations [{}], paddings [{}], spDimsNr: {}",
              ::toString(params.conv.stride, ','),
              ::toString(params.conv.dilation, ','),
              ::toString(params.conv.padding, ','),
              params.conv.spatialDimsNr);
    LOG_TRACE(MME_STACK, "       Tensor x : {}", createTensorString(params.x));
    LOG_TRACE(MME_STACK, "       Tensor w : {}", createTensorString(params.w));
    LOG_TRACE(MME_STACK, "       Tensor y : {}", createTensorString(params.y));
    if (params.strategy.maskedBgemm)
    {
        LOG_TRACE(MME_STACK, "       Tensor xAux : {}", createTensorString(params.xAux));
        LOG_TRACE(MME_STACK, "       Tensor yAux : {}", createTensorString(params.yAux));
        LOG_TRACE(MME_STACK, "       Tensor wAux : {}", createTensorString(params.wAux));
    }
    LOG_TRACE(MME_STACK, "       spBase: {}", params.spBase);
    LOG_TRACE(MME_STACK, "       spSize: {}", params.spSize);
    LOG_TRACE(MME_STACK,
              "       Controls: signal_mode= {}, atomic_add= {}, squash_roi= {}, relu_enable= {}, sbCacheEn= {}",
              toString(params.controls.signalingMode),
              params.controls.atomicAdd,
              params.controls.squashIORois,
              params.controls.reluEn,
              params.controls.sbCacheEn);
    LOG_TRACE(MME_STACK,
              "       Numerics: conversionRoundingMode= {}, accRoundingMode= {}, roundingMode= {}",
              params.controls.conversionRoundingMode,
              params.controls.accRoundingMode,
              params.controls.roundingMode);
    LOG_TRACE(MME_STACK,
              "                 infNanMode: opA= {}, opB= {}, out= {} ",
              params.controls.infNanModeA, params.controls.infNanModeB, params.controls.infNanModeOut);
    LOG_TRACE(MME_STACK,
              "                 clippingEn= {}, clipInfIn= {}",
              params.controls.clippingEn, params.controls.clipInfIn);
    LOG_TRACE(MME_STACK,
              "                 FP8 Bias: opA= {}, opB= {}, out= {}",
              params.controls.fp8BiasIn,
              params.controls.fp8BiasIn2,
              params.controls.fp8BiasOut);
    LOG_TRACE(MME_STACK,
              "       Strategy: geometry={}, pattern={}, pipelineLevel={}, packingFactor={}",
              toString(params.strategy.geometry),
              toString(params.strategy.pattern),
              params.strategy.pipelineLevel,
              params.strategy.packingFactor);
    LOG_TRACE(MME_STACK,
              "                 lowering={}, sbReuse={}, partialsToMemoryEn={}, alignedAddr={}, alignOpt={}",
              params.strategy.loweringEn,
              params.strategy.sbReuse,
              params.strategy.partialsToMemoryEn,
              params.strategy.alignedAddresses,
              params.strategy.recurringMisalignmentOptEn);
    LOG_TRACE(MME_STACK,
              "                 dedwAsBgemm={}, batchConcurrency={}, cdConcurrency={}, flattenEn={}, teAccelerationEn={}",
              params.strategy.dedwAsBgemmEn,
              params.strategy.batchConcurrencyEn,
              params.strategy.cdConcurrencyEn,
              params.strategy.flattenEn,
              params.strategy.teAccelerationEn);
    LOG_TRACE(MME_STACK,
              "                 memset_dedx_void_pixels={}, dedxDynamicPadding={}, maskedBgemm={} ",
              params.strategy.memsetDedxVoidPixels,
              params.strategy.dedxDynamicPadding,
              params.strategy.maskedBgemm);
}

void MmeLogger::printMmeRecipeInfo(MmeCommon::MmeDescriptorGeneratorBase* descGenerator)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(MME_STACK))
    {
        return;
    }
    for (const auto& str : descGenerator->getRecipeDebugInfo())
    {
        LOG_TRACE(MME_STACK, "{}", str);
    }
}

void MmeLogger::printMmeRecipeInfo(gaudi::MmeDescriptorGenerator* descGenerator)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(MME_STACK))
    {
        return;
    }
    for (const auto& str : descGenerator->getRecipeDebugInfo())
    {
        LOG_TRACE(MME_STACK, "{}", str);
    }
}

void MmeLogger::printMmePerf(const MmeCommon::PerfAttr& perfAttr)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(MME_STACK))
    {
        return;
    }

    LOG_TRACE(MME_STACK, "MME Perf Attributes:");
    LOG_TRACE(MME_STACK, "{}", perfAttr.print());
}

void MmeLogger::printMmeDescriptor(const MmeCommon::MmeDescriptorGeneratorBase* descGenerator)
{
    if (GCFG_PRINT_MME_DESCRIPTORS.value() && LOG_LEVEL_AT_LEAST_TRACE(MME_STACK))
    {
        auto dump = descGenerator->dumpDescriptors(false);
        for (const auto& actDump : dump)
        {
            for (const auto& descDump : actDump)
            {
                LOG_TRACE(MME_STACK, "{}", descDump);
            }
        }
    }
}

void MmeLogger::printMmeDescriptor(const gaudi::MmeDescriptorGenerator* descGenerator)
{
    if (GCFG_PRINT_MME_DESCRIPTORS.value() && LOG_LEVEL_AT_LEAST_TRACE(MME_STACK))
    {
        std::vector<std::vector<std::string>> dump = descGenerator->dumpDescriptors(false);
        for (const auto& actDump : dump)
        {
            for (const auto& descDump : actDump)
            {
                LOG_TRACE(MME_STACK, "{}", descDump);
            }
        }
    }
}

void MmeLogger::printDebugInfo(const MmeCommon::PerfAttr&             perfAttr,
                               MmeCommon::MmeDescriptorGeneratorBase* descGenerator)
{
    LOG_DEBUG(MME_STACK, "num of generated activations: {}", descGenerator->getMmeActivationNr());
    printMmeRecipeInfo(descGenerator);
    printMmePerf(perfAttr);
    printMmeDescriptor(descGenerator);
}

void MmeLogger::printDebugInfoGaudi(gaudi::MmeDescriptorGenerator* descGenerator)
{
    LOG_DEBUG(MME_STACK, "num of generated activations: {}", descGenerator->getMmeActivations().size());
    printMmeRecipeInfo(descGenerator);
    printMmeDescriptor(descGenerator);
}

/*
 * Return geometry info - used in trace analyzer
 */
std::string MmeLogger::getGeometryInfo(const MmeLayerParams&       params,
                                       MmeDescriptorGeneratorBase& descGenerator)
{
    std::stringstream ss;

    ss << "MME Geometry: Name=" << toString(descGenerator.getGeometry());
    ss << ", Width=" << std::to_string(descGenerator.getGeometryWidth());
    ss << ", Height=" << std::to_string(descGenerator.getGeometryHeight());
    ss << ", Pattern=" << toString(params.strategy.pattern);
    ss << ", Raster=" << (std::string)(descGenerator.getRecipe().raster ? "True" : "False");
    if (params.isGemmOperation() || params.opType == e_mme_fwd)
    {
        ss << ", Flattening=" << std::to_string(descGenerator.getMMEBrain().getFlatteningFactor());
    }

    ss << ", Aligned Adresses=" << (std::string)(params.strategy.alignedAddresses ? "True" : "False");

    if (params.isGemmOperation() || params.opType == e_mme_dedw)
    {
        ss << ", BatchConcurrency=" << std::to_string(descGenerator.getEffectiveBatchConcurrency());
    }
    if (params.opType == e_mme_dedw)
    {
        ss << ", CDConcurrency=" << std::to_string(descGenerator.getGeometryCdConcurrency());
        ss << ", CDConcurrency dim=" << std::to_string(descGenerator.getSpInterleavingDim(e_mme_op_a));
    }

    if (descGenerator.isAsymPortConfigMode())
    {
        ss << ", asymmetricPortConfig=True";
    }

    return ss.str();
}

/*
 * Return Recipe info - used in trace analyzer
 */
std::string MmeLogger::getRecipeInfo(MmeDescriptorGeneratorBase& descGenerator, const MmeLayerParams& params)
{
    MmeRecipe recipe = descGenerator.getRecipe();

    RecipeType recipeType = recipe.getRecipeType();
    HB_ASSERT((recipeType == e_mme_bgemm_recipe) || (recipeType == e_mme_conv_recipe), "Unsupported recipe type");

    const bool isConv = (recipeType == e_mme_conv_recipe);

    std::stringstream recipeInfo;
    if (isConv && params.opType != e_mme_dedx)
    {
        recipeInfo << ", Lowering Factor=" << (recipe.lowering ? params.w.sizes[DIM_S] : 1);
    }

    EMmeReuseType reuseType = recipe.reuseType();

    recipeInfo << ", Reuse=" + toString(reuseType);
    if (reuseType != e_mme_no_reuse)
    {
        const std::string_view nonSpatialSubviewsStr = isConv ? "CONV" : "BATCH";
        recipeInfo << ", FCD=" << recipe.getFcdSubviews().size() << ", SP=" << recipe.getSpSubviews().size() << ", "
                   << nonSpatialSubviewsStr << "=" << recipe.getNonSpatialSubviews().size();
    }

    if (params.opType == e_mme_fwd || params.opType == e_mme_dedx)
    {
        recipeInfo << ", Packing Factor=" << params.strategy.packingFactor;
        recipeInfo << ", Group Packing Factor=N/A";  // TODO: get value
    }

    if (params.isGemmOperation())
    {
        recipeInfo << ", Batch GEMMs=N/A";  // TODO: get value
        recipeInfo << ", BGemmRollup=N/A";  // TODO: get value
    }

    // Add group packing factor

    return recipeInfo.str();
}

/*
 * Return Recurring Misalignment Info - used in trace analyzer
 */
std::string MmeLogger::getRecurringMisalignmentInfo(MmeDescriptorGeneratorBase& descGenerator,
                                                    const MmeLayerParams&       params,
                                                    const ChipType              chipType)
{
    bool operandA = false;
    bool operandB = false;  // TODO: get recurringMisalignment for B
    bool operandC = false;  // TODO: get recurringMisalignment for C

    std::string mmeRecurringMisslignment;

    operandA = RecurringMisalignmentOptimization::isRecurringMisalignment(params, e_mme_op_a, chipType);
    mmeRecurringMisslignment += ", Recurring Misalignment: operandA=" + (std::string)(operandA ? "True" : "False");
    mmeRecurringMisslignment += ", operandB=N/A";  // TODO: replace with (std::string)(operandB ? "True" : "False");
    mmeRecurringMisslignment += ", operandC=N/A";  // TODO: replace with (std::string)(operandC ? "True" : "False");

    if (operandA && params.opType == e_mme_fwd)
    {
        std::string misalignmentOptFullInfo = descGenerator.getRecurringMisalignmentDebugInfo();
        std::string cutPoints = misalignmentOptFullInfo.substr(misalignmentOptFullInfo.find(", CD cut points=") + 1);
        mmeRecurringMisslignment += ", Recurring Misalignment opt in A operand: ";
        mmeRecurringMisslignment += (misalignmentOptFullInfo.empty() ? "Can not apply" : cutPoints);
    }

    return mmeRecurringMisslignment;
}

/*
 * Return Mme Strategy Attributes - used in trace analyzer
 */
std::string MmeLogger::getMmeStrategyInfo(const MmeLayerParams&       params,
                                          MmeDescriptorGeneratorBase& descGenerator,
                                          const ChipType              chipType)
{
    // Geometry Info
    std::string geometryInfo = getGeometryInfo(params, descGenerator);

    // dcoreConfig Info - only for Gaudi3
    std::string dcoreConfig = "";
    if (chipType == e_mme_Gaudi3)
    {
        dcoreConfig += ", Dcore Config=N/A";  // TODO: get info
    }

    // Recipe Info
    std::string recipeInfo = getRecipeInfo(descGenerator, params);

    // Recurring Misalignment Info
    std::string recurringMisalignmentInfo = getRecurringMisalignmentInfo(descGenerator, params, chipType);

    return geometryInfo + dcoreConfig + recipeInfo + recurringMisalignmentInfo;
}
