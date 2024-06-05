#include "mme_test_manager.h"
#include "device_handler.h"
#include "include/mme_common/mme_common_enum.h"
#include "src/mme_common/common_geo_attr.h"
#include "src/mme_common/mme_hal_factory.h"
#include "include/mme_common/mme_descriptor_generator_base.h"
#include "src/mme_common/mme_geo_factory.h"
#include "print_utils.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include "tensor_comparator.h"
#include "mme_coralhbw_sniffer.h"
#include "mme_corallbw_sniffer.h"
#include "mme_reference.h"
#include "mme_user.h"
#include "json.hpp"
#include "mme_verification/common/mme_reg_write_cmd.h"
#include "sim_tensor_base.h"
#include "common/utils.h"
#include "include/utils/mme_global_conf_manager.h"
#include <random>
#include <stddef.h>
#include <vector>

namespace MmeCommon
{
typedef std::vector<std::unique_ptr<TestResources>> GroupType;

MmeTestManager::MmeTestManager(const ChipType chipType) : m_chipType(chipType), m_mmeHal(getMmeHal(chipType))
{
    // Initialize global configuration
    MmeGlobalConfManager::instance().init("");
}

bool MmeTestManager::verifyTestParams(nlohmann::json& testJson)
{
    // Valid reduction requires that repeats config param must be 1
    if (testJson["repeats"].get<unsigned>() != 1)
    {
        EMmeReductionOp reductionOp = testJson["reductionOp"].get<EMmeReductionOp>();
        switch (reductionOp)
        {
            case EMmeReductionOp::e_mme_reduction_add:
            case EMmeReductionOp::e_mme_reduction_sub:
            case EMmeReductionOp::e_mme_reduction_min:
            case EMmeReductionOp::e_mme_reduction_max:
            case EMmeReductionOp::e_mme_reduction_max_0:
                atomicColoredPrint(COLOR_YELLOW,
                                   "WARNING: param verification failed. [reduction cannot be valid "
                                   "when test repeats is set to more than 1]\n");
                return false;
            case EMmeReductionOp::e_mme_reduction_none:
                return true;
            default:
                MME_ASSERT(0, "Invalid reduction Op value");
        }
    }

    // If the output data type is fp8_143 or fp8_152, then reduction cannot be valid
    bool isOutputTypeFp8 = (testJson["outTypeFloat"].get<EMmeDataType>() == EMmeDataType::e_type_fp8_143) ||
                           (testJson["outTypeFloat"].get<EMmeDataType>() == EMmeDataType::e_type_fp8_152);
    if (isOutputTypeFp8 && (testJson["reductionOp"].get<EMmeReductionOp>() != EMmeReductionOp::e_mme_reduction_none))
    {
        atomicColoredPrint(COLOR_YELLOW,
                           "WARNING: param verification failed. [Reduction is not supported "
                           "when output tensor is of type fp8_143 or fp8_152]\n");
        return false;
    }

    if (testJson["conversionRoundingMode"].get<RoundingMode>() == RoundingMode::StochasticRoundingAndNearest &&
        !isOutputTypeFp8)
    {
        atomicColoredPrint(COLOR_YELLOW,
                           "WARNING: param verification failed. [RSN rounding mode is only "
                           "supported in FP8 output data type]\n");
        return false;
    }

    const unsigned gemmNum = testJson["dualGemm"].get<bool>() ? 2 : 1;
    for (unsigned gemm = 0; gemm < gemmNum; gemm++)
    {
        const std::string gemmSuffix = (gemm == 0) ? "" : "2";

        //Verify sizes.
        const std::vector<int> xSizes = (gemm == 0) ? testJson["xSizes"].get<std::vector<int>>() :
            testJson["xSizes2"].get<std::vector<int>>();
        const std::vector<int> wSizes = (gemm == 0) ? testJson["wSizes"].get<std::vector<int>>() :
            testJson["wSizes2"].get<std::vector<int>>();
        const std::vector<int> ySizes = (gemm == 0) ? testJson["ySizes"].get<std::vector<int>>() :
            testJson["ySizes2"].get<std::vector<int>>();
        for (int i = 0; i < MAX_DIMENSION; ++i)
        {
            if (xSizes[i] < 0)
            {
                atomicColoredPrint(COLOR_YELLOW, "WARNING: param verification failed. [negative offset for X%s tensor]\n",
                    gemmSuffix.c_str());
                return false;
            }
            if (wSizes[i] < 0)
            {
                atomicColoredPrint(COLOR_YELLOW, "WARNING: param verification failed. [negative offset for W%s tensor]\n",
                    gemmSuffix.c_str());
                return false;
            }
            if (ySizes[i] < 0)
            {
                atomicColoredPrint(COLOR_YELLOW, "WARNING: param verification failed. [negative offset for Y%s tensor]\n",
                    gemmSuffix.c_str());
                return false;
            }
        }

        //Verify strides.
        const std::vector<int> xStrides = (gemm == 0) ? testJson["xStrides"].get<std::vector<int>>() :
            testJson["xStrides2"].get<std::vector<int>>();
        std::string errorMessage;
        if (!verifyTensorStrides(xSizes, xStrides, errorMessage))
        {
            atomicColoredPrint(COLOR_YELLOW, "WARNING: param verification failed. [invalid strides for X%s tensor: %s]\n",
                gemmSuffix.c_str(), errorMessage.c_str());
            return false;
        }
        const std::vector<int> wStrides = (gemm == 0) ? testJson["wStrides"].get<std::vector<int>>() :
            testJson["wStrides2"].get<std::vector<int>>();
        if (!verifyTensorStrides(wSizes, wStrides, errorMessage))
        {
            atomicColoredPrint(COLOR_YELLOW, "WARNING: param verification failed. [invalid strides for W%s tensor: %s]\n",
                gemmSuffix.c_str(), errorMessage.c_str());
            return false;
        }
        const std::vector<int> yStrides = (gemm == 0) ? testJson["yStrides"].get<std::vector<int>>() :
            testJson["yStrides2"].get<std::vector<int>>();
        if (!verifyTensorStrides(ySizes, yStrides, errorMessage))
        {
            atomicColoredPrint(COLOR_YELLOW, "WARNING: param verification failed. [invalid strides for Y%s tensor: %s]\n",
                gemmSuffix.c_str(), errorMessage.c_str());
            return false;
        }
    }

    return verifyChipSpecificTestParams(testJson);
}

bool MmeTestManager::verifyTensorStrides(const std::vector<int>& sizes, const std::vector<int>& strides,
    std::string& errorMessage)
{
    if (strides[0] == 0)
    {
        for (unsigned i = 1; i < strides.size(); ++i)
        {
            if (strides[i] != 0)
            {
                errorMessage = "part of strides (but not all) are zero";
                return false;
            }
        }
        return true;
    }

    if (strides[0] != 1)
    {
        errorMessage = "strides[0] != 1";
        return false;
    }

    for (unsigned i = 1; i < strides.size(); ++i)
    {
        if (strides[i] < strides[i - 1] * sizes[i - 1])
        {
            errorMessage = "strides[" + std::to_string(i) + "] < strides[" + std::to_string(i - 1) +
                "] * sizes[" + std::to_string(i - 1) +  "]";
            return false;
        }
    }

    return true;
}

void MmeTestManager::createLfsrValues(uint32_t seed,
                                      LfsrData& lfsrData,
                                      uint32_t euNr,
                                      uint32_t lfsrNumRegs,
                                      bool duplicateForAllCores,
                                      bool configLfsr)
{
    // Init lfsrRegs
    lfsrData.duplicateLfsrValuesForAllCores = duplicateForAllCores;
    uint32_t lfsrBaseVal = configLfsr ? seed : 0xffffffff;
    uint32_t s = lfsrBaseVal;
    lfsrData.lfsrPolynomial.resize(euNr);
    lfsrData.lfsrRegs.resize(euNr);
    for (unsigned core = 0; core < euNr; core++)
    {
        // If we need to duplicate lfsr values for all cores, set s to seed in every
        // iteration Otherwise, set it to seed only in the first iteration
        s = duplicateForAllCores ? lfsrBaseVal : ((core == 0) ? lfsrBaseVal : s);

        lfsrData.lfsrRegs[core].resize(lfsrNumRegs);
        for (unsigned lfsr = 0; lfsr < lfsrNumRegs; lfsr++)
        {
            if (configLfsr) s++;
            lfsrData.lfsrRegs[core][lfsr] = s;  // All cores are set to the same lfsr values
        }
    }

    // Init polynoms
    std::mt19937 mt(seed);
    std::uniform_int_distribution<uint32_t> dist(1, 0xffffffff);

    uint32_t val = (uint32_t) dist(mt);
    for (unsigned core = 0; core < euNr; core++)
    {
        // If we need to duplicate, set all polynomials to the same random number
        // Otherwise, set to a new random number for each core

        lfsrData.lfsrPolynomial[core] = duplicateForAllCores ? val : (uint32_t) dist(mt);
    }
}

// merge the cmds of all tests in group and add configurations to create the
// final program
void MmeTestManager::generateProgram(MmeUser& mmeUser,
                                     const GroupType& group,
                                     SyncObjectManager& soMgr,
                                     std::list<CPProgram>& progs,
                                     std::list<CPProgram>& powerProgs,
                                     bool& firstValidTest,
                                     const LfsrData& lfsrData,
                                     unsigned seed,
                                     unsigned stream,
                                     PmuConfig pmuCfgMode)
{
    mmeUser.initProgram(progs, powerProgs, stream);

    // start building program
    unsigned testIdInGroup = 0;
    for (auto& test : group)
    {
        auto& testJson = *test->testJson;
        auto& dataParams = test->testDataParams;
        auto& cmds = test->cmds;

        mmeUser.generateSingleTestProgram(progs,
                                          powerProgs,
                                          cmds.data(),
                                          soMgr,
                                          dataParams,
                                          firstValidTest,
                                          testJson["clipInfIn"].get<bool>(),
                                          testJson["configLfsr"].get<bool>(),
                                          lfsrData,
                                          seed,
                                          stream,
                                          group.size(),
                                          testIdInGroup,
                                          pmuCfgMode);

        testIdInGroup++;

        firstValidTest = false;
    }
}

void MmeTestManager::gatherDmaBuffers(const std::vector<MmePciDma>& dmaVec,
                                      std::list<Buffer>* inputBuffers,
                                      std::list<Buffer>* outputBuffers)
{
    for (const auto& dmaTran : dmaVec)
    {
        if (dmaTran.size == 0)
            continue;

        Buffer buff;
        buff.size = dmaTran.size;
        buff.hostAddr = dmaTran.hostAddr;
        buff.deviceAddr = dmaTran.deviceAddr;
        buff.isMapped = false;
        if (dmaTran.host2device)
        {
            inputBuffers->push_back(buff);
        }
        else
        {
            outputBuffers->push_back(buff);
            inputBuffers->push_back(buff);
        }
    }
}

uint64_t MmeTestManager::calcProgAddr(const std::list<CPProgram>& progs,
                                      MmeMemoryUsage& groupMemUsage,
                                      const bool programInSram,
                                      const MmeTensorMemoryAttrib& memAttrib)
{
    uint64_t programAddr;
    uint64_t programSize = 0;
    for (auto& prog : progs)
    {
        programSize += prog.getSize();
    }

    if (programInSram)
    {
        programAddr = memAttrib.sramBase;
        programAddr += groupMemUsage.sramUsage;
        if ((programAddr + programSize) < (memAttrib.sramBase + memAttrib.sramSize)) return programAddr;
        atomicColoredPrint(COLOR_YELLOW,
                           "[MME USER] Warning - Program doesnt fit in SRAM - trying to allocate in HBM\n");
    }

    programAddr = memAttrib.hbmBase;
    programAddr += groupMemUsage.hbmUsage;
    MME_ASSERT((programAddr + programSize) < (memAttrib.hbmBase + memAttrib.hbmSize),
               "program address is larger then HBM max address");

    return programAddr;
}

bool MmeTestManager::usesLfsr(const nlohmann::json& testJson)
{
    bool stochasticOperation = false;
    RoundingMode conversionRoundingMode = testJson["conversionRoundingMode"].get<RoundingMode>();
    if (conversionRoundingMode == StochasticRounding || conversionRoundingMode == StochasticRoundingAndNearest)
    {
        stochasticOperation = true;
    }

    if (testJson["stochasticFlush"].get<bool>())
    {
        stochasticOperation = true;
    }

    return stochasticOperation;
}

EMmePattern MmeTestManager::getMmeStackPattern(const nlohmann::json& testJson, const EMmeOpType op)
{
    const EMmePattern convPattern = testJson["convPattern"].get<EMmePattern>();
    const EMmePattern dedwPattern = testJson["dedwPattern"].get<EMmePattern>();
    switch (op)
    {
        case EMmeOpType::e_mme_fwd:
        case EMmeOpType::e_mme_dedx:
        case EMmeOpType::e_mme_transposed_dedx:
            return convPattern;
        case EMmeOpType::e_mme_dedw:
        case EMmeOpType::e_mme_ab:
        case EMmeOpType::e_mme_atb:
        case EMmeOpType::e_mme_abt:
        case EMmeOpType::e_mme_atbt:
        case EMmeOpType::e_mme_reductionAdd:
            return dedwPattern;
        case EMmeOpType::e_mme_memcpy:
        case EMmeOpType::e_mme_trans:
        case EMmeOpType::e_mme_gemm_transpose:
            //  in dma mode pattern is not configurable - always use fck
            return e_mme_sp_reduction_fck;
        default:
            MME_ASSERT(0, "should not get here");
            break;
    }
    return EMmePattern::e_mme_z_reduction_skf;
}

void MmeTestManager::initConvParamsFromJson(const nlohmann::json& testJson, ConvolutionParams* convParams)
{
    convParams->dim = convParams->maxConvDim;
    convParams->paddingValue.int32 = 0;
    auto padding = testJson["padding"].get<std::vector<int>>();
    memcpy(&convParams->padding[0], &padding[0], convParams->maxConvDim * sizeof(convParams->padding[0]));
    convParams->padding[convParams->maxConvDim] = 0;
    auto convStrides = testJson["strides"].get<std::vector<int>>();
    memcpy(&convParams->convStride[0], &convStrides[0], convParams->maxConvDim * sizeof(convParams->convStride[0]));
    convParams->convStride[convParams->maxConvDim] = 1;
    auto dilation = testJson["dilation"].get<std::vector<int>>();
    memcpy(&convParams->dilation[0], &dilation[0], convParams->maxConvDim * sizeof(convParams->dilation[0]));
    convParams->dilation[convParams->maxConvDim] = 1;
}

void MmeTestManager::initStrategyFromJson(const nlohmann::json& testJson, unsigned mmeLimit, MmeStrategy& strategy)
{
    auto op = testJson["operation"].get<EMmeOpType>();

    strategy.mmeLimit = mmeLimit;
    strategy.geometry = testJson["geometry"].get<MmeCommon::EMmeGeometry>();
    strategy.pattern = getMmeStackPattern(testJson, op);
    strategy.pipelineLevel = testJson["pipelineLevel"].get<unsigned>();
    strategy.packingFactor = testJson["packingFactor"].get<unsigned>();
    strategy.reductionLevel = testJson["reductionLevel"].get<unsigned>();
    strategy.isDeterministic = testJson["isDeterministic"].get<bool>();
    strategy.loweringEn = testJson["lowerEn"].get<bool>();
    strategy.sbReuse = testJson["sbReuse"].get<bool>();
    strategy.partialsToMemoryEn = testJson["partialsToMemoryEn"].get<bool>();
    strategy.maskedBgemm = testJson["maskedBgemm"].get<bool>();
    strategy.teAccelerationEn = testJson["teAccel"].get<bool>();
    strategy.alignedAddresses = testJson["alignedAddresses"].get<bool>();
    strategy.unrollEn = testJson["unrollEn"].get<bool>();
    strategy.recurringMisalignmentOptEn = testJson["recurringMisalignmentOptEn"].get<bool>();
    strategy.dualGemm = testJson["dualGemm"].get<bool>();
    strategy.memsetDedxVoidPixels = testJson["memsetVoidPixels"].get<bool>();

    strategy.cdConcurrencyEn = testJson["dedwCDConcurrency"].get<BoolWithUndef>();
    strategy.batchConcurrencyEn = testJson["dedw2x"].get<BoolWithUndef>();
}

unsigned calcPmuSatVal(PmuConfig pmuConfig)
{
    switch (pmuConfig)
    {
        case PMUCFGMODE1:
            return 0xFFFF;
        case PMUCFGMODE2:
            return 1024;
        case PMUCFGMODE3:
            return 512;
        case PMUCFGMODE4:
            return 256;
        case PMUCFGNONE:
            return 0;
        default:
            MME_ASSERT(0, "invalid PMU config");
    }
    return -1U;
}

void MmeTestManager::initControlsFromJson(const nlohmann::json& testJson, MmeControls& controls)
{
    controls.conversionRoundingMode = testJson["conversionRoundingMode"].get<RoundingMode>();
    controls.accRoundingMode = testJson["accRoundingMode"].get<RoundingMode>();
    controls.clippingEn = testJson["clippingEn"].get<bool>() || usesLfsr(testJson);
    controls.clipInfIn = testJson["clipInfIn"].get<bool>();
    controls.infNanModeA = testJson["infNanModeA"].get<InfNanMode>();
    controls.infNanModeB = testJson["infNanModeB"].get<InfNanMode>();
    controls.infNanModeOut = testJson["infNanModeOut"].get<InfNanMode>();
    controls.flushDenormals = testJson["flushDenormals"].get<bool>();
    controls.stochasticFlush = testJson["stochasticFlush"].get<bool>();
    controls.sbCacheEn = testJson["sbCacheEn"].get<bool>();
    controls.reluEn = testJson["reluEn"].get<bool>();
    controls.squashIORois = false;
    controls.slaveSignaling = testJson["slaveSignaling"].get<bool>();
    controls.useSameColorSet = testJson["useSameColorSet"].get<bool>();
    controls.signalingMode = testJson["signalMode"].get<EMmeSignalingMode>();
    controls.pmuSaturationVal = calcPmuSatVal(testJson["pmuConfig"].get<PmuConfig>());
    controls.sbSizeInCLs = testJson["sbSizeInCLs"].get<int>();
    if (controls.signalingMode == e_mme_signaling_once)
    {
        //  when signaling once we have to squash all the ROI into a single one.
        controls.squashIORois = true;
    }
    // set fp8 bias
    controls.fp8BiasIn = DataGenerator::getFpBias(testJson["inTypeFloat"].get<EMmeDataType>(),
                                                  testJson["fp8BiasIn"].get<unsigned>(),
                                                  m_chipType);
    controls.fp8BiasIn2 = DataGenerator::getFpBias(testJson["in2TypeFloat"].get<EMmeDataType>(),
                                                   testJson["fp8BiasIn2"].get<unsigned>(),
                                                   m_chipType);
    controls.fp8BiasOut = DataGenerator::getFpBias(testJson["outTypeFloat"].get<EMmeDataType>(),
                                                   testJson["fp8BiasOut"].get<unsigned>(),
                                                   m_chipType);
}

void MmeTestManager::initTestParamsFromJson(const nlohmann::json& testJson, MmeTestParams* testParams)
{
    testParams->prefetchA = testJson["prefetchOperand"].get<EMmePrefetch>() == EMmePrefetch::e_mme_prefetch_A;
    testParams->prefetchB = testJson["prefetchOperand"].get<EMmePrefetch>() == EMmePrefetch::e_mme_prefetch_B;
    testParams->fullDesc = testJson["fullDesc"].get<bool>();
    testParams->incDec = testJson["incDec"].get<bool>();
    testParams->maskSignals = testJson["loop"].get<bool>();
    testParams->randomMD = false;
    testParams->repeats = testJson["repeats"].get<unsigned>();
    testParams->wkldIdMD = false;
    testParams->traceMode = testJson["traceMode"].get<EMmeTraceMode>();

    testParams->recipeTest = testJson.contains("recipeTest");
    testParams->optimizationTest = testJson.contains("optimizationTest");
    testParams->testOutputTensor = !testJson["skipRun"].get<bool>();

    // Power Management
    testParams->powerTest = testJson["powerTest"].get<bool>();
    testParams->powerLoops = testJson["powerLoops"].get<unsigned>();
    testParams->powerIdleCycles = testJson["powerIdleCycles"].get<unsigned>();
    testParams->powerIdleLoops = testJson["powerIdleLoops"].get<unsigned>();

    // Unit testing
    testParams->useBrain = testJson["useBrain"].get<bool>();
}

void MmeTestManager::initMemoryConfigFromJson(const nlohmann::json& testJson, MmeMemoryConfig& memoryConfig)
{
    memoryConfig.cacheDirective[e_mme_op_a] = testJson["cacheDirectiveA"].get<EMmeCacheDirective>();
    memoryConfig.cacheDirective[e_mme_op_b] = testJson["cacheDirectiveB"].get<EMmeCacheDirective>();
    memoryConfig.cacheDirective[e_mme_op_c] = testJson["cacheDirectiveOut"].get<EMmeCacheDirective>();
    memoryConfig.clss[e_mme_op_a] = testJson["cacheClassA"].get<EMmeCacheClass>();
    memoryConfig.clss[e_mme_op_b] = testJson["cacheClassB"].get<EMmeCacheClass>();
    memoryConfig.clss[e_mme_op_c] = testJson["cacheClassOut"].get<EMmeCacheClass>();
    memoryConfig.reductionOp = testJson["reductionOp"].get<EMmeReductionOp>();
    memoryConfig.reductionRm = testJson["reductionRm"].get<EMmeReductionRm>();
}

static unsigned getCDSize(const EMmeOpType op, const MmeTensorParams& tensorParams)
{
    const SizeArray& wSizes = tensorParams.wHost->getSizes();
    unsigned filterSizes = multiplyElements(std::next(std::next(wSizes.begin())), wSizes.end());
    switch (op)
    {
        case EMmeOpType::e_mme_ab:
        case EMmeOpType::e_mme_abt:
        case EMmeOpType::e_mme_reductionAdd:
            return tensorParams.xHost->getSize(0);
            break;
        case EMmeOpType::e_mme_atb:
        case EMmeOpType::e_mme_atbt:
            return tensorParams.xHost->getSize(1);
            break;
        case EMmeOpType::e_mme_fwd:
            return tensorParams.xHost->getSize(0) * filterSizes;
        case EMmeOpType::e_mme_dedx:
        case EMmeOpType::e_mme_transposed_dedx:
            return tensorParams.yHost->getSize(0) * filterSizes;
            break;
        case EMmeOpType::e_mme_dedw:
        {
            const SizeArray& ySizes = tensorParams.yHost->getSizes();
            return multiplyElements(std::next(ySizes.begin()), ySizes.end());
            break;
        }
        case EMmeOpType::e_mme_gemm_transpose:
        case EMmeOpType::e_mme_memcpy:
        case EMmeOpType::e_mme_trans:
            //  no common dim in dma operations
            return 0;
        default:
            MME_ASSERT(0, "invalid op");
    }
    return 0;
}

static void dumpTensor(const pMMESimTensor& t, std::string fName)
{
    FILE* pFile;
    pFile = fopen(fName.c_str(), "wb");
    if (pFile == nullptr)
    {
        atomicColoredPrint(COLOR_RED, "INFO: Failed opening %s for tensor dump\n", fName.c_str());
        return;
    }
    fwrite(t.get()->data(), sizeof(char), t.get()->getMemorySize(), pFile);
    fclose(pFile);
}
void MmeTestManager::dumpTensors(const MmeDataParams& testDataParams, EMmeOpType op, bool runOnSim, bool runOnChip)
{
    MME_ASSERT(testDataParams.tensorParams.size() == 1,
               "Dump tensor data is currently supported for single set of tensors");

    const MmeTensorParams& tensorParams = testDataParams.tensorParams[0];

    bool isBgemm = (op == EMmeOpType::e_mme_ab || op == EMmeOpType::e_mme_atb || op == EMmeOpType::e_mme_abt ||
                    op == EMmeOpType::e_mme_atbt || op == EMmeOpType::e_mme_reductionAdd || op == e_mme_gemm_transpose);
    bool firstDeviceIsChip = !runOnSim && runOnChip;

    const pMMESimTensor& in0Tenspr = (op == EMmeOpType::e_mme_fwd || op == EMmeOpType::e_mme_dedw || isBgemm)
                                         ? tensorParams.xHost
                                         : tensorParams.wHost;
    const pMMESimTensor& in1Tenspr = (op == EMmeOpType::e_mme_fwd || op == EMmeOpType::e_mme_dedx ||
                                      op == EMmeOpType::e_mme_transposed_dedx || isBgemm)
                                         ? tensorParams.wHost
                                         : tensorParams.yHost;
    const pMMESimTensor& outRefTensor = tensorParams.outRef;
    const pMMESimTensor& outTensor = firstDeviceIsChip ? tensorParams.outDevChip0[0] : tensorParams.outDevSim0;

    dumpTensor(in0Tenspr, "in0_tensor.bin");
    dumpTensor(in1Tenspr, "in1_Tensor.bin");
    dumpTensor(outRefTensor, "out_ref_tensor.bin");
    dumpTensor(outTensor, firstDeviceIsChip ? "out_chip_tensor.bin" : "out_sim_tensor.bin");
}

bool MmeTestManager::doTensorComparison(const MmeDataParams& testDataParams,
                                        const unsigned testCounter,
                                        bool runOnSim,
                                        bool runOnChip,
                                        bool runOnRef,
                                        nlohmann::json& testJson,
                                        const std::string& testInfoStr)
{
    bool equal = true;
    bool compared = false;
    const bool extRefTest = testJson["extRefTest"].get<bool>();
    bool firstDeviceIsChip = !runOnSim && runOnChip;

    if (extRefTest)
    {
        MME_ASSERT(testDataParams.tensorParams.size() == 1, "external reference not supported yet in dual gemm mode");
        FILE* pFile;
        std::string fileName = "mme_output.bin";
        pFile = fopen(fileName.c_str(), "w");
        if (pFile == nullptr)
        {
            atomicColoredPrint(COLOR_RED,
                               "INFO: Failed opening %s for tensor dump for external reference\n",
                               fileName.c_str());
            return false;
        }
        pMMESimTensor firstDevTensor = firstDeviceIsChip ? testDataParams.tensorParams.front().outDevChip0[0]
                                                         : testDataParams.tensorParams.front().outDevSim0;
        fwrite(firstDevTensor.get()->data(), sizeof(char), firstDevTensor.get()->getMemorySize(), pFile);
        fclose(pFile);
        if (equal)
            atomicColoredPrint(COLOR_GREEN,
                               "INFO: Success. Output Dumped, Comparision Skipped (test #%u)\n",
                               testCounter);
        return equal;
    }

    const bool maskBgemm = testJson["maskedBgemm"].get<bool>();
    bool compareBitExact = true;
    const bool sbReuse = testJson["sbReuse"].get<bool>();
    const EMmeOpType op = testJson["operation"].get<EMmeOpType>();
    const bool isConv = (op == EMmeOpType::e_mme_fwd || op == EMmeOpType::e_mme_dedx ||
                         op == EMmeOpType::e_mme_transposed_dedx || op == EMmeOpType::e_mme_dedw);

    if (sbReuse || isConv || usesLfsr(testJson) || testJson["compareDCDCresults"].get<bool>())
    {
        //  convoution and partials create a different order MAC orders
        //  currently stochastic roundings dont produce bit accurate results
        compareBitExact = false;
    }
    const bool compareToRef = runOnRef && !testJson["skipRef"].get<bool>();
    const bool enableSecondaryOutput = testJson["secondaryOutput"].get<bool>();
    unsigned gemmIdx = 0;
    for (auto& tensorParams : testDataParams.tensorParams)
    {
        TensorComparator comparator(getCDSize(op, tensorParams),
                                    tensorParams.outRef->getElementType(),
                                    tensorParams.outRef->getFpBias(),
                                    tensorParams.outRef->getInfNanMode(),
                                    testJson["printAllDiffs"].get<bool>());

        std::string firstDevName = firstDeviceIsChip ? "Device[0]" : "Simulator";
        pMMESimTensor firstDevTensor = firstDeviceIsChip ? tensorParams.outDevChip0[0] : tensorParams.outDevSim0;
    
        if (compareToRef)
        {
            compared = true;
            if (compareBitExact)
            {
                atomicColoredPrint(COLOR_CYAN, "INFO: Comparing bit excat to reference \n");
                equal &= comparator.doCompareBitExact(firstDevTensor,
                                                      "first output of " + firstDevName,
                                                      tensorParams.outRef,
                                                      "Reference",
                                                      testCounter,
                                                      testInfoStr);
            }
            else
            {
                unsigned numPartials = testJson["reductionLevel"].get<unsigned>();
                if (numPartials > 1)
                {
                    // Sanity checks
                    bool isDeterministic = testJson["isDeterministic"].get<bool>();
                    MME_ASSERT(isDeterministic, "Cannot check DCDC results when isDeterministic flag is not set");
                    MME_ASSERT(op == EMmeOpType::e_mme_dedw, "Deterministic cd concurrency is currently limited to dedw");
                    auto& auxTensors = tensorParams.auxTensors;
                    MME_ASSERT(auxTensors[CD_SCRATCHPAD].pTensor && auxTensors[CD_REDUCTION].pTensor,
                           "Expect 2 CD aux tensors in deterministic cd concurrency");

                    atomicColoredPrint(COLOR_CYAN, "INFO: Comparing non bit excat cd scratchpad to reference \n");
                    equal &= comparator.doCompare(auxTensors[CD_SCRATCHPAD].pTensor,
                                                  "cd scratchpad output of " + firstDevName,
                                                  tensorParams.outRef,
                                                  "Reference",
                                                  testCounter,
                                                  testInfoStr,
                                                  numPartials,
                                                  true);

                    if (equal)
                    {
                        atomicColoredPrint(COLOR_CYAN, "INFO: Comparing cd reduction to reference (does not fail the test)\n");
                        comparator.doCompare(firstDevTensor,
                                             "cd reduction output of " + firstDevName,
                                             tensorParams.outRef,
                                             "Reference",
                                             testCounter,
                                             testInfoStr,
                                             1,
                                             false);
                    }
                }
                else 
                {
                    atomicColoredPrint(COLOR_CYAN, "INFO: Comparing non bit excat to reference \n");
                    equal &= comparator.doCompare(firstDevTensor,
                                                  "first output of " + firstDevName,
                                                  tensorParams.outRef,
                                                  "Reference",
                                                  testCounter,
                                                  testInfoStr,
                                                  1,
                                                  true);
                
                }
            }
        }

        if (enableSecondaryOutput)
        {
            auto& secondOutputTensor = firstDeviceIsChip ? tensorParams.outDevChip1[0] : tensorParams.outDevSim1;
            equal &= comparator.doCompareBitExact(firstDevTensor,
                                                  "first output of " + firstDevName,
                                                  secondOutputTensor,
                                                  "second output of " + firstDevName,
                                                  testCounter,
                                                  testInfoStr);
        }
        if (runOnChip)
        {
            compared |= runOnSim;
            unsigned totalDevNr = tensorParams.outDevChip0.size();
            for (unsigned devIdx = firstDeviceIsChip ? 1 : 0; devIdx < totalDevNr; devIdx++)
            {
                std::string devBName = "Device[" + std::to_string(devIdx) + "]";
                equal &= comparator.doCompareBitExact(firstDevTensor,
                                                      "first output of " + firstDevName,
                                                      tensorParams.outDevChip0[devIdx],
                                                      "first output of " + devBName,
                                                      testCounter,
                                                      testInfoStr);

                if (enableSecondaryOutput)
                {
                    equal &= comparator.doCompareBitExact(firstDevTensor,
                                                          "first output of " + firstDevName,
                                                          tensorParams.outDevChip1[devIdx],
                                                          "second output of " + devBName,
                                                          testCounter,
                                                          testInfoStr);
                }
            }
        }
        if (equal)
        {
            if (compared)
            {
                if (testDataParams.tensorParams.size() > 1 && !maskBgemm)
                {
                    atomicColoredPrint(COLOR_GREEN, "INFO: Success. (test #%u, gemm #%u)\n", testCounter, gemmIdx);
                }
                else
                {
                    atomicColoredPrint(COLOR_GREEN, "INFO: Success. (test #%u)\n", testCounter);
                }
            }
            else
            {
                atomicColoredPrint(COLOR_YELLOW,
                                   "WARNING: Results weren't compared against golden model. (test #%u)\n",
                                   testCounter);
            }
        }
        gemmIdx++;
        if (maskBgemm) break;  //  masked bgemm accumulates both gemms into the first gemms output tensor.
    }
    return equal;
}

void MmeTestManager::runReference(const ChipType chipType,
                                  const nlohmann::json& testJson,
                                  const ConvolutionParams& convParams,
                                  MmeDataParams& testDataParams,
                                  MmeMemoryConfig memConfig,
                                  unsigned seed,
                                  unsigned numOfThreads,
                                  uint32_t lfsrNumRegs,
                                  uint32_t* lfsr,
                                  uint32_t polynomial)
{
    const auto op = testJson["operation"].get<EMmeOpType>();
    auto rm = testJson["conversionRoundingMode"].get<RoundingMode>();
    const bool maskBgemm = testJson["maskedBgemm"].get<bool>();
    const bool isOutputTypeFp8 = (testJson["outTypeFloat"].get<EMmeDataType>() == EMmeDataType::e_type_fp8_143) ||
                                 (testJson["outTypeFloat"].get<EMmeDataType>() == EMmeDataType::e_type_fp8_152);
    const bool isConv = (op == EMmeOpType::e_mme_fwd || op == EMmeOpType::e_mme_dedx ||
                         op == EMmeOpType::e_mme_transposed_dedx || op == EMmeOpType::e_mme_dedw);
    const bool isBgemm = (op == EMmeOpType::e_mme_ab || op == EMmeOpType::e_mme_atb || op == EMmeOpType::e_mme_abt ||
                          op == EMmeOpType::e_mme_atbt);
    const bool isReductionAdd = op == EMmeOpType::e_mme_reductionAdd;

    const bool isDma =
        (op == EMmeOpType::e_mme_memcpy || op == EMmeOpType::e_mme_trans || op == EMmeOpType::e_mme_gemm_transpose);
    const bool isDualGemm = testJson["dualGemm"].get<bool>();
    const bool clipEn = testJson["clippingEn"].get<bool>() || usesLfsr(testJson);
    const auto reductionOp = memConfig.reductionOp;
    const auto reductionRm = memConfig.reductionRm;

    CPUCalculator calc(chipType, MME_MAX_TENSOR_DIMS, MME_MAX_CONV_DIMS, seed, lfsrNumRegs, lfsr, polynomial);
    calc.limitNumOfThreads(numOfThreads);

    for (auto& tensorParams : testDataParams.tensorParams)
    {
        SizeArray outSizes = tensorParams.outRef->getSizes();
        unsigned outDim = tensorParams.outRef->getDim();
        const SizeArray& outStrides = tensorParams.outRef->getStrides();

        std::shared_ptr<MmeSimTensor> acc = std::make_shared<MmeSimTensor>(outSizes,
                                                                           outDim,
                                                                           EMmeDataType::e_type_fp32,
                                                                           nullptr,
                                                                           0,
                                                                           MmeCommon::e_mme_full_inf_nan,
                                                                           &outStrides);
        std::shared_ptr<MmeSimTensor> mmeOut = std::make_shared<MmeSimTensor>(outSizes,
                                                                              outDim,
                                                                              tensorParams.outRef->getElementType(),
                                                                              nullptr,
                                                                              tensorParams.outRef->getFpBias(),
                                                                              tensorParams.outRef->getInfNanMode(),
                                                                              &outStrides);

        if (isConv)
        {
            calc.doConvolution(*acc,
                               *tensorParams.xHost,
                               *tensorParams.wHost,
                               *tensorParams.yHost,
                               convParams,
                               op,
                               RoundingMode::RoundToNearest,
                               clipEn,
                               testJson["clipInfIn"].get<bool>());
        }
        else if (isBgemm)
        {
            calc.doBatchGemm(*acc,
                             *tensorParams.xHost,
                             *tensorParams.wHost,
                             op,
                             RoundingMode::RoundToNearest,
                             clipEn,
                             testJson["clipInfIn"].get<bool>());
        }
        else if (isReductionAdd)
        {
            MmeSimTensor xTensor = *tensorParams.xHost;
            MmeSimTensor wTensor = *tensorParams.wHost;
            MmeSimTensor outputTensor = *acc;
            unsigned packingFactor = testJson["packingFactor"].get<unsigned>();
            unsigned reductionLevel = testJson["reductionLevel"].get<unsigned>();
            unsigned numOutputElements = outputTensor.getSizeInElements();
            // Reshape tensors to 2d
            wTensor.getSizes() = {numOutputElements / packingFactor, reductionLevel * packingFactor, 1, 1, 1};
            wTensor.setStridesAndGetTensorSize(nullptr, wTensor.getDim());
            outputTensor.getSizes() = {numOutputElements / packingFactor, packingFactor, 1, 1, 1};
            outputTensor.setStridesAndGetTensorSize(nullptr, outputTensor.getDim());

            calc.doBatchGemm(outputTensor, xTensor, wTensor, EMmeOpType::e_mme_ab);
        }
        else if (isDma)
        {
            calc.doDma(*mmeOut, *tensorParams.xHost, op);
        }
        else
        {
            MME_ASSERT(0, "invalid operation for CPU calculator");
        }

        if (maskBgemm)
        {
            // Perform the bgemm on the masked tensors
            std::shared_ptr<MmeSimTensor> maskAcc = std::make_shared<MmeSimTensor>(outSizes,
                                                                                   outDim,
                                                                                   EMmeDataType::e_type_fp32,
                                                                                   nullptr,
                                                                                   0,
                                                                                   MmeCommon::e_mme_full_inf_nan,
                                                                                   &outStrides);
            auto auxTensors = testDataParams.tensorParams[0].auxTensors;
            calc.doBatchGemm(*maskAcc,
                             *auxTensors[MASKED_BGEMM_A].pTensor,
                             *auxTensors[MASKED_BGEMM_B].pTensor,
                             op,
                             RoundingMode::RoundToNearest,
                             clipEn,
                             testJson["clipInfIn"].get<bool>());
            //  use reduction logic to perform the mask-bgemm accumulation
            calc.doMemoryWrite(acc, maskAcc, e_mme_reduction_add, e_mme_reduction_round_half_to_nearest_even);
        }
        if (!isDma)
        {
            calc.doActivation(mmeOut,
                              acc,
                              nullptr,
                              nullptr,
                              testJson["reluEn"].get<bool>(),
                              rm,
                              nullptr,
                              clipEn,
                              testJson["clipInfIn"].get<bool>(),
                              testJson["flushDenormals"].get<bool>(),
                              testJson["stochasticFlush"].get<bool>());
        }
        //  in masked bgemm both gemms are accumulated into the firsts output.
        auto& outRef = maskBgemm ? testDataParams.tensorParams.front().outRef : tensorParams.outRef;
        calc.doMemoryWrite(outRef, mmeOut, reductionOp, reductionRm, clipEn);
    }
}

bool MmeTestManager::createTestActivations(TestResources& testResources,
                                           ConvolutionParams& convParams,
                                           const MmeTensorMemoryAttrib& memAttrib,
                                           unsigned& actNum)
{
    nlohmann::json& testJson = *testResources.testJson;
    testResources.testDataParams.testJson = testResources.testJson;
    auto op = testJson["operation"].get<EMmeOpType>();

    MmeStrategy strategy;
    initStrategyFromJson(testJson, testResources.mmeLimit, strategy);

    MmeControls controls;
    initControlsFromJson(testJson, controls);

    MmeMemoryConfig memoryConfig;
    initMemoryConfigFromJson(testJson, memoryConfig);

    MmeTestParams testParams;
    initTestParamsFromJson(testJson, &testParams);

    m_mmeUser->doTensorAllocation(op,
                            memAttrib,
                            strategy,
                            testResources.testDataParams,
                            testResources.testMemUsage);

    bool status = m_mmeUser->createActivations(op,
                                               convParams,
                                               memAttrib,
                                               strategy,
                                               controls,
                                               memoryConfig,
                                               testResources.testDataParams,
                                               testResources.testMemUsage,
                                               actNum,
                                               testParams,
                                               testResources.testId);
    if (!status)
    {
        return false;
    }

    if (testParams.recipeTest)
    {
        atomicColoredPrint(COLOR_CYAN, "INFO: Running recipe test. (test #%u)\n", testResources.testId);
        std::string errorStr;
        if (!runRecipeTest(testJson["recipeTest"], errorStr))
        {
            atomicColoredPrint(COLOR_RED, "Recipe test failed: %s\n", errorStr.c_str());
            return false;
        }
        atomicColoredPrint(COLOR_GREEN, "INFO: Recipe test passed.\n");
    }
    if (testParams.optimizationTest)
    {
        atomicColoredPrint(COLOR_CYAN, "INFO: Running optimization test. (test #%u)\n", testResources.testId);
        std::string errorStr;
        if (!runOptimizationTest(testJson["optimizationTest"], errorStr))
        {
            atomicColoredPrint(COLOR_RED, "Optimization test failed: %s\n", errorStr.c_str());
            return false;
        }
        atomicColoredPrint(COLOR_GREEN, "INFO: Optimization test passed.\n");
    }

    return true;
}

void MmeTestManager::patchTestCmds(TestResources& testResources, MmeMemoryUsage& groupMemUsage, bool& firstValidTest)
{
    nlohmann::json& testJson = *testResources.testJson;

    // update usage to allocate tensor according to group usage
    uint64_t testSramOffset = testJson["testSramOffset"];
    uint64_t testHbmOffset = testJson["testHbmOffset"];
    if (testSramOffset == 0 && testHbmOffset == 0)
    {
        testJson["testSramOffset"] = groupMemUsage.sramUsage;
        testJson["testHbmOffset"] = groupMemUsage.hbmUsage;
    }
    else
    {
        // we have values, make sure that if we are in the middle of a group these
        // values are consistent
        if (groupMemUsage.sramUsage == 0 && groupMemUsage.hbmUsage == 0)
        {
            // set offset for patching
            groupMemUsage.sramUsage = testJson["testSramOffset"];
            groupMemUsage.hbmUsage = testJson["testHbmOffset"];
        }
        else
        {
            // this assert is to prevent weird behaviour when running reproductions of
            // paritial groups. this code was build in such a manner that using these
            // fields in JSON is only accepted if this is a single test.
            MME_ASSERT(testJson["testSramOffset"] == groupMemUsage.sramUsage, "single test attribute used in a group");
            MME_ASSERT(testJson["testHbmOffset"] == groupMemUsage.hbmUsage, "single test attribute used in a group");
        }
    }

    // patch DMA list
    for (auto& dma : testResources.testMemUsage.dmaList)
    {
        if (dma.isSram)
        {
            dma.deviceAddr += groupMemUsage.sramUsage;
        }
        else
        {
            dma.deviceAddr += groupMemUsage.hbmUsage;
        }
    }

    MmeTestParams testParams;
    initTestParamsFromJson(testJson, &testParams);

    m_mmeUser->patchTensors(testResources.testDataParams, groupMemUsage.sramUsage, groupMemUsage.hbmUsage);
    m_mmeUser->buildCmds(testParams, testResources.cmds.data(), testResources.testDataParams, firstValidTest, testResources.testMemUsage);
}

void MmeTestManager::allocateSyncObjects(TestResources& testResources, unsigned mmeNr)
{
    nlohmann::json& testJson = *testResources.testJson;
    MmeDataParams& testDataParams = testResources.testDataParams;

    m_syncObjectManager->alignSoIdx(testJson, c_so_group_size);  // can be reduced to 2 or 4 (if
    testDataParams.syncObjects.resize(mmeNr);
    testDataParams.soValues.resize(mmeNr);

    for (int mme = 0; mme < mmeNr; mme++)
    {
        // secondary output is enabled)
        testResources.numOfSoIdx++;
        m_syncObjectManager->getSoIdx(testDataParams.syncObjects[mme].Primary);

        if (testJson["slaveSignaling"].get<bool>())
        {
            testResources.numOfSoIdx++;
            m_syncObjectManager->getSoIdx(testDataParams.syncObjects[mme].PrimarySlave);
        }

        if (testJson["secondaryOutput"].get<bool>())
        {
            // TODO the logic below appears to be Gaudi2 specific, perhaps update once sram/cache in better understood.
            auto operation = testJson["operation"].get<EMmeOpType>();
            EMmeOperand outputOperand = getOutputFromOperation(operation);
            if (testDataParams.operandInSram[outputOperand] != testDataParams.operandInSram[e_mme_op_o] &&
                !testJson["useSameColorSet"].get<bool>())
            {
                // we need a different SO only if the outputs use different colors.
                // which happens only if they are stored in different memory regions.
                // otherwise both output will sync on the same SO (color).
                testResources.numOfSoIdx++;
                m_syncObjectManager->getSoIdx(testDataParams.syncObjects[mme].Secondary);

                if (testJson["slaveSignaling"].get<bool>())
                {
                    testResources.numOfSoIdx++;
                    m_syncObjectManager->getSoIdx(testDataParams.syncObjects[mme].SecondarySlave);
                }
            }
        }
    }
}

void clearOutputTensors(const GroupType& group, unsigned gemm = 0)
{
    for (auto& test : group)
    {
        auto& testDataParams = test->testDataParams;
        auto& testJson = *test->testJson;
        std::shared_ptr<MmeSimTensor> firstOutputPtr;
        bool enableSecondaryOutput = testJson["secondaryOutput"].get<bool>();

        EMmeOpType op = testJson["operation"].get<EMmeOpType>();
        switch (op)
        {
            case EMmeOpType::e_mme_memcpy:
            case EMmeOpType::e_mme_trans:
            case EMmeOpType::e_mme_gemm_transpose:
            case EMmeOpType::e_mme_fwd:
            case EMmeOpType::e_mme_ab:
            case EMmeOpType::e_mme_atb:
            case EMmeOpType::e_mme_abt:
            case EMmeOpType::e_mme_atbt:
            case EMmeOpType::e_mme_reductionAdd:
                firstOutputPtr = testDataParams.tensorParams[gemm].yHost;
                break;
            case EMmeOpType::e_mme_dedx:
            case EMmeOpType::e_mme_transposed_dedx:
                firstOutputPtr = testDataParams.tensorParams[gemm].xHost;
                break;
            case EMmeOpType::e_mme_dedw:
                firstOutputPtr = testDataParams.tensorParams[gemm].wHost;
                break;
            default:
                MME_ASSERT(0, "should not get here");
        }

        memset(firstOutputPtr->data(), 0x77, firstOutputPtr->getMemorySize());
        if (enableSecondaryOutput)
        {
            memset(testDataParams.tensorParams[gemm].oHost->data(),
                   0x77,
                   testDataParams.tensorParams[gemm].oHost->getMemorySize());
        }
    }
}
void MmeTestManager::copyOutputTensors(const GroupType& group, bool useSimulatorDev, bool hostToDevice, unsigned devIdx)
{
    for (auto& test : group)
    {
        auto& testDataParams = test->testDataParams;
        auto& testJson = *test->testJson;
        std::shared_ptr<MmeSimTensor> firstOutputPtr;
        bool enableSecondaryOutput = testJson["secondaryOutput"].get<bool>();

        EMmeOpType op = testJson["operation"].get<EMmeOpType>();
        for (auto& tensorParams : testDataParams.tensorParams)
        {
            switch (op)
            {
                case EMmeOpType::e_mme_memcpy:
                case EMmeOpType::e_mme_trans:
                case EMmeOpType::e_mme_gemm_transpose:
                case EMmeOpType::e_mme_fwd:
                case EMmeOpType::e_mme_ab:
                case EMmeOpType::e_mme_atb:
                case EMmeOpType::e_mme_abt:
                case EMmeOpType::e_mme_atbt:
                case EMmeOpType::e_mme_reductionAdd:
                    firstOutputPtr = tensorParams.yHost;
                    break;
                case EMmeOpType::e_mme_dedx:
                case EMmeOpType::e_mme_transposed_dedx:
                    firstOutputPtr = tensorParams.xHost;
                    break;
                case EMmeOpType::e_mme_dedw:
                    firstOutputPtr = tensorParams.wHost;
                    break;
                default:
                    MME_ASSERT(0, "should not get here");
            }

            auto* src = hostToDevice ? firstOutputPtr->data()
                                     : (useSimulatorDev ? tensorParams.outDevSim0->data()
                                                        : tensorParams.outDevChip0[devIdx]->data());
            auto* dst = hostToDevice ? (useSimulatorDev ? tensorParams.outDevSim0->data()
                                                        : tensorParams.outDevChip0[devIdx]->data())
                                     : firstOutputPtr->data();
            memcpy(dst, src, firstOutputPtr->getMemorySize());

            if (enableSecondaryOutput)
            {
                auto* secondOutSrc = hostToDevice ? tensorParams.oHost->data()
                                                  : (useSimulatorDev ? tensorParams.outDevSim1->data()
                                                                     : tensorParams.outDevChip1[devIdx]->data());
                auto* secondOutDst = hostToDevice ? (useSimulatorDev ? tensorParams.outDevSim1->data()
                                                                     : tensorParams.outDevChip1[devIdx]->data())
                                                  : tensorParams.oHost->data();
                memcpy(secondOutDst, secondOutSrc, tensorParams.oHost->getMemorySize());
            }
        }
    }
}

bool MmeTestManager::runRecipeTest(const json& testJson, std::string& errorStr)
{
    MmeRecipe recipe = m_mmeUser->getRecipe();

    if (testJson.contains("sbUtilization"))
    {
        if ((unsigned)(recipe.sbUtilization() * 100) != testJson["sbUtilization"])
        {
            return false;
        }
    }
    if (testJson.contains("reuse"))
    {
        if (recipe.reuseType() != testJson["reuse"])
        {
            return false;
        }
    }
    if (!unitTestValue(testJson, "spSplits", recipe.getSpSubviews().size(), errorStr))
    {
        return false;
    }
    if (!unitTestValue(testJson, "cdSplits", recipe.getNonSpatialSubviews().size(), errorStr))
    {
        return false;
    }
    return true;
}

bool MmeTestManager::unitTestValue(const json& testJson,
                                   const std::string& field,
                                   unsigned actualVal,
                                   std::string& errorStr)
{
    if (testJson.contains(field))
    {
        int expectedVal = testJson[field];

        if (actualVal != (unsigned) (expectedVal))
        {
            errorStr =
                "Actual " + field + " " + std::to_string(actualVal) + ", expected " + std::to_string(expectedVal);
            return false;
        }
    }
    return true;
}

bool MmeTestManager::unitTestArray(const json& testJson,
                                   const std::string& field,
                                   std::vector<unsigned> actualVal,
                                   std::string& errorStr)
{
    if (testJson.contains(field))
    {
        std::vector<unsigned> expectedVal = testJson[field];
        if (actualVal.size() != expectedVal.size())
        {
            errorStr = field + " lists are of different sizes: expected: " + std::to_string(expectedVal.size()) +
                       ", actual: " + std::to_string(actualVal.size());
            return false;
        }
        int listSize = actualVal.size();
        for (int i = 0; i < listSize; i++)
        {
            if (actualVal[i] != expectedVal[i])
            {
                errorStr = "Element " + std::to_string(i) + ": Actual " + field + " " + std::to_string(actualVal[i]) +
                           ", expected " + std::to_string(expectedVal[i]);
                return false;
            }
        }
    }
    return true;
}

// Extract the cut points from all the sub-problems
std::vector<unsigned> getCdCutPointsFromSubProblems(const MmeCommon::ConvSubProblemContainer subProblems,
                                                    const CommonGeoAttr& geoAttr,
                                                    const MmeCommon::MmeHalReader& mmeHal)
{
    std::vector<unsigned> cdCutPoints;
    for (auto singleSubProblem : subProblems)
    {
        const MmeLayerParams params = singleSubProblem.params;
        unsigned cdCutPoint = RecurringMisalignmentOptimization::getCutPointPerSubProblem(params, geoAttr, mmeHal);
        cdCutPoints.push_back(cdCutPoint);
    }
    return cdCutPoints;
}

bool MmeTestManager::runOptimizationTest(const json& testJson, std::string& errorStr)
{
    const MmeRecipe& recipe = m_mmeUser->getRecipe();
    auto params = m_mmeUser->getParams(0);
    const upCommonGeoAttr& geoAttr = MmeCommon::getGeoAttr(m_chipType, params);

    if (!unitTestValue(testJson, "cdConcurrency", geoAttr->getGeometryCdConcurrency(), errorStr))
    {
        return false;
    }
    if (!unitTestValue(testJson, "batchConcurrency", geoAttr->getEffectiveBatchConcurrency(), errorStr))
    {
        return false;
    }
    if (!unitTestValue(testJson, "geometryHeight", geoAttr->getGeometryHeight(), errorStr))
    {
        return false;
    }
    if (!unitTestValue(testJson, "doubleAccums", geoAttr->getDoubleAccumsBit(), errorStr))
    {
        return false;
    }
    if (!unitTestValue(testJson, "spInterleavingDim", geoAttr->getSpInterleavingDim(MmeCommon::e_mme_op_a), errorStr))
    {
        return false;
    }
    if (geoAttr->supportsConcurrency() && !unitTestValue(testJson, "concurrencyDim", geoAttr->getConcurrentDim(), errorStr))
    {
        return false;
    }
    if (!unitTestValue(testJson, "asymmetricPortMode", geoAttr->isAsymPortConfigMode() ? 1 : 0, errorStr))
    {
        return false;
    }

    MmeCommon::ConvSubProblemContainer subProblems(m_mmeHal.getChipType(), params);
    unsigned numSubProblems = subProblems.size();
    std::vector<unsigned> cdCutPoints = getCdCutPointsFromSubProblems(subProblems, *geoAttr, m_mmeHal);
    if (!unitTestValue(testJson, "recurringMisalignmentSubProblems", numSubProblems, errorStr))
    {
        return false;
    }
    if (!unitTestArray(testJson, "recurringMisalignmentCutPoints", cdCutPoints, errorStr))
    {
        return false;
    }

    return true;
}

bool MmeTestManager::runAndCompareGroup(const GroupType& group,
                                        SyncInfo& groupSI,
                                        unsigned groupId,
                                        uint64_t programAddr,
                                        std::list<CPProgram>& progs,
                                        std::list<CPProgram>& powerProgs,
                                        unsigned stream,
                                        CoralMmeHBWSniffer& hbwSniffer,
                                        CoralMmeLBWSniffer& lbwSniffer,
                                        MMEConfigPrinter& printer,
                                        const std::string& dumpDir,
                                        const unsigned seed,
                                        const unsigned mmeLimit)
{
    if (m_powerTest)
    {
        MME_ASSERT(group.size() == 1, "Power test must have a single group.");
    }

    // Build a print string for all tests in group
    std::stringstream groupStr;
    groupStr << "group #" << groupId << " [";
    for (auto i = 0; i < group.size(); i++)
    {
        groupStr << group[i]->testDataParams.wkldId;
        if (i < group.size() - 1)
        {
            groupStr << ", ";
        }
    }
    const auto groupInfoStr = groupStr.str() + "]";
    atomicColoredPrint(COLOR_CYAN, "INFO: Creating commands and program. (%s)\n", groupInfoStr.c_str());

    // build DMA buffers
    MmeTestAllocator allocator;
    Buffer programBuffer;
    std::list<Device::QueueWkld> devicePrograms;
    std::list<RegisterInfo> setupRegs;
    std::list<Buffer> initBuffers;
    std::list<Buffer> inputBuffers;
    std::list<Buffer> outputBuffers;
    for (auto& test : group)
    {
        gatherDmaBuffers(test->testMemUsage.dmaList, &inputBuffers, &outputBuffers);
    }
    if (m_devHandler->isRunOnSim())
    {
        MME_ASSERT(!m_powerTest, "cannot run power test on simulator");
        // copy from simulator output to host
        copyOutputTensors(group, /*useSimulatorDev=*/true, /*hostToDev=*/false);
        atomicColoredPrint(COLOR_CYAN, "INFO: Sending workload to Simulator. (%s)\n", groupInfoStr.c_str());

        // in sim devices there is no driver to config the following.
        auto simProgram = m_mmeUser->createSimulatorProgram(progs, seed);
        createAndExecuteProgram(m_devHandler->getSimDevice(),
                                stream,
                                programAddr,
                                &simProgram,
                                &groupSI,
                                0,  // Arbitration
                                &inputBuffers,
                                &outputBuffers,
                                &initBuffers,
                                &setupRegs,
                                &programBuffer,
                                &devicePrograms,
                                &allocator);

        if (hbwSniffer.isEnabled())
        {
            MME_ASSERT(!dumpDir.empty(), "dump dir should not be empty");
            hbwSniffer.generateDump(dumpDir);
            hbwSniffer.disable();  // dump only the first test
        }

        if (lbwSniffer.isEnabled())
        {
            MME_ASSERT(!dumpDir.empty(), "dump dir should not be empty");
            lbwSniffer.generateDumpFile(dumpDir + "/act_seq.txt");
            lbwSniffer.disable();  // dump only the first test
        }

        // copy from host to simulator output
        copyOutputTensors(group,
                          /*useSimulatorDevice=*/true,
                          /*hostToDevice=*/true);

        atomicColoredPrint(COLOR_CYAN, "INFO: Simulator run completed. (%s)\n", groupInfoStr.c_str());
    }

    // run on Driver devices
    const auto& driverDevices = m_devHandler->getDriverDevices();
    for (unsigned devIdx = 0; devIdx < m_devHandler->getNumOfDriverDevices(); devIdx++)
    {
        atomicColoredPrint(COLOR_CYAN, "INFO: Sending workload to Device[%d]. (%s)\n", devIdx, groupInfoStr.c_str());
        // copy from device output to host
        copyOutputTensors(group, /*useSimulatorDev=*/false, /*hostToDevice=*/false, devIdx);

        if (m_powerTest && !m_scalFw)
        {
            createAndExecuteProgram(driverDevices[devIdx],
                                    stream,
                                    programAddr,
                                    &progs,
                                    &groupSI,
                                    nullptr,  // Arbitration
                                    &inputBuffers,
                                    nullptr);  // outputBuffers
            atomicColoredPrint(COLOR_CYAN, "INFO: Starting power test\n");
            createAndExecuteProgram(driverDevices[devIdx],
                                    stream,
                                    programAddr,
                                    &powerProgs,
                                    &groupSI,
                                    nullptr,  // Arbitration
                                    nullptr,  // inputBuffers,
                                    &outputBuffers);
        }
        else
        {
            createAndExecuteProgram(driverDevices[devIdx],
                                    stream,
                                    programAddr,
                                    &progs,
                                    &groupSI,
                                    nullptr,  // Arbitration
                                    &inputBuffers,
                                    &outputBuffers);
        }

        // copy from host output to device
        copyOutputTensors(group,
                          /*useSimulatorDevice=*/false,
                          /*hostToDevice*/ true,
                          devIdx);

        atomicColoredPrint(COLOR_CYAN, "INFO: Device[%d] run completed. (%s)\n", devIdx, groupInfoStr.c_str());
    }

    bool equalTensors = true;
    for (auto& test : group)
    {
        nlohmann::json& testJson = *test->testJson;
        MmeDataParams& testDataParams = test->testDataParams;
        unsigned testCounter = testDataParams.wkldId;
        equalTensors = doTensorComparison(testDataParams,
                                          testCounter,
                                          m_devHandler->isRunOnSim(),
                                          m_devHandler->isRunOnChip(),
                                          m_devHandler->isRunOnRef(),
                                          testJson,
                                          printer.dump(testJson));
        // Dump input and output tensors
        if (testJson["dumpTensorData"].get<bool>())
        {
            dumpTensors(test->testDataParams,
                        testJson["operation"].get<EMmeOpType>(),
                        m_devHandler->isRunOnSim(),
                        m_devHandler->isRunOnChip());
        }

        if (!equalTensors) break;
    }

    return equalTensors;
}

bool MmeTestManager::shouldAddNullDescToGroup(const bool allowedToAddNullDescriptors,
                                              const bool addNullDescToTest,
                                              const bool nullDescInGroup)
{
    bool shouldAddTest = (rand() % 10) == 0;  // once every 10 tests.

    // add null desc if we should, if we are allowed by Json, and if we haven't
    // already added to group. unless we wre force by Json
    return (shouldAddTest && allowedToAddNullDescriptors && !nullDescInGroup) || addNullDescToTest;
}

bool MmeTestManager::canAddTestToGroup(const GroupType& group,
                                       const unsigned currentGroupId,
                                       MmeMemoryUsage& groupMemUsage,
                                       const TestResources& curTest,
                                       const unsigned testLimit,
                                       const MmeTensorMemoryAttrib& memAttrib,
                                       const bool programInSram,
                                       const bool groupClipInfIn,
                                       const bool testClipInfIn)
{
    const auto& currTestJson = *curTest.testJson;
    const auto testGroupId = currTestJson["groupId"].get<unsigned>();
    bool canAddCurTest = true;

    if (group.size() > testLimit)
    {
        if (currentGroupId == testGroupId)
        {
            atomicColoredPrint(COLOR_YELLOW,
                               "INFO: forcing test into group - test has same group ID as current group.\n");
            canAddCurTest = true;
        }
        else
        {
            atomicColoredPrint(COLOR_YELLOW, "INFO: closing group - exceeded test limit.\n");
            canAddCurTest = false;
        }
    }

    if (groupClipInfIn != testClipInfIn)
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: closing group - current test flips clipInfIn value.\n");
        canAddCurTest = false;
    }

    if (canAddCurTest)
    {
        // make sure new test doesnt use any SoIdx used by the group
        unsigned currFirstSoIdx = currTestJson["firstSoIdx"].get<unsigned>();
        for (auto& test : group)
        {
            const auto& testJson = *test->testJson;
            if (testJson["firstSoIdx"].get<unsigned>() < currFirstSoIdx)
            {
                MME_ASSERT((testJson["firstSoIdx"].get<unsigned>() + test->numOfSoIdx <= currFirstSoIdx),
                           "test has the same poleSoIdx like another test in group");
            }
            else
            {
                MME_ASSERT((currFirstSoIdx + curTest.numOfSoIdx <= testJson["firstSoIdx"].get<unsigned>()),
                           "test has the same poleSoIdx like another test in group");
            }
        }

        // make sure we have enough monitors left
        unsigned monitorIdx = m_syncObjectManager->getCurrMonitorIdx();
        unsigned monitorsPerTest = 5;  // each test can use up to 5 monitors
        if (monitorIdx + monitorsPerTest >= m_mmeHal.getMonitorNr()) return false;
    }

    // Check if new test violates memory capacity
    uint64_t totalSramUsage = groupMemUsage.sramUsage + curTest.testMemUsage.sramUsage;
    uint64_t totalHbmUsage = groupMemUsage.hbmUsage + curTest.testMemUsage.hbmUsage;
    const unsigned estimatedProgramSize = 0x10000;
    // Determine membership current test still fit into hbm/sram
    uint64_t effectiveSramSize = memAttrib.sramSize - (programInSram ? estimatedProgramSize : 0);
    uint64_t effectiveHbmSize = memAttrib.hbmSize - (programInSram ? 0 : estimatedProgramSize);
    if ((totalSramUsage > effectiveSramSize) || (totalHbmUsage > effectiveHbmSize))
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: closing group - current test exceeds mememory capacity.\n");
        canAddCurTest = false;
    }

    return canAddCurTest;
}

bool MmeTestManager::shouldRunGroup(const GroupType& group,
                                    unsigned testLimit,
                                    bool canAddCurTest,
                                    bool currTestForceCloseGroup)
{
    bool canExecuteGroup = false;
    if (canAddCurTest)
    {
        if (group.size() == testLimit || currTestForceCloseGroup)
        {
            canExecuteGroup = true;
        }
    }
    else
    {
        canExecuteGroup = !group.empty();
    }
    return canExecuteGroup;
}

void MmeTestManager::updateGroupMemUsage(MmeMemoryUsage& groupMemUsage, MmeMemoryUsage& testMemUsage)
{
    groupMemUsage.sramUsage += testMemUsage.sramUsage;
    groupMemUsage.hbmUsage += testMemUsage.hbmUsage;
}

// When MMU is disabled, discard what device returned(virtual address) and use what user put in test json.
uint64_t MmeTestManager::getHbmBaseAddress(uint64_t hbmBase)
{
    bool first = true, mmuEnabled = false;
    for (auto& dev : m_devHandler->getDriverDevices())
    {
        // The indication of MMU enabled is given from the device
        if (dev->getMmuEnabledValue())
        {
            if (first) hbmBase = dev->dramMemoryAllocAndMap();
            else if (hbmBase != dev->dramMemoryAllocAndMap())
            {
                MME_ASSERT(0,
                           "device MMU HBM base has to "
                           "be equal for all driver devices");
            }
            mmuEnabled = true;
        }
        first = false;
    }

    return hbmBase;
}

void MmeTestManager::initMemAttrib(const nlohmann::json& testJson, MmeTensorMemoryAttrib& memAttrib)
{
    memAttrib.hbmSize = testJson["hbmSize"].get<uint64_t>();
    memAttrib.hbmBase = getHbmBaseAddress(testJson["hbmBase"].get<uint64_t>());

    if (testJson.contains("sramSize"))
    {
        memAttrib.sramSize = testJson["sramSize"].get<uint64_t>();
        MME_ASSERT(memAttrib.sramSize <= m_mmeHal.getSramSize(m_devHandler->getDieNr()), "invalid Sram size value");
    }
    else
    {
        memAttrib.sramSize = m_mmeHal.getSramSize(m_devHandler->getDieNr());
    }
    if (testJson.contains("sramBase"))
    {
        memAttrib.sramBase = testJson["sramBase"].get<uint64_t>();
    }
    else
    {
        memAttrib.sramBase = m_mmeHal.getSramStart(m_devHandler->getDieNr());
        MME_ASSERT(memAttrib.sramBase >= m_mmeHal.getSramStart(m_devHandler->getDieNr()), "invalid Sram base value");
    }
}

bool MmeTestManager::runTests(std::vector<nlohmann::json>& testsParams,
                              const std::string& dumpDir,
                              const std::string& dumpUnit,
                              const EMmeDump dumpMmes,
                              const unsigned mmeDumpIdx,
                              const std::string& lfsrDir,
                              const DeviceType devTypeA,
                              const DeviceType devTypeB,
                              std::vector<unsigned>& deviceIdxs,
                              const unsigned seed,
                              const unsigned numOfThreads,
                              unsigned mmeLimit,
                              const bool checkRoi,
                              const bool chipAlternative)
{
    const auto& globalTestParams = testsParams.front();
    mmeLimit = mmeLimit == 0 ? m_mmeHal.getMmeNr() : mmeLimit;
    unsigned dieNr = mmeLimit > m_mmeHal.getMmePerDie() ? m_mmeHal.getDieNr() : 1;
    const unsigned stream = 0;
    const uint64_t smBase = globalTestParams["smBase"].get<uint64_t>();
    const uint64_t pqBaseAddr =
        globalTestParams.contains("pqBaseAddr") ? globalTestParams["pqBaseAddr"].get<uint64_t>() : 0;

    makeMmeUser(mmeLimit);
    makeDeviceHandler(devTypeA, devTypeB, deviceIdxs);
    makeSyncObjectManager(smBase, mmeLimit);
    m_syncObjectManager->resetTestGroup(stream);
    m_mmeUser->setSyncObjectManager(m_syncObjectManager.get());
    m_mmeUser->setDoStaticConfig(m_devHandler->isRunOnSim() && !m_devHandler->isRunOnChip());
    m_mmeUser->setscalFw(m_scalFw);

    bool exitStatus = true;
    CoralMmeHBWSniffer hbwSniffer;
    CoralMmeLBWSniffer lbwSniffer;
    srand(seed);

    m_devHandler->setChipAlternative(chipAlternative);
    m_devHandler->createDevices(pqBaseAddr, dieNr);
    if (!dumpDir.empty())
    {
        m_devHandler->configureMeshSniffersAndDumpDir(dumpMmes, mmeDumpIdx, hbwSniffer, lbwSniffer, dumpDir, dumpUnit);
    }

    exitStatus = m_devHandler->openDevices();
    if (!exitStatus) return exitStatus;
    if (m_mmeUser->canDoStaticConfig())
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Initializing LFSRs.\n");
    }
    else
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Skipping Initializing LFSRs - cannot do static config of regs.\n");
    }

    fixCacheModeAlloc(testsParams);

    LfsrData lfsrData;
    // If any of the cores is reference, we want to duplicate the lfsr values for
    // all cores to ensure consistency between the reference and the device
    createLfsrValues(seed,
                     lfsrData,
                     m_mmeHal.getEuNr(),
                     m_mmeHal.getLFSRSeedsNr(),
                     m_devHandler->isRunOnRef(),
                     globalTestParams["configLfsr"].get<bool>());

    MmeTensorMemoryAttrib memAttrib;
    initMemAttrib(globalTestParams, memAttrib);

    const unsigned b2bTestsNumLimit = checkRoi ? 1 : globalTestParams["b2bTestsNumLimit"].get<unsigned>();
    const bool programInSram = globalTestParams["programInSram"].get<unsigned>();
    const bool addNullDescriptors = globalTestParams["addNullDescriptors"].get<bool>();
    const bool nullDescBeforeTest = globalTestParams["nullDescBeforeTest"].get<bool>();
    const unsigned nullDescNum = globalTestParams["nullDescNum"].get<unsigned>();
    bool groupClipInfInConfig = globalTestParams["clipInfIn"].get<bool>();
    MME_ASSERT(b2bTestsNumLimit > 0, "b2b limit should not be 0");
    if (checkRoi)
    {
        atomicColoredPrint(COLOR_CYAN, "INFO: running checkRoi, limiting b2bTestsNumLimit to 1\n");
    }
    const PmuConfig pmuCfgMode = globalTestParams["pmuConfig"].get<PmuConfig>();
    m_powerTest = globalTestParams["powerTest"].get<bool>();
    m_mmeUser->setPowerTest(m_powerTest);

    unsigned testCounter = 0;
    bool firstValidMmeTest = true;
    bool firstValidDmaTest = true;
    bool firstValidGroup = true;
    bool nullDescInGroup = false;
    MMEConfigPrinter printer;
    unsigned groupId = 0;
    GroupType group;
    MmeMemoryUsage groupMemUsage;

    atomicColoredPrint(COLOR_CYAN, "INFO: Number of tests: %u\n", (unsigned) testsParams.size());
    for (nlohmann::json& testJson : testsParams)
    {
        if (testJson["skipTest"])
            continue;
        if (!testJson["skipOnVerificationFailure"] && !verifyTestParams(testJson))
        {
            exitStatus = false;
            atomicColoredPrint(COLOR_RED, "WARNING: Test details:\n%s\n\n", printer.dump(testJson).c_str());
        }
    }
    if (!exitStatus)
    {
        atomicColoredPrint(COLOR_RED, "Param verification failed. Tests were not run.\n");
        return false;
    }

    for (auto& testJson : testsParams)
    {
        // begin new test
        TestResources testResources;
        testResources.mmeLimit = mmeLimit;
        testResources.cmds.resize(mmeLimit);
        testResources.testJson = &testJson;
        testResources.testId = testCounter;

        atomicColoredPrint(COLOR_CYAN, "INFO: Starting test: (test #%u)\n", testCounter);
        atomicColoredPrint(COLOR_MAGENTA, "%s", printer.dump(testJson).c_str());

        if (testJson["skipTest"])
        {
            atomicColoredPrint(COLOR_YELLOW,
                               "WARNING: The test was intentionally skipped by the "
                               "user via 'skipTest' flag.\n");
            atomicColoredPrint(COLOR_YELLOW, "Test details:\n%s\n\n", printer.dump(testJson).c_str());
            testCounter++;
            continue;
        }
        if (testJson["skipOnVerificationFailure"] && !verifyTestParams(testJson))
        {
            atomicColoredPrint(COLOR_YELLOW, "WARNING: The test params verification failed so the test "
                "was skipped via 'skipOnVerificationFailure' flag.\n");
            testCounter++;
            continue;
        }

        const EMmeOpType op = testJson["operation"].get<EMmeOpType>();
        bool isDmaTest = op == e_mme_memcpy || op == e_mme_trans;
        bool& firstValidTest = isDmaTest ? firstValidDmaTest : firstValidMmeTest;
        m_mmeUser->setDmaMode(isDmaTest);

        ConvolutionParams convParams;
        initConvParamsFromJson(testJson, &convParams);

        testJson["seed"] = seed;

        DataGenerator dataGenerator(m_chipType, testCounter, seed);
        if (!dataGenerator.createAndInitDataParams(testJson,
                                                   m_devHandler->isRunOnSim(),
                                                   m_devHandler->getNumOfDriverDevices()))
        {
            atomicColoredPrint(COLOR_RED, "Test details:\n%s\n\n", printer.dump(testJson).c_str());
            exitStatus = false;
            break;
        }

        testResources.testDataParams = dataGenerator.getParams();
        allocateSyncObjects(testResources, mmeLimit);

        if (checkRoi)
        {
            // generalize to support gaudi3 once we bring on its mem checker
            unsigned euNr = m_mmeHal.getEuNr();
            if (!testJson["slaveSignaling"])
            {
                euNr /= 2;
            }
            testResources.testDataParams.memAccessChecker = createAccessChecker(euNr);
            for (unsigned mmeIdx = 0; mmeIdx < m_mmeHal.getEuNr(); mmeIdx++)
            {
                m_devHandler->getSimDevice()->getCluster()->setMmeDebugMemAccess(
                    mmeIdx,
                    (void*) testResources.testDataParams.memAccessChecker.get());
            }
        }

        atomicColoredPrint(COLOR_CYAN, "INFO: Generating Activations. (test #%u)\n", testCounter);
        unsigned actNum = 0;
        if (!createTestActivations(testResources, convParams, memAttrib, actNum))
        {
            atomicColoredPrint(COLOR_RED, "Test details:\n%s\n\n", printer.dump(testJson).c_str());
            exitStatus = false;
            break;
        }
        if (testJson["skipRun"].get<bool>())
        {
            atomicColoredPrint(COLOR_CYAN, "INFO: Skipping output tensors test.\n");
            testCounter++;
            continue;
        }

        if (!actNum)
        {
            atomicColoredPrint(COLOR_YELLOW,
                               "WARNING: output tensors test skipped. [Operation is out of bound "
                               "- no descs were created]\n");
            atomicColoredPrint(COLOR_YELLOW, "Test details:\n%s\n\n", printer.dump(testJson).c_str());
            testCounter++;
            continue;
        }
        atomicColoredPrint(COLOR_CYAN, "INFO: Activations Generated.\n");

        if (m_devHandler->isRunOnRef() && !testJson["skipRef"].get<bool>() )
        {
            atomicColoredPrint(COLOR_CYAN, "INFO: Running reference. (test #%u)\n", testCounter);
            if (usesLfsr(testJson))
            {
                MME_ASSERT(lfsrData.duplicateLfsrValuesForAllCores,
                           "using different lfsr values per core not supported in reference");
            }
            runReference(m_chipType,
                         testJson,
                         convParams,
                         testResources.testDataParams,
                         m_mmeUser->getParams(0).memoryCfg,
                         seed,
                         numOfThreads,
                         m_mmeHal.getLFSRSeedsNr(),
                         lfsrData.lfsrRegs[0].data(),
                         lfsrData.lfsrPolynomial[0]);
            atomicColoredPrint(COLOR_CYAN, "INFO: Reference completed.\n");
        }

        bool addTest = canAddTestToGroup(group,
                                         groupId,
                                         groupMemUsage,
                                         testResources,
                                         b2bTestsNumLimit,
                                         memAttrib,
                                         programInSram,
                                         groupClipInfInConfig,
                                         testJson["clipInfIn"].get<bool>());
        if (addTest)
        {
            atomicColoredPrint(COLOR_CYAN, "INFO: adding test #%u to group #%d\n", testCounter, groupId);

            bool addNullDesc =
                shouldAddNullDescToGroup(addNullDescriptors, testJson["testHasNullDesc"].get<bool>(), nullDescInGroup);
            if (addNullDesc && nullDescBeforeTest)
            {
                atomicColoredPrint(COLOR_CYAN, "INFO: adding %u nullDescs before test #%u\n", nullDescNum, testCounter);
                for (unsigned desc = 0; desc < nullDescNum; desc++)
                {
                    // create commands and patch the SO address
                    testJson["testHasNullDesc"] = true;
                    nullDescInGroup = true;
                    m_mmeUser->createNullDescCmds(testResources.testDataParams, testResources.cmds.data());
                }
            }

            patchTestCmds(testResources, groupMemUsage, firstValidTest);
            updateGroupMemUsage(groupMemUsage, testResources.testMemUsage);

            if (addNullDesc && !nullDescBeforeTest)
            {
                atomicColoredPrint(COLOR_CYAN, "INFO: adding %u nullDescs after test #%u\n", nullDescNum, testCounter);

                for (unsigned desc = 0; desc < nullDescNum; desc++)
                {
                    // create commands and patch the SO address
                    testJson["testHasNullDesc"] = true;
                    nullDescInGroup = true;
                    m_mmeUser->createNullDescCmds(testResources.testDataParams, testResources.cmds.data());
                }
            }

            // add test to group
            group.push_back(std::make_unique<TestResources>(testResources));
            testJson["groupId"] = groupId;
        }

        bool runGroup = shouldRunGroup(group, b2bTestsNumLimit, addTest, testJson["forceCloseGroup"].get<bool>());
        if (runGroup)
        {
            std::list<CPProgram> progs;
            std::list<CPProgram> powerProgs;
            SyncInfo groupSI = m_syncObjectManager->createGroupSyncObj(group.size());
            generateProgram(*m_mmeUser,
                            group,
                            *m_syncObjectManager,
                            progs,
                            powerProgs,
                            firstValidGroup,
                            lfsrData,
                            seed,
                            stream,
                            pmuCfgMode);
            m_syncObjectManager->addGroupMonitor(progs.front(), stream);
            uint64_t programAddr = calcProgAddr(progs, groupMemUsage, programInSram, memAttrib);
            exitStatus = runAndCompareGroup(group,
                                            groupSI,
                                            groupId,
                                            programAddr,
                                            progs,
                                            powerProgs,
                                            stream,
                                            hbwSniffer,
                                            lbwSniffer,
                                            printer,
                                            dumpDir,
                                            seed,
                                            mmeLimit);

            // reset group
            groupId++;
            group.clear();
            m_syncObjectManager->resetTestGroup(stream);
            groupMemUsage.sramUsage = 0;
            groupMemUsage.hbmUsage = 0;
            nullDescInGroup = false;
        }

        if (!exitStatus) break;

        if (!addTest)
        {
            atomicColoredPrint(COLOR_CYAN, "INFO: adding test #%u to group #%d\n", testCounter, groupId);

            // update usage to allocate tensor according to group usage
            patchTestCmds(testResources, groupMemUsage, firstValidTest);
            updateGroupMemUsage(groupMemUsage, testResources.testMemUsage);

            // add test to group
            group.push_back(std::make_unique<TestResources>(testResources));
            testJson["groupId"] = groupId;

            // set the groups clipInfIn config
            groupClipInfInConfig = testJson["clipInfIn"].get<bool>();
        }

        testCounter++;
    }

    if (!group.empty() && exitStatus)
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: running remaining group.\n");
        std::list<CPProgram> progs;
        std::list<CPProgram> powerProgs;
        SyncInfo groupSI = m_syncObjectManager->createGroupSyncObj(group.size());
        generateProgram(*m_mmeUser,
                        group,
                        *m_syncObjectManager,
                        progs,
                        powerProgs,
                        firstValidGroup,
                        lfsrData,
                        seed,
                        stream,
                        pmuCfgMode);

        m_syncObjectManager->addGroupMonitor(progs.front(), stream);
        uint64_t programAddr = calcProgAddr(progs, groupMemUsage, programInSram, memAttrib);
        exitStatus = runAndCompareGroup(group,
                                        groupSI,
                                        groupId,
                                        programAddr,
                                        progs,
                                        powerProgs,
                                        stream,
                                        hbwSniffer,
                                        lbwSniffer,
                                        printer,
                                        dumpDir,
                                        seed,
                                        mmeLimit);
    }

    atomicColoredPrint(COLOR_YELLOW, "INFO: Test done.\n");
    return exitStatus;
}

void MmeTestManager::fixCacheModeAlloc(std::vector<nlohmann::json>& tests)
{
    // need to fix allocation in cache only for gaudi3 and up platforms , and only for driver devices.
    if (m_chipType < e_mme_Gaudi3 || m_devHandler->getNumOfDriverDevices() == 0)
    {
        return;
    }
    // all devices have the same cache mode settings.
    auto* device = m_devHandler->getDriverDevices().front();
    if (device->isCacheMode())
    {
        atomicColoredPrint(COLOR_YELLOW,
                           "INFO: Test Changed - device is initalized to cache mode - "
                           "Allocating all data to HBM.\n");
        // in cache mode we cant put data in sram - its HW managed.
        for (auto& test : tests)
        {
            test["programInSram"] = false;
            test["xInSram"] = false;
            test["wInSram"] = false;
            test["yInSram"] = false;
            test["oInSram"] = false;
        }
    }
}

unsigned MmeTestManager::getMmePerDie() const
{
    return m_mmeHal.getMmePerDie();
}

unsigned MmeTestManager::getDieNr() const
{
    return m_mmeHal.getDieNr();
}

unsigned MmeTestManager::getMmeNr() const
{
    return m_mmeHal.getMmeNr();
}

unsigned MmeTestManager::getEuNr() const
{
    return m_mmeHal.getEuNr();
}

unsigned MmeTestManager::getLFSRSeedsNr() const
{
    return m_mmeHal.getLFSRSeedsNr();
}

uint64_t MmeTestManager::getSramStart(unsigned dieNr) const
{
    return m_mmeHal.getSramStart();
}

uint64_t MmeTestManager::getSramSize(unsigned dieNr) const
{
    return m_mmeHal.getSramSize();
}

}  // namespace MmeCommon
