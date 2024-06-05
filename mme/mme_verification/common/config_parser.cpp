#include <cstdint>
#include <sstream>
#include <list>
#include "config_parser.h"
#include "include/mme_assert.h"
#include "include/general_utils.h"
#include "drm/habanalabs_accel.h"
#include "include/mme_common/mme_common_enum.h"
#include "print_utils.h"
#include "src/mme_common/mme_hal_factory.h"
#include "utils/logger.h"

namespace MmeCommon
{
// save space for program
#define PROG_BUFFER               (1024 * 1024 * 1024)

// random threshold sizes
#define BGEMM_MAX_BATCH_SIZE        4
#define BGEMM_MIN_BATCH_SIZE        1
#define BGEMM_MAX_GEMM_SIZE         512
#define BGEMM_MIN_GEMM_SIZE         1

#define CONV_MAX_SIZE               32
#define CONV_MIN_SIZE               1
#define CONV_MAX_BATCH_SIZE         3
#define CONV_MIN_BATCH_SIZE         1
#define CONV_MAX_KERNEL_SIZE        4
#define CONV_MIN_KERNEL_SIZE        1
#define CONV_MAX_COMMON_DIM_SIZE    64
#define CONV_MIN_COMMON_DIM_SIZE    1
#define CONV_MAX_K_SIZE             512
#define CONV_MIN_K_SIZE             1

#define PADDING_MAX_SIZE            3
#define PADDING_MIN_SIZE            0
#define STRIDES_MAX_SIZE            5
#define STRIDES_MIN_SIZE            1
#define DILATION_MAX_SIZE           5
#define DILATION_MIN_SIZE           1

void MMETestConfigParser::setDefaults(json& config)
{
    config["in2TypeFloat"] = EMmeDataType::e_types_nr;
    config["dualGemm"] = false;
    config["decMode"] = false;
    config["lowerEn"] = false;
    config["unrollEn"] = false;
    config["recurringMisalignmentOptEn"] = false;
    config["dedw2x"] = BoolWithUndef::Undefined;
    config["teAccel"] = false;
    config["sbReuse"] = false;
    config["partialsToMemoryEn"] = false;
    config["maskedBgemm"] = false;
    config["dedwCDConcurrency"] = BoolWithUndef::Undefined;
    config["isDeterministic"] = false;
    config["compareDCDCresults"] = false;
    config["signalMode"] = EMmeSignalingMode::e_mme_signaling_once;
    config["slaveSignaling"] = false;
    config["useSameColorSet"] = false;
    config["pipelineLevel"] = 1;
    config["packingFactor"] = 1;
    config["reductionLevel"] = 1;
    config["traceMode"] = EMmeTraceMode::e_mme_trace_mode_none;
    config["sizeInBytes"] = false;  // doesnt exists
    config["memsetVoidPixels"] = false;
    config["repeats"] = 1;
    config["convPattern"] = EMmePattern::e_mme_patterns_nr;
    config["dedwPattern"] = EMmePattern::e_mme_patterns_nr;
    config["geometry"] = EMmeGeometry::e_mme_geometry_nr;
    config["conversionRoundingMode"] = RoundingMode::RoundToNearest;
    config["accRoundingMode"] = RoundingMode::RoundToNearest;
    config["clippingEn"] = false;
    config["clipInfIn"] = false;
    config["infNanModeA"] = InfNanMode::e_mme_full_inf_nan;
    config["infNanModeB"] = InfNanMode::e_mme_full_inf_nan;
    config["infNanModeOut"] = InfNanMode::e_mme_full_inf_nan;
    config["flushDenormals"] = false;
    config["stochasticFlush"] = false;
    config["sbCacheEn"] = true;
    config["xInSram"] = true;
    config["yInSram"] = true;
    config["wInSram"] = true;
    config["oInSram"] = true;
    config["secondaryOutput"] = false;
    config["alignedAddresses"] = true;
    config["memsetOutput"] = false;
    config["reductionOp"] = EMmeReductionOp::e_mme_reduction_none;
    config["reductionRm"] = EMmeReductionRm::e_mme_reduction_round_nr;
    config["cacheDirectiveA"] = EMmeCacheDirective::HomeAllocate;
    config["cacheDirectiveB"] = EMmeCacheDirective::HomeAllocate;
    config["cacheDirectiveOut"] = EMmeCacheDirective::HomeAllocate;
    config["cacheClassA"] = EMmeCacheClass::Normal;
    config["cacheClassB"] = EMmeCacheClass::Normal;
    config["cacheClassOut"] = EMmeCacheClass::Normal;
    config["incDec"] = false;
    config["loop"] = false;
    config["prefetchOperand"] = MmeCommon::EMmePrefetch::e_mme_prefetch_none;
    config["fullDesc"] = false;
    config["testHasNullDesc"] = false;
    config["dualNullDesc"] = false;
    config["skipRef"] = false;
    config["skipRun"] = false;
    config["printAllDiffs"] = false;
    config["extRefTest"] = false;
    config["dumpTensorData"] = false;
    config["skipTest"] = false;
    config["reluEn"] = false;
    config["fp8BiasIn"] = EXPONENT_BIAS_FP8_152_15;
    config["fp8BiasIn2"] = -1;  //  default value is invalid, will be caught later and set as fp8BiasIn
    config["fp8BiasOut"] = EXPONENT_BIAS_FP8_152_15;
    config["firstSoIdx"] = -1;
    config["groupId"] = -1;
    config["testSramOffset"] = 0;
    config["testHbmOffset"] = 0;
    config["forceCloseGroup"] = false;
    config["powerLoops"] = 1;
    config["powerIdleCycles"] = 0;
    config["powerIdleLoops"] = 0;
    config["pmuConfig"] = PMUCFGNONE;
    config["useBrain"] = false;
    config["sbSizeInCLs"] = 0;

    for (int dim = 0; dim < MME_MAX_CONV_DIMS - 1; dim++)
    {
        config["padding"].push_back(0);
        config["dilation"].push_back(1);
        config["strides"].push_back(1);
    }
    for (int dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        config["xSizes"].push_back(0);
        config["wSizes"].push_back(0);
        config["ySizes"].push_back(0);
        config["xSizes2"].push_back(0);
        config["wSizes2"].push_back(0);
        config["ySizes2"].push_back(0);

        config["xStrides"].push_back(0);
        config["wStrides"].push_back(0);
        config["yStrides"].push_back(0);
        config["xStrides2"].push_back(0);
        config["yStrides2"].push_back(0);
        config["wStrides2"].push_back(0);
    }

    //  set default data randomizaition
    config["xMean"] = 0;
    config["wMean"] = 0;
    config["yMean"] = 0;
    config["xStd"] = 0;
    config["wStd"] = 0;
    config["yStd"] = 0;
    config["xMinVal"] = -1;
    config["wMinVal"] = -1;
    config["yMinVal"] = -1;
    config["xMaxVal"] = 10;
    config["wMaxVal"] = 10;
    config["yMaxVal"] = 10;
}

bool MMETestConfigParser::parseJsonFile(const std::string& configFilePath, unsigned repeats, unsigned b2bTestLimit)
{
    // parse file as json
    std::ifstream file(configFilePath.c_str());
    json jsonTests;
    MME_ASSERT(file.good(), fmt::format("Failed to open config file - {}", configFilePath).c_str());
    LOG_TRACE(MME_CONFIG_PARSER, "Parsing config file {}", configFilePath);
    file >> jsonTests;

    return parse(jsonTests, repeats, b2bTestLimit);
}

bool MMETestConfigParser::parse(const json& inputJson, unsigned repeats, unsigned b2bTestLimit)
{
    MME_ASSERT(repeats > 0, "repeats cannot be 0");
    MME_ASSERT(m_initialSeed != 0, "must initialize the random seed first");
    bool status = parseGlobalConfig(inputJson, b2bTestLimit);
    if (!status) return status;

    auto tests = inputJson["tests"].get<std::vector<json>>();
    // iterate #repeats on tests.
    for (unsigned rep = 0; rep < repeats; rep++)
    {
        // iterate over all given tests
        for (const auto& test : tests)
        {
            // create all cartesian product tests from a specific tests.
            // and call configureTest callback method on each given config test.
            auto configureTestCallback = std::mem_fn(&MMETestConfigParser::configureTest);
            makeCartesianProductTests(test, m_initialSeed + m_tests.size(), configureTestCallback);
        }
    }
    LOG_TRACE(MME_CONFIG_PARSER, "Config file contain {} tests", getParsedTests().size());
    return status;
}

bool MMETestConfigParser::parseGlobalConfig(json jsonCfg, unsigned b2bTestLimitArg)
{
    try
    {
        if (jsonCfg.contains("pqBaseAddr"))
        {
            m_globalCfg["pqBaseAddr"] = convertToInt(jsonCfg["pqBaseAddr"].get<std::string>());
        }

        if (jsonCfg.contains("sramBase"))
        {
            uint64_t sramBaseVal = convertToInt(jsonCfg["sramBase"].get<std::string>());
            MME_ASSERT(sramBaseVal >= m_mmeHal.getSramStart(),
                       fmt::format("sram base address should be at least {:#x}",
                                   m_mmeHal.getSramStart()).c_str()
                       );
            m_globalCfg["sramBase"] = sramBaseVal;
        }
        if (jsonCfg.contains("sramSize"))
        {
            uint64_t sramSizeVal = convertToInt(jsonCfg["sramSize"].get<std::string>());
            MME_ASSERT(sramSizeVal <= m_mmeHal.getSramSize(),
                       fmt::format("sram base address should be at most {:#x}",
                                   m_mmeHal.getSramSize()).c_str());
            m_globalCfg["sramSize"] = sramSizeVal;
        }
        uint64_t hbmStart = m_mmeHal.getHBMStart() + PROG_BUFFER;
        uint64_t hbmBaseVal =
            (jsonCfg.contains("hbmBase")) ? convertToInt(jsonCfg["hbmBase"].get<std::string>()) : hbmStart;
        MME_ASSERT(hbmBaseVal >= hbmStart,
                   fmt::format("HBM base address should be at least {:#x}",
                               hbmStart).c_str());
        m_globalCfg["hbmBase"] = hbmBaseVal;
        uint64_t smBaseVal = (jsonCfg.contains("smBase")) ? convertToInt(jsonCfg["smBase"].get<std::string>())
                                                          : m_mmeHal.getSMStart();
        MME_ASSERT(smBaseVal >= m_mmeHal.getSMStart(),
                   fmt::format("SM base address should be at least {:#x}",
                               m_mmeHal.getSMStart()).c_str());
        m_globalCfg["smBase"] = smBaseVal;
        uint64_t hbmSizeVal =
            (jsonCfg.contains("hbmSize")) ? convertToInt(jsonCfg["hbmSize"].get<std::string>()) : m_mmeHal.getHBMSize();
        m_globalCfg["hbmSize"] = hbmSizeVal;

        bool shuffleVal = (jsonCfg.contains("shuffle")) ? convertToBool(jsonCfg["shuffle"].get<std::string>()) : false;
        m_globalCfg["shuffle"] = shuffleVal;
        bool programInSramVal =
            (jsonCfg.contains("programInSram")) ? convertToBool(jsonCfg["programInSram"].get<std::string>()) : true;
        m_globalCfg["programInSram"] = programInSramVal;

        m_globalCfg["b2bTestsNumLimit"] = getB2BTestNumLimit(jsonCfg, b2bTestLimitArg);
        m_globalCfg["addNullDescriptors"] = (jsonCfg.contains("addNullDescriptors"))
                                                ? convertToBool(jsonCfg["addNullDescriptors"].get<std::string>())
                                                : false;
        m_globalCfg["nullDescBeforeTest"] = (jsonCfg.contains("nullDescBeforeTest"))
                                                ? convertToBool(jsonCfg["nullDescBeforeTest"].get<std::string>())
                                                : true;
        m_globalCfg["nullDescNum"] =
            (jsonCfg.contains("nullDescNum")) ? convertToInt(jsonCfg["nullDescNum"].get<std::string>()) : 1;
        bool powerTest =
            (jsonCfg.contains("powerTest")) ? convertToBool(jsonCfg["powerTest"].get<std::string>()) : false;
        m_globalCfg["powerTest"] = powerTest;

        m_globalCfg["configLfsr"] =
            (jsonCfg.contains("configLfsr")) ? convertToBool(jsonCfg["configLfsr"].get<std::string>()) : false;
        m_globalCfg["cacheMode"] = getCacheModeVal(jsonCfg);

        const bool skipOnVerificationFailureVal = (jsonCfg.contains("skipOnVerificationFailure")) ?
            convertToBool(jsonCfg["skipOnVerificationFailure"].get<std::string>()) : false;
        m_globalCfg["skipOnVerificationFailure"] = skipOnVerificationFailureVal;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}

template<typename CallBackFunction>
void MMETestConfigParser::makeCartesianProductTests(const json& testConfig,
                                                    const unsigned int seed,
                                                    CallBackFunction& callback)
{
    std::vector<int> counters(testConfig.size(), 0);
    int idx = counters.size();
    unsigned localSeed = seed;
    while (idx >= 0)
    {
        if (idx == counters.size())
        {
            callback(this, testConfig, counters, localSeed++);
            idx--;
        }
        else
        {
            counters[idx]++;
            int currentParamListSize = testConfig.at(idx).items().begin().value().size();
            const std::string& name = testConfig.at(idx).items().begin().key();
            if (currentParamListSize == counters[idx] || name == "xSizes" || name == "ySizes" || name == "wSizes" ||
                name == "xSizes2" || name == "ySizes2" || name == "wSizes2" || name == "dilation" ||
                name == "xStrides" || name == "yStrides" || name == "wStrides" ||
                name == "xStrides2" || name == "yStrides2" || name == "wStrides2" ||
                name == "strides" || name == "padding" || name == "recipeTest" || name == "optimizationTest")
            {
                counters[idx] = 0;
                idx--;
            }
            else
            {
                idx = counters.size();
            }
        }
    }
}

void MMETestConfigParser::configureTest(const json& testConfig,
                                        const std::vector<int>& valueCounters,
                                        const unsigned int seed)
{
    json parsedConfig = m_globalCfg;
    // skip test if cannot parse test params correctly.
    if (parseTestParams(testConfig, valueCounters, seed, parsedConfig))
    {
        m_tests.push_back(parsedConfig);
    }
}

void MMETestConfigParser::fixConvParams(json& parsedConfig) const
{
    // DEDX with dilation and strides s.t. gcd(dilation[i], strides[i]) > 1 for some 0 <= i <= 2
    // is still not supported (except in gaudi2 and new gaudi flow). To allow testing with random dilation and strides,
    // this method fixes the dilation to ensure gcd==1
    if (m_mmeHal.getChipType() == e_mme_Gaudi && getenv("ENABLE_USE_NEW_DESCRIPTORS") == nullptr)
    {
        auto dilation = parsedConfig["dilation"].get<std::vector<int>>();
        auto strides = parsedConfig["strides"].get<std::vector<int>>();
        for (unsigned i = 0; i < MME_MAX_CONV_DIMS - 1; ++i)
        {
            unsigned gcd = std::__gcd(strides[i], dilation[i]);
            while (gcd != 0 && gcd != 1)
            {
                dilation[i] /= gcd;
                gcd = std::__gcd(strides[i], dilation[i]);
            }
        }
        if (dilation != parsedConfig["dilation"])
        {
            atomicColoredPrint(COLOR_YELLOW,
                               "WARNING: test modified. dilation and stride must not have any dividers\n");
            parsedConfig["dilation"] = dilation;
        }
    }

    if (parsedConfig["signalMode"].get<int>() == EMmeSignalingMode::e_mme_signaling_partial)
    {
        //  signal partial is broken, many test use random for this field so to avoid skipping many tests
        //  fix this field. once it is fixed change it back
        parsedConfig["signalMode"] = EMmeSignalingMode::e_mme_signaling_output;
    }

    if (parsedConfig["in2TypeFloat"].get<EMmeDataType>() == EMmeDataType::e_types_nr)
    {
        // second input type wasnt set, set it as first input
        parsedConfig["in2TypeFloat"] = parsedConfig["inTypeFloat"];
    }

    if (parsedConfig["fp8BiasIn2"].get<int>() == -1)
    {
        // second input bias wasnt set, set it as first inputs bias
        parsedConfig["fp8BiasIn2"] = parsedConfig["fp8BiasIn"];
    }

    if (parsedConfig["conversionRoundingMode"].get<int>() == 5)
    {
        // due to a gap in the enum the rounding mode could get an invalid value.
        parsedConfig["conversionRoundingMode"] = RoundToNearest;
    }
    if (parsedConfig["accRoundingMode"].get<int>() == 5)
    {
        // due to a gap in the enum the rounding mode could get an invalid value.
        parsedConfig["accRoundingMode"] = RoundToNearest;
    }
    if (m_mmeHal.getChipType() == e_mme_Gaudi3 && parsedConfig["cacheMode"] == true)
    {
        if (parsedConfig["programInSram"] == true || parsedConfig["xInSram"] == true ||
            parsedConfig["wInSram"] == true || parsedConfig["yInSram"] == true || parsedConfig["oInSram"] == true)
        {
            atomicColoredPrint(
                COLOR_YELLOW,
                "WARNING: test modified. chip running in cache mode - allocating all data and program in HBM\n");
        }
        // in cache mode we cant put data in sram - its HW managed.
        parsedConfig["programInSram"] = false;
        parsedConfig["xInSram"] = false;
        parsedConfig["wInSram"] = false;
        parsedConfig["yInSram"] = false;
        parsedConfig["oInSram"] = false;
    }
}

bool MMETestConfigParser::parseTestParams(const json& rawTestCfgs,
                                          const std::vector<int>& valueCounters,
                                          const unsigned int seed,
                                          json& parsedConfig)
{
    MME_ASSERT(valueCounters.size() == rawTestCfgs.size(), "");

    std::string operandA;
    std::string operandB;
    std::string operandOther;
    setDefaults(parsedConfig);

    std::uniform_int_distribution<uint32_t> uniformIntDistribution(std::numeric_limits<uint32_t>::min(),
                                                                   std::numeric_limits<uint32_t>::max());
    parsedConfig["id"] = uniformIntDistribution(m_randomGenerator);

    // counter is tracking which value should take from the value list for a specific key.
    auto valueIdxCounter = valueCounters.begin();
    for (const auto& config : rawTestCfgs)
    {
        std::string key = config.begin().key();
        const json& valueList = config.front();

        if (key == "recipeTest")
        {
            parseJsonArgForRecipeTest(key, parsedConfig, valueList);
            valueIdxCounter++;
            continue;
        }
        if (key == "optimizationTest")
        {
            parseJsonArgForOptimizationTest(key, parsedConfig, valueList);
            valueIdxCounter++;
            continue;
        }

        unsigned valueIndex = *valueIdxCounter;
        std::string valueStr = valueList.at(valueIndex).get<std::string>();
        parseTestName(key, parsedConfig, valueStr, valueList.size());
        parseMmeCommonValueListArg("inTypeFloat",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeDataType::e_type_first_float_type,
                                   EMmeDataType::e_type_last_float_type,
                                   m_typeNameMap);
        parseMmeCommonValueListArg("in2TypeFloat",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeDataType::e_type_first_float_type,
                                   EMmeDataType::e_type_last_float_type,
                                   m_typeNameMap);
        parseMmeCommonValueListArg("outTypeFloat",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeDataType::e_type_first_float_type,
                                   EMmeDataType::e_type_last_output_float_type,
                                   m_typeNameMap);
        parseNumArgWithRandomFpBias("fp8BiasIn", parsedConfig, key, valueStr, true, false);
        parseNumArgWithRandomFpBias("fp8BiasIn2", parsedConfig, key, valueStr, true, true);
        parseNumArgWithRandomFpBias("fp8BiasOut", parsedConfig, key, valueStr, false, false);
        parseOperationListArg("operation", parsedConfig, key, valueStr, m_operationMap);
        parseMmeCommonValueListArg("signalMode",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeSignalingMode::e_mme_signaling_once,
                                   EMmeSignalingMode::e_mme_signaling_nr,
                                   m_signalModeMap);
        parseMmeCommonValueListArg("traceMode",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeTraceMode::e_mme_trace_mode_none,
                                   EMmeTraceMode::e_mme_trace_mode_nr,
                                   m_traceModeMap);
        parseMmeCommonValueListArg("convPattern",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmePattern::e_patterns_first_fwd,
                                   EMmePattern::e_patterns_last_fwd,
                                   m_convPatternMap);
        parseMmeCommonValueListArg("dedwPattern",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmePattern::e_patterns_first_dedw,
                                   EMmePattern::e_patterns_last_dedw,
                                   m_convPatternMap);
        parseMmeCommonValueListArg("geometry",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   MmeCommon::EMmeGeometry::e_first_geometry_gaudi2,
                                   MmeCommon::EMmeGeometry::e_last_geometry_gaudi2,
                                   m_geometryMap);
        parseMmeCommonValueListArg("conversionRoundingMode",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   0,
                                   RoundingMode::RoundingMode_nr,
                                   m_roundingModeMap);
        parseMmeCommonValueListArg("accRoundingMode",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   0,
                                   RoundingMode::RoundingMode_nr,
                                   m_roundingModeMap);
        parseBoolArg("dualGemm", parsedConfig, key, valueStr);
        parseBoolArg("DecMode", parsedConfig, key, valueStr);
        parseBoolArg("slaveSignaling", parsedConfig, key, valueStr);
        parseBoolArg("useSameColorSet", parsedConfig, key, valueStr);
        parseBoolArg("memsetOutput", parsedConfig, key, valueStr);
        parseMmeCommonValueListArg("reductionOp",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   0,
                                   EMmeReductionOp::e_mme_reduction_nr,
                                   m_reductionOpMap);
        parseMmeCommonValueListArg("reductionRm",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   0,
                                   EMmeReductionRm::e_mme_reduction_round_nr,
                                   m_reductionRmMap);
        parseMmeCommonValueListArg("cacheDirectiveA",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeCacheDirective::SkipCache,
                                   EMmeCacheDirective::NR,
                                   m_cacheDirectiveMap);
        parseMmeCommonValueListArg("cacheDirectiveB",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeCacheDirective::SkipCache,
                                   EMmeCacheDirective::NR,
                                   m_cacheDirectiveMap);
        parseMmeCommonValueListArg("cacheDirectiveOut",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeCacheDirective::SkipCache,
                                   EMmeCacheDirective::NR,
                                   m_cacheDirectiveMap);
        parseMmeCommonValueListArg("cacheClassA",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeCacheClass::Low,
                                   EMmeCacheClass::Reserved,
                                   m_cacheClassMap);
        parseMmeCommonValueListArg("cacheClassB",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeCacheClass::Low,
                                   EMmeCacheClass::Reserved,
                                   m_cacheClassMap);
        parseMmeCommonValueListArg("cacheClassOut",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   EMmeCacheClass::Low,
                                   EMmeCacheClass::Reserved,
                                   m_cacheClassMap);
        parseBoolArg("incDec", parsedConfig, key, valueStr);
        parseBoolArg("loop", parsedConfig, key, valueStr);
        parseMmeCommonValueListArg("prefetchOperand",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   MmeCommon::EMmePrefetch::e_mme_prefetch_none,
                                   MmeCommon::EMmePrefetch::e_mme_prefetch_nr,
                                   m_prefetchMap);
        parseBoolArg("fullDesc", parsedConfig, key, valueStr);
        parseBoolArg("testHasNullDesc", parsedConfig, key, valueStr);
        parseBoolArg("dualNullDesc", parsedConfig, key, valueStr);
        parseBoolArg("skipRef", parsedConfig, key, valueStr);
        parseBoolArg("skipRun", parsedConfig, key, valueStr);
        parseBoolArg("printAllDiffs", parsedConfig, key, valueStr);
        parseBoolArg("extRefTest", parsedConfig, key, valueStr);
        parseBoolArg("dumpTensorData", parsedConfig, key, valueStr);
        parseBoolArg("skipTest", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("reluEn", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("lowerEn", parsedConfig, key, valueStr);
        parseMmeCommonValueListArg("dedwCDConcurrency",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   BoolWithUndef::TurnedOff,
                                   BoolWithUndef::Undefined,
                                   m_boolWithUndefMap);
        parseMmeCommonValueListArg("dedw2x",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   BoolWithUndef::TurnedOff,
                                   BoolWithUndef::Undefined,
                                   m_boolWithUndefMap);
        parseBoolArgWithRandom("unrollEn", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("recurringMisalignmentOptEn", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("teAccel", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("clippingEn", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("clipInfIn", parsedConfig, key, valueStr);
        parseMmeCommonValueListArg("infNanModeA",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   MmeCommon::InfNanMode::e_mme_full_inf_nan,
                                   MmeCommon::InfNanMode::e_mme_infNan_nr,
                                   m_InfNanModeMap);
        parseMmeCommonValueListArg("infNanModeB",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   MmeCommon::InfNanMode::e_mme_full_inf_nan,
                                   MmeCommon::InfNanMode::e_mme_infNan_nr,
                                   m_InfNanModeMap);
        parseMmeCommonValueListArg("infNanModeOut",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   MmeCommon::InfNanMode::e_mme_full_inf_nan,
                                   MmeCommon::InfNanMode::e_mme_infNan_nr,
                                   m_InfNanModeMap);
        parseBoolArgWithRandom("flushDenormals", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("stochasticFlush", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("sbCacheEn", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("sbReuse", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("partialsToMemoryEn", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("maskedBgemm", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("memsetVoidPixels", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("xInSram", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("wInSram", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("yInSram", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("oInSram", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("secondaryOutput", parsedConfig, key, valueStr);
        parseBoolArgWithRandom("alignedAddresses", parsedConfig, key, valueStr);
        parseNumArg("repeats", parsedConfig, key, valueStr);
        parseIntArrayWithRandom("strides", parsedConfig, key, config, STRIDES_MIN_SIZE, STRIDES_MAX_SIZE);
        parseIntArrayWithRandom("dilation", parsedConfig, key, config, DILATION_MIN_SIZE, DILATION_MAX_SIZE);
        parseIntArrayWithRandom("padding", parsedConfig, key, config, PADDING_MIN_SIZE, PADDING_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("ySizes", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("wSizes", parsedConfig, key, config, CONV_MIN_KERNEL_SIZE, CONV_MAX_KERNEL_SIZE);
        parseShapeArrayWithRandomDimSize("xSizes", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("ySizes2", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("wSizes2",
                                       parsedConfig,
                                       key,
                                       config,
                                       CONV_MIN_KERNEL_SIZE,
                                       CONV_MAX_KERNEL_SIZE);
        parseShapeArrayWithRandomDimSize("xSizes2", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("xStrides", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("wStrides", parsedConfig, key, config, CONV_MIN_KERNEL_SIZE, CONV_MAX_KERNEL_SIZE);
        parseShapeArrayWithRandomDimSize("yStrides", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("xStrides2", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseShapeArrayWithRandomDimSize("wStrides2", parsedConfig, key, config, CONV_MIN_KERNEL_SIZE, CONV_MAX_KERNEL_SIZE);
        parseShapeArrayWithRandomDimSize("yStrides2", parsedConfig, key, config, CONV_MIN_SIZE, CONV_MAX_SIZE);
        parseNumArg("sbSizeInCLs", parsedConfig, key, valueStr);
        parseNumArg("xMean", parsedConfig, key, valueStr);
        parseNumArg("wMean", parsedConfig, key, valueStr);
        parseNumArg("yMean", parsedConfig, key, valueStr);
        parseNumArg("xStd", parsedConfig, key, valueStr);
        parseNumArg("wStd", parsedConfig, key, valueStr);
        parseNumArg("yStd", parsedConfig, key, valueStr);
        parseNumArg("xMinVal", parsedConfig, key, valueStr);
        parseNumArg("wMinVal", parsedConfig, key, valueStr);
        parseNumArg("yMinVal", parsedConfig, key, valueStr);
        parseNumArg("xMaxVal", parsedConfig, key, valueStr);
        parseNumArg("wMaxVal", parsedConfig, key, valueStr);
        parseNumArg("yMaxVal", parsedConfig, key, valueStr);
        parseNumArg("firstSoIdx", parsedConfig, key, valueStr);
        parseNumArg("groupId", parsedConfig, key, valueStr);
        parseNumArg("pipelineLevel", parsedConfig, key, valueStr);
        parseNumArg("packingFactor", parsedConfig, key, valueStr);
        parseNumArg("reductionLevel", parsedConfig, key, valueStr);
        parseBoolArg("isDeterministic", parsedConfig, key, valueStr);
        parseBoolArg("compareDCDCresults", parsedConfig, key, valueStr);
        parseNumArg("testHbmOffset", parsedConfig, key, valueStr);
        parseNumArg("testSramOffset", parsedConfig, key, valueStr);
        parseBoolArg("forceCloseGroup", parsedConfig, key, valueStr);
        parseNumArg("powerLoops", parsedConfig, key, valueStr);
        parseNumArg("powerIdleCycles", parsedConfig, key, valueStr);
        parseNumArg("powerIdleLoops", parsedConfig, key, valueStr);
        parseMmeCommonValueListArg("pmuConfig",
                                   parsedConfig,
                                   key,
                                   valueStr,
                                   PmuConfig::PMUCFGNONE,
                                   PmuConfig::PMUCFGMODE4,
                                   m_pmuConfigMap);
        parseBoolArgWithRandom("useBrain", parsedConfig, key, valueStr);
        valueIdxCounter++;
    }
    fixConvParams(parsedConfig);
    m_randomSizesMap.clear();
    m_orderConstrains.clear();

    return true;
}

int64_t MMETestConfigParser::convertToInt(const std::string& str)
{
    return std::strtoll(str.c_str(), nullptr, 0);
}

bool MMETestConfigParser::convertToBool(const std::string& str)
{
    bool ret = false;
    std::vector<std::string> trueValues = {"true", "True", "1"};
    for (auto val : trueValues)
    {
        if (str == val)
        {
            ret = true;
            break;
        }
    }
    return ret;
}

void MMETestConfigParser::parseBoolArg(const std::string& keyName,
                                       json& parsedConfig,
                                       const std::string& key,
                                       const std::string& value)
{
    if (key != keyName) return;
    bool val = convertToBool(value);
    parsedConfig[keyName] = val;
}

void MMETestConfigParser::parseBoolArgWithRandom(const std::string& keyName,
                                                 json& parsedConfig,
                                                 const std::string& key,
                                                 const std::string& value)
{
    if (key != keyName) return;
    std::string valueStr = value;
    if (value == "random")
    {
        valueStr = (std::uniform_int_distribution<unsigned>(0, 1))(m_randomGenerator) ? "true" : "false";
    }
    parseBoolArg(keyName, parsedConfig, key, valueStr);
}

void MMETestConfigParser::parseNumArg(const std::string& keyName,
                                      json& parsedConfig,
                                      const std::string& key,
                                      const std::string& value)
{
    if (key != keyName) return;
    parsedConfig[keyName] = std::strtof(value.c_str(), nullptr);
}

void MMETestConfigParser::parseOperationListArg(const std::string& keyName,
                                                json& parsedConfig,
                                                const std::string& key,
                                                const std::string& value,
                                                std::unordered_map<std::string, EMmeOpType>& nameMap)
{
    if (keyName != key) return;
    if (value == "random")
    {
        unsigned maxValue = (unsigned) EMmeOpType::e_mme_atbt;
        unsigned enumVal = std::uniform_int_distribution<unsigned>(0, maxValue)(m_randomGenerator);
        parsedConfig[keyName] = (EMmeOpType) enumVal;
    }
    else if (value == "random_bgemm" || value == "random_gemm")
    {
        unsigned maxValue = (unsigned) EMmeOpType::e_mme_atbt;
        unsigned minValue = (unsigned) EMmeOpType::e_mme_ab;
        unsigned enumVal = std::uniform_int_distribution<unsigned>(minValue, maxValue)(m_randomGenerator);
        parsedConfig[keyName] = (EMmeOpType) enumVal;
    }
    else if (value == "random_conv")
    {
        unsigned maxValue = (unsigned) EMmeOpType::e_mme_dedw;
        unsigned enumVal = std::uniform_int_distribution<unsigned>(0, maxValue)(m_randomGenerator);
        parsedConfig[keyName] = (EMmeOpType) enumVal;
    }
    else
    {
        auto it = nameMap.find(value);
        if (it == nameMap.end())
        {
            MME_ASSERT(it != nameMap.end(), "reached end of map");
        }
        parsedConfig[keyName] = nameMap[value];
    }
}

template<typename enumT>
void MMETestConfigParser::parseMmeCommonValueListArg(const std::string& keyName,
                                                     json& parsedConfig,
                                                     const std::string& key,
                                                     const std::string& value,
                                                     const unsigned enumMinValue,
                                                     const unsigned enumMaxValue,
                                                     std::unordered_map<std::string, enumT>& nameMap)
{
    if (keyName != key) return;
    if (value == "random")
    {
        unsigned enumVal = std::uniform_int_distribution<unsigned>(enumMinValue, enumMaxValue - 1)(m_randomGenerator);
        parsedConfig[keyName] = enumVal;
    }
    else if (value == "asInput")
    {
        MME_ASSERT((key == "outTypeFloat") || (key == "in2TypeFloat"),
                   "asInput is allowed only in outTypeFloat and in2TypeFloat");
        parsedConfig[key] = parsedConfig["inTypeFloat"];
    }
    else
    {
        auto it = nameMap.find(value);
        MME_ASSERT(it != nameMap.end(), "invalid argument");
        parsedConfig[keyName] = nameMap[value];
    }
}

template<typename enumT>
void MMETestConfigParser::parseValueListArg(const std::string& keyName,
                                            json& parsedConfig,
                                            const std::string& key,
                                            const std::string& value,
                                            std::unordered_map<std::string, enumT>& nameMap)
{
    if (keyName != key) return;
    if (value == "random")
    {
        unsigned maxValue = (unsigned) enumT::size - 1;
        unsigned enumVal = std::uniform_int_distribution<unsigned>(0, maxValue)(m_randomGenerator);
        parsedConfig[keyName] = (enumT) enumVal;
    }
    else if (value == "asInput")
    {
        MME_ASSERT((key == "outTypeFloat"), "asInput is allowed only in outTypeFloat");
        parsedConfig["outTypeFloat"] = parsedConfig["inTypeFloat"];
    }
    else
    {
        auto it = nameMap.find(value);
        MME_ASSERT(it != nameMap.end(), "reached end of map");
        parsedConfig[keyName] = nameMap[value];
    }
}

void MMETestConfigParser::parseNumArgWithRandomFpBias(const std::string& keyName,
                                                      json& parsedConfig,
                                                      const std::string& key,
                                                      const std::string& value,
                                                      bool isInput,
                                                      bool isSecondInput)
{
    if (keyName != key) return;
    if (value == "random")
    {
        if (!isInput)
        {
            if (parsedConfig["outTypeFloat"] == EMmeDataType::e_type_fp16)
            {
                parsedConfig[keyName] = EXPONENT_BIAS_FP16_15;
            }
            else if (parsedConfig["outTypeFloat"] == EMmeDataType::e_type_ufp16)
            {
                parsedConfig[keyName] = EXPONENT_BIAS_UFP16_31;
            }
        }
        else if (isInput && !isSecondInput)
        {
            if (parsedConfig["inTypeFloat"] == EMmeDataType::e_type_fp16)
            {
                parsedConfig[keyName] = EXPONENT_BIAS_FP16_15;
            }
            else if (parsedConfig["inTypeFloat"] == EMmeDataType::e_type_ufp16)
            {
                parsedConfig[keyName] = EXPONENT_BIAS_UFP16_31;
            }
        }
        else if (isInput && isSecondInput)
        {
            if (parsedConfig["in2TypeFloat"] == EMmeDataType::e_type_fp16)
            {
                parsedConfig[keyName] = EXPONENT_BIAS_FP16_15;
            }
            else if (parsedConfig["in2TypeFloat"] == EMmeDataType::e_type_ufp16)
            {
                parsedConfig[keyName] = EXPONENT_BIAS_UFP16_31;
            }
        }
        /* Exponent bias of fp8_152 is between 1-30. */
        else if (!isInput && parsedConfig["outTypeFloat"] == EMmeDataType::e_type_fp8_152)
        {
            unsigned val = std::uniform_int_distribution<unsigned>(EXPONENT_BIAS_FP8_152_MIN_VALUE,
                                                                   EXPONENT_BIAS_FP8_152_MAX_VALUE)(m_randomGenerator);
            parsedConfig[keyName] = val;
        }
        else if (isInput && parsedConfig["inTypeFloat"] == EMmeDataType::e_type_fp8_152)
        {
            parsedConfig[keyName] = EXPONENT_BIAS_FP8_152_15;
        }
        else
        {
            /* Exponent bias of fp8_143 is 3/7/11/15 */
            unsigned indexType =
                std::uniform_int_distribution<unsigned>(0, (EXPONENT_BIAS_FP8_143_TYPES - 1))(m_randomGenerator);
            parsedConfig[keyName] = m_exponentBiasFpTypes[indexType];
        }
    }
    else if (value == "asInput")
    {
        MME_ASSERT((key == "fp8BiasIn2"), "asInput is allowed only in fp8BiasIn2");
        parsedConfig["fp8BiasIn2"] = parsedConfig["fp8BiasIn"];
    }
    else
    {
        parsedConfig[keyName] = std::stoul(value.c_str(), nullptr);
    }
}

int MMETestConfigParser::getRandomSize(std::string type, int minValue, int maxValue)
{
    std::unordered_map<std::string, int>::const_iterator iter = m_randomSizesMap.find(type);

    if (iter == m_randomSizesMap.end())
    {
        int randomVal = std::uniform_int_distribution<int>(minValue, maxValue)(m_randomGenerator);
        m_randomSizesMap.insert(std::make_pair(type, randomVal));
        return randomVal;
    }

    return iter->second;
}

bool MMETestConfigParser::checkXSpSizeOrderConstrain()
{
    // dilation, strides, padding, ySizes and wSizes MUST appear in the json file before xSizes when using random
    const std::list<std::string> fieldsMustBefore = {"dilation", "strides", "padding", "ySizes", "wSizes"};
    auto count = 0;
    for (auto f : m_orderConstrains)
    {
        if ((std::find(fieldsMustBefore.begin(), fieldsMustBefore.end(), f) != fieldsMustBefore.end()))
        {

            ++count;
        }
    }
    if(count != fieldsMustBefore.size())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: Out of order!! dilation, strides, padding, ySizes and wSizes MUST appear in the json file before xSizes when using random.\n");
        return false;
    }

    return true;
}

void MMETestConfigParser::calculateXSpSize(unsigned xDim, json& parsedConfig)
{
    if(xDim > DIM_C && xDim < DIM_B)
    {
        if(!checkXSpSizeOrderConstrain())
        {
            m_orderConstrains.erase("xSizes");
            return;
        }
        fixConvParams(parsedConfig);
        const unsigned yDim = xDim;
        const unsigned wDim = xDim + 1;
        const unsigned convDim = xDim - 1; //stride, dilation, padding
        unsigned ofm = parsedConfig["ySizes"][yDim];
        unsigned s = parsedConfig["strides"][convDim];
        unsigned w = parsedConfig["wSizes"][wDim];
        unsigned pad = parsedConfig["padding"][convDim];
        unsigned dil = parsedConfig["dilation"][convDim];
        // x Sizes calculate according the formula
        parsedConfig["xSizes"][xDim] = (((ofm - 1) * s) + (w - 1) * dil + 1 - (2 * pad));
    }
    return;
}

void MMETestConfigParser::randomConv(const std::string& keyName, unsigned dim, json& parsedConfig, int minValue, int maxValue)
{
    if (keyName == "xSizes" && (dim > DIM_C && dim < DIM_B))
    {
        // constraint - x Sizes calculate according the formula based on ofm, kernel, stride, padding and diliation
        calculateXSpSize(dim, parsedConfig);
    }
    else if (keyName == "wSizes" && dim >= DIM_S && dim <= DIM_Q)
    {
        // constraint - kernel size is larger than padding size
        if( m_orderConstrains.find("padding") == m_orderConstrains.end())
        {
            atomicColoredPrint(COLOR_YELLOW, "WARNING: Out of order!! padding MUST appear in the json file before wSizes when using random.\n");
            m_orderConstrains.erase("wSizes");
            return;
        }
        unsigned pad = parsedConfig["padding"][dim - DIM_S];
        unsigned wSize;
        if(maxValue < pad + 1)
        {
            wSize = pad + 1;
        }
        else
        {
            wSize = std::uniform_int_distribution<int>(pad + 1 , maxValue)(m_randomGenerator);
        }
        parsedConfig[keyName].push_back(wSize);
    }
    else if ((keyName == "wSizes" && dim == WEIGHT_DIM_C) || (keyName == "xSizes" && dim == DIM_C))
    {
        // constraint - wSizes & xSizes with the same common dim
        parsedConfig[keyName].push_back(getRandomSize("random_common_dim", CONV_MIN_COMMON_DIM_SIZE, CONV_MAX_COMMON_DIM_SIZE));
    }
    else if ((keyName == "ySizes" || keyName == "wSizes")  && dim == DIM_K)
    {
        // constraint - ySizes & wSizes with the same width
        parsedConfig[keyName].push_back(getRandomSize("random_k_size", CONV_MIN_K_SIZE, CONV_MAX_K_SIZE));
    }
    else if ((keyName == "ySizes" || keyName == "xSizes") && dim == DIM_B)
    {
        // constraint - ySizes & xSizes with the same batch size
        parsedConfig[keyName].push_back(getRandomSize("random_batch_size", CONV_MIN_BATCH_SIZE, CONV_MAX_BATCH_SIZE));
    }
    else
    {
        int randomVal = std::uniform_int_distribution<int>(minValue, maxValue)(m_randomGenerator);
        parsedConfig[keyName].push_back(randomVal);
    }
    return;
}

void MMETestConfigParser::parseBgemmRandomDim(const std::string& testedTensor, unsigned testedDim, json& parsedConfig, std::string type, unsigned dimOp1, unsigned dimOp2)
{
    std::string tensorOp1;
    std::string tensorOp2;
    if(type == "random_common_dim")
    {
        tensorOp1 = "xSizes";
        tensorOp2 = "wSizes";
    }
    else if(type == "random_output_width")
    {
        tensorOp1 = "wSizes";
        tensorOp2 = "ySizes";
    }
    else if(type == "random_output_height")
    {
        tensorOp1 = "ySizes";
        tensorOp2 = "xSizes";
    }

    if ((testedTensor == tensorOp1 && testedDim == dimOp1) || (testedTensor == tensorOp2 && testedDim == dimOp2))
    {
        parsedConfig[testedTensor].push_back(getRandomSize(type, BGEMM_MIN_GEMM_SIZE, BGEMM_MAX_GEMM_SIZE));
    }
    return;
}

void MMETestConfigParser::randomBgemm(const std::string& keyName, unsigned dim, json& parsedConfig)
{

    if(dim <= WEIGHT_DIM_C)
    {
        EMmeOpType opType = parsedConfig["operation"];
        switch (opType)
        {
            case EMmeOpType::e_mme_ab:
                // constraint - commonDimOK = (xSizes[0] == wSizes[1])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_common_dim", 0, 1);
                // constraint - outputWidthOK = (ySizes[0] == wSizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_output_width", 0, 0);
                // constraint - outputHeightOK = (ySizes[1] == xSizes[1])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_output_height", 1, 1);
                break;
            case EMmeOpType::e_mme_abt:
                // constraint - commonDimOK = (xSizes[0] == wSizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig,"random_common_dim",  0, 0);
                // constraint - outputWidthOK = (wSizes[1] == ySizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig,"random_output_width", 1,  0);
                // constraint - outputHeightOK = (ySizes[1] == xSizes[1])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_output_height", 1, 1);
                break;
            case EMmeOpType::e_mme_atb:
                // constraint - commonDimOK = (xSizes[1] == wSizes[1])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_common_dim",  1, 1);
                // constraint - outputWidthOK = (wSizes[0] == ySizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_output_width",  0,  0);
                // constraint - outputHeightOK = (ySizes[1] == xSizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_output_height", 1,  0);
                break;
            case EMmeOpType::e_mme_atbt:
                // constraint - commonDimOK = (xSizes[1] == wSizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig,"random_common_dim", 1,  0);
                // constraint - outputWidthOK = (wSizes[1] == ySizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig,"random_output_width", 1, 0);
                // constraint - outputHeightOK = (ySizes[1] == xSizes[0])
                parseBgemmRandomDim(keyName, dim, parsedConfig, "random_output_height", 1, 0);
                break;
            default:
                MME_ASSERT(0, "should not get here");
        }
    }
    else
    {
        // constraint - ySizes & xSizes & wSizes with the same batch i size
        parsedConfig[keyName].push_back(getRandomSize("random_batch_size_"+ std::to_string(dim) , BGEMM_MIN_BATCH_SIZE, BGEMM_MAX_BATCH_SIZE));
    }
    return;
}

bool MMETestConfigParser::isConvOperation(json& parsedConfig)
{
    return (parsedConfig["operation"] == EMmeOpType::e_mme_fwd || parsedConfig["operation"] == EMmeOpType::e_mme_dedx ||
            parsedConfig["operation"] == EMmeOpType::e_mme_transposed_dedx ||
            parsedConfig["operation"] == EMmeOpType::e_mme_dedw);
}

void MMETestConfigParser::parseShapeArrayWithRandomDimSize(const std::string& keyName,
                                                         json& parsedConfig,
                                                         const std::string& key,
                                                         const json& testJson,
                                                         int minValue,
                                                         int maxValue)
{
    if (keyName != key) return;

    m_orderConstrains.insert(keyName);
    parsedConfig[keyName].clear();

    const std::vector<std::string>& valueVector = testJson[key].get<std::vector<std::string>>();
    unsigned currIndex = 0;
    for (auto& val : valueVector)
    {
        if (val == "random")
        {
            if(isConvOperation(parsedConfig))
            {
                randomConv(keyName, currIndex, parsedConfig, minValue, maxValue);
            }
            else
            {
                randomBgemm(keyName, currIndex, parsedConfig);
            }
        }
        else
        {
            parsedConfig[key].push_back(std::stoi(val));
        }
        currIndex++;
    }

    // pad shape vector with 1's on uninitialized dims.
    if (valueVector.size() < MAX_DIMENSION)
    {
        for (unsigned idx = valueVector.size(); idx < MAX_DIMENSION; idx++)
        {
            parsedConfig[key].push_back(1);
        }
    }
}

void MMETestConfigParser::parseIntArrayWithRandom(const std::string& keyName,
                                                  json& parsedConfig,
                                                  const std::string& key,
                                                  const json& testJson,
                                                  int minValue,
                                                  int maxValue)
{
    if (keyName != key) return;

    m_orderConstrains.insert(keyName);
    parsedConfig[keyName].clear();

    const std::vector<std::string>& valueVector = testJson[key].get<std::vector<std::string>>();
    for (auto& val : valueVector)
    {
        if (val == "random")
        {
            int randomVal = std::uniform_int_distribution<int>(minValue, maxValue)(m_randomGenerator);
            parsedConfig[keyName].push_back(randomVal);
        }
        else
        {
            parsedConfig[key].push_back(strtof(val.c_str(), nullptr));
        }
    }
}

void MMETestConfigParser::parseJsonArgForOptimizationTest(const std::string& key, json& parsedConfig, const json& value)
{
    if (value.empty()) return;

    parsedConfig[key] = json::object();
    for (json::const_iterator it = value.begin(); it != value.end(); ++it)
    {
        const std::string subKey = it.key();
        const json& subValue = it.value();

        if (subKey == "recurringMisalignmentCutPoints")
        {
            parseIntArrayWithRandom("recurringMisalignmentCutPoints", parsedConfig[key], subKey, value);
            continue;
        }

        const std::string subValueStr = subValue.get<std::string>();
        if (subKey == "cdConcurrency")
        {
            parseNumArg("cdConcurrency", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "batchConcurrency")
        {
            parseNumArg("batchConcurrency", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "geometryHeight")
        {
            parseNumArg("geometryHeight", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "doubleAccums")
        {
            parseBoolArg("doubleAccums", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "concurrencyDim")
        {
            parseNumArg("concurrencyDim", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "spInterleavingDim")
        {
            parseNumArg("spInterleavingDim", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "asymmetricPortMode")
        {
            parseBoolArg("asymmetricPortMode", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "recurringMisalignmentSubProblems")
        {
            parseNumArg("recurringMisalignmentSubProblems", parsedConfig[key], subKey, subValueStr);
        }
        else
        {
            parsedConfig[key][subKey] = subValueStr;
        }
    }
}

void MMETestConfigParser::parseJsonArgForRecipeTest(const std::string& key, json& parsedConfig, const json& value)
{
    if (value.empty()) return;

    parsedConfig[key] = json::object();
    for (json::const_iterator it = value.begin(); it != value.end(); ++it)
    {
        const std::string subKey = it.key();
        const json& subValue = it.value();
        const std::string subValueStr = subValue.get<std::string>();
        if (subKey == "reuse")
        {
            parseMmeCommonValueListArg("reuse",
                                       parsedConfig[key],
                                       subKey,
                                       subValueStr,
                                       EMmeReuseType::e_mme_no_reuse,
                                       EMmeReuseType::e_mme_2d_reuse,
                                       m_reuseTypeMap);
        }
        else if (subKey == "spSplits")
        {
            parseNumArg("spSplits", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "cdSplits")
        {
            parseNumArg("cdSplits", parsedConfig[key], subKey, subValueStr);
        }
        else if (subKey == "sbUtilization")
        {
            parseNumArg("sbUtilization", parsedConfig[key], subKey, subValueStr);
        }
        else
        {
            parsedConfig[key][subKey] = subValueStr;
        }
    }
}

void MMETestConfigParser::parseTestName(const std::string& key,
                                        json& parsedConfig,
                                        const std::string& value,
                                        unsigned int numOfValues)
{
    if (key != "testName") return;
    MME_ASSERT(numOfValues == 1, "attribute testName must have exactly one value");
    parsedConfig["testName"].push_back(value);
}

MMETestConfigParser::MMETestConfigParser(ChipType chipType) : m_mmeHal(getMmeHal(chipType))
{
    m_typeNameMap["fp16"] = EMmeDataType::e_type_fp16;
    m_typeNameMap["ufp16"] = EMmeDataType::e_type_ufp16;
    m_typeNameMap["bf16"] = EMmeDataType::e_type_bf16;
    m_typeNameMap["tf32"] = EMmeDataType::e_type_tf32;
    m_typeNameMap["fp32"] = EMmeDataType::e_type_fp32;
    m_typeNameMap["fp32_non_ieee"] = EMmeDataType::e_type_fp32;
    m_typeNameMap["fp32_ieee"] = EMmeDataType::e_type_fp32_ieee;
    m_typeNameMap["fp8_143"] = EMmeDataType::e_type_fp8_143;
    m_typeNameMap["fp8_152"] = EMmeDataType::e_type_fp8_152;
    m_operationMap["memcpy"] = EMmeOpType::e_mme_memcpy;
    m_operationMap["transpose"] = EMmeOpType::e_mme_trans;
    m_operationMap["gemm_transpose"] = EMmeOpType::e_mme_gemm_transpose;
    m_operationMap["fwd"] = EMmeOpType::e_mme_fwd;
    m_operationMap["dedx"] = EMmeOpType::e_mme_dedx;
    m_operationMap["transposed_dedx"] = EMmeOpType::e_mme_transposed_dedx;
    m_operationMap["dedw"] = EMmeOpType::e_mme_dedw;
    m_operationMap["ab"] = EMmeOpType::e_mme_ab;
    m_operationMap["atb"] = EMmeOpType::e_mme_atb;
    m_operationMap["abt"] = EMmeOpType::e_mme_abt;
    m_operationMap["atbt"] = EMmeOpType::e_mme_atbt;
    m_operationMap["reductionAdd"] = EMmeOpType::e_mme_reductionAdd;
    m_signalModeMap["output"] = EMmeSignalingMode::e_mme_signaling_output;
    m_signalModeMap["desc"] = EMmeSignalingMode::e_mme_signaling_desc;
    m_signalModeMap["descWithStore"] = EMmeSignalingMode::e_mme_signaling_desc_with_store;
    m_signalModeMap["chunk"] = EMmeSignalingMode::e_mme_signaling_chunk;
    m_signalModeMap["once"] = EMmeSignalingMode::e_mme_signaling_once;
    m_signalModeMap["amount"] = EMmeSignalingMode::e_mme_signaling_amount;
    m_signalModeMap["partial"] = EMmeSignalingMode::e_mme_signaling_partial;
    m_traceModeMap["none"] = EMmeTraceMode::e_mme_trace_mode_none;
    m_traceModeMap["node"] = EMmeTraceMode::e_mme_trace_mode_layer_act;
    m_traceModeMap["desc"] = EMmeTraceMode::e_mme_trace_mode_desc;
    m_traceModeMap["advanced"] = EMmeTraceMode::e_mme_trace_mode_advanced;
    m_convPatternMap["skf"] = EMmePattern::e_mme_z_reduction_skf;
    m_convPatternMap["ksf"] = EMmePattern::e_mme_z_reduction_ksf;
    m_convPatternMap["kfc"] = EMmePattern::e_mme_sp_reduction_kfc;
    m_convPatternMap["fkc"] = EMmePattern::e_mme_sp_reduction_fkc;
    m_convPatternMap["fck"] = EMmePattern::e_mme_sp_reduction_fck;
    m_convPatternMap["cfk"] = EMmePattern::e_mme_sp_reduction_cfk;
    m_convPatternMap["kcf"] = EMmePattern::e_mme_sp_reduction_kcf;
    m_convPatternMap["ckf"] = EMmePattern::e_mme_sp_reduction_ckf;
    m_geometryMap["4xw"] = EMmeGeometry::e_mme_geometry_4xw;
    m_geometryMap["2xw"] = EMmeGeometry::e_mme_geometry_2xw;
    m_geometryMap["2xh"] = EMmeGeometry::e_mme_geometry_2xh;
    m_geometryMap["4xh"] = EMmeGeometry::e_mme_geometry_4xh;
    m_roundingModeMap["rn"] = RoundingMode::RoundToNearest;
    m_roundingModeMap["rz"] = RoundingMode::RoundToZero;
    m_roundingModeMap["ru"] = RoundingMode::RoundUp;
    m_roundingModeMap["rd"] = RoundingMode::RoundDown;
    m_roundingModeMap["rs"] = RoundingMode::StochasticRounding;
    m_roundingModeMap["rhaz"] = RoundingMode::RoundAwayFromZero;
    m_roundingModeMap["rsn"] = RoundingMode::StochasticRoundingAndNearest;
    m_reductionOpMap["add"] = EMmeReductionOp::e_mme_reduction_add;
    m_reductionOpMap["sub"] = EMmeReductionOp::e_mme_reduction_sub;
    m_reductionOpMap["min"] = EMmeReductionOp::e_mme_reduction_min;
    m_reductionOpMap["max"] = EMmeReductionOp::e_mme_reduction_max;
    m_reductionOpMap["max_0"] = EMmeReductionOp::e_mme_reduction_max_0;
    m_reductionOpMap["none"] = EMmeReductionOp::e_mme_reduction_none;
    m_reductionRmMap["rn"] = EMmeReductionRm::e_mme_reduction_round_half_to_nearest_even;
    m_reductionRmMap["rz"] = EMmeReductionRm::e_mme_reduction_round_to_zero;
    m_reductionRmMap["ru"] = EMmeReductionRm::e_mme_reduction_round_up;
    m_reductionRmMap["rd"] = EMmeReductionRm::e_mme_reduction_round_down;
    m_cacheDirectiveMap["skipCache"] = EMmeCacheDirective::SkipCache;
    m_cacheDirectiveMap["noAlloc"] = EMmeCacheDirective::NoAllocate;
    m_cacheDirectiveMap["homeAlloc"] = EMmeCacheDirective::HomeAllocate;
    m_cacheDirectiveMap["dcoreAlloc"] = EMmeCacheDirective::DcoreAllocate;
    m_cacheDirectiveMap["sharedAlloc"] = EMmeCacheDirective::SharedAllocate;
    m_cacheClassMap["low"] = EMmeCacheClass::Low;
    m_cacheClassMap["normal"] = EMmeCacheClass::Normal;
    m_cacheClassMap["high"] = EMmeCacheClass::High;
    m_prefetchMap["A"] = EMmePrefetch::e_mme_prefetch_A;
    m_prefetchMap["B"] = EMmePrefetch::e_mme_prefetch_B;
    m_prefetchMap["none"] = EMmePrefetch::e_mme_prefetch_none;
    m_reuseTypeMap["none"] = EMmeReuseType::e_mme_no_reuse;
    m_reuseTypeMap["A"] = EMmeReuseType::e_mme_1d_reuse_a;
    m_reuseTypeMap["B"] = EMmeReuseType::e_mme_1d_reuse_b;
    m_reuseTypeMap["AB"] = EMmeReuseType::e_mme_2d_reuse_ab;
    m_reuseTypeMap["BA"] = EMmeReuseType::e_mme_2d_reuse_ba;
    m_boolWithUndefMap["true"] = BoolWithUndef::TurnedOn;
    m_boolWithUndefMap["false"] = BoolWithUndef::TurnedOff;
    m_boolWithUndefMap["undef"] = BoolWithUndef::Undefined;

    m_InfNanModeMap["fullInfNan"] = InfNanMode::e_mme_full_inf_nan;
    m_InfNanModeMap["noInfNan"] = InfNanMode::e_mme_no_inf_nan;
    m_InfNanModeMap["minInfNan"] = InfNanMode::e_mme_minimal_inf_nan;

    m_pmuConfigMap = {
        {"none", PMUCFGNONE},
        {"mode1", PMUCFGMODE1},
        {"mode2", PMUCFGMODE2},
        {"mode3", PMUCFGMODE3},
        {"mode4", PMUCFGMODE4},
    };
}

void MMETestConfigParser::setSeed(unsigned int seed)
{
    m_initialSeed = seed;
    m_randomGenerator = std::mt19937(seed);
}

unsigned MMETestConfigParser::getB2BTestNumLimit(const json& jsonCfg, unsigned int b2bTestLimitArg)
{
    unsigned b2bTestLimit = 1;  // run a single test per program as default
    // give precedence by order - env variable, program argument, json attribute
    if (getenv("MME_TEST_B2B_LIMIT"))
    {
        b2bTestLimit = std::stoi(getenv("MME_TEST_B2B_LIMIT"));
    }
    else if (b2bTestLimitArg > 1)
    {
        b2bTestLimit = b2bTestLimitArg;
    }
    else if (jsonCfg.contains("b2bTestsNumLimit"))
    {
        b2bTestLimit = convertToInt(jsonCfg["b2bTestsNumLimit"].get<std::string>());
    }
    return b2bTestLimit;
}

MMEConfigPrinter::MMEConfigPrinter(bool dumpDefaults) : m_dumpDefaults(dumpDefaults)
{
    m_typeNameReverseMap[EMmeDataType::e_type_fp16] = "fp16";
    m_typeNameReverseMap[EMmeDataType::e_type_ufp16] = "ufp16";
    m_typeNameReverseMap[EMmeDataType::e_type_bf16] = "bf16";
    m_typeNameReverseMap[EMmeDataType::e_type_tf32] = "tf32";
    m_typeNameReverseMap[EMmeDataType::e_type_fp32] = "fp32";
    m_typeNameReverseMap[EMmeDataType::e_type_fp32_ieee] = "fp32_ieee";
    m_typeNameReverseMap[EMmeDataType::e_type_fp8_143] = "fp8_143";
    m_typeNameReverseMap[EMmeDataType::e_type_fp8_152] = "fp8_152";
    m_typeNameReverseMap[EMmeDataType::e_types_nr] = "asInput";
    m_operationReverseMap[EMmeOpType::e_mme_memcpy] = "memcpy";
    m_operationReverseMap[EMmeOpType::e_mme_trans] = "transpose";
    m_operationReverseMap[EMmeOpType::e_mme_gemm_transpose] = "gemm_transpose";
    m_operationReverseMap[EMmeOpType::e_mme_fwd] = "fwd";
    m_operationReverseMap[EMmeOpType::e_mme_dedx] = "dedx";
    m_operationReverseMap[EMmeOpType::e_mme_transposed_dedx] = "transposed_dedx";
    m_operationReverseMap[EMmeOpType::e_mme_dedw] = "dedw";
    m_operationReverseMap[EMmeOpType::e_mme_ab] = "ab";
    m_operationReverseMap[EMmeOpType::e_mme_atb] = "atb";
    m_operationReverseMap[EMmeOpType::e_mme_abt] = "abt";
    m_operationReverseMap[EMmeOpType::e_mme_atbt] = "atbt";
    m_operationReverseMap[EMmeOpType::e_mme_reductionAdd] = "reductionAdd";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_output] = "output";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_desc] = "desc";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_desc_with_store] = "descWithStore";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_chunk] = "chunk";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_once] = "once";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_amount] = "amount";
    m_signalModeReverseMap[EMmeSignalingMode::e_mme_signaling_partial] = "partial";
    m_traceModeReverseMap[EMmeTraceMode::e_mme_trace_mode_none] = "none";
    m_traceModeReverseMap[EMmeTraceMode::e_mme_trace_mode_desc] = "desc";
    m_traceModeReverseMap[EMmeTraceMode::e_mme_trace_mode_layer_act] = "node";
    m_traceModeReverseMap[EMmeTraceMode::e_mme_trace_mode_advanced] = "advanced";
    m_convPatternReverseMap[EMmePattern::e_mme_z_reduction_skf] = "skf";
    m_convPatternReverseMap[EMmePattern::e_mme_z_reduction_ksf] = "ksf";
    m_convPatternReverseMap[EMmePattern::e_mme_sp_reduction_kfc] = "kfc";
    m_convPatternReverseMap[EMmePattern::e_mme_sp_reduction_fkc] = "fkc";
    m_convPatternReverseMap[EMmePattern::e_mme_sp_reduction_fck] = "fck";
    m_convPatternReverseMap[EMmePattern::e_mme_sp_reduction_cfk] = "cfk";
    m_convPatternReverseMap[EMmePattern::e_mme_sp_reduction_kcf] = "kcf";
    m_convPatternReverseMap[EMmePattern::e_mme_sp_reduction_ckf] = "ckf";
    m_convPatternReverseMap[EMmePattern::e_mme_patterns_nr] = "not specified";
    m_geometryReverseMap[EMmeGeometry::e_mme_geometry_4xw] = "4xw";
    m_geometryReverseMap[EMmeGeometry::e_mme_geometry_2xw] = "2xw";
    m_geometryReverseMap[EMmeGeometry::e_mme_geometry_2xh] = "2xh";
    m_geometryReverseMap[EMmeGeometry::e_mme_geometry_4xh] = "4xh";
    m_geometryReverseMap[EMmeGeometry::e_mme_geometry_nr] = "not specified";
    m_roundingModeReverseMap[RoundingMode::RoundToNearest] = "rn";
    m_roundingModeReverseMap[RoundingMode::RoundToZero] = "rz";
    m_roundingModeReverseMap[RoundingMode::RoundUp] = "ru";
    m_roundingModeReverseMap[RoundingMode::RoundDown] = "rd";
    m_roundingModeReverseMap[RoundingMode::StochasticRounding] = "rs";
    m_roundingModeReverseMap[RoundingMode::RoundAwayFromZero] = "rhaz";
    m_roundingModeReverseMap[RoundingMode::StochasticRoundingAndNearest] = "rsn";
    m_reductionOpReverseMap[EMmeReductionOp::e_mme_reduction_add] = "add";
    m_reductionOpReverseMap[EMmeReductionOp::e_mme_reduction_sub] = "sub";
    m_reductionOpReverseMap[EMmeReductionOp::e_mme_reduction_min] = "min";
    m_reductionOpReverseMap[EMmeReductionOp::e_mme_reduction_max] = "max";
    m_reductionOpReverseMap[EMmeReductionOp::e_mme_reduction_max_0] = "max_0";
    m_reductionOpReverseMap[EMmeReductionOp::e_mme_reduction_none] = "none";
    m_reductionRmReverseMap[EMmeReductionRm::e_mme_reduction_round_half_to_nearest_even] = "rn";
    m_reductionRmReverseMap[EMmeReductionRm::e_mme_reduction_round_to_zero] = "rz";
    m_reductionRmReverseMap[EMmeReductionRm::e_mme_reduction_round_up] = "ru";
    m_reductionRmReverseMap[EMmeReductionRm::e_mme_reduction_round_down] = "rd";
    m_reductionRmReverseMap[EMmeReductionRm::e_mme_reduction_round_nr] = "not specified";
    m_cacheDirectiveReverseMap[EMmeCacheDirective::SkipCache] = "skipCache";
    m_cacheDirectiveReverseMap[EMmeCacheDirective::NoAllocate] = "noAlloc";
    m_cacheDirectiveReverseMap[EMmeCacheDirective::HomeAllocate] = "homeAlloc";
    m_cacheDirectiveReverseMap[EMmeCacheDirective::DcoreAllocate] = "dcoreAlloc";
    m_cacheDirectiveReverseMap[EMmeCacheDirective::SharedAllocate] = "sharedAlloc";
    m_cacheClassReverseMap[EMmeCacheClass::Low] = "low";
    m_cacheClassReverseMap[EMmeCacheClass::Normal] = "normal";
    m_cacheClassReverseMap[EMmeCacheClass::High] = "high";
    m_prefetchReverseMap[EMmePrefetch::e_mme_prefetch_none] = "none";
    m_prefetchReverseMap[EMmePrefetch::e_mme_prefetch_A] = "A";
    m_prefetchReverseMap[EMmePrefetch::e_mme_prefetch_B] = "B";
    m_infNanModeReverseMap[InfNanMode::e_mme_full_inf_nan] = "fullInfNan";
    m_infNanModeReverseMap[InfNanMode::e_mme_no_inf_nan] = "noInfNan";
    m_infNanModeReverseMap[InfNanMode::e_mme_minimal_inf_nan] = "minInfNan";
    m_reuseTypeReverseMap[EMmeReuseType::e_mme_no_reuse] = "none";
    m_reuseTypeReverseMap[EMmeReuseType::e_mme_1d_reuse_a] = "A";
    m_reuseTypeReverseMap[EMmeReuseType::e_mme_1d_reuse_b] = "B";
    m_reuseTypeReverseMap[EMmeReuseType::e_mme_2d_reuse_ab] = "AB";
    m_reuseTypeReverseMap[EMmeReuseType::e_mme_2d_reuse_ba] = "BA";
    m_boolWithUndefReverseMap[BoolWithUndef::TurnedOff] = "false";
    m_boolWithUndefReverseMap[BoolWithUndef::TurnedOn] = "true";
    m_boolWithUndefReverseMap[BoolWithUndef::Undefined] = "undef";
    m_pmuConfigReverseMap = {
        {PMUCFGNONE, "none"},
        {PMUCFGMODE1, "mode1"},
        {PMUCFGMODE2, "mode2"},
        {PMUCFGMODE3, "mode3"},
        {PMUCFGMODE4, "mode4"},
    };
}

std::string MMEConfigPrinter::dump(const json& source)
{
    json rerunableJson = getRerunableJson(source);
    std::stringstream ss;
    std::string prettyJsonStr;
    std::string::size_type pos = 0;
    prettyJson(prettyJsonStr, rerunableJson.dump(), pos);
    ss << prettyJsonStr << std::endl;
    return ss.str();
}

std::string MMEConfigPrinter::dumpAndSerialize(testVector& sourceVector)
{
    json output;
    for (const auto& test : sourceVector)
    {
        json rerunableJson = getRerunableJson(test);
        for (const auto& item : rerunableJson.items())
        {
            if (item.key() == "tests") {
                output["tests"].push_back(item.value().front());
            }
            else
            {
                output[item.key()] = item.value();
            }
        }
    }
    std::stringstream ss;
    std::string prettyJsonStr;
    std::string::size_type pos = 0;
    prettyJson(prettyJsonStr, output.dump(), pos);
    ss << prettyJsonStr << std::endl;
    return ss.str();
}

json MMEConfigPrinter::getRerunableJson(const json& source)
{
    json rerunableJson, defaultJson;
    MMETestConfigParser::setDefaults(defaultJson);
    // global config
    if (source.contains("pqBaseAddr"))
    {
        addHex(rerunableJson, source, defaultJson, "pqBaseAddr", false);
    }

    addHex(rerunableJson, source, defaultJson, "sramBase", false);
    addHex(rerunableJson, source, defaultJson, "hbmBase", false);
    addHex(rerunableJson, source, defaultJson, "smBase", false);
    addHex(rerunableJson, source, defaultJson, "sramSize", false);
    addHex(rerunableJson, source, defaultJson, "hbmSize", false);
    addBool(rerunableJson, source, defaultJson, "shuffle", false);
    addBool(rerunableJson, source, defaultJson, "programInSram", false);
    addBool(rerunableJson, source, defaultJson, "cacheMode", false);
    addBool(rerunableJson, source, defaultJson, "addNullDescriptors", false);
    addBool(rerunableJson, source, defaultJson, "nullDescBeforeTest", false);
    addInt(rerunableJson, source, defaultJson, "nullDescNum", false);
    addInt(rerunableJson, source, defaultJson, "b2bTestsNumLimit", false);
    addInt(rerunableJson, source, defaultJson, "seed", false);
    addBool(rerunableJson, source, defaultJson, "configLfsr", false);
    addBool(rerunableJson, source, defaultJson, "powerTest", false);
    addBool(rerunableJson, source, defaultJson, "skipOnVerificationFailure", false);

    json testJson;
    addString(testJson, source, defaultJson, "testName");
    addValueList(testJson, source, defaultJson, "inTypeFloat", m_typeNameReverseMap);
    addValueList(testJson, source, defaultJson, "in2TypeFloat", m_typeNameReverseMap);
    addValueList(testJson, source, defaultJson, "outTypeFloat", m_typeNameReverseMap);
    addInt(testJson, source, defaultJson, "fp8BiasIn");
    addInt(testJson, source, defaultJson, "fp8BiasIn2");
    addInt(testJson, source, defaultJson, "fp8BiasOut");
    addValueList(testJson, source, defaultJson, "operation", m_operationReverseMap);
    addValueList(testJson, source, defaultJson, "signalMode", m_signalModeReverseMap);
    addValueList(testJson, source, defaultJson, "traceMode", m_traceModeReverseMap);
    addValueList(testJson, source, defaultJson, "convPattern", m_convPatternReverseMap);
    addValueList(testJson, source, defaultJson, "dedwPattern", m_convPatternReverseMap);
    addValueList(testJson, source, defaultJson, "geometry", m_geometryReverseMap);
    addValueList(testJson, source, defaultJson, "conversionRoundingMode", m_roundingModeReverseMap);
    addValueList(testJson, source, defaultJson, "accRoundingMode", m_roundingModeReverseMap);
    addBool(testJson, source, defaultJson, "dualGemm");
    addBool(testJson, source, defaultJson, "decMode");
    addBool(testJson, source, defaultJson, "slaveSignaling");
    addBool(testJson, source, defaultJson, "useSameColorSet");
    addBool(testJson, source, defaultJson, "memsetOutput");
    addValueList(testJson, source, defaultJson, "reductionOp", m_reductionOpReverseMap);
    addValueList(testJson, source, defaultJson, "reductionRm", m_reductionRmReverseMap);
    addValueList(testJson, source, defaultJson, "cacheDirectiveA", m_cacheDirectiveReverseMap);
    addValueList(testJson, source, defaultJson, "cacheDirectiveB", m_cacheDirectiveReverseMap);
    addValueList(testJson, source, defaultJson, "cacheDirectiveOut", m_cacheDirectiveReverseMap);
    addValueList(testJson, source, defaultJson, "cacheClassA", m_cacheClassReverseMap);
    addValueList(testJson, source, defaultJson, "cacheClassB", m_cacheClassReverseMap);
    addValueList(testJson, source, defaultJson, "cacheClassOut", m_cacheClassReverseMap);
    addBool(testJson, source, defaultJson, "incDec");
    addBool(testJson, source, defaultJson, "loop");
    addValueList(testJson, source, defaultJson, "prefetchOperand", m_prefetchReverseMap);
    addBool(testJson, source, defaultJson, "fullDesc");
    addBool(testJson, source, defaultJson, "testHasNullDesc");
    addBool(testJson, source, defaultJson, "dualNullDesc");
    addBool(testJson, source, defaultJson, "skipRef");
    addBool(testJson, source, defaultJson, "skipRun");
    addBool(testJson, source, defaultJson, "printAllDiffs");
    addBool(testJson, source, defaultJson, "extRefTest");
    addBool(testJson, source, defaultJson, "skipTest");
    addBool(testJson, source, defaultJson, "reluEn");
    addBool(testJson, source, defaultJson, "lowerEn");
    addBool(testJson, source, defaultJson, "unrollEn");
    addBool(testJson, source, defaultJson, "recurringMisalignmentOptEn");
    addValueList(testJson, source, defaultJson, "dedwCDConcurrency", m_boolWithUndefReverseMap);
    addValueList(testJson, source, defaultJson, "dedw2x", m_boolWithUndefReverseMap);
    addBool(testJson, source, defaultJson, "teAccel");
    addBool(testJson, source, defaultJson, "clippingEn");
    addBool(testJson, source, defaultJson, "clipInfIn");
    addValueList(testJson, source, defaultJson, "infNanModeA", m_infNanModeReverseMap);
    addValueList(testJson, source, defaultJson, "infNanModeB", m_infNanModeReverseMap);
    addValueList(testJson, source, defaultJson, "infNanModeOut", m_infNanModeReverseMap);
    addBool(testJson, source, defaultJson, "flushDenormals");
    addBool(testJson, source, defaultJson, "stochasticFlush");
    addBool(testJson, source, defaultJson, "sbCacheEn");
    addBool(testJson, source, defaultJson, "sbReuse");
    addBool(testJson, source, defaultJson, "partialsToMemoryEn");
    addBool(testJson, source, defaultJson, "maskedBgemm");
    addBool(testJson, source, defaultJson, "memsetVoidPixels");
    addBool(testJson, source, defaultJson, "xInSram");
    addBool(testJson, source, defaultJson, "wInSram");
    addBool(testJson, source, defaultJson, "yInSram");
    addBool(testJson, source, defaultJson, "oInSram");
    addBool(testJson, source, defaultJson, "secondaryOutput");
    addBool(testJson, source, defaultJson, "alignedAddresses");
    addBool(testJson, source, defaultJson, "useBrain");
    addInt(testJson, source, defaultJson, "repeats");
    addIntArray(testJson, source, defaultJson, "strides");
    addIntArray(testJson, source, defaultJson, "dilation");
    addIntArray(testJson, source, defaultJson, "padding");
    addIntArray(testJson, source, defaultJson, "ySizes");
    addIntArray(testJson, source, defaultJson, "wSizes");
    addIntArray(testJson, source, defaultJson, "xSizes");
    addIntArray(testJson, source, defaultJson, "ySizes2");
    addIntArray(testJson, source, defaultJson, "wSizes2");
    addIntArray(testJson, source, defaultJson, "xSizes2");
    addIntArray(testJson, source, defaultJson, "xStrides");
    addIntArray(testJson, source, defaultJson, "wStrides");
    addIntArray(testJson, source, defaultJson, "yStrides");
    addIntArray(testJson, source, defaultJson, "xStrides2");
    addIntArray(testJson, source, defaultJson, "wStrides2");
    addIntArray(testJson, source, defaultJson, "yStrides2");
    addFloat(testJson, source, defaultJson, "sbSizeInCLs");
    addFloat(testJson, source, defaultJson, "xMean");
    addFloat(testJson, source, defaultJson, "wMean");
    addFloat(testJson, source, defaultJson, "yMean");
    addFloat(testJson, source, defaultJson, "xStd");
    addFloat(testJson, source, defaultJson, "wStd");
    addFloat(testJson, source, defaultJson, "yStd");
    addFloat(testJson, source, defaultJson, "xMinVal");
    addFloat(testJson, source, defaultJson, "wMinVal");
    addFloat(testJson, source, defaultJson, "yMinVal");
    addFloat(testJson, source, defaultJson, "xMaxVal");
    addFloat(testJson, source, defaultJson, "wMaxVal");
    addFloat(testJson, source, defaultJson, "yMaxVal");
    addInt(testJson, source, defaultJson, "firstSoIdx");
    addInt(testJson, source, defaultJson, "groupId");
    addInt(testJson, source, defaultJson, "pipelineLevel");
    addInt(testJson, source, defaultJson, "packingFactor");
    addInt(testJson, source, defaultJson, "reductionLevel");
    addBool(testJson, source, defaultJson, "isDeterministic");
    addBool(testJson, source, defaultJson, "compareDCDCresults");
    addInt(testJson, source, defaultJson, "testSramOffset");
    addInt(testJson, source, defaultJson, "testHbmOffset");
    addBool(testJson, source, defaultJson, "forceCloseGroup");
    addInt(testJson, source, defaultJson, "id");
    addValueList(testJson, source, defaultJson, "pmuConfig", m_pmuConfigReverseMap);
    addInt(testJson, source, defaultJson, "powerLoops");
    addInt(testJson, source, defaultJson, "powerIdleCycles");
    addInt(testJson, source, defaultJson, "powerIdleLoops");
    addRecipeTestValues(testJson, source);
    addOptimizationTestValues(testJson, source);
    rerunableJson["tests"].push_back(testJson);
    return rerunableJson;
}

void MMEConfigPrinter::addIntArray(json& dest,
                                   const json& source,
                                   const json& defaultJson,
                                   const std::string& attribute,
                                   bool asJson)
{
    if (source.count(attribute) == 0) return;
    if (!m_dumpDefaults && defaultJson.contains(attribute))
    {
        auto srcVec = source.at(attribute).get<std::vector<int>>();
        auto defaultVec = defaultJson.at(attribute).get<std::vector<int>>();
        bool diff = false;
        for (unsigned i = 0; i < srcVec.size(); i++)
        {
            if (srcVec[i] != defaultVec[i])
            {
                diff = true;
                break;
            }
        }
        if (!diff) return;
    }

    if (asJson)
    {
        json j;
        for (auto& val : source.at(attribute).get<std::vector<int>>())
        {
            std::string stringVal = std::to_string(val);
            j[attribute].push_back(stringVal);
        }
        dest.push_back(j);
    }
    else
    {
        for (auto& val : source.at(attribute).get<std::vector<int>>())
        {
            std::string stringVal = std::to_string(val);
            dest[attribute].push_back(stringVal);
        }
    }
}
void MMEConfigPrinter::addFloat(json& dest,
                                const json& source,
                                const json& defaultJson,
                                const std::string& attribute,
                                bool asJson)
{
    if (source.count(attribute) == 0) return;
    float val = source.at(attribute).get<float>();
    if (val > UINT16_MAX)
    {
        return addHex(dest, source, defaultJson, attribute, asJson);
    }
    float defaultVal = defaultJson.at(attribute).get<float>();
    if (!m_dumpDefaults && val == defaultVal) return;  // dont print default value
    std::stringstream ss;
    ss << val;
    std::string strVal = ss.str();
    if (asJson)
    {
        json j;
        j[attribute].push_back(strVal);
        dest.push_back(j);
    }
    else
    {
        dest[attribute] = strVal;
    }
}

void MMEConfigPrinter::addInt(json& dest,
                              const json& source,
                              const json& defaultJson,
                              const std::string& attribute,
                              bool asJson)
{
    if (source.count(attribute) == 0) return;
    int val = source.at(attribute).get<int>();
    if (!m_dumpDefaults && defaultJson.contains(attribute))
    {
        int defaultVal = defaultJson.at(attribute).get<int>();
        if (val == defaultVal) return;  // dont print default value
        if (!attribute.compare("fp8BiasIn2"))
        {
            // special case for second operand bias - only print if its different than first operand bias
            if (val == source.at("fp8BiasIn").get<int>()) return;
        }
    }
    std::string strVal = std::to_string(val);
    if (asJson)
    {
        json j;
        j[attribute].push_back(strVal);
        dest.push_back(j);
    }
    else
    {
        dest[attribute] = strVal;
    }
}

void MMEConfigPrinter::addHex(json& dest,
                              const json& source,
                              const json& defaultJson,
                              const std::string& attribute,
                              bool asJson)
{
    if (source.count(attribute) == 0) return;
    std::stringstream ss;
    int64_t val = source.at(attribute).get<int64_t>();
    if (!m_dumpDefaults && defaultJson.contains(attribute))
    {
        int64_t defaultVal = defaultJson.at(attribute).get<int64_t>();
        if (val == defaultVal) return;  // dont print default value
    }
    ss << "0x" << std::hex << val;
    if (asJson)
    {
        json j;
        j[attribute].push_back(ss.str());
        dest.push_back(j);
    }
    else
    {
        dest[attribute] = ss.str();
    }
}

void MMEConfigPrinter::addString(json& dest,
                                 const json& source,
                                 const json& defaultJson,
                                 const std::string& attribute,
                                 bool asJson)
{
    if (source.count(attribute) == 0) return;

    const auto& strVec = source.at(attribute).get<std::vector<std::string>>();
    MME_ASSERT(strVec.size() == 1, "should have only single string attribute");
    const std::string& strVal = strVec.front();

    if (asJson)
    {
        json j;
        j[attribute].push_back(strVal);
        dest.push_back(j);
    }
    else
    {
        dest[attribute] = strVal;
    }
}

void MMEConfigPrinter::addBool(json& dest,
                               const json& source,
                               const json& defaultJson,
                               const std::string& attribute,
                               bool asJson)
{
    if (source.count(attribute) == 0) return;
    bool val = source.at(attribute).get<bool>();
    if (!m_dumpDefaults && defaultJson.contains(attribute))
    {
        bool defaultVal = defaultJson.at(attribute).get<bool>();
        if (val == defaultVal) return;  // dont print default value
    }
    std::string strVal = val ? "true" : "false";
    if (asJson)
    {
        json j;
        j[attribute].push_back(strVal);
        dest.push_back(j);
    }
    else
    {
        dest[attribute] = strVal;
    }
}

void MMEConfigPrinter::addRecipeTestValues(json& dest, const json& source, bool asJson)
{
    const std::string& key = "recipeTest";
    if (source.count(key) == 0 || source[key].empty()) return;

    json recipeTestJson;
    recipeTestJson[key] = json::object();
    addValueList(recipeTestJson[key], source[key], json::object(), "reuse", m_reuseTypeReverseMap, false);
    addInt(recipeTestJson[key], source[key], json::object(), "spSplits", false);
    addInt(recipeTestJson[key], source[key], json::object(), "cdSplits", false);
    addInt(recipeTestJson[key], source[key], json::object(), "sbUtilization", false);

    if (!recipeTestJson[key].empty())
    {
        dest.push_back(recipeTestJson);
    }
}

void MMEConfigPrinter::addOptimizationTestValues(json& dest, const json& source, bool asJson)
{
    const std::string& key = "optimizationTest";
    if (source.count(key) == 0 || source[key].empty()) return;

    json optimizationTestJson;
    optimizationTestJson[key] = json::object();
    addBool(optimizationTestJson[key], source[key], json::object(), "doubleAccums", false);
    addInt(optimizationTestJson[key], source[key], json::object(), "cdConcurrency", false);
    addInt(optimizationTestJson[key], source[key], json::object(), "batchConcurrency", false);
    addInt(optimizationTestJson[key], source[key], json::object(), "geometryHeight", false);
    addInt(optimizationTestJson[key], source[key], json::object(), "spInterleavingDim", false);
    addInt(optimizationTestJson[key], source[key], json::object(), "concurrencyDim", false);
    addBool(optimizationTestJson[key], source[key], json::object(), "asymmetricPortMode", false);
    addInt(optimizationTestJson[key], source[key], json::object(), "recurringMisalignmentSubProblems", false);
    addIntArray(optimizationTestJson[key], source[key], json::object(), "recurringMisalignmentCutPoints", false);

    if (!optimizationTestJson[key].empty())
    {
        dest.push_back(optimizationTestJson);
    }
}

template<typename T>
void MMEConfigPrinter::addValueList(json& dest,
                                    const json& source,
                                    const json& defaultJson,
                                    const std::string& attribute,
                                    std::unordered_map<T, std::string, EnumClassHash> toStringMap,
                                    bool asJson)
{
    if (source.count(attribute) == 0) return;
    T val = source.at(attribute).get<T>();
    if (!m_dumpDefaults && defaultJson.contains(attribute))
    {
        T defaultVal = defaultJson.at(attribute).get<T>();
        if (val == defaultVal) return;  // dont print default value
        if (!attribute.compare("in2TypeFloat"))
        {
            // special case for second operand DT - only print if its different than first operand DT
            if (val == source.at("inTypeFloat").get<int>()) return;
        }
    }
    MME_ASSERT(toStringMap.count(val) != 0, "string map is empty");
    std::string strVal = std::string(toStringMap[val]);
    if (asJson)
    {
        json j;
        j[attribute].push_back(strVal);
        dest.push_back(j);
    }
    else
    {
        dest[attribute] = strVal;
    }
}

void MMEConfigPrinter::prettyJson(std::string& pretty_json,
                                  const std::string& json_string,
                                  std::string::size_type& pos,
                                  size_t le,
                                  size_t indent)
{
    bool in_string = false;
    unsigned openBraces = 0;
    for (; pos < json_string.length(); ++pos)
    {
        char curr_char = json_string[pos];
        switch (curr_char)
        {
            case '[':
                openBraces++;
                break;
            case ']':
                openBraces--;
                break;
            case ' ':
            case '\t':
            case '\r':
            case '\n':
                if (!in_string) continue;
                break;
            case '{':
                pretty_json.append(1, curr_char);
                if (!in_string)
                {
                    prettyJson(pretty_json, json_string, ++pos, le + 1, indent);
                }
                continue;
                break;
            case '}':
                if (!in_string)
                {
                    pretty_json.append(1, curr_char);
                    return;
                }
                break;
            case ':':
                if (!in_string)
                {
                    pretty_json.append(1, curr_char);
                    pretty_json.append(1, ' ');
                    continue;
                }
                break;
            case ',':
                if (!in_string && openBraces != 1)
                {
                    pretty_json.append(1, curr_char);
                    pretty_json.append(1, '\n');
                    pretty_json.append(le * indent, ' ');
                    continue;
                }
                break;
            case '"':
            {
                char last_char = (pos == 0) ? '\0' : json_string[pos - 1];
                if (last_char != '\\')
                {
                    pretty_json.append(1, curr_char);
                    in_string = !in_string;
                }
                continue;
                break;
            }
            default:
                break;
        }
        pretty_json.append(1, curr_char);
    }
}

bool MMETestConfigParser::getCacheModeVal(const json& jsonCfg)
{
    bool cacheModeVal = false;
    // give precedence by order - env variable, program argument, json attribute
    if (getenv("MME_CACHE_MODE"))
    {
        cacheModeVal = std::stoi(getenv("MME_CACHE_MODE")) != 0;
    }
    if (jsonCfg.contains("cacheMode"))
    {
        if (getenv("MME_CACHE_MODE"))
        {
            atomicColoredPrint(COLOR_YELLOW,
                               "WARNING: test modified. cache mode was taken from env variable: MME_CACHE_MODE\n");
        }
        else
        {
            cacheModeVal = convertToBool(jsonCfg["cacheMode"].get<std::string>());
        }
    }
    return cacheModeVal;
}
}  // namespace MmeCommon
