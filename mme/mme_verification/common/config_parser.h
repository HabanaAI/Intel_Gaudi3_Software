#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <random>
#include <set>
#include "json.hpp"
#include "include/mme_common//mme_common_enum.h"
#include "include/mme_common/recipe.h"

namespace MmeCommon
{
class MmeHalReader;

using json = nlohmann::json;
using testVector = std::vector<json>;

struct EnumClassHash
{
    template<typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

/*
 * Responsible for parsing the config JSON file and create the tests.
 * creating a cartesian product of all options,
 * so if the json has 2 options specified in a specific key - 2 tests will be generated for each option.
 */
class MMETestConfigParser
{
public:
    MMETestConfigParser(ChipType chipType);
    ~MMETestConfigParser() = default;
    void setSeed(unsigned seed);
    bool parseJsonFile(const std::string& configFilePath, unsigned repeats, unsigned b2bTestLimit = 1);
    bool parse(const json& inputJson, unsigned repeates, unsigned b2bTestLimit = 1);
    testVector& getParsedTests() { return m_tests; };
    static void setDefaults(json& config);
private:
    bool parseGlobalConfig(json jsonCfg, unsigned b2bTestLimitArg = 1);
    template<typename CallBackFunction>
    void makeCartesianProductTests(const json& testConfig, unsigned seed, CallBackFunction& callback);

    void configureTest(const json& testConfig, const std::vector<int>& valueCounters, unsigned seed);
    bool parseTestParams(const json& rawTestCfgs,
                         const std::vector<int>& valueCounters,
                         const unsigned seed,
                         json& parsedConfig);
    void fixConvParams(json& parsedConfig) const;

    int64_t convertToInt(const std::string& str);
    bool convertToBool(const std::string& str);

    // parse argument methods
    void parseBoolArg(const std::string& keyName, json& parsedConfig, const std::string& key, const std::string& value);
    void parseBoolArgWithRandom(const std::string& keyName,
                                json& parsedConfig,
                                const std::string& key,
                                const std::string& value);
    void parseNumArg(const std::string& keyName, json& parsedConfig, const std::string& key, const std::string& value);
    template<typename enumT>
    void parseValueListArg(const std::string& keyName,
                           json& parsedConfig,
                           const std::string& key,
                           const std::string& value,
                           std::unordered_map<std::string, enumT>& nameMap);

    template<typename enumT>
    void parseMmeCommonValueListArg(const std::string& keyName,
                                    json& parsedConfig,
                                    const std::string& key,
                                    const std::string& value,
                                    const unsigned enumMinValue,
                                    const unsigned enumMaxValue,
                                    std::unordered_map<std::string, enumT>& nameMap);

    void parseOperationListArg(const std::string& keyName,
                               json& parsedConfig,
                               const std::string& key,
                               const std::string& value,
                               std::unordered_map<std::string, EMmeOpType>& nameMap);
    void parseNumArgWithRandomFpBias(const std::string& keyName,
                                     json& parsedConfig,
                                     const std::string& key,
                                     const std::string& value,
                                     bool isInput,
                                     bool isSecondInput);
    void parseIntArrayWithRandom(const std::string& keyName,
                                 json& parsedConfig,
                                 const std::string& key,
                                 const json& testJson,
                                 int minValue = 0,
                                 int maxValue = 0);
    void parseShapeArrayWithRandomDimSize(const std::string& keyName,
                                        json& parsedConfig,
                                        const std::string& key,
                                        const json& testJson,
                                        int minValue,
                                        int maxValue);
    int getRandomSize(std::string type, int minValue, int maxValue);
    void parseJsonArgForOptimizationTest(const std::string& key, json& parsedConfig, const json& value);
    void parseJsonArgForRecipeTest(const std::string& key, json& parsedConfig, const json& value);
    void parseTestName(const std::string& key, json& parsedConfig, const std::string& value, unsigned numOfValues);
    bool checkXSpSizeOrderConstrain();
    void calculateXSpSize(unsigned dim, json& parsedConfig);
    bool isConvOperation(json& parsedConfig);
    void randomConv(const std::string& keyName, unsigned dim, json& parsedConfig, int minValue, int maxValue);
    void randomBgemm(const std::string& keyName, unsigned dim, json& parsedConfig);
    void parseBgemmRandomDim(const std::string& testedTensor, unsigned testedDim, json& parsedConfig, std::string type, unsigned dimOp1, unsigned dimOp2);
    unsigned getB2BTestNumLimit(const json& jsonCfg, unsigned b2bTestLimitArg);
    bool getCacheModeVal(const json& jsonCfg);

    unsigned m_initialSeed = 0;
    const MmeCommon::MmeHalReader& m_mmeHal;
    std::mt19937 m_randomGenerator;
    testVector m_tests;
    json m_globalCfg;
    std::unordered_map<std::string, EMmeDataType> m_typeNameMap;
    std::unordered_map<std::string, EMmeOpType> m_operationMap;
    std::unordered_map<std::string, EMmeSignalingMode> m_signalModeMap;
    std::unordered_map<std::string, EMmeTraceMode> m_traceModeMap;
    std::unordered_map<std::string, EMmePattern> m_convPatternMap;
    std::unordered_map<std::string, EMmePattern> m_dedwPatternMap;
    std::unordered_map<std::string, EMmeGeometry> m_geometryMap;
    std::unordered_map<std::string, RoundingMode> m_roundingModeMap;
    const std::array<unsigned, EXPONENT_BIAS_FP8_143_TYPES> m_exponentBiasFpTypes = {
        {EXPONENT_BIAS_FP8_143_3, EXPONENT_BIAS_FP8_143_7, EXPONENT_BIAS_FP8_143_11, EXPONENT_BIAS_FP8_143_15}};
    std::unordered_map<std::string, EMmeReductionOp> m_reductionOpMap;
    std::unordered_map<std::string, EMmeReductionRm> m_reductionRmMap;
    std::unordered_map<std::string, EMmeCacheDirective> m_cacheDirectiveMap;
    std::unordered_map<std::string, EMmeCacheClass> m_cacheClassMap;
    std::unordered_map<std::string, EMmePrefetch> m_prefetchMap;
    std::unordered_map<std::string, InfNanMode> m_InfNanModeMap;
    std::unordered_map<std::string, int> m_randomSizesMap;
    std::set<std::string > m_orderConstrains;
    std::unordered_map<std::string, PmuConfig> m_pmuConfigMap;
    std::unordered_map<std::string, EMmeReuseType> m_reuseTypeMap;
    std::unordered_map<std::string, BoolWithUndef> m_boolWithUndefMap;
};

class MMEConfigPrinter
{
public:
    MMEConfigPrinter(bool dumpDefaults = false);
    ~MMEConfigPrinter() = default;
    std::string dump(const json& source);
    std::string dumpAndSerialize(testVector& source);
    void prettyJson(std::string& pretty_json,
                    const std::string& json_string,
                    std::string::size_type& pos,
                    size_t le = 0,
                    size_t indent = 3);

private:
    json getRerunableJson(const json& source);
    void
    addIntArray(json& dest, const json& source, const json& defaults, const std::string& attribute, bool asJson = true);
    void
    addFloat(json& dest, const json& source, const json& defaults, const std::string& attribute, bool asJson = true);
    void addInt(json& dest, const json& source, const json& defaults, const std::string& attribute, bool asJson = true);
    void addHex(json& dest, const json& source, const json& defaults, const std::string& attribute, bool asJson = true);
    void
    addBool(json& dest, const json& source, const json& defaults, const std::string& attribute, bool asJson = true);
    void
    addString(json& dest, const json& source, const json& defaults, const std::string& attribute, bool asJson = true);
    void addRecipeTestValues(json& dest, const json& source, bool asJson = true);
    void addOptimizationTestValues(json& dest, const json& source, bool asJson = true);
    template<typename T>
    void addValueList(json& dest,
                      const json& source,
                      const json& defaults,
                      const std::string& attribute,
                      std::unordered_map<T, std::string, EnumClassHash> toStringMap,
                      bool asJson = true);
    std::unordered_map<EMmeDataType, std::string, EnumClassHash> m_typeNameReverseMap;
    std::unordered_map<EMmeOpType, std::string, EnumClassHash> m_operationReverseMap;
    std::unordered_map<EMmeSignalingMode, std::string, EnumClassHash> m_signalModeReverseMap;
    std::unordered_map<EMmeTraceMode, std::string, EnumClassHash> m_traceModeReverseMap;
    std::unordered_map<EMmePattern, std::string, EnumClassHash> m_convPatternReverseMap;
    std::unordered_map<EMmePattern, std::string, EnumClassHash> m_dedwPatternReverseMap;
    std::unordered_map<EMmeGeometry, std::string, EnumClassHash> m_geometryReverseMap;
    std::unordered_map<RoundingMode, std::string, EnumClassHash> m_roundingModeReverseMap;
    std::unordered_map<EMmeReductionOp, std::string, EnumClassHash> m_reductionOpReverseMap;
    std::unordered_map<EMmeReductionRm, std::string, EnumClassHash> m_reductionRmReverseMap;
    std::unordered_map<EMmeCacheDirective, std::string, EnumClassHash> m_cacheDirectiveReverseMap;
    std::unordered_map<EMmeCacheClass, std::string, EnumClassHash> m_cacheClassReverseMap;
    std::unordered_map<EMmePrefetch, std::string, EnumClassHash> m_prefetchReverseMap;
    std::unordered_map<InfNanMode, std::string, EnumClassHash> m_infNanModeReverseMap;
    std::unordered_map<EMmeReuseType, std::string, EnumClassHash> m_reuseTypeReverseMap;
    std::unordered_map<PmuConfig, std::string, EnumClassHash> m_pmuConfigReverseMap;
    std::unordered_map<BoolWithUndef, std::string, EnumClassHash> m_boolWithUndefReverseMap;
    bool m_dumpDefaults = false;
};

}  // namespace MmeCommon
