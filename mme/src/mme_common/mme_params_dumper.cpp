#include "mme_params_dumper.h"
#include <fstream>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include "include/mme_common/mme_common_enum.h"
#include "print_utils.h"
using namespace std;

namespace MmeCommon
{
void MmeParamsDumper::dumpMmeParamsJson()
{
    char* paramsDumpPath = getenv("MME_PARAMS_DUMP_JSON_FILE_PATH");
    std::string filePath;
    if (paramsDumpPath != nullptr)
    {
        filePath = paramsDumpPath;
    }
    else
    {
        filePath = "mme_params.json";
    }

    Json jsonObject;
    std::ifstream prevJsonFile(filePath);
    if (prevJsonFile.good())
    {
        // get existing json so we can append our params object to list of previous params objects
        prevJsonFile >> jsonObject;
    }

    createMmeParamsJson();
    jsonObject["mmeParams"].push_back(m_paramsJson);

    std::ofstream jsonFile;
    jsonFile.open(filePath, std::ios::trunc);
    if (jsonFile.fail())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: failed to open json file: %s\n", filePath.c_str());
        return;
    }

    jsonFile << jsonObject.dump(4);
}

void MmeParamsDumper::createMmeParamsJson()
{
    m_paramsJson["nodeName"] = m_params.nodeName;
    m_paramsJson["opType"] = getOpTypeName(m_params.opType);
    m_paramsJson["spBase"] = m_params.spBase;
    m_paramsJson["spSize"] = m_params.spSize;

    createTensorJson(m_params.x, "x");
    createTensorJson(m_params.y, "y");
    createTensorJson(m_params.w, "w");

    for (int i = 0; i < MME_MAX_CONV_DIMS - 1; i++)
    {
        m_paramsJson["conv"]["stride"].push_back(m_params.conv.stride[i]);
        m_paramsJson["conv"]["dilation"].push_back(m_params.conv.dilation[i]);
        m_paramsJson["conv"]["padding"].push_back(m_params.conv.padding[i]);
    }

    m_paramsJson["reductionOp"] = getReductionOpName(m_params.memoryCfg.reductionOp);
    m_paramsJson["reductionRm"] = getReductionRoundingModeName(m_params.memoryCfg.reductionRm);

    createControlsJson();
    createStrategyJson();
}

void MmeParamsDumper::createTensorJson(const MmeTensorView& tensor, const std::string& tensorName)
{
    m_paramsJson[tensorName]["elementType"] = getDataTypeName(tensor.elementType);

    for (int i = 0; i < MME_MAX_TENSOR_DIMS; i++)
    {
        m_paramsJson[tensorName]["sizes"].push_back(tensor.sizes[i]);
        m_paramsJson[tensorName]["bases"].push_back(tensor.bases[i]);
        m_paramsJson[tensorName]["strides"].push_back(tensor.strides[i]);
    }
}

void MmeParamsDumper::createControlsJson()
{
    m_paramsJson["controls"]["roundingMode"] = getRoundingModeName(m_params.controls.roundingMode);
    m_paramsJson["controls"]["conversionRoundingMode"] = getRoundingModeName(m_params.controls.conversionRoundingMode);
    m_paramsJson["controls"]["accRoundingMode"] = getRoundingModeName(m_params.controls.accRoundingMode);
    m_paramsJson["controls"]["signalingMode"] = getSignalingModeName(m_params.controls.signalingMode);
    m_paramsJson["controls"]["signalAmount"] = m_params.controls.signalAmount;
    m_paramsJson["controls"]["slaveSignaling"] = m_params.controls.slaveSignaling;
    m_paramsJson["controls"]["useSameColorSet"] = m_params.controls.useSameColorSet;
    m_paramsJson["controls"]["atomicAdd"] = m_params.controls.atomicAdd;
    m_paramsJson["controls"]["squashIORois"] = m_params.controls.squashIORois;
    m_paramsJson["controls"]["reluEn"] = m_params.controls.reluEn;
    m_paramsJson["controls"]["fp8BiasIn"] = m_params.controls.fp8BiasIn;
    m_paramsJson["controls"]["fp8BiasIn2"] = m_params.controls.fp8BiasIn2;
    m_paramsJson["controls"]["fp8BiasOut"] = m_params.controls.fp8BiasOut;
    m_paramsJson["controls"]["clippingEn"] = m_params.controls.clippingEn;
    m_paramsJson["controls"]["clipInfIn"] = m_params.controls.clipInfIn;
    m_paramsJson["controls"]["infNanModeA"] = m_params.controls.infNanModeA;
    m_paramsJson["controls"]["infNanModeB"] = m_params.controls.infNanModeB;
    m_paramsJson["controls"]["infNanModeOut"] = m_params.controls.infNanModeOut;
    m_paramsJson["controls"]["flushDenormals"] = m_params.controls.flushDenormals;
    m_paramsJson["controls"]["stochasticFlush"] = m_params.controls.stochasticFlush;
    m_paramsJson["controls"]["pmuSaturationVal"] = m_params.controls.pmuSaturationVal;
}

void MmeParamsDumper::createStrategyJson()
{
    m_paramsJson["strategy"]["geometry"] = getGeometryName(m_params.strategy.geometry);
    m_paramsJson["strategy"]["pattern"] = getPatternName(m_params.strategy.pattern);
    m_paramsJson["strategy"]["pipelineLevel"] = m_params.strategy.pipelineLevel;
    m_paramsJson["strategy"]["packingFactor"] = m_params.strategy.packingFactor;
    m_paramsJson["strategy"]["reductionLevel"] = m_params.strategy.reductionLevel;
    m_paramsJson["strategy"]["loweringEn"] = m_params.strategy.loweringEn;
    m_paramsJson["strategy"]["sbReuse"] = m_params.strategy.sbReuse;
    m_paramsJson["strategy"]["alignedAddresses"] = m_params.strategy.alignedAddresses;
    m_paramsJson["strategy"]["unrollEn"] = m_params.strategy.unrollEn;
    m_paramsJson["strategy"]["partial"] = m_params.strategy.partial;
    m_paramsJson["strategy"]["signalPartial"] = m_params.strategy.signalPartial;
    m_paramsJson["strategy"]["memsetDedxVoidPixels"] = m_params.strategy.memsetDedxVoidPixels;
}

std::string MmeParamsDumper::getOpTypeName(const EMmeOpType& opType)
{
    const std::unordered_map<EMmeOpType, std::string, std::hash<int>> opTypesNames(
        {{EMmeOpType::e_mme_fwd, "fwd"},
         {EMmeOpType::e_mme_dedx, "dedx"},
         {EMmeOpType::e_mme_transposed_dedx, "transposed_dedx"},
         {EMmeOpType::e_mme_dedw, "dedw"},
         {EMmeOpType::e_mme_deterministic_dedw, "deterministic_dedw"},
         {EMmeOpType::e_mme_ab, "ab"},
         {EMmeOpType::e_mme_abt, "abt"},
         {EMmeOpType::e_mme_atb, "atb"},
         {EMmeOpType::e_mme_atbt, "atbt"},
         {EMmeOpType::e_mme_reductionAdd, "reductionAdd"}});

    auto opTypeName = opTypesNames.find(opType);

    if (opTypeName == opTypesNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid operation type: %d\n", opType);
        return "";
    }
    return opTypeName->second;
}

std::string MmeParamsDumper::getDataTypeName(const EMmeDataType& dataType)
{
    const std::unordered_map<EMmeDataType, std::string, std::hash<int>> dataTypesNames({
        {EMmeDataType::e_type_fp16, "fp16"},
        {EMmeDataType::e_type_ufp16, "ufp16"},
        {EMmeDataType::e_type_bf16, "bf16"},
        {EMmeDataType::e_type_fp32, "fp32"},
        {EMmeDataType::e_type_tf32, "tf32"},
        {EMmeDataType::e_type_fp8_143, "fp8_143"},
        {EMmeDataType::e_type_fp8_152, "fp8_152"},
        {EMmeDataType::e_type_fp32_ieee, "fp32_ieee"},
    });

    auto dataTypeName = dataTypesNames.find(dataType);

    if (dataTypeName == dataTypesNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid data type: %d\n", dataType);
        return "";
    }
    return dataTypeName->second;
}

std::string MmeParamsDumper::getReductionOpName(const EMmeReductionOp& reductionOp)
{
    const std::unordered_map<EMmeReductionOp, std::string, std::hash<int>> reductionOpsNames(
        {{EMmeReductionOp::e_mme_reduction_add, "add"},
         {EMmeReductionOp::e_mme_reduction_sub, "sub"},
         {EMmeReductionOp::e_mme_reduction_min, "min"},
         {EMmeReductionOp::e_mme_reduction_max, "max"},
         {EMmeReductionOp::e_mme_reduction_max_0, "max_0"},
         {EMmeReductionOp::e_mme_reduction_none, "none"}});

    auto reductionOpName = reductionOpsNames.find(reductionOp);

    if (reductionOpName == reductionOpsNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid reduction operation: %d\n", reductionOp);
        return "";
    }
    return reductionOpName->second;
}

std::string MmeParamsDumper::getReductionRoundingModeName(const EMmeReductionRm& reductionRm)
{
    const std::unordered_map<EMmeReductionRm, std::string, std::hash<int>> roundingModesNames(
        {{EMmeReductionRm::e_mme_reduction_round_half_to_nearest_even, "round_half_to_nearest_even"},
         {EMmeReductionRm::e_mme_reduction_round_to_zero, "round_to_zero"},
         {EMmeReductionRm::e_mme_reduction_round_up, "round_up"},
         {EMmeReductionRm::e_mme_reduction_round_down, "round_down"}});

    auto roundingModeName = roundingModesNames.find(reductionRm);

    if (roundingModeName == roundingModesNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid reduction rounding mode: %d\n", reductionRm);
        return "";
    }
    return roundingModeName->second;
}

std::string MmeParamsDumper::getRoundingModeName(const RoundingMode& roundingMode)
{
    const std::unordered_map<RoundingMode, std::string, std::hash<int>> roundingModesNames(
        {{RoundingMode::RoundToNearest, "RoundToNearest"},
         {RoundingMode::RoundToZero, "RoundToZero"},
         {RoundingMode::RoundUp, "RoundUp"},
         {RoundingMode::RoundDown, "RoundDown"},
         {RoundingMode::StochasticRounding, "StochasticRounding"},
         {RoundingMode::RoundAwayFromZero, "RoundAwayFromZero"},
         {RoundingMode::StochasticRoundingAndNearest, "StochasticRoundingAndNearest"}});

    auto roundingModeName = roundingModesNames.find(roundingMode);

    if (roundingModeName == roundingModesNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid rounding mode: %d\n", roundingMode);
        return "";
    }
    return roundingModeName->second;
}

std::string MmeParamsDumper::getSignalingModeName(const EMmeSignalingMode& signalingMode)
{
    const std::unordered_map<EMmeSignalingMode, std::string, std::hash<int>> signalingModesNames(
        {{EMmeSignalingMode::e_mme_signaling_none, "none"},
         {EMmeSignalingMode::e_mme_signaling_once, "once"},
         {EMmeSignalingMode::e_mme_signaling_desc, "desc"},
         {EMmeSignalingMode::e_mme_signaling_desc_with_store, "desc_with_store"},
         {EMmeSignalingMode::e_mme_signaling_chunk, "chunk"},
         {EMmeSignalingMode::e_mme_signaling_output, "output"},
         {EMmeSignalingMode::e_mme_signaling_partial, "partial"},
         {EMmeSignalingMode::e_mme_signaling_amount, "amount"}});

    auto signalingModeName = signalingModesNames.find(signalingMode);

    if (signalingModeName == signalingModesNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid signaling mode: %d\n", signalingMode);
        return "";
    }
    return signalingModeName->second;
}

std::string MmeParamsDumper::getGeometryName(const EMmeGeometry& geometry)
{
    const std::unordered_map<EMmeGeometry, std::string, std::hash<int>> geometriesNames(
        {{EMmeGeometry::e_mme_geometry_4xw, "4xw"},
         {EMmeGeometry::e_mme_geometry_2xw, "2xw"},
         {EMmeGeometry::e_mme_geometry_2xh, "2xh"},
         {EMmeGeometry::e_mme_geometry_4xh, "4xh"}});

    auto geometryName = geometriesNames.find(geometry);

    if (geometryName == geometriesNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid geometry: %d\n", geometry);
        return "";
    }
    return geometryName->second;
}

std::string MmeParamsDumper::getPatternName(const EMmePattern& pattern)
{
    const std::unordered_map<EMmePattern, std::string, std::hash<int>> patternsNames(
        {{EMmePattern::e_mme_sp_reduction_kfc, "kfc"},
         {EMmePattern::e_mme_sp_reduction_fkc, "fkc"},
         {EMmePattern::e_mme_sp_reduction_fck, "fck"},
         {EMmePattern::e_mme_sp_reduction_cfk, "cfk"},
         {EMmePattern::e_mme_sp_reduction_kcf, "kcf"},
         {EMmePattern::e_mme_sp_reduction_ckf, "ckf"},
         {EMmePattern::e_mme_z_reduction_ksf, "ksf"},
         {EMmePattern::e_mme_z_reduction_skf, "skf"}});

    auto patternName = patternsNames.find(pattern);

    if (patternName == patternsNames.end())
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: invalid pattern: %d\n", pattern);
        return "";
    }
    return patternName->second;
}

static std::string getBoolFieldStr(unsigned boolField)
{
    return ((boolField == 1) ? std::to_string(1) : std::to_string(0));
}

// This function dumps the key fields of the params file to cfg file in Gaudi format. The file
// name includes a running counter to enable dumping all the params of a single network.
// Important: The output is not a complete test, but rather the key fields
// To produce a full mme gaudi test, we need to set the global fields, and the rest of the fields
// For that we can use the script under mme_verification/common/create_mme_gaudi_test.py which
// reads a set of dumps as produced by this function and creates a single complete test
void MmeParamsDumper::dumpMmeParamsForGaudiCfg(std::string opStrCfg, std::string nodeName)
{
    const MmeLayerParams params = getParams();

    if ((opStrCfg.compare("all") != 0) && (opStrCfg.compare(getOpTypeName(params.opType)) != 0))
    {
        return;  // Do not dump the params
    }

    std::string operation = "operation=" + getOpTypeName(params.opType);
    std::string inType = "inType=";
    std::string outType = "outType=";
    std::string convPattern = "convPattern=" + getPatternName(params.strategy.pattern);
    std::string bgemmPattern = "convPattern=" + getPatternName(params.strategy.pattern);
    std::string dedwPattern = "dedwPattern=" + getPatternName(params.strategy.pattern);
    std::string geometry = "geometry=";

    switch (params.opType)
    {
        case MmeCommon::e_mme_fwd:
            inType += (params.x.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            outType += (params.y.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            break;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            inType += (params.y.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            outType += (params.x.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            break;
        case MmeCommon::e_mme_dedw:
            inType += (params.x.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            outType += (params.w.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            break;
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_reductionAdd:
            inType += (params.x.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            outType += (params.y.elementType == EMmeDataType::e_type_bf16) ? "bfloat" : "fp32";
            break;
        default:
            printf("Error! opType is %d\n", params.opType);
    }
    switch (params.strategy.geometry)
    {
        case e_mme_geometry_4wx1h:
            geometry += "4w1h";
            break;
        case e_mme_geometry_2wx2h:
            geometry += "2w2h";
            break;
        case e_mme_geometry_1wx4h:
            geometry += "1w4h";
            break;
        default:
            printf("Error! geometry is %d\n", params.strategy.geometry);
    }

    std::string xSizes = "xSizes=" + arrayToStr(params.x.sizes.data(), MAX_DIMENSION);
    std::string ySizes = "ySizes=" + arrayToStr(params.y.sizes.data(), MAX_DIMENSION);
    std::string wSizes = "wSizes=" + arrayToStr(params.w.sizes.data(), MAX_DIMENSION);
    std::string dilation = "dilation=" + arrayToStr(params.conv.dilation.data(), MME_MAX_CONV_DIMS - 1);
    std::string strides = "strides=" + arrayToStr(params.conv.stride.data(), MME_MAX_CONV_DIMS - 1);
    std::string padding = "padding=" + arrayToStr((unsigned*) (params.conv.padding.data()), MME_MAX_CONV_DIMS - 1);

    std::string sbReuse = "sbReuse=" + getBoolFieldStr(params.strategy.sbReuse);
    std::string unrollEn = "unrollEn=" + getBoolFieldStr(params.strategy.unrollEn);
    std::string reluEn = "reluEn=" + getBoolFieldStr(params.controls.reluEn);
    std::string lowerEn = "lowerEn=" + getBoolFieldStr(params.strategy.loweringEn);

    static unsigned nodeCount = 0;
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
    std::string fileName = "gaudi_node_" + ((nodeName.compare("") == 0) ? getOpTypeName(params.opType) : nodeName) +
                           "_" + std::to_string(nodeCount++) + ".cfg";

    ofstream dumpFile;
    dumpFile.open(fileName);

    dumpFile << operation + "\n";
    dumpFile << sbReuse + "\n";
    dumpFile << unrollEn + "\n";
    dumpFile << reluEn + "\n";
    dumpFile << lowerEn + "\n";
    dumpFile << xSizes + "\n";
    dumpFile << wSizes + "\n";
    dumpFile << ySizes + "\n";
    dumpFile << dilation + "\n";
    dumpFile << strides + "\n";
    dumpFile << padding + "\n";
    dumpFile << inType + "\n";
    dumpFile << outType + "\n";
    dumpFile << convPattern + "\n";
    dumpFile << dedwPattern + "\n";
    dumpFile << bgemmPattern + "\n";
    dumpFile << geometry + "\n";

    dumpFile.close();
}
}  // namespace MmeCommon