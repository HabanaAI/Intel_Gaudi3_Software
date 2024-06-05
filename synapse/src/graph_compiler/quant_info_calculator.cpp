#include "habana_graph.h"
#include "habana_pass.h"
#include "json_utils.h"
#include "quantization_utils.h"
#include "quant_info_calculator.h"
#include "quantization_data.h"
#include "synapse_common_types.h"

using namespace QuantizationUtils;

const eQuantDataType QuantInfoCalculator::supportedDTypes[NUM_SUPPORTED_QUANTIZATION_TYPES] = {quant_type_fp8_143,
                                                                                               quant_type_fp8_152};

QuantizationMap QuantInfoCalculator::basicQuantizationMap(const TensorPtr& tensor)
{
    QuantizationMap basicQuantMap;
    for (eQuantDataType dtype : QuantInfoCalculator::supportedDTypes)
    {
        basicQuantMap[dtype] = QuantizationData();
        basicQuantMap[dtype].reset(1, dtype);
        basicQuantMap[dtype].setDefaultExpBias();
    }

    return basicQuantMap;
}

void QuantInfoCalculator::calcWeightMMEQuantInfo(const NodePtr node, bool perChannel, bool isGlobalSparsityWeights,
                                                 TensorVector& tensorsToHandle, double backoffFactor)
{
    TensorPtr weightTensor = node->getInput(TENSOR_WEIGHT);
    HB_ASSERT_PTR(weightTensor);

    // Quant info for pre-quantized tensors should be provided by the user, no need to re-calculate.
    if (isQuantDtype(weightTensor->getElementType()) && weightTensor->isDataTypeMatchData()) return;

    if (!weightTensor->getPerChannelDynamicRange().isSet && weightTensor->isStaticParam() && perChannel)
    {
        // A static tensor that was added after calcDynamicRange and hence missing its per channel dynamic range
        calcWeightMMEPerChannelDynamicRange(node);
    }

    bool isSparsityWeights = isGlobalSparsityWeights || weightTensor->isSparsityWeights();
    if (weightTensor->getPerChannelDynamicRange().isSet)
    {
        for (eQuantDataType dtype : supportedDTypes)
        {
            QuantizationData quantInfo    = weightTensor->getQuantizationParams(dtype);
            calcWeightScales(node, quantInfo, backoffFactor, isSparsityWeights);
            weightTensor->setQuantizationParams(quantInfo);
        }
        weightTensor->setPerChannelQuant(true, true);
    }
    else
    {
        calcTensorGlobalQuantInfo(weightTensor, backoffFactor, isSparsityWeights);
        weightTensor->setPerChannelQuant(false);
    }

    removeTensorFromTensorList(tensorsToHandle, weightTensor);
}

void QuantInfoCalculator::calcGlobalQuantInfoPerDType(TensorPtr tensor,
                                                      QuantizationData& quantInfo,
                                                      double backoffFactor,
                                                      bool isSparsityWeights)
{
    if (tensor == nullptr) return;

    quantInfo.reset(1, quantInfo.m_qDataType);

    if (!tensor->getDynamicRange().isSet)
    {
        HB_ASSERT(!GCFG_ENABLE_CALC_DYNAMIC_RANGE.value() || !tensor->isStaticParam(),
                  "Dynamic range for static tensor {} shall be calculated in calcDynamicRange pass.",
                  tensor->getName());

        LOG_DEBUG(QUANT,
                    "Cannot calculate quantization info for dynamic tensor {}, dtype={}, using defaults",
                    tensor->getName(),
                    QuantizationData::getDataTypeString(quantInfo.m_qDataType));

        quantInfo.setDefaultExpBias();
        tensor->setQuantizationParams(quantInfo);
        return;
    }
    double minVal = tensor->getDynamicRange().min;
    double maxVal = tensor->getDynamicRange().max;
    float  absMax = std::max(std::abs(minVal), std::abs(maxVal));
    if (!calcExpBias(absMax,
                     quantInfo.m_qDataType,
                     quantInfo.getChannelParams(0).expBias,
                     backoffFactor))
    {
        LOG_WARN(QUANT, "Cannot calculate quantization info for tensor {}, dtype={}",
                 tensor->getName(), QuantizationData::getDataTypeString(quantInfo.m_qDataType));
        return;
    }
    QuantizationUtils::calcScaleForFp8_143(absMax, quantInfo, backoffFactor);
    LOG_TRACE(QUANT,
              "Calculated global quantization for tensor {} in dtype={}: expBias={}, drange=({},{}), scale={}",
              tensor->getName(),
              QuantizationData::getDataTypeString(quantInfo.m_qDataType),
              quantInfo.expBias(),
              tensor->getDynamicRange().min,
              tensor->getDynamicRange().max,
              quantInfo.scale());
    tensor->setQuantizationParams(quantInfo);
}

void QuantInfoCalculator::calcTensorGlobalQuantInfo(TensorPtr tensor, double backoffFactor, bool isSparsityWeights)
{
    if (tensor == nullptr)
    {
        return;
    }

    for (eQuantDataType dtype : supportedDTypes)
    {
        QuantizationData quantInfo = tensor->getQuantizationParams(dtype);

        if (quantInfo.m_isUserQuantInfo)
        {
            LOG_TRACE(QUANT,
                      "Quantization info was provided by user for tensor {} in dtype={}: expBias={}, scale={}",
                      tensor->getName(),
                      QuantizationData::getDataTypeString(quantInfo.m_qDataType),
                      quantInfo.expBias(),
                      quantInfo.scale());
        }
        else if (quantInfo.isPerChannel())
        {
            LOG_DEBUG(QUANT,
                      "Quantization for tensor {} is calculated Per-Channel, will not calculate globally.",
                      tensor->getName());
        }
        else
        {
            LOG_TRACE(QUANT, "Calculate quantization info dtype {} for {} tensor",
                      QuantizationData::getDataTypeString(quantInfo.m_qDataType), tensor->getName());
            calcGlobalQuantInfoPerDType(tensor, quantInfo, backoffFactor, isSparsityWeights);
        }
    }
}

bool QuantInfoCalculator::initQuantParamsLoadingMode()
{
    const auto scalesFilesPath  = GCFG_QUANTIZATION_PARAMS_PATH.getValueStr();
    if (scalesFilesPath != std::string())
    {
        m_quantizationParamsPath = scalesFilesPath;
        json_utils::Json json = json_utils::jsonFromFile(m_quantizationParamsPath);
        if (json.empty())
        {
            LOG_DEBUG(QUANT, "Json quantization params file {} is empty", m_quantizationParamsPath);
            return false;
        }
        LOG_DEBUG(QUANT, "Initializing mode for quantization params file {}", m_quantizationParamsPath);
        for (auto& jsonIter : json_utils::Json::iterator_wrapper(json))
        {
            if (jsonIter.key() == "Mode")
            {
                m_mode = jsonIter.value();
                break;
            }
        }
        if (m_mode == "")
        {
            LOG_WARN(QUANT, "Failed to load the mode from quantization params file {}", m_quantizationParamsPath);
            return false;
        }
        if (m_mode != "DynamicRange" && m_mode != "Scale")
        {
            LOG_WARN(QUANT, "The mode {} is not supported in the calcQuantInfo pass", m_mode);
            return false;
        }
    }
    return true;
}

bool QuantInfoCalculator::loadQuantParamsByMode(const HabanaGraph& graph)
{
    if (m_mode == "Scale")
    {
        if (!loadQuantizationParams(graph, m_quantizationParamsPath, m_mode))
        {
            LOG_WARN(QUANT, "Failed to load quantization params from files in path {}", m_quantizationParamsPath);
            return false;
        }
        LOG_DEBUG(QUANT, "Finished loading quantization params from json file");
    }
    return true;
}

bool QuantInfoCalculator::tryLoadQuantParamsFromFile(const HabanaGraph& graph)
{
    return initQuantParamsLoadingMode() && loadQuantParamsByMode(graph);
}

bool QuantInfoCalculator::runCalcQuantInfo(HabanaGraph& g)
{
    TensorVector tensorsToHandle;
    for (TensorPtr t : g.getTensors())
    {
        if (t != nullptr)
        {
            tensorsToHandle.push_back(t);
        }
    }

    // Only for MME and Filter2d nodes
    bool isGlobalSparsityWeights = GCFG_ENABLE_SPARSITY_WEIGHTS.value();
    LOG_TRACE(QUANT, "Global sparsity weights quantization is {}abled, "
                     "(ENABLE_SPARSITY_WEIGHTS={})",
              (isGlobalSparsityWeights ? "en" : "dis"), (isGlobalSparsityWeights ? "true" : "false"));
    const NodeVector graphNodes = g.getExeSortedNodes();
    double backoffFactor = g.getTraits().backoffFactor();
    for (const NodePtr& n : graphNodes)
    {
        if (!shouldQuantizeAsMMENode(g, n))
        {
            continue;
        }
        // calc quant params for MME/filter2d static inputs,
        // possibly with per-channel and sparsity, and remove from tensorsToHandle list.
        calcWeightMMEQuantInfo(n, canApplyPerChannelQuant(g, n), isGlobalSparsityWeights, tensorsToHandle, backoffFactor);
    }

    for (TensorPtr t : tensorsToHandle)
    {
        //calc quant params for all other tensors and remove from tensorsToHandle list -
        if (!t->isStaticParam() || (t->isStaticParam() && !t->isDataTypeMatchData()) ||
            t->getDynamicRange().isSet || (t->isStaticParam() && t->getBufferDataType() == syn_type_float))
        {
            calcTensorGlobalQuantInfo(t, backoffFactor);
        }
    }

    for (const TensorPtr& t : g.getTensors())
    {
        if (t == nullptr) continue;
        t->setMeasuredQuantizationParams();
    }

    //load quanitazation params from file in scale mode
    if(!tryLoadQuantParamsFromFile(g)) return false;

    return true;
}
