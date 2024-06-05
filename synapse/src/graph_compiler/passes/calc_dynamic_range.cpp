#include "habana_graph.h"
#include "quantization_utils.h"
#include "json_utils.h"

using namespace QuantizationUtils;

bool isDynamicRangeFileExist(const std::string& dynamicRangeFilePath)
{
    if (dynamicRangeFilePath == std::string()) return false;

    std::string mode;
    json_utils::Json json = json_utils::jsonFromFile(dynamicRangeFilePath);
    if (json.empty())
    {
        LOG_DEBUG(QUANT, "Json quantization params file {} is empty", dynamicRangeFilePath);
        return false;
    }
    LOG_DEBUG(QUANT, "Initializing mode for quantization params file {}", dynamicRangeFilePath);
    for (auto& jsonIter : json_utils::Json::iterator_wrapper(json))
    {
        if (jsonIter.key() == "Mode")
        {
            mode = jsonIter.value();
            break;
        }
    }
    if (mode == "")
    {
        LOG_WARN(QUANT, "Failed to load mode from dynamic ranges file {}", dynamicRangeFilePath);
        return false;
    }
    if (mode != "DynamicRange")
    {
        LOG_DEBUG(QUANT, "The mode {} is not supported for calcDynamicRange pass", mode);
        return false;
    }

    return true;
}

bool calcDynamicRange(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(QUANT,
                  "Calc dynamic range is enabled in synapse only for Inference with Quantization Mode. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    const std::string dynamicRangeFilePath  = GCFG_QUANTIZATION_PARAMS_PATH.getValueStr();
    if (isDynamicRangeFileExist(dynamicRangeFilePath))
    {
        if (!QuantizationUtils::loadQuantizationParams(g, dynamicRangeFilePath, "DynamicRange"))
        {
            LOG_WARN(QUANT, "Failed to load dynamic ranges from files in path {}", dynamicRangeFilePath);
            return false;
        }
        LOG_DEBUG(QUANT, "Finished loading dynamic ranges from json file {}", dynamicRangeFilePath);
    }

    if (!GCFG_ENABLE_CALC_DYNAMIC_RANGE.value())
    {
        LOG_DEBUG(QUANT,
                  "Calc dynamic range for static tensors is disabled. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    for (TensorPtr tensor : g.getTensors())
    {
        if (tensor == nullptr) continue;
        if (!tensor->isStaticParam()) continue;
        if (!tensor->getDynamicRange().isSet)
        {
            calcPerTensorDynamicRange(tensor);
        }
    }
    const NodeVector graphNodes = g.getExeSortedNodes();
    for (const NodePtr& node : graphNodes)
    {
        if (GCFG_PER_CHANNEL_SCALING.value() && shouldQuantizeAsMMENode(g, node))
        {
            calcWeightMMEPerChannelDynamicRange(node);
        }
    }
    return true;
}
