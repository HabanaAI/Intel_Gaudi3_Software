#pragma once

#include "fmt-9.1.0/include/fmt/core.h"
#include "json.hpp"
#include "habana_graph.h"

#include <fstream>

std::string_view getStringFromSynDataType(synDataType type);

inline void dumpJsonToFile(nlohmann_hcl::json& jsonObject, bool isPerChannel, std::string_view fileName)
{
    std::string jsonPath = fmt::format("{}{}", fileName, isPerChannel ? "_pc_quant_info.json" : "_quant_info.json");
    LOG_TRACE(QUANT, "{} Writing quant info to Json file {}", HLLOG_FUNC, jsonPath);

    std::ofstream jsonFile(jsonPath);
    if (jsonFile)
    {
        jsonFile << jsonObject.dump(4);
    }
    jsonFile.close();
    if (!jsonFile)
    {
        LOG_ERR(QUANT, "Cannot dump quant info to file {}", jsonPath);
    }
}

inline void dumpQuantInfoToJson(const HabanaGraph& g, std::string_view jsonFileName = "")
{
    LOG_DEBUG(QUANT, "{} - Started", HLLOG_FUNC);

    if (jsonFileName.empty())
    {
        jsonFileName = g.getRecipeName();
    }

    nlohmann_hcl::json globalJson;
    nlohmann_hcl::json perChannelJson;

    for (auto& tensor: g.getTensors())
    {
        if (tensor == nullptr) continue;
        std::string tensorName = tensor->getName();
        LOG_TRACE(QUANT, "{} start dumping tensor {} quant info to json", HLLOG_FUNC, tensorName);

        nlohmann_hcl::json tensorJson;
        tensorJson["dtype"] = std::string(getStringFromSynDataType(tensor->getElementType()));

        QuantizationMap tensorQuantInfo = tensor->getAllQuantizationParams();
        auto typeQuantInfo              = tensorQuantInfo[QuantizationData::synTypeToQuantType(tensor->getElementType())];
        if (typeQuantInfo.isPerChannel())
        {
            tensorJson["pc_zps"]       = typeQuantInfo.getZpVector();
            tensorJson["pc_scales"]    = typeQuantInfo.getScaleVector();
            perChannelJson[tensorName] = tensorJson;
        }
        else
        {
            if (typeQuantInfo.numOfParams() > 1)
            {
                LOG_ERR(QUANT, "Not dumping Tensor {}, expected global quant, got channels num {}.",
                        tensorName, typeQuantInfo.m_numChannels);
                continue;
            }
            tensorJson["zp"]       = typeQuantInfo.zp(0);
            tensorJson["scale"]    = typeQuantInfo.scale(0);
            globalJson[tensorName] = tensorJson;
        }
        LOG_TRACE(QUANT, "{} end dumping tensor {} quant info to json", HLLOG_FUNC, tensorName);
    }
    if (!globalJson.empty())
    {
        dumpJsonToFile(globalJson, false, jsonFileName);
    }
    if (!perChannelJson.empty())
    {
        dumpJsonToFile(perChannelJson, true, jsonFileName);
    }
    LOG_DEBUG(QUANT, "{} - Finished", HLLOG_FUNC);
}
