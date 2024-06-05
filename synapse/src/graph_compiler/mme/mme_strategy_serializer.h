#pragma once

#include "defs.h"
#include "graph_compiler/mme/mme_utils.h"
#include "graph_serialize/include/graph_serializer/strategy_serializer.h"
#include "json_utils.h"
#include "include/mme_common/mme_common_enum.h"

#include <map>
#include <optional>
#include <string>

namespace graph_serialize
{
// This class is the frontend for serializing MME strategies
class MmeStrategySerializer final
{
public:
    static void
    processNewStrategy(MmeCommon::MmeStrategy& strategy, const std::string& graphName, const std::string& nodeName)
    {
        static const std::string strategyName = "MME Strategy";
        auto&                    serializer   = graph_serialize::StrategySerializer::getInstance();
        if (serializer.isImportingEnabled())
        {
            const graph_serialize::Json& importedData =
                serializer.getSerializationInfo(strategyName, graphName, nodeName);
            if (!json_utils::isEmpty(importedData))
            {
                importStrategy(strategy, importedData);
            }
        }
        if (serializer.isExportingEnabled())
        {
            graph_serialize::Json& dataToExport =
                serializer.createNewSerializationInfo(strategyName, graphName, nodeName);
            exportStrategy(strategy, dataToExport);
        }
    }

private:
    MmeStrategySerializer() = default;  // No instance is expected to be created

    static void importStrategy(MmeCommon::MmeStrategy& strategy, const graph_serialize::Json& importedData)
    {
        HB_ASSERT(!json_utils::isEmpty(importedData), "Invalid override info");
        // Import geometry
        {
            std::optional<std::string> geoStr = json_utils::get_opt<std::string>(importedData, "geometry");
            if (geoStr.has_value())
            {
                strategy.geometry = mme_utils::strToGeometry(geoStr.value());
            }
        }

        // TODO: Import more strategies here..
    }

    static void exportStrategy(const MmeCommon::MmeStrategy& strategy, graph_serialize::Json& dataToExport)
    {
        HB_ASSERT(json_utils::isEmpty(dataToExport), "Invalid override info");
        dataToExport["geometry"] = mme_utils::toString(strategy.geometry);

        // TODO: Export more strategies here..
    }
};

}  // namespace graph_serialize
