#include "tpc_json_serializer.h"
#include "habana_graph.h"
#include "types.h"
#include <cstddef>

template<class T>
static std::vector<uint8_t> getRawData(T* obj, int size)
{
    const uint8_t* p = (uint8_t*)(obj);
    return std::vector<uint8_t>(p, p + size);
}

template<class T>
static std::vector<uint8_t> getRawData(const T& obj)
{
    return getRawData(&obj, sizeof(T));
}

nlohmann_hcl::json TpcJsonSerializer::serialize(const HabanaGraph& habanaGraph)
{
    std::vector<nlohmann_hcl::json> nodes;
    for (const auto& n : habanaGraph.getNodes())
    {
        if (!HabanaGraph::runsOnTPC(n)) continue;

        TPCNode&    tpcNode                = *static_cast<TPCNode*>(n.get());
        auto&       jNode                  = nodes.emplace_back();
        const auto& instance               = tpcNode.getInstance();
        const auto& glueParams             = tpcNode.getSucceededGlueParams();
        auto        nodeParams             = glueParams.nodeParams;
        jNode["guid"]                      = tpcNode.getGUID();
        jNode["name"]                      = tpcNode.getNodeName();
        jNode["HabanaKernelParams"]        = getRawData(glueParams);
        jNode["HabanaKernelInstantiation"] = getRawData(instance);
        jNode["kernel"]                    = getRawData(instance.kernel);
        jNode["kernelElf"]                 = getRawData(instance.kernel.kernelElf, instance.kernel.elfSize);
        jNode["elfSize"]                   = getRawData(instance.kernel.elfSize);
        jNode["nodeParams"]                = getRawData(nodeParams.nodeParams, nodeParams.nodeParamsSize);
        jNode["nodeParamsSize"]            = getRawData(glueParams.nodeParams.nodeParamsSize);
        jNode["userParams"]                = tpcNode.getParamsRawData();
    }
    nlohmann_hcl::json ret;
    ret["kernelElfoffsetInKernelInst"] = offsetof(gcapi::HabanaKernelInstantiation_t, kernelElf);
    ret["nodes"]                       = nodes;
    return ret;
}
