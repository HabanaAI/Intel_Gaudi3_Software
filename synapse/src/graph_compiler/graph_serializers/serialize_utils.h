#pragma once

#include "json.hpp"
#include "habana_nodes/node.h"
#include "tensor.h"

namespace graph_serializer
{
template<class TVector>
std::vector<std::string> getTensorsNames(const TVector& tensors, bool skipNullTensors)
{
    static_assert(std::is_same<typename TVector::value_type, Tensor*>::value ||
                      std::is_same<typename TVector::value_type, TensorPtr>::value,
                  "getTensorsNames supports only Tensor* and TensorPtr");
    std::vector<std::string> ret;
    for (const auto& t : tensors)
    {
        const std::string name = t ? t->getName() : "";
        if (!name.empty() || !skipNullTensors)
        {
            ret.push_back(name);
        }
    }
    return ret;
}

std::vector<uint64_t> getStrides(const Tensor* tensor);

std::vector<uint8_t> getPermutation(const Tensor* tensor);

std::vector<char> getConstData(const Tensor* tensor);

void serializeQuantParams(nlohmann_hcl::json& tensor, const Tensor* t);

// TODO: move to synapse/src/graph_serialize/graph_serializer.cpp when removing DUMP_PRE_GRAPHS
nlohmann_hcl::json serializeTensor(const Tensor* t);

std::set<std::string> getNodeNames(const NodeSet& nodes);

std::string getPerforationDebugInfo(const NodePtr& node);

std::vector<std::string> getMmeRecipeDebugInfo(const HabanaGraph& graph, const NodePtr& node);
}  // namespace graph_serializer