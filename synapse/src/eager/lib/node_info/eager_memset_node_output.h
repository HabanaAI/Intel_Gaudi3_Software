#pragma once

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"

// synapse-internal passes includes (relative to src/)
#include "graph_compiler/passes/memset_node_output.h"
#include "include/tensor.h"

namespace eager_mode
{

class EagerMemsetNodeOutputManager : public MemsetNodeOutputManager
{
public:
    EagerMemsetNodeOutputManager(EagerGraph& g, const TensorPtr& tensor) : MemsetNodeOutputManager(g, tensor) {}
    std::string getNodeName(std::string_view /*prefix*/, std::string_view /*suffix*/) override
    {
        return std::string(static_cast<eager_mode::EagerGraph&>(m_graph).getNextNodeName());
    }
    void setTensorName(Tensor& tensor, std::string_view /*prefix*/, std::string_view /*suffix*/) override
    {
        tensor.setName(static_cast<eager_mode::EagerGraph&>(m_graph).getNextTensorName(), true);
    }
};

}  // namespace eager_mode
