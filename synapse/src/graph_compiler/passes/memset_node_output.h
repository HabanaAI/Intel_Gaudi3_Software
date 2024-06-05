#pragma once

#include "types.h"

#include <tuple>

class HabanaGraph;

class MemsetNodeOutputManager
{
public:
    MemsetNodeOutputManager(HabanaGraph& g, const TensorPtr& tensor);
    std::tuple<TensorPtr, NodePtr, NodePtr> extract(const Node& node);
    NodePtr createMemsetForReduction(const TensorPtr& zerosOutput, const std::string& baseName);

protected:
    virtual std::string getNodeName(std::string_view prefix, std::string_view suffix);
    virtual void        setTensorName(Tensor& tensor, std::string_view prefix, std::string_view suffix);

    HabanaGraph&       m_graph;
    const TensorPtr&   m_tensor;
    bool    m_preferTpc = true;
};

bool memsetNodeOutput(HabanaGraph& g);