#pragma once

#include "graph_compiler/types.h"
#include "tensor.h"
#include "node.h"

class HabanaGraph;

// Extract a sub-graph from MME node if it has bias
// This logic should be implementd as override at class MultiNode derivative:
//     virtual NodeList extract(const HabanaGraph&)
class MmeBiasNodeHandler
{
public:
    MmeBiasNodeHandler(const NodePtr& node) : m_node(node), m_bias(node->getInput(TENSOR_BIAS)) {}
    static bool canExtract(const NodePtr& node)
    {
        const TensorPtr& bias = node->getInput(TENSOR_BIAS);
        return !(bias == nullptr || bias->isShapeTensor() || node->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM);
    }
    NodePtr extract();                    // Eager mode version, returns the new node and modifies the I/O of BGEMM
    bool    extract(HabanaGraph& graph);  // Graph mode version, takes care of everything

private:
    void               calcExtract();
    static std::string getAddFwdGuid(synDataType type);

private:
    const NodePtr& m_node;
    TensorPtr      m_bias      = nullptr;
    TensorPtr      m_mmeOutput = nullptr;
    NodePtr        m_addNode   = nullptr;  // The new node
};