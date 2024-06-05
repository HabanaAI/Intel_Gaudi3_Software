#pragma once

#include "graph_compiler/types.h"
#include <memory>

class HabanaGraph;
class BatchGemmNode;

// Extract a sub-graph from MME node if it has broadcast
// This logic should be implementd as override at class MultiNode derivative:
//     virtual NodeList extract(const HabanaGraph&)
class MmeBroadcastBatchGemmNodeHandler
{
public:
    MmeBroadcastBatchGemmNodeHandler(const NodePtr& node);
    bool            canExtract() const;
    // Eager mode version, returns the new node and modifies the I/O of BGEMM
    const std::pair<NodePtr, NodeVector> extract();
    bool            extract(HabanaGraph& graph);  // Graph mode version, takes care of everything

private:
    using CrossDimArray = std::array<bool, SYN_MAX_TENSOR_DIM>;

    void                         calcExtract();
    static std::pair<bool, bool> calcBroadcastNodeOutDims(const SizeArray& opASizes,
                                                          const SizeArray& opBSizes,
                                                          SizeArray&       bcastAOutSizes,
                                                          SizeArray&       bcastBOutSizes,
                                                          CrossDimArray&   A2BIndices,
                                                          CrossDimArray&   B2AIndices);
    void                         reshapeForExplicitBroadcast(const TensorPtr& tensor);

private:
    const std::shared_ptr<BatchGemmNode> m_node;
    NodeVector                           m_newNodes;
    NodePtr                              m_newBatchGemmNode;
};