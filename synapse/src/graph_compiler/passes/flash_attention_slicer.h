#pragma once

#include "habana_graph.h"

// Slice SDPA nodes on outer batch dim as a temp solution until it will be implemented in cguid.
class FlashAttentionSlicer
{
public:
    FlashAttentionSlicer(HabanaGraph& graph) : m_graph(graph) {}

    void sliceFlashAttentionNodes();

private:
    bool       matchPattern(const NodePtr& node) const;
    TSize      numSlices(const NodePtr& node) const;
    NodeVector sliceNode(const NodePtr& node) const;
    NodeVector splitInputs(const NodePtr& node, const NodeVector& slicedNodes) const;
    NodeVector concatOutputs(const NodePtr& node, const NodeVector& slicedNodes) const;

    HabanaGraph&                     m_graph;
    const std::array<std::string, 2> m_guids     = {"sdpa_fwd_bf16", "sdpa_bwd_bf16"};
    const unsigned                   m_slicedDim = DIM_B;
};