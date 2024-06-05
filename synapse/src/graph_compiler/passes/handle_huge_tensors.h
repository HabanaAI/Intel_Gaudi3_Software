#pragma once

#include "huge_tensor_slicer.h"
#include "types.h"

class HabanaGraph;
class HugeTensorHandler
{
public:
    HugeTensorHandler(const HabanaGraph& graph) : m_graph(graph), m_hugeTensorSlicer() {}
    void handleHugeTensors(HabanaGraph& g);

    // may be used directly per node (for eager)
    bool       shouldHandleHugeTensor(const NodePtr& n);
    NodeVector extractNodeWithHugeTensors(const NodePtr& n);

private:
    OptionalTensorSplitSuggestion generateSlicingHint(const HabanaGraph& g, const NodePtr& n) const;
    std::optional<NSizeArray>
    findChunkSizeOfAggregatedTensor(const HabanaGraph& g, const TensorPtr& tensor, bool checkForSplit) const;

    const HabanaGraph& m_graph;
    HugeTensorSlicer   m_hugeTensorSlicer;
};