
#pragma once

#include "brain_data.h"
#include "habana_graph.h"
#include "types.h"

namespace gc::layered_brain
{
// The bundle persistent tensors handler is responsible to identify the input BPTs and the output BPTs of the bundle.
// For input BPT, a Fork node (read TensorView) will be created to connect all its slices to the original tensor.
// For output BPT, a Join node (write TensorView) will be created to connect all its slices to the original tensor.
class BPTHandler
{
public:
    BPTHandler(const NodePtr& node);

    BPTHandler(const HabanaGraph& graph, const BundleIdx bundleIdx, const NodeVector& bundleNodes);

    bool isBPT(const TensorPtr& tensor) const;

    void addTensorSlice(const TensorPtr& origTensor, const TensorPtr& slicedTensor, const OffsetArray& sliceOffset);

    NodeVector createForkAndJoinNodes() const;

private:
    using TensorSliceOffset = std::pair<TensorPtr, OffsetArray>;

    bool    isInputBPT(const TensorPtr& tensor) const;
    bool    isOutputBPT(const TensorPtr& tensor) const;
    NodePtr createTensorViewNode(const TensorPtr&                   origTensor,
                                 const std::set<TensorSliceOffset>& slicedTensors,
                                 bool                               isRealTensorInput) const;

    const std::optional<BundleIdx> m_bundleIdx;

    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;

    // Map from big tensor to its slices (views)
    TensorToItemOrderedMap<std::set<TensorSliceOffset>> m_forkSlices;
    TensorToItemOrderedMap<std::set<TensorSliceOffset>> m_joinSlices;
};

}  // namespace gc::layered_brain