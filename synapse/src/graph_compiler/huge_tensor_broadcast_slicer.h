#pragma once

#include "huge_tensor_node_slicer.h"

class BroadcastNode;

class HugeTensorBroadcastSlicer : public HugeTensorNodeSlicerBase
{
    using LinkedTensors       = std::pair<TensorPtr, TensorVector>;
    using LinkedTensorsVector = llvm_vecsmall::SmallVector<LinkedTensors, EXPECTED_NUM_OF_SLICES>;

public:
    HugeTensorBroadcastSlicer(BroadcastNode* node, const OptionalTensorSplitSuggestion& splitPattern);

    NodeVector slice();

    static bool doesRequireSlicing(BroadcastNode* node);

private:
    static bool         isHugeTensorForBroadcast(BroadcastNode* node, const TensorPtr& t, const NSizeArray& sizes);
    bool         isHugeTensorForBroadcast(const NSizeArray& sizes) const;
    TSize        findOptimalChunkSize(const NSizeArray& chunkSize, const unsigned dim) const;
    LinkedTensorsVector linkOutputsToInputs(const SlicedTensorVector&                inputs,
                                            const SlicedTensorVector&                outputs,
                                            const std::optional<SlicedTensorVector>& shapes) const;
    bool                existsValidSplitPattern() const;

    NodeVector sliceBroadcast();

    BroadcastNode*                m_node;
    OptionalTensorSplitSuggestion m_splitPattern;
    std::vector<bool>             m_isBroadcastedDim;
};