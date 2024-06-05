#pragma once

#include "huge_tensor_node_slicer.h"
#include "transpose_permutation.h"

class TransposeNode;

class HugeTensorTransposeSlicer : public HugeTensorNodeSlicerBase
{
public:
    HugeTensorTransposeSlicer(TransposeNode* node, const OptionalTensorSplitSuggestion& splitPattern)
    : m_node(node), m_splitPattern(splitPattern)
    {
    }
    NodeVector slice();

    static bool doesRequireSlicing(TransposeNode* node);

private:
    static bool isHugeTensorForTranspose(TransposeNode* node, const TensorPtr& t, const NSizeArray& sizes);
    bool isHugeTensorForTranspose(const NSizeArray& sizes) const;

    // Since transpose input and output dimensions order is different, each dimension value is min(input, output)
    // so the split order is from "most" outer to inner
    static DimVector getTransposeSplitOrder(const TransposePermutationArray& permutation);
    TSize            findOptimalChunkSize(const NSizeArray& chunkSize, const unsigned dim) const;

    NodeVector sliceTranspose();

    TransposeNode*                m_node;
    OptionalTensorSplitSuggestion m_splitPattern;
};
