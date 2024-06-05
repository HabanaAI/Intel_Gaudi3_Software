#pragma once
#include "types.h"

#include <optional>

class HabanaGraph;
struct TensorSplitSuggestion
{
    TensorPtr  tensor;
    NSizeArray chunkSize;
};
using OptionalTensorSplitSuggestion = std::optional<TensorSplitSuggestion>;

class HugeTensorSlicer
{
public:
    HugeTensorSlicer() {}

    // return 'true' if 'node' has huge operands and needs slicing
    bool doesRequireSlicing(const NodePtr& node);

    // return the sliced sub-graph created out of 'node'. use 'splitSuggestion' when possible
    NodeVector sliceNode(const NodePtr& node, const OptionalTensorSplitSuggestion& splitSuggestion = std::nullopt);
};