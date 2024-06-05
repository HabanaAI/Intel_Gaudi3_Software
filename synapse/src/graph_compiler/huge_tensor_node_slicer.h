#pragma once

#include "huge_tensor_slicer.h"
#include <functional>

class HugeTensorNodeSlicerBase
{
protected:
    HugeTensorNodeSlicerBase() {}

    struct SlicedTensor
    {
        TensorPtr   tensor;
        NCoordArray coordinates;

        bool operator<(const SlicedTensor& o) { return coordinates < o.coordinates; }  // Ignore "tensor"
    };
    using SlicedTensorVector = std::vector<SlicedTensor>;

    // Given chunk size, tensor and direction (concat/split) it return "sub-graph" of aggregation nodes,
    // where the aggregation is from the outer dimension to the inner dimension.
    // The "sub-graph" edges are the original tensor and the returned tensors.
    static std::pair<NodeVector, SlicedTensorVector> sliceTensor(const TensorSplitSuggestion& suggestion, bool isInput);
    static NStrideArray calculateDefaultStrides(const unsigned elementSize, const NSizeArray& sizes);

    static TSize findOptimalChunkSize(const NSizeArray&                        chunkSize,
                                      const unsigned                           dim,
                                      const std::function<bool(const TSize&)>& isValidChuckSize);

    static constexpr unsigned EXPECTED_NUM_OF_SLICES = 4;
};
