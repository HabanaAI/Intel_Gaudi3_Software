
#pragma once

#include "brain_data.h"
#include "strategy.h"
#include "types.h"

namespace gc::layered_brain
{
// The sliced tensor generator is responsible to create a sliced tensor from original tensor and BVD coordinate.
// It maintains a sliced tensor DB to avoid creation of the same tensor twice.
// The slice size and offset is calculated according to BVDs and slicing strategy (assuming a strict mapping - no
// overlap/offset).
class SlicedTensorGenerator
{
public:
    // TODO: SW-120740 - high rank support
    using OffsetArray = ::OffsetArray;
    using SizeArray   = ::SizeArray;

    SlicedTensorGenerator(const BundleIdx bundleIdx) : m_bundleIdx(bundleIdx) {};

    SlicedTensorGenerator() : m_bundleIdx(std::nullopt) {};

    std::pair<TensorPtr, OffsetArray> getSlicedTensor(const NodePtr&   origNode,
                                                      const NodeTile&  slicedNodeTile,
                                                      const TensorPtr& origTensor,
                                                      const BVDCoord&  bvdCoord);

private:
    using BigTensorCoord    = std::pair<TensorPtr, BVDCoord>;
    using SliceTensorOffset = std::pair<TensorPtr, OffsetArray>;  // Offset in tensor elements

    std::pair<SizeArray, OffsetArray>
    calcSliceSizeAndOffset(const NodePtr& origNode, const NodeTile& slicedNodeTile, const TensorPtr& origTensor) const;

    const std::optional<BundleIdx> m_bundleIdx;

    std::map<BigTensorCoord, SliceTensorOffset> m_slicedTensorsDB;
};

}  // namespace gc::layered_brain