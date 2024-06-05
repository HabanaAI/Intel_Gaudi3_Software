#pragma once

#include "types.h"
#include "tile.h"

namespace gc::layered_brain
{
class ReducedDimsDetector
{
public:
    using Dim = gc::access_pattern::Dim;

    ReducedDimsDetector(const NodePtr& node);

    std::unordered_set<Dim> getReducedNodeDims() const;
    std::unordered_set<Dim> getReducedNodeDimsForOutput(const TensorPtr& output) const;

protected:
    // Return node dims which are missing in the output
    std::unordered_set<Dim> getMissingNodeDimsInTensor(const TensorPtr& tensor) const;

    // Check if the output is RMW for TPC node
    bool isOutputRmw(const TensorPtr& output) const;

    const NodePtr           m_node;
    std::unordered_set<Dim> m_allSliceableNodeDims;
};

}  // namespace gc::layered_brain