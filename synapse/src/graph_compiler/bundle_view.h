#pragma once

#include "types.h"
#include "access_pattern.h"  // For Dim

using namespace gc::access_pattern;

// The ‘Bundle View’ concept saves the need to project a single node slicing onto the rest of the bundle nodes.
// This concept uses access pattern mapping of dimensions to tie together dimensions of different tensors that
// are sliced and traversed together, since they’re mapped to each other.
// This construct causes slicing decision to one node to immediately apply to all other required (strict) slicing to
// achieve the correct division across the bundle.

using GranularityPerTensorDim = std::map<std::pair<TensorPtr, Dim>, TensorTile::Size>;
using GranularityPerNodeDim   = std::map<std::pair<NodePtr, Dim>, NodeTile::Size>;

using BundleViewId = uint32_t;
using BVDSet       = std::unordered_set<BundleViewId>;

struct BundleView
{
    BundleViewId id;
    // For all these tensor/node dimensions, the granularity of slicing would be aggregated by the common ISMR LCM
    // procedure, and that a single granularity multiplier is enough to express the slicing of all of them.
    GranularityPerTensorDim tensorDimsGranularity;
    GranularityPerNodeDim   nodeDimsGranularity;

    uint64_t resolution;  // Max granularity multiplier
};

using BVDPerTensorDim = std::map<std::pair<TensorPtr, Dim>, BundleViewId>;
using BVDPerNodeDim   = std::map<std::pair<NodePtr, Dim>, BundleViewId>;

class BundleViewContainer
{
public:
    BundleViewContainer(uint32_t numOfBundleViews);
    void              logBundleViews() const;
    uint32_t          getNumOfBundleViews() const;
    const BundleView& getBundleView(BundleViewId bvdId) const;
    BundleViewId      getBVDForTensorDim(const TensorPtr& tensor, Dim tensorDim) const;
    BundleViewId      getBVDForNodeDim(const NodePtr& node, Dim nodeDim) const;
    bool              isNodeDimMappedToBVD(const NodePtr& node, Dim nodeDim) const;
    bool              isTensorMappedToBVD(const TensorPtr& tensor, BundleViewId bvdId) const;
    std::vector<Dim>  getNodeDimsInBVD(BundleViewId bvdId, const NodePtr& node) const;
    BVDSet            getNodesBVDs(const NodeVector& nodes) const;
    TensorTile::Size  getGranularityForTensorDim(const TensorPtr& tensor, Dim tensorDim) const;
    NodeTile::Size    getGranularityForNodeDim(const NodePtr& node, Dim nodeDim) const;
    void mapTensorDimToBVD(const TensorPtr& tensor, Dim tensorDim, BundleViewId bvdId, TensorTile::Size granularity);
    void mapNodeDimToBVD(const NodePtr& node, Dim nodeDim, BundleViewId bvdId, NodeTile::Size granularity);
    BVDSet getBvdsForNodeDims(const NodePtr& node, const std::unordered_set<Dim>& nodeDims) const;

private:
    std::vector<BundleView> m_bundleViews;
    BVDPerTensorDim         m_tensorDimToBvd;
    BVDPerNodeDim           m_nodeDimToBvd;
};

using BundleViewContainerPtr = std::shared_ptr<BundleViewContainer>;