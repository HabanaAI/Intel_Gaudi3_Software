#pragma once

#include "defs.h"
#include "tile.h"
#include "types.h"
#include "tensor.h"
#include <unordered_map>

namespace gc::access_pattern
{
struct TensorAccessPattern
{
    virtual ~TensorAccessPattern() = default;

    // Get a minimal size tile in the offset of the first tile in the node geometry
    virtual TensorTile getGranularity() const = 0;

    // Apply the access pattern mapping to the node tile and return the tensor tile that results.
    virtual TensorTile getTensorTile(const NodeTile& nodeTile) const = 0;

    // This query assumes that the tensor tile was obtained using the granularity provided by getGranularity.
    // Since the nodeTile -> tensorTile mapping is not always invertible, the full node resolution is used for
    // dimensions that can't be deduced from the tensor tile.
    virtual NodeTile getNodeTile(const TensorTile& tensorTile, const NodeTile::Geometry& nodeResolution) const = 0;

    // Get the index-space dimension that this tensor dimension is mapped to.
    virtual Dim getIndexSpaceDim(Dim tensorDim) const = 0;
};
typedef std::shared_ptr<TensorAccessPattern> TensorAccessPatternPtr;

class NodeAccessPattern final
{
public:
    using Resolution = NodeTile::Geometry;
    using TilePerTensor = std::unordered_map<TensorPtr, TensorTile>;

    template<typename Iter>
    explicit NodeAccessPattern(Iter resolutionGeometryBegin, const Iter& resolutionGeometryEnd)
    : m_nodeResolution(resolutionGeometryBegin, resolutionGeometryEnd), m_fullNodeTile(m_nodeResolution)
    {
    }

    void addTensorAccessPattern(const TensorPtr& tensor, const TensorAccessPatternPtr& accessPattern);

    // Get a virtual work division space that can be mapped to tensor regions for work distribution (index space)
    const Resolution& getNodeResolution() const { return m_nodeResolution; }

    // Get the minimal tensor tile. When slicing the tensor, it is expected that the slices would be multiples of this
    // tile.
    TensorTile getTensorGranularity(const TensorPtr& tensor) const;

    // Get the tensor tile used for the part of the work that's described by the node resolution tile.
    TensorTile getTensorTile(const TensorPtr& tensor, const NodeTile& nodeTile) const;

    // Given a tensor tile SLICED TO MULTIPLES OF THE GRANULARITY, get the minimal node resolution tile which works on
    // this tensor tile.
    NodeTile getNodeTile(const TensorPtr& tensor, const TensorTile& tensorTile) const;

    // Given a set of tensors and their tensor tile SLICED TO MULTIPLES OF THE GRANULARITY, get the node resolution tile
    // which works on the LCM of each input tensor tile.
    NodeTile getLcmNodeTile(const TilePerTensor& tensorTiles) const;

    // Tensor tile intersection between adjacent tensor granules.
    IntersectionTile getTensorOverlap(const TensorPtr& tensor) const;

    // Given a tensor and a dimension on which it is split/sliced return the dimensions in a different tensor that
    // should be splitted/sliced accordingly
    MultiDims getTensorMatchingSlicedDims(const TensorPtr& queriedTensor,
                                          const TensorPtr& givenTensor,
                                          Dim              givenSlicingDim) const;

    // Given a tensor and a dimension return the index-space dimension that this tensor dimension is mapped to.
    Dim getIndexSpaceDim(const TensorPtr& tensor, Dim tensorDim) const;

    // Given an index-space dimension and a tensor return the tensor's dimensions that this node dim is mapped to.
    // If the tensor has no mapped dims, the return value will be empty.
    MultiDims getTensorDims(Dim indexSpaceDim, const TensorPtr& tensor) const;

    // Query to check if a tensor is mapped to an access pattern in this node's access patterns map.
    bool hasAccessPattern(const TensorPtr& tensor) const;

private:
    Resolution m_nodeResolution;
    NodeTile   m_fullNodeTile;  // Useful shortcut to creating a node tile from the resolution

    std::map<TensorPtr, TensorAccessPatternPtr, TensorComparator> m_tensorAccessPatterns;

    void     validateNodeTile(const NodeTile& nodeTile) const;
    NodeTile getNextNodeGranule(const NodeTile& nodeGranule) const;
};
typedef std::shared_ptr<NodeAccessPattern> NodeAccessPatternPtr;

}  // namespace gc::access_pattern