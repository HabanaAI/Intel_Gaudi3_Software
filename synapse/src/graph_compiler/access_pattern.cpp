#include "access_pattern.h"
#include "utils.h"

namespace gc::access_pattern
{
template<typename ContainerType>
ContainerType lcm(const ContainerType& numbers1, const ContainerType& numbers2)
{
    HB_ASSERT(numbers1.size() == numbers2.size(), "LCM inputs sizes should match");
    ContainerType ret = numbers1;
    for (auto i = 0; i < numbers1.size(); i++)
    {
        ret[i] = std::lcm(numbers1[i], numbers2[i]);
    }
    return ret;
}

void NodeAccessPattern::addTensorAccessPattern(const TensorPtr& tensor, const TensorAccessPatternPtr& accessPattern)
{
    m_tensorAccessPatterns[tensor] = accessPattern;
}

TensorTile NodeAccessPattern::getTensorGranularity(const TensorPtr& tensor) const
{
    HB_ASSERT_PTR(tensor);
    auto it = m_tensorAccessPatterns.find(tensor);
    HB_ASSERT(it != m_tensorAccessPatterns.end(), "Failed to find tensor access pattern");
    return it->second->getGranularity();
}

NodeTile NodeAccessPattern::getNodeTile(const TensorPtr& tensor, const TensorTile& tensorTile) const
{
    HB_ASSERT_PTR(tensor);
    auto it = m_tensorAccessPatterns.find(tensor);
    HB_ASSERT(it != m_tensorAccessPatterns.end(), "Failed to find tensor access pattern");
    return it->second->getNodeTile(tensorTile, m_nodeResolution);
}

NodeTile NodeAccessPattern::getLcmNodeTile(const TilePerTensor& tensorTiles) const
{
    NodeTile lcmTile(m_nodeResolution.size(), 1, 0);
    // Init the min granularity to 1 for all dims which are not affected by the tensor tile, so the LCM won't be
    // affected
    NodeTile::Geometry baseResolution = lcmTile.geometry;
    NodeTile::Offset   commonOffset;
    for (auto& tensorAndTile : tensorTiles)
    {
        const TensorPtr& tensor = tensorAndTile.first;
        HB_ASSERT_PTR(tensor);
        HB_ASSERT(tensorAndTile.second.offset == TensorTile::Offset(tensor->getDim(), 0), "expected zero offset for tensor {}", tensor->getName());
        TensorTile fullTensorTile = getTensorTile(tensor, m_fullNodeTile);

        IntersectionTile clipped = intersect(fullTensorTile, tensorAndTile.second);
        TensorTile clippedTensorTile(tensor->getDim(), clipped.geometry, clipped.offset);

        // Init LCM tile to be the full dim when tensor mapping is not exact
        IntersectionTile tensorOverlap = getTensorOverlap(tensor);
        TensorTile       tensorGranule = getTensorGranularity(tensor);

        for (Dim tensorDim = 0; tensorDim != tensorOverlap.geometry.size(); ++tensorDim)
        {
            // check if dim is exact (no padding, overlap)
            // overlap => intersection size != 0, padding => offset != 0
            if (tensorOverlap.geometry[tensorDim] != 0 || tensorGranule.offset[tensorDim] != 0)
            {
                clippedTensorTile.geometry[tensorDim] = tensor->getSizeInElements(tensorDim);
            }
        }

        auto it = m_tensorAccessPatterns.find(tensor);
        HB_ASSERT(it != m_tensorAccessPatterns.end(), "Failed to find tensor access pattern");
        const NodeTile&  nodeTile = it->second->getNodeTile(clippedTensorTile, baseResolution);
        lcmTile.geometry = lcm(nodeTile.geometry, lcmTile.geometry);
        if (commonOffset.empty())
        {
            commonOffset = nodeTile.offset;
        }
        else
        {
            for (auto nodeDim = 0; nodeDim < m_nodeResolution.size(); nodeDim++)
            {
                if (commonOffset.at(nodeDim) != nodeTile.offset.at(nodeDim))
                {
                    // Currently this method is called from LCM calculator with zero offset for
                    // each tensor tile instead of real offsets.
                    // As a result we might get a conflict here for dims with non-strict mapping (AP with offsets).
                    // Since these dims are not expected to be sliced (blocked in the bundlizer) -
                    // their LCM is not relevant and full resolution will be returned to mark them
                    // as unsliceable.
                    // TODO: SW-152071 for clean solution.
                    lcmTile.geometry.at(nodeDim) = m_nodeResolution.at(nodeDim);
                }
            }
        }
    }
    return lcmTile;
}

TensorTile NodeAccessPattern::getTensorTile(const TensorPtr& tensor, const NodeTile& nodeTile) const
{
    validateNodeTile(nodeTile);
    HB_ASSERT_PTR(tensor);
    auto it = m_tensorAccessPatterns.find(tensor);
    HB_ASSERT(it != m_tensorAccessPatterns.end(), "Failed to find tensor access pattern");
    return it->second->getTensorTile(nodeTile);
}

void NodeAccessPattern::validateNodeTile(const NodeTile& nodeTile) const
{
    HB_ASSERT(nodeTile.geometry.size() == m_nodeResolution.size(),
              "Node tile geometry must have the same rank as the index space geometry. Node tile rank: {}, Index space "
              "rank: {}",
              nodeTile.geometry.size(),
              m_nodeResolution.size());

    HB_ASSERT(nodeTile.offset.size() == m_nodeResolution.size(),
              "Node tile offset must have the same rank as the index space geometry. Node tile rank: {}, Index space "
              "rank: {}",
              nodeTile.offset.size(),
              m_nodeResolution.size());
}

IntersectionTile NodeAccessPattern::getTensorOverlap(const TensorPtr& tensor) const
{
    // 1x1x1x.. tile at offset 0x0x0x...
    NodeTile firstNodeTile(NodeTile::Geometry(m_nodeResolution.size(), 1),
                           NodeTile::Offset(m_nodeResolution.size(), 0));
    const NodeTile& nextNodeTile = getNextNodeGranule(firstNodeTile);

    const TensorTile& firstTensorTile = getTensorTile(tensor, firstNodeTile);
    const TensorTile& nextTensorTile  = getTensorTile(tensor, nextNodeTile);

    return intersect(firstTensorTile, nextTensorTile);
}

// Advance the offset of _all_ the dimensions that didn't reach their limit
NodeTile NodeAccessPattern::getNextNodeGranule(const NodeTile& nodeGranule) const
{
    NodeTile nextGranule = nodeGranule;
    auto&    offset      = nextGranule.offset;
    for (Dim dim = 0; dim < offset.size(); dim++)
    {
        // Do not advance offset of dimensions that reached the limit (max offset is dimSize - 1)
        if (offset[dim] < m_nodeResolution[dim] - 1)
        {
            offset[dim]++;
        }
    }
    return nextGranule;
}

MultiDims NodeAccessPattern::getTensorMatchingSlicedDims(const TensorPtr& queriedTensor,
                                                         const TensorPtr& givenTensor,
                                                         Dim              givenSlicingDim) const
{
    // If the full node tile does not map to the entire queried tensor, some dimensions may seem sliced, although they
    // are not. To find actual sliced dimensions, need to compare to the actual range of the tensor that's mapped from a
    // full node resolution tile.
    const TensorTile& effectiveQueriedTensorFullTile = getTensorTile(queriedTensor, m_fullNodeTile);

    // Create a tile that covers a single granule at the given slicing dim and is full in the other dims
    const TensorTile& givenTensorGranularity  = getTensorGranularity(givenTensor);
    TensorTile        givenTensorTile         = getTensorTile(givenTensor, m_fullNodeTile);
    givenTensorTile.geometry[givenSlicingDim] = givenTensorGranularity.geometry[givenSlicingDim];
    givenTensorTile.offset[givenSlicingDim]   = givenTensorGranularity.offset[givenSlicingDim];

    // Map the created given tensor tile to a queried-tensor tile
    const NodeTile&   nodeTile          = getNodeTile(givenTensor, givenTensorTile);
    const TensorTile& queriedTensorTile = getTensorTile(queriedTensor, nodeTile);

    // Collect the dimensions in which the mapped queried tensor tile is smaller than the effective full queried tensor
    // tile.
    MultiDims matchingDims;
    for (Dim dim = 0; dim < queriedTensor->getDim(); ++dim)
    {
        if (queriedTensorTile.geometry[dim] < effectiveQueriedTensorFullTile.geometry[dim])
        {
            matchingDims.push_back(dim);
        }
    }
    return matchingDims;
}

Dim NodeAccessPattern::getIndexSpaceDim(const TensorPtr& tensor, Dim tensorDim) const
{
    HB_ASSERT_PTR(tensor);
    HB_ASSERT(tensorDim < tensor->getDim(),
              "Invalid tensor dimension {}, tensor rank is {}",
              tensorDim,
              tensor->getDim());
    auto it = m_tensorAccessPatterns.find(tensor);
    HB_ASSERT(it != m_tensorAccessPatterns.end(), "Failed to find tensor access pattern");
    Dim indexSpaceDim = it->second->getIndexSpaceDim(tensorDim);
    HB_ASSERT(indexSpaceDim < m_nodeResolution.size(),
              "Invalid index-space dim {}, node resolution rank is {}",
              indexSpaceDim,
              m_nodeResolution.size());
    return indexSpaceDim;
}

MultiDims NodeAccessPattern::getTensorDims(Dim indexSpaceDim, const TensorPtr& tensor) const
{
    MultiDims tensorDims;
    for (auto tensorDim = 0; tensorDim < tensor->getDim(); ++tensorDim)
    {
        const auto mappedIndexSpaceDim = getIndexSpaceDim(tensor, tensorDim);
        if (mappedIndexSpaceDim == indexSpaceDim)
        {
            tensorDims.push_back(tensorDim);
        }
    }
    return tensorDims;
}

bool NodeAccessPattern::hasAccessPattern(const TensorPtr& tensor) const
{
    return m_tensorAccessPatterns.find(tensor) != m_tensorAccessPatterns.end();
}

}  // namespace gc::access_pattern