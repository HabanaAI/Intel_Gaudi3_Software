#pragma once

#include "node.h"

using namespace gc::access_pattern;
using TileSizePerTensor = std::map<TensorPtr, TensorTile::Geometry>;
using TileSizePerNode   = std::map<NodePtr, NodeTile::Geometry>;

class CommonTileSizeCalculator
{
public:
    static std::pair<TileSizePerTensor, TileSizePerNode> getMinCommonTilesSizes(const NodeSet&     nodesToIntersect,
                                                                                const TensorSet&   tensorsToIntersect,
                                                                                const HabanaGraph& graph);

    static NodeSet   getConnectedNodesInGroup(const TensorPtr& t, const NodeSet& nodes, const HabanaGraph& graph);
    static TensorSet getConnectedTensorsInGroup(const NodePtr& n, const TensorSet& tensors);
    static void      printInputGraph(const NodeSet& nodes, const TensorSet& tensors, const HabanaGraph& graph);
};