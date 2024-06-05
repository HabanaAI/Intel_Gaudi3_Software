#include "common_tile_size_calculator.h"
#include "habana_graph.h"

template<typename GeometryType>
static void printTileProjection(const GeometryType& projectedTile, const GeometryType& accumulatedTile)
{
    LOG_TRACE(TILE_SIZE_CALC,
              "\t Projected tile ({}) -> Accumulated tile ({}) ",
              toString(projectedTile, ','),
              toString(accumulatedTile, ','));
}

NodeSet
CommonTileSizeCalculator::getConnectedNodesInGroup(const TensorPtr& t, const NodeSet& nodes, const HabanaGraph& graph)
{
    NodeSet connectedNodes;
    for (auto n : graph.getTensorConsumers(t))
    {
        if (nodes.find(n) != nodes.end())
        {
            connectedNodes.insert(n);
        }
    }
    auto prod = graph.getTensorProducer(t);
    if (nodes.find(prod) != nodes.end())
    {
        connectedNodes.insert(prod);
    }
    return connectedNodes;
}

TensorSet CommonTileSizeCalculator::getConnectedTensorsInGroup(const NodePtr& n, const TensorSet& tensors)
{
    TensorSet connectedTensors;
    for (auto t : n->getOperands())
    {
        if (tensors.find(t) != tensors.end())
        {
            connectedTensors.insert(t);
        }
    }
    return connectedTensors;
}

bool validateNodes(const NodeSet& connectedNodes)
{
    for (const auto& n : connectedNodes)
    {
        if (n->getNodeAccessPattern() == nullptr)
        {
            LOG_ERR(TILE_SIZE_CALC, "{}: Node without access pattern found - {}", HLLOG_FUNC, n->getNodeName());
            return false;
        }
    }

    return true;
}

// TODO - unify with the access pattern instance. Problematic in utils.h as it's included from HCL
// which is C++11, and std::lcm is C++17
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

// TODO SW-75960 - refactor maps and create objects
std::pair<TileSizePerTensor, TileSizePerNode>
CommonTileSizeCalculator::getMinCommonTilesSizes(const NodeSet&     connectedNodes,
                                                 const TensorSet&   connectingTensors,
                                                 const HabanaGraph& graph)
{
    printInputGraph(connectedNodes, connectingTensors, graph);
    HB_ASSERT(validateNodes(connectedNodes), "Node without access pattern found, can't handle");

    std::map<NodePtr, NodeTile::Geometry>     isrPerNode;
    std::map<TensorPtr, TensorTile::Geometry> ismrPerTensor;

    // Init nodes ISR to 1
    std::for_each(connectedNodes.begin(), connectedNodes.end(), [&](const NodePtr& n) {
        isrPerNode[n] = NodeTile::Geometry(n->getNodeAccessPattern()->getNodeResolution().size(), 1);
    });

    // Init tensors ISMR to 1
    std::for_each(connectingTensors.begin(), connectingTensors.end(), [&](const TensorPtr& t) {
        ismrPerTensor[t] = TensorTile::Geometry(t->getDim(), 1);
    });

    // Handle a graph of a single node without additional constraints - the node granularity is its index space
    if (connectedNodes.size() == 1 && connectingTensors.empty())
    {
        return {ismrPerTensor, isrPerNode};
    }

    LOG_TRACE(TILE_SIZE_CALC, "Calculate common tile size - Stage 1");

    // Stage 1
    // while (|nodesToHandle| + |tensorsToHandle| > 1)
    //   foreach t: tensorsToHandle
    //      foreach n: {t connected nodes in nodesToHandle}
    //          t.ismr = LCM(t.ismr, projectNodeRoiToTensorRoi(n.isr))
    //          if n doesn't have other connected tensors in tensorsToHandle - remove from nodesToHandle
    //   foreach n : nodesToHandle
    //      foreach t : {n connected tensors in tensorsToHandle}
    //          n.isr = LCM(n.isr, projectTensorRoiToNodeRoi(t.ismr))
    //          if t doesn't have other connected nodes in nodesToHandle - remove from tensorsToHandle
    NodeSet   nodesToHandle(connectedNodes);
    TensorSet tensorsToHandle(connectingTensors);
    unsigned  prevNumVertices = 0;
    while (nodesToHandle.size() + tensorsToHandle.size() > 1)
    {
        HB_ASSERT(prevNumVertices != nodesToHandle.size() + tensorsToHandle.size(),
                  "calculator stuck in endless loop - input has cycle or multiple connected components ({} vertices)",
                  nodesToHandle.size() + tensorsToHandle.size());
        prevNumVertices = nodesToHandle.size() + tensorsToHandle.size();

        LOG_TRACE(TILE_SIZE_CALC, "Stage 1: Iterate all active tensors");
        for (const TensorPtr& t : tensorsToHandle)
        {
            LOG_TRACE(TILE_SIZE_CALC, "Project active connected nodes on tensor {}", t->getName());
            for (const NodePtr& n : getConnectedNodesInGroup(t, nodesToHandle, graph))
            {
                LOG_TRACE(TILE_SIZE_CALC, "\t Project node {}", n->getNodeName());
                TensorTile::Geometry projectedIsmr =
                    n->getNodeAccessPattern()->getTensorTile(t, isrPerNode[n]).geometry;
                ismrPerTensor[t] = lcm(ismrPerTensor[t], projectedIsmr);
                printTileProjection(projectedIsmr, ismrPerTensor[t]);
                if (getConnectedTensorsInGroup(n, tensorsToHandle).size() == 1)  // n's last operand to handle
                {
                    nodesToHandle.erase(n);
                }
            }
        }
        LOG_TRACE(TILE_SIZE_CALC, "Stage 1: Iterate all active nodes");
        for (const NodePtr& n : nodesToHandle)
        {
            LOG_TRACE(TILE_SIZE_CALC, "Project active connected tensors on node {}", n->getNodeName());
            NodeAccessPattern::TilePerTensor tensorTiles;
            for (const TensorPtr& t : getConnectedTensorsInGroup(n, tensorsToHandle))
            {
                LOG_TRACE(TILE_SIZE_CALC, "\t Project tensor {}", t->getName());
                tensorTiles.emplace(t, TensorTile(ismrPerTensor[t]));
                if (getConnectedNodesInGroup(t, nodesToHandle, graph).size() == 1)  // t's last node to handle
                {
                    tensorsToHandle.erase(t);
                }
            }
            NodeTile::Geometry projectedIsr = n->getNodeAccessPattern()->getLcmNodeTile(tensorTiles).geometry;
            isrPerNode[n] = lcm(isrPerNode[n], projectedIsr);
            printTileProjection(projectedIsr, isrPerNode[n]);
        }
    }

    LOG_TRACE(TILE_SIZE_CALC, "Calculate common tile size - Stage 2");

    // Stage 2
    // Init all nodes and tensors "done" flag to false
    // while (completedNodesAndTensors < |connectedNodes| + |connectingTensors|)
    //  foreach t : tensorsToHandle
    //      foreach n: {t connected nodes in connectedNodes}
    //          if (n not done)
    //              mark n as done, completedNodesAndTensors++
    //              n.isr = projectTensorRoiToNodeRoi(t.ismr)
    //              add n back to nodesToHandle
    //  clear tensorsToHandle
    //  foreach n : nodesToHandle
    //      foreach t : {n connected tensors in connectingTensors}
    //          if (t not done)
    //              mark t as done, completedNodesAndTensors++
    //              t.ismr = projectNodeRoiToTensorRoi(n.isr)
    //              add t back to tensorsToHandle
    //  clear nodesToHandle

    std::map<NodePtr, bool>   donePerNode;
    std::map<TensorPtr, bool> donePerTensor;
    std::for_each(connectedNodes.begin(), connectedNodes.end(), [&](const NodePtr& n) { donePerNode[n] = false; });
    std::for_each(connectingTensors.begin(), connectingTensors.end(), [&](const TensorPtr& t) {
        donePerTensor[t] = false;
    });
    unsigned completedNodesAndTensors = 0;
    while (completedNodesAndTensors < connectedNodes.size() + connectingTensors.size())
    {
        LOG_TRACE(TILE_SIZE_CALC, "Stage 2: Iterate all active tensors");
        for (const TensorPtr& t : tensorsToHandle)
        {
            LOG_TRACE(TILE_SIZE_CALC, "Project tensor {} on its non finished connected nodes", t->getName());
            for (const NodePtr& n : getConnectedNodesInGroup(t, connectedNodes, graph))
            {
                if (!donePerNode[n])
                {
                    LOG_TRACE(TILE_SIZE_CALC, "\t Project on node {}", n->getNodeName());
                    donePerNode[n] = true;
                    completedNodesAndTensors++;
                    NodeAccessPattern::TilePerTensor tensorTiles;
                    tensorTiles.emplace(t, TensorTile(ismrPerTensor[t]));
                    NodeTile::Geometry projectedIsr = n->getNodeAccessPattern()->getLcmNodeTile(tensorTiles).geometry;
                    isrPerNode[n]                   = lcm(isrPerNode[n], projectedIsr);
                    printTileProjection(projectedIsr, isrPerNode[n]);
                    nodesToHandle.insert(n);
                }
            }
        }
        tensorsToHandle.clear();
        LOG_TRACE(TILE_SIZE_CALC, "Stage 2: Iterate all active nodes");
        for (const NodePtr& n : nodesToHandle)
        {
            LOG_TRACE(TILE_SIZE_CALC, "Project node {} on its non finished connected tensors", n->getNodeName());
            for (const TensorPtr& t : getConnectedTensorsInGroup(n, connectingTensors))
            {
                if (!donePerTensor[t])
                {
                    LOG_TRACE(TILE_SIZE_CALC, "\t Project on tensor {}", t->getName());
                    donePerTensor[t] = true;
                    completedNodesAndTensors++;
                    TensorTile::Geometry projectedIsmr =
                        n->getNodeAccessPattern()->getTensorTile(t, isrPerNode[n]).geometry;
                    ismrPerTensor[t] = lcm(ismrPerTensor[t], projectedIsmr);
                    printTileProjection(projectedIsmr, ismrPerTensor[t]);
                    tensorsToHandle.insert(t);
                }
            }
        }
        nodesToHandle.clear();
    }
    return {ismrPerTensor, isrPerNode};
}

void CommonTileSizeCalculator::printInputGraph(const NodeSet& nodes, const TensorSet& tensors, const HabanaGraph& graph)
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(TILE_SIZE_CALC)) return;

    for (const NodePtr& n : nodes)
    {
        LOG_DEBUG(TILE_SIZE_CALC, "node {} connected to tensors:", n->getNodeName());
        for (const TensorPtr& t : getConnectedTensorsInGroup(n, tensors))
        {
            LOG_DEBUG(TILE_SIZE_CALC, "  {}", t->getName());
        }
    }
    for (const TensorPtr& t : tensors)
    {
        LOG_DEBUG(TILE_SIZE_CALC, "tensor {} connected to nodes:", t->getName());
        for (const NodePtr& n : getConnectedNodesInGroup(t, nodes, graph))
        {
            LOG_DEBUG(TILE_SIZE_CALC, "  {}", n->getNodeName());
        }
    }
}