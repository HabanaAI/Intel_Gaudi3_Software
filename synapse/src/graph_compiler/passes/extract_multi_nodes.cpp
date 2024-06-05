#include "habana_graph.h"
#include "gaudi_graph.h"
#include "multi_node.h"
#include "graph_editor.h"

namespace extract_multi_nodes
{
using MultiNodePtr   = std::shared_ptr<MultiNode>;
using ExtractionList = std::deque<MultiNodePtr>;

void addToExtractionList(ExtractionList& toExtract, const NodePtr& n, bool skipDataMovementOps)
{
    if (!n || !n->isMultiNode()) return;
    MultiNodePtr multiNode = std::dynamic_pointer_cast<MultiNode>(n);
    HB_ASSERT_PTR(multiNode);

    if (skipDataMovementOps && multiNode->isDataMovementMultiNode()) return;
    toExtract.push_back(multiNode);
}

// Atomic nodes are pairs of marked nodes that are required to be adjacent to each other when compilation is finished.
// Tensor will be marked as connectAtomicNodes when connecting two atomic nodes, during multi node extraction.
// The following function goes over the extracted nodes from a given multi node, and saves the marked
// atomic nodes in a data structure, as part of the graph annotation struct.
void addAtomicNodes(HabanaGraph& graph)
{
    if (!GCFG_ENABLE_ATOMIC_NODES_VALIDATION.value())
    {
        return;
    }

    for (const auto& connectingTensor : graph.getTensors())
    {
        if (!connectingTensor) continue;

        if (connectingTensor->getTensorAnnotation().connectAtomicNodes)
        {
            NodePtr producerNode = graph.getTensorProducer(connectingTensor);
            if (!producerNode) continue;
            for (const NodePtr& consumerNode : graph.getTensorConsumers(connectingTensor))
            {
                graph.getGraphAnnotation().addAtomicNodesPair(producerNode, consumerNode);
            }
        }
    }
}

bool extractMultiNodes(HabanaGraph& graph, bool skipDataMovementOps)
{
    // Todo [SW-101946] Need to make setLogicalBeforePhysicalTranspose dependant on Extract Multi Nodes directly
    bool res = setLogicalBeforePhysicalTranspose(graph);
    if (!res) return false;

    ExtractionList toExtract;
    for (const NodePtr& n : graph.getNodes())
    {
        addToExtractionList(toExtract, n, skipDataMovementOps);
    }

    while (!toExtract.empty())
    {
        MultiNodePtr multiNode = toExtract.front();
        toExtract.pop_front();

        MultiNode::MultiNodeDependencies dependencies;

        NodeList extractedNodes = multiNode->extract(graph, dependencies);

        if (extractedNodes.empty())
        {
            LOG_ERR(GC, "{}: Cannot extract nodes from node {}", HLLOG_FUNC, multiNode->getNodeName());
            return false;
        }

        for (auto& n : extractedNodes)
        {
            n->getNodeAnnotation().isExtracted = true;
            if (n->isTranspose() && !n->isLogicalOperation())
            {
                graph.turnOnPredicate(PREDICATE_ID_PHYSICAL_TRANSPOSE_NODE_CREATED);
            }

            addToExtractionList(toExtract, n, skipDataMovementOps);
        }

        if (GraphEditor::replaceNodes(graph, {multiNode}, extractedNodes) != REPLACE_NODE_SUCCESS)
        {
            LOG_ERR(GC, "{}: Cannot replace nodes for node {}", HLLOG_FUNC, multiNode->getNodeName());
            return false;
        }

        for (const auto& dep: dependencies)
        {
            graph.addControlDependency(dep.blocking, dep.blocked, dep.type);
        }
    }

    if (graph.getDeviceType() != synDeviceGaudi) // [SW-159339] WA - remove when root caused and fixed
    {
        // It can be that the replacement above created new redundant nodes.
        // for example if the original node has more than one producer and was replaced with identity to one of them,
        // his other producers can now be redundant.
        graph.turnOnPredicate(PREDICATE_ID_ELIMINATE_REDUNDANT_NODES);
    }

    addAtomicNodes(graph);

    return true;
}
};  // namespace extract_multi_nodes

bool extractMultiNodes(HabanaGraph& graph)
{
    return extract_multi_nodes::extractMultiNodes(graph, true /* skipDataMovementOps */);
}

bool extractDataMovementMultiNodes(HabanaGraph& graph)
{
    return extract_multi_nodes::extractMultiNodes(graph, false /* skipDataMovementOps */);
}
