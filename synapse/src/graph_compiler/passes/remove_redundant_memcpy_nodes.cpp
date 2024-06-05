#include "habana_graph.h"
#include "habana_pass.h"
#include "graph_editor.h"

bool isMemcpyValidForRemoval(const NodePtr& n)
{
    // Memcpies in the pre-graph that change location are not redundant since they might be used to copy tensors to rmw
    // sections
    if (n->getInput(0)->location() != n->getOutput(0)->location()) return false;
    // Memcpies that change the tensors dtype are translated to cast nodes, so they should not be removed
    if (n->getInput(0)->getElementType() != n->getOutput(0)->getElementType()) return false;
    // memcopies that are tpc dynamic memory ops hold valuable data
    auto tpcMemcpyNode = dynamic_cast<const TPCMemcpyNode*>(n.get());
    if (tpcMemcpyNode && tpcMemcpyNode->isDynamicMemoryOp()) return false;
    return true;
}

// This pass removes redundant memcpy nodes that were added by the user. It has HIGH priority since it should run before
// any pass that might add memcpy nodes.
bool removeRedundantMemcpyNodes(HabanaGraph& g)
{
    if (!GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY.value()) return true;
    NodeSet nodes = g.getNodes();

    for (const auto& n : nodes)
    {
        if (isMemcpy(*n) && isMemcpyValidForRemoval(n))
        {
            LOG_DEBUG(GC, "Attempting to remove redundant memcpy node '{}'", n->getNodeName());
            GraphEditor::removeOneToOneNode(g, n);
        }
    }

    return true;
}