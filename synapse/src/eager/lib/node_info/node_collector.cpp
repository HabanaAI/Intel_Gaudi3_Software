#include "node_collector.h"

// eager includes (relative to src/eager/lib/)
#include "node_info/exec_schedule.h"
#include "node_info/node_displacement.h"
#include "node_info/transpose_fuser.h"
#include "utils/general_defs.h"

namespace eager_mode
{
NodeCollector::NodeCollector(NodeDisplacement& nodeDisplacement)
: m_nodeDisplacement(nodeDisplacement), m_nodes(nodeDisplacement.m_nodes)
{
}

// postpone gemm and transpose handling to the end to allow for transpose fusion
// to gemm\batch gemm. We can't do it on the original graph as some frequent
// gemms \ batch gemms in BERT are extracted in complex guid (for instance addmm_bf16)
// and transposes might also be introduced through decoded strided view\ insert operations.
// we postpone gemm \ batch gemm as well as transpose fusion can affect later optimizations
// on those operations (though currently we do not make any for Eager).
bool NodeCollector::isNodeTypeWithPostponedProcessing(const EagerNode& node) const
{
    switch (node->getNodeType())
    {
        case Node::TYPE_GEMM:
        case Node::TYPE_BATCH_GEMM:
        case Node::TYPE_INTERNAL_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

bool NodeCollector::shouldPostponeNodeProcessing(const EagerNode& node) const
{
    return m_userNodeExtractionDone ? false : isNodeTypeWithPostponedProcessing(node);
}

void NodeCollector::collectNode(EagerNode node, bool isLogical)
{
    EAGER_ASSERT(m_nodes.empty() || m_userNodeExtractionDone,
                 "nodes container should be empty when collecting initial nodes");
    if (isLogical)
    {
        EAGER_ASSERT(m_injectionIdx.has_value() == false, "Unexpected logical node injection");
        m_logicalNodesPresent = true;
    }
    m_nodeTypes.set(node->getNodeType());

    // The node is collected unless it's a memcpy resulting from the logical pass,
    // in which case it's saved to the list of nodes to be injected.
    if (m_injectionIdx)
    {
        m_nodesToInject.emplace_back(*m_injectionIdx, std::move(node));
    }
    else
    {
        m_collectedNodes.push_back(std::move(node));
    }
}

bool NodeCollector::downloadExtractedNodes(unsigned userNodeIdx)
{
    EAGER_ASSERT(m_userNodeExtractionDone, "invalid flow for downloadExtractedNodes");
    EAGER_ASSERT(userNodeIdx < m_userNodeBoundaries.size(),
                 "invalid user node index supplied to downloadExtractedNodes");

    auto appendNode = [this](EagerNode&& node) {
        EAGER_ASSERT(node.getEngineType() == EngineType::MME ||
                         node.get()->getNodeType() != Node::TYPE_INTERNAL_TRANSPOSE,
                     "internal transpose nodes should no longer be present");
        m_nodes.push_back(std::move(node));
    };

    unsigned extractedNodeIndex = (userNodeIdx == 0) ? 0 : m_userNodeBoundaries[userNodeIdx - 1];
    unsigned extractedNodeEnd   = m_userNodeBoundaries[userNodeIdx];
    for (; extractedNodeIndex < extractedNodeEnd; ++extractedNodeIndex)
    {
        EagerNode&& node = std::move(m_collectedNodes[extractedNodeIndex]);
        EAGER_ASSERT_PTR(node);
        // skip over dropped nodes during optimization phases such as transpose fusion
        if (node.isInvalidated()) continue;
        if (isNodeTypeWithPostponedProcessing(node))
        {
            // now actually extract the internal transpose node and add the newly
            // added nodes to the end of m_collectedNodes vector.
            unsigned extractedTransposeNodeIndex = m_collectedNodes.size();
            if (!m_nodeDisplacement.processNewNode(node, false /*userNode*/))
            {
                return false;
            }
            for (; extractedTransposeNodeIndex < m_collectedNodes.size(); ++extractedTransposeNodeIndex)
            {
                appendNode(std::move(m_collectedNodes[extractedTransposeNodeIndex]));
            }
        }
        else
        {
            appendNode(std::move(node));
        }
    }
    return true;
}

bool NodeCollector::processUserNode(EagerNode& node)
{
    bool success = m_nodeDisplacement.processNewNode(node, true);
    // mark extracted nodes boundary corresponding to the processed user node
    m_userNodeBoundaries.push_back(m_collectedNodes.size());
    return success;
}

void NodeCollector::fuseTransposes()
{
    EAGER_ASSERT(m_userNodeExtractionDone, "invalid flow for downloadExtractedNodes");
    EAGER_ASSERT(m_nodes.empty(), "nodes container should be empty before transpose fusion");
    if (!GCFG_ENABLE_EAGER_NODE_DISPLACEMENT_OPTIMIZATIONS.value() || !GCFG_ENABLE_TRANSPOSE_FUSION_IN_EAGER.value())
        return;
    if (!m_nodeTypes[Node::TYPE_INTERNAL_TRANSPOSE]) return;
    if (!m_nodeTypes[Node::TYPE_GEMM] && !m_nodeTypes[Node::TYPE_BATCH_GEMM]) return;
    EagerTransposeFuser transposeFuser(m_collectedNodes);
    transposeFuser.fuseTransposes();
}

void NodeCollector::injectNodes(ExecScheduler& execSequencer, bool bwdPass)
{
    if (m_nodesToInject.empty()) return;

    // Bwd inserts nodes in reverse order and injectNodes requires them to be sorted in increasing order
    if (bwdPass)
    {
        std::reverse(m_nodesToInject.begin(), m_nodesToInject.end());
    }

#ifndef NDEBUG
    {
        LOG_DEBUG(EAGER, "Nodes to be injected after {} logical pass", bwdPass ? "BWD" : "FWD");
        size_t i = 0;
        for (const auto& vv : m_nodesToInject)
        {
            LOG_DEBUG(EAGER,
                      "Node #{:03}/{:03}: ({}, {})",
                      i++,
                      m_nodesToInject.size(),
                      vv.first,
                      vv.second->getNodeName());
        }
    }
#endif  // NDEBUG

    for (auto& vv : m_nodesToInject)
    {
        // we need to update the physical nodes nr and overall tensors
        m_nodes.injectNode(vv.second);
    }

    execSequencer.injectNodes(m_nodes, m_nodesToInject);
    EAGER_ASSERT(m_nodesToInject.empty(), "injectNodes must consume the nodes");
}

}  // namespace eager_mode
