#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/eager_node.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"

// std includes
#include <bitset>
#include <cstddef>
#include <optional>
#include <utility>

namespace eager_mode
{
class ExecScheduler;
class NodeDisplacement;

// This class acts as a temporary container for extracted user nodes prior to their addition
// to the node container.
// The reasoning for it, is that some circumstances such as transpose fusion require a wider visibility and might result
// in us dropping or replacing a node by a set of other nodes. So we wish to avoid collecting statistics and making
// topological ordering and dependencies analysis for nodes which might be dropped. This is achieved by collecting the
// nodes into a simple small vector and marking the boundaries between extracted nodes corresponding to different user
// nodes.
class NodeCollector
{
public:
    explicit NodeCollector(NodeDisplacement& nodeDisplacement);
    bool shouldPostponeNodeProcessing(const EagerNode& node) const;
    // passing EagerNode by value intentionally to avoid invalidation,
    // in case the original node is re-added and container has to grow
    // and invalidate the previous buffer.
    void collectNode(EagerNode node, bool isLogical);
    void markUserNodeExtractionCompletion() { m_userNodeExtractionDone = true; }
    bool downloadExtractedNodes(unsigned userNodeIdx);
    bool processLogicalNodes(ExecScheduler& execSequencer);
    bool processUserNode(EagerNode& node);
    void fuseTransposes();
    void injectNodes(ExecScheduler& execSequencer, bool bwdPass);

    bool hasLogicalNodes() const { return m_logicalNodesPresent; }

    void setInjectionIdx(size_t idx)
    {
        EAGER_ASSERT(!m_injectionIdx, "Injection idx already set");
        EAGER_ASSERT(idx <= m_nodes.size(), "Injection idx out of range");
        m_injectionIdx = idx;
    }

    void resetInjectionIdx()
    {
        EAGER_ASSERT(m_injectionIdx, "Injection idx not set");
        m_injectionIdx.reset();
    }

private:
    bool isNodeTypeWithPostponedProcessing(const EagerNode& node) const;

private:
    bool                        m_userNodeExtractionDone = false;
    bool                        m_logicalNodesPresent    = false;
    std::bitset<Node::TYPE_MAX> m_nodeTypes              = {};
    NodeDisplacement&           m_nodeDisplacement;
    EagerNodes&                 m_nodes;
    EagerNodesVec               m_collectedNodes;
    VecNodes<unsigned>          m_userNodeBoundaries;

    // When set, we're in the middle of a logical node extraction and,
    // the value is the index before which the new nodes will be injected.
    std::optional<std::size_t> m_injectionIdx {};
    // A collection of the nodes extracted in the logical pass to be injected into m_nodes,
    // consists of the index in m_nodes before which the node is to be injected and the node itself.
    VecNodes<std::pair<std::size_t, EagerNode>> m_nodesToInject {};
};

}  // namespace eager_mode
