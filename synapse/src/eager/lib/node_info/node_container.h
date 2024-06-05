#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/duplication_map.h"
#include "node_info/eager_node.h"
#include "node_info/exec_schedule.h"
#include "node_info/node_displacement.h"
#include "node_info/node_info_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_graph.h"
#include "graph_compiler/types.h"

namespace eager_mode
{
class EagerGraph;

// Store all nodes and provide related queries and operations.
class NodesContainer
{
public:
    NodesContainer(EagerGraph& eagerGraph);
    NodesContainer(const NodesContainer& other, EagerGraph& eagerGraph);

    bool             isAddingNewNodesEnabled() const { return m_orgNodes.isAddingNewNodesEnabled(); }
    bool             areOriginalNodesSupported() const { return m_areOrgNodesSupported; }
    bool             areOriginalTensorsSupported() const { return m_orgNodes.getTensors().areAllTensorsSupported(); }
    bool             prepareForGraphDuplication() { return lockAndSortUserNodes(); }

    NodesNrType            getOriginalNodesNr() const { return m_orgNodes.size(); }
    const EagerTensorsSet& getOriginalTensors() const { return m_orgNodes.getTensors(); }
    bool isEagerCompilationSupported() const { return areOriginalNodesSupported() && areOriginalTensorsSupported(); }
    bool              addNewNode(const EagerNode& node);
    const EagerNodes& getNodes() const { return m_nodes; }
    bool              downloadOriginalNodesToHabanaGraph(HabanaGraph& graph);
    bool              downloadOriginalNodesToEagerGraph();
    const EagerNode*  findNodeByID(synNodeId nodeID) const;

    const GlobalDependencies& getGlobalDependencies() const { return m_execSequencer.getGlobalDependencies(); }
    bool                      performMaxShapeInference();

    // Used by Synapse duplicate API to retrieve nodes and tensors mappings between original graph
    // and newly created duplicate graph.
    void getDuplicateMappings(const NodesContainer&             orig,
                              HabanaGraph::TensorPtrMappingVec& tensorsMap,
                              HabanaGraph::NodeIdMappingVec&    nodesMap);

    template<class ContainerType>
    inline static void printNodesWithTensorDetails(const ContainerType& nodes, std::string_view label);

    template<class ContainerType>
    void printNodes(const ContainerType& nodes, const std::string& label);

private:
    void addNewOriginalNode(const EagerNode& node);
    bool processLogicalNodes();
    bool lockAndSortUserNodes();

private:
    EagerGraph&       m_eagerGraph;
    const uint64_t    m_tensorSizeThresholdForParallelExec;  // Caching the parallel exec threshold (used at C'tor only)
    EagerNodesBuilder m_orgNodes;                            // List of original nodes
    EagerNodesBuilder m_nodes;                               // Current nodes after downloading original nodes to Eager
    NodeDisplacement  m_nodeDisplacement;                    // Extract nodes and check if they are supported
    bool              m_areOrgNodesSupported = true;         // Are original nodes added so far supported by Eager
    ExecScheduler     m_execSequencer;                       // Keep it member variable to defer its destruction
    DuplicationMap    m_duplicationMap;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of debug templates
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class ContainerType>
void NodesContainer::printNodes(const ContainerType& nodes, const std::string& label)
{
    llvm_vecsmall::SmallVector<std::string, defaultMaxNodesPerGraph> guids;
    for (const auto& node : nodes)
    {
        guids.push_back(node->getGUID());
    }
    LOG_INFO(EAGER, "{{ \"{}\": [{}] }}", label, fmt::join(guids.begin(), guids.end(), ", "));
}

template<class ContainerType>
void NodesContainer::printNodesWithTensorDetails(const ContainerType& nodes, std::string_view label)
{
    auto printTensors = [](std::string_view label, const TensorVector& tensors) {
        for (const TensorPtr& tensor : tensors)
        {
            if (tensor == nullptr) return;
            if (tensor->isAliasedTensor())
            {
                const TensorPtr& realTensor = Tensor::getRealTensor(tensor);
                EAGER_ASSERT_PTR(realTensor);
                LOG_INFO(EAGER, "       {}: \"{}\"  --->  \"{}\"", label, tensor->getName(), realTensor->getName());
            }
            else
            {
                LOG_INFO(EAGER, "       {}: \"{}\"", label, tensor->getName());
            }
        }
    };

    LOG_INFO(EAGER, "Printing {} with tensor details", label);
    NodesNrType index = 0;
    for (const auto& node : nodes)
    {
        LOG_INFO(EAGER,
                 "  {}) ID:{} \"{}\" [\"{}\", {}, {}]",
                 index++,
                 node->getId(),
                 node->getNodeName(),
                 node->getGUID(),
                 node->getNodeTypeStr(),
                 node->isLogicalOperation() ? "logical" : node->getEngineTypeStr());
        printTensors("i", node->getInputs());
        printTensors("o", node->getOutputs());
    }
}

}  // namespace eager_mode
