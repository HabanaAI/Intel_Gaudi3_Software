#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/eager_node.h"
#include "node_info/node_info_defs.h"
#include "utils/general_defs.h"

class Tensor;
class Node;

///////////////////////////////////////////////////////////////////////////////////////////////////
// reorder_defs
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace eager_mode
{
namespace reorder_defs
{
// Representation of nodes array
template<class Type>
struct NodesArr
{
    Type*       arr  = nullptr;  // Array of nodes
    NodesNrType size = 0;        // Size of the array
};

// Representation of the roots queue
struct RootsQueue
{
    NodesNrType head = -1;  // Index at TopologyReorder::m_sequence that has inDegree=0
    NodesNrType tail = -1;  // Index of last TopologyReorder::m_sequence that has inDegree=0
};

// Mapping tensor to its producer
struct TensorToProducerInfo
{
    TensorToProducerInfo(NodesNrType producer, Tensor* t) : producerNodeId(producer), producedTensor(t)
    {
        EAGER_ASSERT_PTR(t);
    }
    bool operator<(const Tensor* t) const { return producedTensor < t; }  // For std::lower_bound
    bool operator<(const TensorToProducerInfo& c) const { return producedTensor < c.producedTensor; }  // For std::sort

    NodesNrType producerNodeId;  // Internal id of the node that produced the tensor. Useful for reduction op support
    Tensor*     producedTensor;  // Pointer to a tensor that had been produced by some node (a non-graph input)
};

// Representation of tensor consumptions. It includes only tensors that are produced by nodes,
// each one is associated with its produced node id and mapped to all nodes that consume it.
// Outputs of the graph will be included but their consumer node list will be empty.
struct TensorConsumptionInfo : public TensorToProducerInfo
{
    TensorConsumptionInfo(NodesNrType producer, Tensor* t) : TensorToProducerInfo(producer, t) {}
    bool operator<(const Tensor* t) const { return TensorToProducerInfo::operator<(t); }
    bool operator<(const TensorConsumptionInfo& c) const { return TensorToProducerInfo::operator<(c); }

    llvm_vecsmall::SmallVector<NodesNrType, defaultMaxNodesPerGraph> consumers;  // Nodes that consume the tensor above
};
// Consumption info per output of each node (the non-graph inputs)
using ConsumptionInfo = llvm_vecsmall::SmallVector<TensorConsumptionInfo, defaultOutputsPerGraph>;
}  // namespace reorder_defs

///////////////////////////////////////////////////////////////////////////////////////////////////
// TopologyReorder
///////////////////////////////////////////////////////////////////////////////////////////////////

// Utility class to apply in-place topology reorder on arbitrary range of nodes
class TopologyReorder
{
public:
    bool                                              reorder(EagerNode* nodeArr, NodesNrType nodesNr);
    const reorder_defs::ConsumptionInfo&              getConsumptionInfo() const { return m_consumptionInfo; }
    const VecNodes<NodesNrType>&                      getSequence() const { return m_sequence; }
    inline const reorder_defs::TensorConsumptionInfo* getTensorConsumptionInfo(const Tensor* t) const;

private:
    bool initConsumptionInfo();
    bool fillConsumptionInfo();
    bool initRootsQueue();
    bool completeReordering();
    void permuteResult();

private:
    inline reorder_defs::TensorConsumptionInfo* getTensorConsumptionInfo(const Tensor* t);
    static void printNode(const EagerNode& node, NodesNrType oldIdx, NodesNrType newIdx);

private:
    reorder_defs::NodesArr<EagerNode> m_nodes;       // Array of input nodes
    reorder_defs::RootsQueue          m_rootsQueue;  // A queue to track roots removal
    reorder_defs::ConsumptionInfo     m_consumptionInfo;
    VecNodes<NodesNrType>             m_sequence;  // Node indices as they supposed to be executed
    VecNodes<TensorsNrType>           m_inDegree;  // Number of inputs per node to track roots removal
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// GlobalDependencies
///////////////////////////////////////////////////////////////////////////////////////////////////

// This class serves as a comprehensive repository of dependency information for the non-original graph nodes.
// It can offer efficient access to critical details regarding node relationships, such as:
// - Identifying nodes dependent on a specific one.
// - Determining the nodes upon which a given node depends.
// - Locating the most recent node in the execution sequence that produces a tensor consumed by a given node.
// These queries are powered by a continuously updated dataset (TensorConsumptionInfo) received from ExecScheduler
// class. The incoming information is processed in O(n) time complexity, where 'n' represents the data's size. This
// class guarantees that all queries can be executed in O(1) time complexity.
class GlobalDependencies
{
public:
    const VecNodes<NodesNrType>& getLatestPhysicalProducers() const { return m_latestPhysicalProducers; }

protected:
    VecNodes<NodesNrType> m_latestPhysicalProducers;  // For each node store its last physical producer or -1 if none
};

// This class maintains a map for all non-graph-input tensors.
// It provides a query to map a given tensor to its producer. In this case the map must be sorted.
class ProducedTensorsMap final
{
public:
    inline void add(const TensorPtr& tensor, NodesNrType producer);
    void        add(const reorder_defs::ConsumptionInfo& consumptionInfo,
                    const VecNodes<NodesNrType>&         invSequence,
                    NodesNrType                          existingNodesNr);
    void        sort();
    NodesNrType findProducer(const TensorPtr& tensor) const;

private:
    bool m_isSorted = false;  // Is the map sorted according to tensors
    using TensorToProducerInfoMap =
        llvm_vecsmall::SmallVector<reorder_defs::TensorToProducerInfo, defaultOutputsPerGraph>;
    TensorToProducerInfoMap m_producedTensorsMap;  // Pointers to all tensors and their producers
};

// Internal class to build m_latestPhysicalProducers
class GlobalDependenciesBuilder final : public GlobalDependencies
{
public:
    GlobalDependenciesBuilder(bool isParallelExecutionPossible);
    bool isReEnablingParallelExecPossible() const
    {
        return m_isParallelExecutionPossible && !m_enableParallelExecution;
    }
    void enableParallelExecution() { m_enableParallelExecution = true; }
    void disableParallelExecution() { m_enableParallelExecution = false; }

    void processSingleNode(const EagerNode& node);
    void processTwoNodes(const EagerNode& node1, const EagerNode& node2, bool isNode2DependOnNode1);
    void processThreeNodesAndMore(EagerNode* nodeArr, NodesNrType nodesNr, const TopologyReorder& rangeReorder);
    void redoSerialDependencies(NodesNrType nodeNr);
    void finalize(const EagerNodesBuilder& nodes);

private:
    void processSingleNodeForSerialExecution(const EagerNode& node);
    void processSingleNodeForParallelExecution(const EagerNode& node);
    void addNewNode(const EagerNode& node, NodesNrType producerId = -1);
    void fillMissingProducers();
    void fixDependenciesOnLogicalOps(const EagerNodesBuilder& nodes);

private:
    const bool         m_isParallelExecutionPossible;  // Does GCFG settings allow parallel execution
    bool               m_enableParallelExecution;      // Flag to determine type of execution on device
    ProducedTensorsMap m_producedTensorsMap;           // Mapping tensors seen so far to their producers

    // List of nodes that have not associated with physical producer.
    // The list contains nodes and their position in the execution order.
    struct MissingProducerInfo
    {
        MissingProducerInfo(const EagerNode& node, NodesNrType sequenceId) : node(node), sequenceId(sequenceId) {}
        const EagerNode& node;        // A node that has missing producer
        NodesNrType      sequenceId;  // The execution order of the node
    };
    VecNodes<MissingProducerInfo> m_missingProducers;  // All nodes that have missing physical producer
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// ExecScheduler
///////////////////////////////////////////////////////////////////////////////////////////////////

// Apply in-place topological reordering on nodes. There are three usages in Eager:
//  1. Reorder original nodes via reorderAll(...)
//  2. Reorder logical nodes via reorderLogicalNodes(...)
//  3. While downloading original nodes to graph, apply topological nodes after NodeDisplacement::processNewNode(...)
//     via reorderLast(...)
// The idea behind this separation:
// In case of node extraction, no matter if it's occurs via complex GUID or internal logic the resulted sub-graph brings
// new relation between the extracted nodes, those nodes can consume tensors produced by previous nodes or graph inputs,
// so it's feasible to reorder them without considering other nodes in the graph.
// Usually to sort multiple sub-graphs is much faster than to extract and handle all nodes then reorder them, as number
// of comparisons raises and so the runtime. Similar reasoning for collecting dependency info.
class ExecScheduler
{
public:
    ExecScheduler(bool isParallelExecutionPossible) : m_globalDependencies(isParallelExecutionPossible) {}
    bool                      reorderAll(EagerNodesBuilder& nodes);
    bool                      reorderLast(EagerNodesBuilder& nodes);
    void                      disableProcessingNewNodes(const EagerNodesBuilder& nodes);
    const GlobalDependencies& getGlobalDependencies() const { return m_globalDependencies; }

    void injectNodes(EagerNodes& nodes, VecNodes<std::pair<std::size_t, EagerNode>>& nodesToInject);
    void redoSerialDependencies(NodesNrType nodeNr) { m_globalDependencies.redoSerialDependencies(nodeNr); }

private:
    static bool reorderTwoNodes(EagerNode& node1, EagerNode& node2, const EagerTensorsSetBuilder& tensorsSet);
    enum ReorderTwoNodesRes
    {
        SUCCESS,
        FAIL,
        SWAP  // It determines a direct dependency of node2 on node1
    };
    static ReorderTwoNodesRes reorderTwoNodes(EagerNode& node1, EagerNode& node2);
    static bool               isOrdered(const Node& node1, const Node& node2);

private:
    bool                               m_processNewNodesEn   = true;   // Allow processing new nodes
    bool                               m_isReorderAllInvoked = false;  // Was reorderAll(...) invoked?
    TopologyReorder                    m_rangeReorder;
    reorder_defs::NodesArr<EagerNodes> m_nodesArrPtr;         // Pointer to nodes array used by reorderLast(...)
    GlobalDependenciesBuilder          m_globalDependencies;  // Graph-level information on node dependencies
};

}  // namespace eager_mode