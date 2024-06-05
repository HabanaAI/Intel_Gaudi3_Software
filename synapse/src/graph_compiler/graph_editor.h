#pragma once

#include "habana_nodes/node.h"
#include "types.h"

class HabanaGraph;

typedef enum
{
    REPLACE_NODE_SUCCESS,
    REPLACE_FAILED_INVALID_NEW_NODES,
    REPLACE_FAILED_INGROUP_DEPS
} ReplaceNodeReturnStatus;

class GraphEditor
{
public:
    static std::string_view replaceNodeReturnStatusToString(ReplaceNodeReturnStatus status);

    static bool canEliminateTensor(const HabanaGraph& g, const TensorPtr t, unsigned maxOutConsumers = 1, bool eliminateStaticParams=false);

    static pNode checkNodeMemsetForReduction(const HabanaGraph& graph, const pNode& node);

    template<typename NodeContainer = NodeVector>
    static void removeMemsetReductionFromFusion(HabanaGraph&         g,
                                                const NodeContainer& newNodes,
                                                NodeSet&             fusedBlocked,
                                                NodeSet&             fusedBlocking);

    // Insert memcopies in / out to the tensor indices, return the new memcpy nodes
    static NodeList insertMemcpies(HabanaGraph& g,
                                   NodePtr currNode,
                                   bool insertForInputs,
                                   const std::vector<unsigned>& tensorIndex,
                                   TensorLocation location = UNDEFINED_LOCATION);

    // Insert memcpy in / out to the tensor index, return the new memcpy node
    static NodePtr insertMemcpy(HabanaGraph& g, NodePtr currNode, bool insertForInput,
                                const unsigned tensorIndex, TensorLocation location = UNDEFINED_LOCATION);

    // Insert memcpy in / out to the first occurrence of old tensor, return the new memcpy node
    static NodePtr insertMemcpy(HabanaGraph& g, NodePtr currNode, bool insertForInput,
                                const TensorPtr& oldTensor, TensorLocation location = UNDEFINED_LOCATION);

    static NodePtr insertMemcpyForInput(HabanaGraph& g, NodePtr currNode,
                                        const unsigned tensorIndex, TensorLocation location = UNDEFINED_LOCATION);
    static NodePtr insertMemcpyForOutput(HabanaGraph& g, NodePtr prevNode,
                                         const unsigned tensorIndex, TensorLocation location = UNDEFINED_LOCATION);
    // Insert memcpy to the first occurrence of original tensor, return the new memcpy node
    static NodePtr insertMemcpyForInput(HabanaGraph& g, NodePtr currNode,
                                        const TensorPtr& oldTensor, TensorLocation location = UNDEFINED_LOCATION);
    static NodePtr insertMemcpyForOutput(HabanaGraph& g, NodePtr prevNode,
                                         const TensorPtr& oldTensor, TensorLocation location = UNDEFINED_LOCATION);

    // Checks if certain set of nodes contains edges that may create loop after fusion
    static bool isPossibleLoopDueToExtEdges(HabanaGraph& g, const NodeSet& nodesSet);

    // Check if fusing the node set will create a loop
    static bool isLoopDueFusionInNodeSet(HabanaGraph& g, const NodeSet& nodesSet);

    // Check if fusing the node set will create a loop if not then the graph is updated replacing the nodeSet with the
    // fused node
    static bool
    isLoopDueFusionInUpdateGraph(HabanaGraph& g, const NodeSet& nodesSet, NodePtr& fusedNode, bool updateGraph);

    // Detects dependencies between the nodes in the group
    template<typename NodeContainer = NodeVector>
    static bool isInGroupDependencies(HabanaGraph& g, const NodeContainer& nodes);

    // Gets a list of nodes and returns a list of the first MME node, TPC node, and 6 DMA nodes
    template<typename NodeContainer = NodeVector>
    static void
    getFirstNodesPerEngineForBundle(HabanaGraph& g, const NodeContainer& nodes, NodeSet& firstNodesPerEngine);

    // Update new nodes with the maximum wait cycles of the old nodes
    template<typename NodeContainer1 = NodeVector, typename NodeContainer2 = NodeVector>
    static void updateWaitCycles(const NodeContainer1& oldNodes, const NodeContainer2& newNodes);

    template<typename NodeContainer>
    static std::unordered_set<synNodeId> getUniqueFlashAttentionParentIds(const HabanaGraph&   g,
                                                                          const NodeContainer& nodes);

    // replace a group of nodes by another group of nodes  in the graph, while keeping control dependencies properties.
    template<typename NodeContainer1 = NodeVector, typename NodeContainer2 = NodeVector>
    static ReplaceNodeReturnStatus
    replaceNodes(HabanaGraph& g, const NodeContainer1& oldNodes, const NodeContainer2& newNodes, bool isSlicer = false);

    // Perform editFunc on a node. Graph relationships will be updated after editing.
    static void editNode(HabanaGraph& g, const NodePtr& node, std::function<void(const NodePtr&)> editFunc);
    static void editNode(HabanaGraph& g, const NodePtr& node, std::function<void()> editFunc);

    // Remove a group of nodes in the graph, while keeping control dependencies properties.
    template<typename NodeContainer = NodeVector>
    static void removeNodes(HabanaGraph& g, const NodeContainer& nodes);

    // Move the consumers of the removed nodes[i] to newProducer[i] if it was provided
    template<typename NodeContainer1 = NodeVector, typename NodeContainer2 = NodeVector>
    static void removeNodes(HabanaGraph& g, const NodeContainer1& nodes, const NodeContainer2& newProducers);

    // check if node can be removed while moving its control edges to surrounding nodes without any risks
    static bool canRemoveNodeControl(const HabanaGraph& g, const NodePtr& node);

    // Remove a single one-to-one node in the graph, while keeping control dependencies properties
    static void removeOneToOneNode(HabanaGraph& g, const NodePtr& node);

    // Replace a tensor with new tensor in a node. Graph relationships will be updated after replacing the tensors.
    static void replaceTensor(HabanaGraph& g, const NodePtr& node, TensorPtr oldTensor, TensorPtr newTensor);

    // Replace a tensor with new tensor for all nodes in the graph. Graph relationships will be updated after replacing
    // the tensors.
    static void replaceTensor(HabanaGraph& g, TensorPtr oldTensor, TensorPtr newTensor);

    // Replace an input tensor at index with new tensor t. Graph relationships will be updated after replacing input tensor.
    static void replaceInput(HabanaGraph& g, const NodePtr& node, unsigned int index, TensorPtr t, Node::eTensorType tensorType = Node::TENSOR_TYPE_DATA);

    // Replace an output tensor at index with new tensor t. Graph relationships will be updated after replacing output tensor.
    static void replaceOutput(HabanaGraph& g, const NodePtr& node, unsigned int index, TensorPtr t);

    // add node to the graph, maintaining control dependencies correctness
    static bool addNode(HabanaGraph& g, const NodePtr& node, bool setParentId = true);

    template<typename NodeContainer = NodeVector>
    static bool addNodes(HabanaGraph& g, const NodeContainer& nodes, bool setParentId = true);

    // remove node from the graph, maintaining control dependencies correctness
    static void removeNode(HabanaGraph& g, const NodePtr& node, const NodePtr& newProducer = nullptr);

    // recalculate and add node control dependencies according to memory coherence
    static void addNodeControlDependencies(HabanaGraph& g, const NodePtr& node);

    // remove existing control edges, recalculate them according to memory coherence, and add them back.
    static void recalculateNodeControlDependencies(HabanaGraph& g, const NodePtr& node);

private:
    template<typename NodeContainer1 = NodeVector, typename NodeContainer2 = NodeVector>
    static bool isOneForOneReplacement(const NodeContainer1& oldNodes, const NodeContainer2& newNodes);
    template<typename NodeContainer1 = NodeVector, typename NodeContainer2 = NodeVector>
    static bool isTrivialReplacement(const NodeContainer1& oldNodes, const NodeContainer2& newNodes);
    static void replaceTrivialNodes(HabanaGraph& g, const NodePtr& oldNode, const NodePtr& newNode);
};
