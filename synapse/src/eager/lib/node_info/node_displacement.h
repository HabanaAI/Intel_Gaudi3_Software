#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/const_tensor_optimizer.h"
#include "node_info/eager_complex_guid_extractor.h"
#include "node_info/eager_node.h"
#include "node_info/node_collector.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

class HugeTensorHandler;
struct TransposeNodeParams;

namespace eager_mode
{
class EagerGraph;
class ExecScheduler;
class NodesContainerBackend;

// Given a node this class can handle it in various ways:
//  - Keep it
//  - Reject it (unsupported)
//  - Replace it with another node
//  - Extract multiple nodes
//  - Modify its tensors
class NodeDisplacement
{
public:
    NodeDisplacement(EagerGraph& eagerGraph, EagerNodes& curNodes);

    // This interface sets processing node mode to USER
    bool isUserNodeSupported(const EagerNode& node) const { return isNodeSupported(node, true); }
    bool processUserNode(EagerNode& node) { return m_nodeCollector.processUserNode(node); }
    bool downloadExtractedNodes(unsigned userNodeIdx) { return m_nodeCollector.downloadExtractedNodes(userNodeIdx); }
    void markUserNodeExtractionCompletion() { m_nodeCollector.markUserNodeExtractionCompletion(); }
    bool processLogicalNodes(ExecScheduler& execSequencer);
    void fuseTransposes() { return m_nodeCollector.fuseTransposes(); }

private:
    bool addInternalNode(EagerNode& node);
    bool isNodeSupported(const EagerNode& node, bool userNode) const;
    bool processNewNode(EagerNode& node, bool userNode);

    // This enum represent the result of processing a node
    enum class AddNodeResult
    {
        SUCCESS_ADD_REQUIRED,     // Add the node to a container
        SUCCESS_NO_ADD_REQUIRED,  // No need to add a node to a container or continue processing it
        FAIL,                     // A problem with the node that requires taking an action
    };

    AddNodeResult   processNewUserNode(const EagerNode& node);
    AddNodeResult   handleSpecialOperators(const EagerNode& node, bool handleOnlyConstOptimization, bool userNode);
    AddNodeResult   splitBatchNorm(const EagerNode& node);
    static NodeList splitBatchNormFwd(const TPCNode& fullBatchNorm, synDataType dtype);
    static NodeList splitBatchNormBwd(const TPCNode& fullBatchNorm, synDataType dtype);
    bool            processTransposeNode(const TransposeNodeParams& nodeParams);
    AddNodeResult   processNewBroadcastNode(const EagerNode& node);
    AddNodeResult   processNewMmeNode(const EagerNode& node);
    AddNodeResult   processNewTpcNode(EagerNode& node, bool userNode);
    AddNodeResult   processNewComplexGuidNode(const EagerNode& node);
    AddNodeResult   processStridedOp(const EagerNode& node);
    bool            addNodeDefault(pNode node);
    bool            processLogicalOp(const EagerNode& node, size_t nodeIdx, bool bwdPass);
    bool            handleMmeConcurrency(const EagerNode& node);
    bool            handleHugeTensor(const EagerNode& node, HugeTensorHandler& hugeTensorHandler);
    bool            wasExtractionPostponed(const EagerNode& node) const;
    bool editConsumersForZST(const EagerNode& node, const TensorPtr& subTensor);

    NodePtr createPhysicalMemcpy(const TensorPtr& input, const TensorPtr& output, std::string_view name);
    NodePtr getPhysicalMemset(synDataType         elementType,
                              const TensorVector& inputs,
                              const TensorVector& outputs,
                              std::string_view    nodeName);

    // Create memcpy node with clone of the original tensor
    // return the memcpy node and the cloned tensor
    std::pair<EagerNode, TensorPtr> createMemcpyNode(const TensorPtr& orig, bool copyToOrig);

    template<Node::eParamUsage USAGE>
    bool insertMemcpyNodes(const EagerNode& node, const LogicalOpNode::IndicesVec& indices);

    bool hasLogicalNodes() const { return m_nodeCollector.hasLogicalNodes(); }
    bool handle64bitPrecisionNode(EagerNode& node);
    bool handle64bitPrecisionNode(EagerNode& node, std::string_view newGuid);
    bool handle64bitPrecisionBroadcastNode(EagerNode& node, std::string_view newGuid);

private:
    // External vars to be initialized by constructor
    EagerGraph& m_eagerGraph;
    EagerNodes& m_nodes;

    ConstantTensorOptimizer   m_constantTensorOptimizer;          // utility class to optimize out constant\cast
    const bool                m_enableComplexGuidLib;             // Interact with complex GUID library
    bool                      m_isInternalNodeCheckDone = false;  // Flag to force robust adding new internal node
    bool                      m_disableSuggestedManipulation = false;
    NodeCollector             m_nodeCollector;
    EagerComplexGuidExtractor m_complexGuidExtractor;             // A single CGUID extractor

    friend class NodesContainer;
    friend class NodeCollector;
};

}  // namespace eager_mode
