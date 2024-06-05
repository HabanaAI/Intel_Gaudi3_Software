#include "eager_complex_guid_extractor.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_graph.h"
#include "graph_compiler/passes/ir_translation/synapse_graph_wrapper.hpp"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "utils/general_defs.h"

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerIRToSynapseTranslator
///////////////////////////////////////////////////////////////////////////////////////////////////
bool EagerIRToSynapseTranslator::createGCNode(const gc_protocol::ProtocolNode& node, NodePtr& createdNode)
{
    if (unlikely(node.isShapeManipulationOp)) return true;
    std::string_view nodeGuid(node.guid.data, node.guid.size);
    if (!m_constantTensorOptimizer.tryReplaceNodeByConstTensor(m_inputs,
                                                               m_outputs,
                                                               node.userParams.nodeParams,
                                                               node.userParams.nodeParamsSize,
                                                               nodeGuid))
    {
        createdNode = NodeFactory::createNode(m_inputs,
                                              m_outputs,
                                              node.userParams.nodeParams,
                                              node.userParams.nodeParamsSize,
                                              nodeGuid,
                                              static_cast<eager_mode::EagerGraph*>(m_originalGraph)->getNextNodeName());
        if (unlikely(!createdNode)) return false;
    }
    return true;
}

bool EagerIRToSynapseTranslator::startNodeTranslationToSynapse(HabanaGraph* graph, const NodePtr& origNode)
{
    LOG_DEBUG(GC_TRANSLATION, "Starting translation from protocolGraph to Synapse");
    m_originalGraph = graph;
    // Iterate over node's input and output tensors to store data.
    for (const auto& tensor : origNode->getInputs())
    {
        storeTensorData(tensor);
    }
    for (const auto& tensor : origNode->getOutputs())
    {
        storeTensorData(tensor);
    }
    m_originalNode = origNode;
    // foreachNode will invoke handleNode method implemented below
    if (!m_protocolGraph.foreachNode(*this))
    {
        return false;
    }

    return true;
}

bool EagerIRToSynapseTranslator::startTranslationToSynapse(HabanaGraph* graph)
{
    EAGER_ASSERT(false, "startTranslationToSynapse is not supported for EagerIRToSynapseTranslator");
    return false;
}

// Invoked from (protocolGraph::foreachNode).
// After GC node is created, it is added to m_createdNodes.
bool EagerIRToSynapseTranslator::handleNode(const ProtocolNode& irNode)
{
    NodePtr newNode = nullptr;
    if (!createGCNodeAndTensors(irNode, newNode)) return false;
    if (!newNode) return true;
    if (irNode.blockingNodeIds.begin() != irNode.blockingNodeIds.end())
    {
        EAGER_ASSERT(false, "cguid blockingNodeIds is not empty");
        m_irIdToGCNodeBlockingNodes.emplace(
            irNode.id,
            ir_translation_defs::IdsVector(irNode.blockingNodeIds.begin(), irNode.blockingNodeIds.end()));
    }
    m_createdNodes.push_back(std::move(newNode));
    m_irIdToGCNodeIdx.emplace(irNode.id, m_createdNodes.size() - 1);
    return true;
}

void EagerIRToSynapseTranslator::setGCTensorName(const ProtocolTensor& /*irTensor*/, Tensor& gcTensor)
{
    gcTensor.setName(static_cast<eager_mode::EagerGraph*>(m_originalGraph)->getNextTensorName(), true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerComplexGuidExtractor
///////////////////////////////////////////////////////////////////////////////////////////////////
tpc_lib_api::GlueCodeReturn EagerComplexGuidExtractor::calcExtract(HabanaGraph* g, const NodePtr& node)
{
    auto synToIrNodeWrapper = SynapseNodeWrapper(m_deviceID, true);
    synToIrNodeWrapper.setNode(node);
    auto extractRetCode = extractComplexGuid(synToIrNodeWrapper);
    // If CGUID doesn't return a success code (failure/graph unchanged), early exit.
    if (extractRetCode != tpc_lib_api::GLUE_SUCCESS)
    {
        // If CGUID hasn't changed the graph, this isn't a failure.
        if (extractRetCode != tpc_lib_api::GLUE_CGUID_GRAPH_UNCHANGED)
        {
            extractRetCode = tpc_lib_api::GLUE_FAILED;
        }
        return extractRetCode;
    }
    // Translated extracted protocol graph to synapse graph
    // Eager translator doesn't use node replacement, it returns extracted nodes instead.
    m_eagerProtocolToSynapseTranslator.emplace(*m_extractedGraph, m_constantTensorOptimizer);
    if (!m_eagerProtocolToSynapseTranslator->startNodeTranslationToSynapse(g, node))
    {
        LOG_ERR(GC_COMPLEX_GUID, "Translation for eager graph after CGUID extraction has failed");
        return tpc_lib_api::GLUE_FAILED;
    }
    if (!validateExtractedGraph(m_eagerProtocolToSynapseTranslator->getCreatedTensors()))
    {
        LOG_ERR(GC_COMPLEX_GUID, "Validation of graph extracted from node {} failed", node->getNodeName());
        return tpc_lib_api::GLUE_FAILED;
    }
    return extractRetCode;
}

bool EagerComplexGuidExtractor::isNodeNeedsExtract(const NodePtr&        node,
                                                   ComplexGUIDType       type,
                                                   tpc_lib_api::DeviceId deviceId)
{
    EAGER_ASSERT(type == FUNCTIONAL_COMPLEX_GUID, "performance cguid extraction is not supported for Eager");
    return KernelDB::instance().isSupportedFunctionalComplexGuid(node->getGUIDAndHash(), deviceId);
}

}  // namespace eager_mode