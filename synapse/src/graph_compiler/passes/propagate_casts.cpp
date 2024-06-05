#include "defs.h"
#include "node.h"
#include "passes.h"
#include "habana_graph.h"
#include "synapse_common_types.h"
#include "scoped_configuration_change.h"
#include "types.h"
#include <algorithm>
#include "node_utils.h"

static bool isCastCandidateForPropagation(const NodePtr& n, const HabanaGraph& g)
{
    if ((!n->isCast() && !isConvertToFp8Node(n)) || g.getNumberOfTensorConsumers(n->getOutput(0)) == 0)
    {
        return false;
    }

    // Can try propagate only casts which will decrease size of the tensor.
    return (n->getOutput(0)->getElementSizeInBytes() < n->getInput(0)->getElementSizeInBytes());
}

// Check if node can contribute to future synapse optimizations, such as fusion passes.
static bool canBeOptimized(const NodePtr& n)
{
    // Transpose can be fused to MME.
    // Identity broadcast prevents fusions and will be removed anyway in later passes
    // so propagating cast above it and setting it to fp8 has no effect.
    return (n->isTranspose() ||
            (n->getNodeType() == Node::TYPE_BROADCAST && (n->getInput(0)->getShape() == n->getOutput(0)->getShape())));
}

// We can add node to cast propagation chain if all following conditions hold:
// 1. Node has single input and single output
// 2. Node output have single consumer
// 3. Node input and output element types are same
// 4. Node input isn't persistent
// 5. Node type is logical or it can allow future optimization
static bool canAddToPropagationChain(const NodePtr& n, const HabanaGraph& g)
{
    return (n != nullptr && n->getOutputs().size() == 1 && n->getInputs().size() == 1 &&
            g.getTensorConsumers(n->getOutput(0)).size() == 1 &&
            n->getInput(0)->getElementType() == n->getOutput(0)->getElementType() && !n->getInput(0)->isPersistent() &&
            (n->getNodeType() == Node::TYPE_SLICE || n->isLogicalOperation() || canBeOptimized(n)));
}

// Create nodes propagation chain.
// While conditions hold, keep adding nodes to the chain.
static NodeVector createPropagationChain(const NodePtr& castNode, const HabanaGraph& g)
{
    LOG_TRACE(DATA_TYPES, "Creating nodes chain to propagate cast");
    NodePtr    currentNode = g.getTensorProducer(castNode->getInput(0));
    NodeVector propagationChainNodes;
    while (canAddToPropagationChain(currentNode, g))
    {
        LOG_DEBUG(DATA_TYPES, "Can add current node {} to propagation chain", currentNode->getNodeName());

        propagationChainNodes.push_back(currentNode);
        // get the next node in the chain
        currentNode = g.getTensorProducer(propagationChainNodes.back()->getInput(0));
    }
    return propagationChainNodes;
}

static bool shouldPerformPropagation(NodeVector& propagationChain, const HabanaGraph& g)
{
    bool optimized = std::any_of(propagationChain.begin(), propagationChain.end(), [](const NodePtr n) {
        return canBeOptimized(n);
    });
    if (optimized) return true;
    NodePtr currentNode = g.getTensorProducer(propagationChain.back()->getInput(0));
    bool    mmeCastNode = currentNode->isCast() || g.runsOnMME(currentNode) || isFp8MmeCguid(currentNode) ||
                       isConvertFromFp8Node(currentNode);
    return mmeCastNode;
}

// Propagate casts with high precision to the above producer node to allow
// performance optimizations such as MME-Transpose fusion and tpc nodes
// fusion.
bool propagateCastNodes(HabanaGraph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(DATA_TYPES,
                  "Propagate cast nodes is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }
    if (!GCFG_PROPAGATE_CASTS.value())
    {
        LOG_DEBUG(DATA_TYPES, "Pass {} isn't configured to run.", HLLOG_FUNC);
        return true;
    }

    // Collect cast nodes which are candidates for propagation.
    const NodeSet& nodes = g.getNodes();
    NodeVector     castCandidateForPropagation;
    for (const NodePtr& node : nodes)
    {
        if (!isCastCandidateForPropagation(node, g)) continue;
        LOG_TRACE(DATA_TYPES, "Cast node {} is candidate for propagation", node->getNodeName());
        castCandidateForPropagation.push_back(node);
    }
    // TODO: Remove once [SW-136615] is done
    // This is need allow temporary state of ops with output type different from input type,
    // such as broadcast.
    // At the end of propagation of cast node, all the nodes will be in a state of
    // identical input and output type.
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false", true);
    for (const NodePtr& castNode : castCandidateForPropagation)
    {
        LOG_TRACE(DATA_TYPES, "Trying to propagate cast node {}", castNode->getNodeName());
        NodeVector propagationChainNodes = createPropagationChain(castNode, g);
        // Check first if propagation will have potential benefit.
        LOG_TRACE(DATA_TYPES, "Checking if should propagate cast node {}", castNode->getNodeName());
        if (propagationChainNodes.empty() || !shouldPerformPropagation(propagationChainNodes, g))
        {
            LOG_TRACE(DATA_TYPES, "Shouldn't propagate cast node {}", castNode->getNodeName());
            continue;
        }
        LOG_DEBUG(DATA_TYPES,
                  "Propagating cast node {} above {} nodes",
                  castNode->getNodeName(),
                  propagationChainNodes.size());

        // Propagate cast to end of the nodes chain
        synDataType typeToUpdate = castNode->getOutput(0)->getElementType();
        // replace start of the chain tensors
        NodePtr castConsumer = g.getTensorConsumers(castNode->getOutput(0)).front();
        auto    castConsumerQuantParams =
            castNode->getOutput(0)->getAllQuantizationParams();  // need to set this to the new cast output
        unsigned  inputIndex = castConsumer->getInputIndexOfTensor(castNode->getOutput(0));
        TensorPtr newInput   = castConsumer->getInput(inputIndex)->clone(false, false);
        GraphEditor::replaceOutput(g, propagationChainNodes.front(), 0, newInput);
        for (auto& castConsumer : g.getTensorConsumers(castNode->getOutput(0)))
        {
            inputIndex = castConsumer->getInputIndexOfTensor(castNode->getOutput(0));
            GraphEditor::replaceInput(g, castConsumer, inputIndex, newInput);
        }

        // replace end of the chain tensors
        TensorPtr chainEndInput = propagationChainNodes.back()->getInput(0);
        TensorPtr newCastOutput = chainEndInput->clone(false, false);
        GraphEditor::replaceInput(g, propagationChainNodes.back(), 0, newCastOutput);
        // replace cast tensors
        GraphEditor::replaceInput(g, castNode, 0, chainEndInput);
        GraphEditor::replaceOutput(g, castNode, 0, newCastOutput);
        newCastOutput->setAllQuantizationParams(castConsumerQuantParams);
        newCastOutput->changeDefaultElementType(typeToUpdate);
        LOG_TRACE(DATA_TYPES, "Finished propagating cast node {}", castNode->getNodeName());

        // update chain nodes intermediate outputs data type
        for (const NodePtr& chainNode : propagationChainNodes)
        {
            if (chainNode->getOutput(0)->getElementType() != typeToUpdate)
            {
                chainNode->getOutput(0)->changeDefaultElementType(typeToUpdate);
                chainNode->getOutput(0)->setAllQuantizationParams(castConsumerQuantParams);
            }
        }
    }
    return true;
}