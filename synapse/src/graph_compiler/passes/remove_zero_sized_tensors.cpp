#include "remove_zero_sized_tensors.h"

#include "defs.h"
#include "graph_editor.h"
#include "node_factory.h"
#include "node_visitor.h"
#include "tensor.h"
#include "types.h"

#include <algorithm>
#include <utility>

ZeroSizedTensorRemover::ZeroSizedTensorRemover(HabanaGraph& g) : m_deviceType(deviceTypeToDeviceID(g.getDeviceType()))
{
}

bool ZeroSizedTensorRemover::tpcIndexSpaceIsZero(const TPCNode& tpcNode)
{
    KernelInstantiationWrapper kernelInstance;
    bool                       instantiated = tpcNode.getInfoInstance(kernelInstance, m_deviceType, false);

    // There are cases when TPC nodes have to be handled by complex GUID before they can be instantiated.
    // In this case, their index space cannot be determined, and they will be handled by this pass after the complex
    // GUID pass is over.
    if (!instantiated)
    {
        LOG_DEBUG(ZST_REMOVER,
                  "Failed to instantiate the TPC node ({}) GUID ({}) in order to determine kernel's indexspace",
                  tpcNode.getNodeName(),
                  tpcNode.getGUID());
        return false;
    }

    auto& instance = kernelInstance.getInstance();
    return (multiplyElements(instance.indexSpaceGeometry, instance.indexSpaceGeometry + instance.indexSpaceRank) == 0);
}

bool ZeroSizedTensorRemover::handleZeroOutputTensorsCase(const NodePtr& node)
{
    LOG_DEBUG(ZST_REMOVER,
              "Found a node {} that all its output tensors are zero-sized; The node will be replaced.",
              node->getNodeName());
    return true;
}

bool ZeroSizedTensorRemover::handleNonZeroOutputTensorCase(const NodePtr&     node,
                                                           /*OUT*/ TensorPtr& subTensor)
{
    if (HabanaGraph::runsOnMME(node))
    {  // TODO distinguish between zero CD and non-zero CD. (SW-57848)
        LOG_DEBUG(ZST_REMOVER,
                  "The node, '{}' is an MME node with a zero-sized input tensor; The node will be replaced.",
                  node->getNodeName());
        return true;
    }
    if (node->isMemset() || node->isDma())
    {
        LOG_DEBUG(ZST_REMOVER,
                  "The node, '{}' is a DMA node with a zero-sized input tensor; The node will be removed.",
                  node->getNodeName());
        return true;
    }  // TODO Add support for logical nodes (SW-64768)
    if (HabanaGraph::runsOnTPC(node))
    {
        if (tpcIndexSpaceIsZero(static_cast<const TPCNode&>(*node)))
        {
            // Tpc node can't and won't handle zero-sized tensors. Replace the node.
            LOG_DEBUG(ZST_REMOVER,
                      "The node, '{}' has index space geometry which is 0; The node will be removed.",
                      node->getNodeName());
            return true;
        }
        LOG_DEBUG(ZST_REMOVER,
                  "The node, '{}' has index space geometry greater than 0; The node will be called as planned.",
                  node->getNodeName());
        return false;
    }
    if (node->getNodeType() == Node::TYPE_STRIDED_INSERT || node->getNodeType() == Node::TYPE_SLICE_INSERT)
    {
        const auto& originInput = node->getInput(0);
        const auto& insertInput = node->getInput(1);

        // Note: The single output isn't ZST since that is covered by handleZeroOutputTensorsCase
        HB_ASSERT(!originInput->isZeroSizedDataTensor(),
                  "in node {} insert tensor should have the same shape as output tensor",
                  node->getNodeName());
        if (insertInput->isZeroSizedDataTensor())  // TODO: wait, isn't this guaranteed? why do we check this?
        {
            // We reply on them having the same dataType layout etc. as the output, which it'll replace
            subTensor = originInput;  // The tensor will be connected directly to all of node's outputs consumers
            return true;
        }
    }
    return false;
}

/**
 * @brief Check whether a node with 1+ zero sized tensors should be removed and how to handle
 *        its outputs with non 0 amount of consumers.
 *
 * @param node      [in]  A node with zero sized tenosr(s) as either input/outputs
 * @param subTensor [out] dontcare if retvalue is false, otherwise the node should be removed and
 *                        - if not set, mark all consumers inputs from the node as const
 *                        - if set, replace the SINGLE output of the node with subTensor for all of
 *                          its consumers
 *
 * @return true if \p node should be dropped and its outputs handled
 */
bool ZeroSizedTensorRemover::handleZeroSizedOperand(const NodePtr& node, /*OUT*/ TensorPtr& subTensor)
{
    subTensor.reset();

    const TensorVector& outputTensors = node->getOutputs();
    bool                allOutputZst  = std::all_of(outputTensors.begin(), outputTensors.end(), [](const TensorPtr& t) {
        return t->isZeroSizedDataTensor();
    });

    return allOutputZst ? handleZeroOutputTensorsCase(node) : handleNonZeroOutputTensorCase(node, subTensor);
}

TensorPtr ZeroSizedTensorRemover::findFirstZST(const NodePtr& node)
{
    for (const auto& tv : {node->getInputs(), node->getOutputs()})
    {
        for (const TensorPtr& t : tv)
        {
            if (t && t->isZeroSizedDataTensor()) return t;
        }
    }
    return nullptr;
}



// Adjust consumed node outputs and remove it
static void editGraphForZSTNode(HabanaGraph& g, const NodePtr& n, const TensorPtr& subTensor)
{
    // Note that while the node is removed from the graph,
    // there's another ref to it from the calling func so it won't be freed.
    GraphEditor::removeNode(g, n);
    if (g.getNumNodes() == 0)
    {
        LOG_TRACE(ZST_REMOVER,
                  "After removing nodes with zero-sized tensors, there are no nodes left in the graph."
                  "Recipe left empty and still valid.");
    }

    const TensorVector& outputs = n->getOutputs();
    if (subTensor)
    {
        HB_ASSERT(outputs.size() == 1, "Expected a single output in case of subTensor");
        const TensorPtr& t = outputs.front();

        LOG_DEBUG(ZST_REMOVER,
                  "Since node '{}' removed from the graph, output tensor '{}' consumers switching to using its "
                  "input '{}' directly",
                  n->getNodeName(),
                  t->getName(),
                  subTensor);

        for (const NodePtr& consumer : g.getTensorConsumers(t))
        {
            for (size_t i = 0; i < consumer->getNumInputs(); ++i)
            {
                if (consumer->getInput(i) == t) GraphEditor::replaceInput(g, consumer, i, subTensor);
            }
        }
    }
    else
    {
        for (const TensorPtr& t : outputs)
        {
            if (!t || g.getNumberOfTensorConsumers(t) == 0 || t->isShapeTensor() || t->isZeroSizedDataTensor())
            {
                continue;
            }

            // TODO: if a removed ZST node has a non-ZDT dynamic shaped tensor,
            //       we need to have some inference mechanism per such guid of it's size,
            //       a generic memset wouln't do.
            HB_ASSERT(!t->isDynamicShape(), "Unsupported case (Where a simple memset node wouldn't suffice)");

            const auto memsetNode = NodeFactory::createNode({},
                                                            {t},
                                                            nullptr,
                                                            0,
                                                            NodeFactory::memsetNodeTypeName,
                                                            fmt::format("{}_memset", t->getName()));
            LOG_DEBUG(ZST_REMOVER,
                      "Since node '{}' removed from the graph, and output tensor '{}' isn't ZST nor shape, add "
                      "memset node '{}' to 0 it",
                      n->getNodeName(),
                      t->getName(),
                      memsetNode->getNodeName());
            const bool addedSuccessfully = GraphEditor::addNode(g, memsetNode);

            // Note that it's possible to return false and thus have the pass fail in such a case but a pass failure
            // would lead to a failed compilation, just as this would.
            HB_ASSERT(addedSuccessfully,
                      "Unexpecedly failed to validate and add memset node on addition for ZST nodes non-ZST output");
        }
    }
}

// Note: Zero-sized tensors might result from operators such as "where".
bool removeZeroSizedTensors(HabanaGraph& g)
{
    const bool disabledZstSupport = GCFG_DISABLE_ZST_SUPPORT.value();

    ZeroSizedTensorRemover optimizer(g);

    // Copy graph nodes since they may change during the pass, invalidating the sorted node cache.
    auto fn = [](const NodePtr& node){return ZeroSizedTensorRemover::hasZST(node);};
    NodeVector sortedNodes = g.getTopoSortedNodesCond(fn);

    for (const NodePtr& n : sortedNodes)
    {
        TensorPtr t = ZeroSizedTensorRemover::findFirstZST(n);
        if (t == nullptr) continue;

        if (unlikely(disabledZstSupport))
        {
            LOG_ERR(GC,
                    "Found a zero-sized tensor {} in the graph, while it is not supported "
                    "(DISABLE_ZST_SUPPORT=false). Enable the feature and run again.",
                    t->getName());
            return false;
        }
        LOG_TRACE(ZST_REMOVER, "Node \"{}\" contains a zero-sized operand: \"{}\"", n->getNodeName(), t->getName());

        TensorPtr subTensor;
        if (optimizer.handleZeroSizedOperand(n, subTensor))
        {
            editGraphForZSTNode(g, n, subTensor);
        }
    }
    return true;
}
