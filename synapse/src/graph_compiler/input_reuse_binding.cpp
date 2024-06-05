#include "input_reuse_binding.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "identity_node.h"
#include "log_manager.h"
#include "handle_ctrl_edges_for_logical_ops.h"

bool InputInplaceReuseBinding::outputViableForInplace(const HabanaGraph& g,
                                                      const NodePtr&     node,
                                                      const TensorPtr&   nodeOutput) const
{
    return true;  // Binding => no choice - must be viable.
}

ReusePairsMap InputInplaceReuseBinding::getReusePairs(const NodePtr& node)
{
    return node->getReusableInputBinding();
}

void InputInplaceReuseBinding::validateOtherConsumerOrder(const HabanaGraph& g,
                                                          const NodePtr&     node,
                                                          const NodeSet&     consumers)
{
    for (const NodePtr& consumer : consumers)
    {
        if (consumer == node) continue;
        HB_ASSERT(g.getNumberOfPaths(consumer, node, Node::TENSOR_TYPE_ALL) > 0,
                  "node: {}, is missing a control edge from {}",
                  node->getNodeName(),
                  consumer->getNodeName());
    }
}

bool InputInplaceReuseBinding::applyReuse(HabanaGraph&        g,
                                          const NodePtr&      node,
                                          const TensorPtr&    nodeOutput,
                                          const TensorVector& reuseCandidates)
{
    for (const TensorPtr& t : reuseCandidates)
    {
        if (isAlreadyReused(g, t, *node))
        {
            validateOtherConsumerOrder(g, node, CtrlEdgesHandler(g).getRealConsumers(t));
            return true;
        }
        TensorPtr inputTensor  = t;
        TensorPtr outputTensor = nodeOutput;

        // In case the input tensor can't be reused, add memcpy after the input tensor and use memcpy's output as new
        // input tensor of the node
        if (g.getNumberOfTensorConsumers(inputTensor) > 1 || g.isUserManagedDram(inputTensor) ||
            inputTensor->isStaticParam())
        {
            NodePtr memcopy = GraphEditor::insertMemcpyForInput(g, node, inputTensor);
            inputTensor     = memcopy->getOutput(0);
        }

        // In case of aliased output, or persistent tensor or some memory overlap between the tensors then:
        // add memcpy before the output tensor and use memcpy's input as new output tensor of the node
        if (outputTensor->isAliasedTensor() || g.isUserManagedDram(outputTensor))
        {
            NodePtr memcopy = GraphEditor::insertMemcpyForOutput(g, node, outputTensor);
            outputTensor    = memcopy->getInput(0);
        }

        LOG_INFO(GC,
                 "Apply inplace reuse for binding request: node = {}, set tensor {} as aliased by tensor {}",
                 node->getNodeName(),
                 inputTensor->getName(),
                 outputTensor->getName());

        // Reaching this point implies gc will reuse one of the TPC node input tensors
        // to which the output will be written, Hence we set the output tensor as an alias of the input tensor.
        // Following [SW-72646] we also set the TPC node input tensor as RealInLogical such that
        // if the TPC node's producer is a logical op that hadn't been performed yet, a memcpy
        // will be planted between the logical node and it's TPC node consumer during
        // handle logical operations pass. The memcpy will ensure that the TPC node input tensor parameters
        // (such as strides) will not be modified when executing the (TPC producer) logical operation.
        IdentityNode::runLogicalOperation(inputTensor, outputTensor);
        inputTensor->setIsRealInLogical(true);
        return true;
    }
    HB_ASSERT(0, "Inplace reuse binding request failed for node {}", node->getNodeName());
    return false;
}

bool inPlaceInputReuseBinding(HabanaGraph& g)
{
    InputInplaceReuseBinding inplaceReuse;
    return inplaceReuse.runInputInplaceReuse(g);
}
