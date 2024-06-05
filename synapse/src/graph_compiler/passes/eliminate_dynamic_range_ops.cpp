#include "graph.h"
#include "passes.h"
#include "dynamic_range_node.hpp"

bool validateConsecutiveNodes(HabanaGraph& g, const NodePtr& n)
{
    // Consecutive dynamic range nodes validation.
    TensorPtr outputTensor    = n->getOutput(0);
    NodeList  outputConsumers = g.getTensorConsumers(outputTensor);
    for (auto node : outputConsumers)
    {
        if (node->getNodeType() == Node::TYPE_DYNAMIC_RANGE)
        {
            LOG_ERR(HABANA_NODE, "Cannot have two consecutive DynamicRangeNodes");
            return false;
        }
    }
    return true;
}

bool eliminateDynamicRangeOps(HabanaGraph& g)
{
    if (!GCFG_ENABLE_ELIMINATE_DYNAMIC_RANGE_OP.value())
    {
        LOG_DEBUG(QUANT, "Eliminate dynamic range nodes pass is disabled");
        return true;
    }
    const NodeVector graphNodes = g.getExeSortedNodes();

    for (const NodePtr& n : graphNodes)
    {
        if (n->getNodeType() == Node::TYPE_DYNAMIC_RANGE)
        {
            auto dynamicRangeNode = dynamic_cast<DynamicRangeNode*>(n.get());
            HB_ASSERT_PTR(dynamicRangeNode);
            // Check again if the node is still valid (wasn't improperly modified after creation),
            // and check it has no consecutive dynamic range nodes.
            if (!dynamicRangeNode->validateNode() || !validateConsecutiveNodes(g, n))
            {
                return false;
            }
            // A valid node, has a single non-null input and a single non-null output (only one can be persistent).
            TensorPtr inputTensor  = n->getInput(0);
            TensorPtr outputTensor = n->getOutput(0);
            auto inputProducer = g.getTensorProducer(inputTensor);
            // Regularly, the output tensor should be deleted, unless it was persistent.
            if (outputTensor->isPersistent())
            {
                if (!inputProducer)
                {
                    LOG_ERR(HABANA_NODE, "DynamicRangeNode output is persistent, and input has no producer (illegal)");
                    return false;
                }
                // Output tensor now has the dynamic range parameters
                outputTensor->setDynamicRange(dynamicRangeNode->getRange());
                // Input tensor should be deleted, and its producer becomes output tensor's producer.
                GraphEditor::removeNode(g, n);
                GraphEditor::replaceTensor(g, inputProducer, inputTensor, outputTensor);
            }
            // Output should be deleted, and all its consumers will consume input tensor instead.
            else
            {
                // Input tensor now has the dynamic range parameters
                inputTensor->setDynamicRange(dynamicRangeNode->getRange());
                // All output tensor's consumers will now consume the input instead.
                GraphEditor::removeNode(g, n, inputProducer);
                // if inputProducer is null, outputConsumers won't be updated in removeNode (has to be done manually).
                if (!inputProducer)
                {
                    auto outputConsumers = g.getTensorConsumers(outputTensor);
                    for (NodePtr consumer : outputConsumers)
                    {
                        GraphEditor::replaceTensor(g, consumer, outputTensor, inputTensor);
                    }
                }
            }
        }
    }
    return true;
}