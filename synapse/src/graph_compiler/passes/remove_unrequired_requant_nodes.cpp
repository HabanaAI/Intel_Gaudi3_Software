#include "habana_graph.h"

#include "node.h"
#include "habana_pass.h"
#include "node_factory.h"
#include <memory>
#include "graph_editor.h"

bool keepRequant(HabanaGraph& g, pTensor requantInput, pTensor requantOutput)
{
    const NodePtr      requantInputNode  = g.getTensorProducer(requantInput);
    NodePtr            requantOutputNode = nullptr;
    std::list<NodePtr> outputList        = g.getTensorConsumers(requantOutput);
    if (!outputList.empty())
    {
        requantOutputNode = outputList.back();
    }

    if (requantInputNode == nullptr || requantOutputNode == nullptr)
    {
        return false;
    }

    if (!requantInputNode->isLogicalOperation() || !requantOutputNode->isLogicalOperation())
    {
        return false;
    }

    std::shared_ptr<LogicalOpNode> logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(requantInputNode);
    HB_ASSERT(logicalNode != nullptr, "requant input is not logical node even though it was marked as logical");
    if (!logicalNode->incompatibleWithNextNode(requantOutputNode->getNodeType()))
    {
        return false;
    }
    // if the input node is handled logical op, it means that the tensors are aliased and we can't remove the requant.
    if (!logicalNode->getRunLogicalOperationDone())
    {
        return false;
    }

    return true;
}

void removeQuantNode(NodePtr         requantNode,
                     const NodeList& nodesToChange,
                     pTensor         oldTensor,
                     pTensor         newTensor,
                     HabanaGraph&    g)
{
    // remove the unrequired requant node
    GraphEditor::removeNode(g, requantNode);

    for (NodePtr nodeToChange : nodesToChange)
    {
        // Change the nodes tensors so it will fit new graph
        GraphEditor::removeNode(g, nodeToChange);
        nodeToChange->replaceTensor(oldTensor, newTensor);
        GraphEditor::addNode(g, nodeToChange);

        if (nodeToChange->isCast())
        {
            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(nodeToChange);
            HB_ASSERT_PTR(tpcNode);
            tpcNode->resetInstantiated();
        }
    }
}

bool canMergeRequantToNode(const NodePtr& testNode)
{
    return (testNode->isCast() && testNode->getNodeAnnotation().insertedNode);
}

bool removeUnrequiredRequants(HabanaGraph& g)
{
    // Remove Requant nodes that have Cast consumers
    // as Cast nodes can perform Requant their selves.
    // Or on the other hand, remove requants that have cast producers

    // the data type does not make any difference, as matching will be on the guid prefix.
    static const char* requantGuid = "requant_i8";
    bool               pattern1Status;
    NodeSet            matchingNodes;
    GraphPtr           pattern1 = std::make_shared<Graph>();
    {
        // Create requant node
        auto reqIn = std::make_shared<Tensor>();
        auto reqOu = std::make_shared<Tensor>();

        NodePtr patternRequantNode = NodeFactory::createGenericTPCNode({reqIn}, {reqOu}, nullptr, requantGuid, "");

        pattern1Status = pattern1->addNode(patternRequantNode);
    }

    if (pattern1Status)
    {
        matchingNodes = g.matchPatternWithSingleOutputNode(pattern1.get(), NodeTypeMatchingFunc);
    }
    else
    {
        LOG_DEBUG(GC, "Pattern build failed for RemoveUnrequiredRequants pass");
    }

    unsigned int removedNodes = 0;

    // find all matches for req pattern
    for (const NodePtr& requantNode : matchingNodes)
    {
        const pTensor requantInput  = requantNode->getInput(0);
        const pTensor requantOutput = requantNode->getOutput(0);

        if (keepRequant(g, requantInput, requantOutput))
        {
            continue;
        }

        const NodeList outputConsumers = g.getTensorConsumers(requantOutput);

        // checking for preceding cast node
        // if the input to the requant is going only to the requant, and the producer of the input is cast node
        // only then the requant node can be removed.
        if (GraphEditor::canEliminateTensor(g, requantInput) &&
            canMergeRequantToNode(g.getTensorProducer(requantInput)))
        {
            LOG_DEBUG(GC,
                      "Removing requant node {} due to preceding cast node {}",
                      requantNode->getNodeName(),
                      g.getTensorProducer(requantInput)->getNodeName());

            NodeList nodesToChange(1, g.getTensorProducer(requantInput));
            removeQuantNode(requantNode, nodesToChange, requantInput, requantOutput, g);
            ++removedNodes;

            continue;
        }

        // check for following cast node
        // only if requant output has only one consumer and it is cast.
        if (GraphEditor::canEliminateTensor(g, requantOutput) && outputConsumers.size() != 0 &&
            canMergeRequantToNode(outputConsumers.front()))
        {
            LOG_DEBUG(GC,
                      "Removing requant node {} due to following cast node {}",
                      requantNode->getNodeName(),
                      outputConsumers.front()->getNodeName());

            NodeList nodesToChange(1, outputConsumers.front());
            removeQuantNode(requantNode, nodesToChange, requantOutput, requantInput, g);
            ++removedNodes;

            continue;
        }

        // Check for requant nodes where input and output has the same quantization info on selected dtype.
        if (requantInput->getQuantizationParams() == requantOutput->getQuantizationParams() &&
            !g.isOutputTensor(requantOutput) && !g.isInputTensor(requantInput))
        {
            bool canRemoveRequant = true;

            for (NodePtr outputConsumer : outputConsumers)
            {
                if (outputConsumer->isCast() && !outputConsumer->getNodeAnnotation().insertedNode)
                {
                    canRemoveRequant = false;
                }
            }

            if (canRemoveRequant)
            {
                LOG_DEBUG(GC,
                          "Removing requant node {} due to same selected dtype quantization info on input and output ",
                          requantNode->getNodeName());

                removeQuantNode(requantNode, outputConsumers, requantOutput, requantInput, g);
                ++removedNodes;

                continue;
            }
        }
    }

    LOG_DEBUG(GC, "Removed {} unrequired requant nodes", removedNodes);

    return true;
}
