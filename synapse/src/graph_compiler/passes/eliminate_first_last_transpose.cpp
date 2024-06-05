#include "habana_graph.h"
#include "habana_pass.h"
#include "transpose_node.h"
#include "node_factory.h"

#include "graph_editor.h"

typedef std::list<std::tuple<pNode, pTensor, pTensor>> NodesWithTensorsToReplace;

static void collectTransposeInfo(const HabanaGraph& g,
                                 const pNode& node,
                                 NodeList& transposesToRemove,
                                 NodesWithTensorsToReplace& nodesWithTensorsToReplace,
                                 bool input)
{
    if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE &&
        node->getNodeAnnotation().insertedNode)
    {
        HB_ASSERT(node->getInputs().size() == 1 && node->getOutputs().size() == 1, "must be 1 input and 1 output");
        pTensor inTensor  = node->getInput(0);
        pTensor outTensor = node->getOutput(0);
        if (inTensor->isStaticParam())
        {
            return;
        }
        NodeList nodesToUpdate;
        pTensor graphIOTensor, graphInternalTensor;
        if (input)
        {
            graphIOTensor       = inTensor;
            graphInternalTensor = outTensor;

            nodesToUpdate = g.getTensorConsumers(outTensor);
        }
        else
        {
            graphIOTensor       = outTensor;
            graphInternalTensor = inTensor;

            pNode producer = g.getTensorProducer(inTensor);
            nodesToUpdate.push_back(producer);
        }
        TSize sizes[Tensor::c_tensorMaxDim];
        graphInternalTensor->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);
        graphIOTensor->reshape(graphIOTensor->getDim(), sizes, nullptr);

        LOG_DEBUG(DATA_LAYOUT,
                  "Eliminate transpose {} on {} {}",
                  node->getNodeName(),
                  input ? "input" : "output",
                  graphIOTensor->getName());
        transposesToRemove.push_back(node);
        for (auto const& nodeToUpdate : nodesToUpdate)
        {
            nodesWithTensorsToReplace.emplace_back(std::make_tuple(nodeToUpdate, graphInternalTensor, graphIOTensor));
        }
    }
}

bool eliminateFirstLastTranspose(HabanaGraph& g)
{
    NodeList inputTransposesToRemove;
    NodeList outputTransposesToRemove;

    NodesWithTensorsToReplace nodesWithTensorsToReplace;

    if (GCFG_ELIMINATE_FIRST_TRANSPOSE.value())
    {
        LOG_TRACE(DATA_LAYOUT, "EliminateFirstLastTranspose: finding transposes on inputs");

        for (auto input : g.getGraphInputs())
        {
            std::list<pNode> nodes = g.getTensorConsumers(input);
            if (nodes.size() != 1)
            {
                continue;
            }
            pNode node = nodes.front();
            collectTransposeInfo(g, node, inputTransposesToRemove, nodesWithTensorsToReplace, true);
        }
    }
    if (GCFG_ELIMINATE_LAST_TRANSPOSE.value())
    {
        LOG_TRACE(DATA_LAYOUT, "EliminateFirstLastTranspose: finding transposes on outputs");

        for (auto input : g.getGraphOutputs())
        {
            pNode node =  g.getTensorProducer(input);
            collectTransposeInfo(g, node, outputTransposesToRemove, nodesWithTensorsToReplace, false);
        }
    }
    GraphEditor::removeNodes(g, outputTransposesToRemove);
    for (auto nodeWithTensors : nodesWithTensorsToReplace)
    {
        pNode node;
        pTensor oldTensor, newTensor;
        std::tie(node, oldTensor, newTensor) = nodeWithTensors;
        GraphEditor::replaceTensor(g, node, oldTensor, newTensor);
    }
    GraphEditor::removeNodes(g, inputTransposesToRemove);
    return true;
}
