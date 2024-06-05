#include "habana_graph.h"
#include "habana_nodes.h"

static bool inputShapeTensorExist(const NodePtr& node)
{
    auto inputs = node->getInputs();
    for (auto input : inputs)
    {
        auto tensorType = input->getTensorType();
        if (tensorType == OUTPUT_DESCRIBING_SHAPE_TENSOR)
        {
            return true;
        }
    }
    return false;
}

static bool outputsConnectedToNodes(const TensorVector& outputs, const HabanaGraph& graph)
{
    return std::any_of(outputs.begin(), outputs.end(), [&](const TensorPtr& output) {
        return graph.getTensorConsumers(output).size() > 0;
    });
}

static bool addOutputTensorsToList(const HabanaGraph& graph, const NodePtr& node, TensorVector& outputsList)
{
    if (inputShapeTensorExist(node))
    {  // no need for special node handling since it has a dynamic input shape tensor.
        return false;
    }

    auto nodeOutputs = node->getOutputs();
    if (node->isDynamicShape() && outputsConnectedToNodes(nodeOutputs, graph))
    {
        outputsList.insert(outputsList.end(),
                           std::make_move_iterator(nodeOutputs.begin()),
                           std::make_move_iterator(nodeOutputs.end()));
    }
    return true;
}

static void addPostSifUpdatesToList(NodePtr node, TensorVector& linkedOutputsList)
{
    auto shapeNode      = node->getShapeNode();
    auto postSifUpdates = shapeNode->getPostSifUpdates();
    for (auto postSifUpdate : postSifUpdates)
    {
        linkedOutputsList.push_back(postSifUpdate.second);
    }
}

static void
gatherLinkedAndShaplessTensors(HabanaGraph& graph, TensorVector& shaplessOutputs, TensorVector& linkedOutputs)
{
    for (auto node : graph.getNodes())
    {
        bool addedOutputsToWatchList = false;

        if (node->isMemset())
        {
            addedOutputsToWatchList = addOutputTensorsToList(graph, node, shaplessOutputs);
        }

        if (!addedOutputsToWatchList)
        {  // no outputs were added, this could means the node is not a memset node OR that it has a shape
           // input node and could be used for post sif updates link
            addPostSifUpdatesToList(node, linkedOutputs);
        }
    }
}

static bool verifyShaplessOutputsAreLinked(TensorVector& shaplessOutputs, TensorVector& linkedOutputs)
{
    for (auto shapelessOutput : shaplessOutputs)
    {
        if (std::find(linkedOutputs.begin(), linkedOutputs.end(), shapelessOutput) == linkedOutputs.end())
        {
            return false;
        }
    }
    return true;
}

bool verifyMemsetOutputShapes(HabanaGraph& graph)
{
    TensorVector linkedOutputs;
    TensorVector shaplessOutputs;

    gatherLinkedAndShaplessTensors(graph, shaplessOutputs, linkedOutputs);
    return verifyShaplessOutputsAreLinked(shaplessOutputs, linkedOutputs);
}