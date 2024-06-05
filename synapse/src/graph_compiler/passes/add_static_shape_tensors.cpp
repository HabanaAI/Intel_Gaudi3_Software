#include <habana_graph.h>
#include "graph_editor.h"

#include <log_manager.h>

static void addShapeTensorFromStaticTensor(HabanaGraph& graph, const NodePtr& node, const TensorPtr& originalTensor)
{
    auto newShapeTensor = originalTensor->cloneGeometry();
    newShapeTensor->setShapeTensor(SHAPE_TENSOR);
    // Mark tensor as internally created (needed for runtime)
    newShapeTensor->setProp(synTensorProperty::synTensorInternalNoProducer);
    auto newTensorIndex = node->getNumInputs();
    GraphEditor::editNode(graph, node, [&](const NodePtr& n){n->addInput(newTensorIndex, newShapeTensor);});
}

static bool isNonInferableConstNode(HabanaGraph& graph, const NodePtr& node, TensorPtr& outTakeShapeFrom)
{
    // Currently only recognizes a TPC constant node
    // More node types may be added in the future
    if (graph.runsOnTPC(node))
    {
        const auto& tpcNode = static_cast<TPCNode&>(*node);
        if (tpcNode.getGUIDWithoutDtype() == "constant" && tpcNode.getNumInputs() == 0)
        {
            HB_ASSERT(tpcNode.getNumOutputs() == 1, "Constant node {} with numOutputs != 1", tpcNode.getNodeName());
            outTakeShapeFrom = tpcNode.getOutput(0);
            HB_ASSERT(!outTakeShapeFrom->isDynamicShape(),
                      "Constant node {} with no inputs but with dynamic output",
                      tpcNode.getNodeName());
            return true;
        }
    }
    outTakeShapeFrom = nullptr;
    return false;
}

bool addStaticShapeTensors(HabanaGraph& graph)
{
    if (!graph.isDynamicShape())
    {
        return true;
    }

    const auto graphNodes = graph.getNodes();
    for (const NodePtr& node : graphNodes)
    {
        TensorPtr tensorToTakeShapeFrom; // filled by the call below
        if (isNonInferableConstNode(graph, node, tensorToTakeShapeFrom))
        {
            addShapeTensorFromStaticTensor(graph, node, tensorToTakeShapeFrom);
        }
    }

    return true;
}
