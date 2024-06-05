#include "replace_ops_with_logical_ops.h"

#include "defs.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "types.h"
#include "utils.h"

#include <numeric>

namespace replace_ops_with_logical_ops
{

bool canBeLogicalBroadcast(pNode node)
{
    // Pending dynamic shapes support
    if (node->getNodeType() != Node::TYPE_USER) return false;

    std::shared_ptr<TPCNode> pTpcNode = std::dynamic_pointer_cast<TPCNode>(node);
    if (pTpcNode == nullptr) return false;

    if (pTpcNode->getGUIDWithoutDtype() != "tile_fwd") return false;

    auto pParams = static_cast<ns_TileKernel::ParamsV2*>(pTpcNode->getParams());
    if (pParams == nullptr) return false;

    if ((node->getNumInputs() != 1) || (node->getNumOutputs() != 1)) return false;

    auto inputTensor  = node->getInput(0);
    auto outputTensor = node->getOutput(0);

    auto inputDim  = inputTensor->getDim();
    auto outputDim = outputTensor->getDim();

    auto repeatSize = pTpcNode->getParamsSize() / sizeof(pParams->repeat[0]);
    if (repeatSize < outputDim)
    {
        LOG_DEBUG(GC,
                  "{}() returned false for node {}. Tile repeat dim {} is smaller than output dim {}.",
                  HLLOG_FUNC,
                  node->getNodeName(),
                  inputDim,
                  outputDim);
        return false;
    }

    if (inputDim > outputDim)
    {
        LOG_DEBUG(GC,
                  "{}() returned false for node {}. Input dim {} is larger than output dim {}.",
                  HLLOG_FUNC,
                  node->getNodeName(),
                  inputDim,
                  outputDim);
        return false;
    }

    if (pParams->repeat[0] != 1 && !GCFG_MAKE_BROADCAST_PHYSICAL.value())
    {
        LOG_DEBUG(GC,
                  "{}() returned false for node {}. Tile op on FCD can't be converted to logical broadcast.",
                  HLLOG_FUNC,
                  node->getNodeName());
        return false;
    }

    int nonShapeInputs  = node->getNumInputs() - node->getNumInputsShapeTensors();
    int nonShapeOutputs = node->getNumOutputs() - node->getNumOutputsShapeTensors();

    if (nonShapeInputs != 1 || nonShapeOutputs != 1)
    {
        LOG_DEBUG(GC,
                  "{}: can't replace tile with broadcast. Non shape inputs: {}. Non shape outputs {}.",
                  HLLOG_FUNC,
                  nonShapeInputs,
                  nonShapeOutputs);
        return false;
    }

    if (node->isDynamicShape())
    {
        LOG_DEBUG(GC, "{}: can't replace dynamic tile with broadcast", HLLOG_FUNC);
        return false;
    }

    if (std::accumulate(pParams->repeat, pParams->repeat + outputDim, 1, std::multiplies<int>()) == 1)
    {
        LOG_DEBUG(GC, "{}() returned false for node {}. Tile is trivial.", HLLOG_FUNC, node->getNodeName());
        return false;
    }

    for (unsigned i = 0; i < outputDim; i++)
    {
        if (inputTensor->getDim() < i)
        {
            break;
        }
        if ((pParams->repeat[i] != 1) && (inputTensor->getSizeInElements(i) != 1))
        {
            LOG_DEBUG(GC,
                      "{}() returned false for node {}. Tile params don't correspond to input sizes.",
                      HLLOG_FUNC,
                      node->getNodeName());
            return false;
        }
    }

    return true;
}

bool replaceTileWithLogicalBroadcast(const NodePtr& node, HabanaGraph& g)
{
    pNode logicalBroadcastNode = NodeFactory::createNode(node->getInputs(),
                                                         node->getOutputs(),
                                                         nullptr,
                                                         NodeFactory::broadcastNodeTypeName,
                                                         node->getNodeName());

    return (GraphEditor::replaceNodes(g, {node}, {logicalBroadcastNode}) == REPLACE_NODE_SUCCESS);
}

void createBroadcastNode(const TensorPtr& input,
                         const TensorPtr& output,
                         unsigned         axis,
                         unsigned         size,
                         NodeList&        newNode)
{
    output->setName(input->getName() + "_broadcasted_" + std::to_string(input->getId()));

    auto outputShape  = input->getNSizesInElements();
    outputShape[axis] = size;

    output->reshape(input->getDim(), outputShape.data(), nullptr);
    newNode.push_back(NodeFactory::createNode({input},
                                              {output},
                                              nullptr,
                                              NodeFactory::broadcastNodeTypeName,
                                              input->getName() + "_broadcast_" + std::to_string(input->getId())));
}

/*  for concat node with input tenor of shape [W, X, Y, Z] that is N times
   input, and concatenate dim 1 (for example) the output tensor shape is [W, X *
   N, Y, Z] and we must insert N - 1 memcopies to the graph instead we replace
   the concat node with: reshape [W * X, 1, Y * Z] -> broadcast [W * X, N, Y *
   N] -> reshape [W, X * N, Y, Z] and at most 2 memcopies will insert to the
   graph

    return the "output" tensor of the broadcast sequence
*/
static TensorPtr createBroadcastSequence(NodeList& newNodes, const TensorPtr& inputTensor, unsigned axis, unsigned size)
{
    TensorPtr t1 = inputTensor->clone(false, false);

    // In case broadcast axis has a single element, there is no need of wrapping reshape nodes.
    if (inputTensor->getSizeInElements(axis) == 1)
    {
        createBroadcastNode(inputTensor, t1, axis, size, newNodes);
        return t1;
    }

    // calculates new dim sizes
    TSize newDims[3] = {1, 1, 1};
    for (unsigned dim = 0; dim <= axis; ++dim)
    {
        newDims[0] *= inputTensor->getSizeInElements(dim);
    }
    for (unsigned dim = axis + 1; dim < inputTensor->getDim(); ++dim)
    {
        newDims[2] *= inputTensor->getSizeInElements(dim);
    }
    // t1 is the tensor between first reshape and broadcast
    t1->reshape(3, newDims, nullptr);
    t1->setName(inputTensor->getName() + "_reshaped_0_" + std::to_string(t1->getId()));

    TensorPtr t2 = inputTensor->clone(false, false);  // tensor between broadcast and second reshape

    newNodes.push_back(
        NodeFactory::createNode(TensorVector({inputTensor}),
                                TensorVector({t1}),
                                nullptr,
                                NodeFactory::reshapeNodeTypeName,
                                inputTensor->getName() + "_first_reshape_" + std::to_string(t2->getId())));

    constexpr unsigned broadcastAxis = 1;
    createBroadcastNode(t1, t2, broadcastAxis, size, newNodes);

    TensorPtr t3       = inputTensor->clone(false, false);  // tensor after second reshape (output of sequence)
    SizeArray t3_sizes = inputTensor->getAllSizesInElements();
    t3_sizes[axis] *= size;
    t3->reshape(inputTensor->getDim(), t3_sizes.data(), nullptr);
    t3->setName(inputTensor->getName() + "_reshaped_1_" + std::to_string(t1->getId()));

    newNodes.push_back(
        NodeFactory::createNode(TensorVector({t2}),
                                TensorVector({t3}),
                                nullptr,
                                NodeFactory::reshapeNodeTypeName,
                                inputTensor->getName() + "_second_reshape_" + std::to_string(t2->getId())));

    return t3;
}

// try replace concat with broadcast if all inputs are same
static void optimizeConcatOp(HabanaGraph& g, const std::shared_ptr<LogicalOpNode>& logicalNode)
{
    // TODO add support for dynamic shape
    if (g.isDynamicShape()) return;
    // node isn't concat node
    if (logicalNode->getNodeType() != Node::eNodeType::TYPE_INTERNAL_CONCAT) return;
    const TensorVector& inputs = logicalNode->getInputs();
    // small concat, optimization isn't needed
    if (inputs.size() <= 2) return;
    // quick check if there are no adjacent inputs
    if (std::adjacent_find(inputs.begin(), inputs.end()) == inputs.end()) return;

    synConcatenateParams concatDim;
    concatDim.axis = (reinterpret_cast<ConcatenateNode*>(logicalNode.get()))->getAggregationDim();
    NodeList     newNodes;         // nodes that should replace the concat
    TensorVector newConcatInputs;  // inputs to the new concat if needed

    // loop over the inputs to find sequences of inputs that are the same tensor and replace them with broadcast
    unsigned startIndex = 0;
    unsigned endIndex   = 1;
    while (endIndex <= inputs.size())
    {
        if (endIndex != inputs.size() && inputs.at(endIndex) == inputs.at(startIndex))  // input is same
        {
            ++endIndex;
            continue;
        }
        // input is different so we check if optimization is needed
        if (endIndex - startIndex > 2)
        {
            newConcatInputs.push_back(
                createBroadcastSequence(newNodes, inputs.at(startIndex), concatDim.axis, endIndex - startIndex));
            startIndex = endIndex;  // update the index to create new sequence
        }
        else  // no optimization needed
        {
            while (startIndex < endIndex)
            {
                newConcatInputs.push_back(inputs.at(startIndex));  // push unoptimized inputs to new concat input vector
                ++startIndex;
            }
        }
        ++endIndex;
    }
    if (newConcatInputs.size() == 1)  // all inputs are same tensor, we can use the broadcast without new concat
    {
        newNodes.back()->replaceOutput(0, logicalNode->getOutput(0));
    }
    else if (newConcatInputs.size() != inputs.size())  // some inputs are optimized, create new concat to connect them
    {
        newNodes.push_back(NodeFactory::createNode(newConcatInputs,
                                                   TensorVector({logicalNode->getOutput(0)}),
                                                   &concatDim,
                                                   NodeFactory::concatenateNodeLogicalInternalTypeName,
                                                   logicalNode->getNodeName()));
    }

    if (!newNodes.empty())
    {
        auto ret = GraphEditor::replaceNodes(g, {logicalNode}, newNodes);
        HB_ASSERT(ret == REPLACE_NODE_SUCCESS, "failed to replace nodes");
    }
}

bool tryReplaceMemcopyWithIdentity(HabanaGraph& g, const NodePtr& node)
{
    if (node->getNodeType() != Node::TYPE_MEMCOPY) return false;
    // if any of the tensors is not a data tensor (H2D for example) we need an actual memcopy
    if (node->getInput(0)->getTensorType() != DATA_TENSOR) return false;
    if (node->getOutput(0)->getTensorType() != DATA_TENSOR) return false;
    // if both input and output are persistent we will need a memcopy anyway
    if (node->getInput(0)->isUserManagedDram() && node->getOutput(0)->isUserManagedDram()) return false;
    // this case will be converted to a physical cast
    if (node->getInput(0)->getElementType() != node->getOutput(0)->getElementType()) return false;
    // if actually memory movement is performed
    if (node->getInput(0)->inSram() || node->getOutput(0)->inSram()) return false;
    // if one of the operands i strided we might need this memcopy
    if (!node->getInput(0)->isTrivialStrided()) return false;
    if (!node->getOutput(0)->isTrivialStrided()) return false;

    NodePtr identity = NodeFactory::createNode(node->getInputs(),
                                               node->getOutputs(),
                                               nullptr,
                                               NodeFactory::identityNodeTypeName,
                                               node->getNodeName());
    bool    status   = GraphEditor::replaceNodes(g, {node}, {identity});
    HB_ASSERT(status == REPLACE_NODE_SUCCESS,
              "{}: failed replacing memcopy {} with identity",
              __func__,
              node->getNodeName());
    LOG_DEBUG(GC, "Succesfully replaced memcopy not replace memcopy {} with identity", node->getNodeName());
    return true;
}

bool replaceOpsWithLogicalOps(HabanaGraph& g)
{
    NodeVector allNodes = g.getExeSortedNodes();
    for (auto node : allNodes)
    {
        // try replace tile with broadcast
        if (canBeLogicalBroadcast(node))
        {
            LOG_DEBUG(GC,
                      "Replacing tile node: {} with broadcast node: {}",
                      node->getNodeName(),
                      NodeFactory::broadcastNodeTypeName);

            if (!replaceTileWithLogicalBroadcast(node, g))
            {
                LOG_ERR(GC, "Failed to replace tile node with broadcast node");
                return false;
            }
        }
        // try replace concat with broadcast
        if (node->getNodeType() == Node::TYPE_INTERNAL_CONCAT)
        {
            if (!node->isLogicalOperation() || node->isDebug()) continue;
            auto logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(node);
            HB_ASSERT_PTR(logicalNode);
            if (logicalNode->getRunLogicalOperationDone()) continue;
            optimizeConcatOp(g, logicalNode);
        }
        // try replace memcopy with identity    - TODO remove when [SW-98183] is merged
        if (GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY.value() && node->getNodeType() == Node::TYPE_MEMCOPY)
        {
            if (!tryReplaceMemcopyWithIdentity(g, node))
            {
                LOG_DEBUG(GC, "Could not replace memcopy {} with identity", node->getNodeName());
            }
        }
    }
    return true;
}
};  // namespace replace_ops_with_logical_ops

bool replaceOpsWithLogicalOps(HabanaGraph& g)
{
    return replace_ops_with_logical_ops::replaceOpsWithLogicalOps(g);
}