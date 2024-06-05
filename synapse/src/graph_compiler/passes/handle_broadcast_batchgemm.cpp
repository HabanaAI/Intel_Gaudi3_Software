#include "handle_broadcast_batchgemm.h"

#include <tuple>
#include <utility>
#include "habana_graph.h"
#include "log_manager.h"
#include "node.h"
#include "node_visitor.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types.h"

MmeBroadcastBatchGemmNodeHandler::MmeBroadcastBatchGemmNodeHandler(const NodePtr& node)
: m_node(std::dynamic_pointer_cast<BatchGemmNode>(node))
{
    HB_ASSERT_PTR(node);
}

/** Batchgemm node is valid for broadcast if either:
 * 1. Its layout is partial broadcast
 * 2. Node is BATCH_GEMM_DEDW, since it requires broadcast for full or partial layout- as it can't be flattened and
 * handled as the other ops full broadcast are handled.
 */
bool MmeBroadcastBatchGemmNodeHandler::canExtract() const
{
    if (m_node == nullptr)
    {
        return false;
    }
    if (m_node->getNodeType() == Node::TYPE_BATCH_GEMM_DEDW || GCFG_ADD_EXPLICIT_BROADCAST_FOR_BGEMM.value())
    {
        if (m_node->isFullBroadcastLayout()) return true;
    }
    if (m_node->isPartialBroadcastLayout()) return true;
    return false;
}

const std::pair<NodePtr, NodeVector> MmeBroadcastBatchGemmNodeHandler::extract()
{
    calcExtract();
    return make_pair(m_newBatchGemmNode, m_newNodes);
}

bool MmeBroadcastBatchGemmNodeHandler::extract(HabanaGraph& graph)
{
    calcExtract();
    NodeList newNodesList(m_newNodes.begin(), m_newNodes.end());
    newNodesList.push_back(m_newBatchGemmNode);
    return (GraphEditor::replaceNodes(graph, {m_node}, newNodesList) == REPLACE_NODE_SUCCESS);
}

/** Calculate the new (=broadcasted) batch dimensions of the operand.
 * For any batch dimension that is == 1 in one operand and != 1 in the other,
 * broadcast that dimension accordingly.
 * For example:
 * opA dims = [4, 1, 2, H, C]
 * opB dims = [1, 3, 2, C, K]
 * Their new dimensions after broadcast:
 * opA new dims = [4,  *3*, 2, H, C]
 * opB new dims = [*4*, 3,  2, C, K]
 */

using CrossDimArray = std::array<bool, SYN_MAX_TENSOR_DIM>;

std::pair<bool, bool> MmeBroadcastBatchGemmNodeHandler::calcBroadcastNodeOutDims(const SizeArray& opASizes,
                                                                                 const SizeArray& opBSizes,
                                                                                 SizeArray&       bcastAOutSizes,
                                                                                 SizeArray&       bcastBOutSizes,
                                                                                 CrossDimArray&   A2BIndices,
                                                                                 CrossDimArray&   B2AIndices)
{
    bool broadcastOpA = false;
    bool broadcastOpB = false;
    for (int dim = DIM_GEMM_BATCH; dim < MAX_DIMENSIONS_NUM; dim++)
    {
        if (opASizes[dim] == opBSizes[dim]) continue;
        if (opASizes[dim] == 1)
        {
            broadcastOpA        = true;
            bcastAOutSizes[dim] = opBSizes[dim];
            B2AIndices[dim]     = true;
        }
        if (opBSizes[dim] == 1)
        {
            broadcastOpB        = true;
            bcastBOutSizes[dim] = opASizes[dim];
            A2BIndices[dim]     = true;
        }
    }
    return std::make_pair(broadcastOpA, broadcastOpB);
}

void MmeBroadcastBatchGemmNodeHandler::calcExtract()
{
    bool      broadcastOpA;
    bool      broadcastOpB;
    auto          opA      = m_node->getInput(TENSOR_IFM);
    auto          opB      = m_node->getInput(TENSOR_WEIGHT);
    SizeArray opASizes = opA->getAllSizesInElements();
    SizeArray opBSizes = opB->getAllSizesInElements();
    SizeArray bcastAOutSizes(opASizes);
    SizeArray bcastBOutSizes(opBSizes);
    CrossDimArray A2BIndices {};
    CrossDimArray B2AIndices {};

    std::tie(broadcastOpA, broadcastOpB) =
        calcBroadcastNodeOutDims(opASizes, opBSizes, bcastAOutSizes, bcastBOutSizes, A2BIndices, B2AIndices);

    TensorPtr bcastOpAOut, bcastOpBOut;

    TensorPtr bcastOpAShapeInput, bcastOpBShapeInput;

    if ((broadcastOpA || broadcastOpB) && (opA->isDynamicShape() || opB->isDynamicShape()))
    {
        // shape extraction!
        TensorPtr opAShape = opA->cloneGeometry();
        opAShape->setShapeTensor(SHAPE_TENSOR);

        TensorPtr opBShape = opB->cloneGeometry();
        opBShape->setShapeTensor(SHAPE_TENSOR);

        bcastOpAShapeInput = std::make_shared<Tensor>(opA->getDim(), bcastAOutSizes.data(), opA->getElementType());
        bcastOpAShapeInput->setShapeTensor(SHAPE_TENSOR);

        bcastOpBShapeInput = std::make_shared<Tensor>(opB->getDim(), bcastBOutSizes.data(), opB->getElementType());
        bcastOpBShapeInput->setShapeTensor(SHAPE_TENSOR);
        SifMergeShapesMetadata mergeShapeParamsA;
        SifMergeShapesMetadata mergeShapeParamsB;
        mergeShapeParamsA.outputDim = bcastOpAShapeInput->getDim();
        mergeShapeParamsB.outputDim = bcastOpBShapeInput->getDim();

        for (unsigned i = 0; i < SYN_MAX_TENSOR_DIM; ++i)
        {
            mergeShapeParamsA.dimMap[i].inputIdx = B2AIndices[i] ? 1 : 0;
            mergeShapeParamsB.dimMap[i].inputIdx = A2BIndices[i] ? 0 : 1;
            mergeShapeParamsA.dimMap[i].dimIdx   = i;
            mergeShapeParamsB.dimMap[i].dimIdx   = i;
        }

        NodePtr extractShapeA = NodeFactory::createNode({opA},
                                                        {opAShape},
                                                        nullptr,
                                                        NodeFactory::extractShapeNodeTypeName,
                                                        m_node->getNodeName() + "_extractShapeA");

        NodePtr extractShapeB = NodeFactory::createNode({opB},
                                                        {opBShape},
                                                        nullptr,
                                                        NodeFactory::extractShapeNodeTypeName,
                                                        m_node->getNodeName() + "_extractShapeB");

        NodePtr mergeShapeA = NodeFactory::createNode({opAShape, opBShape},
                                                      {bcastOpAShapeInput},
                                                      &mergeShapeParamsA,
                                                      NodeFactory::mergeShapesNodeTypeName,
                                                      m_node->getNodeName() + "_mergeShapesA");

        NodePtr mergeShapeB = NodeFactory::createNode({opAShape, opBShape},
                                                      {bcastOpBShapeInput},
                                                      &mergeShapeParamsB,
                                                      NodeFactory::mergeShapesNodeTypeName,
                                                      m_node->getNodeName() + "_mergeShapesB");

        m_newNodes.push_back(extractShapeA);
        m_newNodes.push_back(extractShapeB);
        m_newNodes.push_back(mergeShapeA);
        m_newNodes.push_back(mergeShapeB);
    }
    const bool canExplicitBroadcast = GCFG_ADD_EXPLICIT_BROADCAST_FOR_BGEMM.value() && m_node->isFullBroadcastLayout();
    if (broadcastOpA)
    {
        bcastOpAOut = std::make_shared<Tensor>(opA->getDim(), bcastAOutSizes.data(), opA->getElementType());
        if (canExplicitBroadcast)
        {
            reshapeForExplicitBroadcast(bcastOpAOut);
        }
        TensorVector inputs = {opA};
        if (bcastOpAShapeInput) inputs.push_back(bcastOpAShapeInput);
        pNode broadcastNode = NodeFactory::createNode(std::move(inputs),
                                                      {bcastOpAOut},
                                                      nullptr,
                                                      NodeFactory::broadcastNodeTypeName,
                                                      m_node->getNodeName() + "_broadcastOpA");
        LOG_TRACE(GC,
                  "Added broadcast node {} for operand {} of node {}",
                  broadcastNode->getNodeName(),
                  opA->getName(),
                  m_node->getNodeName());
        m_newNodes.push_back(broadcastNode);
    }
    if (broadcastOpB)
    {
        bcastOpBOut = std::make_shared<Tensor>(opB->getDim(), bcastBOutSizes.data(), opB->getElementType());
        if (canExplicitBroadcast)
        {
            reshapeForExplicitBroadcast(bcastOpBOut);
        }
        TensorVector inputs = {opB};
        if (bcastOpBShapeInput) inputs.push_back(bcastOpBShapeInput);
        pNode broadcastNode = NodeFactory::createNode(std::move(inputs),
                                                      {bcastOpBOut},
                                                      nullptr,
                                                      NodeFactory::broadcastNodeTypeName,
                                                      m_node->getNodeName() + "_broadcastOpB");
        LOG_TRACE(GC,
                  "Added broadcast node {} for operand {} of node {}",
                  broadcastNode->getNodeName(),
                  opB->getName(),
                  m_node->getNodeName());
        m_newNodes.push_back(broadcastNode);
    }

    // Create new batch gemm node - it will be symmetrical
    synGEMMParams params = m_node->getGEMMParams();
    m_newBatchGemmNode   = NodeFactory::createNode({broadcastOpA ? bcastOpAOut : opA, broadcastOpB ? bcastOpBOut : opB},
                                                 {m_node->getOutput(TENSOR_OFM)},
                                                 &params,
                                                 m_node->getGUID(),
                                                 m_node->getNodeName() + "_symmetrical");
}

void MmeBroadcastBatchGemmNodeHandler::reshapeForExplicitBroadcast(const TensorPtr& tensor)
{
    auto inA = m_node->getInput(TENSOR_IFM);
    auto inB = m_node->getInput(TENSOR_WEIGHT);

    unsigned   maxDim = std::max(inA->getDim(), inB->getDim());
    NSizeArray array;
    for (unsigned dim = 0; dim < maxDim; dim++)
    {
        if (dim < DIM_GEMM_BATCH)
        {
            array[dim] = tensor->getSizeInElements(dim);
        }
        else
        {
            array[dim] = std::max(inA->getSizeInElements(dim), inB->getSizeInElements(dim));
        }
    }
    tensor->reshape(maxDim, array.data(), nullptr);
}

bool handleBroadcastBatchGemm(HabanaGraph& g)
{
    if (!GCFG_ENABLE_IN_GRAPH_BROADCAST_FOR_BGEMM.value()) return true;

    NodeSet nodes = g.getNodes();
    for (const pNode& n : nodes)
    {
        MmeBroadcastBatchGemmNodeHandler broadcastHandler(n);
        if (broadcastHandler.canExtract())
        {
            CHECK_RET_FALSE(broadcastHandler.extract(g), "Failed broadcasting batchgemm node {}", n->getNodeName());
        }
    }
    return true;
}
