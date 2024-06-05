#include "broadcast_node.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "hal_reader/hal_reader.h"
#include "node.h"
#include "node_factory.h"
#include "settable.h"
#include "transpose_utils.h"

unsigned getFirstBroadcastedDim(const pNode& broadcastNode)
{
    pTensor in  = broadcastNode->getInput(TENSOR_IFM);
    pTensor out = broadcastNode->getOutput(TENSOR_OFM);

    for (unsigned i = 0; i < Tensor::c_tensorMaxDim; ++i)
    {
        if (in->getSizeInElements(i) != out->getSizeInElements(i))
        {
            return i;
        }
    }
    return 0;
}

// A helper function to see if the FCD is broacasted,
// Where 1,1,1-> 1,1024,1 is technically not on FCD, but still, all dimensions are 1's untill fcd, so
// The real broadcast is on the FCD.

bool isFcdBroadcasted(const pNode& broadcastNode)
{
    pTensor in  = broadcastNode->getInput(TENSOR_IFM);
    pTensor out = broadcastNode->getOutput(TENSOR_OFM);

    unsigned firstBroadcastDim = getFirstBroadcastedDim(broadcastNode);
    for (unsigned i = 0; i < firstBroadcastDim; i++)
    {
        if (in->getSizeInElements(i) != 1)
        {
            return false;
        }
    }
    return true;
}

Settable<unsigned> getLargestDimAfterBroadcast(pNode broadcastNode)
{
    Settable<unsigned> retval;
    unsigned           firstBroadcastDim = getFirstBroadcastedDim(broadcastNode);
    pTensor            in                = broadcastNode->getInput(TENSOR_IFM);
    unsigned           maxDimSize        = 1;

    for (unsigned i = firstBroadcastDim; i < Tensor::c_tensorMaxDim; i++)
    {
        if (in->getSizeInElements(i) > maxDimSize)
        {
            retval     = i;
            maxDimSize = in->getSizeInBytes(i);
        }
    }
    return retval;
}

// If broadcast is on FCD we use stride 0 on a physical DMA operation.
// This results in reading 1 byte from a Cache line at a time, and horrible performance.
// To avoid this, we do transpose to a bigger FCD, then broadcast, and then transpose back.
// Exapmle: [1,256,12]->(broadcast)->[1024,256,12] will become:
//[1,256,12]->(transpose)->[256,1,12]->(broadcast)->[256,1024,12]->(transpose)->[1024,256,12]

bool transposeFcdBroadcast(HabanaGraph& g)
{
    const NodeVector nodes = g.getExeSortedNodes();
    for (const auto& node : nodes)
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_BROADCAST || node->getNodeType() == Node::TYPE_FCD_BROADCAST ||
            node->getNodeType() == Node::TYPE_BROADCAST)
        {
            if (node->getNodeType() == Node::TYPE_BROADCAST && GCFG_MAKE_BROADCAST_PHYSICAL.value())
            {
                continue;
            }
            TensorPtr in    = node->getInput(0);
            TensorPtr out   = node->getOutput(0);
            TensorPtr shape = nullptr;
            if (node->getInputs().size() > 1)
            {
                if (node->getInput(1)->getTensorType() == OUTPUT_DESCRIBING_SHAPE_TENSOR)
                {
                    shape = node->getInput(1);
                }
            }
            if (!isFcdBroadcasted(node))
            {
                continue;
            }
            // We can only transpose nodes with same dimensionality
            if (in->getDim() != out->getDim())
            {
                continue;
            }

            Settable<unsigned> largerstDim       = getLargestDimAfterBroadcast(node);
            unsigned           firstBroadcastDim = getFirstBroadcastedDim(node);
            // For now optimize if largest dim is greater than CL. We can later handle a case of
            // Where only a multiplication of dimensions is larger than CL and flatten them.
            if (!largerstDim.is_set())
            {
                continue;
            }

            TransposePermutationArray permutationArray;
            permutationArray                      = getIdentityPermutation(in->getDim());
            permutationArray[largerstDim.value()] = (TransposePermutationDim)firstBroadcastDim;
            permutationArray[firstBroadcastDim]   = (TransposePermutationDim)largerstDim.value();

            synTransposeParams params;
            params.tensorDim = in->getDim();
            memcpy(params.permutation,
                   permutationArray.data(),
                   std::min(sizeof(params.permutation),
                            permutationArray.size() * sizeof(TransposePermutationArray::value_type)));
            NSizeArray transposedTensorSizes    = in->getNSizesInElements();
            NSizeArray transposedBroadcastSizes = out->getNSizesInElements();
            transposedTensorSizes              = applyPermutationOnSizes(transposedTensorSizes, permutationArray);
            transposedBroadcastSizes           = applyPermutationOnSizes(transposedBroadcastSizes, permutationArray);
            TensorPtr transposeOutTensor =
                std::make_shared<Tensor>(in->getDim(), transposedTensorSizes.data(), in->getElementType());
            NodePtr  transposeInput = NodeFactory::createNode({in},
                                                             {transposeOutTensor},
                                                             &params,
                                                             NodeFactory::transposeNodeTypeName,
                                                             in->getName() + "transpose");
            NodeList nodesToAdd;
            nodesToAdd.push_back(transposeInput);
            TensorPtr transposedBroadcastOut = out->cloneGeometry();
            transposedBroadcastOut->reshape(out->getDim(), transposedBroadcastSizes.data(), nullptr);
            if (shape != nullptr)
            {
                TensorPtr broadcastShapeOut = shape->clone();
                broadcastShapeOut->reshape(in->getDim(), transposedBroadcastSizes.data(), nullptr);
                NodePtr extractTransposeShape = NodeFactory::createNode({shape},
                                                                        {broadcastShapeOut},
                                                                        &params,
                                                                        NodeFactory::transposedShapeNodeTypeName,
                                                                        shape->getName() + "transpose_shape");
                NodePtr transposedBroadcast =
                    NodePtr(new LogicalBroadcastNode({transposeOutTensor, broadcastShapeOut},
                                                     {transposedBroadcastOut},
                                                     node->getNodeName() + "_transposed_broadcast"));
                nodesToAdd.push_back(extractTransposeShape);
                nodesToAdd.push_back(transposedBroadcast);
            }
            else
            {
                NodePtr transposedBroadcast =
                    NodePtr(new LogicalBroadcastNode({transposeOutTensor},
                                                     {transposedBroadcastOut},
                                                     node->getNodeName() + "_transposed_broadcast"));
                nodesToAdd.push_back(transposedBroadcast);
            }
            NodePtr transposeBack = NodeFactory::createNode({transposedBroadcastOut},
                                                            {out},
                                                            &params,
                                                            NodeFactory::transposeNodeTypeName,
                                                            node->getNodeName() + "_transposed_broadcast_back");
            nodesToAdd.push_back(transposeBack);

            auto status = GraphEditor::replaceNodes(g, {node}, nodesToAdd);
            if (status != REPLACE_NODE_SUCCESS)
            {
                return false;
            }
        }
    }
    return true;
}
