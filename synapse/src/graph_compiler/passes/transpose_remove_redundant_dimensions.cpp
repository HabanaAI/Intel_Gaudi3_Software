#include <habana_nodes/node_factory.h>
#include "habana_pass.h"
#include "synapse_common_types.hpp"
#include "transpose_node.h"
#include "habana_graph.h"
#include "graph_editor.h"
#include "transpose_utils.h"

static void pushOnesToEnd(NSizeArray& max_arr, NSizeArray& min_arr, unsigned dim)
{
    int count = 0;
    // Traverse the array. If element encountered is non-
    // one, then replace the element at index 'count'
    // with this element
    for (int i = 0; i < dim; i++)
    {
        if (max_arr[i] != 1)
        {
            max_arr[count] = max_arr[i];  // here count is incremented
            min_arr[count] = min_arr[i];
            count++;
        }
    }
    // Now all non-one elements have been shifted to
    // front and  'count' is set as index of first 1.
    // Make all elements 1 from count to end.
    while (count < dim)
    {
        max_arr[count] = 1;
        min_arr[count] = 1;
        count++;
    }
}

bool transposeRemoveRedundantDimensions(HabanaGraph& g)
{
    auto nodes = g.getNodes();
    for (const auto& node : nodes)
    {
        auto transposeNode = std::dynamic_pointer_cast<TransposeNode>(node);
        if (!transposeNode || node->getInput(TENSOR_IFM)->isShapeTensor())
        {
            continue;
        }
        HB_ASSERT(transposeNode->getNumInputs() == 1, "num of inputs must be 1");
        auto                  in = transposeNode->getInput(0);
        std::vector<unsigned> emptyDims;
        for (int i = 0; i < in->getDim(); ++i)
        {
            if (in->getSizeInElements(i) == 1) emptyDims.push_back(i);
        }

        // whether no size 1 dims, or all size 1 dims are at the end.
        if (emptyDims.empty() || in->getDim() - emptyDims.size() == emptyDims[0]) continue;

        TransposePermutationArray newArray = transposeNode->permutation();
        TransposePermutationArray originalArray = transposeNode->permutation();

        // Delete
        for (auto& value : emptyDims)
        {
            newArray.erase(std::remove(newArray.begin(), newArray.end(), value), newArray.end());
        }

        for (auto& value : newArray)
        {
            unsigned reducedBy =
                std::count_if(emptyDims.begin(), emptyDims.end(), [&](unsigned& deleted) { return deleted < value; });
            value = TransposePermutationDim((int) value - reducedBy);
        }

        for (int j = newArray.size(); j < in->getDim(); ++j)
        {
            newArray.emplace_back(TransposePermutationDim(j));
        }
        NSizeArray sizeArray    = in->getNSizesInElements();
        NSizeArray minSizeArray = in->getNMinimalSizesInElements();
        // Move all 1s to end;
        pushOnesToEnd(sizeArray, minSizeArray ,in->getDim());
        GraphEditor::editNode(g, transposeNode, [&](){
            transposeNode->setPermutation(newArray);
        });
        TensorPtr squeezedIn = in->cloneGeometry();
        squeezedIn->reshape(in->getDim(), sizeArray.data(), nullptr, minSizeArray.data());

        NodePtr squeezeNode = NodeFactory::createNode({in}, {squeezedIn}, nullptr, NodeFactory::squeezeNodeTypeName,
                transposeNode->getNodeName() + "_squeeze_in");

        GraphEditor::replaceTensor(g, transposeNode, transposeNode->getInput(TENSOR_IFM), squeezedIn);
        NSizeArray sizeArrayOut    = transposeNode->getOutput(0)->getNSizesInElements();
        NSizeArray minSizeArrayOut = transposeNode->getOutput(0)->getNMinimalSizesInElements();
        pushOnesToEnd(sizeArrayOut, minSizeArrayOut ,transposeNode->getOutput(0)->getDim());
        TensorPtr squeezedOut = transposeNode->getOutput(0)->cloneGeometry();
        squeezedOut->reshape(transposeNode->getOutput(0)->getDim(), sizeArrayOut.data(), nullptr, minSizeArrayOut.data());
        pTensor originalOut = transposeNode->getOutput(0);

        TensorPtr shapeOut = transposeNode->getOutput(0)->cloneGeometry();
        shapeOut->setShapeTensor(SHAPE_TENSOR);
        synTransposeParamsNDims params = permutationToParams(originalArray);

        NodePtr extractTranspose = NodeFactory::createNode({in},
                                                           {shapeOut},
                                                           &params,
                                                           NodeFactory::transposedShapeNodeTypeName,
                                                           transposeNode->getNodeName() + "transpose_shape");

        bool enforceLogical = true;
        NodePtr reshapeOut = NodeFactory::createNode({squeezedOut, shapeOut}, {originalOut}, &enforceLogical, NodeFactory::reshapeNodeTypeName,
                                transposeNode->getNodeName() + "_reshape_out");
        GraphEditor::replaceTensor(g, transposeNode, transposeNode->getOutput(TENSOR_OFM), squeezedOut);

        // Maintain tracking of origin nodes for debug purposes
        const auto& transposeNodeOriginNodes = transposeNode->getOriginNodes();
        extractTranspose->setOriginNodes(transposeNodeOriginNodes);
        squeezeNode->setOriginNodes(transposeNodeOriginNodes);
        reshapeOut->setOriginNodes(transposeNodeOriginNodes);

        GraphEditor::addNodes(g, {extractTranspose, squeezeNode, reshapeOut});
    }
    return true;
}
