#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"

class slice : public GraphOptimizerTest
{
};

void calcOutputSize(synSliceParams params, unsigned dim, TSize* outputSize)
{
    for (unsigned i = 0; i < dim; i++)
    {
        unsigned axis = params.axes[i];
        outputSize[axis] = (params.ends[i] - params.starts[i]) / params.steps[i];
    }
}

void createGraph(TSize inputSize[], TSize outputSize[], synSliceParams params,
                 unsigned numberOfDim, unsigned totalNumberOfNodes)
{
    GaudiGraph                 graph;
    CompilationHalReaderSetter compHalReaderSetter(&graph);
    TensorPtr sliceInput  = std::make_shared<Tensor>(numberOfDim, inputSize, syn_type_single);
    TensorPtr sliceOutput = std::make_shared<Tensor>(numberOfDim, outputSize, syn_type_single);

    NodePtr sliceNode = NodeFactory::createNode({sliceInput}, {sliceOutput},
                                                   &params, NodeFactory::sliceNodeTypeName, "slice_node");
    GraphEditor::addNode(graph, sliceNode);

    ASSERT_TRUE(extractMultiNodes(graph));
    ASSERT_TRUE(extractDataMovementMultiNodes(graph));

    const NodeVector& nodes = graph.getExeSortedNodes();
    ASSERT_EQ(graph.getNumNodes(), totalNumberOfNodes) << "Total node number should be equal to " << totalNumberOfNodes;

    if (totalNumberOfNodes != 1)
    {
        //Dma node should be present and transpose
        auto it = find_if(nodes.begin(), nodes.end(), [](const pNode& node) {
            std::shared_ptr<DMANode> current = std::dynamic_pointer_cast<DMANode>(node);
            if (current != nullptr)
            {
                return current->isTranspose();
            }
            return false;
        });
        ASSERT_FALSE(it == nodes.end());
    }

    TSize outputFromGraph[MAX_DIMENSIONS_NUM];
    nodes.back()->getOutput(0)->getAllSizesInElements(outputFromGraph, MAX_DIMENSIONS_NUM);

    for (unsigned i = 0; i < numberOfDim; i++) // check that the output shape is as planned after transpose
    {
        ASSERT_EQ(outputFromGraph[i], outputSize[i]) << "Output size isn't equal to the original output size";
    }
}

TEST_F(slice, slice_4d)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    synSliceParams sliceParams = {{0, 1, 2, 3}, {0, 0, 0, 0}, {4, 2, 2, 4}, {2, 2, 1, 1}};

    unsigned dim = 4;
    TSize inputSize[]  = {4, 2, 2, 4};
    TSize outputSize[dim];
    calcOutputSize(sliceParams, dim, outputSize);
    createGraph(inputSize, outputSize, sliceParams, dim, 11);
}

TEST_F(slice, slice_4d_axis_order)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    synSliceParams sliceParams = {{2, 1, 0, 3}, {0, 0, 0, 0}, {2, 2, 4, 4}, {1, 2, 2, 1}};

    unsigned dim = 4;
    TSize inputSize[]  = {4, 2, 2, 4};
    TSize outputSize[dim];
    calcOutputSize(sliceParams, dim, outputSize);
    createGraph(inputSize, outputSize, sliceParams, dim, 11);
}

TEST_F(slice, slice_3d)
{
    synSliceParams sliceParams = {{0, 1, 2}, {0, 0, 0}, {4, 2, 2}, {2, 2, 1}};

    unsigned dim = 3;
    TSize inputSize[]  = {4, 2, 2};
    TSize outputSize[dim];
    calcOutputSize(sliceParams, dim, outputSize);
    createGraph(inputSize, outputSize, sliceParams, dim, 12);
}

TEST_F(slice, logical_slice)
{
    synSliceParams sliceParams = {{0, 1, 2}, {0, 0, 0}, {4, 2, 2}, {1, 2, 1}};

    unsigned dim = 3;
    TSize inputSize[]  = {4, 2, 2};
    TSize outputSize[dim];
    calcOutputSize(sliceParams, dim, outputSize);
    createGraph(inputSize, outputSize, sliceParams, dim, 1);
}

TEST_F(slice, DISABLED_fcd_step_larger_than_1)
{
    synSliceParams sliceParams = {{0, 1, 2, 3}, {0, 0, 0, 0}, {4, 2, 2, 4}, {2, 2, 2, 2}}; // fcd step is 2 and no other axis with step 1 to transpose with

    unsigned dim = 4;
    TSize inputSize[]  = {4, 2, 2, 4};
    TSize outputSize[] = {0, 0, 0, 0};

    calcOutputSize(sliceParams, dim, outputSize);
    GaudiGraph graph;

    TensorPtr sliceInput  = std::make_shared<Tensor>(dim, inputSize, syn_type_single);
    TensorPtr sliceOutput = std::make_shared<Tensor>(dim, outputSize, syn_type_single);

    NodePtr sliceNode = NodeFactory::createNode({sliceInput}, {sliceOutput},
                                              &sliceParams, NodeFactory::sliceNodeTypeName, "slice_node");
    GraphEditor::addNode(graph, sliceNode);

    ASSERT_TRUE(graph.compile());
    ASSERT_EQ(graph.getNumNodes(), 6);  // dma_up, expand, slice, memcpy, dma_down

    synSliceParams validSliceParams = {{0, 1, 2, 3}, {0, 0, 0, 0}, {4, 2, 2, 4}, {1, 2, 2, 2}}; // fcd step is 1

    calcOutputSize(validSliceParams, dim, outputSize);
    GaudiGraph validGraph;

    TensorPtr validSliceInput  = std::make_shared<Tensor>(dim, inputSize, syn_type_single);
    TensorPtr validSliceOutput = std::make_shared<Tensor>(dim, outputSize, syn_type_single);

    NodePtr validSliceNode = NodeFactory::createNode({validSliceInput}, {validSliceOutput},
                                              &validSliceParams, NodeFactory::sliceNodeTypeName, "valid_slice_node");
    GraphEditor::addNode(validGraph, validSliceNode);

    ASSERT_TRUE(validGraph.compile());

}