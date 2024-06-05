#include <graph_compiler/types.h>
#include <tensor.h>
#include <platform/gaudi/graph_compiler/gaudi_graph.h>
#include <graph_compiler/habana_nodes/node_factory.h>
#include "graph_optimizer_test.h"

class ShapeOperationsTest : public GraphOptimizerTest
{
public:
    static TensorPtr createTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes = {})
    {
        if (minSizes.empty())
        {
            return std::make_shared<Tensor>(maxSizes.size(), maxSizes.data(), syn_type_float);
        }
        else
        {
            return std::make_shared<Tensor>(maxSizes.size(), maxSizes.data(), syn_type_float, minSizes.data());
        }
    }

    static TensorPtr createPersistentTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes = {})
    {
        auto tensor = createTensor(maxSizes, minSizes);
        synMemoryDescriptor memDesc(true /* persistent */);
        tensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
        tensor->setMemoryDescriptor(memDesc);
        tensor->map();
        return tensor;
    }

    static TensorPtr createShapeTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes)
    {
        return std::make_shared<Tensor>(maxSizes.size(),
                                        maxSizes.data(),
                                        syn_type_float,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        minSizes.data(),
                                        SHAPE_TENSOR);
    }
};

TEST_F(ShapeOperationsTest, extract_shape_node_should_forward_input_shape)
{
    const std::vector<TSize> maxSizes = {100, 200};
    const std::vector<TSize> minSizes = {50, 60};

    auto input = createPersistentTensor(maxSizes, minSizes);
    auto shape = createShapeTensor(maxSizes, minSizes);
    auto memsetOut = createPersistentTensor(maxSizes, minSizes);

    auto extractshape = NodeFactory::createNode({input},
                                                {shape},
                                                nullptr,
                                                NodeFactory::extractShapeNodeTypeName,
                                                "extract_shape");
    auto memset = NodeFactory::createNode({shape},
                                          {memsetOut},
                                          nullptr,
                                          NodeFactory::memsetNodeTypeName,
                                          "extract_shape");

    GaudiGraph graph;
    ASSERT_TRUE(GraphEditor::addNode(graph, extractshape));
    ASSERT_TRUE(GraphEditor::addNode(graph, memset));

    ASSERT_TRUE(graph.compile());

    const auto& allNodes = graph.getExeSortedNodes();
    ASSERT_EQ(allNodes.size(), 4);
    for (const auto& tensor : {shape, memsetOut})
    {
        ASSERT_EQ(tensor->getDim(), minSizes.size());
        for (unsigned dim = 0; dim < tensor->getDim(); dim++)
        {
            ASSERT_EQ(tensor->getMinimalSizeInElements(dim), minSizes[dim]);
        }
    }
}

TEST_F(ShapeOperationsTest, merge_shapes_node)
{
    TensorPtr in1 = createShapeTensor({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
    TensorPtr in2 = createShapeTensor({11, 12, 13, 14, 15}, {11, 12, 13, 14, 15});
    TensorPtr out = createShapeTensor({1, 12, 42, 2, 15}, {1, 1, 1, 1, 1});

    SifMergeShapesMetadata params;
    params.fillValue          = 42;
    params.outputDim          = 5;
    params.dimMap[0].inputIdx = 0;
    params.dimMap[0].dimIdx   = 0;
    params.dimMap[1].inputIdx = 1;
    params.dimMap[1].dimIdx   = 1;
    params.dimMap[2].inputIdx = -1;
    params.dimMap[3].inputIdx = 0;
    params.dimMap[3].dimIdx   = 1;
    params.dimMap[4].inputIdx = 1;
    params.dimMap[4].dimIdx   = 4;
    // params.dimMap[5].inputIdx = 0;
    // params.dimMap[5].dimIdx   = 0;

    NodePtr node = NodeFactory::createNode({in1, in2}, {out}, &params, NodeFactory::mergeShapesNodeTypeName, "merge");
    node->inferOutputsShape(synDeviceGaudi, false /* infer max */);

    for (unsigned dim = 0; dim < out->getDim(); dim++)
    {
        ASSERT_EQ(out->getSizeInElements(dim), out->getMinimalSizeInElements(dim)) << "wrong size at dim " << dim;
    }
}
