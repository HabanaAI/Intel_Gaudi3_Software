#include <memory>
#include <gtest/gtest.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "habana_nodes.h"
#include "sim_graph.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"
#include "gaudi_graph.h"
#include "compilation_hal_reader.h"
#include "scoped_configuration_change.h"

bool sliceGraphToSRAMCapacity(HabanaGraph& g);
bool internalTensorsDynamicShape(HabanaGraph& g);

namespace gaudi
{
    bool loadTpcKernels(GaudiGraph& g);
}

class ValidatePostSlicingSizes : public GraphOptimizerTest
{
public:

    TensorPtr createTensor(const std::vector<TSize>& shape,
            const std::vector<TSize>& minShape,
            bool  isPersistent,
            synTensorType tensorType = DATA_TENSOR)
    {
        synMemoryDescriptor memDesc(isPersistent);
        auto tensor = std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_bf16);
        tensor->setMemoryDescriptor(memDesc);
        if (isPersistent)
        {
            tensor->setMemorySectionID(m_memorySectionId++);
        }
        tensor->setDramOffset(0x1000000);
        tensor->map();
        tensor->setMinSize(minShape.data());
        tensor->setTensorType(tensorType);
        return tensor;
    }

private:

    int m_memorySectionId = 0;

};

TEST_F(ValidatePostSlicingSizes, simple)
{
    ScopedConfigurationChange PostSlicingShapeCheck("ENFORCE_POST_SLICING_SHAPE_CHECK", "true");

    const TSize MAX_SIZE = 640;
    const std::vector<TSize> sizes = { MAX_SIZE, MAX_SIZE };

    const TSize MIN_SIZE = 16;
    const std::vector<TSize> minSizes = { MIN_SIZE, MIN_SIZE };

    const TSize BAD_SIZE = 1330;
    const std::vector<TSize> badSizes = { BAD_SIZE, BAD_SIZE };

    TensorPtr A = createTensor(sizes, minSizes, true);
    TensorPtr B = createTensor(sizes, minSizes, true);
    TensorPtr C = createTensor(sizes, minSizes, false);
    TensorPtr D = createTensor(sizes, minSizes, false);
    TensorPtr E = createTensor(sizes, minSizes, false);
    TensorPtr F = createTensor(sizes, minSizes, false);
    TensorPtr G = createTensor(sizes, minSizes, true);

    NodePtr relu1 = NodeFactory::createNode({A}, {C}, nullptr, "relu_fwd_bf16", "");
    NodePtr relu2 = NodeFactory::createNode({B}, {D}, nullptr, "relu_fwd_bf16", "");
    NodePtr gemm  = NodeFactory::createNode({C, D}, {E}, nullptr, NodeFactory::gemmNodeTypeName, "");
    NodePtr relu3 = NodeFactory::createNode({E}, {F}, nullptr, "relu_fwd_bf16", "");
    NodePtr copy  = NodeFactory::createNode({F}, {G}, nullptr, NodeFactory::memcpyNodeTypeName, "");

    GaudiGraph graph;
    CompilationHalReaderSetter compHalReaderSetter(&graph);

    ASSERT_TRUE(GraphEditor::addNode(graph, relu1));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu2));
    ASSERT_TRUE(GraphEditor::addNode(graph, relu3));
    ASSERT_TRUE(GraphEditor::addNode(graph, gemm));
    ASSERT_TRUE(GraphEditor::addNode(graph, copy));

    ASSERT_TRUE(gaudi::loadTpcKernels(graph));

    ASSERT_TRUE(internalTensorsDynamicShape(graph));
    ASSERT_TRUE(sliceGraphToSRAMCapacity(graph));
    ASSERT_TRUE(internalTensorsDynamicShape(graph));

    // Check pre slicing sizes, it should succeed
    ASSERT_TRUE(validatePreSlicingSizes(graph));

    // Check that the slicer did create some bundles
    // and sliced GEMM and Relu nodes
    bool haveBundles = false;
    unsigned nGemms = 0;
    unsigned nRelus = 0;
    for (const auto& n : graph.getExeSortedNodes())
    {
        if (n->getNodeAnnotation().bundleInfo.is_set())
        {
            if (!haveBundles)
            {
                // Deliberately introduce an error in a bundle
                // The pre slicing sizes validator should not
                // be able to catch it
                n->getOutput(0)->setMinSize(badSizes.data());
                haveBundles=true;
            }
        }

        if (n->getNodeType() == Node::eNodeType::TYPE_GEMM)
        {
            ++nGemms;
        }
        else if (n->getGUID().compare(0, 4, "relu") == 0)
        {
            ++nRelus;
        }
    }
    ASSERT_TRUE(haveBundles);
    ASSERT_TRUE(nGemms > 1);
    ASSERT_TRUE(nRelus > 1);
    // Cannot catch errors in tensors added/sliced by the slicer
    ASSERT_TRUE(validatePreSlicingSizes(graph));

    // Make sure tensor F is now produced by an internal concat node
    auto concatNode = graph.getTensorProducer(F);
    ASSERT_TRUE(concatNode->getNodeType() == Node::eNodeType::TYPE_INTERNAL_CONCAT);
    // Deliberately introduce an error outside of a bundle
    F->setMinSize(badSizes.data());

    // Now check pre slicing sizes, it should fail
    ASSERT_FALSE(validatePreSlicingSizes(graph));
}
