#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "params_file_manager.h"

class Gaudi2DsdDmaTest : public GraphOptimizerTest
{
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
    }
};

TEST_F(Gaudi2DsdDmaTest, basic)
{
    const unsigned tensor_dim           = 3;
    const TSize    maxSizes[tensor_dim] = {3, 4, 5};
    const TSize    minSizes[tensor_dim] = {2, 2, 2};
    Gaudi2Graph    g;
    TensorPtr      i = TensorPtr(
        new Tensor(tensor_dim, maxSizes, syn_type_single, nullptr, nullptr, true, false, INVALID_BATCH_POS, minSizes));
    TensorPtr o = TensorPtr(
        new Tensor(tensor_dim, maxSizes, syn_type_single, nullptr, nullptr, false, false, INVALID_BATCH_POS, minSizes));
    NodePtr             n = NodeFactory::createNode({i}, {o}, nullptr, "memcpy", "node1");
    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i->setDramOffset(0x1000);
    o->setDramOffset(0x2000);
    i->setMemoryDescriptor(memDesc);
    o->setMemoryDescriptor(memDesc);
    GraphEditor::addNode(g, n);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
}
