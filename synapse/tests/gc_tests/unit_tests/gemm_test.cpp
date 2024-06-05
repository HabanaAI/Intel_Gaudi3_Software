#include <memory>
#include <gtest/gtest.h>
#include <math.h>
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "habana_nodes.h"
#include "sim_graph.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"
#include "scoped_configuration_change.h"

class GEMM : public GraphOptimizerTest {};

TEST_F(GEMM, cpu)
{
    bool ret = true;
    const TSize TEST_SIZE = 64;
    bf16_t* a = new bf16_t[TEST_SIZE * TEST_SIZE];
    bf16_t* b = new bf16_t[2 * TEST_SIZE * TEST_SIZE];

    for (TSize i = 0; i < TEST_SIZE * TEST_SIZE; ++i)
    {
        a[i] = bf16_t((float)i);
    }
    // create unit matrix
    for (TSize i = 0; i < TEST_SIZE; ++i)
    {
        for (TSize j = 0; j < TEST_SIZE; ++j)
        {
            if (i == j)
            {
                b[TEST_SIZE*j + i] = bf16_t(1.f);
            }
            else
            {
                b[TEST_SIZE*j + i] = bf16_t(0.f);
            }
        }
    }

    const TSize sizes[] = { TEST_SIZE, TEST_SIZE };

    TensorPtr A = TensorPtr(new Tensor(2U, sizes, syn_type_bf16, reinterpret_cast<char*>(a)));
    TensorPtr B = TensorPtr(new Tensor(2U, sizes, syn_type_bf16, reinterpret_cast<char*>(b)));
    TensorPtr C = TensorPtr(new Tensor(2U, sizes, syn_type_float));

    SimGraph g;
    synGEMMParams params;
    NodePtr       n  = NodeFactory::createNode({A, B, nullptr}, {C}, &params, NodeFactory::gemmNodeTypeName, "");
    GraphEditor::addNode(g, n);
    ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    ret = g.execute();
    ASSERT_EQ(ret, true) << "Failed to execute graph";

    //Validate the output
    float* c = reinterpret_cast<float*>(C->map());
    for (unsigned i = 0; i < TEST_SIZE * TEST_SIZE; ++i)
    {
        ASSERT_TRUE(c[i] == a[i].toFloat()) << "Wrong output at " << i << " Out: " << c[i] << " Ref: " << a[i].toFloat();
    }


    delete[] a;
    delete[] b;
}
