#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "tensor.h"
#include "types_exception.h"

#include <gtest/gtest.h>

class MMEGraphTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<const char*, synDataType, bool>>
{
public:
    void runTest();

private:
    static constexpr std::array<synDataType, 3> fp32Types {{syn_type_single, syn_type_tf32, syn_type_hb_float}};

    bool isFP32DataType(synDataType type)
    {
        return (std::find(fp32Types.begin(), fp32Types.end(), type) != fp32Types.end());
    }
};

void MMEGraphTest::runTest()
{
    auto guid     = std::get<0>(GetParam());
    auto dataType = std::get<1>(GetParam());
    auto gaudi2b  = std::get<2>(GetParam());

    Gaudi2Graph g;

    if (gaudi2b)
    {
        g.setFP32LimitedDevice();
    }

    const TSize sizes_x[] = {256, 256, 1, 1};
    const TSize sizes_w[] = {256, 256, 1, 1};
    const TSize sizes_y[] = {256, 256, 1, 1};

    TensorPtr x = TensorPtr(new Tensor(4U, sizes_x, dataType));
    TensorPtr w = TensorPtr(new Tensor(4U, sizes_w, dataType));
    TensorPtr y = TensorPtr(new Tensor(4U, sizes_y, dataType));

    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    x->setDramOffset(0x1000);
    w->setDramOffset(0x2000);
    y->setDramOffset(0x42000);
    x->setMemoryDescriptor(memDesc);
    w->setMemoryDescriptor(memDesc);
    y->setMemoryDescriptor(memDesc);

    synConvolutionParams params {};

    NodePtr conv = NodeFactory::createNode({x, w}, {y}, &params, guid, "mme_node");

    if (gaudi2b && isFP32DataType(dataType))
    {
        EXPECT_THROW({ GraphEditor::addNode(g, conv); }, DeviceLimitationFP32Exception);
    }
    else
    {
        EXPECT_NO_THROW({
            ASSERT_TRUE(GraphEditor::addNode(g, conv));
            ASSERT_TRUE(g.compile());
        });
    }
}

TEST_P(MMEGraphTest, gaudi2_mme_node_gc_test)
{
    runTest();
};

INSTANTIATE_TEST_SUITE_P(,
                         MMEGraphTest,
                         ::testing::Combine(::testing::ValuesIn({NodeFactory::convolutionNodeTypeName,
                                                                 NodeFactory::deDxNodeTypeName,
                                                                 NodeFactory::deDwNodeTypeName}),
                                            ::testing::ValuesIn({syn_type_fp8_143,
                                                                 syn_type_fp8_152,
                                                                 syn_type_bf16,
                                                                 syn_type_fp16,
                                                                 syn_type_tf32,
                                                                 syn_type_single,
                                                                 syn_type_hb_float}),
                                            ::testing::ValuesIn({false, true})));
