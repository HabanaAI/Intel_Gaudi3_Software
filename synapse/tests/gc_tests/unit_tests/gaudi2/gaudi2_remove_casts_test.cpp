#include "gaudi2_graph.h"
#include "node_factory.h"
#include "graph_optimizer_test.h"
#include "synapse_common_types.h"

class Gaudi2OptimizeCastsTest : public GraphOptimizerTest
{
public:
    void setPersistent(TensorPtr& tensor)
    {
        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000);
    }
    void validateNumOfCastNodes(unsigned expected)
    {
        unsigned castNodeCount = 0;
        for (const auto& node : m_graph.getExeSortedNodes())
        {
            if ((node != nullptr) && (node->isCast()))
            {
                ++castNodeCount;
            }
        }
        ASSERT_EQ(castNodeCount, expected)
            << "found " << castNodeCount << " cast nodes in the graph but expected to " << expected;
    }
    Gaudi2Graph m_graph;
};

// Dedw node is sliced, which adds a reduction + cast node from fp32 to bf16 (cast is after the reduction). The next
// cast is opposite - bf16 to fp32 - so they should be removed.
TEST_F(Gaudi2OptimizeCastsTest, remove_opposite_casts_after_reduction)
{
    const unsigned       dims = 4;
    const TSize          b = 32, h = 16, w = 16;
    const TSize          c = 64, k = 4;
    synConvolutionParams params;
    const TSize          xSizes[]   = {c, w, h, b};
    const TSize          dwSizes[]  = {k, c, params.kW, params.kH};
    const TSize          dySizes[]  = {k,
                                convOutputDimSize(w, params.kW, params.dW, params.padL + params.padR, params.dilW),
                                convOutputDimSize(h, params.kH, params.dH, params.padT + params.padB, params.dilH),
                                b};
    TSize                inputSize  = b * h * w * c * dataTypeSizeInBytes(syn_type_bf16);
    TSize                outputSize = params.kW * params.kH * k * c * dataTypeSizeInBytes(syn_type_bf16);

    setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, std::to_string(inputSize / 2 + outputSize));
    auto x  = std::make_shared<Tensor>(dims, xSizes, syn_type_bf16);
    auto dy = std::make_shared<Tensor>(dims, dySizes, syn_type_bf16);
    auto dw = std::make_shared<Tensor>(dims, dwSizes, syn_type_bf16);
    setPersistent(dy);
    setPersistent(x);

    // Dedw will be sliced, so a reduction will be added + cast from fp32 to bf16 after the reduction.
    pNode dedwNode = NodeFactory::createNode({dy, x}, {dw}, &params, NodeFactory::deDwNodeTypeName, "dedw");

    // This cast is expected to be optimized out (it is an opposite cast to the one created by the sram management)
    auto  castOut  = std::make_shared<Tensor>(dims, dwSizes, syn_type_single);
    pNode castNode = NodeFactory::createNode({dw}, {castOut}, nullptr, "cast_bf16_to_f32", "cast");

    ASSERT_TRUE(GraphEditor::addNode(m_graph, dedwNode));
    ASSERT_TRUE(GraphEditor::addNode(m_graph, castNode));

    validateNumOfCastNodes(1);
    ASSERT_TRUE(m_graph.compile());
    validateNumOfCastNodes(0);
}