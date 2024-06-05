#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types_exception.h"

#include <array>

namespace
{
class frobeniusNormNodeTest : public GraphOptimizerTest
{
};

TEST_F(frobeniusNormNodeTest, test_no_params)
{
    std::array<TSize, 4> inSize  = {2, 1, 1, 3};
    std::array<TSize, 1> outSize = {1};

    TensorPtr inTensor  = std::make_shared<Tensor>(inSize.size(), inSize.data(), syn_type_bf16);
    TensorPtr outTensor = std::make_shared<Tensor>(outSize.size(), outSize.data(), syn_type_bf16);

    NodePtr frobeniusNorm = NodeFactory::createNode({inTensor},
                                                    {outTensor},
                                                    nullptr,
                                                    NodeFactory::FrobeniusNormTypeName,
                                                    "frobenius_norm_fwd");
    EXPECT_NE(frobeniusNorm, nullptr);
}

TEST_F(frobeniusNormNodeTest, test_invalid_node)
{
    std::array<TSize, 4> inSize  = {2, 1, 1, 3};
    std::array<TSize, 1> outSize = {1};

    TensorPtr inTensor  = std::make_shared<Tensor>(inSize.size(), inSize.data(), syn_type_float);
    TensorPtr outTensor = std::make_shared<Tensor>(outSize.size(), outSize.data(), syn_type_bf16);

    try
    {
        pNode frobeniusNorm = NodeFactory::createNode({inTensor},
                                                      {outTensor},
                                                      nullptr,
                                                      NodeFactory::FrobeniusNormTypeName,
                                                      "frobenius_norm_fwd");
    }
    catch (const InvalidNodeParamsException& e)
    {
        FAIL();
    }
}

}  // namespace