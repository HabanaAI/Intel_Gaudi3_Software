#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "squeeze_node.h"
#include "tensor.h"
#include "types_exception.h"

class SqueezeNodeTest : public GraphOptimizerTest
{
};

TEST_F(SqueezeNodeTest, test_no_params)
{
    TSize inSize[] = {2, 1, 1, 3};
    TSize outSize[] = {2, 3};

    pTensor inTensor(new Tensor(4, inSize, syn_type_bf16));
    pTensor outTensor(new Tensor(2, outSize, syn_type_bf16));

    pNode squeeze = NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::squeezeNodeTypeName, "");
    ASSERT_TRUE(squeeze != nullptr);
}

TEST_F(SqueezeNodeTest, test_with_params)
{
    TSize inSize[] = {2, 1, 1, 3};
    TSize outSize[] = {2, 1 ,3};

    pTensor inTensor(new Tensor(4, inSize, syn_type_bf16));
    pTensor outTensor(new Tensor(3, outSize, syn_type_bf16));
    unsigned axis = 1;

    pNode squeeze = NodeFactory::createNode({inTensor}, {outTensor}, &axis, NodeFactory::squeezeNodeTypeName, "");
    ASSERT_TRUE(squeeze != nullptr);
}



TEST_F(SqueezeNodeTest, test_invalid_node)
{
    TSize inSize[] = {2, 2, 1, 3};
    TSize outSize[] = {2, 1 ,3};

    pTensor inTensor(new Tensor(4, inSize, syn_type_bf16));
    pTensor outTensor(new Tensor(3, outSize, syn_type_bf16));
    unsigned axis = 1;
    bool isValidNode = true;
    try
    {
        pNode squeeze = NodeFactory::createNode({inTensor}, {outTensor}, &axis, NodeFactory::squeezeNodeTypeName, "");
    } catch(const InvalidNodeParamsException& e)
    {
        isValidNode = false;
    }
    ASSERT_EQ(isValidNode, false);
}
