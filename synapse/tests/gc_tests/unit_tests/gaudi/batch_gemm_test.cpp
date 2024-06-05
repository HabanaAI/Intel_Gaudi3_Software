#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "utils.h"
#include <vector>

class GaudiBatchGemmNode : public GraphOptimizerTest
{

};

TEST_F(GaudiBatchGemmNode, batch_gemm_3d_symmetric)
{
    unsigned dims = 3;
    const SizeVector            input0Size {256, 128, 2};
    const SizeVector            input1Size {256, 256, 2};
    const TSize                 outputSize[] = {256, 128, 2, 0, 0};

    TensorPtr input0Tensor = TensorPtr(new Tensor(dims, input0Size.data(), syn_type_fixed));
    TensorPtr input1Tensor = TensorPtr(new Tensor(dims, input1Size.data(), syn_type_fixed));
    TensorPtr outputTensor = TensorPtr(new Tensor(dims, outputSize, syn_type_int32));

    synGEMMParams                  params;
    NodePtr                        node = NodeFactory::createNode({input0Tensor, input1Tensor},
                                           {outputTensor},
                                           &params,
                                           NodeFactory::batchGemmNodeTypeName,
                                           "");
    std::shared_ptr<BatchGemmNode> bgn  = std::dynamic_pointer_cast<BatchGemmNode>(node);

    bool isAsymmetricNode = bgn->isFullBroadcastLayout();
    bool isSymmetricNode  = bgn->isSymmetricLayout();

    bool isAsymmetricStatic = BatchGemmNode::isFullBroadcastLayout(input0Size, input1Size);
    bool isSymmetricStatic  = BatchGemmNode::isSymmetricLayout(input0Size, input1Size);

    ASSERT_FALSE(isAsymmetricNode);
    ASSERT_TRUE(isSymmetricNode);

    ASSERT_EQ(isAsymmetricNode, isAsymmetricStatic);
    ASSERT_EQ(isSymmetricNode, isSymmetricStatic);
}

TEST_F(GaudiBatchGemmNode, batch_gemm_5d_symmetric)
{
    unsigned                    dims = 5;
    const SizeVector            input0Size {256, 128, 2, 4, 5};
    const SizeVector            input1Size {256, 256, 2, 4, 5};
    const TSize                 outputSize[] = {256, 128, 2, 4, 5};

    TensorPtr input0Tensor = TensorPtr(new Tensor(dims, input0Size.data(), syn_type_fixed));
    TensorPtr input1Tensor = TensorPtr(new Tensor(dims, input1Size.data(), syn_type_fixed));
    TensorPtr outputTensor = TensorPtr(new Tensor(dims, outputSize, syn_type_int32));

    synGEMMParams                  params;
    NodePtr                        node = NodeFactory::createNode({input0Tensor, input1Tensor},
                                           {outputTensor},
                                           &params,
                                           NodeFactory::batchGemmNodeTypeName,
                                           "");
    std::shared_ptr<BatchGemmNode> bgn  = std::dynamic_pointer_cast<BatchGemmNode>(node);

    bool isAsymmetricNode = bgn->isFullBroadcastLayout();
    bool isSymmetricNode  = bgn->isSymmetricLayout();

    bool isAsymmetricStatic = BatchGemmNode::isFullBroadcastLayout(input0Size, input1Size);
    bool isSymmetricStatic  = BatchGemmNode::isSymmetricLayout(input0Size, input1Size);

    ASSERT_FALSE(isAsymmetricNode);
    ASSERT_TRUE(isSymmetricNode);

    ASSERT_EQ(isAsymmetricNode, isAsymmetricStatic);
    ASSERT_EQ(isSymmetricNode, isSymmetricStatic);
}

TEST_F(GaudiBatchGemmNode, batch_gemm_5d_asymmetric)
{
    unsigned                    dims = 5;
    const SizeVector            input0Size {256, 128, 2, 4, 5};
    const SizeVector            input1Size {256, 256, 1, 1, 1};
    const TSize                 outputSize[] = {256, 128, 2, 4, 5};

    TensorPtr inputTensor  = TensorPtr(new Tensor(dims, input0Size.data(), syn_type_fixed));
    TensorPtr weightTensor = TensorPtr(new Tensor(dims, input1Size.data(), syn_type_fixed));
    TensorPtr outputTensor = TensorPtr(new Tensor(dims, outputSize, syn_type_int32));

    synGEMMParams                  params;
    NodePtr                        node = NodeFactory::createNode({inputTensor, weightTensor},
                                           {outputTensor},
                                           &params,
                                           NodeFactory::batchGemmNodeTypeName,
                                           "");
    std::shared_ptr<BatchGemmNode> bgn  = std::dynamic_pointer_cast<BatchGemmNode>(node);

    bool isAsymmetricNode = bgn->isFullBroadcastLayout();
    bool isSymmetricNode  = bgn->isSymmetricLayout();

    bool isAsymmetricStatic = BatchGemmNode::isFullBroadcastLayout(input0Size, input1Size);
    bool isSymmetricStatic  = BatchGemmNode::isSymmetricLayout(input0Size, input1Size);

    ASSERT_TRUE(isAsymmetricNode);
    ASSERT_FALSE(isSymmetricNode);

    ASSERT_EQ(isAsymmetricNode, isAsymmetricStatic);
    ASSERT_EQ(isSymmetricNode, isSymmetricStatic);
}