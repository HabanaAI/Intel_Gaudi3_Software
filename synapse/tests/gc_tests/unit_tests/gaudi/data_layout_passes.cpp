#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include <gtest/gtest.h>
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "test_utils.h"
#include "types.h"
#include "synapse_test.hpp"
#include "syn_singleton.hpp"

using namespace std;

class GaudiLayoutPasses : public GraphOptimizerTest
{
protected:
    bool isDmaTranspose(NodePtr n)
    {
        if (n != nullptr && n->getNodeTypeStr() == "DmaTranspose")
        {
            return true;
        }
        return false;
    }

    void setTensorPersistent(const TensorPtr& t)
    {
        synMemoryDescriptor persistentMemoryDesc(true);
        t->setMemoryDescriptor(persistentMemoryDesc);
        t->setMemorySectionID(m_persistentSectionId++);
    }

    uint64_t m_persistentSectionId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
};

TEST_F(GaudiLayoutPasses, gaudi_wrap_conv_weight_with_transposes)
{
    GaudiGraph g;

    const TSize kW    = 3;
    const TSize kH    = 3;
    const TSize dW    = 1;
    const TSize dH    = 1;
    const TSize nOFM  = 8;
    const TSize wOFM  = 5;
    const TSize hOFM  = 5;
    const TSize nIFM  = 8;
    const TSize padH  = 0;
    const TSize padW  = 0;
    const TSize batch = 1;

    synConvolutionParams params;
    params.dH   = dH;
    params.dW   = dW;
    params.kH   = kH;
    params.kW   = kW;
    params.padT = padH;
    params.padB = padH;
    params.padL = padW;
    params.padR = padW;
    params.dilH = 1;
    params.dilW = 1;

    // o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
    const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

    float weights[nIFM * nOFM * params.kW * params.kH];
    float ifm[nIFM * wIFM * hIFM * batch];

    // Pass input as NCHW(same as used in PT)
    const TSize i_sizes[] = {wIFM, hIFM, nIFM, batch};
    const TSize o_sizes[] = {wOFM, hOFM, nOFM, batch};
    // Pass weight as KCRS(same as used in PT)
    const TSize w_sizes[] = {params.kH, params.kW, nIFM, nOFM};

    TensorPtr T1 = std::make_shared<Tensor>(4U, i_sizes, syn_type_single, reinterpret_cast<char*>(ifm));
    T1->setName("T1");
    setTensorPersistent(T1);

    TensorPtr W1 = std::make_shared<Tensor>(4U, w_sizes, syn_type_single, reinterpret_cast<char*>(weights));
    W1->setName("W1");
    setTensorPersistent(W1);

    TensorPtr O1 = std::make_shared<Tensor>(4U, o_sizes, syn_type_single);
    O1->setName("O1");
    setTensorPersistent(O1);

    NodePtr              conv1 = NodeFactory::createNode({T1, W1, nullptr, nullptr},
                                            {O1},
                                            &params,
                                            NodeFactory::convolutionNodeTypeName,
                                            "conv1");
    Node::NodeProperties p;

    // Setting conv inputs to be as NCHW/KCRS same as PT will pass
    p.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("SRCK"), gc::Layout(), gc::Layout("WHCN")};
    p.outputLayouts = {gc::Layout("WHCN")};
    conv1->setInputLayouts(p.inputLayouts);
    conv1->setOutputLayouts(p.outputLayouts);
    GraphEditor::addNode(g, conv1);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    // Expecting Synapse to insert 3 transposes
    // 1. input to NHWC
    // 2. weight to RSCK
    // 3. output back to NCHW
    unsigned int transposeNodeCount = 0;
    for (const auto& node : g.getExeSortedNodes())
    {
        if (node != nullptr && isDmaTranspose(node))
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 3) << "{} unexpected transpose nodes found in graph" << transposeNodeCount;
}

TEST_F(GaudiLayoutPasses, gaudi_wrap_conv_relu_with_transposes)
{
    GaudiGraph g;

    const TSize kW    = 3;
    const TSize kH    = 3;
    const TSize dW    = 1;
    const TSize dH    = 1;
    const TSize nOFM  = 8;
    const TSize wOFM  = 5;
    const TSize hOFM  = 5;
    const TSize nIFM  = 8;
    const TSize padH  = 0;
    const TSize padW  = 0;
    const TSize batch = 1;

    synConvolutionParams params;
    params.dH   = dH;
    params.dW   = dW;
    params.kH   = kH;
    params.kW   = kW;
    params.padT = padH;
    params.padB = padH;
    params.padL = padW;
    params.padR = padW;
    params.dilH = 1;
    params.dilW = 1;

    // o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
    const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

    float weights[nIFM * nOFM * params.kW * params.kH];
    float ifm[nIFM * wIFM * hIFM * batch];

    // Pass input as NCHW
    const TSize i_sizes[] = {wIFM, hIFM, nIFM, batch};
    const TSize o_sizes[] = {wOFM, hOFM, nOFM, batch};
    const TSize w_sizes[] = {nOFM, nIFM, params.kH, params.kW};

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr T1 = std::make_shared<Tensor>(4U, i_sizes, syn_type_single, reinterpret_cast<char*>(ifm));
    T1->setName("T1");
    setTensorPersistent(T1);

    TensorPtr W1 = std::make_shared<Tensor>(4U, w_sizes, syn_type_single, reinterpret_cast<char*>(weights));
    W1->setName("W1");
    setTensorPersistent(W1);

    TensorPtr O1 = std::make_shared<Tensor>(4U, o_sizes, syn_type_single);
    O1->setName("O1");

    TensorPtr RELU_O1 = std::make_shared<Tensor>(4U, o_sizes, syn_type_single);
    RELU_O1->setName("RELU_1");
    setTensorPersistent(RELU_O1);

    NodePtr              conv1 = NodeFactory::createNode({T1, W1, nullptr, nullptr},
                                            {O1},
                                            &params,
                                            NodeFactory::convolutionNodeTypeName,
                                            "conv1");
    Node::NodeProperties p;
    // NCHW
    p.inputLayouts  = {gc::Layout("WHCN"), gc::Layout("KCSR"), gc::Layout(), gc::Layout("WHCN")};
    p.outputLayouts = {gc::Layout("WHCN")};
    conv1->setInputLayouts(p.inputLayouts);
    conv1->setOutputLayouts(p.outputLayouts);
    GraphEditor::addNode(g, conv1);

    NodePtr              relu1 = NodeFactory::createNode({O1}, {RELU_O1}, nullptr, "relu_fwd_f32", "relu1");
    Node::NodeProperties p_relu;
    GraphEditor::addNode(g, relu1);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    // Expecting Synapse to insert 4 transposes
    // 1. input to NHWC
    // 2. output of conv back to NCHW
    // 3+4. 2 transposes around Relu, which is don't care
    // Transposes 2+3 are identity, hence will be removed by pass
    // We'll end up with only 2 left, 1 + 4.
    unsigned int transposeNodeCount = 0;
    for (const auto& node : g.getExeSortedNodes())
    {
        if (node != nullptr && isDmaTranspose(node))
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 2) << "{} unexpected transpose nodes found in graph" << transposeNodeCount;
}

TEST_F(GaudiLayoutPasses, gaudi_maxpool_with_transposes)
{
    GaudiGraph g;

    const TSize kW    = 3;
    const TSize kH    = 3;
    const TSize inZ   = 2;
    const TSize inW   = 4;
    const TSize inH   = 4;
    const TSize batch = 1;

    SpatialReduction2DDef kernel_params;
    kernel_params.pad_w_begin = 0;
    kernel_params.pad_h_end   = 0;
    kernel_params.pad_w_end   = 0;
    kernel_params.pad_h_begin = 0;
    kernel_params.kernel_w    = kW;
    kernel_params.kernel_h    = kH;
    kernel_params.stride_w    = 1;
    kernel_params.stride_h    = 1;
    kernel_params.dilation_w  = 1;
    kernel_params.dilation_h  = 1;

    // Tensor size [NCHW]
    TSize inTensorSize[4]     = {inW, inH, inZ, batch};
    TSize outTensorSize[2][4] = {{2, 2, inZ, batch}, {2, 2, inZ, batch}};

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr T1 =
        std::make_shared<Tensor>(4U, inTensorSize, syn_type_single);  // , reinterpret_cast<char*>(inputBufferSize));
    T1->setName("T1");
    setTensorPersistent(T1);

    TensorPtr O1 = std::make_shared<Tensor>(4U, outTensorSize[0], syn_type_uint8);
    O1->setName("O1");
    setTensorPersistent(O1);

    TensorPtr O2 = std::make_shared<Tensor>(4U, outTensorSize[1], syn_type_single);
    O2->setName("O2");
    setTensorPersistent(O2);

    NodePtr maxpool = NodeFactory::createNode({T1}, {O1, O2}, &kernel_params, "maxpool_2d_fwd_f32", "maxpool1");
    Node::NodeProperties p;
    // NCHW
    p.inputLayouts  = {gc::Layout("WHCN")};
    p.outputLayouts = {gc::Layout("WHCN"), gc::Layout("WHCN")};
    maxpool->setInputLayouts(p.inputLayouts);
    maxpool->setOutputLayouts(p.outputLayouts);
    GraphEditor::addNode(g, maxpool);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    // Expecting Synapse to insert 3 transposes
    // 1. input to NHWC
    // 2. output of maxpool to NCHW X2
    unsigned int transposeNodeCount = 0;
    for (const auto& node : g.getExeSortedNodes())
    {
        if (node != nullptr && isDmaTranspose(node))
        {
            ++transposeNodeCount;
        }
    }
    ASSERT_EQ(transposeNodeCount, 3) << "{} unexpected transpose nodes found in graph" << transposeNodeCount;
}