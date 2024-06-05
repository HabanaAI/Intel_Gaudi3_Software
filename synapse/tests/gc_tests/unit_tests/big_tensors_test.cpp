#include "graph_optimizer_test.h"
#include "node_annotation.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "utils.h"

#include <habana_global_conf.h>
#include <platform/gaudi/graph_compiler/gaudi_graph.h>
#include <node_factory.h>

class BigTensorsTest : public GraphOptimizerTest
{
    using BaseClass = GraphOptimizerTest;

protected:

    void SetUp() override
    {
        BaseClass::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT, "false");
        setGlobalConfForTest(GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING, "false");
        setGlobalConfForTest(GCFG_IGNORE_INDEX_SPACE_FOR_SLICING, "false");
    }

    TensorPtr createTensor(SizeVector shape, synDataType dataType, const char* name = nullptr)
    {
        auto tensor = std::make_shared<Tensor>(shape.size(), shape.data(), dataType);
        if (name)
        {
            tensor->setName(name);
        }
        return tensor;
    }
    TensorPtr createPersistentTensor(SizeVector shape, synDataType dataType, const char* name = nullptr)
    {
        auto tensor = createTensor(shape, dataType, name);

        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setMemorySectionID(m_sectionId++);
        tensor->setDramOffset(0x10000000 * m_sectionId);
        tensor->map();
        return tensor;
    }

    uint64_t m_sectionId = 0;
};

TEST_F(BigTensorsTest, slicer_should_not_reshape_conv_when_stitching_tpc_producer)
{
    // Given graph relu->conv with 4D tensors
    GaudiGraph graph;

    SizeVector fmSize  = {256, 32, 32, 64};
    SizeVector wghSize = {256, 256, 1, 1};

    TensorPtr in     = createPersistentTensor(fmSize, syn_type_float);
    TensorPtr convIn = createTensor(fmSize, syn_type_float);
    auto      relu   = NodeFactory::createNode({in}, {convIn}, nullptr, "relu_fwd_f32", "reluIn");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));

    TensorPtr            wgh     = createPersistentTensor(wghSize, syn_type_float);
    TensorPtr            convOut = createPersistentTensor(fmSize, syn_type_float);
    synConvolutionParams convParams {};

    auto conv =
        NodeFactory::createNode({convIn, wgh}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(graph, conv));

    // When compiling the graph
    ASSERT_TRUE(graph.compile());

    // Then make sure the TPC is stitched and MME is not reshaped.
    Settable<uint32_t> mmeBundleID;
    for (const auto& node : graph.getNodes())
    {
        if (graph.runsOnMME(node))
        {
            const auto& ifm = node->getInput(TENSOR_IFM);
            ASSERT_NE(ifm->getSizeInElements(DIM_H), 1) << "Unexpected: MME was flattened";
            const auto& nodeBundleInfo = node->getNodeAnnotation().bundleInfo;
            ASSERT_TRUE(nodeBundleInfo.is_set());
            if (mmeBundleID.is_set())
            {
                ASSERT_EQ(nodeBundleInfo->bundleIndex, mmeBundleID.value());
            }
            mmeBundleID = nodeBundleInfo->bundleIndex;
        }
    }
    ASSERT_TRUE(mmeBundleID.is_set()) << "No MME node was encountered";
    for (const auto& node : graph.getNodes())
    {
        if (graph.runsOnTPC(node))
        {
            const auto& nodeBundleInfo = node->getNodeAnnotation().bundleInfo;
            ASSERT_TRUE(nodeBundleInfo.is_set()) << "Unexpected: TPC is not stitched";
            ASSERT_EQ(nodeBundleInfo->bundleIndex, mmeBundleID.value()) << "TPC was not stitched to the MME bundle";
        }
    }
}

TEST_F(BigTensorsTest, slicer_should_not_reshape_conv_when_stitching_tpc_consumer)
{
    // Given graph conv->relu with 4D tensors
    GaudiGraph graph;

    SizeVector fmSize  = {256, 32, 32, 64};
    SizeVector wghSize = {256, 256, 1, 1};

    TensorPtr            convIn  = createPersistentTensor(fmSize, syn_type_float);
    TensorPtr            wgh     = createPersistentTensor(wghSize, syn_type_float);
    TensorPtr            convOut = createTensor(fmSize, syn_type_float);
    synConvolutionParams convParams {};

    auto conv =
        NodeFactory::createNode({convIn, wgh}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(graph, conv));

    TensorPtr out  = createPersistentTensor(fmSize, syn_type_float);
    auto      relu = NodeFactory::createNode({convOut}, {out}, nullptr, "relu_fwd_f32", "reluIn");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));

    // When compiling the graph
    ASSERT_TRUE(graph.compile());

    // Then make sure the TPC is stitched and MME is not reshaped.
    Settable<uint32_t> mmeBundleID;
    for (const auto& node : graph.getNodes())
    {
        if (graph.runsOnMME(node))
        {
            const auto& ifm = node->getInput(TENSOR_IFM);
            ASSERT_NE(ifm->getSizeInElements(DIM_H), 1) << "Unexpected: MME was flattened";
            const auto& nodeBundleInfo = node->getNodeAnnotation().bundleInfo;
            ASSERT_TRUE(nodeBundleInfo.is_set());
            if (mmeBundleID.is_set())
            {
                ASSERT_EQ(nodeBundleInfo->bundleIndex, mmeBundleID.value());
            }
            mmeBundleID = nodeBundleInfo->bundleIndex;
        }
    }
    ASSERT_TRUE(mmeBundleID.is_set()) << "No MME node was encountered";
    for (const auto& node : graph.getNodes())
    {
        if (graph.runsOnTPC(node))
        {
            const auto& nodeBundleInfo = node->getNodeAnnotation().bundleInfo;
            ASSERT_TRUE(nodeBundleInfo.is_set()) << "Unexpected: TPC is not stitched";
            ASSERT_EQ(nodeBundleInfo->bundleIndex, mmeBundleID.value()) << "TPC was not stitched to the MME bundle";
        }
    }
}

TEST_F(BigTensorsTest, slicer_should_stitch_tpc_producer_to_reshaped_mme)
{
    // Given graph relu->reshape(4d->2d)->conv
    GaudiGraph graph;

    SizeVector fmSize         = {256, 32, 32, 64};
    SizeVector reshapedFMSize = {256, multiplyElements(std::next(fmSize.begin()), fmSize.end()), 1, 1};
    SizeVector wghSize        = {256, 256, 1, 1};

    TensorPtr in        = createPersistentTensor(fmSize, syn_type_float, "in");
    TensorPtr reshapeIn = createTensor(fmSize, syn_type_float, "reshapeIn");
    auto      inRelu    = NodeFactory::createNode({in}, {reshapeIn}, nullptr, "relu_fwd_f32", "reluIn");
    ASSERT_TRUE(GraphEditor::addNode(graph, inRelu));

    TensorPtr convIn       = createTensor(reshapedFMSize, syn_type_float, "convIn");
    TensorPtr reshapeShape = std::make_shared<Tensor>(reshapedFMSize.size(),
                                                      reshapedFMSize.data(),
                                                      syn_type_float,
                                                      nullptr,
                                                      nullptr,
                                                      false,
                                                      true,
                                                      INVALID_BATCH_POS,
                                                      reshapedFMSize.data(),
                                                      OUTPUT_DESCRIBING_SHAPE_TENSOR);
    reshapeShape->setName("shape_tensor");
    auto reshape = NodeFactory::createNode({reshapeIn, reshapeShape},
                                           {convIn},
                                           nullptr,
                                           NodeFactory::reshapeNodeTypeName,
                                           "reshape");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape));

    TensorPtr            wgh     = createPersistentTensor(wghSize, syn_type_float, "wgh");
    TensorPtr            convOut = createPersistentTensor(reshapedFMSize, syn_type_float, "convOut");
    synConvolutionParams convParams {};

    auto conv =
        NodeFactory::createNode({convIn, wgh}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(graph, conv));

    // When compiling the graph
    ASSERT_TRUE(graph.compile());

    // Then make sure the TPC is stitched to MME, the reshape should not be removed out from the bundle.
    Settable<BundleInfo> bundleInfo;
    unsigned             numOfMmeNodes     = 0;
    unsigned             numOfTpcNodes     = 0;
    unsigned             numOfReshapeNodes = 0;
    for (const auto& node : graph.getNodes())
    {
        const auto& nodeBundleInfo = node->getNodeAnnotation().bundleInfo;
        ASSERT_TRUE(nodeBundleInfo.is_set());
        if (bundleInfo.is_set())
        {
            ASSERT_EQ(bundleInfo->bundleIndex, nodeBundleInfo->bundleIndex);
        }
        else
        {
            bundleInfo = nodeBundleInfo;
        }
        if (graph.runsOnMME(node))
        {
            numOfMmeNodes++;
        }
        else if (graph.runsOnTPC(node))
        {
            numOfTpcNodes++;
        }
        else if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            numOfReshapeNodes++;
            const auto& reshapeProducers = graph.getNodeProducers(node);
            ASSERT_TRUE(std::any_of(reshapeProducers.begin(), reshapeProducers.end(), [&](const NodePtr& n) {
                return graph.runsOnTPC(n);
            }));
            const auto& reshapeConsumers = graph.getNodeConsumers(node);
            ASSERT_EQ(reshapeConsumers.size(), 1);
            ASSERT_TRUE(graph.runsOnMME(*reshapeConsumers.begin()));
        }
    }
    ASSERT_TRUE(bundleInfo.is_set());
    ASSERT_TRUE(numOfMmeNodes > 0);
    ASSERT_EQ(numOfMmeNodes, numOfReshapeNodes);
    ASSERT_EQ(numOfTpcNodes, numOfReshapeNodes);
}

TEST_F(BigTensorsTest, slicer_should_stitch_tpc_consumer_to_reshaped_mme)
{
    // Given graph conv->reshape(2d->4d)->relu
    GaudiGraph graph;

    SizeVector fmSize         = {256, 32, 32, 64};
    SizeVector reshapedFMSize = {256, multiplyElements(std::next(fmSize.begin()), fmSize.end()), 1, 1};
    SizeVector wghSize        = {256, 256, 1, 1};

    TensorPtr            convIn  = createPersistentTensor(reshapedFMSize, syn_type_float, "convIn");
    TensorPtr            wgh     = createPersistentTensor(wghSize, syn_type_float, "wgh");
    TensorPtr            convOut = createTensor(reshapedFMSize, syn_type_float, "convOut");
    synConvolutionParams convParams {};

    auto conv =
        NodeFactory::createNode({convIn, wgh}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(graph, conv));

    TensorPtr reshapeShape = std::make_shared<Tensor>(fmSize.size(),
                                                      fmSize.data(),
                                                      syn_type_float,
                                                      nullptr,
                                                      nullptr,
                                                      false,
                                                      true,
                                                      INVALID_BATCH_POS,
                                                      fmSize.data(),
                                                      OUTPUT_DESCRIBING_SHAPE_TENSOR);
    reshapeShape->setName("shape_tensor");
    TensorPtr reluIn  = createTensor(fmSize, syn_type_float, "reluIn");
    auto      reshape = NodeFactory::createNode({convOut, reshapeShape},
                                           {reluIn},
                                           nullptr,
                                           NodeFactory::reshapeNodeTypeName,
                                           "reshape");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape));

    TensorPtr reluOut = createPersistentTensor(fmSize, syn_type_float, "reshapeIn");
    auto      relu    = NodeFactory::createNode({reluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));

    // When compiling the graph
    ASSERT_TRUE(graph.compile());

    // Then make sure the TPC is stitched to MME, the reshape should not be removed out from the bundle.
    Settable<BundleInfo> bundleInfo;
    unsigned             numOfMmeNodes     = 0;
    unsigned             numOfTpcNodes     = 0;
    unsigned             numOfReshapeNodes = 0;
    for (const auto& node : graph.getNodes())
    {
        const auto& nodeBundleInfo = node->getNodeAnnotation().bundleInfo;
        ASSERT_TRUE(nodeBundleInfo.is_set());
        if (bundleInfo.is_set())
        {
            ASSERT_EQ(bundleInfo->bundleIndex, nodeBundleInfo->bundleIndex);
        }
        else
        {
            bundleInfo = nodeBundleInfo;
        }
        if (graph.runsOnMME(node))
        {
            numOfMmeNodes++;
        }
        else if (graph.runsOnTPC(node))
        {
            numOfTpcNodes++;
        }
        else if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            numOfReshapeNodes++;
            const auto& reshapeProducers = graph.getNodeProducers(node);
            ASSERT_TRUE(std::any_of(reshapeProducers.begin(), reshapeProducers.end(), [&](const NodePtr& n) {
                return graph.runsOnMME(n);
            }));
            const auto& reshapeConsumers = graph.getNodeConsumers(node);
            ASSERT_EQ(reshapeConsumers.size(), 1);
            ASSERT_TRUE(graph.runsOnTPC(*reshapeConsumers.begin()));
        }
    }
    ASSERT_TRUE(bundleInfo.is_set());
    ASSERT_TRUE(numOfMmeNodes > 0);
    ASSERT_EQ(numOfMmeNodes, numOfReshapeNodes);
    ASSERT_EQ(numOfTpcNodes, numOfReshapeNodes);
}

TEST_F(BigTensorsTest, slicer_should_stitch_tpc_producer_to_tiny_reshaped_mme)
{
    // Given graph relu->reshape(4d->2d)->conv with a tiny feature map size
    GaudiGraph graph;

    SizeVector fmSize         = {256, 16, 16, 1};
    SizeVector reshapedFMSize = {256, multiplyElements(std::next(fmSize.begin()), fmSize.end()), 1, 1};
    SizeVector wghSize        = {256, 256, 1, 1};

    TensorPtr in        = createPersistentTensor(fmSize, syn_type_float, "in");
    TensorPtr reshapeIn = createTensor(fmSize, syn_type_float, "reshapeIn");
    auto      inRelu    = NodeFactory::createNode({in}, {reshapeIn}, nullptr, "relu_fwd_f32", "reluIn");
    ASSERT_TRUE(GraphEditor::addNode(graph, inRelu));

    TensorPtr convIn       = createTensor(reshapedFMSize, syn_type_float, "convIn");
    TensorPtr reshapeShape = std::make_shared<Tensor>(reshapedFMSize.size(),
                                                      reshapedFMSize.data(),
                                                      syn_type_float,
                                                      nullptr,
                                                      nullptr,
                                                      false,
                                                      true,
                                                      INVALID_BATCH_POS,
                                                      reshapedFMSize.data(),
                                                      OUTPUT_DESCRIBING_SHAPE_TENSOR);
    reshapeShape->setName("shape_tensor");
    auto reshape = NodeFactory::createNode({reshapeIn, reshapeShape},
                                           {convIn},
                                           nullptr,
                                           NodeFactory::reshapeNodeTypeName,
                                           "reshape");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape));

    TensorPtr            wgh     = createPersistentTensor(wghSize, syn_type_float, "wgh");
    TensorPtr            convOut = createPersistentTensor(reshapedFMSize, syn_type_float, "convOut");
    synConvolutionParams convParams {};

    auto conv =
        NodeFactory::createNode({convIn, wgh}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(graph, conv));

    // When compiling the graph
    ASSERT_TRUE(graph.compile());

    // Then make sure the TPC does get stitched to MME and there's no slicing
    NodePtr mmeNode = nullptr;
    NodePtr tpcNode = nullptr;
    NodePtr rspNode = nullptr;
    for (const auto& node : graph.getNodes())
    {
        if (graph.runsOnMME(node))
        {
            ASSERT_EQ(mmeNode, nullptr) << "Expecting a single MME node in post graph";
            mmeNode = node;
        }
        if (graph.runsOnTPC(node))
        {
            ASSERT_EQ(nullptr, tpcNode) << "Expecting a single TPC node in post graph";
            tpcNode = node;
        }
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            ASSERT_EQ(nullptr, rspNode) << "Expecting a single reshape node in post graph";
            rspNode = node;
        }
    }
    ASSERT_NE(nullptr, mmeNode) << "MME node wasn't found";
    ASSERT_NE(nullptr, tpcNode) << "TPC node wasn't found";
    ASSERT_NE(nullptr, rspNode) << "Reshape node wasn't found";

    ASSERT_TRUE(mmeNode->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(tpcNode->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(rspNode->getNodeAnnotation().bundleInfo.is_set());

    ASSERT_EQ(mmeNode->getNodeAnnotation().bundleInfo->bundleIndex,
              tpcNode->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_EQ(mmeNode->getNodeAnnotation().bundleInfo->bundleIndex,
              rspNode->getNodeAnnotation().bundleInfo->bundleIndex);
}

TEST_F(BigTensorsTest, slicer_should_stitch_tpc_consumer_to_tiny_reshaped_mme)
{
    // Given graph conv->reshape(4d->2d)->relu with a tiny feature map size
    GaudiGraph graph;

    SizeVector fmSize         = {256, 16, 16, 1};
    SizeVector reshapedFMSize = {256, multiplyElements(std::next(fmSize.begin()), fmSize.end()), 1, 1};
    SizeVector wghSize        = {256, 256, 1, 1};

    TensorPtr            convIn  = createPersistentTensor(reshapedFMSize, syn_type_float, "convIn");
    TensorPtr            wgh     = createPersistentTensor(wghSize, syn_type_float, "wgh");
    TensorPtr            convOut = createTensor(reshapedFMSize, syn_type_float, "convOut");
    synConvolutionParams convParams {};

    auto conv =
        NodeFactory::createNode({convIn, wgh}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(graph, conv));

    TensorPtr reshapeShape = std::make_shared<Tensor>(fmSize.size(),
                                                      fmSize.data(),
                                                      syn_type_float,
                                                      nullptr,
                                                      nullptr,
                                                      false,
                                                      true,
                                                      INVALID_BATCH_POS,
                                                      fmSize.data(),
                                                      OUTPUT_DESCRIBING_SHAPE_TENSOR);
    reshapeShape->setName("shape_tensor");
    TensorPtr reluIn  = createTensor(fmSize, syn_type_float, "reluIn");
    auto      reshape = NodeFactory::createNode({convOut, reshapeShape},
                                           {reluIn},
                                           nullptr,
                                           NodeFactory::reshapeNodeTypeName,
                                           "reshape");
    ASSERT_TRUE(GraphEditor::addNode(graph, reshape));

    TensorPtr reluOut = createPersistentTensor(fmSize, syn_type_float, "reshapeIn");
    auto      relu    = NodeFactory::createNode({reluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu");
    ASSERT_TRUE(GraphEditor::addNode(graph, relu));

    // When compiling the graph
    ASSERT_TRUE(graph.compile());

    // Then make sure the TPC does get stitched to MME and there's no slicing
    NodePtr mmeNode = nullptr;
    NodePtr tpcNode = nullptr;
    NodePtr rspNode = nullptr;
    for (const auto& node : graph.getNodes())
    {
        if (graph.runsOnMME(node))
        {
            ASSERT_EQ(mmeNode, nullptr) << "Expecting a single MME node in post graph";
            mmeNode = node;
        }
        if (graph.runsOnTPC(node))
        {
            ASSERT_EQ(nullptr, tpcNode) << "Expecting a single TPC node in post graph";
            tpcNode = node;
        }
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            ASSERT_EQ(nullptr, rspNode) << "Expecting a single reshape node in post graph";
            rspNode = node;
        }
    }
    ASSERT_NE(nullptr, mmeNode) << "MME node wasn't found";
    ASSERT_NE(nullptr, tpcNode) << "TPC node wasn't found";
    ASSERT_NE(nullptr, rspNode) << "Reshape node wasn't found";

    ASSERT_TRUE(mmeNode->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(tpcNode->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(rspNode->getNodeAnnotation().bundleInfo.is_set());

    ASSERT_EQ(mmeNode->getNodeAnnotation().bundleInfo->bundleIndex,
              tpcNode->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_EQ(mmeNode->getNodeAnnotation().bundleInfo->bundleIndex,
              rspNode->getNodeAnnotation().bundleInfo->bundleIndex);
}