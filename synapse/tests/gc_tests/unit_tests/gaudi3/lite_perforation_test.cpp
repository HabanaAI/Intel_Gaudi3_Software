//
// This file contains unit tests for the lite perforation
// temporary mechanism, to allow locality in Gaudi3 using
// legacy GC brains.
//

#include "compilation_hal_reader.h"
#include "graph_optimizer_test.h"
#include "gaudi3_graph.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"
#include "platform/gaudi3/graph_compiler/passes.h"

#include "brain_conf.h"

class LitePerforationTest : public GraphOptimizerTest
{
public:
    Gaudi3Graph                                graph;
    CompilationHalReaderSetter                 halReaderSetter {&graph};

    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_LAYERED_PIPELINE_BRAIN, "false");
        setGlobalConfForTest(GCFG_ENABLE_PIPELINE_MANAGEMENT, "true");
        setGlobalConfForTest(GCFG_ENABLE_BRAIN_LOCALITY_HINTS_ANNOTATION, "true");
    }
};

TEST_F(LitePerforationTest, post_slicing_big_nodes_and_tensor_should_have_lp_annotations)
{
    const char* reluGuid = "relu_fwd_bf16";
    const char* castGuid = "cast_f32_to_bf16";

    SizeVector fmSize  = {512, 32768};
    SizeVector wghSize = {512, 512};

    // Given graph:      [in]->Relu->[t]->GEMM->[out]
    //                                      ^
    //                                      |
    //                    [wgh]->cast->[wgh_bf16]
    auto        in       = std::make_shared<Tensor>(fmSize.size(), fmSize.data(), syn_type_bf16);
    auto        t        = std::make_shared<Tensor>(fmSize.size(), fmSize.data(), syn_type_bf16);
    auto        out      = std::make_shared<Tensor>(fmSize.size(), fmSize.data(), syn_type_float);
    auto        wgh      = std::make_shared<Tensor>(wghSize.size(), wghSize.data(), syn_type_float);
    auto        wgh_bf16 = std::make_shared<Tensor>(wghSize.size(), wghSize.data(), syn_type_bf16);

    in->setName("in", true);
    t->setName("t", true);
    out->setName("out", true);
    wgh->setName("wgh", true);
    wgh_bf16->setName("wgh_bf16", true);

    synMemoryDescriptor memDesc(true);
    in->setMemoryDescriptor(memDesc);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    wgh->setMemoryDescriptor(memDesc);
    wgh->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    NodePtr relu = NodeFactory::createNode({in}, {t}, nullptr, reluGuid, "ReLU");
    NodePtr cast = NodeFactory::createNode({wgh}, {wgh_bf16}, nullptr, castGuid, "cast");

    synGEMMParams gemmParams {};
    NodePtr gemm = NodeFactory::createNode({t, wgh_bf16}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "GEMM");

    ASSERT_TRUE(GraphEditor::addNode(graph, relu));
    ASSERT_TRUE(GraphEditor::addNode(graph, cast));
    ASSERT_TRUE(GraphEditor::addNode(graph, gemm));

    // When slicing
    gaudi3::loadTpcKernels(graph);
    sliceGraphForPipeline(graph);

    // Expect each slice to either have perforation information or have big origin with perforation information
    int numReluSlices = 0;
    int numGemmSlices = 0;
    const gc::access_pattern::TensorTile::Size reluGranularity =
        relu->getNodeAccessPattern()->getTensorGranularity(t).geometry[1];
    for (const NodePtr& n : graph.getNodes())
    {
        const auto& perforationInfo = n->getNodeAnnotation().perforation.has_value()
                                          ? n->getNodeAnnotation().perforation
                                          : n->getNodeAnnotation().origBigNode->getNodeAnnotation().perforation;
        if (n->getGUID() == reluGuid)
        {
            numReluSlices++;
            ASSERT_TRUE(perforationInfo.has_value());
            EXPECT_EQ(1, perforationInfo->indexSpaceDim);
            EXPECT_EQ(1, perforationInfo->granularity);
            const auto& tensorLPAnnotations = n->getInput(0)->getTensorAnnotation().perforation;
            ASSERT_FALSE(tensorLPAnnotations.has_value())
                << "Only intermediate tensors are expected to have LP annotations";
        }
        else if (n->getGUID() == NodeFactory::gemmNodeTypeName)
        {
            numGemmSlices++;
            ASSERT_TRUE(perforationInfo.has_value());
            ASSERT_EQ(3, perforationInfo->indexSpaceDim);
            // The GEMM would have EW granularity on the input height, so the common granularity would be derived from
            // the relu only and will project on the GEMM node granularity.
            ASSERT_EQ(reluGranularity, perforationInfo->granularity);
            // Tensor clones inherit annotations from their source, so the indication appear on each slice.
            const auto& input0LPAnnotations = n->getInput(0)->getTensorAnnotation().perforation;
            ASSERT_TRUE(input0LPAnnotations.has_value()) << "All intermediate tensors should have annotations";
            EXPECT_TRUE(input0LPAnnotations->cached);
            EXPECT_TRUE(input0LPAnnotations->sliced);
            const auto& input1LPAnnotations = n->getInput(1)->getTensorAnnotation().perforation;
            ASSERT_TRUE(input1LPAnnotations.has_value()) << "All intermediate tensors should have annotations";
            EXPECT_TRUE(input1LPAnnotations->cached);
            EXPECT_FALSE(input1LPAnnotations->sliced);
        }
    }
    EXPECT_GT(numReluSlices, 1);
    EXPECT_GT(numGemmSlices, 1);
}