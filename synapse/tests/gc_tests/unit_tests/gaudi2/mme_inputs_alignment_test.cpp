#include "gaudi2_graph.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "pipeline_management_fe_test.h"
#include "synapse_common_types.h"
#include "types.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/passes.h"

class MmeAlignmentTest : public PipelineManagementTest
{
};

TEST_F(MmeAlignmentTest, both_tensors_should_be_aligned)
{
    Gaudi2Graph graph;
    std::vector<TSize> castSizes = {256,10000};
    std::vector<TSize> packingOutSizes = {1600,1600};
    std::vector<TSize> expandDimsOutSizes = {1600, 1600, 1};
    std::vector<TSize> reshapeOpAInSizes = {1600, 128, 64};
    std::vector<TSize> reshapeOpAOutSizes = {1600, 8192};
    std::vector<TSize> gemmOutSizes = {1600, 8192};

    auto castIn  = createTensor(castSizes, syn_type_single, false);
    auto castOut  = createTensor(castSizes, syn_type_fp8_152, false);
    auto packingOut = createTensor(packingOutSizes, syn_type_fp8_152, false);
    auto expandDimsOut = createTensor(expandDimsOutSizes, syn_type_fp8_152, false);
    auto reshapeOpAIn = createTensor(reshapeOpAInSizes, syn_type_fp8_152, false);
    auto reshapeOpAOut = createTensor(reshapeOpAOutSizes, syn_type_fp8_152, false);
    auto gemmOut = createTensor(gemmOutSizes, syn_type_fp8_152, false);

    synGEMMParams gemmParams {};
    unsigned      expandDim = 2;
    NodePtr       cast = NodeFactory::createNode({castIn}, {castOut}, nullptr, "cast_f32_to_f8", "cast");
    NodePtr       reshapeOpB = NodeFactory::createNode({castOut}, {packingOut}, nullptr, "reshape", "reshapeOpB");
    NodePtr       reshapeOpA = NodeFactory::createNode({reshapeOpAIn}, {reshapeOpAOut}, nullptr, "reshape", "reshapeOpA");
    NodePtr       expandDims = NodeFactory::createNode({packingOut}, {expandDimsOut}, &expandDim, "expand_dims", "expandDims");
    NodePtr       gemm = NodeFactory::createNode({reshapeOpAOut, expandDimsOut}, {gemmOut}, &gemmParams, "gemm", "gemm");

    ASSERT_TRUE(GraphEditor::addNode(graph, cast));
    ASSERT_TRUE(GraphEditor::addNode(graph, reshapeOpB));
    ASSERT_TRUE(GraphEditor::addNode(graph, reshapeOpA));
    ASSERT_TRUE(GraphEditor::addNode(graph, expandDims));
    ASSERT_TRUE(GraphEditor::addNode(graph, gemm));

    ASSERT_TRUE(gaudi2::loadTpcKernels(graph));
    sliceGraphForPipeline(graph);

    // Assert bundling assumptions:
    ASSERT_TRUE(cast->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(reshapeOpB->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_FALSE(reshapeOpA->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(expandDims->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(gemm->getNodeAnnotation().bundleInfo.is_set());

    // Assert both inputs are aligned and in sram:
    // Since this bundle places both mme inputs in sram, alignment is possible:
    // For input 0 - it is BPT but is in sram, so alignment is allowed.
    // For input 1 - it has a logical producer and there is enough sram available, so alignment is allowed.
    for (const auto& node : graph.getNodes())
    {
        if (HabanaGraph::runsOnMME(node))
        {
            for (const auto& in : node->getInputs())
            {
                ASSERT_TRUE(in->inSram());
                ASSERT_TRUE(in->getStrideInBytes(1) % graph.getHALReader()->getCacheLineSizeInBytes() == 0);
                ASSERT_TRUE(in->isRealInLogical());
            }
        }
    }
}

TEST_F(MmeAlignmentTest, input_0_should_not_be_aligned)
{
    Gaudi2Graph graph;

    std::vector<TSize> castSizes = {1600,1600};
    std::vector<TSize> expandDimsOutSizes = {1600, 1600, 1};
    std::vector<TSize> reshapeOpAInSizes = {1600, 128, 64};
    std::vector<TSize> reshapeOpAOutSizes = {1600, 8192};
    std::vector<TSize> gemmOutSizes = {1600, 8192};

    auto castIn  = createTensor(castSizes, syn_type_single, false);
    auto castOut  = createTensor(castSizes, syn_type_fp8_152, false);
    auto expandDimsOut = createTensor(expandDimsOutSizes, syn_type_fp8_152, false);
    auto reshapeOpAIn = createTensor(reshapeOpAInSizes, syn_type_fp8_152, false);
    auto reshapeOpAOut = createTensor(reshapeOpAOutSizes, syn_type_fp8_152, false);
    auto gemmOut = createTensor(gemmOutSizes, syn_type_fp8_152, false);

    synGEMMParams gemmParams {};
    unsigned      expandDim = 2;
    NodePtr       cast = NodeFactory::createNode({castIn}, {castOut}, nullptr, "cast_f32_to_f8", "cast");
    NodePtr       reshapeOpA = NodeFactory::createNode({reshapeOpAIn}, {reshapeOpAOut}, nullptr, "reshape", "reshapeOpA");
    NodePtr       expandDims = NodeFactory::createNode({castOut}, {expandDimsOut}, &expandDim, "expand_dims", "expandDims");
    NodePtr       gemm = NodeFactory::createNode({reshapeOpAOut, expandDimsOut}, {gemmOut}, &gemmParams, "gemm", "gemm");

    ASSERT_TRUE(GraphEditor::addNode(graph, cast));
    ASSERT_TRUE(GraphEditor::addNode(graph, reshapeOpA));
    ASSERT_TRUE(GraphEditor::addNode(graph, expandDims));
    ASSERT_TRUE(GraphEditor::addNode(graph, gemm));

    ASSERT_TRUE(gaudi2::loadTpcKernels(graph));
    sliceGraphForPipeline(graph);

    // Assert bundling assumptions:
    ASSERT_TRUE(cast->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_FALSE(reshapeOpA->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(expandDims->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(gemm->getNodeAnnotation().bundleInfo.is_set());

    // Assert alignment:
    // For input 0 - it is BPT and not in sram, so alignment is not allowed.
    // Input 1's alignment therefore depends on the slice size, so it doesn't matter.
    for (const auto& node : graph.getNodes())
    {
        if (HabanaGraph::runsOnMME(node))
        {
            ASSERT_FALSE(node->getInput(0)->inSram());
            ASSERT_TRUE(node->getInput(0)->getStrideInBytes(1) % graph.getHALReader()->getCacheLineSizeInBytes() != 0);
        }
    }
}