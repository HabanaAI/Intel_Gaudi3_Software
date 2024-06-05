#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "node_factory.h"
#include "gaudi3_pipeline_management_fe_test.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "hal_reader/gaudi3/hal_reader.h"

#include "brain_conf.h"

void Gaudi3PipelineManagementTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    setGlobalConfForTest(GCFG_ENABLE_PIPELINE_MANAGEMENT, "true");
    setGlobalConfForTest(GCFG_ENABLE_LAYERED_PIPELINE_BRAIN, "false");
    CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
}

void Gaudi3PipelineManagementTest::TearDown()
{
    GraphOptimizerTest::TearDown();
}

TensorPtr
Gaudi3PipelineManagementTest::createTensor(std::vector<TSize> shape, synDataType dataType, bool isPersistent /*= true*/)
{
    synMemoryDescriptor memDesc(isPersistent);
    auto                tensor = std::make_shared<Tensor>(shape.size(), shape.data(), dataType);
    tensor->setMemoryDescriptor(memDesc);
    if (isPersistent)
    {
        tensor->setMemorySectionID(m_memorySectionId++);
    }
    tensor->map();
    return tensor;
}

TEST_F(Gaudi3PipelineManagementTest, transpose_logic_should_be_bundled)
{
    Gaudi3Graph g;
    setGlobalConfForTest(GCFG_ENABLE_BUNDLE_TRANSPOSE, "true");

    synGEMMParams gemmParams {};
    TensorPtr     tensor0 = createTensor({16, 4, 8, 128}, syn_type_single, true);
    TensorPtr     tensor1 = createTensor({16, 4, 8, 128}, syn_type_single, false);
    TensorPtr     tensor2 = createTensor({16, 8, 4, 128}, syn_type_single, false);
    TensorPtr     tensor3 = createTensor({16, 8, 4, 128}, syn_type_single, true);
    TensorPtr     tensor4 = createTensor({16, 8, 4, 128}, syn_type_single, true);

    TransposePermutationArray permutation({TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch});

    pNode relu = NodeFactory::createNode({tensor0}, {tensor1}, nullptr, "relu_fwd_f32", "relu");
    pNode transpose =
        NodeFactory::createNode({tensor1}, {tensor2}, &permutation, "transpose_logic", "logical_transpose");
    NodePtr batchGemm = NodeFactory::createNode({tensor2, tensor3},
                                                {tensor4},
                                                &gemmParams,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "batch_gemm");

    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(GraphEditor::addNode(g, transpose));
    ASSERT_TRUE(GraphEditor::addNode(g, batchGemm));

    ASSERT_TRUE(gaudi3::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    const auto& graphNodes = g.getExeSortedNodes();
    const auto  it         = std::find_if(graphNodes.begin(), graphNodes.end(), [](const auto& node) {
        return node && node->isTranspose();
    });
    ASSERT_TRUE(it != graphNodes.end()) << "Expecting a transpose node";
    const auto& transposeNode = *it;
    ASSERT_TRUE(transposeNode->isLogicalOperation()) << "Expecting logical transpose node";
    ASSERT_TRUE(transposeNode->getNodeAnnotation().bundleInfo.is_set()) << "Expecting transpose bundled";
}