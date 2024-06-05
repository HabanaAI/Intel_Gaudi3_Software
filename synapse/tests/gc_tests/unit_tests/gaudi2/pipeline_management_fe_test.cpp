#include "gtest/gtest.h"
#include <algorithm>
#include <optional>
#include <string>
#include "bundle_plane_graph.h"
#include "passes/sram_management/slicing_brain.h"
#include "passes/sram_management/bundle_slicer.h"
#include "perf_lib_layer_params.h"
#include "pipeline_bundlizer.h"
#include "pipeline_management_fe_test.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "tensor.h"
#include "node_factory.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "transpose_utils.h"

void PipelineManagementTest::SetUp()
{
    GraphOptimizerTest::SetUp();
    setGlobalConfForTest(GCFG_ENABLE_PIPELINE_MANAGEMENT, "true");
    CompilationHalReader::setHalReader(Gaudi2HalReader::instance());
}

void PipelineManagementTest::TearDown()
{
    GraphOptimizerTest::TearDown();
}

TensorPtr
PipelineManagementTest::createTensor(std::vector<TSize> shape, synDataType dataType, bool isPersistent /*= true*/)
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

struct BundleNodesComposition
{
    bool hasMme = false;
    bool hasTpc = false;
    bool hasReshape = false;
};
using NodeCountersPerBundleMap = std::map<unsigned, BundleNodesComposition>;
using BundleNodeCounters       = std::pair<unsigned, BundleNodesComposition>;

std::shared_ptr<NodeCountersPerBundleMap> getNodeCountersPerBundle(HabanaGraph& g)
{
    auto nodeCountersPerBundleId = std::make_shared<NodeCountersPerBundleMap>();
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (n->getNodeAnnotation().bundleInfo.is_set())
        {
            unsigned bundleIdx      = n->getNodeAnnotation().bundleInfo->bundleIndex;
            auto     bundleCounters = nodeCountersPerBundleId->find(bundleIdx);
            if (bundleCounters == nodeCountersPerBundleId->end())
            {
                // add a new bundle
                (*nodeCountersPerBundleId)[bundleIdx] = {};
            }
            (*nodeCountersPerBundleId)[bundleIdx].hasMme =
                (*nodeCountersPerBundleId)[bundleIdx].hasMme || HabanaGraph::runsOnMME(n);
            (*nodeCountersPerBundleId)[bundleIdx].hasTpc =
                (*nodeCountersPerBundleId)[bundleIdx].hasTpc || HabanaGraph::runsOnTPC(n);
            (*nodeCountersPerBundleId)[bundleIdx].hasReshape =
                (*nodeCountersPerBundleId)[bundleIdx].hasReshape || (n->getNodeType() == Node::TYPE_INTERNAL_RESHAPE);
        }
    }
    return nodeCountersPerBundleId;
}

enum NodeTypesInBundle
{
    bundleIncludesMme,
    bundleIncludesTpc,
    bundleIncludesMmeAndTpc,
    bundleIncludesMmeAndTpcAndReshape,
    bundleIncludesMmeOnly
};

bool findBundleWithNodesTypes(std::shared_ptr<NodeCountersPerBundleMap>& counters, NodeTypesInBundle expectedNodeTypes)
{
    auto iter = std::find_if(counters->begin(), counters->end(), [&](const BundleNodeCounters& bundleCounters) {
        // Check for match of this bundle according to requested nodes composition
        switch (expectedNodeTypes)
        {
            case bundleIncludesMme:
                return bundleCounters.second.hasMme;
            case bundleIncludesTpc:
                return bundleCounters.second.hasTpc;
            case bundleIncludesMmeAndTpc:
                return bundleCounters.second.hasMme && bundleCounters.second.hasTpc;
            case bundleIncludesMmeAndTpcAndReshape:
                return bundleCounters.second.hasMme && bundleCounters.second.hasTpc && bundleCounters.second.hasReshape;
            case bundleIncludesMmeOnly:
                return bundleCounters.second.hasMme && !bundleCounters.second.hasTpc;
            default:
                // invalid bundle type
                return false;
        }
        return false;
    });
    return (iter != counters->end());
}

bool isMmeInputIdxInSram(HabanaGraph& g, std::optional<unsigned> inputIdxInSram)
{
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (g.runsOnMME(n))
        {
            if (inputIdxInSram)
            {
                // validate the specific input index is in SRAM
                if (!n->getInput(*inputIdxInSram)->inSram()) return false;
            }
            else
            {
                // validate that one of the node's inputs is in SRAM, doesn't matter which
                if (!n->getInput(0)->inSram() && !n->getInput(1)->inSram()) return false;
            }
        }
    }
    return true;
}

bool isMmeOutputInSram(HabanaGraph& g)
{
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (g.runsOnMME(n))
        {
            if (!n->getOutput(0)->inSram()) return false;
        }
    }
    return true;
}

bool isReshapeOutputInSram(HabanaGraph& g)
{
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            if (!n->getOutput(0)->inSram()) return false;
        }
    }
    return true;
}

static bool isBGemmSlicedOnAllBatchDims(const NodePtr&               bGemm,
                                 const unsigned               inputIndex,
                                 const std::vector<TSize>&    originalShape,
                                 const std::vector<unsigned>& batchDims)
{
    return std::all_of(batchDims.begin(),
                       batchDims.end(),
                       [&bGemm, &inputIndex, &originalShape](const unsigned batchDim) {
                           return bGemm->getInput(inputIndex)->getSizeInElements(batchDim) < originalShape[batchDim];
                       });
}

// Check that MME with a valid TPC producer are bundled
TEST_F(PipelineManagementTest, mme_with_tpc_producer_bundled)
{
    Gaudi2Graph g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    setGlobalConfForTest(GCFG_MIN_CYCLES_FOR_MME_SLICING, "0");

    // Gemm
    synGEMMParams gemmParams {};
    TensorPtr     relu_out = createTensor({256, 256}, syn_type_bf16);
    TensorPtr     b        = createTensor({256, 256}, syn_type_bf16);
    TensorPtr     gemmOut  = createTensor({256, 256}, syn_type_bf16);
    NodePtr       gemm =
        NodeFactory::createNode({relu_out, b}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm));

    // Relu producer
    TensorPtr relu_in = createTensor({256, 256}, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({relu_in}, {relu_out}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // assert the MME input for the TPC producer is in SRAM
    ASSERT_TRUE(isMmeInputIdxInSram(g, 0));
}

// Check that MME with a producer which can't be sliced are not bundled together
TEST_F(PipelineManagementTest, mme_with_unsliceable_producer_not_bundled)
{
    Gaudi2Graph g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    synGEMMParams gemmParams {};
    TensorPtr     reduceIn  = createTensor({256, 256}, syn_type_bf16);
    TensorPtr     reduceOut = createTensor({256, 1}, syn_type_bf16);
    TensorPtr     b         = createTensor({256, 256}, syn_type_bf16);
    TensorPtr     gemmOut   = createTensor({256, 1}, syn_type_bf16);
    NodePtr       gemm =
        NodeFactory::createNode({reduceOut, b}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm));

    // Non seperable producer
    ns_Reduction::Params params;
    params.reductionDimension = 1;
    NodePtr avgpool =
        NodeFactory::createGenericTPCNode({reduceIn}, {reduceOut}, &params, "reduce_sum_fwd_bf16", "reduce_sum");
    ASSERT_TRUE(GraphEditor::addNode(g, avgpool));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the TPC node is not bundled
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is no bundle with TPC
    ASSERT_TRUE(!findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesTpc));
}

// Check that MME with TPC producer, which is shared with another MME, is bundled only with one of them
TEST_F(PipelineManagementTest, mme_with_bundled_tpc_producer_not_bundled)
{
    Gaudi2Graph g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    setGlobalConfForTest(GCFG_MIN_CYCLES_FOR_MME_SLICING, "0");

    // Gemm
    synGEMMParams gemmParams {};
    TensorPtr     relu_out = createTensor({256, 256}, syn_type_bf16);
    TensorPtr     b1       = createTensor({256, 256}, syn_type_bf16);
    TensorPtr     gemmOut1 = createTensor({256, 256}, syn_type_bf16);
    NodePtr       gemm1 =
        NodeFactory::createNode({relu_out, b1}, {gemmOut1}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm1");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm1));

    // Another Gemm
    TensorPtr a2       = createTensor({256, 256}, syn_type_bf16);
    TensorPtr gemmOut2 = createTensor({256, 256}, syn_type_bf16);
    NodePtr   gemm2 =
        NodeFactory::createNode({a2, relu_out}, {gemmOut2}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm2");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm2));

    // Relu producer
    TensorPtr relu_in = createTensor({256, 256}, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({relu_in}, {relu_out}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert gemm+relu nodes are bundled together, while the other gemm is not bundled with a producer
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is 1 bundle - MME + TPC producer
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle is with MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
}

// Check that MME with a valid TPC consumer are bundled
TEST_F(PipelineManagementTest, mme_with_tpc_consumer_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Conv
    synConvolutionParams  params {};
    std::vector<TSize>    aSizes   = {512, 28, 28, 256};
    std::vector<TSize>    bSizes   = {256, 512, 1, 1};
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             a        = createTensor(aSizes, syn_type_bf16);
    TensorPtr             b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16);
    NodePtr conv = NodeFactory::createNode({a, b}, {convOut}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    // Relu consumer
    TensorPtr reluOut = createTensor(outSizes, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({convOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // Assert the MME output for the TPC consumer is in SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
}

// Check that MME with a consumer which can't be sliced are not bundled together
TEST_F(PipelineManagementTest, mme_with_unsliceable_consumer_not_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    aSizes   = {512, 28, 28, 256};
    std::vector<TSize>    bSizes   = {256, 512, 1, 1};
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             a        = createTensor(aSizes, syn_type_bf16);
    TensorPtr             b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16);
    NodePtr               conv =
        NodeFactory::createNode({a, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    // Non seperable consumer
    std::vector<TSize>    reducedSizes = {256, 28, 28, 1};
    TensorPtr             reduceOut    = createTensor(reducedSizes, syn_type_bf16);
    ns_Reduction::Params  consumerParams;
    consumerParams.reductionDimension = 3;
    NodePtr reduceSum =
        NodeFactory::createGenericTPCNode({convOut}, {reduceOut}, &consumerParams, "reduce_sum_fwd_bf16", "reduce_sum");
    ASSERT_TRUE(GraphEditor::addNode(g, reduceSum));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the TPC node is not bundled
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is no bundle with TPC
    ASSERT_TRUE(!findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesTpc));
}

// Check that MME with a valid TPC consumer chain are bundled
TEST_F(PipelineManagementTest, mme_with_tpc_consumer_chain_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    aSizes   = {512, 28, 28, 256};
    std::vector<TSize>    bSizes   = {256, 512, 1, 1};
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             a        = createTensor(aSizes, syn_type_bf16);
    TensorPtr             b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16);
    NodePtr               conv =
        NodeFactory::createNode({a, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    // Reshape
    std::vector<TSize>    reshapedSizes = {256 * 28, 28, 256};
    TensorPtr             reshapeOut    = createTensor(reshapedSizes, syn_type_bf16);
    NodePtr               reshape = NodeFactory::createNode({convOut}, {reshapeOut}, nullptr, "reshape", "reshape");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));

    // Relu consumer
    TensorPtr reluOut = createTensor(reshapedSizes, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({reshapeOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 3 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME, TPC and reshape
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpcAndReshape));
    // Assert the MME output for the TPC consumer is in SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
    // Assert reshape output is in SRAM
    ASSERT_TRUE(isReshapeOutputInSram(g));
}
// Check that MME with a valid TPC producer and consumer chains are bundled
TEST_F(PipelineManagementTest, mme_with_tpc_producer_and_consumer_chain_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Relu producer for first input
    std::vector<TSize>    aReshapedSizes = {512 * 28, 28, 256};
    TensorPtr             aIn            = createTensor(aReshapedSizes, syn_type_bf16);
    TensorPtr             a              = createTensor(aReshapedSizes, syn_type_bf16);
    NodePtr               reluA = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(g, reluA));

    // Reshape for first input
    std::vector<TSize>    aSizes    = {512, 28, 28, 256};
    TensorPtr             reshapedA = createTensor(aSizes, syn_type_bf16);
    NodePtr               reshapeA  = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeA));

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    bSizes   = {256, 512, 1, 1};
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16);
    NodePtr               conv =
        NodeFactory::createNode({reshapedA, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    // Reshape output
    std::vector<TSize>    outReshapedSizes = {256 * 28, 28, 256};
    TensorPtr             reshapedOut      = createTensor(outReshapedSizes, syn_type_bf16);
    NodePtr reshapeOut = NodeFactory::createNode({convOut}, {reshapedOut}, nullptr, "reshape", "reshapeOut");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeOut));

    // Relu consumer
    TensorPtr reluOut = createTensor(outReshapedSizes, syn_type_bf16);
    NodePtr   reluConsumer =
        NodeFactory::createGenericTPCNode({reshapedOut}, {reluOut}, nullptr, "relu_fwd_bf16", "reluConsumer");
    ASSERT_TRUE(GraphEditor::addNode(g, reluConsumer));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 5 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME, TPC and reshape
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpcAndReshape));
    // Assert the MME output for the TPC consumer is in SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
    // Assert reshape output is in SRAM
    ASSERT_TRUE(isReshapeOutputInSram(g));
}

// Check that MME with 2 valid TPC producer chains and consumer chain are bundled
TEST_F(PipelineManagementTest, mme_with_two_tpc_producers_and_consumer_chain_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Relu producer for first input
    std::vector<TSize>    aReshapedSizes = {512 * 28, 28, 256};
    TensorPtr             aIn            = createTensor(aReshapedSizes, syn_type_bf16);
    TensorPtr             a              = createTensor(aReshapedSizes, syn_type_bf16);
    NodePtr               reluA = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(g, reluA));

    // Reshape for first input
    std::vector<TSize>    aSizes    = {512, 28, 28, 256};
    TensorPtr             reshapedA = createTensor(aSizes, syn_type_bf16);
    NodePtr               reshapeA  = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeA));

    // Relu producer for second input
    std::vector<TSize>    bSizes = {256, 512, 1, 1};
    TensorPtr             bIn    = createTensor(bSizes, syn_type_bf16);
    TensorPtr             b      = createTensor(bSizes, syn_type_bf16);
    NodePtr               reluB  = NodeFactory::createGenericTPCNode({bIn}, {b}, nullptr, "relu_fwd_bf16", "reluB");
    ASSERT_TRUE(GraphEditor::addNode(g, reluB));

    // Conv
    synConvolutionParams  convParams {};
    std::vector<TSize>    outSizes = {256, 28, 28, 256};
    TensorPtr             convOut  = createTensor(outSizes, syn_type_bf16);
    NodePtr               conv =
        NodeFactory::createNode({reshapedA, b}, {convOut}, &convParams, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(GraphEditor::addNode(g, conv));

    // Reshape output
    std::vector<TSize>    outReshapedSizes = {256 * 28, 28, 256};
    TensorPtr             reshapedOut      = createTensor(outReshapedSizes, syn_type_bf16);
    NodePtr reshapeOut = NodeFactory::createNode({convOut}, {reshapedOut}, nullptr, "reshape", "reshapeOut");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeOut));

    // Relu consumer
    TensorPtr reluOut = createTensor(outReshapedSizes, syn_type_bf16);
    NodePtr   reluConsumer =
        NodeFactory::createGenericTPCNode({reshapedOut}, {reluOut}, nullptr, "relu_fwd_bf16", "reluConsumer");
    ASSERT_TRUE(GraphEditor::addNode(g, reluConsumer));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 6 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME, TPC and reshape
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpcAndReshape));
    // Assert the MME output for the TPC consumer is in SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
    // Assert reshape output is in SRAM
    ASSERT_TRUE(isReshapeOutputInSram(g));
}
// Check that MME with a valid logic operation and TPC consumers chain are bundled by language bundlizer
TEST_F(PipelineManagementTest, bgemm_with_logic_op_and_tpc_consumer_chain_bundled_by_language)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // BGEMM
    synConvolutionParams bgemmParams {};
    std::vector<TSize>   aSizes   = {512, 28, 28, 256};
    std::vector<TSize>   bSizes   = {256, 512, 28, 256};
    std::vector<TSize>   outSizes = {256, 28, 28, 256};
    TensorPtr            a        = createTensor(aSizes, syn_type_bf16);
    TensorPtr            b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr            bgemmOut = createTensor(outSizes, syn_type_bf16);
    NodePtr              bgemm =
        NodeFactory::createNode({a, b}, {bgemmOut}, &bgemmParams, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    // Reshape
    std::vector<TSize> reshapedSizes = {256 * 28, 28, 256};
    TensorPtr          reshapeOut    = createTensor(reshapedSizes, syn_type_bf16);
    NodePtr            reshape       = NodeFactory::createNode({bgemmOut}, {reshapeOut}, nullptr, "reshape", "reshape");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));

    // Relu consumer
    TensorPtr reluOut = createTensor(reshapedSizes, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({reshapeOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 3 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME, TPC and reshape
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpcAndReshape));
    // Assert the MME output for the TPC consumer is in SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
    // Assert reshape output is in SRAM
    ASSERT_TRUE(isReshapeOutputInSram(g));
}

// Check that MME with a valid TPC consumer chain are bundled by language bundlizer
TEST_F(PipelineManagementTest, bgemm_with_tpc_consumer_chain_bundled_by_language)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    setGlobalConfForTest(GCFG_ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS, "true");

    // BGEMM
    synConvolutionParams bgemmParams {};
    std::vector<TSize>   aSizes   = {512, 28, 28, 256};
    std::vector<TSize>   bSizes   = {256, 512, 28, 256};
    std::vector<TSize>   outSizes = {256, 28, 28, 256};
    TensorPtr            a        = createTensor(aSizes, syn_type_bf16);
    TensorPtr            b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr            bgemmOut = createTensor(outSizes, syn_type_bf16);
    NodePtr              bgemm =
        NodeFactory::createNode({a, b}, {bgemmOut}, &bgemmParams, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    // Relu consumer
    TensorPtr reluOut = createTensor(outSizes, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({bgemmOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // Assert the MME output for the TPC consumer is in SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
}

// Check that bgemm with a valid TPC producers and consumers chains will bundle all producers and consumers by language
// bundlizer
TEST_F(PipelineManagementTest, bgemm_with_tpc_producer_and_consumer_chain_should_bundle_all_by_language)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    setGlobalConfForTest(GCFG_ENABLE_BUNDLES_WITH_CONSUMERS_AND_PRODUCERS, "true");

    // Relu producer for first input
    std::vector<TSize> aSizes = {512 * 28, 28, 256};
    TensorPtr          aIn    = createTensor(aSizes, syn_type_bf16);
    TensorPtr          a      = createTensor(aSizes, syn_type_bf16);
    NodePtr            reluA  = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(g, reluA));
    // Reshape for first input
    std::vector<TSize> aReshapedSizes = {512, 28, 28, 256};
    TensorPtr          reshapedA      = createTensor(aReshapedSizes, syn_type_bf16);
    NodePtr            reshapeA       = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeA));

    // BGEMM
    synConvolutionParams bgemmParams {};
    std::vector<TSize>   bSizes   = {256, 512, 28, 256};
    std::vector<TSize>   outSizes = {256, 28, 28, 256};
    TensorPtr            b        = createTensor(bSizes, syn_type_bf16);
    TensorPtr            bgemmOut = createTensor(outSizes, syn_type_bf16);
    NodePtr              bgemm =
        NodeFactory::createNode({reshapedA, b}, {bgemmOut}, &bgemmParams, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    // Reshape output
    std::vector<TSize> outReshapedSizes = {256 * 28, 28, 256};
    TensorPtr          reshapedOut      = createTensor(outReshapedSizes, syn_type_bf16);
    NodePtr reshapeOut = NodeFactory::createNode({bgemmOut}, {reshapedOut}, nullptr, "reshape", "reshapeOut");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeOut));

    // Relu consumer
    TensorPtr reluOut = createTensor(outReshapedSizes, syn_type_bf16);
    NodePtr   reluConsumer =
        NodeFactory::createGenericTPCNode({reshapedOut}, {reluOut}, nullptr, "relu_fwd_bf16", "reluConsumer");
    ASSERT_TRUE(GraphEditor::addNode(g, reluConsumer));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 5 nodes are not bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_FALSE(std::any_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return !n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME, TPC and reshape (from producer)
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpcAndReshape));
    // Assert the MME output for the TPC consumer is SRAM
    ASSERT_TRUE(isMmeOutputInSram(g));
    // Assert reshape output of consumer chain is in SRAM
    ASSERT_TRUE(isReshapeOutputInSram(g));
}

TEST_F(PipelineManagementTest,
       tpc_node_between_2_mme_nodes_should_bundle_the_tpc_with_the_second_mme_as_a_producer_by_language)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Relu producer for first input
    std::vector<TSize> aSizes = {512 * 28, 28, 256};
    TensorPtr          aIn    = createTensor(aSizes, syn_type_bf16);
    TensorPtr          a      = createTensor(aSizes, syn_type_bf16);
    NodePtr            reluA  = NodeFactory::createGenericTPCNode({aIn}, {a}, nullptr, "relu_fwd_bf16", "reluA");
    ASSERT_TRUE(GraphEditor::addNode(g, reluA));
    // Reshape for first input
    std::vector<TSize> aReshapedSizes = {512, 28, 28, 256};
    TensorPtr          reshapedA      = createTensor(aReshapedSizes, syn_type_bf16);
    NodePtr            reshapeA       = NodeFactory::createNode({a}, {reshapedA}, nullptr, "reshape", "reshapeA");
    ASSERT_TRUE(GraphEditor::addNode(g, reshapeA));

    // BGEMM A
    synConvolutionParams bgemmParams {};
    std::vector<TSize>   bSizes    = {256, 512, 28, 256};
    std::vector<TSize>   outSizes  = {256, 28, 28, 256};
    TensorPtr            b         = createTensor(bSizes, syn_type_bf16);
    TensorPtr            bgemmAOut = createTensor(outSizes, syn_type_bf16);
    NodePtr              bgemm     = NodeFactory::createNode({reshapedA, b},
                                                             {bgemmAOut},
                                            &bgemmParams,
                                            NodeFactory::batchGemmNodeTypeName,
                                            "bgemmA");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    // Relu consumer of bgemmA
    TensorPtr reluBOut = createTensor(outSizes, syn_type_bf16);
    NodePtr   reluB    = NodeFactory::createGenericTPCNode({bgemmAOut}, {reluBOut}, nullptr, "relu_fwd_bf16", "reluB");
    ASSERT_TRUE(GraphEditor::addNode(g, reluB));

    // BGEMM B
    synConvolutionParams bgemmBParams {};
    std::vector<TSize>   cSizes    = {28, 28, 28, 256};
    std::vector<TSize>   outBSizes = {28, 256, 28, 256};
    TensorPtr            c         = createTensor(bSizes, syn_type_bf16);
    TensorPtr            bgemmBOut = createTensor(outBSizes, syn_type_bf16);
    NodePtr              bgemmB    = NodeFactory::createNode({c, reluBOut},
                                                             {bgemmBOut},
                                             &bgemmBParams,
                                             NodeFactory::batchGemmNodeTypeName,
                                             "bgemmB");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemmB));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there are 2 bundles
    ASSERT_EQ(nodeCountersPerBundleId->size(), 2);
    // reluB should be bundle as bgemmB producer and not as bgemmA consumer
    ASSERT_FALSE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeOnly));
}

TEST_F(PipelineManagementTest, tpc_reshape_mme_bundle_test)
{
    Gaudi2Graph                graph;
    CompilationHalReaderSetter compHalReaderSetter(&graph);

    std::vector<TSize> ifmShape         = {256, 4096};
    std::vector<TSize> ofmShape         = {512, 4096};
    std::vector<TSize> wghShape         = {ofmShape.front(), ifmShape.front()};
    std::vector<TSize> reshapedIfmShape = {ifmShape.front(), 1, 1, ifmShape.back()};

    auto reluIn  = createTensor(reshapedIfmShape, syn_type_single, true);
    auto reluOut = createTensor(reshapedIfmShape, syn_type_single, false);
    auto gemmIfm = createTensor(ifmShape, syn_type_single, false);
    auto gemmWgh = createTensor(wghShape, syn_type_single, true);
    auto gemmOfm = createTensor(ofmShape, syn_type_single, true);

    NodePtr relu = NodeFactory::createNode({reluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu");
    NodePtr reshape =
        NodeFactory::createNode({reluOut}, {gemmIfm}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    synGEMMParams gemmParams {};
    NodePtr       gemm =
        NodeFactory::createNode({gemmIfm, gemmWgh}, {gemmOfm}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");

    GraphEditor::addNode(graph, relu);
    GraphEditor::addNode(graph, reshape);
    GraphEditor::addNode(graph, gemm);

    ASSERT_TRUE(gaudi2::loadTpcKernels(graph));
    ASSERT_TRUE(sliceGraphForPipeline(graph));
}

TEST_F(PipelineManagementTest, shared_mme_with_producer_and_consumer_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // BN producer for first input
    ns_BatchNormKernel::ParamsV2 bnParams = {};
    bnParams.momentum                     = 0.1;
    bnParams.threshold.f                  = 1e-07;
    bnParams.epsilon                      = 1e-05;
    bnParams.isTraining                   = true;

    std::vector<TSize>    bnSizes       = {64, 56, 56, 256};
    std::vector<TSize>    bnSizes2      = {64};
    TensorPtr             bnIn0         = createTensor(bnSizes, syn_type_bf16);
    TensorPtr             bnIn1         = createTensor(bnSizes, syn_type_bf16);
    TensorPtr             bnIn2         = createTensor(bnSizes2, syn_type_float);
    TensorPtr             bnIn3         = createTensor(bnSizes2, syn_type_float);
    TensorPtr             bnIn4         = createTensor(bnSizes2, syn_type_float);
    TensorPtr             bnOut0        = createTensor(bnSizes, syn_type_bf16, false);
    TensorPtr             bnOut1        = createTensor(bnSizes2, syn_type_float, false);
    TensorPtr             bnOut2        = createTensor(bnSizes2, syn_type_float, false);
    NodePtr               bnBwdProducer = NodeFactory::createNode({bnIn0, bnIn1, bnIn2, bnIn3, bnIn4},
                                                    {bnOut0, bnOut1, bnOut2},
                                                    &bnParams,
                                                    "batch_norm_bwd_bf16",
                                                    "bnBwdProducer");
    ASSERT_TRUE(GraphEditor::addNode(g, bnBwdProducer));

    // Dedx
    synConvolutionParams  convParams {};
    std::vector<TSize>    wSizes  = {64, 256, 1, 1};
    std::vector<TSize>    dxSizes = {256, 56, 56, 256};
    TensorPtr             w       = createTensor(wSizes, syn_type_bf16);
    TensorPtr             dx      = createTensor(dxSizes, syn_type_bf16, false);
    NodePtr dedx = NodeFactory::createNode({bnOut0, w}, {dx}, &convParams, NodeFactory::deDxNodeTypeName, "dedx");
    ASSERT_TRUE(GraphEditor::addNode(g, dedx));

    // Dedw
    TensorPtr x    = createTensor(dxSizes, syn_type_bf16);
    TensorPtr dw   = createTensor(wSizes, syn_type_bf16, false);
    NodePtr   dedw = NodeFactory::createNode({bnOut0, x}, {dw}, &convParams, NodeFactory::deDwNodeTypeName, "dedw");
    ASSERT_TRUE(GraphEditor::addNode(g, dedw));

    // Add consumer
    TensorPtr addIn0 = createTensor(dxSizes, syn_type_bf16);
    TensorPtr addOut = createTensor(dxSizes, syn_type_bf16);
    NodePtr   add    = NodeFactory::createNode({addIn0, dx}, {addOut}, nullptr, "add_fwd_bf16", "addConsumer");
    ASSERT_TRUE(GraphEditor::addNode(g, add));

    ASSERT_TRUE(g.compile());

    // Assert the 6 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is one bundles
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert that exactly one bundle has MME, TPC and reshape
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // Assert dedx and dedw outputs are in SRAM
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (g.runsOnMME(n))
        {
            ASSERT_TRUE(n->getOutput(0)->inSram());
        }
    }
}

// Check that Broadcast batch-gemm with a valid TPC producer are bundled
TEST_F(PipelineManagementTest, broadcast_bgemm_with_tpc_producer_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    setGlobalConfForTest(GCFG_MIN_CYCLES_FOR_MME_SLICING, "0");

    // Batch gemm
    synGEMMParams gemmParams {};
    TensorPtr     relu_out   = createTensor({64, 64, 16}, syn_type_bf16);
    TensorPtr     b          = createTensor({64, 64}, syn_type_bf16);
    TensorPtr     gemmOut    = createTensor({64, 64, 16}, syn_type_bf16);
    NodePtr       batch_gemm = NodeFactory::createNode({relu_out, b},
                                                 {gemmOut},
                                                 &gemmParams,
                                                 NodeFactory::batchGemmNodeTypeName,
                                                 "batch_gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, batch_gemm));

    // Relu producer
    TensorPtr relu_in = createTensor({64, 64, 16}, syn_type_bf16);
    NodePtr   relu    = NodeFactory::createGenericTPCNode({relu_in}, {relu_out}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(alignAsymmetricBgemm(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // assert the MME input for the TPC producer is in SRAM
    ASSERT_TRUE(isMmeInputIdxInSram(g, 0));
}

TEST_F(PipelineManagementTest, language_bundlizer_toggle_bundle_transpose_on)
{
    Gaudi2Graph g;
    setGlobalConfForTest(GCFG_ENABLE_BUNDLE_TRANSPOSE, "true");

    TensorPtr g_0_tensor_20 = createTensor({96, 200704, 1, 1}, syn_type_single, true);
    TensorPtr g_0_tensor_21 = createTensor({96}, syn_type_single, true);
    TensorPtr g_0_tensor_22 = createTensor({96}, syn_type_single, true);
    TensorPtr g_0_tensor_23 = createTensor({96, 200704, 1, 1}, syn_type_single, false);
    TensorPtr g_0_tensor_24_1645_0_0_norm1_aten_native_layer_norm =
        createTensor({1, 200704, 1, 1}, syn_type_single, false);
    TensorPtr g_0_tensor_25_1646_0_0_norm1_aten_native_layer_norm =
        createTensor({1, 200704, 1, 1}, syn_type_single, false);

    unsigned char g_0_0_0_norm1_layer_norm_fwd_f32_13_0_params[] = {1, 0, 0, 0, 172, 197, 39, 55};

    NodePtr layerNorm = NodeFactory::createNode({g_0_tensor_20, g_0_tensor_21, g_0_tensor_22},
                                                {g_0_tensor_23,
                                                 g_0_tensor_24_1645_0_0_norm1_aten_native_layer_norm,
                                                 g_0_tensor_25_1646_0_0_norm1_aten_native_layer_norm},
                                                (void*)g_0_0_0_norm1_layer_norm_fwd_f32_13_0_params,
                                                sizeof(g_0_0_0_norm1_layer_norm_fwd_f32_13_0_params),
                                                "layer_norm_fwd_f32",
                                                "g_0_0_0_norm1_layer_norm_fwd_f32_13_0");
    ASSERT_TRUE(GraphEditor::addNode(g, layerNorm));

    TensorPtr g_0_tensor_26_1644_0_0_norm1_aten_native_layer_norm =
        createTensor({96, 3136, 64}, syn_type_single, false);

    NodePtr reshape = NodeFactory::createNode({g_0_tensor_23},
                                              {g_0_tensor_26_1644_0_0_norm1_aten_native_layer_norm},
                                              nullptr,
                                              0,
                                              "reshape",
                                              "g_0_0_0_norm1_reshape_14_0");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));

    TensorPtr g_0_tensor_27_1652_0_0_aten_view = createTensor({96, 7, 8, 7, 512}, syn_type_single, false);
    NodePtr   reshape1 = NodeFactory::createNode({g_0_tensor_26_1644_0_0_norm1_aten_native_layer_norm},
                                               {g_0_tensor_27_1652_0_0_aten_view},
                                               nullptr,
                                               0,
                                               "reshape",
                                               "g_0_0_0_reshape_15_0");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape1));

    TensorPtr g_0_tensor_28_1655_0_0_aten_permute = createTensor({96, 7, 7, 8, 512}, syn_type_single, false);

    unsigned char g_0_0_0_transpose_16_0_params[] = {0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0,
                                                     2, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0};

    NodePtr transpose = NodeFactory::createNode({g_0_tensor_27_1652_0_0_aten_view},
                                                {g_0_tensor_28_1655_0_0_aten_permute},
                                                (void*)g_0_0_0_transpose_16_0_params,
                                                sizeof(g_0_0_0_transpose_16_0_params),
                                                "transpose",
                                                "g_0_0_0_transpose_16_0");
    ASSERT_TRUE(GraphEditor::addNode(g, transpose));

    TensorPtr g_0_tensor_29_1661_0_0_attn_qkv_aten_view = createTensor({96, 49, 4096}, syn_type_single, false);

    NodePtr reshape2 = NodeFactory::createNode({g_0_tensor_28_1655_0_0_aten_permute},
                                               {g_0_tensor_29_1661_0_0_attn_qkv_aten_view},
                                               nullptr,
                                               "reshape",
                                               "g_0_0_0_attn_qkv_reshape_17_0");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape2));

    TensorPtr g_0_tensor_1_1664_0_0_attn_qkv_aten_t       = createTensor({288, 96}, syn_type_single, true);
    TensorPtr g_0_tensor_30_1667_0_0_attn_qkv_aten_matmul = createTensor({288, 49, 4096}, syn_type_single, false);
    NodePtr   bgemm =
        NodeFactory::createNode({g_0_tensor_29_1661_0_0_attn_qkv_aten_view, g_0_tensor_1_1664_0_0_attn_qkv_aten_t},
                                {g_0_tensor_30_1667_0_0_attn_qkv_aten_matmul},
                                nullptr,
                                "batch_gemm",
                                "g_0_0_0_attn_qkv_batch_gemm_18_0");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    TensorPtr g_0_tensor_31                             = createTensor({288}, syn_type_single, true);
    TensorPtr g_0_tensor_32_1667_0_0_attn_qkv_aten_add_ = createTensor({288, 49, 4096}, syn_type_single, true);
    NodePtr   add = NodeFactory::createGenericTPCNode({g_0_tensor_30_1667_0_0_attn_qkv_aten_matmul, g_0_tensor_31},
                                                    {g_0_tensor_32_1667_0_0_attn_qkv_aten_add_},
                                                    nullptr,
                                                    "add_fwd_f32",
                                                    "g_0_add_fwd_f32_20_0");
    ASSERT_TRUE(GraphEditor::addNode(g, add));
    ASSERT_TRUE(g.compile());

    const auto& graphNodes = g.getExeSortedNodes();
    const auto  it         = std::find_if(graphNodes.begin(), graphNodes.end(), [](const auto& node) {
        return node && node->isTranspose();
    });

    ASSERT_TRUE(it != graphNodes.end()) << "Expecting a transpose node";
    const auto& transposeNode = *it;
    ASSERT_TRUE(transposeNode->getNodeAnnotation().bundleInfo.is_set()) << "Expecting transpose bundled";
}

TEST_F(PipelineManagementTest, language_bundlizer_toggle_bundle_transpose_off)
{
    Gaudi2Graph g;
    setGlobalConfForTest(GCFG_ENABLE_BUNDLE_TRANSPOSE, "false");

    TensorPtr g_0_tensor_20 = createTensor({96, 200704, 1, 1}, syn_type_single, true);
    TensorPtr g_0_tensor_21 = createTensor({96}, syn_type_single, true);
    TensorPtr g_0_tensor_22 = createTensor({96}, syn_type_single, true);
    TensorPtr g_0_tensor_23 = createTensor({96, 200704, 1, 1}, syn_type_single, false);
    TensorPtr g_0_tensor_24_1645_0_0_norm1_aten_native_layer_norm =
        createTensor({1, 200704, 1, 1}, syn_type_single, false);
    TensorPtr g_0_tensor_25_1646_0_0_norm1_aten_native_layer_norm =
        createTensor({1, 200704, 1, 1}, syn_type_single, false);

    unsigned char g_0_0_0_norm1_layer_norm_fwd_f32_13_0_params[] = {1, 0, 0, 0, 172, 197, 39, 55};

    NodePtr layerNorm = NodeFactory::createNode({g_0_tensor_20, g_0_tensor_21, g_0_tensor_22},
                                                {g_0_tensor_23,
                                                 g_0_tensor_24_1645_0_0_norm1_aten_native_layer_norm,
                                                 g_0_tensor_25_1646_0_0_norm1_aten_native_layer_norm},
                                                (void*)g_0_0_0_norm1_layer_norm_fwd_f32_13_0_params,
                                                sizeof(g_0_0_0_norm1_layer_norm_fwd_f32_13_0_params),
                                                "layer_norm_fwd_f32",
                                                "g_0_0_0_norm1_layer_norm_fwd_f32_13_0");
    ASSERT_TRUE(GraphEditor::addNode(g, layerNorm));

    TensorPtr g_0_tensor_26_1644_0_0_norm1_aten_native_layer_norm =
        createTensor({96, 3136, 64}, syn_type_single, false);

    NodePtr reshape = NodeFactory::createNode({g_0_tensor_23},
                                              {g_0_tensor_26_1644_0_0_norm1_aten_native_layer_norm},
                                              nullptr,
                                              0,
                                              "reshape",
                                              "g_0_0_0_norm1_reshape_14_0");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));

    TensorPtr g_0_tensor_27_1652_0_0_aten_view = createTensor({96, 7, 8, 7, 512}, syn_type_single, false);
    NodePtr   reshape1 = NodeFactory::createNode({g_0_tensor_26_1644_0_0_norm1_aten_native_layer_norm},
                                               {g_0_tensor_27_1652_0_0_aten_view},
                                               nullptr,
                                               0,
                                               "reshape",
                                               "g_0_0_0_reshape_15_0");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape1));

    TensorPtr g_0_tensor_28_1655_0_0_aten_permute = createTensor({96, 7, 7, 8, 512}, syn_type_single, false);

    unsigned char g_0_0_0_transpose_16_0_params[] = {0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0,
                                                     2, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0};

    NodePtr transpose = NodeFactory::createNode({g_0_tensor_27_1652_0_0_aten_view},
                                                {g_0_tensor_28_1655_0_0_aten_permute},
                                                (void*)g_0_0_0_transpose_16_0_params,
                                                sizeof(g_0_0_0_transpose_16_0_params),
                                                "transpose",
                                                "g_0_0_0_transpose_16_0");
    ASSERT_TRUE(GraphEditor::addNode(g, transpose));

    TensorPtr g_0_tensor_29_1661_0_0_attn_qkv_aten_view = createTensor({96, 49, 4096}, syn_type_single, false);

    NodePtr reshape2 = NodeFactory::createNode({g_0_tensor_28_1655_0_0_aten_permute},
                                               {g_0_tensor_29_1661_0_0_attn_qkv_aten_view},
                                               nullptr,
                                               "reshape",
                                               "g_0_0_0_attn_qkv_reshape_17_0");
    ASSERT_TRUE(GraphEditor::addNode(g, reshape2));

    TensorPtr g_0_tensor_1_1664_0_0_attn_qkv_aten_t       = createTensor({288, 96}, syn_type_single, true);
    TensorPtr g_0_tensor_30_1667_0_0_attn_qkv_aten_matmul = createTensor({288, 49, 4096}, syn_type_single, false);
    NodePtr   bgemm =
        NodeFactory::createNode({g_0_tensor_29_1661_0_0_attn_qkv_aten_view, g_0_tensor_1_1664_0_0_attn_qkv_aten_t},
                                {g_0_tensor_30_1667_0_0_attn_qkv_aten_matmul},
                                nullptr,
                                "batch_gemm",
                                "g_0_0_0_attn_qkv_batch_gemm_18_0");
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    TensorPtr g_0_tensor_31                             = createTensor({288}, syn_type_single, true);
    TensorPtr g_0_tensor_32_1667_0_0_attn_qkv_aten_add_ = createTensor({288, 49, 4096}, syn_type_single, true);
    NodePtr   add = NodeFactory::createGenericTPCNode({g_0_tensor_30_1667_0_0_attn_qkv_aten_matmul, g_0_tensor_31},
                                                    {g_0_tensor_32_1667_0_0_attn_qkv_aten_add_},
                                                    nullptr,
                                                    "add_fwd_f32",
                                                    "g_0_add_fwd_f32_20_0");
    ASSERT_TRUE(GraphEditor::addNode(g, add));
    ASSERT_TRUE(g.compile());

    const auto& graphNodes = g.getExeSortedNodes();
    const auto  it         = std::find_if(graphNodes.begin(), graphNodes.end(), [](const auto& node) {
        return node && node->isTranspose();
    });

    ASSERT_TRUE(it != graphNodes.end()) << "Expecting a transpose node";
    const auto& transposeNode = *it;
    ASSERT_TRUE(!transposeNode->getNodeAnnotation().bundleInfo.is_set()) << "Expecting transpose not bundled";
}

// Check that masked batch-gemm with a valid TPC producer are bundled and masks are sliced correctly
TEST_F(PipelineManagementTest, masked_bgemm_with_tpc_producer_slice_masks)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Masked batch gemm
    synGEMMParams gemmParams {};
    unsigned      batch           = 16;
    TensorPtr     reluOut         = createTensor({128, 128, 16, batch}, syn_type_bf16);
    TensorPtr     b               = createTensor({128, 128, 16, batch}, syn_type_bf16);
    TensorPtr     maskA           = createTensor({13, 128, 1, batch}, syn_type_bf16);
    TensorPtr     maskB           = createTensor({128, 13, 1, batch}, syn_type_bf16);
    TensorPtr     gemmOut         = createTensor({128, 128, 16, batch}, syn_type_bf16);
    NodePtr       maskedbatchGemm = NodeFactory::createNode({reluOut, b, maskA, maskB},
                                                      {gemmOut},
                                                      &gemmParams,
                                                      NodeFactory::maskedBatchGemmNodeTypeName,
                                                      "masked_batch_gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, maskedbatchGemm));

    // Relu producer
    TensorPtr reluIn = createTensor({128, 128, 16, batch}, syn_type_bf16);
    NodePtr   relu   = NodeFactory::createGenericTPCNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // assert the masked bgemm inputs are sliced correctly
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
        {
            // assert the external batch dim is sliced
            Dim      slicedBatchDim  = DIM_GEMM_BATCH + 1;
            unsigned slicedBatchSize = n->getInput(0)->getSizeInElements(slicedBatchDim);
            ASSERT_LT(slicedBatchSize, batch);
            // assert all inputs are sliced the same on the batch dim (including masts)
            for (const TensorPtr& input : n->getInputs())
            {
                ASSERT_EQ(input->getSizeInElements(slicedBatchDim), slicedBatchSize);
            }
        }
    }
}

// Check that masked bgemm is sliced on the internal batch dim when its external batch is degenerated
TEST_F(PipelineManagementTest, masked_bgemm_sliced_on_internal_batch_dim_when_external_degenerated)
{
    Gaudi2Graph g;

    // Masked batch gemm
    synGEMMParams params(true, false);
    TSize         internalBatchSize = 12;
    TensorPtr     a                 = createTensor({2048, 128, internalBatchSize, 1}, syn_type_bf16);
    TensorPtr     b                 = createTensor({2048, 128, internalBatchSize, 1}, syn_type_bf16);
    TensorPtr     maskA             = createTensor({2048, 6, 1, 1}, syn_type_bf16);
    TensorPtr     maskB             = createTensor({2048, 6, 1, 1}, syn_type_bf16);
    TensorPtr     maskedBGemmOut    = createTensor({2048, 2048, internalBatchSize, 1}, syn_type_bf16);
    NodePtr       maskedBGemm       = NodeFactory::createNode({a, b, maskA, maskB},
                                                              {maskedBGemmOut},
                                                  &params,
                                                  NodeFactory::maskedBatchGemmNodeTypeName,
                                                  "MaskedBGemm");
    ASSERT_TRUE(GraphEditor::addNode(g, maskedBGemm));

    ASSERT_TRUE(sliceGraphForPipeline(g));
    // assert the masked bgemm inputs are sliced correctly
    for (NodePtr n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
        {
            // assert the internal batch dim is sliced
            Dim      slicedBatchDim  = DIM_GEMM_BATCH;
            unsigned slicedBatchSize = n->getInput(0)->getSizeInElements(slicedBatchDim);
            ASSERT_LT(slicedBatchSize, internalBatchSize);
            // assert input 1 is sliced the same on the internal batch dim.
            ASSERT_EQ(n->getInput(1)->getSizeInElements(slicedBatchDim), slicedBatchSize);
        }
    }
}

// Check that transposed dedx with a valid TPC producer bundled
TEST_F(PipelineManagementTest, transposed_dedx_with_tpc_producer_to_w_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    synConvolutionParams convParams {};
    convParams.kH    = 1;
    convParams.kW    = 1;
    TSize channelsIn = 500, channelsOut = 1000, batch = 27, spatial = 154;

    // Relu producer for second input
    std::vector<TSize> wSizes = {channelsOut, channelsIn, convParams.kW, convParams.kH};
    // swap for transposed dedx
    std::swap(wSizes[WEIGHT_DIM_K], wSizes[WEIGHT_DIM_C]);
    TensorPtr reluIn = createTensor(wSizes, syn_type_single);
    TensorPtr w      = createTensor(wSizes, syn_type_single);
    NodePtr   reluW  = NodeFactory::createGenericTPCNode({reluIn}, {w}, nullptr, "relu_fwd_f32", "reluW");
    ASSERT_TRUE(GraphEditor::addNode(g, reluW));

    std::vector<TSize> dxSizes = {channelsIn, spatial, spatial, batch};
    std::vector<TSize> ySizes  = {channelsOut, spatial, spatial, batch};
    TensorPtr          dy      = createTensor(ySizes, syn_type_single);
    TensorPtr          dx      = createTensor(dxSizes, syn_type_single, false);
    NodePtr            dedxT =
        NodeFactory::createNode({dy, w}, {dx}, &convParams, NodeFactory::transposedDeDxNodeTypeName, "transposedDedx");

    ASSERT_TRUE(GraphEditor::addNode(g, dedxT));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert all nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
}

// Check that transposed dedx with a valid TPC producer bundled
TEST_F(PipelineManagementTest, transposed_dedx_with_tpc_producer_to_dy_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    synConvolutionParams convParams {};
    convParams.kH    = 1;
    convParams.kW    = 1;
    TSize channelsIn = 3, channelsOut = 256, batch = 27, spatial = 154;

    // Relu producer for first input
    std::vector<TSize> ySizes = {channelsOut, spatial, spatial, batch};
    TensorPtr          dy     = createTensor(ySizes, syn_type_bf16);
    TensorPtr          reluIn = createTensor(ySizes, syn_type_bf16);
    NodePtr            relu   = NodeFactory::createGenericTPCNode({reluIn}, {dy}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    std::vector<TSize> wSizes = {channelsOut, channelsIn, convParams.kW, convParams.kH};
    // swap for transposed dedx
    std::swap(wSizes[WEIGHT_DIM_K], wSizes[WEIGHT_DIM_C]);
    std::vector<TSize> dxSizes = {channelsIn, spatial, spatial, batch};
    TensorPtr          w       = createTensor(wSizes, syn_type_bf16);
    TensorPtr          dx      = createTensor(dxSizes, syn_type_bf16, false);
    NodePtr            dedxT =
        NodeFactory::createNode({dy, w}, {dx}, &convParams, NodeFactory::transposedDeDxNodeTypeName, "transposedDedx");

    ASSERT_TRUE(GraphEditor::addNode(g, dedxT));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert all nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
}

// Check that transposed dedx is bundled wish shared dedw
TEST_F(PipelineManagementTest, transposed_dedx_with_shared_dedw_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    synConvolutionParams convParams {};
    convParams.kH    = 1;
    convParams.kW    = 1;
    TSize channelsIn = 3, channelsOut = 256, batch = 27, spatial = 154;

    std::vector<TSize> wSizes           = {channelsOut, channelsIn, convParams.kW, convParams.kH};
    std::vector<TSize> transposedWSizes = wSizes;
    // swap for transposed dedx
    std::swap(transposedWSizes[WEIGHT_DIM_K], transposedWSizes[WEIGHT_DIM_C]);
    TensorPtr          w       = createTensor(transposedWSizes, syn_type_bf16);
    std::vector<TSize> dxSizes = {channelsIn, spatial, spatial, batch};
    std::vector<TSize> ySizes  = {channelsOut, spatial, spatial, batch};
    TensorPtr          dy      = createTensor(ySizes, syn_type_bf16);
    TensorPtr          dx      = createTensor(dxSizes, syn_type_bf16, false);
    NodePtr            dedxT =
        NodeFactory::createNode({dy, w}, {dx}, &convParams, NodeFactory::transposedDeDxNodeTypeName, "transposedDedx");
    ASSERT_TRUE(GraphEditor::addNode(g, dedxT));

    // Dedw
    TensorPtr x    = createTensor(dxSizes, syn_type_bf16);
    TensorPtr dw   = createTensor(wSizes, syn_type_bf16, false);
    NodePtr   dedw = NodeFactory::createNode({dy, x}, {dw}, &convParams, NodeFactory::deDwNodeTypeName, "dedw");
    ASSERT_TRUE(GraphEditor::addNode(g, dedw));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert all nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // Assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // Assert all the nodes in the graph are bundled
    const NodeSet& nodes = g.getNodes();
    ASSERT_TRUE(std::all_of(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set();
    }));
    // Assert this bundle has MME
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMme));
}

// Check that Broadcast batch-gemm with dims size diff 2 with a valid TPC producer are bundled
TEST_F(PipelineManagementTest, broadcast_bgemm_diff_2_with_tpc_producer_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    setGlobalConfForTest(GCFG_MIN_CYCLES_FOR_MME_SLICING, "0");

    // Batch gemm
    synGEMMParams gemmParams {};
    TensorPtr     reluOut = createTensor({64, 64, 16, 4}, syn_type_bf16);
    TensorPtr     b       = createTensor({64, 64}, syn_type_bf16);
    TensorPtr     gemmOut = createTensor({64, 64, 16, 4}, syn_type_bf16);
    NodePtr       batchGemm =
        NodeFactory::createNode({reluOut, b}, {gemmOut}, &gemmParams, NodeFactory::batchGemmNodeTypeName, "batch_gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, batchGemm));

    // Relu producer
    TensorPtr reluIn = createTensor({64, 64, 16, 4}, syn_type_bf16);
    NodePtr   relu   = NodeFactory::createGenericTPCNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(alignAsymmetricBgemm(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // assert the MME input for the TPC producer is in SRAM
    ASSERT_TRUE(isMmeInputIdxInSram(g, 0));
}

// Check that Broadcast batch-gemm with dims size diff 2 with a TPC producer for the weights are bundled
TEST_F(PipelineManagementTest, broadcast_bgemm_diff_2_with_weights_tpc_producer_bundled)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // Batch gemm
    synGEMMParams gemmParams {};
    TensorPtr     reluOut = createTensor({1024, 1024}, syn_type_bf16);
    TensorPtr     a       = createTensor({1024, 1024, 16, 4}, syn_type_bf16);
    TensorPtr     gemmOut = createTensor({1024, 1024, 16, 4}, syn_type_bf16);
    NodePtr       batch_gemm =
        NodeFactory::createNode({a, reluOut}, {gemmOut}, &gemmParams, NodeFactory::batchGemmNodeTypeName, "batch_gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, batch_gemm));

    // Relu producer
    TensorPtr reluIn = createTensor({1024, 1024}, syn_type_bf16);
    NodePtr   relu   = NodeFactory::createGenericTPCNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(alignAsymmetricBgemm(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // Assert the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    // assert the MME input for the TPC producer is in SRAM
    ASSERT_TRUE(isMmeInputIdxInSram(g, 1));
}

struct DedwDedxSharedInputTestParams
{
    std::vector<TSize> xSizes;
    std::vector<TSize> wSizes;
    std::vector<TSize> ySizes;
    unsigned kernelSize;
    unsigned stride;
    unsigned pad;
    bool     bundleTogether;
};

class DedwDedxSharedInputTest
: public PipelineManagementTest
, public testing::WithParamInterface<DedwDedxSharedInputTestParams>
{
public:

    void runSingleTest()
    {
        Gaudi2Graph                g;
        CompilationHalReaderSetter compHalReaderSetter(&g);

        // Create DEDX and dEDW nodes, sharing the same dy tensor
        synConvolution3DParams convParams = createConvParams();
        createTensors();
        NodePtr dedx = NodeFactory::createNode({m_dy, m_w}, {m_dx}, &convParams, NodeFactory::deDx3DNodeTypeName, "3D_dedx");
        ASSERT_TRUE(GraphEditor::addNode(g, dedx));
        NodePtr dedw = NodeFactory::createNode({m_dy, m_x}, {m_dw}, &convParams, NodeFactory::deDw3DNodeTypeName, "3D_dedw");
        ASSERT_TRUE(GraphEditor::addNode(g, dedw));

        // Run pipeline managemnt
        ASSERT_TRUE(gaudi2::loadTpcKernels(g));
        ASSERT_TRUE(sliceGraphForPipeline(g));

        // Validate the 2 nodes were bundled together or separately, as expected
        validateBundles(g);
    }

protected:

    synConvolution3DParams createConvParams() const
    {
        synConvolution3DParams convParams {};
        convParams.kernel[CONV_KERNEL_WIDTH]  = GetParam().kernelSize;
        convParams.kernel[CONV_KERNEL_HEIGHT] = GetParam().kernelSize;
        convParams.kernel[CONV_KERNEL_DEPTH]  = GetParam().kernelSize;
        convParams.stride[CONV_STRIDE_HEIGHT] = GetParam().stride;
        convParams.stride[CONV_STRIDE_WIDTH]  = GetParam().stride;
        convParams.stride[CONV_STRIDE_DEPTH]  = GetParam().stride;
        convParams.padding[CONV_PAD_LEFT]     = GetParam().pad;
        convParams.padding[CONV_PAD_RIGHT]    = GetParam().pad;
        convParams.padding[CONV_PAD_TOP]      = GetParam().pad;
        convParams.padding[CONV_PAD_BOTTOM]   = GetParam().pad;
        convParams.padding[CONV_PAD_FRONT]    = GetParam().pad;
        convParams.padding[CONV_PAD_BACK]     = GetParam().pad;
        return convParams;
    }

    void createTensors()
    {
        m_dy = createTensor(GetParam().ySizes, syn_type_bf16);
        m_w  = createTensor(GetParam().wSizes, syn_type_bf16);
        m_dx = createTensor(GetParam().xSizes, syn_type_bf16);
        m_x  = createTensor(GetParam().xSizes, syn_type_bf16);
        m_dw = createTensor(GetParam().wSizes, syn_type_bf16);
    }

    void validateBundles(HabanaGraph& g) const
    {
        std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
        if (GetParam().bundleTogether)
        {
            // Assert the 2 nodes are bundled together - there is 1 bundle
            ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
            // Assert the MME inputs are in SRAM for dy
            ASSERT_TRUE(isMmeInputIdxInSram(g, TENSOR_DEDY));
        }
        else
        {
            // Assert the 2 nodes are bundled separately - there are 2 bundles
            ASSERT_EQ(nodeCountersPerBundleId->size(), 2);
            // Assert the MME inputs are in SRAM for any of the inputs
            ASSERT_TRUE(isMmeInputIdxInSram(g, std::nullopt));
        }
    }

    TensorPtr m_dy;
    TensorPtr m_dx;
    TensorPtr m_dw;
    TensorPtr m_w;
    TensorPtr m_x;
};

TEST_P(DedwDedxSharedInputTest, bundle_test)
{
    runSingleTest();
}

// Check that dedx and dedw don't bundle together if dedw prefers to place X in SRAM
// The nodes are taken from UNET3D
INSTANTIATE_TEST_SUITE_P(separate_bundles_3d,
                         DedwDedxSharedInputTest,
                         ::testing::Values(DedwDedxSharedInputTestParams {{64, 64, 64, 64, 7}, {128, 64, 3, 3, 3}, {128, 32, 32, 32, 7}, 3, 2, 1, false},
                                           DedwDedxSharedInputTestParams {{32, 128, 128, 128, 7}, {64, 32, 3, 3, 3}, {64, 64, 64, 64, 7}, 3, 2, 1, false}));

// Check that dedx and dedw bundle together, and dedw doesn't prefer to place X in SRAM
// The nodes are taken from UNET3D, and perf was better when bundled together.
INSTANTIATE_TEST_SUITE_P(bundle_together_3d,
                         DedwDedxSharedInputTest,
                         ::testing::Values(DedwDedxSharedInputTestParams {{128, 32, 32, 32, 7}, {256, 128, 3, 3, 3}, {256, 16, 16, 16, 7}, 3, 2, 1, true},
                                           DedwDedxSharedInputTestParams {{64, 64, 64, 64, 7}, {64, 64, 3, 3, 3}, {64, 64, 64, 64, 7}, 3, 1, 1, true},
                                           DedwDedxSharedInputTestParams {{128, 64, 64, 64, 7}, {128, 64, 3, 3, 3}, {64, 64, 64, 64, 7}, 3, 1, 1, true},
                                           DedwDedxSharedInputTestParams {{32, 128, 128, 128, 7}, {3, 32, 1, 1, 1}, {3, 128, 128, 128, 7}, 1, 1, 0, true}));

struct CLAlignmentCalcTestParams
{
    TSize       singleLineAlignmentBytes;
    unsigned    numCacheLinesForDim0;
    TSize       nonSlicedDimsSize;
    unsigned    slicedDim;
    TSize       slicedDimSize;
    synDataType dataType;
};

class CLAlignmentCalcTest
: public PipelineManagementTest
, public testing::WithParamInterface<CLAlignmentCalcTestParams>
{
public:
    void runSingleTest()
    {
        const unsigned sizeToAlignTo = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
        // CL size aligned dim - alignment
        TSize dim0Size = GetParam().numCacheLinesForDim0 * sizeToAlignTo - GetParam().singleLineAlignmentBytes;
        std::vector<TSize> sizes = {dim0Size, GetParam().nonSlicedDimsSize, GetParam().nonSlicedDimsSize};
        TensorPtr          t     = createTensor(sizes, GetParam().dataType);

        std::map<unsigned, TSize> sizePerDim    = {{GetParam().slicedDim, GetParam().slicedDimSize}};
        TStride                   alignmentSize = SlicedOperandUtils::getSliceAlignmentSize(t, sizePerDim);
        unsigned elementSize = dataTypeSizeInBytes(GetParam().dataType);
        // Assuming 1 dim is sliced and the other is not sliced, and both non sliced sizes are the same
        TStride expectedSize =
            GetParam().singleLineAlignmentBytes * GetParam().slicedDimSize * GetParam().nonSlicedDimsSize * elementSize;
        ASSERT_EQ(alignmentSize, expectedSize);
    }
};

TEST_F(PipelineManagementTest, gemm_with_2_sliced_producers_chains)
{
    // The bundle is taken from GPT3 FP8 model: GEMM node + TPC producer for each operand.
    // Both operands can fit SRAM when operand B is sliced (operand B can't fit to SRAM unsliced).
    // Operand A is expected to be sliced as well to improve the TPC-MME pipeline.

    setGlobalConfForTest(GCFG_ENABLE_SLICING_BOTH_PRODUCER_CHAINS, "true");
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // GEMM
    synGEMMParams gemmParams {false, true};
    TensorPtr     a    = createTensor({6144, 2048}, syn_type_fp8_152, false);
    TensorPtr     b    = createTensor({6144, 12288}, syn_type_fp8_152, false);
    TensorPtr     out  = createTensor({12288, 2048}, syn_type_bf16, true);
    NodePtr       gemm = NodeFactory::createNode({a, b}, {out}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(gemm);
    ASSERT_TRUE(GraphEditor::addNode(g, gemm));

    // Producer chain for operand A: cast -> reshape
    TensorPtr reshapeIn = createTensor({6144, 1, 2048}, syn_type_fp8_152, false);
    NodePtr   reshape = NodeFactory::createNode({reshapeIn}, {a}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape);
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));
    TensorPtr castAIn = createTensor({6144, 1, 2048}, syn_type_bf16, true);
    NodePtr   castA   = NodeFactory::createGenericTPCNode({castAIn}, {reshapeIn}, nullptr, "cast_bf16_to_f8", "castA");
    ASSERT_TRUE(castA);
    ASSERT_TRUE(GraphEditor::addNode(g, castA));

    // Producer chain for operand B: cast
    TensorPtr castBIn = createTensor({6144, 12288}, syn_type_bf16, true);
    NodePtr   castB   = NodeFactory::createGenericTPCNode({castBIn}, {b}, nullptr, "cast_bf16_to_f8", "castB");
    ASSERT_TRUE(castB);
    ASSERT_TRUE(GraphEditor::addNode(g, castB));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));

    SlicingBrain   dummyBrain(g);  // Required to intialize SlicingBrain::knobs - for SRAM capacity
    BPGraphContext bpgCtxt(g);
    const auto&    bundles = MantaRayBundlizer(g).generateBundles();
    ASSERT_EQ(bundles.size(), 1);
    PipelineBundlePtr bundle = bundles.front().first;
    ASSERT_TRUE(bundle);
    ASSERT_EQ(bundle->getNodes().size(), g.getNodes().size());  // All nodes should be bundled together
    BundleSolverPtr solver = bundles.front().second;
    ASSERT_TRUE(solver);
    BundleStrategyPtr solution = solver->solveBundle();
    ASSERT_TRUE(solution);

    // Both GEMM's inputs are expected to be sliced on the non-common dim
    const auto& slicedOpA = solution->getSlicingData().getSlicedOperand(gemm->getInput(0));
    const auto& slicedOpB = solution->getSlicingData().getSlicedOperand(gemm->getInput(1));
    const auto& slicedOut = solution->getSlicingData().getSlicedOperand(gemm->getOutput(0));
    ASSERT_TRUE(slicedOpA);
    ASSERT_TRUE(slicedOpB);
    ASSERT_TRUE(slicedOut);
    ASSERT_FALSE(SlicedOperandUtils::isTriviallySliced(slicedOpA));
    ASSERT_FALSE(SlicedOperandUtils::isTriviallySliced(slicedOpB));
    MmeDimController dimController(gemm);
    ASSERT_EQ(slicedOpA->chunkDimensions[dimController.nonCommonDimOperandA().front()],
              slicedOut->chunkDimensions[dimController.heightOutput().front()]);
    ASSERT_EQ(slicedOpB->chunkDimensions[dimController.nonCommonDimOperandB().front()],
              slicedOut->chunkDimensions[dimController.widthOutput().front()]);

    for (const auto& slicedOperand : solution->getSlicingData().getSlicedOperands())
    {
        const auto numSlices = SlicedOperandUtils::nofSlices(slicedOperand);
        ASSERT_GT(numSlices, 1);  // All operands are expected to be sliced
        if (slicedOperand->originalTensor == gemm->getOutput(0) ||
            slicedOperand->originalTensor == castA->getInput(0) || slicedOperand->originalTensor == castB->getInput(0))
        {
            ASSERT_FALSE(slicedOperand->resideInSRAM);  // BPTs should be placed in HBM
        }
        else
        {
            ASSERT_TRUE(slicedOperand->resideInSRAM);
            if (slicedOperand->originalTensor == castB->getOutput(0))
            {
                // Double buffer should be set to operand B producer chain
                ASSERT_EQ(slicedOperand->numOfBuffers, 2);
            }
            else
            {
                // Operand A producers chain should be placed concurrently in SRAM
                ASSERT_EQ(slicedOperand->numOfBuffers, numSlices);
            }
        }
    }

    solver->fillBundleSolution(solution);
    BundleSlicer::sliceBundle(*bundle, g);

    for (const auto& n : g.getNodes())
    {
        ASSERT_TRUE(n->getNodeType() != Node::TYPE_MEMCOPY);  // No SRAM<->HBM DMA copies are expected
    }
}

TEST_F(PipelineManagementTest, bgemm_with_spatial_and_batch_dim_slicing_non_shared_expansion_not_allowed)
{
    // The bundle is taken from stable diffusion model: BGEMM node chooses to slice opA on dims 2 and 1, and since dim 1
    // is not mapped to any dim in opB, the expansion should consider only dim2, and not include any producers
    setGlobalConfForTest(GCFG_ENABLE_SLICING_BOTH_PRODUCER_CHAINS, "true");
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // BGEMM
    synGEMMParams bgemmParams {false, false};
    TensorPtr     a    = createTensor({4096, 4096, 5}, syn_type_bf16, true);
    TensorPtr     b    = createTensor({64, 4096, 5}, syn_type_bf16, false);
    TensorPtr     out  = createTensor({4096, 4096, 5}, syn_type_bf16, false);
    NodePtr       bgemm = NodeFactory::createNode({a, b}, {out}, &bgemmParams, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(bgemm);
    ASSERT_TRUE(GraphEditor::addNode(g, bgemm));

    // Producer chain for operand B: cast -> reshape -> transpose. Reshape node in0 dim 1 is mapped  to reshape out dim2, and since
    // dim1 is inner dim, it can't be added to the producer chain
    TensorPtr reshapeIn = createTensor({64, 5, 4096}, syn_type_bf16, false);
    NodePtr   reshape = NodeFactory::createNode({reshapeIn}, {b}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape);
    ASSERT_TRUE(GraphEditor::addNode(g, reshape));
    TensorPtr castBIn = createTensor({64, 5, 4096}, syn_type_single, true);
    NodePtr   castB   = NodeFactory::createGenericTPCNode({castBIn}, {reshapeIn}, nullptr, "cast_f32_to_bf16", "cast");
    ASSERT_TRUE(castB);
    ASSERT_TRUE(GraphEditor::addNode(g, castB));

    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    SlicingBrain   dummyBrain(g);  // Required to intialize SlicingBrain::knobs - for SRAM capacity
    BPGraphContext bpgCtxt(g);
    const auto&    bundles = MantaRayBundlizer(g).generateBundles();
    ASSERT_EQ(bundles.size(), 1);
    PipelineBundlePtr bundle = bundles.front().first;
    ASSERT_EQ(bundle->getNodes().size(), 1) << "Expecting single mme bundle";
    ASSERT_TRUE(bgemm->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_FALSE(reshape->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_FALSE(castB->getNodeAnnotation().bundleInfo.is_set());

    BundleSolverPtr solver = bundles.front().second;
    ASSERT_TRUE(solver);
    BundleStrategyPtr solution = solver->solveBundle();
    ASSERT_TRUE(solution);

    // Bgemm in0 is expected to be slicd on dim 2 and 1, and in1 is expected to be sliced on dim 2 only
    const auto& slicedOpA = solution->getSlicingData().getSlicedOperand(bgemm->getInput(0));
    const auto& slicedOpB = solution->getSlicingData().getSlicedOperand(bgemm->getInput(1));
    const auto& slicedOut = solution->getSlicingData().getSlicedOperand(bgemm->getOutput(0));
    ASSERT_TRUE(slicedOpA);
    ASSERT_TRUE(slicedOpB);
    ASSERT_TRUE(slicedOut);
    MmeDimController dimController(bgemm);
    ASSERT_TRUE(SlicedOperandUtils::isSlicedOnDimension(slicedOpA, dimController.nonCommonDimOperandA().back())) << "Expecting A to be sliced on non common dim";
    ASSERT_TRUE(SlicedOperandUtils::isSlicedOnDimension(slicedOpA, dimController.batchDim().back())) << "Expecting A to be sliced on batch dim";
    ASSERT_TRUE(SlicedOperandUtils::isSlicedOnDimension(slicedOpB, dimController.batchDim().back())) << "Expecting B to be sliced on batch dim";
}

TEST_P(CLAlignmentCalcTest, test_alignment_size_calc)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(cl_alignment_for_slice,
                         CLAlignmentCalcTest,
                         ::testing::Values(CLAlignmentCalcTestParams {20, 3, 1024, 2, 10, syn_type_bf16},
                                           CLAlignmentCalcTestParams {15, 1, 960, 1, 19, syn_type_single},
                                           CLAlignmentCalcTestParams {37, 2, 555, 2, 64, syn_type_bf16}));

// Check that masks of masked bgemm in sram.
TEST_F(PipelineManagementTest, masked_bgemm_masks_in_sram)
{
    Gaudi2Graph g;

    // masked batch gemm
    const synGEMMParams params(true, false);
    const TSize         internalBatchSize = 12;
    const TensorPtr     a                 = createTensor({2048, 128, internalBatchSize, 1}, syn_type_bf16);
    const TensorPtr     b                 = createTensor({2048, 128, internalBatchSize, 1}, syn_type_bf16);
    const TensorPtr     maskA             = createTensor({2048, 6, 1, 1}, syn_type_bf16);
    const TensorPtr     maskB             = createTensor({2048, 6, 1, 1}, syn_type_bf16);
    const TensorPtr     maskedBGemmOut    = createTensor({2048, 2048, internalBatchSize, 1}, syn_type_bf16);
    const NodePtr       maskedBGemm       = NodeFactory::createNode({a, b, maskA, maskB},
                                                                    {maskedBGemmOut},
                                                                    &params,
                                                                    NodeFactory::maskedBatchGemmNodeTypeName,
                                                                    "MaskedBGemm");
    ASSERT_TRUE(GraphEditor::addNode(g, maskedBGemm));

    ASSERT_TRUE(sliceGraphForPipeline(g));
    // check the node is bundled
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME only
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeOnly));

    for (const auto& n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
        {
            ASSERT_TRUE(n->getInput(TENSOR_IFM)->inSram() || n->getInput(TENSOR_WEIGHT)->inSram());
            ASSERT_TRUE(n->getInput(TENSOR_AUX_BGEMM_MASK_A)->inSram());
            ASSERT_TRUE(n->getInput(TENSOR_AUX_BGEMM_MASK_B)->inSram());
        }
    }
}

// Check that in bundle contains masked bgemm and tpc consumer the masks are in sram.
TEST_F(PipelineManagementTest, masked_bgemm_with_tpc_consumer_masks_in_sram)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    // masked batch gemm
    const synGEMMParams params(true, false);
    const TSize         internalBatchSize = 12;
    const TensorPtr     a                 = createTensor({2048, 128, internalBatchSize, 1}, syn_type_bf16);
    const TensorPtr     b                 = createTensor({2048, 128, internalBatchSize, 1}, syn_type_bf16);
    const TensorPtr     maskA             = createTensor({2048, 2048, 1, 1}, syn_type_bf16);
    const TensorPtr     maskB             = createTensor({2048, 2048, 1, 1}, syn_type_bf16);
    const TensorPtr     maskedBGemmOut    = createTensor({2048, 2048, internalBatchSize, 1}, syn_type_bf16);
    const NodePtr       maskedBGemm       = NodeFactory::createNode({a, b, maskA, maskB},
                                                                    {maskedBGemmOut},
                                                                    &params,
                                                                    NodeFactory::maskedBatchGemmNodeTypeName,
                                                                    "MaskedBGemm");
    ASSERT_TRUE(GraphEditor::addNode(g, maskedBGemm));

    // relu consumer
    const TensorPtr reluOut = createTensor({2048, 2048, internalBatchSize, 1}, syn_type_bf16);
    const NodePtr   relu =
        NodeFactory::createGenericTPCNode({maskedBGemmOut}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));
    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // check the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));

    for (const auto& n : g.getExeSortedNodes())
    {
        if (n->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
        {
            ASSERT_TRUE(n->getInput(TENSOR_IFM)->inSram() || n->getInput(TENSOR_WEIGHT)->inSram());
            ASSERT_TRUE(n->getInput(TENSOR_AUX_BGEMM_MASK_A)->inSram());
            ASSERT_TRUE(n->getInput(TENSOR_AUX_BGEMM_MASK_B)->inSram());
            ASSERT_TRUE(n->getOutput(TENSOR_OFM)->inSram());
        }
    }
}

// Check that bgemm is sliced on 2 dims when the master operand is too big to sram
TEST_F(PipelineManagementTest, bgemm_with_tpc_producer_sliced_on_two_dims)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    constexpr unsigned internalBatchDim  = 2;
    constexpr unsigned internalBatchSize = 16;
    constexpr unsigned externalBatchDim  = 3;
    constexpr unsigned externalBatchSize = 10;

    // relu producer
    const std::vector<TSize> reluTensorsShape = {2048, 2048, internalBatchSize, externalBatchSize};
    const TensorPtr          reluIn  = createTensor(reluTensorsShape, syn_type_bf16);
    const TensorPtr          reluOut = createTensor(reluTensorsShape, syn_type_bf16);
    const NodePtr relu = NodeFactory::createGenericTPCNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    const synGEMMParams params(false, false);
    const TensorPtr     bGemmOut = createTensor({64, 2048, internalBatchSize, externalBatchSize}, syn_type_bf16);
    const TensorPtr     b        = createTensor({64, 2048, internalBatchSize, externalBatchSize}, syn_type_bf16);
    const NodePtr       bGemm =
        NodeFactory::createNode({reluOut, b}, {bGemmOut}, &params, NodeFactory::batchGemmNodeTypeName, "bGemm");
    ASSERT_TRUE(GraphEditor::addNode(g, bGemm));
    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // check the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    for (const auto& n : g.getNodes())
    {
        if (n->getNodeType() == Node::TYPE_BATCH_GEMM)
        {
            ASSERT_TRUE(isBGemmSlicedOnAllBatchDims(n, TENSOR_IFM, reluTensorsShape, {internalBatchDim, externalBatchDim}));
            ASSERT_TRUE(n->getInput(TENSOR_IFM)->inSram());
        }
    }
}

// Check that bgemms are sliced on 2 dims when the master operand is too big to sram
TEST_F(PipelineManagementTest, two_bgemm_with_tpc_producer_sliced_on_two_dims)
{
    Gaudi2Graph                g;
    CompilationHalReaderSetter compHalReaderSetter(&g);

    constexpr unsigned internalBatchDim  = 2;
    constexpr unsigned internalBatchSize = 16;
    constexpr unsigned externalBatchDim  = 3;
    constexpr unsigned externalBatchSize = 10;

    // relu producer
    const std::vector<TSize> reluTensorsShape = {2048, 2048, internalBatchSize, externalBatchSize};
    const TensorPtr          reluIn  = createTensor(reluTensorsShape, syn_type_bf16);
    const TensorPtr          reluOut = createTensor(reluTensorsShape, syn_type_bf16);
    const NodePtr relu = NodeFactory::createGenericTPCNode({reluIn}, {reluOut}, nullptr, "relu_fwd_bf16", "relu");
    ASSERT_TRUE(GraphEditor::addNode(g, relu));

    const synGEMMParams params1(false, false);
    const TensorPtr     bGemmOut1 = createTensor({64, 2048, internalBatchSize, externalBatchSize}, syn_type_bf16);
    const TensorPtr     b1        = createTensor({64, 2048, internalBatchSize, externalBatchSize}, syn_type_bf16);
    const NodePtr       bGemm1 =
        NodeFactory::createNode({reluOut, b1}, {bGemmOut1}, &params1, NodeFactory::batchGemmNodeTypeName, "bGemm1");
    ASSERT_TRUE(GraphEditor::addNode(g, bGemm1));

    const synGEMMParams params2(false, false);
    const TensorPtr     bGemmOut2 = createTensor({64, 2048, internalBatchSize, externalBatchSize}, syn_type_bf16);
    const TensorPtr     b2        = createTensor({64, 2048, internalBatchSize, externalBatchSize}, syn_type_bf16);
    const NodePtr       bGemm2 =
        NodeFactory::createNode({reluOut, b2}, {bGemmOut2}, &params2, NodeFactory::batchGemmNodeTypeName, "bGemm2");
    ASSERT_TRUE(GraphEditor::addNode(g, bGemm2));
    ASSERT_TRUE(gaudi2::loadTpcKernels(g));
    ASSERT_TRUE(sliceGraphForPipeline(g));

    // check the 2 nodes are bundled together
    std::shared_ptr<NodeCountersPerBundleMap> nodeCountersPerBundleId = getNodeCountersPerBundle(g);
    // assert there is a single bundle
    ASSERT_EQ(nodeCountersPerBundleId->size(), 1);
    // assert this bundle has MME and TPC
    ASSERT_TRUE(findBundleWithNodesTypes(nodeCountersPerBundleId, bundleIncludesMmeAndTpc));
    for (const auto& n : g.getNodes())
    {
        if (n->getNodeType() == Node::TYPE_BATCH_GEMM)
        {
            ASSERT_TRUE(isBGemmSlicedOnAllBatchDims(n, TENSOR_IFM, reluTensorsShape, {internalBatchDim, externalBatchDim}));
            ASSERT_TRUE(n->getInput(TENSOR_IFM)->inSram());
        }
    }
}