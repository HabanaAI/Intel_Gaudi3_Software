#include "code_generator.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "gaudi_graph.h"
#include "perf_lib_layer_params.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include <optional>

class ComplexGuidTest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_F(ComplexGuidTest, exclude_from_handle_tpc_rmw_kernels_pass)
{
    // Kernels used for complex GUIDs may require reducible memory via the legacy rmwMask.
    // The handleTpcRmwKernels pass should not handle these nodes.
    // This test checks that if a node is a user of a RMW section - the pass ignores it
    // and doesn't create a reduction node (TYPE_INTERNAL_REDUCTION).

    setGlobalConfForTest(GCFG_MAX_RMW_TENSOR_BYTES, "20");

    GaudiGraph gaudiGraph;

    std::vector<TSize> inSizes  = {1000, 100};
    std::vector<TSize> outSizes = {1, 100};
    const auto         rmwSectionId = gaudiGraph.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);

    pTensor             reduceSumIn(new Tensor(inSizes.size(), inSizes.data(), syn_type_float));
    synMemoryDescriptor reduceSumInDesc(true);
    reduceSumIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    reduceSumIn->setMemoryDescriptor(reduceSumInDesc);

    pTensor memsetOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    memsetOut->setTensorInSram();
    memsetOut->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSectionId);
    memsetOut->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pTensor reduceSumOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    reduceSumOut->setTensorInSram();
    reduceSumOut->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSectionId);
    reduceSumOut->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pTensor             finalOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    synMemoryDescriptor finalOutDesc(true);
    finalOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    finalOut->setMemoryDescriptor(finalOutDesc);

    pNode                memset = NodeFactory::createNode({}, {memsetOut}, nullptr, "memset", "memset");
    ns_Reduction::Params params = {0};
    pNode                reduceSum =
        NodeFactory::createNode({reduceSumIn}, {reduceSumOut}, &params, "reduce_sum_square_fwd_f32", "reduce_sum");
    pNode memcopy = NodeFactory::createNode({reduceSumOut}, {finalOut}, nullptr, "memcpy", "memcpy");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memset));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, reduceSum));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopy));

    gaudiGraph.addControlDependency({memset}, {reduceSum});

    ASSERT_TRUE(gaudiGraph.compile()) << "failed to compile graph";

    ASSERT_TRUE(reduceSumOut->inSram());

    bool isInternalReductionAdded = false;

    // Make sure internal reduction is not added
    // Also, make sure the 3 nodes are in the same bundle, since they are users of the same rmw section.
    std::optional<unsigned> bundleIdx;
    for (const auto& node : gaudiGraph.getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            isInternalReductionAdded = true;
        }
        ASSERT_TRUE(node->getNodeAnnotation().bundleInfo.is_set());
        if (!bundleIdx.has_value())
        {
            bundleIdx.emplace(node->getNodeAnnotation().bundleInfo->bundleIndex);
        }
        else
        {
            ASSERT_EQ(bundleIdx, node->getNodeAnnotation().bundleInfo->bundleIndex);
        }
    }

    ASSERT_FALSE(isInternalReductionAdded);

    ASSERT_GE(gaudiGraph.getExeSortedNodes().size(), 3);
}

TEST_F(ComplexGuidTest, exclude_from_memset_node_output_pass)
{
    // Kernels used for complex GUIDs may require to memset their output using the legacy mode.
    // The memsetNodeOutput pass should not handle these nodes.
    // This test checks that if a node is a user of a RMW section - the pass ignores it
    // and doesn't create a reduction node (TYPE_INTERNAL_REDUCTION).

    GaudiGraph gaudiGraph;

    std::vector<TSize> inSizes  = {10, 1, 1, 1};
    std::vector<TSize> outSizes = {15, 1, 1, 1};

    const auto rmwSectionId = gaudiGraph.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);

    pTensor             gradIn(new Tensor(inSizes.size(), inSizes.data(), syn_type_float));
    synMemoryDescriptor gradDesc(true);
    gradIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    gradIn->setMemoryDescriptor(gradDesc);

    pTensor             indicesIn(new Tensor(inSizes.size(), inSizes.data(), syn_type_int32));
    synMemoryDescriptor indicesDesc(true);
    indicesIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    indicesIn->setMemoryDescriptor(indicesDesc);

    pTensor sortOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    sortOut->setTensorInSram();
    sortOut->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSectionId);
    sortOut->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pTensor             finalOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    synMemoryDescriptor finalOutDesc(true);
    finalOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    finalOut->setMemoryDescriptor(finalOutDesc);

    ns_SortBwd::Params params;
    params.axis = 0;
    pNode sort = NodeFactory::createGenericTPCNode({gradIn, indicesIn}, {sortOut}, &params, "sort_bwd_f32", "sortNode");
    pNode memcopy = NodeFactory::createNode({sortOut}, {finalOut}, nullptr, "memcpy", "memcpy");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, sort));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopy));

    ASSERT_TRUE(gaudiGraph.compile()) << "failed to compile graph";

    ASSERT_TRUE(sortOut->inSram());

    bool isInternalReductionAdded = false;

    for (const auto& node : gaudiGraph.getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            isInternalReductionAdded = true;
        }
    }

    ASSERT_FALSE(isInternalReductionAdded);
}

TEST_F(ComplexGuidTest, bundle_complex_guid)
{
    // To prevent rmw section users from mixing with other bundles, rmw section users should be bundled together.
    // This test verifies that the complex guid bundles are created as expected.

    GaudiGraph                      gaudiGraph;
    std::vector<TSize>              sizes         = {16, 16};
    const auto rmwSection1Id = gaudiGraph.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);
    const auto rmwSection2Id = gaudiGraph.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);

    // We have the following node sequence:
    // [in] -> memcpy1 -> relu1 -> gemm -> relu2 -> memcpy2 -> [out]
    // relu1 input belong to first rmw section
    // relu2 output belong to second rmw section

    pTensor             in(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    synMemoryDescriptor inDesc(true);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    in->setMemoryDescriptor(inDesc);

    pTensor relu1In(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    relu1In->setTensorInSram();
    relu1In->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSection1Id);
    relu1In->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pNode memcpy1 = NodeFactory::createNode({in}, {relu1In}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy1");
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcpy1));

    pTensor relu1Out(new Tensor(sizes.size(), sizes.data(), syn_type_float));

    pNode relu1 = NodeFactory::createNode({relu1In}, {relu1Out}, nullptr, "relu_fwd_f32", "relu1");
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, relu1));

    synGEMMParams       gemmParams {};
    pTensor             gemmIn(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    synMemoryDescriptor gemmInDesc(true);
    gemmIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    gemmIn->setMemoryDescriptor(gemmInDesc);
    pTensor gemmOut(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    pNode   gemm =
        NodeFactory::createNode({relu1Out, gemmIn}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, gemm));

    pTensor relu2Out(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    relu2Out->setTensorInSram();
    relu2Out->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSection2Id);
    relu2Out->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pNode relu2 = NodeFactory::createNode({gemmOut}, {relu2Out}, nullptr, "relu_fwd_f32", "relu2");
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, relu2));

    pTensor             out(new Tensor(sizes.size(), sizes.data(), syn_type_float));
    synMemoryDescriptor outDesc(true);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    out->setMemoryDescriptor(outDesc);

    pNode memcpy2 = NodeFactory::createNode({relu2Out}, {out}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy2");
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcpy2));

    ASSERT_TRUE(gaudiGraph.compile()) << "failed to compile graph";

    ASSERT_TRUE(relu1In->inSram());
    ASSERT_TRUE(relu2Out->inSram());

    // Expected bundling status:
    // First bundle : memcpy1 + relu1 (users of the first rmw section).
    // Second bundle : gemm
    // Third bundle : relu2 + memcpy2 (users of the second rmw section).

    std::string nodeNameBase;
    // Check whether or not, node's name is based on a given string
    const auto isSubstringOfNodeName = [&nodeNameBase](const NodePtr& nodePtr) {
        const auto nodeName = nodePtr->getNodeName();
        return nodeName.find(nodeNameBase) != std::string::npos;
    };

    const auto exeSortedNodes = gaudiGraph.getExeSortedNodes();

    nodeNameBase              = "memcpy1";
    const auto memCpy1NodeItr = std::find_if(exeSortedNodes.begin(), exeSortedNodes.end(), isSubstringOfNodeName);
    ASSERT_NE(memCpy1NodeItr, exeSortedNodes.end()) << "Expected memcpy node in the graph, couldn't find it";

    nodeNameBase            = "relu1";
    const auto relu1NodeItr = std::find_if(exeSortedNodes.begin(), exeSortedNodes.end(), isSubstringOfNodeName);
    ASSERT_NE(relu1NodeItr, exeSortedNodes.end()) << "Expected relu node in the graph, couldn't find it";

    // Assure relu1 and memcpy1 share the same bundle
    ASSERT_TRUE((*memCpy1NodeItr)->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE((*relu1NodeItr)->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_EQ((*memCpy1NodeItr)->getNodeAnnotation().bundleInfo->bundleIndex,
              (*relu1NodeItr)->getNodeAnnotation().bundleInfo->bundleIndex);
    const auto bundle1Idx = (*memCpy1NodeItr)->getNodeAnnotation().bundleInfo->bundleIndex;

    nodeNameBase           = "gemm";
    const auto gemmNodeItr = std::find_if(exeSortedNodes.begin(), exeSortedNodes.end(), isSubstringOfNodeName);
    ASSERT_NE(gemmNodeItr, exeSortedNodes.end()) << "Expected gemm node in the graph, couldn't find it";
    ASSERT_TRUE((*gemmNodeItr)->getNodeAnnotation().bundleInfo.is_set());
    const auto bundle2Idx = (*gemmNodeItr)->getNodeAnnotation().bundleInfo->bundleIndex;

    nodeNameBase              = "memcpy2";
    const auto memCpy2NodeItr = std::find_if(exeSortedNodes.begin(), exeSortedNodes.end(), isSubstringOfNodeName);
    ASSERT_NE(memCpy2NodeItr, exeSortedNodes.end()) << "Expected second memcpy node in the graph, couldn't find it";

    nodeNameBase            = "relu2";
    const auto relu2NodeItr = std::find_if(exeSortedNodes.begin(), exeSortedNodes.end(), isSubstringOfNodeName);
    ASSERT_NE(relu2NodeItr, exeSortedNodes.end()) << "Expected second relu node in the graph, couldn't find it";

    // Assure relu2 and memcpy2 share the same bundle
    ASSERT_TRUE((*memCpy2NodeItr)->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE((*relu2NodeItr)->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_EQ((*memCpy2NodeItr)->getNodeAnnotation().bundleInfo->bundleIndex,
              (*relu2NodeItr)->getNodeAnnotation().bundleInfo->bundleIndex);
    const auto bundle3Idx = (*memCpy2NodeItr)->getNodeAnnotation().bundleInfo->bundleIndex;

    // Assure three different bundles
    ASSERT_NE(bundle1Idx, bundle2Idx);
    ASSERT_NE(bundle1Idx, bundle3Idx);
    ASSERT_NE(bundle2Idx, bundle3Idx);
}

TEST_F(ComplexGuidTest, DISABLED_crop_and_resize_rmw)  // Disabled until SW-86610 resolved
{
    GaudiGraph gaudiGraph;

    const TSize        n = 512, c = 4, h = 8, w = 8;
    std::vector<TSize> gradsSizes   = {c, w, h, n};
    std::vector<TSize> boxesSizes   = {4, n};
    std::vector<TSize> indicesSizes = {n};
    std::vector<TSize> outSizes     = {c, 32, 32, 1};

    const auto rmwSectionId = gaudiGraph.getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);

    pTensor             grads(new Tensor(gradsSizes.size(), gradsSizes.data(), syn_type_float));
    synMemoryDescriptor gradsDesc(true);
    grads->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    grads->setMemoryDescriptor(gradsDesc);

    pTensor             boxes(new Tensor(boxesSizes.size(), boxesSizes.data(), syn_type_float));
    synMemoryDescriptor boxesDesc(true);
    boxes->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    boxes->setMemoryDescriptor(boxesDesc);

    pTensor             indices(new Tensor(indicesSizes.size(), indicesSizes.data(), syn_type_int32));
    synMemoryDescriptor indicesDesc(true);
    indices->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    indices->setMemoryDescriptor(indicesDesc);

    pTensor imageSize(new Tensor(outSizes.size(), outSizes.data(), syn_type_uint32));
    //imageSize->setShapeTensor(INPUT_DESCRIBING_SHAPE_TENSOR); <- Replace with H2D when possible

    pTensor shapeSplitOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_uint32));
    shapeSplitOut->setShapeTensor(OUTPUT_DESCRIBING_SHAPE_TENSOR);

    pTensor cropAndResizeOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    cropAndResizeOut->setTensorInSram();
    cropAndResizeOut->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSectionId);
    cropAndResizeOut->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pTensor memsetOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    memsetOut->setTensorInSram();
    memsetOut->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(rmwSectionId);
    memsetOut->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.set(0);

    pTensor             finalOut(new Tensor(outSizes.size(), outSizes.data(), syn_type_float));
    synMemoryDescriptor finalOutDesc(true);
    finalOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    finalOut->setMemoryDescriptor(finalOutDesc);

    pNode memset = NodeFactory::createNode({}, {memsetOut}, nullptr, "memset", "memset");
    ns_CropAndResizeBwdKernel::ParamsSplitedOutput cropAndResizeParams;
    std::memset(&cropAndResizeParams, 0, sizeof(ns_CropAndResizeBwdKernel::ParamsSplitedOutput));
    cropAndResizeParams.mode         = CropAndResizeMode_t::CROP_AND_RESIZE_MODE_BILINEAR;
    cropAndResizeParams.isValidCount = false;
    pNode cropAndResize              = NodeFactory::createNode({grads, boxes, indices, imageSize, shapeSplitOut},
                                                               {cropAndResizeOut},
                                                  &cropAndResizeParams,
                                                  "crop_and_resize_bwd_f32",
                                                  "crop_and_resize");
    pNode memcopy = NodeFactory::createNode({cropAndResizeOut}, {finalOut}, nullptr, "memcpy", "memcpy");

    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memset));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, cropAndResize));
    ASSERT_TRUE(GraphEditor::addNode(gaudiGraph, memcopy));

    gaudiGraph.addControlDependency({memset}, {cropAndResize});

    ASSERT_TRUE(gaudiGraph.compile()) << "failed to compile graph";

    ASSERT_TRUE(cropAndResizeOut->inSram());

    // The 3 nodes are expected to be in the same bundle since they are users of the same rmw section.
    ASSERT_TRUE(memset->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(cropAndResize->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(memcopy->getNodeAnnotation().bundleInfo.is_set());
    auto bundleIdx = memset->getNodeAnnotation().bundleInfo->bundleIndex;
    ASSERT_EQ(bundleIdx, cropAndResize->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_EQ(bundleIdx, memcopy->getNodeAnnotation().bundleInfo->bundleIndex);
}