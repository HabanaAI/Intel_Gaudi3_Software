#include "graph_optimizer_test.h"
#include "graph_compiler/passes/complex_guid_extractor.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "test_utils.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "generic_graph_test.h"
#include "ir_translation/graph_comparator.hpp"
#include "infer_shape_node.h"

class ComplexGuidExtractorDummyTest : public GenericGraphTest
{
public:
    static void SetUpTestSuite()
    {
        // create folder for build temp files and build the dummy lib
        int ret = system("cd $SYNAPSE_ROOT/tests/dummyComplexGuid/ && mkdir -p .build"
                         "&& cd .build && cmake .. && make");
        LOG_DEBUG(GO_TEST, "dummy lib compile return val - {}", ret);
    }

    void SetUp() override
    {
        GTEST_SKIP() << "Not running dummy CGUID tests";
        initConfigAndKernelDB(ComplexGUIDExtractorModeEnabledDummy);
        GenericGraphTest::SetUp();
        m_prev_mode = GCFG_COMPLEX_GUID_EXTRACTOR_MODE.value();
        std::shared_ptr<HalReader> halReader;
        halReader = GaudiHalReader::instance(m_graph->getDeviceType());
        CompilationHalReader::setHalReader(halReader);
    }

    void TearDown() override
    {
        resetConfigAndKernelDB(m_prev_mode);
        GenericGraphTest::TearDown();
    }

    static void TearDownTestSuite()
    {
        // remove dummy lib and temp files
        int ret = system("rm $BUILD_ROOT_LATEST/libdummyComplexGuid.so &&"
                         "cd $SYNAPSE_ROOT/tests/dummyComplexGuid/ && rm -r .build/");
        LOG_DEBUG(GO_TEST, "dummy lib remove return val - {}", ret);
    }

    void initConfigAndKernelDB(uint64_t mode)
    {
        GCFG_COMPLEX_GUID_EXTRACTOR_MODE.setValue(mode);
        GCFG_COMPLEX_GUID_LIB_NAME.setValue(dummyLibName);
    }

    NodePtr initNodeAndTensors(const char* guid, bool isOutputRMW = false)
    {
        const TSize C = 3;
        const TSize W = 3;
        const TSize H = 3;
        const TSize N = 1;

        char in1[C * W * H * N];

        const TSize sizes[] = {C, W, H, N};

        uint64_t            memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2;
        synMemoryDescriptor persistentMemoryDesc(true);

        TensorPtr x = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
        x->setName("x");
        x->setDramOffset(0x7000);
        x->setMemorySectionID(memSecId++);
        x->setMemoryDescriptor(persistentMemoryDesc);

        TensorPtr y = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
        y->setName("y");
        y->setDramOffset(0x8000);
        y->setMemorySectionID(memSecId++);
        y->setMemoryDescriptor(persistentMemoryDesc);

        TensorPtr out = TensorPtr(new Tensor(4U, sizes, syn_type_float, reinterpret_cast<char*>(in1)));
        out->setName("out");
        if (isOutputRMW)
        {
            out->setTensorInSram();
            auto& nonPersistentSectionInfo = out->getTensorAnnotation().nonPersistentSectionInfo;
            nonPersistentSectionInfo.sectionId.set(++memSecId);
            nonPersistentSectionInfo.offsetFromBase.set(0x1000);
        }
        else  // persistent
        {
            out->setDramOffset(0x9000);
            out->setMemorySectionID(memSecId++);
            out->setMemoryDescriptor(persistentMemoryDesc);
        }
        NodePtr complexAddNode = NodeFactory::createGenericTPCNode({x, y}, {out}, nullptr, guid, "complexGuidNode");
        return complexAddNode;
    }

    void testShapeManipulationNodes(ComplexGUIDType type, synDeviceType deviceType)
    {
        const int dummyShapeInferenceUniqueId = 12345;
        // A shapeManipulationNode with isShapeManipulation set to true, which will be replaced with inferShapeNode
        NodePtr shapeManipulationNode    = initNodeAndTensors("logical_shape_manipulation");
        // shapeManipulationNode as a TPCNode, to be able to set multiSifInfo
        auto    shapeManipulationTPCNode = dynamic_cast<TPCNode*>(shapeManipulationNode.get());
        ASSERT_TRUE(shapeManipulationTPCNode != nullptr);
        // output has to be a shape tensor
        shapeManipulationTPCNode->getOutput(0)->setShapeTensor(SHAPE_TENSOR);
        auto inputs = shapeManipulationTPCNode->getInputs();
        auto outputs = shapeManipulationTPCNode->getOutputs();
        // Add shapeManipulationNode to graph, and execute functional extraction pass
        ASSERT_TRUE(GraphEditor::addNode(*m_graph, shapeManipulationNode));
        if (type == FUNCTIONAL_COMPLEX_GUID)
        {
            ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
        }
        else
        {
            ASSERT_TRUE(extractPerformanceComplexGuidNodes(*m_graph));
        }
        // check it was replaced with a single shapeManipulationNode
        auto nodes = m_graph->getExeSortedNodes();
        ASSERT_EQ(nodes.size(), 1);
        // check shapeManipulationNode is inferShape shapeManipulationNode
        auto inferShapeNode = dynamic_cast<InferShapeNode*>((*nodes.begin()).get());
        ASSERT_TRUE(inferShapeNode != nullptr);
        // check inputs/outputs were copied
        ASSERT_EQ(inferShapeNode->getInputs(), inputs);
        ASSERT_EQ(inferShapeNode->getOutputs(), outputs);
        // Check if the guid of fuser node was saved in params
        auto rawData = inferShapeNode->getParamsRawData();
        auto params = reinterpret_cast<InferShapeParams*>(rawData.data());
        ASSERT_EQ(strcmp(params->sifGUID, "logical_shape_manipulation"), 0);
        // isTpc should always true, and it has a single sif.
        ASSERT_TRUE(params->isTpc);
        ASSERT_EQ(params->multiSifInfo, nullptr);
        // Get sif id for fuser node from kernel DB, and check it is the value that was stored in dummy CGUID
        tpc_lib_api::UniqueShapeInferenceHash sifID;
        KernelDB::instance().GetKernelShapeInferenceFunctionID(deviceTypeToDeviceID(deviceType),
                                                               params->sifGUID,
                                                               &sifID);
        ASSERT_EQ(sifID.Value, dummyShapeInferenceUniqueId);
    }

    void resetConfigAndKernelDB(uint64_t previousMode)
    {
        GCFG_COMPLEX_GUID_EXTRACTOR_MODE.setValue(previousMode);
        GCFG_COMPLEX_GUID_LIB_NAME.setValue(ComplexGuidLibName);
    }

    // TODO: Should be changed when enabling those tests, and CGUID is ready with a default name.
    const std::string ComplexGuidLibName  = "libComplexGuid.so";
    const std::string dummyLibName        = "libdummyComplexGuid.so";

protected:
    uint64_t m_prev_mode;
};

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_valid_ndims_tensor)
{
    /*
     * Verify that pass doesn't fail when there is a new tensor of with more than 5 dims for a TPCNode
     */

    NodePtr complexAddNode = initNodeAndTensors("valid_nDims_complex_add_f32");
    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // Should NOT fail due to nDims validation, as complex add f32 is extracted to a TPC node which can have nDims
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_invalid_ndims_tensor)
{
    /*
     * Verify that pass fails when there is a new tensor of non-Ndims node with more than 5 dims
     */

    NodePtr complexAddNode = initNodeAndTensors("invalid_nDims_norm_moments_fwd_f32_unconnected_f32");
    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // Should fail due to nDims validation, as one the extracted can't have nDims
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_invalid_tensor_property)
{
    /*
     * Verify that pass fails when there is a new tensor with invalid property
     */

    //  TODO SW-55946 make this a ComplexGuidExtractorWithUnifiedFuser,
    //  once GC tensor creation is correctly vaildated in tpc fuser pass.

    NodePtr complexAddNode = initNodeAndTensors("invalid_prop_complex_add_f32");
    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));  // fails due to new tensor with invalid property
}

TEST_P(ComplexGuidExtractorDummyTest, test_unconnected_norm_moments_cluster)
{
    /*
     * Extract norm_moments to add->gemm->neg, so an unconnected cluster is created: {add, neg).
     * Verify an exception is thrown in tpcFuser pass.
     */
    // guid has f32 twice, so it will be recognized by both cguid clustering and addNode logic
    NodePtr complexAddNode = initNodeAndTensors("norm_moments_fwd_f32_unconnected_f32");
    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    auto nodeId = complexAddNode->getId();
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
    ASSERT_TRUE((*m_graph).getExeSortedNodes().size() > 1);
    try
    {
        TPCClusterConstructor clusterConstructor(*m_graph);
        clusterConstructor.computeClusters();
        FAIL() << "Expected exception after connectivity check";
    }
    catch (std::exception& exception)
    {
        ASSERT_EQ(exception.what(), fmt::format("Cluster from complex guid Id {} is not connected", nodeId))
            << "Expected specific connectivity exception";
    }
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass)
{
    auto guid = "complex_add_f32";
    NodePtr complexAddNode = initNodeAndTensors(guid);

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 3);
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_two_RMW_sections)
{
    /*
     * Verify that pass fails when RMW tensors of same extracted node have different sections
     */

    NodePtr complexAddNode = initNodeAndTensors("two_RMW_complex_add_f32");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to tensors in different RMW sections
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_dropped_external_tensor)
{
    /*
     * Verify that pass fails when one of the external tensors wasn't returned by complex GUID
     */

    NodePtr complexAddNode = initNodeAndTensors("dropped_external_tensor_complex_add_f32");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to external tensor wasn't returned by complex GUID
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_new_output_RMW_tensor_no_alias)
{
    /*
     * Verify that pass fails when there is a new RMW output tensor which is not alias to other RMW output
     * Alias difference is not same section
     */

    NodePtr complexAddNode = initNodeAndTensors("invalid_new_RMW_output_complex_add_f32");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to new RMW output which is not alias to other RMW output
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_new_output_RMW_tensor_no_alias_offset)
{
    /*
     * Verify that pass fails when there is a new RMW output tensor which is not alias to other RMW output.
     * Alias difference is not same offset
     */
    NodePtr complexAddNode = initNodeAndTensors("invalid_new_RMW_output_offset_complex_add_f32", true);
    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to new RMW output which is not alias to other RMW output
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_new_output_persistent_tensor_no_alias)
{
    /*
     * Verify that pass fails when there is a new RMW output tensor which is not alias to other RMW output
     * Alias difference is not same section
     */

    NodePtr complexAddNode = initNodeAndTensors("invalid_new_persistent_output_complex_add_f32");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to new RMW output which is not alias to other RMW output
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_new_output_persistent_tensor_no_alias_offset)
{
    /*
     * Verify that pass fails when there is a new RMW output tensor which is not alias to other RMW output
     * Alias difference is not same section
     */

    NodePtr complexAddNode = initNodeAndTensors("invalid_new_persistent_output_offset_complex_add_f32");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to new RMW output which is not alias to other RMW output
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_pass_new_input_tensor)
{
    /*
     * Verify that pass fails when there is a new input tensor which doesn't have static data
     */

    NodePtr complexAddNode = initNodeAndTensors("invalid_new_input_complex_add_f32");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // fails due to new input tensor which doesn't have static data
    ASSERT_FALSE(extractFunctionalComplexGuidNodes(*m_graph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_non_default_strides)
{
    /*
     * Verify that pass succeeds and an intermediate tensor with non-default strides is extracted.
     * The extracted node is tpcnop, to allow stride manipulation without worrying for correctnes.
     * The original cguid node is complex_nop_strides which exctracted to 2 consecutive tpcnops.
     * First tpcnop output tensor has non-default strides.
     * The non-default strides are twice the size of default strides (expect FCD, which isn't supported until SW-43104).
     */

    const TSize C = 3;
    const TSize W = 3;
    const TSize H = 3;
    const TSize N = 1;

    char in1[C * W * H * N];

    const TSize sizes[] = {C, W, H, N};

    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor x = pTensor(new Tensor(4U, sizes, syn_type_int32, reinterpret_cast<char*>(in1)));
    x->setName("x");
    x->setDramOffset(0x7000);
    x->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    x->setMemoryDescriptor(persistentMemoryDesc);

    pTensor out = pTensor(new Tensor(4U, sizes, syn_type_int32, reinterpret_cast<char*>(in1)));
    out->setName("out");
    out->setDramOffset(0x9000);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    out->setMemoryDescriptor(persistentMemoryDesc);

    NodePtr complexAddNode =
        NodeFactory::createGenericTPCNode({x}, {out}, nullptr, "complex_nop_strides_f32", "complexGuidNode");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 2);

    auto allTensors = (*m_graph).getTensors();
    // get the intermediate tensor with non-default strides.
    auto intermediateTensors = (*m_graph).getGraphIntermediates();
    ASSERT_EQ(intermediateTensors.size(), 1);
    TStride inputStrides[Tensor::c_numOfNStrides];
    TStride intermediateStrides[Tensor::c_numOfNStrides];
    allTensors.find(x)->get()->getNStridesInBytes(inputStrides);
    intermediateTensors.front()->getNStridesInBytes(intermediateStrides);

    //  verify that intermediate strides are twice the of the default. skip FCD since it must be as element size.
    for (unsigned i = 1; i < HABANA_DIM_MAX; i++)
    {
        ASSERT_TRUE(intermediateStrides[i] == 2 * inputStrides[i]);
    }
}

TEST_P(ComplexGuidExtractorDummyTest, test_shape_manipulation_node_functional)
{
    testShapeManipulationNodes(FUNCTIONAL_COMPLEX_GUID, GetParam());
}

TEST_P(ComplexGuidExtractorDummyTest, test_shape_manipulation_node_performance)
{
    testShapeManipulationNodes(PERFORMANCE_COMPLEX_GUID, GetParam());
}

TEST_P(ComplexGuidExtractorDummyTest, test_complex_extractor_compile)
{
    auto guid = "complex_add_f32";
    NodePtr complexAddNode = initNodeAndTensors(guid);

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    // disable TPC fuser, so it won't try to cluster the extracted nodes
    setGlobalConfForTest(GCFG_RUN_TPC_FUSER, "false");
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    ASSERT_TRUE((*m_graph).compile());
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 3);
}

TEST_P(ComplexGuidExtractorDummyTest, test_cguid_functional_extractor_pass_with_performance_guid)
{
    GraphComparator comparator;
    NodePtr dummyNode = initNodeAndTensors("dummy_performance_guid");

    ASSERT_TRUE(dummyNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, dummyNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    auto preExtractionGraph = m_graph->clone();
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_TRUE(comparator.compareGraphs(*m_graph, *preExtractionGraph));
}

TEST_P(ComplexGuidExtractorDummyTest, test_cguid_performance_extractor_pass_with_functional_guid)
{
    GraphComparator comparator;
    NodePtr dummyNode = initNodeAndTensors("dummy_functional_guid");

    ASSERT_TRUE(dummyNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, dummyNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    auto preExtractionGraph = m_graph->clone();
    ASSERT_TRUE(extractPerformanceComplexGuidNodes(*m_graph));
    ASSERT_TRUE(comparator.compareGraphs(*m_graph, *preExtractionGraph));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
}

TEST_P(ComplexGuidExtractorDummyTest, test_cguid_performance_extractor_pass)
{
    auto    guid           = "complex_add_f32";
    NodePtr complexAddNode = initNodeAndTensors(guid);

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 1);
    ASSERT_TRUE(extractPerformanceComplexGuidNodes(*m_graph));
    ASSERT_EQ((*m_graph).getExeSortedNodes().size(), 3);
}

TEST_P(ComplexGuidExtractorDummyTest, test_unchanged_graph)
{
    GraphComparator comparator;
    NodePtr complexAddNode = initNodeAndTensors("graph_unchanged_dummy");

    ASSERT_TRUE(complexAddNode != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, complexAddNode));
    auto preExtractionGraph = m_graph->clone();
    ASSERT_TRUE(extractFunctionalComplexGuidNodes(*m_graph));
    ASSERT_TRUE(extractPerformanceComplexGuidNodes(*m_graph));
    ASSERT_TRUE(comparator.compareGraphs(*m_graph, *preExtractionGraph));
}

/*
 * to run a single test with a specific device type:
 * ComplexGuidExtractorDummyTest.<test_name>/_<deviceType> (e.g. _GaudiM)
 */
INSTANTIATE_TEST_SUITE_P(,
                         ComplexGuidExtractorDummyTest,
                         ::testing::Values(synDeviceGaudi, synDeviceGaudi2),
                         ComplexGuidExtractorDummyTest::GetName());
