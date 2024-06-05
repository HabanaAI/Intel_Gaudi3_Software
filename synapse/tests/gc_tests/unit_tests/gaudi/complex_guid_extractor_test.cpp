#include "generic_graph_test.h"
#include "graph_compiler/passes/complex_guid_extractor.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

class ComplexGuidExtractorGaudiOnlyTest : public GenericGraphTest
{
    void SetUp() override
    {
        GenericGraphTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_P(ComplexGuidExtractorGaudiOnlyTest, test_complex_guid_with_dynamic_shape)
{
    // testing a complex guid with a dynamic shape

    std::string guid = "batched_nms_fwd_f32";  // this is a complex guid with dynamic shape

    // check kernel DB dynamic shapes APIs
    ASSERT_TRUE(KernelDB::instance().isDynamicShapeKernel(guid, tpc_lib_api::DEVICE_ID_GAUDI));

    tpc_lib_api::UniqueShapeInferenceHash sifID;

    ASSERT_TRUE(KernelDB::instance().GetKernelShapeInferenceFunctionID(tpc_lib_api::DEVICE_ID_GAUDI, guid, &sifID));
    ASSERT_NE(sifID.Value, 0);  // 0 is default if no sif exists

    unsigned sifLibraryVersion;
    sifLibraryVersion = KernelDB::instance().GetLibraryVersion(tpc_lib_api::DEVICE_ID_GAUDI, guid);
    ASSERT_NE(sifLibraryVersion, 0);  // 0 is default if no sif exists

    /*
     * From below we just create tensors and the batched_nms node, and verifing compilation succeeds.
     * Tensors configurations taken from tpc_fuser/mlir/pytenettests/ComplexGuidTests/Misc/BatchedNMSTest.hpp
     */
    synMemoryDescriptor persistentMemoryDesc(true);

    TSize N = 24;
    TSize C = 24;

    TSize inp1Size[2] = {4, N};
    char  in1[4 * N * sizeof(float)];
    TSize inp2Size[1] = {N};
    char  in2[N * sizeof(float)];
    TSize inp3Size[1] = {N};
    char  in3[N * sizeof(float)];
    TSize inp4Size[1] = {N};
    char  in4[N * sizeof(float)];
    TSize inp5Size[1] = {N * C};
    char  in5[N * C * sizeof(float)];
    TSize out1Size[1] = {N * C};
    char  out1[N * C * sizeof(float)];
    TSize out2Size[1] = {5};
    char  out2[5 * sizeof(float)];

    uint64_t memSecId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4;

    pTensor inBox = pTensor(new Tensor(2U, inp1Size, syn_type_float, reinterpret_cast<char*>(in1)));
    inBox->setName("inBox");
    inBox->setMemoryDescriptor(persistentMemoryDesc);
    inBox->setDramOffset(0x1000);
    inBox->setMemorySectionID(memSecId++);

    pTensor inScores = pTensor(new Tensor(1U, inp2Size, syn_type_float, reinterpret_cast<char*>(in2)));
    inScores->setName("inBox");
    inScores->setMemoryDescriptor(persistentMemoryDesc);
    inScores->setDramOffset(0x10000);
    inScores->setMemorySectionID(memSecId++);

    pTensor inClasses = pTensor(new Tensor(1U, inp3Size, syn_type_int32, reinterpret_cast<char*>(in3)));
    inClasses->setName("inClasses");
    inClasses->setMemoryDescriptor(persistentMemoryDesc);
    inClasses->setDramOffset(0x20000);
    inClasses->setMemorySectionID(memSecId++);

    pTensor shape1 = pTensor(new Tensor(1U,
                                        inp4Size,
                                        syn_type_int32,
                                        reinterpret_cast<char*>(in4),
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        nullptr,
                                        SHAPE_TENSOR));
    shape1->setName("shape1");
    shape1->setMemoryDescriptor(persistentMemoryDesc);
    shape1->setDramOffset(0x30000);
    shape1->setMemorySectionID(memSecId++);

    pTensor shape2 = pTensor(new Tensor(1U,
                                        inp5Size,
                                        syn_type_int32,
                                        reinterpret_cast<char*>(in5),
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        nullptr,
                                        SHAPE_TENSOR));
    shape2->setName("shape2");
    shape2->setMemoryDescriptor(persistentMemoryDesc);
    shape2->setDramOffset(0x40000);
    shape2->setMemorySectionID(memSecId++);

    pTensor outIdx1 = pTensor(new Tensor(1U, out1Size, syn_type_int32, reinterpret_cast<char*>(out1)));
    outIdx1->setName("outIdx1");
    outIdx1->setMemoryDescriptor(persistentMemoryDesc);
    outIdx1->setDramOffset(0x50000);
    outIdx1->setMemorySectionID(memSecId++);

    pTensor outIdx2 = pTensor(new Tensor(1U, out2Size, syn_type_int32, reinterpret_cast<char*>(out2)));
    outIdx2->setName("outIdx1");
    outIdx2->setMemoryDescriptor(persistentMemoryDesc);
    outIdx2->setDramOffset(0x60000);
    outIdx2->setMemorySectionID(memSecId++);

    ns_BatchedNmsKernel::Params userParams;
    userParams.nms_threshold   = 0.35;
    userParams.max_num_classes = C;

    NodePtr batchedNms = NodeFactory::createNode({inBox, inScores, inClasses, shape1, shape2},
                                                 {outIdx1, outIdx2},
                                                 (ns_BatchedNmsKernel::Params*)&userParams,
                                                 guid.c_str(),
                                                 "batchedNmsNode");

    ASSERT_TRUE(GraphEditor::addNode(*m_graph, batchedNms));
    ASSERT_EQ(m_graph->getExeSortedNodes().size(), 1);
    ASSERT_TRUE(m_graph->compile());
    // Check if CGUID has replaced the node with other nodes, and the original one isn't one of them.
    const auto& graphNodes = m_graph->getExeSortedNodes();
    ASSERT_GT(graphNodes.size(), 1);
    ASSERT_TRUE(std::none_of(graphNodes.begin(), graphNodes.end(), [&](const NodePtr& n) {
        return n->getNodeName() == batchedNms->getNodeName() || n->getGUID() == batchedNms->getGUID();
    }));
}

INSTANTIATE_TEST_SUITE_P(,
                         ComplexGuidExtractorGaudiOnlyTest,
                         ::testing::Values(synDeviceGaudi),
                         ComplexGuidExtractorGaudiOnlyTest::GetName());