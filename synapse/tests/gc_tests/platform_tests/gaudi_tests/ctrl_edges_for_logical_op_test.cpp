#include <cstddef>

#include "data_layout_test_infra.h"
#include "gc_gaudi_test_infra.h"
#include "syn_singleton.hpp"

static unsigned countNumRealNodes(const synGraphHandle& handle)
{
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(handle);
    unsigned           ret   = 0;
    for (const NodePtr& n : graph->getNodes())
    {
        if (n && !n->isLogicalOperation())
        {
            ret++;
        }
    }
    return ret;
}

TEST_F_GC(SynTrainingTestInfra, test_ctrl_edge_for_logical_op1)
{
    // [SW-70846]
    unsigned section  = createSection(16);
    unsigned sizes[]  = {4};
    unsigned tensor_0 = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            "tensor_0",
                                            0,
                                            0,
                                            &section);

    unsigned  tensor_1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    synNodeId relu_id;
    addNodeToGraph("relu_fwd_f32", {tensor_0}, {tensor_1}, nullptr, 0, "relu", 0, &relu_id);

    unsigned           tensor_2  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    synStridedOpParams sv_params = {0};
    sv_params.strides[0]         = 1;
    addNodeToGraph("strided_view", {tensor_0}, {tensor_2}, &sv_params, sizeof(sv_params));

    unsigned tensor_3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, 1);
    unsigned tensor_4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("mult_fwd_f32", {tensor_2, tensor_3}, {tensor_4});

    unsigned tensor_5 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("strided_insert", {tensor_0, tensor_4}, {tensor_5}, &sv_params, sizeof(sv_params));

    unsigned tensor_6 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("strided_view", {tensor_5}, {tensor_6}, &sv_params, sizeof(sv_params));

    unsigned tensor_7 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, 1);
    unsigned tensor_8 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("mult_fwd_f32", {tensor_6, tensor_7}, {tensor_8});

    unsigned  tensor_9 = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            "tensor_9",
                                            0,
                                            0,
                                            &section);
    synNodeId si_id;
    addNodeToGraph("strided_insert",
                   {tensor_5, tensor_8},
                   {tensor_9},
                   &sv_params,
                   sizeof(sv_params),
                   "insert",
                   0,
                   &si_id);
    setNodeDependency(&relu_id, &si_id, 1, 1);

    compileTopology("ctrl_edge_for_logical_op1", 0);

    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 3);  // verify no internal memcopy nodes were added
    runTopology();

    float* tensor0Data = (float*)m_hostBuffers[tensor_0];
    float* tensor1Data = (float*)m_hostBuffers[tensor_1];
    float* tensor3Data = (float*)m_hostBuffers[tensor_3];
    float* tensor7Data = (float*)m_hostBuffers[tensor_7];
    float* tensor9Data = (float*)m_hostBuffers[tensor_9];
    for (unsigned i = 0; i < sizes[0]; i++)
    {
        float expected = tensor0Data[i] * tensor3Data[i] * tensor7Data[i];
        ASSERT_EQ(expected, tensor9Data[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << tensor9Data[i];

        expected = std::max(tensor0Data[i], 0.f);
        ASSERT_EQ(expected, tensor1Data[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << tensor1Data[i];
    }
}

TEST_F_GC(SynTrainingTestInfra, test_ctrl_edge_for_logical_op2)
{
    // don't optimize out the strided insert in this test
    ScopedConfigurationChange optimizeSi("ENABLE_OPTIMIZE_STRIDED_INSERT", "false");

    // [SW-70846]
    unsigned section3      = createSection(16);
    unsigned section4      = createSection(16);
    unsigned section5      = createSection(16);
    unsigned sizes[]       = {4};
    float    tensor0Data[] = {0.1, 0.2, 0.3, 0.4};
    unsigned tensor0       = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           tensor0Data,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor0",
                                           0,
                                           0,
                                           &section3);

    unsigned  tensor1 = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor1",
                                           0,
                                           0,
                                           &section4);
    synNodeId reshapeID;
    addNodeToGraph("reshape", {tensor0}, {tensor1}, nullptr, 0, "reshape1", 0, &reshapeID);

    float    tensor2Data[] = {0.123, 0.234, 0.345, 0.456};
    unsigned tensor2       = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           tensor2Data,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor2",
                                           0,
                                           0,
                                           &section5);
    unsigned tensor3       = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor3",
                                           0,
                                           0,
                                           &section5);
    addNodeToGraph("mult_fwd_f32", {tensor1, tensor2}, {tensor3}, nullptr, 0, "mul1");

    unsigned           tensor4  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    synStridedOpParams siParams = {0};
    siParams.strides[0]         = 1;
    synNodeId si1ID;
    addNodeToGraph("strided_insert", {tensor0, tensor3}, {tensor4}, &siParams, sizeof(siParams), "SI1", 0, &si1ID);

    unsigned tensor5 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("reshape", {tensor4}, {tensor5}, nullptr, 0, "reshape2");

    float    tensor6Data[] = {0.8, 0.7, 0.6, 0.5};
    unsigned tensor6       = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, tensor6Data, sizes, 1);
    unsigned tensor7       = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("mult_fwd_f32", {tensor5, tensor6}, {tensor7}, nullptr, 0, "mul2");

    unsigned  tensor8 = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor8",
                                           0,
                                           0,
                                           &section3);
    synNodeId si2ID;
    addNodeToGraph("strided_insert", {tensor4, tensor7}, {tensor8}, &siParams, sizeof(siParams), "SI2", 0, &si2ID);

    setNodeDependency(&reshapeID, &si2ID, 1, 1);
    setNodeDependency(&si1ID, &si2ID, 1, 1);

    compileTopology("ctrl_edge_for_logical_op2", 0);

    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle),
              4);  // verify only 2 internal memcopy nodes were added, and reshape nodes are gone

    runTopology();

    float* tensor8Data = (float*)m_hostBuffers[tensor8];
    for (unsigned i = 0; i < sizes[0]; i++)
    {
        float expected = tensor0Data[i] * tensor2Data[i] * tensor6Data[i];
        ASSERT_EQ(expected, tensor8Data[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << tensor8Data[i];
    }
}

TEST_F_GC(SynTrainingTestInfra, test_ctrl_edge_for_logical_op3)
{
    // The SpillPersistentTensors Pass pass can replace persistent tensors with an intermediate ones
    // Thus, we disable it in order to check the diffusion of control edges between persistent tensors
    ScopedConfigurationChange enableSpillPersistentTensorsPass("SPILL_PERSISTENT_TENSORS", "false");

    // [SW-70846]
    unsigned section3      = createSection(16);
    unsigned section4      = createSection(16);
    unsigned section5      = createSection(16);
    unsigned section6      = createSection(16);
    unsigned sizes[]       = {4};
    float    tensor0Data[] = {0.1, 0.2, 0.3, 0.4};
    unsigned tensor0       = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           tensor0Data,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor0",
                                           0,
                                           0,
                                           &section3);

    unsigned  tensor1 = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor1",
                                           0,
                                           0,
                                           &section4);
    synNodeId reluId;
    addNodeToGraph("relu_fwd_f32", {tensor0}, {tensor1}, nullptr, 0, "relu", 0, &reluId);

    unsigned  tensor2 = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor2",
                                           0,
                                           0,
                                           &section5);
    synNodeId reshapeID;
    addNodeToGraph("reshape", {tensor0}, {tensor2}, nullptr, 0, "reshape1", 0, &reshapeID);

    float    tensor3Data[] = {0.123, 0.234, 0.345, 0.456};
    unsigned tensor3       = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_FROM_INITIALIZER,
                                           tensor3Data,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor3",
                                           0,
                                           0,
                                           &section6);
    unsigned tensor4       = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor4",
                                           0,
                                           0,
                                           &section5);
    addNodeToGraph("mult_fwd_f32", {tensor2, tensor3}, {tensor4}, nullptr, 0, "mul1");

    unsigned           tensor5  = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor5",
                                           0,
                                           0,
                                           &section3);
    synStridedOpParams siParams = {0};
    siParams.strides[0]         = 1;
    synNodeId si1ID;
    addNodeToGraph("strided_insert", {tensor0, tensor4}, {tensor5}, &siParams, sizeof(siParams), "SI1", 0, &si1ID);

    setNodeDependency(&reshapeID, &si1ID, 1, 1);
    setNodeDependency(&reluId, &si1ID, 1, 1);

    compileTopology("ctrl_edge_for_logical_op3", 0);

    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 4);  // verify 2 internal memcopy nodes were added.

    runTopology();

    float* tensor5Data = (float*)m_hostBuffers[tensor5];
    float* tensor1Data = (float*)m_hostBuffers[tensor1];
    for (unsigned i = 0; i < sizes[0]; i++)
    {
        float expected = tensor0Data[i] * tensor3Data[i];
        ASSERT_EQ(expected, tensor5Data[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << tensor5Data[i];

        expected = std::max(tensor0Data[i], 0.f);
        ASSERT_EQ(expected, tensor1Data[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << tensor1Data[i];
    }
}

TEST_F_GC(SynTrainingTestInfra, test_ctrl_edge_for_logical4)
{
    unsigned tensor_0_sizes[] = {8, 256};
    unsigned tensor_0 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_0_sizes, 2, syn_type_int32);

    unsigned tensor_1_sizes[] = {4, 64};
    unsigned tensor_1 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_1_sizes, 2, syn_type_int32);

    unsigned tensor_2_sizes[] = {4, 64};
    unsigned tensor_2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_2_sizes, 2, syn_type_int32);

    ns_GatherKernel::Params gatherParams = {.axis = 0};
    addNodeToGraph("gather_elements_fwd_i32",
                   {tensor_0, tensor_1},
                   {tensor_2},
                   &gatherParams,
                   sizeof(gatherParams),
                   "n0__hpu_gather_elements_gather_elements_fwd_i32");

    unsigned tensor_3_sizes[] = {256};
    unsigned tensor_3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_3_sizes, 1, syn_type_int32);

    addNodeToGraph("reshape", {tensor_2}, {tensor_3}, nullptr, 0, "Reshape2_0");

    unsigned tensor_4_sizes[] = {31, 16, 256};
    unsigned tensor_4 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_4_sizes, 3, syn_type_single);

    unsigned tensor_5_sizes[] = {5, 16, 256};
    unsigned tensor_5 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_5_sizes, 3, syn_type_single);

    synStridedOpParams svParams = {0};
    svParams.strides[0]         = 1;
    svParams.strides[1]         = 31;
    svParams.strides[2]         = 296;
    addNodeToGraph("strided_view", {tensor_4}, {tensor_5}, &svParams, sizeof(svParams), "StridedView3");

    // create tensor_6 tensor
    unsigned tensor_6_sizes[] = {5, 16, 256};
    unsigned tensor_6 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_6_sizes, 3, syn_type_single);
    ns_GatherKernel::Params gatherParams2 = {.axis = 2};
    addNodeToGraph("gather_fwd_f32",
                   {tensor_5, tensor_3},
                   {tensor_6},
                   &gatherParams2,
                   sizeof(gatherParams2),
                   "n1__aten_index_select_gather_fwd_f32_complex_gather_fwd_f32_2");

    // create tensor_7 tensor
    unsigned tensor_7_sizes[] = {31, 16, 256};
    unsigned tensor_7 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_7_sizes, 3, syn_type_single);
    addNodeToGraph("strided_insert", {tensor_4, tensor_6}, {tensor_7}, &svParams, sizeof(svParams), "StridedInsert6");

    // create tensor_8 tensor
    unsigned tensor_8_sizes[] = {4, 64};
    unsigned tensor_8 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_8_sizes, 2, syn_type_int32);

    synStridedOpParams svParams2 = {0};
    svParams2.strides[0]         = 1;
    svParams2.strides[1]         = 4;
    addNodeToGraph("strided_view", {tensor_2}, {tensor_8}, &svParams2, sizeof(svParams2), "StridedView7");

    // create tensor_9 tensor
    unsigned tensor_9_sizes[] = {1, 64};
    unsigned tensor_9 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_9_sizes, 2, syn_type_int32);

    // create tensor_10 tensor
    unsigned tensor_10_sizes[] = {4, 64};
    unsigned tensor_10 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_10_sizes, 2, syn_type_int32);

    addNodeToGraph("add_fwd_i32", {tensor_8, tensor_9}, {tensor_10}, nullptr, 0, "n2__aten_add__add_fwd_i30");

    // create tensor_11 tensor
    unsigned tensor_11_sizes[] = {4, 64};
    unsigned tensor_11 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_11_sizes, 2, syn_type_int32);

    addNodeToGraph("strided_insert",
                   {tensor_2, tensor_10},
                   {tensor_11},
                   &svParams2,
                   sizeof(svParams2),
                   "StridedInsert9");

    compileTopology("test_ctrl_edge_for_logical4", 0);

    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(getGraph(0).graphHandle);

    for (const NodePtr& n : graph->getNodes())
    {
        if (n->getNodeType() == Node::TYPE_STRIDED_INSERT)
        {
            const TensorPtr& original = n->getInput(0);
            const TensorPtr& insert   = n->getInput(1);

            // strided insert is the only consumer of these inputs
            EXPECT_EQ(graph->getNumberOfTensorConsumers(original), 1);
            EXPECT_EQ(graph->getNumberOfTensorConsumers(insert), 1);

            // producer order is correct
            const auto realOriginalProducers = graph->getRealProducers(original);
            ASSERT_EQ(realOriginalProducers.size(), 1);
            const NodePtr& originalProducer = *realOriginalProducers.begin();

            const auto realInsertProducers = graph->getRealProducers(insert);
            ASSERT_EQ(realInsertProducers.size(), 1);
            const NodePtr& insertProducer = *realInsertProducers.begin();
            EXPECT_NE(originalProducer, nullptr);
            EXPECT_NE(insertProducer, nullptr);

            // check control edges between original producer and insert producer
            EXPECT_NE(graph->getNumberOfPaths(originalProducer, insertProducer, Node::TENSOR_TYPE_ALL), 0);
        }
    }
}

/*
              t0
          sectionX       +-------+
        +--------------->+       |   t1
                         |  ADD  +----------->
                    +--->+       |
                    |    +--+----+
                    |       |
         +-------+  |       |  CTRL
   t2    |       |  |       + - - - - - +
+------->+ ReLU  +--+                   |
         |       |  |                   |
         +-------+  |                   |
                    |               +---v----+     t4
                    |               |        |  sectionX
                    +--------------->Identity+------------->
                                    |        |
                                    +--------+

removing identity node will cause a cycle
*/
TEST_F_GC(SynTrainingTestInfra, test_ctrl_edge_for_logical5)
{
    unsigned sizes[] = {4};
    unsigned section = createSection(static_cast<uint64_t>(sizes[0]) * 4);

    unsigned tensor2 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor2");
    unsigned tensor3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1);
    addNodeToGraph("relu_fwd_f32", {tensor2}, {tensor3}, nullptr, 0, "relu");

    unsigned tensor0 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor0",
                                           0,
                                           0,
                                           &section);

    unsigned tensor1 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, 1, syn_type_single, nullptr, "tensor1");
    synNodeId addId;
    addNodeToGraph("add_fwd_f32", {tensor0, tensor3}, {tensor1}, nullptr, 0, "add", 0, &addId);

    unsigned  tensor4 = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           "tensor4",
                                           0,
                                           0,
                                           &section);
    synNodeId identityId;
    addNodeToGraph("identity", {tensor3}, {tensor4}, nullptr, 0, "identity", 0, &identityId);

    setNodeDependency(&addId, &identityId, 1, 1);

    compileAndRun();

    const float* tensor0Data = (const float*)m_hostBuffers[tensor0];
    const float* tensor1Data = (const float*)m_hostBuffers[tensor1];
    const float* tensor2Data = (const float*)m_hostBuffers[tensor2];
    const float* tensor4Data = (const float*)m_hostBuffers[tensor4];
    for (unsigned i = 0; i < sizes[0]; i++)
    {
        float intermediateResult = std::max(0.f, tensor2Data[i]);
        EXPECT_EQ(tensor1Data[i], intermediateResult + tensor0Data[i]) << "index " << i;
        EXPECT_EQ(tensor4Data[i], intermediateResult) << "index " << i;
    }
}

TEST_F_GC(SynTrainingTestInfra, test_ctrl_edge_for_logical6)  // [SW-103495]
{
    unsigned sizesIn[]  = {64, 4, 2};
    unsigned sizesOut[] = {8, 8, 4, 2};
    unsigned section    = createSection(static_cast<uint64_t>(sizesIn[0] * sizesIn[1] * sizesIn[2]) * 4);

    unsigned tensor0_0 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             sizesIn,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "tensor0_0");
    unsigned tensor0_1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             sizesIn,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "tensor0_1");
    unsigned tensor1   = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           sizesIn,
                                           3,
                                           syn_type_single,
                                           nullptr,
                                           "tensor1",
                                           0,
                                           0,
                                           &section);
    unsigned tensor2   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesIn, 3);
    unsigned tensor3   = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           sizesOut,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           "tensor3",
                                           0,
                                           0,
                                           &section);

    addNodeToGraph("add_fwd_f32", {tensor0_0, tensor0_1}, {tensor1}, nullptr, 0, "add1");
    addNodeToGraph("relu_fwd_f32", {tensor1}, {tensor2}, nullptr, 0, "relu2");
    addNodeToGraph("reshape", {tensor2}, {tensor3}, nullptr, 0, "reshape");

    compileAndRun();

    const float* tensor0_0Data = (const float*)m_hostBuffers[tensor0_0];
    const float* tensor0_1Data = (const float*)m_hostBuffers[tensor0_1];
    const float* tensor1Data   = (const float*)m_hostBuffers[tensor1];
    const float* tensor3Data   = (const float*)m_hostBuffers[tensor3];
    for (unsigned i = 0; i < sizesIn[0] * sizesIn[1] * sizesIn[2]; i++)
    {
        float expected = std::max(0.f, tensor0_0Data[i] + tensor0_1Data[i]);
        EXPECT_EQ(tensor1Data[i], expected) << "index " << i;
        EXPECT_EQ(tensor3Data[i], expected) << "index " << i;
    }

    // verify a single internal memcopy nodes were added, and that both relu nodes were fused
    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 2);
}

/*
                               [in2]
                             +----------+
                                        |
                                        v
                +------+   [in1']    +------+  [add2Out]
             +->+ ReLU +------------>+ Add2 +------>
             |  +------+          |  +------+
     [in1]   |                    |
  +----------+                    +------+
             |                           V
             |  +--------+  [I-out]  +------+  [add1Out]
             +->+Identity+---------->+ Add1 +------>
                +--------+           +------+

    [in1] and [in1'] share the same memory, so the user must add a control edge between (Identity) and (ReLU).
    From the user's perspective, Identity is a REAL node - so this is legal.
    Since Identity is a logical node in synapse, GC must propagate the control edge to the REAL consumer (Add1).
    this test checks that the control edge propagates correctly to (Add1) and
    that GC adds a spill memcopy (to avoid the cycle). resulting in:

                       [in1']^
                             |
                           +----+       [in2]
                           |copy|        +
                           +----+        |
                             ^           v
                 +------+    |        +------+  [add2Out]
              +->+ ReLU +------------>+ Add2 +------>
              |  +------+     [t]  |  +------+
      [in1]   |                    |
   +----------+                    +------+
              |                           V
              |  +--------+  [I-out]  +------+  [add1Out]
              +->+Identity+---------->+ Add1 +------>
                 +--------+           +------+

    * with a control edge between (Add1) and (copy)

    In addition, we use permutation on inputs, and allow permutation on [in1'] so that to verify the logical transpose
    nodes (added by data layout flow) are handled correctly.
*/
TEST_F_GC(SynGaudiDataLayoutTest, test_ctrl_edge_for_logical7)  // [SW-106187]
{
    TestSizeVec operandSizes {2, 3, 4, 5};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned section = createSection(multiplyElements(operandSizes) * 4);

    unsigned in1 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       operandSizes.data(),
                                       operandSizes.size(),
                                       syn_type_single,
                                       nullptr,
                                       "in1",
                                       0,
                                       0,
                                       &section);
    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       operandSizes.data(),
                                       operandSizes.size(),
                                       syn_type_single,
                                       nullptr,
                                       "in2");

    const char* in1TagName = "in1Tag";
    unsigned    in1Tag     = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          operandSizes.data(),
                                          operandSizes.size(),
                                          syn_type_single,
                                          nullptr,
                                          in1TagName,
                                          0,
                                          0,
                                          &section);

    unsigned add1Out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           operandSizes.data(),
                                           operandSizes.size(),
                                           syn_type_single,
                                           nullptr,
                                           "add1Out");

    unsigned add2Out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           operandSizes.data(),
                                           operandSizes.size(),
                                           syn_type_single,
                                           nullptr,
                                           "add2Out");

    unsigned identityOut = createTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        operandSizes.data(),
                                        operandSizes.size(),
                                        syn_type_single);

    setPermutation(in1, perm);
    setPermutation(in2, perm);
    synTensorSetAllowPermutation(m_tensors[in1Tag], true);
    synTensorSetAllowPermutation(m_tensors[add1Out], true);
    synTensorSetAllowPermutation(m_tensors[add2Out], true);
    synNodeId identityId;
    synNodeId reluId;

    addNodeToGraph("relu_fwd_f32", {in1}, {in1Tag}, nullptr, 0, "relu", 0, &reluId);
    addNodeToGraph("identity", {in1}, {identityOut}, nullptr, 0, "I", 0, &identityId);
    addNodeToGraph("add_fwd_f32", {identityOut, in1Tag}, {add1Out}, nullptr, 0, "add1");
    addNodeToGraph("add_fwd_f32", {in2, in1Tag}, {add2Out}, nullptr, 0, "add2");

    setNodeDependency(&identityId, &reluId, 1, 1);

    compileAndRun();

    float* in1Data     = castHostBuffer<float>(in1);
    float* in2Data     = castHostBuffer<float>(in2);
    float* in1TagData  = castHostBuffer<float>(in1Tag);
    float* add1OutData = castHostBuffer<float>(add1Out);
    float* add2OutData = castHostBuffer<float>(add2Out);

    ASSERT_TRUE(isPermuted(in1TagName, perm));

    // transpose shape to NHWC
    perm.permuteShape(operandSizes.data(), operandSizes.size());
    transposeBuffer(operandSizes.data(), operandSizes.size(), in1Data, perm.getInversePermutation());
    transposeBuffer(operandSizes.data(), operandSizes.size(), in2Data, perm.getInversePermutation());
    transposeBuffer(operandSizes.data(), operandSizes.size(), in1TagData, perm.getInversePermutation());
    transposeBuffer(operandSizes.data(), operandSizes.size(), add1OutData, perm.getInversePermutation());
    transposeBuffer(operandSizes.data(), operandSizes.size(), add2OutData, perm.getInversePermutation());

    const auto nElem = multiplyElements(operandSizes.begin(), operandSizes.end());
    for (int i = 0; i < nElem; i++)
    {
        EXPECT_FLOAT_EQ(in1TagData[i], std::max(0.f, in1Data[i]));
        EXPECT_FLOAT_EQ(add1OutData[i], in1TagData[i] + in1Data[i]);
        EXPECT_FLOAT_EQ(add2OutData[i], in1TagData[i] + in2Data[i]);
    }

    // verify a single internal memcopy nodes were added
    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 4);
}

/*
Pre Graph:
          +----------+     In1
          | Constant +------------+
          +----------+            |
                                  |
                                  v
                   In2       +----+--------+    Out
             +---------------> SliceInsert +------------>
                             +-------------+

Since In1 and In2 are permuted, after data layout passes we will get:

                  +---+     In1
                +>+ T +---------->
                | +---+
+----------+    |
| Constant +----+------------------v
+----------+                       |
                                   |
                                   v
              +---+     In2   +----+--------+     +---+     Out
         +--->+ T +---------->+ SliceInsert +---->+ T +------>
              +---+           +-------------+     +---+

Make sure that the new producer of In1 is considered when checking if to insert a copy before the SliceInsert.
Otherwise, it might consume the wrong data (if the other SliceInsert producer finishes writing) -
 leading to a unhandled Write After Read dependency.

*/
TEST_F_GC(SynGaudiDataLayoutTest, test_ctrl_edge_for_logical8)  // [SW-116094]
{
    TestSizeVec realSizes {2, 3, 4, 5};
    TestSizeVec insertSizes {2, 3, 4, 3};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned in1 = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       realSizes.data(),
                                       realSizes.size(),
                                       syn_type_single,
                                       nullptr,
                                       "in1");

    unsigned in2 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       insertSizes.data(),
                                       insertSizes.size(),
                                       syn_type_single,
                                       nullptr,
                                       "in2");

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       realSizes.data(),
                                       realSizes.size(),
                                       syn_type_single,
                                       nullptr,
                                       "out");

    setPermutation(in1, perm);
    setPermutation(in2, perm);
    synTensorSetAllowPermutation(m_tensors[out], true);

    ns_ConstantKernel::Params constParams = {};
    constParams.constant.f                = 0;
    addNodeToGraph("constant_f32", {}, {in1}, &constParams, sizeof(constParams));

    synSliceParams sliceParams = {0};
    for (int i = 0; i < realSizes.size(); i++)
    {
        sliceParams.axes[i]  = i;
        sliceParams.ends[i]  = insertSizes[i];
        sliceParams.steps[i] = 1;
    }

    addNodeToGraph("slice_insert", {in1, in2}, {out}, &sliceParams, sizeof(sliceParams), "slice_insert");

    compileAndRun();

    float* in2Data = castHostBuffer<float>(in2);
    float* outData = castHostBuffer<float>(out);

    ASSERT_TRUE(isPermuted("out", perm));

    // transpose shape to NHWC
    perm.permuteShape(realSizes.data(), realSizes.size());
    perm.permuteShape(insertSizes.data(), insertSizes.size());
    transposeBuffer(insertSizes.data(), insertSizes.size(), in2Data, perm.getInversePermutation());
    transposeBuffer(realSizes.data(), realSizes.size(), outData, perm.getInversePermutation());

    const auto nElemInsert = multiplyElements(realSizes.begin(), realSizes.end());
    const auto nElemReal   = multiplyElements(insertSizes.begin(), insertSizes.end());
    for (int i = 0; i < nElemInsert; i++)
    {
        EXPECT_FLOAT_EQ(outData[i], in2Data[i]);
    }
    for (int i = nElemInsert; i < nElemReal; i++)
    {
        EXPECT_FLOAT_EQ(outData[i], 0.);
    }

    // verify 2 internal memcopy nodes were added
    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 3);
}

/*
                        +-------+  t2
                     +->+ ReLU3 +------>
                     |  +-------+
                     |
                     |
 t0  +-------+  t5   |  +---------+ t3 (same memory section as t4)
+--->+ ReLU1 +-------+->+ Reshape +---->
     +-------+          +---+-----+
                           CTRL
                            v
                   t1   +---+---+  t4
                  +---->+ ReLU2 +------>
                        +-------+
*/
class SynGaudiLogicalOpCtrlEdgeTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<bool>
{
};

TEST_P_GC(SynGaudiLogicalOpCtrlEdgeTest, test_ctrl_edge_for_logical9)  // [SW-122930]
{
    bool                      negativeTest = GetParam();
    ScopedConfigurationChange fuser_pass("RUN_TPC_FUSER", "false");  // not interested
    ScopedConfigurationChange spill_persistent_pass("SPILL_PERSISTENT_TENSORS", "false");  // don't avoid the issue
    ScopedConfigurationChange disable_pass("HANDLE_MEMORY_COHERENCE", negativeTest ? "false" : "true");

    unsigned sizesIn[]  = {32, 2};
    unsigned sizesOut[] = {64};
    unsigned section    = createSection(static_cast<uint64_t>(sizesOut[0]) * 4);

    unsigned t0 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesIn,
                                      2,
                                      syn_type_single,
                                      nullptr,
                                      "t0");
    unsigned t1 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesOut,
                                      1,
                                      syn_type_single,
                                      nullptr,
                                      "t1");
    unsigned t2 = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesIn,
                                      2,
                                      syn_type_single,
                                      nullptr,
                                      "t2");
    unsigned t3 = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesOut,
                                      1,
                                      syn_type_single,
                                      nullptr,
                                      "t3",
                                      0,
                                      0,
                                      &section);
    unsigned t4 = createPersistTensor(OUTPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesOut,
                                      1,
                                      syn_type_single,
                                      nullptr,
                                      "t4",
                                      0,
                                      0,
                                      &section);
    unsigned t5 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesIn, 2);

    synNodeId id1, id2, id3;
    addNodeToGraph("relu_fwd_f32", {t1}, {t4}, nullptr, 0, "relu2", 0, &id2);
    addNodeToGraph("relu_fwd_f32", {t0}, {t5}, nullptr, 0, "relu1");
    addNodeToGraph("reshape", {t5}, {t3}, nullptr, 0, "reshape", 0, &id1);
    addNodeToGraph("relu_fwd_f32", {t5}, {t2}, nullptr, 0, "relu3", 0, &id3);
    setNodeDependency(&id1, &id2, 1, 1);

    if (negativeTest)
    {
        GraphData& graphData = getGraph(0);
        synStatus  status    = synGraphCompile(&graphData.recipeHandle, graphData.graphHandle, "", nullptr);
        ASSERT_TRUE(status != synSuccess);
        return;
    }

    compileAndRun();

    const float* t0Data = (const float*)m_hostBuffers[t0];
    const float* t1Data = (const float*)m_hostBuffers[t1];
    const float* t2Data = (const float*)m_hostBuffers[t2];
    const float* t3Data = (const float*)m_hostBuffers[t3];
    const float* t4Data = (const float*)m_hostBuffers[t4];
    for (unsigned i = 0; i < sizesOut[0]; i++)
    {
        ASSERT_EQ(t2Data[i], std::max(t0Data[i], 0.f)) << "index " << i;
        ASSERT_EQ(t3Data[i], std::max(t1Data[i], 0.f)) << "index " << i;
        ASSERT_EQ(t4Data[i], std::max(t1Data[i], 0.f)) << "index " << i;
    }

    // verify no extra memcopies nodes were added
    ASSERT_EQ(countNumRealNodes(getGraph(0).graphHandle), 3);
}

INSTANTIATE_TEST_SUITE_P(test_ctrl_edge_for_logical9, SynGaudiLogicalOpCtrlEdgeTest, testing::Values(false, true));