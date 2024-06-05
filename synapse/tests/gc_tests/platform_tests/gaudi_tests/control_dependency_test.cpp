#include "gc_gaudi_test_infra.h"
#include "scoped_configuration_change.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "log_manager.h"
#include <fstream>
#include "node_factory.h"
#include "habana_global_conf_runtime.h"

class SynTrainingControlDependencyTest : public SynGaudiTestInfra
{
protected:
    virtual void SetUpTest() override
    {
        SynGaudiTestInfra::SetUpTest();

        synConfigurationGet("MAKE_CTRL_DEP_SOFT", m_origCtrlDepSoft, sizeof(m_origCtrlDepSoft));
        synConfigurationGet("SRAM_SLICER_MAX_CAPACITY_BYTES", m_origSramCapStr, sizeof(m_origSramCapStr));
        synConfigurationGet("SRAM_SLICER_BUNDLE_EXPANSION_ENABLED",
                            m_origTpcSlicingEnabled,
                            sizeof(m_origTpcSlicingEnabled));

        synConfigurationSet("MAKE_CTRL_DEP_SOFT", "false");  // to keep global config of hard control dep
        synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", "20000");
        synConfigurationSet("SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", "True");

        //This test checks that control nodes existence really effects the execution order, the test uses overlapping tensors for it's checks
        //and determines which node was executed last by the values in the tensor. Disable validation for this test.
    }
    virtual void TearDownTest() override
    {
        synConfigurationSet("MAKE_CTRL_DEP_SOFT", m_origCtrlDepSoft);
        synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", m_origSramCapStr);
        synConfigurationSet("SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", m_origTpcSlicingEnabled);
        SynGaudiTestInfra::TearDownTest();
    }

public:
    void run_ctrl_dep_case_conv_neg_sequence(bool withCtrlDep, bool pipeliningDisabled = false);
    void run_ctrl_dep_logical_op_test(bool withCtrlDep);

protected:
    char m_origSramCapStr[128];
    char m_origTpcSlicingEnabled[128];
    char m_origCtrlDepSoft[128];
};

TEST_F_GC(SynTrainingTestInfra, check_api_node_blocks_itself)
{
    synStatus      status;

    unsigned batch = 1;
    unsigned K     = 16;
    unsigned oW    = 50;
    unsigned oH    = 50;

    unsigned ofmSizes[] = {batch, oH, oW, K};

    // Prepare some descriptors
    const unsigned NUM_NEG_NODES = 10;
    char           negNodeGuid[] = "neg_fwd_f32";
    uint64_t       negNodeIds[NUM_NEG_NODES];
    synGraphHandle graphA;

    status = synGraphCreate(&graphA, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

    for (unsigned i = 0; i < NUM_NEG_NODES; i++)
    {
        synTensor in[]  = {createTrainingTensor(4U,
                                                syn_type_single,
                                                ofmSizes,
                                                true,
                                                ("neg_in_" + std::to_string(i)).c_str(),
                                                graphA)};
        synTensor out[] = {createTrainingTensor(4U,
                                                syn_type_single,
                                                ofmSizes,
                                                true,
                                                ("neg_out_" + std::to_string(i)).c_str(),
                                                graphA)};

        status = synNodeCreateWithId(graphA,
                                     in,
                                     out,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     negNodeGuid,
                                     ("neg_node_" + std::to_string(i)).c_str(),
                                     &negNodeIds[i],
                                     nullptr,
                                     nullptr);
        ASSERT_EQ(status, synSuccess) << "Failed to create neg Node";
    }
    synNodeId blocking[] = {negNodeIds[0], negNodeIds[1], negNodeIds[7], negNodeIds[4]};
    synNodeId blocked[] = {negNodeIds[3], negNodeIds[9], negNodeIds[5], negNodeIds[1]};
    status = synNodeDependencySet(graphA, blocking, blocked, 4, 4);
    ASSERT_EQ(status, synInvalidArgument) << "Dependency creation should fail";
    synGraphDestroy(graphA);
}

void SynTrainingControlDependencyTest::run_ctrl_dep_case_conv_neg_sequence(bool withCtrlDep, bool pipeliningDisabled)
{
    synStatus      status   = synSuccess;

    const unsigned deviceId = _getDeviceId();

    GCFG_CHECK_SECTION_OVERLAP.setValue(false);

    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 3;
    params.kW   = 3;
    params.dilH = 1;
    params.dilW = 1;
    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    unsigned batch = 1;
    unsigned C = 3;
    unsigned W = 112;
    unsigned H = 112;
    unsigned K = 512;
    unsigned kW = params.kW;
    unsigned kH = params.kH;
    unsigned oW = convOutputDimSize(W, params.kW, params.dW, params.padL + params.padR, params.dilW);
    unsigned oH = convOutputDimSize(H, params.kH, params.dH, params.padT + params.padB, params.dilH);

    unsigned dim = 4;
    unsigned ifmSizes[] = {batch, H, W, C};
    unsigned wSizes[] = {kH, kW, C, K};
    unsigned ofmSizes[] = {batch, oH, oW, K};

    unsigned ifmNumElem = 1, wNumElem = 1, ofmNumElem = 1;
    for (unsigned i = 0; i < dim; i++)
    {
        ifmNumElem *= ifmSizes[i];
        wNumElem *= wSizes[i];
        ofmNumElem *= ofmSizes[i];
    }

    unsigned ifmTotalSize = ifmNumElem * sizeof(float);
    unsigned wTotalSize = wNumElem * sizeof(float);
    unsigned ofmTotalSize = ofmNumElem * sizeof(float);

    // Prepare some descriptors
    synTensor ifm, w, ofm, neg;
    uint64_t ifm_mem, w_mem, ofm_mem, neg_mem;

    synGraphHandle graphA;
    status = synGraphCreate(&graphA, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create gaudi graph";

    status = synDeviceMalloc(deviceId, ifmTotalSize + ofmTotalSize, 0, 0, &ifm_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory for IFM";
    // we would like to have neg output and conv input on the same memory address
    neg_mem = ifm_mem;
    synSectionHandle  section;
    synSectionHandle* sectionPtr = nullptr;
    if (withCtrlDep)
    {
        synSectionCreate(&section, 0, graphA);
        sectionPtr = &section;
    }

    status = synDeviceMalloc(deviceId, wTotalSize , 0, 0, &w_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory for W";

    status = synDeviceMalloc(deviceId, ofmTotalSize, 0, 0, &ofm_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory for OFM";

    ifm = createTrainingTensor(4U, syn_type_single, ifmSizes, true, "ifm", graphA, sectionPtr);
    w   = createTrainingTensor(4U, syn_type_single, wSizes, true, "w", graphA);
    ofm = createTrainingTensor(4U, syn_type_single, ofmSizes, true, "ofm", graphA);
    neg = createTrainingTensor(4U, syn_type_single, ofmSizes, true, "neg_out", graphA, sectionPtr);

    synTensor convInputs[2] = {ifm, w};
    synTensor convOutput[1] = {ofm};


    synTensor negInputs[1]  = {ofm};
    synTensor negOutputs[1] = {neg};
    char      negNodeGuid[] = "neg_fwd_f32";
    uint64_t negNodeId, convNodeId;
    status = synNodeCreateWithId(graphA,
                                 negInputs,
                                 negOutputs,
                                 1,
                                 1,
                                 nullptr,
                                 0,
                                 negNodeGuid,
                                 "neg_node",
                                 &negNodeId,
                                 nullptr,
                                 nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create neg Node";

    status = synNodeCreateWithId(graphA,
                                 convInputs,
                                 convOutput,
                                 2,
                                 1,
                                 &params,
                                 sizeof(synConvolutionParams),
                                 NodeFactory::convolutionNodeTypeName,
                                 "conv",
                                 &convNodeId,
                                 nullptr,
                                 nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create Conv Node";

    if (withCtrlDep)
    {
        uint64_t blockingNodesId[] = {convNodeId};
        uint64_t blockedNodesId[]  = {negNodeId};
        status = synNodeDependencySet(graphA, blockingNodesId, blockedNodesId, 1, 1);
    }
    synRecipeHandle recipeHandle;

    status = synGraphCompile(&recipeHandle, graphA, GetTestFileName().c_str(), 0);
    ASSERT_EQ(status , synSuccess) << "Failed to compile graph";

    uint64_t topologyWorkspaceSize = 0;

    status = synWorkspaceGetSize(&topologyWorkspaceSize, recipeHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to get workspace's memory-size";

    uint64_t topologyWorkspaceBuffer = 0;

    if (topologyWorkspaceSize != 0)
    {
        status = synDeviceMalloc(deviceId, topologyWorkspaceSize, 0, 0, &topologyWorkspaceBuffer);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace buffer";
    }

    float* ifmBuffer;
    float* wBuffer;
    float* ofmBuffer;
    float* negBuffer;

    status = synHostMalloc(deviceId, ifmTotalSize , 0, (void**)&ifmBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed malloc ifm buffer";
    status = synHostMalloc(deviceId, wTotalSize , 0, (void**)&wBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed malloc w buffer";
    status = synHostMalloc(deviceId, ofmTotalSize , 0, (void**)&ofmBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed malloc ofm buffer";
    status = synHostMalloc(deviceId, ofmTotalSize , 0, (void**)&negBuffer);
    ASSERT_EQ(status, synSuccess) << "Failed malloc neg buffer";

    synEventHandle eventHandle;
    status = synEventCreate(&eventHandle, deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed create event";

    // initializing input buffer with ones, and output with zeros:

    float expectedVal = params.kH * params.kW * C;
    for (unsigned i = 0; i < ifmNumElem; i++)
    {
        ifmBuffer[i] = 1;
    }
    for (unsigned i = 0; i < wNumElem; i++)
    {
        wBuffer[i] = 1;
    }
    for (unsigned i = 0; i < ofmNumElem; i++)
    {
        ofmBuffer[i] = 0;
        negBuffer[i] = 0;
    }

    status = synMemCopyAsync( m_streamHandleDownload,
                              (uint64_t) ifmBuffer,
                              ifmTotalSize,
                              ifm_mem,
                              HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy ifm to the device";

    status = synMemCopyAsync( m_streamHandleDownload,
                              (uint64_t) wBuffer,
                              wTotalSize,
                              w_mem,
                              HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy w to the device";

    status = synEventRecord(eventHandle, m_streamHandleDownload);
    ASSERT_EQ(status, synSuccess) << "Failed record-event (copy to the device)";

    status = synStreamWaitEvent(m_streamHandleCompute, eventHandle, 0);
    ASSERT_EQ(status, synSuccess) << "Failed stream-wait-event (completion of copy to the device)";

    synLaunchTensorInfo ifmTensor        = {"ifm", ifm_mem, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synLaunchTensorInfo wTensor          = {"w", w_mem, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synLaunchTensorInfo ofmTensor        = {"ofm", ofm_mem, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synLaunchTensorInfo negTensor        = {"neg_out", neg_mem, DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
    synLaunchTensorInfo conTensorsList[] = {ifmTensor, wTensor, ofmTensor, negTensor};

    uint32_t totalNumOfTensors = sizeof(conTensorsList) / sizeof(synLaunchTensorInfo);
    prepareTensorInfo(recipeHandle, conTensorsList, totalNumOfTensors);
    status =
        synLaunch(m_streamHandleCompute, conTensorsList, totalNumOfTensors, topologyWorkspaceBuffer, recipeHandle, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to enqueue";

    status = synEventRecord(eventHandle, m_streamHandleCompute);
    ASSERT_EQ(status, synSuccess) << "Failed record-event (enqueue)";

    status = synStreamWaitEvent(m_streamHandleUpload, eventHandle, 0);
    ASSERT_EQ(status, synSuccess) << "Failed stream-wait-event (completion of compute)";

    status = synMemCopyAsync(m_streamHandleUpload,
                             ofm_mem,
                             ofmTotalSize,
                             (uint64_t)ofmBuffer,
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";

    status = synMemCopyAsync(m_streamHandleUpload,
                             neg_mem,
                             ofmTotalSize,
                             (uint64_t)negBuffer,
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";

    // waiting for the completion of last operation
    status = synStreamSynchronize(m_streamHandleUpload);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (copy from the device)";

    // make sure compute job finished
    int maxIter = 20;
    int iter    = 0;
    while (iter < maxIter)
    {
        status = synEventQuery(eventHandle);
        if (status == synSuccess) break;
        usleep(1000);
        iter++;
    }
    ASSERT_NE(iter, maxIter) << "compute didn't finished in time 20 milliseconds";

    status = synStreamQuery(m_streamHandleUpload);
    ASSERT_EQ(status, synSuccess) << "StreamQuery fails when event already concluded";

    // Validate:
    float* pOfmBuffer = (float*)ofmBuffer;
    float* pNegBuffer = (float*)negBuffer;
    if (withCtrlDep || pipeliningDisabled)
    {
        for (unsigned i = 0; i < ofmNumElem; i++)
        {
            // the expected value is 27, as kernel of 3*3*3 with one over IFM of ones will give 27.
            ASSERT_EQ(*pOfmBuffer, expectedVal);
            ASSERT_EQ(*pNegBuffer, -expectedVal);
            pOfmBuffer++;
            pNegBuffer++;
        }
    }
    else
    {
        bool isEqual = true;
        for (unsigned i = 0; i < ofmNumElem; i++)
        {
            // the expected value is 27, as kernel of 3*3*3 with one over IFM of ones will give 27.
            isEqual &= (*pOfmBuffer == expectedVal);
            isEqual &= (*pNegBuffer == -expectedVal);
            pOfmBuffer++;
            pNegBuffer++;
        }
        ASSERT_EQ(isEqual, false) << "Expecting outputs to differ from reference";
    }
    synRecipeDestroy(recipeHandle);
    synGraphDestroy(graphA);
}

// TODO [SW-100504] enable this test on gaudi3
TEST_F_GC(SynTrainingControlDependencyTest, parallel_conv_neg_expect_fail_2, {'-', synDeviceGaudi3})
{
    // Pipline mangement must be disabled because its slicing on batch dim is not enough to cause the input overriding.
    ScopedConfigurationChange slice_disable("ENABLE_PIPELINE_MANAGEMENT", "false");
    // Must enable spatial slice to cause input overriding.
    ScopedConfigurationChange spatialSliceEnable("SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED", "true");
    ScopedConfigurationChange validate_memory_section_disable("VALIDATE_MEMORY_SECTION_TENSORS", "false");
    ScopedConfigurationChange handle_memory_coherence_disable("HANDLE_MEMORY_COHERENCE", "false");
    run_ctrl_dep_case_conv_neg_sequence(false);
}

TEST_F_GC(SynTrainingControlDependencyTest, parallel_conv_neg_expect_pass_2)
{
    run_ctrl_dep_case_conv_neg_sequence(true);
}

/*
 * Three Add ops followed by reshape:
 *   input1/2/3 , out_add1/2/3, are persistent with same mem address
 *   reshape outputs are aliases to the same address due to logical op
 * Concat of reshape outputs.
 * if execution of add ops will be pipelined before reshaped tensor values will be set concat output will be wrong.
 * need control edges to from reshapex to addX+1.
 * pass of logical op with ctrl-dep should add memcpy for each reshape,
 * so data will be coped into different addresses and concat output will be correct.
 */
void SynTrainingControlDependencyTest::run_ctrl_dep_logical_op_test(bool withCtrlDep)
{
    unsigned sizes[] = {4, 4, 1, 1};
    unsigned reshaped_sizes[] = {1, 16, 1, 1};
    unsigned concat_sizes[] = {1, 48, 1, 1};
    unsigned dims = 2;
    const unsigned total_size_input = 4 * 4;

    unsigned add1Arg = createConstTensor(MEM_INIT_ALL_ONES, nullptr, sizes,
                                         dims, syn_type_single);
    unsigned add2Arg = createConstTensor(MEM_INIT_ALL_ONES, nullptr, sizes,
                                         dims, syn_type_single);
    unsigned add3Arg = createConstTensor(MEM_INIT_ALL_ONES, nullptr, sizes,
                                         dims, syn_type_single);

    uint64_t memSize      = getMemorySize(sizes, syn_type_single, dims);
    unsigned sectionIndex = createSection(memSize);
    unsigned input1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, dims, syn_type_single,
                                          nullptr, "input1", 0, 0, &sectionIndex);
    unsigned input2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, dims, syn_type_single,
                                          nullptr, "input2", 0, 0, &sectionIndex);
    unsigned input3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, dims, syn_type_single,
                                          nullptr, "input3", 0, 0, &sectionIndex);

    unsigned out_add1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, dims, syn_type_single,
                                            nullptr, "out_add1", 0, 0, &sectionIndex);
    unsigned out_add2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, dims, syn_type_single,
                                            nullptr, "out_add2", 0, 0, &sectionIndex);
    unsigned out_add3= createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizes, dims, syn_type_single,
                                           nullptr, "out_add3", 0, 0, &sectionIndex);

    unsigned in_reshape1 = connectOutputTensorToInputTensor(out_add1);
    unsigned in_reshape2 = connectOutputTensorToInputTensor(out_add2);
    unsigned in_reshape3 = connectOutputTensorToInputTensor(out_add3);

    unsigned out_reshape1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                         reshaped_sizes, dims, syn_type_float);
    unsigned out_reshape2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                         reshaped_sizes, dims, syn_type_float);
    unsigned out_reshape3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                         reshaped_sizes, dims, syn_type_float);

    unsigned in_concat1 =  connectOutputTensorToInputTensor(out_reshape1);
    unsigned in_concat2 =  connectOutputTensorToInputTensor(out_reshape2);
    unsigned in_concat3 =  connectOutputTensorToInputTensor(out_reshape3);

    unsigned out_concat = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, concat_sizes, dims,
                                              syn_type_single, nullptr, "out_concat");

    synNodeId add1Id,add2Id, add3Id, reshape1Id, reshape2Id, reshape3Id;
    addNodeToGraph("add_fwd_f32", {input1, add1Arg}, {out_add1}, nullptr, 0, "add1", 0, &add1Id);
    addNodeToGraph("add_fwd_f32", {input2, add2Arg}, {out_add2}, nullptr, 0, "add2", 0, &add2Id);
    addNodeToGraph("add_fwd_f32", {input3, add3Arg}, {out_add3}, nullptr, 0, "add3", 0, &add3Id);

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {in_reshape1}, {out_reshape1}, nullptr, 0, "reshape1", 0, &reshape1Id);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {in_reshape2}, {out_reshape2}, nullptr, 0, "reshape2", 0, &reshape2Id);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {in_reshape3}, {out_reshape3}, nullptr, 0, "reshape3", 0, &reshape3Id);

    unsigned concatDim = 1;
    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {in_concat1, in_concat2, in_concat3},
                   {out_concat},
                   &concatDim,
                   sizeof(concatDim),
                   "concat");

    if (withCtrlDep)
    {
        setNodeDependency(&reshape1Id, &add2Id, 1, 1);
        setNodeDependency(&reshape2Id, &add3Id, 1, 1);
    }
    compileAndRun();

    float_t* pOutputBuffer = (float_t*)m_hostBuffers[out_concat];

    for (unsigned i = 0; i < 3; i++)
    {
        float_t expected_output = withCtrlDep ? 2 + i : 4;
        unsigned base_idx = total_size_input * i;
        for (unsigned j = 0; j < total_size_input; j++)
        {
            ASSERT_EQ(pOutputBuffer[base_idx + j], expected_output) << "Unexpected output";
        }
    }
}

TEST_F_GC(SynTrainingControlDependencyTest, check_logical_nodes_ctrl_deps_pass)
{
    run_ctrl_dep_logical_op_test(true);
}
