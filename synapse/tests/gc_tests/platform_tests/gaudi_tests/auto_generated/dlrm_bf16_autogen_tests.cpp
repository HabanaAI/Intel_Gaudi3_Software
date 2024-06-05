


/***************************************************************************************
 ***************************************************************************************
 ***************************************************************************************
 ***                                                                                 ***
 ***        This file is auto-generated. DO NOT EDIT!!!                              ***
 ***                                                                                 ***
 ***        To update this file use json_to_synapse.py script from the gc_tools      ***
 ***                                                                                 ***
 ***************************************************************************************
 ***************************************************************************************
 ***************************************************************************************/
#include "synapse_api.h"
#include "graph_manager.h"
#include "perf_lib_layer_params.h"
#include "../gc_autogen_test.h"

class SynGaudiDLRMTestBFloat16 : public SynGaudiAutoGenTest
{
public:
    SynGaudiDLRMTestBFloat16() { setSupportedDevices({synDeviceGaudi}); }
};

TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Linear_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * bot_l_0_linear node
     * inputs: [X_int(128, 1, 1, 13)(dtype=bf16), bot_l_0_linear_weight(1, 1, 13, 512)(dtype=bf16), bot_l_0_linear_bias[512](dtype=bf16)]
     * output: [bot_l_0_linear_output(128, 1, 1, 512)(dtype=bf16)]
     *************/
    synConvolutionParams bot_l_0_linear_kernel_params;
    bot_l_0_linear_kernel_params.dH = 1;
    bot_l_0_linear_kernel_params.dW = 1;
    bot_l_0_linear_kernel_params.kH = 1;
    bot_l_0_linear_kernel_params.kW = 1;
    bot_l_0_linear_kernel_params.padT = 0;
    bot_l_0_linear_kernel_params.padB = 0;
    bot_l_0_linear_kernel_params.padL = 0;
    bot_l_0_linear_kernel_params.padR = 0;
    bot_l_0_linear_kernel_params.dilH = 1;
    bot_l_0_linear_kernel_params.dilW = 1;

    // create X_int tensor
    pManagedTensor X_int = ManagedTensor::createManagedTensor("X_int", {128, 1, 1, 13}, syn_type_bf16, true, this);
    synLaunchTensorInfo X_int_tr_info = {"X_int", X_int->getDramAddress()};

    // create bot_l_0_linear_weight tensor
    pManagedTensor bot_l_0_linear_weight = ManagedTensor::createManagedTensor("bot_l_0_linear_weight", {1, 1, 13, 512}, syn_type_bf16, true, this);
    synLaunchTensorInfo bot_l_0_linear_weight_tr_info = {"bot_l_0_linear_weight",
                                                         bot_l_0_linear_weight->getDramAddress()};

    // create bot_l_0_linear_bias tensor
    pManagedTensor bot_l_0_linear_bias = ManagedTensor::createManagedTensor("bot_l_0_linear_bias", {512,}, syn_type_bf16, true, this);
    synLaunchTensorInfo bot_l_0_linear_bias_tr_info = {"bot_l_0_linear_bias", bot_l_0_linear_bias->getDramAddress()};

    // create bot_l_0_linear_output tensor
    pManagedTensor bot_l_0_linear_output = ManagedTensor::createManagedTensor("bot_l_0_linear_output", {128, 1, 1, 512}, syn_type_bf16, true, this);
    synLaunchTensorInfo                         bot_l_0_linear_output_tr_info = {"bot_l_0_linear_output",
                                                         bot_l_0_linear_output->getDramAddress()};
    ManagedNodeWithParams<synConvolutionParams> bot_l_0_linear({X_int, bot_l_0_linear_weight, bot_l_0_linear_bias}, {bot_l_0_linear_output}, "bot_l_0_linear", "spatial_convolution", graphHandle, bot_l_0_linear_kernel_params);
    bot_l_0_linear.createNode();


    // generate graph
    SetTestFileName("dlrm_Linear_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor X_int from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, X_int->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/X_int", typed_data, X_int->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, X_int->getDramAddress(), X_int->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(X_int_tr_info);

    // init tensor bot_l_0_linear_weight from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, bot_l_0_linear_weight->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bot_l_0_linear_weight", typed_data, bot_l_0_linear_weight->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, bot_l_0_linear_weight->getDramAddress(), bot_l_0_linear_weight->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(bot_l_0_linear_weight_tr_info);

    // init tensor bot_l_0_linear_bias from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, bot_l_0_linear_bias->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bot_l_0_linear_bias", typed_data, bot_l_0_linear_bias->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, bot_l_0_linear_bias->getDramAddress(), bot_l_0_linear_bias->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(bot_l_0_linear_bias_tr_info);

    // List outputs
    graph_outputs.push_back(bot_l_0_linear_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor bot_l_0_linear_output from file
    {
        bfloat16* ref_arr = new bfloat16[bot_l_0_linear_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, bot_l_0_linear_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bot_l_0_linear_output", ref_arr, bot_l_0_linear_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(bot_l_0_linear_output->getDramAddress(), data, bot_l_0_linear_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: bot_l_0_linear_output");
            validateResult(ref_arr, typed_data, bot_l_0_linear_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: bot_l_0_linear_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_ReLU_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * bot_l_0_relu node
     * inputs: [bot_l_0_linear_output(128, 1, 1, 512)(dtype=bf16)]
     * output: [bot_l_0_relu_output(128, 1, 1, 512)(dtype=bf16)]
     *************/

    // create bot_l_0_linear_output tensor
    pManagedTensor bot_l_0_linear_output = ManagedTensor::createManagedTensor("bot_l_0_linear_output", {128, 1, 1, 512}, syn_type_bf16, true, this);
    synLaunchTensorInfo bot_l_0_linear_output_tr_info = {"bot_l_0_linear_output",
                                                         bot_l_0_linear_output->getDramAddress()};

    // create bot_l_0_relu_output tensor
    pManagedTensor bot_l_0_relu_output = ManagedTensor::createManagedTensor("bot_l_0_relu_output", {128, 1, 1, 512}, syn_type_bf16, true, this);
    synLaunchTensorInfo bot_l_0_relu_output_tr_info = {"bot_l_0_relu_output", bot_l_0_relu_output->getDramAddress()};
    ManagedNode bot_l_0_relu({bot_l_0_linear_output}, {bot_l_0_relu_output}, "bot_l_0_relu", "relu_fwd_bf16", graphHandle);
    bot_l_0_relu.createNode();


    // generate graph
    SetTestFileName("dlrm_ReLU_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor bot_l_0_linear_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, bot_l_0_linear_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bot_l_0_linear_output", typed_data, bot_l_0_linear_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, bot_l_0_linear_output->getDramAddress(), bot_l_0_linear_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(bot_l_0_linear_output_tr_info);

    // List outputs
    graph_outputs.push_back(bot_l_0_relu_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor bot_l_0_relu_output from file
    {
        bfloat16* ref_arr = new bfloat16[bot_l_0_relu_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, bot_l_0_relu_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bot_l_0_relu_output", ref_arr, bot_l_0_relu_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(bot_l_0_relu_output->getDramAddress(), data, bot_l_0_relu_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: bot_l_0_relu_output");
            validateResult(ref_arr, typed_data, bot_l_0_relu_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: bot_l_0_relu_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, DISABLED_dlrm_EmbeddingBag_fwd) // disabled since guid was removed
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * emb_0_embbag node
     * inputs: [emb_0_embbag_weight[1460, 32](dtype=bf16), lS_indices_0[128](dtype=int32), lS_offset_0[128](dtype=int32)]
     * output: [emb_0_embbag_output(128, 32)(dtype=bf16)]
     *************/
    ns_EmbeddingBagWithSgdKernel::Params emb_0_embbag_kernel_params;
    emb_0_embbag_kernel_params.mode = EMBEDDING_BAG_MODE_SUM;
    emb_0_embbag_kernel_params.sgd.wd = 0;
    emb_0_embbag_kernel_params.sgd.mom = 0;
    emb_0_embbag_kernel_params.sgd.damp = 0;
    emb_0_embbag_kernel_params.sgd.nesterov = false;

    // create emb_0_embbag_weight tensor
    pManagedTensor emb_0_embbag_weight = ManagedTensor::createManagedTensor("emb_0_embbag_weight", {1460, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_0_embbag_weight_tr_info = {"emb_0_embbag_weight", emb_0_embbag_weight->getDramAddress()};

    // create lS_indices_0 tensor
    pManagedTensor lS_indices_0 = ManagedTensor::createManagedTensor("lS_indices_0", {128,}, syn_type_int32, true, this);
    synLaunchTensorInfo lS_indices_0_tr_info = {"lS_indices_0", lS_indices_0->getDramAddress()};

    // create lS_offset_0 tensor
    pManagedTensor lS_offset_0 = ManagedTensor::createManagedTensor("lS_offset_0", {128,}, syn_type_int32, true, this);
    synLaunchTensorInfo lS_offset_0_tr_info = {"lS_offset_0", lS_offset_0->getDramAddress()};

    // create emb_0_embbag_output tensor
    pManagedTensor emb_0_embbag_output = ManagedTensor::createManagedTensor("emb_0_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_0_embbag_output_tr_info = {"emb_0_embbag_output", emb_0_embbag_output->getDramAddress()};
    ManagedNodeWithParams<ns_EmbeddingBagWithSgdKernel::Params> emb_0_embbag({emb_0_embbag_weight, lS_indices_0, lS_offset_0}, {emb_0_embbag_output}, "emb_0_embbag", "embedding_bag_sgd_fwd_bf16", graphHandle, emb_0_embbag_kernel_params);
    emb_0_embbag.createNode();


    // generate graph
    SetTestFileName("dlrm_EmbeddingBag_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor lS_indices_0 from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, lS_indices_0->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/lS_indices_0", typed_data, lS_indices_0->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, lS_indices_0->getDramAddress(), lS_indices_0->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(lS_indices_0_tr_info);

    // init tensor lS_offset_0 from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, lS_offset_0->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/lS_offset_0", typed_data, lS_offset_0->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, lS_offset_0->getDramAddress(), lS_offset_0->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(lS_offset_0_tr_info);

    // init tensor emb_0_embbag_weight from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_weight->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_weight", typed_data, emb_0_embbag_weight->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_weight->getDramAddress(), emb_0_embbag_weight->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_weight_tr_info);

    // List outputs
    graph_outputs.push_back(emb_0_embbag_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor emb_0_embbag_output from file
    {
        bfloat16* ref_arr = new bfloat16[emb_0_embbag_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_output", ref_arr, emb_0_embbag_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(emb_0_embbag_output->getDramAddress(), data, emb_0_embbag_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: emb_0_embbag_output");
            validateResult(ref_arr, typed_data, emb_0_embbag_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: emb_0_embbag_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Concat_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * concat_interact0 node
     * inputs: [bot_l_4_relu_output[128, 32](dtype=bf16), emb_0_embbag_output[128, 32](dtype=bf16), emb_1_embbag_output[128, 32](dtype=bf16), emb_2_embbag_output[128, 32](dtype=bf16), emb_3_embbag_output[128, 32](dtype=bf16), emb_4_embbag_output[128, 32](dtype=bf16), emb_5_embbag_output[128, 32](dtype=bf16), emb_6_embbag_output[128, 32](dtype=bf16), emb_7_embbag_output[128, 32](dtype=bf16), emb_8_embbag_output[128, 32](dtype=bf16), emb_9_embbag_output[128, 32](dtype=bf16), emb_10_embbag_output[128, 32](dtype=bf16), emb_11_embbag_output[128, 32](dtype=bf16), emb_12_embbag_output[128, 32](dtype=bf16), emb_13_embbag_output[128, 32](dtype=bf16), emb_14_embbag_output[128, 32](dtype=bf16), emb_15_embbag_output[128, 32](dtype=bf16), emb_16_embbag_output[128, 32](dtype=bf16), emb_17_embbag_output[128, 32](dtype=bf16), emb_18_embbag_output[128, 32](dtype=bf16), emb_19_embbag_output[128, 32](dtype=bf16), emb_20_embbag_output[128, 32](dtype=bf16), emb_21_embbag_output[128, 32](dtype=bf16), emb_22_embbag_output[128, 32](dtype=bf16), emb_23_embbag_output[128, 32](dtype=bf16), emb_24_embbag_output[128, 32](dtype=bf16), emb_25_embbag_output[128, 32](dtype=bf16)]
     * output: [concat_interact0_output(128, 864)(dtype=bf16)]
     *************/

    // create bot_l_4_relu_output tensor
    pManagedTensor bot_l_4_relu_output = ManagedTensor::createManagedTensor("bot_l_4_relu_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo bot_l_4_relu_output_tr_info = {"bot_l_4_relu_output", bot_l_4_relu_output->getDramAddress()};

    // create emb_0_embbag_output tensor
    pManagedTensor emb_0_embbag_output = ManagedTensor::createManagedTensor("emb_0_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_0_embbag_output_tr_info = {"emb_0_embbag_output", emb_0_embbag_output->getDramAddress()};

    // create emb_1_embbag_output tensor
    pManagedTensor emb_1_embbag_output = ManagedTensor::createManagedTensor("emb_1_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_1_embbag_output_tr_info = {"emb_1_embbag_output", emb_1_embbag_output->getDramAddress()};

    // create emb_2_embbag_output tensor
    pManagedTensor emb_2_embbag_output = ManagedTensor::createManagedTensor("emb_2_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_2_embbag_output_tr_info = {"emb_2_embbag_output", emb_2_embbag_output->getDramAddress()};

    // create emb_3_embbag_output tensor
    pManagedTensor emb_3_embbag_output = ManagedTensor::createManagedTensor("emb_3_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_3_embbag_output_tr_info = {"emb_3_embbag_output", emb_3_embbag_output->getDramAddress()};

    // create emb_4_embbag_output tensor
    pManagedTensor emb_4_embbag_output = ManagedTensor::createManagedTensor("emb_4_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_4_embbag_output_tr_info = {"emb_4_embbag_output", emb_4_embbag_output->getDramAddress()};

    // create emb_5_embbag_output tensor
    pManagedTensor emb_5_embbag_output = ManagedTensor::createManagedTensor("emb_5_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_5_embbag_output_tr_info = {"emb_5_embbag_output", emb_5_embbag_output->getDramAddress()};

    // create emb_6_embbag_output tensor
    pManagedTensor emb_6_embbag_output = ManagedTensor::createManagedTensor("emb_6_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_6_embbag_output_tr_info = {"emb_6_embbag_output", emb_6_embbag_output->getDramAddress()};

    // create emb_7_embbag_output tensor
    pManagedTensor emb_7_embbag_output = ManagedTensor::createManagedTensor("emb_7_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_7_embbag_output_tr_info = {"emb_7_embbag_output", emb_7_embbag_output->getDramAddress()};

    // create emb_8_embbag_output tensor
    pManagedTensor emb_8_embbag_output = ManagedTensor::createManagedTensor("emb_8_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_8_embbag_output_tr_info = {"emb_8_embbag_output", emb_8_embbag_output->getDramAddress()};

    // create emb_9_embbag_output tensor
    pManagedTensor emb_9_embbag_output = ManagedTensor::createManagedTensor("emb_9_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_9_embbag_output_tr_info = {"emb_9_embbag_output", emb_9_embbag_output->getDramAddress()};

    // create emb_10_embbag_output tensor
    pManagedTensor emb_10_embbag_output = ManagedTensor::createManagedTensor("emb_10_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_10_embbag_output_tr_info = {"emb_10_embbag_output", emb_10_embbag_output->getDramAddress()};

    // create emb_11_embbag_output tensor
    pManagedTensor emb_11_embbag_output = ManagedTensor::createManagedTensor("emb_11_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_11_embbag_output_tr_info = {"emb_11_embbag_output", emb_11_embbag_output->getDramAddress()};

    // create emb_12_embbag_output tensor
    pManagedTensor emb_12_embbag_output = ManagedTensor::createManagedTensor("emb_12_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_12_embbag_output_tr_info = {"emb_12_embbag_output", emb_12_embbag_output->getDramAddress()};

    // create emb_13_embbag_output tensor
    pManagedTensor emb_13_embbag_output = ManagedTensor::createManagedTensor("emb_13_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_13_embbag_output_tr_info = {"emb_13_embbag_output", emb_13_embbag_output->getDramAddress()};

    // create emb_14_embbag_output tensor
    pManagedTensor emb_14_embbag_output = ManagedTensor::createManagedTensor("emb_14_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_14_embbag_output_tr_info = {"emb_14_embbag_output", emb_14_embbag_output->getDramAddress()};

    // create emb_15_embbag_output tensor
    pManagedTensor emb_15_embbag_output = ManagedTensor::createManagedTensor("emb_15_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_15_embbag_output_tr_info = {"emb_15_embbag_output", emb_15_embbag_output->getDramAddress()};

    // create emb_16_embbag_output tensor
    pManagedTensor emb_16_embbag_output = ManagedTensor::createManagedTensor("emb_16_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_16_embbag_output_tr_info = {"emb_16_embbag_output", emb_16_embbag_output->getDramAddress()};

    // create emb_17_embbag_output tensor
    pManagedTensor emb_17_embbag_output = ManagedTensor::createManagedTensor("emb_17_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_17_embbag_output_tr_info = {"emb_17_embbag_output", emb_17_embbag_output->getDramAddress()};

    // create emb_18_embbag_output tensor
    pManagedTensor emb_18_embbag_output = ManagedTensor::createManagedTensor("emb_18_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_18_embbag_output_tr_info = {"emb_18_embbag_output", emb_18_embbag_output->getDramAddress()};

    // create emb_19_embbag_output tensor
    pManagedTensor emb_19_embbag_output = ManagedTensor::createManagedTensor("emb_19_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_19_embbag_output_tr_info = {"emb_19_embbag_output", emb_19_embbag_output->getDramAddress()};

    // create emb_20_embbag_output tensor
    pManagedTensor emb_20_embbag_output = ManagedTensor::createManagedTensor("emb_20_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_20_embbag_output_tr_info = {"emb_20_embbag_output", emb_20_embbag_output->getDramAddress()};

    // create emb_21_embbag_output tensor
    pManagedTensor emb_21_embbag_output = ManagedTensor::createManagedTensor("emb_21_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_21_embbag_output_tr_info = {"emb_21_embbag_output", emb_21_embbag_output->getDramAddress()};

    // create emb_22_embbag_output tensor
    pManagedTensor emb_22_embbag_output = ManagedTensor::createManagedTensor("emb_22_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_22_embbag_output_tr_info = {"emb_22_embbag_output", emb_22_embbag_output->getDramAddress()};

    // create emb_23_embbag_output tensor
    pManagedTensor emb_23_embbag_output = ManagedTensor::createManagedTensor("emb_23_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_23_embbag_output_tr_info = {"emb_23_embbag_output", emb_23_embbag_output->getDramAddress()};

    // create emb_24_embbag_output tensor
    pManagedTensor emb_24_embbag_output = ManagedTensor::createManagedTensor("emb_24_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_24_embbag_output_tr_info = {"emb_24_embbag_output", emb_24_embbag_output->getDramAddress()};

    // create emb_25_embbag_output tensor
    pManagedTensor emb_25_embbag_output = ManagedTensor::createManagedTensor("emb_25_embbag_output", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_25_embbag_output_tr_info = {"emb_25_embbag_output", emb_25_embbag_output->getDramAddress()};

    // create concat_interact0_output tensor
    pManagedTensor concat_interact0_output = ManagedTensor::createManagedTensor("concat_interact0_output", {128, 864}, syn_type_bf16, true, this);
    synLaunchTensorInfo concat_interact0_output_tr_info = {"concat_interact0_output",
                                                           concat_interact0_output->getDramAddress()};
    ManagedNode concat_interact0({bot_l_4_relu_output, emb_0_embbag_output, emb_1_embbag_output, emb_2_embbag_output, emb_3_embbag_output, emb_4_embbag_output, emb_5_embbag_output, emb_6_embbag_output, emb_7_embbag_output, emb_8_embbag_output, emb_9_embbag_output, emb_10_embbag_output, emb_11_embbag_output, emb_12_embbag_output, emb_13_embbag_output, emb_14_embbag_output, emb_15_embbag_output, emb_16_embbag_output, emb_17_embbag_output, emb_18_embbag_output, emb_19_embbag_output, emb_20_embbag_output, emb_21_embbag_output, emb_22_embbag_output, emb_23_embbag_output, emb_24_embbag_output, emb_25_embbag_output}, {concat_interact0_output}, "concat_interact0", "concat", graphHandle);
    concat_interact0.createNode();


    // generate graph
    SetTestFileName("dlrm_Concat_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor bot_l_4_relu_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, bot_l_4_relu_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bot_l_4_relu_output", typed_data, bot_l_4_relu_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, bot_l_4_relu_output->getDramAddress(), bot_l_4_relu_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(bot_l_4_relu_output_tr_info);

    // init tensor emb_0_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_output", typed_data, emb_0_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_output->getDramAddress(), emb_0_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_output_tr_info);

    // init tensor emb_1_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_1_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_1_embbag_output", typed_data, emb_1_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_1_embbag_output->getDramAddress(), emb_1_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_1_embbag_output_tr_info);

    // init tensor emb_2_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_2_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_2_embbag_output", typed_data, emb_2_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_2_embbag_output->getDramAddress(), emb_2_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_2_embbag_output_tr_info);

    // init tensor emb_3_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_3_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_3_embbag_output", typed_data, emb_3_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_3_embbag_output->getDramAddress(), emb_3_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_3_embbag_output_tr_info);

    // init tensor emb_4_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_4_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_4_embbag_output", typed_data, emb_4_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_4_embbag_output->getDramAddress(), emb_4_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_4_embbag_output_tr_info);

    // init tensor emb_5_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_5_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_5_embbag_output", typed_data, emb_5_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_5_embbag_output->getDramAddress(), emb_5_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_5_embbag_output_tr_info);

    // init tensor emb_6_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_6_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_6_embbag_output", typed_data, emb_6_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_6_embbag_output->getDramAddress(), emb_6_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_6_embbag_output_tr_info);

    // init tensor emb_7_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_7_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_7_embbag_output", typed_data, emb_7_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_7_embbag_output->getDramAddress(), emb_7_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_7_embbag_output_tr_info);

    // init tensor emb_8_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_8_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_8_embbag_output", typed_data, emb_8_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_8_embbag_output->getDramAddress(), emb_8_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_8_embbag_output_tr_info);

    // init tensor emb_9_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_9_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_9_embbag_output", typed_data, emb_9_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_9_embbag_output->getDramAddress(), emb_9_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_9_embbag_output_tr_info);

    // init tensor emb_10_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_10_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_10_embbag_output", typed_data, emb_10_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_10_embbag_output->getDramAddress(), emb_10_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_10_embbag_output_tr_info);

    // init tensor emb_11_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_11_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_11_embbag_output", typed_data, emb_11_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_11_embbag_output->getDramAddress(), emb_11_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_11_embbag_output_tr_info);

    // init tensor emb_12_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_12_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_12_embbag_output", typed_data, emb_12_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_12_embbag_output->getDramAddress(), emb_12_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_12_embbag_output_tr_info);

    // init tensor emb_13_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_13_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_13_embbag_output", typed_data, emb_13_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_13_embbag_output->getDramAddress(), emb_13_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_13_embbag_output_tr_info);

    // init tensor emb_14_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_14_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_14_embbag_output", typed_data, emb_14_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_14_embbag_output->getDramAddress(), emb_14_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_14_embbag_output_tr_info);

    // init tensor emb_15_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_15_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_15_embbag_output", typed_data, emb_15_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_15_embbag_output->getDramAddress(), emb_15_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_15_embbag_output_tr_info);

    // init tensor emb_16_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_16_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_16_embbag_output", typed_data, emb_16_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_16_embbag_output->getDramAddress(), emb_16_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_16_embbag_output_tr_info);

    // init tensor emb_17_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_17_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_17_embbag_output", typed_data, emb_17_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_17_embbag_output->getDramAddress(), emb_17_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_17_embbag_output_tr_info);

    // init tensor emb_18_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_18_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_18_embbag_output", typed_data, emb_18_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_18_embbag_output->getDramAddress(), emb_18_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_18_embbag_output_tr_info);

    // init tensor emb_19_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_19_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_19_embbag_output", typed_data, emb_19_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_19_embbag_output->getDramAddress(), emb_19_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_19_embbag_output_tr_info);

    // init tensor emb_20_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_20_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_20_embbag_output", typed_data, emb_20_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_20_embbag_output->getDramAddress(), emb_20_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_20_embbag_output_tr_info);

    // init tensor emb_21_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_21_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_21_embbag_output", typed_data, emb_21_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_21_embbag_output->getDramAddress(), emb_21_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_21_embbag_output_tr_info);

    // init tensor emb_22_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_22_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_22_embbag_output", typed_data, emb_22_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_22_embbag_output->getDramAddress(), emb_22_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_22_embbag_output_tr_info);

    // init tensor emb_23_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_23_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_23_embbag_output", typed_data, emb_23_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_23_embbag_output->getDramAddress(), emb_23_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_23_embbag_output_tr_info);

    // init tensor emb_24_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_24_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_24_embbag_output", typed_data, emb_24_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_24_embbag_output->getDramAddress(), emb_24_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_24_embbag_output_tr_info);

    // init tensor emb_25_embbag_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_25_embbag_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_25_embbag_output", typed_data, emb_25_embbag_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_25_embbag_output->getDramAddress(), emb_25_embbag_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_25_embbag_output_tr_info);

    // List outputs
    graph_outputs.push_back(concat_interact0_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor concat_interact0_output from file
    {
        bfloat16* ref_arr = new bfloat16[concat_interact0_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, concat_interact0_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/concat_interact0_output", ref_arr, concat_interact0_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(concat_interact0_output->getDramAddress(), data, concat_interact0_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: concat_interact0_output");
            validateResult(ref_arr, typed_data, concat_interact0_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: concat_interact0_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Reshape_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * reshape_bmm_fwd node
     * inputs: [concat_interact0_output[128, 864](dtype=bf16)]
     * output: [reshape_bmm_output(128, 27, 32)(dtype=bf16)]
     *************/

    // create concat_interact0_output tensor
    pManagedTensor concat_interact0_output = ManagedTensor::createManagedTensor("concat_interact0_output", {128, 864}, syn_type_bf16, true, this);
    synLaunchTensorInfo concat_interact0_output_tr_info = {"concat_interact0_output",
                                                           concat_interact0_output->getDramAddress()};

    // create reshape_bmm_output tensor
    pManagedTensor reshape_bmm_output = ManagedTensor::createManagedTensor("reshape_bmm_output", {128, 27, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_bmm_output_tr_info = {"reshape_bmm_output", reshape_bmm_output->getDramAddress()};
    ManagedNode reshape_bmm_fwd({concat_interact0_output}, {reshape_bmm_output}, "reshape_bmm_fwd", "reshape", graphHandle);
    reshape_bmm_fwd.createNode();


    // generate graph
    SetTestFileName("dlrm_Reshape_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor concat_interact0_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, concat_interact0_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/concat_interact0_output", typed_data, concat_interact0_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, concat_interact0_output->getDramAddress(), concat_interact0_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(concat_interact0_output_tr_info);

    // List outputs
    graph_outputs.push_back(reshape_bmm_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor reshape_bmm_output from file
    {
        bfloat16* ref_arr = new bfloat16[reshape_bmm_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_bmm_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_bmm_output", ref_arr, reshape_bmm_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(reshape_bmm_output->getDramAddress(), data, reshape_bmm_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: reshape_bmm_output");
            validateResult(ref_arr, typed_data, reshape_bmm_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: reshape_bmm_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_BMM_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * bmm_interact node
     * inputs: [reshape_bmm_output[128, 27, 32](dtype=bf16), reshape_bmm_output[128, 27, 32](dtype=bf16)]
     * output: [bmm_interact_output(128, 27, 27)(dtype=bf16)]
     *************/
    synGEMMParams bmm_interact_kernel_params;
    bmm_interact_kernel_params.transpose_a = false;
    bmm_interact_kernel_params.transpose_b = true;

    // create reshape_bmm_output tensor
    pManagedTensor reshape_bmm_output = ManagedTensor::createManagedTensor("reshape_bmm_output", {128, 27, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_bmm_output_tr_info = {"reshape_bmm_output", reshape_bmm_output->getDramAddress()};

    // create bmm_interact_output tensor
    pManagedTensor bmm_interact_output = ManagedTensor::createManagedTensor("bmm_interact_output", {128, 27, 27}, syn_type_bf16, true, this);
    synLaunchTensorInfo bmm_interact_output_tr_info = {"bmm_interact_output", bmm_interact_output->getDramAddress()};
    ManagedNodeWithParams<synGEMMParams> bmm_interact({reshape_bmm_output, reshape_bmm_output}, {bmm_interact_output}, "bmm_interact", "batch_gemm", graphHandle, bmm_interact_kernel_params);
    bmm_interact.createNode();


    // generate graph
    SetTestFileName("dlrm_BMM_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor reshape_bmm_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_bmm_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_bmm_output", typed_data, reshape_bmm_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, reshape_bmm_output->getDramAddress(), reshape_bmm_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(reshape_bmm_output_tr_info);

    // List outputs
    graph_outputs.push_back(bmm_interact_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor bmm_interact_output from file
    {
        bfloat16* ref_arr = new bfloat16[bmm_interact_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, bmm_interact_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bmm_interact_output", ref_arr, bmm_interact_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(bmm_interact_output->getDramAddress(), data, bmm_interact_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: bmm_interact_output");
            validateResult(ref_arr, typed_data, bmm_interact_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: bmm_interact_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_gather_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * tril_indices_fwd node
     * inputs: [reshape_tril_output[128, 729](dtype=bf16), tril_indices[351](dtype=int32)]
     * output: [tril_indices_output(128, 351)(dtype=bf16)]
     *************/
    ns_ScatterKernel::Params tril_indices_fwd_kernel_params;
    tril_indices_fwd_kernel_params.axis = 0;

    // create reshape_tril_output tensor
    pManagedTensor reshape_tril_output = ManagedTensor::createManagedTensor("reshape_tril_output", {128, 729}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_tril_output_tr_info = {"reshape_tril_output", reshape_tril_output->getDramAddress()};

    // create tril_indices tensor
    pManagedTensor tril_indices = ManagedTensor::createManagedTensor("tril_indices", {351,}, syn_type_int32, true, this);
    synLaunchTensorInfo tril_indices_tr_info = {"tril_indices", tril_indices->getDramAddress()};

    // create tril_indices_output tensor
    pManagedTensor tril_indices_output = ManagedTensor::createManagedTensor("tril_indices_output", {128, 351}, syn_type_bf16, true, this);
    synLaunchTensorInfo tril_indices_output_tr_info = {"tril_indices_output", tril_indices_output->getDramAddress()};
    ManagedNodeWithParams<ns_ScatterKernel::Params> tril_indices_fwd({reshape_tril_output, tril_indices}, {tril_indices_output}, "tril_indices_fwd", "gather_fwd_bf16", graphHandle, tril_indices_fwd_kernel_params);
    tril_indices_fwd.createNode();


    // generate graph
    SetTestFileName("dlrm_gather_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor reshape_tril_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_tril_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_tril_output", typed_data, reshape_tril_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, reshape_tril_output->getDramAddress(), reshape_tril_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(reshape_tril_output_tr_info);

    // init tensor tril_indices from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, tril_indices->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/tril_indices", typed_data, tril_indices->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, tril_indices->getDramAddress(), tril_indices->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(tril_indices_tr_info);

    // List outputs
    graph_outputs.push_back(tril_indices_output_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor tril_indices_output from file
    {
        bfloat16* ref_arr = new bfloat16[tril_indices_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, tril_indices_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/tril_indices_output", ref_arr, tril_indices_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(tril_indices_output->getDramAddress(), data, tril_indices_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: tril_indices_output");
            validateResult(ref_arr, typed_data, tril_indices_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: tril_indices_output");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, DISABLED_dlrm_BinaryCrossEntropyLoss_fwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * binary_cross_entropy_loss_fwd node
     * inputs: [top_l_10_linear_output[1, 128](dtype=bf16), target[1, 128](dtype=bf16)]
     * output: [binary_cross_entropy_loss_output(1,)(dtype=bf16), binary_cross_entropy_loss_saved_sigmoid(1, 128)(dtype=bf16)]
     *************/
    ns_BinaryCrossEntropy::Params binary_cross_entropy_loss_fwd_kernel_params;
    binary_cross_entropy_loss_fwd_kernel_params.isWeightsUsed = false;
    binary_cross_entropy_loss_fwd_kernel_params.mode = CROSS_ENTROPY_MODE_MEAN;

    // create top_l_10_linear_output tensor
    pManagedTensor top_l_10_linear_output = ManagedTensor::createManagedTensor("top_l_10_linear_output", {1, 128}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_10_linear_output_tr_info = {"top_l_10_linear_output",
                                                          top_l_10_linear_output->getDramAddress()};

    // create target tensor
    pManagedTensor target = ManagedTensor::createManagedTensor("target", {1, 128}, syn_type_bf16, true, this);
    synLaunchTensorInfo target_tr_info = {"target", target->getDramAddress()};

    // create binary_cross_entropy_loss_output tensor
    pManagedTensor binary_cross_entropy_loss_output = ManagedTensor::createManagedTensor("binary_cross_entropy_loss_output", {1,}, syn_type_bf16, true, this);
    synLaunchTensorInfo binary_cross_entropy_loss_output_tr_info = {"binary_cross_entropy_loss_output",
                                                                    binary_cross_entropy_loss_output->getDramAddress()};

    // create binary_cross_entropy_loss_saved_sigmoid tensor
    pManagedTensor binary_cross_entropy_loss_saved_sigmoid = ManagedTensor::createManagedTensor("binary_cross_entropy_loss_saved_sigmoid", {1, 128}, syn_type_bf16, true, this);
    synLaunchTensorInfo binary_cross_entropy_loss_saved_sigmoid_tr_info = {
        "binary_cross_entropy_loss_saved_sigmoid",
        binary_cross_entropy_loss_saved_sigmoid->getDramAddress()};
    ManagedNodeWithParams<ns_BinaryCrossEntropy::Params> binary_cross_entropy_loss_fwd({top_l_10_linear_output, target}, {binary_cross_entropy_loss_output, binary_cross_entropy_loss_saved_sigmoid}, "binary_cross_entropy_loss_fwd", "binary_cross_entropy_fwd_bf16", graphHandle, binary_cross_entropy_loss_fwd_kernel_params);
    binary_cross_entropy_loss_fwd.createNode();


    // generate graph
    SetTestFileName("dlrm_BinaryCrossEntropyLoss_fwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor top_l_10_linear_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_output", typed_data, top_l_10_linear_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_10_linear_output->getDramAddress(), top_l_10_linear_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_10_linear_output_tr_info);

    // init tensor target from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, target->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/target", typed_data, target->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, target->getDramAddress(), target->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(target_tr_info);

    // List outputs
    graph_outputs.push_back(binary_cross_entropy_loss_output_tr_info);
    graph_outputs.push_back(binary_cross_entropy_loss_saved_sigmoid_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor binary_cross_entropy_loss_output from file
    {
        bfloat16* ref_arr = new bfloat16[binary_cross_entropy_loss_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, binary_cross_entropy_loss_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/binary_cross_entropy_loss_output", ref_arr, binary_cross_entropy_loss_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(binary_cross_entropy_loss_output->getDramAddress(), data, binary_cross_entropy_loss_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: binary_cross_entropy_loss_output");
            validateResult(ref_arr, typed_data, binary_cross_entropy_loss_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: binary_cross_entropy_loss_output");
        }
        delete[] ref_arr;
    }
    // data check for tensor binary_cross_entropy_loss_saved_sigmoid from file
    {
        bfloat16* ref_arr = new bfloat16[binary_cross_entropy_loss_saved_sigmoid->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, binary_cross_entropy_loss_saved_sigmoid->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/binary_cross_entropy_loss_saved_sigmoid", ref_arr, binary_cross_entropy_loss_saved_sigmoid->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(binary_cross_entropy_loss_saved_sigmoid->getDramAddress(), data, binary_cross_entropy_loss_saved_sigmoid->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: binary_cross_entropy_loss_saved_sigmoid");
            validateResult(ref_arr, typed_data, binary_cross_entropy_loss_saved_sigmoid->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: binary_cross_entropy_loss_saved_sigmoid");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_BinaryCrossEntropyLoss_bwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * binary_cross_entropy_loss_bwd node
     * inputs: [top_l_10_linear_output[1, 128](dtype=bf16), target[1, 128](dtype=bf16)]
     * output: [binary_cross_entropy_loss_grad_input(1, 128)(dtype=bf16)]
     *************/
    ns_BinaryCrossEntropy::Params binary_cross_entropy_loss_bwd_kernel_params;
    binary_cross_entropy_loss_bwd_kernel_params.isWeightsUsed = false;
    binary_cross_entropy_loss_bwd_kernel_params.mode = CROSS_ENTROPY_MODE_MEAN;

    // create top_l_10_linear_output tensor
    pManagedTensor top_l_10_linear_output = ManagedTensor::createManagedTensor("top_l_10_linear_output", {1, 128}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_10_linear_output_tr_info = {"top_l_10_linear_output",
                                                          top_l_10_linear_output->getDramAddress()};

    // create target tensor
    pManagedTensor target = ManagedTensor::createManagedTensor("target", {1, 128}, syn_type_bf16, true, this);
    synLaunchTensorInfo target_tr_info = {"target", target->getDramAddress()};

    // create binary_cross_entropy_loss_grad_input tensor
    pManagedTensor binary_cross_entropy_loss_grad_input = ManagedTensor::createManagedTensor("binary_cross_entropy_loss_grad_input", {1, 128}, syn_type_bf16, true, this);
    synLaunchTensorInfo binary_cross_entropy_loss_grad_input_tr_info = {
        "binary_cross_entropy_loss_grad_input",
        binary_cross_entropy_loss_grad_input->getDramAddress()};
    ManagedNodeWithParams<ns_BinaryCrossEntropy::Params> binary_cross_entropy_loss_bwd({top_l_10_linear_output, target}, {binary_cross_entropy_loss_grad_input}, "binary_cross_entropy_loss_bwd", "binary_cross_entropy_bwd_bf16", graphHandle, binary_cross_entropy_loss_bwd_kernel_params);
    binary_cross_entropy_loss_bwd.createNode();


    // generate graph
    SetTestFileName("dlrm_BinaryCrossEntropyLoss_bwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor top_l_10_linear_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_output", typed_data, top_l_10_linear_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_10_linear_output->getDramAddress(), top_l_10_linear_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_10_linear_output_tr_info);

    // init tensor target from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, target->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/target", typed_data, target->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, target->getDramAddress(), target->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(target_tr_info);

    // List outputs
    graph_outputs.push_back(binary_cross_entropy_loss_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor binary_cross_entropy_loss_grad_input from file
    {
        bfloat16* ref_arr = new bfloat16[binary_cross_entropy_loss_grad_input->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, binary_cross_entropy_loss_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/binary_cross_entropy_loss_grad_input", ref_arr, binary_cross_entropy_loss_grad_input->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(binary_cross_entropy_loss_grad_input->getDramAddress(), data, binary_cross_entropy_loss_grad_input->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: binary_cross_entropy_loss_grad_input");
            validateResult(ref_arr, typed_data, binary_cross_entropy_loss_grad_input->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: binary_cross_entropy_loss_grad_input");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Linear_dedx)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * top_l_10_linear_dedx node
     * inputs: [binary_cross_entropy_loss_grad_input(128, 1, 1, 1)(dtype=bf16), top_l_10_linear_weight(1, 1, 1024, 1)(dtype=bf16)]
     * output: [top_l_10_linear_grad_input(128, 1, 1, 1024)(dtype=bf16)]
     *************/
    synConvolutionParams top_l_10_linear_dedx_kernel_params;
    top_l_10_linear_dedx_kernel_params.dH = 1;
    top_l_10_linear_dedx_kernel_params.dW = 1;
    top_l_10_linear_dedx_kernel_params.kH = 1;
    top_l_10_linear_dedx_kernel_params.kW = 1;
    top_l_10_linear_dedx_kernel_params.padT = 0;
    top_l_10_linear_dedx_kernel_params.padB = 0;
    top_l_10_linear_dedx_kernel_params.padL = 0;
    top_l_10_linear_dedx_kernel_params.padR = 0;
    top_l_10_linear_dedx_kernel_params.dilH = 1;
    top_l_10_linear_dedx_kernel_params.dilW = 1;

    // create binary_cross_entropy_loss_grad_input tensor
    pManagedTensor binary_cross_entropy_loss_grad_input = ManagedTensor::createManagedTensor("binary_cross_entropy_loss_grad_input", {128, 1, 1, 1}, syn_type_bf16, true, this);
    synLaunchTensorInfo binary_cross_entropy_loss_grad_input_tr_info = {
        "binary_cross_entropy_loss_grad_input",
        binary_cross_entropy_loss_grad_input->getDramAddress()};

    // create top_l_10_linear_weight tensor
    pManagedTensor top_l_10_linear_weight = ManagedTensor::createManagedTensor("top_l_10_linear_weight", {1, 1, 1024, 1}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_10_linear_weight_tr_info = {"top_l_10_linear_weight",
                                                          top_l_10_linear_weight->getDramAddress()};

    // create top_l_10_linear_grad_input tensor
    pManagedTensor top_l_10_linear_grad_input = ManagedTensor::createManagedTensor("top_l_10_linear_grad_input", {128, 1, 1, 1024}, syn_type_bf16, true, this);
    synLaunchTensorInfo                         top_l_10_linear_grad_input_tr_info = {"top_l_10_linear_grad_input",
                                                              top_l_10_linear_grad_input->getDramAddress()};
    ManagedNodeWithParams<synConvolutionParams> top_l_10_linear_dedx({binary_cross_entropy_loss_grad_input, top_l_10_linear_weight}, {top_l_10_linear_grad_input}, "top_l_10_linear_dedx", "dedx", graphHandle, top_l_10_linear_dedx_kernel_params);
    top_l_10_linear_dedx.createNode();


    // generate graph
    SetTestFileName("dlrm_Linear_dedx_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor binary_cross_entropy_loss_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, binary_cross_entropy_loss_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/binary_cross_entropy_loss_grad_input", typed_data, binary_cross_entropy_loss_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, binary_cross_entropy_loss_grad_input->getDramAddress(), binary_cross_entropy_loss_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(binary_cross_entropy_loss_grad_input_tr_info);

    // init tensor top_l_10_linear_weight from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_weight->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_weight", typed_data, top_l_10_linear_weight->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_10_linear_weight->getDramAddress(), top_l_10_linear_weight->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_10_linear_weight_tr_info);

    // List outputs
    graph_outputs.push_back(top_l_10_linear_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor top_l_10_linear_grad_input from file
    {
        bfloat16* ref_arr = new bfloat16[top_l_10_linear_grad_input->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_grad_input", ref_arr, top_l_10_linear_grad_input->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(top_l_10_linear_grad_input->getDramAddress(), data, top_l_10_linear_grad_input->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: top_l_10_linear_grad_input");
            validateResult(ref_arr, typed_data, top_l_10_linear_grad_input->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: top_l_10_linear_grad_input");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Linear_dedw)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * top_l_10_linear_dedw node
     * inputs: [binary_cross_entropy_loss_grad_input(128, 1, 1, 1)(dtype=bf16), top_l_8_relu_output(128, 1, 1, 1024)(dtype=bf16)]
     * output: [top_l_10_linear_weight_grad(1, 1, 1024, 1)(dtype=float32)]
     *************/
    synConvolutionParams top_l_10_linear_dedw_kernel_params;
    top_l_10_linear_dedw_kernel_params.dH = 1;
    top_l_10_linear_dedw_kernel_params.dW = 1;
    top_l_10_linear_dedw_kernel_params.kH = 1;
    top_l_10_linear_dedw_kernel_params.kW = 1;
    top_l_10_linear_dedw_kernel_params.padT = 0;
    top_l_10_linear_dedw_kernel_params.padB = 0;
    top_l_10_linear_dedw_kernel_params.padL = 0;
    top_l_10_linear_dedw_kernel_params.padR = 0;
    top_l_10_linear_dedw_kernel_params.dilH = 1;
    top_l_10_linear_dedw_kernel_params.dilW = 1;

    // create binary_cross_entropy_loss_grad_input tensor
    pManagedTensor binary_cross_entropy_loss_grad_input = ManagedTensor::createManagedTensor("binary_cross_entropy_loss_grad_input", {128, 1, 1, 1}, syn_type_bf16, true, this);
    synLaunchTensorInfo binary_cross_entropy_loss_grad_input_tr_info = {
        "binary_cross_entropy_loss_grad_input",
        binary_cross_entropy_loss_grad_input->getDramAddress()};

    // create top_l_8_relu_output tensor
    pManagedTensor top_l_8_relu_output = ManagedTensor::createManagedTensor("top_l_8_relu_output", {128, 1, 1, 1024}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_8_relu_output_tr_info = {"top_l_8_relu_output", top_l_8_relu_output->getDramAddress()};

    // create top_l_10_linear_weight_grad tensor
    pManagedTensor top_l_10_linear_weight_grad = ManagedTensor::createManagedTensor("top_l_10_linear_weight_grad", {1, 1, 1024, 1}, syn_type_single, true, this);
    synLaunchTensorInfo                         top_l_10_linear_weight_grad_tr_info = {"top_l_10_linear_weight_grad",
                                                               top_l_10_linear_weight_grad->getDramAddress()};
    ManagedNodeWithParams<synConvolutionParams> top_l_10_linear_dedw({binary_cross_entropy_loss_grad_input, top_l_8_relu_output}, {top_l_10_linear_weight_grad}, "top_l_10_linear_dedw", "dedw", graphHandle, top_l_10_linear_dedw_kernel_params);
    top_l_10_linear_dedw.createNode();


    // generate graph
    SetTestFileName("dlrm_Linear_dedw_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor binary_cross_entropy_loss_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, binary_cross_entropy_loss_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/binary_cross_entropy_loss_grad_input", typed_data, binary_cross_entropy_loss_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, binary_cross_entropy_loss_grad_input->getDramAddress(), binary_cross_entropy_loss_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(binary_cross_entropy_loss_grad_input_tr_info);

    // init tensor top_l_8_relu_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_8_relu_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_8_relu_output", typed_data, top_l_8_relu_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_8_relu_output->getDramAddress(), top_l_8_relu_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_8_relu_output_tr_info);

    // List outputs
    graph_outputs.push_back(top_l_10_linear_weight_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor top_l_10_linear_weight_grad from file
    {
        float* ref_arr = new float[top_l_10_linear_weight_grad->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_weight_grad->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_weight_grad", ref_arr, top_l_10_linear_weight_grad->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(top_l_10_linear_weight_grad->getDramAddress(), data, top_l_10_linear_weight_grad->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: top_l_10_linear_weight_grad");
            validateResult(ref_arr, typed_data, top_l_10_linear_weight_grad->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: top_l_10_linear_weight_grad");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Linear_dedb)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * top_l_10_linear_dedb node
     * inputs: [binary_cross_entropy_loss_grad_input[128, 1](dtype=bf16)]
     * output: [top_l_10_linear_bias_grad(1,)(dtype=float32)]
     *************/
    ns_Reduction::Params top_l_10_linear_dedb_kernel_params;
    top_l_10_linear_dedb_kernel_params.reductionDimension = 1;

    // create binary_cross_entropy_loss_grad_input tensor
    pManagedTensor binary_cross_entropy_loss_grad_input = ManagedTensor::createManagedTensor("binary_cross_entropy_loss_grad_input", {128, 1}, syn_type_bf16, true, this);
    synLaunchTensorInfo binary_cross_entropy_loss_grad_input_tr_info = {
        "binary_cross_entropy_loss_grad_input",
        binary_cross_entropy_loss_grad_input->getDramAddress()};

    // create top_l_10_linear_bias_grad tensor
    pManagedTensor top_l_10_linear_bias_grad = ManagedTensor::createManagedTensor("top_l_10_linear_bias_grad", {1,}, syn_type_single, true, this);
    synLaunchTensorInfo                         top_l_10_linear_bias_grad_tr_info = {"top_l_10_linear_bias_grad",
                                                             top_l_10_linear_bias_grad->getDramAddress()};
    ManagedNodeWithParams<ns_Reduction::Params> top_l_10_linear_dedb({binary_cross_entropy_loss_grad_input}, {top_l_10_linear_bias_grad}, "top_l_10_linear_dedb", "reduce_sum_fwd_bf16", graphHandle, top_l_10_linear_dedb_kernel_params);
    top_l_10_linear_dedb.createNode();


    // generate graph
    SetTestFileName("dlrm_Linear_dedb_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor binary_cross_entropy_loss_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, binary_cross_entropy_loss_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/binary_cross_entropy_loss_grad_input", typed_data, binary_cross_entropy_loss_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, binary_cross_entropy_loss_grad_input->getDramAddress(), binary_cross_entropy_loss_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(binary_cross_entropy_loss_grad_input_tr_info);

    // List outputs
    graph_outputs.push_back(top_l_10_linear_bias_grad_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor top_l_10_linear_bias_grad from file
    {
        float* ref_arr = new float[top_l_10_linear_bias_grad->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_bias_grad->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_bias_grad", ref_arr, top_l_10_linear_bias_grad->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(top_l_10_linear_bias_grad->getDramAddress(), data, top_l_10_linear_bias_grad->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: top_l_10_linear_bias_grad");
            validateResult(ref_arr, typed_data, top_l_10_linear_bias_grad->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: top_l_10_linear_bias_grad");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_ReLU_bwd)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * top_l_8_relu_bwd node
     * inputs: [top_l_10_linear_grad_input(128, 1, 1, 1024)(dtype=bf16), top_l_8_relu_output(128, 1, 1, 1024)(dtype=bf16)]
     * output: [top_l_8_relu_grad_input(128, 1, 1, 1024)(dtype=bf16)]
     *************/

    // create top_l_10_linear_grad_input tensor
    pManagedTensor top_l_10_linear_grad_input = ManagedTensor::createManagedTensor("top_l_10_linear_grad_input", {128, 1, 1, 1024}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_10_linear_grad_input_tr_info = {"top_l_10_linear_grad_input",
                                                              top_l_10_linear_grad_input->getDramAddress()};

    // create top_l_8_relu_output tensor
    pManagedTensor top_l_8_relu_output = ManagedTensor::createManagedTensor("top_l_8_relu_output", {128, 1, 1, 1024}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_8_relu_output_tr_info = {"top_l_8_relu_output", top_l_8_relu_output->getDramAddress()};

    // create top_l_8_relu_grad_input tensor
    pManagedTensor top_l_8_relu_grad_input = ManagedTensor::createManagedTensor("top_l_8_relu_grad_input", {128, 1, 1, 1024}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_8_relu_grad_input_tr_info = {"top_l_8_relu_grad_input",
                                                           top_l_8_relu_grad_input->getDramAddress()};
    ManagedNode top_l_8_relu_bwd({top_l_10_linear_grad_input, top_l_8_relu_output}, {top_l_8_relu_grad_input}, "top_l_8_relu_bwd", "relu_bwd_bf16", graphHandle);
    top_l_8_relu_bwd.createNode();


    // generate graph
    SetTestFileName("dlrm_ReLU_bwd_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor top_l_10_linear_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_10_linear_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_10_linear_grad_input", typed_data, top_l_10_linear_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_10_linear_grad_input->getDramAddress(), top_l_10_linear_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_10_linear_grad_input_tr_info);

    // init tensor top_l_8_relu_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_8_relu_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_8_relu_output", typed_data, top_l_8_relu_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_8_relu_output->getDramAddress(), top_l_8_relu_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_8_relu_output_tr_info);

    // List outputs
    graph_outputs.push_back(top_l_8_relu_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor top_l_8_relu_grad_input from file
    {
        bfloat16* ref_arr = new bfloat16[top_l_8_relu_grad_input->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_8_relu_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_8_relu_grad_input", ref_arr, top_l_8_relu_grad_input->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(top_l_8_relu_grad_input->getDramAddress(), data, top_l_8_relu_grad_input->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: top_l_8_relu_grad_input");
            validateResult(ref_arr, typed_data, top_l_8_relu_grad_input->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: top_l_8_relu_grad_input");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Concat_dedx)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * concat_interact1_dedx node
     * inputs: [top_l_0_linear_grad_input[128, 383](dtype=bf16)]
     * output: [concat_interact1_grad_input0(128, 32)(dtype=bf16), concat_interact1_grad_input1(128, 351)(dtype=bf16)]
     *************/
    unsigned concat_interact1_dedx_kernel_params;
    concat_interact1_dedx_kernel_params = 0;

    // create top_l_0_linear_grad_input tensor
    pManagedTensor top_l_0_linear_grad_input = ManagedTensor::createManagedTensor("top_l_0_linear_grad_input", {128, 383}, syn_type_bf16, true, this);
    synLaunchTensorInfo top_l_0_linear_grad_input_tr_info = {"top_l_0_linear_grad_input",
                                                             top_l_0_linear_grad_input->getDramAddress()};

    // create concat_interact1_grad_input0 tensor
    pManagedTensor concat_interact1_grad_input0 = ManagedTensor::createManagedTensor("concat_interact1_grad_input0", {128, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo concat_interact1_grad_input0_tr_info = {"concat_interact1_grad_input0",
                                                                concat_interact1_grad_input0->getDramAddress()};

    // create concat_interact1_grad_input1 tensor
    pManagedTensor concat_interact1_grad_input1 = ManagedTensor::createManagedTensor("concat_interact1_grad_input1", {128, 351}, syn_type_bf16, true, this);
    synLaunchTensorInfo             concat_interact1_grad_input1_tr_info = {"concat_interact1_grad_input1",
                                                                concat_interact1_grad_input1->getDramAddress()};
    ManagedNodeWithParams<unsigned> concat_interact1_dedx({top_l_0_linear_grad_input}, {concat_interact1_grad_input0, concat_interact1_grad_input1}, "concat_interact1_dedx", "split", graphHandle, concat_interact1_dedx_kernel_params);
    concat_interact1_dedx.createNode();


    // generate graph
    SetTestFileName("dlrm_Concat_dedx_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor top_l_0_linear_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, top_l_0_linear_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/top_l_0_linear_grad_input", typed_data, top_l_0_linear_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, top_l_0_linear_grad_input->getDramAddress(), top_l_0_linear_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(top_l_0_linear_grad_input_tr_info);

    // List outputs
    graph_outputs.push_back(concat_interact1_grad_input0_tr_info);
    graph_outputs.push_back(concat_interact1_grad_input1_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor concat_interact1_grad_input0 from file
    {
        bfloat16* ref_arr = new bfloat16[concat_interact1_grad_input0->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, concat_interact1_grad_input0->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/concat_interact1_grad_input0", ref_arr, concat_interact1_grad_input0->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(concat_interact1_grad_input0->getDramAddress(), data, concat_interact1_grad_input0->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: concat_interact1_grad_input0");
            validateResult(ref_arr, typed_data, concat_interact1_grad_input0->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: concat_interact1_grad_input0");
        }
        delete[] ref_arr;
    }
    // data check for tensor concat_interact1_grad_input1 from file
    {
        bfloat16* ref_arr = new bfloat16[concat_interact1_grad_input1->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, concat_interact1_grad_input1->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/concat_interact1_grad_input1", ref_arr, concat_interact1_grad_input1->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(concat_interact1_grad_input1->getDramAddress(), data, concat_interact1_grad_input1->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: concat_interact1_grad_input1");
            validateResult(ref_arr, typed_data, concat_interact1_grad_input1->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: concat_interact1_grad_input1");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_gather_dedx)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * tril_indices_dedx node
     * inputs: [tril_grad_buffer[128, 729](dtype=bf16), tril_indices_broadcast[128, 351](dtype=int32), concat_interact1_grad_input1[128, 351](dtype=bf16)]
     * output: [tril_indices_grad_input(128, 729)(dtype=bf16)]
     *************/
    ns_ScatterKernel::Params tril_indices_dedx_kernel_params;
    tril_indices_dedx_kernel_params.axis = 0;

    // create tril_grad_buffer tensor
    pManagedTensor tril_grad_buffer = ManagedTensor::createManagedTensor("tril_grad_buffer", {128, 729}, syn_type_bf16, true, this);
    synLaunchTensorInfo tril_grad_buffer_tr_info = {"tril_grad_buffer", tril_grad_buffer->getDramAddress()};

    // create tril_indices_broadcast tensor
    pManagedTensor tril_indices_broadcast = ManagedTensor::createManagedTensor("tril_indices_broadcast", {128, 351}, syn_type_int32, true, this);
    synLaunchTensorInfo tril_indices_broadcast_tr_info = {"tril_indices_broadcast",
                                                          tril_indices_broadcast->getDramAddress()};

    // create concat_interact1_grad_input1 tensor
    pManagedTensor concat_interact1_grad_input1 = ManagedTensor::createManagedTensor("concat_interact1_grad_input1", {128, 351}, syn_type_bf16, true, this);
    synLaunchTensorInfo concat_interact1_grad_input1_tr_info = {"concat_interact1_grad_input1",
                                                                concat_interact1_grad_input1->getDramAddress()};

    // create tril_indices_grad_input tensor
    pManagedTensor tril_indices_grad_input = ManagedTensor::createManagedTensor("tril_indices_grad_input", {128, 729}, syn_type_bf16, true, this);
    synLaunchTensorInfo                             tril_indices_grad_input_tr_info = {"tril_indices_grad_input",
                                                           tril_indices_grad_input->getDramAddress()};
    ManagedNodeWithParams<ns_ScatterKernel::Params> tril_indices_dedx({tril_grad_buffer, tril_indices_broadcast, concat_interact1_grad_input1}, {tril_indices_grad_input}, "tril_indices_dedx", "gather_bwd_bf16", graphHandle, tril_indices_dedx_kernel_params);
    tril_indices_dedx.createNode();


    // generate graph
    SetTestFileName("dlrm_gather_dedx_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor tril_grad_buffer from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, tril_grad_buffer->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        memset(data, 0, tril_grad_buffer->getTotalSizeInBytes());
        downloadTensorData(data, tril_grad_buffer->getDramAddress(), tril_grad_buffer->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(tril_grad_buffer_tr_info);

    // init tensor tril_indices_broadcast from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, tril_indices_broadcast->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/tril_indices_broadcast", typed_data, tril_indices_broadcast->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, tril_indices_broadcast->getDramAddress(), tril_indices_broadcast->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(tril_indices_broadcast_tr_info);

    // init tensor concat_interact1_grad_input1 from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, concat_interact1_grad_input1->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/concat_interact1_grad_input1", typed_data, concat_interact1_grad_input1->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, concat_interact1_grad_input1->getDramAddress(), concat_interact1_grad_input1->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(concat_interact1_grad_input1_tr_info);

    // List outputs
    graph_outputs.push_back(tril_indices_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor tril_indices_grad_input from file
    {
        bfloat16* ref_arr = new bfloat16[tril_indices_grad_input->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, tril_indices_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/tril_indices_grad_input", ref_arr, tril_indices_grad_input->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(tril_indices_grad_input->getDramAddress(), data, tril_indices_grad_input->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: tril_indices_grad_input");
            validateResult(ref_arr, typed_data, tril_indices_grad_input->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: tril_indices_grad_input");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_Reshape_dedx)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * reshape_tril_dedx node
     * inputs: [tril_indices_grad_input[128, 729](dtype=bf16)]
     * output: [reshape_tril_grad_input(128, 27, 27)(dtype=bf16)]
     *************/

    // create tril_indices_grad_input tensor
    pManagedTensor tril_indices_grad_input = ManagedTensor::createManagedTensor("tril_indices_grad_input", {128, 729}, syn_type_bf16, true, this);
    synLaunchTensorInfo tril_indices_grad_input_tr_info = {"tril_indices_grad_input",
                                                           tril_indices_grad_input->getDramAddress()};

    // create reshape_tril_grad_input tensor
    pManagedTensor reshape_tril_grad_input = ManagedTensor::createManagedTensor("reshape_tril_grad_input", {128, 27, 27}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_tril_grad_input_tr_info = {"reshape_tril_grad_input",
                                                           reshape_tril_grad_input->getDramAddress()};
    ManagedNode reshape_tril_dedx({tril_indices_grad_input}, {reshape_tril_grad_input}, "reshape_tril_dedx", "reshape", graphHandle);
    reshape_tril_dedx.createNode();


    // generate graph
    SetTestFileName("dlrm_Reshape_dedx_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor tril_indices_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, tril_indices_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/tril_indices_grad_input", typed_data, tril_indices_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, tril_indices_grad_input->getDramAddress(), tril_indices_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(tril_indices_grad_input_tr_info);

    // List outputs
    graph_outputs.push_back(reshape_tril_grad_input_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor reshape_tril_grad_input from file
    {
        bfloat16* ref_arr = new bfloat16[reshape_tril_grad_input->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_tril_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_tril_grad_input", ref_arr, reshape_tril_grad_input->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(reshape_tril_grad_input->getDramAddress(), data, reshape_tril_grad_input->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: reshape_tril_grad_input");
            validateResult(ref_arr, typed_data, reshape_tril_grad_input->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: reshape_tril_grad_input");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_BMM_dedx)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * bmm_interact_dedx node
     * inputs: [reshape_tril_grad_input[128, 27, 27](dtype=bf16), reshape_bmm_output[128, 27, 32](dtype=bf16)]
     * output: [bmm_interact_grad_input0(128, 27, 32)(dtype=float32)]
     *************/
    synGEMMParams bmm_interact_dedx_kernel_params;
    bmm_interact_dedx_kernel_params.transpose_a = false;
    bmm_interact_dedx_kernel_params.transpose_b = true;

    // create reshape_tril_grad_input tensor
    pManagedTensor reshape_tril_grad_input =
        ManagedTensor::createManagedTensor("reshape_tril_grad_input", {128, 27, 27}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_tril_grad_input_tr_info = {"reshape_tril_grad_input",
                                                           reshape_tril_grad_input->getDramAddress()};

    // create reshape_bmm_output tensor
    pManagedTensor reshape_bmm_output =
        ManagedTensor::createManagedTensor("reshape_bmm_output", {128, 32, 27}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_bmm_output_tr_info = {"reshape_bmm_output", reshape_bmm_output->getDramAddress()};

    // create bmm_interact_grad_input0 tensor
    pManagedTensor bmm_interact_grad_input0 =
        ManagedTensor::createManagedTensor("bmm_interact_grad_input0", {128, 27, 32}, syn_type_single, true, this);
    synLaunchTensorInfo                  bmm_interact_grad_input0_tr_info = {"bmm_interact_grad_input0",
                                                            bmm_interact_grad_input0->getDramAddress()};
    ManagedNodeWithParams<synGEMMParams> bmm_interact_dedx({reshape_tril_grad_input, reshape_bmm_output},
                                                           {bmm_interact_grad_input0},
                                                           "bmm_interact_dedx",
                                                           "batch_gemm_dedx",
                                                           graphHandle,
                                                           bmm_interact_dedx_kernel_params);
    bmm_interact_dedx.createNode();


    // generate graph
    SetTestFileName("dlrm_BMM_dedx_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor reshape_tril_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_tril_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_tril_grad_input", typed_data, reshape_tril_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, reshape_tril_grad_input->getDramAddress(), reshape_tril_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(reshape_tril_grad_input_tr_info);

    // init tensor reshape_bmm_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_bmm_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_bmm_output", typed_data, reshape_bmm_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, reshape_bmm_output->getDramAddress(), reshape_bmm_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(reshape_bmm_output_tr_info);

    // List outputs
    graph_outputs.push_back(bmm_interact_grad_input0_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor bmm_interact_grad_input0 from file
    {
        float* ref_arr = new float[bmm_interact_grad_input0->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, bmm_interact_grad_input0->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bmm_interact_grad_input0", ref_arr, bmm_interact_grad_input0->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(bmm_interact_grad_input0->getDramAddress(), data, bmm_interact_grad_input0->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: bmm_interact_grad_input0");
            validateResult(ref_arr, typed_data, bmm_interact_grad_input0->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: bmm_interact_grad_input0");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_BMM_dedw)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * bmm_interact_dedw node
     * inputs: [reshape_tril_grad_input[128, 27, 27](dtype=bf16), reshape_bmm_output[128, 27, 32](dtype=bf16)]
     * output: [bmm_interact_grad_input1(128, 32, 27)(dtype=float32)]
     *************/
    synGEMMParams bmm_interact_dedw_kernel_params;
    bmm_interact_dedw_kernel_params.transpose_a = true;
    bmm_interact_dedw_kernel_params.transpose_b = false;

    // create reshape_tril_grad_input tensor
    pManagedTensor reshape_tril_grad_input = ManagedTensor::createManagedTensor("reshape_tril_grad_input", {128, 27, 27}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_tril_grad_input_tr_info = {"reshape_tril_grad_input",
                                                           reshape_tril_grad_input->getDramAddress()};

    // create reshape_bmm_output tensor
    pManagedTensor reshape_bmm_output = ManagedTensor::createManagedTensor("reshape_bmm_output", {128, 27, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo reshape_bmm_output_tr_info = {"reshape_bmm_output", reshape_bmm_output->getDramAddress()};

    // create bmm_interact_grad_input1 tensor
    pManagedTensor bmm_interact_grad_input1 = ManagedTensor::createManagedTensor("bmm_interact_grad_input1", {128, 32, 27}, syn_type_single, true, this);
    synLaunchTensorInfo                  bmm_interact_grad_input1_tr_info = {"bmm_interact_grad_input1",
                                                            bmm_interact_grad_input1->getDramAddress()};
    ManagedNodeWithParams<synGEMMParams> bmm_interact_dedw({reshape_tril_grad_input, reshape_bmm_output}, {bmm_interact_grad_input1}, "bmm_interact_dedw", "batch_gemm_dedw", graphHandle, bmm_interact_dedw_kernel_params);
    bmm_interact_dedw.createNode();


    // generate graph
    SetTestFileName("dlrm_BMM_dedw_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor reshape_tril_grad_input from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_tril_grad_input->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_tril_grad_input", typed_data, reshape_tril_grad_input->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, reshape_tril_grad_input->getDramAddress(), reshape_tril_grad_input->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(reshape_tril_grad_input_tr_info);

    // init tensor reshape_bmm_output from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, reshape_bmm_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        bfloat16* typed_data = static_cast<bfloat16*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/reshape_bmm_output", typed_data, reshape_bmm_output->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, reshape_bmm_output->getDramAddress(), reshape_bmm_output->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(reshape_bmm_output_tr_info);

    // List outputs
    graph_outputs.push_back(bmm_interact_grad_input1_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor bmm_interact_grad_input1 from file
    {
        float* ref_arr = new float[bmm_interact_grad_input1->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, bmm_interact_grad_input1->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bmm_interact_grad_input1_transp", ref_arr, bmm_interact_grad_input1->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(bmm_interact_grad_input1->getDramAddress(), data, bmm_interact_grad_input1->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: bmm_interact_grad_input1");
            validateResult(ref_arr, typed_data, bmm_interact_grad_input1->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: bmm_interact_grad_input1");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, dlrm_transpose)
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * bmm_dedw_out_transpose node
     * inputs: [bmm_interact_grad_input1[128, 32, 27](dtype=float32)]
     * output: [bmm_interact_grad_input1_transpose(128, 27, 32)(dtype=float32)]
     *************/
    ns_TransposeKernel::Params bmm_dedw_out_transpose_kernel_params;
    bmm_dedw_out_transpose_kernel_params.axes[0] = 1;
    bmm_dedw_out_transpose_kernel_params.axes[1] = 0;
    bmm_dedw_out_transpose_kernel_params.axes[2] = 2;
    bmm_dedw_out_transpose_kernel_params.axes[3] = 3;

    // create bmm_interact_grad_input1 tensor
    pManagedTensor bmm_interact_grad_input1 = ManagedTensor::createManagedTensor("bmm_interact_grad_input1", {128, 32, 27}, syn_type_single, true, this);
    synLaunchTensorInfo bmm_interact_grad_input1_tr_info = {"bmm_interact_grad_input1",
                                                            bmm_interact_grad_input1->getDramAddress()};

    // create bmm_interact_grad_input1_transpose tensor
    pManagedTensor bmm_interact_grad_input1_transpose = ManagedTensor::createManagedTensor("bmm_interact_grad_input1_transpose", {128, 27, 32}, syn_type_single, true, this);
    synLaunchTensorInfo bmm_interact_grad_input1_transpose_tr_info = {
        "bmm_interact_grad_input1_transpose",
        bmm_interact_grad_input1_transpose->getDramAddress()};
    ManagedNodeWithParams<ns_TransposeKernel::Params> bmm_dedw_out_transpose({bmm_interact_grad_input1}, {bmm_interact_grad_input1_transpose}, "bmm_dedw_out_transpose", "transpose_fwd_f32", graphHandle, bmm_dedw_out_transpose_kernel_params);
    bmm_dedw_out_transpose.createNode();


    // generate graph
    SetTestFileName("dlrm_transpose_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor bmm_interact_grad_input1 from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, bmm_interact_grad_input1->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bmm_interact_grad_input1", typed_data, bmm_interact_grad_input1->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, bmm_interact_grad_input1->getDramAddress(), bmm_interact_grad_input1->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(bmm_interact_grad_input1_tr_info);

    // List outputs
    graph_outputs.push_back(bmm_interact_grad_input1_transpose_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor bmm_interact_grad_input1_transpose from file
    {
        float* ref_arr = new float[bmm_interact_grad_input1_transpose->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, bmm_interact_grad_input1_transpose->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/bmm_interact_grad_input1_transpose", ref_arr, bmm_interact_grad_input1_transpose->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(bmm_interact_grad_input1_transpose->getDramAddress(), data, bmm_interact_grad_input1_transpose->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: bmm_interact_grad_input1_transpose");
            validateResult(ref_arr, typed_data, bmm_interact_grad_input1_transpose->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: bmm_interact_grad_input1_transpose");
        }
        delete[] ref_arr;
    }

    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}


TEST_F_GC(SynGaudiDLRMTestBFloat16, DISABLED_dlrm_EmbeddingBag_dedw) // disabled since guid was removed
{
    uint32_t deviceId = _getDeviceId();
    synGraphHandle graphHandle;
    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    synStatus status = synGraphCreate(&graphHandle, synDeviceGaudi);
    ASSERT_TRUE(status == synSuccess && "synGraphCreate failed!");
    clearDramMap();
    /*************
     * emb_0_embbag_dedw_SLS node
     * inputs: [concat_interact0_grad_input1[128, 32](dtype=float32), emb_0_embbag_dedw_grad_indices[128](dtype=int32), emb_0_embbag_dedw_lengths[128](dtype=int32), emb_0_embbag_valid_indices[1](dtype=int32)]
     * output: [emb_0_embbag_dedw_grad_inter(128, 32)(dtype=float32)]
     *************/
    ns_SparseLengthsSum::Params emb_0_embbag_dedw_SLS_kernel_params;
    emb_0_embbag_dedw_SLS_kernel_params.mode = EMBEDDED_SC_ZP;

    // create concat_interact0_grad_input1 tensor
    pManagedTensor concat_interact0_grad_input1 = ManagedTensor::createManagedTensor("concat_interact0_grad_input1", {128, 32}, syn_type_single, true, this);
    synLaunchTensorInfo concat_interact0_grad_input1_tr_info = {"concat_interact0_grad_input1",
                                                                concat_interact0_grad_input1->getDramAddress()};

    // create emb_0_embbag_dedw_grad_indices tensor
    pManagedTensor emb_0_embbag_dedw_grad_indices = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_grad_indices", {128,}, syn_type_int32, true, this);
    synLaunchTensorInfo emb_0_embbag_dedw_grad_indices_tr_info = {"emb_0_embbag_dedw_grad_indices",
                                                                  emb_0_embbag_dedw_grad_indices->getDramAddress()};

    // create emb_0_embbag_dedw_lengths tensor
    pManagedTensor emb_0_embbag_dedw_lengths = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_lengths", {128,}, syn_type_int32, true, this);
    synLaunchTensorInfo emb_0_embbag_dedw_lengths_tr_info = {"emb_0_embbag_dedw_lengths",
                                                             emb_0_embbag_dedw_lengths->getDramAddress()};

    // create emb_0_embbag_valid_indices tensor
    pManagedTensor emb_0_embbag_valid_indices = ManagedTensor::createManagedTensor("emb_0_embbag_valid_indices", {1,}, syn_type_int32, true, this);
    synLaunchTensorInfo emb_0_embbag_valid_indices_tr_info = {"emb_0_embbag_valid_indices",
                                                              emb_0_embbag_valid_indices->getDramAddress()};

    // create emb_0_embbag_dedw_grad_inter tensor
    pManagedTensor emb_0_embbag_dedw_grad_inter = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_grad_inter", {128, 32}, syn_type_single, false, this);
    ManagedNodeWithParams<ns_SparseLengthsSum::Params> emb_0_embbag_dedw_SLS({concat_interact0_grad_input1, emb_0_embbag_dedw_grad_indices, emb_0_embbag_dedw_lengths, emb_0_embbag_valid_indices}, {emb_0_embbag_dedw_grad_inter}, "emb_0_embbag_dedw_SLS", "sparse_lengths_sum_fwd_f32", graphHandle, emb_0_embbag_dedw_SLS_kernel_params);
    emb_0_embbag_dedw_SLS.createNode();

    /*************
     * emb_0_embbag_dedw_EMB node
     * inputs: [emb_0_embbag_dedw_grad_inter(128, 32)(dtype=float32), emb_0_embbag_weight_golden[1460, 32](dtype=float32), emb_0_embbag_momentum[1460, 32](dtype=float32), emb_0_embbag_dedw_param_table_indices[128](dtype=int32), emb_0_offset_shape[128](dtype=int32), emb_0_embbag_timestamp[1](dtype=int32), emb_0_embbag_rate[1](dtype=float32), emb_0_embbag_valid_indices[1](dtype=int32), emb_0_embbag_valid_indices[1](dtype=int32)]
     * output: [emb_0_embbag_dedw_weight_output_golden(1460, 32)(dtype=float32), emb_0_embbag_dedw_momentum_output(1460, 32)(dtype=float32), emb_0_embbag_dedw_weight_output(1460, 32)(dtype=bf16)]
     *************/
    ns_EmbeddingBagWithSgdKernel::Params emb_0_embbag_dedw_EMB_kernel_params;
    emb_0_embbag_dedw_EMB_kernel_params.mode = EMBEDDING_BAG_MODE_SUM;
    emb_0_embbag_dedw_EMB_kernel_params.sgd.wd = 0;
    emb_0_embbag_dedw_EMB_kernel_params.sgd.mom = 1;
    emb_0_embbag_dedw_EMB_kernel_params.sgd.damp = 0;
    emb_0_embbag_dedw_EMB_kernel_params.sgd.nesterov = false;

    // create emb_0_embbag_weight_golden tensor
    pManagedTensor emb_0_embbag_weight_golden = ManagedTensor::createManagedTensor("emb_0_embbag_weight_golden", {1460, 32}, syn_type_single, true, this);
    synLaunchTensorInfo emb_0_embbag_weight_golden_tr_info = {"emb_0_embbag_weight_golden",
                                                              emb_0_embbag_weight_golden->getDramAddress()};

    // create emb_0_embbag_momentum tensor
    pManagedTensor emb_0_embbag_momentum = ManagedTensor::createManagedTensor("emb_0_embbag_momentum", {1460, 32}, syn_type_single, true, this);
    synLaunchTensorInfo emb_0_embbag_momentum_tr_info = {"emb_0_embbag_momentum",
                                                         emb_0_embbag_momentum->getDramAddress()};

    // create emb_0_embbag_dedw_param_table_indices tensor
    pManagedTensor emb_0_embbag_dedw_param_table_indices = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_param_table_indices", {128,}, syn_type_int32, true, this);
    synLaunchTensorInfo emb_0_embbag_dedw_param_table_indices_tr_info = {
        "emb_0_embbag_dedw_param_table_indices",
        emb_0_embbag_dedw_param_table_indices->getDramAddress()};

    // create emb_0_offset_shape tensor
    pManagedTensor emb_0_offset_shape = ManagedTensor::createManagedTensor("emb_0_offset_shape", {128,}, syn_type_int32, true, this);
    synLaunchTensorInfo emb_0_offset_shape_tr_info = {"emb_0_offset_shape", emb_0_offset_shape->getDramAddress()};

    // create emb_0_embbag_timestamp tensor
    pManagedTensor emb_0_embbag_timestamp = ManagedTensor::createManagedTensor("emb_0_embbag_timestamp", {1,}, syn_type_int32, true, this);
    synLaunchTensorInfo emb_0_embbag_timestamp_tr_info = {"emb_0_embbag_timestamp",
                                                          emb_0_embbag_timestamp->getDramAddress()};

    // create emb_0_embbag_rate tensor
    pManagedTensor emb_0_embbag_rate = ManagedTensor::createManagedTensor("emb_0_embbag_rate", {1,}, syn_type_single, true, this);
    synLaunchTensorInfo emb_0_embbag_rate_tr_info = {"emb_0_embbag_rate", emb_0_embbag_rate->getDramAddress()};

    // create emb_0_embbag_dedw_weight_output_golden tensor
    pManagedTensor emb_0_embbag_dedw_weight_output_golden = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_weight_output_golden", {1460, 32}, syn_type_single, true, this);
    synLaunchTensorInfo emb_0_embbag_dedw_weight_output_golden_tr_info = {
        "emb_0_embbag_dedw_weight_output_golden",
        emb_0_embbag_dedw_weight_output_golden->getDramAddress()};

    // create emb_0_embbag_dedw_momentum_output tensor
    pManagedTensor emb_0_embbag_dedw_momentum_output = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_momentum_output", {1460, 32}, syn_type_single, true, this);
    synLaunchTensorInfo emb_0_embbag_dedw_momentum_output_tr_info = {
        "emb_0_embbag_dedw_momentum_output",
        emb_0_embbag_dedw_momentum_output->getDramAddress()};

    // create emb_0_embbag_dedw_weight_output tensor
    pManagedTensor emb_0_embbag_dedw_weight_output = ManagedTensor::createManagedTensor("emb_0_embbag_dedw_weight_output", {1460, 32}, syn_type_bf16, true, this);
    synLaunchTensorInfo emb_0_embbag_dedw_weight_output_tr_info = {"emb_0_embbag_dedw_weight_output",
                                                                   emb_0_embbag_dedw_weight_output->getDramAddress()};
    ManagedNodeWithParams<ns_EmbeddingBagWithSgdKernel::Params> emb_0_embbag_dedw_EMB({emb_0_embbag_dedw_grad_inter, emb_0_embbag_weight_golden, emb_0_embbag_momentum, emb_0_embbag_dedw_param_table_indices, emb_0_offset_shape, emb_0_embbag_timestamp, emb_0_embbag_rate, emb_0_embbag_valid_indices, emb_0_embbag_valid_indices}, {emb_0_embbag_dedw_weight_output_golden, emb_0_embbag_dedw_momentum_output, emb_0_embbag_dedw_weight_output}, "emb_0_embbag_dedw_EMB", "embedding_bag_sgd_bwd_f32", graphHandle, emb_0_embbag_dedw_EMB_kernel_params);
    emb_0_embbag_dedw_EMB.createNode();


    // generate graph
    SetTestFileName("dlrm_EmbeddingBag_dedw_bf16");
    LaunchInfo launchInfo = compileAllocateAndLoadGraph(graphHandle);
    TensorInfoList graph_inputs;
    TensorInfoList graph_outputs;

    // Init tensors data from files

    // init tensor concat_interact0_grad_input1 from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, concat_interact0_grad_input1->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/concat_interact0_grad_input1", typed_data, concat_interact0_grad_input1->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, concat_interact0_grad_input1->getDramAddress(), concat_interact0_grad_input1->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(concat_interact0_grad_input1_tr_info);

    // init tensor emb_0_embbag_dedw_grad_indices from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_dedw_grad_indices->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_dedw_grad_indices", typed_data, emb_0_embbag_dedw_grad_indices->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_dedw_grad_indices->getDramAddress(), emb_0_embbag_dedw_grad_indices->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_dedw_grad_indices_tr_info);

    // init tensor emb_0_embbag_dedw_lengths from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_dedw_lengths->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_dedw_lengths", typed_data, emb_0_embbag_dedw_lengths->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_dedw_lengths->getDramAddress(), emb_0_embbag_dedw_lengths->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_dedw_lengths_tr_info);

    // init tensor emb_0_embbag_valid_indices from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_valid_indices->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_valid_indices", typed_data, emb_0_embbag_valid_indices->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_valid_indices->getDramAddress(), emb_0_embbag_valid_indices->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_valid_indices_tr_info);

    // init tensor emb_0_embbag_momentum from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_momentum->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_momentum", typed_data, emb_0_embbag_momentum->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_momentum->getDramAddress(), emb_0_embbag_momentum->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_momentum_tr_info);

    // init tensor emb_0_embbag_dedw_param_table_indices from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_dedw_param_table_indices->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_dedw_param_table_indices", typed_data, emb_0_embbag_dedw_param_table_indices->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_dedw_param_table_indices->getDramAddress(), emb_0_embbag_dedw_param_table_indices->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_dedw_param_table_indices_tr_info);

    // init tensor emb_0_offset_shape from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_offset_shape->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_offset_shape", typed_data, emb_0_offset_shape->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_offset_shape->getDramAddress(), emb_0_offset_shape->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_offset_shape_tr_info);

    // init tensor emb_0_embbag_timestamp from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_timestamp->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        int32_t* typed_data = static_cast<int32_t*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_timestamp", typed_data, emb_0_embbag_timestamp->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_timestamp->getDramAddress(), emb_0_embbag_timestamp->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_timestamp_tr_info);

    // init tensor emb_0_embbag_rate from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_rate->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_rate", typed_data, emb_0_embbag_rate->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_rate->getDramAddress(), emb_0_embbag_rate->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_rate_tr_info);

    // init tensor emb_0_embbag_weight from file
    {
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_weight_golden->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_weight", typed_data, emb_0_embbag_weight_golden->getNumberOfElements());
        ASSERT_TRUE(file_res);
        downloadTensorData(data, emb_0_embbag_weight_golden->getDramAddress(), emb_0_embbag_weight_golden->getTotalSizeInBytes());
        status = synHostFree(deviceId, data, 0);
        ASSERT_TRUE(status == synSuccess && "synHostFree failed");
    }
    graph_inputs.push_back(emb_0_embbag_weight_golden_tr_info);

    // List outputs
    graph_outputs.push_back(emb_0_embbag_dedw_weight_output_tr_info);
    graph_outputs.push_back(emb_0_embbag_dedw_momentum_output_tr_info);
    graph_outputs.push_back(emb_0_embbag_dedw_weight_output_golden_tr_info);

    // Go! Go! Go!
    executeTraining(launchInfo, graph_inputs, graph_outputs);

    // Check data against files
    // data check for tensor emb_0_embbag_dedw_momentum_output from file
    {
        float* ref_arr = new float[emb_0_embbag_dedw_momentum_output->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_dedw_momentum_output->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_dedw_momentum_output", ref_arr, emb_0_embbag_dedw_momentum_output->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(emb_0_embbag_dedw_momentum_output->getDramAddress(), data, emb_0_embbag_dedw_momentum_output->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: emb_0_embbag_dedw_momentum_output");
            validateResult(ref_arr, typed_data, emb_0_embbag_dedw_momentum_output->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: emb_0_embbag_dedw_momentum_output");
        }
        delete[] ref_arr;
    }
    // data check for tensor emb_0_embbag_weight_golden from file
    {
        float* ref_arr = new float[emb_0_embbag_weight_golden->getNumberOfElements()];
        void* data = nullptr;
        status = synHostMalloc(deviceId, emb_0_embbag_weight_golden->getTotalSizeInBytes(), 0, &data);
        ASSERT_TRUE(status == synSuccess && "synHostMalloc failed");
        float* typed_data = static_cast<float*>(data);
        bool file_res = read_file("/software/dlrm_traininig/sw_extracted_tensors/emb_0_embbag_dedw_weight_output", ref_arr, emb_0_embbag_dedw_weight_output_golden->getNumberOfElements());
        (void)file_res;
        if (file_res)
        {
            uploadTensorData(emb_0_embbag_weight_golden->getDramAddress(), data, emb_0_embbag_weight_golden->getTotalSizeInBytes());
            LOG_WARN(SYN_TEST, "validating: emb_0_embbag_weight_golden");
            validateResult(ref_arr, typed_data, emb_0_embbag_weight_golden->getNumberOfElements());
        } else {
            LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: emb_0_embbag_weight_golden");
        }
        delete[] ref_arr;
    }


    clearDramMap();
    status = synRecipeDestroy(launchInfo.m_recipeHandle);
    ASSERT_TRUE(status == synSuccess);
    status = synGraphDestroy(graphHandle);
    ASSERT_TRUE(status == synSuccess);
}
