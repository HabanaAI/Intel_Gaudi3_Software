#include "gc_autogen_test.h"
#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynTrainingTpcTestInfra, maxpool_relu_forward_L2)
{
    const char* envSoftwareLfsData = std::getenv("SOFTWARE_LFS_DATA");
    ASSERT_TRUE(envSoftwareLfsData) << "SOFTWARE_LFS_DATA is not set!";
    std::string softwareLfsData = envSoftwareLfsData;
    std::string path            = softwareLfsData + "/synapse/tests/gaudi/maxpool_relu/";

    /*************
     * worker_0_maxpool node
     * inputs: [input(64, 112, 112, 64)(dtype=bf16)]
     * output: [worker_0_maxpoolmax_indices(64, 56, 56, 64)(dtype=uint16), worker_0_maxpool_output(64, 56, 56, 64)(dtype=bf16)]
     *************/
    ns_SpatialReduction::Params worker_0_maxpool_kernel_params;
    worker_0_maxpool_kernel_params.kernel_w = 3;
    worker_0_maxpool_kernel_params.kernel_h = 3;
    worker_0_maxpool_kernel_params.stride_w = 2;
    worker_0_maxpool_kernel_params.stride_h = 2;
    worker_0_maxpool_kernel_params.pad_w_begin = 1;
    worker_0_maxpool_kernel_params.pad_w_end = 1;
    worker_0_maxpool_kernel_params.pad_h_begin = 1;
    worker_0_maxpool_kernel_params.pad_h_end = 1;
    worker_0_maxpool_kernel_params.dilation_w = 1;
    worker_0_maxpool_kernel_params.dilation_h = 1;
    worker_0_maxpool_kernel_params.pooling_convention = POOLING_CONVENTION_VALID;

    // create input tensor
    unsigned input_sizes[4] = {64, 112, 112, 64};
    const unsigned input_size = 64*112*112*64;
    bfloat16* input_val = new bfloat16[input_size];
    bool ret = read_file(path + "input", input_val, input_size);
    ASSERT_EQ(ret, true) << "Failed to read input";
    float* inputBuffer = new float[input_size];
    // copy the file content into input buffer
    for (unsigned iter = 0; iter < input_size; iter++)
    {
        inputBuffer[iter] = (float)input_val[iter];
    }

    unsigned inputTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputBuffer, input_sizes, 4,
                                                syn_type_bf16, nullptr, "worker_0_maxpool_in_vec");

    // create worker_0_maxpoolmax_indices tensor
    unsigned worker_0_maxpoolmax_indices_sizes[4] = {64, 56, 56, 64};

    unsigned indicesOutputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                        worker_0_maxpoolmax_indices_sizes, 4,
                                                        syn_type_int16, nullptr, "worker_0_maxpoolmax_indices");

    // create worker_0_maxpool_output tensor
    unsigned worker_0_maxpool_output_sizes[] = {64, 56, 56, 64};
    const unsigned worker_0_maxpool_output_size = 64*56*56*64;

    unsigned outputTensor  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                 worker_0_maxpool_output_sizes, 4, syn_type_bf16, nullptr,
                                                 "worker_0_maxpoolmax_output");

    addNodeToGraph("maxpool_2d_fwd_bf16", {inputTensor}, {indicesOutputTensor, outputTensor},
                   &worker_0_maxpool_kernel_params, sizeof(ns_SpatialReduction::Params));

    /*************
     * layer1_0_relu1 node
     * inputs: [worker_0_maxpool_output(64, 56, 56, 64)(dtype=bf16)]
     * output: [layer1_0_relu1_output(64, 56, 56, 64)(dtype=bf16)]
     *************/

    // create layer1_0_relu1_output tensor
    unsigned layer1_0_relu1_output_sizes[4] = {64, 56, 56, 64};
    unsigned layer1_0_relu1_output_size = 64*56*56*64;

    unsigned reluOutputTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                    layer1_0_relu1_output_sizes, 4, syn_type_bf16,
                                                    nullptr, "layer1_0_relu1_output");

    unsigned reluInputTensor = connectOutputTensorToInputTensor(outputTensor);
    addNodeToGraph("relu_fwd_bf16", {reluInputTensor}, {reluOutputTensor}, nullptr);

    compileAndRun();

    // validate - check data against files
    bfloat16* ref_arr = new bfloat16[worker_0_maxpool_output_size];
    ret = read_file(path + "worker_0_maxpool_output", ref_arr, worker_0_maxpool_output_size);
    ASSERT_EQ(ret, true) << "Failed to read output";
    bfloat16* worker_0_maxpoolmax_output_val = (bfloat16*)m_hostBuffers[outputTensor];

    for (unsigned int index = 0; index < worker_0_maxpool_output_size; ++index)
    {
        float diff = std::abs((float)ref_arr[index] - (float)worker_0_maxpoolmax_output_val[index]);
        ASSERT_FALSE(std::isnan(diff)) << "Computed maxpool values are incorrect";
        ASSERT_TRUE(diff < 0.01) << "Computed maxpool values are too far from reference";
    }

    bfloat16* relu_ref_arr = new bfloat16[layer1_0_relu1_output_size];
    ret = read_file(path + "layer1_0_relu1_output", relu_ref_arr, layer1_0_relu1_output_size);
    ASSERT_EQ(ret, true) << "Failed to read relu output";
    bfloat16* layer1_0_relu1_output_val = (bfloat16*)m_hostBuffers[reluOutputTensor];

    validateResult(relu_ref_arr, layer1_0_relu1_output_val, layer1_0_relu1_output_size);

    delete[] ref_arr;
    delete[] relu_ref_arr;
    delete[] input_val;
    delete[] inputBuffer;

}
