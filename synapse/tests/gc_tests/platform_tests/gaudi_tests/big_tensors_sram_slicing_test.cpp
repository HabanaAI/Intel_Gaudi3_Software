#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"

class SynGaudiBigTensorsSramSlicingTests : public SynGaudiTwoRunCompareTest
{
public:
    void setConfigsForTest()
    {
        // Compare unsliced graphs
        addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

        // to slicing in big-small config:
        addConfigurationToRun(SECOND_RUN, "IGNORE_INDEX_SPACE_FOR_SLICING", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_SLICER_RESHAPE_ALIGNMENT", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_BATCH_NORM_MEMCPY_FUSION", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "true");
    }
};

// The following tests are relevant for Gaudi1 only, where big-small infra is not yet enabled by default.

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, stitching_consumer_with_overlap_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_tensor_223_t30748_n16326__aten_threshold_backward_0_max_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_223_t30748_n16326__aten_threshold_backward_0_min_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_223_t30748_n16326__aten_threshold_backward_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_223_t30748_n16326__aten_threshold_backward_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_223_t30748_n16326__aten_threshold_backward_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_223_t30748_n16326__aten_threshold_backward_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_225_1571_max_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_225_1571_min_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_225_1571             = createTensors(1,
                                                 INPUT_TENSOR,
                                                 true,
                                                 "g_0_tensor_225_1571",
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 g_0_tensor_225_1571_max_sizes,
                                                 4,
                                                 syn_type_bf16,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_tensor_225_1571_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n16328__aten_convolution_backward_overrideable_dedw_0_id;
    unsigned char g_0_n16328__aten_convolution_backward_overrideable_dedw_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0,   1,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 160, 72, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_tensor_223_t30748_n16326__aten_threshold_backward_0, g_0_tensor_225_1571},
                   {g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1},
                   (void*)g_0_n16328__aten_convolution_backward_overrideable_dedw_0_params,
                   104,
                   "g_0_n16328__aten_convolution_backward_overrideable_dedw_0",
                   0 /*graphIndex*/,
                   &g_0_n16328__aten_convolution_backward_overrideable_dedw_0_id);

    unsigned g_0_tensor_226_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_tensor_226_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_tensor_226             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_226",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_226_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_226_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0_max_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0_min_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n16328__aten_convolution_backward_overrideable_dedx_0_id;
    unsigned char g_0_n16328__aten_convolution_backward_overrideable_dedx_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0,   1,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 108, 87, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_tensor_223_t30748_n16326__aten_threshold_backward_0, g_0_tensor_226},
                   {g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0},
                   (void*)g_0_n16328__aten_convolution_backward_overrideable_dedx_0_params,
                   104,
                   "g_0_n16328__aten_convolution_backward_overrideable_dedx_0",
                   0 /*graphIndex*/,
                   &g_0_n16328__aten_convolution_backward_overrideable_dedx_0_id);

    unsigned g_0_tensor_325_1451_max_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_325_1451_min_sizes[] = {256, 21, 13, 4};
    unsigned g_0_tensor_325_1451             = createTensors(1,
                                                 INPUT_TENSOR,
                                                 true,
                                                 "g_0_tensor_325_1451",
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 g_0_tensor_325_1451_max_sizes,
                                                 4,
                                                 syn_type_int16,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_tensor_325_1451_min_sizes,
                                                 synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0_max_sizes[] = {256, 42, 25, 4};
    unsigned g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0_min_sizes[] = {256, 42, 25, 4};
    unsigned g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n16560__aten_max_pool2d_with_indices_backward_maxpool_2d_bwd_bf16_0_id;
    unsigned char g_0_n16560__aten_max_pool2d_with_indices_backward_maxpool_2d_bwd_bf16_0_params[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_bwd_bf16",
                   {g_0_tensor_228_t30755_n16328__aten_convolution_backward_overrideable_0, g_0_tensor_325_1451},
                   {g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0},
                   (void*)g_0_n16560__aten_max_pool2d_with_indices_backward_maxpool_2d_bwd_bf16_0_params,
                   44,
                   "g_0_n16560__aten_max_pool2d_with_indices_backward_maxpool_2d_bwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_n16560__aten_max_pool2d_with_indices_backward_maxpool_2d_bwd_bf16_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_tensor_326_t31119_n16560__aten_max_pool2d_with_indices_backward_0,
                        g_0_tensor_227_t30756_n16328__aten_convolution_backward_overrideable_1});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, stitching_node_with_zero_sized_shape_tensor, {synDeviceGaudi})
{
    unsigned
        g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_HabanaInstanceNormGrad_1_max_sizes
            [] = {256, 64, 64, 2};
    unsigned
        g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_HabanaInstanceNormGrad_1_min_sizes
            [] = {256, 64, 64, 2};
    unsigned g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_HabanaInstanceNormGrad_1 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_"
            "HabanaInstanceNormGrad_1",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_HabanaInstanceNormGrad_1_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_HabanaInstanceNormGrad_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_max_sizes[] =
            {256, 66, 66, 2};
    unsigned
        g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_min_sizes[] =
            {256, 66, 66, 2};
    unsigned g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput =
        createTensors(
            1,
            INPUT_TENSOR,
            false,
            "g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_max_sizes,
            4,
            syn_type_uint32,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_min_sizes,
            synTensorType::SHAPE_TENSOR)[0];

    unsigned
        g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0_max_sizes
            [] = {256, 66, 66, 2};
    unsigned
        g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0_min_sizes
            [] = {256, 66, 66, 2};
    unsigned g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_dedx_n627_0_id;
    unsigned char
        g_0_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_dedx_n627_0_params
            [] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,  0,   0,   0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 178, 91, 1,   0,   0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,  253, 127, 0, 0};
    addNodeToGraph(
        "dedx",
        {g_0_t1516_StatefulPartitionedCall_gradient_tape_generator_X_habana_instance_normalization_9_HabanaInstanceNormGrad_1,
         g_0_t629_StatefulPartitionedCall_generator_X_conv2d_13_Conv2D_1_Cast_0,
         g_0_t1520_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput},
        {g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0},
        (void*)
            g_0_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_dedx_n627_0_params,
        72,
        "g_0_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_dedx_n627_0",
        0 /*graphIndex*/,
        &g_0_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_dedx_n627_0_id);

    unsigned
        g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_max_sizes[] =
            {256, 66, 66, 2, 1};
    unsigned
        g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_min_sizes[] =
            {256, 66, 66, 2, 1};
    unsigned g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1 =
        createTensors(
            1,
            INPUT_TENSOR,
            false,
            "g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_max_sizes,
            5,
            syn_type_uint32,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_min_sizes,
            synTensorType::SHAPE_TENSOR)[0];

    unsigned
        g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_max_sizes[] =
            {256, 66, 66, 2, 1};
    unsigned
        g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_min_sizes[] =
            {256, 66, 66, 2, 1};
    unsigned g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_max_sizes,
            5,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_reshape_n628_0_id;
    addNodeToGraph(
        "reshape",
        {g_0_t1519_StatefulPartitionedCall_gradient_tape_generator_X_conv2d_13_Conv2D_1_Conv2DBackpropInput_0,
         g_0_t1523_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1},
        {g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1},
        nullptr,
        0,
        "g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_reshape_n628_0",
        0 /*graphIndex*/,
        &g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_reshape_n628_0_id);

    unsigned
        g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_data[] =
            {0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0};
    unsigned
        g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_size[] =
            {10};

    unsigned g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1 =
        createHost2DeviceTensor(
            INPUT_TENSOR,
            g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_size,
            g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_data,
            1,
            "g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1");

    unsigned
        g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_max_sizes[] =
            {256, 64, 64, 2, 1};
    unsigned
        g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_min_sizes[] =
            {256, 64, 64, 2, 1};
    unsigned g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_max_sizes,
            5,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_pad_bwd_bf16_n630_0_id;
    unsigned char
        g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_pad_bwd_bf16_n630_0_params
            [] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    addNodeToGraph(
        "pad_bwd_bf16",
        {g_0_t1522_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1,
         g_0_t1526_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1},
        {g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1},
        (void*)
            g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_pad_bwd_bf16_n630_0_params,
        48,
        "g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_pad_bwd_bf16_n630_0",
        0 /*graphIndex*/,
        &g_0_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1_pad_bwd_bf16_n630_0_id);

    setConfigsForTest();

    compareRunsResults(
        {g_0_t1524_StatefulPartitionedCall_gradient_tape_generator_X_tf_compat_v1_pad_1_MirrorPadGrad_1});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, avgpool_conv_bundle, {synDeviceGaudi})
{
    unsigned g_0_layer4_2_relu3_output_max_sizes[] = {2048, 7, 7, 64};
    unsigned g_0_layer4_2_relu3_output_min_sizes[] = {2048, 7, 7, 64};
    unsigned g_0_layer4_2_relu3_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_relu3_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_relu3_output_max_sizes,
                                                       4,
                                                       syn_type_bf16,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_relu3_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_worker_0_avgpool_output_max_sizes[] = {2048, 1, 1, 64};
    unsigned      g_0_worker_0_avgpool_output_min_sizes[] = {2048, 1, 1, 64};
    unsigned      g_0_worker_0_avgpool_output             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_worker_0_avgpool_output",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_worker_0_avgpool_output_max_sizes,
                                                         4,
                                                         syn_type_bf16,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_worker_0_avgpool_output_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_worker_0_avgpool_0_id;
    unsigned char g_0_worker_0_avgpool_0_params[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     7, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                     1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("avg_pool_2d_fwd_bf16",
                   {g_0_layer4_2_relu3_output},
                   {g_0_worker_0_avgpool_output},
                   (void*)g_0_worker_0_avgpool_0_params,
                   48,
                   "g_0_worker_0_avgpool_0",
                   0 /*graphIndex*/,
                   &g_0_worker_0_avgpool_0_id);

    unsigned g_0_worker_0_fc_weight_max_sizes[] = {1000, 2048, 1, 1};
    unsigned g_0_worker_0_fc_weight_min_sizes[] = {1000, 2048, 1, 1};
    unsigned g_0_worker_0_fc_weight             = createTensors(1,
                                                    INPUT_TENSOR,
                                                    true,
                                                    "g_0_worker_0_fc_weight",
                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                    nullptr,
                                                    g_0_worker_0_fc_weight_max_sizes,
                                                    4,
                                                    syn_type_bf16,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    g_0_worker_0_fc_weight_min_sizes,
                                                    synTensorType::DATA_TENSOR)[0];

    unsigned g_0_worker_0_fc_bias_max_sizes[] = {1000};
    unsigned g_0_worker_0_fc_bias_min_sizes[] = {1000};
    unsigned g_0_worker_0_fc_bias             = createTensors(1,
                                                  INPUT_TENSOR,
                                                  true,
                                                  "g_0_worker_0_fc_bias",
                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                  nullptr,
                                                  g_0_worker_0_fc_bias_max_sizes,
                                                  1,
                                                  syn_type_bf16,
                                                  nullptr,
                                                  0,
                                                  0,
                                                  nullptr,
                                                  false,
                                                  g_0_worker_0_fc_bias_min_sizes,
                                                  synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_worker_0_fc_output_max_sizes[] = {1000, 1, 1, 64};
    unsigned      g_0_worker_0_fc_output_min_sizes[] = {1000, 1, 1, 64};
    unsigned      g_0_worker_0_fc_output             = createTensors(1,
                                                    OUTPUT_TENSOR,
                                                    true,
                                                    "g_0_worker_0_fc_output",
                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                    nullptr,
                                                    g_0_worker_0_fc_output_max_sizes,
                                                    4,
                                                    syn_type_bf16,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    g_0_worker_0_fc_output_min_sizes,
                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_worker_0_fc_0_id;
    unsigned char g_0_worker_0_fc_0_params[] = {1, 0, 0, 0, 1, 0, 0,   0,   1, 0, 0, 0, 1, 0, 0,   0,   0, 0,
                                                0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 1,   0,   0, 0,
                                                1, 0, 0, 0, 0, 0, 227, 229, 1, 0, 0, 0, 0, 0, 0,   0,   0, 0,
                                                0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 1, 0, 0, 0, 254, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_worker_0_avgpool_output, g_0_worker_0_fc_weight, g_0_worker_0_fc_bias},
                   {g_0_worker_0_fc_output},
                   (void*)g_0_worker_0_fc_0_params,
                   72,
                   "g_0_worker_0_fc_0",
                   0 /*graphIndex*/,
                   &g_0_worker_0_fc_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_worker_0_fc_output});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, conv_instance_norm_bundle, {synDeviceGaudi})
{
    unsigned g_0_tensor_19_t1170_n371__aten_leaky_relu__0_max_sizes[] = {32, 160, 192, 64};
    unsigned g_0_tensor_19_t1170_n371__aten_leaky_relu__0_min_sizes[] = {32, 160, 192, 64};
    unsigned g_0_tensor_19_t1170_n371__aten_leaky_relu__0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_19_t1170_n371__aten_leaky_relu__0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_19_t1170_n371__aten_leaky_relu__0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_19_t1170_n371__aten_leaky_relu__0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_21_t1176_n374__hpu_cast_0_max_sizes[] = {64, 32, 3, 3};
    unsigned g_0_tensor_21_t1176_n374__hpu_cast_0_min_sizes[] = {64, 32, 3, 3};
    unsigned g_0_tensor_21_t1176_n374__hpu_cast_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_tensor_21_t1176_n374__hpu_cast_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_tensor_21_t1176_n374__hpu_cast_0_max_sizes,
                                                                  4,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_tensor_21_t1176_n374__hpu_cast_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0_max_sizes[] = {64, 80, 96, 64};
    unsigned g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0_min_sizes[] = {64, 80, 96, 64};
    unsigned g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n375__aten_convolution_overrideable_spatial_convolution_0_id;
    unsigned char g_0_n375__aten_convolution_overrideable_spatial_convolution_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 193, 227, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   252, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_tensor_19_t1170_n371__aten_leaky_relu__0, g_0_tensor_21_t1176_n374__hpu_cast_0},
                   {g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0},
                   (void*)g_0_n375__aten_convolution_overrideable_spatial_convolution_0_params,
                   72,
                   "g_0_n375__aten_convolution_overrideable_spatial_convolution_0",
                   0 /*graphIndex*/,
                   &g_0_n375__aten_convolution_overrideable_spatial_convolution_0_id);

    unsigned g_0_tensor_24_max_sizes[] = {64};
    unsigned g_0_tensor_24_min_sizes[] = {64};
    unsigned g_0_tensor_24             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_24",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_24_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_24_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_23_max_sizes[] = {64};
    unsigned g_0_tensor_23_min_sizes[] = {64};
    unsigned g_0_tensor_23             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_23",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_23_max_sizes,
                                           1,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_23_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_25_t1187_n377__hpu_instance_norm_0_max_sizes[] = {64, 80, 96, 64};
    unsigned g_0_tensor_25_t1187_n377__hpu_instance_norm_0_min_sizes[] = {64, 80, 96, 64};
    unsigned g_0_tensor_25_t1187_n377__hpu_instance_norm_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_25_t1187_n377__hpu_instance_norm_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_25_t1187_n377__hpu_instance_norm_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_25_t1187_n377__hpu_instance_norm_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_26_t1188_n377__hpu_instance_norm_1_max_sizes[] = {64, 64};
    unsigned g_0_tensor_26_t1188_n377__hpu_instance_norm_1_min_sizes[] = {64, 64};
    unsigned g_0_tensor_26_t1188_n377__hpu_instance_norm_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_26_t1188_n377__hpu_instance_norm_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_26_t1188_n377__hpu_instance_norm_1_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_26_t1188_n377__hpu_instance_norm_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_27_t1189_n377__hpu_instance_norm_2_max_sizes[] = {64, 64};
    unsigned g_0_tensor_27_t1189_n377__hpu_instance_norm_2_min_sizes[] = {64, 64};
    unsigned g_0_tensor_27_t1189_n377__hpu_instance_norm_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_27_t1189_n377__hpu_instance_norm_2",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_27_t1189_n377__hpu_instance_norm_2_max_sizes,
                      2,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_27_t1189_n377__hpu_instance_norm_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    synNodeId     g_0_n377__hpu_instance_norm_instance_norm_fwd_bf16_0_id;
    unsigned char g_0_n377__hpu_instance_norm_instance_norm_fwd_bf16_0_params[] = {102, 102, 102, 63, 172, 197, 39, 55};
    addNodeToGraph("instance_norm_fwd_bf16",
                   {g_0_tensor_22_t1180_n375__aten_convolution_overrideable_0, g_0_tensor_24, g_0_tensor_23},
                   {g_0_tensor_25_t1187_n377__hpu_instance_norm_0,
                    g_0_tensor_26_t1188_n377__hpu_instance_norm_1,
                    g_0_tensor_27_t1189_n377__hpu_instance_norm_2},
                   (void*)g_0_n377__hpu_instance_norm_instance_norm_fwd_bf16_0_params,
                   8,
                   "g_0_n377__hpu_instance_norm_instance_norm_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_n377__hpu_instance_norm_instance_norm_fwd_bf16_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_tensor_25_t1187_n377__hpu_instance_norm_0,
                        g_0_tensor_26_t1188_n377__hpu_instance_norm_1,
                        g_0_tensor_27_t1189_n377__hpu_instance_norm_2});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, dedx_bn_bwd_bundle_L2, {synDeviceGaudi})
{
    unsigned g_0_layer4_2_conv2_output_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_conv2_output_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_conv2_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_conv2_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_conv2_output_max_sizes,
                                                       4,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_conv2_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_relu2_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_relu2_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_relu2_grad_input             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer4_2_relu2_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer4_2_relu2_grad_input_max_sizes,
                                                           4,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer4_2_relu2_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn2_saved_mean_max_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn2_saved_mean_min_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn2_saved_mean             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer4_2_bn2_saved_mean",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer4_2_bn2_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer4_2_bn2_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn2_saved_var_max_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn2_saved_var_min_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn2_saved_var             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_layer4_2_bn2_saved_var",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer4_2_bn2_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer4_2_bn2_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn2_weight_max_sizes[] = {512};
    unsigned g_0_layer4_2_bn2_weight_min_sizes[] = {512};
    unsigned g_0_layer4_2_bn2_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer4_2_bn2_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer4_2_bn2_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer4_2_bn2_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn2_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_bn2_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_bn2_grad_input             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         false,
                                                         "g_0_layer4_2_bn2_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer4_2_bn2_grad_input_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer4_2_bn2_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn2_bias_grad_max_sizes[] = {512};
    unsigned g_0_layer4_2_bn2_bias_grad_min_sizes[] = {512};
    unsigned g_0_layer4_2_bn2_bias_grad             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer4_2_bn2_bias_grad",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer4_2_bn2_bias_grad_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer4_2_bn2_bias_grad_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_layer4_2_bn2_weight_grad_max_sizes[] = {512};
    unsigned  g_0_layer4_2_bn2_weight_grad_min_sizes[] = {512};
    unsigned  g_0_layer4_2_bn2_weight_grad             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          true,
                                                          "g_0_layer4_2_bn2_weight_grad",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer4_2_bn2_weight_grad_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer4_2_bn2_weight_grad_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer4_2_bn2_bwd_0_id;
    addNodeToGraph("batch_norm_bwd_f32",
                   {g_0_layer4_2_conv2_output,
                    g_0_layer4_2_relu2_grad_input,
                    g_0_layer4_2_bn2_saved_mean,
                    g_0_layer4_2_bn2_saved_var,
                    g_0_layer4_2_bn2_weight},
                   {g_0_layer4_2_bn2_grad_input, g_0_layer4_2_bn2_bias_grad, g_0_layer4_2_bn2_weight_grad},
                   nullptr,
                   0,
                   "g_0_layer4_2_bn2_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer4_2_bn2_bwd_0_id);

    unsigned g_0_layer4_2_conv2_weight_max_sizes[] = {512, 512, 3, 3};
    unsigned g_0_layer4_2_conv2_weight_min_sizes[] = {512, 512, 3, 3};
    unsigned g_0_layer4_2_conv2_weight             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_conv2_weight",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_conv2_weight_max_sizes,
                                                       4,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_conv2_weight_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_layer4_2_conv2_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned      g_0_layer4_2_conv2_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned      g_0_layer4_2_conv2_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_layer4_2_conv2_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer4_2_conv2_grad_input_max_sizes,
                                                           4,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer4_2_conv2_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_layer4_2_conv2_dedx_0_id;
    unsigned char g_0_layer4_2_conv2_dedx_0_params[] = {3, 0, 0, 0, 3, 0, 0,  0,   1, 0, 0, 0, 1, 0, 0,   0,   1, 0,
                                                        0, 0, 1, 0, 0, 0, 1,  0,   0, 0, 1, 0, 0, 0, 1,   0,   0, 0,
                                                        1, 0, 0, 0, 0, 0, 28, 127, 1, 0, 0, 0, 0, 0, 0,   0,   0, 0,
                                                        0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 1, 0, 0, 0, 255, 127, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_layer4_2_bn2_grad_input, g_0_layer4_2_conv2_weight},
                   {g_0_layer4_2_conv2_grad_input},
                   (void*)g_0_layer4_2_conv2_dedx_0_params,
                   72,
                   "g_0_layer4_2_conv2_dedx_0",
                   0 /*graphIndex*/,
                   &g_0_layer4_2_conv2_dedx_0_id);

    unsigned g_0_layer4_2_relu1_output_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_relu1_output_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_relu1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_relu1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_relu1_output_max_sizes,
                                                       4,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_relu1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_layer4_2_relu1_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned  g_0_layer4_2_relu1_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned  g_0_layer4_2_relu1_grad_input             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           false,
                                                           "g_0_layer4_2_relu1_grad_input",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer4_2_relu1_grad_input_max_sizes,
                                                           4,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer4_2_relu1_grad_input_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer4_2_relu1_bwd_0_id;
    addNodeToGraph("relu_bwd_f32",
                   {g_0_layer4_2_conv2_grad_input, g_0_layer4_2_relu1_output},
                   {g_0_layer4_2_relu1_grad_input},
                   nullptr,
                   0,
                   "g_0_layer4_2_relu1_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer4_2_relu1_bwd_0_id);

    unsigned g_0_layer4_2_conv1_output_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_conv1_output_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_conv1_output             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_layer4_2_conv1_output",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_layer4_2_conv1_output_max_sizes,
                                                       4,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_layer4_2_conv1_output_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn1_saved_mean_max_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn1_saved_mean_min_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn1_saved_mean             = createTensors(1,
                                                         INPUT_TENSOR,
                                                         true,
                                                         "g_0_layer4_2_bn1_saved_mean",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer4_2_bn1_saved_mean_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer4_2_bn1_saved_mean_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn1_saved_var_max_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn1_saved_var_min_sizes[] = {512, 1, 1, 1};
    unsigned g_0_layer4_2_bn1_saved_var             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_layer4_2_bn1_saved_var",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer4_2_bn1_saved_var_max_sizes,
                                                        4,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer4_2_bn1_saved_var_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn1_weight_max_sizes[] = {512};
    unsigned g_0_layer4_2_bn1_weight_min_sizes[] = {512};
    unsigned g_0_layer4_2_bn1_weight             = createTensors(1,
                                                     INPUT_TENSOR,
                                                     true,
                                                     "g_0_layer4_2_bn1_weight",
                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                     nullptr,
                                                     g_0_layer4_2_bn1_weight_max_sizes,
                                                     1,
                                                     syn_type_single,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     g_0_layer4_2_bn1_weight_min_sizes,
                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn1_grad_input_max_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_bn1_grad_input_min_sizes[] = {512, 7, 7, 64};
    unsigned g_0_layer4_2_bn1_grad_input             = createTensors(1,
                                                         OUTPUT_TENSOR,
                                                         true,
                                                         "g_0_layer4_2_bn1_grad_input",
                                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                         nullptr,
                                                         g_0_layer4_2_bn1_grad_input_max_sizes,
                                                         4,
                                                         syn_type_single,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         g_0_layer4_2_bn1_grad_input_min_sizes,
                                                         synTensorType::DATA_TENSOR)[0];

    unsigned g_0_layer4_2_bn1_bias_grad_max_sizes[] = {512};
    unsigned g_0_layer4_2_bn1_bias_grad_min_sizes[] = {512};
    unsigned g_0_layer4_2_bn1_bias_grad             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        true,
                                                        "g_0_layer4_2_bn1_bias_grad",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_layer4_2_bn1_bias_grad_max_sizes,
                                                        1,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_layer4_2_bn1_bias_grad_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_layer4_2_bn1_weight_grad_max_sizes[] = {512};
    unsigned  g_0_layer4_2_bn1_weight_grad_min_sizes[] = {512};
    unsigned  g_0_layer4_2_bn1_weight_grad             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          true,
                                                          "g_0_layer4_2_bn1_weight_grad",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_layer4_2_bn1_weight_grad_max_sizes,
                                                          1,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_layer4_2_bn1_weight_grad_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_layer4_2_bn1_bwd_0_id;
    addNodeToGraph("batch_norm_bwd_f32",
                   {g_0_layer4_2_conv1_output,
                    g_0_layer4_2_relu1_grad_input,
                    g_0_layer4_2_bn1_saved_mean,
                    g_0_layer4_2_bn1_saved_var,
                    g_0_layer4_2_bn1_weight},
                   {g_0_layer4_2_bn1_grad_input, g_0_layer4_2_bn1_bias_grad, g_0_layer4_2_bn1_weight_grad},
                   nullptr,
                   0,
                   "g_0_layer4_2_bn1_bwd_0",
                   0 /*graphIndex*/,
                   &g_0_layer4_2_bn1_bwd_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_layer4_2_bn2_bias_grad,
                        g_0_layer4_2_bn2_weight_grad,
                        g_0_layer4_2_bn1_grad_input,
                        g_0_layer4_2_bn1_bias_grad,
                        g_0_layer4_2_bn1_weight_grad});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, mult_gelu_tpc_bundle_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_tensor_1747_t9686_n1002__aten_mul_0_max_sizes[] = {768, 127, 32};
    unsigned g_0_tensor_1747_t9686_n1002__aten_mul_0_min_sizes[] = {768, 127, 32};
    unsigned g_0_tensor_1747_t9686_n1002__aten_mul_0             = createTensors(1,
                                                                     INPUT_TENSOR,
                                                                     true,
                                                                     "g_0_tensor_1747_t9686_n1002__aten_mul_0",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_tensor_1747_t9686_n1002__aten_mul_0_max_sizes,
                                                                     3,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_tensor_1747_t9686_n1002__aten_mul_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_1754_max_sizes[] = {3072, 768};
    unsigned g_0_tensor_1754_min_sizes[] = {3072, 768};
    unsigned g_0_tensor_1754             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1754",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1754_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1754_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_1755_max_sizes[] = {3072, 127, 32};
    unsigned      g_0_tensor_1755_min_sizes[] = {3072, 127, 32};
    unsigned      g_0_tensor_1755             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             false,
                                             "g_0_tensor_1755",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1755_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1755_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n1004__hpu_matmul_backward_batch_gemm_0_id;
    unsigned char g_0_n1004__hpu_matmul_backward_batch_gemm_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_1747_t9686_n1002__aten_mul_0, g_0_tensor_1754},
                   {g_0_tensor_1755},
                   (void*)g_0_n1004__hpu_matmul_backward_batch_gemm_0_params,
                   2,
                   "g_0_n1004__hpu_matmul_backward_batch_gemm_0",
                   0 /*graphIndex*/,
                   &g_0_n1004__hpu_matmul_backward_batch_gemm_0_id);

    unsigned g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0_max_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0_min_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_n1004__hpu_matmul_backward_identity_0_id;
    addNodeToGraph("identity",
                   {g_0_tensor_1755},
                   {g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0},
                   nullptr,
                   0,
                   "g_0_n1004__hpu_matmul_backward_identity_0",
                   0 /*graphIndex*/,
                   &g_0_n1004__hpu_matmul_backward_identity_0_id);

    unsigned g_0_tensor_1767_t9711__0_max_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1767_t9711__0_min_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1767_t9711__0             = createTensors(1,
                                                      INPUT_TENSOR,
                                                      true,
                                                      "g_0_tensor_1767_t9711__0",
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      g_0_tensor_1767_t9711__0_max_sizes,
                                                      3,
                                                      syn_type_bf16,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      false,
                                                      g_0_tensor_1767_t9711__0_min_sizes,
                                                      synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_tensor_1768_t9714_n1006__aten_mul_0_max_sizes[] = {3072, 127, 32};
    unsigned  g_0_tensor_1768_t9714_n1006__aten_mul_0_min_sizes[] = {3072, 127, 32};
    unsigned  g_0_tensor_1768_t9714_n1006__aten_mul_0             = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_tensor_1768_t9714_n1006__aten_mul_0",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_tensor_1768_t9714_n1006__aten_mul_0_max_sizes,
                                                                     3,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_tensor_1768_t9714_n1006__aten_mul_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_n1006__aten_mul_mult_fwd_bf16_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_1756_t9698_n1004__hpu_matmul_backward_0, g_0_tensor_1767_t9711__0},
                   {g_0_tensor_1768_t9714_n1006__aten_mul_0},
                   nullptr,
                   0,
                   "g_0_n1006__aten_mul_mult_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_n1006__aten_mul_mult_fwd_bf16_0_id);

    unsigned g_0_tensor_1769_max_sizes[] = {1};
    unsigned g_0_tensor_1769_min_sizes[] = {1};
    unsigned g_0_tensor_1769             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1769",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1769_max_sizes,
                                             1,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1769_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_tensor_1770_t9718_n1007__aten_mul_0_max_sizes[] = {3072, 127, 32};
    unsigned  g_0_tensor_1770_t9718_n1007__aten_mul_0_min_sizes[] = {3072, 127, 32};
    unsigned  g_0_tensor_1770_t9718_n1007__aten_mul_0             = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_tensor_1770_t9718_n1007__aten_mul_0",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_tensor_1770_t9718_n1007__aten_mul_0_max_sizes,
                                                                     3,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_tensor_1770_t9718_n1007__aten_mul_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_n1007__aten_mul_mult_fwd_bf16_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {g_0_tensor_1768_t9714_n1006__aten_mul_0, g_0_tensor_1769},
                   {g_0_tensor_1770_t9718_n1007__aten_mul_0},
                   nullptr,
                   0,
                   "g_0_n1007__aten_mul_mult_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_n1007__aten_mul_mult_fwd_bf16_0_id);

    unsigned g_0_tensor_1771_max_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1771_min_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1771             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1771",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1771_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1771_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_1780_max_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1780_min_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1780             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1780",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1780_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1780_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0_max_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0_min_sizes[] = {3072, 127, 32};
    unsigned g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_n1008__aten_gelu_backward_gelu_bwd_bf16_0_id;
    addNodeToGraph("gelu_bwd_bf16",
                   {g_0_tensor_1770_t9718_n1007__aten_mul_0, g_0_tensor_1771, g_0_tensor_1780},
                   {g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0},
                   nullptr,
                   0,
                   "g_0_n1008__aten_gelu_backward_gelu_bwd_bf16_0",
                   0 /*graphIndex*/,
                   &g_0_n1008__aten_gelu_backward_gelu_bwd_bf16_0_id);

    unsigned g_0_tensor_1788_max_sizes[] = {768, 3072};
    unsigned g_0_tensor_1788_min_sizes[] = {768, 3072};
    unsigned g_0_tensor_1788             = createTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1788",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1788_max_sizes,
                                             2,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1788_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_tensor_1789_max_sizes[] = {768, 127, 32};
    unsigned      g_0_tensor_1789_min_sizes[] = {768, 127, 32};
    unsigned      g_0_tensor_1789             = createTensors(1,
                                             OUTPUT_TENSOR,
                                             true,
                                             "g_0_tensor_1789",
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             g_0_tensor_1789_max_sizes,
                                             3,
                                             syn_type_bf16,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             false,
                                             g_0_tensor_1789_min_sizes,
                                             synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_n1010__hpu_matmul_backward_batch_gemm_0_id;
    unsigned char g_0_n1010__hpu_matmul_backward_batch_gemm_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {g_0_tensor_1781_t9721_n1008__aten_gelu_backward_0, g_0_tensor_1788},
                   {g_0_tensor_1789},
                   (void*)g_0_n1010__hpu_matmul_backward_batch_gemm_0_params,
                   2,
                   "g_0_n1010__hpu_matmul_backward_batch_gemm_0",
                   0 /*graphIndex*/,
                   &g_0_n1010__hpu_matmul_backward_batch_gemm_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_tensor_1789});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, conv_add_with_shared_operand, {synDeviceGaudi})
{
    unsigned g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0_max_sizes[] = {64, 8, 8, 8};
    unsigned g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0_min_sizes[] = {64, 8, 8, 8};
    unsigned g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1762_l2loss_149_readvariableop_0_max_sizes[] = {36, 64, 1, 1};
    unsigned g_0_t1762_l2loss_149_readvariableop_0_min_sizes[] = {36, 64, 1, 1};
    unsigned g_0_t1762_l2loss_149_readvariableop_0             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_t1762_l2loss_149_readvariableop_0",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_t1762_l2loss_149_readvariableop_0_max_sizes,
                                                                   4,
                                                                   syn_type_single,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_t1762_l2loss_149_readvariableop_0_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t9281_box_net_box_predict_3_separable_conv2d_0_max_sizes[] = {36, 8, 8, 8};
    unsigned g_0_t9281_box_net_box_predict_3_separable_conv2d_0_min_sizes[] = {36, 8, 8, 8};
    unsigned g_0_t9281_box_net_box_predict_3_separable_conv2d_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t9281_box_net_box_predict_3_separable_conv2d_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t9281_box_net_box_predict_3_separable_conv2d_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t9281_box_net_box_predict_3_separable_conv2d_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_box_net_box_predict_3_separable_conv2d_spatial_convolution_n5195_0_id;
    unsigned char g_0_box_net_box_predict_3_separable_conv2d_spatial_convolution_n5195_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 157, 232, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   253, 127, 0, 0};
    addNodeToGraph(
        "spatial_convolution",
        {g_0_t9280_box_net_box_predict_3_separable_conv2d_depthwise_0, g_0_t1762_l2loss_149_readvariableop_0},
        {g_0_t9281_box_net_box_predict_3_separable_conv2d_0},
        (void*)g_0_box_net_box_predict_3_separable_conv2d_spatial_convolution_n5195_0_params,
        72,
        "g_0_box_net_box_predict_3_separable_conv2d_spatial_convolution_n5195_0",
        0 /*graphIndex*/,
        &g_0_box_net_box_predict_3_separable_conv2d_spatial_convolution_n5195_0_id);

    unsigned g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0_max_sizes[] = {64, 4, 4, 8};
    unsigned g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0_min_sizes[] = {64, 4, 4, 8};
    unsigned g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t9668_box_net_box_predict_4_separable_conv2d_0_max_sizes[] = {36, 4, 4, 8};
    unsigned g_0_t9668_box_net_box_predict_4_separable_conv2d_0_min_sizes[] = {36, 4, 4, 8};
    unsigned g_0_t9668_box_net_box_predict_4_separable_conv2d_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t9668_box_net_box_predict_4_separable_conv2d_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t9668_box_net_box_predict_4_separable_conv2d_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t9668_box_net_box_predict_4_separable_conv2d_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_box_net_box_predict_4_separable_conv2d_spatial_convolution_n5490_0_id;
    unsigned char g_0_box_net_box_predict_4_separable_conv2d_spatial_convolution_n5490_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 157, 232, 1,   0,   0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   253, 127, 0, 0};
    addNodeToGraph(
        "spatial_convolution",
        {g_0_t9667_box_net_box_predict_4_separable_conv2d_depthwise_0, g_0_t1762_l2loss_149_readvariableop_0},
        {g_0_t9668_box_net_box_predict_4_separable_conv2d_0},
        (void*)g_0_box_net_box_predict_4_separable_conv2d_spatial_convolution_n5490_0_params,
        72,
        "g_0_box_net_box_predict_4_separable_conv2d_spatial_convolution_n5490_0",
        0 /*graphIndex*/,
        &g_0_box_net_box_predict_4_separable_conv2d_spatial_convolution_n5490_0_id);

    unsigned g_0_t9670_box_net_box_predict_4_BiasAdd_max_sizes[] = {36, 1, 1, 1};
    unsigned g_0_t9670_box_net_box_predict_4_BiasAdd_min_sizes[] = {36, 1, 1, 1};
    unsigned g_0_t9670_box_net_box_predict_4_BiasAdd             = createTensors(1,
                                                                     INPUT_TENSOR,
                                                                     true,
                                                                     "g_0_t9670_box_net_box_predict_4_BiasAdd",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_t9670_box_net_box_predict_4_BiasAdd_max_sizes,
                                                                     4,
                                                                     syn_type_single,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_t9670_box_net_box_predict_4_BiasAdd_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t9669_box_net_box_predict_4_BiasAdd_0_max_sizes[] = {36, 4, 4, 8};
    unsigned g_0_t9669_box_net_box_predict_4_BiasAdd_0_min_sizes[] = {36, 4, 4, 8};
    unsigned g_0_t9669_box_net_box_predict_4_BiasAdd_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t9669_box_net_box_predict_4_BiasAdd_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t9669_box_net_box_predict_4_BiasAdd_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t9669_box_net_box_predict_4_BiasAdd_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_box_net_box_predict_4_BiasAdd_add_fwd_f32_n5492_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t9668_box_net_box_predict_4_separable_conv2d_0, g_0_t9670_box_net_box_predict_4_BiasAdd},
                   {g_0_t9669_box_net_box_predict_4_BiasAdd_0},
                   nullptr,
                   0,
                   "g_0_box_net_box_predict_4_BiasAdd_add_fwd_f32_n5492_0",
                   0 /*graphIndex*/,
                   &g_0_box_net_box_predict_4_BiasAdd_add_fwd_f32_n5492_0_id);

    unsigned g_0_t9282_box_net_box_predict_3_BiasAdd_0_max_sizes[] = {36, 8, 8, 8};
    unsigned g_0_t9282_box_net_box_predict_3_BiasAdd_0_min_sizes[] = {36, 8, 8, 8};
    unsigned g_0_t9282_box_net_box_predict_3_BiasAdd_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t9282_box_net_box_predict_3_BiasAdd_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t9282_box_net_box_predict_3_BiasAdd_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t9282_box_net_box_predict_3_BiasAdd_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_box_net_box_predict_3_BiasAdd_add_fwd_f32_n5197_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t9281_box_net_box_predict_3_separable_conv2d_0, g_0_t9670_box_net_box_predict_4_BiasAdd},
                   {g_0_t9282_box_net_box_predict_3_BiasAdd_0},
                   nullptr,
                   0,
                   "g_0_box_net_box_predict_3_BiasAdd_add_fwd_f32_n5197_0",
                   0 /*graphIndex*/,
                   &g_0_box_net_box_predict_3_BiasAdd_add_fwd_f32_n5197_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_t9669_box_net_box_predict_4_BiasAdd_0, g_0_t9282_box_net_box_predict_3_BiasAdd_0});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, dedx_dedw_bn_with_shared_operand, {synDeviceGaudi})
{
    unsigned
        g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes
            [] = {384, 14, 14, 96};
    unsigned
        g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes
            [] = {384, 14, 14, 96};
    unsigned g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_"
            "FusedBatchNormGradV3_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0_max_sizes[] =
        {384, 64, 1, 1};
    unsigned g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0_min_sizes[] =
        {384, 64, 1, 1};
    unsigned g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_max_sizes[] = {64,
                                                                                                                    14,
                                                                                                                    14,
                                                                                                                    96};
    unsigned g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_min_sizes[] = {64,
                                                                                                                    14,
                                                                                                                    14,
                                                                                                                    96};
    unsigned g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0_max_sizes[] =
        {64, 14, 14, 96};
    unsigned g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0_min_sizes[] =
        {64, 14, 14, 96};
    unsigned g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0 = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_dedx_n2516_0_id;
    unsigned char
        g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_dedx_n2516_0_params[] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,  0,   0,   0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 169, 94, 1,   0,   0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,  253, 127, 0, 0};
    addNodeToGraph(
        "dedx",
        {g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t1656_MobilenetV2_expanded_conv_10_expand_Conv2D_ReadVariableOp_fp32_to_bf16_cast_435_0,
         g_0_t4548_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput},
        {g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0},
        (void*)g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_dedx_n2516_0_params,
        72,
        "g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_dedx_n2516_0",
        0 /*graphIndex*/,
        &g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_dedx_n2516_0_id);

    unsigned g_0_t2467_MobilenetV2_expanded_conv_9_add_0_max_sizes[] = {64, 14, 14, 96};
    unsigned g_0_t2467_MobilenetV2_expanded_conv_9_add_0_min_sizes[] = {64, 14, 14, 96};
    unsigned g_0_t2467_MobilenetV2_expanded_conv_9_add_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2467_MobilenetV2_expanded_conv_9_add_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2467_MobilenetV2_expanded_conv_9_add_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2467_MobilenetV2_expanded_conv_9_add_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes[] =
        {384, 64, 1, 1};
    unsigned g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes[] =
        {384, 64, 1, 1};
    unsigned g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_dedw_n2517_0_id;
    unsigned char
        g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_dedw_n2517_0_params[] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,  0,   0,   0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 169, 94, 1,   0,   0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,  253, 127, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_t4506_gradients_MobilenetV2_expanded_conv_10_expand_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t2467_MobilenetV2_expanded_conv_9_add_0},
        {g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0},
        (void*)g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_dedw_n2517_0_params,
        72,
        "g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_dedw_n2517_0",
        0 /*graphIndex*/,
        &g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_dedw_n2517_0_id);

    unsigned g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0_max_sizes[] = {64, 14, 14, 96};
    unsigned g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0_min_sizes[] = {64, 14, 14, 96};
    unsigned g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1_max_sizes[] = {64};
    unsigned g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1_min_sizes[] = {64};
    unsigned g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes
            [] = {64};
    unsigned
        g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes
            [] = {64};
    unsigned g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3 =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_"
            "FusedBatchNormGradV3",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes
            [] = {64, 14, 14, 96};
    unsigned
        g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes
            [] = {64, 14, 14, 96};
    unsigned g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_"
            "FusedBatchNormGradV3_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes
            [] = {64};
    unsigned
        g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes
            [] = {64};
    unsigned g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_"
            "FusedBatchNormGradV3_2",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes
            [] = {64};
    unsigned
        g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes
            [] = {64};
    unsigned g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_"
            "FusedBatchNormGradV3_1",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_max_sizes,
            1,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    synNodeId
        g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0_id;
    unsigned char
        g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 111, 18, 131, 58, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t2443_MobilenetV2_expanded_conv_9_project_Conv2D_0,
         g_0_t4547_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropInput_0,
         g_0_t2445_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_1,
         g_0_t4575_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3,
         g_0_t1035_mobilenetv2_expanded_conv_9_project_batchnorm_readvariableop_0},
        {g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2,
         g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1},
        (void*)
            g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0_params,
        16,
        "g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_"
        "norm_bwd_bf16_n2539_0",
        0 /*graphIndex*/,
        &g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0_id);

    synNodeId
        blocking_g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0
            [] = {g_0_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_dedw_n2517_0_id};
    setNodeDependency(
        blocking_g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0,
        &g_0_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_batch_norm_bwd_bf16_n2539_0_id,
        1,
        1);

    setConfigsForTest();

    compareRunsResults(
        {g_0_t4549_gradients_MobilenetV2_expanded_conv_10_expand_Conv2D_grad_Conv2DBackpropFilter_0,
         g_0_t4569_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_0,
         g_0_t4571_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_2,
         g_0_t4570_gradients_MobilenetV2_expanded_conv_9_project_BatchNorm_FusedBatchNormV3_grad_FusedBatchNormGradV3_1});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, tpc_bundle_with_parallel_nodes_ASIC_CI, {synDeviceGaudi})
{
    ScopedConfigurationChange fuserCfg("RUN_TPC_FUSER", "false");

    unsigned gemmIn1Sizes[] = {4096, 7168};
    unsigned gemmIn1        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmIn1",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemmIn1Sizes,
                                     2,
                                     syn_type_bf16)[0];

    unsigned gemmIn2Sizes[] = {1024, 4096};
    unsigned gemmIn2        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "gemmIn2",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemmIn2Sizes,
                                     2,
                                     syn_type_bf16)[0];

    unsigned gemmOutSizes[] = {1024, 7168};
    unsigned gemmOut        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "gemmOut",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     gemmOutSizes,
                                     2,
                                     syn_type_bf16)[0];

    unsigned char gemmParams[] = {0, 0};
    addNodeToGraph("gemm", {gemmIn1, gemmIn2}, {gemmOut}, (void*)gemmParams, 2, "gemm");

    unsigned add1InSizes[] = {1024, 1};
    unsigned add1In        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "add1In",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    add1InSizes,
                                    2,
                                    syn_type_bf16)[0];

    unsigned add1OutSizes[] = {1024, 7168};
    unsigned add1Out        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "add1Out",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     add1OutSizes,
                                     2,
                                     syn_type_bf16)[0];

    addNodeToGraph("add_fwd_bf16", {gemmOut, add1In}, {add1Out}, nullptr, 0, "add1");

    unsigned relu1OutSizes[] = {1024, 7168};
    unsigned relu1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "relu1Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      relu1OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("relu_fwd_bf16", {add1Out}, {relu1Out}, nullptr, 0, "relu1");

    unsigned add2InSizes[] = {1024, 7168};
    unsigned add2In        = createTensors(1,
                                    INPUT_TENSOR,
                                    true,
                                    "add2In",
                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                    nullptr,
                                    add2InSizes,
                                    2,
                                    syn_type_bf16)[0];

    unsigned add2OutSizes[] = {1024, 7168};
    unsigned add2Out        = createTensors(1,
                                     OUTPUT_TENSOR,
                                     false,
                                     "add2Out",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     add2OutSizes,
                                     2,
                                     syn_type_bf16)[0];

    addNodeToGraph("add_fwd_bf16", {add2In, relu1Out}, {add2Out}, nullptr, 0, "add2");

    unsigned relu2OutSizes[] = {1024, 7168};
    unsigned relu2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "relu2Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      relu2OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("relu_fwd_bf16", {add2Out}, {relu2Out}, nullptr, 0, "relu2");

    unsigned cast1InSizes[] = {1024, 1024};
    unsigned cast1In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "cast1In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     cast1InSizes,
                                     2,
                                     syn_type_single)[0];

    unsigned cast1OutSizes[] = {1024, 1024};
    unsigned cast1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "cast1Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      cast1OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("cast_f32_to_bf16", {cast1In}, {cast1Out}, nullptr, 0, "cast1");

    unsigned gemm1OutSizes[] = {1024, 7168};
    unsigned gemm1Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm1Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gemm1OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    unsigned char gemm1Params[] = {0, 0};
    addNodeToGraph("gemm", {relu2Out, cast1Out}, {gemm1Out}, (void*)gemm1Params, 2, "gemm1");

    unsigned cast2InSizes[] = {1024, 1024};
    unsigned cast2In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "cast2In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     cast2InSizes,
                                     2,
                                     syn_type_single)[0];

    unsigned cast2OutSizes[] = {1024, 1024};
    unsigned cast2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "cast2Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      cast2OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("cast_f32_to_bf16", {cast2In}, {cast2Out}, nullptr, 0, "cast2");

    unsigned gemm2OutSizes[] = {1024, 7168};
    unsigned gemm2Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm2Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gemm2OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    unsigned char gemm2Params[] = {0, 0};
    addNodeToGraph("gemm", {relu2Out, cast2Out}, {gemm2Out}, (void*)gemm2Params, 2, "gemm2");

    unsigned cast3InSizes[] = {1024, 1024};
    unsigned cast3In        = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "cast3In",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     cast3InSizes,
                                     2,
                                     syn_type_single)[0];

    unsigned cast3OutSizes[] = {1024, 1024};
    unsigned cast3Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "cast3Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      cast3OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    addNodeToGraph("cast_f32_to_bf16", {cast3In}, {cast3Out}, nullptr, 0, "cast3");

    unsigned gemm3OutSizes[] = {1024, 7168};
    unsigned gemm3Out        = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "gemm3Out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      gemm3OutSizes,
                                      2,
                                      syn_type_bf16)[0];

    unsigned char gemm3Params[] = {0, 0};
    addNodeToGraph("gemm", {relu2Out, cast3Out}, {gemm3Out}, (void*)gemm3Params, 2, "gemm3");

    setConfigsForTest();

    compareRunsResults({gemm1Out, gemm2Out, gemm3Out});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, mme_with_reshaped_producers_and_consumer_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_Tensor_0_max_sizes[] = {14336, 28, 256};
    unsigned g_0_Tensor_0_min_sizes[] = {14336, 28, 256};
    unsigned g_0_Tensor_0             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_Tensor_0",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_0_max_sizes,
                                          3,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_0_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_Tensor_1_max_sizes[] = {14336, 28, 256};
    unsigned  g_0_Tensor_1_min_sizes[] = {14336, 28, 256};
    unsigned  g_0_Tensor_1             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "g_0_Tensor_1",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_1_max_sizes,
                                          3,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_1_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_reluA_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_Tensor_0},
                   {g_0_Tensor_1},
                   nullptr,
                   0,
                   "g_0_reluA_0",
                   0 /*graphIndex*/,
                   &g_0_reluA_0_id);

    unsigned  g_0_Tensor_2_max_sizes[] = {512, 28, 28, 256};
    unsigned  g_0_Tensor_2_min_sizes[] = {512, 28, 28, 256};
    unsigned  g_0_Tensor_2             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "g_0_Tensor_2",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_2_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_2_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_reshapeA_0_id;
    addNodeToGraph("reshape",
                   {g_0_Tensor_1},
                   {g_0_Tensor_2},
                   nullptr,
                   0,
                   "g_0_reshapeA_0",
                   0 /*graphIndex*/,
                   &g_0_reshapeA_0_id);

    unsigned g_0_Tensor_3_max_sizes[] = {256, 4, 128, 1};
    unsigned g_0_Tensor_3_min_sizes[] = {256, 4, 128, 1};
    unsigned g_0_Tensor_3             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_Tensor_3",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_3_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_3_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_Tensor_4_max_sizes[] = {256, 4, 128, 1};
    unsigned  g_0_Tensor_4_min_sizes[] = {256, 4, 128, 1};
    unsigned  g_0_Tensor_4             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "g_0_Tensor_4",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_4_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_4_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_reluB_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_Tensor_3},
                   {g_0_Tensor_4},
                   nullptr,
                   0,
                   "g_0_reluB_0",
                   0 /*graphIndex*/,
                   &g_0_reluB_0_id);

    unsigned  g_0_Tensor_5_max_sizes[] = {256, 512, 1, 1};
    unsigned  g_0_Tensor_5_min_sizes[] = {256, 512, 1, 1};
    unsigned  g_0_Tensor_5             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "g_0_Tensor_5",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_5_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_5_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_reshapeB_0_id;
    addNodeToGraph("reshape",
                   {g_0_Tensor_4},
                   {g_0_Tensor_5},
                   nullptr,
                   0,
                   "g_0_reshapeB_0",
                   0 /*graphIndex*/,
                   &g_0_reshapeB_0_id);

    unsigned      g_0_Tensor_6_max_sizes[] = {256, 28, 28, 256};
    unsigned      g_0_Tensor_6_min_sizes[] = {256, 28, 28, 256};
    unsigned      g_0_Tensor_6             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "g_0_Tensor_6",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_6_max_sizes,
                                          4,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_6_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_conv_0_id;
    unsigned char g_0_conv_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,  1,  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 46, 83, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_Tensor_2, g_0_Tensor_5},
                   {g_0_Tensor_6},
                   (void*)g_0_conv_0_params,
                   104,
                   "g_0_conv_0",
                   0 /*graphIndex*/,
                   &g_0_conv_0_id);

    unsigned  g_0_Tensor_7_max_sizes[] = {7168, 28, 256};
    unsigned  g_0_Tensor_7_min_sizes[] = {7168, 28, 256};
    unsigned  g_0_Tensor_7             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          false,
                                          "g_0_Tensor_7",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_7_max_sizes,
                                          3,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_7_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_reshapeOut_0_id;
    addNodeToGraph("reshape",
                   {g_0_Tensor_6},
                   {g_0_Tensor_7},
                   nullptr,
                   0,
                   "g_0_reshapeOut_0",
                   0 /*graphIndex*/,
                   &g_0_reshapeOut_0_id);

    unsigned  g_0_Tensor_8_max_sizes[] = {7168, 28, 256};
    unsigned  g_0_Tensor_8_min_sizes[] = {7168, 28, 256};
    unsigned  g_0_Tensor_8             = createTensors(1,
                                          OUTPUT_TENSOR,
                                          true,
                                          "g_0_Tensor_8",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_Tensor_8_max_sizes,
                                          3,
                                          syn_type_bf16,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_Tensor_8_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_reluConsumer_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_Tensor_7},
                   {g_0_Tensor_8},
                   nullptr,
                   0,
                   "g_0_reluConsumer_0",
                   0 /*graphIndex*/,
                   &g_0_reluConsumer_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_Tensor_8});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests,
          resnet_bwd_shared_input_bpt_for_two_bundled_nodes_ASIC_CI,
          {synDeviceGaudi})
{
    unsigned g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0_max_sizes[] = {256, 14, 14, 256};
    unsigned g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0_min_sizes[] = {256, 14, 14, 256};
    unsigned g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0_max_sizes[] = {256,
                                                                                                          14,
                                                                                                          14,
                                                                                                          256};
    unsigned g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0_min_sizes[] = {256,
                                                                                                          14,
                                                                                                          14,
                                                                                                          256};
    unsigned g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3_max_sizes[] = {256};
    unsigned g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3_min_sizes[] = {256};
    unsigned g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_max_sizes[] = {
        256};
    unsigned g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_min_sizes[] = {
        256};
    unsigned g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0_max_sizes[] =
        {256, 14, 14, 256};
    unsigned g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0_min_sizes[] =
        {256, 14, 14, 256};
    unsigned g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0 = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2_max_sizes[] = {
        256};
    unsigned g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2_min_sizes[] = {
        256};
    unsigned g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1_max_sizes[] = {
        256};
    unsigned g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1_min_sizes[] = {
        256};
    unsigned g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1081_0_id;
    unsigned char
        g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1081_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t1577_while_body__1_while_resnet50_res4c_branch2b_Conv2D_0,
         g_0_t2468_while_body__1_gradient_tape_while_resnet50_activation_29_ReluGrad_0,
         g_0_t1586_while_body__1_while_resnet50_bn4c_branch2b_FusedBatchNormV3_3,
         g_0_t2475_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3,
         g_0_t697_while_body__1_while_resnet50_bn4c_branch2b_readvariableop_0},
        {g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0,
         g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2,
         g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1},
        (void*)
            g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1081_0_params,
        16,
        "g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1081_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1081_0_id);

    unsigned g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0_max_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0_min_sizes[] = {256, 256, 3, 3};
    unsigned g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_max_sizes[] =
            {256, 14, 14, 256};
    unsigned
        g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_min_sizes[] =
            {256, 14, 14, 256};
    unsigned g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput =
        createTensors(
            1,
            INPUT_TENSOR,
            false,
            "g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_max_sizes,
            4,
            syn_type_uint32,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_min_sizes,
            synTensorType::SHAPE_TENSOR)[0];

    unsigned
        g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0_max_sizes[] =
            {256, 14, 14, 256};
    unsigned
        g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0_min_sizes[] =
            {256, 14, 14, 256};
    unsigned g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_dedx_n1082_0_id;
    unsigned char
        g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_dedx_n1082_0_params[] =
            {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 209, 216, 1,   0,   0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   255, 127, 0, 0};
    addNodeToGraph(
        "dedx",
        {g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0,
         g_0_t881_while_body__1_while_resnet50_res4c_branch2b_Conv2D_Cast_0,
         g_0_t2479_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput},
        {g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0},
        (void*)
            g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_dedx_n1082_0_params,
        72,
        "g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_dedx_n1082_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_dedx_n1082_0_id);

    unsigned g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0_max_sizes[] = {256, 14, 14, 256};
    unsigned g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0_min_sizes[] = {256, 14, 14, 256};
    unsigned g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0_max_sizes[] =
            {256, 256, 3, 3};
    unsigned
        g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0_min_sizes[] =
            {256, 256, 3, 3};
    unsigned g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_dedw_n1095_0_id;
    unsigned char
        g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_dedw_n1095_0_params
            [] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,   1,   0,   0, 0,
                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 209, 216, 1,   0,   0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   255, 127, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_t2469_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_0,
         g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0},
        {g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0},
        (void*)
            g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_dedw_n1095_0_params,
        72,
        "g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_dedw_n1095_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_dedw_n1095_0_id);

    unsigned g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0_max_sizes[] = {256,
                                                                                                                   256,
                                                                                                                   3,
                                                                                                                   3};
    unsigned g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0_min_sizes[] = {256,
                                                                                                                   256,
                                                                                                                   3,
                                                                                                                   3};
    unsigned g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_cast_bf16_to_f32_n1096_0_id;
    addNodeToGraph(
        "cast_bf16_to_f32",
        {g_0_t2501_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropFilter_0},
        {g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0},
        nullptr,
        0,
        "g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_cast_bf16_to_f32_n1096_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_cast_bf16_to_f32_n1096_0_id);

    unsigned g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0_max_sizes[] = {256,
                                                                                                          14,
                                                                                                          14,
                                                                                                          256};
    unsigned g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0_min_sizes[] = {256,
                                                                                                          14,
                                                                                                          14,
                                                                                                          256};
    unsigned g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_relu_bwd_bf16_n1083_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_t2478_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Conv2DBackpropInput_0,
                    g_0_t1576_while_body__1_while_resnet50_activation_28_Relu_0},
                   {g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0},
                   nullptr,
                   0,
                   "g_0_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_relu_bwd_bf16_n1083_0",
                   0 /*graphIndex*/,
                   &g_0_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_relu_bwd_bf16_n1083_0_id);

    unsigned g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0_max_sizes[] = {256, 14, 14, 256};
    unsigned g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0_min_sizes[] = {256, 14, 14, 256};
    unsigned g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3_max_sizes[] = {256};
    unsigned g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3_min_sizes[] = {256};
    unsigned g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_max_sizes[] = {
        256};
    unsigned g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_min_sizes[] = {
        256};
    unsigned g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0_max_sizes[] =
        {256, 14, 14, 256};
    unsigned g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0_min_sizes[] =
        {256, 14, 14, 256};
    unsigned g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2_max_sizes[] = {
        256};
    unsigned g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2_min_sizes[] = {
        256};
    unsigned g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1_max_sizes[] = {
        256};
    unsigned g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1_min_sizes[] = {
        256};
    unsigned g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1 = createTensors(
        1,
        OUTPUT_TENSOR,
        true,
        "g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1_max_sizes,
        1,
        syn_type_single,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId
        g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1088_0_id;
    unsigned char
        g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1088_0_params
            [] = {149, 191, 214, 51, 205, 204, 204, 61, 159, 240, 39, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t1556_while_body__1_while_resnet50_res4c_branch2a_Conv2D_0,
         g_0_t2480_while_body__1_gradient_tape_while_resnet50_activation_28_ReluGrad_0,
         g_0_t1565_while_body__1_while_resnet50_bn4c_branch2a_FusedBatchNormV3_3,
         g_0_t2487_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3,
         g_0_t693_while_body__1_while_resnet50_bn4c_branch2a_readvariableop_0},
        {g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0,
         g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2,
         g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1},
        (void*)
            g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1088_0_params,
        16,
        "g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1088_0",
        0 /*graphIndex*/,
        &g_0_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_batch_norm_bwd_bf16_n1088_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_t2471_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_2,
                        g_0_t2470_while_body__1_gradient_tape_while_resnet50_bn4c_branch2b_FusedBatchNormGradV3_1,
                        g_0_t2503_while_body__1_gradient_tape_while_resnet50_res4c_branch2b_Conv2D_Cast_Cast_0,
                        g_0_t2481_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_0,
                        g_0_t2483_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_2,
                        g_0_t2482_while_body__1_gradient_tape_while_resnet50_bn4c_branch2a_FusedBatchNormGradV3_1});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, densenet_shared_input_bpt_for_two_bundled_nodes_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_t7161_densenet_conv3_2_x1_Conv2D_0_max_sizes[] = {128, 28, 28, 256};
    unsigned g_0_t7161_densenet_conv3_2_x1_Conv2D_0_min_sizes[] = {128, 28, 28, 256};
    unsigned g_0_t7161_densenet_conv3_2_x1_Conv2D_0             = createTensors(1,
                                                                    INPUT_TENSOR,
                                                                    true,
                                                                    "g_0_t7161_densenet_conv3_2_x1_Conv2D_0",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_t7161_densenet_conv3_2_x1_Conv2D_0_max_sizes,
                                                                    4,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_t7161_densenet_conv3_2_x1_Conv2D_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0_max_sizes[] = {128, 28, 28, 256};
    unsigned g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0_min_sizes[] = {128, 28, 28, 256};
    unsigned g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3_max_sizes[] = {128};
    unsigned g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3_min_sizes[] = {128};
    unsigned g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_max_sizes[] = {128};
    unsigned g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_min_sizes[] = {128};
    unsigned g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0_max_sizes[] = {128, 28, 28, 256};
    unsigned g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0_min_sizes[] = {128, 28, 28, 256};
    unsigned g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2_max_sizes[] = {128};
    unsigned g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2_min_sizes[] = {128};
    unsigned g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1_max_sizes[] = {128};
    unsigned g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1_min_sizes[] = {128};
    unsigned g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6385_0_id;
    unsigned char g_0_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6385_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 164, 140, 56, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t7161_densenet_conv3_2_x1_Conv2D_0,
         g_0_t12844_gradient_tape_densenet_relu3_2_x2_ReluGrad_0,
         g_0_t7170_densenet_conv3_2_x2_bn_FusedBatchNormV3_3,
         g_0_t12851_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3,
         g_0_t1964_conv3_2_x2_bn_gamma_regularizer_square_readvariableop_0},
        {g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0,
         g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2,
         g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1},
        (void*)g_0_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6385_0_params,
        16,
        "g_0_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6385_0",
        0 /*graphIndex*/,
        &g_0_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6385_0_id);

    unsigned g_0_t7160_densenet_relu3_2_x1_Relu_0_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t7160_densenet_relu3_2_x1_Relu_0_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t7160_densenet_relu3_2_x1_Relu_0             = createTensors(1,
                                                                  INPUT_TENSOR,
                                                                  true,
                                                                  "g_0_t7160_densenet_relu3_2_x1_Relu_0",
                                                                  MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                  nullptr,
                                                                  g_0_t7160_densenet_relu3_2_x1_Relu_0_max_sizes,
                                                                  4,
                                                                  syn_type_bf16,
                                                                  nullptr,
                                                                  0,
                                                                  0,
                                                                  nullptr,
                                                                  false,
                                                                  g_0_t7160_densenet_relu3_2_x1_Relu_0_min_sizes,
                                                                  synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0_max_sizes[] = {128, 160, 1, 1};
    unsigned g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0_min_sizes[] = {128, 160, 1, 1};
    unsigned g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_dedw_n6386_0_id;
    unsigned char g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_dedw_n6386_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,   1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 144, 206, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0, g_0_t7160_densenet_relu3_2_x1_Relu_0},
        {g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0},
        (void*)g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_dedw_n6386_0_params,
        104,
        "g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_dedw_n6386_0",
        0 /*graphIndex*/,
        &g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_dedw_n6386_0_id);

    unsigned g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0_max_sizes[] = {128, 160, 1, 1};
    unsigned g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0_min_sizes[] = {128, 160, 1, 1};
    unsigned g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_dedx_n6387_0_id;
    unsigned char g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_dedx_n6387_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,  1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 61, 102, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_t12845_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_0,
                    g_0_t4096_densenet_conv3_2_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_814_0,
                    g_0_t12856_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput},
                   {g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0},
                   (void*)g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_dedx_n6387_0_params,
                   104,
                   "g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_dedx_n6387_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_dedx_n6387_0_id);

    unsigned g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_densenet_relu3_2_x1_ReluGrad_relu_bwd_bf16_n6388_0_id;
    addNodeToGraph("relu_bwd_bf16",
                   {g_0_t12855_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropInput_0,
                    g_0_t7160_densenet_relu3_2_x1_Relu_0},
                   {g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0},
                   nullptr,
                   0,
                   "g_0_gradient_tape_densenet_relu3_2_x1_ReluGrad_relu_bwd_bf16_n6388_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_densenet_relu3_2_x1_ReluGrad_relu_bwd_bf16_n6388_0_id);

    unsigned g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3_max_sizes[] = {160};
    unsigned g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3_min_sizes[] = {160};
    unsigned g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_max_sizes[] = {160};
    unsigned g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_min_sizes[] = {160};
    unsigned g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0_max_sizes[] = {160};
    unsigned g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0_min_sizes[] = {160};
    unsigned g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2_max_sizes[] = {160};
    unsigned g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2_min_sizes[] = {160};
    unsigned g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1_max_sizes[] = {160};
    unsigned g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1_min_sizes[] = {160};
    unsigned g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6393_0_id;
    unsigned char g_0_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6393_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 164, 140, 56, 55, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_bf16",
        {g_0_t7140_densenet_concatenate_6_concat_fp32_to_bf16_cast_813_0,
         g_0_t12857_gradient_tape_densenet_relu3_2_x1_ReluGrad_0,
         g_0_t7149_densenet_conv3_2_x1_bn_FusedBatchNormV3_3,
         g_0_t12864_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3,
         g_0_t1961_conv3_2_x1_bn_gamma_regularizer_square_readvariableop_0},
        {g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0,
         g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2,
         g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1},
        (void*)g_0_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6393_0_params,
        16,
        "g_0_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6393_0",
        0 /*graphIndex*/,
        &g_0_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_batch_norm_bwd_bf16_n6393_0_id);

    unsigned g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0_max_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0_min_sizes[] = {160, 28, 28, 256};
    unsigned g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_t12867_SGD_gradients_AddN_50_0_max_sizes[] = {160, 28, 28, 256};
    unsigned  g_0_t12867_SGD_gradients_AddN_50_0_min_sizes[] = {160, 28, 28, 256};
    unsigned  g_0_t12867_SGD_gradients_AddN_50_0             = createTensors(1,
                                                                OUTPUT_TENSOR,
                                                                true,
                                                                "g_0_t12867_SGD_gradients_AddN_50_0",
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                g_0_t12867_SGD_gradients_AddN_50_0_max_sizes,
                                                                4,
                                                                syn_type_bf16,
                                                                nullptr,
                                                                0,
                                                                0,
                                                                nullptr,
                                                                false,
                                                                g_0_t12867_SGD_gradients_AddN_50_0_min_sizes,
                                                                synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_SGD_gradients_AddN_50_add_fwd_bf16_n6394_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_t12835_gradient_tape_densenet_concatenate_7_Slice_0,
                    g_0_t12858_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_0},
                   {g_0_t12867_SGD_gradients_AddN_50_0},
                   nullptr,
                   0,
                   "g_0_SGD_gradients_AddN_50_add_fwd_bf16_n6394_0",
                   0 /*graphIndex*/,
                   &g_0_SGD_gradients_AddN_50_add_fwd_bf16_n6394_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_t12847_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_2,
                        g_0_t12846_gradient_tape_densenet_conv3_2_x2_bn_FusedBatchNormGradV3_1,
                        g_0_t12854_gradient_tape_densenet_conv3_2_x1_Conv2D_Conv2DBackpropFilter_0,
                        g_0_t12860_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_2,
                        g_0_t12859_gradient_tape_densenet_conv3_2_x1_bn_FusedBatchNormGradV3_1,
                        g_0_t12867_SGD_gradients_AddN_50_0});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, densenet_consumer_with_two_same_input_tensors_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0_max_sizes[] = {128, 992, 1, 1};
    unsigned g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0_min_sizes[] = {128, 992, 1, 1};
    unsigned g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0_max_sizes[] = {128,
                                                                                                          992,
                                                                                                          1,
                                                                                                          1};
    unsigned g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0_min_sizes[] = {128,
                                                                                                          992,
                                                                                                          1,
                                                                                                          1};
    unsigned g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_cast_f32_to_bf16_n10332_0_id;
    addNodeToGraph("cast_f32_to_bf16",
                   {g_0_t17438_densenet_conv5_16_x1_conv2d_readvariableop_0},
                   {g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0},
                   nullptr,
                   0,
                   "g_0_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_cast_f32_to_bf16_n10332_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_cast_f32_to_bf16_n10332_0_id);

    unsigned g_0_t19497_densenet_relu5_16_x1_Relu_0_max_sizes[] = {992, 7, 7, 256};
    unsigned g_0_t19497_densenet_relu5_16_x1_Relu_0_min_sizes[] = {992, 7, 7, 256};
    unsigned g_0_t19497_densenet_relu5_16_x1_Relu_0             = createTensors(1,
                                                                    INPUT_TENSOR,
                                                                    true,
                                                                    "g_0_t19497_densenet_relu5_16_x1_Relu_0",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_t19497_densenet_relu5_16_x1_Relu_0_max_sizes,
                                                                    4,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_t19497_densenet_relu5_16_x1_Relu_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t19498_densenet_conv5_16_x1_Conv2D_0_max_sizes[] = {128, 7, 7, 256};
    unsigned g_0_t19498_densenet_conv5_16_x1_Conv2D_0_min_sizes[] = {128, 7, 7, 256};
    unsigned g_0_t19498_densenet_conv5_16_x1_Conv2D_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t19498_densenet_conv5_16_x1_Conv2D_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t19498_densenet_conv5_16_x1_Conv2D_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t19498_densenet_conv5_16_x1_Conv2D_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_densenet_conv5_16_x1_Conv2D_spatial_convolution_n10938_0_id;
    unsigned char g_0_densenet_conv5_16_x1_Conv2D_spatial_convolution_n10938_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,   1,  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 134, 86, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t19497_densenet_relu5_16_x1_Relu_0,
                    g_0_t18108_densenet_conv5_16_x1_Conv2D_ReadVariableOp_fp32_to_bf16_cast_849_0},
                   {g_0_t19498_densenet_conv5_16_x1_Conv2D_0},
                   (void*)g_0_densenet_conv5_16_x1_Conv2D_spatial_convolution_n10938_0_params,
                   104,
                   "g_0_densenet_conv5_16_x1_Conv2D_spatial_convolution_n10938_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv5_16_x1_Conv2D_spatial_convolution_n10938_0_id);

    unsigned g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0_max_sizes[] = {128};
    unsigned g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0_min_sizes[] = {128};
    unsigned g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {128};
    unsigned g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {128};
    unsigned g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0_max_sizes[] = {128, 7, 7, 256};
    unsigned g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0_min_sizes[] = {128, 7, 7, 256};
    unsigned g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes[] = {128};
    unsigned g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes[] = {128};
    unsigned g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes[] = {128};
    unsigned g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes[] = {128};
    unsigned g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes[] = {128};
    unsigned g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes[] = {128};
    unsigned g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes[] = {128};
    unsigned g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes[] = {128};
    unsigned g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_densenet_conv5_16_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n10939_0_id;
    unsigned char g_0_densenet_conv5_16_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n10939_0_params[] =
        {149, 191, 214, 51, 0, 0, 128, 63, 164, 140, 56, 55, 0, 0, 0, 0};
    addNodeToGraph("batch_norm_fwd_bf16",
                   {g_0_t19498_densenet_conv5_16_x1_Conv2D_0,
                    g_0_t17978_densenet_conv5_16_x2_bn_readvariableop_1_0,
                    g_0_t17977_densenet_conv5_16_x2_bn_readvariableop_0,
                    g_0_t17979_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_0,
                    g_0_t17980_densenet_conv5_16_x2_bn_fusedbatchnormv3_readvariableop_1_0},
                   {g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0,
                    g_0_t19501_densenet_conv5_16_x2_bn_FusedBatchNormV3,
                    g_0_t19502_densenet_conv5_16_x2_bn_FusedBatchNormV3,
                    g_0_t19503_densenet_conv5_16_x2_bn_FusedBatchNormV3,
                    g_0_t19504_densenet_conv5_16_x2_bn_FusedBatchNormV3},
                   (void*)g_0_densenet_conv5_16_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n10939_0_params,
                   16,
                   "g_0_densenet_conv5_16_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n10939_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_conv5_16_x2_bn_FusedBatchNormV3_batch_norm_fwd_bf16_n10939_0_id);

    unsigned  g_0_t19505_densenet_relu5_16_x2_Relu_0_max_sizes[] = {128, 7, 7, 256};
    unsigned  g_0_t19505_densenet_relu5_16_x2_Relu_0_min_sizes[] = {128, 7, 7, 256};
    unsigned  g_0_t19505_densenet_relu5_16_x2_Relu_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    true,
                                                                    "g_0_t19505_densenet_relu5_16_x2_Relu_0",
                                                                    MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                    nullptr,
                                                                    g_0_t19505_densenet_relu5_16_x2_Relu_0_max_sizes,
                                                                    4,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_t19505_densenet_relu5_16_x2_Relu_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_densenet_relu5_16_x2_Relu_relu_fwd_bf16_n10940_0_id;
    addNodeToGraph("relu_fwd_bf16",
                   {g_0_t19499_densenet_conv5_16_x2_bn_FusedBatchNormV3_0},
                   {g_0_t19505_densenet_relu5_16_x2_Relu_0},
                   nullptr,
                   0,
                   "g_0_densenet_relu5_16_x2_Relu_relu_fwd_bf16_n10940_0",
                   0 /*graphIndex*/,
                   &g_0_densenet_relu5_16_x2_Relu_relu_fwd_bf16_n10940_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_t19505_densenet_relu5_16_x2_Relu_0});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, orange_shared_input_bpt_for_two_bundled_nodes_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_t3129_model_resUnit5_stride1_BiasAdd_0_max_sizes[] = {256, 28, 16, 32};
    unsigned g_0_t3129_model_resUnit5_stride1_BiasAdd_0_min_sizes[] = {256, 28, 16, 32};
    unsigned g_0_t3129_model_resUnit5_stride1_BiasAdd_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3129_model_resUnit5_stride1_BiasAdd_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3129_model_resUnit5_stride1_BiasAdd_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3129_model_resUnit5_stride1_BiasAdd_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0_max_sizes[] = {256, 28, 16, 32};
    unsigned g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0_min_sizes[] = {256, 28, 16, 32};
    unsigned g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3_max_sizes[] = {256};
    unsigned g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3_min_sizes[] = {256};
    unsigned g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_max_sizes[] = {256};
    unsigned g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_min_sizes[] = {256};
    unsigned g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1749_model_resunit5_stride1_bn_readvariableop_0_max_sizes[] = {256};
    unsigned g_0_t1749_model_resunit5_stride1_bn_readvariableop_0_min_sizes[] = {256};
    unsigned g_0_t1749_model_resunit5_stride1_bn_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1749_model_resunit5_stride1_bn_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1749_model_resunit5_stride1_bn_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1749_model_resunit5_stride1_bn_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0_max_sizes[] = {256, 28, 16, 32};
    unsigned g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0_min_sizes[] = {256, 28, 16, 32};
    unsigned g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2_max_sizes[] = {256};
    unsigned g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2_min_sizes[] = {256};
    unsigned g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1_max_sizes[] = {256};
    unsigned g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1_min_sizes[] = {256};
    unsigned g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3842_0_id;
    unsigned char g_0_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3842_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 111, 18, 131, 58, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_f32",
        {g_0_t3129_model_resUnit5_stride1_BiasAdd_0,
         g_0_t6857_gradient_tape_model_resUnit5_stride1_relu_activation_ReluGrad_0,
         g_0_t3140_model_resUnit5_stride1_bn_FusedBatchNormV3_3,
         g_0_t6864_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3,
         g_0_t1749_model_resunit5_stride1_bn_readvariableop_0},
        {g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0,
         g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2,
         g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1},
        (void*)g_0_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3842_0_params,
        16,
        "g_0_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3842_0",
        0 /*graphIndex*/,
        &g_0_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3842_0_id);

    unsigned g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0_max_sizes[] = {256, 128, 5, 1};
    unsigned g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0_min_sizes[] = {256, 128, 5, 1};
    unsigned g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0_max_sizes[] = {128,
                                                                                                        56,
                                                                                                        16,
                                                                                                        32};
    unsigned g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0_min_sizes[] = {128,
                                                                                                        56,
                                                                                                        16,
                                                                                                        32};
    unsigned g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_dedx_n3860_0_id;
    unsigned char g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_dedx_n3860_0_params[] = {
        5, 0, 0, 0, 1, 0, 0, 0,  2,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 96, 94, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0,
                    g_0_t1748_model_resunit5_stride1_conv2d_readvariableop_0,
                    g_0_t6889_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput},
                   {g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0},
                   (void*)g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_dedx_n3860_0_params,
                   104,
                   "g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_dedx_n3860_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_dedx_n3860_0_id);

    unsigned g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0_max_sizes[] = {256,
                                                                                                         128,
                                                                                                         5,
                                                                                                         1};
    unsigned g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0_min_sizes[] = {256,
                                                                                                         128,
                                                                                                         5,
                                                                                                         1};
    unsigned g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_dedw_n3843_0_id;
    unsigned char g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_dedw_n3843_0_params[] = {
        5, 0, 0, 0, 1, 0, 0, 0,  2,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 45, 22, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_t6858_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_0,
                    g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0},
                   {g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0},
                   (void*)g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_dedw_n3843_0_params,
                   104,
                   "g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_dedw_n3843_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_dedw_n3843_0_id);

    unsigned g_0_t6891_RectifiedAdam_gradients_AddN_25_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6891_RectifiedAdam_gradients_AddN_25_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6891_RectifiedAdam_gradients_AddN_25 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6891_RectifiedAdam_gradients_AddN_25",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6891_RectifiedAdam_gradients_AddN_25_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6891_RectifiedAdam_gradients_AddN_25_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6890_RectifiedAdam_gradients_AddN_25_0_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6890_RectifiedAdam_gradients_AddN_25_0_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6890_RectifiedAdam_gradients_AddN_25_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6890_RectifiedAdam_gradients_AddN_25_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6890_RectifiedAdam_gradients_AddN_25_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6890_RectifiedAdam_gradients_AddN_25_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_RectifiedAdam_gradients_AddN_25_add_fwd_f32_n3862_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t6888_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropInput_0,
                    g_0_t6891_RectifiedAdam_gradients_AddN_25},
                   {g_0_t6890_RectifiedAdam_gradients_AddN_25_0},
                   nullptr,
                   0,
                   "g_0_RectifiedAdam_gradients_AddN_25_add_fwd_f32_n3862_0",
                   0 /*graphIndex*/,
                   &g_0_RectifiedAdam_gradients_AddN_25_add_fwd_f32_n3862_0_id);

    unsigned g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_relu_bwd_f32_n3863_0_id;
    addNodeToGraph("relu_bwd_f32",
                   {g_0_t6890_RectifiedAdam_gradients_AddN_25_0, g_0_t3127_model_resUnit4_redu_relu_activation_Relu_0},
                   {g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0},
                   nullptr,
                   0,
                   "g_0_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_relu_bwd_f32_n3863_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_relu_bwd_f32_n3863_0_id);

    unsigned g_0_t3105_model_resUnit4_redu_BiasAdd_0_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t3105_model_resUnit4_redu_BiasAdd_0_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t3105_model_resUnit4_redu_BiasAdd_0             = createTensors(1,
                                                                     INPUT_TENSOR,
                                                                     true,
                                                                     "g_0_t3105_model_resUnit4_redu_BiasAdd_0",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_t3105_model_resUnit4_redu_BiasAdd_0_max_sizes,
                                                                     4,
                                                                     syn_type_single,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_t3105_model_resUnit4_redu_BiasAdd_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3_max_sizes[] = {128};
    unsigned g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3_min_sizes[] = {128};
    unsigned g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_max_sizes[] = {128};
    unsigned g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_min_sizes[] = {128};
    unsigned g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1744_model_resunit4_redu_bn_readvariableop_0_max_sizes[] = {128};
    unsigned g_0_t1744_model_resunit4_redu_bn_readvariableop_0_min_sizes[] = {128};
    unsigned g_0_t1744_model_resunit4_redu_bn_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1744_model_resunit4_redu_bn_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1744_model_resunit4_redu_bn_readvariableop_0_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1744_model_resunit4_redu_bn_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0_max_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0_min_sizes[] = {128, 56, 16, 32};
    unsigned g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2_max_sizes[] = {128};
    unsigned g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2_min_sizes[] = {128};
    unsigned g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1_max_sizes[] = {128};
    unsigned g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1_min_sizes[] = {128};
    unsigned g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1_max_sizes,
                      1,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3868_0_id;
    unsigned char g_0_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3868_0_params[] =
        {149, 191, 214, 51, 205, 204, 204, 61, 111, 18, 131, 58, 1, 0, 0, 0};
    addNodeToGraph(
        "batch_norm_bwd_f32",
        {g_0_t3105_model_resUnit4_redu_BiasAdd_0,
         g_0_t6892_gradient_tape_model_resUnit4_redu_relu_activation_ReluGrad_0,
         g_0_t3116_model_resUnit4_redu_bn_FusedBatchNormV3_3,
         g_0_t6899_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3,
         g_0_t1744_model_resunit4_redu_bn_readvariableop_0},
        {g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0,
         g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2,
         g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1},
        (void*)g_0_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3868_0_params,
        16,
        "g_0_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3868_0",
        0 /*graphIndex*/,
        &g_0_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_batch_norm_bwd_f32_n3868_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_t6860_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_2,
                        g_0_t6859_gradient_tape_model_resUnit5_stride1_bn_FusedBatchNormGradV3_1,
                        g_0_t6867_gradient_tape_model_resUnit5_stride1_Conv2D_Conv2DBackpropFilter_0,
                        g_0_t6893_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_0,
                        g_0_t6895_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_2,
                        g_0_t6894_gradient_tape_model_resUnit4_redu_bn_FusedBatchNormGradV3_1});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, flattenable_dedw_sliced_on_spatial_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_t7506_optimizers_gradients_AddN_1_0_max_sizes[] = {1, 512, 512, 16};
    unsigned g_0_t7506_optimizers_gradients_AddN_1_0_min_sizes[] = {1, 512, 512, 16};
    unsigned g_0_t7506_optimizers_gradients_AddN_1_0             = createTensors(1,
                                                                     INPUT_TENSOR,
                                                                     true,
                                                                     "g_0_t7506_optimizers_gradients_AddN_1_0",
                                                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                     nullptr,
                                                                     g_0_t7506_optimizers_gradients_AddN_1_0_max_sizes,
                                                                     4,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_t7506_optimizers_gradients_AddN_1_0_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t6915_unet_v1_ouputs_block_act2_relu_0_max_sizes[] = {32, 512, 512, 16};
    unsigned g_0_t6915_unet_v1_ouputs_block_act2_relu_0_min_sizes[] = {32, 512, 512, 16};
    unsigned g_0_t6915_unet_v1_ouputs_block_act2_relu_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t6915_unet_v1_ouputs_block_act2_relu_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t6915_unet_v1_ouputs_block_act2_relu_0_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t6915_unet_v1_ouputs_block_act2_relu_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes[] =
            {1, 32, 1, 1};
    unsigned
        g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes[] =
            {1, 32, 1, 1};
    unsigned g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0 =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_dedw_n4776_0_id;
    unsigned char
        g_0_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_dedw_n4776_0_params[] =
            {1, 0, 0, 0, 1, 0, 0, 0,  1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 61, 102, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_t7506_optimizers_gradients_AddN_1_0, g_0_t6915_unet_v1_ouputs_block_act2_relu_0},
        {g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0},
        (void*)
            g_0_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_dedw_n4776_0_params,
        104,
        "g_0_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_dedw_n4776_0",
        0 /*graphIndex*/,
        &g_0_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_dedw_n4776_0_id);

    setConfigsForTest();

    compareRunsResults(
        {g_0_t7517_optimizers_gradients_UNet_v1_ouputs_block_conv2d_2_Conv2D_grad_Conv2DBackpropFilter_0});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, ssd_shared_input_conv_dedx_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_t1825_huber_loss_Minimum_0_max_sizes[] = {4, 8732, 128};
    unsigned g_0_t1825_huber_loss_Minimum_0_min_sizes[] = {4, 8732, 128};
    unsigned g_0_t1825_huber_loss_Minimum_0             = createTensors(1,
                                                            INPUT_TENSOR,
                                                            true,
                                                            "g_0_t1825_huber_loss_Minimum_0",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr,
                                                            g_0_t1825_huber_loss_Minimum_0_max_sizes,
                                                            3,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_t1825_huber_loss_Minimum_0_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0_max_sizes[] = {4, 8732, 128};
    unsigned g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0_min_sizes[] = {4, 8732, 128};
    unsigned g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0_max_sizes,
                      3,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_mult_fwd_f32_n615_0_id;
    addNodeToGraph("mult_fwd_f32",
                   {g_0_t1825_huber_loss_Minimum_0, g_0_t1825_huber_loss_Minimum_0},
                   {g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0},
                   nullptr,
                   0,
                   "g_0_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_mult_fwd_f32_n615_0",
                   0 /*graphIndex*/,
                   &g_0_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_mult_fwd_f32_n615_0_id);

    unsigned g_0_t1824_huber_loss_Abs_0_max_sizes[] = {4, 8732, 128};
    unsigned g_0_t1824_huber_loss_Abs_0_min_sizes[] = {4, 8732, 128};
    unsigned g_0_t1824_huber_loss_Abs_0             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_t1824_huber_loss_Abs_0",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_t1824_huber_loss_Abs_0_max_sizes,
                                                        3,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_t1824_huber_loss_Abs_0_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_t1827_huber_loss_Sub_1_0_max_sizes[] = {4, 8732, 128};
    unsigned  g_0_t1827_huber_loss_Sub_1_0_min_sizes[] = {4, 8732, 128};
    unsigned  g_0_t1827_huber_loss_Sub_1_0             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          false,
                                                          "g_0_t1827_huber_loss_Sub_1_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t1827_huber_loss_Sub_1_0_max_sizes,
                                                          3,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t1827_huber_loss_Sub_1_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_huber_loss_Sub_1_sub_fwd_f32_n616_0_id;
    addNodeToGraph("sub_fwd_f32",
                   {g_0_t1824_huber_loss_Abs_0, g_0_t1825_huber_loss_Minimum_0},
                   {g_0_t1827_huber_loss_Sub_1_0},
                   nullptr,
                   0,
                   "g_0_huber_loss_Sub_1_sub_fwd_f32_n616_0",
                   0 /*graphIndex*/,
                   &g_0_huber_loss_Sub_1_sub_fwd_f32_n616_0_id);

    unsigned g_0_t1851_huber_loss_Mul_1_max_sizes[] = {1, 1, 1};
    unsigned g_0_t1851_huber_loss_Mul_1_min_sizes[] = {1, 1, 1};
    unsigned g_0_t1851_huber_loss_Mul_1             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_t1851_huber_loss_Mul_1",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_t1851_huber_loss_Mul_1_max_sizes,
                                                        3,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_t1851_huber_loss_Mul_1_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];

    unsigned  g_0_t1850_huber_loss_Mul_1_0_max_sizes[] = {4, 8732, 128};
    unsigned  g_0_t1850_huber_loss_Mul_1_0_min_sizes[] = {4, 8732, 128};
    unsigned  g_0_t1850_huber_loss_Mul_1_0             = createTensors(1,
                                                          OUTPUT_TENSOR,
                                                          false,
                                                          "g_0_t1850_huber_loss_Mul_1_0",
                                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                          nullptr,
                                                          g_0_t1850_huber_loss_Mul_1_0_max_sizes,
                                                          3,
                                                          syn_type_single,
                                                          nullptr,
                                                          0,
                                                          0,
                                                          nullptr,
                                                          false,
                                                          g_0_t1850_huber_loss_Mul_1_0_min_sizes,
                                                          synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_huber_loss_Mul_1_mult_fwd_f32_n631_0_id;
    addNodeToGraph("mult_fwd_f32",
                   {g_0_t1826_huber_loss_ArithmeticOptimizer_ReplaceMulWithSquare_Mul_0, g_0_t1851_huber_loss_Mul_1},
                   {g_0_t1850_huber_loss_Mul_1_0},
                   nullptr,
                   0,
                   "g_0_huber_loss_Mul_1_mult_fwd_f32_n631_0",
                   0 /*graphIndex*/,
                   &g_0_huber_loss_Mul_1_mult_fwd_f32_n631_0_id);

    unsigned  g_0_t1853_huber_loss_Add_0_max_sizes[] = {4, 8732, 128};
    unsigned  g_0_t1853_huber_loss_Add_0_min_sizes[] = {4, 8732, 128};
    unsigned  g_0_t1853_huber_loss_Add_0             = createTensors(1,
                                                        OUTPUT_TENSOR,
                                                        false,
                                                        "g_0_t1853_huber_loss_Add_0",
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        g_0_t1853_huber_loss_Add_0_max_sizes,
                                                        3,
                                                        syn_type_single,
                                                        nullptr,
                                                        0,
                                                        0,
                                                        nullptr,
                                                        false,
                                                        g_0_t1853_huber_loss_Add_0_min_sizes,
                                                        synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_huber_loss_Add_add_fwd_f32_n632_0_id;
    addNodeToGraph("add_fwd_f32",
                   {g_0_t1850_huber_loss_Mul_1_0, g_0_t1827_huber_loss_Sub_1_0},
                   {g_0_t1853_huber_loss_Add_0},
                   nullptr,
                   0,
                   "g_0_huber_loss_Add_add_fwd_f32_n632_0",
                   0 /*graphIndex*/,
                   &g_0_huber_loss_Add_add_fwd_f32_n632_0_id);

    unsigned g_0_t1855_Reshape_1_max_sizes[] = {4, 1, 8732, 128};
    unsigned g_0_t1855_Reshape_1_min_sizes[] = {4, 1, 8732, 128};
    unsigned g_0_t1855_Reshape_1             = createTensors(1,
                                                 INPUT_TENSOR,
                                                 false,
                                                 "g_0_t1855_Reshape_1",
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 g_0_t1855_Reshape_1_max_sizes,
                                                 4,
                                                 syn_type_uint32,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 false,
                                                 g_0_t1855_Reshape_1_min_sizes,
                                                 synTensorType::SHAPE_TENSOR)[0];

    unsigned  g_0_t1854_Reshape_1_0_max_sizes[] = {4, 1, 8732, 128};
    unsigned  g_0_t1854_Reshape_1_0_min_sizes[] = {4, 1, 8732, 128};
    unsigned  g_0_t1854_Reshape_1_0             = createTensors(1,
                                                   OUTPUT_TENSOR,
                                                   false,
                                                   "g_0_t1854_Reshape_1_0",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_t1854_Reshape_1_0_max_sizes,
                                                   4,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_t1854_Reshape_1_0_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_Reshape_1_reshape_n633_0_id;
    addNodeToGraph("reshape",
                   {g_0_t1853_huber_loss_Add_0, g_0_t1855_Reshape_1},
                   {g_0_t1854_Reshape_1_0},
                   nullptr,
                   0,
                   "g_0_Reshape_1_reshape_n633_0",
                   0 /*graphIndex*/,
                   &g_0_Reshape_1_reshape_n633_0_id);

    unsigned g_0_t1882_gradients_Reshape_2_grad_Reshape_0_max_sizes[] = {1, 1, 8732, 128};
    unsigned g_0_t1882_gradients_Reshape_2_grad_Reshape_0_min_sizes[] = {1, 1, 8732, 128};
    unsigned g_0_t1882_gradients_Reshape_2_grad_Reshape_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t1882_gradients_Reshape_2_grad_Reshape_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1882_gradients_Reshape_2_grad_Reshape_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1882_gradients_Reshape_2_grad_Reshape_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1856_reduce_sum_filter_0_max_sizes[] = {1, 4, 1, 1};
    unsigned g_0_t1856_reduce_sum_filter_0_min_sizes[] = {1, 4, 1, 1};
    unsigned g_0_t1856_reduce_sum_filter_0             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_t1856_reduce_sum_filter_0",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_t1856_reduce_sum_filter_0_max_sizes,
                                                           4,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           true,
                                                           g_0_t1856_reduce_sum_filter_0_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

    unsigned g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput_max_sizes[] = {4, 1, 8732, 128};
    unsigned g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput_min_sizes[] = {4, 1, 8732, 128};
    unsigned g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput =
        createTensors(1,
                      INPUT_TENSOR,
                      false,
                      "g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput_max_sizes,
                      4,
                      syn_type_uint32,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput_min_sizes,
                      synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0_max_sizes[] = {4, 1, 8732, 128};
    unsigned g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0_min_sizes[] = {4, 1, 8732, 128};
    unsigned g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0_id;
    unsigned char g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,   1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 156, 170, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_t1882_gradients_Reshape_2_grad_Reshape_0,
                    g_0_t1856_reduce_sum_filter_0,
                    g_0_t1885_gradients_smooth_l1_grad_Conv2DBackpropInput},
                   {g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0},
                   (void*)g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0_params,
                   104,
                   "g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0",
                   0 /*graphIndex*/,
                   &g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0_id);

    unsigned      g_0_t1857_smooth_l1_0_max_sizes[] = {1, 1, 8732, 128};
    unsigned      g_0_t1857_smooth_l1_0_min_sizes[] = {1, 1, 8732, 128};
    unsigned      g_0_t1857_smooth_l1_0             = createTensors(1,
                                                   OUTPUT_TENSOR,
                                                   true,
                                                   "g_0_t1857_smooth_l1_0",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   g_0_t1857_smooth_l1_0_max_sizes,
                                                   4,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   g_0_t1857_smooth_l1_0_min_sizes,
                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_smooth_l1_spatial_convolution_n634_0_id;
    unsigned char g_0_smooth_l1_spatial_convolution_n634_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,   1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 156, 170, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t1854_Reshape_1_0, g_0_t1856_reduce_sum_filter_0},
                   {g_0_t1857_smooth_l1_0},
                   (void*)g_0_smooth_l1_spatial_convolution_n634_0_params,
                   104,
                   "g_0_smooth_l1_spatial_convolution_n634_0",
                   0 /*graphIndex*/,
                   &g_0_smooth_l1_spatial_convolution_n634_0_id);

    synNodeId blocking_g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0[] = {
        g_0_Reshape_1_reshape_n633_0_id};
    setNodeDependency(blocking_g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0,
                      &g_0_gradients_smooth_l1_grad_Conv2DBackpropInput_dedx_n650_0_id,
                      1,
                      1);

    setConfigsForTest();

    compareRunsResults({g_0_t1884_gradients_smooth_l1_grad_Conv2DBackpropInput_0, g_0_t1857_smooth_l1_0});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, maxpool_with_shape_tensor_ASIC_CI, {synDeviceGaudi})
{
    const char* dedwInputLayouts[]  = {"WHCN", "WHCN"};
    const char* dedwOutputLayouts[] = {"SRCK"};

    const char* dedxInputLayouts[]  = {"WHCN", "SRCK", "WHCN"};
    const char* dedxOutputLayouts[] = {"WHCN"};

    const char* maxpoolInputLayouts[]  = {"WHCN", "WHCN", "WHCN"};
    const char* maxpoolOutputLayouts[] = {"WHCN"};

    unsigned g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_max_sizes[] =
        {19, 13, 256, 2};
    unsigned g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_min_sizes[] =
        {18, 12, 256, 2};
    unsigned g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward = createTensors(
        1,
        INPUT_TENSOR,
        true,
        "g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_min_sizes,
        synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_275_max_sizes[] = {19, 13, 256, 2};
    unsigned g_0_tensor_275_min_sizes[] = {18, 12, 256, 2};
    unsigned g_0_tensor_275             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_275",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_275_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_275_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes
            [] = {3, 3, 256, 256};
    unsigned
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes
            [] = {3, 3, 256, 256};
    unsigned g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_id;
    unsigned char g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0,   1,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 109, 46, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward, g_0_tensor_275},
        {g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable},
        (void*)g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_params,
        104,
        "g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0",
        0 /*graphIndex*/,
        &g_0_gradient_proposal_generator_rpn_head_conv_dedw_10010_0_id,
        dedwInputLayouts,
        dedwOutputLayouts);

    unsigned g_0_tensor_276_max_sizes[] = {3, 3, 256, 256};
    unsigned g_0_tensor_276_min_sizes[] = {3, 3, 256, 256};
    unsigned g_0_tensor_276             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_276",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_276_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_276_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_278_max_sizes[] = {19, 13, 256, 2};
    unsigned g_0_tensor_278_min_sizes[] = {18, 12, 256, 2};
    unsigned g_0_tensor_278             = createTensors(1,
                                            INPUT_TENSOR,
                                            false,
                                            "g_0_tensor_278",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_278_max_sizes,
                                            4,
                                            syn_type_uint32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_278_min_sizes,
                                            synTensorType::SHAPE_TENSOR)[0];

    unsigned
        g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes
            [] = {19, 13, 256, 2};
    unsigned
        g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes
            [] = {18, 12, 256, 2};
    unsigned g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            false,
            "g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_proposal_generator_rpn_head_conv_dedx_10011_0_id;
    unsigned char g_0_gradient_proposal_generator_rpn_head_conv_dedx_10011_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedx",
        {g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward,
         g_0_tensor_276,
         g_0_tensor_278},
        {g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable},
        (void*)g_0_gradient_proposal_generator_rpn_head_conv_dedx_10011_0_params,
        104,
        "g_0_gradient_proposal_generator_rpn_head_conv_dedx_10011_0",
        0 /*graphIndex*/,
        &g_0_gradient_proposal_generator_rpn_head_conv_dedx_10011_0_id,
        dedxInputLayouts,
        dedxOutputLayouts);

    unsigned g_0_tensor_410_max_sizes[] = {19, 13, 256, 2};
    unsigned g_0_tensor_410_min_sizes[] = {18, 12, 256, 2};
    unsigned g_0_tensor_410             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_410",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_410_max_sizes,
                                            4,
                                            syn_type_int16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_410_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_411_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_411_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_411             = createTensors(1,
                                            INPUT_TENSOR,
                                            false,
                                            "g_0_tensor_411",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_411_max_sizes,
                                            4,
                                            syn_type_uint32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_411_min_sizes,
                                            synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_max_sizes[] =
        {38, 25, 256, 2};
    unsigned g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_min_sizes[] =
        {36, 24, 256, 2};
    unsigned g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward = createTensors(
        1,
        OUTPUT_TENSOR,
        false,
        "g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward",
        MEM_INIT_RANDOM_WITH_NEGATIVE,
        nullptr,
        g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_max_sizes,
        4,
        syn_type_bf16,
        nullptr,
        0,
        0,
        nullptr,
        false,
        g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward_min_sizes,
        synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_id;
    unsigned char g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_params[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "maxpool_2d_bwd_bf16",
        {g_0_tensor_279_id_43410_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
         g_0_tensor_410,
         g_0_tensor_411},
        {g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward},
        (void*)g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_params,
        44,
        "g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0",
        0 /*graphIndex*/,
        &g_0_gradient_backbone_top_block_maxpool_2d_bwd_bf16_10098_0_id,
        maxpoolInputLayouts,
        maxpoolOutputLayouts);

    unsigned g_0_tensor_413_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_413_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_413             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_413",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_413_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_413_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes
            [] = {38, 25, 256, 2};
    unsigned
        g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes
            [] = {36, 24, 256, 2};
    unsigned g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable =
        createTensors(
            1,
            INPUT_TENSOR,
            true,
            "g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_proposal_generator_rpn_head_conv_add_fwd_bf16_10099_0_id;
    addNodeToGraph(
        "add_fwd_bf16",
        {g_0_tensor_413,
         g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable},
        {g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add},
        nullptr,
        0,
        "g_0_gradient_proposal_generator_rpn_head_conv_add_fwd_bf16_10099_0",
        0 /*graphIndex*/,
        &g_0_gradient_proposal_generator_rpn_head_conv_add_fwd_bf16_10099_0_id);

    unsigned g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add_max_sizes,
                      4,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_backbone_top_block_add_fwd_bf16_10100_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {g_0_tensor_414_id_43491_gradient_proposal_generator_rpn_head_conv_aten__add,
                    g_0_tensor_412_id_43786_gradient_backbone_top_block_aten__max_pool2d_with_indices_backward},
                   {g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add},
                   nullptr,
                   0,
                   "g_0_gradient_backbone_top_block_add_fwd_bf16_10100_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_backbone_top_block_add_fwd_bf16_10100_0_id);

    unsigned g_0_tensor_416_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_416_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_416             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_416",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_416_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_416_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes[] =
        {3, 3, 256, 256};
    unsigned g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes[] =
        {3, 3, 256, 256};
    unsigned g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_backbone_fpn_output5_dedw_10101_0_id;
    unsigned char g_0_gradient_backbone_fpn_output5_dedw_10101_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0,  1,   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 39, 166, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw",
                   {g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add, g_0_tensor_416},
                   {g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable},
                   (void*)g_0_gradient_backbone_fpn_output5_dedw_10101_0_params,
                   104,
                   "g_0_gradient_backbone_fpn_output5_dedw_10101_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_backbone_fpn_output5_dedw_10101_0_id,
                   dedwInputLayouts,
                   dedwOutputLayouts);

    unsigned g_0_tensor_417_max_sizes[] = {3, 3, 256, 256};
    unsigned g_0_tensor_417_min_sizes[] = {3, 3, 256, 256};
    unsigned g_0_tensor_417             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_tensor_417",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_417_max_sizes,
                                            4,
                                            syn_type_bf16,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_417_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_419_max_sizes[] = {38, 25, 256, 2};
    unsigned g_0_tensor_419_min_sizes[] = {36, 24, 256, 2};
    unsigned g_0_tensor_419             = createTensors(1,
                                            INPUT_TENSOR,
                                            false,
                                            "g_0_tensor_419",
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            g_0_tensor_419_max_sizes,
                                            4,
                                            syn_type_uint32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_tensor_419_min_sizes,
                                            synTensorType::SHAPE_TENSOR)[0];

    unsigned g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes[] =
        {38, 25, 256, 2};
    unsigned g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes[] =
        {36, 24, 256, 2};
    unsigned g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_bf16,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_backbone_fpn_output5_dedx_10102_0_id;
    unsigned char g_0_gradient_backbone_fpn_output5_dedx_10102_0_params[] = {
        3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx",
                   {g_0_tensor_415_id_43789_gradient_backbone_top_block_aten__add, g_0_tensor_417, g_0_tensor_419},
                   {g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable},
                   (void*)g_0_gradient_backbone_fpn_output5_dedx_10102_0_params,
                   104,
                   "g_0_gradient_backbone_fpn_output5_dedx_10102_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_backbone_fpn_output5_dedx_10102_0_id,
                   dedxInputLayouts,
                   dedxOutputLayouts);

    setConfigsForTest();

    setActualSizes(
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward,
        g_0_tensor_274_id_43402_gradient_proposal_generator_rpn_head_conv_aten__threshold_backward_max_sizes);
    setActualSizes(g_0_tensor_275, g_0_tensor_275_max_sizes);
    setActualSizes(
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
        g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes);
    setActualSizes(g_0_tensor_276, g_0_tensor_276_max_sizes);
    setActualSizes(g_0_tensor_410, g_0_tensor_410_max_sizes);
    setActualSizes(g_0_tensor_413, g_0_tensor_413_max_sizes);
    setActualSizes(
        g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
        g_0_tensor_266_id_43485_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable_max_sizes);
    setActualSizes(g_0_tensor_416, g_0_tensor_416_max_sizes);
    setActualSizes(g_0_tensor_417, g_0_tensor_417_max_sizes);
    setActualSizes(
        g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable,
        g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable_max_sizes);
    setActualSizes(g_0_tensor_411, g_0_tensor_411_max_sizes);
    setActualSizes(g_0_tensor_278, g_0_tensor_278_max_sizes);
    setActualSizes(g_0_tensor_419, g_0_tensor_419_max_sizes);

    compareRunsResults(
        {g_0_tensor_277_id_43411_gradient_proposal_generator_rpn_head_conv_aten__convolution_backward_overrideable,
         g_0_tensor_418_id_43938_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable,
         g_0_tensor_420_id_43937_gradient_backbone_fpn_output5_aten__convolution_backward_overrideable});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, dedx_dedw_with_shared_input_from_unet3d_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_0_max_sizes[] = {4, 32, 32, 32, 2};
    unsigned g_0_0_min_sizes[] = {4, 32, 32, 32, 2};
    unsigned g_0_0             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "g_0_0",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   g_0_0_max_sizes,
                                   5,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_0_0_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned g_0_1_max_sizes[] = {128, 32, 32, 32, 2};
    unsigned g_0_1_min_sizes[] = {128, 32, 32, 32, 2};
    unsigned g_0_1             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "g_0_1",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   g_0_1_max_sizes,
                                   5,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_0_1_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_3_max_sizes[] = {4, 128, 1, 1, 1};
    unsigned      g_0_3_min_sizes[] = {4, 128, 1, 1, 1};
    unsigned      g_0_3             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "g_0_3",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   g_0_3_max_sizes,
                                   5,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_0_3_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_model_dedw3d_239_0_id;
    unsigned char g_0_gradient_model_dedw3d_239_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedw3d",
                   {g_0_0, g_0_1},
                   {g_0_3},
                   (void*)g_0_gradient_model_dedw3d_239_0_params,
                   128,
                   "g_0_gradient_model_dedw3d_239_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_model_dedw3d_239_0_id);

    unsigned g_0_2_max_sizes[] = {4, 128, 1, 1, 1};
    unsigned g_0_2_min_sizes[] = {4, 128, 1, 1, 1};
    unsigned g_0_2             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "g_0_2",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   g_0_2_max_sizes,
                                   5,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_0_2_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    unsigned      g_0_4_max_sizes[] = {128, 32, 32, 32, 2};
    unsigned      g_0_4_min_sizes[] = {128, 32, 32, 32, 2};
    unsigned      g_0_4             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "g_0_4",
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   g_0_4_max_sizes,
                                   5,
                                   syn_type_single,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   g_0_4_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_model_dedx3d_240_0_id;
    unsigned char g_0_gradient_model_dedx3d_240_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("dedx3d",
                   {g_0_0, g_0_2},
                   {g_0_4},
                   (void*)g_0_gradient_model_dedx3d_240_0_params,
                   128,
                   "g_0_gradient_model_dedx3d_240_0",
                   0 /*graphIndex*/,
                   &g_0_gradient_model_dedx3d_240_0_id);

    setConfigsForTest();

    compareRunsResults({g_0_3, g_0_4});
}

TEST_F_GC(SynGaudiBigTensorsSramSlicingTests, dedx_dedw_with_shared_input_from_maskrcnn_ASIC_CI, {synDeviceGaudi})
{
    unsigned g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view_max_sizes[] = {12,
                                                                                                                 164,
                                                                                                                 100,
                                                                                                                 2};
    unsigned g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view_min_sizes[] = {12,
                                                                                                                 164,
                                                                                                                 100,
                                                                                                                 2};
    unsigned g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    unsigned g_0_tensor_17_max_sizes[] = {256, 164, 100, 2};
    unsigned g_0_tensor_17_min_sizes[] = {256, 164, 100, 2};
    unsigned g_0_tensor_17             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_17",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_17_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_17_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_max_sizes
            [] = {12, 256, 1, 1};
    unsigned
        g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_min_sizes
            [] = {12, 256, 1, 1};
    unsigned g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_"
            "overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedw_3006_0_id;
    unsigned char g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedw_3006_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,   1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 117, 183, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedw",
        {g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view, g_0_tensor_17},
        {g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable},
        (void*)g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedw_3006_0_params,
        104,
        "g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedw_3006_0",
        0 /*graphIndex*/,
        &g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedw_3006_0_id);

    unsigned g_0_tensor_6_max_sizes[] = {12, 256, 1, 1};
    unsigned g_0_tensor_6_min_sizes[] = {12, 256, 1, 1};
    unsigned g_0_tensor_6             = createTensors(1,
                                          INPUT_TENSOR,
                                          true,
                                          "g_0_tensor_6",
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          g_0_tensor_6_max_sizes,
                                          4,
                                          syn_type_single,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          false,
                                          g_0_tensor_6_min_sizes,
                                          synTensorType::DATA_TENSOR)[0];

    unsigned
        g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_max_sizes
            [] = {256, 164, 100, 2};
    unsigned
        g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_min_sizes
            [] = {256, 164, 100, 2};
    unsigned g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable =
        createTensors(
            1,
            OUTPUT_TENSOR,
            true,
            "g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_"
            "overrideable",
            MEM_INIT_RANDOM_WITH_NEGATIVE,
            nullptr,
            g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_max_sizes,
            4,
            syn_type_single,
            nullptr,
            0,
            0,
            nullptr,
            false,
            g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable_min_sizes,
            synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedx_3007_0_id;
    unsigned char g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedx_3007_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph(
        "dedx",
        {g_0_tensor_16_id_11536_gradient_proposal_generator_rpn_head_anchor_deltas_aten__view, g_0_tensor_6},
        {g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable},
        (void*)g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedx_3007_0_params,
        104,
        "g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedx_3007_0",
        0 /*graphIndex*/,
        &g_0_gradient_proposal_generator_rpn_head_anchor_deltas_dedx_3007_0_id);

    setConfigsForTest();

    compareRunsResults(
        {g_0_tensor_18_id_11746_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable,
         g_0_tensor_19_id_11745_gradient_proposal_generator_rpn_head_anchor_deltas_aten__convolution_backward_overrideable});
}
