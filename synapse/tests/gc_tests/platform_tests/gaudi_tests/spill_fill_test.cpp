#include "global_conf_test_setter.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "node_factory.h"
#include "synapse_common_types.hpp"
#include "tensor.h"
#include <syn_singleton.hpp>

class SynGaudiSpillFillTest : public SynGaudiTwoRunCompareTest
{
public:
    void SetUp() override
    {
        // The flag COMPLEXGUID_USER_ALLOWLIST can't be set using the GlobalConfTestSetter because it is not synapse
        // configuration, so setting it directly using setenv.
        // COMPLEXGUID_USER_ALLOWLIST=<Space seperated guid string without the data type to enable ComplexGuid which are
        // disabled by default>
        m_complexGuidPrevCfg = std::getenv("COMPLEXGUID_USER_ALLOWLIST");
        setenv("COMPLEXGUID_USER_ALLOWLIST", "batch_norm_fwd batch_norm_bwd layer_norm_fwd", true);
        SynGaudiTwoRunCompareTest::SetUp();
    }

    void TearDown() override
    {
        SynGaudiTwoRunCompareTest::TearDown();
        // Reset env var to its original value
        if (m_complexGuidPrevCfg) setenv("COMPLEXGUID_USER_ALLOWLIST", m_complexGuidPrevCfg, true);
    }

    void runTest(const std::vector<unsigned>& outputToCompareIdx)
    {
        GlobalConfTestSetter conf("NORM_MOMENTS_CLUSTERING", "true");
        GlobalConfTestSetter conf2("ENABLE_SLICE_NORM_BUNDLING", "true");
        GlobalConfTestSetter conf3("ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");

        addConfigurationToRun(FIRST_RUN, "ENABLE_SPILL_FILL_FUSION", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_SPILL_FILL_FUSION", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "true");

        compareRunsResults(outputToCompareIdx);
    }
    const char* m_complexGuidPrevCfg;
};

TEST_F_GC(SynGaudiSpillFillTest, conv_and_norm_moments)
{
    // Graph #0
    /*************
     * g_0_model_conv2d_Conv2D_spatial_convolution_n16_0 node
     * inputs:
     *     g_0_t66_iteratorgetnext_0[64, 256, 256, 32] (dtype=float32)
     *     g_0_t67_model_conv2d_conv2d_readvariableop_0[64, 64, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_t76_model_conv2d_Conv2D_0[64, 256, 256, 32] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t66_iteratorgetnext_0 tensor
    unsigned g_0_t66_iteratorgetnext_0_max_sizes[] = {64, 256, 256, 32};
    unsigned g_0_t66_iteratorgetnext_0_min_sizes[] = {64, 256, 256, 32};
    unsigned g_0_t66_iteratorgetnext_0             = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_t66_iteratorgetnext_0",
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_t66_iteratorgetnext_0_max_sizes,
                                                       4,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_t66_iteratorgetnext_0_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_t67_model_conv2d_conv2d_readvariableop_0 tensor
    unsigned g_0_t67_model_conv2d_conv2d_readvariableop_0_max_sizes[] = {64, 64, 1, 1};
    unsigned g_0_t67_model_conv2d_conv2d_readvariableop_0_min_sizes[] = {64, 64, 1, 1};
    unsigned g_0_t67_model_conv2d_conv2d_readvariableop_0 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "g_0_t67_model_conv2d_conv2d_readvariableop_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t67_model_conv2d_conv2d_readvariableop_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t67_model_conv2d_conv2d_readvariableop_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t76_model_conv2d_Conv2D_0 tensor
    unsigned      g_0_t76_model_conv2d_Conv2D_0_max_sizes[] = {64, 256, 256, 32};
    unsigned      g_0_t76_model_conv2d_Conv2D_0_min_sizes[] = {64, 256, 256, 32};
    unsigned      g_0_t76_model_conv2d_Conv2D_0             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           true,
                                                           "g_0_t76_model_conv2d_Conv2D_0",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_t76_model_conv2d_Conv2D_0_max_sizes,
                                                           4,
                                                           syn_type_single,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_t76_model_conv2d_Conv2D_0_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_id;
    unsigned char g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_params[] = {
        1, 0, 0, 0, 1, 0, 0, 0,   1,   0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 161, 117, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {g_0_t66_iteratorgetnext_0, g_0_t67_model_conv2d_conv2d_readvariableop_0},
                   {g_0_t76_model_conv2d_Conv2D_0},
                   (void*)g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_params,
                   104,
                   "g_0_model_conv2d_Conv2D_spatial_convolution_n16_0",
                   0 /*graphIndex*/,
                   &g_0_model_conv2d_Conv2D_spatial_convolution_n16_0_id);

    /*************
     * g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0 node
     * inputs:
     *     g_0_t76_model_conv2d_Conv2D_0[64, 256, 256, 32] (dtype=float32)
     * outputs:
     *     g_0_t77_model_batch_normalization_moments_mean_0[64, 1, 1, 1] (dtype=float32)
     *     g_0_t78_model_batch_normalization_moments_variance_0[64, 1, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t77_model_batch_normalization_moments_mean_0 tensor
    unsigned g_0_t77_model_batch_normalization_moments_mean_0_max_sizes[] = {64, 1, 1, 1};
    unsigned g_0_t77_model_batch_normalization_moments_mean_0_min_sizes[] = {64, 1, 1, 1};
    unsigned g_0_t77_model_batch_normalization_moments_mean_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_t77_model_batch_normalization_moments_mean_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t77_model_batch_normalization_moments_mean_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t77_model_batch_normalization_moments_mean_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create g_0_t78_model_batch_normalization_moments_variance_0 tensor
    unsigned g_0_t78_model_batch_normalization_moments_variance_0_max_sizes[] = {64, 1, 1, 1};
    unsigned g_0_t78_model_batch_normalization_moments_variance_0_min_sizes[] = {64, 1, 1, 1};
    unsigned g_0_t78_model_batch_normalization_moments_variance_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "g_0_t78_model_batch_normalization_moments_variance_0",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_t78_model_batch_normalization_moments_variance_0_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_t78_model_batch_normalization_moments_variance_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_id;
    unsigned char
        g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_params[] = {14,
                                                                                                                  0,
                                                                                                                  0,
                                                                                                                  0};
    addNodeToGraph(
        "norm_moments_fwd_f32",
        {g_0_t76_model_conv2d_Conv2D_0},
        {g_0_t77_model_batch_normalization_moments_mean_0, g_0_t78_model_batch_normalization_moments_variance_0},
        (void*)g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_params,
        4,
        "g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0",
        0 /*graphIndex*/,
        &g_0_model_batch_normalization_moments_variance_habana_norm_moments_norm_moments_fwd_f32_n17_0_id);

    runTest({g_0_t77_model_batch_normalization_moments_mean_0, g_0_t76_model_conv2d_Conv2D_0});
}

TEST_F_GC(SynGaudiSpillFillTest, fuse_spill_relu)
{
    /*************
     * relu_node node
     * inputs:
     *     relu_in[64, 56, 56, 256] (dtype=bf16)
     * outputs:
     *     relu_out[64, 56, 56, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *     relu_ctrl[] (dtype=invalid)
     *************/
    // create relu_in tensor
    unsigned relu_in_max_sizes[] = {64, 56, 56, 256};
    unsigned relu_in_min_sizes[] = {64, 56, 56, 256};
    unsigned relu_in             = createTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "relu_in",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     relu_in_max_sizes,
                                     4,
                                     syn_type_bf16,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     relu_in_min_sizes,
                                     synTensorType::DATA_TENSOR)[0];
    // create relu_out tensor
    unsigned  relu_out_max_sizes[] = {64, 56, 56, 256};
    unsigned  relu_out_min_sizes[] = {64, 56, 56, 256};
    unsigned  relu_out             = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "relu_out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      relu_out_max_sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      relu_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId relu_id;
    addNodeToGraph("relu_fwd_bf16", {relu_in}, {relu_out}, nullptr, 0, "relu_node", 0 /*graphIndex*/, &relu_id);

    /*************
     * conv_node node
     * inputs:
     *     relu_out[64, 56, 56, 256] (dtype=bf16)
     *     conv_in1[64, 64, 3, 3] (dtype=bf16)
     * outputs:
     *     conv_out[64, 56, 56, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create conv_in1 tensor
    unsigned conv_in1_max_sizes[] = {64, 64, 3, 3};
    unsigned conv_in1_min_sizes[] = {64, 64, 3, 3};
    unsigned conv_in1             = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "conv_in1",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      conv_in1_max_sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv_in1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create conv_out tensor
    unsigned      conv_out_max_sizes[] = {64, 56, 56, 256};
    unsigned      conv_out_min_sizes[] = {64, 56, 56, 256};
    unsigned      conv_out             = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "conv_out",
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      conv_out_max_sizes,
                                      4,
                                      syn_type_bf16,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      conv_out_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId     conv_id;
    unsigned char conv_params[] = {3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,   0,  1,   0,   0, 0,
                                   1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 118, 41, 1,   0,   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,  252, 127, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {relu_out, conv_in1},
                   {conv_out},
                   (void*)conv_params,
                   72,
                   "conv_node",
                   0 /*graphIndex*/,
                   &conv_id);
    runTest({conv_out, relu_out});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, conv_batch_norm_fusion_fp32_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_f32_n298_0 node
     * inputs:
     *     g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0[64, 56, 56, 256] (dtype=f32)
     *     g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0[64] (dtype=float32)
     *     g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0[64] (dtype=float32)
     *     g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0[64] (dtype=float32)
     *     g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0[64] (dtype=float32)
     * outputs:
     *     g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0[64, 56, 56, 256] (dtype=f32)
     *     g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3[64] (dtype=float32)
     *     g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3[64] (dtype=float32)
     *     g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1[64] (dtype=float32)
     *     g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0 tensor
    unsigned g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0 = createTensors(1,
                                                                                       INPUT_TENSOR,
                                                                                       true,
                                                                                       "g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0",
                                                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                       nullptr,
                                                                                       g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_max_sizes,
                                                                                       4,
                                                                                       syn_type_single,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0 tensor
    unsigned g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0 = createTensors(1,
                                                                                                INPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0 tensor
    unsigned g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0 = createTensors(1,
                                                                                              INPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0 = createTensors(1,
                                                                                                               INPUT_TENSOR,
                                                                                                               true,
                                                                                                               "g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0",
                                                                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                               nullptr,
                                                                                                               g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_max_sizes,
                                                                                                               1,
                                                                                                               syn_type_single,
                                                                                                               nullptr,
                                                                                                               0,
                                                                                                               0,
                                                                                                               nullptr,
                                                                                                               false,
                                                                                                               g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_min_sizes,
                                                                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0 = createTensors(1,
                                                                                                                 INPUT_TENSOR,
                                                                                                                 true,
                                                                                                                 "g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0",
                                                                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                                 nullptr,
                                                                                                                 g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                                                                                                                 1,
                                                                                                                 syn_type_single,
                                                                                                                 nullptr,
                                                                                                                 0,
                                                                                                                 0,
                                                                                                                 nullptr,
                                                                                                                 false,
                                                                                                                 g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                                                                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0 tensor
    unsigned g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                false,
                                                                                                "g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_max_sizes,
                                                                                                4,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3 tensor
    unsigned g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_max_sizes[] = {64};
    unsigned g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_min_sizes[] = {64};
    unsigned g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 tensor
    unsigned g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1 tensor
    unsigned g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_max_sizes[] = {64};
    unsigned g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_min_sizes[] = {64};
    unsigned g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 tensor
    unsigned g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_f32_n298_0_id;
    unsigned char g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_f32_n298_0_params[] = {149,191,214,51,205,204,204,61,159,240,39,55,1,0,0,0};
    addNodeToGraph("batch_norm_fwd_f32", {g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0, g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0, g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0, g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0, g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0}, {g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0, g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3, g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3, g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1, g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3}, (void*)g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_f32_n298_0_params, 16, "g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_f32_n298_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_f32_n298_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_f32_n306_0 node
     * inputs:
     *     g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0[64, 56, 56, 256] (dtype=f32)
     * outputs:
     *     g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0[64, 56, 56, 256] (dtype=f32)
     * ctrl inputs:
     * ctrl outputs:
     *     g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_f32_n306_control_edge_2206[] (dtype=invalid)
     *************/

    // create g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0 tensor
    unsigned g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0 = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   false,
                                                                                   "g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0",
                                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                   nullptr,
                                                                                   g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_max_sizes,
                                                                                   4,
                                                                                   syn_type_single,
                                                                                   nullptr,
                                                                                   0,
                                                                                   0,
                                                                                   nullptr,
                                                                                   false,
                                                                                   g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_min_sizes,
                                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_f32_n306_0_id;
    addNodeToGraph("relu_fwd_f32", {g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0}, {g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0}, nullptr, 0, "g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_f32_n306_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_f32_n306_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0 node
     * inputs:
     *     g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0[64, 56, 56, 256] (dtype=f32)
     *     g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0[64, 64, 3, 3] (dtype=f32)
     * outputs:
     *     g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0[64, 56, 56, 256] (dtype=f32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0 tensor
    unsigned g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_max_sizes[] = {64,64,3,3};
    unsigned g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_min_sizes[] = {64,64,3,3};
    unsigned g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0 = createTensors(1,
                                                                                            INPUT_TENSOR,
                                                                                            true,
                                                                                            "g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0",
                                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                            nullptr,
                                                                                            g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_max_sizes,
                                                                                            4,
                                                                                            syn_type_single,
                                                                                            nullptr,
                                                                                            0,
                                                                                            0,
                                                                                            nullptr,
                                                                                            false,
                                                                                            g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_min_sizes,
                                                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0 tensor
    unsigned g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0 = createTensors(1,
                                                                                       OUTPUT_TENSOR,
                                                                                       true,
                                                                                       "g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0",
                                                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                       nullptr,
                                                                                       g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_max_sizes,
                                                                                       4,
                                                                                       syn_type_single,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_id;
    unsigned char g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,118,41,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0, g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0}, {g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0}, (void*)g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_params, 72, "g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_f32_n308_0 node
     * inputs:
     *     g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0[64, 56, 56, 256] (dtype=f32)
     *     g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0[64] (dtype=float32)
     *     g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0[64] (dtype=float32)
     *     g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0[64] (dtype=float32)
     *     g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0[64] (dtype=float32)
     * outputs:
     *     g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0[64, 56, 56, 256] (dtype=f32)
     *     g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3[64] (dtype=float32)
     *     g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3[64] (dtype=float32)
     *     g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1[64] (dtype=float32)
     *     g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0 tensor
    unsigned g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0 = createTensors(1,
                                                                                                INPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0 tensor
    unsigned g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0 = createTensors(1,
                                                                                              INPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0 = createTensors(1,
                                                                                                               INPUT_TENSOR,
                                                                                                               true,
                                                                                                               "g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0",
                                                                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                               nullptr,
                                                                                                               g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_max_sizes,
                                                                                                               1,
                                                                                                               syn_type_single,
                                                                                                               nullptr,
                                                                                                               0,
                                                                                                               0,
                                                                                                               nullptr,
                                                                                                               false,
                                                                                                               g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_min_sizes,
                                                                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0 = createTensors(1,
                                                                                                                 INPUT_TENSOR,
                                                                                                                 true,
                                                                                                                 "g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0",
                                                                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                                 nullptr,
                                                                                                                 g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                                                                                                                 1,
                                                                                                                 syn_type_single,
                                                                                                                 nullptr,
                                                                                                                 0,
                                                                                                                 0,
                                                                                                                 nullptr,
                                                                                                                 false,
                                                                                                                 g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                                                                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0 tensor
    unsigned g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_max_sizes,
                                                                                                4,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3 tensor
    unsigned g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_max_sizes[] = {64};
    unsigned g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_min_sizes[] = {64};
    unsigned g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 tensor
    unsigned g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1 tensor
    unsigned g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_max_sizes[] = {64};
    unsigned g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_min_sizes[] = {64};
    unsigned g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 tensor
    unsigned g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_f32_n308_0_id;
    unsigned char g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_f32_n308_0_params[] = {149,191,214,51,205,204,204,61,159,240,39,55,1,0,0,0};
    addNodeToGraph("batch_norm_fwd_f32", {g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0, g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0, g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0, g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0, g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0}, {g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0, g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3, g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3, g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1, g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3}, (void*)g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_f32_n308_0_params, 16, "g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_f32_n308_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_f32_n308_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_SPILL_FILL_FUSION", "true");
    addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "false");

    addConfigurationToRun(SECOND_RUN, "ENABLE_SPILL_FILL_FUSION", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "true");

    compareRunsResults({g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0, g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0});
}

TEST_F_GC(SynGaudiSpillFillTest, conv_batch_norm_fusion_bf_16_ASIC_CI)
{
    // Graph #0

    /*************
     * g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_bf16_n298_0 node
     * inputs:
     *     g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0[64] (dtype=float32)
     *     g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0[64] (dtype=float32)
     *     g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0[64] (dtype=float32)
     *     g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0[64] (dtype=float32)
     * outputs:
     *     g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3[64] (dtype=float32)
     *     g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3[64] (dtype=float32)
     *     g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1[64] (dtype=float32)
     *     g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0 tensor
    unsigned g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0 = createTensors(1,
                                                                                       INPUT_TENSOR,
                                                                                       true,
                                                                                       "g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0",
                                                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                       nullptr,
                                                                                       g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_max_sizes,
                                                                                       4,
                                                                                       syn_type_bf16,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0 tensor
    unsigned g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0 = createTensors(1,
                                                                                                INPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0 tensor
    unsigned g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0 = createTensors(1,
                                                                                              INPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0 = createTensors(1,
                                                                                                               INPUT_TENSOR,
                                                                                                               true,
                                                                                                               "g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0",
                                                                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                               nullptr,
                                                                                                               g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_max_sizes,
                                                                                                               1,
                                                                                                               syn_type_single,
                                                                                                               nullptr,
                                                                                                               0,
                                                                                                               0,
                                                                                                               nullptr,
                                                                                                               false,
                                                                                                               g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0_min_sizes,
                                                                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0 = createTensors(1,
                                                                                                                 INPUT_TENSOR,
                                                                                                                 true,
                                                                                                                 "g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0",
                                                                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                                 nullptr,
                                                                                                                 g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                                                                                                                 1,
                                                                                                                 syn_type_single,
                                                                                                                 nullptr,
                                                                                                                 0,
                                                                                                                 0,
                                                                                                                 nullptr,
                                                                                                                 false,
                                                                                                                 g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                                                                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0 tensor
    unsigned g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                false,
                                                                                                "g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_max_sizes,
                                                                                                4,
                                                                                                syn_type_bf16,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3 tensor
    unsigned g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_max_sizes[] = {64};
    unsigned g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_min_sizes[] = {64};
    unsigned g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 tensor
    unsigned g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1 tensor
    unsigned g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_max_sizes[] = {64};
    unsigned g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_min_sizes[] = {64};
    unsigned g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 tensor
    unsigned g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_bf16_n298_0_id;
    unsigned char g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_bf16_n298_0_params[] = {149,191,214,51,205,204,204,61,159,240,39,55,1,0,0,0};
    addNodeToGraph("batch_norm_fwd_bf16", {g_0_t921_while_body__1_while_resnet50_res2a_branch2a_Conv2D_0, g_0_t578_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_1_0, g_0_t577_while_body__1_while_resnet50_bn2a_branch2a_readvariableop_0, g_0_t579_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_0, g_0_t580_while_body__1_while_resnet50_bn2a_branch2a_fusedbatchnormv3_readvariableop_1_0}, {g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0, g_0_t949_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_3, g_0_t951_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3, g_0_t942_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_1, g_0_t948_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3}, (void*)g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_bf16_n298_0_params, 16, "g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_bf16_n298_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_batch_norm_fwd_bf16_n298_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_bf16_n306_0 node
     * inputs:
     *     g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0[64, 56, 56, 256] (dtype=bf16)
     * outputs:
     *     g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0[64, 56, 56, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *     g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_bf16_n306_control_edge_2206[] (dtype=invalid)
     *************/

    // create g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0 tensor
    unsigned g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0 = createTensors(1,
                                                                                   OUTPUT_TENSOR,
                                                                                   false,
                                                                                   "g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0",
                                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                   nullptr,
                                                                                   g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_max_sizes,
                                                                                   4,
                                                                                   syn_type_bf16,
                                                                                   nullptr,
                                                                                   0,
                                                                                   0,
                                                                                   nullptr,
                                                                                   false,
                                                                                   g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0_min_sizes,
                                                                                   synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_bf16_n306_0_id;
    addNodeToGraph("relu_fwd_bf16", {g_0_t941_while_body__1_while_resnet50_bn2a_branch2a_FusedBatchNormV3_0}, {g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0}, nullptr, 0, "g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_bf16_n306_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_activation_1_Relu_relu_fwd_bf16_n306_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0 node
     * inputs:
     *     g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0[64, 64, 3, 3] (dtype=bf16)
     * outputs:
     *     g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0[64, 56, 56, 256] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0 tensor
    unsigned g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_max_sizes[] = {64,64,3,3};
    unsigned g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_min_sizes[] = {64,64,3,3};
    unsigned g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0 = createTensors(1,
                                                                                            INPUT_TENSOR,
                                                                                            true,
                                                                                            "g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0",
                                                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                            nullptr,
                                                                                            g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_max_sizes,
                                                                                            4,
                                                                                            syn_type_bf16,
                                                                                            nullptr,
                                                                                            0,
                                                                                            0,
                                                                                            nullptr,
                                                                                            false,
                                                                                            g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0_min_sizes,
                                                                                            synTensorType::DATA_TENSOR)[0];

    // create g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0 tensor
    unsigned g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0 = createTensors(1,
                                                                                       OUTPUT_TENSOR,
                                                                                       true,
                                                                                       "g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0",
                                                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                       nullptr,
                                                                                       g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_max_sizes,
                                                                                       4,
                                                                                       syn_type_bf16,
                                                                                       nullptr,
                                                                                       0,
                                                                                       0,
                                                                                       nullptr,
                                                                                       false,
                                                                                       g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0_min_sizes,
                                                                                       synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_id;
    unsigned char g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,118,41,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {g_0_t960_while_body__1_while_resnet50_activation_1_Relu_0, g_0_t827_while_body__1_while_resnet50_res2a_branch2b_Conv2D_Cast_0}, {g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0}, (void*)g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_params, 72, "g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_res2a_branch2b_Conv2D_spatial_convolution_n307_0_id);

    /*************
     * g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0 node
     * inputs:
     *     g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0[64] (dtype=float32)
     *     g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0[64] (dtype=float32)
     *     g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0[64] (dtype=float32)
     *     g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0[64] (dtype=float32)
     * outputs:
     *     g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0[64, 56, 56, 256] (dtype=bf16)
     *     g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3[64] (dtype=float32)
     *     g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3[64] (dtype=float32)
     *     g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1[64] (dtype=float32)
     *     g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3[64] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0 tensor
    unsigned g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0 = createTensors(1,
                                                                                                INPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0 tensor
    unsigned g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0 = createTensors(1,
                                                                                              INPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0 tensor
    unsigned g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_max_sizes[] = {64};
    unsigned g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_min_sizes[] = {64};
    unsigned g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0 = createTensors(1,
                                                                                                               INPUT_TENSOR,
                                                                                                               true,
                                                                                                               "g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0",
                                                                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                               nullptr,
                                                                                                               g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_max_sizes,
                                                                                                               1,
                                                                                                               syn_type_single,
                                                                                                               nullptr,
                                                                                                               0,
                                                                                                               0,
                                                                                                               nullptr,
                                                                                                               false,
                                                                                                               g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0_min_sizes,
                                                                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0 tensor
    unsigned g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_max_sizes[] = {64};
    unsigned g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_min_sizes[] = {64};
    unsigned g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0 = createTensors(1,
                                                                                                                 INPUT_TENSOR,
                                                                                                                 true,
                                                                                                                 "g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0",
                                                                                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                                 nullptr,
                                                                                                                 g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_max_sizes,
                                                                                                                 1,
                                                                                                                 syn_type_single,
                                                                                                                 nullptr,
                                                                                                                 0,
                                                                                                                 0,
                                                                                                                 nullptr,
                                                                                                                 false,
                                                                                                                 g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0_min_sizes,
                                                                                                                 synTensorType::DATA_TENSOR)[0];

    // create g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0 tensor
    unsigned g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_max_sizes[] = {64,56,56,256};
    unsigned g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_min_sizes[] = {64,56,56,256};
    unsigned g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_max_sizes,
                                                                                                4,
                                                                                                syn_type_bf16,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3 tensor
    unsigned g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_max_sizes[] = {64};
    unsigned g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_min_sizes[] = {64};
    unsigned g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 tensor
    unsigned g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];

    // create g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1 tensor
    unsigned g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_max_sizes[] = {64};
    unsigned g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_min_sizes[] = {64};
    unsigned g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1 = createTensors(1,
                                                                                                OUTPUT_TENSOR,
                                                                                                true,
                                                                                                "g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1",
                                                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                                nullptr,
                                                                                                g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_max_sizes,
                                                                                                1,
                                                                                                syn_type_single,
                                                                                                nullptr,
                                                                                                0,
                                                                                                0,
                                                                                                nullptr,
                                                                                                false,
                                                                                                g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1_min_sizes,
                                                                                                synTensorType::DATA_TENSOR)[0];

    // create g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 tensor
    unsigned g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes[] = {64};
    unsigned g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes[] = {64};
    unsigned g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3 = createTensors(1,
                                                                                              OUTPUT_TENSOR,
                                                                                              true,
                                                                                              "g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3",
                                                                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                              nullptr,
                                                                                              g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_max_sizes,
                                                                                              1,
                                                                                              syn_type_single,
                                                                                              nullptr,
                                                                                              0,
                                                                                              0,
                                                                                              nullptr,
                                                                                              false,
                                                                                              g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_min_sizes,
                                                                                              synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_id;
    unsigned char g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_params[] = {149,191,214,51,205,204,204,61,159,240,39,55,1,0,0,0};
    addNodeToGraph("batch_norm_fwd_bf16", {g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0, g_0_t582_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_1_0, g_0_t581_while_body__1_while_resnet50_bn2a_branch2b_readvariableop_0, g_0_t583_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_0, g_0_t584_while_body__1_while_resnet50_bn2a_branch2b_fusedbatchnormv3_readvariableop_1_0}, {g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0, g_0_t970_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_3, g_0_t972_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3, g_0_t963_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_1, g_0_t969_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3}, (void*)g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_params, 16, "g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0", 0 /*graphIndex*/, &g_0_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_batch_norm_fwd_bf16_n308_0_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_SPILL_FILL_FUSION", "true");
    addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "false");

    addConfigurationToRun(SECOND_RUN, "ENABLE_SPILL_FILL_FUSION", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BUNDLE_EVICTION_FUSING", "true");

    compareRunsResults({g_0_t961_while_body__1_while_resnet50_res2a_branch2b_Conv2D_0, g_0_t962_while_body__1_while_resnet50_bn2a_branch2b_FusedBatchNormV3_0});
}