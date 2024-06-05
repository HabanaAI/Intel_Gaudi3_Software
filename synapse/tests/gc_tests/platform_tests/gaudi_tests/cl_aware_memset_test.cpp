#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "test_types.hpp"
#include <cstddef>

class SynTrainingClAwareMemsetUnitTest : public SynGaudiTwoRunCompareTest
{
public:
    void setConfigsForTest()
    {
        addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_BUNDLE_MEMORY_MANAGEMENT", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_REPLACE_MEMSET_BY_CACHE_WARMUP", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_CACHE_WARMUP_ON_SINGLE_DCORE", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_ADD_CACHE_WARMUP", "true");

        addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "true");
        addConfigurationToRun(SECOND_RUN, "ENABLE_BUNDLE_MEMORY_MANAGEMENT", "true");
        addConfigurationToRun(SECOND_RUN, "ENABLE_ADD_CACHE_WARMUP", "false");
    }

    void bundledLayerNorm()
    {
        // This sequence taken from bert and should create partials writes of LN outputs
        // create layer_norm_input1 tensor
        unsigned layer_norm_input1_sizes[] = {1024, 14336, 1, 1};
        unsigned layer_norm_input1         = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "layer_norm_input1",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   layer_norm_input1_sizes,
                                                   4,
                                                   syn_type_bf16,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   layer_norm_input1_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

        // create layer_norm_input2 tensor
        unsigned layer_norm_input2_sizes[] = {1024};
        unsigned layer_norm_input2         = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "layer_norm_input2",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   layer_norm_input2_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   layer_norm_input2_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

        // create layer_norm_input3 tensor
        unsigned layer_norm_input3_sizes[] = {1024};
        unsigned layer_norm_input3         = createTensors(1,
                                                   INPUT_TENSOR,
                                                   true,
                                                   "layer_norm_input3",
                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                   nullptr,
                                                   layer_norm_input3_sizes,
                                                   1,
                                                   syn_type_single,
                                                   nullptr,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   false,
                                                   layer_norm_input3_sizes,
                                                   synTensorType::DATA_TENSOR)[0];

        // create layer_norm_output1 tensor
        unsigned layer_norm_output1_sizes[] = {1024, 14336, 1, 1};
        unsigned layer_norm_output1         = createTensors(1,
                                                    OUTPUT_TENSOR,
                                                    false,
                                                    "layer_norm_output1",
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    layer_norm_output1_sizes,
                                                    4,
                                                    syn_type_bf16,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    layer_norm_output1_sizes,
                                                    synTensorType::DATA_TENSOR)[0];

        // create layer_norm_output2 tensor
        unsigned layer_norm_output2_sizes[] = {1, 14336, 1, 1};
        unsigned layer_norm_output2         = createTensors(1,
                                                    OUTPUT_TENSOR,
                                                    true,
                                                    "layer_norm_output2",
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    layer_norm_output2_sizes,
                                                    4,
                                                    syn_type_single,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    layer_norm_output2_sizes,
                                                    synTensorType::DATA_TENSOR)[0];

        // create layer_norm_output3 tensor
        unsigned      layer_norm_output3_sizes[]   = {1, 14336, 1, 1};
        unsigned      layer_norm_output3           = createTensors(1,
                                                    OUTPUT_TENSOR,
                                                    true,
                                                    "layer_norm_output3",
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    layer_norm_output3_sizes,
                                                    4,
                                                    syn_type_single,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    false,
                                                    layer_norm_output3_sizes,
                                                    synTensorType::DATA_TENSOR)[0];
        unsigned char layer_norm_fwd_bf16_params[] = {1, 0, 0, 0, 204, 188, 140, 43};
        synNodeId     layerNormId;
        addNodeToGraph("layer_norm_fwd_bf16",
                       {layer_norm_input1, layer_norm_input2, layer_norm_input3},
                       {layer_norm_output1, layer_norm_output2, layer_norm_output3},
                       layer_norm_fwd_bf16_params,
                       8,
                       "layer_norm",
                       0,
                       &layerNormId);

        // create bgemm_opA tensor
        unsigned  bgemm_opA_sizes[] = {1024, 512, 28};
        unsigned  bgemm_opA         = createTensors(1,
                                           OUTPUT_TENSOR,
                                           false,
                                           "bgemm_opA",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           bgemm_opA_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_opA_sizes,
                                           synTensorType::DATA_TENSOR)[0];
        synNodeId reshapeId;
        addNodeToGraph("reshape", {layer_norm_output1}, {bgemm_opA}, nullptr, 0, "reshape", 0, &reshapeId);

        // create bgemm_opB tensor
        unsigned bgemm_opB_sizes[] = {1024, 4096};
        unsigned bgemm_opB         = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "bgemm_opB",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           bgemm_opB_sizes,
                                           2,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_opB_sizes,
                                           synTensorType::DATA_TENSOR)[0];

        // create bgemm_out tensor
        unsigned      bgemm_out_sizes[] = {4096, 512, 28};
        unsigned      bgemm_out         = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "bgemm_out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           bgemm_out_sizes,
                                           3,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_out_sizes,
                                           synTensorType::DATA_TENSOR)[0];
        unsigned char bgemm_params[]    = {0, 1};
        synNodeId     bgemmId;
        addNodeToGraph("batch_gemm",
                       {bgemm_opA, bgemm_opB},
                       {bgemm_out},
                       (void*)bgemm_params,
                       2,
                       "batch_gemm",
                       0,
                       &bgemmId);
        setConfigsForTest();
        // bgemm output not relevant for comparison because just those LN outputs have partials writes
        compareRunsResults({layer_norm_output2, layer_norm_output3});
    }

    void convPacking()
    {
        // Graph #0
        /*************
         * g_0_layer1_0_conv1_0 node
         * inputs:
         *     g_0_worker_0_maxpool_output[64, 56, 56, 64] (dtype=bf16)
         *     g_0_layer1_0_conv1_weight[64, 64, 1, 1] (dtype=bf16)
         * outputs:
         *     g_0_layer1_0_conv1_output[64, 56, 56, 64] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_worker_0_maxpool_output tensor
        unsigned g_0_worker_0_maxpool_output_max_sizes[] = {64, 56, 56, 64};
        unsigned g_0_worker_0_maxpool_output_min_sizes[] = {64, 56, 56, 64};
        unsigned g_0_worker_0_maxpool_output             = createTensors(1,
                                                             INPUT_TENSOR,
                                                             true,
                                                             "g_0_worker_0_maxpool_output",
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             g_0_worker_0_maxpool_output_max_sizes,
                                                             4,
                                                             syn_type_bf16,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             g_0_worker_0_maxpool_output_min_sizes,
                                                             synTensorType::DATA_TENSOR)[0];

        // create g_0_layer1_0_conv1_weight tensor
        unsigned g_0_layer1_0_conv1_weight_max_sizes[] = {64, 64, 1, 1};
        unsigned g_0_layer1_0_conv1_weight_min_sizes[] = {64, 64, 1, 1};
        unsigned g_0_layer1_0_conv1_weight             = createTensors(1,
                                                           INPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_0_conv1_weight",
                                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                           nullptr,
                                                           g_0_layer1_0_conv1_weight_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_0_conv1_weight_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];

        // create g_0_layer1_0_conv1_output tensor
        unsigned      g_0_layer1_0_conv1_output_max_sizes[] = {64, 56, 56, 64};
        unsigned      g_0_layer1_0_conv1_output_min_sizes[] = {64, 56, 56, 64};
        unsigned      g_0_layer1_0_conv1_output             = createTensors(1,
                                                           OUTPUT_TENSOR,
                                                           true,
                                                           "g_0_layer1_0_conv1_output",
                                                           MEM_INIT_ALL_ZERO,
                                                           nullptr,
                                                           g_0_layer1_0_conv1_output_max_sizes,
                                                           4,
                                                           syn_type_bf16,
                                                           nullptr,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           false,
                                                           g_0_layer1_0_conv1_output_min_sizes,
                                                           synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_layer1_0_conv1_0_id;
        unsigned char g_0_layer1_0_conv1_0_params[] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        addNodeToGraph("spatial_convolution",
                       {g_0_worker_0_maxpool_output, g_0_layer1_0_conv1_weight},
                       {g_0_layer1_0_conv1_output},
                       (void*)g_0_layer1_0_conv1_0_params,
                       112,
                       "g_0_layer1_0_conv1_0",
                       0 /*graphIndex*/,
                       &g_0_layer1_0_conv1_0_id);

        setConfigsForTest();

        compareRunsResults({g_0_layer1_0_conv1_output});
    }
    void bundledCrossEntropy()
    {
        /*************
         * g_0_cross_entropy_loss0_log_softmax_bwd_0 node
         * inputs:
         *     g_0_cross_entropy_loss0_logs_output[1000, 64] (dtype=bf16)
         *     g_0_target[64] (dtype=int32)
         * outputs:
         *     g_0_cross_entropy_loss0_grad_input[1000, 64] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_cross_entropy_loss0_logs_output tensor
        unsigned g_0_cross_entropy_loss0_logs_output_max_sizes[] = {1000, 64};
        unsigned g_0_cross_entropy_loss0_logs_output_min_sizes[] = {1000, 64};
        unsigned g_0_cross_entropy_loss0_logs_output             = createTensors(1,
                                                                     INPUT_TENSOR,
                                                                     true,
                                                                     "g_0_cross_entropy_loss0_logs_output",
                                                                     MEM_INIT_ALL_ZERO,
                                                                     nullptr,
                                                                     g_0_cross_entropy_loss0_logs_output_max_sizes,
                                                                     2,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_cross_entropy_loss0_logs_output_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];

        // create g_0_target tensor
        unsigned g_0_target_max_sizes[] = {64};
        unsigned g_0_target_min_sizes[] = {64};
        unsigned g_0_target             = createTensors(1,
                                            INPUT_TENSOR,
                                            true,
                                            "g_0_target",
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            g_0_target_max_sizes,
                                            1,
                                            syn_type_int32,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            false,
                                            g_0_target_min_sizes,
                                            synTensorType::DATA_TENSOR)[0];

        // create g_0_cross_entropy_loss0_grad_input tensor
        unsigned      g_0_cross_entropy_loss0_grad_input_max_sizes[] = {1000, 64};
        unsigned      g_0_cross_entropy_loss0_grad_input_min_sizes[] = {1000, 64};
        unsigned      g_0_cross_entropy_loss0_grad_input             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "g_0_cross_entropy_loss0_grad_input",
                                                                    MEM_INIT_ALL_ZERO,
                                                                    nullptr,
                                                                    g_0_cross_entropy_loss0_grad_input_max_sizes,
                                                                    2,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    g_0_cross_entropy_loss0_grad_input_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_cross_entropy_loss0_log_softmax_bwd_0_id;
        unsigned char g_0_cross_entropy_loss0_log_softmax_bwd_0_params[] = {1, 0, 0, 0, 64, 0, 0, 0};
        addNodeToGraph("softmax_cross_entropy_bwd_bf16",
                       {g_0_cross_entropy_loss0_logs_output, g_0_target},
                       {g_0_cross_entropy_loss0_grad_input},
                       (void*)g_0_cross_entropy_loss0_log_softmax_bwd_0_params,
                       8,
                       "g_0_cross_entropy_loss0_log_softmax_bwd_0",
                       0 /*graphIndex*/,
                       &g_0_cross_entropy_loss0_log_softmax_bwd_0_id);

        /*************
         * g_0_worker_0_fc_dedx_reshape_0 node
         * inputs:
         *     g_0_cross_entropy_loss0_grad_input[1000, 64] (dtype=bf16)
         * outputs:
         *     g_0_worker_0_fc_dedx_tensor_reshape[1000, 1, 1, 64] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_worker_0_fc_dedx_tensor_reshape tensor
        unsigned  g_0_worker_0_fc_dedx_tensor_reshape_max_sizes[] = {1000, 1, 1, 64};
        unsigned  g_0_worker_0_fc_dedx_tensor_reshape_min_sizes[] = {1000, 1, 1, 64};
        unsigned  g_0_worker_0_fc_dedx_tensor_reshape             = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_worker_0_fc_dedx_tensor_reshape",
                                                                     MEM_INIT_ALL_ZERO,
                                                                     nullptr,
                                                                     g_0_worker_0_fc_dedx_tensor_reshape_max_sizes,
                                                                     4,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_worker_0_fc_dedx_tensor_reshape_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
        synNodeId g_0_worker_0_fc_dedx_reshape_0_id;
        addNodeToGraph("reshape",
                       {g_0_cross_entropy_loss0_grad_input},
                       {g_0_worker_0_fc_dedx_tensor_reshape},
                       nullptr,
                       0,
                       "g_0_worker_0_fc_dedx_reshape_0",
                       0 /*graphIndex*/,
                       &g_0_worker_0_fc_dedx_reshape_0_id);

        /*************
         * g_0_worker_0_fc_dedx_0 node
         * inputs:
         *     g_0_worker_0_fc_dedx_tensor_reshape[1000, 1, 1, 64] (dtype=bf16)
         *     g_0_worker_0_fc_weight[1000, 2048, 1, 1] (dtype=bf16)
         * outputs:
         *     g_0_worker_0_fc_grad_input[2048, 1, 1, 64] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_worker_0_fc_weight tensor
        unsigned g_0_worker_0_fc_weight_max_sizes[] = {1000, 2048, 1, 1};
        unsigned g_0_worker_0_fc_weight_min_sizes[] = {1000, 2048, 1, 1};
        unsigned g_0_worker_0_fc_weight             = createTensors(1,
                                                        INPUT_TENSOR,
                                                        true,
                                                        "g_0_worker_0_fc_weight",
                                                        MEM_INIT_ALL_ZERO,
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

        // create g_0_worker_0_fc_grad_input tensor
        unsigned      g_0_worker_0_fc_grad_input_max_sizes[] = {2048, 1, 1, 64};
        unsigned      g_0_worker_0_fc_grad_input_min_sizes[] = {2048, 1, 1, 64};
        unsigned      g_0_worker_0_fc_grad_input             = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_worker_0_fc_grad_input",
                                                            MEM_INIT_ALL_ZERO,
                                                            nullptr,
                                                            g_0_worker_0_fc_grad_input_max_sizes,
                                                            4,
                                                            syn_type_bf16,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_worker_0_fc_grad_input_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_worker_0_fc_dedx_0_id;
        unsigned char g_0_worker_0_fc_dedx_0_params[] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        addNodeToGraph("dedx",
                       {g_0_worker_0_fc_dedx_tensor_reshape, g_0_worker_0_fc_weight},
                       {g_0_worker_0_fc_grad_input},
                       (void*)g_0_worker_0_fc_dedx_0_params,
                       112,
                       "g_0_worker_0_fc_dedx_0",
                       0 /*graphIndex*/,
                       &g_0_worker_0_fc_dedx_0_id);

        /*************
         * g_0_worker_0_fc_dedw_reshape_0 node
         * inputs:
         *     g_0_cross_entropy_loss0_grad_input[1000, 64] (dtype=bf16)
         * outputs:
         *     g_0_worker_0_fc_dedw_tensor_reshape[1000, 1, 1, 64] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_worker_0_fc_dedw_tensor_reshape tensor
        unsigned  g_0_worker_0_fc_dedw_tensor_reshape_max_sizes[] = {1000, 1, 1, 64};
        unsigned  g_0_worker_0_fc_dedw_tensor_reshape_min_sizes[] = {1000, 1, 1, 64};
        unsigned  g_0_worker_0_fc_dedw_tensor_reshape             = createTensors(1,
                                                                     OUTPUT_TENSOR,
                                                                     false,
                                                                     "g_0_worker_0_fc_dedw_tensor_reshape",
                                                                     MEM_INIT_ALL_ZERO,
                                                                     nullptr,
                                                                     g_0_worker_0_fc_dedw_tensor_reshape_max_sizes,
                                                                     4,
                                                                     syn_type_bf16,
                                                                     nullptr,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     false,
                                                                     g_0_worker_0_fc_dedw_tensor_reshape_min_sizes,
                                                                     synTensorType::DATA_TENSOR)[0];
        synNodeId g_0_worker_0_fc_dedw_reshape_0_id;
        addNodeToGraph("reshape",
                       {g_0_cross_entropy_loss0_grad_input},
                       {g_0_worker_0_fc_dedw_tensor_reshape},
                       nullptr,
                       0,
                       "g_0_worker_0_fc_dedw_reshape_0",
                       0 /*graphIndex*/,
                       &g_0_worker_0_fc_dedw_reshape_0_id);

        /*************
         * g_0_worker_0_fc_dedw_0 node
         * inputs:
         *     g_0_worker_0_fc_dedw_tensor_reshape[1000, 1, 1, 64] (dtype=bf16)
         *     g_0_worker_0_avgpool_output[2048, 1, 1, 64] (dtype=bf16)
         * outputs:
         *     g_0_worker_0_fc_weight_grad[1000, 2048, 1, 1] (dtype=float32)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_worker_0_avgpool_output tensor
        unsigned g_0_worker_0_avgpool_output_max_sizes[] = {2048, 1, 1, 64};
        unsigned g_0_worker_0_avgpool_output_min_sizes[] = {2048, 1, 1, 64};
        unsigned g_0_worker_0_avgpool_output             = createTensors(1,
                                                             INPUT_TENSOR,
                                                             true,
                                                             "g_0_worker_0_avgpool_output",
                                                             MEM_INIT_ALL_ZERO,
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

        // create g_0_worker_0_fc_weight_grad tensor
        unsigned      g_0_worker_0_fc_weight_grad_max_sizes[] = {1000, 2048, 1, 1};
        unsigned      g_0_worker_0_fc_weight_grad_min_sizes[] = {1000, 2048, 1, 1};
        unsigned      g_0_worker_0_fc_weight_grad             = createTensors(1,
                                                             OUTPUT_TENSOR,
                                                             true,
                                                             "g_0_worker_0_fc_weight_grad",
                                                             MEM_INIT_ALL_ZERO,
                                                             nullptr,
                                                             g_0_worker_0_fc_weight_grad_max_sizes,
                                                             4,
                                                             syn_type_single,
                                                             nullptr,
                                                             0,
                                                             0,
                                                             nullptr,
                                                             false,
                                                             g_0_worker_0_fc_weight_grad_min_sizes,
                                                             synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_worker_0_fc_dedw_0_id;
        unsigned char g_0_worker_0_fc_dedw_0_params[] = {
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        addNodeToGraph("dedw",
                       {g_0_worker_0_fc_dedw_tensor_reshape, g_0_worker_0_avgpool_output},
                       {g_0_worker_0_fc_weight_grad},
                       (void*)g_0_worker_0_fc_dedw_0_params,
                       112,
                       "g_0_worker_0_fc_dedw_0",
                       0 /*graphIndex*/,
                       &g_0_worker_0_fc_dedw_0_id);
        setConfigsForTest();

        addConfigurationToRun(FIRST_RUN, "ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE", "0");
        addConfigurationToRun(SECOND_RUN, "ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE", "0");

        compareRunsResults({g_0_worker_0_fc_weight_grad});
    }

    void bundledInstanceNorm()
    {
        // Graph #0

        /*************
         * g_0_gradient_model_2_transp_conv_spatial_convolution_251_0 node
         * inputs:
         *     g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice[40, 48, 128, 64] (dtype=bf16)
         *     g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast[2, 2, 128, 256] (dtype=bf16)
         * outputs:
         *     g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable[20, 24, 256,
         *64] (dtype=bf16) ctrl inputs: ctrl outputs:
         *************/

        // create g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice tensor
        unsigned g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice_max_sizes[] = {40, 48, 128, 64};
        unsigned g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice_min_sizes[] = {40, 48, 128, 64};
        unsigned g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice_max_sizes,
                          4,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast tensor
        unsigned g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast_max_sizes[] = {2, 2, 128, 256};
        unsigned g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast_min_sizes[] = {2, 2, 128, 256};
        unsigned g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast_max_sizes,
                          4,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable tensor
        unsigned
            g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable_max_sizes[] =
                {20, 24, 256, 64};
        unsigned
            g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable_min_sizes[] =
                {20, 24, 256, 64};
        unsigned g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable =
            createTensors(
                1,
                OUTPUT_TENSOR,
                false,
                "g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable",
                MEM_INIT_RANDOM_WITH_NEGATIVE,
                nullptr,
                g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable_max_sizes,
                4,
                syn_type_bf16,
                nullptr,
                0,
                0,
                nullptr,
                false,
                g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable_min_sizes,
                synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_gradient_model_2_transp_conv_spatial_convolution_251_0_id;
        unsigned char g_0_gradient_model_2_transp_conv_spatial_convolution_251_0_params[] = {
            2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        const char* convInputLayouts[]  = {"WHCN", "SRCK"};
        const char* convOutputLayouts[] = {"WHCN"};
        addNodeToGraph("spatial_convolution",
                       {g_0_tensor_443_id_1755_gradient_model_2_transp_conv_aten__slice,
                        g_0_tensor_102_id_1244_model_2_transp_conv_hpu__cast},
                       {g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable},
                       (void*)g_0_gradient_model_2_transp_conv_spatial_convolution_251_0_params,
                       112,
                       "g_0_gradient_model_2_transp_conv_spatial_convolution_251_0",
                       0 /*graphIndex*/,
                       &g_0_gradient_model_2_transp_conv_spatial_convolution_251_0_id,
                       convInputLayouts,
                       convOutputLayouts);

        /*************
         * g_0_gradient_model_2_transp_conv_leakyrelu_bwd_bf16_259_0 node
         * inputs:
         *     g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable[20, 24, 256,
         *64] (dtype=bf16) g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu_[20, 24, 256, 64]
         *(dtype=bf16) outputs: g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward[20, 24,
         *256, 64] (dtype=bf16) ctrl inputs: ctrl outputs:
         *************/

        // create g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu_ tensor
        unsigned g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu__max_sizes[] = {20,
                                                                                                        24,
                                                                                                        256,
                                                                                                        64};
        unsigned g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu__min_sizes[] = {20,
                                                                                                        24,
                                                                                                        256,
                                                                                                        64};
        unsigned g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu_ =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu_",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu__max_sizes,
                          4,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu__min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward tensor
        unsigned g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward_max_sizes[] = {20,
                                                                                                              24,
                                                                                                              256,
                                                                                                              64};
        unsigned g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward_min_sizes[] = {20,
                                                                                                              24,
                                                                                                              256,
                                                                                                              64};
        unsigned g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward =
            createTensors(1,
                          OUTPUT_TENSOR,
                          false,
                          "g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward_max_sizes,
                          4,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward_min_sizes,
                          synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_gradient_model_2_transp_conv_leakyrelu_bwd_bf16_259_0_id;
        unsigned char g_0_gradient_model_2_transp_conv_leakyrelu_bwd_bf16_259_0_params[] =
            {0, 0, 0, 64, 225, 122, 132, 63};
        addNodeToGraph("leakyrelu_bwd_bf16",
                       {g_0_tensor_444_id_1759_gradient_model_2_transp_conv_aten__convolution_backward_overrideable,
                        g_0_tensor_208_id_1235_model_1_conv_block_conv2_lrelu_aten__leaky_relu_},
                       {g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward},
                       (void*)g_0_gradient_model_2_transp_conv_leakyrelu_bwd_bf16_259_0_params,
                       8,
                       "g_0_gradient_model_2_transp_conv_leakyrelu_bwd_bf16_259_0",
                       0 /*graphIndex*/,
                       &g_0_gradient_model_2_transp_conv_leakyrelu_bwd_bf16_259_0_id);

        /*************
         * g_0_gradient_model_1_instance_norm_bwd_bf16_260_0 node
         * inputs:
         *     g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable[20, 24, 256, 64]
         *(dtype=bf16) g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward[20, 24, 256, 64]
         *(dtype=bf16) g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm[256, 64] (dtype=float32)
         *     g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm[256, 64] (dtype=float32)
         *     g_0_tensor_203[256] (dtype=float32)
         * outputs:
         *     g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward[20, 24, 256, 64] (dtype=bf16)
         *     g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward[256] (dtype=float32)
         *     g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward[256] (dtype=float32)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable tensor
        unsigned g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable_max_sizes[] = {20,
                                                                                                                    24,
                                                                                                                    256,
                                                                                                                    64};
        unsigned g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable_min_sizes[] = {20,
                                                                                                                    24,
                                                                                                                    256,
                                                                                                                    64};
        unsigned g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable_max_sizes,
                          4,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm tensor
        unsigned g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm_max_sizes[] = {256, 64};
        unsigned g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm_min_sizes[] = {256, 64};
        unsigned g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm_max_sizes,
                          2,
                          syn_type_single,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm tensor
        unsigned g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm_max_sizes[] = {256, 64};
        unsigned g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm_min_sizes[] = {256, 64};
        unsigned g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm_max_sizes,
                          2,
                          syn_type_single,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_203 tensor
        unsigned g_0_tensor_203_max_sizes[] = {256};
        unsigned g_0_tensor_203_min_sizes[] = {256};
        unsigned g_0_tensor_203             = createTensors(1,
                                                INPUT_TENSOR,
                                                true,
                                                "g_0_tensor_203",
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                g_0_tensor_203_max_sizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                false,
                                                g_0_tensor_203_min_sizes,
                                                synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward tensor
        unsigned g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward_max_sizes[] = {20, 24, 256, 64};
        unsigned g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward_min_sizes[] = {20, 24, 256, 64};
        unsigned g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward =
            createTensors(1,
                          OUTPUT_TENSOR,
                          true,
                          "g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward_max_sizes,
                          4,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward tensor
        unsigned g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward_max_sizes[] = {256};
        unsigned g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward_min_sizes[] = {256};
        unsigned g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward =
            createTensors(1,
                          OUTPUT_TENSOR,
                          true,
                          "g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward_max_sizes,
                          1,
                          syn_type_single,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward tensor
        unsigned g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward_max_sizes[] = {256};
        unsigned g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward_min_sizes[] = {256};
        unsigned g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward =
            createTensors(1,
                          OUTPUT_TENSOR,
                          true,
                          "g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward_max_sizes,
                          1,
                          syn_type_single,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward_min_sizes,
                          synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_gradient_model_1_instance_norm_bwd_bf16_260_0_id;
        unsigned char g_0_gradient_model_1_instance_norm_bwd_bf16_260_0_params[] =
            {102, 102, 102, 63, 172, 197, 39, 55};

        const char* normInputLayouts[]  = {"WHCN", "WHCN", "", "", ""};
        const char* normOutputLayouts[] = {"WHCN", "", ""};
        addNodeToGraph("instance_norm_bwd_bf16",
                       {g_0_tensor_202_id_1233_model_1_conv_block_conv2_conv_aten__convolution_overrideable,
                        g_0_tensor_452_id_1771_gradient_model_2_transp_conv_aten__leaky_relu_backward,
                        g_0_tensor_206_id_1237_model_1_conv_block_conv2_norm_hpu__instance_norm,
                        g_0_tensor_207_id_1239_model_1_conv_block_conv2_norm_hpu__instance_norm,
                        g_0_tensor_203},
                       {g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward,
                        g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward,
                        g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward},
                       (void*)g_0_gradient_model_1_instance_norm_bwd_bf16_260_0_params,
                       8,
                       "g_0_gradient_model_1_instance_norm_bwd_bf16_260_0",
                       0 /*graphIndex*/,
                       &g_0_gradient_model_1_instance_norm_bwd_bf16_260_0_id,
                       normInputLayouts,
                       normOutputLayouts);

        setConfigsForTest();

        compareRunsResults({g_0_tensor_453_id_1775_gradient_model_1_hpu__instance_norm_backward,
                            g_0_tensor_454_id_1777_gradient_model_1_hpu__instance_norm_backward,
                            g_0_tensor_455_id_1779_gradient_model_1_hpu__instance_norm_backward});
    }

    void bundleAddWithPartialsOutput()
    {
        // This sequence taken from pt_bart_1x with graph editor tool with the below command:
        // ./graph_editor.py -n lm_head/batch_gemm/1185 -m pt_bart_1x -g 0 -o bundle_100.json -r 1 --fwd
        // The test was generated by generate_gd_cpp with the below command:
        //./generate_gd_cpp.py -f bundle_100 -o partials_hybrid_kernel.cpp -g gaudi-single
        // The test should add hybrid cache warmup
        // Graph #0

        /*************
         * g_0_lm_head_batch_gemm_1185_0 node
         * inputs:
         *     g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm[768, 127, 32]
         *(dtype=bf16) g_0_tensor_1895_id_5456_lm_head_aten__t[50265, 768] (dtype=bf16) outputs:
         *     g_0_tensor_1896_id_5458_lm_head_aten__matmul[50265, 127, 32] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm tensor
        unsigned g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm_max_sizes[] = {768,
                                                                                                                 127,
                                                                                                                 32};
        unsigned g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm_min_sizes[] = {768,
                                                                                                                 127,
                                                                                                                 32};
        unsigned g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm_max_sizes,
                          3,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_1895_id_5456_lm_head_aten__t tensor
        unsigned g_0_tensor_1895_id_5456_lm_head_aten__t_max_sizes[] = {50265, 768};
        unsigned g_0_tensor_1895_id_5456_lm_head_aten__t_min_sizes[] = {50265, 768};
        unsigned g_0_tensor_1895_id_5456_lm_head_aten__t =
            createTensors(1,
                          INPUT_TENSOR,
                          true,
                          "g_0_tensor_1895_id_5456_lm_head_aten__t",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_1895_id_5456_lm_head_aten__t_max_sizes,
                          2,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_1895_id_5456_lm_head_aten__t_min_sizes,
                          synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_1896_id_5458_lm_head_aten__matmul tensor
        unsigned g_0_tensor_1896_id_5458_lm_head_aten__matmul_max_sizes[] = {50265, 127, 32};
        unsigned g_0_tensor_1896_id_5458_lm_head_aten__matmul_min_sizes[] = {50265, 127, 32};
        unsigned g_0_tensor_1896_id_5458_lm_head_aten__matmul =
            createTensors(1,
                          OUTPUT_TENSOR,
                          false,
                          "g_0_tensor_1896_id_5458_lm_head_aten__matmul",
                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                          nullptr,
                          g_0_tensor_1896_id_5458_lm_head_aten__matmul_max_sizes,
                          3,
                          syn_type_bf16,
                          nullptr,
                          0,
                          0,
                          nullptr,
                          false,
                          g_0_tensor_1896_id_5458_lm_head_aten__matmul_min_sizes,
                          synTensorType::DATA_TENSOR)[0];
        synNodeId     g_0_lm_head_batch_gemm_1185_0_id;
        unsigned char g_0_lm_head_batch_gemm_1185_0_params[] = {0, 0};
        addNodeToGraph("batch_gemm",
                       {g_0_tensor_1890_id_5447_model_decoder_5_final_layer_norm_aten__native_layer_norm,
                        g_0_tensor_1895_id_5456_lm_head_aten__t},
                       {g_0_tensor_1896_id_5458_lm_head_aten__matmul},
                       (void*)g_0_lm_head_batch_gemm_1185_0_params,
                       2,
                       "g_0_lm_head_batch_gemm_1185_0",
                       0 /*graphIndex*/,
                       &g_0_lm_head_batch_gemm_1185_0_id);

        /*************
         * g_0__add_fwd_bf16_1186_0 node
         * inputs:
         *     g_0_tensor_1896_id_5458_lm_head_aten__matmul[50265, 127, 32] (dtype=bf16)
         *     g_0_tensor_1894_id_5462_hpu__cast[50265, 1] (dtype=bf16)
         * outputs:
         *     g_0_tensor_1897_id_5463_aten__add[50265, 127, 32] (dtype=bf16)
         * ctrl inputs:
         * ctrl outputs:
         *************/

        // create g_0_tensor_1894_id_5462_hpu__cast tensor
        unsigned g_0_tensor_1894_id_5462_hpu__cast_max_sizes[] = {50265, 1};
        unsigned g_0_tensor_1894_id_5462_hpu__cast_min_sizes[] = {50265, 1};
        unsigned g_0_tensor_1894_id_5462_hpu__cast             = createTensors(1,
                                                                   INPUT_TENSOR,
                                                                   true,
                                                                   "g_0_tensor_1894_id_5462_hpu__cast",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_tensor_1894_id_5462_hpu__cast_max_sizes,
                                                                   2,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_tensor_1894_id_5462_hpu__cast_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];

        // create g_0_tensor_1897_id_5463_aten__add tensor
        unsigned  g_0_tensor_1897_id_5463_aten__add_max_sizes[] = {50265, 127, 32};
        unsigned  g_0_tensor_1897_id_5463_aten__add_min_sizes[] = {50265, 127, 32};
        unsigned  g_0_tensor_1897_id_5463_aten__add             = createTensors(1,
                                                                   OUTPUT_TENSOR,
                                                                   true,
                                                                   "g_0_tensor_1897_id_5463_aten__add",
                                                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                   nullptr,
                                                                   g_0_tensor_1897_id_5463_aten__add_max_sizes,
                                                                   3,
                                                                   syn_type_bf16,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   g_0_tensor_1897_id_5463_aten__add_min_sizes,
                                                                   synTensorType::DATA_TENSOR)[0];
        synNodeId g_0__add_fwd_bf16_1186_0_id;
        addNodeToGraph("add_fwd_bf16",
                       {g_0_tensor_1896_id_5458_lm_head_aten__matmul, g_0_tensor_1894_id_5462_hpu__cast},
                       {g_0_tensor_1897_id_5463_aten__add},
                       nullptr,
                       0,
                       "g_0__add_fwd_bf16_1186_0",
                       0 /*graphIndex*/,
                       &g_0__add_fwd_bf16_1186_0_id);
        setConfigsForTest();

        compareRunsResults({g_0_tensor_1897_id_5463_aten__add});
    }

    void mmeOutputWithUnalignedFcdStrides()
    {
        // Test case where we have one mme dcore only working.
        // This scenario might be used in case of partials

        // create bgemm_opA tensor
        unsigned bgemm_opA_sizes[] = {64, 128, 12, 32};
        unsigned bgemm_opA         = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "bgemm_opA",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           bgemm_opA_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_opA_sizes,
                                           synTensorType::DATA_TENSOR)[0];

        // create bgemm_opB tensor
        unsigned bgemm_opB_sizes[] = {30, 64, 12, 32};
        unsigned bgemm_opB         = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "bgemm_opB",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           bgemm_opB_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_opB_sizes,
                                           synTensorType::DATA_TENSOR)[0];

        // create bgemm_out tensor
        unsigned      bgemm_out_sizes[] = {30, 128, 12, 32};
        unsigned      bgemm_out         = createTensors(1,
                                           OUTPUT_TENSOR,
                                           true,
                                           "bgemm_out",
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           bgemm_out_sizes,
                                           4,
                                           syn_type_bf16,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           bgemm_out_sizes,
                                           synTensorType::DATA_TENSOR)[0];
        unsigned char bgemm_params[]    = {0, 0};
        synNodeId     bgemmId;
        addNodeToGraph("batch_gemm",
                       {bgemm_opA, bgemm_opB},
                       {bgemm_out},
                       (void*)bgemm_params,
                       2,
                       "batch_gemm",
                       0,
                       &bgemmId);
        setConfigsForTest();
        // perforation disabled to assure partials handling sets this node to run on single dcore
        addConfigurationToRun(FIRST_RUN, "ENABLE_LAYERED_BRAIN_PERFORATION", "false");
        // assure node is handled regardless of bundling decision
        addConfigurationToRun(FIRST_RUN, "ENABLE_LB_PARTIALS_WRITE_UNBUNDELED_NODES_HANDLING", "true");
        addConfigurationToRun(FIRST_RUN, "ENABLE_LB_PARTIALS_WRITE_MME_HANDLING", "true");
        compareRunsResults({bgemm_out});
    }
};

TEST_F_GC(SynTrainingClAwareMemsetUnitTest, clAwareHybridMemsetMemget_bundled_add_ASIC, {synDeviceGaudi3})
{
    bundleAddWithPartialsOutput();
}

TEST_F_GC(SynTrainingClAwareMemsetUnitTest, clAwareMemget_bundled_layer_norm_fwd_ASIC, {synDeviceGaudi3})
{
    bundledLayerNorm();
}

// TODO SW-170102 - irrelevant test since conv weight packing kernel changed. Need to find a different test
TEST_F_GC(SynTrainingClAwareMemsetUnitTest, DISABLED_clAwareMemset_spatial_convolution_ASIC, {synDeviceGaudi3})
{
    // this scenario tests existing memset replacement with CL aware memset
    convPacking();
}

TEST_F_GC(SynTrainingClAwareMemsetUnitTest, clAwareMemget_cross_entropy_ASIC, {synDeviceGaudi3})
{
    bundledCrossEntropy();
}

TEST_F_GC(SynTrainingClAwareMemsetUnitTest, cacheWarmup_singleDcore_ASIC, {synDeviceGaudi3})
{
    bundledInstanceNorm();
}

TEST_F_GC(SynTrainingClAwareMemsetUnitTest, mme_singleDcore_ASIC, {synDeviceGaudi3})
{
    mmeOutputWithUnalignedFcdStrides();
}