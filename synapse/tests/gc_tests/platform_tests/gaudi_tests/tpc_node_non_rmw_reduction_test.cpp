#include "gc_gaudi_test_infra.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "synapse_common_types.hpp"

// This test reproduces bug from [SW-98982]
TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_node_non_rmw_reduction)
{
    unsigned tensor_0_sizes[] = {8208, 64};
    unsigned tensor_0 =
        createTensors(1, INPUT_TENSOR, true, "tensor_0", MEM_INIT_ALL_ZERO, nullptr, tensor_0_sizes, 2)[0];

    unsigned tensor_1_sizes[] = {512, 8208};
    unsigned tensor_1 =
        createTensors(1, OUTPUT_TENSOR, false, "tensor_1", MEM_INIT_ALL_ZERO, nullptr, tensor_1_sizes, 2)[0];

    unsigned tensor_2_sizes[] = {512, 64};
    unsigned tensor_2 =
        createTensors(1, OUTPUT_TENSOR, true, "tensor_2", MEM_INIT_ALL_ZERO, nullptr, tensor_2_sizes, 2)[0];

    synGEMMParams gemmNodeParams;
    addNodeToGraph("gemm", {tensor_0, tensor_1}, {tensor_2}, &gemmNodeParams, sizeof(gemmNodeParams), "gemmNode");

    unsigned tensor_3_sizes[] = {8208, 512};
    unsigned tensor_3 =
        createTensors(1, INPUT_TENSOR, true, "tensor_3", MEM_INIT_ALL_ZERO, nullptr, tensor_3_sizes, 2)[0];

    unsigned tensor_4_sizes[] = {1};
    unsigned tensor_4 =
        createTensors(1, INPUT_TENSOR, true, "tensor_4", MEM_INIT_ALL_ZERO, nullptr, tensor_4_sizes, 1)[0];

    unsigned tensor_5_sizes[] = {8208, 512};
    unsigned tensor_5 =
        createTensors(1, OUTPUT_TENSOR, false, "tensor_5", MEM_INIT_ALL_ZERO, nullptr, tensor_5_sizes, 2)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_3, tensor_4}, {tensor_5}, nullptr, 0, "multNode1");

    unsigned tensor_6_sizes[] = {1};
    unsigned tensor_6 =
        createTensors(1, INPUT_TENSOR, true, "tensor_6", MEM_INIT_ALL_ZERO, nullptr, tensor_6_sizes, 1)[0];

    unsigned tensor_7_sizes[] = {1};
    unsigned tensor_7 =
        createTensors(1, OUTPUT_TENSOR, true, "tensor_7", MEM_INIT_ALL_ZERO, nullptr, tensor_7_sizes, 1)[0];
    addNodeToGraph("sqrt_fwd_f32", {tensor_6}, {tensor_7}, nullptr, 0, "sqrtNode");

    unsigned tensor_8_sizes[] = {8208, 512};
    unsigned tensor_8 =
        createTensors(1, OUTPUT_TENSOR, false, "tensor_8", MEM_INIT_ALL_ZERO, nullptr, tensor_8_sizes, 2)[0];

    addNodeToGraph("mult_fwd_f32", {tensor_5, tensor_7}, {tensor_8}, nullptr, 0, "multNodeId2");

    unsigned tensor_9_sizes[] = {1};
    unsigned tensor_9 =
        createTensors(1, INPUT_TENSOR, true, "tensor_9", MEM_INIT_ALL_ZERO, nullptr, tensor_9_sizes, 1)[0];

    unsigned tensor_10_sizes[] = {8208, 512};
    unsigned tensor_10 =
        createTensors(1, OUTPUT_TENSOR, true, "tensor_10", MEM_INIT_ALL_ZERO, nullptr, tensor_10_sizes, 2)[0];

    addNodeToGraph("mult_fwd_f32", {tensor_8, tensor_9}, {tensor_10}, nullptr, 0, "multNode3");

    synTransposeParams transposNodeParams = {{TPD_Width, TPD_Channel}, 2};

    addNodeToGraph("transpose",
                   {tensor_10},
                   {tensor_1},
                   &transposNodeParams,
                   sizeof(transposNodeParams),
                   "transposeNode");

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "18874368");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({tensor_7, tensor_2});
}