#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynGaudiTestInfra, pytorch_transformer, {synDeviceGaudi})
{
    // Graph #0

    /*************
     * g_0_Transpose107_0 node
     * inputs:
     *     g_0_tensor_0[12, 3] min[6, 3] (dtype=float32)
     * outputs:
     *     g_0_tensor_2[3, 12] min[3, 6] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_0 tensor
    unsigned g_0_tensor_0_max_sizes[] = {12,3};
    unsigned g_0_tensor_0_min_sizes[] = {6,3};
    unsigned g_0_tensor_0 = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "g_0_tensor_0",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      g_0_tensor_0_max_sizes,
                                      2,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_0_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_2 tensor
    unsigned g_0_tensor_2_max_sizes[] = {3,12};
    unsigned g_0_tensor_2_min_sizes[] = {3,6};
    unsigned g_0_tensor_2 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "g_0_tensor_2",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      g_0_tensor_2_max_sizes,
                                      2,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_2_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_Transpose107_0_id;
    unsigned char g_0_Transpose107_0_params[] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_0}, {g_0_tensor_2}, (void*)g_0_Transpose107_0_params, 24, "g_0_Transpose107_0", 0 /*graphIndex*/, &g_0_Transpose107_0_id);

    /*************
     * g_0_Transpose108_0 node
     * inputs:
     *     g_0_tensor_1[3, 12, 2] min[3, 6, 2] (dtype=float32)
     * outputs:
     *     g_0_tensor_3[12, 3, 2] min[6, 3, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_1 tensor
    unsigned g_0_tensor_1_max_sizes[] = {3,12,2};
    unsigned g_0_tensor_1_min_sizes[] = {3,6,2};
    unsigned g_0_tensor_1 = createTensors(1,
                                      INPUT_TENSOR,
                                      true,
                                      "g_0_tensor_1",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      g_0_tensor_1_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_1_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_3 tensor
    unsigned g_0_tensor_3_max_sizes[] = {12,3,2};
    unsigned g_0_tensor_3_min_sizes[] = {6,3,2};
    unsigned g_0_tensor_3 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "g_0_tensor_3",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      g_0_tensor_3_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_3_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_Transpose108_0_id;
    unsigned char g_0_Transpose108_0_params[] = {1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_1}, {g_0_tensor_3}, (void*)g_0_Transpose108_0_params, 24, "g_0_Transpose108_0", 0 /*graphIndex*/, &g_0_Transpose108_0_id);

    /*************
     * g_0_BatchGemm109_0 node
     * inputs:
     *     g_0_tensor_3[12, 3, 2] min[6, 3, 2] (dtype=float32)
     *     g_0_tensor_2[3, 12] min[3, 6] (dtype=float32)
     * outputs:
     *     g_0_tensor_4[3, 3, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_4 tensor
    unsigned g_0_tensor_4_max_sizes[] = {3,3,2};
    unsigned g_0_tensor_4_min_sizes[] = {3,3,2};
    unsigned g_0_tensor_4 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false,
                                      "g_0_tensor_4",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      g_0_tensor_4_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_4_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_BatchGemm109_0_id;
    unsigned char g_0_BatchGemm109_0_params[] = {0,0};
    addNodeToGraph("batch_gemm", {g_0_tensor_3, g_0_tensor_2}, {g_0_tensor_4}, (void*)g_0_BatchGemm109_0_params, 2, "g_0_BatchGemm109_0", 0 /*graphIndex*/, &g_0_BatchGemm109_0_id);

    /*************
     * g_0_Transpose110_0 node
     * inputs:
     *     g_0_tensor_4[3, 3, 2] (dtype=float32)
     * outputs:
     *     g_0_tensor_5[3, 3, 2] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_5 tensor
    unsigned g_0_tensor_5_max_sizes[] = {3,3,2};
    unsigned g_0_tensor_5_min_sizes[] = {3,3,2};
    unsigned g_0_tensor_5 = createTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "g_0_tensor_5",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      g_0_tensor_5_max_sizes,
                                      3,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      g_0_tensor_5_min_sizes,
                                      synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_Transpose110_0_id;
    unsigned char g_0_Transpose110_0_params[] = {1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_4}, {g_0_tensor_5}, (void*)g_0_Transpose110_0_params, 24, "g_0_Transpose110_0", 0 /*graphIndex*/, &g_0_Transpose110_0_id);


    compileTopology("pytorch_transformer", 0);
}
