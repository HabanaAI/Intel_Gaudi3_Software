#include "gc_gaudi_test_infra.h"
//A reproducer for [SW-97959] Error on CD slicing solver when CD is big and there are 2 batch dimensions.
TEST_F_GC(SynGaudiTestInfra, big_cd_bgemm)
{
    // Graph #0

    /*************
     * g_0_gradient_lm_head_batch_gemm_2997_0 node
     * inputs:
     *     g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose[250012, 128, 2, 4] (dtype=float32)
     *     g_0_tensor_4607[1024, 250012] (dtype=float32)
     * outputs:
     *     g_0_tensor_4608[1024, 128, 2, 4] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose tensor
    unsigned g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose_max_sizes[] = {250012,128,2,4};
    unsigned g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose_min_sizes[] = {250012,128,2,4};
    unsigned g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose = createTensors(1,
                                                                                   INPUT_TENSOR,
                                                                                   true,
                                                                                   "g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose",
                                                                                   MEM_INIT_ALL_ZERO,
                                                                                   nullptr,
                                                                                   g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose_max_sizes,
                                                                                   4,
                                                                                   syn_type_single,
                                                                                   nullptr,
                                                                                   0,
                                                                                   0,
                                                                                   nullptr,
                                                                                   false,
                                                                                   g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose_min_sizes,
                                                                                   synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4607 tensor
    unsigned g_0_tensor_4607_max_sizes[] = {1024,250012};
    unsigned g_0_tensor_4607_min_sizes[] = {1024,250012};
    unsigned g_0_tensor_4607 = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "g_0_tensor_4607",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         g_0_tensor_4607_max_sizes,
                                         2,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_4607_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_4608 tensor
    unsigned g_0_tensor_4608_max_sizes[] = {1024,128,2,4};
    unsigned g_0_tensor_4608_min_sizes[] = {1024,128,2,4};
    unsigned g_0_tensor_4608 = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "g_0_tensor_4608",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         g_0_tensor_4608_max_sizes,
                                         4,
                                         syn_type_single,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         g_0_tensor_4608_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId g_0_gradient_lm_head_batch_gemm_2997_0_id;
    unsigned char g_0_gradient_lm_head_batch_gemm_2997_0_params[] = {0,0};
    addNodeToGraph("batch_gemm", {g_0_tensor_4606_id_15188_gradient_lm_head_aten__transpose, g_0_tensor_4607}, {g_0_tensor_4608}, (void*)g_0_gradient_lm_head_batch_gemm_2997_0_params, 2, "g_0_gradient_lm_head_batch_gemm_2997_0", 0 /*graphIndex*/, &g_0_gradient_lm_head_batch_gemm_2997_0_id);


    compileTopology("big_cd_bgemm", 0);
}
