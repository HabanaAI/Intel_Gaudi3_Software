#include "gc_gaudi_test_infra.h"
#include <math.h>
#include "scoped_configuration_change.h"
TEST_F_GC(SynGaudiTestInfra, transpose_bgemm)
{
    // A reproducer for [SW-94068]- When MME Flattens a strided tensor which is input to BGEMM it caused incorrect results.
    // Transpose on FCD is used to create strides. SRAM is disabled so that no DMA will make the tensor dense.
    ScopedConfigurationChange sramSlicerCapacity("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    // Graph #0

    /*************
     * g_0__transpose_33_0 node
     * inputs:
     *     g_0_tensor_44[1024, 66, 4] (dtype=float32)
     * outputs:
     *     g_0_tensor_52_id_4495_aten__transpose_1[1024, 4, 66] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_44 tensor
    unsigned g_0_tensor_44_max_sizes[] = {1024,66,4};
    unsigned g_0_tensor_44_min_sizes[] = {1024,66,4};
    unsigned g_0_tensor_44 = createTensors(1,
                                       INPUT_TENSOR,
                                       true,
                                       "g_0_tensor_44",
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       g_0_tensor_44_max_sizes,
                                       3,
                                       syn_type_single,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       false,
                                       g_0_tensor_44_min_sizes,
                                       synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_52_id_4495_aten__transpose_1 tensor
    unsigned g_0_tensor_52_id_4495_aten__transpose_1_max_sizes[] = {1024,4,66};
    unsigned g_0_tensor_52_id_4495_aten__transpose_1_min_sizes[] = {1024,4,66};
    unsigned g_0_tensor_52_id_4495_aten__transpose_1 = createTensors(1,
                                                                 OUTPUT_TENSOR,
                                                                 false,
                                                                 "g_0_tensor_52_id_4495_aten__transpose_1",
                                                                 MEM_INIT_ALL_ZERO,
                                                                 nullptr,
                                                                 g_0_tensor_52_id_4495_aten__transpose_1_max_sizes,
                                                                 3,
                                                                 syn_type_single,
                                                                 nullptr,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 false,
                                                                 g_0_tensor_52_id_4495_aten__transpose_1_min_sizes,
                                                                 synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__transpose_33_0_id;
    unsigned char g_0__transpose_33_0_params[] = {0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0};
    addNodeToGraph("transpose", {g_0_tensor_44}, {g_0_tensor_52_id_4495_aten__transpose_1}, (void*)g_0__transpose_33_0_params, 24, "g_0__transpose_33_0", 0 /*graphIndex*/, &g_0__transpose_33_0_id);

    /*************
     * g_0__batch_gemm_34_0 node
     * inputs:
     *     g_0_tensor_52_id_4495_aten__transpose_1[1024, 4, 66] (dtype=float32)
     *     g_0_tensor_51_id_4518_aten__t[1024, 1024] (dtype=float32)
     * outputs:
     *     g_0_tensor_53_id_4521_aten__matmul[1024, 4, 66] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create g_0_tensor_51_id_4518_aten__t tensor
    unsigned g_0_tensor_51_id_4518_aten__t_max_sizes[] = {1024,1024};
    unsigned g_0_tensor_51_id_4518_aten__t_min_sizes[] = {1024,1024};
    unsigned g_0_tensor_51_id_4518_aten__t = createTensors(1,
                                                       INPUT_TENSOR,
                                                       true,
                                                       "g_0_tensor_51_id_4518_aten__t",
                                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       g_0_tensor_51_id_4518_aten__t_max_sizes,
                                                       2,
                                                       syn_type_single,
                                                       nullptr,
                                                       0,
                                                       0,
                                                       nullptr,
                                                       false,
                                                       g_0_tensor_51_id_4518_aten__t_min_sizes,
                                                       synTensorType::DATA_TENSOR)[0];





    // create g_0_tensor_53_id_4521_aten__matmul tensor
    unsigned g_0_tensor_53_id_4521_aten__matmul_max_sizes[] = {1024,4,66};
    unsigned g_0_tensor_53_id_4521_aten__matmul_min_sizes[] = {1024,4,66};
    unsigned g_0_tensor_53_id_4521_aten__matmul = createTensors(1,
                                                            OUTPUT_TENSOR,
                                                            true,
                                                            "g_0_tensor_53_id_4521_aten__matmul",
                                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                                 nullptr,
                                                            g_0_tensor_53_id_4521_aten__matmul_max_sizes,
                                                            3,
                                                            syn_type_single,
                                                            nullptr,
                                                            0,
                                                            0,
                                                            nullptr,
                                                            false,
                                                            g_0_tensor_53_id_4521_aten__matmul_min_sizes,
                                                            synTensorType::DATA_TENSOR)[0];
    synNodeId g_0__batch_gemm_34_0_id;
    unsigned char g_0__batch_gemm_34_0_params[] = {0,0};
    addNodeToGraph("batch_gemm", {g_0_tensor_52_id_4495_aten__transpose_1, g_0_tensor_51_id_4518_aten__t}, {g_0_tensor_53_id_4521_aten__matmul}, (void*)g_0__batch_gemm_34_0_params, 2, "g_0__batch_gemm_34_0", 0 /*graphIndex*/, &g_0__batch_gemm_34_0_id);


    compileTopology("transpose_bgemm", 0);
    runTopology(0, true);


    float* out = castHostOutBuffer<float>(g_0_tensor_53_id_4521_aten__matmul);

    for (unsigned i=0; i < 1024*4*66; i++)
    {
        ASSERT_EQ(out[i], 0) << "Non zero value at index " << i;
    }
}
