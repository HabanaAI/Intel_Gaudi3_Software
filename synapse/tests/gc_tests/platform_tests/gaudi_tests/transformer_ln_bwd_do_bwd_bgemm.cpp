#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
TEST_F_GC(SynTrainingTestInfra, transformer_ln_bwd_do_bwd_bgemm)
{
    // Graph #0

    /*************
     * n72205__aten_native_layer_norm_backward_layer_norm_bwd_bf16_0 node
     * inputs:
     *     tensor_1456[1024, 3936, 1, 1] (dtype=bf16)
     *     tensor_1457[1024, 3936, 1, 1] (dtype=bf16)
     *     tensor_1459[1, 3936, 1, 1] (dtype=f32)
     *     tensor_1460[1, 3936, 1, 1] (dtype=f32)
     *     tensor_1458[1024] (dtype=bf16)
     * outputs:
     *     tensor_1461[1024, 3936, 1, 1] (dtype=f32)
     *     tensor_1462[1024] (dtype=bf16)
     *     tensor_1463[1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1456 tensor
    unsigned tensor_1456_max_sizes[] = {1024, 3936, 1, 1};
    unsigned tensor_1456_min_sizes[] = {1024, 3936, 1, 1};
    unsigned tensor_1456             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1456",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1456_max_sizes,
                                         4,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1456_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1457 tensor
    unsigned tensor_1457_max_sizes[] = {1024, 3936, 1, 1};
    unsigned tensor_1457_min_sizes[] = {1024, 3936, 1, 1};
    unsigned tensor_1457             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1457",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1457_max_sizes,
                                         4,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1457_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1459 tensor
    unsigned tensor_1459_max_sizes[] = {1, 3936, 1, 1};
    unsigned tensor_1459_min_sizes[] = {1, 3936, 1, 1};
    unsigned tensor_1459             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1459",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1459_max_sizes,
                                         4,
                                         syn_type_float,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1459_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1460 tensor
    unsigned tensor_1460_max_sizes[] = {1, 3936, 1, 1};
    unsigned tensor_1460_min_sizes[] = {1, 3936, 1, 1};
    unsigned tensor_1460             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1460",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1460_max_sizes,
                                         4,
                                         syn_type_float,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1460_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1458 tensor
    unsigned tensor_1458_max_sizes[] = {1024};
    unsigned tensor_1458_min_sizes[] = {1024};
    unsigned tensor_1458             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1458",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1458_max_sizes,
                                         1,
                                         syn_type_float,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1458_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1461 tensor
    unsigned tensor_1461_max_sizes[] = {1024, 3936, 1, 1};
    unsigned tensor_1461_min_sizes[] = {1024, 3936, 1, 1};
    unsigned tensor_1461             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "tensor_1461",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1461_max_sizes,
                                         4,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1461_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1462 tensor
    unsigned tensor_1462_max_sizes[] = {1024};
    unsigned tensor_1462_min_sizes[] = {1024};
    unsigned tensor_1462             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "tensor_1462",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1462_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1462_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1463 tensor
    unsigned      tensor_1463_max_sizes[] = {1024};
    unsigned      tensor_1463_min_sizes[] = {1024};
    unsigned      tensor_1463             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "tensor_1463",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1463_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1463_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     n72205__aten_native_layer_norm_backward_layer_norm_bwd_bf16_0_id;
    unsigned char n72205__aten_native_layer_norm_backward_layer_norm_bwd_bf16_0_params[] =
        {0, 171, 55, 66, 235, 127, 0, 0};
    addNodeToGraph("layer_norm_bwd_bf16",
                   {tensor_1456, tensor_1457, tensor_1459, tensor_1460, tensor_1458},
                   {tensor_1461, tensor_1462, tensor_1463},
                   (void*)n72205__aten_native_layer_norm_backward_layer_norm_bwd_bf16_0_params,
                   8,
                   "n72205__aten_native_layer_norm_backward_layer_norm_bwd_bf16_0",
                   0 /*graphIndex*/,
                   &n72205__aten_native_layer_norm_backward_layer_norm_bwd_bf16_0_id);

    /*************
     * n72205__aten_native_layer_norm_backward_reshape_252073_0 node
     * inputs:
     *     tensor_1461[1024, 3936, 1, 1] (dtype=bf16)
     * outputs:
     *     tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0[1024, 96, 41] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0 tensor
    unsigned tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0_max_sizes[] = {1024, 96, 41};
    unsigned tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0_min_sizes[] = {1024, 96, 41};
    unsigned tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0_max_sizes,
                      3,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId n72205__aten_native_layer_norm_backward_reshape_252073_0_id;
    addNodeToGraph("reshape",
                   {tensor_1461},
                   {tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0},
                   nullptr,
                   0,
                   "n72205__aten_native_layer_norm_backward_reshape_252073_0",
                   0 /*graphIndex*/,
                   &n72205__aten_native_layer_norm_backward_reshape_252073_0_id);

    /*************
     * n72205__aten_native_layer_norm_backward_reshape_252074_0 node
     * inputs:
     *     tensor_1463[1024] (dtype=bf16)
     * outputs:
     *     tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1[1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1 tensor
    unsigned tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1_max_sizes[] = {1024};
    unsigned tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1_min_sizes[] = {1024};
    unsigned tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId n72205__aten_native_layer_norm_backward_reshape_252074_0_id;
    addNodeToGraph("reshape",
                   {tensor_1463},
                   {tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1},
                   nullptr,
                   0,
                   "n72205__aten_native_layer_norm_backward_reshape_252074_0",
                   0 /*graphIndex*/,
                   &n72205__aten_native_layer_norm_backward_reshape_252074_0_id);

    /*************
     * n72205__aten_native_layer_norm_backward_reshape_252075_0 node
     * inputs:
     *     tensor_1462[1024] (dtype=bf16)
     * outputs:
     *     tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2[1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2 tensor
    unsigned tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2_max_sizes[] = {1024};
    unsigned tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2_min_sizes[] = {1024};
    unsigned tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2 =
        createTensors(1,
                      OUTPUT_TENSOR,
                      false,
                      "tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2_max_sizes,
                      1,
                      syn_type_bf16,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId n72205__aten_native_layer_norm_backward_reshape_252075_0_id;
    addNodeToGraph("reshape",
                   {tensor_1462},
                   {tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2},
                   nullptr,
                   0,
                   "n72205__aten_native_layer_norm_backward_reshape_252075_0",
                   0 /*graphIndex*/,
                   &n72205__aten_native_layer_norm_backward_reshape_252075_0_id);

    /*************
     * n72211__aten_add__add_fwd_bf16_0 node
     * inputs:
     *     tensor_1467[1024] (dtype=bf16)
     *     tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1[1024] (dtype=bf16)
     * outputs:
     *     tensor_1468[1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1467 tensor
    unsigned tensor_1467_max_sizes[] = {1024};
    unsigned tensor_1467_min_sizes[] = {1024};
    unsigned tensor_1467             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1467",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1467_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1467_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1468 tensor
    unsigned  tensor_1468_max_sizes[] = {1024};
    unsigned  tensor_1468_min_sizes[] = {1024};
    unsigned  tensor_1468             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "tensor_1468",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1468_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1468_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId n72211__aten_add__add_fwd_bf16_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {tensor_1467, tensor_1465_t102881_n72205__aten_native_layer_norm_backward_1},
                   {tensor_1468},
                   nullptr,
                   0,
                   "n72211__aten_add__add_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &n72211__aten_add__add_fwd_bf16_0_id);

    /*************
     * n72214__aten_add__add_fwd_bf16_0 node
     * inputs:
     *     tensor_1469[1024] (dtype=bf16)
     *     tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2[1024] (dtype=bf16)
     * outputs:
     *     tensor_1470[1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1469 tensor
    unsigned tensor_1469_max_sizes[] = {1024};
    unsigned tensor_1469_min_sizes[] = {1024};
    unsigned tensor_1469             = createTensors(1,
                                         INPUT_TENSOR,
                                         true,
                                         "tensor_1469",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1469_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1469_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];

    // create tensor_1470 tensor
    unsigned  tensor_1470_max_sizes[] = {1024};
    unsigned  tensor_1470_min_sizes[] = {1024};
    unsigned  tensor_1470             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "tensor_1470",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1470_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1470_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId n72214__aten_add__add_fwd_bf16_0_id;
    addNodeToGraph("add_fwd_bf16",
                   {tensor_1469, tensor_1466_t102882_n72205__aten_native_layer_norm_backward_2},
                   {tensor_1470},
                   nullptr,
                   0,
                   "n72214__aten_add__add_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &n72214__aten_add__add_fwd_bf16_0_id);

    /*************
     * n72216__hpu_cast_cast_i8_to_bf16_0 node
     * inputs:
     *     tensor_1374_t102692_n72073__hpu__fused_dropout_1[1024, 96, 41] (dtype=int8)
     * outputs:
     *     tensor_1471_t102889_n72216__hpu_cast_0[1024, 96, 41] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1374_t102692_n72073__hpu__fused_dropout_1 tensor
    unsigned tensor_1374_t102692_n72073__hpu__fused_dropout_1_max_sizes[] = {1024, 96, 41};
    unsigned tensor_1374_t102692_n72073__hpu__fused_dropout_1_min_sizes[] = {1024, 96, 41};
    unsigned tensor_1374_t102692_n72073__hpu__fused_dropout_1 =
        createTensors(1,
                      INPUT_TENSOR,
                      true,
                      "tensor_1374_t102692_n72073__hpu__fused_dropout_1",
                      MEM_INIT_ALL_ZERO,
                      nullptr,
                      tensor_1374_t102692_n72073__hpu__fused_dropout_1_max_sizes,
                      3,
                      syn_type_int8,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      tensor_1374_t102692_n72073__hpu__fused_dropout_1_min_sizes,
                      synTensorType::DATA_TENSOR)[0];

    // create tensor_1471_t102889_n72216__hpu_cast_0 tensor
    unsigned      tensor_1471_t102889_n72216__hpu_cast_0_max_sizes[] = {1024, 96, 41};
    unsigned      tensor_1471_t102889_n72216__hpu_cast_0_min_sizes[] = {1024, 96, 41};
    unsigned      tensor_1471_t102889_n72216__hpu_cast_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "tensor_1471_t102889_n72216__hpu_cast_0",
                                                                    MEM_INIT_ALL_ZERO,
                                                                    nullptr,
                                                                    tensor_1471_t102889_n72216__hpu_cast_0_max_sizes,
                                                                    3,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    tensor_1471_t102889_n72216__hpu_cast_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId     n72216__hpu_cast_cast_i8_to_bf16_0_id;
    unsigned char n72216__hpu_cast_cast_i8_to_bf16_0_params[] = {4, 0, 0, 0};
    addNodeToGraph("cast_i8_to_bf16",
                   {tensor_1374_t102692_n72073__hpu__fused_dropout_1},
                   {tensor_1471_t102889_n72216__hpu_cast_0},
                   (void*)n72216__hpu_cast_cast_i8_to_bf16_0_params,
                   4,
                   "n72216__hpu_cast_cast_i8_to_bf16_0",
                   0 /*graphIndex*/,
                   &n72216__hpu_cast_cast_i8_to_bf16_0_id);

    /*************
     * n72217__aten_mul_mult_fwd_bf16_0 node
     * inputs:
     *     tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0[1024, 96, 41] (dtype=bf16)
     *     tensor_1471_t102889_n72216__hpu_cast_0[1024, 96, 41] (dtype=bf16)
     * outputs:
     *     tensor_1472_t102892_n72217__aten_mul_0[1024, 96, 41] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1472_t102892_n72217__aten_mul_0 tensor
    unsigned  tensor_1472_t102892_n72217__aten_mul_0_max_sizes[] = {1024, 96, 41};
    unsigned  tensor_1472_t102892_n72217__aten_mul_0_min_sizes[] = {1024, 96, 41};
    unsigned  tensor_1472_t102892_n72217__aten_mul_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "tensor_1472_t102892_n72217__aten_mul_0",
                                                                    MEM_INIT_ALL_ZERO,
                                                                    nullptr,
                                                                    tensor_1472_t102892_n72217__aten_mul_0_max_sizes,
                                                                    3,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    tensor_1472_t102892_n72217__aten_mul_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId n72217__aten_mul_mult_fwd_bf16_0_id;
    addNodeToGraph(
        "mult_fwd_bf16",
        {tensor_1464_t102880_n72205__aten_native_layer_norm_backward_0, tensor_1471_t102889_n72216__hpu_cast_0},
        {tensor_1472_t102892_n72217__aten_mul_0},
        nullptr,
        0,
        "n72217__aten_mul_mult_fwd_bf16_0",
        0 /*graphIndex*/,
        &n72217__aten_mul_mult_fwd_bf16_0_id);

    /*************
     * n72220__aten_mul_constant_bf16_0 node
     * inputs:
     * outputs:
     *     tensor_1473[1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1473 tensor
    unsigned      tensor_1473_max_sizes[] = {1};
    unsigned      tensor_1473_min_sizes[] = {1};
    unsigned      tensor_1473             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "tensor_1473",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1473_max_sizes,
                                         1,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1473_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     n72220__aten_mul_constant_bf16_0_id;
    unsigned char n72220__aten_mul_constant_bf16_0_params[] = {110, 219, 182, 63};
    addNodeToGraph("constant_bf16",
                   {},
                   {tensor_1473},
                   (void*)n72220__aten_mul_constant_bf16_0_params,
                   4,
                   "n72220__aten_mul_constant_bf16_0",
                   0 /*graphIndex*/,
                   &n72220__aten_mul_constant_bf16_0_id);

    /*************
     * n72220__aten_mul_mult_fwd_bf16_0 node
     * inputs:
     *     tensor_1472_t102892_n72217__aten_mul_0[1024, 96, 41] (dtype=bf16)
     *     tensor_1473[1] (dtype=bf16)
     * outputs:
     *     tensor_1474_t102896_n72220__aten_mul_0[1024, 96, 41] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1474_t102896_n72220__aten_mul_0 tensor
    unsigned  tensor_1474_t102896_n72220__aten_mul_0_max_sizes[] = {1024, 96, 41};
    unsigned  tensor_1474_t102896_n72220__aten_mul_0_min_sizes[] = {1024, 96, 41};
    unsigned  tensor_1474_t102896_n72220__aten_mul_0             = createTensors(1,
                                                                    OUTPUT_TENSOR,
                                                                    false,
                                                                    "tensor_1474_t102896_n72220__aten_mul_0",
                                                                    MEM_INIT_ALL_ZERO,
                                                                    nullptr,
                                                                    tensor_1474_t102896_n72220__aten_mul_0_max_sizes,
                                                                    3,
                                                                    syn_type_bf16,
                                                                    nullptr,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    false,
                                                                    tensor_1474_t102896_n72220__aten_mul_0_min_sizes,
                                                                    synTensorType::DATA_TENSOR)[0];
    synNodeId n72220__aten_mul_mult_fwd_bf16_0_id;
    addNodeToGraph("mult_fwd_bf16",
                   {tensor_1472_t102892_n72217__aten_mul_0, tensor_1473},
                   {tensor_1474_t102896_n72220__aten_mul_0},
                   nullptr,
                   0,
                   "n72220__aten_mul_mult_fwd_bf16_0",
                   0 /*graphIndex*/,
                   &n72220__aten_mul_mult_fwd_bf16_0_id);

    /*************
     * n72235__hpu_matmul_backward_transpose_0 node
     * inputs:
     *     tensor_1_t102678_n72065__aten_t_0[1024, 4096] (dtype=bf16)
     * outputs:
     *     tensor_1475[4096, 1024] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1_t102678_n72065__aten_t_0 tensor
    unsigned tensor_1_t102678_n72065__aten_t_0_max_sizes[] = {1024, 4096};
    unsigned tensor_1_t102678_n72065__aten_t_0_min_sizes[] = {1024, 4096};
    unsigned tensor_1_t102678_n72065__aten_t_0             = createTensors(1,
                                                               INPUT_TENSOR,
                                                               true,
                                                               "tensor_1_t102678_n72065__aten_t_0",
                                                               MEM_INIT_ALL_ZERO,
                                                               nullptr,
                                                               tensor_1_t102678_n72065__aten_t_0_max_sizes,
                                                               2,
                                                               syn_type_bf16,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               tensor_1_t102678_n72065__aten_t_0_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];

    // create tensor_1475 tensor
    unsigned      tensor_1475_max_sizes[] = {4096, 1024};
    unsigned      tensor_1475_min_sizes[] = {4096, 1024};
    unsigned      tensor_1475             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false,
                                         "tensor_1475",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1475_max_sizes,
                                         2,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1475_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     n72235__hpu_matmul_backward_transpose_0_id;
    unsigned char n72235__hpu_matmul_backward_transpose_0_params[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0};
    addNodeToGraph("transpose",
                   {tensor_1_t102678_n72065__aten_t_0},
                   {tensor_1475},
                   (void*)n72235__hpu_matmul_backward_transpose_0_params,
                   24,
                   "n72235__hpu_matmul_backward_transpose_0",
                   0 /*graphIndex*/,
                   &n72235__hpu_matmul_backward_transpose_0_id);

    /*************
     * n72235__hpu_matmul_backward_batch_gemm_0 node
     * inputs:
     *     tensor_1474_t102896_n72220__aten_mul_0[1024, 96, 41] (dtype=bf16)
     *     tensor_1475[4096, 1024] (dtype=bf16)
     * outputs:
     *     tensor_1476[4096, 96, 41] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create tensor_1476 tensor
    unsigned      tensor_1476_max_sizes[] = {4096, 96, 41};
    unsigned      tensor_1476_min_sizes[] = {4096, 96, 41};
    unsigned      tensor_1476             = createTensors(1,
                                         OUTPUT_TENSOR,
                                         true,
                                         "tensor_1476",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         tensor_1476_max_sizes,
                                         3,
                                         syn_type_bf16,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         false,
                                         tensor_1476_min_sizes,
                                         synTensorType::DATA_TENSOR)[0];
    synNodeId     n72235__hpu_matmul_backward_batch_gemm_0_id;
    unsigned char n72235__hpu_matmul_backward_batch_gemm_0_params[] = {0, 0};
    addNodeToGraph("batch_gemm",
                   {tensor_1474_t102896_n72220__aten_mul_0, tensor_1475},
                   {tensor_1476},
                   (void*)n72235__hpu_matmul_backward_batch_gemm_0_params,
                   2,
                   "n72235__hpu_matmul_backward_batch_gemm_0",
                   0 /*graphIndex*/,
                   &n72235__hpu_matmul_backward_batch_gemm_0_id);

    compileTopology("transformer_ln_bwd_do_bwd_bgemm", 0);
}
