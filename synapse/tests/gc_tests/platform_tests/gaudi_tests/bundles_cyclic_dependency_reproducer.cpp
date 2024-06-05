#include "gc_gaudi_test_infra.h"

TEST_F_GC(SynGaudiTestInfra, bundles_cyclic_dependency_reproducer)
{
    /*************
     * TPC3632 node
     * inputs: [tensor_4284[768, 4096](dtype=float32)]
     * output: [tensor_4292[768, 4096](dtype=bf16)]
     *************/

    // create tensor_4284 tensor
    unsigned tensor_4284_sizes[] = {768,4096};
    unsigned tensor_4284 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4284",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4284_sizes,
                                        2,
                                        syn_type_single)[0];

    // create tensor_4292 tensor
    unsigned tensor_4292_sizes[] = {768,4096};
    unsigned tensor_4292 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4292",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4292_sizes,
                                        2,
                                        syn_type_bf16)[0];
    addNodeToGraph("cast_f32_to_bf16", {tensor_4284}, {tensor_4292}, nullptr, 0, "TPC3632");

    /*************
     * GEMM3633 node
     * inputs: [tensor_4292[768, 4096](dtype=bf16), tensor_92[768, 768](dtype=bf16)]
     * output: [tensor_4293[768, 4096](dtype=bf16)]
     *************/

    // create tensor_92 tensor
    unsigned tensor_92_sizes[] = {768,768};
    unsigned tensor_92 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_92",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_92_sizes,
                                        2,
                                        syn_type_bf16)[0];

    // create tensor_4293 tensor
    unsigned tensor_4293_sizes[] = {768,4096};
    unsigned tensor_4293 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4293",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4293_sizes,
                                        2,
                                        syn_type_bf16)[0];
    unsigned char GEMM3633_params[] = {0,1};
    addNodeToGraph("gemm", {tensor_4292, tensor_92}, {tensor_4293}, (void*)GEMM3633_params, 2, "GEMM3633");

    /*************
     * GEMM3634 node
     * inputs: [tensor_1705[768, 4096](dtype=bf16), tensor_4292[768, 4096](dtype=bf16)]
     * output: [tensor_4294[768, 768](dtype=bf16)]
     *************/

    // create tensor_1705 tensor
    unsigned tensor_1705_sizes[] = {768,4096};
    unsigned tensor_1705 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_1705",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_1705_sizes,
                                        2,
                                        syn_type_bf16)[0];

    // create tensor_4294 tensor
    unsigned tensor_4294_sizes[] = {768,768};
    unsigned tensor_4294 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4294",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4294_sizes,
                                        2,
                                        syn_type_bf16)[0];
    unsigned char GEMM3634_params[] = {1,0};
    addNodeToGraph("gemm", {tensor_1705, tensor_4292}, {tensor_4294}, (void*)GEMM3634_params, 2, "GEMM3634");

    /*************
     * TPC3635 node
     * inputs: [tensor_4294[768, 768](dtype=bf16)]
     * output: [tensor_4295[768, 768](dtype=float32)]
     *************/

    // create tensor_4295 tensor
    unsigned tensor_4295_sizes[] = {768,768};
    unsigned tensor_4295 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4295",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4295_sizes,
                                        2,
                                        syn_type_single)[0];
    addNodeToGraph("cast_bf16_to_f32", {tensor_4294}, {tensor_4295}, nullptr, 0, "TPC3635");

    /*************
     * TPC3664 node
     * inputs: [tensor_4316[768, 4096](dtype=float32)]
     * output: [tensor_4324[768, 4096](dtype=bf16)]
     *************/

    // create tensor_4316 tensor
    unsigned tensor_4316_sizes[] = {768,4096};
    unsigned tensor_4316 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4316",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4316_sizes,
                                        2,
                                        syn_type_single)[0];

    // create tensor_4324 tensor
    unsigned tensor_4324_sizes[] = {768,4096};
    unsigned tensor_4324 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4324",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4324_sizes,
                                        2,
                                        syn_type_bf16)[0];
    addNodeToGraph("cast_f32_to_bf16", {tensor_4316}, {tensor_4324}, nullptr, 0, "TPC3664");

    /*************
     * GEMM3665 node
     * inputs: [tensor_1705[768, 4096](dtype=bf16), tensor_4324[768, 4096](dtype=bf16)]
     * output: [tensor_4325[768, 768](dtype=bf16)]
     *************/

    // create tensor_4325 tensor
    unsigned tensor_4325_sizes[] = {768,768};
    unsigned tensor_4325 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4325",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4325_sizes,
                                        2,
                                        syn_type_bf16)[0];
    unsigned char GEMM3665_params[] = {1,0};
    addNodeToGraph("gemm", {tensor_1705, tensor_4324}, {tensor_4325}, (void*)GEMM3665_params, 2, "GEMM3665");

    /*************
     * TPC3666 node
     * inputs: [tensor_4325[768, 768](dtype=bf16)]
     * output: [tensor_4326[768, 768](dtype=float32)]
     *************/

    // create tensor_4326 tensor
    unsigned tensor_4326_sizes[] = {768,768};
    unsigned tensor_4326 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4326",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4326_sizes,
                                        2,
                                        syn_type_single)[0];
    addNodeToGraph("cast_bf16_to_f32", {tensor_4325}, {tensor_4326}, nullptr, 0, "TPC3666");

    /*************
     * GEMM3671 node
     * inputs: [tensor_4324[768, 4096](dtype=bf16), tensor_64[768, 768](dtype=bf16)]
     * output: [tensor_4331[768, 4096](dtype=bf16)]
     *************/

    // create tensor_64 tensor
    unsigned tensor_64_sizes[] = {768,768};
    unsigned tensor_64 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_64",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_64_sizes,
                                        2,
                                        syn_type_bf16)[0];

    // create tensor_4331 tensor
    unsigned tensor_4331_sizes[] = {768,4096};
    unsigned tensor_4331 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4331",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4331_sizes,
                                        2,
                                        syn_type_bf16)[0];
    unsigned char GEMM3671_params[] = {0,1};
    addNodeToGraph("gemm", {tensor_4324, tensor_64}, {tensor_4331}, (void*)GEMM3671_params, 2, "GEMM3671");

    /*************
     * GEMM3687 node
     * inputs: [tensor_1705[768, 4096](dtype=bf16), tensor_4346[768, 4096](dtype=bf16)]
     * output: [tensor_4347[768, 768](dtype=bf16)]
     *************/

    // create tensor_4346 tensor
    unsigned tensor_4346_sizes[] = {768,4096};
    unsigned tensor_4346 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4346",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4346_sizes,
                                        2,
                                        syn_type_bf16)[0];

    // create tensor_4347 tensor
    unsigned tensor_4347_sizes[] = {768,768};
    unsigned tensor_4347 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4347",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4347_sizes,
                                        2,
                                        syn_type_bf16)[0];
    unsigned char GEMM3687_params[] = {1,0};
    addNodeToGraph("gemm", {tensor_1705, tensor_4346}, {tensor_4347}, (void*)GEMM3687_params, 2, "GEMM3687");

    /*************
     * TPC3688 node
     * inputs: [tensor_4347[768, 768](dtype=bf16)]
     * output: [tensor_4348[768, 768](dtype=float32)]
     *************/

    // create tensor_4348 tensor
    unsigned tensor_4348_sizes[] = {768,768};
    unsigned tensor_4348 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4348",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4348_sizes,
                                        2,
                                        syn_type_single)[0];
    addNodeToGraph("cast_bf16_to_f32", {tensor_4347}, {tensor_4348}, nullptr, 0, "TPC3688");

    /*************
     * GEMM3693 node
     * inputs: [tensor_4346[768, 4096](dtype=bf16), tensor_484[768, 768](dtype=bf16)]
     * output: [tensor_4353[768, 4096](dtype=bf16)]
     *************/

    // create tensor_484 tensor
    unsigned tensor_484_sizes[] = {768,768};
    unsigned tensor_484 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_484",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_484_sizes,
                                        2,
                                        syn_type_bf16)[0];

    // create tensor_4353 tensor
    unsigned tensor_4353_sizes[] = {768,4096};
    unsigned tensor_4353 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_4353",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4353_sizes,
                                        2,
                                        syn_type_bf16)[0];
    unsigned char GEMM3693_params[] = {0,1};
    addNodeToGraph("gemm", {tensor_4346, tensor_484}, {tensor_4353}, (void*)GEMM3693_params, 2, "GEMM3693");

    /*************
     * TPC3702 node
     * inputs: [tensor_4245[768, 4096](dtype=bf16), tensor_4353[768, 4096](dtype=bf16)]
     * output: [tensor_4363[768, 4096](dtype=bf16)]
     *************/

    // create tensor_4245 tensor
    unsigned tensor_4245_sizes[] = {768,4096};
    unsigned tensor_4245 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4245",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4245_sizes,
                                        2,
                                        syn_type_bf16)[0];

    // create tensor_4363 tensor
    unsigned tensor_4363_sizes[] = {768,4096};
    unsigned tensor_4363 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4363",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4363_sizes,
                                        2,
                                        syn_type_bf16)[0];
    addNodeToGraph("add_fwd_bf16", {tensor_4245, tensor_4353}, {tensor_4363}, nullptr, 0, "TPC3702");

    /*************
     * TPC3703 node
     * inputs: [tensor_4331[768, 4096](dtype=bf16), tensor_4293[768, 4096](dtype=bf16)]
     * output: [tensor_4364[768, 4096](dtype=bf16)]
     *************/

    // create tensor_4364 tensor
    unsigned tensor_4364_sizes[] = {768,4096};
    unsigned tensor_4364 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_4364",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_4364_sizes,
                                        2,
                                        syn_type_bf16)[0];
    addNodeToGraph("add_fwd_bf16", {tensor_4331, tensor_4293}, {tensor_4364}, nullptr, 0, "TPC3703");


    compileTopology("bundles_cyclic_dependency_reproducer");
}
