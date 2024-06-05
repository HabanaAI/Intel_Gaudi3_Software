#include "syn_gaudi_two_run_compare_test.h"

TEST_F_GC(SynGaudiTwoRunCompareTest, tpc_bundle_fused_and_layer_norm_fwd_ASIC_CI)
{
    /*************
     * GEMM42649 node
     * inputs: [tensor_2088[768, 4096](dtype=bf16), tensor_2089[768, 768](dtype=bf16)]
     * output: [tensor_2090[768, 4096](dtype=bf16)]
     *************/

    // create tensor_2088 tensor
    unsigned tensor_2088_sizes[] = {768,4096};
    unsigned tensor_2088 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2088",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2088_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2089 tensor
    unsigned tensor_2089_sizes[] = {768,768};
    unsigned tensor_2089 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2089",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2089_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2090 tensor
    unsigned tensor_2090_sizes[] = {768,4096};
    unsigned tensor_2090 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2090",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2090_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char GEMM42649_params[] = {0,0};
    addNodeToGraph("gemm", {tensor_2088, tensor_2089}, {tensor_2090}, (void*)GEMM42649_params, 2, "GEMM42649");

    /*************
     * Reshape42650 node
     * inputs: [tensor_2091[768](dtype=bf16)]
     * output: [tensor_2093[768, 1](dtype=bf16)]
     *************/

    // create tensor_2091 tensor
    unsigned tensor_2091_sizes[] = {768};
    unsigned tensor_2091 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2091",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2091_sizes,
                                        1,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2093 tensor
    unsigned tensor_2093_sizes[] = {768,1};
    unsigned tensor_2093 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2093",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2093_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_2091}, {tensor_2093}, nullptr, 0, "Reshape42650");

    /*************
     * TPC42651 node
     * inputs: [tensor_2090[768, 4096](dtype=bf16), tensor_2093[768, 1](dtype=bf16)]
     * output: [tensor_2092[768, 4096](dtype=bf16)]
     *************/

    // create tensor_2092 tensor
    unsigned tensor_2092_sizes[] = {768,4096};
    unsigned tensor_2092 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2092",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2092_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_bf16", {tensor_2090, tensor_2093}, {tensor_2092}, nullptr, 0, "TPC42651");

    /*************
     * TPC42652 node
     * inputs: [tensor_2092[768, 4096](dtype=bf16)]
     * output: [tensor_2094[768, 4096](dtype=bf16), tensor_2095[768, 4096](dtype=int8)]
     *************/

    // create tensor_2094 tensor
    unsigned tensor_2094_sizes[] = {768,4096};
    unsigned tensor_2094 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2094",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2094_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2095 tensor
    unsigned tensor_2095_sizes[] = {768,4096};
    unsigned tensor_2095 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2095",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_2095_sizes,
                                        2,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC42652_params[] = {205,204,204,61,118,83,159,3};
    addNodeToGraph("dropout_fwd_bf16", {tensor_2092}, {tensor_2094, tensor_2095}, (void*)TPC42652_params, 8, "TPC42652");

    /*************
     * TPC42653 node
     * inputs: [tensor_2056[768, 4096](dtype=bf16), tensor_2094[768, 4096](dtype=bf16)]
     * output: [tensor_2096[768, 4096](dtype=bf16)]
     *************/

    // create tensor_2056 tensor
    unsigned tensor_2056_sizes[] = {768,4096};
    unsigned tensor_2056 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2056",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2056_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2096 tensor
    unsigned tensor_2096_sizes[] = {768,4096};
    unsigned tensor_2096 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2096",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2096_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_bf16", {tensor_2056, tensor_2094}, {tensor_2096}, nullptr, 0, "TPC42653");

    /*************
     * Reshape42654 node
     * inputs: [tensor_2096[768, 4096](dtype=bf16)]
     * output: [tensor_2102[768, 1, 1, 4096](dtype=bf16)]
     *************/

    // create tensor_2102 tensor
    unsigned tensor_2102_sizes[] = {768,1,1,4096};
    unsigned tensor_2102 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2102",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2102_sizes,
                                        4,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_2096}, {tensor_2102}, nullptr, 0, "Reshape42654");

    /*************
     * TPC42658 node
     * inputs: [tensor_2102[768, 1, 1, 4096](dtype=bf16), tensor_2097[768](dtype=float32), tensor_2098[768](dtype=float32)]
     * output: [tensor_2103[768, 1, 1, 4096](dtype=bf16), tensor_2104[1, 1, 1, 4096](dtype=float32), tensor_2105[1, 1, 1, 4096](dtype=float32)]
     *************/

    // create tensor_2097 tensor
    unsigned tensor_2097_sizes[] = {768};
    unsigned tensor_2097 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2097",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2097_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2098 tensor
    unsigned tensor_2098_sizes[] = {768};
    unsigned tensor_2098 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2098",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2098_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2103 tensor
    unsigned tensor_2103_sizes[] = {768,1,1,4096};
    unsigned tensor_2103 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2103",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2103_sizes,
                                        4,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2104 tensor
    unsigned tensor_2104_sizes[] = {1,1,1,4096};
    unsigned tensor_2104 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2104",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2104_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2105 tensor
    unsigned tensor_2105_sizes[] = {1,1,1,4096};
    unsigned tensor_2105 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2105",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2105_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC42658_params[] = {1,237,12,181,172,197,39,55};
    addNodeToGraph("layer_norm_fwd_bf16", {tensor_2102, tensor_2097, tensor_2098}, {tensor_2103, tensor_2104, tensor_2105}, (void*)TPC42658_params, 8, "TPC42658");

    /*************
     * Reshape42655 node
     * inputs: [tensor_2103[768, 1, 1, 4096](dtype=bf16)]
     * output: [tensor_2099[768, 4096](dtype=bf16)]
     *************/

    // create tensor_2099 tensor
    unsigned tensor_2099_sizes[] = {768,4096};
    unsigned tensor_2099 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_2099",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2099_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_2103}, {tensor_2099}, nullptr, 0, "Reshape42655");

    /*************
     * Reshape42656 node
     * inputs: [tensor_2104[1, 1, 1, 4096](dtype=float32)]
     * output: [tensor_2100[1, 4096](dtype=float32)]
     *************/

    // create tensor_2100 tensor
    unsigned tensor_2100_sizes[] = {1,4096};
    unsigned tensor_2100 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2100",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2100_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_2104}, {tensor_2100}, nullptr, 0, "Reshape42656");

    /*************
     * Reshape42657 node
     * inputs: [tensor_2105[1, 1, 1, 4096](dtype=float32)]
     * output: [tensor_2101[1, 4096](dtype=float32)]
     *************/

    // create tensor_2101 tensor
    unsigned tensor_2101_sizes[] = {1,4096};
    unsigned tensor_2101 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2101",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2101_sizes,
                                        2,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_2105}, {tensor_2101}, nullptr, 0, "Reshape42657");

    /*************
     * GEMM42659 node
     * inputs: [tensor_2099[768, 4096](dtype=bf16), tensor_2106[3072, 768](dtype=bf16)]
     * output: [tensor_2107[3072, 4096](dtype=bf16)]
     *************/

    // create tensor_2106 tensor
    unsigned tensor_2106_sizes[] = {3072,768};
    unsigned tensor_2106 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2106",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2106_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_2107 tensor
    unsigned tensor_2107_sizes[] = {3072,4096};
    unsigned tensor_2107 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_2107",
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr, // initializer
                                        tensor_2107_sizes,
                                        2,
                                        syn_type_bf16,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char GEMM42659_params[] = {0,0};
    addNodeToGraph("gemm", {tensor_2099, tensor_2106}, {tensor_2107}, (void*)GEMM42659_params, 2, "GEMM42659");

    addConfigurationToRun(FIRST_RUN, "ENABLE_TPC_BUNDLES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_TPC_BUNDLES", "1");

    compareRunsResults({tensor_2107});
}
