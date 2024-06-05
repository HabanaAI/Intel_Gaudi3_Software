#include "gc_gaudi_test_infra.h"

class SynTrainingTFVidarTest : public SynTrainingTestInfra
{
};

// Temporariy disabling test on Gaudi2 due to SW-94033
TEST_F_GC(SynTrainingTFVidarTest, lord_vidar_ASIC_CI)
{
    /*************
     * TPC97891 node
     * inputs: [tensor_23415[216, 144, 3, 3](dtype=float32), tensor_23415[216, 144, 3, 3](dtype=float32)]
     * output: [tensor_23416[216, 144, 3, 3](dtype=float32)]
     *************/

    // create tensor_23415 tensor
    unsigned tensor_23415_sizes[] = {216,144,3,3};
    unsigned tensor_23415 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23415",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23415_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23416 tensor
    unsigned tensor_23416_sizes[] = {216,144,3,3};
    unsigned tensor_23416 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23416",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23416_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23415, tensor_23415}, {tensor_23416}, nullptr, 0, "TPC97891");

    /*************
     * TPC97892 node
     * inputs: [tensor_23417[24, 24, 3, 3](dtype=float32), tensor_23417[24, 24, 3, 3](dtype=float32)]
     * output: [tensor_23418[24, 24, 3, 3](dtype=float32)]
     *************/

    // create tensor_23417 tensor
    unsigned tensor_23417_sizes[] = {24,24,3,3};
    unsigned tensor_23417 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23417",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23417_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23418 tensor
    unsigned tensor_23418_sizes[] = {24,24,3,3};
    unsigned tensor_23418 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23418",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23418_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23417, tensor_23417}, {tensor_23418}, nullptr, 0, "TPC97892");

    /*************
     * TPC97893 node
     * inputs: [tensor_23419[8, 1, 3, 3](dtype=float32), tensor_23419[8, 1, 3, 3](dtype=float32)]
     * output: [tensor_23420[8, 1, 3, 3](dtype=float32)]
     *************/

    // create tensor_23419 tensor
    unsigned tensor_23419_sizes[] = {8,1,3,3};
    unsigned tensor_23419 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23419",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23419_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23420 tensor
    unsigned tensor_23420_sizes[] = {8,1,3,3};
    unsigned tensor_23420 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23420",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23420_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23419, tensor_23419}, {tensor_23420}, nullptr, 0, "TPC97893");

    /*************
     * TPC97894 node
     * inputs: [tensor_23421[144, 432, 3, 3](dtype=float32), tensor_23421[144, 432, 3, 3](dtype=float32)]
     * output: [tensor_23422[144, 432, 3, 3](dtype=float32)]
     *************/

    // create tensor_23421 tensor
    unsigned tensor_23421_sizes[] = {144,432,3,3};
    unsigned tensor_23421 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23421",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23421_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23422 tensor
    unsigned tensor_23422_sizes[] = {144,432,3,3};
    unsigned tensor_23422 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23422",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23422_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23421, tensor_23421}, {tensor_23422}, nullptr, 0, "TPC97894");

    /*************
     * TPC97895 node
     * inputs: [tensor_23423[96, 96, 3, 3](dtype=float32), tensor_23423[96, 96, 3, 3](dtype=float32)]
     * output: [tensor_23424[96, 96, 3, 3](dtype=float32)]
     *************/

    // create tensor_23423 tensor
    unsigned tensor_23423_sizes[] = {96,96,3,3};
    unsigned tensor_23423 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23423",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23423_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23424 tensor
    unsigned tensor_23424_sizes[] = {96,96,3,3};
    unsigned tensor_23424 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23424",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23424_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23423, tensor_23423}, {tensor_23424}, nullptr, 0, "TPC97895");

    /*************
     * TPC97896 node
     * inputs: [tensor_23425[8, 8, 3, 3](dtype=float32), tensor_23425[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23426[8, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23425 tensor
    unsigned tensor_23425_sizes[] = {8,8,3,3};
    unsigned tensor_23425 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23425",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23425_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23426 tensor
    unsigned tensor_23426_sizes[] = {8,8,3,3};
    unsigned tensor_23426 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23426",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23426_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23425, tensor_23425}, {tensor_23426}, nullptr, 0, "TPC97896");

    /*************
     * TPC97897 node
     * inputs: [tensor_23427[144, 144, 3, 3](dtype=float32), tensor_23427[144, 144, 3, 3](dtype=float32)]
     * output: [tensor_23428[144, 144, 3, 3](dtype=float32)]
     *************/

    // create tensor_23427 tensor
    unsigned tensor_23427_sizes[] = {144,144,3,3};
    unsigned tensor_23427 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23427",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23427_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23428 tensor
    unsigned tensor_23428_sizes[] = {144,144,3,3};
    unsigned tensor_23428 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23428",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23428_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23427, tensor_23427}, {tensor_23428}, nullptr, 0, "TPC97897");

    /*************
     * TPC97898 node
     * inputs: [tensor_23429[64, 64, 3, 3](dtype=float32), tensor_23429[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23430[64, 64, 3, 3](dtype=float32)]
     *************/

    // create tensor_23429 tensor
    unsigned tensor_23429_sizes[] = {64,64,3,3};
    unsigned tensor_23429 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23429",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23429_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23430 tensor
    unsigned tensor_23430_sizes[] = {64,64,3,3};
    unsigned tensor_23430 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23430",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23430_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23429, tensor_23429}, {tensor_23430}, nullptr, 0, "TPC97898");

    /*************
     * TPC97899 node
     * inputs: [tensor_23431[8, 8, 3, 3](dtype=float32), tensor_23431[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23432[8, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23431 tensor
    unsigned tensor_23431_sizes[] = {8,8,3,3};
    unsigned tensor_23431 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23431",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23431_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23432 tensor
    unsigned tensor_23432_sizes[] = {8,8,3,3};
    unsigned tensor_23432 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23432",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23432_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23431, tensor_23431}, {tensor_23432}, nullptr, 0, "TPC97899");

    /*************
     * TPC97900 node
     * inputs: [tensor_23433[16, 8, 3, 3](dtype=float32), tensor_23433[16, 8, 3, 3](dtype=float32)]
     * output: [tensor_23434[16, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23433 tensor
    unsigned tensor_23433_sizes[] = {16,8,3,3};
    unsigned tensor_23433 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23433",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23433_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23434 tensor
    unsigned tensor_23434_sizes[] = {16,8,3,3};
    unsigned tensor_23434 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23434",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23434_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23433, tensor_23433}, {tensor_23434}, nullptr, 0, "TPC97900");

    /*************
     * TPC97901 node
     * inputs: [tensor_23435[8, 24, 3, 3](dtype=float32), tensor_23435[8, 24, 3, 3](dtype=float32)]
     * output: [tensor_23436[8, 24, 3, 3](dtype=float32)]
     *************/

    // create tensor_23435 tensor
    unsigned tensor_23435_sizes[] = {8,24,3,3};
    unsigned tensor_23435 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23435",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23435_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23436 tensor
    unsigned tensor_23436_sizes[] = {8,24,3,3};
    unsigned tensor_23436 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23436",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23436_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23435, tensor_23435}, {tensor_23436}, nullptr, 0, "TPC97901");

    /*************
     * TPC97902 node
     * inputs: [tensor_23437[96, 96, 3, 3](dtype=float32), tensor_23437[96, 96, 3, 3](dtype=float32)]
     * output: [tensor_23438[96, 96, 3, 3](dtype=float32)]
     *************/

    // create tensor_23437 tensor
    unsigned tensor_23437_sizes[] = {96,96,3,3};
    unsigned tensor_23437 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23437",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23437_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23438 tensor
    unsigned tensor_23438_sizes[] = {96,96,3,3};
    unsigned tensor_23438 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23438",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23438_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23437, tensor_23437}, {tensor_23438}, nullptr, 0, "TPC97902");

    /*************
     * TPC97903 node
     * inputs: [tensor_23439[96, 64, 3, 3](dtype=float32), tensor_23439[96, 64, 3, 3](dtype=float32)]
     * output: [tensor_23440[96, 64, 3, 3](dtype=float32)]
     *************/

    // create tensor_23439 tensor
    unsigned tensor_23439_sizes[] = {96,64,3,3};
    unsigned tensor_23439 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23439",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23439_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23440 tensor
    unsigned tensor_23440_sizes[] = {96,64,3,3};
    unsigned tensor_23440 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23440",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23440_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23439, tensor_23439}, {tensor_23440}, nullptr, 0, "TPC97903");

    /*************
     * TPC97904 node
     * inputs: [tensor_23441[16, 8, 3, 3](dtype=float32), tensor_23441[16, 8, 3, 3](dtype=float32)]
     * output: [tensor_23442[16, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23441 tensor
    unsigned tensor_23441_sizes[] = {16,8,3,3};
    unsigned tensor_23441 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23441",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23441_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23442 tensor
    unsigned tensor_23442_sizes[] = {16,8,3,3};
    unsigned tensor_23442 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23442",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23442_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23441, tensor_23441}, {tensor_23442}, nullptr, 0, "TPC97904");

    /*************
     * TPC97905 node
     * inputs: [tensor_23443[8, 8, 3, 3](dtype=float32), tensor_23443[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23444[8, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23443 tensor
    unsigned tensor_23443_sizes[] = {8,8,3,3};
    unsigned tensor_23443 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23443",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23443_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23444 tensor
    unsigned tensor_23444_sizes[] = {8,8,3,3};
    unsigned tensor_23444 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23444",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23444_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23443, tensor_23443}, {tensor_23444}, nullptr, 0, "TPC97905");

    /*************
     * TPC97906 node
     * inputs: [tensor_23445[8, 1, 3, 3](dtype=float32), tensor_23445[8, 1, 3, 3](dtype=float32)]
     * output: [tensor_23446[8, 1, 3, 3](dtype=float32)]
     *************/

    // create tensor_23445 tensor
    unsigned tensor_23445_sizes[] = {8,1,3,3};
    unsigned tensor_23445 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23445",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23445_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23446 tensor
    unsigned tensor_23446_sizes[] = {8,1,3,3};
    unsigned tensor_23446 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23446",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23446_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23445, tensor_23445}, {tensor_23446}, nullptr, 0, "TPC97906");

    /*************
     * TPC97907 node
     * inputs: [tensor_23447[8, 8, 3, 3](dtype=float32), tensor_23447[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23448[8, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23447 tensor
    unsigned tensor_23447_sizes[] = {8,8,3,3};
    unsigned tensor_23447 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23447",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23447_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23448 tensor
    unsigned tensor_23448_sizes[] = {8,8,3,3};
    unsigned tensor_23448 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23448",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23448_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23447, tensor_23447}, {tensor_23448}, nullptr, 0, "TPC97907");

    /*************
     * TPC97908 node
     * inputs: [tensor_23449[144, 96, 3, 3](dtype=float32), tensor_23449[144, 96, 3, 3](dtype=float32)]
     * output: [tensor_23450[144, 96, 3, 3](dtype=float32)]
     *************/

    // create tensor_23449 tensor
    unsigned tensor_23449_sizes[] = {144,96,3,3};
    unsigned tensor_23449 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23449",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23449_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23450 tensor
    unsigned tensor_23450_sizes[] = {144,96,3,3};
    unsigned tensor_23450 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23450",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23450_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23449, tensor_23449}, {tensor_23450}, nullptr, 0, "TPC97908");

    /*************
     * TPC97909 node
     * inputs: [tensor_23451[48, 192, 3, 3](dtype=float32), tensor_23451[48, 192, 3, 3](dtype=float32)]
     * output: [tensor_23452[48, 192, 3, 3](dtype=float32)]
     *************/

    // create tensor_23451 tensor
    unsigned tensor_23451_sizes[] = {48,192,3,3};
    unsigned tensor_23451 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23451",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23451_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23452 tensor
    unsigned tensor_23452_sizes[] = {48,192,3,3};
    unsigned tensor_23452 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23452",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23452_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23451, tensor_23451}, {tensor_23452}, nullptr, 0, "TPC97909");

    /*************
     * TPC97910 node
     * inputs: [tensor_23453[16, 16, 3, 3](dtype=float32), tensor_23453[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23454[16, 16, 3, 3](dtype=float32)]
     *************/

    // create tensor_23453 tensor
    unsigned tensor_23453_sizes[] = {16,16,3,3};
    unsigned tensor_23453 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23453",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23453_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23454 tensor
    unsigned tensor_23454_sizes[] = {16,16,3,3};
    unsigned tensor_23454 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23454",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23454_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23453, tensor_23453}, {tensor_23454}, nullptr, 0, "TPC97910");

    /*************
     * TPC97911 node
     * inputs: [tensor_23455[16, 16, 3, 3](dtype=float32), tensor_23455[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23456[16, 16, 3, 3](dtype=float32)]
     *************/

    // create tensor_23455 tensor
    unsigned tensor_23455_sizes[] = {16,16,3,3};
    unsigned tensor_23455 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23455",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23455_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23456 tensor
    unsigned tensor_23456_sizes[] = {16,16,3,3};
    unsigned tensor_23456 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23456",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23456_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23455, tensor_23455}, {tensor_23456}, nullptr, 0, "TPC97911");

    /*************
     * TPC97912 node
     * inputs: [tensor_23457[32, 32, 3, 3](dtype=float32), tensor_23457[32, 32, 3, 3](dtype=float32)]
     * output: [tensor_23458[32, 32, 3, 3](dtype=float32)]
     *************/

    // create tensor_23457 tensor
    unsigned tensor_23457_sizes[] = {32,32,3,3};
    unsigned tensor_23457 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23457",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23457_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23458 tensor
    unsigned tensor_23458_sizes[] = {32,32,3,3};
    unsigned tensor_23458 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23458",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23458_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23457, tensor_23457}, {tensor_23458}, nullptr, 0, "TPC97912");

    /*************
     * TPC97913 node
     * inputs: [tensor_23459[64, 64, 3, 3](dtype=float32), tensor_23459[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23460[64, 64, 3, 3](dtype=float32)]
     *************/

    // create tensor_23459 tensor
    unsigned tensor_23459_sizes[] = {64,64,3,3};
    unsigned tensor_23459 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23459",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23459_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23460 tensor
    unsigned tensor_23460_sizes[] = {64,64,3,3};
    unsigned tensor_23460 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23460",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23460_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23459, tensor_23459}, {tensor_23460}, nullptr, 0, "TPC97913");

    /*************
     * TPC97914 node
     * inputs: [tensor_23461[64, 64, 3, 3](dtype=float32), tensor_23461[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23462[64, 64, 3, 3](dtype=float32)]
     *************/

    // create tensor_23461 tensor
    unsigned tensor_23461_sizes[] = {64,64,3,3};
    unsigned tensor_23461 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23461",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23461_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23462 tensor
    unsigned tensor_23462_sizes[] = {64,64,3,3};
    unsigned tensor_23462 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23462",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23462_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23461, tensor_23461}, {tensor_23462}, nullptr, 0, "TPC97914");

    /*************
     * TPC97915 node
     * inputs: [tensor_23463[48, 48, 3, 3](dtype=float32), tensor_23463[48, 48, 3, 3](dtype=float32)]
     * output: [tensor_23464[48, 48, 3, 3](dtype=float32)]
     *************/

    // create tensor_23463 tensor
    unsigned tensor_23463_sizes[] = {48,48,3,3};
    unsigned tensor_23463 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23463",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23463_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23464 tensor
    unsigned tensor_23464_sizes[] = {48,48,3,3};
    unsigned tensor_23464 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23464",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23464_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23463, tensor_23463}, {tensor_23464}, nullptr, 0, "TPC97915");

    /*************
     * TPC97916 node
     * inputs: [tensor_23465[32, 16, 3, 3](dtype=float32), tensor_23465[32, 16, 3, 3](dtype=float32)]
     * output: [tensor_23466[32, 16, 3, 3](dtype=float32)]
     *************/

    // create tensor_23465 tensor
    unsigned tensor_23465_sizes[] = {32,16,3,3};
    unsigned tensor_23465 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23465",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23465_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23466 tensor
    unsigned tensor_23466_sizes[] = {32,16,3,3};
    unsigned tensor_23466 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23466",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23466_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23465, tensor_23465}, {tensor_23466}, nullptr, 0, "TPC97916");

    /*************
     * TPC97917 node
     * inputs: [tensor_23467[24, 112, 3, 3](dtype=float32), tensor_23467[24, 112, 3, 3](dtype=float32)]
     * output: [tensor_23468[24, 112, 3, 3](dtype=float32)]
     *************/

    // create tensor_23467 tensor
    unsigned tensor_23467_sizes[] = {24,112,3,3};
    unsigned tensor_23467 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23467",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23467_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23468 tensor
    unsigned tensor_23468_sizes[] = {24,112,3,3};
    unsigned tensor_23468 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23468",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23468_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23467, tensor_23467}, {tensor_23468}, nullptr, 0, "TPC97917");

    /*************
     * TPC97918 node
     * inputs: [tensor_23470[16, 16, 3, 3](dtype=float32), tensor_23470[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23471[16, 16, 3, 3](dtype=float32)]
     *************/

    // create tensor_23470 tensor
    unsigned tensor_23470_sizes[] = {16,16,3,3};
    unsigned tensor_23470 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23470",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23470_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23471 tensor
    unsigned tensor_23471_sizes[] = {16,16,3,3};
    unsigned tensor_23471 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23471",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23471_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23470, tensor_23470}, {tensor_23471}, nullptr, 0, "TPC97918");

    /*************
     * TPC97919 node
     * inputs: [tensor_23472[1, 704, 320, 2](dtype=float32), tensor_23473[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23474[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23472 tensor
    unsigned tensor_23472_sizes[] = {1,704,320,2};
    unsigned tensor_23472 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23472",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23472_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23473 tensor
    unsigned tensor_23473_sizes[] = {1,704,320,2};
    unsigned tensor_23473 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23473",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23473_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23474 tensor
    unsigned tensor_23474_sizes[] = {1,704,320,2};
    unsigned tensor_23474 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23474",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23474_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23472, tensor_23473}, {tensor_23474}, nullptr, 0, "TPC97919");

    /*************
     * TPC97920 node
     * inputs: [tensor_23474[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23475[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23475 tensor
    unsigned tensor_23475_sizes[] = {1,704,320,2};
    unsigned tensor_23475 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23475",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23475_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("abs_fwd_f32", {tensor_23474}, {tensor_23475}, nullptr, 0, "TPC97920");

    /*************
     * TPC97921 node
     * inputs: [tensor_23476[16, 40, 3, 3](dtype=float32), tensor_23476[16, 40, 3, 3](dtype=float32)]
     * output: [tensor_23477[16, 40, 3, 3](dtype=float32)]
     *************/

    // create tensor_23476 tensor
    unsigned tensor_23476_sizes[] = {16,40,3,3};
    unsigned tensor_23476 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23476",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23476_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23477 tensor
    unsigned tensor_23477_sizes[] = {16,40,3,3};
    unsigned tensor_23477 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23477",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23477_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23476, tensor_23476}, {tensor_23477}, nullptr, 0, "TPC97921");

    /*************
     * TPC97922 node
     * inputs: [tensor_23480[64, 64, 3, 3](dtype=float32), tensor_23480[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23481[64, 64, 3, 3](dtype=float32)]
     *************/

    // create tensor_23480 tensor
    unsigned tensor_23480_sizes[] = {64,64,3,3};
    unsigned tensor_23480 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23480",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23480_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23481 tensor
    unsigned tensor_23481_sizes[] = {64,64,3,3};
    unsigned tensor_23481 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23481",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23481_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23480, tensor_23480}, {tensor_23481}, nullptr, 0, "TPC97922");

    /*************
     * TPC97923 node
     * inputs: [tensor_23482[1, 704, 320, 2](dtype=float32), tensor_23482[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23483[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23482 tensor
    unsigned tensor_23482_sizes[] = {1,704,320,2};
    unsigned tensor_23482 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23482",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23482_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23483 tensor
    unsigned tensor_23483_sizes[] = {1,704,320,2};
    unsigned tensor_23483 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23483",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23483_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23482, tensor_23482}, {tensor_23483}, nullptr, 0, "TPC97923");

    /*************
     * TPC97924 node
     * inputs: [tensor_23483[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23484[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23484 tensor
    unsigned tensor_23484_sizes[] = {1,704,320,2};
    unsigned tensor_23484 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23484",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23484_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97924_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23483}, {tensor_23484}, (void*)TPC97924_params, 48, "TPC97924");

    /*************
     * TPC97925 node
     * inputs: [tensor_23482[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23485[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23485 tensor
    unsigned tensor_23485_sizes[] = {1,704,320,2};
    unsigned tensor_23485 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23485",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23485_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97925_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23482}, {tensor_23485}, (void*)TPC97925_params, 48, "TPC97925");

    /*************
     * TPC97926 node
     * inputs: [tensor_23485[1, 704, 320, 2](dtype=float32), tensor_23485[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23486[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23486 tensor
    unsigned tensor_23486_sizes[] = {1,704,320,2};
    unsigned tensor_23486 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23486",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23486_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23485, tensor_23485}, {tensor_23486}, nullptr, 0, "TPC97926");

    /*************
     * Transpose97927 node
     * inputs: [tensor_23487[2, 704, 320, 1](dtype=float32)]
     * output: [tensor_23488[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23487 tensor
    unsigned tensor_23487_sizes[] = {2,704,320,1};
    unsigned tensor_23487 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23487",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23487_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23488 tensor
    unsigned tensor_23488_sizes[] = {1,704,320,2};
    unsigned tensor_23488 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23488",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23488_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Transpose97927_params[] = {3,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0};
    addNodeToGraph("transpose", {tensor_23487}, {tensor_23488}, (void*)Transpose97927_params, 24, "Transpose97927");

    /*************
     * TPC97928 node
     * inputs: [tensor_23488[1, 704, 320, 2](dtype=float32), tensor_23488[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23489[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23489 tensor
    unsigned tensor_23489_sizes[] = {1,704,320,2};
    unsigned tensor_23489 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23489",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23489_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23488, tensor_23488}, {tensor_23489}, nullptr, 0, "TPC97928");

    /*************
     * TPC97929 node
     * inputs: [tensor_23489[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23490[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23490 tensor
    unsigned tensor_23490_sizes[] = {1,704,320,2};
    unsigned tensor_23490 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23490",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23490_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97929_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23489}, {tensor_23490}, (void*)TPC97929_params, 48, "TPC97929");

    /*************
     * TPC97930 node
     * inputs: [tensor_23488[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23491[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23491 tensor
    unsigned tensor_23491_sizes[] = {1,704,320,2};
    unsigned tensor_23491 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23491",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23491_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97930_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23488}, {tensor_23491}, (void*)TPC97930_params, 48, "TPC97930");

    /*************
     * TPC97931 node
     * inputs: [tensor_23491[1, 704, 320, 2](dtype=float32), tensor_23491[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23492[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23492 tensor
    unsigned tensor_23492_sizes[] = {1,704,320,2};
    unsigned tensor_23492 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23492",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23492_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23491, tensor_23491}, {tensor_23492}, nullptr, 0, "TPC97931");

    /*************
     * TPC97932 node
     * inputs: [tensor_23490[1, 704, 320, 2](dtype=float32), tensor_23492[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23493[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23493 tensor
    unsigned tensor_23493_sizes[] = {1,704,320,2};
    unsigned tensor_23493 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23493",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23493_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23490, tensor_23492}, {tensor_23493}, nullptr, 0, "TPC97932");

    /*************
     * TPC97933 node
     * inputs: [tensor_23486[1, 704, 320, 2](dtype=float32), tensor_23492[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23494[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23494 tensor
    unsigned tensor_23494_sizes[] = {1,704,320,2};
    unsigned tensor_23494 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23494",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23494_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23486, tensor_23492}, {tensor_23494}, nullptr, 0, "TPC97933");

    /*************
     * TPC97934 node
     * inputs: [tensor_23491[1, 704, 320, 2](dtype=float32), tensor_23485[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23495[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23495 tensor
    unsigned tensor_23495_sizes[] = {1,704,320,2};
    unsigned tensor_23495 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23495",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23495_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23491, tensor_23485}, {tensor_23495}, nullptr, 0, "TPC97934");

    /*************
     * TPC97935 node
     * inputs: [tensor_23482[1, 704, 320, 2](dtype=float32), tensor_23488[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23496[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23496 tensor
    unsigned tensor_23496_sizes[] = {1,704,320,2};
    unsigned tensor_23496 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23496",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23496_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23482, tensor_23488}, {tensor_23496}, nullptr, 0, "TPC97935");

    /*************
     * TPC97936 node
     * inputs: [tensor_23496[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23497[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23497 tensor
    unsigned tensor_23497_sizes[] = {1,704,320,2};
    unsigned tensor_23497 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23497",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23497_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97936_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23496}, {tensor_23497}, (void*)TPC97936_params, 48, "TPC97936");

    /*************
     * TPC97937 node
     * inputs: [tensor_23497[1, 704, 320, 2](dtype=float32), tensor_23495[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23498[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23498 tensor
    unsigned tensor_23498_sizes[] = {1,704,320,2};
    unsigned tensor_23498 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23498",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23498_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23497, tensor_23495}, {tensor_23498}, nullptr, 0, "TPC97937");

    /*************
     * Transpose97938 node
     * inputs: [tensor_23499[2, 704, 320, 1](dtype=float32)]
     * output: [tensor_23500[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23499 tensor
    unsigned tensor_23499_sizes[] = {2,704,320,1};
    unsigned tensor_23499 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23499",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23499_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23500 tensor
    unsigned tensor_23500_sizes[] = {1,704,320,2};
    unsigned tensor_23500 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23500",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23500_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Transpose97938_params[] = {3,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0};
    addNodeToGraph("transpose", {tensor_23499}, {tensor_23500}, (void*)Transpose97938_params, 24, "Transpose97938");

    /*************
     * TPC97939 node
     * inputs: [tensor_23500[1, 704, 320, 2](dtype=float32), tensor_23500[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23501[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23501 tensor
    unsigned tensor_23501_sizes[] = {1,704,320,2};
    unsigned tensor_23501 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23501",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23501_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23500, tensor_23500}, {tensor_23501}, nullptr, 0, "TPC97939");

    /*************
     * TPC97940 node
     * inputs: [tensor_23501[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23502[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23502 tensor
    unsigned tensor_23502_sizes[] = {1,704,320,2};
    unsigned tensor_23502 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23502",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23502_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97940_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23501}, {tensor_23502}, (void*)TPC97940_params, 48, "TPC97940");

    /*************
     * TPC97941 node
     * inputs: [tensor_23500[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23503[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23503 tensor
    unsigned tensor_23503_sizes[] = {1,704,320,2};
    unsigned tensor_23503 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23503",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23503_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97941_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23500}, {tensor_23503}, (void*)TPC97941_params, 48, "TPC97941");

    /*************
     * TPC97942 node
     * inputs: [tensor_23503[1, 704, 320, 2](dtype=float32), tensor_23503[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23504[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23504 tensor
    unsigned tensor_23504_sizes[] = {1,704,320,2};
    unsigned tensor_23504 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23504",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23504_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23503, tensor_23503}, {tensor_23504}, nullptr, 0, "TPC97942");

    /*************
     * TPC97943 node
     * inputs: [tensor_23502[1, 704, 320, 2](dtype=float32), tensor_23504[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23505[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23505 tensor
    unsigned tensor_23505_sizes[] = {1,704,320,2};
    unsigned tensor_23505 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23505",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23505_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23502, tensor_23504}, {tensor_23505}, nullptr, 0, "TPC97943");

    /*************
     * Transpose97944 node
     * inputs: [tensor_23506[2, 704, 320, 1](dtype=float32)]
     * output: [tensor_23507[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23506 tensor
    unsigned tensor_23506_sizes[] = {2,704,320,1};
    unsigned tensor_23506 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23506",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23506_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23507 tensor
    unsigned tensor_23507_sizes[] = {1,704,320,2};
    unsigned tensor_23507 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23507",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23507_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Transpose97944_params[] = {3,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0};
    addNodeToGraph("transpose", {tensor_23506}, {tensor_23507}, (void*)Transpose97944_params, 24, "Transpose97944");

    /*************
     * TPC97945 node
     * inputs: [tensor_23507[1, 704, 320, 2](dtype=float32), tensor_23507[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23508[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23508 tensor
    unsigned tensor_23508_sizes[] = {1,704,320,2};
    unsigned tensor_23508 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23508",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23508_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23507, tensor_23507}, {tensor_23508}, nullptr, 0, "TPC97945");

    /*************
     * TPC97946 node
     * inputs: [tensor_23508[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23509[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23509 tensor
    unsigned tensor_23509_sizes[] = {1,704,320,2};
    unsigned tensor_23509 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23509",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23509_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97946_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23508}, {tensor_23509}, (void*)TPC97946_params, 48, "TPC97946");

    /*************
     * TPC97947 node
     * inputs: [tensor_23507[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23510[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23510 tensor
    unsigned tensor_23510_sizes[] = {1,704,320,2};
    unsigned tensor_23510 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23510",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23510_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97947_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23507}, {tensor_23510}, (void*)TPC97947_params, 48, "TPC97947");

    /*************
     * TPC97948 node
     * inputs: [tensor_23510[1, 704, 320, 2](dtype=float32), tensor_23510[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23511[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23511 tensor
    unsigned tensor_23511_sizes[] = {1,704,320,2};
    unsigned tensor_23511 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23511",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23511_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23510, tensor_23510}, {tensor_23511}, nullptr, 0, "TPC97948");

    /*************
     * TPC97949 node
     * inputs: [tensor_23509[1, 704, 320, 2](dtype=float32), tensor_23511[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23512[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23512 tensor
    unsigned tensor_23512_sizes[] = {1,704,320,2};
    unsigned tensor_23512 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23512",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23512_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23509, tensor_23511}, {tensor_23512}, nullptr, 0, "TPC97949");

    /*************
     * TPC97950 node
     * inputs: [tensor_23514[1, 8, 3, 3](dtype=float32), tensor_23514[1, 8, 3, 3](dtype=float32)]
     * output: [tensor_23515[1, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23514 tensor
    unsigned tensor_23514_sizes[] = {1,8,3,3};
    unsigned tensor_23514 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23514",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23514_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23515 tensor
    unsigned tensor_23515_sizes[] = {1,8,3,3};
    unsigned tensor_23515 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23515",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23515_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23514, tensor_23514}, {tensor_23515}, nullptr, 0, "TPC97950");

    /*************
     * Reshape97951 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23518[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23516 tensor
    unsigned tensor_23516_sizes[] = {1};
    unsigned tensor_23516 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23516",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23516_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23518 tensor
    unsigned tensor_23518_sizes[] = {1,1,1,1};
    unsigned tensor_23518 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23518",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23518_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23518}, nullptr, 0, "Reshape97951");

    /*************
     * TPC97952 node
     * inputs: [tensor_23473[1, 704, 320, 2](dtype=float32), tensor_23518[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23517[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23517 tensor
    unsigned tensor_23517_sizes[] = {1,704,320,2};
    unsigned tensor_23517 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23517",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23517_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("equal_fwd_f32", {tensor_23473, tensor_23518}, {tensor_23517}, nullptr, 0, "TPC97952");

    /*************
     * TPC97953 node
     * inputs: [tensor_23517[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23519[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23519 tensor
    unsigned tensor_23519_sizes[] = {1,704,320,2};
    unsigned tensor_23519 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23519",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23519_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23517}, {tensor_23519}, nullptr, 0, "TPC97953");

    /*************
     * Reshape97954 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23521[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23521 tensor
    unsigned tensor_23521_sizes[] = {1,1,1,1};
    unsigned tensor_23521 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23521",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23521_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23521}, nullptr, 0, "Reshape97954");

    /*************
     * TPC97955 node
     * inputs: [tensor_23482[1, 704, 320, 2](dtype=float32), tensor_23521[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23520[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23520 tensor
    unsigned tensor_23520_sizes[] = {1,704,320,2};
    unsigned tensor_23520 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23520",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23520_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("equal_fwd_f32", {tensor_23482, tensor_23521}, {tensor_23520}, nullptr, 0, "TPC97955");

    /*************
     * TPC97956 node
     * inputs: [tensor_23520[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23522[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23522 tensor
    unsigned tensor_23522_sizes[] = {1,704,320,2};
    unsigned tensor_23522 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23522",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23522_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23520}, {tensor_23522}, nullptr, 0, "TPC97956");

    /*************
     * Reshape97957 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23525[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23523 tensor
    unsigned tensor_23523_sizes[] = {1};
    unsigned tensor_23523 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23523",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23523_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23525 tensor
    unsigned tensor_23525_sizes[] = {1,1,1,1};
    unsigned tensor_23525 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23525",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23525_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23525}, nullptr, 0, "Reshape97957");

    /*************
     * TPC97958 node
     * inputs: [tensor_23525[1, 1, 1, 1](dtype=float32), tensor_23494[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23524[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23524 tensor
    unsigned tensor_23524_sizes[] = {1,704,320,2};
    unsigned tensor_23524 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23524",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23524_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23525, tensor_23494}, {tensor_23524}, nullptr, 0, "TPC97958");

    /*************
     * TPC97959 node
     * inputs: [tensor_23526[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23527[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23526 tensor
    unsigned tensor_23526_sizes[] = {1,704,320,2};
    unsigned tensor_23526 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23526",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23526_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23527 tensor
    unsigned tensor_23527_sizes[] = {1,704,320,2};
    unsigned tensor_23527 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23527",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23527_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23526}, {tensor_23527}, nullptr, 0, "TPC97959");

    /*************
     * Reshape97960 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23529[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23529 tensor
    unsigned tensor_23529_sizes[] = {1,1,1,1};
    unsigned tensor_23529 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23529",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23529_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23529}, nullptr, 0, "Reshape97960");

    /*************
     * TPC97961 node
     * inputs: [tensor_23527[1, 704, 320, 2](dtype=float32), tensor_23529[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23528[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23528 tensor
    unsigned tensor_23528_sizes[] = {1,704,320,2};
    unsigned tensor_23528 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23528",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23528_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23527, tensor_23529}, {tensor_23528}, nullptr, 0, "TPC97961");

    /*************
     * Reshape97962 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23531[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23531 tensor
    unsigned tensor_23531_sizes[] = {1,1,1,1};
    unsigned tensor_23531 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23531",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23531_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23531}, nullptr, 0, "Reshape97962");

    /*************
     * TPC97963 node
     * inputs: [tensor_23531[1, 1, 1, 1](dtype=float32), tensor_23528[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23530[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23530 tensor
    unsigned tensor_23530_sizes[] = {1,704,320,2};
    unsigned tensor_23530 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23530",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23530_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("equal_fwd_f32", {tensor_23531, tensor_23528}, {tensor_23530}, nullptr, 0, "TPC97963");

    /*************
     * TPC97964 node
     * inputs: [tensor_23530[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23532[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23532 tensor
    unsigned tensor_23532_sizes[] = {1,704,320,2};
    unsigned tensor_23532 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23532",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23532_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23530}, {tensor_23532}, nullptr, 0, "TPC97964");

    /*************
     * TPC97965 node
     * inputs: [tensor_23472[1, 704, 320, 2](dtype=float32), tensor_23528[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23533[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23533 tensor
    unsigned tensor_23533_sizes[] = {1,704,320,2};
    unsigned tensor_23533 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23533",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23533_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23472, tensor_23528}, {tensor_23533}, nullptr, 0, "TPC97965");

    /*************
     * TPC97966 node
     * inputs: [tensor_23533[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23534[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23534 tensor
    unsigned tensor_23534_sizes[] = {1,704,320,2};
    unsigned tensor_23534 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23534",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23534_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("abs_fwd_f32", {tensor_23533}, {tensor_23534}, nullptr, 0, "TPC97966");

    /*************
     * TPC97967 node
     * inputs: [tensor_23535[1, 704, 320, 2](dtype=float32), tensor_23500[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23536[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23535 tensor
    unsigned tensor_23535_sizes[] = {1,704,320,2};
    unsigned tensor_23535 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23535",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23535_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23536 tensor
    unsigned tensor_23536_sizes[] = {1,704,320,2};
    unsigned tensor_23536 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23536",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23536_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23535, tensor_23500}, {tensor_23536}, nullptr, 0, "TPC97967");

    /*************
     * TPC97968 node
     * inputs: [tensor_23536[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23537[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23537 tensor
    unsigned tensor_23537_sizes[] = {1,704,320,2};
    unsigned tensor_23537 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23537",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23537_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97968_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23536}, {tensor_23537}, (void*)TPC97968_params, 48, "TPC97968");

    /*************
     * TPC97969 node
     * inputs: [tensor_23535[1, 704, 320, 2](dtype=float32), tensor_23535[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23538[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23538 tensor
    unsigned tensor_23538_sizes[] = {1,704,320,2};
    unsigned tensor_23538 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23538",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23538_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23535, tensor_23535}, {tensor_23538}, nullptr, 0, "TPC97969");

    /*************
     * TPC97970 node
     * inputs: [tensor_23538[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23539[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23539 tensor
    unsigned tensor_23539_sizes[] = {1,704,320,2};
    unsigned tensor_23539 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23539",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23539_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97970_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23538}, {tensor_23539}, (void*)TPC97970_params, 48, "TPC97970");

    /*************
     * TPC97971 node
     * inputs: [tensor_23535[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23540[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23540 tensor
    unsigned tensor_23540_sizes[] = {1,704,320,2};
    unsigned tensor_23540 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23540",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23540_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97971_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23535}, {tensor_23540}, (void*)TPC97971_params, 48, "TPC97971");

    /*************
     * TPC97972 node
     * inputs: [tensor_23503[1, 704, 320, 2](dtype=float32), tensor_23540[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23541[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23541 tensor
    unsigned tensor_23541_sizes[] = {1,704,320,2};
    unsigned tensor_23541 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23541",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23541_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23503, tensor_23540}, {tensor_23541}, nullptr, 0, "TPC97972");

    /*************
     * TPC97973 node
     * inputs: [tensor_23537[1, 704, 320, 2](dtype=float32), tensor_23541[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23542[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23542 tensor
    unsigned tensor_23542_sizes[] = {1,704,320,2};
    unsigned tensor_23542 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23542",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23542_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23537, tensor_23541}, {tensor_23542}, nullptr, 0, "TPC97973");

    /*************
     * TPC97974 node
     * inputs: [tensor_23540[1, 704, 320, 2](dtype=float32), tensor_23540[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23543[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23543 tensor
    unsigned tensor_23543_sizes[] = {1,704,320,2};
    unsigned tensor_23543 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23543",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23543_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23540, tensor_23540}, {tensor_23543}, nullptr, 0, "TPC97974");

    /*************
     * TPC97975 node
     * inputs: [tensor_23543[1, 704, 320, 2](dtype=float32), tensor_23504[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23544[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23544 tensor
    unsigned tensor_23544_sizes[] = {1,704,320,2};
    unsigned tensor_23544 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23544",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23544_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23543, tensor_23504}, {tensor_23544}, nullptr, 0, "TPC97975");

    /*************
     * Reshape97976 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23546[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23546 tensor
    unsigned tensor_23546_sizes[] = {1,1,1,1};
    unsigned tensor_23546 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23546",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23546_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23546}, nullptr, 0, "Reshape97976");

    /*************
     * TPC97977 node
     * inputs: [tensor_23546[1, 1, 1, 1](dtype=float32), tensor_23544[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23545[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23545 tensor
    unsigned tensor_23545_sizes[] = {1,704,320,2};
    unsigned tensor_23545 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23545",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23545_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23546, tensor_23544}, {tensor_23545}, nullptr, 0, "TPC97977");

    /*************
     * Reshape97978 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23548[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23548 tensor
    unsigned tensor_23548_sizes[] = {1,1,1,1};
    unsigned tensor_23548 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23548",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23548_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23548}, nullptr, 0, "Reshape97978");

    /*************
     * TPC97979 node
     * inputs: [tensor_23535[1, 704, 320, 2](dtype=float32), tensor_23548[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23547[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23547 tensor
    unsigned tensor_23547_sizes[] = {1,704,320,2};
    unsigned tensor_23547 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23547",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23547_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("equal_fwd_f32", {tensor_23535, tensor_23548}, {tensor_23547}, nullptr, 0, "TPC97979");

    /*************
     * TPC97980 node
     * inputs: [tensor_23547[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23549[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23549 tensor
    unsigned tensor_23549_sizes[] = {1,704,320,2};
    unsigned tensor_23549 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23549",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23549_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23547}, {tensor_23549}, nullptr, 0, "TPC97980");

    /*************
     * TPC97981 node
     * inputs: [tensor_23550[8, 24, 3, 3](dtype=float32), tensor_23550[8, 24, 3, 3](dtype=float32)]
     * output: [tensor_23551[8, 24, 3, 3](dtype=float32)]
     *************/

    // create tensor_23550 tensor
    unsigned tensor_23550_sizes[] = {8,24,3,3};
    unsigned tensor_23550 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23550",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23550_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23551 tensor
    unsigned tensor_23551_sizes[] = {8,24,3,3};
    unsigned tensor_23551 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23551",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23551_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23550, tensor_23550}, {tensor_23551}, nullptr, 0, "TPC97981");

    /*************
     * Reshape97982 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23554[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23554 tensor
    unsigned tensor_23554_sizes[] = {1,1,1,1};
    unsigned tensor_23554 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23554",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23554_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23554}, nullptr, 0, "Reshape97982");

    /*************
     * TPC97983 node
     * inputs: [tensor_23552[1, 704, 320, 2](dtype=float32), tensor_23554[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23553[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23552 tensor
    unsigned tensor_23552_sizes[] = {1,704,320,2};
    unsigned tensor_23552 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23552",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23552_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23553 tensor
    unsigned tensor_23553_sizes[] = {1,704,320,2};
    unsigned tensor_23553 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23553",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23553_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("equal_fwd_f32", {tensor_23552, tensor_23554}, {tensor_23553}, nullptr, 0, "TPC97983");

    /*************
     * TPC97984 node
     * inputs: [tensor_23553[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23555[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23555 tensor
    unsigned tensor_23555_sizes[] = {1,704,320,2};
    unsigned tensor_23555 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23555",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23555_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23553}, {tensor_23555}, nullptr, 0, "TPC97984");

    /*************
     * TPC97985 node
     * inputs: [tensor_23472[1, 704, 320, 2](dtype=float32), tensor_23552[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23556[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23556 tensor
    unsigned tensor_23556_sizes[] = {1,704,320,2};
    unsigned tensor_23556 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23556",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23556_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23472, tensor_23552}, {tensor_23556}, nullptr, 0, "TPC97985");

    /*************
     * TPC97986 node
     * inputs: [tensor_23556[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23557[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23557 tensor
    unsigned tensor_23557_sizes[] = {1,704,320,2};
    unsigned tensor_23557 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23557",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23557_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("abs_fwd_f32", {tensor_23556}, {tensor_23557}, nullptr, 0, "TPC97986");

    /*************
     * TPC97987 node
     * inputs: [tensor_23560[1, 704, 320, 2](dtype=float32), tensor_23507[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23561[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23560 tensor
    unsigned tensor_23560_sizes[] = {1,704,320,2};
    unsigned tensor_23560 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23560",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23560_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23561 tensor
    unsigned tensor_23561_sizes[] = {1,704,320,2};
    unsigned tensor_23561 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23561",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23561_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23560, tensor_23507}, {tensor_23561}, nullptr, 0, "TPC97987");

    /*************
     * TPC97988 node
     * inputs: [tensor_23561[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23562[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23562 tensor
    unsigned tensor_23562_sizes[] = {1,704,320,2};
    unsigned tensor_23562 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23562",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23562_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97988_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23561}, {tensor_23562}, (void*)TPC97988_params, 48, "TPC97988");

    /*************
     * TPC97989 node
     * inputs: [tensor_23560[1, 704, 320, 2](dtype=float32), tensor_23560[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23563[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23563 tensor
    unsigned tensor_23563_sizes[] = {1,704,320,2};
    unsigned tensor_23563 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23563",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23563_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23560, tensor_23560}, {tensor_23563}, nullptr, 0, "TPC97989");

    /*************
     * TPC97990 node
     * inputs: [tensor_23563[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23564[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23564 tensor
    unsigned tensor_23564_sizes[] = {1,704,320,2};
    unsigned tensor_23564 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23564",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23564_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97990_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23563}, {tensor_23564}, (void*)TPC97990_params, 48, "TPC97990");

    /*************
     * TPC97991 node
     * inputs: [tensor_23560[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23565[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23565 tensor
    unsigned tensor_23565_sizes[] = {1,704,320,2};
    unsigned tensor_23565 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23565",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23565_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC97991_params[] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    addNodeToGraph("avg_pool_2d_fwd_f32", {tensor_23560}, {tensor_23565}, (void*)TPC97991_params, 48, "TPC97991");

    /*************
     * TPC97992 node
     * inputs: [tensor_23510[1, 704, 320, 2](dtype=float32), tensor_23565[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23566[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23566 tensor
    unsigned tensor_23566_sizes[] = {1,704,320,2};
    unsigned tensor_23566 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23566",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23566_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23510, tensor_23565}, {tensor_23566}, nullptr, 0, "TPC97992");

    /*************
     * TPC97993 node
     * inputs: [tensor_23562[1, 704, 320, 2](dtype=float32), tensor_23566[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23567[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23567 tensor
    unsigned tensor_23567_sizes[] = {1,704,320,2};
    unsigned tensor_23567 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23567",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23567_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23562, tensor_23566}, {tensor_23567}, nullptr, 0, "TPC97993");

    /*************
     * TPC97994 node
     * inputs: [tensor_23565[1, 704, 320, 2](dtype=float32), tensor_23565[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23568[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23568 tensor
    unsigned tensor_23568_sizes[] = {1,704,320,2};
    unsigned tensor_23568 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23568",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23568_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23565, tensor_23565}, {tensor_23568}, nullptr, 0, "TPC97994");

    /*************
     * TPC97995 node
     * inputs: [tensor_23568[1, 704, 320, 2](dtype=float32), tensor_23511[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23569[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23569 tensor
    unsigned tensor_23569_sizes[] = {1,704,320,2};
    unsigned tensor_23569 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23569",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23569_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23568, tensor_23511}, {tensor_23569}, nullptr, 0, "TPC97995");

    /*************
     * Reshape97996 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23571[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23571 tensor
    unsigned tensor_23571_sizes[] = {1,1,1,1};
    unsigned tensor_23571 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23571",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23571_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23571}, nullptr, 0, "Reshape97996");

    /*************
     * TPC97997 node
     * inputs: [tensor_23571[1, 1, 1, 1](dtype=float32), tensor_23569[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23570[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23570 tensor
    unsigned tensor_23570_sizes[] = {1,704,320,2};
    unsigned tensor_23570 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23570",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23570_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23571, tensor_23569}, {tensor_23570}, nullptr, 0, "TPC97997");

    /*************
     * Reshape97998 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23573[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23573 tensor
    unsigned tensor_23573_sizes[] = {1,1,1,1};
    unsigned tensor_23573 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23573",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23573_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23573}, nullptr, 0, "Reshape97998");

    /*************
     * TPC97999 node
     * inputs: [tensor_23560[1, 704, 320, 2](dtype=float32), tensor_23573[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23572[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23572 tensor
    unsigned tensor_23572_sizes[] = {1,704,320,2};
    unsigned tensor_23572 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23572",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23572_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("equal_fwd_f32", {tensor_23560, tensor_23573}, {tensor_23572}, nullptr, 0, "TPC97999");

    /*************
     * TPC98000 node
     * inputs: [tensor_23572[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23574[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23574 tensor
    unsigned tensor_23574_sizes[] = {1,704,320,2};
    unsigned tensor_23574 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23574",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23574_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23572}, {tensor_23574}, nullptr, 0, "TPC98000");

    /*************
     * Concatenate98001 node
     * inputs: [tensor_23575[16, 352, 160, 2](dtype=float32), tensor_23576[24, 352, 160, 2](dtype=float32)]
     * output: [tensor_23577[40, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23575 tensor
    unsigned tensor_23575_sizes[] = {16,352,160,2};
    unsigned tensor_23575 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23575",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23575_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23576 tensor
    unsigned tensor_23576_sizes[] = {24,352,160,2};
    unsigned tensor_23576 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23576",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23576_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23577 tensor
    unsigned tensor_23577_sizes[] = {40,352,160,2};
    unsigned tensor_23577 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23577",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23577_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Concatenate98001_params[] = {0,0,0,0};
    addNodeToGraph("concat", {tensor_23575, tensor_23576}, {tensor_23577}, (void*)Concatenate98001_params, 4, "Concatenate98001");

    /*************
     * Convolution98002 node
     * inputs: [tensor_23577[40, 352, 160, 2](dtype=float32), tensor_23578[16, 40, 3, 3](dtype=float32)]
     * output: [tensor_23579[16, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23578 tensor
    unsigned tensor_23578_sizes[] = {16,40,3,3};
    unsigned tensor_23578 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23578",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23578_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23579 tensor
    unsigned tensor_23579_sizes[] = {16,352,160,2};
    unsigned tensor_23579 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23579",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23579_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98002_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,142,14,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,158,237,19,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23577, tensor_23578}, {tensor_23579}, (void*)Convolution98002_params, 88, "Convolution98002");

    /*************
     * Reshape98003 node
     * inputs: [tensor_23580[16](dtype=float32)]
     * output: [tensor_23582[16, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23580 tensor
    unsigned tensor_23580_sizes[] = {16};
    unsigned tensor_23580 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23580",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23580_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23582 tensor
    unsigned tensor_23582_sizes[] = {16,1,1,1};
    unsigned tensor_23582 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23582",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23582_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23580}, {tensor_23582}, nullptr, 0, "Reshape98003");

    /*************
     * TPC98004 node
     * inputs: [tensor_23579[16, 352, 160, 2](dtype=float32), tensor_23582[16, 1, 1, 1](dtype=float32)]
     * output: [tensor_23581[16, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23581 tensor
    unsigned tensor_23581_sizes[] = {16,352,160,2};
    unsigned tensor_23581 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23581",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23581_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23579, tensor_23582}, {tensor_23581}, nullptr, 0, "TPC98004");

    /*************
     * TPC98005 node
     * inputs: [tensor_23581[16, 352, 160, 2](dtype=float32)]
     * output: [tensor_23583[16, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23583 tensor
    unsigned tensor_23583_sizes[] = {16,352,160,2};
    unsigned tensor_23583 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23583",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23583_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("relu_fwd_f32", {tensor_23581}, {tensor_23583}, nullptr, 0, "TPC98005");

    /*************
     * Convolution98006 node
     * inputs: [tensor_23583[16, 352, 160, 2](dtype=float32), tensor_23455[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23584[16, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23584 tensor
    unsigned tensor_23584_sizes[] = {16,352,160,2};
    unsigned tensor_23584 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23584",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23584_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98006_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,53,15,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,231,19,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23583, tensor_23455}, {tensor_23584}, (void*)Convolution98006_params, 88, "Convolution98006");

    /*************
     * Reshape98007 node
     * inputs: [tensor_23585[16](dtype=float32)]
     * output: [tensor_23587[16, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23585 tensor
    unsigned tensor_23585_sizes[] = {16};
    unsigned tensor_23585 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23585",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23585_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23587 tensor
    unsigned tensor_23587_sizes[] = {16,1,1,1};
    unsigned tensor_23587 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23587",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23587_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23585}, {tensor_23587}, nullptr, 0, "Reshape98007");

    /*************
     * TPC98008 node
     * inputs: [tensor_23584[16, 352, 160, 2](dtype=float32), tensor_23587[16, 1, 1, 1](dtype=float32)]
     * output: [tensor_23586[16, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23586 tensor
    unsigned tensor_23586_sizes[] = {16,352,160,2};
    unsigned tensor_23586 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23586",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23586_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23584, tensor_23587}, {tensor_23586}, nullptr, 0, "TPC98008");

    /*************
     * TPC98009 node
     * inputs: [tensor_23586[16, 352, 160, 2](dtype=float32)]
     * output: [tensor_23588[16, 352, 160, 2](dtype=float32)]
     *************/

    // create tensor_23588 tensor
    unsigned tensor_23588_sizes[] = {16,352,160,2};
    unsigned tensor_23588 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23588",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23588_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("relu_fwd_f32", {tensor_23586}, {tensor_23588}, nullptr, 0, "TPC98009");

    /*************
     * TPC98010 node
     * inputs: [tensor_23588[16, 352, 160, 2](dtype=float32)]
     * output: [tensor_23589[16, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23589 tensor
    unsigned tensor_23589_sizes[] = {16,704,320,2};
    unsigned tensor_23589 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23589",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23589_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98010_params[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,3,0,0,0,0,0,155,184,0,0,0,0,192,2,0,0,64,1,0,0,2,0,0,0};
    addNodeToGraph("resize_fwd_f32", {tensor_23588}, {tensor_23589}, (void*)TPC98010_params, 44, "TPC98010");

    /*************
     * Concatenate98011 node
     * inputs: [tensor_23590[8, 704, 320, 2](dtype=float32), tensor_23589[16, 704, 320, 2](dtype=float32)]
     * output: [tensor_23591[24, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23590 tensor
    unsigned tensor_23590_sizes[] = {8,704,320,2};
    unsigned tensor_23590 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23590",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23590_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23591 tensor
    unsigned tensor_23591_sizes[] = {24,704,320,2};
    unsigned tensor_23591 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23591",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23591_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Concatenate98011_params[] = {0,0,0,0};
    addNodeToGraph("concat", {tensor_23590, tensor_23589}, {tensor_23591}, (void*)Concatenate98011_params, 4, "Concatenate98011");

    /*************
     * Convolution98012 node
     * inputs: [tensor_23591[24, 704, 320, 2](dtype=float32), tensor_23550[8, 24, 3, 3](dtype=float32)]
     * output: [tensor_23592[8, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23592 tensor
    unsigned tensor_23592_sizes[] = {8,704,320,2};
    unsigned tensor_23592 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23592",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23592_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98012_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,55,12,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,187,19,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23591, tensor_23550}, {tensor_23592}, (void*)Convolution98012_params, 88, "Convolution98012");

    /*************
     * Reshape98013 node
     * inputs: [tensor_23593[8](dtype=float32)]
     * output: [tensor_23595[8, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23593 tensor
    unsigned tensor_23593_sizes[] = {8};
    unsigned tensor_23593 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23593",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23593_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23595 tensor
    unsigned tensor_23595_sizes[] = {8,1,1,1};
    unsigned tensor_23595 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23595",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23595_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23593}, {tensor_23595}, nullptr, 0, "Reshape98013");

    /*************
     * TPC98014 node
     * inputs: [tensor_23592[8, 704, 320, 2](dtype=float32), tensor_23595[8, 1, 1, 1](dtype=float32)]
     * output: [tensor_23594[8, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23594 tensor
    unsigned tensor_23594_sizes[] = {8,704,320,2};
    unsigned tensor_23594 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23594",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23594_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23592, tensor_23595}, {tensor_23594}, nullptr, 0, "TPC98014");

    /*************
     * TPC98015 node
     * inputs: [tensor_23594[8, 704, 320, 2](dtype=float32)]
     * output: [tensor_23596[8, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23596 tensor
    unsigned tensor_23596_sizes[] = {8,704,320,2};
    unsigned tensor_23596 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23596",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23596_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("relu_fwd_f32", {tensor_23594}, {tensor_23596}, nullptr, 0, "TPC98015");

    /*************
     * TPC98016 node
     * inputs: [tensor_23597[216, 216, 3, 3](dtype=float32), tensor_23597[216, 216, 3, 3](dtype=float32)]
     * output: [tensor_23598[216, 216, 3, 3](dtype=float32)]
     *************/

    // create tensor_23597 tensor
    unsigned tensor_23597_sizes[] = {216,216,3,3};
    unsigned tensor_23597 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23597",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23597_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23598 tensor
    unsigned tensor_23598_sizes[] = {216,216,3,3};
    unsigned tensor_23598 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23598",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23598_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23597, tensor_23597}, {tensor_23598}, nullptr, 0, "TPC98016");

    /*************
     * Reshape98017 node
     * inputs: [tensor_23599[1](dtype=float32)]
     * output: [tensor_23601[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23599 tensor
    unsigned tensor_23599_sizes[] = {1};
    unsigned tensor_23599 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23599",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23599_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23601 tensor
    unsigned tensor_23601_sizes[] = {1,1,1,1};
    unsigned tensor_23601 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23601",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23601_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23599}, {tensor_23601}, nullptr, 0, "Reshape98017");

    /*************
     * TPC98018 node
     * inputs: [tensor_23601[1, 1, 1, 1](dtype=float32), tensor_23486[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23600[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23600 tensor
    unsigned tensor_23600_sizes[] = {1,704,320,2};
    unsigned tensor_23600 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23600",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23600_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23601, tensor_23486}, {tensor_23600}, nullptr, 0, "TPC98018");

    /*************
     * TPC98019 node
     * inputs: [tensor_23484[1, 704, 320, 2](dtype=float32), tensor_23600[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23603[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23603 tensor
    unsigned tensor_23603_sizes[] = {1,704,320,2};
    unsigned tensor_23603 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23603",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23603_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23484, tensor_23600}, {tensor_23603}, nullptr, 0, "TPC98019");

    /*************
     * TPC98020 node
     * inputs: [tensor_23493[1, 704, 320, 2](dtype=float32), tensor_23603[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23602[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23602 tensor
    unsigned tensor_23602_sizes[] = {1,704,320,2};
    unsigned tensor_23602 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23602",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23602_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23493, tensor_23603}, {tensor_23602}, nullptr, 0, "TPC98020");

    /*************
     * TPC98021 node
     * inputs: [tensor_23602[1, 704, 320, 2](dtype=float32), tensor_23524[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23604[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23604 tensor
    unsigned tensor_23604_sizes[] = {1,704,320,2};
    unsigned tensor_23604 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23604",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23604_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23602, tensor_23524}, {tensor_23604}, nullptr, 0, "TPC98021");

    /*************
     * Reshape98022 node
     * inputs: [tensor_23599[1](dtype=float32)]
     * output: [tensor_23606[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23606 tensor
    unsigned tensor_23606_sizes[] = {1,1,1,1};
    unsigned tensor_23606 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23606",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23606_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23599}, {tensor_23606}, nullptr, 0, "Reshape98022");

    /*************
     * TPC98023 node
     * inputs: [tensor_23606[1, 1, 1, 1](dtype=float32), tensor_23543[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23605[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23605 tensor
    unsigned tensor_23605_sizes[] = {1,704,320,2};
    unsigned tensor_23605 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23605",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23605_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23606, tensor_23543}, {tensor_23605}, nullptr, 0, "TPC98023");

    /*************
     * TPC98024 node
     * inputs: [tensor_23539[1, 704, 320, 2](dtype=float32), tensor_23605[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23608[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23608 tensor
    unsigned tensor_23608_sizes[] = {1,704,320,2};
    unsigned tensor_23608 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23608",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23608_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23539, tensor_23605}, {tensor_23608}, nullptr, 0, "TPC98024");

    /*************
     * TPC98025 node
     * inputs: [tensor_23505[1, 704, 320, 2](dtype=float32), tensor_23608[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23607[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23607 tensor
    unsigned tensor_23607_sizes[] = {1,704,320,2};
    unsigned tensor_23607 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23607",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23607_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23505, tensor_23608}, {tensor_23607}, nullptr, 0, "TPC98025");

    /*************
     * TPC98026 node
     * inputs: [tensor_23545[1, 704, 320, 2](dtype=float32), tensor_23607[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23609[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23609 tensor
    unsigned tensor_23609_sizes[] = {1,704,320,2};
    unsigned tensor_23609 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23609",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23609_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23545, tensor_23607}, {tensor_23609}, nullptr, 0, "TPC98026");

    /*************
     * Reshape98027 node
     * inputs: [tensor_23599[1](dtype=float32)]
     * output: [tensor_23611[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23611 tensor
    unsigned tensor_23611_sizes[] = {1,1,1,1};
    unsigned tensor_23611 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23611",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23611_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23599}, {tensor_23611}, nullptr, 0, "Reshape98027");

    /*************
     * TPC98028 node
     * inputs: [tensor_23611[1, 1, 1, 1](dtype=float32), tensor_23568[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23610[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23610 tensor
    unsigned tensor_23610_sizes[] = {1,704,320,2};
    unsigned tensor_23610 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23610",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23610_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23611, tensor_23568}, {tensor_23610}, nullptr, 0, "TPC98028");

    /*************
     * TPC98029 node
     * inputs: [tensor_23564[1, 704, 320, 2](dtype=float32), tensor_23610[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23613[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23613 tensor
    unsigned tensor_23613_sizes[] = {1,704,320,2};
    unsigned tensor_23613 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23613",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23613_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23564, tensor_23610}, {tensor_23613}, nullptr, 0, "TPC98029");

    /*************
     * TPC98030 node
     * inputs: [tensor_23512[1, 704, 320, 2](dtype=float32), tensor_23613[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23612[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23612 tensor
    unsigned tensor_23612_sizes[] = {1,704,320,2};
    unsigned tensor_23612 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23612",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23612_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23512, tensor_23613}, {tensor_23612}, nullptr, 0, "TPC98030");

    /*************
     * TPC98031 node
     * inputs: [tensor_23570[1, 704, 320, 2](dtype=float32), tensor_23612[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23614[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23614 tensor
    unsigned tensor_23614_sizes[] = {1,704,320,2};
    unsigned tensor_23614 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23614",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23614_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23570, tensor_23612}, {tensor_23614}, nullptr, 0, "TPC98031");

    /*************
     * Convolution98032 node
     * inputs: [tensor_23596[8, 704, 320, 2](dtype=float32), tensor_23443[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23615[8, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23615 tensor
    unsigned tensor_23615_sizes[] = {8,704,320,2};
    unsigned tensor_23615 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23615",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23615_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98032_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,89,145,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,0,20,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23596, tensor_23443}, {tensor_23615}, (void*)Convolution98032_params, 88, "Convolution98032");

    /*************
     * Reshape98033 node
     * inputs: [tensor_23616[8](dtype=float32)]
     * output: [tensor_23618[8, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23616 tensor
    unsigned tensor_23616_sizes[] = {8};
    unsigned tensor_23616 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23616",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23616_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23618 tensor
    unsigned tensor_23618_sizes[] = {8,1,1,1};
    unsigned tensor_23618 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23618",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23618_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23616}, {tensor_23618}, nullptr, 0, "Reshape98033");

    /*************
     * TPC98034 node
     * inputs: [tensor_23615[8, 704, 320, 2](dtype=float32), tensor_23618[8, 1, 1, 1](dtype=float32)]
     * output: [tensor_23617[8, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23617 tensor
    unsigned tensor_23617_sizes[] = {8,704,320,2};
    unsigned tensor_23617 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23617",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23617_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23615, tensor_23618}, {tensor_23617}, nullptr, 0, "TPC98034");

    /*************
     * TPC98035 node
     * inputs: [tensor_23617[8, 704, 320, 2](dtype=float32)]
     * output: [tensor_23619[8, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23619 tensor
    unsigned tensor_23619_sizes[] = {8,704,320,2};
    unsigned tensor_23619 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23619",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23619_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("relu_fwd_f32", {tensor_23617}, {tensor_23619}, nullptr, 0, "TPC98035");

    /*************
     * Convolution98036 node
     * inputs: [tensor_23619[8, 704, 320, 2](dtype=float32), tensor_23620[1, 8, 3, 3](dtype=float32)]
     * output: [tensor_23621[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23620 tensor
    unsigned tensor_23620_sizes[] = {1,8,3,3};
    unsigned tensor_23620 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23620",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23620_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23621 tensor
    unsigned tensor_23621_sizes[] = {1,704,320,2};
    unsigned tensor_23621 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23621",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23621_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98036_params[] = {3,0,0,0,3,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,55,13,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,1,20,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23619, tensor_23620}, {tensor_23621}, (void*)Convolution98036_params, 88, "Convolution98036");

    /*************
     * Reshape98037 node
     * inputs: [tensor_23622[1](dtype=float32)]
     * output: [tensor_23624[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23622 tensor
    unsigned tensor_23622_sizes[] = {1};
    unsigned tensor_23622 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23622",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23622_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23624 tensor
    unsigned tensor_23624_sizes[] = {1,1,1,1};
    unsigned tensor_23624 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23624",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23624_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23622}, {tensor_23624}, nullptr, 0, "Reshape98037");

    /*************
     * TPC98038 node
     * inputs: [tensor_23621[1, 704, 320, 2](dtype=float32), tensor_23624[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23623[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23623 tensor
    unsigned tensor_23623_sizes[] = {1,704,320,2};
    unsigned tensor_23623 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23623",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23623_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23621, tensor_23624}, {tensor_23623}, nullptr, 0, "TPC98038");

    /*************
     * Reshape98039 node
     * inputs: [tensor_23456[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23627[2304, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23627 tensor
    unsigned tensor_23627_sizes[] = {2304,1,1,1};
    unsigned tensor_23627 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23627",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23627_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23456}, {tensor_23627}, nullptr, 0, "Reshape98039");

    /*************
     * TPC98040 node
     * inputs: [tensor_23627[2304, 1, 1, 1](dtype=float32)]
     * output: [tensor_23628[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23628 tensor
    unsigned tensor_23628_sizes[] = {1,1,1,1};
    unsigned tensor_23628 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23628",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23628_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98040_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23627}, {tensor_23628}, (void*)TPC98040_params, 4, "TPC98040");

    /*************
     * Reshape98041 node
     * inputs: [tensor_23628[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23626[1](dtype=float32)]
     *************/

    // create tensor_23626 tensor
    unsigned tensor_23626_sizes[] = {1};
    unsigned tensor_23626 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23626",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23626_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23628}, {tensor_23626}, nullptr, 0, "Reshape98041");

    /*************
     * TPC98042 node
     * inputs: [tensor_23626[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23630[1](dtype=float32)]
     *************/

    // create tensor_23630 tensor
    unsigned tensor_23630_sizes[] = {1};
    unsigned tensor_23630 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23630",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23630_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23626, tensor_23523}, {tensor_23630}, nullptr, 0, "TPC98042");

    /*************
     * Reshape98043 node
     * inputs: [tensor_23551[8, 24, 3, 3](dtype=float32)]
     * output: [tensor_23632[1728, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23632 tensor
    unsigned tensor_23632_sizes[] = {1728,1,1,1};
    unsigned tensor_23632 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23632",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23632_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23551}, {tensor_23632}, nullptr, 0, "Reshape98043");

    /*************
     * TPC98044 node
     * inputs: [tensor_23632[1728, 1, 1, 1](dtype=float32)]
     * output: [tensor_23633[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23633 tensor
    unsigned tensor_23633_sizes[] = {1,1,1,1};
    unsigned tensor_23633 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23633",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23633_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98044_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23632}, {tensor_23633}, (void*)TPC98044_params, 4, "TPC98044");

    /*************
     * Reshape98045 node
     * inputs: [tensor_23633[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23631[1](dtype=float32)]
     *************/

    // create tensor_23631 tensor
    unsigned tensor_23631_sizes[] = {1};
    unsigned tensor_23631 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23631",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23631_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23633}, {tensor_23631}, nullptr, 0, "Reshape98045");

    /*************
     * TPC98046 node
     * inputs: [tensor_23631[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23635[1](dtype=float32)]
     *************/

    // create tensor_23635 tensor
    unsigned tensor_23635_sizes[] = {1};
    unsigned tensor_23635 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23635",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23635_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23631, tensor_23523}, {tensor_23635}, nullptr, 0, "TPC98046");

    /*************
     * Reshape98047 node
     * inputs: [tensor_23444[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23637[576, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23637 tensor
    unsigned tensor_23637_sizes[] = {576,1,1,1};
    unsigned tensor_23637 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23637",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23637_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23444}, {tensor_23637}, nullptr, 0, "Reshape98047");

    /*************
     * TPC98048 node
     * inputs: [tensor_23637[576, 1, 1, 1](dtype=float32)]
     * output: [tensor_23638[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23638 tensor
    unsigned tensor_23638_sizes[] = {1,1,1,1};
    unsigned tensor_23638 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23638",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23638_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98048_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23637}, {tensor_23638}, (void*)TPC98048_params, 4, "TPC98048");

    /*************
     * Reshape98049 node
     * inputs: [tensor_23638[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23636[1](dtype=float32)]
     *************/

    // create tensor_23636 tensor
    unsigned tensor_23636_sizes[] = {1};
    unsigned tensor_23636 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23636",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23636_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23638}, {tensor_23636}, nullptr, 0, "Reshape98049");

    /*************
     * TPC98050 node
     * inputs: [tensor_23636[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23640[1](dtype=float32)]
     *************/

    // create tensor_23640 tensor
    unsigned tensor_23640_sizes[] = {1};
    unsigned tensor_23640 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23640",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23640_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23636, tensor_23523}, {tensor_23640}, nullptr, 0, "TPC98050");

    /*************
     * Reshape98051 node
     * inputs: [tensor_23462[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23642[36864, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23642 tensor
    unsigned tensor_23642_sizes[] = {36864,1,1,1};
    unsigned tensor_23642 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23642",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23642_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23462}, {tensor_23642}, nullptr, 0, "Reshape98051");

    /*************
     * TPC98052 node
     * inputs: [tensor_23642[36864, 1, 1, 1](dtype=float32)]
     * output: [tensor_23643[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23643 tensor
    unsigned tensor_23643_sizes[] = {1,1,1,1};
    unsigned tensor_23643 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23643",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23643_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98052_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23642}, {tensor_23643}, (void*)TPC98052_params, 4, "TPC98052");

    /*************
     * Reshape98053 node
     * inputs: [tensor_23643[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23641[1](dtype=float32)]
     *************/

    // create tensor_23641 tensor
    unsigned tensor_23641_sizes[] = {1};
    unsigned tensor_23641 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23641",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23641_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23643}, {tensor_23641}, nullptr, 0, "Reshape98053");

    /*************
     * TPC98054 node
     * inputs: [tensor_23641[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23645[1](dtype=float32)]
     *************/

    // create tensor_23645 tensor
    unsigned tensor_23645_sizes[] = {1};
    unsigned tensor_23645 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23645",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23645_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23641, tensor_23523}, {tensor_23645}, nullptr, 0, "TPC98054");

    /*************
     * Reshape98055 node
     * inputs: [tensor_23468[24, 112, 3, 3](dtype=float32)]
     * output: [tensor_23647[24192, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23647 tensor
    unsigned tensor_23647_sizes[] = {24192,1,1,1};
    unsigned tensor_23647 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23647",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23647_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23468}, {tensor_23647}, nullptr, 0, "Reshape98055");

    /*************
     * TPC98056 node
     * inputs: [tensor_23647[24192, 1, 1, 1](dtype=float32)]
     * output: [tensor_23648[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23648 tensor
    unsigned tensor_23648_sizes[] = {1,1,1,1};
    unsigned tensor_23648 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23648",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23648_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98056_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23647}, {tensor_23648}, (void*)TPC98056_params, 4, "TPC98056");

    /*************
     * Reshape98057 node
     * inputs: [tensor_23648[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23646[1](dtype=float32)]
     *************/

    // create tensor_23646 tensor
    unsigned tensor_23646_sizes[] = {1};
    unsigned tensor_23646 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23646",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23646_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23648}, {tensor_23646}, nullptr, 0, "Reshape98057");

    /*************
     * TPC98058 node
     * inputs: [tensor_23646[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23650[1](dtype=float32)]
     *************/

    // create tensor_23650 tensor
    unsigned tensor_23650_sizes[] = {1};
    unsigned tensor_23650 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23650",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23650_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23646, tensor_23523}, {tensor_23650}, nullptr, 0, "TPC98058");

    /*************
     * Reshape98059 node
     * inputs: [tensor_23418[24, 24, 3, 3](dtype=float32)]
     * output: [tensor_23652[5184, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23652 tensor
    unsigned tensor_23652_sizes[] = {5184,1,1,1};
    unsigned tensor_23652 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23652",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23652_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23418}, {tensor_23652}, nullptr, 0, "Reshape98059");

    /*************
     * TPC98060 node
     * inputs: [tensor_23652[5184, 1, 1, 1](dtype=float32)]
     * output: [tensor_23653[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23653 tensor
    unsigned tensor_23653_sizes[] = {1,1,1,1};
    unsigned tensor_23653 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23653",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23653_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98060_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23652}, {tensor_23653}, (void*)TPC98060_params, 4, "TPC98060");

    /*************
     * Reshape98061 node
     * inputs: [tensor_23653[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23651[1](dtype=float32)]
     *************/

    // create tensor_23651 tensor
    unsigned tensor_23651_sizes[] = {1};
    unsigned tensor_23651 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23651",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23651_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23653}, {tensor_23651}, nullptr, 0, "Reshape98061");

    /*************
     * TPC98062 node
     * inputs: [tensor_23651[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23655[1](dtype=float32)]
     *************/

    // create tensor_23655 tensor
    unsigned tensor_23655_sizes[] = {1};
    unsigned tensor_23655 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23655",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23655_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23651, tensor_23523}, {tensor_23655}, nullptr, 0, "TPC98062");

    /*************
     * Reshape98063 node
     * inputs: [tensor_23424[96, 96, 3, 3](dtype=float32)]
     * output: [tensor_23657[82944, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23657 tensor
    unsigned tensor_23657_sizes[] = {82944,1,1,1};
    unsigned tensor_23657 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23657",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23657_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23424}, {tensor_23657}, nullptr, 0, "Reshape98063");

    /*************
     * TPC98064 node
     * inputs: [tensor_23657[82944, 1, 1, 1](dtype=float32)]
     * output: [tensor_23658[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23658 tensor
    unsigned tensor_23658_sizes[] = {1,1,1,1};
    unsigned tensor_23658 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23658",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23658_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98064_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23657}, {tensor_23658}, (void*)TPC98064_params, 4, "TPC98064");

    /*************
     * Reshape98065 node
     * inputs: [tensor_23658[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23656[1](dtype=float32)]
     *************/

    // create tensor_23656 tensor
    unsigned tensor_23656_sizes[] = {1};
    unsigned tensor_23656 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23656",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23656_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23658}, {tensor_23656}, nullptr, 0, "Reshape98065");

    /*************
     * TPC98066 node
     * inputs: [tensor_23656[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23660[1](dtype=float32)]
     *************/

    // create tensor_23660 tensor
    unsigned tensor_23660_sizes[] = {1};
    unsigned tensor_23660 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23660",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23660_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23656, tensor_23523}, {tensor_23660}, nullptr, 0, "TPC98066");

    /*************
     * Reshape98067 node
     * inputs: [tensor_23426[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23662[576, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23662 tensor
    unsigned tensor_23662_sizes[] = {576,1,1,1};
    unsigned tensor_23662 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23662",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23662_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23426}, {tensor_23662}, nullptr, 0, "Reshape98067");

    /*************
     * TPC98068 node
     * inputs: [tensor_23662[576, 1, 1, 1](dtype=float32)]
     * output: [tensor_23663[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23663 tensor
    unsigned tensor_23663_sizes[] = {1,1,1,1};
    unsigned tensor_23663 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23663",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23663_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98068_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23662}, {tensor_23663}, (void*)TPC98068_params, 4, "TPC98068");

    /*************
     * Reshape98069 node
     * inputs: [tensor_23663[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23661[1](dtype=float32)]
     *************/

    // create tensor_23661 tensor
    unsigned tensor_23661_sizes[] = {1};
    unsigned tensor_23661 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23661",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23661_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23663}, {tensor_23661}, nullptr, 0, "Reshape98069");

    /*************
     * TPC98070 node
     * inputs: [tensor_23661[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23665[1](dtype=float32)]
     *************/

    // create tensor_23665 tensor
    unsigned tensor_23665_sizes[] = {1};
    unsigned tensor_23665 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23665",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23665_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23661, tensor_23523}, {tensor_23665}, nullptr, 0, "TPC98070");

    /*************
     * Reshape98071 node
     * inputs: [tensor_23434[16, 8, 3, 3](dtype=float32)]
     * output: [tensor_23667[1152, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23667 tensor
    unsigned tensor_23667_sizes[] = {1152,1,1,1};
    unsigned tensor_23667 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23667",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23667_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23434}, {tensor_23667}, nullptr, 0, "Reshape98071");

    /*************
     * TPC98072 node
     * inputs: [tensor_23667[1152, 1, 1, 1](dtype=float32)]
     * output: [tensor_23668[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23668 tensor
    unsigned tensor_23668_sizes[] = {1,1,1,1};
    unsigned tensor_23668 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23668",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23668_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98072_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23667}, {tensor_23668}, (void*)TPC98072_params, 4, "TPC98072");

    /*************
     * Reshape98073 node
     * inputs: [tensor_23668[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23666[1](dtype=float32)]
     *************/

    // create tensor_23666 tensor
    unsigned tensor_23666_sizes[] = {1};
    unsigned tensor_23666 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23666",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23666_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23668}, {tensor_23666}, nullptr, 0, "Reshape98073");

    /*************
     * TPC98074 node
     * inputs: [tensor_23666[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23670[1](dtype=float32)]
     *************/

    // create tensor_23670 tensor
    unsigned tensor_23670_sizes[] = {1};
    unsigned tensor_23670 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23670",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23670_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23666, tensor_23523}, {tensor_23670}, nullptr, 0, "TPC98074");

    /*************
     * Reshape98075 node
     * inputs: [tensor_23477[16, 40, 3, 3](dtype=float32)]
     * output: [tensor_23672[5760, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23672 tensor
    unsigned tensor_23672_sizes[] = {5760,1,1,1};
    unsigned tensor_23672 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23672",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23672_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23477}, {tensor_23672}, nullptr, 0, "Reshape98075");

    /*************
     * TPC98076 node
     * inputs: [tensor_23672[5760, 1, 1, 1](dtype=float32)]
     * output: [tensor_23673[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23673 tensor
    unsigned tensor_23673_sizes[] = {1,1,1,1};
    unsigned tensor_23673 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23673",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23673_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98076_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23672}, {tensor_23673}, (void*)TPC98076_params, 4, "TPC98076");

    /*************
     * Reshape98077 node
     * inputs: [tensor_23673[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23671[1](dtype=float32)]
     *************/

    // create tensor_23671 tensor
    unsigned tensor_23671_sizes[] = {1};
    unsigned tensor_23671 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23671",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23671_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23673}, {tensor_23671}, nullptr, 0, "Reshape98077");

    /*************
     * TPC98078 node
     * inputs: [tensor_23671[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23675[1](dtype=float32)]
     *************/

    // create tensor_23675 tensor
    unsigned tensor_23675_sizes[] = {1};
    unsigned tensor_23675 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23675",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23675_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23671, tensor_23523}, {tensor_23675}, nullptr, 0, "TPC98078");

    /*************
     * Reshape98079 node
     * inputs: [tensor_23436[8, 24, 3, 3](dtype=float32)]
     * output: [tensor_23677[1728, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23677 tensor
    unsigned tensor_23677_sizes[] = {1728,1,1,1};
    unsigned tensor_23677 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23677",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23677_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23436}, {tensor_23677}, nullptr, 0, "Reshape98079");

    /*************
     * TPC98080 node
     * inputs: [tensor_23677[1728, 1, 1, 1](dtype=float32)]
     * output: [tensor_23678[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23678 tensor
    unsigned tensor_23678_sizes[] = {1,1,1,1};
    unsigned tensor_23678 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23678",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23678_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98080_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23677}, {tensor_23678}, (void*)TPC98080_params, 4, "TPC98080");

    /*************
     * Reshape98081 node
     * inputs: [tensor_23678[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23676[1](dtype=float32)]
     *************/

    // create tensor_23676 tensor
    unsigned tensor_23676_sizes[] = {1};
    unsigned tensor_23676 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23676",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23676_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23678}, {tensor_23676}, nullptr, 0, "Reshape98081");

    /*************
     * TPC98082 node
     * inputs: [tensor_23676[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23680[1](dtype=float32)]
     *************/

    // create tensor_23680 tensor
    unsigned tensor_23680_sizes[] = {1};
    unsigned tensor_23680 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23680",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23680_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23676, tensor_23523}, {tensor_23680}, nullptr, 0, "TPC98082");

    /*************
     * Reshape98083 node
     * inputs: [tensor_23448[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23682[576, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23682 tensor
    unsigned tensor_23682_sizes[] = {576,1,1,1};
    unsigned tensor_23682 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23682",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23682_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23448}, {tensor_23682}, nullptr, 0, "Reshape98083");

    /*************
     * TPC98084 node
     * inputs: [tensor_23682[576, 1, 1, 1](dtype=float32)]
     * output: [tensor_23683[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23683 tensor
    unsigned tensor_23683_sizes[] = {1,1,1,1};
    unsigned tensor_23683 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23683",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23683_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98084_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23682}, {tensor_23683}, (void*)TPC98084_params, 4, "TPC98084");

    /*************
     * Reshape98085 node
     * inputs: [tensor_23683[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23681[1](dtype=float32)]
     *************/

    // create tensor_23681 tensor
    unsigned tensor_23681_sizes[] = {1};
    unsigned tensor_23681 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23681",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23681_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23683}, {tensor_23681}, nullptr, 0, "Reshape98085");

    /*************
     * TPC98086 node
     * inputs: [tensor_23681[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23685[1](dtype=float32)]
     *************/

    // create tensor_23685 tensor
    unsigned tensor_23685_sizes[] = {1};
    unsigned tensor_23685 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23685",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23685_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23681, tensor_23523}, {tensor_23685}, nullptr, 0, "TPC98086");

    /*************
     * Reshape98087 node
     * inputs: [tensor_23515[1, 8, 3, 3](dtype=float32)]
     * output: [tensor_23687[72, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23687 tensor
    unsigned tensor_23687_sizes[] = {72,1,1,1};
    unsigned tensor_23687 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23687",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23687_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23515}, {tensor_23687}, nullptr, 0, "Reshape98087");

    /*************
     * TPC98088 node
     * inputs: [tensor_23687[72, 1, 1, 1](dtype=float32)]
     * output: [tensor_23688[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23688 tensor
    unsigned tensor_23688_sizes[] = {1,1,1,1};
    unsigned tensor_23688 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23688",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23688_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98088_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23687}, {tensor_23688}, (void*)TPC98088_params, 4, "TPC98088");

    /*************
     * Reshape98089 node
     * inputs: [tensor_23688[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23686[1](dtype=float32)]
     *************/

    // create tensor_23686 tensor
    unsigned tensor_23686_sizes[] = {1};
    unsigned tensor_23686 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23686",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23686_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23688}, {tensor_23686}, nullptr, 0, "Reshape98089");

    /*************
     * TPC98090 node
     * inputs: [tensor_23686[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23690[1](dtype=float32)]
     *************/

    // create tensor_23690 tensor
    unsigned tensor_23690_sizes[] = {1};
    unsigned tensor_23690 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23690",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23690_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23686, tensor_23523}, {tensor_23690}, nullptr, 0, "TPC98090");

    /*************
     * Reshape98091 node
     * inputs: [tensor_23440[96, 64, 3, 3](dtype=float32)]
     * output: [tensor_23692[55296, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23692 tensor
    unsigned tensor_23692_sizes[] = {55296,1,1,1};
    unsigned tensor_23692 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23692",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23692_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23440}, {tensor_23692}, nullptr, 0, "Reshape98091");

    /*************
     * TPC98092 node
     * inputs: [tensor_23692[55296, 1, 1, 1](dtype=float32)]
     * output: [tensor_23693[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23693 tensor
    unsigned tensor_23693_sizes[] = {1,1,1,1};
    unsigned tensor_23693 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23693",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23693_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98092_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23692}, {tensor_23693}, (void*)TPC98092_params, 4, "TPC98092");

    /*************
     * Reshape98093 node
     * inputs: [tensor_23693[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23691[1](dtype=float32)]
     *************/

    // create tensor_23691 tensor
    unsigned tensor_23691_sizes[] = {1};
    unsigned tensor_23691 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23691",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23691_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23693}, {tensor_23691}, nullptr, 0, "Reshape98093");

    /*************
     * TPC98094 node
     * inputs: [tensor_23691[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23695[1](dtype=float32)]
     *************/

    // create tensor_23695 tensor
    unsigned tensor_23695_sizes[] = {1};
    unsigned tensor_23695 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23695",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23695_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23691, tensor_23523}, {tensor_23695}, nullptr, 0, "TPC98094");

    /*************
     * Reshape98095 node
     * inputs: [tensor_23452[48, 192, 3, 3](dtype=float32)]
     * output: [tensor_23697[82944, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23697 tensor
    unsigned tensor_23697_sizes[] = {82944,1,1,1};
    unsigned tensor_23697 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23697",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23697_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23452}, {tensor_23697}, nullptr, 0, "Reshape98095");

    /*************
     * TPC98096 node
     * inputs: [tensor_23697[82944, 1, 1, 1](dtype=float32)]
     * output: [tensor_23698[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23698 tensor
    unsigned tensor_23698_sizes[] = {1,1,1,1};
    unsigned tensor_23698 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23698",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23698_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98096_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23697}, {tensor_23698}, (void*)TPC98096_params, 4, "TPC98096");

    /*************
     * Reshape98097 node
     * inputs: [tensor_23698[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23696[1](dtype=float32)]
     *************/

    // create tensor_23696 tensor
    unsigned tensor_23696_sizes[] = {1};
    unsigned tensor_23696 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23696",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23696_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23698}, {tensor_23696}, nullptr, 0, "Reshape98097");

    /*************
     * TPC98098 node
     * inputs: [tensor_23696[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23700[1](dtype=float32)]
     *************/

    // create tensor_23700 tensor
    unsigned tensor_23700_sizes[] = {1};
    unsigned tensor_23700 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23700",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23700_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23696, tensor_23523}, {tensor_23700}, nullptr, 0, "TPC98098");

    /*************
     * Reshape98099 node
     * inputs: [tensor_23598[216, 216, 3, 3](dtype=float32)]
     * output: [tensor_23702[419904, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23702 tensor
    unsigned tensor_23702_sizes[] = {419904,1,1,1};
    unsigned tensor_23702 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23702",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23702_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23598}, {tensor_23702}, nullptr, 0, "Reshape98099");

    /*************
     * TPC98100 node
     * inputs: [tensor_23702[419904, 1, 1, 1](dtype=float32)]
     * output: [tensor_23703[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23703 tensor
    unsigned tensor_23703_sizes[] = {1,1,1,1};
    unsigned tensor_23703 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23703",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23703_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98100_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23702}, {tensor_23703}, (void*)TPC98100_params, 4, "TPC98100");

    /*************
     * Reshape98101 node
     * inputs: [tensor_23703[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23701[1](dtype=float32)]
     *************/

    // create tensor_23701 tensor
    unsigned tensor_23701_sizes[] = {1};
    unsigned tensor_23701 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23701",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23701_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23703}, {tensor_23701}, nullptr, 0, "Reshape98101");

    /*************
     * TPC98102 node
     * inputs: [tensor_23701[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23705[1](dtype=float32)]
     *************/

    // create tensor_23705 tensor
    unsigned tensor_23705_sizes[] = {1};
    unsigned tensor_23705 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23705",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23705_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23701, tensor_23523}, {tensor_23705}, nullptr, 0, "TPC98102");

    /*************
     * Reshape98103 node
     * inputs: [tensor_23416[216, 144, 3, 3](dtype=float32)]
     * output: [tensor_23707[279936, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23707 tensor
    unsigned tensor_23707_sizes[] = {279936,1,1,1};
    unsigned tensor_23707 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23707",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23707_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23416}, {tensor_23707}, nullptr, 0, "Reshape98103");

    /*************
     * TPC98104 node
     * inputs: [tensor_23707[279936, 1, 1, 1](dtype=float32)]
     * output: [tensor_23708[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23708 tensor
    unsigned tensor_23708_sizes[] = {1,1,1,1};
    unsigned tensor_23708 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23708",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23708_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98104_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23707}, {tensor_23708}, (void*)TPC98104_params, 4, "TPC98104");

    /*************
     * Reshape98105 node
     * inputs: [tensor_23708[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23706[1](dtype=float32)]
     *************/

    // create tensor_23706 tensor
    unsigned tensor_23706_sizes[] = {1};
    unsigned tensor_23706 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23706",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23706_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23708}, {tensor_23706}, nullptr, 0, "Reshape98105");

    /*************
     * TPC98106 node
     * inputs: [tensor_23706[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23710[1](dtype=float32)]
     *************/

    // create tensor_23710 tensor
    unsigned tensor_23710_sizes[] = {1};
    unsigned tensor_23710 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23710",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23710_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23706, tensor_23523}, {tensor_23710}, nullptr, 0, "TPC98106");

    /*************
     * Reshape98107 node
     * inputs: [tensor_23422[144, 432, 3, 3](dtype=float32)]
     * output: [tensor_23712[559872, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23712 tensor
    unsigned tensor_23712_sizes[] = {559872,1,1,1};
    unsigned tensor_23712 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23712",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23712_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23422}, {tensor_23712}, nullptr, 0, "Reshape98107");

    /*************
     * TPC98108 node
     * inputs: [tensor_23712[559872, 1, 1, 1](dtype=float32)]
     * output: [tensor_23713[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23713 tensor
    unsigned tensor_23713_sizes[] = {1,1,1,1};
    unsigned tensor_23713 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23713",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23713_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98108_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23712}, {tensor_23713}, (void*)TPC98108_params, 4, "TPC98108");

    /*************
     * Reshape98109 node
     * inputs: [tensor_23713[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23711[1](dtype=float32)]
     *************/

    // create tensor_23711 tensor
    unsigned tensor_23711_sizes[] = {1};
    unsigned tensor_23711 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23711",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23711_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23713}, {tensor_23711}, nullptr, 0, "Reshape98109");

    /*************
     * TPC98110 node
     * inputs: [tensor_23711[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23715[1](dtype=float32)]
     *************/

    // create tensor_23715 tensor
    unsigned tensor_23715_sizes[] = {1};
    unsigned tensor_23715 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23715",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23715_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23711, tensor_23523}, {tensor_23715}, nullptr, 0, "TPC98110");

    /*************
     * Reshape98111 node
     * inputs: [tensor_23428[144, 144, 3, 3](dtype=float32)]
     * output: [tensor_23717[186624, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23717 tensor
    unsigned tensor_23717_sizes[] = {186624,1,1,1};
    unsigned tensor_23717 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23717",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23717_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23428}, {tensor_23717}, nullptr, 0, "Reshape98111");

    /*************
     * TPC98112 node
     * inputs: [tensor_23717[186624, 1, 1, 1](dtype=float32)]
     * output: [tensor_23718[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23718 tensor
    unsigned tensor_23718_sizes[] = {1,1,1,1};
    unsigned tensor_23718 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23718",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23718_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98112_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23717}, {tensor_23718}, (void*)TPC98112_params, 4, "TPC98112");

    /*************
     * Reshape98113 node
     * inputs: [tensor_23718[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23716[1](dtype=float32)]
     *************/

    // create tensor_23716 tensor
    unsigned tensor_23716_sizes[] = {1};
    unsigned tensor_23716 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23716",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23716_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23718}, {tensor_23716}, nullptr, 0, "Reshape98113");

    /*************
     * TPC98114 node
     * inputs: [tensor_23716[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23720[1](dtype=float32)]
     *************/

    // create tensor_23720 tensor
    unsigned tensor_23720_sizes[] = {1};
    unsigned tensor_23720 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23720",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23720_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23716, tensor_23523}, {tensor_23720}, nullptr, 0, "TPC98114");

    /*************
     * Reshape98115 node
     * inputs: [tensor_23438[96, 96, 3, 3](dtype=float32)]
     * output: [tensor_23722[82944, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23722 tensor
    unsigned tensor_23722_sizes[] = {82944,1,1,1};
    unsigned tensor_23722 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23722",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23722_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23438}, {tensor_23722}, nullptr, 0, "Reshape98115");

    /*************
     * TPC98116 node
     * inputs: [tensor_23722[82944, 1, 1, 1](dtype=float32)]
     * output: [tensor_23723[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23723 tensor
    unsigned tensor_23723_sizes[] = {1,1,1,1};
    unsigned tensor_23723 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23723",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23723_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98116_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23722}, {tensor_23723}, (void*)TPC98116_params, 4, "TPC98116");

    /*************
     * Reshape98117 node
     * inputs: [tensor_23723[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23721[1](dtype=float32)]
     *************/

    // create tensor_23721 tensor
    unsigned tensor_23721_sizes[] = {1};
    unsigned tensor_23721 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23721",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23721_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23723}, {tensor_23721}, nullptr, 0, "Reshape98117");

    /*************
     * TPC98118 node
     * inputs: [tensor_23721[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23725[1](dtype=float32)]
     *************/

    // create tensor_23725 tensor
    unsigned tensor_23725_sizes[] = {1};
    unsigned tensor_23725 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23725",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23725_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23721, tensor_23523}, {tensor_23725}, nullptr, 0, "TPC98118");

    /*************
     * Reshape98119 node
     * inputs: [tensor_23450[144, 96, 3, 3](dtype=float32)]
     * output: [tensor_23727[124416, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23727 tensor
    unsigned tensor_23727_sizes[] = {124416,1,1,1};
    unsigned tensor_23727 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23727",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23727_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23450}, {tensor_23727}, nullptr, 0, "Reshape98119");

    /*************
     * TPC98120 node
     * inputs: [tensor_23727[124416, 1, 1, 1](dtype=float32)]
     * output: [tensor_23728[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23728 tensor
    unsigned tensor_23728_sizes[] = {1,1,1,1};
    unsigned tensor_23728 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23728",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23728_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98120_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23727}, {tensor_23728}, (void*)TPC98120_params, 4, "TPC98120");

    /*************
     * Reshape98121 node
     * inputs: [tensor_23728[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23726[1](dtype=float32)]
     *************/

    // create tensor_23726 tensor
    unsigned tensor_23726_sizes[] = {1};
    unsigned tensor_23726 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23726",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23726_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23728}, {tensor_23726}, nullptr, 0, "Reshape98121");

    /*************
     * TPC98122 node
     * inputs: [tensor_23726[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23730[1](dtype=float32)]
     *************/

    // create tensor_23730 tensor
    unsigned tensor_23730_sizes[] = {1};
    unsigned tensor_23730 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23730",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23730_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23726, tensor_23523}, {tensor_23730}, nullptr, 0, "TPC98122");

    /*************
     * Reshape98123 node
     * inputs: [tensor_23432[8, 8, 3, 3](dtype=float32)]
     * output: [tensor_23732[576, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23732 tensor
    unsigned tensor_23732_sizes[] = {576,1,1,1};
    unsigned tensor_23732 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23732",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23732_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23432}, {tensor_23732}, nullptr, 0, "Reshape98123");

    /*************
     * TPC98124 node
     * inputs: [tensor_23732[576, 1, 1, 1](dtype=float32)]
     * output: [tensor_23733[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23733 tensor
    unsigned tensor_23733_sizes[] = {1,1,1,1};
    unsigned tensor_23733 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23733",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23733_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98124_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23732}, {tensor_23733}, (void*)TPC98124_params, 4, "TPC98124");

    /*************
     * Reshape98125 node
     * inputs: [tensor_23733[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23731[1](dtype=float32)]
     *************/

    // create tensor_23731 tensor
    unsigned tensor_23731_sizes[] = {1};
    unsigned tensor_23731 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23731",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23731_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23733}, {tensor_23731}, nullptr, 0, "Reshape98125");

    /*************
     * TPC98126 node
     * inputs: [tensor_23731[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23735[1](dtype=float32)]
     *************/

    // create tensor_23735 tensor
    unsigned tensor_23735_sizes[] = {1};
    unsigned tensor_23735 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23735",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23735_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23731, tensor_23523}, {tensor_23735}, nullptr, 0, "TPC98126");

    /*************
     * Reshape98127 node
     * inputs: [tensor_23446[8, 1, 3, 3](dtype=float32)]
     * output: [tensor_23737[72, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23737 tensor
    unsigned tensor_23737_sizes[] = {72,1,1,1};
    unsigned tensor_23737 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23737",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23737_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23446}, {tensor_23737}, nullptr, 0, "Reshape98127");

    /*************
     * TPC98128 node
     * inputs: [tensor_23737[72, 1, 1, 1](dtype=float32)]
     * output: [tensor_23738[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23738 tensor
    unsigned tensor_23738_sizes[] = {1,1,1,1};
    unsigned tensor_23738 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23738",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23738_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98128_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23737}, {tensor_23738}, (void*)TPC98128_params, 4, "TPC98128");

    /*************
     * Reshape98129 node
     * inputs: [tensor_23738[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23736[1](dtype=float32)]
     *************/

    // create tensor_23736 tensor
    unsigned tensor_23736_sizes[] = {1};
    unsigned tensor_23736 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23736",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23736_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23738}, {tensor_23736}, nullptr, 0, "Reshape98129");

    /*************
     * TPC98130 node
     * inputs: [tensor_23736[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23740[1](dtype=float32)]
     *************/

    // create tensor_23740 tensor
    unsigned tensor_23740_sizes[] = {1};
    unsigned tensor_23740 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23740",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23740_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23736, tensor_23523}, {tensor_23740}, nullptr, 0, "TPC98130");

    /*************
     * Reshape98131 node
     * inputs: [tensor_23471[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23742[2304, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23742 tensor
    unsigned tensor_23742_sizes[] = {2304,1,1,1};
    unsigned tensor_23742 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23742",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23742_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23471}, {tensor_23742}, nullptr, 0, "Reshape98131");

    /*************
     * TPC98132 node
     * inputs: [tensor_23742[2304, 1, 1, 1](dtype=float32)]
     * output: [tensor_23743[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23743 tensor
    unsigned tensor_23743_sizes[] = {1,1,1,1};
    unsigned tensor_23743 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23743",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23743_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98132_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23742}, {tensor_23743}, (void*)TPC98132_params, 4, "TPC98132");

    /*************
     * Reshape98133 node
     * inputs: [tensor_23743[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23741[1](dtype=float32)]
     *************/

    // create tensor_23741 tensor
    unsigned tensor_23741_sizes[] = {1};
    unsigned tensor_23741 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23741",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23741_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23743}, {tensor_23741}, nullptr, 0, "Reshape98133");

    /*************
     * TPC98134 node
     * inputs: [tensor_23741[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23745[1](dtype=float32)]
     *************/

    // create tensor_23745 tensor
    unsigned tensor_23745_sizes[] = {1};
    unsigned tensor_23745 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23745",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23745_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23741, tensor_23523}, {tensor_23745}, nullptr, 0, "TPC98134");

    /*************
     * Reshape98135 node
     * inputs: [tensor_23458[32, 32, 3, 3](dtype=float32)]
     * output: [tensor_23747[9216, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23747 tensor
    unsigned tensor_23747_sizes[] = {9216,1,1,1};
    unsigned tensor_23747 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23747",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23747_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23458}, {tensor_23747}, nullptr, 0, "Reshape98135");

    /*************
     * TPC98136 node
     * inputs: [tensor_23747[9216, 1, 1, 1](dtype=float32)]
     * output: [tensor_23748[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23748 tensor
    unsigned tensor_23748_sizes[] = {1,1,1,1};
    unsigned tensor_23748 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23748",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23748_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98136_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23747}, {tensor_23748}, (void*)TPC98136_params, 4, "TPC98136");

    /*************
     * Reshape98137 node
     * inputs: [tensor_23748[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23746[1](dtype=float32)]
     *************/

    // create tensor_23746 tensor
    unsigned tensor_23746_sizes[] = {1};
    unsigned tensor_23746 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23746",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23746_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23748}, {tensor_23746}, nullptr, 0, "Reshape98137");

    /*************
     * TPC98138 node
     * inputs: [tensor_23746[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23750[1](dtype=float32)]
     *************/

    // create tensor_23750 tensor
    unsigned tensor_23750_sizes[] = {1};
    unsigned tensor_23750 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23750",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23750_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23746, tensor_23523}, {tensor_23750}, nullptr, 0, "TPC98138");

    /*************
     * Reshape98139 node
     * inputs: [tensor_23430[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23752[36864, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23752 tensor
    unsigned tensor_23752_sizes[] = {36864,1,1,1};
    unsigned tensor_23752 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23752",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23752_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23430}, {tensor_23752}, nullptr, 0, "Reshape98139");

    /*************
     * TPC98140 node
     * inputs: [tensor_23752[36864, 1, 1, 1](dtype=float32)]
     * output: [tensor_23753[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23753 tensor
    unsigned tensor_23753_sizes[] = {1,1,1,1};
    unsigned tensor_23753 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23753",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23753_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98140_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23752}, {tensor_23753}, (void*)TPC98140_params, 4, "TPC98140");

    /*************
     * Reshape98141 node
     * inputs: [tensor_23753[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23751[1](dtype=float32)]
     *************/

    // create tensor_23751 tensor
    unsigned tensor_23751_sizes[] = {1};
    unsigned tensor_23751 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23751",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23751_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23753}, {tensor_23751}, nullptr, 0, "Reshape98141");

    /*************
     * TPC98142 node
     * inputs: [tensor_23751[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23755[1](dtype=float32)]
     *************/

    // create tensor_23755 tensor
    unsigned tensor_23755_sizes[] = {1};
    unsigned tensor_23755 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23755",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23755_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23751, tensor_23523}, {tensor_23755}, nullptr, 0, "TPC98142");

    /*************
     * Reshape98143 node
     * inputs: [tensor_23460[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23757[36864, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23757 tensor
    unsigned tensor_23757_sizes[] = {36864,1,1,1};
    unsigned tensor_23757 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23757",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23757_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23460}, {tensor_23757}, nullptr, 0, "Reshape98143");

    /*************
     * TPC98144 node
     * inputs: [tensor_23757[36864, 1, 1, 1](dtype=float32)]
     * output: [tensor_23758[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23758 tensor
    unsigned tensor_23758_sizes[] = {1,1,1,1};
    unsigned tensor_23758 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23758",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23758_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98144_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23757}, {tensor_23758}, (void*)TPC98144_params, 4, "TPC98144");

    /*************
     * Reshape98145 node
     * inputs: [tensor_23758[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23756[1](dtype=float32)]
     *************/

    // create tensor_23756 tensor
    unsigned tensor_23756_sizes[] = {1};
    unsigned tensor_23756 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23756",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23756_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23758}, {tensor_23756}, nullptr, 0, "Reshape98145");

    /*************
     * TPC98146 node
     * inputs: [tensor_23756[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23760[1](dtype=float32)]
     *************/

    // create tensor_23760 tensor
    unsigned tensor_23760_sizes[] = {1};
    unsigned tensor_23760 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23760",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23760_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23756, tensor_23523}, {tensor_23760}, nullptr, 0, "TPC98146");

    /*************
     * Reshape98147 node
     * inputs: [tensor_23442[16, 8, 3, 3](dtype=float32)]
     * output: [tensor_23762[1152, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23762 tensor
    unsigned tensor_23762_sizes[] = {1152,1,1,1};
    unsigned tensor_23762 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23762",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23762_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23442}, {tensor_23762}, nullptr, 0, "Reshape98147");

    /*************
     * TPC98148 node
     * inputs: [tensor_23762[1152, 1, 1, 1](dtype=float32)]
     * output: [tensor_23763[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23763 tensor
    unsigned tensor_23763_sizes[] = {1,1,1,1};
    unsigned tensor_23763 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23763",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23763_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98148_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23762}, {tensor_23763}, (void*)TPC98148_params, 4, "TPC98148");

    /*************
     * Reshape98149 node
     * inputs: [tensor_23763[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23761[1](dtype=float32)]
     *************/

    // create tensor_23761 tensor
    unsigned tensor_23761_sizes[] = {1};
    unsigned tensor_23761 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23761",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23761_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23763}, {tensor_23761}, nullptr, 0, "Reshape98149");

    /*************
     * TPC98150 node
     * inputs: [tensor_23761[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23765[1](dtype=float32)]
     *************/

    // create tensor_23765 tensor
    unsigned tensor_23765_sizes[] = {1};
    unsigned tensor_23765 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23765",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23765_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23761, tensor_23523}, {tensor_23765}, nullptr, 0, "TPC98150");

    /*************
     * Reshape98151 node
     * inputs: [tensor_23464[48, 48, 3, 3](dtype=float32)]
     * output: [tensor_23767[20736, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23767 tensor
    unsigned tensor_23767_sizes[] = {20736,1,1,1};
    unsigned tensor_23767 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23767",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23767_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23464}, {tensor_23767}, nullptr, 0, "Reshape98151");

    /*************
     * TPC98152 node
     * inputs: [tensor_23767[20736, 1, 1, 1](dtype=float32)]
     * output: [tensor_23768[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23768 tensor
    unsigned tensor_23768_sizes[] = {1,1,1,1};
    unsigned tensor_23768 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23768",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23768_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98152_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23767}, {tensor_23768}, (void*)TPC98152_params, 4, "TPC98152");

    /*************
     * Reshape98153 node
     * inputs: [tensor_23768[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23766[1](dtype=float32)]
     *************/

    // create tensor_23766 tensor
    unsigned tensor_23766_sizes[] = {1};
    unsigned tensor_23766 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23766",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23766_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23768}, {tensor_23766}, nullptr, 0, "Reshape98153");

    /*************
     * TPC98154 node
     * inputs: [tensor_23766[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23770[1](dtype=float32)]
     *************/

    // create tensor_23770 tensor
    unsigned tensor_23770_sizes[] = {1};
    unsigned tensor_23770 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23770",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23770_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23766, tensor_23523}, {tensor_23770}, nullptr, 0, "TPC98154");

    /*************
     * Reshape98155 node
     * inputs: [tensor_23466[32, 16, 3, 3](dtype=float32)]
     * output: [tensor_23772[4608, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23772 tensor
    unsigned tensor_23772_sizes[] = {4608,1,1,1};
    unsigned tensor_23772 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23772",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23772_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23466}, {tensor_23772}, nullptr, 0, "Reshape98155");

    /*************
     * TPC98156 node
     * inputs: [tensor_23772[4608, 1, 1, 1](dtype=float32)]
     * output: [tensor_23773[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23773 tensor
    unsigned tensor_23773_sizes[] = {1,1,1,1};
    unsigned tensor_23773 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23773",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23773_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98156_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23772}, {tensor_23773}, (void*)TPC98156_params, 4, "TPC98156");

    /*************
     * Reshape98157 node
     * inputs: [tensor_23773[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23771[1](dtype=float32)]
     *************/

    // create tensor_23771 tensor
    unsigned tensor_23771_sizes[] = {1};
    unsigned tensor_23771 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23771",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23771_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23773}, {tensor_23771}, nullptr, 0, "Reshape98157");

    /*************
     * TPC98158 node
     * inputs: [tensor_23771[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23775[1](dtype=float32)]
     *************/

    // create tensor_23775 tensor
    unsigned tensor_23775_sizes[] = {1};
    unsigned tensor_23775 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23775",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23775_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23771, tensor_23523}, {tensor_23775}, nullptr, 0, "TPC98158");

    /*************
     * Reshape98159 node
     * inputs: [tensor_23481[64, 64, 3, 3](dtype=float32)]
     * output: [tensor_23777[36864, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23777 tensor
    unsigned tensor_23777_sizes[] = {36864,1,1,1};
    unsigned tensor_23777 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23777",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23777_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23481}, {tensor_23777}, nullptr, 0, "Reshape98159");

    /*************
     * TPC98160 node
     * inputs: [tensor_23777[36864, 1, 1, 1](dtype=float32)]
     * output: [tensor_23778[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23778 tensor
    unsigned tensor_23778_sizes[] = {1,1,1,1};
    unsigned tensor_23778 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23778",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23778_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98160_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23777}, {tensor_23778}, (void*)TPC98160_params, 4, "TPC98160");

    /*************
     * Reshape98161 node
     * inputs: [tensor_23778[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23776[1](dtype=float32)]
     *************/

    // create tensor_23776 tensor
    unsigned tensor_23776_sizes[] = {1};
    unsigned tensor_23776 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23776",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23776_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23778}, {tensor_23776}, nullptr, 0, "Reshape98161");

    /*************
     * TPC98162 node
     * inputs: [tensor_23776[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23780[1](dtype=float32)]
     *************/

    // create tensor_23780 tensor
    unsigned tensor_23780_sizes[] = {1};
    unsigned tensor_23780 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23780",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23780_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23776, tensor_23523}, {tensor_23780}, nullptr, 0, "TPC98162");

    /*************
     * Reshape98163 node
     * inputs: [tensor_23420[8, 1, 3, 3](dtype=float32)]
     * output: [tensor_23782[72, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23782 tensor
    unsigned tensor_23782_sizes[] = {72,1,1,1};
    unsigned tensor_23782 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23782",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23782_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23420}, {tensor_23782}, nullptr, 0, "Reshape98163");

    /*************
     * TPC98164 node
     * inputs: [tensor_23782[72, 1, 1, 1](dtype=float32)]
     * output: [tensor_23783[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23783 tensor
    unsigned tensor_23783_sizes[] = {1,1,1,1};
    unsigned tensor_23783 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23783",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23783_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98164_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23782}, {tensor_23783}, (void*)TPC98164_params, 4, "TPC98164");

    /*************
     * Reshape98165 node
     * inputs: [tensor_23783[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23781[1](dtype=float32)]
     *************/

    // create tensor_23781 tensor
    unsigned tensor_23781_sizes[] = {1};
    unsigned tensor_23781 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23781",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23781_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23783}, {tensor_23781}, nullptr, 0, "Reshape98165");

    /*************
     * TPC98166 node
     * inputs: [tensor_23781[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23785[1](dtype=float32)]
     *************/

    // create tensor_23785 tensor
    unsigned tensor_23785_sizes[] = {1};
    unsigned tensor_23785 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23785",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23785_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23781, tensor_23523}, {tensor_23785}, nullptr, 0, "TPC98166");

    /*************
     * Reshape98167 node
     * inputs: [tensor_23532[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23787[450560, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23787 tensor
    unsigned tensor_23787_sizes[] = {450560,1,1,1};
    unsigned tensor_23787 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23787",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23787_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23532}, {tensor_23787}, nullptr, 0, "Reshape98167");

    /*************
     * TPC98168 node
     * inputs: [tensor_23787[450560, 1, 1, 1](dtype=float32)]
     * output: [tensor_23788[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23788 tensor
    unsigned tensor_23788_sizes[] = {1,1,1,1};
    unsigned tensor_23788 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23788",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23788_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98168_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23787}, {tensor_23788}, (void*)TPC98168_params, 4, "TPC98168");

    /*************
     * Reshape98169 node
     * inputs: [tensor_23788[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23786[1](dtype=float32)]
     *************/

    // create tensor_23786 tensor
    unsigned tensor_23786_sizes[] = {1};
    unsigned tensor_23786 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23786",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23786_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23788}, {tensor_23786}, nullptr, 0, "Reshape98169");

    /*************
     * Reshape98170 node
     * inputs: [tensor_23454[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23791[2304, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23791 tensor
    unsigned tensor_23791_sizes[] = {2304,1,1,1};
    unsigned tensor_23791 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23791",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23791_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23454}, {tensor_23791}, nullptr, 0, "Reshape98170");

    /*************
     * TPC98171 node
     * inputs: [tensor_23791[2304, 1, 1, 1](dtype=float32)]
     * output: [tensor_23792[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23792 tensor
    unsigned tensor_23792_sizes[] = {1,1,1,1};
    unsigned tensor_23792 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23792",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23792_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98171_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23791}, {tensor_23792}, (void*)TPC98171_params, 4, "TPC98171");

    /*************
     * Reshape98172 node
     * inputs: [tensor_23792[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23790[1](dtype=float32)]
     *************/

    // create tensor_23790 tensor
    unsigned tensor_23790_sizes[] = {1};
    unsigned tensor_23790 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23790",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23790_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23792}, {tensor_23790}, nullptr, 0, "Reshape98172");

    /*************
     * TPC98173 node
     * inputs: [tensor_23790[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23794[1](dtype=float32)]
     *************/

    // create tensor_23794 tensor
    unsigned tensor_23794_sizes[] = {1};
    unsigned tensor_23794 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23794",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23794_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23790, tensor_23523}, {tensor_23794}, nullptr, 0, "TPC98173");

    /*************
     * TPC98174 node
     * inputs: [tensor_23796[16, 16, 3, 3](dtype=float32), tensor_23796[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23797[16, 16, 3, 3](dtype=float32)]
     *************/

    // create tensor_23796 tensor
    unsigned tensor_23796_sizes[] = {16,16,3,3};
    unsigned tensor_23796 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23796",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23796_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23797 tensor
    unsigned tensor_23797_sizes[] = {16,16,3,3};
    unsigned tensor_23797 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23797",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23797_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23796, tensor_23796}, {tensor_23797}, nullptr, 0, "TPC98174");

    /*************
     * Reshape98175 node
     * inputs: [tensor_23797[16, 16, 3, 3](dtype=float32)]
     * output: [tensor_23799[2304, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23799 tensor
    unsigned tensor_23799_sizes[] = {2304,1,1,1};
    unsigned tensor_23799 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23799",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23799_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23797}, {tensor_23799}, nullptr, 0, "Reshape98175");

    /*************
     * TPC98176 node
     * inputs: [tensor_23799[2304, 1, 1, 1](dtype=float32)]
     * output: [tensor_23800[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23800 tensor
    unsigned tensor_23800_sizes[] = {1,1,1,1};
    unsigned tensor_23800 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23800",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23800_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98176_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23799}, {tensor_23800}, (void*)TPC98176_params, 4, "TPC98176");

    /*************
     * Reshape98177 node
     * inputs: [tensor_23800[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23798[1](dtype=float32)]
     *************/

    // create tensor_23798 tensor
    unsigned tensor_23798_sizes[] = {1};
    unsigned tensor_23798 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23798",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23798_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23800}, {tensor_23798}, nullptr, 0, "Reshape98177");

    /*************
     * TPC98178 node
     * inputs: [tensor_23798[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23802[1](dtype=float32)]
     *************/

    // create tensor_23802 tensor
    unsigned tensor_23802_sizes[] = {1};
    unsigned tensor_23802 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23802",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23802_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23798, tensor_23523}, {tensor_23802}, nullptr, 0, "TPC98178");

    /*************
     * Reshape98179 node
     * inputs: [tensor_23805[1](dtype=float32)]
     * output: [tensor_23807[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23805 tensor
    unsigned tensor_23805_sizes[] = {1};
    unsigned tensor_23805 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23805",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23805_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23807 tensor
    unsigned tensor_23807_sizes[] = {1,1,1,1};
    unsigned tensor_23807 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23807",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23807_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23805}, {tensor_23807}, nullptr, 0, "Reshape98179");

    /*************
     * TPC98180 node
     * inputs: [tensor_23807[1, 1, 1, 1](dtype=float32), tensor_23485[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23806[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23806 tensor
    unsigned tensor_23806_sizes[] = {1,704,320,2};
    unsigned tensor_23806 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23806",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23806_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23807, tensor_23485}, {tensor_23806}, nullptr, 0, "TPC98180");

    /*************
     * TPC98181 node
     * inputs: [tensor_23806[1, 704, 320, 2](dtype=float32), tensor_23491[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23808[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23808 tensor
    unsigned tensor_23808_sizes[] = {1,704,320,2};
    unsigned tensor_23808 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23808",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23808_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23806, tensor_23491}, {tensor_23808}, nullptr, 0, "TPC98181");

    /*************
     * Reshape98182 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23810[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23810 tensor
    unsigned tensor_23810_sizes[] = {1,1,1,1};
    unsigned tensor_23810 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23810",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23810_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23810}, nullptr, 0, "Reshape98182");

    /*************
     * TPC98183 node
     * inputs: [tensor_23808[1, 704, 320, 2](dtype=float32), tensor_23810[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23809[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23809 tensor
    unsigned tensor_23809_sizes[] = {1,704,320,2};
    unsigned tensor_23809 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23809",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23809_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23808, tensor_23810}, {tensor_23809}, nullptr, 0, "TPC98183");

    /*************
     * Reshape98184 node
     * inputs: [tensor_23805[1](dtype=float32)]
     * output: [tensor_23812[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23812 tensor
    unsigned tensor_23812_sizes[] = {1,1,1,1};
    unsigned tensor_23812 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23812",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23812_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23805}, {tensor_23812}, nullptr, 0, "Reshape98184");

    /*************
     * TPC98185 node
     * inputs: [tensor_23812[1, 1, 1, 1](dtype=float32), tensor_23540[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23811[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23811 tensor
    unsigned tensor_23811_sizes[] = {1,704,320,2};
    unsigned tensor_23811 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23811",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23811_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23812, tensor_23540}, {tensor_23811}, nullptr, 0, "TPC98185");

    /*************
     * TPC98186 node
     * inputs: [tensor_23811[1, 704, 320, 2](dtype=float32), tensor_23503[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23813[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23813 tensor
    unsigned tensor_23813_sizes[] = {1,704,320,2};
    unsigned tensor_23813 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23813",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23813_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23811, tensor_23503}, {tensor_23813}, nullptr, 0, "TPC98186");

    /*************
     * Reshape98187 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23815[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23815 tensor
    unsigned tensor_23815_sizes[] = {1,1,1,1};
    unsigned tensor_23815 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23815",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23815_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23815}, nullptr, 0, "Reshape98187");

    /*************
     * TPC98188 node
     * inputs: [tensor_23813[1, 704, 320, 2](dtype=float32), tensor_23815[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23814[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23814 tensor
    unsigned tensor_23814_sizes[] = {1,704,320,2};
    unsigned tensor_23814 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23814",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23814_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23813, tensor_23815}, {tensor_23814}, nullptr, 0, "TPC98188");

    /*************
     * Reshape98189 node
     * inputs: [tensor_23805[1](dtype=float32)]
     * output: [tensor_23817[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23817 tensor
    unsigned tensor_23817_sizes[] = {1,1,1,1};
    unsigned tensor_23817 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23817",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23817_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23805}, {tensor_23817}, nullptr, 0, "Reshape98189");

    /*************
     * TPC98190 node
     * inputs: [tensor_23817[1, 1, 1, 1](dtype=float32), tensor_23565[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23816[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23816 tensor
    unsigned tensor_23816_sizes[] = {1,704,320,2};
    unsigned tensor_23816 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23816",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23816_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23817, tensor_23565}, {tensor_23816}, nullptr, 0, "TPC98190");

    /*************
     * TPC98191 node
     * inputs: [tensor_23816[1, 704, 320, 2](dtype=float32), tensor_23510[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23818[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23818 tensor
    unsigned tensor_23818_sizes[] = {1,704,320,2};
    unsigned tensor_23818 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23818",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23818_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23816, tensor_23510}, {tensor_23818}, nullptr, 0, "TPC98191");

    /*************
     * Reshape98192 node
     * inputs: [tensor_23523[1](dtype=float32)]
     * output: [tensor_23820[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23820 tensor
    unsigned tensor_23820_sizes[] = {1,1,1,1};
    unsigned tensor_23820 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23820",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23820_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23523}, {tensor_23820}, nullptr, 0, "Reshape98192");

    /*************
     * TPC98193 node
     * inputs: [tensor_23818[1, 704, 320, 2](dtype=float32), tensor_23820[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23819[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23819 tensor
    unsigned tensor_23819_sizes[] = {1,704,320,2};
    unsigned tensor_23819 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23819",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23819_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23818, tensor_23820}, {tensor_23819}, nullptr, 0, "TPC98193");

    /*************
     * Reshape98194 node
     * inputs: [tensor_23805[1](dtype=float32)]
     * output: [tensor_23822[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23822 tensor
    unsigned tensor_23822_sizes[] = {1,1,1,1};
    unsigned tensor_23822 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23822",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23822_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23805}, {tensor_23822}, nullptr, 0, "Reshape98194");

    /*************
     * TPC98195 node
     * inputs: [tensor_23822[1, 1, 1, 1](dtype=float32), tensor_23498[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23821[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23821 tensor
    unsigned tensor_23821_sizes[] = {1,704,320,2};
    unsigned tensor_23821 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23821",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23821_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23822, tensor_23498}, {tensor_23821}, nullptr, 0, "TPC98195");

    /*************
     * Reshape98196 node
     * inputs: [tensor_23599[1](dtype=float32)]
     * output: [tensor_23824[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23824 tensor
    unsigned tensor_23824_sizes[] = {1,1,1,1};
    unsigned tensor_23824 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23824",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23824_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23599}, {tensor_23824}, nullptr, 0, "Reshape98196");

    /*************
     * TPC98197 node
     * inputs: [tensor_23824[1, 1, 1, 1](dtype=float32), tensor_23821[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23823[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23823 tensor
    unsigned tensor_23823_sizes[] = {1,704,320,2};
    unsigned tensor_23823 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23823",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23823_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23824, tensor_23821}, {tensor_23823}, nullptr, 0, "TPC98197");

    /*************
     * TPC98198 node
     * inputs: [tensor_23809[1, 704, 320, 2](dtype=float32), tensor_23823[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23825[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23825 tensor
    unsigned tensor_23825_sizes[] = {1,704,320,2};
    unsigned tensor_23825 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23825",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23825_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23809, tensor_23823}, {tensor_23825}, nullptr, 0, "TPC98198");

    /*************
     * TPC98199 node
     * inputs: [tensor_23825[1, 704, 320, 2](dtype=float32), tensor_23604[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23826[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23826 tensor
    unsigned tensor_23826_sizes[] = {1,704,320,2};
    unsigned tensor_23826 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23826",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23826_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23825, tensor_23604}, {tensor_23826}, nullptr, 0, "TPC98199");

    /*************
     * Reshape98200 node
     * inputs: [tensor_23805[1](dtype=float32)]
     * output: [tensor_23828[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23828 tensor
    unsigned tensor_23828_sizes[] = {1,1,1,1};
    unsigned tensor_23828 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23828",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23828_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23805}, {tensor_23828}, nullptr, 0, "Reshape98200");

    /*************
     * TPC98201 node
     * inputs: [tensor_23828[1, 1, 1, 1](dtype=float32), tensor_23542[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23827[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23827 tensor
    unsigned tensor_23827_sizes[] = {1,704,320,2};
    unsigned tensor_23827 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23827",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23827_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23828, tensor_23542}, {tensor_23827}, nullptr, 0, "TPC98201");

    /*************
     * Reshape98202 node
     * inputs: [tensor_23599[1](dtype=float32)]
     * output: [tensor_23830[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23830 tensor
    unsigned tensor_23830_sizes[] = {1,1,1,1};
    unsigned tensor_23830 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23830",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23830_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23599}, {tensor_23830}, nullptr, 0, "Reshape98202");

    /*************
     * TPC98203 node
     * inputs: [tensor_23830[1, 1, 1, 1](dtype=float32), tensor_23827[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23829[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23829 tensor
    unsigned tensor_23829_sizes[] = {1,704,320,2};
    unsigned tensor_23829 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23829",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23829_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23830, tensor_23827}, {tensor_23829}, nullptr, 0, "TPC98203");

    /*************
     * TPC98204 node
     * inputs: [tensor_23814[1, 704, 320, 2](dtype=float32), tensor_23829[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23831[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23831 tensor
    unsigned tensor_23831_sizes[] = {1,704,320,2};
    unsigned tensor_23831 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23831",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23831_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23814, tensor_23829}, {tensor_23831}, nullptr, 0, "TPC98204");

    /*************
     * TPC98205 node
     * inputs: [tensor_23831[1, 704, 320, 2](dtype=float32), tensor_23609[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23832[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23832 tensor
    unsigned tensor_23832_sizes[] = {1,704,320,2};
    unsigned tensor_23832 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23832",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23832_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23831, tensor_23609}, {tensor_23832}, nullptr, 0, "TPC98205");

    /*************
     * Reshape98206 node
     * inputs: [tensor_23805[1](dtype=float32)]
     * output: [tensor_23834[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23834 tensor
    unsigned tensor_23834_sizes[] = {1,1,1,1};
    unsigned tensor_23834 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23834",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23834_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23805}, {tensor_23834}, nullptr, 0, "Reshape98206");

    /*************
     * TPC98207 node
     * inputs: [tensor_23834[1, 1, 1, 1](dtype=float32), tensor_23567[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23833[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23833 tensor
    unsigned tensor_23833_sizes[] = {1,704,320,2};
    unsigned tensor_23833 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23833",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23833_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23834, tensor_23567}, {tensor_23833}, nullptr, 0, "TPC98207");

    /*************
     * Reshape98208 node
     * inputs: [tensor_23599[1](dtype=float32)]
     * output: [tensor_23836[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23836 tensor
    unsigned tensor_23836_sizes[] = {1,1,1,1};
    unsigned tensor_23836 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23836",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23836_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23599}, {tensor_23836}, nullptr, 0, "Reshape98208");

    /*************
     * TPC98209 node
     * inputs: [tensor_23836[1, 1, 1, 1](dtype=float32), tensor_23833[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23835[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23835 tensor
    unsigned tensor_23835_sizes[] = {1,704,320,2};
    unsigned tensor_23835 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23835",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23835_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23836, tensor_23833}, {tensor_23835}, nullptr, 0, "TPC98209");

    /*************
     * TPC98210 node
     * inputs: [tensor_23819[1, 704, 320, 2](dtype=float32), tensor_23835[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23837[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23837 tensor
    unsigned tensor_23837_sizes[] = {1,704,320,2};
    unsigned tensor_23837 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23837",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23837_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23819, tensor_23835}, {tensor_23837}, nullptr, 0, "TPC98210");

    /*************
     * TPC98211 node
     * inputs: [tensor_23837[1, 704, 320, 2](dtype=float32), tensor_23614[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23838[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23838 tensor
    unsigned tensor_23838_sizes[] = {1,704,320,2};
    unsigned tensor_23838 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23838",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23838_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23837, tensor_23614}, {tensor_23838}, nullptr, 0, "TPC98211");

    /*************
     * TPC98212 node
     * inputs: [tensor_23620[1, 8, 3, 3](dtype=float32), tensor_23620[1, 8, 3, 3](dtype=float32)]
     * output: [tensor_23839[1, 8, 3, 3](dtype=float32)]
     *************/

    // create tensor_23839 tensor
    unsigned tensor_23839_sizes[] = {1,8,3,3};
    unsigned tensor_23839 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23839",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23839_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23620, tensor_23620}, {tensor_23839}, nullptr, 0, "TPC98212");

    /*************
     * Reshape98213 node
     * inputs: [tensor_23839[1, 8, 3, 3](dtype=float32)]
     * output: [tensor_23841[72, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23841 tensor
    unsigned tensor_23841_sizes[] = {72,1,1,1};
    unsigned tensor_23841 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23841",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23841_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23839}, {tensor_23841}, nullptr, 0, "Reshape98213");

    /*************
     * TPC98214 node
     * inputs: [tensor_23841[72, 1, 1, 1](dtype=float32)]
     * output: [tensor_23842[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23842 tensor
    unsigned tensor_23842_sizes[] = {1,1,1,1};
    unsigned tensor_23842 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23842",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23842_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98214_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23841}, {tensor_23842}, (void*)TPC98214_params, 4, "TPC98214");

    /*************
     * Reshape98215 node
     * inputs: [tensor_23842[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23840[1](dtype=float32)]
     *************/

    // create tensor_23840 tensor
    unsigned tensor_23840_sizes[] = {1};
    unsigned tensor_23840 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23840",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23840_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23842}, {tensor_23840}, nullptr, 0, "Reshape98215");

    /*************
     * TPC98216 node
     * inputs: [tensor_23840[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23844[1](dtype=float32)]
     *************/

    // create tensor_23844 tensor
    unsigned tensor_23844_sizes[] = {1};
    unsigned tensor_23844 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23844",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23844_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23840, tensor_23523}, {tensor_23844}, nullptr, 0, "TPC98216");

    /*************
     * TPC98217 node
     * inputs: [tensor_23578[16, 40, 3, 3](dtype=float32), tensor_23578[16, 40, 3, 3](dtype=float32)]
     * output: [tensor_23845[16, 40, 3, 3](dtype=float32)]
     *************/

    // create tensor_23845 tensor
    unsigned tensor_23845_sizes[] = {16,40,3,3};
    unsigned tensor_23845 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23845",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23845_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23578, tensor_23578}, {tensor_23845}, nullptr, 0, "TPC98217");

    /*************
     * Reshape98218 node
     * inputs: [tensor_23845[16, 40, 3, 3](dtype=float32)]
     * output: [tensor_23847[5760, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23847 tensor
    unsigned tensor_23847_sizes[] = {5760,1,1,1};
    unsigned tensor_23847 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23847",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23847_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23845}, {tensor_23847}, nullptr, 0, "Reshape98218");

    /*************
     * TPC98219 node
     * inputs: [tensor_23847[5760, 1, 1, 1](dtype=float32)]
     * output: [tensor_23848[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23848 tensor
    unsigned tensor_23848_sizes[] = {1,1,1,1};
    unsigned tensor_23848 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23848",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23848_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98219_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_23847}, {tensor_23848}, (void*)TPC98219_params, 4, "TPC98219");

    /*************
     * Reshape98220 node
     * inputs: [tensor_23848[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23846[1](dtype=float32)]
     *************/

    // create tensor_23846 tensor
    unsigned tensor_23846_sizes[] = {1};
    unsigned tensor_23846 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23846",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23846_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23848}, {tensor_23846}, nullptr, 0, "Reshape98220");

    /*************
     * TPC98221 node
     * inputs: [tensor_23846[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_23850[1](dtype=float32)]
     *************/

    // create tensor_23850 tensor
    unsigned tensor_23850_sizes[] = {1};
    unsigned tensor_23850 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23850",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23850_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23846, tensor_23523}, {tensor_23850}, nullptr, 0, "TPC98221");

    /*************
     * Reshape98222 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23853[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23851 tensor
    unsigned tensor_23851_sizes[] = {1};
    unsigned tensor_23851 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23851",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23851_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23853 tensor
    unsigned tensor_23853_sizes[] = {1,1,1,1};
    unsigned tensor_23853 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23853",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23853_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23853}, nullptr, 0, "Reshape98222");

    /*************
     * TPC98223 node
     * inputs: [tensor_23853[1, 1, 1, 1](dtype=float32), tensor_23522[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23852[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23852 tensor
    unsigned tensor_23852_sizes[] = {1,704,320,2};
    unsigned tensor_23852 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23852",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23852_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23853, tensor_23522}, {tensor_23852}, nullptr, 0, "TPC98223");

    /*************
     * Reshape98224 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23855[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23855 tensor
    unsigned tensor_23855_sizes[] = {1,1,1,1};
    unsigned tensor_23855 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23855",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23855_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23855}, nullptr, 0, "Reshape98224");

    /*************
     * TPC98225 node
     * inputs: [tensor_23855[1, 1, 1, 1](dtype=float32), tensor_23826[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23854[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23854 tensor
    unsigned tensor_23854_sizes[] = {1,704,320,2};
    unsigned tensor_23854 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23854",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23854_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23855, tensor_23826}, {tensor_23854}, nullptr, 0, "TPC98225");

    /*************
     * Reshape98226 node
     * inputs: [tensor_23803[1](dtype=float32)]
     * output: [tensor_23857[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23803 tensor
    unsigned tensor_23803_sizes[] = {1};
    unsigned tensor_23803 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23803",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23803_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23857 tensor
    unsigned tensor_23857_sizes[] = {1,1,1,1};
    unsigned tensor_23857 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23857",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23857_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23803}, {tensor_23857}, nullptr, 0, "Reshape98226");

    /*************
     * TPC98227 node
     * inputs: [tensor_23857[1, 1, 1, 1](dtype=float32), tensor_23854[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23856[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23856 tensor
    unsigned tensor_23856_sizes[] = {1,704,320,2};
    unsigned tensor_23856 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23856",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23856_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23857, tensor_23854}, {tensor_23856}, nullptr, 0, "TPC98227");

    /*************
     * Reshape98228 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23859[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23859 tensor
    unsigned tensor_23859_sizes[] = {1,1,1,1};
    unsigned tensor_23859 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23859",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23859_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23859}, nullptr, 0, "Reshape98228");

    /*************
     * TPC98229 node
     * inputs: [tensor_23859[1, 1, 1, 1](dtype=float32), tensor_23832[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23858[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23858 tensor
    unsigned tensor_23858_sizes[] = {1,704,320,2};
    unsigned tensor_23858 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23858",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23858_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23859, tensor_23832}, {tensor_23858}, nullptr, 0, "TPC98229");

    /*************
     * Reshape98230 node
     * inputs: [tensor_23803[1](dtype=float32)]
     * output: [tensor_23861[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23861 tensor
    unsigned tensor_23861_sizes[] = {1,1,1,1};
    unsigned tensor_23861 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23861",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23861_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23803}, {tensor_23861}, nullptr, 0, "Reshape98230");

    /*************
     * TPC98231 node
     * inputs: [tensor_23861[1, 1, 1, 1](dtype=float32), tensor_23858[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23860[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23860 tensor
    unsigned tensor_23860_sizes[] = {1,704,320,2};
    unsigned tensor_23860 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23860",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23860_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23861, tensor_23858}, {tensor_23860}, nullptr, 0, "TPC98231");

    /*************
     * Reshape98232 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23863[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23863 tensor
    unsigned tensor_23863_sizes[] = {1,1,1,1};
    unsigned tensor_23863 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23863",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23863_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23863}, nullptr, 0, "Reshape98232");

    /*************
     * TPC98233 node
     * inputs: [tensor_23863[1, 1, 1, 1](dtype=float32), tensor_23838[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23862[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23862 tensor
    unsigned tensor_23862_sizes[] = {1,704,320,2};
    unsigned tensor_23862 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23862",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23862_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23863, tensor_23838}, {tensor_23862}, nullptr, 0, "TPC98233");

    /*************
     * Reshape98234 node
     * inputs: [tensor_23803[1](dtype=float32)]
     * output: [tensor_23865[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23865 tensor
    unsigned tensor_23865_sizes[] = {1,1,1,1};
    unsigned tensor_23865 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23865",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23865_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23803}, {tensor_23865}, nullptr, 0, "Reshape98234");

    /*************
     * TPC98235 node
     * inputs: [tensor_23865[1, 1, 1, 1](dtype=float32), tensor_23862[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23864[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23864 tensor
    unsigned tensor_23864_sizes[] = {1,704,320,2};
    unsigned tensor_23864 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23864",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23864_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23865, tensor_23862}, {tensor_23864}, nullptr, 0, "TPC98235");

    /*************
     * Reshape98236 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23867[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23867 tensor
    unsigned tensor_23867_sizes[] = {1,1,1,1};
    unsigned tensor_23867 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23867",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23867_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23867}, nullptr, 0, "Reshape98236");

    /*************
     * TPC98237 node
     * inputs: [tensor_23867[1, 1, 1, 1](dtype=float32), tensor_23549[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23866[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23866 tensor
    unsigned tensor_23866_sizes[] = {1,704,320,2};
    unsigned tensor_23866 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23866",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23866_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23867, tensor_23549}, {tensor_23866}, nullptr, 0, "TPC98237");

    /*************
     * Reshape98238 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23870[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23870 tensor
    unsigned tensor_23870_sizes[] = {1,1,1,1};
    unsigned tensor_23870 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23870",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23870_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23870}, nullptr, 0, "Reshape98238");

    /*************
     * TPC98239 node
     * inputs: [tensor_23870[1, 1, 1, 1](dtype=float32), tensor_23868[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23869[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23868 tensor
    unsigned tensor_23868_sizes[] = {1,704,320,2};
    unsigned tensor_23868 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23868",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23868_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23869 tensor
    unsigned tensor_23869_sizes[] = {1,704,320,2};
    unsigned tensor_23869 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23869",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23869_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23870, tensor_23868}, {tensor_23869}, nullptr, 0, "TPC98239");

    /*************
     * TPC98240 node
     * inputs: [tensor_23852[1, 704, 320, 2](dtype=float32), tensor_23869[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23871[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23871 tensor
    unsigned tensor_23871_sizes[] = {1,704,320,2};
    unsigned tensor_23871 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23871",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23871_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23852, tensor_23869}, {tensor_23871}, nullptr, 0, "TPC98240");

    /*************
     * Convolution98241 node
     * inputs: [tensor_23871[1, 704, 320, 2](dtype=float32), tensor_23625[1, 1, 9, 9](dtype=float32)]
     * output: [tensor_23872[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23625 tensor
    unsigned tensor_23625_sizes[] = {1,1,9,9};
    unsigned tensor_23625 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23625",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23625_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23872 tensor
    unsigned tensor_23872_sizes[] = {1,704,320,2};
    unsigned tensor_23872 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23872",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23872_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98241_params[] = {9,0,0,0,9,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,1,0,0,0,1,0,0,0,0,1,77,53,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,175,37,20,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23871, tensor_23625}, {tensor_23872}, (void*)Convolution98241_params, 88, "Convolution98241");

    /*************
     * Reshape98242 node
     * inputs: [tensor_23795[1](dtype=float32)]
     * output: [tensor_23874[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23795 tensor
    unsigned tensor_23795_sizes[] = {1};
    unsigned tensor_23795 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23795",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23795_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23874 tensor
    unsigned tensor_23874_sizes[] = {1,1,1,1};
    unsigned tensor_23874 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23874",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23874_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23795}, {tensor_23874}, nullptr, 0, "Reshape98242");

    /*************
     * TPC98243 node
     * inputs: [tensor_23872[1, 704, 320, 2](dtype=float32), tensor_23874[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23873[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23873 tensor
    unsigned tensor_23873_sizes[] = {1,704,320,2};
    unsigned tensor_23873 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23873",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23873_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("less_fwd_f32", {tensor_23872, tensor_23874}, {tensor_23873}, nullptr, 0, "TPC98243");

    /*************
     * TPC98244 node
     * inputs: [tensor_23873[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23875[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23875 tensor
    unsigned tensor_23875_sizes[] = {1,704,320,2};
    unsigned tensor_23875 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23875",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23875_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23873}, {tensor_23875}, nullptr, 0, "TPC98244");

    /*************
     * Reshape98245 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23877[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23877 tensor
    unsigned tensor_23877_sizes[] = {1,1,1,1};
    unsigned tensor_23877 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23877",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23877_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23877}, nullptr, 0, "Reshape98245");

    /*************
     * TPC98246 node
     * inputs: [tensor_23877[1, 1, 1, 1](dtype=float32), tensor_23875[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23876[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23876 tensor
    unsigned tensor_23876_sizes[] = {1,704,320,2};
    unsigned tensor_23876 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23876",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23876_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23877, tensor_23875}, {tensor_23876}, nullptr, 0, "TPC98246");

    /*************
     * Reshape98247 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23879[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23879 tensor
    unsigned tensor_23879_sizes[] = {1,1,1,1};
    unsigned tensor_23879 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23879",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23879_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23879}, nullptr, 0, "Reshape98247");

    /*************
     * TPC98248 node
     * inputs: [tensor_23879[1, 1, 1, 1](dtype=float32), tensor_23876[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23878[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23878 tensor
    unsigned tensor_23878_sizes[] = {1,704,320,2};
    unsigned tensor_23878 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23878",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23878_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23879, tensor_23876}, {tensor_23878}, nullptr, 0, "TPC98248");

    /*************
     * TPC98249 node
     * inputs: [tensor_23880[1, 704, 320, 2](dtype=float32), tensor_23878[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23881[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23880 tensor
    unsigned tensor_23880_sizes[] = {1,704,320,2};
    unsigned tensor_23880 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23880",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23880_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23881 tensor
    unsigned tensor_23881_sizes[] = {1,704,320,2};
    unsigned tensor_23881 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23881",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23881_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23880, tensor_23878}, {tensor_23881}, nullptr, 0, "TPC98249");

    /*************
     * TPC98250 node
     * inputs: [tensor_23869[1, 704, 320, 2](dtype=float32), tensor_23866[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23882[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23882 tensor
    unsigned tensor_23882_sizes[] = {1,704,320,2};
    unsigned tensor_23882 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23882",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23882_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23869, tensor_23866}, {tensor_23882}, nullptr, 0, "TPC98250");

    /*************
     * Convolution98251 node
     * inputs: [tensor_23882[1, 704, 320, 2](dtype=float32), tensor_23625[1, 1, 9, 9](dtype=float32)]
     * output: [tensor_23883[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23883 tensor
    unsigned tensor_23883_sizes[] = {1,704,320,2};
    unsigned tensor_23883 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23883",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23883_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98251_params[] = {9,0,0,0,9,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,1,0,0,0,1,0,0,0,0,1,181,53,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,166,39,20,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23882, tensor_23625}, {tensor_23883}, (void*)Convolution98251_params, 88, "Convolution98251");

    /*************
     * Reshape98252 node
     * inputs: [tensor_23795[1](dtype=float32)]
     * output: [tensor_23885[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23885 tensor
    unsigned tensor_23885_sizes[] = {1,1,1,1};
    unsigned tensor_23885 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23885",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23885_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23795}, {tensor_23885}, nullptr, 0, "Reshape98252");

    /*************
     * TPC98253 node
     * inputs: [tensor_23883[1, 704, 320, 2](dtype=float32), tensor_23885[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23884[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23884 tensor
    unsigned tensor_23884_sizes[] = {1,704,320,2};
    unsigned tensor_23884 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23884",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23884_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("less_fwd_f32", {tensor_23883, tensor_23885}, {tensor_23884}, nullptr, 0, "TPC98253");

    /*************
     * TPC98254 node
     * inputs: [tensor_23884[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23886[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23886 tensor
    unsigned tensor_23886_sizes[] = {1,704,320,2};
    unsigned tensor_23886 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23886",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23886_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23884}, {tensor_23886}, nullptr, 0, "TPC98254");

    /*************
     * Reshape98255 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23888[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23888 tensor
    unsigned tensor_23888_sizes[] = {1,1,1,1};
    unsigned tensor_23888 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23888",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23888_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23888}, nullptr, 0, "Reshape98255");

    /*************
     * TPC98256 node
     * inputs: [tensor_23888[1, 1, 1, 1](dtype=float32), tensor_23886[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23887[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23887 tensor
    unsigned tensor_23887_sizes[] = {1,704,320,2};
    unsigned tensor_23887 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23887",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23887_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23888, tensor_23886}, {tensor_23887}, nullptr, 0, "TPC98256");

    /*************
     * Reshape98257 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23890[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23890 tensor
    unsigned tensor_23890_sizes[] = {1,1,1,1};
    unsigned tensor_23890 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23890",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23890_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23890}, nullptr, 0, "Reshape98257");

    /*************
     * TPC98258 node
     * inputs: [tensor_23890[1, 1, 1, 1](dtype=float32), tensor_23887[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23889[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23889 tensor
    unsigned tensor_23889_sizes[] = {1,704,320,2};
    unsigned tensor_23889 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23889",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23889_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23890, tensor_23887}, {tensor_23889}, nullptr, 0, "TPC98258");

    /*************
     * TPC98259 node
     * inputs: [tensor_23880[1, 704, 320, 2](dtype=float32), tensor_23889[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23891[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23891 tensor
    unsigned tensor_23891_sizes[] = {1,704,320,2};
    unsigned tensor_23891 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23891",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23891_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23880, tensor_23889}, {tensor_23891}, nullptr, 0, "TPC98259");

    /*************
     * TPC98260 node
     * inputs: [tensor_23876[1, 704, 320, 2](dtype=float32), tensor_23887[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23892[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23892 tensor
    unsigned tensor_23892_sizes[] = {1,704,320,2};
    unsigned tensor_23892 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23892",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23892_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23876, tensor_23887}, {tensor_23892}, nullptr, 0, "TPC98260");

    /*************
     * Reshape98261 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23894[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23894 tensor
    unsigned tensor_23894_sizes[] = {1,1,1,1};
    unsigned tensor_23894 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23894",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23894_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23894}, nullptr, 0, "Reshape98261");

    /*************
     * TPC98262 node
     * inputs: [tensor_23856[1, 704, 320, 2](dtype=float32), tensor_23894[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23893[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23893 tensor
    unsigned tensor_23893_sizes[] = {1,704,320,2};
    unsigned tensor_23893 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23893",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23893_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("min_fwd_f32", {tensor_23856, tensor_23894}, {tensor_23893}, nullptr, 0, "TPC98262");

    /*************
     * Reshape98263 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23896[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23896 tensor
    unsigned tensor_23896_sizes[] = {1,1,1,1};
    unsigned tensor_23896 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23896",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23896_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23896}, nullptr, 0, "Reshape98263");

    /*************
     * TPC98264 node
     * inputs: [tensor_23893[1, 704, 320, 2](dtype=float32), tensor_23896[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23895[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23895 tensor
    unsigned tensor_23895_sizes[] = {1,704,320,2};
    unsigned tensor_23895 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23895",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23895_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23893, tensor_23896}, {tensor_23895}, nullptr, 0, "TPC98264");

    /*************
     * TPC98265 node
     * inputs: [tensor_23895[1, 704, 320, 2](dtype=float32), tensor_23881[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23897[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23897 tensor
    unsigned tensor_23897_sizes[] = {1,704,320,2};
    unsigned tensor_23897 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23897",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23897_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23895, tensor_23881}, {tensor_23897}, nullptr, 0, "TPC98265");

    /*************
     * Reshape98266 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23899[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23899 tensor
    unsigned tensor_23899_sizes[] = {1,1,1,1};
    unsigned tensor_23899 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23899",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23899_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23899}, nullptr, 0, "Reshape98266");

    /*************
     * TPC98267 node
     * inputs: [tensor_23860[1, 704, 320, 2](dtype=float32), tensor_23899[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23898[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23898 tensor
    unsigned tensor_23898_sizes[] = {1,704,320,2};
    unsigned tensor_23898 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23898",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23898_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("min_fwd_f32", {tensor_23860, tensor_23899}, {tensor_23898}, nullptr, 0, "TPC98267");

    /*************
     * Reshape98268 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23901[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23901 tensor
    unsigned tensor_23901_sizes[] = {1,1,1,1};
    unsigned tensor_23901 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23901",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23901_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23901}, nullptr, 0, "Reshape98268");

    /*************
     * TPC98269 node
     * inputs: [tensor_23898[1, 704, 320, 2](dtype=float32), tensor_23901[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23900[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23900 tensor
    unsigned tensor_23900_sizes[] = {1,704,320,2};
    unsigned tensor_23900 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23900",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23900_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23898, tensor_23901}, {tensor_23900}, nullptr, 0, "TPC98269");

    /*************
     * TPC98270 node
     * inputs: [tensor_23900[1, 704, 320, 2](dtype=float32), tensor_23891[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23902[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23902 tensor
    unsigned tensor_23902_sizes[] = {1,704,320,2};
    unsigned tensor_23902 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23902",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23902_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23900, tensor_23891}, {tensor_23902}, nullptr, 0, "TPC98270");

    /*************
     * TPC98271 node
     * inputs: [tensor_23897[1, 704, 320, 2](dtype=float32), tensor_23902[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23903[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23903 tensor
    unsigned tensor_23903_sizes[] = {1,704,320,2};
    unsigned tensor_23903 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23903",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23903_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("min_fwd_f32", {tensor_23897, tensor_23902}, {tensor_23903}, nullptr, 0, "TPC98271");

    /*************
     * Reshape98272 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23905[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23905 tensor
    unsigned tensor_23905_sizes[] = {1,1,1,1};
    unsigned tensor_23905 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23905",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23905_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23905}, nullptr, 0, "Reshape98272");

    /*************
     * TPC98273 node
     * inputs: [tensor_23864[1, 704, 320, 2](dtype=float32), tensor_23905[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23904[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23904 tensor
    unsigned tensor_23904_sizes[] = {1,704,320,2};
    unsigned tensor_23904 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23904",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23904_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("min_fwd_f32", {tensor_23864, tensor_23905}, {tensor_23904}, nullptr, 0, "TPC98273");

    /*************
     * Reshape98274 node
     * inputs: [tensor_23516[1](dtype=float32)]
     * output: [tensor_23907[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23907 tensor
    unsigned tensor_23907_sizes[] = {1,1,1,1};
    unsigned tensor_23907 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23907",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23907_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23516}, {tensor_23907}, nullptr, 0, "Reshape98274");

    /*************
     * TPC98275 node
     * inputs: [tensor_23904[1, 704, 320, 2](dtype=float32), tensor_23907[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23906[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23906 tensor
    unsigned tensor_23906_sizes[] = {1,704,320,2};
    unsigned tensor_23906 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23906",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23906_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23904, tensor_23907}, {tensor_23906}, nullptr, 0, "TPC98275");

    /*************
     * Reshape98276 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23909[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23909 tensor
    unsigned tensor_23909_sizes[] = {1,1,1,1};
    unsigned tensor_23909 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23909",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23909_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23909}, nullptr, 0, "Reshape98276");

    /*************
     * TPC98277 node
     * inputs: [tensor_23909[1, 1, 1, 1](dtype=float32), tensor_23532[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23908[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23908 tensor
    unsigned tensor_23908_sizes[] = {1,704,320,2};
    unsigned tensor_23908 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23908",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23908_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23909, tensor_23532}, {tensor_23908}, nullptr, 0, "TPC98277");

    /*************
     * TPC98278 node
     * inputs: [tensor_23534[1, 704, 320, 2](dtype=float32), tensor_23908[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23910[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23910 tensor
    unsigned tensor_23910_sizes[] = {1,704,320,2};
    unsigned tensor_23910 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23910",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23910_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23534, tensor_23908}, {tensor_23910}, nullptr, 0, "TPC98278");

    /*************
     * Reshape98279 node
     * inputs: [tensor_23910[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23912[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23912 tensor
    unsigned tensor_23912_sizes[] = {225280,1,1,2};
    unsigned tensor_23912 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23912",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23912_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23910}, {tensor_23912}, nullptr, 0, "Reshape98279");

    /*************
     * TPC98280 node
     * inputs: [tensor_23912[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_23913[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23913 tensor
    unsigned tensor_23913_sizes[] = {1,1,1,2};
    unsigned tensor_23913 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23913",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23913_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98280_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23912}, {tensor_23913}, (void*)TPC98280_params, 4, "TPC98280");

    /*************
     * Reshape98281 node
     * inputs: [tensor_23913[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_23911[2](dtype=float32)]
     *************/

    // create tensor_23911 tensor
    unsigned tensor_23911_sizes[] = {2};
    unsigned tensor_23911 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23911",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23911_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23913}, {tensor_23911}, nullptr, 0, "Reshape98281");

    /*************
     * Reshape98282 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23916[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23916 tensor
    unsigned tensor_23916_sizes[] = {1,1,1,1};
    unsigned tensor_23916 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23916",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23916_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23916}, nullptr, 0, "Reshape98282");

    /*************
     * TPC98283 node
     * inputs: [tensor_23916[1, 1, 1, 1](dtype=float32), tensor_23574[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23915[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23915 tensor
    unsigned tensor_23915_sizes[] = {1,704,320,2};
    unsigned tensor_23915 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23915",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23915_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23916, tensor_23574}, {tensor_23915}, nullptr, 0, "TPC98283");

    /*************
     * TPC98284 node
     * inputs: [tensor_23869[1, 704, 320, 2](dtype=float32), tensor_23915[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23917[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23917 tensor
    unsigned tensor_23917_sizes[] = {1,704,320,2};
    unsigned tensor_23917 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23917",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23917_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23869, tensor_23915}, {tensor_23917}, nullptr, 0, "TPC98284");

    /*************
     * Convolution98285 node
     * inputs: [tensor_23917[1, 704, 320, 2](dtype=float32), tensor_23625[1, 1, 9, 9](dtype=float32)]
     * output: [tensor_23918[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23918 tensor
    unsigned tensor_23918_sizes[] = {1,704,320,2};
    unsigned tensor_23918 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23918",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23918_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98285_params[] = {9,0,0,0,9,0,0,0,1,0,0,0,1,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,1,0,0,0,1,0,0,0,0,1,44,14,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,46,20,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23917, tensor_23625}, {tensor_23918}, (void*)Convolution98285_params, 88, "Convolution98285");

    /*************
     * Reshape98286 node
     * inputs: [tensor_23795[1](dtype=float32)]
     * output: [tensor_23920[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23920 tensor
    unsigned tensor_23920_sizes[] = {1,1,1,1};
    unsigned tensor_23920 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23920",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23920_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23795}, {tensor_23920}, nullptr, 0, "Reshape98286");

    /*************
     * TPC98287 node
     * inputs: [tensor_23918[1, 704, 320, 2](dtype=float32), tensor_23920[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23919[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23919 tensor
    unsigned tensor_23919_sizes[] = {1,704,320,2};
    unsigned tensor_23919 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23919",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23919_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("less_fwd_f32", {tensor_23918, tensor_23920}, {tensor_23919}, nullptr, 0, "TPC98287");

    /*************
     * TPC98288 node
     * inputs: [tensor_23919[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23921[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23921 tensor
    unsigned tensor_23921_sizes[] = {1,704,320,2};
    unsigned tensor_23921 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23921",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23921_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23919}, {tensor_23921}, nullptr, 0, "TPC98288");

    /*************
     * Reshape98289 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23923[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23923 tensor
    unsigned tensor_23923_sizes[] = {1,1,1,1};
    unsigned tensor_23923 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23923",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23923_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23923}, nullptr, 0, "Reshape98289");

    /*************
     * TPC98290 node
     * inputs: [tensor_23923[1, 1, 1, 1](dtype=float32), tensor_23921[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23922[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23922 tensor
    unsigned tensor_23922_sizes[] = {1,704,320,2};
    unsigned tensor_23922 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23922",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23922_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23923, tensor_23921}, {tensor_23922}, nullptr, 0, "TPC98290");

    /*************
     * Reshape98291 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23925[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23925 tensor
    unsigned tensor_23925_sizes[] = {1,1,1,1};
    unsigned tensor_23925 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23925",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23925_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23925}, nullptr, 0, "Reshape98291");

    /*************
     * TPC98292 node
     * inputs: [tensor_23925[1, 1, 1, 1](dtype=float32), tensor_23922[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23924[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23924 tensor
    unsigned tensor_23924_sizes[] = {1,704,320,2};
    unsigned tensor_23924 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23924",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23924_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23925, tensor_23922}, {tensor_23924}, nullptr, 0, "TPC98292");

    /*************
     * TPC98293 node
     * inputs: [tensor_23880[1, 704, 320, 2](dtype=float32), tensor_23924[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23926[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23926 tensor
    unsigned tensor_23926_sizes[] = {1,704,320,2};
    unsigned tensor_23926 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23926",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23926_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23880, tensor_23924}, {tensor_23926}, nullptr, 0, "TPC98293");

    /*************
     * TPC98294 node
     * inputs: [tensor_23906[1, 704, 320, 2](dtype=float32), tensor_23926[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23927[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23927 tensor
    unsigned tensor_23927_sizes[] = {1,704,320,2};
    unsigned tensor_23927 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23927",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23927_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23906, tensor_23926}, {tensor_23927}, nullptr, 0, "TPC98294");

    /*************
     * TPC98295 node
     * inputs: [tensor_23903[1, 704, 320, 2](dtype=float32), tensor_23927[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23928[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23928 tensor
    unsigned tensor_23928_sizes[] = {1,704,320,2};
    unsigned tensor_23928 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23928",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23928_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("min_fwd_f32", {tensor_23903, tensor_23927}, {tensor_23928}, nullptr, 0, "TPC98295");

    /*************
     * TPC98296 node
     * inputs: [tensor_23892[1, 704, 320, 2](dtype=float32), tensor_23922[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23929[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23929 tensor
    unsigned tensor_23929_sizes[] = {1,704,320,2};
    unsigned tensor_23929 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23929",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23929_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23892, tensor_23922}, {tensor_23929}, nullptr, 0, "TPC98296");

    /*************
     * Convolution98297 node
     * inputs: [tensor_23929[1, 704, 320, 2](dtype=float32), tensor_23559[1, 1, 5, 5](dtype=float32)]
     * output: [tensor_23930[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23559 tensor
    unsigned tensor_23559_sizes[] = {1,1,5,5};
    unsigned tensor_23559 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23559",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23559_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23930 tensor
    unsigned tensor_23930_sizes[] = {1,704,320,2};
    unsigned tensor_23930 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23930",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23930_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Convolution98297_params[] = {5,0,0,0,5,0,0,0,1,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,0,1,135,145,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,141,48,20,109,85,0,0,1,0,0,0,252,127,0,0};
    addNodeToGraph("spatial_convolution", {tensor_23929, tensor_23559}, {tensor_23930}, (void*)Convolution98297_params, 88, "Convolution98297");

    /*************
     * Reshape98298 node
     * inputs: [tensor_23479[1](dtype=float32)]
     * output: [tensor_23932[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23479 tensor
    unsigned tensor_23479_sizes[] = {1};
    unsigned tensor_23479 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23479",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23479_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23932 tensor
    unsigned tensor_23932_sizes[] = {1,1,1,1};
    unsigned tensor_23932 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23932",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23932_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23479}, {tensor_23932}, nullptr, 0, "Reshape98298");

    /*************
     * TPC98299 node
     * inputs: [tensor_23930[1, 704, 320, 2](dtype=float32), tensor_23932[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23931[1, 704, 320, 2](dtype=int8)]
     *************/

    // create tensor_23931 tensor
    unsigned tensor_23931_sizes[] = {1,704,320,2};
    unsigned tensor_23931 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23931",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23931_sizes,
                                        4,
                                        syn_type_int8,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("less_fwd_f32", {tensor_23930, tensor_23932}, {tensor_23931}, nullptr, 0, "TPC98299");

    /*************
     * TPC98300 node
     * inputs: [tensor_23931[1, 704, 320, 2](dtype=int8)]
     * output: [tensor_23933[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23933 tensor
    unsigned tensor_23933_sizes[] = {1,704,320,2};
    unsigned tensor_23933 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23933",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23933_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("cast_i8_to_f32", {tensor_23931}, {tensor_23933}, nullptr, 0, "TPC98300");

    /*************
     * Reshape98301 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23935[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23935 tensor
    unsigned tensor_23935_sizes[] = {1,1,1,1};
    unsigned tensor_23935 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23935",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23935_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23935}, nullptr, 0, "Reshape98301");

    /*************
     * TPC98302 node
     * inputs: [tensor_23935[1, 1, 1, 1](dtype=float32), tensor_23933[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23934[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23934 tensor
    unsigned tensor_23934_sizes[] = {1,704,320,2};
    unsigned tensor_23934 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23934",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23934_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23935, tensor_23933}, {tensor_23934}, nullptr, 0, "TPC98302");

    /*************
     * TPC98303 node
     * inputs: [tensor_23928[1, 704, 320, 2](dtype=float32), tensor_23934[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23936[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23936 tensor
    unsigned tensor_23936_sizes[] = {1,704,320,2};
    unsigned tensor_23936 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23936",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23936_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23928, tensor_23934}, {tensor_23936}, nullptr, 0, "TPC98303");

    /*************
     * Reshape98304 node
     * inputs: [tensor_23936[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23938[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23938 tensor
    unsigned tensor_23938_sizes[] = {225280,1,1,2};
    unsigned tensor_23938 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23938",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23938_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23936}, {tensor_23938}, nullptr, 0, "Reshape98304");

    /*************
     * TPC98305 node
     * inputs: [tensor_23938[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_23939[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23939 tensor
    unsigned tensor_23939_sizes[] = {1,1,1,2};
    unsigned tensor_23939 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23939",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23939_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98305_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23938}, {tensor_23939}, (void*)TPC98305_params, 4, "TPC98305");

    /*************
     * Reshape98306 node
     * inputs: [tensor_23939[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_23937[2](dtype=float32)]
     *************/

    // create tensor_23937 tensor
    unsigned tensor_23937_sizes[] = {2};
    unsigned tensor_23937 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23937",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23937_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23939}, {tensor_23937}, nullptr, 0, "Reshape98306");

    /*************
     * Reshape98307 node
     * inputs: [tensor_23934[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23942[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23942 tensor
    unsigned tensor_23942_sizes[] = {225280,1,1,2};
    unsigned tensor_23942 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23942",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23942_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23934}, {tensor_23942}, nullptr, 0, "Reshape98307");

    /*************
     * TPC98308 node
     * inputs: [tensor_23942[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_23943[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23943 tensor
    unsigned tensor_23943_sizes[] = {1,1,1,2};
    unsigned tensor_23943 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23943",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23943_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98308_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23942}, {tensor_23943}, (void*)TPC98308_params, 4, "TPC98308");

    /*************
     * Reshape98309 node
     * inputs: [tensor_23943[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_23941[2](dtype=float32)]
     *************/

    // create tensor_23941 tensor
    unsigned tensor_23941_sizes[] = {2};
    unsigned tensor_23941 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23941",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23941_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23943}, {tensor_23941}, nullptr, 0, "Reshape98309");

    /*************
     * TPC98310 node
     * inputs: [tensor_23941[2](dtype=float32), tensor_23804[1](dtype=float32)]
     * output: [tensor_23945[2](dtype=float32)]
     *************/

    // create tensor_23804 tensor
    unsigned tensor_23804_sizes[] = {1};
    unsigned tensor_23804 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23804",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23804_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23945 tensor
    unsigned tensor_23945_sizes[] = {2};
    unsigned tensor_23945 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23945",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23945_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23941, tensor_23804}, {tensor_23945}, nullptr, 0, "TPC98310");

    /*************
     * TPC98311 node
     * inputs: [tensor_23937[2](dtype=float32), tensor_23945[2](dtype=float32)]
     * output: [tensor_23946[2](dtype=float32)]
     *************/

    // create tensor_23946 tensor
    unsigned tensor_23946_sizes[] = {2};
    unsigned tensor_23946 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23946",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23946_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23937, tensor_23945}, {tensor_23946}, nullptr, 0, "TPC98311");

    /*************
     * TPC98312 node
     * inputs: [tensor_23805[1](dtype=float32), tensor_23946[2](dtype=float32)]
     * output: [tensor_23947[2](dtype=float32)]
     *************/

    // create tensor_23947 tensor
    unsigned tensor_23947_sizes[] = {2};
    unsigned tensor_23947 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23947",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23947_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23805, tensor_23946}, {tensor_23947}, nullptr, 0, "TPC98312");

    /*************
     * TPC98313 node
     * inputs: [tensor_23947[2](dtype=float32)]
     * output: [tensor_23949[1](dtype=float32)]
     *************/

    // create tensor_23949 tensor
    unsigned tensor_23949_sizes[] = {1};
    unsigned tensor_23949 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23949",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23949_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98313_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23947}, {tensor_23949}, (void*)TPC98313_params, 4, "TPC98313");

    /*************
     * Reshape98314 node
     * inputs: [tensor_23949[1](dtype=float32)]
     * output: [tensor_23948[1](dtype=float32)]
     *************/

    // create tensor_23948 tensor
    unsigned tensor_23948_sizes[] = {1};
    unsigned tensor_23948 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23948",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23948_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23949}, {tensor_23948}, nullptr, 0, "Reshape98314");

    /*************
     * TPC98315 node
     * inputs: [tensor_23851[1](dtype=float32), tensor_23786[1](dtype=float32)]
     * output: [tensor_23951[1](dtype=float32)]
     *************/

    // create tensor_23951 tensor
    unsigned tensor_23951_sizes[] = {1};
    unsigned tensor_23951 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23951",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23951_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23851, tensor_23786}, {tensor_23951}, nullptr, 0, "TPC98315");

    /*************
     * TPC98316 node
     * inputs: [tensor_23951[1](dtype=float32), tensor_23513[1](dtype=float32)]
     * output: [tensor_23952[1](dtype=float32)]
     *************/

    // create tensor_23513 tensor
    unsigned tensor_23513_sizes[] = {1};
    unsigned tensor_23513 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23513",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23513_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23952 tensor
    unsigned tensor_23952_sizes[] = {1};
    unsigned tensor_23952 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23952",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23952_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23951, tensor_23513}, {tensor_23952}, nullptr, 0, "TPC98316");

    /*************
     * TPC98317 node
     * inputs: [tensor_23911[2](dtype=float32), tensor_23952[1](dtype=float32)]
     * output: [tensor_23953[2](dtype=float32)]
     *************/

    // create tensor_23953 tensor
    unsigned tensor_23953_sizes[] = {2};
    unsigned tensor_23953 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23953",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23953_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23911, tensor_23952}, {tensor_23953}, nullptr, 0, "TPC98317");

    /*************
     * TPC98318 node
     * inputs: [tensor_23558[1](dtype=float32), tensor_23953[2](dtype=float32)]
     * output: [tensor_23954[2](dtype=float32)]
     *************/

    // create tensor_23558 tensor
    unsigned tensor_23558_sizes[] = {1};
    unsigned tensor_23558 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23558",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23558_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23954 tensor
    unsigned tensor_23954_sizes[] = {2};
    unsigned tensor_23954 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23954",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23954_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23558, tensor_23953}, {tensor_23954}, nullptr, 0, "TPC98318");

    /*************
     * TPC98319 node
     * inputs: [tensor_23954[2](dtype=float32)]
     * output: [tensor_23956[1](dtype=float32)]
     *************/

    // create tensor_23956 tensor
    unsigned tensor_23956_sizes[] = {1};
    unsigned tensor_23956 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23956",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23956_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98319_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23954}, {tensor_23956}, (void*)TPC98319_params, 4, "TPC98319");

    /*************
     * Reshape98320 node
     * inputs: [tensor_23956[1](dtype=float32)]
     * output: [tensor_23955[1](dtype=float32)]
     *************/

    // create tensor_23955 tensor
    unsigned tensor_23955_sizes[] = {1};
    unsigned tensor_23955 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23955",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23955_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23956}, {tensor_23955}, nullptr, 0, "Reshape98320");

    /*************
     * Reshape98321 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23959[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23959 tensor
    unsigned tensor_23959_sizes[] = {1,1,1,1};
    unsigned tensor_23959 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23959",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23959_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23959}, nullptr, 0, "Reshape98321");

    /*************
     * TPC98322 node
     * inputs: [tensor_23959[1, 1, 1, 1](dtype=float32), tensor_23555[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23958[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23958 tensor
    unsigned tensor_23958_sizes[] = {1,704,320,2};
    unsigned tensor_23958 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23958",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23958_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23959, tensor_23555}, {tensor_23958}, nullptr, 0, "TPC98322");

    /*************
     * Reshape98323 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23961[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23961 tensor
    unsigned tensor_23961_sizes[] = {1,1,1,1};
    unsigned tensor_23961 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23961",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23961_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23961}, nullptr, 0, "Reshape98323");

    /*************
     * TPC98324 node
     * inputs: [tensor_23961[1, 1, 1, 1](dtype=float32), tensor_23519[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23960[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23960 tensor
    unsigned tensor_23960_sizes[] = {1,704,320,2};
    unsigned tensor_23960 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23960",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23960_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23961, tensor_23519}, {tensor_23960}, nullptr, 0, "TPC98324");

    /*************
     * Reshape98325 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_23963[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23963 tensor
    unsigned tensor_23963_sizes[] = {1,1,1,1};
    unsigned tensor_23963 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23963",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23963_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_23963}, nullptr, 0, "Reshape98325");

    /*************
     * TPC98326 node
     * inputs: [tensor_23963[1, 1, 1, 1](dtype=float32), tensor_23960[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23962[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23962 tensor
    unsigned tensor_23962_sizes[] = {1,704,320,2};
    unsigned tensor_23962 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23962",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23962_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23963, tensor_23960}, {tensor_23962}, nullptr, 0, "TPC98326");

    /*************
     * Reshape98327 node
     * inputs: [tensor_23962[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23965[450560, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23965 tensor
    unsigned tensor_23965_sizes[] = {450560,1,1,1};
    unsigned tensor_23965 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23965",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23965_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23962}, {tensor_23965}, nullptr, 0, "Reshape98327");

    /*************
     * TPC98328 node
     * inputs: [tensor_23965[450560, 1, 1, 1](dtype=float32)]
     * output: [tensor_23966[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_23966 tensor
    unsigned tensor_23966_sizes[] = {1,1,1,1};
    unsigned tensor_23966 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23966",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23966_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98328_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23965}, {tensor_23966}, (void*)TPC98328_params, 4, "TPC98328");

    /*************
     * Reshape98329 node
     * inputs: [tensor_23966[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_23964[1](dtype=float32)]
     *************/

    // create tensor_23964 tensor
    unsigned tensor_23964_sizes[] = {1};
    unsigned tensor_23964 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23964",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23964_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23966}, {tensor_23964}, nullptr, 0, "Reshape98329");

    /*************
     * TPC98330 node
     * inputs: [tensor_23851[1](dtype=float32), tensor_23964[1](dtype=float32)]
     * output: [tensor_23968[1](dtype=float32)]
     *************/

    // create tensor_23968 tensor
    unsigned tensor_23968_sizes[] = {1};
    unsigned tensor_23968 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23968",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23968_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23851, tensor_23964}, {tensor_23968}, nullptr, 0, "TPC98330");

    /*************
     * TPC98331 node
     * inputs: [tensor_23968[1](dtype=float32), tensor_23513[1](dtype=float32)]
     * output: [tensor_23969[1](dtype=float32)]
     *************/

    // create tensor_23969 tensor
    unsigned tensor_23969_sizes[] = {1};
    unsigned tensor_23969 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23969",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23969_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23968, tensor_23513}, {tensor_23969}, nullptr, 0, "TPC98331");

    /*************
     * Reshape98332 node
     * inputs: [tensor_23962[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23971[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23971 tensor
    unsigned tensor_23971_sizes[] = {225280,1,1,2};
    unsigned tensor_23971 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23971",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23971_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23962}, {tensor_23971}, nullptr, 0, "Reshape98332");

    /*************
     * TPC98333 node
     * inputs: [tensor_23971[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_23972[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23972 tensor
    unsigned tensor_23972_sizes[] = {1,1,1,2};
    unsigned tensor_23972 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23972",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23972_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98333_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23971}, {tensor_23972}, (void*)TPC98333_params, 4, "TPC98333");

    /*************
     * Reshape98334 node
     * inputs: [tensor_23972[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_23970[2](dtype=float32)]
     *************/

    // create tensor_23970 tensor
    unsigned tensor_23970_sizes[] = {2};
    unsigned tensor_23970 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23970",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23970_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23972}, {tensor_23970}, nullptr, 0, "Reshape98334");

    /*************
     * TPC98335 node
     * inputs: [tensor_23851[1](dtype=float32), tensor_23970[2](dtype=float32)]
     * output: [tensor_23974[2](dtype=float32)]
     *************/

    // create tensor_23974 tensor
    unsigned tensor_23974_sizes[] = {2};
    unsigned tensor_23974 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23974",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23974_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23851, tensor_23970}, {tensor_23974}, nullptr, 0, "TPC98335");

    /*************
     * TPC98336 node
     * inputs: [tensor_23974[2](dtype=float32), tensor_23513[1](dtype=float32)]
     * output: [tensor_23975[2](dtype=float32)]
     *************/

    // create tensor_23975 tensor
    unsigned tensor_23975_sizes[] = {2};
    unsigned tensor_23975 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23975",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23975_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_23974, tensor_23513}, {tensor_23975}, nullptr, 0, "TPC98336");

    /*************
     * TPC98337 node
     * inputs: [tensor_23475[1, 704, 320, 2](dtype=float32), tensor_23960[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23976[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23976 tensor
    unsigned tensor_23976_sizes[] = {1,704,320,2};
    unsigned tensor_23976 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23976",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23976_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23475, tensor_23960}, {tensor_23976}, nullptr, 0, "TPC98337");

    /*************
     * Reshape98338 node
     * inputs: [tensor_23976[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23978[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23978 tensor
    unsigned tensor_23978_sizes[] = {225280,1,1,2};
    unsigned tensor_23978 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23978",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23978_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23976}, {tensor_23978}, nullptr, 0, "Reshape98338");

    /*************
     * TPC98339 node
     * inputs: [tensor_23978[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_23979[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23979 tensor
    unsigned tensor_23979_sizes[] = {1,1,1,2};
    unsigned tensor_23979 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23979",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23979_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98339_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23978}, {tensor_23979}, (void*)TPC98339_params, 4, "TPC98339");

    /*************
     * Reshape98340 node
     * inputs: [tensor_23979[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_23977[2](dtype=float32)]
     *************/

    // create tensor_23977 tensor
    unsigned tensor_23977_sizes[] = {2};
    unsigned tensor_23977 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23977",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23977_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23979}, {tensor_23977}, nullptr, 0, "Reshape98340");

    /*************
     * TPC98341 node
     * inputs: [tensor_23977[2](dtype=float32), tensor_23969[1](dtype=float32)]
     * output: [tensor_23981[2](dtype=float32)]
     *************/

    // create tensor_23981 tensor
    unsigned tensor_23981_sizes[] = {2};
    unsigned tensor_23981 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23981",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23981_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23977, tensor_23969}, {tensor_23981}, nullptr, 0, "TPC98341");

    /*************
     * TPC98342 node
     * inputs: [tensor_23558[1](dtype=float32), tensor_23981[2](dtype=float32)]
     * output: [tensor_23982[2](dtype=float32)]
     *************/

    // create tensor_23982 tensor
    unsigned tensor_23982_sizes[] = {2};
    unsigned tensor_23982 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23982",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23982_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23558, tensor_23981}, {tensor_23982}, nullptr, 0, "TPC98342");

    /*************
     * TPC98343 node
     * inputs: [tensor_23982[2](dtype=float32)]
     * output: [tensor_23984[1](dtype=float32)]
     *************/

    // create tensor_23984 tensor
    unsigned tensor_23984_sizes[] = {1};
    unsigned tensor_23984 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23984",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23984_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98343_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23982}, {tensor_23984}, (void*)TPC98343_params, 4, "TPC98343");

    /*************
     * Reshape98344 node
     * inputs: [tensor_23984[1](dtype=float32)]
     * output: [tensor_23983[1](dtype=float32)]
     *************/

    // create tensor_23983 tensor
    unsigned tensor_23983_sizes[] = {1};
    unsigned tensor_23983 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23983",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23983_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23984}, {tensor_23983}, nullptr, 0, "Reshape98344");

    /*************
     * TPC98345 node
     * inputs: [tensor_23982[2](dtype=float32), tensor_23469[1](dtype=float32)]
     * output: [tensor_23986[2](dtype=float32)]
     *************/

    // create tensor_23469 tensor
    unsigned tensor_23469_sizes[] = {1};
    unsigned tensor_23469 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23469",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23469_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23986 tensor
    unsigned tensor_23986_sizes[] = {2};
    unsigned tensor_23986 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23986",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23986_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23982, tensor_23469}, {tensor_23986}, nullptr, 0, "TPC98345");

    /*************
     * TPC98346 node
     * inputs: [tensor_23623[1, 704, 320, 2](dtype=float32), tensor_23976[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23987[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23987 tensor
    unsigned tensor_23987_sizes[] = {1,704,320,2};
    unsigned tensor_23987 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23987",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23987_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23623, tensor_23976}, {tensor_23987}, nullptr, 0, "TPC98346");

    /*************
     * TPC98347 node
     * inputs: [tensor_23987[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23988[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23988 tensor
    unsigned tensor_23988_sizes[] = {1,704,320,2};
    unsigned tensor_23988 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23988",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23988_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("abs_fwd_f32", {tensor_23987}, {tensor_23988}, nullptr, 0, "TPC98347");

    /*************
     * TPC98348 node
     * inputs: [tensor_23988[1, 704, 320, 2](dtype=float32), tensor_23960[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23989[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_23989 tensor
    unsigned tensor_23989_sizes[] = {1,704,320,2};
    unsigned tensor_23989 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23989",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23989_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23988, tensor_23960}, {tensor_23989}, nullptr, 0, "TPC98348");

    /*************
     * Reshape98349 node
     * inputs: [tensor_23989[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_23991[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23991 tensor
    unsigned tensor_23991_sizes[] = {225280,1,1,2};
    unsigned tensor_23991 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23991",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23991_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23989}, {tensor_23991}, nullptr, 0, "Reshape98349");

    /*************
     * TPC98350 node
     * inputs: [tensor_23991[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_23992[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_23992 tensor
    unsigned tensor_23992_sizes[] = {1,1,1,2};
    unsigned tensor_23992 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23992",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23992_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98350_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23991}, {tensor_23992}, (void*)TPC98350_params, 4, "TPC98350");

    /*************
     * Reshape98351 node
     * inputs: [tensor_23992[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_23990[2](dtype=float32)]
     *************/

    // create tensor_23990 tensor
    unsigned tensor_23990_sizes[] = {2};
    unsigned tensor_23990 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23990",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23990_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23992}, {tensor_23990}, nullptr, 0, "Reshape98351");

    /*************
     * TPC98352 node
     * inputs: [tensor_23990[2](dtype=float32), tensor_23975[2](dtype=float32)]
     * output: [tensor_23994[2](dtype=float32)]
     *************/

    // create tensor_23994 tensor
    unsigned tensor_23994_sizes[] = {2};
    unsigned tensor_23994 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23994",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23994_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_23990, tensor_23975}, {tensor_23994}, nullptr, 0, "TPC98352");

    /*************
     * TPC98353 node
     * inputs: [tensor_23994[2](dtype=float32)]
     * output: [tensor_23996[1](dtype=float32)]
     *************/

    // create tensor_23996 tensor
    unsigned tensor_23996_sizes[] = {1};
    unsigned tensor_23996 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23996",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23996_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98353_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_23994}, {tensor_23996}, (void*)TPC98353_params, 4, "TPC98353");

    /*************
     * Reshape98354 node
     * inputs: [tensor_23996[1](dtype=float32)]
     * output: [tensor_23995[1](dtype=float32)]
     *************/

    // create tensor_23995 tensor
    unsigned tensor_23995_sizes[] = {1};
    unsigned tensor_23995 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23995",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23995_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23996}, {tensor_23995}, nullptr, 0, "Reshape98354");

    /*************
     * TPC98355 node
     * inputs: [tensor_23478[1](dtype=float32), tensor_23994[2](dtype=float32)]
     * output: [tensor_23998[2](dtype=float32)]
     *************/

    // create tensor_23478 tensor
    unsigned tensor_23478_sizes[] = {1};
    unsigned tensor_23478 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23478",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23478_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_23998 tensor
    unsigned tensor_23998_sizes[] = {2};
    unsigned tensor_23998 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_23998",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23998_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23478, tensor_23994}, {tensor_23998}, nullptr, 0, "TPC98355");

    /*************
     * TPC98356 node
     * inputs: [tensor_23999[144, 144, 3, 3](dtype=float32), tensor_23999[144, 144, 3, 3](dtype=float32)]
     * output: [tensor_24000[144, 144, 3, 3](dtype=float32)]
     *************/

    // create tensor_23999 tensor
    unsigned tensor_23999_sizes[] = {144,144,3,3};
    unsigned tensor_23999 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_23999",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_23999_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_24000 tensor
    unsigned tensor_24000_sizes[] = {144,144,3,3};
    unsigned tensor_24000 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24000",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24000_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23999, tensor_23999}, {tensor_24000}, nullptr, 0, "TPC98356");

    /*************
     * Reshape98357 node
     * inputs: [tensor_24000[144, 144, 3, 3](dtype=float32)]
     * output: [tensor_24002[186624, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24002 tensor
    unsigned tensor_24002_sizes[] = {186624,1,1,1};
    unsigned tensor_24002 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24002",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24002_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24000}, {tensor_24002}, nullptr, 0, "Reshape98357");

    /*************
     * TPC98358 node
     * inputs: [tensor_24002[186624, 1, 1, 1](dtype=float32)]
     * output: [tensor_24003[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24003 tensor
    unsigned tensor_24003_sizes[] = {1,1,1,1};
    unsigned tensor_24003 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24003",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24003_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98358_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_24002}, {tensor_24003}, (void*)TPC98358_params, 4, "TPC98358");

    /*************
     * Reshape98359 node
     * inputs: [tensor_24003[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_24001[1](dtype=float32)]
     *************/

    // create tensor_24001 tensor
    unsigned tensor_24001_sizes[] = {1};
    unsigned tensor_24001 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24001",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24001_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24003}, {tensor_24001}, nullptr, 0, "Reshape98359");

    /*************
     * TPC98360 node
     * inputs: [tensor_24001[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_24005[1](dtype=float32)]
     *************/

    // create tensor_24005 tensor
    unsigned tensor_24005_sizes[] = {1};
    unsigned tensor_24005 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24005",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24005_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_24001, tensor_23523}, {tensor_24005}, nullptr, 0, "TPC98360");

    /*************
     * TPC98361 node
     * inputs: [tensor_24006[96, 288, 3, 3](dtype=float32), tensor_24006[96, 288, 3, 3](dtype=float32)]
     * output: [tensor_24007[96, 288, 3, 3](dtype=float32)]
     *************/

    // create tensor_24006 tensor
    unsigned tensor_24006_sizes[] = {96,288,3,3};
    unsigned tensor_24006 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24006",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24006_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_24007 tensor
    unsigned tensor_24007_sizes[] = {96,288,3,3};
    unsigned tensor_24007 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24007",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24007_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_24006, tensor_24006}, {tensor_24007}, nullptr, 0, "TPC98361");

    /*************
     * Reshape98362 node
     * inputs: [tensor_24007[96, 288, 3, 3](dtype=float32)]
     * output: [tensor_24009[248832, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24009 tensor
    unsigned tensor_24009_sizes[] = {248832,1,1,1};
    unsigned tensor_24009 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24009",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24009_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24007}, {tensor_24009}, nullptr, 0, "Reshape98362");

    /*************
     * TPC98363 node
     * inputs: [tensor_24009[248832, 1, 1, 1](dtype=float32)]
     * output: [tensor_24010[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24010 tensor
    unsigned tensor_24010_sizes[] = {1,1,1,1};
    unsigned tensor_24010 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24010",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24010_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98363_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_24009}, {tensor_24010}, (void*)TPC98363_params, 4, "TPC98363");

    /*************
     * Reshape98364 node
     * inputs: [tensor_24010[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_24008[1](dtype=float32)]
     *************/

    // create tensor_24008 tensor
    unsigned tensor_24008_sizes[] = {1};
    unsigned tensor_24008 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24008",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24008_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24010}, {tensor_24008}, nullptr, 0, "Reshape98364");

    /*************
     * TPC98365 node
     * inputs: [tensor_24008[1](dtype=float32), tensor_23523[1](dtype=float32)]
     * output: [tensor_24012[1](dtype=float32)]
     *************/

    // create tensor_24012 tensor
    unsigned tensor_24012_sizes[] = {1};
    unsigned tensor_24012 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24012",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24012_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_24008, tensor_23523}, {tensor_24012}, nullptr, 0, "TPC98365");

    /*************
     * Reshape98366 node
     * inputs: [tensor_23740[1](dtype=float32)]
     * output: [tensor_24014[1](dtype=float32)]
     *************/

    // create tensor_24014 tensor
    unsigned tensor_24014_sizes[] = {1};
    unsigned tensor_24014 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24014",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24014_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23740}, {tensor_24014}, nullptr, 0, "Reshape98366");

    /*************
     * Reshape98367 node
     * inputs: [tensor_23665[1](dtype=float32)]
     * output: [tensor_24015[1](dtype=float32)]
     *************/

    // create tensor_24015 tensor
    unsigned tensor_24015_sizes[] = {1};
    unsigned tensor_24015 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24015",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24015_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23665}, {tensor_24015}, nullptr, 0, "Reshape98367");

    /*************
     * Reshape98368 node
     * inputs: [tensor_23670[1](dtype=float32)]
     * output: [tensor_24016[1](dtype=float32)]
     *************/

    // create tensor_24016 tensor
    unsigned tensor_24016_sizes[] = {1};
    unsigned tensor_24016 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24016",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24016_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23670}, {tensor_24016}, nullptr, 0, "Reshape98368");

    /*************
     * Reshape98369 node
     * inputs: [tensor_23745[1](dtype=float32)]
     * output: [tensor_24017[1](dtype=float32)]
     *************/

    // create tensor_24017 tensor
    unsigned tensor_24017_sizes[] = {1};
    unsigned tensor_24017 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24017",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24017_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23745}, {tensor_24017}, nullptr, 0, "Reshape98369");

    /*************
     * Reshape98370 node
     * inputs: [tensor_23785[1](dtype=float32)]
     * output: [tensor_24018[1](dtype=float32)]
     *************/

    // create tensor_24018 tensor
    unsigned tensor_24018_sizes[] = {1};
    unsigned tensor_24018 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24018",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24018_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23785}, {tensor_24018}, nullptr, 0, "Reshape98370");

    /*************
     * Reshape98371 node
     * inputs: [tensor_23735[1](dtype=float32)]
     * output: [tensor_24019[1](dtype=float32)]
     *************/

    // create tensor_24019 tensor
    unsigned tensor_24019_sizes[] = {1};
    unsigned tensor_24019 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24019",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24019_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23735}, {tensor_24019}, nullptr, 0, "Reshape98371");

    /*************
     * Reshape98372 node
     * inputs: [tensor_23765[1](dtype=float32)]
     * output: [tensor_24020[1](dtype=float32)]
     *************/

    // create tensor_24020 tensor
    unsigned tensor_24020_sizes[] = {1};
    unsigned tensor_24020 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24020",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24020_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23765}, {tensor_24020}, nullptr, 0, "Reshape98372");

    /*************
     * Reshape98373 node
     * inputs: [tensor_23794[1](dtype=float32)]
     * output: [tensor_24021[1](dtype=float32)]
     *************/

    // create tensor_24021 tensor
    unsigned tensor_24021_sizes[] = {1};
    unsigned tensor_24021 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24021",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24021_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23794}, {tensor_24021}, nullptr, 0, "Reshape98373");

    /*************
     * Reshape98374 node
     * inputs: [tensor_23775[1](dtype=float32)]
     * output: [tensor_24022[1](dtype=float32)]
     *************/

    // create tensor_24022 tensor
    unsigned tensor_24022_sizes[] = {1};
    unsigned tensor_24022 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24022",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24022_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23775}, {tensor_24022}, nullptr, 0, "Reshape98374");

    /*************
     * Reshape98375 node
     * inputs: [tensor_23750[1](dtype=float32)]
     * output: [tensor_24023[1](dtype=float32)]
     *************/

    // create tensor_24023 tensor
    unsigned tensor_24023_sizes[] = {1};
    unsigned tensor_24023 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24023",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24023_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23750}, {tensor_24023}, nullptr, 0, "Reshape98375");

    /*************
     * Reshape98376 node
     * inputs: [tensor_23780[1](dtype=float32)]
     * output: [tensor_24024[1](dtype=float32)]
     *************/

    // create tensor_24024 tensor
    unsigned tensor_24024_sizes[] = {1};
    unsigned tensor_24024 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24024",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24024_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23780}, {tensor_24024}, nullptr, 0, "Reshape98376");

    /*************
     * Reshape98377 node
     * inputs: [tensor_23760[1](dtype=float32)]
     * output: [tensor_24025[1](dtype=float32)]
     *************/

    // create tensor_24025 tensor
    unsigned tensor_24025_sizes[] = {1};
    unsigned tensor_24025 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24025",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24025_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23760}, {tensor_24025}, nullptr, 0, "Reshape98377");

    /*************
     * Reshape98378 node
     * inputs: [tensor_23645[1](dtype=float32)]
     * output: [tensor_24026[1](dtype=float32)]
     *************/

    // create tensor_24026 tensor
    unsigned tensor_24026_sizes[] = {1};
    unsigned tensor_24026 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24026",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24026_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23645}, {tensor_24026}, nullptr, 0, "Reshape98378");

    /*************
     * Reshape98379 node
     * inputs: [tensor_23755[1](dtype=float32)]
     * output: [tensor_24027[1](dtype=float32)]
     *************/

    // create tensor_24027 tensor
    unsigned tensor_24027_sizes[] = {1};
    unsigned tensor_24027 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24027",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24027_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23755}, {tensor_24027}, nullptr, 0, "Reshape98379");

    /*************
     * Reshape98380 node
     * inputs: [tensor_23695[1](dtype=float32)]
     * output: [tensor_24028[1](dtype=float32)]
     *************/

    // create tensor_24028 tensor
    unsigned tensor_24028_sizes[] = {1};
    unsigned tensor_24028 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24028",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24028_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23695}, {tensor_24028}, nullptr, 0, "Reshape98380");

    /*************
     * Reshape98381 node
     * inputs: [tensor_23725[1](dtype=float32)]
     * output: [tensor_24029[1](dtype=float32)]
     *************/

    // create tensor_24029 tensor
    unsigned tensor_24029_sizes[] = {1};
    unsigned tensor_24029 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24029",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24029_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23725}, {tensor_24029}, nullptr, 0, "Reshape98381");

    /*************
     * Reshape98382 node
     * inputs: [tensor_23730[1](dtype=float32)]
     * output: [tensor_24030[1](dtype=float32)]
     *************/

    // create tensor_24030 tensor
    unsigned tensor_24030_sizes[] = {1};
    unsigned tensor_24030 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24030",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24030_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23730}, {tensor_24030}, nullptr, 0, "Reshape98382");

    /*************
     * Reshape98383 node
     * inputs: [tensor_24005[1](dtype=float32)]
     * output: [tensor_24031[1](dtype=float32)]
     *************/

    // create tensor_24031 tensor
    unsigned tensor_24031_sizes[] = {1};
    unsigned tensor_24031 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24031",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24031_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24005}, {tensor_24031}, nullptr, 0, "Reshape98383");

    /*************
     * Reshape98384 node
     * inputs: [tensor_23710[1](dtype=float32)]
     * output: [tensor_24032[1](dtype=float32)]
     *************/

    // create tensor_24032 tensor
    unsigned tensor_24032_sizes[] = {1};
    unsigned tensor_24032 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24032",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24032_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23710}, {tensor_24032}, nullptr, 0, "Reshape98384");

    /*************
     * Reshape98385 node
     * inputs: [tensor_23705[1](dtype=float32)]
     * output: [tensor_24033[1](dtype=float32)]
     *************/

    // create tensor_24033 tensor
    unsigned tensor_24033_sizes[] = {1};
    unsigned tensor_24033 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24033",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24033_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23705}, {tensor_24033}, nullptr, 0, "Reshape98385");

    /*************
     * Reshape98386 node
     * inputs: [tensor_23715[1](dtype=float32)]
     * output: [tensor_24034[1](dtype=float32)]
     *************/

    // create tensor_24034 tensor
    unsigned tensor_24034_sizes[] = {1};
    unsigned tensor_24034 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24034",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24034_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23715}, {tensor_24034}, nullptr, 0, "Reshape98386");

    /*************
     * Reshape98387 node
     * inputs: [tensor_23720[1](dtype=float32)]
     * output: [tensor_24035[1](dtype=float32)]
     *************/

    // create tensor_24035 tensor
    unsigned tensor_24035_sizes[] = {1};
    unsigned tensor_24035 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24035",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24035_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23720}, {tensor_24035}, nullptr, 0, "Reshape98387");

    /*************
     * Reshape98388 node
     * inputs: [tensor_24012[1](dtype=float32)]
     * output: [tensor_24036[1](dtype=float32)]
     *************/

    // create tensor_24036 tensor
    unsigned tensor_24036_sizes[] = {1};
    unsigned tensor_24036 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24036",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24036_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24012}, {tensor_24036}, nullptr, 0, "Reshape98388");

    /*************
     * Reshape98389 node
     * inputs: [tensor_23660[1](dtype=float32)]
     * output: [tensor_24037[1](dtype=float32)]
     *************/

    // create tensor_24037 tensor
    unsigned tensor_24037_sizes[] = {1};
    unsigned tensor_24037 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24037",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24037_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23660}, {tensor_24037}, nullptr, 0, "Reshape98389");

    /*************
     * Reshape98390 node
     * inputs: [tensor_23700[1](dtype=float32)]
     * output: [tensor_24038[1](dtype=float32)]
     *************/

    // create tensor_24038 tensor
    unsigned tensor_24038_sizes[] = {1};
    unsigned tensor_24038 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24038",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24038_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23700}, {tensor_24038}, nullptr, 0, "Reshape98390");

    /*************
     * Reshape98391 node
     * inputs: [tensor_23770[1](dtype=float32)]
     * output: [tensor_24039[1](dtype=float32)]
     *************/

    // create tensor_24039 tensor
    unsigned tensor_24039_sizes[] = {1};
    unsigned tensor_24039 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24039",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24039_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23770}, {tensor_24039}, nullptr, 0, "Reshape98391");

    /*************
     * Reshape98392 node
     * inputs: [tensor_23650[1](dtype=float32)]
     * output: [tensor_24040[1](dtype=float32)]
     *************/

    // create tensor_24040 tensor
    unsigned tensor_24040_sizes[] = {1};
    unsigned tensor_24040 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24040",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24040_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23650}, {tensor_24040}, nullptr, 0, "Reshape98392");

    /*************
     * Reshape98393 node
     * inputs: [tensor_23655[1](dtype=float32)]
     * output: [tensor_24041[1](dtype=float32)]
     *************/

    // create tensor_24041 tensor
    unsigned tensor_24041_sizes[] = {1};
    unsigned tensor_24041 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24041",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24041_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23655}, {tensor_24041}, nullptr, 0, "Reshape98393");

    /*************
     * Reshape98394 node
     * inputs: [tensor_23675[1](dtype=float32)]
     * output: [tensor_24042[1](dtype=float32)]
     *************/

    // create tensor_24042 tensor
    unsigned tensor_24042_sizes[] = {1};
    unsigned tensor_24042 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24042",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24042_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23675}, {tensor_24042}, nullptr, 0, "Reshape98394");

    /*************
     * Reshape98395 node
     * inputs: [tensor_23802[1](dtype=float32)]
     * output: [tensor_24043[1](dtype=float32)]
     *************/

    // create tensor_24043 tensor
    unsigned tensor_24043_sizes[] = {1};
    unsigned tensor_24043 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24043",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24043_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23802}, {tensor_24043}, nullptr, 0, "Reshape98395");

    /*************
     * Reshape98396 node
     * inputs: [tensor_23680[1](dtype=float32)]
     * output: [tensor_24044[1](dtype=float32)]
     *************/

    // create tensor_24044 tensor
    unsigned tensor_24044_sizes[] = {1};
    unsigned tensor_24044 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24044",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24044_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23680}, {tensor_24044}, nullptr, 0, "Reshape98396");

    /*************
     * Reshape98397 node
     * inputs: [tensor_23685[1](dtype=float32)]
     * output: [tensor_24045[1](dtype=float32)]
     *************/

    // create tensor_24045 tensor
    unsigned tensor_24045_sizes[] = {1};
    unsigned tensor_24045 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24045",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24045_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23685}, {tensor_24045}, nullptr, 0, "Reshape98397");

    /*************
     * Reshape98398 node
     * inputs: [tensor_23690[1](dtype=float32)]
     * output: [tensor_24046[1](dtype=float32)]
     *************/

    // create tensor_24046 tensor
    unsigned tensor_24046_sizes[] = {1};
    unsigned tensor_24046 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24046",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24046_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23690}, {tensor_24046}, nullptr, 0, "Reshape98398");

    /*************
     * Reshape98399 node
     * inputs: [tensor_23850[1](dtype=float32)]
     * output: [tensor_24047[1](dtype=float32)]
     *************/

    // create tensor_24047 tensor
    unsigned tensor_24047_sizes[] = {1};
    unsigned tensor_24047 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24047",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24047_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23850}, {tensor_24047}, nullptr, 0, "Reshape98399");

    /*************
     * Reshape98400 node
     * inputs: [tensor_23630[1](dtype=float32)]
     * output: [tensor_24048[1](dtype=float32)]
     *************/

    // create tensor_24048 tensor
    unsigned tensor_24048_sizes[] = {1};
    unsigned tensor_24048 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24048",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24048_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23630}, {tensor_24048}, nullptr, 0, "Reshape98400");

    /*************
     * Reshape98401 node
     * inputs: [tensor_23635[1](dtype=float32)]
     * output: [tensor_24049[1](dtype=float32)]
     *************/

    // create tensor_24049 tensor
    unsigned tensor_24049_sizes[] = {1};
    unsigned tensor_24049 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24049",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24049_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23635}, {tensor_24049}, nullptr, 0, "Reshape98401");

    /*************
     * Reshape98402 node
     * inputs: [tensor_23640[1](dtype=float32)]
     * output: [tensor_24050[1](dtype=float32)]
     *************/

    // create tensor_24050 tensor
    unsigned tensor_24050_sizes[] = {1};
    unsigned tensor_24050 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24050",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24050_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23640}, {tensor_24050}, nullptr, 0, "Reshape98402");

    /*************
     * Reshape98403 node
     * inputs: [tensor_23844[1](dtype=float32)]
     * output: [tensor_24051[1](dtype=float32)]
     *************/

    // create tensor_24051 tensor
    unsigned tensor_24051_sizes[] = {1};
    unsigned tensor_24051 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24051",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24051_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23844}, {tensor_24051}, nullptr, 0, "Reshape98403");

    /*************
     * Concatenate98404 node
     * inputs: [tensor_24014[1](dtype=float32), tensor_24015[1](dtype=float32), tensor_24016[1](dtype=float32), tensor_24017[1](dtype=float32), tensor_24018[1](dtype=float32), tensor_24019[1](dtype=float32), tensor_24020[1](dtype=float32), tensor_24021[1](dtype=float32), tensor_24022[1](dtype=float32), tensor_24023[1](dtype=float32), tensor_24024[1](dtype=float32), tensor_24025[1](dtype=float32), tensor_24026[1](dtype=float32), tensor_24027[1](dtype=float32), tensor_24028[1](dtype=float32), tensor_24029[1](dtype=float32), tensor_24030[1](dtype=float32), tensor_24031[1](dtype=float32), tensor_24032[1](dtype=float32), tensor_24033[1](dtype=float32), tensor_24034[1](dtype=float32), tensor_24035[1](dtype=float32), tensor_24036[1](dtype=float32), tensor_24037[1](dtype=float32), tensor_24038[1](dtype=float32), tensor_24039[1](dtype=float32), tensor_24040[1](dtype=float32), tensor_24041[1](dtype=float32), tensor_24042[1](dtype=float32), tensor_24043[1](dtype=float32), tensor_24044[1](dtype=float32), tensor_24045[1](dtype=float32), tensor_24046[1](dtype=float32), tensor_24047[1](dtype=float32), tensor_24048[1](dtype=float32), tensor_24049[1](dtype=float32), tensor_24050[1](dtype=float32), tensor_24051[1](dtype=float32)]
     * output: [tensor_24013[38](dtype=float32)]
     *************/

    // create tensor_24013 tensor
    unsigned tensor_24013_sizes[] = {38};
    unsigned tensor_24013 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24013",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24013_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char Concatenate98404_params[] = {0,0,0,0};
    addNodeToGraph("concat", {tensor_24014, tensor_24015, tensor_24016, tensor_24017, tensor_24018, tensor_24019, tensor_24020, tensor_24021, tensor_24022, tensor_24023, tensor_24024, tensor_24025, tensor_24026, tensor_24027, tensor_24028, tensor_24029, tensor_24030, tensor_24031, tensor_24032, tensor_24033, tensor_24034, tensor_24035, tensor_24036, tensor_24037, tensor_24038, tensor_24039, tensor_24040, tensor_24041, tensor_24042, tensor_24043, tensor_24044, tensor_24045, tensor_24046, tensor_24047, tensor_24048, tensor_24049, tensor_24050, tensor_24051}, {tensor_24013}, (void*)Concatenate98404_params, 4, "Concatenate98404");

    /*************
     * TPC98405 node
     * inputs: [tensor_24013[38](dtype=float32)]
     * output: [tensor_24053[1](dtype=float32)]
     *************/

    // create tensor_24053 tensor
    unsigned tensor_24053_sizes[] = {1};
    unsigned tensor_24053 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24053",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24053_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98405_params[] = {0,0,0,0};
    addNodeToGraph("reduce_sum_fwd_f32", {tensor_24013}, {tensor_24053}, (void*)TPC98405_params, 4, "TPC98405");

    /*************
     * Reshape98406 node
     * inputs: [tensor_24053[1](dtype=float32)]
     * output: [tensor_24052[1](dtype=float32)]
     *************/

    // create tensor_24052 tensor
    unsigned tensor_24052_sizes[] = {1};
    unsigned tensor_24052 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24052",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24052_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24053}, {tensor_24052}, nullptr, 0, "Reshape98406");

    /*************
     * Reshape98407 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_24057[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24057 tensor
    unsigned tensor_24057_sizes[] = {1,1,1,1};
    unsigned tensor_24057 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24057",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24057_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_24057}, nullptr, 0, "Reshape98407");

    /*************
     * TPC98408 node
     * inputs: [tensor_24057[1, 1, 1, 1](dtype=float32), tensor_24055[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24056[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24055 tensor
    unsigned tensor_24055_sizes[] = {1,704,320,2};
    unsigned tensor_24055 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24055",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24055_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_24056 tensor
    unsigned tensor_24056_sizes[] = {1,704,320,2};
    unsigned tensor_24056 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24056",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24056_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_24057, tensor_24055}, {tensor_24056}, nullptr, 0, "TPC98408");

    /*************
     * TPC98409 node
     * inputs: [tensor_23958[1, 704, 320, 2](dtype=float32), tensor_24056[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24058[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24058 tensor
    unsigned tensor_24058_sizes[] = {1,704,320,2};
    unsigned tensor_24058 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24058",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24058_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23958, tensor_24056}, {tensor_24058}, nullptr, 0, "TPC98409");

    /*************
     * Reshape98410 node
     * inputs: [tensor_23851[1](dtype=float32)]
     * output: [tensor_24060[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24060 tensor
    unsigned tensor_24060_sizes[] = {1,1,1,1};
    unsigned tensor_24060 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24060",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24060_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_23851}, {tensor_24060}, nullptr, 0, "Reshape98410");

    /*************
     * TPC98411 node
     * inputs: [tensor_24060[1, 1, 1, 1](dtype=float32), tensor_24058[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24059[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24059 tensor
    unsigned tensor_24059_sizes[] = {1,704,320,2};
    unsigned tensor_24059 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24059",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24059_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_24060, tensor_24058}, {tensor_24059}, nullptr, 0, "TPC98411");

    /*************
     * Reshape98412 node
     * inputs: [tensor_24059[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24062[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_24062 tensor
    unsigned tensor_24062_sizes[] = {225280,1,1,2};
    unsigned tensor_24062 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24062",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24062_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24059}, {tensor_24062}, nullptr, 0, "Reshape98412");

    /*************
     * TPC98413 node
     * inputs: [tensor_24062[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_24063[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_24063 tensor
    unsigned tensor_24063_sizes[] = {1,1,1,2};
    unsigned tensor_24063 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24063",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24063_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98413_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24062}, {tensor_24063}, (void*)TPC98413_params, 4, "TPC98413");

    /*************
     * Reshape98414 node
     * inputs: [tensor_24063[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_24061[2](dtype=float32)]
     *************/

    // create tensor_24061 tensor
    unsigned tensor_24061_sizes[] = {2};
    unsigned tensor_24061 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24061",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24061_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24063}, {tensor_24061}, nullptr, 0, "Reshape98414");

    /*************
     * TPC98415 node
     * inputs: [tensor_23851[1](dtype=float32), tensor_24061[2](dtype=float32)]
     * output: [tensor_24065[2](dtype=float32)]
     *************/

    // create tensor_24065 tensor
    unsigned tensor_24065_sizes[] = {2};
    unsigned tensor_24065 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24065",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24065_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23851, tensor_24061}, {tensor_24065}, nullptr, 0, "TPC98415");

    /*************
     * TPC98416 node
     * inputs: [tensor_24065[2](dtype=float32), tensor_23513[1](dtype=float32)]
     * output: [tensor_24066[2](dtype=float32)]
     *************/

    // create tensor_24066 tensor
    unsigned tensor_24066_sizes[] = {2};
    unsigned tensor_24066 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24066",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24066_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_24065, tensor_23513}, {tensor_24066}, nullptr, 0, "TPC98416");

    /*************
     * Reshape98417 node
     * inputs: [tensor_24059[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24068[450560, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24068 tensor
    unsigned tensor_24068_sizes[] = {450560,1,1,1};
    unsigned tensor_24068 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24068",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24068_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24059}, {tensor_24068}, nullptr, 0, "Reshape98417");

    /*************
     * TPC98418 node
     * inputs: [tensor_24068[450560, 1, 1, 1](dtype=float32)]
     * output: [tensor_24069[1, 1, 1, 1](dtype=float32)]
     *************/

    // create tensor_24069 tensor
    unsigned tensor_24069_sizes[] = {1,1,1,1};
    unsigned tensor_24069 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24069",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24069_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98418_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24068}, {tensor_24069}, (void*)TPC98418_params, 4, "TPC98418");

    /*************
     * Reshape98419 node
     * inputs: [tensor_24069[1, 1, 1, 1](dtype=float32)]
     * output: [tensor_24067[1](dtype=float32)]
     *************/

    // create tensor_24067 tensor
    unsigned tensor_24067_sizes[] = {1};
    unsigned tensor_24067 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24067",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24067_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24069}, {tensor_24067}, nullptr, 0, "Reshape98419");

    /*************
     * TPC98420 node
     * inputs: [tensor_23851[1](dtype=float32), tensor_24067[1](dtype=float32)]
     * output: [tensor_24071[1](dtype=float32)]
     *************/

    // create tensor_24071 tensor
    unsigned tensor_24071_sizes[] = {1};
    unsigned tensor_24071 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24071",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24071_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23851, tensor_24067}, {tensor_24071}, nullptr, 0, "TPC98420");

    /*************
     * TPC98421 node
     * inputs: [tensor_24071[1](dtype=float32), tensor_23513[1](dtype=float32)]
     * output: [tensor_24072[1](dtype=float32)]
     *************/

    // create tensor_24072 tensor
    unsigned tensor_24072_sizes[] = {1};
    unsigned tensor_24072 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24072",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24072_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("max_fwd_f32", {tensor_24071, tensor_23513}, {tensor_24072}, nullptr, 0, "TPC98421");

    /*************
     * TPC98422 node
     * inputs: [tensor_23557[1, 704, 320, 2](dtype=float32), tensor_24058[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24073[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24073 tensor
    unsigned tensor_24073_sizes[] = {1,704,320,2};
    unsigned tensor_24073 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24073",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24073_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23557, tensor_24058}, {tensor_24073}, nullptr, 0, "TPC98422");

    /*************
     * TPC98423 node
     * inputs: [tensor_23623[1, 704, 320, 2](dtype=float32), tensor_24073[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24074[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24074 tensor
    unsigned tensor_24074_sizes[] = {1,704,320,2};
    unsigned tensor_24074 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24074",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24074_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("sub_fwd_f32", {tensor_23623, tensor_24073}, {tensor_24074}, nullptr, 0, "TPC98423");

    /*************
     * TPC98424 node
     * inputs: [tensor_24074[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24075[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24075 tensor
    unsigned tensor_24075_sizes[] = {1,704,320,2};
    unsigned tensor_24075 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24075",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24075_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("abs_fwd_f32", {tensor_24074}, {tensor_24075}, nullptr, 0, "TPC98424");

    /*************
     * Reshape98425 node
     * inputs: [tensor_24073[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24077[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_24077 tensor
    unsigned tensor_24077_sizes[] = {225280,1,1,2};
    unsigned tensor_24077 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24077",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24077_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24073}, {tensor_24077}, nullptr, 0, "Reshape98425");

    /*************
     * TPC98426 node
     * inputs: [tensor_24077[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_24078[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_24078 tensor
    unsigned tensor_24078_sizes[] = {1,1,1,2};
    unsigned tensor_24078 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24078",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24078_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98426_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24077}, {tensor_24078}, (void*)TPC98426_params, 4, "TPC98426");

    /*************
     * Reshape98427 node
     * inputs: [tensor_24078[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_24076[2](dtype=float32)]
     *************/

    // create tensor_24076 tensor
    unsigned tensor_24076_sizes[] = {2};
    unsigned tensor_24076 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24076",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24076_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24078}, {tensor_24076}, nullptr, 0, "Reshape98427");

    /*************
     * TPC98428 node
     * inputs: [tensor_24076[2](dtype=float32), tensor_24072[1](dtype=float32)]
     * output: [tensor_24080[2](dtype=float32)]
     *************/

    // create tensor_24080 tensor
    unsigned tensor_24080_sizes[] = {2};
    unsigned tensor_24080 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24080",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24080_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_24076, tensor_24072}, {tensor_24080}, nullptr, 0, "TPC98428");

    /*************
     * TPC98429 node
     * inputs: [tensor_23558[1](dtype=float32), tensor_24080[2](dtype=float32)]
     * output: [tensor_24081[2](dtype=float32)]
     *************/

    // create tensor_24081 tensor
    unsigned tensor_24081_sizes[] = {2};
    unsigned tensor_24081 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24081",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24081_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23558, tensor_24080}, {tensor_24081}, nullptr, 0, "TPC98429");

    /*************
     * TPC98430 node
     * inputs: [tensor_24081[2](dtype=float32), tensor_23478[1](dtype=float32)]
     * output: [tensor_24082[2](dtype=float32)]
     *************/

    // create tensor_24082 tensor
    unsigned tensor_24082_sizes[] = {2};
    unsigned tensor_24082 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24082",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24082_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_24081, tensor_23478}, {tensor_24082}, nullptr, 0, "TPC98430");

    /*************
     * TPC98431 node
     * inputs: [tensor_24081[2](dtype=float32)]
     * output: [tensor_24084[1](dtype=float32)]
     *************/

    // create tensor_24084 tensor
    unsigned tensor_24084_sizes[] = {1};
    unsigned tensor_24084 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24084",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24084_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98431_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24081}, {tensor_24084}, (void*)TPC98431_params, 4, "TPC98431");

    /*************
     * Reshape98432 node
     * inputs: [tensor_24084[1](dtype=float32)]
     * output: [tensor_24083[1](dtype=float32)]
     *************/

    // create tensor_24083 tensor
    unsigned tensor_24083_sizes[] = {1};
    unsigned tensor_24083 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24083",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24083_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24084}, {tensor_24083}, nullptr, 0, "Reshape98432");

    /*************
     * TPC98433 node
     * inputs: [tensor_24075[1, 704, 320, 2](dtype=float32), tensor_24058[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24086[1, 704, 320, 2](dtype=float32)]
     *************/

    // create tensor_24086 tensor
    unsigned tensor_24086_sizes[] = {1,704,320,2};
    unsigned tensor_24086 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24086",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24086_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_24075, tensor_24058}, {tensor_24086}, nullptr, 0, "TPC98433");

    /*************
     * Reshape98434 node
     * inputs: [tensor_24086[1, 704, 320, 2](dtype=float32)]
     * output: [tensor_24088[225280, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_24088 tensor
    unsigned tensor_24088_sizes[] = {225280,1,1,2};
    unsigned tensor_24088 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24088",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24088_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24086}, {tensor_24088}, nullptr, 0, "Reshape98434");

    /*************
     * TPC98435 node
     * inputs: [tensor_24088[225280, 1, 1, 2](dtype=float32)]
     * output: [tensor_24089[1, 1, 1, 2](dtype=float32)]
     *************/

    // create tensor_24089 tensor
    unsigned tensor_24089_sizes[] = {1,1,1,2};
    unsigned tensor_24089 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24089",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24089_sizes,
                                        4,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98435_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24088}, {tensor_24089}, (void*)TPC98435_params, 4, "TPC98435");

    /*************
     * Reshape98436 node
     * inputs: [tensor_24089[1, 1, 1, 2](dtype=float32)]
     * output: [tensor_24087[2](dtype=float32)]
     *************/

    // create tensor_24087 tensor
    unsigned tensor_24087_sizes[] = {2};
    unsigned tensor_24087 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24087",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24087_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24089}, {tensor_24087}, nullptr, 0, "Reshape98436");

    /*************
     * TPC98437 node
     * inputs: [tensor_24087[2](dtype=float32), tensor_24066[2](dtype=float32)]
     * output: [tensor_24091[2](dtype=float32)]
     *************/

    // create tensor_24091 tensor
    unsigned tensor_24091_sizes[] = {2};
    unsigned tensor_24091 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24091",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24091_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("div_fwd_f32", {tensor_24087, tensor_24066}, {tensor_24091}, nullptr, 0, "TPC98437");

    /*************
     * TPC98438 node
     * inputs: [tensor_23478[1](dtype=float32), tensor_24091[2](dtype=float32)]
     * output: [tensor_24092[2](dtype=float32)]
     *************/

    // create tensor_24092 tensor
    unsigned tensor_24092_sizes[] = {2};
    unsigned tensor_24092 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24092",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24092_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("mult_fwd_f32", {tensor_23478, tensor_24091}, {tensor_24092}, nullptr, 0, "TPC98438");

    /*************
     * TPC98439 node
     * inputs: [tensor_23947[2](dtype=float32), tensor_24082[2](dtype=float32)]
     * output: [tensor_24094[2](dtype=float32)]
     *************/

    // create tensor_24094 tensor
    unsigned tensor_24094_sizes[] = {2};
    unsigned tensor_24094 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24094",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24094_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23947, tensor_24082}, {tensor_24094}, nullptr, 0, "TPC98439");

    /*************
     * TPC98440 node
     * inputs: [tensor_23986[2](dtype=float32), tensor_23954[2](dtype=float32)]
     * output: [tensor_24095[2](dtype=float32)]
     *************/

    // create tensor_24095 tensor
    unsigned tensor_24095_sizes[] = {2};
    unsigned tensor_24095 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24095",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24095_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_23986, tensor_23954}, {tensor_24095}, nullptr, 0, "TPC98440");

    /*************
     * TPC98441 node
     * inputs: [tensor_24092[2](dtype=float32), tensor_23998[2](dtype=float32)]
     * output: [tensor_24096[2](dtype=float32)]
     *************/

    // create tensor_24096 tensor
    unsigned tensor_24096_sizes[] = {2};
    unsigned tensor_24096 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24096",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24096_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_24092, tensor_23998}, {tensor_24096}, nullptr, 0, "TPC98441");

    /*************
     * TPC98442 node
     * inputs: [tensor_24094[2](dtype=float32), tensor_24095[2](dtype=float32)]
     * output: [tensor_24097[2](dtype=float32)]
     *************/

    // create tensor_24097 tensor
    unsigned tensor_24097_sizes[] = {2};
    unsigned tensor_24097 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24097",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24097_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_24094, tensor_24095}, {tensor_24097}, nullptr, 0, "TPC98442");

    /*************
     * TPC98443 node
     * inputs: [tensor_24096[2](dtype=float32), tensor_24097[2](dtype=float32)]
     * output: [tensor_24093[2](dtype=float32)]
     *************/

    // create tensor_24093 tensor
    unsigned tensor_24093_sizes[] = {2};
    unsigned tensor_24093 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24093",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24093_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_24096, tensor_24097}, {tensor_24093}, nullptr, 0, "TPC98443");

    /*************
     * TPC98444 node
     * inputs: [tensor_24093[2](dtype=float32)]
     * output: [tensor_24099[1](dtype=float32)]
     *************/

    // create tensor_24099 tensor
    unsigned tensor_24099_sizes[] = {1};
    unsigned tensor_24099 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24099",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24099_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98444_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24093}, {tensor_24099}, (void*)TPC98444_params, 4, "TPC98444");

    /*************
     * Reshape98445 node
     * inputs: [tensor_24099[1](dtype=float32)]
     * output: [tensor_24098[1](dtype=float32)]
     *************/

    // create tensor_24098 tensor
    unsigned tensor_24098_sizes[] = {1};
    unsigned tensor_24098 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24098",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24098_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24099}, {tensor_24098}, nullptr, 0, "Reshape98445");

    /*************
     * TPC98446 node
     * inputs: [tensor_24098[1](dtype=float32), tensor_24052[1](dtype=float32)]
     * output: [tensor_24101[1](dtype=float32)]
     *************/

    // create tensor_24101 tensor
    unsigned tensor_24101_sizes[] = {1};
    unsigned tensor_24101 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24101",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24101_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_24098, tensor_24052}, {tensor_24101}, nullptr, 0, "TPC98446");

    /*************
     * TPC98447 node
     * inputs: [tensor_24102[1](dtype=float32), tensor_23851[1](dtype=float32)]
     * output: [tensor_24103[1](dtype=float32)]
     *************/

    // create tensor_24102 tensor
    unsigned tensor_24102_sizes[] = {1};
    unsigned tensor_24102 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24102",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24102_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_24103 tensor
    unsigned tensor_24103_sizes[] = {1};
    unsigned tensor_24103 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24103",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24103_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_24102, tensor_23851}, {tensor_24103}, nullptr, 0, "TPC98447");

    /*************
     * TPC98448 node
     * inputs: [tensor_24104[1](dtype=float32), tensor_24101[1](dtype=float32)]
     * output: [tensor_24105[1](dtype=float32)]
     *************/

    // create tensor_24104 tensor
    unsigned tensor_24104_sizes[] = {1};
    unsigned tensor_24104 = createTensors(1,
                                        INPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24104",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24104_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];

    // create tensor_24105 tensor
    unsigned tensor_24105_sizes[] = {1};
    unsigned tensor_24105 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24105",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24105_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("add_fwd_f32", {tensor_24104, tensor_24101}, {tensor_24105}, nullptr, 0, "TPC98448");

    /*************
     * TPC98449 node
     * inputs: [tensor_24091[2](dtype=float32)]
     * output: [tensor_24107[1](dtype=float32)]
     *************/

    // create tensor_24107 tensor
    unsigned tensor_24107_sizes[] = {1};
    unsigned tensor_24107 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, // isPersistent
                                        "tensor_24107",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24107_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    unsigned char TPC98449_params[] = {0,0,0,0};
    addNodeToGraph("reduce_mean_fwd_f32", {tensor_24091}, {tensor_24107}, (void*)TPC98449_params, 4, "TPC98449");

    /*************
     * Reshape98450 node
     * inputs: [tensor_24107[1](dtype=float32)]
     * output: [tensor_24106[1](dtype=float32)]
     *************/

    // create tensor_24106 tensor
    unsigned tensor_24106_sizes[] = {1};
    unsigned tensor_24106 = createTensors(1,
                                        OUTPUT_TENSOR,
                                        true, // isPersistent
                                        "tensor_24106",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr, // initializer
                                        tensor_24106_sizes,
                                        1,
                                        syn_type_single,
                                        nullptr,
                                        0,
                                        0,
                                        nullptr,
                                        false)[0];
    addNodeToGraph("reshape", {tensor_24107}, {tensor_24106}, nullptr, 0, "Reshape98450");


    compileTopology();
}
