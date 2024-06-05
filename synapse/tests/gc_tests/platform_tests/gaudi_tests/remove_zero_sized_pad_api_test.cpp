#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "syn_gaudi_two_run_compare_test.h"

class RemoveZeroSizedPadAPITest : public SynTrainingTestInfra
{
};

TEST_F_GC(RemoveZeroSizedPadAPITest, simple_api_one_zero_pad)
{
    //                 ________            ___
    // Creating:      |        |          |   |
    //          t1->->|Zero_Pad|->->t2->->|Add|->->t4
    //                |________|    t3->->|___|
    //
    //                 ___
    // Expecting:      |   |
    //          t1->->|Add|->->t4
    //          t3->->|___|

    unsigned tensor_size[] = {4, 4, 4, 4};
    unsigned t1            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_int32,
                                      nullptr);
    unsigned t2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_int32, nullptr);
    unsigned t3 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_int32,
                                      nullptr);
    unsigned t4 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_int32, nullptr);

    ns_PadKernelEx::Params params {};
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));

    addNodeToGraph("pad_fwd_i32", {t1}, {t2}, &params, sizeof(params), "pad1");
    addNodeToGraph("add_fwd_i32", {t2, t3}, {t4}, nullptr, 0, "add1");

    compileAndRun();
    int* in1  = castHostBuffer<int>(t1);
    int* in2  = castHostBuffer<int>(t3);
    int* out  = castHostBuffer<int>(t4);
    int  size = tensor_size[0] * tensor_size[1] * tensor_size[2] * tensor_size[3];
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(in1[i] + in2[i], out[i]);
    }
}

TEST_F_GC(RemoveZeroSizedPadAPITest, simple_api_one_zero_pad_2)
{
    //                 ___            ________            ___
    // Creating: t1-->|   |          |        |          |   |
    //                |Add|->->t3->->|Zero_Pad|->->t4->->|Add|->->t6
    //           t2-->|___|          |________|    t5->->|___|
    //
    //                 ___              ___
    // Expecting:     |   |            |   |
    //          t1->->|Add|-->-->t3->->|Add|->->t6
    //          t2->->|___|      t5->->|___|

    unsigned tensor_size[] = {4, 4, 4, 4};
    unsigned t1            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_int32,
                                      nullptr);
    unsigned t2            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_int32,
                                      nullptr);
    unsigned t3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_int32, nullptr);
    unsigned t4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_int32, nullptr);
    unsigned t5 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_int32,
                                      nullptr);
    unsigned t6 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_int32, nullptr);

    ns_PadKernelEx::Params params {};
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));

    addNodeToGraph("add_fwd_i32", {t1, t2}, {t3}, nullptr, 0, "add1");
    addNodeToGraph("pad_fwd_i32", {t3}, {t4}, &params, sizeof(params), "pad1");
    addNodeToGraph("add_fwd_i32", {t4, t5}, {t6}, nullptr, 0, "add2");

    compileAndRun();
    int* in1  = castHostBuffer<int>(t1);
    int* in2  = castHostBuffer<int>(t2);
    int* in3  = castHostBuffer<int>(t5);
    int* out  = castHostBuffer<int>(t6);
    int  size = tensor_size[0] * tensor_size[1] * tensor_size[2] * tensor_size[3];
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(in1[i] + in2[i] + in3[i], out[i]);
    }
}

TEST_F_GC(RemoveZeroSizedPadAPITest, graph_output_api_zero_pad)
{
    //                                                      ___
    //                               |-->-->-->-->-->-->-->|   |
    //                 ___           |    ________         |   |
    // Creating:      |   |          |   |        |        |Add|-->t5
    //          t1->->|Add|->->t3->->|-->|Zero_Pad|-->t4-->|___|
    //          t2->->|___|          |   |________|
    //                               |       ________
    //                               |      |        |
    //                               |-->-->|Zero_Pad|-->t6 (t6 is an output tensor of the graph)
    //                                      |________|
    //
    //                 ___                      ___
    // Expecting:     |   |          |-->-->-->|Add|-->t5
    //          t1->->|Add|->->t6->->|-->-->-->|___|
    //          t2->->|___|

    unsigned tensor_size[] = {4};
    unsigned t1            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      1,
                                      syn_type_int32,
                                      nullptr);
    unsigned t2            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      1,
                                      syn_type_int32,
                                      nullptr);
    unsigned t3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 1, syn_type_int32, nullptr);
    unsigned t4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 1, syn_type_int32, nullptr);
    unsigned t5 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 1, syn_type_int32, nullptr);
    unsigned t6 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 1, syn_type_int32, nullptr);

    ns_PadKernelEx::Params params {};
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));

    addNodeToGraph("add_fwd_i32", {t1, t2}, {t3}, nullptr, 0, "add1");
    addNodeToGraph("pad_fwd_i32", {t3}, {t4}, &params, sizeof(params), "pad1");
    addNodeToGraph("add_fwd_i32", {t3, t4}, {t5}, nullptr, 0, "add2");
    addNodeToGraph("pad_fwd_i32", {t3}, {t6}, &params, sizeof(params), "pad2");

    compileAndRun();
    int* in1  = castHostBuffer<int>(t1);
    int* in2  = castHostBuffer<int>(t2);
    int* out1 = castHostBuffer<int>(t5);
    int* out2 = castHostBuffer<int>(t6);
    int  size = tensor_size[0];
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(2 * (in1[i] + in2[i]), out1[i]);
        ASSERT_EQ(in1[i] + in2[i], out2[i]);
    }
}