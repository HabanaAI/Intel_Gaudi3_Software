#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "syn_gaudi_two_run_compare_test.h"

class SynTrainingTestInfraFP8CastInjection : public SynTrainingTestInfra
{
};

TEST_F_GC(SynTrainingTestInfraFP8CastInjection, simple_fp8_cast_injection_test, {synDeviceGaudi2})
{
    //                 ________            ___
    // Creating:      |        |         |   |
    //          t1->->|Zero_Pad|->->t2->->|Add|->->t4
    //                |________|    t3->->|___|
    //
    //

    setGraphInferenceMode();
    unsigned tensor_size[] = {4, 4, 4, 4};
    unsigned t1            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr);
    unsigned t2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_fp8_152, nullptr);
    unsigned t3 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr);
    unsigned t4 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_fp8_152, nullptr);

    unsigned             tensors[] = {t1, t2, t3, t4};
    synQuantDynamicRange dynamicRange;
    dynamicRange.max = 10;
    dynamicRange.min = -10;
    for (unsigned tensor : tensors)
    {
        ASSERT_EQ(synSuccess,
                  synTensorSetQuantizationData(getTensorByIndex(tensor),
                                               SYN_QUANT_DYNAMIC_RANGE,
                                               &dynamicRange,
                                               sizeof(synQuantDynamicRange)));
    }

    ns_PadKernelEx::Params params {};
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));

    addNodeToGraph("pad_fwd_bf16", {t1}, {t2}, &params, sizeof(params), "pad1");
    addNodeToGraph("add_fwd_bf16", {t2, t3}, {t4}, nullptr, 0, "add1");

    compileAndRun();
    fp8_152_t* in1  = castHostBuffer<fp8_152_t>(t1);
    fp8_152_t* in2  = castHostBuffer<fp8_152_t>(t3);
    fp8_152_t* out  = castHostBuffer<fp8_152_t>(t4);
    int        size = tensor_size[0] * tensor_size[1] * tensor_size[2] * tensor_size[3];
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(in1[i].toFloat() + in2[i].toFloat(), out[i].toFloat());
    }
}

TEST_F_GC(SynTrainingTestInfraFP8CastInjection, simple_fp8_cast_injection_test_2, {synDeviceGaudi2})
{
    //                 ___            ________            ___
    // Creating: t1-->|   |          |        |          |   |
    //                |Add|->->t3->->|Zero_Pad|->->t4->->|Add|->->t6
    //           t2-->|___|          |________|    t5->->|___|
    //

    setGraphInferenceMode();
    unsigned tensor_size[] = {4, 4, 4, 4};
    unsigned t1            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr);
    unsigned t2            = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr);
    unsigned t3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_fp8_152, nullptr);
    unsigned t4 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_fp8_152, nullptr);
    unsigned t5 = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      tensor_size,
                                      4,
                                      syn_type_fp8_152,
                                      nullptr);
    unsigned t6 =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_size, 4, syn_type_fp8_152, nullptr);

    unsigned             tensors[] = {t1, t2, t3, t4, t5, t6};
    synQuantDynamicRange dynamicRange;
    dynamicRange.max = 10;
    dynamicRange.min = -10;
    for (unsigned tensor : tensors)
    {
        ASSERT_EQ(synSuccess,
                  synTensorSetQuantizationData(getTensorByIndex(tensor),
                                               SYN_QUANT_DYNAMIC_RANGE,
                                               &dynamicRange,
                                               sizeof(synQuantDynamicRange)));
    }
    ns_PadKernelEx::Params params {};
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));

    addNodeToGraph("add_fwd_bf16", {t1, t2}, {t3}, nullptr, 0, "add1");
    addNodeToGraph("pad_fwd_bf16", {t3}, {t4}, &params, sizeof(params), "pad1");
    addNodeToGraph("add_fwd_bf16", {t4, t5}, {t6}, nullptr, 0, "add2");

    compileAndRun();
    fp8_152_t* in1  = castHostBuffer<fp8_152_t>(t1);
    fp8_152_t* in2  = castHostBuffer<fp8_152_t>(t2);
    fp8_152_t* in3  = castHostBuffer<fp8_152_t>(t5);
    fp8_152_t* out  = castHostBuffer<fp8_152_t>(t6);
    int        size = tensor_size[0] * tensor_size[1] * tensor_size[2] * tensor_size[3];
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(in1[i].toFloat() + in2[i].toFloat() + in3[i].toFloat(), out[i].toFloat());
    }
}