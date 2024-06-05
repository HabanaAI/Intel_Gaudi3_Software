#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "syn_base_test.h"
#include "infra/gc_synapse_test.h"
#include "tensor_io_indicies.h"
#include "timer.h"
#include <tuple>
#include <deque>
#include <vector>

class SynGaudiTestHost2Device : public SynGaudiTestInfra
{
public:
    SynGaudiTestHost2Device() { setTestPackage(TEST_PACKAGE_DSD); }
};

TEST_F_GC(SynGaudiTestHost2Device, basic, {synDeviceGaudi, synDeviceGaudi2})
{
    const unsigned range_size           = 3;
    unsigned       range_sizes[]        = {range_size};
    unsigned       range_compile_data[] = {0, 100, 1, 0, 2, 1};
    unsigned       range_actual_data[]  = {0, 8, 2};
    unsigned       outputMinSizes[]     = {2};
    unsigned       outputMaxSizes[]     = {100};
    unsigned       outputActualSizes[]  = {4};

    auto t0 = createHost2DeviceTensor(INPUT_TENSOR, range_sizes, range_compile_data, 1, "h2d");
    auto t1 = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  outputMaxSizes,
                                  1,
                                  syn_type_uint32,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  outputMinSizes);

    addNodeToGraph("range_i32", {t0}, {t1});

    compileTopology();

    setActualSizes(t1, outputActualSizes);
    setActualScalarParametersData(t0, range_actual_data, sizeof(range_actual_data));

    runTopology();

    const uint32_t* outBuf = reinterpret_cast<const uint32_t*>(m_hostBuffers[t1]);

    unsigned value = range_actual_data[0];
    unsigned delta = range_actual_data[2];
    (void)outBuf;
    for (unsigned i = 0; i < outputActualSizes[0]; ++i, value += delta)
    {
        ASSERT_EQ(outBuf[i], value) << "at index " << i;
    }
}
TEST_F_GC(SynGaudiTestHost2Device, intermediate_complex_guid, {synDeviceGaudi, synDeviceGaudi2})
{
    unsigned in0_max_sizes[] = {8, 4000, 20, 2, 2};
    unsigned in0_min_sizes[] = {2, 1000, 5, 1, 1};
    unsigned in0_act_sizes[] = {4, 1000, 5, 1, 1};
    unsigned in0             = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "IN0",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 in0_max_sizes,
                                 5,
                                 syn_type_single,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in0_min_sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned in1_max_sizes[] = {8, 1, 20, 2, 2};
    unsigned in1_min_sizes[] = {2, 1, 5, 1, 1};
    unsigned in1_act_sizes[] = {4, 1, 5, 1, 1};
    unsigned in1             = createTensors(1,
                                 INPUT_TENSOR,
                                 true,
                                 "IN1",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 in1_max_sizes,
                                 5,
                                 syn_type_single,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 in1_min_sizes,
                                 synTensorType::DATA_TENSOR)[0];

    unsigned  out_max_sizes[] = {8, 4000, 20, 2, 2};
    unsigned  out_min_sizes[] = {2, 1000, 5, 1, 1};
    unsigned  out_act_sizes[] = {4, 1000, 5, 1, 1};
    unsigned  out             = createTensors(1,
                                 OUTPUT_TENSOR,
                                 true,
                                 "OUT",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 out_max_sizes,
                                 5,
                                 syn_type_single,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 out_min_sizes,
                                 synTensorType::DATA_TENSOR)[0];
    synNodeId add_fwd_complex_0_id;
    addNodeToGraph("add_fwd_f32", {in0, in1}, {out}, nullptr, 0, "add_fwd_complex_0", 0, &add_fwd_complex_0_id);

    compileTopology("intermediate_complex_guid", 0);
    setActualSizes(in0, in0_act_sizes);
    setActualSizes(in1, in1_act_sizes);
    setActualSizes(out, out_act_sizes);
    runTopology();
}

TEST_F_GC(SynGaudiTestHost2Device, big_pad, {synDeviceGaudi, synDeviceGaudi2})
{
    unsigned inputMaxSize     = 270000000;
    unsigned inputMinSize     = inputMaxSize - 1200;
    unsigned padMaxValue      = 150;
    unsigned padMinValue      = 20;
    unsigned inputActualSize  = (inputMinSize + inputMaxSize) / 2;
    unsigned padActualValue   = (padMinValue + padMaxValue) / 2;
    unsigned outputMinSize    = inputMinSize + padMinValue;
    unsigned outputMaxSize    = inputMaxSize + padMaxValue;
    unsigned outputActualSize = inputActualSize + padActualValue;

    const unsigned pad_size      = 10;
    unsigned       pad_sizes[]   = {pad_size};
    unsigned pad_compile_data[]  = {padMaxValue, 0, 0, 0, 0, 0, 0, 0, 0, 0, padMinValue, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned pad_actual_data[]   = {padActualValue, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned inputMinSizes[]     = {inputMinSize};
    unsigned inputMaxSizes[]     = {inputMaxSize};
    unsigned outputMinSizes[]    = {outputMinSize};
    unsigned outputMaxSizes[]    = {outputMaxSize};
    unsigned inputActualSizes[]  = {inputActualSize};
    unsigned outputActualSizes[] = {outputActualSize};

    auto th = createHost2DeviceTensor(INPUT_TENSOR, pad_sizes, pad_compile_data, 1, "h2d");
    auto t1 = createPersistTensor(INPUT_TENSOR,
                                  MEM_INIT_RANDOM_POSITIVE,
                                  nullptr,
                                  inputMaxSizes,
                                  1,
                                  syn_type_single,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  inputMinSizes);
    auto t2 = createPersistTensor(OUTPUT_TENSOR,
                                  MEM_INIT_NONE,
                                  nullptr,
                                  outputMaxSizes,
                                  1,
                                  syn_type_single,
                                  nullptr,
                                  nullptr,
                                  0,
                                  0,
                                  nullptr,
                                  outputMinSizes);

    ns_PadKernelEx::Params padParams {};
    padParams.mode    = PAD_MODE_CONSTANT;
    padParams.value.f = 0.f;

    addNodeToGraph("pad_fwd_f32", {t1, th}, {t2}, &padParams, sizeof padParams, "PAD");

    compileTopology();

    setActualSizes(t1, inputActualSizes);
    setActualSizes(t2, outputActualSizes);
    setActualScalarParametersData(th, pad_actual_data, sizeof(pad_actual_data));

    runTopology();

    const float* inBuf  = reinterpret_cast<const float*>(m_hostBuffers[t1]);
    const float* outBuf = reinterpret_cast<const float*>(m_hostBuffers[t2]);

    for (unsigned i = 0; i < padActualValue; ++i)
    {
        ASSERT_EQ(0, outBuf[i]) << " bad pad at index " << i;
    }
    for (unsigned i = 0; i < inputActualSize; ++i)
    {
        ASSERT_EQ(inBuf[i], outBuf[i + padActualValue]) << " bad original value at index " << i;
    }
}
