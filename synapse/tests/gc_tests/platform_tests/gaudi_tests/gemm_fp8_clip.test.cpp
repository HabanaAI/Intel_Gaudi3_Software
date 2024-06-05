#include "gc_gaudi_test_infra.h"
#include "mme_reference/data_types/fp8.h"
#include "quantization_data.h"
#include <data_types/fp8.h>

template <typename CastFromType, typename CastToType>
void castFp8ToAnyFloat(CastFromType*           fromBuffer,
                       CastToType*             toBuffer,
                       unsigned                elementsNum,
                       const QuantizationData& fromQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(!std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        // cast first to fp32 then to target type
        toBuffer[i] = CastToType(fromBuffer[i].toFloat(fromQuantInfo.expBias()));
    }
}

class SynGaudiGemmfp8ClipTest : public SynTrainingTestInfra
{
};

TEST_F_GC(SynGaudiGemmfp8ClipTest, gemmClipTest, {synDeviceGaudi2, synDeviceGaudi3})
{
    pushGlobalConf("USE_DEFAULT_QUANT_PARAM", "true");
    pushGlobalConf("UPDATE_GRAPH_OUTPUT_MME", "true");

    std::vector<float>       inputBuffer1 = {120};
    std::vector<float>       inputBuffer2 = {120};
    const std::vector<float> expected= {240};
    std::vector<unsigned>    inSizes1     = {1, 1};
    std::vector<unsigned>    inSizes2     = {1, 1};
    std::vector<unsigned>    outSizes     = {1, 1};

    // create graph
    setGraphInferenceModeAndQuantizationEnabled();

    unsigned T1 = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_FROM_INITIALIZER,
                                            inputBuffer1.data(),
                                            inSizes1.data(),
                                            inSizes1.size(),
                                            syn_type_float,
                                            nullptr,
                                            "Input1");

    unsigned T2 = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_FROM_INITIALIZER,
                                            inputBuffer2.data(),
                                            inSizes2.data(),
                                            inSizes2.size(),
                                            syn_type_float,
                                            nullptr,
                                            "Input2");

    unsigned T3 = createPersistTensor(OUTPUT_TENSOR,
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        outSizes.data(),
                                        outSizes.size(),
                                        syn_type_fp8_143,
                                        nullptr,
                                        "Output");

    addNodeToGraph("gemm", {T1, T2}, {T3});

    compileTopology();
    runTopology();

    QuantizationData quant_data;
    quant_data.setExpBias(7);
    std::vector<float> temp = {0};
    castFp8ToAnyFloat<fp8_143_t, float>(castHostBuffer<fp8_143_t>(T3), temp.data(), 1, quant_data);

    ASSERT_EQ(expected[0], float(temp[0]));
}