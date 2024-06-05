#include "gc_dynamic_shapes_infra.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "synapse_common_types.h"
class SynGaudiSimpleDynamicGemm : public SynGaudiDynamicShapesTestsInfra
{
    public:
        SynGaudiSimpleDynamicGemm()
        {
            setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
        }

        virtual void SetUpTest() override
        {
            SynGaudiTestInfra::SetUpTest();
            m_prevEnableSerializeDeserializePass = GCFG_GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS.value();
            GCFG_GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS.setValue(false);
        }

        virtual void TearDownTest() override
        {
            GCFG_GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS.setValue(m_prevEnableSerializeDeserializePass);
            SynGaudiTestInfra::TearDownTest();
        };

    private:

        bool m_prevEnableSerializeDeserializePass;
};

TEST_F_GC(SynGaudiSimpleDynamicGemm, simple_small_gemm)
{
    unsigned xMaxSizes[] = {7,6};
    unsigned xMinSizes[] = {3,2};
    (void)xMinSizes;
    unsigned xActSizes[] = {5,4};

    unsigned yMaxSizes[] = {8,7};
    unsigned yMinSizes[] = {4,3};
    (void)yMinSizes;
    unsigned yActSizes[] = {6,5};

    unsigned outMaxSizes[] = {8,6};
    unsigned outMinSizes[] = {4,2};
    (void)outMinSizes;
    unsigned outActSizes[] = {6,4};

    float xValues[] = {   1,   2,   3,   4,   5,      0, 0,
                          2,   3,   5,   7,  11,      0, 0,
                          1,  -1,   0,   2,  -2,      0, 0,
                          2,   3,  -4,   5,  -6,      0, 0,

                          0,   0,   0,   0,   0,      0, 0,
                          0,   0,   0,   0,   0,      0, 0    };

    float yValues[] = {   2,   1,   0,  -1,  -2,  -3,   0, 9,
                          0,  -1,   2,  -3,   4,  -5,   0, 0,
                         -7,   0,  -6,   0,  -5,   0,   0, 0,
                          1,   1,   1,  -1,  -1,  -1,   0, 0,
                          0,   1,   2,   2,   1,   0,   0, 0,

                          0,   0,   0,   0,   0,   0,   0, 0,
                          0,   0,   0,   0,   0,   0,   0, 0  };

    unsigned xTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, xValues,
                                            xMaxSizes, 2, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, xMinSizes);

    unsigned yTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, yValues,
                                            yMaxSizes, 2, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, yMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            outMaxSizes, 2, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, outMinSizes);

    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {xTensor, yTensor},
                   {outTensor},
                   nullptr, 0);

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(xTensor, xActSizes);
    (void)xActSizes;
    setActualSizes(yTensor, yActSizes);
    (void)yActSizes;
    setActualSizes(outTensor, outActSizes);
    (void)outActSizes;

    auto* xData = castHostInBuffer<float>(xTensor);
    auto* yData = castHostInBuffer<float>(yTensor);
    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* outData = castHostOutBuffer<float>(outTensor);

    // check it manually
    //
    for (int i = 0; i < outActSizes[1]; ++i)
    {
        for (int j = 0; j < outActSizes[0]; ++j)
        {
            float sum = 0;
            for (int k = 0; k < yActSizes[1]; ++k)
            {
                sum += xData[i * xMaxSizes[0] + k] * yData[k * yMaxSizes[0] + j];
            }
            // check for exact equality because our data are nice and small integers
            ASSERT_EQ(sum, outData[i * outMaxSizes[0] + j]) << "Failed at index " << i << ", " << j;

        }
    }
    //
    (void)xData; (void)yData; (void)outData;
}


TEST_F_GC(SynGaudiSimpleDynamicGemm, simple_big_gemm, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{

    const unsigned Mmax = 900, Mmin = 90, Mact = 450;
    const unsigned Nmax = 800, Nmin = 80, Nact = 400;
    const unsigned Kmax = 250, Kmin = 25, Kact = 125;

    unsigned xMaxSizes[] = {Kmax, Nmax};
    unsigned xMinSizes[] = {Kmin, Nmin};
    unsigned xActSizes[] = {Kact, Nact};

    unsigned yMaxSizes[] = {Mmax, Kmax};
    unsigned yMinSizes[] = {Mmin, Kmin};
    unsigned yActSizes[] = {Mact, Kact};


    unsigned outMaxSizes[] = {Mmax, Nmax};
    unsigned outMinSizes[] = {Mmin, Nmin};
    unsigned outActSizes[] = {Mact, Nact};

    // we may or may not use these
    (void)xMinSizes;
    (void)yMinSizes;
    (void)outMinSizes;
    (void)xActSizes;
    (void)yActSizes;
    (void)outActSizes;

    unsigned xTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            xMaxSizes, 2, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, xMinSizes);

    unsigned yTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            yMaxSizes, 2, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, yMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                            outMaxSizes, 2, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, outMinSizes);

    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {xTensor, yTensor},
                   {outTensor},
                   nullptr, 0);

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(xTensor, xActSizes);
    setActualSizes(yTensor, yActSizes);
    setActualSizes(outTensor, outActSizes);


    auto* xData = castHostInBuffer<float>(xTensor);
    auto* yData = castHostInBuffer<float>(yTensor);
    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* outData = castHostOutBuffer<float>(outTensor);

    // check it manually
    //
    for (int i = 0; i < Nact; ++i)
    {
        for (int j = 0; j < Mact; ++j)
        {
            float sum = 0;
            for (int k = 0; k < Kact; ++k)
            {
                sum += xData[i * Kmax + k] * yData[k * Mmax + j];
            }

            float eps = std::max(fabs(sum)/1000, 1e-3);
            // check for exact equality because our data are nice and small integers
            ASSERT_NEAR(sum, outData[i * Mmax + j], eps) << "Failed at index " << i << "," << j;
        }
    }
    //
    (void)xData; (void)yData; (void)outData;
}
