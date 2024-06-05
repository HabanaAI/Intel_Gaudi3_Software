#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

class SynGaudiTPCFuserTestMultiSif : public SynGaudiTestInfra
{
protected:
    virtual void SetUpTest() override
    {
        SynGaudiTestInfra::SetUpTest();
        prev_CFG_RUN_TPC_FUSER     = GCFG_RUN_TPC_FUSER.value();
        prev_ENABLE_INTERNAL_NODES = GCFG_ENABLE_INTERNAL_NODES.value();
        GCFG_RUN_TPC_FUSER.setValue(true);
        GCFG_ENABLE_INTERNAL_NODES.setValue(true);
    }

    virtual void TearDownTest() override
    {
        GCFG_RUN_TPC_FUSER.setValue(prev_CFG_RUN_TPC_FUSER);
        GCFG_ENABLE_INTERNAL_NODES.setValue(prev_ENABLE_INTERNAL_NODES);
        SynGaudiTestInfra::TearDownTest();
    };

public:
    SynGaudiTPCFuserTestMultiSif()
    {
        setTestPackage(TEST_PACKAGE_DSD);
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2});
    }

    unsigned makeTensor(TensorUsage usage,
                        bool        persist,
                        MemInitType initSelect,
                        unsigned    dims,
                        unsigned*   sizes,
                        unsigned*   minSizes,
                        float*      initializer = nullptr)
    {
        return createTensors(1,
                             usage,
                             persist,
                             nullptr,
                             initSelect,
                             initializer,
                             sizes,
                             dims,
                             syn_type_single,
                             nullptr,
                             0,
                             0,
                             nullptr,
                             false,
                             minSizes)[0];
    }

private:
    bool prev_CFG_RUN_TPC_FUSER;
    bool prev_ENABLE_INTERNAL_NODES;
};

TEST_F_GC(SynGaudiTPCFuserTestMultiSif, multisif1)
{
    unsigned maxDims[]    = {256, 64};
    unsigned minDims[]    = {1, 1};
    unsigned actualDims[] = {2, 2};

    unsigned in1  = makeTensor(INPUT_TENSOR, true, MEM_INIT_RANDOM_WITH_NEGATIVE, 2, maxDims, minDims);
    unsigned in2  = makeTensor(INPUT_TENSOR, true, MEM_INIT_RANDOM_WITH_NEGATIVE, 2, maxDims, minDims);
    unsigned in3  = makeTensor(INPUT_TENSOR, true, MEM_INIT_RANDOM_WITH_NEGATIVE, 2, maxDims, minDims);
    unsigned out1 = makeTensor(OUTPUT_TENSOR, false, MEM_INIT_ALL_ZERO, 2, maxDims, minDims);
    unsigned out2 = makeTensor(OUTPUT_TENSOR, true, MEM_INIT_ALL_ZERO, 2, maxDims, minDims);

    addNodeToGraph("add_fwd_f32", {in1, in2}, {out1});
    addNodeToGraph("add_fwd_f32", {in3, out1}, {out2});

    setActualSizes(in1, actualDims);
    setActualSizes(in2, actualDims);
    setActualSizes(in3, actualDims);
    setActualSizes(out1, actualDims);
    setActualSizes(out2, actualDims);

    // TODO verify that fusion has coourred
    // TODO verify that the fused node has multisif data set
    compileAndRun();

    float* input1 = (float*)m_hostBuffers[in1];
    float* input2 = (float*)m_hostBuffers[in2];
    float* input3 = (float*)m_hostBuffers[in3];

    float* output = (float*)m_hostBuffers[out2];

    auto numberOfElements = actualDims[0] * actualDims[1];

    for (unsigned idx = 0; idx < numberOfElements; idx++)
    {
        float expected_out = (*input1 + *input2) + *input3;

        ASSERT_EQ(expected_out, *output) << "OUTPUT3: Mismatch for at index " << idx << " |Expected:" << expected_out
                                         << " |Result: " << *output << " |Operands: " << *input1 << ", " << *input2
                                         << ", " << *input3;

        input1++;
        input2++;
        input3++;

        output++;
    }
}
