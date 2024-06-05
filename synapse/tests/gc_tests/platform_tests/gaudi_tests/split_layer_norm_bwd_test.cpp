#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, layerNormBwdTest_L2)
{
    float ifmValues[4] = {1.0, 2.0, 3.0, 4.0};
    float gradValues[4] = {0.1, 0.2, 0.3, 0.4};
    float meanValues[4] = {1.4, 2.3, 3.2, 4.1};
    float lstdValues[4] = {3.9, 4.8, 1.7, 2.6};
    float gammaValues[1] = {2.0};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned gammaDims[4] = {1, 1, 1, 1};

    unsigned ifmTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmValues, dataDims, 4, syn_type_float);
    unsigned gradInTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, gradValues, dataDims, 4, syn_type_float);
    unsigned meanTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, meanValues, dataDims, 4, syn_type_float);
    unsigned lstdTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, lstdValues, dataDims, 4, syn_type_float);
    unsigned gammaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, gammaValues, gammaDims, 1, syn_type_float);

    unsigned gradOutTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4, syn_type_float);
    unsigned gradBetaTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, gammaDims, 1, syn_type_float);
    unsigned gradGammaTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, gammaDims, 1, syn_type_float);

    ns_LayerNormKernel::Params params;
    params.eps = 0.1;
    params.epsValid = false;

    addNodeToGraph("layer_norm_bwd_f32",
                   {ifmTensor, gradInTensor, meanTensor, lstdTensor, gammaTensor},
                   {gradOutTensor, gradBetaTensor, gradGammaTensor}, &params, sizeof(ns_LayerNormKernel::Params));

    compileTopology();
    runTopology();

    float* pGradOutput = (float*)m_hostBuffers[gradOutTensor];
    float* pBetaOutput = (float*)m_hostBuffers[gradBetaTensor];
    float* pGammaOutput = (float*)m_hostBuffers[gradGammaTensor];

    float gradOutRef[4] = {-1.89821, -3.98131, -0.117912, -0.140608};
    float betaRef[1] = {1};
    float gammaRef[1] = {-0.65};

    validateResult(gradOutRef, pGradOutput, 4);
    validateResult(betaRef, pBetaOutput, 1);
    validateResult(gammaRef, pGammaOutput, 1);
}

TEST_F_GC(SynTrainingTestInfra, layerNormBwdSplitDisabledTest)
{
    pushGlobalConf("GCFG_SKIP_LAYER_NORM_BWD_SPLIT", "1");

    // Should use same test data as in SynGaudiTestInfra.layerNormBwdTest
    float ifmValues[4] = {1.0, 2.0, 3.0, 4.0};
    float gradValues[4] = {0.1, 0.2, 0.3, 0.4};
    float meanValues[4] = {1.4, 2.3, 3.2, 4.1};
    float lstdValues[4] = {3.9, 4.8, 1.7, 2.6};
    float gammaValues[1] = {2.0};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned gammaDims[4] = {1, 1, 1, 1};

    unsigned ifmTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmValues, dataDims, 4, syn_type_float);
    unsigned gradInTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, gradValues, dataDims, 4, syn_type_float);
    unsigned meanTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, meanValues, dataDims, 4, syn_type_float);
    unsigned lstdTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, lstdValues, dataDims, 4, syn_type_float);
    unsigned gammaTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, gammaValues, gammaDims, 1, syn_type_float);

    unsigned gradOutTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4, syn_type_float);
    unsigned gradBetaTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, gammaDims, 1, syn_type_float);
    unsigned gradGammaTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, gammaDims, 1, syn_type_float);

    ns_LayerNormKernel::Params params;
    params.eps = 0.1;
    params.epsValid = false;

    addNodeToGraph("layer_norm_bwd_f32",
                   {ifmTensor, gradInTensor, meanTensor, lstdTensor, gammaTensor},
                   {gradOutTensor, gradBetaTensor, gradGammaTensor}, &params, sizeof(ns_LayerNormKernel::Params));

    compileTopology();
    runTopology();

    float* pGradOutput = (float*)m_hostBuffers[gradOutTensor];
    float* pBetaOutput = (float*)m_hostBuffers[gradBetaTensor];
    float* pGammaOutput = (float*)m_hostBuffers[gradGammaTensor];

    float gradOutRef[4] = {-1.89821, -3.98131, -0.117912, -0.140608};
    float betaRef[1] = {1};
    float gammaRef[1] = {-0.65};

    validateResult(gradOutRef, pGradOutput, 4);
    validateResult(betaRef, pBetaOutput, 1);
    validateResult(gammaRef, pGammaOutput, 1);
}
