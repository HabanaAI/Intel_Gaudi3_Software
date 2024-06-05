#include <fstream>
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"


static void fromFile(std::string path, void* buf, unsigned size)
{
    std::ifstream file(path);
    file.read((char*)buf, size);
    file.close();
}

//static void toFile(std::string path, void* buf, unsigned size)
//{
//    std::ofstream file(path);
//    file.write((char*)buf, size);
//    file.close();
//}

static uint16_t floatToBF16(float in)
{
    uint32_t* val_32b = reinterpret_cast<uint32_t*>(&in);
    return *val_32b >> 16;
}

static void referenceImpl(const float* grad, const float* weightsIn, const float* momentumIn,
                          float lr, float mb, const ns_OptimizerSGD::Params* params, unsigned nElements,
                          float* weightsOut, uint16_t* bfWeightsOut, float* momentumOut)
{
    float oneMinusDamp = 1.f - params->damp;

    for (unsigned elem = 0; elem < nElements; ++elem)
    {
        float wgg = std::fma(params->wd, weightsIn[elem], grad[elem]);

        if (mb)
        {
            //First step
            momentumOut[elem] = wgg;
        }
        else
        {
            float weightedGrad = wgg * oneMinusDamp;
            momentumOut[elem] = std::fma(params->mom, momentumIn[elem], weightedGrad);
        }

        if (params->nesterov)
        {
            wgg = std::fma(params->mom, momentumOut[elem], wgg);
        }

        weightsOut[elem] = std::fma(wgg, -lr, weightsIn[elem]);
        if (bfWeightsOut != nullptr)
        {
            bfWeightsOut[elem] = floatToBF16(weightsOut[elem]);
        }
    }
}

TEST_F_GC(SynTrainingTpcTestInfra, weight_update_L2)
{
    const char* envSoftwareLfsData = std::getenv("SOFTWARE_LFS_DATA");
    ASSERT_TRUE(envSoftwareLfsData) << "SOFTWARE_LFS_DATA is not set!";
    std::string softwareLfsData = envSoftwareLfsData;
    std::string dataPath        = softwareLfsData + "/demos/gaudi/weights_checkpoint/";

    static const unsigned K = 1000;
    static const unsigned C = 2048;
    static const unsigned S = 1;
    static const unsigned R = 1;
    static const unsigned Q = 1;
    static const unsigned nElems = Q * R * S * C * K;
    static const unsigned nBytes = nElems * sizeof(float);
    static unsigned       tensorSizes[] = {K, C, R, S, Q};
    static unsigned       ones[] = {1, 1, 1, 1, 1};

    float* initializer = new float[nElems];

    //5 input tensors, 2/3 output tensors (BF16 output is optional)

    //Weight grad
    fromFile(dataPath + "fc_grad_weights.1", initializer, nBytes);
    auto in1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, initializer, tensorSizes);

    //F32 weights in
    fromFile(dataPath + "fc_golden_weights.0", initializer, nBytes);
    auto in2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, initializer, tensorSizes);

    //Momentum in. Note that this tensor should be ignored in this test
    fromFile(dataPath + "fc_weights_momentum.0", initializer, nBytes);
    auto in3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, initializer, tensorSizes);

    delete[] initializer;

    //Scalar inputs passed as tensors

    //First mini-batch indication. For this test "true", so ignores momentum values completely
    uint32_t firstMB = 1;
    auto in4 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, reinterpret_cast<float*>(&firstMB), ones, DEFAULT_SIZES, syn_type_int32);

    //The learning rate
    float lr = 0.1f;
    auto in5 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, &lr, ones, DEFAULT_SIZES);

    //Outputs: new weights F32, new weights BF16, momentum
    //Todo: seems like the address is ignored, can't alias
    auto out1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensorSizes);
    auto out2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensorSizes, DEFAULT_SIZES, syn_type_bf16);
    auto out3 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensorSizes);
    createGraph();

    ns_OptimizerSGD::Params params;
    params.nesterov = false;
    params.damp = 0.f;
    params.mom = 0.9f;
    params.wd = 1e-4f;

    addNodeToGraph("optimizer_sgd_bwd_f32", {in1, in2, in3, in4, in5}, {out1, out2, out3}, &params, sizeof(params));
    compileAndRun();

    float* pOutputGoldenWeights = (float*)m_hostBuffers[out1];
    float* pOutputMomentum      = (float*)m_hostBuffers[out3];

    float* refGoldenWeights     = new float[nElems];
    float* refMomentum          = new float[nElems];
    float* pGrad                = (float*)m_hostBuffers[in1];
    float* pInputGoldenWeights  = (float*)m_hostBuffers[in2];
    float* pInputMomentum       = (float*)m_hostBuffers[in3];

    referenceImpl(pGrad, pInputGoldenWeights, pInputMomentum, lr, firstMB, &params, nElems, refGoldenWeights, nullptr, refMomentum);

    for (unsigned elem = 0; elem < nElems; ++elem)
    {
        ASSERT_EQ(refGoldenWeights[elem], pOutputGoldenWeights[elem]) << "Golden weights mismatch at " << elem;
        ASSERT_EQ(refMomentum[elem], pOutputMomentum[elem])           << "Momentum mismatch at " << elem;
    }

    delete[] refGoldenWeights;
    delete[] refMomentum;
}
