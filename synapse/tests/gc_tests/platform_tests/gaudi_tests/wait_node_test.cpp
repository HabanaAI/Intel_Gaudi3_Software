#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

class SynTrainingTestWait : public SynTrainingTestInfra
{
public:
    void SetUpTest() override
    {
        SynTrainingTestInfra::SetUpTest();
        GCFG_ENABLE_INTERNAL_NODES.setValue(true);
    }

    void linearDmaNode()
    {
        auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
        auto out = createPersistTensor(OUTPUT_TENSOR);

        synNodeId memcpyNodeId;
        ASSERT_EQ(synSuccess,synNodeCreateWithId(m_graphs[0].graphHandle,
                                                 &m_tensors[in],
                                                 &m_tensors[out],
                                                 1,
                                                 1,
                                                 nullptr,
                                                 0,
                                                 "memcpy",
                                                 "my_memcpy",
                                                 &memcpyNodeId,
                                                 nullptr,
                                                 nullptr)) << "Failed to create node with GUID " << "memcpy";


        struct synWaitParams waitParams(42);

        synNodeId waitNodeId;
        ASSERT_EQ(synSuccess, synNodeCreateWithId(m_graphs[0].graphHandle,
                                                  {},
                                                  {},
                                                  0,
                                                  0,
                                                  (void *)(&waitParams),
                                                  sizeof(waitParams),
                                                  "wait",
                                                  "my_wait",
                                                  &waitNodeId,
                                                  nullptr,
                                                  nullptr)) << "Failed to create node with GUID " << "wait";

        ASSERT_EQ(synSuccess, synNodeDependencySet(m_graphs[0].graphHandle,
                                                   {&waitNodeId},
                                                   {&memcpyNodeId},
                                                   1,
                                                   1)) << "Failed to create control dependency";
        compileTopology();
        runTopology();

        float* pDmaInput  = (float*)m_hostBuffers[in];
        float* pDmaOutput = (float*)m_hostBuffers[out];

        for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
        {
            ASSERT_EQ(*pDmaInput, *pDmaOutput) << "Mismatch at index " << i
                                               << " Expected:"         << *pDmaInput
                                               << " Result: "          << *pDmaOutput;
            pDmaInput++;
            pDmaOutput++;
        }
    }

    void gemmNode()
    {
        std::array<unsigned,2> aSize = {128,256};
        std::array<unsigned,2> bSize = {512,128};
        bool transposeA = false;
        bool transposeB = false;


        const unsigned dims = 2;
        unsigned aDimSizes[]   = {aSize[0], aSize[1], 1, 1};
        unsigned bDimSizes[]   = {bSize[0], bSize[1], 1, 1};
        unsigned outDimSizes[] = {transposeB ? bSize[1] : bSize[0],
                                  transposeA ? aSize[0] : aSize[1],
                                  1, 1};

        unsigned aTensorIndex   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, aDimSizes, dims,
                                                      syn_type_single);
        unsigned bTensorIndex   = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, bDimSizes, dims,
                                                      syn_type_single);
        unsigned outTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outDimSizes, dims,
                                                      syn_type_single);

        TensorIndices inputIndices  = {aTensorIndex, bTensorIndex};
        TensorIndices outputIndices = {outTensorIndex};

        synGEMMParams params;
        params.transpose_a = transposeA;
        params.transpose_b = transposeB;

        synNodeId gemmNodeId;
        ASSERT_EQ(synSuccess, synNodeCreateWithId(m_graphs[0].graphHandle,
                                                  &m_tensors[aTensorIndex],
                                                  &m_tensors[outTensorIndex],
                                                  2,
                                                  1,
                                                  nullptr,
                                                  0,
                                                  "gemm",
                                                  "my_gemm",
                                                  &gemmNodeId,
                                                  nullptr,
                                                  nullptr)) << "Failed to create node with GUID " << "gemm";


        struct synWaitParams waitParams(42);

        synNodeId waitNodeId;
        ASSERT_EQ(synSuccess, synNodeCreateWithId(m_graphs[0].graphHandle,
                                                  {},
                                                  {},
                                                  0,
                                                  0,
                                                  (void *)(&waitParams),
                                                  sizeof(waitParams),
                                                  "wait",
                                                  "my_wait",
                                                  &waitNodeId,
                                                  nullptr,
                                                  nullptr)) << "Failed to create node with GUID " << "wait";

        ASSERT_EQ(synSuccess, synNodeDependencySet(m_graphs[0].graphHandle,
                                                   {&waitNodeId},
                                                   {&gemmNodeId},
                                                   1,
                                                   1)) << "Failed to create control dependency";

        compileAndRun();

        float* matAVal = (float*)m_hostBuffers[aTensorIndex];
        float* matBVal = (float*)m_hostBuffers[bTensorIndex];

        uint64_t matOutSize = getNumberOfElements(outDimSizes);
        float* outRef = new float[matOutSize];

        synTensorDescriptor aDesc   = m_tensorDescs[aTensorIndex];
        synTensorDescriptor bDesc   = m_tensorDescs[bTensorIndex];
        synTensorDescriptor outDesc = m_tensorDescs[outTensorIndex];

        ERepefenceOp op;
        if (transposeA)
        {
            if (transposeB)
            {
                op = REFERENCE_OP_ATBT;
            }
            else
            {
                op = REFERENCE_OP_ATB;
            }
        }
        else
        {
            if (transposeB)
            {
                op = REFERENCE_OP_ABT;
            }
            else
            {
                op = REFERENCE_OP_AB;
            }
        }

        calculateGemm(aDesc, (char*)matAVal, bDesc, (char*)matBVal, outDesc, (char*)outRef, params, op, m_deviceType);

        float* pOutputBuffer = (float*)m_hostBuffers[outTensorIndex];
        for (uint64_t i = 0; i < matOutSize; i++)
        {
            ASSERT_TRUE(float_eq(pOutputBuffer[i], outRef[i])) << "Incorrect matrix value at " << i << " is "
                                                               << pOutputBuffer[i] << " expected " << outRef[i];
        }

        delete[] outRef;
    }

    void negateNode()
    {
        auto in = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
        auto out = createPersistTensor(OUTPUT_TENSOR);

        synNodeId negNodeId;
        ASSERT_EQ(synSuccess,synNodeCreateWithId(m_graphs[0].graphHandle,
                                                 &m_tensors[in],
                                                 &m_tensors[out],
                                                 1,
                                                 1,
                                                 nullptr,
                                                 0,
                                                 "neg_fwd_f32",
                                                 "my_neg",
                                                 &negNodeId,
                                                 nullptr,
                                                 nullptr)) << "Failed to create node with GUID " << "neg_fwd_f32";


        struct synWaitParams waitParams(42);

        synNodeId waitNodeId;
        ASSERT_EQ(synSuccess, synNodeCreateWithId(m_graphs[0].graphHandle,
                                                  {},
                                                  {},
                                                  0,
                                                  0,
                                                  (void *)(&waitParams),
                                                  sizeof(waitParams),
                                                  "wait",
                                                  "my_wait",
                                                  &waitNodeId,
                                                  nullptr,
                                                  nullptr)) << "Failed to create node with GUID " << "wait";

        ASSERT_EQ(synSuccess, synNodeDependencySet(m_graphs[0].graphHandle,
                                                   {&waitNodeId},
                                                   {&negNodeId},
                                                   1,
                                                   1)) << "Failed to create control dependency";
        compileTopology();
        runTopology();

        float* pTpcInput  = (float*)m_hostBuffers[in];
        float* pTpcOutput = (float*)m_hostBuffers[out];

        for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
        {
            ASSERT_EQ(*pTpcInput, -(*pTpcOutput)) << "Mismatch at index " << i
                                                  << " Expected:"         << *pTpcInput
                                                  << " Result: "          << *pTpcOutput;
            pTpcInput++;
            pTpcOutput++;
        }
    }
}; // class SynGaudiTestWait

// This uses dma
TEST_F_GC(SynTrainingTestWait, wait_before_linear_dma)
{
    linearDmaNode();
}

// This uses mme
TEST_F_GC(SynTrainingTestWait, wait_before_gemm)
{
    gemmNode();
}

// This uses tpc
TEST_F_GC(SynTrainingTestWait, wait_before_negate)
{
    negateNode();
}
