#include "infra/cpu_calculator.h"
#include "utils.h"
#include "synapse_test.hpp"
#include "node_factory.h"
#include "gc_gaudi_test_infra.h"

class SynTrainingGraphDuplicateTest : public SynTrainingTestInfra
{
public:
    void fwdConvTest(int iterations, bool duplicateFromDuplicate);
};

void SynTrainingGraphDuplicateTest::fwdConvTest(int iterations, bool duplicateFromDuplicate)
{
    synConvolutionParams params;
    params.kH = 3;
    params.kW = 3;

    const unsigned batch = 1;
    const unsigned nIFM  = 3;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    const unsigned     ofmDataSize = nOFM * wOFM * hOFM * batch;
    std::vector<float> ofmRefBuffer(ofmDataSize);

    // create_tensor's layout
    constexpr unsigned             NUM_DIMS    = 4;
    std::array<unsigned, NUM_DIMS> ifmDimSizes = {nIFM, wIFM, hIFM, batch};
    std::array<unsigned, NUM_DIMS> wghDimSizes = {nOFM, nIFM, params.kW, params.kH};
    std::array<unsigned, NUM_DIMS> ofmDimSizes = {nOFM, wOFM, hOFM, batch};

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                ifmDimSizes.data(),
                                                NUM_DIMS,
                                                syn_type_single);
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                wghDimSizes.data(),
                                                NUM_DIMS,
                                                syn_type_single);
    unsigned yTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes.data(), NUM_DIMS, syn_type_single);

    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];

    calculateFwdConvolution(xDesc,
                            (char*)xData,
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)ofmRefBuffer.data(),
                            params,
                            m_deviceType);

    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params,
                   sizeof(synConvolutionParams));

    auto newGraphIndex = duplicateGraph();

    for (int i = 1; i < iterations; i++)
    {
        newGraphIndex = duplicateGraph();
    }

    if (duplicateFromDuplicate)
    {
        newGraphIndex = duplicateGraph();
    }

    // compile original graph
    compileAndRun();
    // compile duplicated graph
    compileAndRun(newGraphIndex);

    auto yDupTensorIndex = getDuplicateTensorIndex(newGraphIndex, yTensorIndex);
    auto elementsCount   = getNumberOfElements(ofmDimSizes.data(), ofmDimSizes.size());
    for (auto tensorIndex : {yTensorIndex, yDupTensorIndex})
    {
        auto pOutputBuffer = static_cast<float*>(m_hostBuffers[tensorIndex]);
        for (uint64_t i = 0; i < elementsCount; i++)
        {
            ASSERT_LE(abs(pOutputBuffer[i] - ofmRefBuffer[i]), 0.00001)
                << "Mismatch at index " << i << " Result:" << pOutputBuffer[i] << " Ref: " << ofmRefBuffer[i]
                << " tensorIndex: " << tensorIndex;
        }
    }
}

TEST_F_GC(SynTrainingGraphDuplicateTest, DISABLED_fwd_conv_test_simple)
{
    fwdConvTest(1, false);
}

TEST_F_GC(SynTrainingGraphDuplicateTest, DISABLED_fwd_conv_test_dup_of_dup)
{
    fwdConvTest(1, true);
}

TEST_F_GC(SynTrainingGraphDuplicateTest, DISABLED_fwd_conv_test_iters)
{
    fwdConvTest(10, false);
}

TEST_F_GC(SynTrainingGraphDuplicateTest, DISABLED_memcpy_then_neg)
{
    // create_tensor's layout
    constexpr unsigned             NUM_DIMS   = 5;
    std::array<unsigned, NUM_DIMS> fmDimSizes = {3, 4, 5, 6, 7};

    unsigned ifmTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_RANDOM_POSITIVE,
                                                  nullptr,
                                                  fmDimSizes.data(),
                                                  NUM_DIMS,
                                                  syn_type_single);
    unsigned ofmTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fmDimSizes.data(), NUM_DIMS, syn_type_single);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName);

    unsigned negOfmTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fmDimSizes.data(), NUM_DIMS, syn_type_single);
    addNodeToGraph("neg_f32", {ofmTensorIndex}, {negOfmTensorIndex});

    auto newGraphIndex = duplicateGraph();

    // compile original graph
    compileAndRun();
    // compile duplicated graph
    compileAndRun(newGraphIndex);

    auto elementsCount = getNumberOfElements(fmDimSizes.data(), fmDimSizes.size());
    auto pOrigBuffer   = static_cast<float*>(m_hostBuffers[ifmTensorIndex]);

    for (auto tensorIndexPair :
         {std::make_pair(ofmTensorIndex, 1), std::make_pair(ofmTensorIndex, 1), std::make_pair(negOfmTensorIndex, -1)})
    {
        auto pOutputBuffer = static_cast<float*>(m_hostBuffers[tensorIndexPair.first]);
        for (uint64_t i = 0; i < elementsCount; i++)
        {
            ASSERT_EQ(pOutputBuffer[i] * tensorIndexPair.second, pOrigBuffer[i])
                << "Mismatch at index " << i << " Result:" << pOutputBuffer[i] << " Ref: " << pOrigBuffer[i]
                << " tensorIndex: " << tensorIndexPair.first;
        }
    }
}