#include "synapse_api.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, complex_guid_reduce_sum_square)
{
    unsigned inSize  = 1234;
    unsigned outSize = 1;

    std::vector<float> inData(inSize, 0);
    for (auto i = 0; i < inData.size(); i++)
    {
        inData[i] = i % 10;  // to avoid overflow
    }
    auto inIdx = createPersistTensor(INPUT_TENSOR,
                                     MEM_INIT_FROM_INITIALIZER,
                                     inData.data(),
                                     &inSize,
                                     1,
                                     syn_type_single,
                                     nullptr,
                                     "input");
    auto outIdx =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, &outSize, 1, syn_type_single, nullptr, "output");

    auto rmwSectionIdx = createNonPersistentSection();

    auto memsetOutIdx = createTensors(1,
                                      OUTPUT_TENSOR,
                                      false, /*isPersistent*/
                                      "memset_output",
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      &outSize,
                                      1,
                                      syn_type_float,
                                      nullptr,
                                      0,
                                      0, /*offsetInSection*/
                                      &rmwSectionIdx /*sectionIndex*/)[0];

    auto reduceSumOutIdx = createTensors(1,
                                         OUTPUT_TENSOR,
                                         false, /*isPersistent*/
                                         "reduce_sum_output",
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         &outSize,
                                         1,
                                         syn_type_float,
                                         nullptr,
                                         0,
                                         0, /*offsetInSection*/
                                         &rmwSectionIdx /*sectionIndex*/)[0];

    // Create reduce-sum node
    synNodeId            reduceSumId;
    ns_Reduction::Params params = {0};
    addNodeToGraph("reduce_sum_square_fwd_f32",
                   {inIdx},
                   {reduceSumOutIdx},
                   &params,
                   sizeof(params),
                   "reducde_sum",
                   0,
                   &reduceSumId);

    // Create memset node
    synNodeId memsetId;
    addNodeToGraph("memset", {}, {memsetOutIdx}, nullptr, 0, "memset", 0, &memsetId);

    // Create memcopy node
    addNodeToGraph("memcpy", {reduceSumOutIdx}, {outIdx}, nullptr, 0, "memcpy");

    synNodeDependencySet(m_graphs[0].graphHandle, &memsetId, &reduceSumId, 1, 1);

    compileTopology();
    runTopology();

    float* pOutput = castHostBuffer<float>(outIdx);

    float expectedResult = 0;
    for (auto i = 0; i < inData.size(); i++)
    {
        expectedResult += (inData[i] * inData[i]);
    }

    ASSERT_TRUE(float_eq(expectedResult, pOutput[0]));
}

TEST_F_GC(SynTrainingTestInfra, complex_guid_and_mme_L2)
{
    unsigned              b = 1, inH = 16, inW = 16, c = 16, k = 16, kernel = 3;
    synConvolutionParams  params(kernel, kernel, 2, 2, 0, 0, 0, 0, 1, 1);
    unsigned              outH = convOutputDimSize(inH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned              outW = convOutputDimSize(inW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    std::vector<unsigned> xSizes {c, inW, inH, b};
    std::vector<unsigned> wSizes {k, c, kernel, kernel};
    std::vector<unsigned> ySizes {k, outW, outH, b};

    unsigned xIdx = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        xSizes.data(),
                                        xSizes.size(),
                                        syn_type_float);
    unsigned wIdx = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        wSizes.data(),
                                        wSizes.size(),
                                        syn_type_float);
    unsigned yIdx = SynTrainingTestInfra::createTensor(OUTPUT_TENSOR,
                                                       MEM_INIT_ALL_ZERO,
                                                       nullptr,
                                                       ySizes.data(),
                                                       ySizes.size(),
                                                       syn_type_float);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {xIdx, wIdx}, {yIdx}, &params, sizeof(params), "convolution");

    auto rmwSectionIdx = createNonPersistentSection();

    auto reluOutIdx = createTensors(1,
                                    OUTPUT_TENSOR,
                                    false, /*isPersistent*/
                                    "relu_output",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    ySizes.data(),
                                    ySizes.size(),
                                    syn_type_float,
                                    nullptr,
                                    0,
                                    0, /*offsetInSection*/
                                    &rmwSectionIdx /*sectionIndex*/)[0];

    addNodeToGraph("relu_fwd_f32", {yIdx}, {reluOutIdx}, nullptr, 0, "relu");

    unsigned finalOutIdx =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes.data(), ySizes.size(), syn_type_float);

    addNodeToGraph("memcpy", {reluOutIdx}, {finalOutIdx}, nullptr, 0, "memcpy");

    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[xIdx];
    synTensorDescriptor wDesc = m_tensorDescs[wIdx];
    synTensorDescriptor yDesc = m_tensorDescs[yIdx];

    auto xData   = m_hostBuffers[xIdx];
    auto wData   = m_hostBuffers[wIdx];
    auto outData = m_hostBuffers[finalOutIdx];

    uint64_t           outSizeInElements = getNumberOfElements(yDesc.m_sizes, yDesc.m_dims);
    std::vector<float> refConv(outSizeInElements, 0);
    std::vector<float> refConvRelu(outSizeInElements, 0);

    calculateFwdConvolution(xDesc,
                            (char*)xData,
                            wDesc,
                            (char*)wData,
                            yDesc,
                            (char*)refConv.data(),
                            params,
                            m_deviceType);
    calculateRelu(yDesc, refConv.data(), yDesc, refConvRelu.data());

    CoordArray wrongIdx = {0};
    bool       ret      = checkResults(yDesc, (char*) outData, (char*) refConvRelu.data(), wrongIdx);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, yDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, yDesc.m_dataType, outData);
}
