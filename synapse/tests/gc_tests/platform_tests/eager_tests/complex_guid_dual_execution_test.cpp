#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
#include "habana_global_conf.h"

class SynTrainingComplexGuidDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    SynTrainingComplexGuidDualExecutionTest() { ReleaseDevice(); }
};

TEST_F_GC(SynTrainingComplexGuidDualExecutionTest, eager_norm_moments_clustering)

{
    const unsigned c    = 128;
    const unsigned w    = 8;
    const unsigned h    = 4;
    const unsigned b    = 2;
    const unsigned wOut = 1;
    const unsigned hOut = 1;
    // create_tensor's layout
    std::array<unsigned, SYN_MAX_TENSOR_DIM> inSizes  = {c, w, h, b, 1};
    std::array<unsigned, SYN_MAX_TENSOR_DIM> outSizes = {c, wOut, hOut, b, 1};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                         nullptr,
                         inSizes.data(),
                         inSizes.size(),
                         syn_type_single);
    auto meanTensorIndexPair     = createPersistTensors(OUTPUT_TENSOR,
                                                    MEM_INIT_ALL_ZERO,
                                                    nullptr,
                                                    outSizes.data(),
                                                    outSizes.size(),
                                                    syn_type_single);
    auto varianceTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                        MEM_INIT_ALL_ZERO,
                                                        nullptr,
                                                        outSizes.data(),
                                                        outSizes.size(),
                                                        syn_type_single);

    ns_NormMomentsKernel::Params params;
    params.NormAxisBmp = 6;

    addNodesToGraphs("norm_moments_fwd_f32", (void*)&params, sizeof(params));

    compileAndRun();

    std::array<std::pair<TensorIndexPair, std::string>, 2> resultPairs = {
        make_pair(meanTensorIndexPair, std::string("mean tensor")),
        make_pair(varianceTensorIndexPair, std::string("variance tensor"))};

    for (const auto& resultPair : resultPairs)
    {
        auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[resultPair.first.graph]);
        auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[resultPair.first.eager]);
        for (uint64_t i = 0; i < getNumberOfElements(outSizes.data(), outSizes.size()); i++)
        {
            ASSERT_LE(abs(pOutputBufferGraphMode[i] - pOutputBufferEagerMode[i]), 0.00001)
                << resultPair.second << " Graph mode mismatch at index " << i
                << " Graph mode:" << pOutputBufferGraphMode[i] << " Eager mode: " << pOutputBufferEagerMode[i];
        }
    }
}
