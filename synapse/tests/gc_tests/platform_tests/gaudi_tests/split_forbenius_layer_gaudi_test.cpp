#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "log_manager.h"
#include "tensor.h"

namespace
{
class SynTrainingFNTest : public SynTrainingTestInfra
{
public:
    template<typename DATA_TYPE>
    void frobeniusNormFwdTest();

private:
    template<typename DATA_TYPE>
    void validateResults(const DATA_TYPE*         inputFm,
                         std::array<unsigned, 4>& ifmSizes,
                         const DATA_TYPE*         outputRes,
                         std::array<unsigned, 2>& outResSizes);

    template<typename T, typename T_res>
    void macOp(T* op1, T* op2, T_res* op3, T_res* res1);

    template<typename T>
    float getElement(int b, int w, int h, int d, T* data, unsigned arrSizes[]);

    template<typename T>
    void lpNormFroStage1Ref(const T*                       ifmTnsr,  // inputs
                            const std::array<unsigned, 4>& ifmSizes,
                            T* const                       sumPartial,  // outputs
                            const int                      chunkSize);                       // scalar param

    template<typename T>
    void lpNormFroStage2Ref(const T* sumPartial, const std::array<unsigned, 2>& sumPartialSize, float& frobNormRef);
};
}  // namespace

template<typename T, typename T_res>
void SynTrainingFNTest::macOp(T* op1, T* op2, T_res* op3, T_res* res1)
{
    *res1 = (T_res)(*op3) + (T_res)(*op1) * (T_res)(*op2);
}

// stage 1 reference implementation
template<typename T>
void SynTrainingFNTest::lpNormFroStage1Ref(const T*                       ifmTnsr,  // inputs
                                           const std::array<unsigned, 4>& ifmSizes,
                                           T* const                       sumPartial,  // outputs
                                           const int                      chunkSize)                        // scalar param
{
    const int channelSize = ifmSizes[0];
    const int widthSize   = ifmSizes[1];
    const int heightSize  = ifmSizes[2];
    const int batchSize   = ifmSizes[3];
    const int numChunks   = div_round_up(widthSize, chunkSize);

    // iterate over number of chunks
    for (int i = 0; i < numChunks; i++)
    {
        float sum   = 0;
        int   start = i * chunkSize;
        int   end   = start + chunkSize;
        if (end > widthSize) end = widthSize;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < heightSize; h++)
            {
                for (int w = start; w < end; w++)
                {
                    for (int d = 0; d < channelSize; d += 1)
                    {
                        unsigned offset = b * (widthSize * heightSize * channelSize) + d * (widthSize * heightSize) +
                                          w * (heightSize) + h;
                        float aj = (float) (*(ifmTnsr + offset));
                        macOp<float, float>(&aj, &aj, &sum, &sum);
                    }
                }
            }
        }
        // update partial sum
        *(sumPartial + i) = sum;
    }
}

// stage 2 reference implementation
template<typename T>
void SynTrainingFNTest::lpNormFroStage2Ref(const T*                       sumPartial,
                                           const std::array<unsigned, 2>& sumPartialSize,
                                           float&                         frobNormRef)
{
    frobNormRef           = 0.0;
    const int channelSize = sumPartialSize[0];
    const int widthSize   = sumPartialSize[1];

    for (int w = 0; w < widthSize; w++)
    {
        for (int d = 0; d < channelSize; d++)
        {
            unsigned offset = d * widthSize + w;
            float    aj     = (float) (*(sumPartial + offset));
            frobNormRef += aj;
        }
    }
    frobNormRef = powf((float) frobNormRef, 0.5);
}

template<typename DATA_TYPE>
void SynTrainingFNTest::validateResults(const DATA_TYPE*         inputFm,
                                        std::array<unsigned, 4>& ifmSizes,
                                        const DATA_TYPE*         outputRes,
                                        std::array<unsigned, 2>& outResSizes)
{
    int chunkSize = std::max(floor(ifmSizes[1] / 8), 4.0);
    int numChunks = std::ceil(static_cast<float>(ifmSizes[1]) / static_cast<float>(chunkSize));

    std::array<unsigned, 2> partialSumSizes = {1, numChunks};
    unsigned                intermediateSum = createTensors(1,
                                             OUTPUT_TENSOR,
                                             inputFm,
                                             "partial_sum_ref",
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             partialSumSizes.data(),
                                             2,
                                             syn_type_float)[0];

    auto interSum = reinterpret_cast<DATA_TYPE*>(m_hostBuffers[intermediateSum]);

    lpNormFroStage1Ref(inputFm, ifmSizes, interSum, chunkSize);

    float frobNormRef;
    lpNormFroStage2Ref(interSum, partialSumSizes, frobNormRef);
    float              result     = *outputRes;
    static const float maxAbsDiff = 9e-04;

    ASSERT_LE(std::abs((DATA_TYPE)frobNormRef - (DATA_TYPE)result), (DATA_TYPE)maxAbsDiff)
        << " actual results not in the reference range";
}

template<typename DATA_TYPE>
void SynTrainingFNTest::frobeniusNormFwdTest()
{
    std::array<unsigned, 4> fMsizes     = {64, 8, 4, 1};
    std::array<unsigned, 2> oneDimSizes = {1};

    synDataType synDType = std::is_same<DATA_TYPE, float>::value ? syn_type_float : syn_type_bf16;

    unsigned featureMapIn = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                fMsizes.data(),
                                                4,
                                                synDType,
                                                nullptr,
                                                "featureMapIn");

    unsigned scalarOut = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             oneDimSizes.data(),
                                             1,
                                             synDType,
                                             nullptr,
                                             "scalarOut");

    addNodeToGraph("frobenius_norm_fwd", {featureMapIn}, {scalarOut}, nullptr);

    compileTopology();
    runTopology();

    DATA_TYPE* pFmInput  = (DATA_TYPE*) m_hostBuffers[featureMapIn];
    DATA_TYPE* pFmOutput = (DATA_TYPE*) m_hostBuffers[scalarOut];
    validateResults(pFmInput, fMsizes, pFmOutput, oneDimSizes);
}

TEST_F_GC(SynTrainingFNTest, frobenius_norm_fwd_float_test)
{
    frobeniusNormFwdTest<float>();
}

TEST_F_GC(SynTrainingFNTest, frobenius_norm_fwd_bf16_test)
{
    frobeniusNormFwdTest<bfloat16>();
}
