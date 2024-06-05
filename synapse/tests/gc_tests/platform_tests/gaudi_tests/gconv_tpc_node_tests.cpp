#include "compilation_hal_reader.h"
#include "data_type_utils.h"
#include "gc_gaudi_test_infra.h"
#include "gtest/gtest.h"
#include <data_types/fp8.h>
#include <type_traits>
#include "infra/test_device_manager.h"
#include "node_factory.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "synapse_common_types.h"
#include "types_exception.h"

template<typename DType>
static void fillWeightsWithZeroes(std::vector<DType> weights)
{
    if constexpr (std::is_same_v<DType, fp8_152_t> || std::is_same_v<DType, fp8_143_t>)
    {
        std::fill(weights.begin(), weights.end(), static_cast<DType>(0));
    }
    else
    {
        std::fill(weights.begin(), weights.end(), 0);
    }
}

static void addDelim(unsigned groupsPerVector, unsigned kPerGroup, unsigned nPackedGroups, unsigned i)
    __attribute__((unused));

///// Printing functions for debug /////
template<typename DType>
static void printExpectedVsActualTensors(unsigned inputSizeInElems,
                                         unsigned outputSizeInElems,
                                         unsigned groupsPerVector,
                                         unsigned kPerGroup,
                                         unsigned nPackedGroups,
                                         float*   tpcInputData,
                                         DType*   refPaddedWgh,
                                         DType*   tpcOutput) __attribute__((unused));

static void addDelim(unsigned groupsPerVector, unsigned kPerGroup, unsigned nPackedGroups, unsigned i)
{
    if (i % kPerGroup == 0 && i != 0)
    {
        std::cout << " | ";
    }
    if (i % (kPerGroup * groupsPerVector) == 0 && i != 0)
    {
        std::cout << "| ";
    }
    if (i % (kPerGroup * groupsPerVector * nPackedGroups) == 0)
    {
        std::cout << std::endl;
    }
}

template<typename DType>
static void printExpectedVsActualTensors(unsigned inputSizeInElems,
                                         unsigned outputSizeInElems,
                                         unsigned groupsPerVector,
                                         unsigned kPerGroup,
                                         unsigned nPackedGroups,
                                         float*   tpcInputData,
                                         DType*   refPaddedWgh,
                                         DType*   tpcOutput)
{
    std::cout << "INPUT BUFFER:" << std::endl;
    for (int i = 0; i < inputSizeInElems; i++)
    {
        float expected = tpcInputData[i];
        addDelim(groupsPerVector, kPerGroup, nPackedGroups, i);
        std::cout << std::setw(4) << std::setfill(' ') << expected << " ";
    }
    std::cout << "\n\nEXPECTED OUT BUFFER:" << std::endl;
    for (int i = 0; i < outputSizeInElems; i++)
    {
        float expected = float(refPaddedWgh[i]);
        addDelim(groupsPerVector, kPerGroup, nPackedGroups, i);
        std::cout << std::setw(4) << std::setfill(' ') << expected << " ";
    }
    std::cout << "\n\nACTUAL OUT BUFFER:" << std::endl;
    for (int i = 0; i < outputSizeInElems; i++)
    {
        float actual = float(tpcOutput[i]);
        addDelim(groupsPerVector, kPerGroup, nPackedGroups, i);
        std::cout << std::setw(4) << std::setfill(' ') << actual << " ";
    }
    std::cout << std::endl;
}
/////

// pad multiple packed groups
/**
 * @param pWeightData - actual data of the weights WITH PADDING, its type is DType = {float, bfloat16}
 * @param R  - spatial dim R of the weights
 * @param S  - spatial dim S of the weights
 * @param C  - dim C of the padded weight tensor of a single packed group
 * @param K  - dim K of the padded weight tensor of a single packed group
 * @param groupsPerVector - number of groups that are in 1 diagonal (t)
 * @param nPackedGroups - number of diagonals
 * @param kStride - the full K dimension size in elements
 * */
template <typename DType>
static std::vector<DType> UnpadWeightTensor(DType* pWeightData, unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned groupsPerVector, unsigned nPackedGroups, unsigned kStride)
{
    if (K % groupsPerVector != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of output channels");
        throw IllegalGroupParams();
    }
    if (C % groupsPerVector != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of input channels");
        throw IllegalGroupParams();
    }
    unsigned kPerGroup = K / groupsPerVector;
    unsigned cPerGroup = C / groupsPerVector;
    unsigned weightsSpatialSize =  R * S;

    unsigned newWeightsSize = K * C * weightsSpatialSize * nPackedGroups;
    std::vector<DType> newWeights(newWeightsSize);
    fillWeightsWithZeroes(newWeights);

    for (unsigned packedGroupIdx = 0; packedGroupIdx < nPackedGroups; packedGroupIdx++)
    {
        unsigned kOffset = packedGroupIdx * groupsPerVector * kPerGroup;
        for (unsigned g = 0; g < groupsPerVector; ++g)
        {
            for (unsigned spatialIdx = 0; spatialIdx < weightsSpatialSize; ++spatialIdx)
            {
                for (unsigned c = g * cPerGroup; c < (g + 1) * cPerGroup; ++c)
                {
                    DType* dst = newWeights.data() +
                                spatialIdx * cPerGroup * kStride +
                                (c % cPerGroup) * kStride +
                                kOffset +
                                g * kPerGroup;
                    DType* src = pWeightData +
                                spatialIdx * C * kStride +
                                c * kStride +
                                kOffset +
                                g * kPerGroup;
                    memcpy(dst, src, kPerGroup * sizeof(DType));
                }
            }
        }
    }
    return newWeights;
}

//pad multiple packed groups
template <typename DType>
static std::vector<DType> PadWeightTensor(DType* pWeightData, unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned groupsPerVector, unsigned nPackedGroups, unsigned kStride)
{
    if (K % groupsPerVector != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of output channels");
        throw IllegalGroupParams();
    }
    if (C % groupsPerVector != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of input channels");
        throw IllegalGroupParams();
    }
    unsigned kPerGroup = K / groupsPerVector;
    unsigned cPerGroup = C / groupsPerVector;
    unsigned weightsSpatialSize =  R * S;

    unsigned newWeightsSize = K * C * weightsSpatialSize * nPackedGroups;
    std::vector<DType> newWeights(newWeightsSize);
    fillWeightsWithZeroes(newWeights);

    for (unsigned packedGroupIdx = 0; packedGroupIdx < nPackedGroups; packedGroupIdx++)
    {
        unsigned kOffset = packedGroupIdx * groupsPerVector * kPerGroup;
        for (unsigned g = 0; g < groupsPerVector; ++g)
        {
            for (unsigned spatialIdx = 0; spatialIdx < weightsSpatialSize; ++spatialIdx)
            {
                for (unsigned c = g * cPerGroup; c < (g + 1) * cPerGroup; ++c)
                {
                    DType* src = pWeightData +
                                spatialIdx * cPerGroup * kStride +
                                (c % cPerGroup) * kStride +
                                kOffset +
                                g * kPerGroup;
                    DType* dst = newWeights.data() +
                                spatialIdx * C * kStride +
                                c * kStride +
                                kOffset +
                                g * kPerGroup;
                    memcpy(dst, src, kPerGroup * sizeof(DType));
                }
            }
        }
    }
    return newWeights;
}

//pad multiple packed groups
template <typename DType>
static std::vector<DType> CreatePaddedWeightTensor(unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned groupsPerVector, unsigned nPackedGroups, unsigned kStride)
{
    if (K % groupsPerVector != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of output channels");
        throw IllegalGroupParams();
    }
    if (C % groupsPerVector != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of input channels");
        throw IllegalGroupParams();
    }
    unsigned kPerGroup = K / groupsPerVector;
    unsigned cPerGroup = C / groupsPerVector;
    unsigned weightsSpatialSize =  R * S;

    unsigned newWeightsSize = K * C * weightsSpatialSize * nPackedGroups;
    std::vector<DType> newWeights(newWeightsSize);
    std::vector<DType> src(kPerGroup);
    fillWithRandom(src.data(), kPerGroup, {0,4});
    fillWithRandom(newWeights.data(), newWeightsSize, {0,4});
    // for easier debugging, can use 0 as padding value (real scenario with garbage padding):
    // DType padVal = 0;
    // std::fill(newWeights, newWeights + newWeightsSize, padVal);

    for (unsigned packedGroupIdx = 0; packedGroupIdx < nPackedGroups; packedGroupIdx++)
    {
        unsigned kOffset = packedGroupIdx * groupsPerVector * kPerGroup;
        for (unsigned g = 0; g < groupsPerVector; ++g)
        {
            for (unsigned spatialIdx = 0; spatialIdx < weightsSpatialSize; ++spatialIdx)
            {
                for (unsigned c = g * cPerGroup; c < (g + 1) * cPerGroup; ++c)
                {
                    DType* dst = newWeights.data() +
                                spatialIdx * C * kStride +
                                c * kStride +
                                kOffset +
                                g * kPerGroup;
                    memcpy(dst, src.data(), kPerGroup * sizeof(DType));
                }
            }
        }
    }
    return newWeights;
}
class SynGaudiGConvPackingTpcTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<int, int, double>>
{
public:
    SynGaudiGConvPackingTpcTest()
    : m_filterR(std::get<0>(GetParam())),
      m_filterS(std::get<0>(GetParam())),
      m_kPerGroup(std::get<1>(GetParam())),
      m_nOfDiagonals(std::get<2>(GetParam()))
    {
    }

    virtual void runSingleTest() = 0;
    virtual unsigned addTPCNode(TestSizes& wSizes, float* wData, unsigned kPerGroup, unsigned groupsPerVector, unsigned nPackedGroups, unsigned dims = 4) = 0;
    virtual void validateResults(unsigned inDataSize, unsigned outDataSize, float * refInputData, unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned nGroups, unsigned kStride, unsigned groupsForCurConv) = 0;

protected:
    unsigned m_filterR;
    unsigned m_filterS;
    unsigned m_nIFM;
    unsigned m_nOFM;
    unsigned m_nGroups;
    unsigned m_kPerGroup;
    double   m_nOfDiagonals;
    unsigned m_unpaddedWeightsTensor;
    unsigned m_paddedWeightsTensor;
};

template<typename DType>
class SynGaudiFwdDedxGConvTpcTest : public SynGaudiGConvPackingTpcTest
{
public:
    void runSingleTest() override;
    unsigned addTPCNode(TestSizes& wSizes, float* wData, unsigned kPerGroup, unsigned groupsPerVector, unsigned nPackedGroups, unsigned dims = 4) override;
    void validateResults(unsigned inDataSize, unsigned outDataSize, float * refInputData, unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned nGroups, unsigned kStride, unsigned groupsForCurConv) override;
};

template<typename DType>
void SynGaudiFwdDedxGConvTpcTest<DType>::runSingleTest()
{
    CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    unsigned vectorSizeInElements =
        CompilationHalReader::getHalReader()->getMmeMinimalWidthInElems(dataTypeToSynType<DType>());
    m_nGroups                     = std::max(1.0, m_nOfDiagonals * (vectorSizeInElements / m_kPerGroup));
    m_nIFM                        = m_kPerGroup * m_nGroups;
    m_nOFM                        = m_nIFM;

    TestSizes          wghDimSizes         = {m_nOFM, m_nIFM / m_nGroups, m_filterS, m_filterR, 1};
    unsigned           wDataSizeInElements = wghDimSizes[0] * wghDimSizes[1] * wghDimSizes[2] * wghDimSizes[3];
    std::vector<float> wData(wDataSizeInElements);
    fillWithRandom(wData.data(), wDataSizeInElements, {0, 4});

    unsigned C = wghDimSizes[WEIGHT_DIM_C];  //= cPerGroup
    unsigned K = wghDimSizes[WEIGHT_DIM_K];
    unsigned R = wghDimSizes[WEIGHT_DIM_R];
    unsigned S = wghDimSizes[WEIGHT_DIM_S];

    ASSERT_EQ(m_kPerGroup, K / m_nGroups);
    unsigned groupsPerVector = std::min(m_nGroups, std::max(1U, vectorSizeInElements / m_kPerGroup));
    unsigned nPackedGroups   = std::max(1.0, m_nOfDiagonals);  // Num of diagonals

    unsigned tpcOutSize = addTPCNode(wghDimSizes, wData.data(), m_kPerGroup, groupsPerVector, nPackedGroups);

    compileAndRun();
    //expecting 2 diagonals of 3 orig groups each
    validateResults(multiplyElements(wghDimSizes),
                    tpcOutSize,
                    wData.data(),
                    R,
                    S,
                    C,
                    m_kPerGroup,
                    nPackedGroups,
                    K,
                    groupsPerVector);
}

template<typename DType>
unsigned SynGaudiFwdDedxGConvTpcTest<DType>::addTPCNode(TestSizes& wSizes,
                                                        float*     wData,
                                                        unsigned   kPerGroup,
                                                        unsigned   groupsPerVector,
                                                        unsigned   nPackedGroups,
                                                        unsigned   dims)
{
    unsigned params = kPerGroup;
    TestSizes paddedWghSizes;
    paddedWghSizes = wSizes;

    paddedWghSizes[WEIGHT_DIM_K] = kPerGroup * nPackedGroups * groupsPerVector;
    paddedWghSizes[WEIGHT_DIM_C] = wSizes[WEIGHT_DIM_C] * groupsPerVector;

    synDataType type = dataTypeToSynType<DType>();
    std::string guid = fmt::format("gconv_fwd_{}", getDtypeSuffixFromSynDataType(type));

    m_unpaddedWeightsTensor = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_FROM_INITIALIZER,
                                                  wData,
                                                  wSizes.data(),
                                                  dims,
                                                  type,
                                                  nullptr,
                                                  "unpaddedWgh");
    m_paddedWeightsTensor   = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                paddedWghSizes.data(),
                                                dims,
                                                type,
                                                nullptr,
                                                "paddedWgh");

    addNodeToGraph(guid.c_str(),
                   {m_unpaddedWeightsTensor},
                   {m_paddedWeightsTensor},
                   &params,
                   sizeof(params),
                   "tpc_pad_node");

    return multiplyElements(paddedWghSizes);
}

template<typename DType>
void SynGaudiFwdDedxGConvTpcTest<DType>::validateResults(unsigned inDataSize,
                                                         unsigned outDataSize,
                                                         float*   tpcInputData,
                                                         unsigned R,
                                                         unsigned S,
                                                         unsigned cPerGroup,
                                                         unsigned kPerGroup,
                                                         unsigned nPackedGroups,
                                                         unsigned kStride,
                                                         unsigned groupsPerVector)
{
    ASSERT_TRUE((std::is_same_v<DType, float> || std::is_same_v<DType, bfloat16> || std::is_same_v<DType, fp8_152_t> ||
                 std::is_same_v<DType, fp8_143_t>))
        << "only float, bfloat16 and fp8 data types are supported";

    DType* tpcOutput = castHostBuffer<DType>(m_paddedWeightsTensor);
    if (std::is_same_v<DType, float>)
    {
        auto refPaddedWgh = PadWeightTensor<float>(tpcInputData,
                                                   R,
                                                   S,
                                                   cPerGroup * groupsPerVector,
                                                   kPerGroup * groupsPerVector,
                                                   groupsPerVector,
                                                   nPackedGroups,
                                                   kStride);

        // printExpectedVsActualTensors(outDataSize/groupsPerVector, outDataSize, groupsPerVector, kPerGroup,
        // nPackedGroups, tpcInputData, refPaddedWgh.data(), (float*)tpcOutput);

        for (int i = 0; i < outDataSize; i++)
        {
            ASSERT_FLOAT_EQ(float(tpcOutput[i]), refPaddedWgh[i]);
        }
    }
    else
    {
        std::unique_ptr<DType[]> castInputData =
            std::unique_ptr<DType[]>(convertBuffer<DType>(tpcInputData, inDataSize));
        std::vector<DType> refPaddedWgh = PadWeightTensor<DType>(castInputData.get(),
                                                                 R,
                                                                 S,
                                                                 cPerGroup * groupsPerVector,
                                                                 kPerGroup * groupsPerVector,
                                                                 groupsPerVector,
                                                                 nPackedGroups,
                                                                 kStride);

        // prints the input data buffer, expected output, and actual output
        // printExpectedVsActualTensors(outDataSize/groupsPerVector, outDataSize, groupsPerVector, kPerGroup,
        // nPackedGroups, tpcInputData, refPaddedWgh.data(), (bfloat16*)tpcOutput);

        for (int i = 0; i < outDataSize; i++)
        {
            float expected = float(refPaddedWgh[i]);
            float actual   = float(tpcOutput[i]);
            ASSERT_FLOAT_EQ(expected, actual);
        }
    }
}

template<typename DType>
class SynGaudiDedwGConvTpcTest : public SynGaudiGConvPackingTpcTest
{
public:
    void runSingleTest() override;
    unsigned addTPCNode(TestSizes& wSizes, float* wData, unsigned kPerGroup, unsigned groupsPerVector, unsigned nPackedGroups, unsigned dims = 4) override;
    void validateResults(unsigned inDataSize, unsigned outDataSize, float * refInputData, unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned nGroups, unsigned kStride, unsigned groupsForCurConv) override;
};

template<typename DType>
void SynGaudiDedwGConvTpcTest<DType>::runSingleTest()
{
    CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    unsigned vectorSizeInElements =
        CompilationHalReader::getHalReader()->getMmeMinimalWidthInElems(dataTypeToSynType<DType>());
    m_nGroups                     = std::max(1.0, m_nOfDiagonals * (vectorSizeInElements / m_kPerGroup));
    m_nIFM                        = m_kPerGroup * m_nGroups;
    m_nOFM                        = m_nIFM;

    TestSizes unpaddedWghDimSizes = {m_nOFM, m_nIFM / m_nGroups, m_filterS, m_filterR, 1};

    unsigned C = unpaddedWghDimSizes[WEIGHT_DIM_C];  //= cPerGroup
    unsigned K = unpaddedWghDimSizes[WEIGHT_DIM_K];
    unsigned R = unpaddedWghDimSizes[WEIGHT_DIM_R];
    unsigned S = unpaddedWghDimSizes[WEIGHT_DIM_S];

    ASSERT_EQ(m_kPerGroup, K / m_nGroups);
    unsigned groupsPerVector = std::min(m_nGroups, std::max(1U, vectorSizeInElements / m_kPerGroup));
    unsigned nPackedGroups   = std::max(1.0, m_nOfDiagonals);  // Num of diagonals

    TestSizes paddedWghSizes;
    paddedWghSizes = unpaddedWghDimSizes;
    paddedWghSizes[WEIGHT_DIM_K] = m_kPerGroup * nPackedGroups * groupsPerVector;
    paddedWghSizes[WEIGHT_DIM_C] = unpaddedWghDimSizes[WEIGHT_DIM_C] * groupsPerVector;
    auto wData                   = CreatePaddedWeightTensor<float>(R,
                                                 S,
                                                 C * groupsPerVector,
                                                 m_kPerGroup * groupsPerVector,
                                                 groupsPerVector,
                                                 nPackedGroups,
                                                 K);

    unsigned tpcOutSize = addTPCNode(paddedWghSizes, wData.data(), m_kPerGroup, groupsPerVector, nPackedGroups);

    compileAndRun();

    validateResults(multiplyElements(paddedWghSizes),
                    tpcOutSize,
                    wData.data(),
                    R,
                    S,
                    C,
                    m_kPerGroup,
                    nPackedGroups,
                    K,
                    groupsPerVector);
}

template<typename DType>
unsigned SynGaudiDedwGConvTpcTest<DType>::addTPCNode(TestSizes& wSizes,
                                                     float*     wData,
                                                     unsigned   kPerGroup,
                                                     unsigned   groupsPerVector,
                                                     unsigned   nPackedGroups,
                                                     unsigned   dims)
{
    unsigned params = kPerGroup;
    TestSizes unpaddedWghDimSizes = {m_nOFM, m_nIFM / m_nGroups, m_filterR, m_filterS, 1};

    synDataType type = dataTypeToSynType<DType>();
    std::string guid = fmt::format("gconv_bwd_{}", getDtypeSuffixFromSynDataType(type));

    m_paddedWeightsTensor   = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                wData,
                                                wSizes.data(),
                                                dims,
                                                type,
                                                nullptr,
                                                "paddedWgh");
    m_unpaddedWeightsTensor = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  unpaddedWghDimSizes.data(),
                                                  dims,
                                                  type,
                                                  nullptr,
                                                  "unpaddedWgh");

    addNodeToGraph(guid.c_str(),
                   {m_paddedWeightsTensor},
                   {m_unpaddedWeightsTensor},
                   &params,
                   sizeof(params),
                   "tpc_unpad_node");

    return multiplyElements(unpaddedWghDimSizes);
}

template<typename DType>
void SynGaudiDedwGConvTpcTest<DType>::validateResults(unsigned inDataSize,
                                                      unsigned outDataSize,
                                                      float*   tpcInputData,
                                                      unsigned R,
                                                      unsigned S,
                                                      unsigned cPerGroup,
                                                      unsigned kPerGroup,
                                                      unsigned nPackedGroups,
                                                      unsigned kStride,
                                                      unsigned groupsPerVector)
{
    ASSERT_TRUE((std::is_same_v<DType, float> || std::is_same_v<DType, bfloat16> || std::is_same_v<DType, fp8_152_t> ||
                 std::is_same_v<DType, fp8_143_t>))
        << "only float, bfloat16 and fp8 data types are supported";

    DType* tpcOutput = castHostBuffer<DType>(m_unpaddedWeightsTensor);
    if (std::is_same_v<DType, float>)
    {
        auto refUnpaddedWgh = UnpadWeightTensor<float>(tpcInputData,
                                                       R,
                                                       S,
                                                       cPerGroup * groupsPerVector,
                                                       kPerGroup * groupsPerVector,
                                                       groupsPerVector,
                                                       nPackedGroups,
                                                       kStride);

        // printExpectedVsActualTensors(outDataSize/groupsPerVector, outDataSize, groupsPerVector, kPerGroup,
        // nPackedGroups, tpcInputData, refPaddedWgh.data(), (float*)tpcOutput);

        for (int i = 0; i < outDataSize; i++)
        {
            ASSERT_FLOAT_EQ(float(tpcOutput[i]), refUnpaddedWgh[i]);
        }
    }
    else
    {
        auto castInputData  = std::unique_ptr<DType[]>(convertBuffer<DType>(tpcInputData, inDataSize));
        auto refUnpaddedWgh = UnpadWeightTensor<DType>(castInputData.get(),
                                                       R,
                                                       S,
                                                       cPerGroup * groupsPerVector,
                                                       kPerGroup * groupsPerVector,
                                                       groupsPerVector,
                                                       nPackedGroups,
                                                       kStride);

        // prints the input data buffer, expected output, and actual output
        // printExpectedVsActualTensors(outDataSize/groupsPerVector, outDataSize, groupsPerVector, kPerGroup,
        // nPackedGroups, tpcInputData, refPaddedWgh.data(), (bfloat16*)tpcOutput);

        for (int i = 0; i < outDataSize; i++)
        {
            float expected = float(refUnpaddedWgh[i]);
            float actual   = float(tpcOutput[i]);
            ASSERT_FLOAT_EQ(expected, actual);
        }
    }
}

class SynGaudiGConvUnpackingTpcTestBf16 : public SynGaudiDedwGConvTpcTest<bfloat16>
{
};
class SynGaudiGConvPackingTpcTestBf16 : public SynGaudiFwdDedxGConvTpcTest<bfloat16>
{};
class SynGaudiGConvUnpackingTpcTestFloat : public SynGaudiDedwGConvTpcTest<float>
{};
class SynGaudiGConvPackingTpcTestFloat : public SynGaudiFwdDedxGConvTpcTest<float>
{};
class SynGaudiGConvUnpackingTpcTestFP8 : public SynGaudiDedwGConvTpcTest<fp8_152_t>
{
};
class SynGaudiGConvPackingTpcTestFP8 : public SynGaudiFwdDedxGConvTpcTest<fp8_152_t>
{};

class SynGaudiGConvUnpackingTpcTestHFP8 : public SynGaudiDedwGConvTpcTest<fp8_143_t>
{
};
class SynGaudiGConvPackingTpcTestHFP8 : public SynGaudiFwdDedxGConvTpcTest<fp8_143_t>
{
};

TEST_P_GC(SynGaudiGConvPackingTpcTestFP8, fp8_packing, {synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

TEST_P_GC(SynGaudiGConvUnpackingTpcTestFP8, fp8_unpacking, {synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

TEST_P_GC(SynGaudiGConvPackingTpcTestHFP8, h_fp8_packing, {synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

TEST_P_GC(SynGaudiGConvUnpackingTpcTestHFP8, h_fp8_unpacking, {synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

TEST_P_GC(SynGaudiGConvPackingTpcTestBf16, bf16_packing)
{
    runSingleTest();
}

// TODO Test not stable in CI [SW-94182]
TEST_P_GC(SynGaudiGConvUnpackingTpcTestBf16, DISABLED_bf16_unpacking)
{
    runSingleTest();
}

TEST_P_GC(SynGaudiGConvPackingTpcTestFloat, float32_packing)
{
    runSingleTest();
}

TEST_P_GC(SynGaudiGConvUnpackingTpcTestFloat, float32_unpacking)
{
    runSingleTest();
}

/// BF16 ///
/// Packing ///
INSTANTIATE_TEST_SUITE_P(pack_no_remainder_one_diag,
                         SynGaudiGConvPackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(1, 16, 1), std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvPackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvPackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(pack_remainder,
                         SynGaudiGConvPackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(
    gconv_packing_bf16_no_remainder_ASIC,
    SynGaudiGConvPackingTpcTestBf16,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_packing_bf16_remainder_ASIC,
    SynGaudiGConvPackingTpcTestBf16,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));

/// Unpacking ///
INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_one_diag,
                         SynGaudiGConvUnpackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(1, 8, 1),
                                           std::make_tuple(1, 16, 1),
                                           std::make_tuple(2, 16, 1)));
INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvUnpackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvUnpackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_remainder,
                         SynGaudiGConvUnpackingTpcTestBf16,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_bf16_no_remainder_ASIC_CI,
    SynGaudiGConvUnpackingTpcTestBf16,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_bf16_remainder_ASIC_CI,
    SynGaudiGConvUnpackingTpcTestBf16,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));
/// BF16 - End///

/// Float ///
/// Packing ///
INSTANTIATE_TEST_SUITE_P(pack_no_remainder_one_diag,
                         SynGaudiGConvPackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(1, 16, 1), std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(pack_remainder,
                         SynGaudiGConvPackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvPackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvPackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags
INSTANTIATE_TEST_SUITE_P(
    gconv_packing_fp32_no_remainder_L2,
    SynGaudiGConvPackingTpcTestFloat,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_packing_fp32_remainder_L2,
    SynGaudiGConvPackingTpcTestFloat,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));

/// Unpacking ///
INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvUnpackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvUnpackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_one_diag,
                         SynGaudiGConvUnpackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(1, 8, 1),
                                           std::make_tuple(1, 16, 1),
                                           std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(unpack_remainder,
                         SynGaudiGConvUnpackingTpcTestFloat,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_fp32_no_remainder_ASIC_CI,
    SynGaudiGConvUnpackingTpcTestFloat,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_fp32_remainder_L2,
    SynGaudiGConvUnpackingTpcTestFloat,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));
/// Float - End///

/// FP8 ///
/// Packing ///
INSTANTIATE_TEST_SUITE_P(pack_no_remainder_one_diag,
                         SynGaudiGConvPackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(1, 16, 1), std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(pack_remainder,
                         SynGaudiGConvPackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvPackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvPackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags
INSTANTIATE_TEST_SUITE_P(
    gconv_packing_fp32_no_remainder_L2,
    SynGaudiGConvPackingTpcTestFP8,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_packing_fp32_remainder_L2,
    SynGaudiGConvPackingTpcTestFP8,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));

/// Unpacking ///
INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvUnpackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvUnpackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_one_diag,
                         SynGaudiGConvUnpackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(1, 8, 1),
                                           std::make_tuple(1, 16, 1),
                                           std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(unpack_remainder,
                         SynGaudiGConvUnpackingTpcTestFP8,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_fp32_no_remainder_ASIC_CI,
    SynGaudiGConvUnpackingTpcTestFP8,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_fp32_remainder_L2,
    SynGaudiGConvUnpackingTpcTestFP8,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));
/// FP8 - End///

/// Float - End///

/// HFP8 ///
/// Packing ///
INSTANTIATE_TEST_SUITE_P(pack_no_remainder_one_diag,
                         SynGaudiGConvPackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(1, 16, 1), std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(pack_remainder,
                         SynGaudiGConvPackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvPackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(pack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvPackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags
INSTANTIATE_TEST_SUITE_P(
    gconv_packing_fp32_no_remainder_L2,
    SynGaudiGConvPackingTpcTestHFP8,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_packing_fp32_remainder_L2,
    SynGaudiGConvPackingTpcTestHFP8,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));

/// Unpacking ///
INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_aligned,
                         SynGaudiGConvUnpackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(3, 32, 4),    // 4 diags
                                           std::make_tuple(1, 16, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_multiple_diag_unaligned,
                         SynGaudiGConvUnpackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(5, 10, 2),    // 2 diags
                                           std::make_tuple(1, 10, 3),    // 3 diags
                                           std::make_tuple(1, 10, 4),    // 4 diags
                                           std::make_tuple(1, 10, 5)));  // 5 diags

INSTANTIATE_TEST_SUITE_P(unpack_no_remainder_one_diag,
                         SynGaudiGConvUnpackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(1, 8, 1),
                                           std::make_tuple(1, 16, 1),
                                           std::make_tuple(2, 16, 1)));

INSTANTIATE_TEST_SUITE_P(unpack_remainder,
                         SynGaudiGConvUnpackingTpcTestHFP8,
                         ::testing::Values(std::make_tuple(1, 20, 2 / 3)));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_fp32_no_remainder_ASIC_CI,
    SynGaudiGConvUnpackingTpcTestHFP8,
    ::testing::Combine(::testing::Range(1, 3),          // R, S sizes
                       ::testing::Range(1, 20, 3),      // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(1.0, 6.0, 1.0)  // nOfDiagonals
                       ));

INSTANTIATE_TEST_SUITE_P(
    gconv_unpacking_fp32_remainder_L2,
    SynGaudiGConvUnpackingTpcTestHFP8,
    ::testing::Combine(::testing::Range(1, 3),            // R, S sizes
                       ::testing::Range(1, 20, 3),        // C/K factor- c/kPerGroup (C/k = c/k factor * nGroups)
                       ::testing::Range(0.25, 1.0, 0.25)  // nOfDiagonals
                       ));
/// FP8 - End///
