#include "data_layout_test_infra.h"
#include "sizes_nested_loop_generator.h"
#include "utils.h"
#include "h2d_tensors.h"
#include "syn_gaudi_two_run_compare_test.h"

#define SKIP_TEST_IF(cond)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (cond)                                                                                                      \
        {                                                                                                              \
            GTEST_SKIP() << "Skipping test: " << #cond;                                                                \
        }                                                                                                              \
    } while (0)

enum SliceType
{
    SLICE_FWD = 1,
    SLICE_GRAD,
    SLICE_INSERT
};

enum SizeType
{
    MIN = 0,
    MAX,
    ACTUAL,
};

typedef std::array<TestSizeVec, 3> DynamicTestSizes;

class SynTwoRunCompareDynamicSliceTest : public SynGaudiTwoRunCompareTest
{
};

class SynGaudiDynamicSliceTest
: public SynGaudiDataLayoutTest
, public testing::WithParamInterface<std::tuple<SliceType,
                                                bool /* is dynamic size */,
                                                bool /* is dynamic starts */,
                                                bool /* is dynamic steps */,
                                                bool /* is input permuted */,
                                                bool /* is 64bit data */,
                                                std::tuple<TestSizeVec /* realSize */,
                                                           TestSizeVec /* aliasSize */,
                                                           TestSizeVec /* starts */,
                                                           TestSizeVec /* steps */>>>
{
public:
    SynGaudiDynamicSliceTest()
    {
        setTestPackage(TEST_PACKAGE_DSD);
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }
protected:
    DynamicTestSizes               m_starts;
    DynamicTestSizes               m_steps;
    SliceType                      m_type;
    synDataType                    m_dataType;
    std::optional<gc::Permutation> m_optionalInputPermutation = std::nullopt;

    DynamicTestSizes m_sizesAlias;
    DynamicTestSizes m_sizesReal;

    template<class DType>
    void
    checkSlice(const DType* inputData, const DType* outputData, SizesNestedLoopGenerator<unsigned>::Sizes aliasIndex)
    {
        auto&    sizesIn      = (m_type == SLICE_FWD) ? m_sizesReal : m_sizesAlias;
        auto&    sizesOut     = (m_type == SLICE_FWD) ? m_sizesAlias : m_sizesReal;
        uint64_t strideIn     = 1;
        uint64_t strideOut    = 1;
        uint64_t inputOffset  = 0;
        uint64_t outputOffset = 0;
        for (unsigned i = 0; i < aliasIndex.size(); i++)
        {
            if (m_type == SLICE_FWD)
            {
                inputOffset += strideIn * (m_starts[ACTUAL][i] + aliasIndex[i] * m_steps[ACTUAL][i]);
                outputOffset += strideOut * aliasIndex[i];
            }
            else
            {
                outputOffset += strideOut * (m_starts[ACTUAL][i] + aliasIndex[i] * m_steps[ACTUAL][i]);
                inputOffset += strideIn * aliasIndex[i];
            }
            strideIn *= sizesIn[ACTUAL][i];
            strideOut *= sizesOut[ACTUAL][i];
        }
        DType result   = outputData[outputOffset];
        DType expected = inputData[inputOffset];
        ASSERT_EQ(result, expected) << "Mismatch at index " << toString(aliasIndex, ',') << " Expected: " << expected
                                    << " Result: " << result;
    }

    template<class DType>
    unsigned countNumEqual(const DType* outputData, const DType* inputData)
    {
        auto&    sizesOut    = (m_type == SLICE_FWD) ? m_sizesAlias : m_sizesReal;
        unsigned numElements = multiplyElements(sizesOut[ACTUAL].begin(), sizesOut[ACTUAL].end());
        unsigned ret         = 0;
        for (unsigned i = 0; i < numElements; i++)
        {
            DType expected = (inputData != nullptr) ? inputData[i] : DType(0);
            ret += (outputData[i] == expected);
        }
        return ret;
    }

    template<class DType>
    void validateResults(const DynamicTestSizes& sizesIn,
                         const DynamicTestSizes& sizesOut,
                         unsigned                in,
                         unsigned                out,
                         unsigned                originalTensor)
    {
        const DType* outputData = castHostBuffer<DType>(out);
        DType*       inputData  = castHostBuffer<DType>(in);
        if (m_optionalInputPermutation.has_value())
        {
            if (m_type == SLICE_INSERT)
            {
                DType* origTensorData = castHostBuffer<DType>(originalTensor);
                auto   sizesOrig      = sizesOut;
                // transpose orig tensor shape from NCHW NHWC
                permuteShapes(sizesOrig, m_optionalInputPermutation.value());
                // transpose original tensor buffer from NHWC to NCHW
                transposeBuffer(sizesOrig[ACTUAL].data(),
                                sizesOrig[ACTUAL].size(),
                                origTensorData,
                                m_optionalInputPermutation.value().getInversePermutation());
            }
            // transpose input tensor shape from NCHW NHWC
            auto inputSizes = sizesIn;
            permuteShapes(inputSizes, m_optionalInputPermutation.value());
            // transpose in tensor buffer from NHWC to NCHW
            transposeBuffer(inputSizes[ACTUAL].data(),
                            inputSizes[ACTUAL].size(),
                            inputData,
                            m_optionalInputPermutation.value().getInversePermutation());
        }
        // compare elements not mapped to alias tensor-
        // for slice grad: count zeros. for slice insert: count equal elements to original tensor
        if (m_type == SLICE_GRAD || m_type == SLICE_INSERT)
        {
            DType*   compareData      = m_type == SLICE_INSERT ? castHostBuffer<DType>(originalTensor) : nullptr;
            uint32_t numEqual         = countNumEqual(outputData, compareData);
            uint32_t numElementsReal  = multiplyElements(m_sizesReal[ACTUAL].begin(), m_sizesReal[ACTUAL].end());
            uint32_t numElementsAlias = multiplyElements(m_sizesAlias[ACTUAL].begin(), m_sizesAlias[ACTUAL].end());
            ASSERT_GE(numEqual, numElementsReal - numElementsAlias);
        }

        SizesNestedLoopGenerator sizesGenerator(m_sizesAlias[ACTUAL]);
        while (!sizesGenerator.isDone())
        {
            const auto aliasIdx = sizesGenerator.nextState();
            checkSlice(inputData, outputData, aliasIdx);
        }
    }

    std::string getGuid()
    {
        switch (m_type)
        {
            case SLICE_FWD:
                return NodeFactory::sliceNodeTypeName;
            case SLICE_GRAD:
                return NodeFactory::stridedSliceGradNodeTypeName;
            case SLICE_INSERT:
                return NodeFactory::sliceInsertNodeTypeName;
        }
        return "";
    }

    void permuteShapes(DynamicTestSizes& shapes, const gc::Permutation& perm) const
    {
        for (auto& shape : shapes)
        {
            perm.permuteShape(shape.data(), shape.size());
        }
    }

    void runSlice()
    {
        auto&    sizesIn  = (m_type == SLICE_FWD) ? m_sizesReal : m_sizesAlias;
        auto&    sizesOut = (m_type == SLICE_FWD) ? m_sizesAlias : m_sizesReal;

        TensorIndices inputs;
        std::unordered_set<unsigned> shapeTensorIndices {};
        unsigned numElements = multiplyElements(sizesIn[MAX].begin(), sizesIn[MAX].end());
        std::vector<float> inValues(numElements);
        // init to successive values to ease debugging
        std::iota(inValues.begin(), inValues.end(), 1001);
        unsigned in = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_FROM_INITIALIZER,
                                          inValues.data(),
                                          sizesIn[MAX].data(),
                                          sizesIn[MAX].size(),
                                          m_dataType,
                                          nullptr,
                                          "input",
                                          0,
                                          0,
                                          nullptr,
                                          sizesIn[MIN].data());
        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizesOut[MAX].data(),
                                           sizesOut[MAX].size(),
                                           m_dataType,
                                           nullptr,
                                           "out",
                                           0,
                                           0,
                                           nullptr,
                                           sizesOut[MIN].data());

        if (m_type == SLICE_INSERT)
        {
            unsigned numRealElements = multiplyElements(m_sizesReal[MAX].begin(), m_sizesReal[MAX].end());
            std::vector<float> inRealValues(numRealElements);
            // init to successive values to ease debugging
            std::iota(inRealValues.begin(), inRealValues.end(), 2001);
            unsigned original = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    inRealValues.data(),
                                                    m_sizesReal[MAX].data(),
                                                    m_sizesReal[MAX].size(),
                                                    m_dataType,
                                                    nullptr,
                                                    "original",
                                                    0,
                                                    0,
                                                    nullptr,
                                                    m_sizesReal[MIN].data());
            inputs.push_back(original);
            ASSERT_EQ(inputs.size(), 1) << "Expecting original tensor idx to be inserted first!";
            inputs.push_back(in);
        }
        else
        {
            unsigned shapeT =
                createShapeTensor(INPUT_TENSOR, sizesOut[MAX].data(), sizesOut[MIN].data(), sizesOut[MAX].size());
            inputs.push_back(in);
            inputs.push_back(shapeT);
            shapeTensorIndices.insert(shapeT);
        }
        synDynamicSliceDmaH2dTensor paramData[2] = {{0}, {0}};
        paramData[0].dims = m_steps[MAX].size();
        std::copy(m_steps[MAX].begin(), m_steps[MAX].end(), paramData[0].steps);
        std::copy(m_starts[MAX].begin(), m_starts[MAX].end(), paramData[0].starts);
        paramData[1].dims = m_steps[MIN].size();
        std::copy(m_steps[MIN].begin(), m_steps[MIN].end(), paramData[1].steps);
        std::copy(m_starts[MIN].begin(), m_starts[MIN].end(), paramData[1].starts);
        unsigned paramSize[1] = {sizeof(synDynamicSliceDmaH2dTensor)/sizeof(unsigned)};
        unsigned paramT = createHost2DeviceTensor(INPUT_TENSOR, paramSize, (unsigned*)paramData, 1);
        inputs.push_back(paramT);
        shapeTensorIndices.insert(paramT);

        if (m_optionalInputPermutation.has_value())
        {
            for (auto& in : inputs)
            {
                bool isDataTensor = shapeTensorIndices.find(in) == shapeTensorIndices.end();
                if (isDataTensor)
                {
                    setPermutation(in, m_optionalInputPermutation.value());
                }
            }
        }

        addNodeToGraph(getGuid().c_str(), inputs, {out}, nullptr, 0, getGuid().c_str());

        compileTopology();

        if (m_type == SLICE_INSERT)
        {
            // sizes for original tensor
            setActualSizes(inputs[0], m_sizesReal[ACTUAL].data());
        }
        else
        {
            // sizes for shape tensor
            setActualSizes(inputs[1], sizesOut[ACTUAL].data());
        }
        setActualSizes(in, sizesIn[ACTUAL].data());
        setActualSizes(out, sizesOut[ACTUAL].data());
        synDynamicSliceDmaH2dTensor actualData = {0};
        actualData.dims = m_steps[ACTUAL].size();
        std::copy(m_steps[ACTUAL].begin(), m_steps[ACTUAL].end(), actualData.steps);
        std::copy(m_starts[ACTUAL].begin(), m_starts[ACTUAL].end(), actualData.starts);
        setActualScalarParametersData(paramT, &actualData, sizeof(synDynamicSliceDmaH2dTensor));

        runTopology();

        switch (m_dataType)
        {
            case syn_type_int64:
                validateResults<int64_t>(sizesIn, sizesOut, in, out, inputs[0]);
                break;
            case syn_type_float:
                validateResults<float>(sizesIn, sizesOut, in, out, inputs[0]);
                break;
            default:
                HB_ASSERT(0, "non supported data type in test");
        }
    }
};

TEST_P_GC(SynGaudiDynamicSliceTest, slice_test, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    m_type                      = std::get<0>(GetParam());
    bool        dynamicSize     = std::get<1>(GetParam());
    bool        dynamicStart    = std::get<2>(GetParam());
    bool        dynamicStep     = std::get<3>(GetParam());
    bool        isPermutedInput = std::get<4>(GetParam());
    bool        is64Bit         = std::get<5>(GetParam());
    const auto& params          = std::get<6>(GetParam());

    // exceptions
    SKIP_TEST_IF(is64Bit && isPermutedInput);  // [SW-97305] - currently we don't support transpose nodes in 64 bit

    // blacklist tests that do not run on Gaudi3
    //
    auto param = GetParam();
    using ParamType = decltype(param);
    std::vector<ParamType> blacklistedParams
    {
      {SLICE_FWD, true, false, true, false, false, {{ 2, 4, 1, 1 }, { 2, 4, 1, 1 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 }}},
      {SLICE_FWD, true, true,  true, false, false, {{ 2, 4, 1, 1 }, { 2, 4, 1, 1 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 }}},
    };

    SKIP_TEST_IF (m_deviceType == synDeviceGaudi3 &&
                  std::find(blacklistedParams.begin(), blacklistedParams.end(), param) != blacklistedParams.end());

    m_sizesReal[MIN] = m_sizesReal[MAX] = m_sizesReal[ACTUAL] = std::get<0>(params);
    m_sizesAlias[MIN] = m_sizesAlias[MAX] = m_sizesAlias[ACTUAL] = std::get<1>(params);
    m_starts[MIN] = m_starts[MAX] = m_starts[ACTUAL] = std::get<2>(params);
    m_steps[MIN] = m_steps[MAX] = m_steps[ACTUAL] = std::get<3>(params);

    if (dynamicStep)
    {
        m_steps[MAX][0] = 1;
        m_steps[MIN][0] += 1;
        m_steps[MAX][2] = 1;
        m_steps[MIN][2] += 1;
    }
    if (dynamicStart)
    {
        m_starts[MAX][0] = 0;
        m_starts[MIN][0] += 1;
        m_starts[MAX][1] = 0;
        m_starts[MIN][1] += 1;
    }
    if (dynamicSize)
    {
        m_sizesAlias[MAX][0] += 1;
        m_sizesAlias[MAX][3] += 1;
        m_sizesReal[MAX][0] =
            std::max(m_sizesAlias[MAX][0] * m_steps[MAX][0] + m_starts[MAX][0] + 1, m_sizesReal[ACTUAL][0] + 1);
        m_sizesReal[MAX][3] =
            std::max(m_sizesAlias[MAX][3] * m_steps[MAX][3] + m_starts[MAX][3] + 1, m_sizesReal[ACTUAL][3] + 1);
    }
    if (isPermutedInput)
    {
        const gc::Permutation arbitraryPermutation(DimVector {TPD_Width, TPD_Height, TPD_Channel, TPD_4Dim_Batch});
        const auto&           sizesIn = (m_type == SLICE_FWD) ? m_sizesReal[ACTUAL] : m_sizesAlias[ACTUAL];
        HB_ASSERT(sizesIn.size() == arbitraryPermutation.size(),
                  "Expecting input rank {} == const permutation length {}",
                  sizesIn.size(),
                  arbitraryPermutation.size());
        m_optionalInputPermutation = arbitraryPermutation;
    }
    m_dataType = is64Bit ? syn_type_int64 : syn_type_float;

    const bool isDynamic         = dynamicSize || dynamicStart || dynamicStep;
    const bool isLowRankOperands = std::max(m_sizesAlias[MAX].size(), m_sizesReal[MAX].size()) <= SYN_MAX_TENSOR_DIM;
    HB_ASSERT(isLowRankOperands || (!m_optionalInputPermutation.has_value() && !isDynamic),
              "Dynamic/Permuted tensors cannot be of high rank (ndim)");

    runSlice();
}

INSTANTIATE_TEST_SUITE_P(
    dynamic,
    SynGaudiDynamicSliceTest,
    ::testing::Combine(::testing::ValuesIn({SLICE_FWD, SLICE_GRAD, SLICE_INSERT}),             // slice type
                       ::testing::ValuesIn({false, true}),                                     // dynamic size
                       ::testing::ValuesIn({false, true}),                                     // dynamic starts
                       ::testing::ValuesIn({false, true}),                                     // dynamic steps
                       ::testing::ValuesIn({false, true}),                                     // input permutation
                       ::testing::ValuesIn({false, true}),                                     // is 64bit data
                       ::testing::Values(  // real sizes, alias sizes, starts, steps
                           std::make_tuple(TestSizeVec {2, 4, 1, 1},
                                           TestSizeVec {2, 4, 1, 1},
                                           TestSizeVec {0, 0, 0, 0},
                                           TestSizeVec {1, 1, 1, 1}),
                           std::make_tuple(TestSizeVec {6, 6, 1, 1},
                                           TestSizeVec {3, 6, 1, 1},
                                           TestSizeVec {0, 0, 0, 0},
                                           TestSizeVec {2, 1, 1, 1}),
                           std::make_tuple(TestSizeVec {6, 6, 1, 1},
                                           TestSizeVec {4, 2, 1, 1},
                                           TestSizeVec {1, 1, 0, 0},
                                           TestSizeVec {1, 2, 1, 1}),
                           std::make_tuple(TestSizeVec {9, 9, 9, 1},
                                           TestSizeVec {7, 2, 9, 1},
                                           TestSizeVec {1, 1, 0, 0},
                                           TestSizeVec {1, 3, 1, 1}),
                           std::make_tuple(TestSizeVec {8, 8, 8, 8},
                                           TestSizeVec {2, 2, 2, 2},
                                           TestSizeVec {2, 2, 2, 2},
                                           TestSizeVec {2, 2, 2, 2}),
                           std::make_tuple(TestSizeVec {8, 8, 8, 8},
                                           TestSizeVec {2, 2, 2, 2},
                                           TestSizeVec {2, 2, 2, 2},
                                           TestSizeVec {2, 1, 2, 1}))));

// TODO: Enable after [SW-110143] is done
INSTANTIATE_TEST_SUITE_P(
    DISABLED_ndim,
    SynGaudiDynamicSliceTest,
    ::testing::Combine(::testing::ValuesIn({SLICE_FWD, SLICE_GRAD, SLICE_INSERT}),             // slice type
                       ::testing::ValuesIn({false}),                                           // dynamic size
                       ::testing::ValuesIn({false}),                                           // dynamic starts
                       ::testing::ValuesIn({false}),                                           // dynamic steps
                       ::testing::ValuesIn({false}),                                           // input permutation
                       ::testing::ValuesIn({false, true}),                                     // is 64bit data
                       ::testing::Values(  // real sizes,     alias sizes,            starts, steps
                           std::make_tuple(TestSizeVec {2, 4, 1, 1, 1, 1},
                                           TestSizeVec {2, 4, 1, 1, 1, 1},
                                           TestSizeVec {0, 0, 0, 0, 0, 0},
                                           TestSizeVec {1, 1, 1, 1, 1, 1}),
                           std::make_tuple(TestSizeVec {6, 6, 1, 1, 1, 1},
                                           TestSizeVec {3, 6, 1, 1, 1, 1},
                                           TestSizeVec {0, 0, 0, 0, 0, 0},
                                           TestSizeVec {2, 1, 1, 1, 1, 1}),
                           std::make_tuple(TestSizeVec {8, 8, 8, 8, 8, 8},
                                           TestSizeVec {2, 2, 2, 2, 2, 2},
                                           TestSizeVec {2, 2, 2, 2, 2, 2},
                                           TestSizeVec {2, 1, 2, 1, 2, 2}))));

TEST_F_GC(SynTwoRunCompareDynamicSliceTest, static_reshape_optimization)
{

    unsigned i0MaxSizes[] = {1, 2, 29300, 2};
    unsigned i0MinSizes[] = {1, 2, 24100, 2};

    unsigned i1MaxSizes[] = {1, 2, 29200, 2};
    unsigned i1MinSizes[] = {1, 2, 24000, 2};

    unsigned i2Sizes[] = {1, 1, 1, 1};

    unsigned i3MaxSizes[] = {0, 0, 100, 0};
    unsigned i3MinSizes[] = {0, 0, 0, 0};

    unsigned oMaxSizes[] = {1, 2, 29200, 2};
    unsigned oMinSizes[] = {1, 2, 24000, 2};

    unsigned in0 = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       i0MaxSizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       i0MinSizes);
    unsigned in1 = createShapeTensor(INPUT_TENSOR, i1MaxSizes, i1MinSizes, 4, syn_type_uint32);
    unsigned in2 = createShapeTensor(INPUT_TENSOR, i2Sizes, i2Sizes, 4, syn_type_uint32);
    unsigned in3 = createShapeTensor(INPUT_TENSOR, i3MaxSizes, i3MinSizes, 4, syn_type_uint32);
    unsigned out_ =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, oMaxSizes, 4, syn_type_bf16, nullptr, oMinSizes);
    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       oMaxSizes,
                                       4,
                                       syn_type_bf16,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       oMinSizes);

    addNodeToGraph("slice", {in0, in1, in2, in3}, {out_});
    addNodeToGraph("neg_fwd_bf16", {out_}, {out});

    setActualSizes(in0, i0MinSizes);
    setActualSizes(in1, i1MinSizes);
    setActualSizes(in3, i3MaxSizes);
    setActualSizes(out, oMinSizes);

    addConfigurationToRun(FIRST_RUN, "ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION", "true");
    compareRunsResults({out});
}
