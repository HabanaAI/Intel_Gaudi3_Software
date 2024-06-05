#include "gc_dynamic_shapes_infra.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "types.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>

class SynGaudiStridedViewInsertTestDynamic
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<bool /* stridedInsert or stridedView */,
                                                bool /* is dynamic input */,
                                                bool /* is dynamic output */,
                                                bool /* is dynamic view params */,
                                                synDataType /* op data type */,
                                                std::tuple<TestSizes /* inputSizeMin */,
                                                           TestSizes /* outputSize */,
                                                           TestSizes /* strides */,
                                                           TestSizes /* base offset */>>>
{
public:
    SynGaudiStridedViewInsertTestDynamic() { setTestPackage(TEST_PACKAGE_DSD); }
    static const unsigned NUM_DIMS = 4;

protected:
    TestSizes m_sizesIn;
    TestSizes m_sizesView;
    TestSizes m_strides;
    TestSizes m_offset;
    TestSizes m_sizesInMax;
    TestSizes m_sizesViewMax;
    TestSizes m_stridesMax;
    TestSizes m_offsetMax;
    TestSizes m_sizesInMin;
    TestSizes m_sizesViewMin;
    TestSizes m_stridesMin;
    TestSizes m_offsetMin;
    synDataType m_dataType;

    bool isOOB(const TestSizes& real, const TestSizes& view, const TestSizes& strides, unsigned offset)
    {
        uint64_t realTensorElements = std::accumulate(real.begin(), real.end(), 1, std::multiplies<unsigned>());
        uint64_t lastElementOffset  = 0;
        for (unsigned d = 0; d < NUM_DIMS; d++)
        {
            lastElementOffset += (uint64_t)strides[d] * (view[d] - 1);
        }
        if (offset + lastElementOffset >= realTensorElements)
        {
            return true;
        }
        return false;
    }

    template<class DType>
    void validateStridedView(unsigned out, unsigned in)
    {
        const DType* outputData = castHostBuffer<DType>(out);
        const DType* inputData  = castHostBuffer<DType>(in);
        unsigned     outIndex   = 0;
        for (unsigned i = 0; i < m_sizesView[3]; i++)
            for (unsigned j = 0; j < m_sizesView[2]; j++)
                for (unsigned k = 0; k < m_sizesView[1]; k++)
                    for (unsigned l = 0; l < m_sizesView[0]; l++)
                    {
                        unsigned inIndex =
                            i * m_strides[3] + j * m_strides[2] + k * m_strides[1] + l * m_strides[0] + m_offset[0];
                        ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                            << "Mismatch at index " << i << "," << j << "," << k << "," << l
                            << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
                        outIndex++;
                    }
    }

    template<class DType>
    void validateStridedInsert(unsigned out, unsigned inReal, unsigned inView)
    {
        const DType* outputData    = castHostBuffer<DType>(out);
        const DType* inputData     = castHostBuffer<DType>(inReal);
        const DType* inputViewData = castHostBuffer<DType>(inView);
        // count number of diffs between the original tensor and the output tensor
        unsigned numDiffs = 0;
        for (unsigned i = 0; i < m_sizesIn[0] * m_sizesIn[1] * m_sizesIn[2] * m_sizesIn[3]; i++)
        {
            if (outputData[i] != inputData[i])
            {
                numDiffs++;
            }
        }
        ASSERT_LE(numDiffs, m_sizesView[0] * m_sizesView[1] * m_sizesView[2] * m_sizesView[3])
            << "too many diffs in original tensor!";

        unsigned viewIndex = 0;
        for (unsigned i = 0; i < m_sizesView[3]; i++)
            for (unsigned j = 0; j < m_sizesView[2]; j++)
                for (unsigned k = 0; k < m_sizesView[1]; k++)
                    for (unsigned l = 0; l < m_sizesView[0]; l++)
                    {
                        // offset for view index in the output
                        unsigned realOutIndex =
                            i * m_strides[3] + j * m_strides[2] + k * m_strides[1] + l * m_strides[0] + m_offset[0];

                        // compare inserted data
                        DType viewValue = inputViewData[viewIndex];
                        ASSERT_EQ(viewValue, outputData[realOutIndex])
                            << "Mismatch at index " << i << "," << j << "," << k << "," << l
                            << " Expected: " << viewValue << " Result: " << outputData[realOutIndex];
                        viewIndex++;
                    }
    }

    void runStridedView()
    {
        unsigned in  = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          m_sizesInMax.data(),
                                          NUM_DIMS,
                                          m_dataType,
                                          nullptr,
                                          "input",
                                          0,
                                          0,
                                          nullptr,
                                          m_sizesInMin.data());
        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_sizesViewMax.data(),
                                           NUM_DIMS,
                                           m_dataType,
                                           nullptr,
                                           "out",
                                           0,
                                           0,
                                           nullptr,
                                           m_sizesViewMin.data());

        unsigned shapeT  = createShapeTensor(INPUT_TENSOR, m_sizesViewMax.data(), m_sizesViewMin.data(), NUM_DIMS);
        unsigned strideT = createShapeTensor(INPUT_TENSOR, m_stridesMax.data(), m_stridesMin.data(), NUM_DIMS);
        unsigned offsetT = createShapeTensor(INPUT_TENSOR, m_offsetMax.data(), m_offsetMin.data(), 1);

        addNodeToGraph(NodeFactory::stridedViewNodeTypeName,
                       {in, shapeT, strideT, offsetT},
                       {out},
                       nullptr,
                       0,
                       "strided_view");

        compileTopology();

        setActualSizes(in, m_sizesIn.data());
        setActualSizes(out, m_sizesView.data());
        setActualSizes(shapeT, m_sizesView.data());
        setActualSizes(strideT, m_strides.data());
        setActualSizes(offsetT, m_offset.data());

        runTopology();

        switch (m_dataType)
        {
            case syn_type_int64:
                validateStridedView<int64_t>(out, in);
                break;
            case syn_type_float:
                validateStridedView<float>(out, in);
                break;
            default:
                HB_ASSERT(0, "non supported data type in test");
        }
    }

    void runStridedInsert()
    {
        unsigned inReal = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              m_sizesInMax.data(),
                                              NUM_DIMS,
                                              m_dataType,
                                              nullptr,
                                              "input_real",
                                              0,
                                              0,
                                              nullptr,
                                              m_sizesInMin.data());

        unsigned inView = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              m_sizesViewMax.data(),
                                              NUM_DIMS,
                                              m_dataType,
                                              nullptr,
                                              "input_view",
                                              0,
                                              0,
                                              nullptr,
                                              m_sizesViewMin.data());

        unsigned strideT = createShapeTensor(INPUT_TENSOR, m_stridesMax.data(), m_stridesMin.data(), NUM_DIMS);
        unsigned offsetT = createShapeTensor(INPUT_TENSOR, m_offsetMax.data(), m_offsetMin.data(), 1);

        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_sizesInMax.data(),
                                           NUM_DIMS,
                                           m_dataType,
                                           nullptr,
                                           "out",
                                           0,
                                           0,
                                           nullptr,
                                           m_sizesInMin.data());

        addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                       {inReal, inView, strideT, offsetT},
                       {out},
                       nullptr,
                       0,
                       "strided_insert");

        compileTopology();

        setActualSizes(inReal, m_sizesIn.data());
        setActualSizes(inView, m_sizesView.data());
        setActualSizes(out, m_sizesIn.data());
        setActualSizes(strideT, m_strides.data());
        setActualSizes(offsetT, m_offset.data());

        runTopology();

        switch (m_dataType)
        {
            case syn_type_int64:
                validateStridedInsert<int64_t>(out, inReal, inView);
                break;
            case syn_type_float:
                validateStridedInsert<float>(out, inReal, inView);
                break;
            default:
                HB_ASSERT(0, "non supported data type in test");
        }
    }

    void runSingleTest()
    {
        auto        isInsert      = std::get<0>(GetParam());
        auto        dynamicInput  = std::get<1>(GetParam());
        auto        dynamicView   = std::get<2>(GetParam());
        auto        dynamicParams = std::get<3>(GetParam());
        m_dataType                = std::get<4>(GetParam());
        const auto& params        = std::get<5>(GetParam());

        m_sizesIn   = std::get<0>(params);
        m_sizesView = std::get<1>(params);
        m_strides   = std::get<2>(params);
        m_offset    = std::get<3>(params);

        m_sizesInMax   = m_sizesIn;
        m_sizesInMin   = m_sizesIn;
        m_sizesViewMax = m_sizesView;
        m_sizesViewMin = m_sizesView;
        m_stridesMax   = m_strides;
        m_stridesMin   = m_strides;
        m_offsetMax    = m_offset;
        m_offsetMin    = m_offset;

        if (dynamicInput)
        {
            m_sizesInMin[0] -= 1;
            m_sizesInMin[1] -= 1;
            m_sizesInMax[0] += 1;
            m_sizesInMax[1] += 1;
        }
        if (dynamicView)
        {
            m_sizesViewMin[0] -= 1;
            m_sizesViewMin[1] -= 1;
            m_sizesViewMax[0] += 1;
            m_sizesViewMax[1] += 1;
        }
        if (dynamicParams)
        {
            m_stridesMax[0] = 0;
            m_stridesMax[1] = 0;
            m_offsetMax[0]  = 0;
        }

        if (isOOB(m_sizesInMax, m_sizesViewMax, m_stridesMax, m_offsetMax[0]))
        {
            m_sizesViewMax = m_sizesView;
        }
        if (isOOB(m_sizesInMin, m_sizesViewMin, m_stridesMin, m_offsetMin[0]))
        {
            m_sizesInMin = m_sizesIn;
        }

        if (isInsert)
        {
            runStridedView();
        }
        else
        {
            runStridedInsert();
        }
    }
};

TEST_P_GC(SynGaudiStridedViewInsertTestDynamic, viewInsertTest, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynGaudiStridedViewInsertTestDynamic,
    ::testing::Combine(
        ::testing::ValuesIn({true, false}),                      // view or insert
        ::testing::ValuesIn({true, false}),                      // dynamicInput
        ::testing::ValuesIn({true, false}),                      // dynamicView
        ::testing::ValuesIn({true, false}),                      // dynamicParams
        ::testing::ValuesIn({syn_type_single, syn_type_int64}),  // data type
        ::testing::Values(  // input sizes,   output sizes,   strides (elements), offset
            std::make_tuple(TestSizes {2, 4, 1, 1}, TestSizes {2, 2, 1, 1}, TestSizes {1, 4, 4, 4}, TestSizes {0}),
            std::make_tuple(TestSizes {2, 4, 1, 1}, TestSizes {2, 2, 1, 1}, TestSizes {1, 6, 6, 6}, TestSizes {0}),
            std::make_tuple(TestSizes {4, 2, 3, 4}, TestSizes {4, 3, 2, 2}, TestSizes {1, 8, 4, 48}, TestSizes {0}),
            std::make_tuple(TestSizes {2, 4, 3, 1}, TestSizes {2, 2, 3, 1}, TestSizes {1, 4, 8, 8}, TestSizes {2}),
            std::make_tuple(TestSizes {4, 2, 1, 1}, TestSizes {2, 2, 1, 1}, TestSizes {2, 4, 4, 4}, TestSizes {1}))));

class BasicReadWriteTests : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(BasicReadWriteTests, basic_read_node, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned sizesIn[] = {2, 8, 2};  // strides - [1, 2, 16]

    unsigned maxSizesOut[]    = {2, 6, 2};  // strides - [1, 2, 16]
    unsigned minSizesOut[]    = {2, 2, 2};  // strides - [1, 8, 16]
    unsigned actualSizesOut[] = {2, 4, 2};  // strides - [1, 4, 16]

    unsigned minStrides[]    = {1, 8, 16};
    unsigned maxStrides[]    = {1, 2, 16};
    unsigned actualStrides[] = {1, 4, 16};

    unsigned minOffset[]    = {2};
    unsigned maxOffset[]    = {0};
    unsigned actualOffset[] = {2};

    unsigned dim = ARRAY_SIZE(maxSizesOut);

    unsigned inTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizesIn, dim, syn_type_float);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             maxSizesOut,
                                             dim,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             minSizesOut);

    unsigned shape   = createShapeTensor(INPUT_TENSOR, maxSizesOut, minSizesOut, dim);
    unsigned strides = createShapeTensor(INPUT_TENSOR, maxStrides, minStrides, dim);
    unsigned offset  = createShapeTensor(INPUT_TENSOR, maxOffset, minOffset, 1);

    addNodeToGraph(NodeFactory::stridedViewNodeTypeName, {inTensor, shape, strides, offset}, {outTensor}, nullptr, 0);

    compileTopology();

    setActualSizes(shape, actualSizesOut);
    setActualSizes(strides, actualStrides);
    setActualSizes(offset, actualOffset);
    setActualSizes(outTensor, actualSizesOut);
    runTopology(0, true);

    float* inData  = castHostInBuffer<float>(inTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    int count = 0;
    for (int i = 0; i < actualSizesOut[2]; i++)
        for (int j = 0; j < actualSizesOut[1]; j++)
            for (int k = 0; k < actualSizesOut[0]; k++)
            {
                unsigned offset = i * actualStrides[2] + j * actualStrides[1] + k * actualStrides[0] + actualOffset[0];
                ASSERT_EQ(outData[count], inData[offset]) << "Mismatch at output index " << count << ", input index " << offset;
                count++;
            }
}

TEST_F_GC(BasicReadWriteTests, basic_write_node, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned sizesIn[] = {2, 8, 2};  // strides - [1, 2, 16]

    unsigned maxSizesInsert[]    = {2, 6, 2};  // strides - [1, 2, 16]
    unsigned minSizesInsert[]    = {2, 2, 2};  // strides - [1, 8, 16]
    unsigned actualSizesInsert[] = {2, 4, 2};  // strides - [1, 4, 16]

    unsigned minStrides[]    = {1, 8, 16};
    unsigned maxStrides[]    = {1, 2, 16};
    unsigned actualStrides[] = {1, 4, 16};

    unsigned minOffset[]    = {2};
    unsigned maxOffset[]    = {0};
    unsigned actualOffset[] = {2};

    unsigned dim = ARRAY_SIZE(maxSizesInsert);

    unsigned inTensor =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizesIn, dim, syn_type_single);

    unsigned insertTensor = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                maxSizesInsert,
                                                dim,
                                                syn_type_single,
                                                nullptr,
                                                "insert",
                                                0,
                                                0,
                                                nullptr,
                                                minSizesInsert);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesIn, dim, syn_type_single);

    unsigned strides = createShapeTensor(INPUT_TENSOR, maxStrides, minStrides, dim);
    unsigned offset  = createShapeTensor(INPUT_TENSOR, maxOffset, minOffset, 1);

    addNodeToGraph(NodeFactory::stridedInsertNodeTypeName,
                   {inTensor, insertTensor, strides, offset},
                   {outTensor},
                   nullptr,
                   0);
    compileTopology();

    setActualSizes(strides, actualStrides);
    setActualSizes(offset, actualOffset);
    setActualSizes(insertTensor, actualSizesInsert);
    runTopology(0, true);

    float* inData     = castHostInBuffer<float>(inTensor);
    float* insertData = castHostInBuffer<float>(insertTensor);
    float* outData    = castHostOutBuffer<float>(outTensor);

    unsigned diffs = 0;
    for (int i = 0; i < sizesIn[0] * sizesIn[1] * sizesIn[2]; i++)
    {
        if (inData[i] != outData[i])
        {
            diffs++;
        }
    }
    ASSERT_LE(diffs, actualSizesInsert[0] * actualSizesInsert[1] * actualSizesInsert[2]);

    int count = 0;
    for (int i = 0; i < actualSizesInsert[2]; i++)
        for (int j = 0; j < actualSizesInsert[1]; j++)
            for (int k = 0; k < actualSizesInsert[0]; k++)
            {
                unsigned offset = i * actualStrides[2] + j * actualStrides[1] + k * actualStrides[0] + actualOffset[0];
                ASSERT_EQ(outData[offset], insertData[count]) << "Mismatch at output index " << offset << ", input index " << count;
                count++;
            }
}
