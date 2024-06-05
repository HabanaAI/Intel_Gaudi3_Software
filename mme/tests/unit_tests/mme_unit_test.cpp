#include "mme_unit_test.h"
#include "gaudi2/mme.h"
#include "sim_tensor.h"
#include "tensor_comparator.h"
#include "data_types/non_standard_dtypes.h"
#include "include/general_utils.h"
#include "mme_reference.h"
#include <limits>
#include <memory>

using namespace MmeCommon;

TEST_F(MmeUTDataTypeConversionTest, bfloat_to_float)
{
    uint16_t infVal = 0x7f80;
    uint16_t zeroVal = 0x0;
    bf16_t bf16 = bf16_t::max();
    uint32_t fp = (uint32_t) 0x7f7f << 16;
    EXPECT_FALSE(bf16.isInf());
    EXPECT_FLOAT_EQ(bf16.toFloat(), reinterpret_ptr<float>(&fp));

    bf16.value() = infVal;
    EXPECT_TRUE(bf16.isInf());
    EXPECT_FLOAT_EQ(bf16.toFloat(), std::numeric_limits<float>::infinity());

    bf16.value() = zeroVal;
    EXPECT_TRUE(bf16.isZero());
    EXPECT_FLOAT_EQ(bf16.toFloat(), 0.0f);
}

TEST_F(MmeUTDataTypeConversionTest, tfloat32_to_float)
{
    uint32_t infVal = 0x3FC00 << 13;
    uint32_t zeroVal = 0x0;
    tf32_t tf32 = tf32_t::max();
    uint32_t fp = (uint32_t) 0x3FBFF << 13;
    EXPECT_FALSE(tf32.isInf());
    EXPECT_FLOAT_EQ(tf32.toFloat(), reinterpret_ptr<float>(&fp));

    tf32.value() = infVal;
    EXPECT_TRUE(tf32.isInf());
    EXPECT_FLOAT_EQ(tf32.toFloat(), std::numeric_limits<float>::infinity());

    tf32.value() = zeroVal;
    EXPECT_TRUE(tf32.isZero());
    EXPECT_FLOAT_EQ(tf32.toFloat(), 0.0f);
}

TEST_F(MmeUTDataTypeConversionTest, fp16_to_float)
{
    uint32_t infVal = 0x7C00;
    uint32_t zeroVal = 0x0;
    fp16_t fp16 = fp16_t::max();
    uint32_t fp = 0x477FE000;
    EXPECT_FALSE(fp16.isInf());
    EXPECT_FLOAT_EQ(fp16.toFloat(), reinterpret_ptr<float>(&fp));

    fp16.value() = infVal;
    EXPECT_TRUE(fp16.isInf());
    EXPECT_FLOAT_EQ(fp16.toFloat(), std::numeric_limits<float>::infinity());

    fp16.value() = zeroVal;
    EXPECT_TRUE(fp16.isZero());
    EXPECT_FLOAT_EQ(fp16.toFloat(), 0.0f);
}

TEST_F(MmeUTReferenceTest, reference_gemm_simple_test_fp32)
{
    SizeArray aSizes = {5, 1};
    SizeArray bSizes = {1, 5};
    SizeArray ySizes = {1, 1};
    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_fp32);
    MmeSimTensor b = createSimTensor(bSizes, 2, EMmeDataType::e_type_fp32);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    fillSimTensorWithData(10.0f, 5, a);
    fillSimTensorWithData(10.0f, 5, b);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(y, a, b, e_mme_ab);
    ASSERT_NE(y.data(), nullptr);
    ASSERT_EQ(*(float*) y.data(), 500.0f);
}

TEST_F(MmeUTReferenceTest, reference_gemm_api_simple_test_fp32)
{
    SizeArray aSizes = {5, 1};
    SizeArray bSizes = {1, 5};
    SizeArray ySizes = {1, 1};
    Matrix a(EMmeDataType::e_type_fp32, 1, 5);
    Matrix b(EMmeDataType::e_type_fp32, 5, 1);
    Matrix y(EMmeDataType::e_type_fp32, 1, 1);

    float val = 10.0f;
    DataBuffer data(5 * sizeof(float));
    for (unsigned i = 0; i < 5; i++)
    {
        memcpy((float*) data + i, &val, sizeof(float));
    }
    a.getMatrix().data = data;
    b.getMatrix().data = data;

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doGemm(y, a, b, false, false);
    ASSERT_NE(y.getMatrix().data.get(), nullptr);
    ASSERT_EQ(reinterpret_ptr<float>(y.getMatrix().data.get()), 500.0f);
}

TEST_F(MmeUTReferenceTest, reference_gemm_simple_test_bf16)
{
    SizeArray aSizes = {5, 1};
    SizeArray bSizes = {1, 5};
    SizeArray ySizes = {1, 1};
    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_bf16);
    MmeSimTensor b = createSimTensor(bSizes, 2, EMmeDataType::e_type_bf16);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    bf16_t val(10, RoundingMode::RoundToNearest);
    fillSimTensorWithData(val, 5, a);
    fillSimTensorWithData(val, 5, b);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(y, a, b, e_mme_ab);
    ASSERT_NE(y.data(), nullptr);
    float resF = reinterpret_ptr<float>(y.data());
    ASSERT_EQ(resF, 500.0f);
}

TEST_F(MmeUTReferenceTest, reference_gemm_2d_test_fp32)
{
    SizeArray aSizes = {5, 5};
    SizeArray bSizes = {5, 5};
    SizeArray ySizes = {5, 5};
    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_fp32_ieee);
    MmeSimTensor b = createSimTensor(bSizes, 2, EMmeDataType::e_type_fp32_ieee);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    fillSimTensorWithData(10.0f, 25, a);
    fillSimTensorWithData(5.0f, 25, b);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(y, a, b, e_mme_ab);
    ASSERT_NE(y.data(), nullptr);
    for (unsigned i = 0; i < 25; i++)
    {
        ASSERT_EQ(*((float*) y.data() + i), 250.0f);
    }
}

TEST_F(MmeUTReferenceTest, reference_gemm_2d_test_bf16)
{
    SizeArray aSizes = {5, 5};
    SizeArray bSizes = {5, 5};
    SizeArray ySizes = {5, 5};
    MmeSimTensor a   = createSimTensor(aSizes, 2, EMmeDataType::e_type_bf16);
    MmeSimTensor b   = createSimTensor(bSizes, 2, EMmeDataType::e_type_bf16);
    MmeSimTensor acc = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    MmeSimTensor y   = createSimTensor(ySizes, 2, EMmeDataType::e_type_bf16);
    bf16_t valA(10.0f), valB(5.0f);
    fillSimTensorWithData(valA, 25, a);
    fillSimTensorWithData(valB, 25, b);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(acc, a, b, e_mme_ab);
    pMMESimTensor pY = std::make_shared<MmeSimTensor>(y);
    pMMESimTensor pAcc = std::make_shared<MmeSimTensor>(acc);
    calculator.doActivation(pY, pAcc, nullptr, nullptr, false);
    ASSERT_NE(pY->data(), nullptr);
    for (unsigned i = 0; i < 25; i++)
    {
        bf16_t res(*(reinterpret_cast<uint16_t*>(pY->data()) + i));
        float resF = (float) res;
        ASSERT_EQ(resF, 250.0f);
    }
}
template<typename T>
static float getFloatVal(const T& val)
{
    return (float) val;
}
template<>
float getFloatVal<fp8_152_t>(const fp8_152_t& val)
{
    return val.toFloat(EXPONENT_BIAS_FP8_152_15);
}
template<>
float getFloatVal<fp8_143_t>(const fp8_143_t& val)
{
    return val.toFloat(7);
}

TYPED_TEST_CASE(MmeUTRandomTensorTest, Gaudi2Types);
TYPED_TEST(MmeUTRandomTensorTest, fill_sim_tensor)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    constexpr unsigned fillValue = 1;
    EMmeDataType type = this->getTensorType();
    pMMESimTensor tensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    tensor->setFpBias(this->getDefaultBias());
    RandomSimTensorGenerator generator(seed);
    generator.fill(tensor, fillValue);
    unsigned sizeInElements = tensor->getMemorySize() / tensor->getElementSize();
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        TypeParam actualVal = (reinterpret_ptr_with_index<TypeParam>(tensor->data(), element));
        float fVal = getFloatVal(actualVal);
        ASSERT_FLOAT_EQ(fVal, (float) fillValue) << "element - " << element << "type - " << type;
    }
}

TYPED_TEST(MmeUTRandomTensorTest, random_tensor_uniform)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    EMmeDataType type = this->getTensorType();
    pMMESimTensor tensor;
    RandomSimTensorGenerator generator(seed);
    if (std::is_same<TypeParam, fp8_152_t>::value)
    {
        tensor = std::make_shared<MmeSimTensor>(sizes, 4, type, nullptr, EXPONENT_BIAS_FP8_152_15);
    }
    else if (std::is_same<TypeParam, fp8_143_t>::value)
    {
        tensor = std::make_shared<MmeSimTensor>(sizes, 4, type, nullptr, 7);
    }
    else {
        tensor = std::make_shared<MmeSimTensor>(sizes, 4, type, nullptr);
    }
    generator.generateUniform(tensor, 0, 10);
    unsigned sizeInElements = tensor->getMemorySize() / tensor->getElementSize();
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        TypeParam actualVal = (reinterpret_ptr_with_index<TypeParam>(tensor->data(), element));
        float fVal = getFloatVal(actualVal);
        ASSERT_TRUE(fVal >= 0.0f && fVal <= 10.0f)
            << "element - " << element << " fVal - " << fVal << " type - " << type;
    }
}

TEST_F(MMEUnitTest, fill_and_compare_float)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    constexpr unsigned fillValue = 1035;  // sum of 1..45
    EMmeDataType type = EMmeDataType::e_type_fp32;
    pMMESimTensor fillTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    pMMESimTensor accumulatedTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    RandomSimTensorGenerator generator(seed);
    // fill tensor with initial value
    generator.fill(fillTensor, fillValue);
    // build accumulate tensor by adding one fillValue times
    unsigned sizeInElements = accumulatedTensor->getSizeInElements();
    // calculate value -  sum of 1...45
    float accumulatedVal = 0;
    for (unsigned count = 1; count < 46; count++)
    {
        accumulatedVal += (float) count;
    }
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        float* actualValP = &(reinterpret_ptr_with_index<float>(accumulatedTensor->data(), element));
        *actualValP = accumulatedVal;
    }

    TensorComparator comparator(false, type);
    ASSERT_TRUE(comparator.doCompare(fillTensor, "fillTensor", accumulatedTensor, "accumulatedTensor"));
}

TEST_F(MMEUnitTest, fill_and_compare_bfloat16)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    constexpr unsigned fillValue = 1035;  // sum of 1..45
    EMmeDataType type = EMmeDataType::e_type_bf16;
    pMMESimTensor fillTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    pMMESimTensor accumulatedTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    RandomSimTensorGenerator generator(seed);
    // fill tensor with initial value
    generator.fill(fillTensor, fillValue);
    // build accumulate tensor by adding one fillValue times
    unsigned sizeInElements = accumulatedTensor->getSizeInElements();
    // calculate value -  sum of 1...45
    bf16_t accumulatedVal(0.0f);
    for (unsigned count = 1; count < 46; count++)
    {
        accumulatedVal = bfloat16::add(accumulatedVal, bf16_t((float) count));
    }
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        bf16_t* actualValP = &(reinterpret_ptr_with_index<bf16_t>(accumulatedTensor->data(), element));
        *actualValP = accumulatedVal;
    }

    TensorComparator comparator(false, type);
    ASSERT_TRUE(comparator.doCompare(fillTensor, "fillTensor", accumulatedTensor, "accumulatedTensor"));
}

TEST_F(MMEUnitTest, fill_and_compare_fp16)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    constexpr unsigned fillValue = 1035;  // sum of 1..45
    EMmeDataType type = EMmeDataType::e_type_fp16;
    pMMESimTensor fillTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    pMMESimTensor accumulatedTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    RandomSimTensorGenerator generator(seed);
    // fill tensor with initial value
    generator.fill(fillTensor, fillValue);
    // build accumulate tensor by adding one fillValue times
    unsigned sizeInElements = accumulatedTensor->getSizeInElements();
    // calculate value -  sum of 1...45
    fp16_t accumulatedVal(1.0f);
    for (unsigned count = 2; count < 46; count++)
    {
        accumulatedVal = fp16_t::add(accumulatedVal, fp16_t((float) count));
    }
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        fp16_t* actualValP = &(reinterpret_ptr_with_index<fp16_t>(accumulatedTensor->data(), element));
        *actualValP = accumulatedVal;
    }

    TensorComparator comparator(false, type);
    ASSERT_TRUE(comparator.doCompare(fillTensor, "fillTensor", accumulatedTensor, "accumulatedTensor"));
}

TEST_F(MMEUnitTest, fill_and_compare_tf32)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    constexpr unsigned fillValue = 1035;  // sum of 1..45
    EMmeDataType type = EMmeDataType::e_type_tf32;
    pMMESimTensor fillTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    pMMESimTensor accumulatedTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    RandomSimTensorGenerator generator(seed);
    // fill tensor with initial value
    generator.fill(fillTensor, fillValue);
    // build accumulate tensor by adding one fillValue times
    unsigned sizeInElements = accumulatedTensor->getSizeInElements();
    // calculate value -  sum of 1...45
    tf32_t accumulatedVal(0.0f);
    for (unsigned count = 1; count < 46; count++)
    {
        accumulatedVal = tf32_t::add(accumulatedVal, tf32_t((float) count));
    }
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        tf32_t* actualValP = &(reinterpret_ptr_with_index<tf32_t>(accumulatedTensor->data(), element));
        *actualValP = accumulatedVal;
    }

    TensorComparator comparator(false, type);
    ASSERT_TRUE(comparator.doCompare(fillTensor, "fillTensor", accumulatedTensor, "accumulatedTensor"));
}

TEST_F(MMEUnitTest, comparison_failure_bf16)
{
    SizeArray sizes = {64, 32, 32, 64, 1};
    constexpr unsigned seed = 1;
    constexpr unsigned fillValue = 1035;  // sum of 1..45
    EMmeDataType type = EMmeDataType::e_type_bf16;
    pMMESimTensor fillTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    pMMESimTensor accumulatedTensor = std::make_shared<MmeSimTensor>(sizes, 4, type);
    RandomSimTensorGenerator generator(seed);
    // fill tensor with initial value
    generator.fill(fillTensor, fillValue);
    // build accumulate tensor by adding one fillValue times
    unsigned sizeInElements = accumulatedTensor->getSizeInElements();
    // calculate value -  sum of 1...45
    bf16_t accumulatedVal(0.0f);
    for (unsigned count = 1; count < 46; count++)
    {
        accumulatedVal = bfloat16::add(accumulatedVal, bf16_t((float) count));
    }
    for (unsigned element = 0; element < sizeInElements; element++)
    {
        if (element == 1000)
        {
            accumulatedVal = bfloat16::add(accumulatedVal, bf16_t(100.0f));
        }
        bf16_t* actualValP = &(reinterpret_ptr_with_index<bf16_t>(accumulatedTensor->data(), element));
        *actualValP = accumulatedVal;
    }

    TensorComparator comparator(false, type);
    ASSERT_FALSE(comparator.doCompare(fillTensor, "fillTensor", accumulatedTensor, "accumulatedTensor"));
    Settable<SizeArray> diffElement = comparator.getDiffElement();
    ASSERT_TRUE(diffElement.is_set());
    // check that diffElement points to the correct offset
    ASSERT_EQ(diffElement.value()[0] + diffElement.value()[1] * sizes[0], 1000);
    ASSERT_EQ(diffElement.value()[2], 0);
    ASSERT_EQ(diffElement.value()[3], 0);
    ASSERT_EQ(diffElement.value()[4], 0);
}

TEST_F(MMEUnitTest, fp8_143_convert)
{
    // FP32 Input to be converted to fp8
    float f32 = -0.25;
    // Representation of -0.25 as FP8-143: sign=1 | exponent=0 | mantissa=1
    uint8_t f8_143_expected = 0x88;
    // Convert FP32 to FP8-143
    fp8_143_t f8(f32, RoundingMode::RoundToNearest, 3);
    // Get the representation of FP8-143 and check correctness
    uint8_t f8_143_actual = (uint8_t) f8;
    ASSERT_EQ(f8_143_actual, f8_143_expected);
    // Convert FP8-143 to FP32 and check correctness
    float res = f8.toFloat(3);
    ASSERT_EQ(res, f32);
}

TEST_F(MMEUnitTest, fp8_143_mul)
{
    // Input in FP32: a * b
    float a = 0.25;
    float b = -0.5;
    float ref = a * b;

    // Convert FP32 to FP8-143 and multiply
    fp8_143_t fa(a, RoundingMode::RoundToNearest, 15);
    fp8_143_t fb(b, RoundingMode::RoundToNearest, 15);
    float res = fa.toFloat(15) * fb.toFloat(15);

    ASSERT_EQ(res, ref);
}

TEST_F(MmeUTReferenceTest, reference_gemm_simple_test_fp8_143)
{
    SizeArray aSizes = {2, 1};
    SizeArray bSizes = {1, 2};
    SizeArray ySizes = {1, 1};
    static constexpr unsigned bias = 7;
    fp8_143_t valA = fp8_143_t(0.25f, RoundingMode::RoundToNearest, bias);
    fp8_143_t valB = fp8_143_t(-0.5f, RoundingMode::RoundToNearest, bias);
    float expectedOut = valA.toFloat(bias) * valB.toFloat(bias) * 2;

    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_fp8_143);
    MmeSimTensor b = createSimTensor(bSizes, 2, EMmeDataType::e_type_fp8_143);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    fillSimTensorWithData(valA, 2, a, bias);
    fillSimTensorWithData(valB, 2, b, bias);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(y, a, b, e_mme_ab);
    ASSERT_NE(y.data(), nullptr);
    float resF = reinterpret_ptr<float>(y.data());
    fp8_143_t out = fp8_143_t(resF, RoundingMode::RoundToNearest, bias);
    ASSERT_EQ(resF, expectedOut);
}

TEST_F(MmeUTReferenceTest, reference_conv_fwd_fp8_143)
{
    SizeArray aSizes = {5, 5};
    SizeArray wSizes = {5, 5};
    SizeArray ySizes = {5, 5};
    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_fp8_143);
    MmeSimTensor w = createSimTensor(wSizes, 2, EMmeDataType::e_type_fp8_143);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    fillSimTensorWithData((uint8_t) fp8_143_t(0.5f, RoundingMode::RoundToNearest, 3), 25, a);
    fillSimTensorWithData((uint8_t) fp8_143_t(-0.5f, RoundingMode::RoundToNearest, 3), 4, w);

    ConvolutionParams params;
    params.dim = 2;
    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doConvolution(y, a, w, y, params, EMmeOpType::e_mme_fwd, RoundingMode::RoundToNearest);
    ASSERT_NE(y.data(), nullptr);
}

TEST_F(MMEUnitTest, fp8_152_convert)
{
    // FP32 Input to be converted to fp8
    float f32 = -0.5;
    // Representation of -0.5 as FP8-152: sign=1 | exponent=0 | mantissa=1
    uint8_t f8_152_expected = 0xB8;
    // Convert FP32 to FP8-152
    fp8_152_t f8(f32, RoundingMode::RoundToNearest, 15);
    // Get the representation of FP8-152 and check correctness
    uint8_t f8_152_actual = (uint8_t) f8;
    ASSERT_EQ(f8_152_actual, f8_152_expected);
    // Convert FP8-152 to FP32 and check correctness
    float res = f8.toFloat(15);
    ASSERT_EQ(res, f32);
}

TEST_F(MMEUnitTest, fp8_152_mul)
{
    // Input in FP32: a * b
    float a = 0.5;
    float b = -0.5;
    float ref = a * b;

    // Convert FP32 to FP8-152 and multiply
    fp8_152_t fa(a);
    fp8_152_t fb(b);
    float res = fa * fb;

    ASSERT_EQ(res, ref);
}

TEST_F(MmeUTReferenceTest, reference_gemm_simple_test_fp8_152)
{
    SizeArray aSizes = {2, 1};
    SizeArray bSizes = {1, 2};
    SizeArray ySizes = {1, 1};
    static constexpr unsigned bias = EXPONENT_BIAS_FP8_152_15;
    fp8_152_t valA = fp8_152_t(0.5f, RoundingMode::RoundToNearest, bias);
    fp8_152_t valB = fp8_152_t(-0.5f, RoundingMode::RoundToNearest, bias);
    float expectedOut = valA.toFloat(bias) * valB.toFloat(bias) * 2;

    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_fp8_152);
    MmeSimTensor b = createSimTensor(bSizes, 2, EMmeDataType::e_type_fp8_152);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    fillSimTensorWithData(valA, 2, a, bias);
    fillSimTensorWithData(valB, 2, b, bias);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doBatchGemm(y, a, b, e_mme_ab);
    ASSERT_NE(y.data(), nullptr);
    float resF = reinterpret_ptr<float>(y.data());
    ASSERT_EQ(resF, expectedOut);
}

TEST_F(MmeUTReferenceTest, reference_conv_fwd_fp8_152)
{
    SizeArray aSizes = {5, 5};
    SizeArray wSizes = {5, 5};
    SizeArray ySizes = {5, 5};
    MmeSimTensor a = createSimTensor(aSizes, 2, EMmeDataType::e_type_fp8_152);
    MmeSimTensor w = createSimTensor(wSizes, 2, EMmeDataType::e_type_fp8_152);
    MmeSimTensor y = createSimTensor(ySizes, 2, EMmeDataType::e_type_fp32);
    fillSimTensorWithData((uint8_t) fp8_152_t(0.5f, RoundingMode::RoundToNearest, EXPONENT_BIAS_FP8_152_15), 25, a);
    fillSimTensorWithData((uint8_t) fp8_152_t(-0.5f, RoundingMode::RoundToNearest, EXPONENT_BIAS_FP8_152_15), 4, w);

    ConvolutionParams params;
    params.dim = 2;
    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doConvolution(y, a, w, y, params, EMmeOpType::e_mme_fwd, RoundingMode::RoundToNearest);
    ASSERT_NE(y.data(), nullptr);
}

TEST_F(MmeUTReferenceTest, reference_gemm_api)
{
    bool transposeA = true;
    bool transposeB = false;
    unsigned int size_a1 = 128;
    unsigned int size_common_dimension5 = 128;
    unsigned int size_b1 = 256;
    SizeArray aSizes = {size_common_dimension5, size_a1};
    SizeArray bSizes = {size_b1, size_common_dimension5};
    SizeArray ySizes = {size_b1, size_a1};

    fp16_t val(10, RoundingMode::RoundToNearest);
    DataBuffer data_a(size_common_dimension5 * size_a1 * sizeof(uint16_t));
    DataBuffer data_b(size_common_dimension5 * size_b1 * sizeof(uint16_t));

    for (unsigned i = 0; i < size_a1; i++)
    {
        for (unsigned k = 0; k < size_common_dimension5; k++)
        {
            memcpy((uint16_t*) data_a + i * size_common_dimension5 + k, &val, sizeof(uint16_t));
        }
    }
    for (unsigned h = 0; h < size_common_dimension5; h++)
    {
        for (unsigned p = 0; p < size_b1; p++)
        {
            memcpy((uint16_t*) data_b + h * size_b1 + p, &val,
                   sizeof(uint16_t));  // copy all matrix
        }
    }
    Matrix a(EMmeDataType::e_type_fp16, size_a1, size_common_dimension5, data_a.get());
    Matrix b(EMmeDataType::e_type_fp16, size_common_dimension5, size_b1, data_b.get());
    Matrix y(EMmeDataType::e_type_fp32, size_a1, size_b1);

    CPUCalculator calculator(e_mme_Gaudi2, Gaudi2::Mme::c_mme_max_tensor_dims, Gaudi2::Mme::c_mme_max_conv_dims);
    calculator.doGemm(y, a, b, transposeA, transposeB);
}
