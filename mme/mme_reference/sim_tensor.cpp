#include "sim_tensor.h"
#include "data_types/non_standard_dtypes.h"
#include "mme_common_utils.h"
#include "print_utils.h"
#include <cstdint>

using namespace MmeCommon;

MmeSimTensor::MmeSimTensor(const SizeArray& sizes,
                           int dim,
                           EMmeDataType type,
                           char* data,
                           unsigned fpBias,
                           MmeCommon::InfNanMode infNanMode,
                           const SizeArray* strides)
: MMESimTensorBase(sizes, dim, type, fpBias, infNanMode, data, strides)
{
    checkTypeAndSetSize(type);
    uint64_t tensorSize = setStridesAndGetTensorSize(strides, dim);
    setData(tensorSize, data, 0, false);
}

MmeSimTensor::MmeSimTensor(const int* sizes,
                           int dim,
                           EMmeDataType type,
                           char* data,
                           int* strides,
                           int alignment,
                           const bool shouldCopyData)
: MMESimTensorBase(sizes, dim, type, data, strides)
{
    uint64_t tensorSize = 0;
    SizeArray stridesArray = {0};
    checkTypeAndSetSize(type);
    if (strides != nullptr)
    {
        for (unsigned i = 0; i < MAX_DIMENSION; ++i)
        {
            stridesArray[i] = strides[i];
        }
        tensorSize = setStridesAndGetTensorSize(&stridesArray, dim);
    }
    else
    {
        tensorSize = setStridesAndGetTensorSize(nullptr, dim);
    }
    setData(tensorSize, data, alignment, shouldCopyData);
}

void MmeSimTensor::checkTypeAndSetSize(EMmeDataType type)
{
    switch (type)
    {
        case e_type_fp8_143:
        case e_type_fp8_152:
            setSize(sizeof(fp8_152_t));
            break;
        case e_type_bf16:
            setSize(sizeof(bf16_t));
            break;
        case e_type_ufp16:
        case e_type_fp16:
            setSize(sizeof(fp16_t));
            break;
        case e_type_int16:
        case e_type_uint16:
            setSize(sizeof(int16_t));
            break;
        case e_type_fp32:
        case e_type_fp32_ieee:
            setSize(sizeof(float));
            break;
        case e_type_tf32:
            setSize(sizeof(tf32_t));
            break;
        case e_type_int4:
        case e_type_uint4:
            setSize(sizeof(int8_t));
            break;
        case e_type_int8:
        case e_type_uint8:
            setSize(sizeof(int8_t));
            break;
        case e_type_int32:
        case e_type_int32_26:
        case e_type_int32_16:
            setSize(sizeof(int32_t));
            break;
        default:
            MME_ASSERT(0, "Type is not supported yet");
    }
}

void MmeSimTensor::memsetTensor(const char* value)
{
    uint8_t byteValue = (uint8_t)(*value);
    memset(data(), byteValue, getSizeInElements());
}

template<typename T>
void RandomSimTensorGenerator::generateInternal(pMMESimTensor& tensor,
                                                Operation op,
                                                float minValue,
                                                float maxValue,
                                                float mean,
                                                float stdDev,
                                                unsigned reductionPacking,
                                                unsigned reductionLevel)
{
    MME_ASSERT(minValue <= maxValue, "min value is larger then max value");
    if (op == Operation::NormalDistribution && ((mean < minValue) || (mean > maxValue)))
    {
        atomicColoredPrint(COLOR_YELLOW, "WARNING: Wrong test. Mean is outside [MinValue..MaxValue] range.\n");
    }

    float typeMinVal = getLowestVal<T>(tensor->getFpBias(), tensor->getInfNanMode());  // lowest negative number
    float typeMaxVal = getMaxVal<T>(tensor->getFpBias(), tensor->getInfNanMode());  // highest positive number
    typeMinVal = std::max(minValue, typeMinVal);
    typeMaxVal = std::min(maxValue, typeMaxVal);
    std::uniform_real_distribution<float> uniformDistribution(typeMinVal, typeMaxVal);
    std::normal_distribution<float> normalDistribution(mean, stdDev);
    DataBuffer buffer(tensor->getMemorySize());
    T* it = (T*) buffer;
    uint64_t sizeInElements = tensor->getSizeInElements();
    T value = T(0.0f);
    unsigned rowSize = reductionPacking * reductionLevel;
    unsigned row, col, colWithinPxP;

    for (uint64_t currentElement = 0; currentElement < sizeInElements; currentElement++)
    {
        // constant value (fill) or random.
        switch (op)
        {
            case Operation::UnitMatrix:
                // initialize the first value to 1, the rest to 0
                getValue<T>(currentElement == 0 ? maxValue : minValue,
                            &value,
                            tensor->getFpBias(),
                            tensor->getInfNanMode());
                break;
            case Operation::Fill:
                getValue<T>(typeMinVal, &value, tensor->getFpBias(), tensor->getInfNanMode());
                break;
            case Operation::UniformDistribution:
            {
                float randVal = uniformDistribution(m_randomGen);
                getValue<T>(randVal, &value, tensor->getFpBias(), tensor->getInfNanMode());
                break;
            }
            case Operation::NormalDistribution:
            {
                float randVal = normalDistribution(m_randomGen);
                // cut value to min\max val
                randVal = std::min(std::max(randVal, typeMinVal), typeMaxVal);
                getValue<T>(randVal, &value, tensor->getFpBias(), tensor->getInfNanMode());
                break;
            }
            case Operation::FillForReductionPacking:
                // The packing aux tensor targets summing up the N partial results using GEMM with the packing aux as A and
                // the primary aux tensor as B.
                // Naive implementation is gemm of {4, Q*R*S*C*K,1,1,1} x {4, Q*R*S*C*K, 1, 1, 1}, where the A tensor is all 1's.
                // Much more efficient is to pack the GEMM such that we produce PackingFactor results in a single step (instead of 1),
                // and setting A tensor to 1's and 0's to achieve the correct result.
                // For example, by setting the packing tensor (A) to {4*Q*R, Q*R, 1, 1, 1} x {S*C*K, 4*Q*R, 1, 1, 1}. The packing
                // tensor should be set to:
                //
                //              N * PackingFactor (4*Q*R)
                //        | 1      | 1      | 1      | 1      |
                //        |  1     |  1     |  1     |  1     |
                //        |   1    |   1    |   1    |   1    |
                //        |  ...   |  ...   |  ...   |  ...   |  PackingFactor (Q * R)
                //        |   ...  |   ...  |   ...  |   ...  |
                //        |      1 |      1 |      1 |      1 |
                //
                // Therefore, any element which belong to the diagon as above should be 1, others should be 0
                row = currentElement / rowSize;
                col = currentElement % rowSize;
                // Divide the tensor intp PxP boxes
                colWithinPxP = col % reductionPacking;
                // The element should be set to 1 if it is on the slant (row = colWithinPxP), otherwise 0
                getValue<T>(row == colWithinPxP ? 1.0f : 0.0f, &value, tensor->getFpBias(), tensor->getInfNanMode());
                break;
            default:
                MME_ASSERT(0, "Illegal fill operation");
        }  // end switch

        *it = value;
        it++;
    }
    tensor->setData(buffer);
}

#define REGISTER_TYPE_DATA_GEN(enumType, realType)                                                                     \
    m_funcMap[enumType] = std::mem_fn(&RandomSimTensorGenerator::generateInternal<realType>);

RandomSimTensorGenerator::RandomSimTensorGenerator(unsigned int seed) : m_randomGen(seed)
{
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_fp16, fp16_t);
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_ufp16, ufp16_t);
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_bf16, bf16_t);
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_fp32_ieee, fp32_t);
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_fp32, fp32_t);
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_tf32, fp32_t);  //  tf32 values will be generated in runtime by the MME
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_fp8_143, fp8_143_t);
    REGISTER_TYPE_DATA_GEN(EMmeDataType::e_type_fp8_152, fp8_152_t);
}

void RandomSimTensorGenerator::generateUniform(pMMESimTensor& tensor, float minValue, float maxValue)
{
    // redirect to the correct type specification function
    auto func = m_funcMap[tensor->getElementType()];
    return func(this,
                tensor,
                Operation::UniformDistribution,
                minValue,
                maxValue,
                0 /* mean - not used*/,
                1 /*stdDev - not used*/,
                1 /* packingFactor - not used*/,
                1 /* reductionLevel */);
}

void RandomSimTensorGenerator::generateNormal(pMMESimTensor& tensor,
                                              float minValue,
                                              float maxValue,
                                              float mean,
                                              float stdDev)
{
    // redirect to the correct type specification function
    auto func = m_funcMap[tensor->getElementType()];
    return func(this, tensor, Operation::NormalDistribution, minValue, maxValue, mean, stdDev, 1, 1);
}

void RandomSimTensorGenerator::fill(pMMESimTensor& tensor, float value)
{
    // fill tensor with constant value
    auto func = m_funcMap[tensor->getElementType()];
    return func(this, tensor, Operation::Fill, value, value,
          0 /*mean - not used*/, 1 /*stdDev - not used*/, 1 /* packingFactor - not used */, 1 /* reductionLevel */);
}

void RandomSimTensorGenerator::fillForReductionPacking(pMMESimTensor& tensor, unsigned packingFactor, unsigned reductionLevel)
{
    // fill tensor with constant value
    auto func = m_funcMap[tensor->getElementType()];
    return func(this, tensor, Operation::FillForReductionPacking, 0 /* not used */, 0 /* not used */,
                 0 /*mean - not used*/, 1 /*stdDev - not used*/, packingFactor, reductionLevel);
}

void RandomSimTensorGenerator::generateUnitMatrix(pMMESimTensor& tensor, float unitValue, float otherValues)
{
    // fill tensor with constant value
    auto func = m_funcMap[tensor->getElementType()];
    return func(this,
                tensor,
                Operation::UnitMatrix,
                otherValues,
                unitValue,
                0 /*mean - not used*/,
                1 /*stdDev - not used*/,
                1 /* packingFactor - not used */,
                1 /* reductionLevel */);
}

void RandomSimTensorGenerator::duplicate(pMMESimTensor& srcTensor, pMMESimTensor& dstTensor)
{
    uint8_t* srcData = (uint8_t*) srcTensor->data();
    uint8_t* dstData = (uint8_t*) dstTensor->data();
    uint64_t numBytes = srcTensor->getMemorySize();
    MME_ASSERT(srcTensor->getMemorySize() == dstTensor->getMemorySize(), "Cannot duplicate tensors of different sizes");
    memcpy(dstData, srcData, numBytes);
}

template<>
void RandomSimTensorGenerator::getValue<fp8_152_t>(float floatVal,
                                                   fp8_152_t* outVal,
                                                   unsigned fpBias,
                                                   MmeCommon::InfNanMode infNanMode)
{
    *outVal = fp8_152_t(floatVal, RoundingMode::RoundToNearest, fpBias, 0, false, false, false, false, infNanMode);
}
template<>
void RandomSimTensorGenerator::getValue<fp8_143_t>(float floatVal,
                                                   fp8_143_t* outVal,
                                                   unsigned fpBias,
                                                   MmeCommon::InfNanMode infNanMode)
{
    *outVal = fp8_143_t(floatVal, RoundingMode::RoundToNearest, fpBias, 0, false, false, false, false, infNanMode);
}
template<>
void RandomSimTensorGenerator::getValue<fp16_t>(float floatVal,
                                                fp16_t* outVal,
                                                unsigned fpBias,
                                                MmeCommon::InfNanMode infNanMode)
{
    *outVal = fp16_t(floatVal, RoundingMode::RoundToNearest, fpBias, 0, false, false, false, infNanMode);
}

template<>
float RandomSimTensorGenerator::getMinVal<fp8_152_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp8_152_t::min(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getMinVal<fp8_143_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp8_143_t::min(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getMinVal<fp16_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp16_t::min(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getMaxVal<fp8_152_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp8_152_t::max(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getMaxVal<fp8_143_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp8_143_t::max(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getMaxVal<fp16_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp16_t::max(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getLowestVal<fp8_152_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp8_152_t::lowest(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getLowestVal<fp8_143_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp8_143_t::lowest(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
template<>
float RandomSimTensorGenerator::getLowestVal<fp16_t>(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
{
    return fp16_t::lowest(fpBias, infNanMode).toFloat(fpBias, infNanMode);
}
CommonRefMatrix& CommonRefMatrix::operator=(const CommonRefMatrix& other)
{
    if (&other != this)
    {
        shape = other.shape;
        data = other.data;
        sizeOfDataType = other.sizeOfDataType;
    }
    return *this;
}

CommonRefMatrix::CommonRefMatrix(const CommonRefMatrix& other)
{
    shape = other.shape;
    data = other.data;
    sizeOfDataType = other.sizeOfDataType;
}
