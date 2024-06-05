#pragma once

#include <map>
#include <memory>
#include "sim_tensor_base.h"
#include "include/mme_assert.h"

// common mme tensor representation
class MmeSimTensor : public MMESimTensorBase
{
public:
    MmeSimTensor() : MMESimTensorBase() {}
    MmeSimTensor(const MmeCommon::SizeArray& sizes,
                 int dim,
                 MmeCommon::EMmeDataType type,
                 char* data = nullptr,
                 unsigned fpBias = 0,
                 MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan,
                 const MmeCommon::SizeArray* strides = nullptr);
    MmeSimTensor(MMESimTensorBase* other, int* offsets, int dim, const MmeCommon::SizeArray* sizes)
    : MMESimTensorBase(other, offsets, dim, sizes)
    {
    }
    MmeSimTensor(const MMESimTensorBase* other, char* data) : MMESimTensorBase(other)
    {
        checkTypeAndSetSize(other->getElementType());
        uint64_t tensorSize = setStridesAndGetTensorSize(&other->getStrides(), other->getDim());
        setData(tensorSize, data);
    }
    // backward compatibility constructor
    // In gaudi, the default for the alignment was 128. It is now changed to 0.
    // todo AlonG: verify that things work when the default for alignment is 0
    MmeSimTensor(const int* sizes,
                 int dim,
                 MmeCommon::EMmeDataType type,
                 char* data = 0,
                 int* strides = 0,
                 int alignment = 0,
                 const bool shouldCopyData = true);
    MmeSimTensor(MmeSimTensor* other, int* offsets, int dim, int* sizes) : MMESimTensorBase(other, offsets, dim, sizes)
    {
    }
    virtual ~MmeSimTensor() = default;
    void memsetTensor(const char* value);

    typedef int16_t bf16_t;
    typedef int32_t f32_t;

private:
    void checkTypeAndSetSize(MmeCommon::EMmeDataType type);
};

using pMMESimTensor = std::shared_ptr<MmeSimTensor>;

// [0] - height, [1] - width
using MatrixShape = std::array<uint64_t, 2>;
// simple matrix representation.
struct CommonRefMatrix
{
    CommonRefMatrix() = default;
    explicit CommonRefMatrix(unsigned sizeOfDataType) : sizeOfDataType(sizeOfDataType) {}
    CommonRefMatrix(const CommonRefMatrix& other);
    ~CommonRefMatrix() = default;
    CommonRefMatrix& operator=(const CommonRefMatrix& other);
    MatrixShape shape;
    DataBuffer data;
    unsigned sizeOfDataType = 0;
    uint64_t getHeight() const { return shape[0]; }
    uint64_t getWidth() const { return shape[1]; }
    void setShape(uint64_t height, uint64_t width) { shape = {height, width}; }
    byte* getElementAt(uint64_t height, uint64_t width)
    {
        uint64_t offset = height * getWidth() + width;
        assert(offset * sizeOfDataType < data.sizeInBytes());
        return data[offset * sizeOfDataType];
    }
    const byte* getElementAt(uint64_t height, uint64_t width) const
    {
        uint64_t offset = height * getWidth() + width;
        MME_ASSERT(offset * sizeOfDataType < data.sizeInBytes(), "height\\width out of bounds");
        return data[offset * sizeOfDataType];
    }
    const uint64_t getMemorySize() const
    {
        return MmeCommon::multiplyElements(shape.begin(), shape.end()) * sizeOfDataType;
    }

    void doTranspose()
    {
        DataBuffer tposData(data.sizeInBytes());
        for (uint64_t h = 0; h < getHeight(); h++)
        {
            for (uint64_t w = 0; w < getWidth(); w++)
            {
                for (unsigned byte = 0; byte < sizeOfDataType; byte++)
                {
                    uint64_t offset = (w * getHeight() + h) * sizeOfDataType;
                    *tposData[offset + byte] = *(getElementAt(h, w) + byte);
                }
            }
        }
        data = tposData;

        setShape(getWidth(), getHeight());
    }
};

class RandomSimTensorGenerator
{
public:
    enum class Operation
    {
        UniformDistribution,
        NormalDistribution,
        Fill,
        FillForReductionPacking,
        UnitMatrix
    };
    using generateFunc =
        std::function<void(RandomSimTensorGenerator*, pMMESimTensor&, Operation, float, float, float, float, unsigned, unsigned)>;
    using fillReductionPackingFunc =
        std::function<void(RandomSimTensorGenerator*, pMMESimTensor&)>;
    RandomSimTensorGenerator(unsigned int seed);
    ~RandomSimTensorGenerator() = default;
    void generateUniform(pMMESimTensor& tensor, float minValue, float maxValue);
    void generateNormal(pMMESimTensor& tensor, float minValue, float maxValue, float mean, float stdDev);
    void generateUnitMatrix(pMMESimTensor& tensor, float unitValue, float otherValues);
    void fill(pMMESimTensor& tensor, float value);
    void fillForReductionPacking(pMMESimTensor& tensor, unsigned packingFactor, unsigned reductionLevel);
    void duplicate(pMMESimTensor& srcTensor, pMMESimTensor& dstTensor);

private:
    template<typename T>
    void generateInternal(pMMESimTensor& tensor,
                          Operation op,
                          float minValue,
                          float maxValue,
                          float mean = 0,
                          float stdDev = 1,
                          unsigned reductionPacking = 1,
                          unsigned reductionLevel = 1);
    template <class T>
    void fillReductionPackingTensorInternal(pMMESimTensor& tensor, unsigned reductionPacking, unsigned reductionLevel);
    template<typename T>
    void getValue(float floatVal, T* outVal, unsigned fpBias, MmeCommon::InfNanMode infNanMode)
    {
        *outVal = T(floatVal);
    }
    template<typename T>
    float getMinVal(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
    {
        return T::min().toFloat();
    }
    template<typename T>
    float getMaxVal(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
    {
        return T::max().toFloat();
    }
    template<typename T>
    float getLowestVal(unsigned fpBias, MmeCommon::InfNanMode infNanMode)
    {
        return T::lowest().toFloat();
    }
    std::mt19937 m_randomGen;
    std::map<MmeCommon::EMmeDataType, generateFunc> m_funcMap;
};
