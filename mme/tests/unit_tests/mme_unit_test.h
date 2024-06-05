#pragma once

#include <gtest/gtest.h>
#include "include/mme_common/mme_common_enum.h"
#include "sim_tensor.h"
#include "data_types/non_standard_dtypes.h"
#include "mme_assert.h"

class MMEUnitTest : public ::testing::Test
{
public:
    MMEUnitTest() = default;
    virtual ~MMEUnitTest() = default;
protected:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

class MmeUTDataTypeConversionTest : public MMEUnitTest
{
};

typedef ::testing::Types<fp32_t, bf16_t, tf32_t, fp16_t, fp8_152_t, fp8_143_t> Gaudi2Types;

template<typename T>
class MmeUTRandomTensorTest : public MMEUnitTest
{
protected:
    MmeCommon::EMmeDataType getTensorType()
    {
        if (std::is_same<T, float>() || std::is_same<T, fp32_t>())
            return MmeCommon::EMmeDataType::e_type_fp32;
        else if (std::is_same<T, bf16_t>())
            return MmeCommon::EMmeDataType::e_type_bf16;
        else if (std::is_same<T, fp16_t>())
            return MmeCommon::EMmeDataType::e_type_fp16;
        else if (std::is_same<T, ufp16_t>())
            return MmeCommon::EMmeDataType::e_type_ufp16;
        else if (std::is_same<T, tf32_t>())
            return MmeCommon::EMmeDataType::e_type_tf32;
        else if (std::is_same<T, fp8_152_t>())
            return MmeCommon::EMmeDataType::e_type_fp8_152;
        else if (std::is_same<T, fp8_143_t>())
            return MmeCommon::EMmeDataType::e_type_fp8_143;
        else
            MME_ASSERT(0, "invalid data type");
        return MmeCommon::EMmeDataType::e_type_bf16;
    }

    unsigned getDefaultBias()
    {
        if (std::is_same<T, fp16_t>())
            return 15;
        else if (std::is_same<T, ufp16_t>())
            return 31;
        else if (std::is_same<T, fp8_152_t>())
            return 15;
        else if (std::is_same<T, fp8_143_t>())
            return 7;
        else
            return 0;
    }
};


class MmeUTReferenceTest : public MMEUnitTest
{
public:
    MmeSimTensor createSimTensor(MmeCommon::SizeArray size, unsigned dim, MmeCommon::EMmeDataType type)
    {
        for (unsigned i = 0; i < MAX_DIMENSION; ++i)
        {
            if (size[i] == 0) size[i] = 1;
        }
        return {size, dim, type};
    }
    template<typename T>
    void fillSimTensorWithData(T val, unsigned sizeInElements, MmeSimTensor& tensor, unsigned expBias = 0)
    {
        DataBuffer data(sizeInElements * sizeof(T));
        for (unsigned i = 0; i < sizeInElements; i++)
        {
            memcpy((T*) data + i, &val, sizeof(T));
        }
        tensor.setData(data);
        if (tensor.getElementType() == MmeCommon::e_type_fp8_152 || tensor.getElementType() == MmeCommon::e_type_fp8_143)
        {
            tensor.setFpBias(expBias);
        }
    }
};
