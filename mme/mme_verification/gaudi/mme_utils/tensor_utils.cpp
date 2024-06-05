
#include "gaudi/headers/tensor_utils.h"
#include "convolution_params.h"
#include "sim_tensor.h"
#include "mme_assert.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <stdlib.h>

#ifdef WIN32
#include <windows.h>
#undef min
#undef max
#endif

using namespace MmeCommon;

static MmeSimTensor::f32_t getRandomf32(MmeSimTensor::f32_t range)
{
    union
    {
        uint8_t b[4];
        MmeSimTensor::f32_t f32;
    } num;

    for (int i = 0; i < sizeof(num); i++)
    {
        num.b[i] = rand();
    }

    if (num.f32 < 0)
    {
        num.f32 = -num.f32;
    }

    return (num.f32 % range);
}

static MmeSimTensor::bf16_t getRandombf16(MmeSimTensor::f32_t range)
{
    union
    {
        uint8_t b[2];
        MmeSimTensor::bf16_t bf16;
    } num;

    for (int i = 0; i < sizeof(num); i++)
    {
        num.b[i] = rand();
    }

    if (num.bf16 < 0)
    {
        num.bf16 = -num.bf16;
    }

    return (num.bf16 % range);
}

static void genRandomValues_int(MmeSimTensor* t, MmeSimTensor::f32_t minValue, MmeSimTensor::f32_t maxValue)
{
    if (!t->getSizeInElements()) return;

    int idx[MmeSimTensor::c_tensorMaxDim] = {0};
    int dim = -1;

    MmeSimTensor::f32_t range = maxValue - minValue + 1;

    while (dim < t->getDim())
    {
        if (dim < 0)
        {
            switch (t->getElementType())
            {
                case e_type_fp32:
                {
                    MmeSimTensor::f32_t val = getRandomf32(range) + minValue;
                    *(MmeSimTensor::f32_t*) (t->getElementAt(idx)) = val;
                    break;
                }
                case e_type_bf16:
                {
                    MmeSimTensor::bf16_t val = getRandombf16(range) + minValue;
                    *(MmeSimTensor::bf16_t*) (t->getElementAt(idx)) = val;
                    break;
                }
                default:
                    MME_ASSERT(0, "invalid data type");
            }

            dim = 0;
        }
        else
        {
            idx[dim]++;
            if (idx[dim] >= t->getSize(dim))
            {
                idx[dim] = 0;
                dim++;
            }
            else
            {
                dim = -1;
            }
        }
    }
}

union bigNum_t
{
    uint64_t ddw[2];
    uint8_t b[16];
};

static void genRandom_bigNum(bigNum_t* num)
{
    for (unsigned i = 0; i < sizeof(num->b); i++)
    {
        num->b[i] = rand();
    }
    num->b[15] &= 0x3f;
}

static double bigNum2double(const bigNum_t* num)
{
    const double maxU64 = (double) UINT64_MAX + 1.0;
    double ret = num->ddw[0] + (num->ddw[1] * maxU64);
    ret /= (double) UINT64_MAX + ((double) UINT64_MAX / 4 * maxU64);
    return ret;
}

static float scaleDouble(const double num, const float d, const float u)
{
    double ret = ((u - d) * num) + d;
    return ret;
}

static void
genRandomValues_float(MmeSimTensor* t, MmeSimTensor::f32_t minValue, MmeSimTensor::f32_t maxValue, unsigned scale)
{
    if (!t->getSizeInElements()) return;

    int idx[MmeSimTensor::c_tensorMaxDim] = {0};
    int dim = -1;

    while (dim < t->getDim())
    {
        if (dim < 0)
        {
            bigNum_t numi;
            genRandom_bigNum(&numi);
            double numd = bigNum2double(&numi);

            union
            {
                float f;
                MmeSimTensor::f32_t tf32;
                MmeSimTensor::bf16_t tf16[2];
            } numf;

            numf.f = scaleDouble(numd, *(float*) &minValue, *(float*) &maxValue);

            switch (t->getElementType())
            {
                case e_type_fp32:
                {
                    *(MmeSimTensor::f32_t*) (t->getElementAt(idx)) = numf.tf32;
                    break;
                }
                case e_type_bf16:
                {
                    *(MmeSimTensor::bf16_t*) (t->getElementAt(idx)) = numf.tf16[1];
                    break;
                }
                default:
                    MME_ASSERT(0, "invalid data type");
            }

            dim = 0;
        }
        else
        {
            idx[dim]++;
            if (idx[dim] >= t->getSize(dim))
            {
                idx[dim] = 0;
                dim++;
            }
            else
            {
                dim = -1;
            }
        }
    }
}

void genRandomValues(MmeSimTensor* t,
                     MmeSimTensor::f32_t minValue,
                     MmeSimTensor::f32_t maxValue,
                     bool fp,
                     unsigned scale)
{
    if (fp)
    {
        genRandomValues_float(t, minValue, maxValue, scale);
    }
    else
    {
        genRandomValues_int(t, minValue, maxValue);
    }
}

bool transposeDims(int dim1, int dim2, SizeArray strides, SizeArray sizes)
{
    int stride = strides[dim1];
    int size = sizes[dim1];
    strides[dim1] = strides[dim2];
    sizes[dim1] = sizes[dim2];
    strides[dim2] = stride;
    sizes[dim2] = size;
    return true;
}
