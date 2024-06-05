#include <ctgmath>
#include <cassert>
#include "mme_reference/data_types/bfloat16.h"
#include <float.h>
#include "cpu_calculator.h"

int32_t satRndDblHighMul(int32_t a, int32_t b)
{
    assert(b > 0);
    int64_t a_64(a);
    int64_t b_64(b);
    int64_t ab_64 = a_64 * b_64;
    ab_64 += (ab_64 & (1 << 30));
    return (int32_t)(ab_64 >> 31);
}

int32_t rndDivByPOT(int32_t x, int exponent)
{
    assert(exponent >= 0);
    assert(exponent <= 31);
    const int32_t mask = (1ll << exponent) - 1;
    const int32_t remainder = x & mask;
    int32_t threshold = mask >> 1;
    threshold += (x < 0) ? 1 : 0;
    int32_t res = x >> exponent;
    res += (remainder > threshold) ? 1 : 0;
    return res;
}

int32_t satRndMulByPOT(int32_t x, int exponent)
{
    int64_t res = (x < 0) ? -(((int64_t)-x) << exponent) : ((int64_t)x) << exponent;
    return saturate<int32_t>(res);
}

int64_t scaleCIn(int64_t val, double scaleX, double scaleW, double scaleCIn)
{
    HB_ASSERT(scaleX && scaleW, "cannot scale cin with zero scaleX or scaleW");
    double scaleFactor = scaleCIn / (scaleX * scaleW);
    return shiftAndScale(val, scaleFactor);
}

/*
 * Float saturate.
 * Saturates small numbers to 0 like HW default.
 */
template<>
float saturate<float>(float in)
{
    if (in > 0)
    {
        if (in > std::numeric_limits<float>::max())
        {
            return std::numeric_limits<float>::max();
        }
        if (in < std::numeric_limits<float>::min())
        {
            return 0;
        }
    }
    else
    {
        if (in > -std::numeric_limits<float>::min())
        {
            return 0;
        }
        if (in < -std::numeric_limits<float>::max())
        {
            return -std::numeric_limits<float>::max();
        }
    }

    return in;
}

template<>
bfloat16 saturate<bfloat16, long int>(long int in)
{
    if (in > bfloat16::max())
    {
        return bfloat16::max();
    }

    if (in < bfloat16::min())
    {
        return bfloat16::min();
    }

    return bfloat16((float)in);
}


static const int32_t FLOAT_PRECISION = 1000;

template<>
int64_t getIntRep(float value)
{
    return round(value * FLOAT_PRECISION);
}

template<>
float getRepFromInt(int64_t value)
{
    return (float)value / FLOAT_PRECISION;
}
