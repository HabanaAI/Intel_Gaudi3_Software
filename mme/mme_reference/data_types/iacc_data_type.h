#pragma once

#include "include/general_utils.h"

template<int64_t MIN_VAL, int64_t MAX_VAL, bool SATURATE>
struct iacc32_t
{
    int32_t v;

    iacc32_t<MIN_VAL, MAX_VAL, SATURATE>() : v(0) {}

    iacc32_t<MIN_VAL, MAX_VAL, SATURATE>(const int64_t val) : v(val)
    {
        if (!SATURATE)
        {
            MME_ASSERT(val >= MIN_VAL, "out of range value");
            MME_ASSERT(val <= MAX_VAL, "out of range value");
        }
        else
        {
            if (val > MAX_VAL) v = MAX_VAL;
            else if (val < MIN_VAL)
                v = MIN_VAL;
        }
    }

    iacc32_t<MIN_VAL, MAX_VAL, SATURATE>(const iacc32_t<MIN_VAL, MAX_VAL, SATURATE>& other) : v(other.v)
    {
        if (!SATURATE)
        {
            MME_ASSERT(v >= MIN_VAL, "out of range value");
            MME_ASSERT(v <= MAX_VAL, "out of range value");
        }
        else
        {
            if (v > MAX_VAL) v = MAX_VAL;
            else if (v < MIN_VAL)
                v = MIN_VAL;
        }
    }

    int64_t operator+(const int64_t rhs) const { return rhs + v; }

    iacc32_t<MIN_VAL, MAX_VAL, SATURATE> operator+(const iacc32_t<MIN_VAL, MAX_VAL, SATURATE>& rhs) const
    {
        iacc32_t<MIN_VAL, MAX_VAL, SATURATE> ret;
        ret = (int64_t) v + (int64_t) rhs.v;
        return ret;
    }

    void operator=(const int64_t rhs)
    {
        int32_t newVal = rhs;
        if (!SATURATE)
        {
            MME_ASSERT(rhs >= MIN_VAL, "out of range value");
            MME_ASSERT(rhs <= MAX_VAL, "out of range value");
        }
        else
        {
            if (rhs > MAX_VAL) newVal = MAX_VAL;
            else if (rhs < MIN_VAL)
                newVal = MIN_VAL;
        }

        v = newVal;
    }

    void operator=(const iacc32_t<MIN_VAL, MAX_VAL, SATURATE>& rhs)
    {
        int32_t newVal = rhs.v;
        if (!SATURATE)
        {
            MME_ASSERT(rhs.v >= MIN_VAL, "out of range value");
            MME_ASSERT(rhs.v <= MAX_VAL, "out of range value");
        }
        else
        {
            if (rhs.v > MAX_VAL) newVal = MAX_VAL;
            else if (rhs.v < MIN_VAL)
                newVal = MIN_VAL;
        }

        v = newVal;
    }

    void operator+=(const iacc32_t<MIN_VAL, MAX_VAL, SATURATE>& rhs)
    {
        int64_t tmp = (int64_t) v + (int64_t) rhs.v;
        if (!SATURATE)
        {
            MME_ASSERT(tmp >= MIN_VAL, "out of range value");
            MME_ASSERT(tmp <= MAX_VAL, "out of range value");
        }
        else
        {
            if (rhs.v > MAX_VAL) tmp = MAX_VAL;
            else if (rhs.v < MIN_VAL)
                tmp = MIN_VAL;
        }

        v = tmp;
    }

    void operator+=(const int64_t rhs)
    {
        int64_t tmp = (int64_t) v + rhs;

        if (!SATURATE)
        {
            MME_ASSERT(tmp >= MIN_VAL, "out of range value");
            MME_ASSERT(tmp <= MAX_VAL, "out of range value");
        }
        else
        {
            if (tmp > MAX_VAL) tmp = MAX_VAL;
            else if (tmp < MIN_VAL)
                tmp = MIN_VAL;
        }

        v = tmp;
    }

    void operator-=(const int64_t rhs)
    {
        int64_t tmp = (int64_t) v - rhs;

        if (!SATURATE)
        {
            MME_ASSERT(tmp >= MIN_VAL, "out of range value");
            MME_ASSERT(tmp <= MAX_VAL, "out of range value");
        }
        else
        {
            if (tmp > MAX_VAL) tmp = MAX_VAL;
            else if (tmp < MIN_VAL)
                tmp = MIN_VAL;
        }

        v = tmp;
    }
};

static const int64_t INT26_MAX = (1 << 25) - 1;
static const int64_t INT26_MIN = ~INT26_MAX;
typedef iacc32_t<INT16_MIN, INT16_MAX, false> iacc32_16_t;
typedef iacc32_t<INT26_MIN, INT26_MAX, false> iacc32_26_t;
typedef iacc32_t<INT32_MIN, INT32_MAX, false> iacc32_32_t;
typedef iacc32_t<INT32_MIN, INT32_MAX, true> iacc32_32_saturate_t;

static_assert(sizeof(iacc32_32_t) == sizeof(uint32_t), "");
