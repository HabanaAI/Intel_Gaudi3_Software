#pragma once

#include "include/general_utils.h"

template<typename T>
struct i4x2_t
{
    T i0 : 4;
    T i1 : 4;

    int64_t operator*(const i4x2_t<T>& rhs) const
    {
        return (((int64_t) i0) * ((int64_t) rhs.i0) + ((int64_t) i1) * ((int64_t) rhs.i1));
    }

    int64_t operator+(const int64_t& rhs) const { return rhs + ((int64_t) i0) + ((int64_t) i1); }

    i4x2_t<T>() : i0(0), i1(0) {}
    i4x2_t<T>(int i) : i0(0), i1(0) { MME_ASSERT(i == 0, "int initialization is only allowed for 0"); }
    i4x2_t<T>(const i4x2_t<T>& other) : i0(other.i0), i1(other.i1) {};
    void operator=(const i4x2_t& rhs)
    {
        i0 = rhs.i0;
        i1 = rhs.i1;
    }

    // the following operators are never called.
    void operator=(const int val)
    {
        i0 = 0;
        i1 = 1;
        MME_ASSERT(val == 0, "int assign is only allowed for 0");
    }
    void operator+=(const i4x2_t<T>& rhs)  = delete;

    template<typename OutputT>
    static OutputT fma(const i4x2_t& a, const i4x2_t& b, const OutputT& c, MmeCommon::RoundingMode& rm)
    {
        OutputT ret = (a * b) + c;
        return ret;
    }

    template<typename OutputT>
    static OutputT fma_vec(const i4x2_t* a, const i4x2_t* b, unsigned cdSize, MmeCommon::RoundingMode rm)
    {
        OutputT  c(0);
        for (int i = 0; i < cdSize; i++)
        {
            c = fma(a[i], b[i], c, rm);
        }
        return c;
    }
};

typedef i4x2_t<uint8_t> uint4x2_t;
typedef i4x2_t<int8_t> int4x2_t;
