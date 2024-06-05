#ifndef MME__GENERAL_UTILS_H
#define MME__GENERAL_UTILS_H

#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>
#include "include/mme_assert.h"

//  Bgemm Dims

namespace MmeCommon
{
#define MAX_DIMENSION 5
using SizeArray = std::array<unsigned, MAX_DIMENSION>;

// dimension defines -
typedef enum
{
    //  conv input dims
    DIM_C = 0,
    DIM_W = 1,
    DIM_H = 2,
    DIM_D = 3,
    DIM_B = 4,
    //  conv Weight dims
    DIM_Q = 4,
    DIM_R = 3,
    DIM_S = 2,
    WEIGHT_DIM_C = 1,
    DIM_K = 0,
    //  Bgemm dims
    GEMM_DIM_B3 = 4,
    GEMM_DIM_B2 = 3,
    GEMM_DIM_B1 = 2,
    GEMM_DIM_H = 1,
    GEMM_DIM_W = 0,
    //  general definitions
    FIRST_SP_DIM = 1,
} MmeDimsIndex;

typedef enum {
    MASKED_BGEMM_A,
    MASKED_BGEMM_B,
    CD_SCRATCHPAD,
    CD_REDUCTION,
    AUX_TENSOR_MAX_NUM
} MmeAuxTensorIdx;

// util functions copied from synapse
inline bool isPowerOf2(uint64_t num)
{
    if (num == 0) return false;
    return (num & (num - 1)) == 0;
}

inline uint64_t div_round_up(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

inline int64_t div_round_down(int64_t n, int64_t d)
{
    return ((n >= 0) ? ((n) / (d)) : (((n) - (d) + 1) / (d)));
}

inline int64_t mod_neg(int64_t n, int64_t d)
{
    return (((d) + ((n) % (d))) % (d));
}

inline uint64_t round_to_multiple(uint64_t a, uint64_t mul)
{
    return mul * div_round_up(a, mul);
}

inline uint64_t round_down_to_multiple(uint64_t a, uint64_t mul)
{
    return mul * div_round_down(a, mul);
}

inline unsigned alignToVal_NonPowerOf2(unsigned num, unsigned alignment)
{
    return div_round_up(num, alignment) * alignment;
}

inline unsigned alignToVal(unsigned num, unsigned alignment)
{
    MME_ASSERT(isPowerOf2(alignment), "can only align to power of 2");
    return (num + alignment - 1) & ~(alignment - 1);
}

template<typename ITER_TYPE>
uint64_t multiplyElements(ITER_TYPE begin, ITER_TYPE end)
{
    return std::accumulate(begin,
                           end,
                           (uint64_t) 1,  // acc
                           [](uint64_t acc, uint64_t val) { return acc * val; });
}

// bypass debian8.9 error with memset
template<typename T>
inline void setValue(T* dest, int val, int sizeInElements)
{
    for (unsigned i = 0; i < sizeInElements; ++i)
    {
        dest[i] = val;
    }
}

// Return 'true' if the given array has at least one zero value in it or 'false' otherwise.
// 'start' is the first index where the search starts.
// 'end' is the last (included) index where the search ends.
inline bool hasZeros(const SizeArray& arr, unsigned start = 0, unsigned end = MAX_DIMENSION - 1)
{
    MME_ASSERT((start < MAX_DIMENSION) && (end < MAX_DIMENSION), "invalid start\\end dims");
    for (unsigned i = start; i <= end; i++)
    {
        if (arr[i] == 0)
        {
            return true;
        }
    }
    return false;
}

// Return true if all values of an array are equal to a given value
template<class T>
bool allValuesEqualTo(const T* a, unsigned size, unsigned v)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != v)
        {
            return false;
        }
    }
    return true;
}

// Produce a size array which has n at index 0, the rest indices are padded with p
inline SizeArray uint2SizeArr(unsigned n, unsigned p)
{
    SizeArray arr = {0};
    arr[0] = n;
    std::fill(arr.begin() + 1, arr.end(), p);
    return arr;
}

// When slicer doesn't split the tensor into multiple subviews, strides become contiguous.
// This simple function calculates them.
inline void calcContiguousStrides(SizeArray& strides, const SizeArray& sizes)
{
    strides[0] = 1;
    for (unsigned dim = 0; dim < MAX_DIMENSION - 1; dim++)
    {
        strides[dim + 1] = strides[dim] * sizes[dim];
    }
}

}  // namespace MmeCommon

/* reinterpretation tools */
template<typename T, typename U>
inline T& reinterpret_ptr(U* ptr)
{
    static_assert(std::is_const<U>() == std::is_const<T>(), "cannot add\\remove const");
    return *reinterpret_cast<T*>(ptr);
}

template<typename T, typename U>
inline T& reinterpret_ptr_with_index(U* ptr, unsigned i)
{
    static_assert(std::is_const<U>() == std::is_const<T>(), "cannot add\\remove const");
    return *(reinterpret_cast<T*>(ptr) + i);
}

inline std::string arrayToStr(const unsigned* a, unsigned size)
{
    std::string str;
    for (int i = 0; i < size; i++)
    {
        str += std::to_string(a[i]);
        if (i < (size - 1))
        {
            str += ", ";
        }
    }
    return str;
}

template<typename ITER>
inline std::string arrayToStr(ITER begin, ITER end)
{
    std::string str;
    for (auto i = begin; i != end; i++)
    {
        str += std::to_string(*i);
        const auto& next = std::next(i);
        if (next != end)
        {
            str += ", ";
        }
    }
    return str;
}

inline std::vector<unsigned> getAllDivisors(unsigned v)
{
    std::vector<unsigned> divisors;
    // run until sqrt(v) and add both the divisor and its complementary
    unsigned maxV = (unsigned) std::sqrt((float) v);
    for (int i = 1; i <= maxV; i++)
    {
        if (v % i == 0)
        {
            divisors.push_back(i);
            if (i != v / i)
            {
                divisors.push_back(v / i);
            }
        }
    }
    return divisors;
}

template <typename T>
inline void assertOrAssign(std::optional<T>& variable, T& value, std::string errMsg)
{
    if (!variable.has_value())
    {
        variable = value;
    }
    else
    {
        MME_ASSERT(*variable == value, errMsg.c_str());
    }
}

inline bool validateTranspose(const MmeCommon::SizeArray& inSizes, const MmeCommon::SizeArray& outSizes)
{
    if (inSizes.at(0) != outSizes.at(1) || inSizes.at(1) != outSizes.at(0))
    {
        return false;
    }
    for (auto dim = 2; dim < MAX_DIMENSION; ++dim)
    {
        if (inSizes.at(dim) != outSizes.at(dim))
        {
            return false;
        }
    }
    return true;
}

#endif //MME__GENERAL_UTILS_H
