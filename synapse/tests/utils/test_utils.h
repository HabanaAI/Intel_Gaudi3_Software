#pragma once

#include "infra/cpu_calculator.h"

#include "log_manager.h"
#include "mme_reference/data_types/fp8.h"
#include "test_types.hpp"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <stdlib.h>
#include <type_traits>
#include <vector>

#define UNUSED(x) ((void)x)

std::string deviceTypeToString(const synDeviceType& deviceType);

typedef std::vector<synQuantizationParams> QuantParamsVector;

bool float_eq(float a, float b, float eps=0.001f);

void init_3d_data(float* p, unsigned W, unsigned H, unsigned D);

// Note that uniform_int_distribution generates random numbers on closed interval [x,y] whereas
// uniform_real_distribution generates numbers on half-open interval [x,y) so the end-point will
// not be included. Therefore, the uniform_distribution template is a bit missleading since
// depending on its concrete type, the end-point may be included or not.
// *** For our testing purposes this should not make a difference and thus is acceptable ***
template <typename T>
using uniform_distribution =
    typename std::conditional<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>
    >::type;

template<typename T>
inline bool isValidValue(T value)
{
    return true;
}

inline bool isValidValue(float value)
{
    return std::isnormal(value) || value == 0;
}

inline bool isValidValue(double value)
{
    return std::isnormal(value) || value == 0;
}

inline bool isValidValue(fp8_152_t value)
{
    if (value.isInf() || value.isNan())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template<typename t, typename dist_type = t>
inline void
fillWithRandom(std::default_random_engine& generator, t* p, uint64_t size, std::pair<dist_type, dist_type> minMax)
{
    if (minMax.first > minMax.second)
    {
        std::swap(minMax.first, minMax.second);
    }

    uniform_distribution<dist_type>     distribution(minMax.first, minMax.second);

    for (uint64_t i = 0; i < size; ++i)
    {
        do{
            p[i] = t(distribution(generator));
        } while (!isValidValue(p[i]));
    }
}

template<typename t, typename dist_type = t>
inline void fillWithRandom(t* p, uint64_t size, std::pair<dist_type, dist_type> minMax)
{
    std::default_random_engine generator;  // NOLINT(cert-msc32-c,cert-msc51-cpp) - deterministic on purpose
    fillWithRandom(generator, p, size, minMax);
}

/*
 * usefull for radnom data for tensor that needs to be quantized with default quant info (zp=0 , scale=1)
 */
void fillWithRandomRoundedFloats(float* p, uint64_t size, std::pair<float, float> minMax);

template<typename T>
inline void fillWithRandom(T* p, uint64_t size)
{
    fillWithRandom(p, size, {0, std::numeric_limits<T>::max()});
}

template<>
inline void fillWithRandom<float>(float* p, uint64_t size)
{
    fillWithRandom(p, size, {0, 1});
}

template<typename T>
inline void fillWithRandomNeg(T* p, uint64_t size)
{
    fillWithRandom(p, size, {std::numeric_limits<T>::min(), std::numeric_limits<T>::max()});
}

void fillWithRandom(void* p, uint64_t size, synDataType dataType);

template<typename T>
inline T rand_in_interval(T min, T max)
{
    T value;
    do{
        value = (T)(((double)rand() / ((double)RAND_MAX + 1) * (max - min)) + min);
    } while(!isValidValue(value));
    return value;
}

// fill buffer with running numbers from minVal to maxVal-1
// multiplier - multiply the result by this
// offset - add offset to the result
// negativesFreq - fill with negative numbers each negativeFreq elements
template<typename T>
void fillBufferWithRunningNumbers(T* buffer, int elementCnt, int maxVal = INT_MAX, double multiplier = 1,
                                         int offset = 0, int negativesFreq = 0 /* 0 = no negatives*/)
{
    for (int i = 0; i < elementCnt; ++i)
    {
        if (negativesFreq && (i % negativesFreq == negativesFreq - 1))
        {
            buffer[i] = -((i % maxVal * multiplier) + offset);
        }
        else
        {
            buffer[i] = (i % maxVal * multiplier) + offset;
        }
    }

}

class Test_Random_Number_Creator
{
public:
    Test_Random_Number_Creator (const std::array<int,2>& range)
    : m_range (range)
    {}

    int operator () ()
    {
        int distanceFromMin = m_range[MAX_RANGE_POS] - m_range[MIN_RANGE_POS] + 1;

        return (rand()) % distanceFromMin + m_range[MIN_RANGE_POS];
    }

private:
    const std::array<int,2> m_range;

    static const int MIN_RANGE_POS = 0;
    static const int MAX_RANGE_POS = 1;
};

std::vector<char*> AllocateMemory(std::vector<uint64_t> vec);
void FreeMemory(std::vector<char*> vec);

template<class T>
T* generateValuesArray(uint32_t array_size,
                       const std::array<int,2>& range)
{
    T* arr = new T[array_size];
    std::generate (arr, arr + array_size, Test_Random_Number_Creator (range));

    return arr;
}

void* generateValuesArray(uint32_t array_size,
                          synDataType array_type,
                          const std::array<int,2>& range);

NodePtr getConvNodeWithGoyaLayouts(const TensorPtr& IFM, const TensorPtr& weights, const TensorPtr& bias, const TensorPtr& OFM,
                                   const synConvolutionParams& params, const std::string& name);

NodePtr getConvPlusNodeWithGoyaLayouts(const TensorPtr& IFM, const TensorPtr& weights, const TensorPtr& bias, const TensorPtr& OFM,
                                       const TensorPtr& cin, const synConvolutionParams& params, const std::string& name);

uint32_t getElementSizeInBytes(synDataType type);

// TODO:  Move index to TOffset array [SW-117362]
template<class T>
T getValueFromBuffer(const TSize* sizes, const int* index, int dim, void* buffer)
{
    TOffset indexTemp[SYN_MAX_TENSOR_DIM];
    castNcopy(indexTemp, index, SYN_MAX_TENSOR_DIM);
    TSize absSize = calcAbsSize(sizes, indexTemp, dim);

    char* ptr = (char*)buffer + absSize * sizeof(T);

    return *(T*)ptr;
}

template<typename T>
void printTensor(const unsigned int* tensor_dim, const T* tensor)
{
    const unsigned int seperatorSize = tensor_dim[TPD_Height] * tensor_dim[TPD_Width] * tensor_dim[TPD_Channel] + 2 * (tensor_dim[TPD_Channel] - 1);
    std::ostringstream seperator;
    seperator << std::setw(seperatorSize) << std::setfill('-');

    for (int batch = 0; batch < tensor_dim[TPD_4Dim_Batch]; ++batch)
    {
        int next_batch = batch * tensor_dim[TPD_Width] * tensor_dim[TPD_Height]*tensor_dim[TPD_Channel];
        for (int height = 0; height < tensor_dim[TPD_Height]; ++height)
        {
            std::ostringstream output;
            for (int channel = 0 ; channel < tensor_dim[TPD_Channel]; ++channel)
            {
                if (channel > 0)
                {
                    output << " |";
                }

                for (int width = 0; width < tensor_dim[TPD_Width]; ++width)
                {
                    int next_index = channel + width *tensor_dim[TPD_Channel] + height*tensor_dim[TPD_Width] * tensor_dim[TPD_Channel] + next_batch;
                    output << std::setw(4) << static_cast<int>(tensor[next_index]) << " " ;
                }
            }
            LOG_DEBUG(SYN_TEST, "{}", output.str ());
        }
        LOG_DEBUG(SYN_TEST,"{}", seperator.str());
    }
}

float dequantize(int8_t value, double zp, double scale);
float dequantizeQuantize(float value, double scaleIn, double scaleOut, double zpIn, double zpOut);
float roundHalfAwayFromZero(float value);
float clamp(synDataType dType, float value);

template<typename T>
T requant(const T value, const QuantParamsVector& paramsIn, const QuantParamsVector& paramsOut)
{
    HB_ASSERT(paramsIn.size() == 1 && paramsOut.size() == 1, "Only global quantization is supported");
    float interVal = static_cast<float>(value);
    interVal = dequantizeQuantize(interVal, paramsIn[0].m_scale, paramsOut[0].m_scale, paramsIn[0].m_zp, paramsOut[0].m_zp);
    interVal = roundHalfAwayFromZero(interVal);
    interVal = clamp(paramsOut[0].m_qDataType, interVal);
    return static_cast<T>(interVal);
}

std::array<unsigned, (uint32_t)synapse::LogManager::LogType::LOG_MAX> testsEnableAllSynLogs(int newLevel);
void testRecoverSynLogsLevel(const std::array<unsigned, (uint32_t)synapse::LogManager::LogType::LOG_MAX>& prev);

void setTensorAsPersistent(TensorPtr& tensor, unsigned tensorsCount = 0);

uint64_t getNumberOfElements(const unsigned* sizes, unsigned dims);
uint64_t getNumberOfElements(const TSize* sizes, unsigned dims);

void randomBufferValues(MemInitType initSelect,
                        synDataType type,
                        uint64_t    size,
                        void*       output,
                        bool        isDefaultGenerator = true);

#include "test_utils.inl"