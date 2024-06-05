#pragma once
#include "gtest/gtest.h"
#include "habana_global_conf.h"

static float    pearsonThreshold     = 0.95;
static float    epsilonAbsoluteError = 1e-3;
static float    l2NormMinRatio       = 1.0 - 0.2;
static unsigned maxLenAbsError       = 1;
static unsigned maxLenAverageVsStdev = 15;
static float    monotonicRange       = 1e-3;

template<typename TensorDataType>

static bool isMonotonic(TensorDataType values[], uint64_t length)
{
    if (length == 0)
    {
        return false;
    }
    float minimum = values[0];
    float maximum = values[0];

    for (uint64_t i = 0; i < length; i++)
    {
        float convertedValueI = (float)values[i];
        if (convertedValueI < minimum)
        {
            minimum = convertedValueI;
        }
        if (convertedValueI > maximum)
        {
            maximum = convertedValueI;
        }
    }

    return std::abs(maximum - minimum) < monotonicRange;
}

template<typename TensorDataType>
static float calculateL2Norm(TensorDataType values[], uint64_t length)
{
    float sumSquared = 0;
    for (uint64_t idx = 0; idx < length; idx++)
    {
        sumSquared += (float)values[idx] * (float)values[idx];
    }
    return sqrt(sumSquared);
};

template<typename TensorDataType, typename ReferenceDataType = float>
static void
validateL2Norm(ReferenceDataType expected[], TensorDataType result[], uint64_t length, std::string tensorName)
{
    float l2NormExpected = calculateL2Norm(expected, length);
    float l2NormResult   = calculateL2Norm(result, length);

    if (l2NormExpected == 0 && l2NormResult == 0)
    {
        return;
    }

    if (l2NormExpected > l2NormResult)
    {
        std::swap(l2NormExpected, l2NormResult);
    }

    if (GCFG_DEBUG_MODE.value() > 0)
    {
        if (l2NormMinRatio > l2NormExpected / l2NormResult)
        {
            std::cout << "RESNET-DBG: incorrect result l2norm: " << tensorName << ": " << l2NormMinRatio << ", "
                      <<  l2NormExpected / l2NormResult << std::endl;
        }
        else
        {
            std::cout << "RESNET-DBG: good result l2norm: " << tensorName << ": " << l2NormMinRatio << ", "
                      <<  l2NormExpected / l2NormResult << std::endl;
        }
    }
    else
    {
        ASSERT_LE(l2NormMinRatio, l2NormExpected / l2NormResult) << " incorrect result for tensor " << tensorName;
    }
}

template<typename TensorDataType, typename ReferenceDataType = float>
static void
validatePearson(ReferenceDataType expected[], TensorDataType result[], uint64_t length, std::string tensorName)
{
    double sumResult = 0, sumExpected = 0, sumResultXSumExpected = 0;
    double squareSumResult = 0, squareSumExpected = 0;

    for (uint64_t i = 0; i < length; i++)
    {
        sumResult = sumResult + (double)result[i];

        sumExpected = sumExpected + (double)expected[i];

        sumResultXSumExpected = sumResultXSumExpected + (double)result[i] * (double)expected[i];

        squareSumResult   = squareSumResult + (double)result[i] * (double)result[i];
        squareSumExpected = squareSumExpected + (double)expected[i] * (double)expected[i];
    }

    // If all values are zeros correlation is 1.
    if (squareSumResult == 0 && squareSumExpected == 0)
    {
        return;
    }
    else
    {
        // use formula for calculating correlation coefficient.
        double corr = (double)(length * sumResultXSumExpected - sumResult * sumExpected) /
                      sqrt((length * squareSumResult - sumResult * sumResult) *
                           (length * squareSumExpected - sumExpected * sumExpected));

        if (GCFG_DEBUG_MODE.value() > 0)
        {
            if (pearsonThreshold > corr)
            {
                std::cout << "RESNET-DBG: incorrect result pearson: " << tensorName << ": " << pearsonThreshold
                          << ", " <<  corr << std::endl;
            }
            else
            {
                std::cout << "RESNET-DBG: good result pearson: " << tensorName << ": " << pearsonThreshold << ", "
                          <<  corr << std::endl;
            }
        }
        else
        {
            ASSERT_LE(pearsonThreshold, corr) << " incorrect result for tensor " << tensorName;
        }
    }
}

template<typename TensorDataType, typename ReferenceDataType = float>
static void
validateAvgErrorVsStdev(ReferenceDataType expected[], TensorDataType result[], uint64_t length, std::string tensorName)
{
    float var     = 0.0f;
    float mean    = 0.0f;
    float avg_err = 0.0f;

    for (uint64_t idx = 0; idx < length; idx++)
    {
        float ref = expected[idx];
        float x   = result[idx];
        var += ref * ref / length;
        mean += ref / length;
        avg_err += std::abs(ref - x) / ((float)length);
    }
    var -= mean * mean;
    ASSERT_LE(avg_err, std::sqrt(var) * 0.5) << " incorrect result for tensor " << tensorName;
};

template<typename TensorDataType, typename ReferenceDataType = float>
static void validateAbsoluteErrorVsEpsilon(ReferenceDataType expected[],
                                           TensorDataType    result[],
                                           uint64_t          length,
                                           std::string       tensorName)
{
    for (uint64_t idx = 0; idx < length; idx++)
    {
        float ref = expected[idx];
        float x   = result[idx];
        ASSERT_LE(std::abs(ref - x), epsilonAbsoluteError) << " incorrect result for tensor " << tensorName;
    }
};

template<typename TensorDataType, typename ReferenceDataType = float>
void validateResult(ReferenceDataType expected[], TensorDataType result[], uint64_t length, std::string tensorName = "")
{
    if (length <= maxLenAbsError || isMonotonic(expected, length))
    {
        validateAbsoluteErrorVsEpsilon(expected, result, length, tensorName);
    }
    else if (length <= maxLenAverageVsStdev)
    {
        validateAvgErrorVsStdev(expected, result, length, tensorName);
    }
    else
    {
        validatePearson(expected, result, length, tensorName);
        validateL2Norm(expected, result, length, tensorName);
    }
}