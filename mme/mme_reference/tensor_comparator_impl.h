#pragma once
#include "tensor_comparator.h"
#include <cmath>
#include <cstdint>

// Compare tensors element by element.
// if there is a difference diffElement & message are set.

class TensorComparatorImpl
{
public:
    explicit TensorComparatorImpl(unsigned CDSize,
                              MmeCommon::EMmeDataType dtType,
                              unsigned expBias = 0,
                              MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan,
                              bool printAllDiffs = false,
                              bool exitOnFailure = false);
    ~TensorComparatorImpl() = default;
    bool doCompare(const std::shared_ptr<MmeSimTensor>& a,
                   const std::string& tensorAName,
                   const std::shared_ptr<MmeSimTensor>& b,
                   const std::string& tensorBName,
                   const unsigned testCounter = 0,
                   const std::string& testInfoStr = "",
                   const unsigned numPartials = 1,
                   const bool failOnMismatch = true);
    bool doCompareBitExact(const std::shared_ptr<MmeSimTensor>& a,
                           const std::string& tensorAName,
                           const std::shared_ptr<MmeSimTensor>& b,
                           const std::string& tensorBName,
                           const unsigned testCounter = 0,
                           const std::string& testInfoStr = "");
    bool doCompare(const Matrix& a, const std::string& matrixAName, const Matrix& b, const std::string& matrixBName);
    bool
    doCompareBitExact(const Matrix& a, const std::string& matrixAName, const Matrix& b, const std::string& matrixBName);

    Settable<MmeCommon::SizeArray> getDiffElement() const;

private:
    void printAndExit(const std::string& tensorAName,
                      const std::string& tensorBName,
                      const unsigned testCounter,
                      const std::string& testInfoStr,
                      const bool failOnMismatch = true);
    template<typename T, typename U>
    bool compare(const U& a, const U& b, unsigned numPartials = 1);

    template<typename T>
    bool compareByType(const T& a, const T& b, unsigned ulpMaxDiff) const;

    template<typename T>
    int32_t getULPDistance(const T& a, const T& b) const;

    template<typename T>
    bool cmpAllClose(const T& a, const T& b) const;

    template<typename T, typename U>
    void setPrintMessage(const std::shared_ptr<U>& a,
                         const std::string& tensorAName,
                         const std::shared_ptr<U>& b,
                         const std::string& tensorBName,
                         unsigned numPartials = 1,
                         unsigned           partialOffset = 0);
    std::string getPrintMessage() const
    {
        MME_ASSERT(!m_printMsg.empty(), "error message empty");
        return m_printMsg;
    }
    template<typename T>
    float getFloatVal(const T& val) const
    {
        return (float) val;
    }
    void setMaxUlp(MmeCommon::EMmeDataType dtType);
    void setAbsTol(MmeCommon::EMmeDataType dtType);

    static bool isElementStrideHole(std::shared_ptr<MmeSimTensor> tensor, unsigned rawIndex);

private:
    unsigned m_maxULP;
    static constexpr unsigned m_minULP = 4;
    static constexpr float m_fp8ThresholdCompareWithULP = 0.05;
    static constexpr float m_ThresholdCompareWithULP = 2.0;
    float m_lowestNToCompareWithULP = m_ThresholdCompareWithULP;
    float m_absTolerance = 0.05f;
    float m_relTolerance = 0.1f;

    unsigned m_fpBias = 0;
    MmeCommon::InfNanMode m_infNanMode;
    bool m_exitOnFailure;
    bool m_printAllDiffs;
    std::vector<MmeCommon::SizeArray> m_diffArray = {{0}};
    std::string m_printMsg = "";
    unsigned cdSize = 0;
};

template<>
int32_t TensorComparatorImpl::getULPDistance(const float& a, const float& b) const;
