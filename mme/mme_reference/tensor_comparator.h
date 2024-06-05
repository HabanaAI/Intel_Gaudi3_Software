#pragma once
#include "include/mme_common/mme_common_enum.h"
#include "settable.h"
#include <cmath>
#include <cstdint>
#include <memory>

// Compare tensors element by element.
// if there is a difference diffElement & message are set.
class MmeSimTensor;
class Matrix;
class TensorComparatorImpl;
class TensorComparator
{
public:
    explicit TensorComparator(unsigned CDSize,
                              MmeCommon::EMmeDataType dtType,
                              unsigned expBias = 0,
                              MmeCommon::InfNanMode infNanMode = MmeCommon::e_mme_full_inf_nan,
                              bool printAllDiffs = false,
                              bool exitOnFailure = false);
    ~TensorComparator();
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
    std::unique_ptr<TensorComparatorImpl> m_impl;
};
