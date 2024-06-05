#include "tensor_comparator_impl.h"
#include "sim_tensor.h"
#include "data_types/non_standard_dtypes.h"
#include "mme_reference.h"
#include "print_utils.h"
#include <sstream>

using namespace MmeCommon;

TensorComparatorImpl::TensorComparatorImpl(unsigned CDSize,
                                           EMmeDataType dtType,
                                           unsigned expBias,
                                           MmeCommon::InfNanMode infNanMode,
                                           bool printAllDiffs,
                                           bool exitOnFailure)
: m_fpBias(expBias),
  m_infNanMode(infNanMode),
  m_printAllDiffs(printAllDiffs),
  m_exitOnFailure(exitOnFailure),
  cdSize(CDSize)
{
    setMaxUlp(dtType);
    setAbsTol(dtType);
}

void TensorComparatorImpl::setMaxUlp(EMmeDataType dtType)
{
    // limit maxULP according to the data types precision.
    unsigned maxULP = cdSize * 5;
    switch (dtType)
    {
        case EMmeDataType::e_type_fp8_152:
            m_lowestNToCompareWithULP = m_fp8ThresholdCompareWithULP;
            maxULP = maxULP >> 21;
            break;
        case EMmeDataType::e_type_fp8_143:
            m_lowestNToCompareWithULP = m_fp8ThresholdCompareWithULP;
            maxULP = maxULP >> 20;
            break;
        case EMmeDataType::e_type_bf16:
            maxULP = maxULP >> 16;
            break;
        case EMmeDataType::e_type_fp16:
        case EMmeDataType::e_type_ufp16:
            maxULP = maxULP >> 13;
            break;
        case EMmeDataType::e_type_fp32:
        case EMmeDataType::e_type_fp32_ieee:
            break;
        default:
            MME_ASSERT(0, "invalid data type");
    }

    m_maxULP = std::max(maxULP, (unsigned) m_minULP);
}

void TensorComparatorImpl::setAbsTol(EMmeDataType dtType)
{
    // limit absTolerance according to the data types precision.
    float minRepresntable = 0;
    switch (dtType)
    {
        case EMmeDataType::e_type_fp8_152:
        {
            fp8_152_t minVal((uint8_t) m_minULP);
            minRepresntable = minVal.toFloat(m_fpBias, m_infNanMode);
            break;
        }
        case EMmeDataType::e_type_fp8_143:
        {
            fp8_143_t minVal((uint8_t) m_minULP);
            minRepresntable = minVal.toFloat(m_fpBias, m_infNanMode);
            break;
        }
        case EMmeDataType::e_type_fp16:
        {
            fp16_t minVal((uint16_t) m_minULP);
            minRepresntable = minVal.toFloat(m_fpBias, m_infNanMode);
            break;
        }
        case EMmeDataType::e_type_ufp16:
        {
            ufp16_t minVal((uint16_t) m_minULP);
            minRepresntable = minVal.toFloat(m_fpBias, m_infNanMode);
            break;
        }
        case EMmeDataType::e_type_bf16:
        {
            bf16_t minVal((uint16_t) m_minULP);
            minRepresntable = minVal.toFloat();
            break;
        }
        case EMmeDataType::e_type_fp32:
        case EMmeDataType::e_type_fp32_ieee:
        {
            fp32_t minVal((uint32_t) m_minULP);
            minRepresntable = minVal.toFloat();
            break;
        }
        default:
            MME_ASSERT(0, "invalid data type");
    }

    m_absTolerance = std::max(m_absTolerance, minRepresntable);
}

bool TensorComparatorImpl::doCompare(const pMMESimTensor& a,
                                     const std::string& tensorAName,
                                     const pMMESimTensor& b,
                                     const std::string& tensorBName,
                                     const unsigned testCounter,
                                     const std::string& testInfoStr,
                                     const unsigned numPartials,
                                     const bool failOnMismatch)
{
    MME_ASSERT(a->getElementType() == b->getElementType(),
           "Element type of sim tensors should match to perform proper comparison");
    atomicColoredPrint(COLOR_CYAN,
                       "INFO: Comparing the results of %s to %s. (test #%u)\n",
                       tensorAName.c_str(),
                       tensorBName.c_str(),
                       testCounter);
    bool equal = false;
    m_diffArray.clear();
    m_printMsg.clear();
    unsigned partialOffset = a->getSize(MME_MAX_TENSOR_DIMS -1);

    switch (a->getElementType())
    {
        case EMmeDataType::e_type_fp8_143:
            equal = compare<fp8_143_t>(a, b, numPartials);
            if (!equal) setPrintMessage<fp8_143_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        case EMmeDataType::e_type_fp8_152:
            equal = compare<fp8_152_t>(a, b, numPartials);
            if (!equal) setPrintMessage<fp8_152_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        case EMmeDataType::e_type_bf16:
            equal = compare<bf16_t>(a, b, numPartials);
            if (!equal) setPrintMessage<bf16_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        case EMmeDataType::e_type_fp16:
            equal = compare<fp16_t>(a, b, numPartials);
            if (!equal) setPrintMessage<fp16_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        case EMmeDataType::e_type_ufp16:
            equal = compare<ufp16_t>(a, b, numPartials);
            if (!equal) setPrintMessage<ufp16_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        case EMmeDataType::e_type_fp32:
            equal = compare<fp32_t>(a, b, numPartials);
            if (!equal) setPrintMessage<fp32_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        case EMmeDataType::e_type_tf32:
            equal = compare<fp32_t>(a, b, numPartials);
            if (!equal) setPrintMessage<fp32_t>(a, tensorAName, b, tensorBName, numPartials, partialOffset);
            break;
        default:
            MME_ASSERT(0, "invalid data type");
    }

    if (!equal)
    {
        printAndExit(tensorAName, tensorBName, testCounter, testInfoStr, failOnMismatch);
    }

    return equal;
}

bool TensorComparatorImpl::doCompareBitExact(const pMMESimTensor& a,
                                             const std::string& tensorAName,
                                             const pMMESimTensor& b,
                                             const std::string& tensorBName,
                                             const unsigned testCounter,
                                             const std::string& testInfoStr)
{
    atomicColoredPrint(COLOR_CYAN,
                       "INFO: Comparing the results of %s to %s. (test #%u)\n",
                       tensorAName.c_str(),
                       tensorBName.c_str(),
                       testCounter);

    MME_ASSERT(a->getMemorySize() == b->getMemorySize(), "Memory size of A and B doesnt match");
    m_diffArray.clear();
    char* t0Data = a.get()->data();
    char* t1Data = b.get()->data();

    if (0 == memcmp(t0Data, t1Data, a.get()->getMemorySize())) return true;

    // Find the mismatch
    unsigned elementSize = a.get()->getElementSize();
    bool found = false;
    for (unsigned pixel = 0; pixel < a.get()->getSizeInElements(); pixel++)
    {
        if (isElementStrideHole(a, pixel))
        {
            continue;
        }

        unsigned index = pixel * elementSize;
        if (memcmp(&t0Data[index], &t1Data[index], elementSize))
        {
            m_diffArray.push_back(a.get()->getOffsetOfIndex(pixel));
            found = true;
            if (!m_printAllDiffs) break;
        }
    }
    if (!found)
    {
        return true;
    }

    switch (a->getElementType())
    {
        case EMmeDataType::e_type_fp8_143:
            setPrintMessage<fp8_143_t>(a, tensorAName, b, tensorBName);
            break;
        case EMmeDataType::e_type_fp8_152:
            setPrintMessage<fp8_152_t>(a, tensorAName, b, tensorBName);
            break;
        case EMmeDataType::e_type_bf16:
            setPrintMessage<bf16_t>(a, tensorAName, b, tensorBName);
            break;
        case EMmeDataType::e_type_fp16:
            setPrintMessage<fp16_t>(a, tensorAName, b, tensorBName);
            break;
        case EMmeDataType::e_type_ufp16:
            setPrintMessage<ufp16_t>(a, tensorAName, b, tensorBName);
            break;
        case EMmeDataType::e_type_fp32:
            setPrintMessage<fp32_t>(a, tensorAName, b, tensorBName);
            break;
        case EMmeDataType::e_type_tf32:
            setPrintMessage<fp32_t>(a, tensorAName, b, tensorBName);
            break;
        default:
            MME_ASSERT(0, "invalid data type");
    }

    printAndExit(tensorAName, tensorBName, testCounter, testInfoStr);

    return false;
}

bool TensorComparatorImpl::doCompare(const Matrix& a,
                                     const std::string& matrixAName,
                                     const Matrix& b,
                                     const std::string& matrixBName)
{
    MME_ASSERT(a.getElementType() == b.getElementType(),
           "Element type of matrices should match to perform proper comparison");
    atomicColoredPrint(COLOR_CYAN,
                       "INFO: Comparing the results of %s to %s.\n",
                       matrixAName.c_str(),
                       matrixBName.c_str());
    bool equal = false;
    m_diffArray.clear();
    m_printMsg.clear();
    pCommonMatrix matA = std::make_shared<Matrix>(a);
    pCommonMatrix matB = std::make_shared<Matrix>(b);
    switch (a.getElementType())
    {
        case EMmeDataType::e_type_fp8_143:
            equal = compare<fp8_143_t>(matA, matB);
            if (!equal) setPrintMessage<fp8_143_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_fp8_152:
            equal = compare<fp8_152_t>(matA, matB);
            if (!equal) setPrintMessage<fp8_152_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_bf16:
            equal = compare<bf16_t>(matA, matB);
            if (!equal) setPrintMessage<bf16_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_fp16:
            equal = compare<fp16_t>(matA, matB);
            if (!equal) setPrintMessage<fp16_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_ufp16:
            equal = compare<fp16_t>(matA, matB);
            if (!equal) setPrintMessage<ufp16_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_fp32:
            equal = compare<fp32_t>(matA, matB);
            if (!equal) setPrintMessage<fp32_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_tf32:
            equal = compare<fp32_t>(matA, matB);
            if (!equal) setPrintMessage<fp32_t>(matA, matrixAName, matB, matrixBName);
            break;
        default:
            MME_ASSERT(0, "not supported");
    }

    if (!equal)
    {
        printAndExit(matrixAName, matrixBName, 0, "");
    }

    return equal;
}

bool TensorComparatorImpl::doCompareBitExact(const Matrix& a,
                                             const std::string& matrixAName,
                                             const Matrix& b,
                                             const std::string& matrixBName)
{
    atomicColoredPrint(COLOR_CYAN,
                       "INFO: Comparing the results of %s to %s.\n",
                       matrixAName.c_str(),
                       matrixBName.c_str());
    m_diffArray.clear();
    unsigned elementSizeA = getElementSize(a.getElementType());
    unsigned elementSizeB = getElementSize(b.getElementType());
    uint64_t sizeA = a.getSizeInElements() * elementSizeA;
    uint64_t sizeB = b.getSizeInElements() * elementSizeB;
    MME_ASSERT(sizeA == sizeB, "Memory size of A and B doesnt match");
    const char* t0Data = a.data();
    const char* t1Data = b.data();

    if (0 == memcmp(t0Data, t1Data, sizeA)) return true;

    // Find the mismatch
    bool found = false;
    for (unsigned pixel = 0; pixel < a.getSizeInElements(); pixel++)
    {
        unsigned index = pixel * elementSizeA;
        if (memcmp(&t0Data[index], &t1Data[index], elementSizeA))
        {
            m_diffArray.push_back(a.getOffsetOfIndex(pixel));
            found = true;
            if (!m_printAllDiffs) break;
        }
    }
    MME_ASSERT(found, "mismatch not found");

    pCommonMatrix matA = std::make_shared<Matrix>(a);
    pCommonMatrix matB = std::make_shared<Matrix>(b);

    switch (a.getElementType())
    {
        case EMmeDataType::e_type_fp8_143:
            setPrintMessage<fp8_143_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_fp8_152:
            setPrintMessage<fp8_152_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_bf16:
            setPrintMessage<bf16_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_fp16:
            setPrintMessage<fp16_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_ufp16:
            setPrintMessage<ufp16_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_fp32:
            setPrintMessage<fp32_t>(matA, matrixAName, matB, matrixBName);
            break;
        case EMmeDataType::e_type_tf32:
            setPrintMessage<fp32_t>(matA, matrixAName, matB, matrixBName);
            break;
        default:
            MME_ASSERT(0, "invalid data type");
    }

    printAndExit(matrixAName, matrixBName, 0, "");

    return false;
}

// Compare two tensors.
// In case of numPartials > 1, a holds N partial results, and b holds the sum of the partial results. The function
// compares the sum of the N partial values of a in the corresponding locations within the partials with the value in b
template<typename T, typename U>
bool TensorComparatorImpl::compare(const U& a, const U& b, const unsigned numPartials)
{
    unsigned aNumElements = a->getSizeInElements();
    unsigned bNumElements = b->getSizeInElements();
    MME_ASSERT(aNumElements == bNumElements * numPartials, "tensor sizes do not match");
    unsigned element = 0;
    unsigned end = bNumElements;

    bool equal = true;
    while (element != end)
    {
        T pixelA;
        if (numPartials == 1)
        {
            pixelA = reinterpret_ptr<T>(a->getElementAt(a->getOffsetOfIndex(element)));
        }
        else
        {
            float pixelAfloat = 0.0f;
            for (int n = 0; n < numPartials; n++)
            {
                unsigned partialElement = bNumElements * n + element;
                pixelAfloat += (float)(reinterpret_ptr<T>(a->getElementAt(a->getOffsetOfIndex(partialElement))));
            }
            pixelA = T(pixelAfloat);
        }
        T pixelB = reinterpret_ptr<T>(b->getElementAt(b->getOffsetOfIndex(element)));
        if (!compareByType(pixelA, pixelB, m_maxULP))
        {
            equal = false;
            m_diffArray.push_back(a->getOffsetOfIndex(element));
            if (!m_printAllDiffs) return equal;
        }
        element++;
    }
    return equal;
}

template<typename T>
bool TensorComparatorImpl::compareByType(const T& a, const T& b, unsigned ulpMaxDiff) const
{
    // use allComp method when signs are different or values low.
    float aValF = getFloatVal(a), bValF = getFloatVal(b);

    bool result;

    if (a.isNan(m_infNanMode) || b.isNan(m_infNanMode))
    {
        return false;
    }
    if (((aValF < 0) != (bValF < 0)) || ((aValF == 0) || (bValF == 0)) ||
        (std::abs(aValF - bValF) < m_lowestNToCompareWithULP))
    {
        return cmpAllClose(a, b);
    }
    else
    {
        int32_t distance = 0;
        distance = getULPDistance<T>(a, b);
        return distance <= ulpMaxDiff;
    }
    return result;
}

template<typename T>
int32_t TensorComparatorImpl::getULPDistance(const T& a, const T& b) const
{
    // skip if they are equal
    if (a == b) return 0;
    // return max for Nan or inf
    const int32_t max = std::numeric_limits<int32_t>::max();
    float aValFloat = getFloatVal(a);
    float bValFloat = getFloatVal(b);
    if (a.isNan(m_infNanMode) || b.isNan(m_infNanMode)) return max;
    if (a.isInf(m_infNanMode) || b.isInf(m_infNanMode)) return max;

    // dont compare different signs
    MME_ASSERT((getFloatVal(a) < 0) == (getFloatVal(b) < 0), "values should be of the same sign");

    // get abs(distance)
    int32_t distance = a.value() - b.value();
    if (distance < 0) distance = -distance;

    return distance;
}

template<>
int32_t TensorComparatorImpl::getULPDistance<float>(const float& a, const float& b) const
{
    // skip if they are equal
    if (a == b) return 0;
    // return max for Nan or inf
    const int32_t max = std::numeric_limits<int32_t>::max();
    float aValFloat = getFloatVal(a);
    float bValFloat = getFloatVal(b);
    if (std::isnan(aValFloat) || std::isnan(bValFloat)) return max;
    if (std::isinf(aValFloat) || std::isinf(bValFloat)) return max;
    // type punning - move from float to bit representation
    union floatInt32Pun
    {
        float f;
        uint32_t i;
    };
    floatInt32Pun aP, bP;
    aP.f = a;
    bP.f = b;
    // dont compare different signs
    if ((aP.i < 0) != (bP.i < 0)) return max;
    // get abs(distance)
    int32_t distance = aP.i - bP.i;
    if (distance < 0) distance = -distance;
    return distance;
}

template<typename T, typename U>
void TensorComparatorImpl::setPrintMessage(const std::shared_ptr<U>& a,
                                           const std::string& tensorAName,
                                           const std::shared_ptr<U>& b,
                                           const std::string& tensorBName,
                                           unsigned           numPartials,
                                           unsigned           partialOffset)
{
    std::stringstream ss;
    MME_ASSERT(!m_diffArray.empty(), "diff array is empty!");

    if (m_printAllDiffs)
    {
        ss << "Found " << m_diffArray.size() << " mismatches" << std::endl;
    }

    for (auto diff : m_diffArray)
    {
        T valA;
        std::vector<T> aPartialValues(numPartials);
        if (numPartials == 1)
        {
            valA = reinterpret_ptr<T>(a->getElementAt(diff));
        }
        else
        {
            SizeArray partialIdx = diff;
            unsigned numElementsB = b.get()->getSizeInElements();
            for (unsigned n=0; n<numPartials; n++)
            {
                partialIdx[partialIdx.size() - 1] = diff[partialIdx.size() - 1] + n * partialOffset;
                aPartialValues[n] = reinterpret_ptr<T>(a->getElementAt(partialIdx));
            }
        }
        T valB = reinterpret_ptr<T>(b->getElementAt(diff));
        float fValA = getFloatVal(valA), fValB = getFloatVal(valB);

        ss << "Mismatch at element  [" << arrayToStr(diff.begin(), diff.end()) << "]" << std::endl;
        ss << tensorAName << " value: " << fValA << " (0x" << std::hex << (unsigned) valA.value() << ")" << std::endl;
        if (numPartials != 1)
        {
            ss << tensorAName << "      partial values are: ";
            for (unsigned n=0; n<numPartials; n++)
            {
                float partialFloatVal = getFloatVal(aPartialValues[n]);
                ss << partialFloatVal << " (0x" << std::hex << (unsigned) aPartialValues[n].value() << "),   ";
            }
            ss << std::endl;
        }
        ss << tensorBName << " value: " << fValB << " (0x" << std::hex << (unsigned) valB.value() << ")" << std::endl;
    }
    m_printMsg = ss.str();
}

void TensorComparatorImpl::printAndExit(const std::string& tensorAName,
                                        const std::string& tensorBName,
                                        const unsigned int testCounter,
                                        const std::string& testInfoStr,
                                        const bool failOnMismatch)
{
    std::stringstream ss;
    if ( failOnMismatch)
    {
        ss << "ERROR: Comparison mismatch. " << tensorAName << " <-> " << tensorBName << ". (test #" << testCounter << ")"
           << std::endl;
        ss << "Test details: " << std::endl << testInfoStr << std::endl;
    }
    else 
    {
        ss << "Comparison mismatch ** Allowed, NO ERROR **. " << tensorAName << " <-> " << tensorBName << ". (test #" << testCounter << ")"
           << std::endl;    
    }
    ss << getPrintMessage();

    if ( failOnMismatch)
    {
        atomicColoredPrint(COLOR_RED, "%s", ss.str().c_str());
        if (m_exitOnFailure)
        {
            exit(1);
        }
    }
    else    // do not fail the test
    {
        atomicColoredPrint(COLOR_YELLOW, "%s", ss.str().c_str());    
    }
}

template<typename T>
bool TensorComparatorImpl::cmpAllClose(const T& a, const T& b) const
{
    bool isEqual = false;
    if (a.isInf(m_infNanMode) || b.isInf(m_infNanMode)) return isEqual;
    float aValFloat = getFloatVal(a);
    float bValFloat = getFloatVal(b);
    if (!a.isNan(m_infNanMode) && !b.isNan(m_infNanMode))
    {
        float absDiff = std::abs(aValFloat - bValFloat);
        float maxDiff = (m_absTolerance + (m_relTolerance * std::min(std::abs(aValFloat), std::abs(bValFloat))));
        isEqual = (absDiff <= maxDiff);
    }
    return isEqual;
}

template<>
float TensorComparatorImpl::getFloatVal(const fp8_152_t& val) const
{
    return val.toFloat(m_fpBias, m_infNanMode);
}
template<>
float TensorComparatorImpl::getFloatVal(const fp8_143_t& val) const
{
    return val.toFloat(m_fpBias, m_infNanMode);
}
template<>
float TensorComparatorImpl::getFloatVal(const fp16_t& val) const
{
    return val.toFloat(m_fpBias, m_infNanMode);
}

Settable<SizeArray> TensorComparatorImpl::getDiffElement() const
{
    Settable<SizeArray> diffElement;

    if (!m_diffArray.empty())
    {
        diffElement = m_diffArray[0];
    }

    return diffElement;
}

bool TensorComparatorImpl::isElementStrideHole(pMMESimTensor tensor, unsigned rawIndex)
{
    const unsigned dim = tensor->getDim();
    const MmeCommon::SizeArray& sizes = tensor->getSizes();
    const MmeCommon::SizeArray& strides = tensor->getStrides();
    unsigned localIndex = rawIndex;
    for (int i = dim - 1; i > 0; --i)
    {
        localIndex %= strides[i];
        if (localIndex >= strides[i - 1] * sizes[i - 1])
        {
            return true;
        }
    }

    return false;
}

 TensorComparator::TensorComparator(unsigned CDSize,
                              MmeCommon::EMmeDataType dtType,
                              unsigned expBias,
                              MmeCommon::InfNanMode infNanMode,
                              bool printAllDiffs,
                              bool exitOnFailure)
{
    m_impl = std::make_unique<TensorComparatorImpl>(CDSize,
                                                    dtType,
                                                    expBias,
                                                    infNanMode,
                                                    printAllDiffs,
                                                    exitOnFailure);
}

TensorComparator::~TensorComparator() = default;

bool TensorComparator::doCompare(const pMMESimTensor& a,
                                 const std::string& tensorAName,
                                 const pMMESimTensor& b,
                                 const std::string& tensorBName,
                                 const unsigned testCounter,
                                 const std::string& testInfoStr,
                                 const unsigned numPartials,
                                 const bool failOnMismatch)
{
    return m_impl->doCompare(a, tensorAName,
                             b, tensorBName,
                             testCounter,
                             testInfoStr,
                             numPartials,
                             failOnMismatch);
}

bool TensorComparator::doCompareBitExact(const pMMESimTensor& a,
                                         const std::string& tensorAName,
                                         const pMMESimTensor& b,
                                         const std::string& tensorBName,
                                         const unsigned testCounter,
                                         const std::string& testInfoStr)
{
    return m_impl->doCompareBitExact(a, tensorAName,
                                     b, tensorBName,
                                     testCounter,
                                     testInfoStr);
}

bool TensorComparator::doCompare(const Matrix& a,
                                 const std::string& matrixAName,
                                 const Matrix& b,
                                 const std::string& matrixBName)
{
    return m_impl->doCompare(a, matrixAName,
                             b, matrixBName);
}
bool TensorComparator::doCompareBitExact(const Matrix& a,
                                         const std::string& matrixAName,
                                         const Matrix& b,
                                         const std::string& matrixBName)
{
    return m_impl->doCompareBitExact(a, matrixAName,
                                     b, matrixBName);
}

Settable<MmeCommon::SizeArray> TensorComparator::getDiffElement() const
{
    return m_impl->getDiffElement();
}