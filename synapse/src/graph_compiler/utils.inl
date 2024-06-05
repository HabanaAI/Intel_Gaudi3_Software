#pragma once

#include <algorithm>
#include <deque>
#include <numeric>
#include <sstream>

#include "synapse_types.h"
#include <cstdint>
#include <limits>
#include "syn_data_type_type_conversions.h"
#include "infra/defs.h"
#include "limits_4bit.h"

#include "event_triggered_logger.hpp"

template<typename valType>
bool valueIsInRangeOfType(valType val, synDataType type)
{
    switch (type) {
        case syn_type_int4:
            return val <= INT4_MAX_VAL && val >= INT4_MIN_VAL;
        case syn_type_uint4:
            return val <= UINT4_MAX_VAL && val >= UINT4_MIN_VAL;
        case syn_type_fixed:
            return val <= std::numeric_limits<int8_t>::max() && val >= std::numeric_limits<int8_t>::min();
        case syn_type_uint8:
            return val <= std::numeric_limits<uint8_t>::max() && val >= std::numeric_limits<uint8_t>::min();
        case syn_type_int16:
            return val <= std::numeric_limits<int16_t>::max() && val >= std::numeric_limits<int16_t>::min();
        case syn_type_uint16:
            return val <= std::numeric_limits<uint16_t>::max() && val >= std::numeric_limits<uint16_t>::min();
        case syn_type_int32:
            return val <= std::numeric_limits<int32_t>::max() && val >= std::numeric_limits<int32_t>::min();
        case syn_type_uint32:
            return val <= std::numeric_limits<uint32_t>::max() && val >= std::numeric_limits<uint32_t>::min();
        case syn_type_int64:
            return val <= std::numeric_limits<int64_t>::max() && val >= std::numeric_limits<int64_t>::min();
        case syn_type_uint64:
            return val <= std::numeric_limits<uint64_t>::max() && val >= std::numeric_limits<uint64_t>::min();
        case syn_type_single:
        //Bf16 has the same range as a float just a smaller precision.
        case syn_type_bf16:
            return val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest();
        case syn_type_fp16:
            return val <= float(fp16_t::max()) && val >= float(fp16_t::lowest());
        case syn_type_na:
            return false;
        default:
            return false;
    }
}

/*
 * converts to uint32 according to synDataType.
 * also calculates size (in bytes) of the converted variable
 */
template<typename valType>
uint32_t convertToUint32(valType val, synDataType type, size_t& sizeInBytes)
{
    uint32_t retVal = 0;
    switch (type)
    {
        case syn_type_int4:
        case syn_type_uint4:
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
            retVal = (uint8_t)val;
            sizeInBytes = sizeof(uint8_t);
            break;
        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_fp16:
            retVal = (uint16_t)val;
            sizeInBytes = sizeof(uint16_t);
            break;
        case syn_type_single:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_tf32:
        case syn_type_hb_float:
        case syn_type_int64:
        case syn_type_uint64:
            retVal = (uint32_t)val;
            sizeInBytes = sizeof(uint32_t);
            break;
        default:
            HB_ASSERT(false, "wrong data type");
    }
    return retVal;
}

template<class valType>
void padBuffWithValueInternal(void* buff, unsigned elements, valType val, synDataType type)
{
    HB_ASSERT(buff, "invalid buffer");

    size_t   valueBytes = 0;
    uint32_t valueInType = convertToUint32(val, type, valueBytes); //modifies valueBytes
    if (type == syn_type_uint4 || type == syn_type_int4)
    {
        HB_ASSERT(elements % 2 == 0, "Invalid argument: element (Not divisable by 2)");
        elements /= 2;
    }
    for (uint32_t i = 0; i < elements; ++i)
    {
        memcpy((uint8_t*)buff + i*valueBytes, &valueInType, valueBytes);
    }
}

template<class valType>
void padBuffWithValue(void* buff, unsigned elements, valType val, synDataType type)
{
    padBuffWithValueInternal(buff, elements, val, type);
}

/*
 * padBuffWithValue template specializations for double & float data types.
 * double/float require casting to int to avoid casting negative double/float to unsigned, which is undefined behaviour.
 * Conversion to int truncates the value towards zero.
 */
template<> inline
void padBuffWithValue(void* buff, unsigned elements, double val, synDataType type)
{
    padBuffWithValueInternal(buff, elements, (int)val, type);
}

template<> inline
void padBuffWithValue(void* buff, unsigned elements, float val, synDataType type)
{
    padBuffWithValueInternal(buff, elements, (int)val, type);
}



template<typename ITER_TYPE>
uint64_t multiplyElements(ITER_TYPE begin, ITER_TYPE end)
{
    return std::accumulate(begin,
                           end,
                           1ULL, // acc
                           [](uint64_t acc, uint64_t val){return acc * val;});
}

template<typename CONTAINER>
uint64_t multiplyElements(CONTAINER container)
{
    using std::begin;
    using std::end;
    return multiplyElements(begin(container), end(container));
}

template<typename COPY_FROM_TYPE, typename COPY_TO_TYPE>
void copyDataFromTo(void* dstData, void* srcData, uint32_t size)
{
    COPY_FROM_TYPE*  src = (COPY_FROM_TYPE*) srcData;
    COPY_TO_TYPE*    dst = (COPY_TO_TYPE*) dstData;

    for( unsigned i = 0; i < size; i++, src++, dst++ )
    {
        *dst = *src;
    }
}

template<typename COPY_FROM_TYPE>
void copyDataFrom(void* dstData, void* srcData, synDataType dstType, uint32_t size)
{
    switch(dstType)
    {
        case syn_type_int4:
        case syn_type_fixed:
        {
            copyDataFromTo<COPY_FROM_TYPE, int8_t>( dstData, srcData, size );
            break;
        }
        case syn_type_uint4:
        case syn_type_uint8:
        {
            copyDataFromTo<COPY_FROM_TYPE, uint8_t>( dstData, srcData, size );
            break;
        }
        case syn_type_int16:
        {
            copyDataFromTo<COPY_FROM_TYPE, int16_t>( dstData, srcData, size );
            break;
        }
        case syn_type_bf16:
        case syn_type_fp16:
        case syn_type_uint16:
        {
            copyDataFromTo<COPY_FROM_TYPE, uint16_t>( dstData, srcData, size );
            break;
        }
        case syn_type_int32:
        {
            copyDataFromTo<COPY_FROM_TYPE, int32_t>( dstData, srcData, size );
            break;
        }
        case syn_type_uint32:
        {
            copyDataFromTo<COPY_FROM_TYPE, uint32_t>( dstData, srcData, size );
            break;
        }
        case syn_type_single:
        {
            copyDataFromTo<COPY_FROM_TYPE, int32_t>( dstData, srcData, size );
            break;
        }
        case syn_type_int64:
        {
            copyDataFromTo<COPY_FROM_TYPE, int64_t>( dstData, srcData, size );
            break;
        }
        case syn_type_uint64:
        {
            copyDataFromTo<COPY_FROM_TYPE, uint64_t>( dstData, srcData, size );
            break;
        }
        case syn_type_na:
        default:
        {
            HB_ASSERT(false, "Invalid data type");
        }
    }
}

template<class CONTAINER>
std::string toString(const CONTAINER& container, char delimiter)
{
    return toString(container.begin(), container.end(), delimiter);
}

template<class T_ITER>
std::string toString(T_ITER begin, T_ITER end, char delimiter)
{
    std::ostringstream out;
    auto iter = begin;
    while (iter != end)
    {
        out << *iter;
        ++iter;
        if (iter != end)
        {
            out.put(delimiter);
        }
    }
    return out.str();
}

template<>
inline std::string toString<const unsigned char*>(const unsigned char* begin, const unsigned char* end, char delimiter)
{
    std::ostringstream out;
    auto iter = begin;
    while (iter != end)
    {
        out << (unsigned)*iter;
        ++iter;
        if (iter != end)
        {
            out.put(delimiter);
        }
    }
    return out.str();
}

template<class CONTAINER, class GETTER>
std::string toString(const CONTAINER& container, char delimiter, GETTER getter)
{
    return toString(container.begin(), container.end(), delimiter, getter);
}

template<class T_ITER, class GETTER>
std::string toString(T_ITER begin, T_ITER end, char delimiter, GETTER getter)
{
    std::ostringstream out;
    auto iter = begin;
    while (iter != end)
    {
        out << getter(*iter);
        ++iter;
        if (iter != end)
        {
            out.put(delimiter);
        }
    }
    return out.str();
}

inline const std::string& toString(synDeviceType deviceType)
{
    static const std::string deviceTypeStrings[] = {"",
                                                    "",
                                                    "Gaudi",
                                                    "Gaudi2000M",
                                                    "Gaudi2",
                                                    "Gaudi3",
                                                    "synDeviceEmulator",
                                                    "InvalidDevice"};
    static constexpr uint32_t nbSupportedDevices = sizeof(deviceTypeStrings) / sizeof(deviceTypeStrings[0]);
    static_assert(nbSupportedDevices == synDeviceTypeSize);
    return deviceTypeStrings[deviceType];
}

template<class T, size_t N>
std::string arrayToString(const T (&array)[N], char delimiter)
{
    return toString(std::begin(array), std::end(array), delimiter);
}

template<typename valType>
valType clip(const valType& n, const valType& lower, const valType& upper)
{
    return std::max(lower, std::min(n, upper));
}

template<typename ValType>
ValType gcd(ValType a, ValType b)
{
    if (b == 0) return a;
    return gcd(b, a % b);
}

template<typename ValType>
ValType gcd(std::set<ValType>  elements)
{
    if( elements.size() == 0 )
    {
        // Undefined
        return 0;
    }

    ValType gcdCandidate = *(elements.begin());

    // After the first loop, gcdCandidate will still be elements[0]...
    for( auto element : elements )
    {
        gcdCandidate = gcd( element, gcdCandidate );
        if( gcdCandidate == 1 )
        {
            break;
        }
    }

    return gcdCandidate;
}

template<class valType>
constexpr synDataType dataTypeToSynType()
{
    return asSynType<valType>();
}

template<typename arrayType>
std::string getDimStr(const arrayType* arr, const unsigned dim, bool isFCDLast = false)
{
    std::stringstream tensorSizes;

    if (dim > 0)
    {
        tensorSizes << "[";
    }

    for (int i=0; i < dim; ++i)
    {
        int j = isFCDLast ? dim-1-i : i;
        tensorSizes << arr[j];
        std::string next = (i < dim-1)? ", " : "]";
        tensorSizes << next;
    }

    return tensorSizes.str();
}

template<typename valType>
void dequeSimpleMerge(std::deque<valType>&    dst,
                      std::deque<valType>&    src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

inline uint64_t lowestOneBit(uint64_t n)
{
    return (n & (~(n - 1)));
}

inline uint64_t highestOneBit(uint64_t i) {
    // For example, if we had 0100000100000b, it will go:
    i |= (i >>  1); // 0110000110000b
    i |= (i >>  2); // 0111100111100b
    i |= (i >>  4); // 011111111111b
    i |= (i >>  8); // 011111111111b
    i |= (i >> 16); // 011111111111b
    i |= (i >> 32);
    return i - (i >> 1); // 011111111111b - 01111111111b = 010000000000b
}

template<typename Container, typename T>
size_t index_of(Container&& container, T&& value)
{
    auto it = std::find(container.begin(), container.end(), value);
    if (it == container.end())
        return -1;
    return std::distance(container.begin(), it);
}

template<class T_CONT>
bool replaceFirst(T_CONT& cont,const typename T_CONT::value_type& oldItem, const typename T_CONT::value_type& newItem)
{
    auto pos = std::find(cont.begin(), cont.end(), oldItem);
    if (pos != cont.end())
    {
        *pos = newItem;
        return true;
    }

    return false;
}

template<typename ITER_TYPE, typename T>
bool areAllElementsEqual(ITER_TYPE begin, ITER_TYPE end, T value)
{
    return std::all_of(begin, end,[value](T i){return i == value;});
}

template<class CONTAINER>
bool areAllElementsUnique(const CONTAINER& container)
{
    std::set<typename CONTAINER::value_type> encounteredElements;
    for (auto& val : container)
    {
        bool newInsert = encounteredElements.insert(val).second;
        if (!newInsert)
        {
            return false;
        }
    }
    return true;
}

template<typename TO, typename FROM>
std::unique_ptr<TO> static_unique_pointer_cast(std::unique_ptr<FROM>&& old)
{
    return std::unique_ptr<TO>{static_cast<TO*>(old.release())};
}

template<typename It>
typename std::iterator_traits<It>::difference_type countNon1Elements(It first, It last)
{
    return std::count_if(first, last, [](uint32_t v) {return v != 1;});
}

template<class CONTAINER, class Pred>
void eraseIf(CONTAINER& container, Pred pred)
{
    auto it = container.begin();
    while (it != container.end())
    {
        if (pred(*it))
        {
            it = container.erase(it);
        }
        else
        {
            it++;
        }
    }
}

#define MIN_VALID_PKT_SIZE      ((int32_t) 0)
#define MAX_INVALID_PKT_SIZE    (MIN_VALID_PKT_SIZE - 1)

template<typename CP_DMA_PKT_TYPE, typename LIN_DMA_PKT_TYPE, typename WBULK_PKT_TYPE, typename ARB_POINT_PKT_TYPE,
         unsigned CP_DMA_OP_CODE,  unsigned LIN_DMA_OP_CODE,  unsigned WBULK_OP_CODE,  unsigned ARB_POINT_OP_CODE>
bool isValidPacket(const uint32_t*&                pCurrentPacket,
                   int64_t&                        leftBufferSize,
                   utilPacketType&                 packetType,
                   uint64_t&                       pktBufferAddress,
                   uint64_t&                       pktBufferSize,
                   const std::vector<int32_t>&     packetTypesSize,
                   const std::vector<std::string>& packetsName,
                   ePacketValidationLoggingMode    loggingMode = PKT_VAIDATION_LOGGING_MODE_DISABLED)
{
    bool shouldLog = (loggingMode == PKT_VAIDATION_LOGGING_MODE_ENABLED);

    if (packetTypesSize.size() == 0)
    {
        return false;
    }

    packetType = UTIL_PKT_TYPE_OTHER;

    uint32_t opcode = ((CP_DMA_PKT_TYPE*) pCurrentPacket)->opcode;
    if ((opcode < packetTypesSize.size()) &&
        (opcode < packetsName.size()) &&
        (opcode > 0))
    {
        uint32_t packetSize = packetTypesSize[opcode];
        if (unlikely(packetSize < MIN_VALID_PKT_SIZE))
        {
            if (unlikely(shouldLog))
            {
                ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                        GC,
                        "Invalid packet size. opcode: {}",
                        opcode);
            }

            return false;
        }

        if (opcode == WBULK_OP_CODE)
        {
            packetSize = 2 * sizeof(uint32_t) + ((WBULK_PKT_TYPE*) pCurrentPacket)->size64 * sizeof(uint64_t);
        }
        else if (opcode == CP_DMA_OP_CODE)
        {
            pktBufferAddress = ((CP_DMA_PKT_TYPE*) pCurrentPacket)->src_addr;
            pktBufferSize    = ((CP_DMA_PKT_TYPE*) pCurrentPacket)->tsize;
            packetType         = UTIL_PKT_TYPE_CP_DMA;
        }
        else if (opcode == LIN_DMA_OP_CODE)
        {
            pktBufferAddress = ((LIN_DMA_PKT_TYPE*) pCurrentPacket)->src_addr;
            pktBufferSize    = ((LIN_DMA_PKT_TYPE*) pCurrentPacket)->tsize;
            packetType         = UTIL_PKT_TYPE_LDMA;
        }
        else if (opcode == ARB_POINT_OP_CODE)
        {
            if (((ARB_POINT_PKT_TYPE*) pCurrentPacket)->rls)
            {
                packetType = UTIL_PKT_TYPE_ARB_CLEAR;
            }
            else
            {
                packetType = UTIL_PKT_TYPE_ARB_SET;
            }
        }

        leftBufferSize -= packetSize;
        pCurrentPacket += (packetSize / sizeof(uint32_t));
    }
    else
    {
        if (unlikely(shouldLog))
        {
            ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                    GC,
                    "Invalid opcode: {}",
                    opcode);
        }

        return false;
    }

    return true;
}

template<typename CP_DMA_PKT_TYPE, typename LIN_DMA_PKT_TYPE, typename WBULK_PKT_TYPE, typename ARB_POINT_PKT_TYPE,
         unsigned CP_DMA_OP_CODE,  unsigned LIN_DMA_OP_CODE,  unsigned WBULK_OP_CODE, unsigned ARB_POINT_OP_CODE>
bool checkForUndefinedOpcode(const void*&                    pCommandsBuffer,
                             uint64_t                        bufferSize,
                             const std::vector<int32_t>&     packetTypesSize,
                             const std::vector<std::string>& packetsName,
                             ePacketValidationLoggingMode    loggingMode = PKT_VAIDATION_LOGGING_MODE_DISABLED)
{
    bool shouldLog = (loggingMode == PKT_VAIDATION_LOGGING_MODE_ENABLED);

    const uint32_t* pCurrentPacket = nullptr;
    int64_t         leftBufferSize = 0;

    bool isFailureFound = false;

    do
    {
        pCurrentPacket = (const uint32_t*) pCommandsBuffer;
        leftBufferSize = bufferSize;

        while (leftBufferSize > 0)
        {
            utilPacketType packetType       = UTIL_PKT_TYPE_OTHER;
            uint64_t       pktBufferAddress = 0;
            uint64_t       pktBufferSize    = 0;

            bool isValid = isValidPacket<CP_DMA_PKT_TYPE, LIN_DMA_PKT_TYPE, WBULK_PKT_TYPE, ARB_POINT_PKT_TYPE,
                                        CP_DMA_OP_CODE,  LIN_DMA_OP_CODE,  WBULK_OP_CODE, ARB_POINT_OP_CODE>(
                                        pCurrentPacket,
                                        leftBufferSize,
                                        packetType,
                                        pktBufferAddress,
                                        pktBufferSize,
                                        packetTypesSize,
                                        packetsName,
                                        shouldLog ? PKT_VAIDATION_LOGGING_MODE_ENABLED :
                                                    PKT_VAIDATION_LOGGING_MODE_DISABLED);
            if (!isValid)
            {
                if (unlikely(shouldLog))
                {
                    ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                            GC,
                            "Invalid packet at buffer offset {} (buffer size {}) from buffer 0x{:x}",
                            bufferSize - leftBufferSize,
                            bufferSize,
                            (uint64_t) pCommandsBuffer);
                }

                isFailureFound = true;
                break;
            }
        }

        if (leftBufferSize < 0)
        {
            if (unlikely(shouldLog))
            {
                ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                        GC,
                        "Buffer size does not fit");
            }

            isFailureFound = true;
        }

        // In case failure found, we did not yet logged it, and we should log-upon-failure,
        // Set shouldLog to true (otherwise - false)
        if ((loggingMode == PKT_VAIDATION_LOGGING_MODE_UPON_FAILURE) &&
            (isFailureFound) &&
            (!shouldLog))
        {
            shouldLog = true;
        }
        else
        {
            shouldLog = false;
        }
    } while (shouldLog);

    return !isFailureFound;
}

// TODO[c++20]: use std::string_view::starts_with
constexpr bool startsWith(std::string_view str, std::string_view prefix)
{
    return str.substr(0, prefix.size()) == prefix;
}

// Given range with the following properties:
//  (1) good < bad.
//  (2) isGood(good) = true.
//  (3) isGood(bad)  = false.
//  (4) if a < b and isGood(b) = true, then also isGood(a) = true.
// Then the function returns the value "v" such that isGood(v) = true and isGood(v + 1) = false.
// The complexity of the function is log(bad - good).
template<typename T>
T binarySearch(T good, T bad, std::function<bool(const T&)> isGood)
{
    HB_ASSERT(good < bad, "Good must be less than bad");
    while (bad - good != 1)
    {
        HB_ASSERT(std::numeric_limits<T>::max() - good >= bad, "Overflow detected");
        T val = (bad + good) / 2;
        if (isGood(val))
        {
            good = val;
        }
        else
        {
            bad = val;
        }
    }
    return good;
}

inline bool areStridesAscending(const NStrideArray& strides, unsigned rank)
{
    for (unsigned strideIdx = 1; strideIdx <= rank; strideIdx++)
    {
        if (strides.at(strideIdx) < strides.at(strideIdx - 1)) return false;
    }
    return true;
}