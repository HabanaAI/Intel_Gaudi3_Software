#pragma once

//A set of general-purpose utility functions
#include "drm/habanalabs_accel.h"
#include "gc_interface.h" // TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include "habana_device_types.h"
#include "hal_reader/hal_reader.h"
#include "infra/defs.h"
#include "math_utils.h"
#include "settable.h"
#include "synapse_common_types.h"
#include "synapse_types.h"
#include "tensor_shape.h"
#include "tpc_kernel_lib_interface.h"
#include "type_utils.h"
#include "types.h"
#include <dlfcn.h>

#include <cmath>
#include <limits>
#include <set>
#include <string_view>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if MAGIC_ENUM_SUPPORTED
#include <magic_enum-0.8.1/include/magic_enum.hpp>
#endif

class HabanaGraph;

#define NCHW_N_DIM 3
#define NCHW_C_DIM 2
#define NCHW_H_DIM 1
#define NCHW_W_DIM 0

typedef void* libHandle;
typedef void* fnHandle;

inline constexpr auto ENGINES_ID_SIZE_MAX = std::max<uint32_t>({gaudi_engine_id::GAUDI_ENGINE_ID_SIZE,
                                                                gaudi2_engine_id::GAUDI2_ENGINE_ID_SIZE,
                                                                gaudi3_engine_id::GAUDI3_ENGINE_ID_SIZE});
inline constexpr auto QUEUE_ID_SIZE_MAX =
    std::max<uint32_t>(gaudi_queue_id::GAUDI_QUEUE_ID_SIZE, gaudi2_queue_id::GAUDI2_QUEUE_ID_SIZE);

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#define varoffsetof(t, m) ((size_t)(&(((t*)0)->m)))
#define CEIL(size, base) ((size + base - 1) / base)

// Basic precompile max calculation (due to definition on third-party, adding the "SYN_" prefix)
#define SYN_MAX(a, b) (((a) > (b)) ? (a) : (b))

//A macro for not using parameters but not getting warned about it.
#define UNUSED(x) ((void)x)

#define TRANSLATE_ENUM_TO_STRING(x) \
    case x: \
        return #x;

#define CHECK_MAX_VAL(variable, type) HB_ASSERT(variable < std::numeric_limits<type>::max(), "max val exceeded")

// Copy src array to dst array with implicit casting
#define castNcopy(dst, src, size)          \
    for (size_t kk = 0; kk < size; kk++) { \
        dst[kk] = src[kk];                 \
    }

inline bool isAddressRangeOverlaps(uint64_t firstBaseAddress,
                                   uint64_t firstEndAddress,
                                   uint64_t secondBaseAddress,
                                   uint64_t secondEndAddress)
{
    if ((firstBaseAddress > secondEndAddress) || (secondBaseAddress > firstEndAddress))
    {
        return false;
    }

    return true;
}

inline uint64_t getAddressFromSplitParts( uint32_t highAddressPart, uint32_t lowAddressPart )
{
    return ( ( ( (uint64_t) highAddressPart ) << 32 ) & 0xFFFFFFFF00000000 ) | ( lowAddressPart );
}

inline bool isBitSelected(uint16_t data, uint32_t index)
{
    return bool((data & (0x1 << index)));
}

typedef union
{
    void* p;
    uint64_t u64;
    uint32_t u32[2];
} ptrToInt;

// The following can only be used for contiguous fields
// For simplicity, there are no verifications
uint32_t    getBitFieldValue(uint32_t registerVal, uint32_t fieldOffset, uint32_t numOfBits);
void        setBitFieldValue(uint32_t* pRegisterVal, uint32_t fieldOffset, uint32_t numOfBits, uint32_t newVal);

template<typename ValType>
ValType gcd(ValType a, ValType b);

// The following GCD is by definition not-utilized, and should be used where readability is prefered over performance
// In case of performance preference, the suggestion is to run over GCD candidates (from largest until 1), and find one that all agree upon
template<typename ValType>
ValType gcd(std::set<ValType> elements);

unsigned    next_power_of_two(unsigned a);
const char* getDeviceName(HabanaDeviceType type);
void        copyString(std::string_view input, char* dst, size_t dst_size);
void        copyStringSafe(std::string_view input, char* dst, size_t dst_size, size_t maxSize);
libHandle LoadSharedObject(const char* name);
fnHandle  GetFunction(libHandle handle, const char* name);
void      UnloadSharedObject(libHandle handle);
void      SafeMemcpy(void* dst, size_t dstSize, const void* src, size_t srcSize);
double    calculateExp(double num);
uint8_t   findFirstSetBit(unsigned tpcMask);
uint32_t  turnOffMSBs(uint32_t mask, unsigned numBitsToUnset);
void      dumpBufferToFile(const char* fileName, const void* buf, unsigned size);
uint32_t  crc32(const void* buff, uint64_t size, uint32_t prevCrc=0);
void      parsePrintf(uint32_t* buff, std::vector<std::string>& finalPrints);
template<typename valType>
bool      valueIsInRangeOfType(valType val, synDataType type);
template<typename valType>
void      padBuffWithValue(void *buff, unsigned elements, valType v, synDataType type);
unsigned  calcPaddingSize(uint64_t size, uint64_t alignment);
unsigned  alignSizeDown(unsigned size, uint64_t alignment);
unsigned  alignSizeUp(unsigned size, uint64_t alignment);

template<typename ITER_TYPE>
uint64_t multiplyElements(ITER_TYPE begin, ITER_TYPE end);

void findSpatialIndex(const TSize originalSize[SYN_MAX_TENSOR_DIM],
                      int spatialIndex[SYN_MAX_TENSOR_DIM - 1],
                      TSize spatialSize);

void findIndex(const TSize* sizes, uint32_t dimNum, uint64_t sizePos, TOffset* outIndex);

TSize calcAbsSize(const TSize* sizes, const TOffset* index, unsigned int dimNum);

int convInputDimSize(unsigned int outputSize,
                     unsigned int weightSize,
                     unsigned int stride,
                     int padding,
                     unsigned int dilation);

int convOutputDimSize(TSize inputSize,
                      unsigned int weightSize,
                      unsigned int stride,
                      int padding,
                      unsigned int dilation);

TSize convOutputDimSizeSamePadding(TSize inputSize, unsigned int stride);

void convPaddingSamePadding(unsigned int  inputSize,
                            unsigned int  weightSize,
                            unsigned int  dilation,
                            unsigned int  stride,
                            unsigned int& before,
                            unsigned int& after);

void copyData( void*        dstData,
               void*        srcData,
               synDataType  dstType,
               synDataType  srcType,
               uint64_t     size );

bool NodeTypeMatchingFunc(const NodePtr& a, const NodePtr& b);

bool NodeTypeMatchingFuncWithPrecision(const NodePtr& a, const NodePtr& b);

float bToMb(unsigned bytes);

float bToKb(unsigned bytes);

std::vector<std::string> splitString(const std::string& str, char delimiter);

// This method fills a multiplication of 64-Elements vectors (TPC vector-size)
// inputSize and outputSize are in elements
bool swizzle(int32_t* output, int32_t* input, unsigned inputSize, unsigned outputSize, synDataType targetDataType);

bool getSwizzleOutputNumOfElement(unsigned& swizzleOutputNumOfElements, unsigned numOfInputElements, synDataType targetDataType);

std::string_view getHighestGUIDDataType(const std::vector<std::string>& dtypes);
synDataType      getHighestGUIDDataType(const std::vector<synDataType>& dtypes);

template<class CONTAINER>
std::string toString(const CONTAINER& container, char delimiter);

template<class T_ITER>
std::string toString(T_ITER begin, T_ITER end, char delimiter);

template<class CONTAINER, class GETTER>
std::string toString(const CONTAINER& container, char delimiter, GETTER getter);

template<class T_ITER, class GETTER>
std::string toString(T_ITER begin, T_ITER end, char delimiter, GETTER getter);

inline const std::string& toString(synDeviceType deviceType);

template<class T, size_t N>
std::string arrayToString(const T (&array)[N], char delimiter);

std::vector<TSize> splitToChunks(TSize    numSamples,
                                 unsigned numOfChunks,
                                 unsigned firstBiggerIndex = 0,
                                 unsigned numOfPhysicalEngs = 0);

std::vector<TSize> splitToChunksWithIndexes(TSize     numSamples,
                                            unsigned  numOfChunks,
                                            unsigned  firstBiggerIndex,
                                            unsigned& firstMinimalIndex);

/**
    Given an interval, values outside the interval are clipped to the interval edges.

    @param n input number.
    @param lower minimum value
    @param upper maximum value
    @return value n is replaced with lower if n < lower, upper if n > upper, n otherwise
*/
template<typename valType>
valType clip(const valType& n, const valType& lower, const valType& upper);

bool findSingleNonOneDim(pTensor &weightsTensor, unsigned &nonOneDim);

bool isMemcpy(const Node& n);
bool isTpcMemcpy(const NodePtr& n);

bool fitsInBits(unsigned val, unsigned bits);
unsigned countSetBits(uint64_t val, unsigned numOfBits);

bool isHostDma(HabanaDeviceType type);
bool isCompletionQueue(HabanaDeviceType type);

deviceAddrOffset getVirtualAddressForMemoryID(uint64_t memId, uint64_t offset = 0);
uint64_t         getMemoryIDFromVirtualAddress(deviceAddrOffset virtualAddr);
void             getSectionInfoFromVirtualAddress(deviceAddrOffset virtualAddr, uint16_t& memId, uint64_t& offset);
deviceAddrOffset maskOutMemoryID(deviceAddrOffset virtualAddr);
deviceAddrOffset maskOutMemoryID(uint32_t virtualAddrHigh32bits);
uint64_t         getMaxMemorySectionID();
uint64_t         getSramMemoryID();
bool             assignVirtualAddressToUserPersistentTensor(pTensor t);
std::string_view getMemorySectionNameForMemoryID(uint64_t memId, const TensorPtr& t = TensorPtr());

template <typename T>
bool allClose(T a, T b, double relTolerance = 1e-05f, double absTolerance = 1e-08f)
{
    return (double)std::fabs(a - b) <= (absTolerance + relTolerance * (double)std::fabs(b));
}

bool copyTensorData(pTensor& dest, pTensor& src);

bool verifyDeviceShapeTensor(unsigned         dims,
                             const TSize*     maxSizes,
                             synDataType      dataType,
                             std::string_view tensorName,
                             const TSize*     minSizes);

int16_t castFloatToBFloat16(float floatVal, bool bigEndian = false);

void getConvolutionSize(const SizeArray&              xSize,
                        uint32_t                      yChannel,
                        const synConvolutionParamsV2& convParams,
                        SizeArray&                    wSizeOut,
                        SizeArray&                    ySizeOut);

unsigned getNumEnginesForDeviceType(HabanaDeviceType deviceType, const HalReader& halReader);

enum class insertNodeLocation
{
    AFTER_INPUT,
    AFTER_OUTPUT
};

TensorPtr createExpandedTensor(const TensorPtr& tensor, unsigned dim, unsigned fillValue = 1);
TensorPtr createSqueezedTensor(const TensorPtr& tensor, const Settable<unsigned> dim = Settable<unsigned>());
TensorPtr createFlattenedTensor(const TensorPtr& tensor, unsigned axis);

std::tuple<TensorPtr, NodePtr> expandTensor(const TensorPtr& tensor, unsigned dim);
std::tuple<TensorPtr, NodePtr> expandShapeTensorWithValue(const TensorPtr& expandIn, unsigned dim, TSize fillValue);
std::tuple<TensorPtr, NodePtr> squeezeTensor(const TensorPtr& tensor, const unsigned* const dim);
std::tuple<TensorPtr, NodePtr, NodePtr>
                              reinterpret64BitTensor(const TensorPtr& tensor, bool from64Bit, synDataType type);
std::pair<NodePtr, TensorPtr> reinterpretTensor(const TensorPtr& tensor, bool isInput, synDataType type);
TensorPtr createHost2DeviceTensor(synDataType dtype, uint64_t sizeInElements, const std::string& name);

void insertReshapeNodeAfter(HabanaGraph&       g,
                            const pTensor&     input,
                            const pNode&       n,
                            const std::string& name,
                            const NSizeArray&  sizes,
                            insertNodeLocation location);

std::pair<pNode, pTensor> createReshapeNode(const pTensor& input, const std::string& name, const SizeArray& sizes, insertNodeLocation location, bool enforceLogical = false);
std::pair<pNode, pTensor> createReshapeNode(const pTensor&     input,
                                            const std::string& name,
                                            const NSizeArray&  sizes,
                                            insertNodeLocation location,
                                            bool               enforceLogical = false);

struct SynapsePointerHasher
{
    template<typename T>
    size_t operator()(T&& object) const
    {
        return object->getHash();
    }
};

struct SynapsePointerEqualTo
{
    template<typename T>
    size_t operator()(T&& lhs, T&& rhs) const
    {
        return *lhs == *rhs;
    }
};

bool validateConvPadding(const SizeArray&                xSize,
                         const SizeArray&                wSize,
                         const SizeArray&                ySize,
                         uint32_t                        dimNum,
                         const synConvolution3DParamsV2& params);

bool validateConvolutionSize(const SizeArray&                xSize,
                             const SizeArray&                wSize,
                             const SizeArray&                ySize,
                             uint32_t                        dimNum,
                             const synConvolution3DParamsV2& params);

bool validateTransposedDedxSize(const SizeArray&                xSize,
                                const SizeArray&                wSize,
                                const SizeArray&                ySize,
                                uint32_t                        dimNum,
                                const synConvolution3DParamsV2& params);

template<class T_CONT>
bool replaceFirst(T_CONT& cont, const typename T_CONT::value_type& oldItem, const typename T_CONT::value_type& newItem);

std::set<unsigned> dimIndexArray(const SizeArray& sizes, unsigned dimI, unsigned d);

bool extractScalarFromStaticTensor(const pTensor& tensor, float& scalarVal);

bool changeTensorElementTypeSafe(pTensor tensor, synDataType type);

template<class T>
class ArrayDeletor
{
public:
    void operator()(T* p)
    {
        delete[] p;
    }
};

// LL to prevent overflow in expressions like 'size * BITS_PER_BYTE / elementSizeInBits'
constexpr auto BITS_PER_BYTE    = 8ULL;
constexpr auto BITS_IN_UINT32   = 32;
constexpr auto BITS_IN_UINT64   = 64;
constexpr auto BITS_IN_UNSIGNED = sizeof(unsigned) * 8;
template <typename T>
inline T bitsToByte(T value)
{
    return value / BITS_PER_BYTE;
}

template <typename T>
inline T safeBitsToByte(T value)
{
    HB_ASSERT(value % BITS_PER_BYTE == 0, "Can't cast bits to byte");
    return bitsToByte(value);
}

inline unsigned safeSizeInBitsToElements(unsigned vectorSize, unsigned elementSizeInBits)
{
    HB_ASSERT(vectorSize * BITS_PER_BYTE % elementSizeInBits == 0, "Cannot calculate number of elements");
    return vectorSize * BITS_PER_BYTE / elementSizeInBits;
}

template <typename T>
inline T safeBytesToBits(T value)
{
    HB_ASSERT(value < std::numeric_limits<T>::max() / BITS_PER_BYTE, "cannot convert bytes to bits");
    return value * BITS_PER_BYTE;
}

inline bool isTypeFloat(synDataType type)
{
    return type == syn_type_single || type == syn_type_bf16 || type == syn_type_fp16 || type == syn_type_fp8_152 ||
           type == syn_type_fp8_143;
}

inline bool is8BitFloat(synDataType type)
{
    return type == syn_type_fp8_152 || type == syn_type_fp8_143;
}

inline int negMod(int a, int b)
{
    return (b + (a % b)) % b;
}

const char* synTensorType2Txt(synTensorType type);

synTensorProperty quantizationPropertyToTensorProperty(synQuantizationProperty quantProp);

// TODO: [SW-166081] Remove when fuser is moved to protocolIR
static inline gcapi::DeviceId_t newGlueCodeToOldDeviceId(tpc_lib_api::DeviceId deviceId)
{
    switch (deviceId)
    {
        case tpc_lib_api::DEVICE_ID_GAUDI:
            return gcapi::DEVICE_ID_GAUDI;
        case tpc_lib_api::DEVICE_ID_GAUDI2:
            return gcapi::DEVICE_ID_GAUDI2;
        case tpc_lib_api::DEVICE_ID_GAUDI3:
            return gcapi::DEVICE_ID_GAUDI3;
        default:
            HB_ASSERT(false, "Unsupported device id {}", deviceId);
            return gcapi::DEVICE_ID_MAX;
    }
}

inline tpc_lib_api::DeviceId deviceTypeToDeviceID(synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return tpc_lib_api::DEVICE_ID_GAUDI;
        case synDeviceGaudi2:
            return tpc_lib_api::DEVICE_ID_GAUDI2;
        case synDeviceGaudi3:
            return tpc_lib_api::DEVICE_ID_GAUDI3;
        // for unit tests that do not define a device
        case synDeviceEmulator:
        case synDeviceTypeInvalid:
            return tpc_lib_api::DEVICE_ID_GAUDI;
        default:
            HB_ASSERT(false, "Unsupported device type {}", deviceType);
            return tpc_lib_api::DEVICE_ID_MAX;
    }
}

static inline synDeviceType deviceIDToDeviceType(tpc_lib_api::DeviceId deviceId)
{
    switch (deviceId)
    {
        case tpc_lib_api::DEVICE_ID_GAUDI:
            return synDeviceGaudi;
        case tpc_lib_api::DEVICE_ID_GAUDI2:
            return synDeviceGaudi2;
        case tpc_lib_api::DEVICE_ID_GAUDI3:
            return synDeviceGaudi3;
        default:
            HB_ASSERT(false, "Unsupported device id {}", deviceId);
            return synDeviceTypeInvalid;
    }
}

bool isOpSupportsNdims(const NodePtr& node);
bool isTensorDimsValidForNode(const TensorPtr& tensor, const NodePtr& node, bool checkNDims);

/**
 * @brief Returns the minimal dimensions, as well as the sizes and strides, needed to represent the same data as the
 * original tensor. If we have a static dimension where strides[i] * size[i] == stride[i + 1] we can merge him into the
 * higher dimension.
 * @return std::tuple<TensorShape, NStrideArray> The shape and strides of the minimal dimensions.
 */
std::pair<TensorShape, NStrideArray> mergeDenseDimensions(const TensorPtr& t);

// https://en.wikipedia.org/wiki/Filename#Comparison_of_file_name_limitations
// POSIX "Fully portable filenames" allows [0-9A-Za-z._-] in filenames so that
// every other char is replaced by this function with '_'. Leading '-' is also
// replaced. But the max 14 char length is ignored by this function.
std::string sanitizeFileName(std::string fileName);

bool       isDirectory(const std::string& path);
bool       isFusedKernel(std::string_view guid);
SizeVector toSizeVector(const TensorPtr& tensor);
SizeVector toSizeVector(const SizeArray& sizes, unsigned int rank);
SizeVector toSizeVector(const TSize sizes[SYN_MAX_TENSOR_DIM], unsigned int rank);

// Template class that inherits from some Base class and add to it a "isDestroyed"
// member. It is useful to tackle edge case when a static thread local instance has
// been destroyed before a regular static instance. See KernelDB::instance() for example.
template<class Base>
struct TWithDestroyedFlag : Base
{
    using Base::Base;  // make all ctors of base available
    // volatile because assignment in dtor to a non volatile is ignored by a compiler
    volatile bool isDestroyed = false;
    ~TWithDestroyedFlag() { isDestroyed = true; }
};

#if MAGIC_ENUM_SUPPORTED
template<class T>
std::string_view enumToString(T enumeration)
{
    return magic_enum::enum_name(enumeration);
}
#else
template<class T>
T enumToString(T enumeration)
{
    return enumeration;
}
#endif

void* allocateBufferForSynType(synDataType type, uint64_t elementsNum);

bool isInferenceQuantization(const HabanaGraph& g);

bool isGuidPrefix(const NodePtr& node, std::string_view prefix );

// List of well-known memory sections
inline constexpr std::string_view WORKSPACE_MEMORY_SECTION_NAME    = "workspace";
inline constexpr std::string_view PROGRAM_MEMORY_SECTION_NAME      = "program";
inline constexpr std::string_view PROGRAM_DATA_MEMORY_SECTION_NAME = "program-data";

template<class T, size_t N>
bool doesArrayContainZeros(const T (&array)[N])
{
    return std::any_of(std::cbegin(array), std::cend(array), [](T v) { return v == 0; });
}

#include  "utils.inl"
