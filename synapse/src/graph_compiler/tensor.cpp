#include "tensor.h"

#include "compilation_hal_reader.h"
#include "data_type_utils.h"
#include "define_synapse_common.hpp"
#include "defs.h"
#include "node.h"
#include "quantization_data.h"
#include "quantization_utils.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "transpose_utils.h"
#include "types.h"
#include "types_exception.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

std::atomic<uint64_t> Tensor::m_nextId {0};

const TensorPtr& Tensor::getRealTensor(const TensorPtr& tensor)
{
    const TensorPtr* realTensor = &tensor;
    while ((*realTensor)->isAliasedTensor())
    {
        realTensor = &((*realTensor)->getAliasTensor());
    }
    return *realTensor;
}

// Tensor Class methods

Tensor::Tensor(synDataType type, const char* tensorName, synTensorType tensorType, int32_t graphID)
: m_id(m_nextId++),
  m_name(tensorName ? tensorName : fmt::format("tensor-{}", m_id)),
  m_type(type),
  m_memSectionId(MEMORY_ID_RESERVED_FOR_WORKSPACE),
  m_graphID(graphID),
  m_tensorType(tensorType)
{
    memset(m_strides, 0, sizeof(m_strides));
    memset(m_origStrides, 0, sizeof(m_origStrides));
    m_quantizationParamsMap.emplace(QuantizationData::synTypeToQuantType(m_type), m_type);
}

Tensor::Tensor(synTensorType tensorType, int32_t graphID, const char* tensorName)
: Tensor(syn_type_na, tensorName, tensorType, graphID)
{
}

Tensor::Tensor(unsigned       dim,
               const TSize*   sizes,
               synDataType    type,
               char*          data,
               const TStride* strides,
               bool           isOutput,
               bool           isInput,
               unsigned       batchPos,
               const TSize*   minSizes,
               synTensorType  tensorType)
: m_id(m_nextId++),
  m_name(fmt::format("tensor-{}", m_id)),
  m_type(type),
  m_data(data),
  m_batchPos(batchPos),
  m_memSectionId(MEMORY_ID_RESERVED_FOR_WORKSPACE),
  m_tensorType(tensorType)
{
    HB_ASSERT(dim <= c_tensorMaxNDim, "dimension is bigger than maximum dimensions");

    reshape(dim, sizes, strides, minSizes);

    // Required since some tests go around the API, using the c'tor directly
    setProp(synTensorPropGeometryMin);
    setProp(synTensorPropGeometryMax);
    setProp(synTensorPropGeometryDim);

    //We don't enforce inputs- it is meaningless.
    UNUSED(isInput);
    if (isOutput)
    {
        enforceOutput();
    }
    m_quantizationParamsMap.emplace(QuantizationData::synTypeToQuantType(m_type), m_type);
}

Tensor::Tensor(unsigned dim, const TSize* sizes, synDataType type, const TSize* minSizes)
: Tensor(dim, sizes, type)
{
    if (minSizes != nullptr)
    {
        m_shape.setMinSize(minSizes);
    }
}

Tensor::Tensor(const Tensor& t, const TransposePermutationArray& permutation, const std::string& name)
: m_id(m_nextId++),
  m_name(name.empty() ? fmt::format("{}_c{}", t.m_name, m_id) : name),
  m_type(t.m_type),
  m_batchPos(t.m_batchPos),
  m_annotation(t.m_annotation.memory.alignment),
  m_quantizationParamsMap(t.m_quantizationParamsMap),
  m_memSectionId(MEMORY_ID_RESERVED_FOR_WORKSPACE),
  m_graphID(t.m_graphID),
  m_tensorType(t.m_tensorType),
  m_dynamicRange(t.m_dynamicRange),
  m_perChannelDynamicRange(t.m_perChannelDynamicRange)
{
    const NSizeArray& sizes         = t.getAllNSizesInElements();
    NSizeArray        PermutedSizes = applyPermutationOnSizes(sizes, permutation, false);
    if (!t.isDynamicShape())
    {
        reshape(permutation.size(), PermutedSizes.data(), nullptr, nullptr);
    }
    else
    {
        const NSizeArray& minSizes         = t.getNMinimalSizesInElements();
        NSizeArray        PermutedMinSizes = applyPermutationOnSizes(minSizes, permutation, false);
        reshape(permutation.size(), PermutedSizes.data(), nullptr, PermutedMinSizes.data());
    }

    auto propertyMask = synTensorPropGeometryMin | synTensorPropGeometryMax | synTensorPropGeometryDim;
    m_propsSet        = t.m_propsSet & propertyMask;
}

Tensor::Tensor(const TensorShape& shape, synDataType type)
: m_id(m_nextId++),
  m_name(fmt::format("tensor-{}", m_id)),
  m_type(type),
  m_memSectionId(MEMORY_ID_RESERVED_FOR_WORKSPACE)
{
    HB_ASSERT(m_shape.getDim() <= c_tensorMaxNDim, "dimension is bigger than maximum dimensions");

    reshape(shape.getDim(), shape.getNSizes().data(), nullptr, shape.getNMinSizes().data());
    // Required since some tests go around the API, using the c'tor directly
    setProp(synTensorPropGeometryMin);
    setProp(synTensorPropGeometryMax);
    setProp(synTensorPropGeometryDim);
    m_quantizationParamsMap.emplace(QuantizationData::synTypeToQuantType(m_type), m_type);
}

/* Tensor copy constructor.
 * Data belongs to the original tensor, so the copied tensor should not free the buffer,
 * even if original tensor was set with shouldFreeBuffer.
 */
Tensor::Tensor(const Tensor& t, bool copyAddresses /*false*/, bool copyData /*true*/, TensorNameClonePolicy namePolicy)
: m_id(m_nextId++),
  m_name(getClonedTensorName(t, namePolicy)),
  m_type(t.m_type),
  m_outputState(t.m_outputState),
  m_inputState(t.m_inputState),
  m_isWeights(copyData ? t.m_isWeights : false),
  m_isBias(copyData ? t.m_isBias : false),
  m_isParam(copyData ? t.m_isParam : false),
  m_unitMatrix(copyData ? t.m_unitMatrix : false),
  m_perChannelQuant(t.m_perChannelQuant),
  m_shape(t.m_shape),
  m_bufferSizeInBytes(t.m_bufferSizeInBytes),
  m_deviceSizeInBytes(t.m_deviceSizeInBytes),
  m_bufferDataType(t.m_bufferDataType),
  m_batchPos(t.m_batchPos),
  m_aliasInfo(),
  m_annotation(t.m_annotation),
  m_quantizationParamsMap(t.m_quantizationParamsMap),
  m_memSectionId(t.m_memSectionId),
  m_graphID(t.m_graphID),
  m_internalSectionHandle(t.m_internalSectionHandle),
  m_memoryDescriptor(t.m_memoryDescriptor),
  m_tensorType(t.m_tensorType),
  m_ctrlEdgeType(t.m_ctrlEdgeType),
  m_dynamicRange(t.m_dynamicRange),
  m_perChannelDynamicRange(t.m_perChannelDynamicRange),
  m_propsSet(t.m_propsSet),
  m_prefStride(t.m_prefStride),
  m_isDropped(t.m_isDropped),
  m_isRequantLocked(t.m_isRequantLocked),
  m_isInt8FixedPoint(t.m_isInt8FixedPoint),
  m_isInt16Limited(t.m_isInt16Limited),
  m_isSparsityWeights(t.m_isSparsityWeights),
  m_lockProps(t.m_lockProps),
  m_twinHost2DeviceTensor(t.m_twinHost2DeviceTensor),
  m_permutation(t.m_permutation)
{
    memcpy(m_strides, t.m_strides, sizeof(m_strides));
    memcpy(m_origStrides, t.m_origStrides, sizeof(m_origStrides));

    if (t.m_data != nullptr && copyData && t.isStaticParam())
    {
        bind(new char[m_deviceSizeInBytes], true);
        memcpy(m_data, t.m_data, m_deviceSizeInBytes);
    }

    if (copyAddresses)
    {
        if (t.m_sramOffset.is_set())
        {
            setSramOffset(t.m_sramOffset.value());
        }
        if (t.m_dramOffset.is_set())
        {
            setDramOffset(t.m_dramOffset.value());
        }
    }

    if (t.isHost2DeviceTensor() && m_data == nullptr)
    {
        // for H2D tensors, the data pointer is mandatory
        // if a deep copy was not made above, do a shallow copy here
        bind(t.getData(), false);
    }
}

Tensor::~Tensor()
{
    if (m_shouldFreeBuffer && m_data != nullptr)
    {
        delete[] m_data;
        m_data = nullptr;
    }
}

void Tensor::setName(std::string_view name, bool forceExact)
{
    if (m_lockName)
    {
        LOG_TRACE(GC, "Skipping tensor rename from {} to {}, tensor name was set by the user", m_name, name);
        return;
    }

    if (forceExact)
    {
        LOG_TRACE(GC, "Renaming tensor {} to {} (forceExact)", m_name, name);
        m_lockName = true;
        m_name = name;
    }
    else
    {
        std::string newName = fmt::format("{}_{}", name, m_id);
        LOG_TRACE(GC, "Renaming tensor {} to {}", m_name, newName);
        m_name = std::move(newName);
    }

}

void Tensor::setDenseStrides()
{
    LOG_TRACE(GC, "{}: tensor: {}", HLLOG_FUNC, getName());

    unsetPermutation();

    auto sizes = getNSizesInElements();
    reshape(getDim(), sizes.data(), nullptr);
    if (isType4Bit() && isCondensed4Bit())
    {
        HB_ASSERT(m_annotation.info4Bit.condensedDim != UNINITIALIZED_CONDENSED_DIM,
                  "Tensor {} was condensed but dimension wasn't updated", getName());
        // need to condense again since the reshape has expanded the condensed strides
        m_annotation.info4Bit.isCondensed = false;
        condenseStridesTo4Bit(m_annotation.info4Bit.condensedDim);
    }
}

// API setter
void Tensor::setGeometry(unsigned dim, const TSize* sizes, synGeometryType geometryType)
{
    HB_ASSERT(dim <= c_tensorMaxNDim, "dimension is bigger than maximum dimensions");
    HB_ASSERT_PTR(sizes);

    if (isPropSet(synTensorPropGeometryDim) && dim != getDim())
    {
        LOG_INFO(GC, "Modifying tensor \"{}\" dim number from {} to {}", getName(), getDim(), dim);
    }

    switch (geometryType)
    {
        case synGeometryMinSizes:
        {
            if (LOG_LEVEL_AT_LEAST_INFO(GC) && isPropSet(synTensorPropGeometryMin))
            {
                const auto& minSizes = getAllMinimalSizesInElements();
                if (dim != getDim() || !std::equal(minSizes.begin(), minSizes.begin() + dim, sizes))
                {
                    LOG_INFO(GC,
                             "Modifying tensor \"{}\" min sizes from [{}] to [{}]",
                             getName(),
                             fmt::join(minSizes.begin(), minSizes.begin() + getDim(), ","),
                             fmt::join(sizes, sizes + dim, ","));
                }
            }
            m_shape.setDim(dim);
            setProp(synTensorPropGeometryDim);
            m_shape.setMinSize(sizes);
            setProp(synTensorPropGeometryMin);
            break;
        }
        case synGeometryMaxSizes:
        {
            if (LOG_LEVEL_AT_LEAST_INFO(GC) && isPropSet(synTensorPropGeometryMax))
            {
                const auto& maxSizes = getAllSizesInElements();
                if (dim != getDim() || !std::equal(maxSizes.begin(), maxSizes.begin() + dim, sizes))
                {
                    LOG_INFO(GC,
                             "Modifying tensor \"{}\" max sizes from [{}] to [{}]",
                             getName(),
                             fmt::join(maxSizes.begin(), maxSizes.begin() + getDim(), ","),
                             fmt::join(sizes, sizes + dim, ","));
                }
            }
            m_shape.setDim(dim);
            setProp(synTensorPropGeometryDim);
            m_shape.setSize(sizes);
            setProp(synTensorPropGeometryMin);
            setProp(synTensorPropGeometryMax);
            break;
        }
        case synGeometryDims:
            m_shape.setDim(dim);
            setProp(synTensorPropGeometryDim);
            break;
        default:
        {
            HB_ASSERT(0, "Invalid geometry type");
            break;
        }
    }
    if (m_duplicatedTensor) setDefaultStrides();
}

void Tensor::reshape(unsigned dim, const TSize* maxSizes, const TStride* strides, const TSize* minSizes)
{
    HB_ASSERT(dim <= c_tensorMaxNDim, "dimension is bigger than maximum dimensions");

    m_shape.setDim(dim);

    if (maxSizes != nullptr)
    {
        m_shape.setSize(maxSizes);
    }

    if (minSizes != nullptr)
    {
        setMinSize(minSizes);
    }

    if (strides != nullptr)
    {
        HB_ASSERT(dim > 0, "cannot reshape to dimension 0");
        memcpy(m_strides, strides, sizeof(TStride) * (dim + 1));
        calculateDegenerateStrides(dim + 1);
    }
    else
    {
        setDefaultStrides();
    }
}

bool Tensor::resizeDims(unsigned dims)
{
    unsigned curDims = getDim();
    if (curDims == dims) return true;

    auto sizes = getNSizesInElements();
    if (curDims < dims)
    {
        // reshape the tensor by adding 1's as the sizes of the dimensions to the right
        for (unsigned i = curDims; i < dims; ++i)
        {
            sizes[i] = 1;
        }
        reshape(dims, sizes.data(), nullptr);
    }
    else if (dims < curDims)
    {
        for (unsigned i = dims; i < curDims; ++i)
        {
            if (sizes[i] != 1)
            {
                LOG_TRACE(GC,
                          "Can't resize from {} dimensions to {} dimensions, since the dimensions to be removed aren't 1", curDims, dims);
                return false;
            }
        }
        reshape(dims, sizes.data(), nullptr);
    }
    LOG_TRACE(GC, "Resized tensor {} from {} dimensions to {} dimensions", getName(), curDims, dims);
    return true;
}

/* try to align the given initialStrides to cache line size
     param sizes - tensor sizes
     param initial strides - tensor strides before alignment
     param numOfDims - tensor dims
     param dimToAlign - first dim to align
     return the aligned strides or {} if no alignment was needed
 */
std::optional<StrideArray> Tensor::getCacheLineAlignedStrides(const SizeArray&   sizes,
                                                              const StrideArray& initialStrides,
                                                              unsigned           numOfDims,
                                                              unsigned           dimToAlign /* = 0 */)
{
    const unsigned sizeToAlignTo = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();

    // check if already aligned
    if (initialStrides[dimToAlign + 1] % sizeToAlignTo == 0) return std::optional<StrideArray>{};

    // aligning doesn't improve when size is smaller than cache-line
    if (initialStrides[dimToAlign + 1] < sizeToAlignTo) return std::optional<StrideArray>{};

    // find next aligned size bigger than current size in bytes of the given dimension:
    unsigned alignedStrideVal = ((initialStrides[dimToAlign + 1] / sizeToAlignTo) * sizeToAlignTo) + sizeToAlignTo;

    // fill alignedStrides from initialStrides and then update according to alignment
    StrideArray alignedStrides = initialStrides;
    alignedStrides[dimToAlign + 1] = alignedStrideVal;
    for (unsigned dim = dimToAlign + 1; dim < numOfDims; ++dim)
    {
        alignedStrides[dim + 1] = alignedStrides[dim] * sizes[dim];
    }
    return std::optional<StrideArray>{alignedStrides};
}

// Modify the tensor's strides such that they will align to cache line size, this function may increase the tensor size
void Tensor::alignStridesToCacheLine()
{
    HB_ASSERT(!isPersistent() && !isPartOfRMWSection(),
              "aligning strides may increase size and thus disallowed for persistent tensor and tensor in RMW section");
    auto newStrides = getCacheLineAlignedStrides(getAllSizesInElements(), getAllStridesInBytes(), getDim());
    if (newStrides) reshape(getDim(), getAllSizesInElements().data(), newStrides->data());
}

void Tensor::bind(void* p, bool shouldFreeBuffer)
{
    HB_ASSERT(!isBound(), "Tensor already bound");
    m_data             = static_cast<char*>(p);
    m_shouldFreeBuffer = shouldFreeBuffer;
}

void Tensor::unbind()
{
    HB_ASSERT(isBound(), "Tensor wasn't bound");
    if (m_shouldFreeBuffer)
    {
        delete[] m_data;
    }
    m_data = nullptr;
}

void Tensor::rebind(TSize size)
{
    unbind();
    char* newData = new char[size];
    bind(newData, true);
    m_bufferSizeInBytes = size;
}

void Tensor::setTensorBuffer(void* ptr, uint64_t size, synDataType bufferDataType, bool copyBuffer)
{
    if (isBound())
    {
        unbind();
    }

    if (copyBuffer)
    {
        bind(new char[size], true);
        memcpy(m_data, ptr, size);
    }
    else
    {
        bind(ptr, copyBuffer);
    }

    m_bufferSizeInBytes = size;
    m_bufferDataType    = bufferDataType;
}

void* Tensor::map()
{
    if (getData() == nullptr)
    {
        AllocateHostMemory();
    }
    return getAddress();
}

void Tensor::getAllSizesInElements(TSize* sizes, unsigned count) const
{
    HB_ASSERT(getDim() <= count, "This function is deprecated for dimension {}", getDim());
    memcpy(sizes, m_shape.getNSizes().data(), sizeof(TSize) * count);
}

void Tensor::getAllNSizesInElements(TSize sizes[c_tensorMaxNDim]) const
{
    HB_ASSERT(getDim() <= c_tensorMaxNDim, "This function is deprecated for dimension {}", getDim());
    memcpy(sizes, m_shape.getNSizes().data(), sizeof(TSize) * c_tensorMaxNDim);
}

void Tensor::getAllSizesInElementsCondensed(TSize* sizes, unsigned count) const
{
    HB_ASSERT(getDim() <= count, "Buffer isn't big enough for {} dims", getDim());
    getAllSizesInElements(sizes, count);
    // tensor of 4 bit data type are packed as pairs in a byte,
    // TPC and DMA expect the FCD to be half the size
    // MME expects half the size in FCD (for activations) or dim 1 (for static weights)
    if (isType4Bit())
    {
        matchSizesTo4Bit(sizes, count);
    }
}

uint32_t Tensor::getNon1SizeDimsCount() const
{
    auto maxSizes = m_shape.getMaxSizes();
    return countNon1Elements(maxSizes.begin(), maxSizes.begin() + m_shape.getDim());
}

void Tensor::getAllMinimalSizesInElementsCondensed(TSize* sizes, unsigned count) const
{
    HB_ASSERT(getDim() <= count, "This function is deprecated for dimension {}", getDim());
    getAllMinimalSizesInElements(sizes, count);
    // tensor of 4 bit data type are packed as pairs in a byte,
    // TPC and DMA expect the FCD to be half the size
    // MME expects half the size in FCD (for activations) or dim 1 (for static weights)
    if (isType4Bit())
    {
        matchSizesTo4Bit(sizes, count);
    }
}

uint64_t Tensor::getTotalElements() const
{
    // When type isn't defined - can't rely on strides
    if (m_type == syn_type_na)
    {
        return getDenseSizeInElements();
    }
    if (m_shape.getDim() == 1)
    {
        return m_shape.getSize(DIM_C);
    }
    return BITS_PER_BYTE * getMaxStride() / getElementSizeInBits();
}

uint64_t Tensor::getMinimalElements() const
{
    uint64_t ret = 1;
    for (unsigned dim = 0; dim < m_shape.getDim(); ++dim)
    {
        ret *= getMinimalSizeInElements(dim);
    }
    return ret;
}

uint64_t Tensor::getDenseStrideInElements(int dim) const
{
    uint64_t ret = 1;

    if (dim > m_shape.getDim())
    {
        LOG_ERR(GC,
                "{}: tensor: {}, dim={}, expected a dim that is smaller than: {}",
                HLLOG_FUNC,
                getName(),
                dim,
                m_shape.getDim());
        throw InvalidTensorParamsException();
    }

    for (int i = 0; i < dim; ++i)
    {
        ret *= getSizeInElements(i);
    }
    return ret;
}

StrideArray Tensor::getAllStridesInBytes() const
{
    HB_ASSERT(getDim() <= c_tensorMaxDim, "This function is deprecated for dimension {}", getDim());

    StrideArray bytesStrides = {};
    std::copy(m_strides, m_strides + c_numOfStrides, bytesStrides.data());
    return bytesStrides;
}

TStride Tensor::calcStrideInElements(TStride stride) const
{
    if (isCondensed4Bit())
    {
        // if the Tensor is 4 bit condensed, the stride can be treated as byte
        return stride;
    }
    return BITS_PER_BYTE * stride / getElementSizeInBits();
}

TStride Tensor::getStrideInElements(unsigned dim) const
{
    HB_ASSERT(dim < c_numOfNStrides, "dimension is bigger than number of strides");
    return calcStrideInElements(m_strides[dim]);
}

template<int N, typename T>
void Tensor::getFirstNStridesInElements(T& strides) const
{
    if (isCondensed4Bit())
    {
        for (unsigned dim = 0; dim < N; ++dim)
        {
            strides[dim] = m_strides[dim];
        }
    }
    else
    {
      auto elementSizeInBits = getElementSizeInBits();
      // clang-format off
        switch (elementSizeInBits) {
        case 4:  for (unsigned dim = 0; dim < N; ++dim) strides[dim] = m_strides[dim] * (CHAR_BIT / 4);  break;
        case 8:  for (unsigned dim = 0; dim < N; ++dim) strides[dim] = m_strides[dim] * (CHAR_BIT / 8);  break;
        case 16: for (unsigned dim = 0; dim < N; ++dim) strides[dim] = m_strides[dim] / (16 / CHAR_BIT); break;
        case 32: for (unsigned dim = 0; dim < N; ++dim) strides[dim] = m_strides[dim] / (32 / CHAR_BIT); break;
        case 64: for (unsigned dim = 0; dim < N; ++dim) strides[dim] = m_strides[dim] / (64 / CHAR_BIT); break;
        default: HB_ASSERT(0, "");
        }
      // clang-format on
    }
}

void Tensor::getAllStridesInElements(TStride elemStrides[c_numOfStrides]) const
{
    HB_ASSERT(getDim() <= c_tensorMaxDim, "This function is deprecated for dimension {}", getDim());
    getFirstNStridesInElements<c_numOfStrides>(elemStrides);
}

NStrideArray Tensor::getNStridesInElements() const
{
    NStrideArray elemNStrides;
    getFirstNStridesInElements<c_numOfNStrides>(elemNStrides);
    return elemNStrides;
}

uint64_t Tensor::getSampleSize() const
{
    uint64_t ret = 1;
    for (uint32_t layoutIndex = 0; layoutIndex < DIM_B; ++layoutIndex)
    {
        ret *= getSizeInElements(layoutIndex);
    }
    return ret;
};

void Tensor::setPerChannelQuant(bool perChannel, bool force)
{
    if (perChannel && !isStaticParam())
    {
        if (force == false)
        {
            LOG_WARN(GC, "Cannot set per channel quant for {} tensor - only static tensors are supported", getName());
        }
        else
        {
            for (const auto& quantization : m_quantizationParamsMap)
            {
                if (quantization.second.isPerChannel())
                {
                    m_perChannelQuant = perChannel;
                    return;
                }
            }
            LOG_WARN(GC, "Cannot set per channel quant for {} tensor - all quant data is per-tensor", getName());
        }
    }
    else
    {
        m_perChannelQuant = perChannel;
    }
}

void Tensor::setSwizzled(uint64_t numOfSwizzleElements)
{
    m_annotation.dataInfo.isSwizzled = true;

    m_deviceSizeInBytes = safeBitsToByte(static_cast<uint64_t>(numOfSwizzleElements) * dataTypeToSizeInBits(m_type));
}

bool Tensor::isStridedOnFCD() const
{
    return (m_type != syn_type_na) && (m_strides[0] != dataTypeSizeInBytes(m_type, true)) && !isZeroSizedDataTensor() &&
           std::any_of(m_strides, m_strides + c_numOfNStrides, [](TStride s) { return s != 0; });
}

void Tensor::setAsAuxTensor(bool isScratchPad)
{
    m_annotation.isAuxTensor = true;
    m_annotation.isScratchPadAuxTensor = isScratchPad;
    setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
}

void Tensor::setElementType(synDataType type)
{
    HB_ASSERT(type != syn_type_na, "Type cannot be changed to syn_type_na");
    if (m_type != syn_type_na && (getElementSizeInBits() != dataTypeToSizeInBits(type)))
    {
        HB_ASSERT(getData() == nullptr, "Data is invalid when size of element type is changed");
    }
    m_type = type;
    setDefaultStrides();  // changing dtype causes us to modify the strides accordingly
}

/* Finish setting properties for a tensor after being assigned to a node.
 * assign degenerate strides
 */
void Tensor::calculateDegenerateStrides(unsigned dim, TStride maxStride, uint64_t lastElementOffset)
{
    if (dim == 0) return;

    HB_ASSERT(dim < c_numOfNStrides, "illegal dim!");

    const NSizeArray& sizesInElements = getAllNSizesInElements();

    if (!maxStride)
    {
        for (unsigned d = 0; d < dim; d++)
        {
            maxStride = std::max(maxStride, m_strides[d] * sizesInElements[d]);
        }
    }

    if (!isZeroSizedDataTensor())
    {
        if (!lastElementOffset)
        {
            for (unsigned d = 0; d < getDim(); d++)
            {
                lastElementOffset += m_strides[d] * (sizesInElements[d] - 1);
            }
        }
        m_deviceSizeInBytes = lastElementOffset + dataTypeSizeInBytes(m_type, true);
    }

    // Initialize degenerate dimensions
    std::fill(std::begin(m_strides) + dim, std::begin(m_strides) + c_numOfNStrides, maxStride);
    // save the calculated strides
    memcpy(m_origStrides, m_strides, sizeof(m_strides));
}

void Tensor::setDefaultStrides()
{
    if (m_type == syn_type_na) return;
    TStride stride    = isType4Bit() ? 1 : getElementSizeInBytes();  // 4bit strides are handled in a separate pass
    unsigned dim      = getDim();
    TStride maxStride = 0;
    uint64_t lastElementOffset = 0;
    auto setDefaultStrides = [this, &stride, &maxStride, &lastElementOffset, dim](const NSizeArray& sizesInElements) {
        for (unsigned d = 0; d <= dim; ++d)
        {
            m_strides[d] = stride;
            stride *= (TStride)sizesInElements[d];
            lastElementOffset += stride - m_strides[d];
            if (stride > maxStride) maxStride = stride;
        }
        LOG_TRACE(GC, "setting default strides ({}) in tensor {}", toString(m_strides, m_strides + dim, ','), m_name);
    };

    if (m_permutation)
    {
        NSizeArray sizesInElements = getAllNSizesInElements();
        // applying user-given permutation on the sizes (shape)
        SizeVector vec(sizesInElements.begin(), sizesInElements.begin() + dim);
        m_permutation->permuteShape(vec);

        std::copy(vec.begin(), vec.end(), sizesInElements.begin());
        LOG_TRACE(GC,
                  "applying user-given permutation on the shape in tensor {} to get shape ({}) for strides calculation",
                  m_name,
                  toString(sizesInElements.begin(), sizesInElements.begin() + dim, ','));
        // calculate strides
        setDefaultStrides(sizesInElements);

        // applying the inverse permutation on the calculated strides
        gc::Permutation invPermutation = m_permutation->getInversePermutation();
        invPermutation.permuteShape(m_strides, dim);
        LOG_TRACE(GC,
                  "applying inverse permutation on the strides in tensor {} to get these strides: ({})",
                  m_name,
                  toString(m_strides, m_strides + dim, ','));
    }
    else
    {
        const NSizeArray& sizesInElements = getAllNSizesInElements();
        // calculate strides
        setDefaultStrides(sizesInElements);
    }
    calculateDegenerateStrides(dim, maxStride, lastElementOffset);
}

bool Tensor::setPermutation(const gc::Permutation& permutation)
{
    LOG_TRACE(GC, "Trying to set this permutation in tensor {}: {}", getName(), permutation.toString());
    if (!permutation.isValidPermutation()) return false;
    m_permutation = permutation;
    return true;
}

// API setter.
void Tensor::setDeviceLayout(synDataType type, const uint32_t strides[HABANA_DIM_MAX])
{
    static_assert(HABANA_DIM_MAX + 1 == c_numOfNStrides, "number of strides is inconsistent");
    m_type = type;
    if (strides)
    {
        std::copy(strides, strides + HABANA_DIM_MAX, m_strides);
        if (isShapeTensor() && !isTrivialStrided())
        {
            LOG_WARN(SYN_API,
                     "irregular strides are not allowed for shape tensors! setting regular strides. tensor {}",
                     getName());
            setDefaultStrides();
        }
    }
}

TensorLocation Tensor::getTensorAllocatedLocation() const
{
    if (_isAliasedTensor())
    {
        return m_aliasInfo.pAliasedTensor->getTensorAllocatedLocation();
    }

    if (m_sramOffset.is_set())
    {
        return TENSOR_IN_SRAM;
    }
    else if (m_dramOffset.is_set())
    {
        return TENSOR_IN_DRAM;
    }
    else
    {
        return UNDEFINED_LOCATION;
    }
}

std::string_view Tensor::getTensorLocationString() const
{
    if (_isAliasedTensor())
    {
        return m_aliasInfo.pAliasedTensor->getTensorLocationString();
    }

    switch (getTensorAllocatedLocation())
    {  // clang-format off
        case TENSOR_IN_SRAM:     return "in SRAM";
        case TENSOR_IN_DRAM:     return "in DRAM";
        case UNDEFINED_LOCATION: return "not allocated";
        default: HB_ASSERT(false, "");
    }  // clang-format on
}

TensorLocation Tensor::location() const
{
    if (_isAliasedTensor())
    {
        return m_aliasInfo.pAliasedTensor->location();
    }
    return m_annotation.memory.location;
}

void Tensor::unsetSramOffset()

{
    if (_isAliasedTensor())
    {
        m_aliasInfo.pAliasedTensor->unsetSramOffset();
    }
    else
    {
        m_sramOffset.unset();
    }
}

void Tensor::unsetDramOffset()

{
    if (_isAliasedTensor())
    {
        m_aliasInfo.pAliasedTensor->unsetDramOffset();
    }
    else
    {
        m_dramOffset.unset();
    }
}

deviceAddrOffset Tensor::getSramOffset() const
{
    if (_isAliasedTensor())
    {
        return m_aliasInfo.pAliasedTensor->getSramOffset() + m_aliasInfo.byteOffset;
    }

    HB_ASSERT(m_sramOffset.is_set(), "{}: SRAM offset is not set! Tensor: {}", __FUNCTION__, getName());

    return m_sramOffset.value();
}

deviceAddrOffset Tensor::getDramOffset() const
{
    if (_isAliasedTensor())
    {
        return m_aliasInfo.pAliasedTensor->getDramOffset() + m_aliasInfo.byteOffset;
    }
    HB_ASSERT(m_dramOffset.is_set(), "{}: DRAM offset is not set! Tensor: {}", __FUNCTION__, getName());
    return m_dramOffset.value();
}

deviceAddrOffset Tensor::getTensorOffset() const
{
    if (tensorAllocatedInDram())
    {
        return getDramOffset();
    }
    if (tensorAllocatedInSram())
    {
        return getSramOffset();
    }
    return -1;  // invalidAddress
}

void Tensor::setSramOffset(deviceAddrOffset offset)
{
    m_sramOffset.set(offset);
    if (inConstSection()) return;
    setMemorySectionID(getSramMemoryID());
}

void Tensor::setTensorInSram()
{
    setMemorySectionID(getSramMemoryID());
    getTensorAnnotation().memory.location = TENSOR_IN_SRAM;
}

void Tensor::setTensorInWorkspace()
{
    setAsNonUserManaged();
    setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
    getTensorAnnotation().memory.location = TENSOR_IN_DRAM;
}

std::vector<Tensor::AddressRange> Tensor::getAddressRange() const
{
    uint64_t elementSizeInBytes = getElementSizeInBytes();
    uint64_t maxStrides         = getTotalElements();
    auto     sizes              = m_shape.getSizes();
    uint64_t totalElements      = multiplyElements(sizes.begin(), sizes.end());
    uint64_t contiguousElements = totalElements;
    if (contiguousElements > maxStrides)
    {
        // The tensor is inflated to a size bigger than the original size (broadcast)
        HB_ASSERT(isAliasedTensor(), "tensor isn't aliased");
        totalElements      = getAliasTensor()->getTotalElements();
        contiguousElements = totalElements;
    }
    else if (contiguousElements < maxStrides)
    {
        if (isAliasedTensor())
        {
            // Tensor is spread over memory space (concat on an inner dim)
            contiguousElements = 1;
            for (unsigned int dim = 0;
                 dim < m_shape.getDim() && contiguousElements == m_strides[dim] / elementSizeInBytes;
                 ++dim)
            {
                contiguousElements *= m_shape.getSize(dim);
            }
        }
        else
        {
            contiguousElements = maxStrides;
        }
    }
    std::vector<AddressRange> addressesRange;
    addressesRange.reserve(totalElements / contiguousElements);

    uint64_t                baseAddress = getTensorOffset();
    for (uint64_t elementsPassed = 0; elementsPassed < totalElements; elementsPassed += contiguousElements)
    {
        uint64_t distanceToElementInBytes = _getDistanceToElementInBytes(elementsPassed);
        uint64_t startAddress             = baseAddress + distanceToElementInBytes;
        uint64_t endAddress = startAddress + std::min(contiguousElements * elementSizeInBytes, m_deviceSizeInBytes);
        addressesRange.emplace_back(startAddress, endAddress);
    }
    return addressesRange;
}

char* Tensor::getData() const
{
    if (getHostAliasTensor() != nullptr)
    {
        HB_ASSERT(m_data == nullptr, "Expected host aliases not to have data of their own");
        char* data = getHostAliasTensor()->getData();
        return data + getHostAliasOffset();
    }
    return m_data;
}

void Tensor::setAsSliceSubTensor(const TensorPtr& aliasTensor, uint64_t byteOffset, const TStride* strides)
{
    _updateAliasDeviceInfo(aliasTensor, ALIAS_TENSOR_TYPE_SLICE, byteOffset);

    memcpy(m_strides, strides, sizeof(m_strides));
}

void Tensor::setAsConcatSubTensor(const TensorPtr& aliasTensor, uint64_t byteOffset, unsigned concatDim)
{
    if (_isAliasedTensor())
    {
        //Concat is the one case where we allow alias composition if the original tensor is dense
        HB_ASSERT(isDenseLayout(), "Unsupported alias combination");
        m_aliasInfo.pAliasedTensor->setAsConcatSubTensor(aliasTensor, byteOffset, concatDim);
        //Todo: is it better to keep pointing to my parent or point to my new alias?
        resetAliasing();
    }
    _updateAliasDeviceInfo(aliasTensor, ALIAS_TENSOR_TYPE_CONCAT, byteOffset);

    //Set strides to reflect strides as the alias tensor
    for (unsigned dim = 0; dim < m_shape.getDim() + 1; ++dim)
    {
        m_strides[dim] = aliasTensor->getStrideInBytes(dim);
    }
}

void Tensor::setAsFlattenSubTensor(const TensorPtr& aliasTensor, bool isInput, unsigned axis)
{
    _updateAliasDeviceInfo(aliasTensor, ALIAS_TENSOR_TYPE_FLAT);

    if (isInput)
    {
        std::shared_ptr<Tensor> validPtr = std::make_shared<Tensor>(*this);
        aliasTensor->validateFlattenSubTensor(validPtr, axis);
    }
    else
    {
        validateFlattenSubTensor(aliasTensor, axis);
    }
}

void Tensor::validateFlattenSubTensor(const TensorPtr& aliasTensor, unsigned axis)
{
    bool axisValidation = axis < aliasTensor->getDim();
    bool isAliasDense   = aliasTensor->isDenseLayout();

    if (!axisValidation || !isAliasDense)
    {
        LOG_ERR(GC,
                "{}: tensor={}, axis={}, dim={}, isAliasDense={}",
                HLLOG_FUNC,
                getName(),
                axis,
                aliasTensor->getDim(),
                isAliasDense);
        HB_ASSERT(false, "validateFlattenSubTensor: precondition validation failed");
    }

    /* Compute the number of elements until axis (including)*/
    uint64_t left = aliasTensor->getDenseStrideInElements(axis + 1);
    /* Compute the number of elements after axis */
    uint64_t right = aliasTensor->getDenseSizeInElements() / left;

    if (left != getSizeInElements(0))
    {
        LOG_ERR(GC,
                "{}: tensor={} Mismatch size until axis, between tensor and its alias flatten-tensor",
                HLLOG_FUNC,
                getName());
        LOG_ERR(GC, "{}: left={} expected={}", HLLOG_FUNC, left, getSizeInElements(0));
        HB_ASSERT(false, "validateFlattenSubTensor: size mismatch");
    }

    if (right != getSizeInElements(1))
    {
        LOG_ERR(GC,
                "{}: tensor={} Mismatch size from axis, between tensor and its alias flatten-tensor",
                HLLOG_FUNC,
                getName());
        LOG_ERR(GC, "{}: right={} expected={}", HLLOG_FUNC, right, getSizeInElements(1));
        HB_ASSERT(false, "validateFlattenSubTensor: size mismatch");
    }
}

void Tensor::setAsHostAliasedSubTensor(const TensorPtr& aliasTensor, uint64_t byteOffset)
{
    if (m_hostAliasInfo.m_parentTensor != nullptr)
    {
        LOG_WARN(GC,
                 "{} replacing host alias from {} to {}",
                 m_name,
                 m_hostAliasInfo.m_parentTensor->getName(),
                 aliasTensor->getName());
    }
    if (getData() != nullptr)
    {
        LOG_WARN(GC, "Setting a bound tensor as alias");
        unbind();
    }
    m_hostAliasInfo.m_parentTensor = aliasTensor;
    m_hostAliasInfo.m_byteOffset   = byteOffset;
    m_shouldFreeBuffer             = false;
}

void Tensor::setDramAllocatedTensor(const TensorPtr& parentDramTensor, uint64_t byteOffset)
{
    HB_ASSERT(isStaticParam(), "DRAM allocated tensor is not a static tensors");

    if (getDramAllocatedTensor() != nullptr)
    {
        LOG_WARN(GC,
                 "{} replacing dram allocated from {} to {}",
                 m_name,
                 m_dramAllocatedTensor.m_parentTensor->getName(),
                 parentDramTensor->getName());
    }

    m_dramAllocatedTensor.m_parentTensor = parentDramTensor;
    m_dramAllocatedTensor.m_byteOffset   = byteOffset;
}

void Tensor::resetAliasing()
{
    m_aliasInfo = AliasTensor();
    resetStrides();
}

bool Tensor::isDenseLayout(unsigned lastDim) const
{
    if (m_type == syn_type_na || lastDim == 0 || isZeroSizedDataTensor())
    {
        return true;
    }
    const auto& sizes = getAllNSizesInElements();

    auto isDense = [this, lastDim](const auto& sizes) {
        TSize firstStride = sizes[0];
        if (!isType4Bit())  // 4bit sizes are handled below
        {
            firstStride *= getElementSizeInBytes();
        }
        // The last stride is always the size of the tensor, there is no reason to check it
        if (firstStride != m_strides[1]) return false;
        for (unsigned dim = 1; dim < lastDim; ++dim)
        {
            if (sizes[dim] * m_strides[dim] != m_strides[dim + 1]) return false;
        }
        return true;
    };

    if (!isCondensed4Bit())
    {
        return isDense(sizes);
    }
    else
    {
        // if the dim is condensed, we must adjust the stride
        auto           condensedSizes = sizes;
        const unsigned condensedIndex = m_annotation.info4Bit.condensedDim;
        if (condensedIndex < lastDim)
        {
            condensedSizes[condensedIndex] /= 2;
        }
        return isDense(condensedSizes);
    }
}

bool Tensor::isDenseLayout() const
{
    if (isZeroSizedDataTensor())
    {
        return true;
    }

    unsigned lastValidSizeIndex = 0;
    for (int i = (int)m_shape.getDim() - 1; i >= 0; --i)
    {
        if (m_shape.getSize(i) != 1)
        {
            lastValidSizeIndex = i;
            break;
        }
    }

    return isDenseLayout(lastValidSizeIndex);
}

ReductionInfo Tensor::getRealReductionInfo(bool checkSetOp) const
{
    bool          isReductionEnabled = false;
    ReductionInfo info               = {};
    const Tensor* aliasedTensor      = this;
    // loop until we found an alias node with isReductionEnabled=true or no more aliases in the chain
    while (!isReductionEnabled && aliasedTensor != nullptr)
    {
        info               = aliasedTensor->getTensorAnnotation().tensorReductionInfo;
        bool isSet         = ReductionInfo::isReductionSet(info.reductionOperation);
        isReductionEnabled = info.isReductionEnabled || (checkSetOp && isSet);
        aliasedTensor = aliasedTensor->isAliasedTensor() ? aliasedTensor->getAliasTensor().get() : nullptr;
    }

    return info;
}

bool Tensor::isReductionEnabled(bool checkSetOp) const
{
    auto info = getRealReductionInfo(checkSetOp);
    if (checkSetOp)
    {
        return info.isReductionEnabled;
    }
    else  // check only real RMW reduction HW operations
    {
        return info.isReductionEnabled && ReductionInfo::isRMWReduction(info.reductionOperation);
    }
}

bool Tensor::compareGeometry(const Tensor& o) const
{
    unsigned dim = std::max(getDim(), o.getDim());
    const auto& sizes      = getAllNSizesInElements();
    const auto& otherSizes = o.getAllNSizesInElements();
    return std::equal(sizes.begin(), sizes.begin() + dim, otherSizes.begin());
}

bool Tensor::compareGeometryWithTranspose(const Tensor& o, const TransposePermutationArray& transposeParams) const
{
    unsigned dim = std::max(getDim(), o.getDim());
    const auto& sizes      = getAllNSizesInElements();
    const auto& otherSizes = o.getAllNSizesInElements();
    for (unsigned d = 0; d < dim; ++d)
    {
        if (sizes[transposeParams[d]] != otherSizes[d]) return false;
    }
    return true;
}

bool Tensor::operator==(const Tensor& o) const
{
    if (&o == this)
    {
        return true;
    }
    //Tensors are equal if they have the same data type and the same geometry
    if (getElementType() != o.getElementType()) return false;
    return compareGeometry(o);
}

void Tensor::AllocateHostMemory()
{
    HB_ASSERT(getData() == nullptr, "should be nullptr");
    if (m_hostAliasInfo.m_parentTensor != nullptr)
    {
        m_hostAliasInfo.m_parentTensor->AllocateHostMemory();
    }
    else
    {
        m_data             = new char[m_deviceSizeInBytes];
        m_shouldFreeBuffer = true;
    }
}

void Tensor::_updateAliasDeviceInfo(const TensorPtr& aliasTensor, AliasTensorType aliasType, uint64_t byteOffset)
{
    HB_ASSERT(!_isAliasedTensor() || (m_aliasInfo.pAliasedTensor == aliasTensor &&
                                      m_aliasInfo.aliasTensorType == aliasType && m_aliasInfo.byteOffset == byteOffset),
              "Tensor is already set as a different alias");

    m_aliasInfo.pAliasedTensor  = aliasTensor;
    m_aliasInfo.aliasTensorType = aliasType;
    m_aliasInfo.byteOffset      = byteOffset;
}

void Tensor::cloneAliasInfo(const TensorPtr& other, const TensorPtr& alias /*= nullptr*/)
{
    _updateAliasDeviceInfo(alias == nullptr ? other->getAliasTensor() : alias,
                           other->_getAliasedTensorType(),
                           other->_getAliasedByteOffset());
    memcpy(m_strides, other->m_strides, sizeof(TStride) * Tensor::c_numOfNStrides);
}

void Tensor::_debugPrintAliasInfo() const
{
    LOG_DEBUG(GC, "  AliasInfo: {}", (void*) &m_aliasInfo);
    LOG_DEBUG(GC, "      AliasedTensor: {}", (void*)m_aliasInfo.pAliasedTensor.get());
    LOG_DEBUG(GC, "      Bytes offset: {}", m_aliasInfo.byteOffset);
}

void Tensor::cloneHostAliasInfo(const TensorPtr& other, const TensorPtr& alias /*= nullptr */)
{
    m_hostAliasInfo.m_byteOffset   = other->getHostAliasOffset();
    m_hostAliasInfo.m_parentTensor = alias != nullptr ? alias : other->getHostAliasTensor();
}

uint64_t Tensor::_getDistanceToElementInBytes(uint64_t elementNum) const
{
    TOffset elementIndex[Tensor::c_tensorMaxNDim];
    findIndex(m_shape.getSizes().data(), m_shape.getDim(), elementNum, elementIndex);

    uint64_t distanceInBytes = 0;
    for (unsigned int dim = 0; dim < m_shape.getDim(); ++dim)
    {
        distanceInBytes += m_strides[dim] * elementIndex[dim];
    }
    return distanceInBytes;
}

void Tensor::debugPrint() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;

    TensorAnnotation ann = getTensorAnnotation();
    LOG_DEBUG(GC, "  {}, {}", getName(), (void*) this);
    LOG_DEBUG(GC, "      Number of dimensions: {}", m_shape.getDim());
    auto sizes = m_shape.getNSizes();
    LOG_DEBUG(GC, "      Sizes: {}", toString(sizes.begin(), sizes.begin() + getDim(), ','));
    LOG_DEBUG(GC, "      Strides: {}", toString(m_strides, m_strides + getDim() + 1, ','));
    LOG_DEBUG(GC, "      isWeight {}, isBias {}, isStaticParams {}", isWeights(), m_isBias, isStaticParam());
    LOG_DEBUG(GC, "      Host Address: {}", getAddress());
    if (!isShapeTensor())
    {
        LOG_DEBUG(GC, "      Tensor is in {}", tensorAllocatedInSram() ? "SRAM" : "DRAM");
        LOG_DEBUG(GC, "      {} Offset: 0x{:x}", tensorAllocatedInSram() ? "SRAM" : "DRAM", getTensorOffset());
    }
    LOG_DEBUG(GC, "      Size in bytes: {}", getTotalSizeInBytes());
    LOG_DEBUG(GC, "      Data Type: {}", getStringFromSynDataType(m_type));
    LOG_DEBUG(GC, "      Memory ID: {}", m_memSectionId);
    LOG_DEBUG(GC, "      Quantization: ");
    for (auto quantization : m_quantizationParamsMap)
    {
        LOG_DEBUG(GC,
                  "        Data type = {}, scale={}, zp={}",
                  getStringFromSynDataType(QuantizationData::quantTypeToSynType((eQuantDataType)quantization.first)),
                  quantization.second.scale(),
                  quantization.second.zp());
    }
    LOG_DEBUG(GC, "      {}", ann.memorySpaceInfo.prefetchInfo.prefetch ? "Prefetched" : "Not-Prefetched");
    if (isAliasedTensor())
    {
        _debugPrintAliasInfo();
    }
    if (hasDramAllocatedTensor())
    {
        LOG_DEBUG(GC,
                  "      hasDramAllocatedTensor: {}, offset {}",
                  getDramAllocatedTensor()->getName(),
                  getDramAllocatedTensorOffset());
    }
}

void Tensor::setConnectAtomicNodes()
{
    LOG_TRACE(GC, "{} tensor was set as connecting two atomic nodes", getName());
    m_annotation.connectAtomicNodes = true;
}


void Tensor::setQuantizationParams(const synQuantMetadata& params)
{
    HB_ASSERT(params.dataType != syn_type_na,
              "Trying to set tensor quantization params for invalid data type {}",
              params.dataType);
    HB_ASSERT_PTR(params.zpScales);

    QuantizationData quantData(params);
    m_quantizationParamsMap[QuantizationData::synTypeToQuantType(params.dataType)] = quantData;
}

void Tensor::setQuantizationParams(const synFpQuantMetadata& params)
{
    HB_ASSERT(params.dataType != syn_type_na,
              "Trying to set tensor quantization params for invalid data type {}",
              params.dataType);
    HB_ASSERT_PTR(params.fpQuantParams);

    for (int i = 0; i < params.numFpQuantParams; i++)
    {
        HB_ASSERT(QuantizationUtils::expBiasIsInRangeOfType(params.fpQuantParams[i].expBias, params.dataType),
                  "expBias: {} is not supported for data type {}",
                  params.fpQuantParams[i].expBias,
                  params.dataType);
    }

    QuantizationData quantData(params);
    m_quantizationParamsMap[QuantizationData::synTypeToQuantType(params.dataType)] = quantData;
}

void Tensor::setQuantizationParams(const synPerChannelQuantizationParams& params)
{
    HB_ASSERT(params.m_qDataType != syn_type_na,
              "Trying to set tensor quantization params for invalid data type {}",
              params.m_qDataType);

    for (int i = 0; i < params.m_numChannels; i++)
    {
        HB_ASSERT(QuantizationUtils::expBiasIsInRangeOfType(params.m_pcExpBias[i], params.m_qDataType),
                  "expBias: {} is not supported for data type {}",
                  params.m_pcExpBias[i],
                  params.m_qDataType);
    }

    QuantizationData quantData(params);
    m_quantizationParamsMap[QuantizationData::synTypeToQuantType(params.m_qDataType)] = quantData;
}

void Tensor::setQuantizationParams(const synQuantizationParams& params)
{
    HB_ASSERT(params.m_qDataType != syn_type_na,
              "Trying to set tensor quantization params for invalid data type {}",
              params.m_qDataType);
    HB_ASSERT(QuantizationUtils::expBiasIsInRangeOfType(params.m_expBias, params.m_qDataType),
              "expBias: {} is not supported for data type {}",
              params.m_expBias,
              params.m_qDataType);
    QuantizationData quantData(params);
    m_quantizationParamsMap[QuantizationData::synTypeToQuantType(params.m_qDataType)] = quantData;
}

void Tensor::setQuantizationParams(const QuantizationData& params)
{
    HB_ASSERT(params.getSynDataType() != syn_type_na,
              "Trying to set tensor quantization params for invalid data type {}",
              params.getSynDataType());

    m_quantizationParamsMap[params.m_qDataType] = params;
}

void Tensor::setAllQuantizationParams(const QuantizationMap& params)
{
    for (const std::pair<const uint32_t, QuantizationData>& quantization : params)
    {
        m_quantizationParamsMap[quantization.first] = quantization.second;
    }
}

bool Tensor::setDynamicRange(DynamicRange dynamicRange)
{
    if (dynamicRange.max >= dynamicRange.min)
    {
        m_dynamicRange = dynamicRange;
        return true;
    }
    LOG_TRACE(GC, "Dynamic range wasn't set: max[{}] < min[{}]", dynamicRange.max, dynamicRange.min);
    return false;
}

bool Tensor::setPerChannelDynamicRange(const synPerChannelDynamicRange& perChannelDynamicRange)
{
    m_perChannelDynamicRange.numChannels = perChannelDynamicRange.numChannels;

    for (unsigned i = 0; i < perChannelDynamicRange.numChannels; i++)
    {
        if (perChannelDynamicRange.ranges[i].max < perChannelDynamicRange.ranges[i].min)
        {
            LOG_TRACE(GC,
                      "Per channel dynamic range wasn't set. "
                      "Ranges in channel {} are invalid: max[{}] < min[{}]",
                      i,
                      perChannelDynamicRange.ranges[i].max,
                      perChannelDynamicRange.ranges[i].min);
            return false;
        }
        m_perChannelDynamicRange.ranges.push_back({perChannelDynamicRange.ranges[i].min, perChannelDynamicRange.ranges[i].max});
    }
    return true;
}

bool Tensor::setPerChannelDynamicRange(const PerChannelDynamicRange& perChannelDynamicRange)
{
    HB_ASSERT(perChannelDynamicRange.numChannels == perChannelDynamicRange.ranges.size(),
              "mismatch between number of channels and the ranges vector size in tensor {}",
              getName());

    for (unsigned i = 0; i < perChannelDynamicRange.numChannels; i++)
    {
        if (perChannelDynamicRange.ranges[i].max < perChannelDynamicRange.ranges[i].min)
        {
            LOG_TRACE(GC,
                      "Per channel dynamic range wasn't set. "
                      "Ranges in channel {} are invalid: max[{}] < min[{}]",
                      i,
                      perChannelDynamicRange.ranges[i].max,
                      perChannelDynamicRange.ranges[i].min);
            return false;
        }
    }
    m_perChannelDynamicRange = perChannelDynamicRange;
    return true;
}

double Tensor::getZeroPoint(unsigned index) const
{
    auto it = m_quantizationParamsMap.find(QuantizationData::synTypeToQuantType(m_type));
    if (it == m_quantizationParamsMap.end())
    {
        return 0;
    }
    return it->second.zp(index);
}

double Tensor::getZeroPoint(synDataType type, unsigned index) const
{
    auto it = m_quantizationParamsMap.find(QuantizationData::synTypeToQuantType(type));
    if (it == m_quantizationParamsMap.end())
    {
        HB_ASSERT(false,
                  "Trying to get zp at dtype {} with non existing quant params for tensor {}",
                  getStringFromSynDataType(type),
                  getName());
        return 0;
    }
    return it->second.zp(index);
}

double Tensor::getScale(unsigned index) const
{
    auto it = m_quantizationParamsMap.find(QuantizationData::synTypeToQuantType(m_type));
    if (it == m_quantizationParamsMap.end())
    {
        return 1.0;
    }
    return it->second.scale(index);
}

double Tensor::getScale(synDataType type, unsigned index) const
{
    auto it = m_quantizationParamsMap.find(QuantizationData::synTypeToQuantType(type));
    if (it == m_quantizationParamsMap.end())
    {
        HB_ASSERT(false,
                  "Trying to get scale at dtype {} with non existing quant params for tensor {}",
                  getStringFromSynDataType(type),
                  getName());
        return 1.0;
    }
    return it->second.scale(index);
}

void Tensor::setScale(double newScale, unsigned index)
{
    HB_ASSERT(m_quantizationParamsMap.count(QuantizationData::synTypeToQuantType(m_type)) != 0,
              "Trying to set scale at dtype {} with non existing quant params for tensor {}",
              getStringFromSynDataType(m_type),
              getName());
    m_quantizationParamsMap.at(QuantizationData::synTypeToQuantType(m_type)).setScale(newScale, index);
}

bool Tensor::setScaleForDtype(double newScale, synDataType dtype /*= syn_type_fp8_143*/)
{
    if (m_quantizationParamsMap.count(QuantizationData::synTypeToQuantType(dtype)) == 0)
    {
        LOG_WARN(QUANT,
                 "{}: Trying to set scale at dtype {} with non existing quant params for tensor {},",
                 HLLOG_FUNC,
                 getStringFromSynDataType(dtype),
                 getName());
        return false;
    }
    m_quantizationParamsMap.at(QuantizationData::synTypeToQuantType(dtype)).setScale(newScale);
    return true;
}

unsigned Tensor::getExpBias(unsigned index) const
{
    auto it = m_quantizationParamsMap.find(QuantizationData::synTypeToQuantType(m_type));
    if (it == m_quantizationParamsMap.end())
    {
        return QuantizationData::getDefaultExpBias(m_type);
    }
    return it->second.expBias(index);
}

unsigned Tensor::getExpBias(synDataType type, unsigned index) const
{
    HB_ASSERT(type == syn_type_fp8_143 || type == syn_type_fp8_152,
              "Trying to get exponent bias at dtype {} for tensor {}, dtype expected to be fp8 variant",
              getStringFromSynDataType(type),
              getName());
    auto it = m_quantizationParamsMap.find(QuantizationData::synTypeToQuantType(type));
    if (it == m_quantizationParamsMap.end())
    {
        HB_ASSERT(false,
                  "Trying to get exponent bias at dtype {} with non existing quant params for tensor {}",
                  getStringFromSynDataType(type),
                  getName());
        return QuantizationData::getDefaultExpBias(type);
    }
    return it->second.expBias(index);
}

void Tensor::setExpBias(unsigned newExpBias, unsigned index)
{
    HB_ASSERT(m_quantizationParamsMap.count(QuantizationData::synTypeToQuantType(m_type)) != 0,
              "Trying to set exponent bias at dtype {} with non existing quant params for tensor {}",
              getStringFromSynDataType(m_type),
              getName());
    m_quantizationParamsMap.at(QuantizationData::synTypeToQuantType(m_type)).setExpBias(newExpBias, index);
}

bool Tensor::setExpBiasForDtype(unsigned newExpBias, synDataType dtype /*= syn_type_fp8_143*/)
{
    if (m_quantizationParamsMap.count(QuantizationData::synTypeToQuantType(dtype)) == 0)
    {
        LOG_WARN(QUANT,
                 "{}: Trying to set exponent bias at dtype {} with non existing quant params for tensor {},",
                 HLLOG_FUNC,
                 getStringFromSynDataType(dtype),
                 getName());
        return false;
    }
    m_quantizationParamsMap.at(QuantizationData::synTypeToQuantType(dtype)).setExpBias(newExpBias);
    return true;
}

void Tensor::setZeroPoint(double newZeroPoint, unsigned index)
{
    HB_ASSERT(m_quantizationParamsMap.count(QuantizationData::synTypeToQuantType(m_type)) != 0,
              "Trying to set zp at dtype {} with non existing quant params for tensor {}",
              getStringFromSynDataType(m_type),
              getName());
    m_quantizationParamsMap.at(QuantizationData::synTypeToQuantType(m_type)).setZp(newZeroPoint, index);
}

const QuantizationData& Tensor::getQuantizationParams(synDataType dataType) const
{
    eQuantDataType quantDataType = QuantizationData::synTypeToQuantType(dataType);
    return getQuantizationParams(quantDataType);
}

const QuantizationData& Tensor::getQuantizationParams(eQuantDataType dataType) const
{
    auto iter = m_quantizationParamsMap.find(dataType);
    if (iter == m_quantizationParamsMap.end())
    {
        // return defaults (scale=1, zp=0, expBias=EXP_BIAS_143_DEFAULT) for unquantized types
        return QuantizationData::defaultQuantizationData[dataType];
    }
    return iter->second;
}

bool Tensor::isQuantizationParamsExist(synDataType dataType) const
{
    eQuantDataType quantDataType = QuantizationData::synTypeToQuantType(dataType);
    return m_quantizationParamsMap.count(quantDataType) != 0;
}

std::string Tensor::getAliasedTensorTypeStr() const
{
    switch (m_aliasInfo.aliasTensorType)
    {
        case ALIAS_TENSOR_TYPE_NONE:
            return "";
        case ALIAS_TENSOR_TYPE_ALIAS:
            return "alias";
        case ALIAS_TENSOR_TYPE_CONCAT:
            return "concat";
        case ALIAS_TENSOR_TYPE_FLAT:
            return "flat";
        case ALIAS_TENSOR_TYPE_SLICE:
            return "slice";
        default:
            return "";
    }
}

std::shared_ptr<Tensor> Tensor::clone(bool                  copyAddresses /*false*/,
                                      bool                  copyData /*true*/,
                                      bool                  keepPersistent /*false*/,
                                      TensorNameClonePolicy namePolicy) const
{
    std::shared_ptr<Tensor> pClone = std::make_shared<Tensor>(*this, copyAddresses, copyData, namePolicy);

    if (!keepPersistent)
    {
        // Cloned tensor should not be persistent and should be created in workspace
        pClone->setAsNonUserManaged();
        pClone->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
        pClone->m_outputState                                 = IOClassification::AUTO;
        pClone->getTensorAnnotation().memory.allowPermutation = false;
    }

    // the new tensor has no one aliased to him, so make sure to reset this annotation.
    pClone->setIsRealInLogical(false);
    pClone->setIsRealInAliasing(false);
    pClone->getTensorAnnotation().tensorReductionInfo = ReductionInfo();
    pClone->getTensorAnnotation().m_preSlicingSize.reset();

    return pClone;
}

std::shared_ptr<Tensor> Tensor::cloneGeometry() const
{
    std::shared_ptr<Tensor> pClone = std::make_shared<Tensor>(m_shape, m_type);
    pClone->setBatchPos(m_batchPos);

    if (isPropSet(synTensorPropGeometryMin))
    {
        pClone->setProp(synTensorPropGeometryMin);
    }
    if (isPropSet(synTensorPropGeometryMax))
    {
        pClone->setProp(synTensorPropGeometryMax);
    }
    if (isPropSet(synTensorPropGeometryDim))
    {
        pClone->setProp(synTensorPropGeometryDim);
    }
    pClone->getTensorAnnotation().m_preSlicingSize.reset();

    return pClone;
}

std::string Tensor::getDimSizesStr(bool isFCDLast, bool minSizes) const
{
    unsigned          dims  = m_shape.getDim();
    const NSizeArray& sizes = minSizes ? m_shape.getNMinSizes() : m_shape.getNSizes();
    return "[" +
           (isFCDLast ? toString(sizes.rend() - dims, sizes.rend(), ',')
                      : toString(sizes.begin(), sizes.begin() + dims, ',')) +
           "]";
}

std::string Tensor::getStridesStr(bool isFCDLast) const
{
    return getDimStr(m_strides, m_shape.getDim() + 1, isFCDLast);
}

void Tensor::mergeTensorsAttributes(const pTensor& otherTensor)
{
    /* Batch Position */
    unsigned baseBatchPos  = getBatchPos();
    unsigned otherBatchPos = otherTensor->getBatchPos();
    if (baseBatchPos != INVALID_BATCH_POS && otherBatchPos != INVALID_BATCH_POS)
    {
        HB_ASSERT(baseBatchPos == otherBatchPos, "Batch position should be the same");
    }
    else if (otherBatchPos != INVALID_BATCH_POS)
    {
        setBatchPos(otherBatchPos);
    }

    /* Model param */
    if (m_isBias || otherTensor->m_isBias) setAsBias();
    if (isWeights() || otherTensor->isWeights()) setAsWeights();
    if (isStaticParam() || otherTensor->isStaticParam()) setAsStaticParam();

    /* Output state */
    if (isEnforcedOutput() || otherTensor->isEnforcedOutput())
    {
        enforceOutput();
    }
    else if (isMaskedOutput() || otherTensor->isMaskedOutput())
    {
        maskOutput();
    }
}

void Tensor::setUserContext(const TensorExUserContext& userContext)
{
    m_annotation.userContext = userContext;

    if (userContext.strides)
    {
        auto sizes = getNSizesInElements();
        TStride tensorStrides[c_numOfNStrides] = {0};
        memcpy(tensorStrides + 1, userContext.strides, getDim() * sizeof(TStride));
        tensorStrides[0] = getElementSizeInBytes();
        HB_ASSERT(!isStridedOnFCD(), "user context for strided fcd tensor");
        reshape(getDim(), sizes.data(), tensorStrides);
    }
}

void Tensor::setShapeTensor(synTensorType shapeTensorType)
{
    HB_ASSERT(shapeTensorType == OUTPUT_DESCRIBING_SHAPE_TENSOR || shapeTensorType == INPUT_DESCRIBING_SHAPE_TENSOR ||
                  shapeTensorType == HOST_SHAPE_TENSOR,
              "Set tensor as shape tensor with unexpected tensor type");
    m_tensorType = shapeTensorType;
}

void Tensor::setAsNonUserManaged()
{
    m_memoryDescriptor.m_isPersistent = false;
    m_isExternal                      = false;
    if (!getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.is_set())
    {
        getTensorAnnotation().nonPersistentSectionInfo.sectionId.unset();
        getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.unset();
    }
}

void Tensor::lockQuantization(NodePtr lockingNode)
{
    HB_ASSERT(m_quantizationParamsMap.size() > 0, "quantization params map is empty");
    HB_ASSERT(lockingNode != nullptr, "lockingNode is nullptr");
    LOG_TRACE(QUANT, "Tensor {} quantization is locked by {}", getName(), lockingNode->getNodeName());
    m_quantizationLockingNodes.insert(lockingNode->getId());
}

void Tensor::requantLock(NodePtr lockingNode)
{
    m_isRequantLocked = true;
    m_quantizationLockingNodes.clear();
    // TODO: [SW-82762] Remove unnecessary lockingNode parameter
    if (lockingNode == nullptr)  // if node doesn't exist, i.e., tensor is a graph input - use dummy node ID
    {
        m_quantizationLockingNodes.insert(0);
    }
    else
    {
        lockQuantization(lockingNode);
    }
    LOG_TRACE(QUANT, "Tensor {} quantization is locked for requant", getName());
}

void Tensor::saveConflicts()
{
    for (const synNodeId& lockingNode : m_quantizationLockingNodes)
    {
        m_conflictQuants[lockingNode] = m_quantizationParamsMap;
    }
}

void Tensor::saveConflicts(NodePtr node, QuantizationMap& quantizationMap)
{
    for (const synNodeId& lockingNode : m_quantizationLockingNodes)
    {
        m_conflictQuants[lockingNode] = m_quantizationParamsMap;
    }

    m_conflictQuants[node->getId()] = quantizationMap;
}

void Tensor::revertQuantization()
{
    if (m_measuredQuantizationMap.empty())
    {
        LOG_WARN(QUANT, "{}: Empty measured quantization map for tensor: {}", HLLOG_FUNC, getName());
        return;
    }

    setAllQuantizationParams(m_measuredQuantizationMap);
}

void Tensor::setMeasuredQuantizationParams(QuantizationMap& measuredQuantizationMap)
{
    setAllQuantizationParams(measuredQuantizationMap);
    m_measuredQuantizationMap = measuredQuantizationMap;
}

void Tensor::resetQuantization()
{
    m_quantizationParamsMap.clear();
    m_measuredQuantizationMap.clear();
    m_quantizationLockingNodes.clear();
    m_conflictQuants.clear();
    m_isRequantLocked = false;
}

void Tensor::enforceInt8FixedPoint()
{
    if (m_quantizationParamsMap.find(quant_type_int8fp) == m_quantizationParamsMap.end())
    {
        LOG_WARN(GC, "enforceInt8FixedPoint: int8fp quantization does not exist for tensor {}", getName());
        return;
    }
    m_isInt8FixedPoint                                   = true;
    m_quantizationParamsMap[quant_type_int8]             = m_quantizationParamsMap[quant_type_int8fp];
    m_quantizationParamsMap[quant_type_int8].m_qDataType = quant_type_int8;
}

void Tensor::enforceInt16Limited()
{
    if (m_quantizationParamsMap.find(quant_type_int16ltd) == m_quantizationParamsMap.end())
    {
        LOG_WARN(GC, "enforceInt16Limited: int16ltd quantization does not exist for tensor {}", getName());
        return;
    }
    m_isInt16Limited                                      = true;
    m_quantizationParamsMap[quant_type_int16]             = m_quantizationParamsMap[quant_type_int16ltd];
    m_quantizationParamsMap[quant_type_int16].m_qDataType = quant_type_int16;
}

void Tensor::condenseStridesTo4Bit(CondenseDimIndex condensedDimIndex)
{
    HB_ASSERT(isType4Bit(), "Tensor {} doesn't have 4 bit data type ", getName());
    HB_ASSERT(!isCondensed4Bit(), "Tensor {} already condensed ", getName());

    LOG_TRACE(GC, "{} for tensor {}, condensed dim: {}", HLLOG_FUNC, getName(), condensedDimIndex);

    for (unsigned i = condensedDimIndex + 1; i < c_numOfNStrides; i++)
    {
        m_strides[i] /= 2;
    }

    // update relevant members
    TStride maxStride   = getMaxStride();
    m_deviceSizeInBytes = maxStride;
    // need to save the dim for future use (for example sizes manipulation - getAllSizesInElementsCondensed function)
    m_annotation.info4Bit.condensedDim = condensedDimIndex;
    m_annotation.info4Bit.isCondensed  = true;
}

void Tensor::matchSizesTo4Bit(TSize* sizes, unsigned count) const
{
    HB_ASSERT(getDim() <= count, "This function is deprecated for dimension {}", getDim());
    HB_ASSERT(isCondensed4Bit(), "Tensor {} is not yet condensed", getName());
    HB_ASSERT(m_annotation.info4Bit.condensedDim != UNINITIALIZED_CONDENSED_DIM,
              "Tensor {} was condensed but dimension wasn't updated", getName());

    LOG_TRACE(GC, "{} for tensor {}, condensed dim: {}", HLLOG_FUNC, getName(), m_annotation.info4Bit.condensedDim);

    CondenseDimIndex condensedDim = m_annotation.info4Bit.condensedDim;
    HB_ASSERT(sizes[condensedDim] % 2 == 0, "Size of dim {} must be even when data type is 4 bit", condensedDim);
    sizes[condensedDim] /= 2;
}

// BHW == 1 and the depth == other.depth
bool Tensor::is1DAndSameDepthAsOther(const TensorPtr& other) const
{
    const SizeArray& sizes = getAllSizesInElements();

    return (getSizeInElements(0) == other->getSizeInElements(0)) &&
           (multiplyElements(sizes.begin() + 1, sizes.end()) == 1);
}

bool Tensor::isPropsValid()
{
    if (!(m_propsSet & synTensorPropGeometryDim))
    {
        LOG_ERR(GC, "tensor {} does not have geometry", m_name);
        return false;
    }

    if (!(m_propsSet & synTensorPropHostPtr) && isStaticParam())
    {
        LOG_ERR(GC, "tensor {} is static but lack a host pointer", m_name);
        return false;
    }

    if ((m_propsSet & synTensorPropSection) && (!isAssignedToConstSection()) && (m_propsSet & synTensorPropHostPtr))
    {
        LOG_ERR(GC, "tensor {} cannot have host pointer and a non-const section at the same time", m_name);
        return false;
    }
    /* TODO SW-35198 restore below validation after solve failing tests mentioned in JIRA
    if ((m_propsSet & synTensorPropHostPtr) &&
        (getBufferSizeInBytes() != getDenseSizeInElements() * getDataTypeSizeInBytes(getBufferDataType())))
    {
        LOG_ERR(GC,
                "tensor {} buffer size {} doesn't match expected size {} according to tensor dimensions",
                m_name,
                getBufferSizeInBytes(),
                getDenseSizeInElements() * getDataTypeSizeInBytes(getBufferDataType()))
        return false;
    }
    */

    if (!(m_propsSet & synTensorPropGeometryMax))
    {
        if (isPersistent())
        {
            LOG_ERR(GC, "tensor {} is persistent but synGeometrySizes were not provided", m_name);
            return false;
        }
        if (isModelParameter())
        {
            LOG_ERR(GC, "tensor {} is a model parameter but synGeometrySizes were not provided", m_name);
            return false;
        }
    }

    if (isPersistent() && !(m_propsSet & synTensorPropName))
    {
        LOG_ERR(GC, "tensor {} is persistent but no name is given", m_name);
        return false;
    }

    if (getDim() == 0)
    {
        LOG_ERR(GC, "tensor {} dimension not initialized", m_name);
        return false;
    }

    if (isStaticParam() && !isDataTypeMatchData() && m_bufferDataType != syn_type_float)
    {
        LOG_ERR(GC,
                "tensor {} unsupported buffer data type that does not match device data type (must be float).",
                m_name);
        return false;
    }

    if (isPerChannelQuant() && !isStaticParam())
    {
        LOG_ERR(GC, "tensor {} is quantized per-channel but not specified as static", m_name);
        return false;
    }

    if (isDynamicShape() && (m_tensorType == DATA_TENSOR))
    {
        LOG_WARN(GC, "tensor {} has dynamic sizes but is specified as static - Ignoring dynamicity", m_name);
    }

    if (m_tensorType == DEVICE_SHAPE_TENSOR && !verifyDeviceShapeTensor(m_shape.getDim(),
                                                                        m_shape.getNSizes().data(),
                                                                        m_type,
                                                                        m_name,
                                                                        m_shape.getMinSizes().data()))
    {
        LOG_ERR(SYN_API, "Tensor {} of type DEVICE_SHAPE_TENSOR is invalid.", m_name);
        return false;
    }

    if (m_permutation)
    {
        if (!isPersistent())
        {
            LOG_ERR(GC, "Tensor {} cannot have a permutation unless it is persistent", m_name);
            return false;
        }
        if (m_permutation->size() != getDim())
        {
            LOG_ERR(GC,
                    "Tensor {} has permutation with different number of dims ({}) than its shape ({})",
                    m_name,
                    m_permutation->size(),
                    getDim());
            return false;
        }
    }

    if (m_isExternal && !isPersistent())
    {
        LOG_ERR(SYN_API, "Tensor {} cannot be marked as external event tensor without being persistent.", m_name);
        return false;
    }

    if (getTensorAnnotation().memory.allowPermutation && !isPersistent())
    {
        LOG_ERR(SYN_API, "Tensor {} cannot be marked as allow strided without being persistent", getName());
        return false;
    }

    return true;
}

void Tensor::lockPropsAndFinalizeTensor()
{
    m_lockProps = true;

    // the degenerate strides of a tensor cannot be calculated before it has a geometry
    // so we finalize the tensor strides (as GC's expects them) only when the tensor is locked.
    static constexpr TStride ZEROS[c_numOfNStrides] = {};
    if (memcmp(m_strides, ZEROS, sizeof(m_strides)) == 0)
    {
        setDefaultStrides();
    }
    else
    {
        calculateDegenerateStrides(getDim());
    }
}

bool Tensor::isPartOfRMWSection() const
{
    const auto& sectionInfo = getTensorAnnotation().nonPersistentSectionInfo;
    return inSram() &&
           sectionInfo.sectionId.is_set() &&
           sectionInfo.offsetFromBase.is_set() &&
           !sectionInfo.bufferingLevel.is_set();
}

void Tensor::setSectionHandle(const SlotMapItemSptr<InternalSectionHandle>& sectionHandle)
{
    m_internalSectionHandle = sectionHandle;
}

const SlotMapItemSptr<InternalSectionHandle>& Tensor::getSectionHandle() const
{
    return m_internalSectionHandle;
}

bool Tensor::inConstSection() const
{
    // The code below checked if the sectionPtr is nullptr even before the section handle change,
    // so I assume it is legal not to have a section here. Not logging an error in case sectionPtr is nullptr
    return (m_internalSectionHandle && m_internalSectionHandle->getConst() && getData() &&
            getMemorySectionID() >= MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
}

bool Tensor::isAssignedToConstSection() const
{
    return m_internalSectionHandle && m_internalSectionHandle->getConst();
}

void Tensor::changeDefaultElementType(synDataType type, bool ignoreDataTypeMatch)
{
    if (getElementType() == type) return;

    if (!ignoreDataTypeMatch && isStaticParam() && isDataTypeMatchData())
    {
        LOG_WARN(DATA_TYPES,
                 "quantized tensor {} element type cannot be changed from {} to {}",
                 getName(),
                 getElementType(),
                 type);
        return;
    }

    m_type = type;
    setDefaultStrides();  // changing dtype causes us to modify the strides accordingly
}

bool Tensor::isUserManagedDram() const
{
    if (isPersistent()) return true;
    const auto& sectionInfo = getTensorAnnotation().nonPersistentSectionInfo;
    return inDram() && sectionInfo.sectionId.is_set() && sectionInfo.offsetFromBase.is_set() &&
           !sectionInfo.bufferingLevel.is_set();
}

char* Tensor::getHostMaxData() const
{
    const Tensor* h2dTensor = m_twinHost2DeviceTensor ? m_twinHost2DeviceTensor.get() : this;
    HB_ASSERT(h2dTensor->isHostShapeTensor() || h2dTensor->isHost2DeviceTensor(),
              "Function called for a non-host tensor {}",
              m_name);
    // Max data is at the beginning of data
    return h2dTensor->getData();
}

bool Tensor::hasHostData() const
{
    const Tensor* h2dTensor = m_twinHost2DeviceTensor ? m_twinHost2DeviceTensor.get() : this;
    return (h2dTensor->isHostShapeTensor() || h2dTensor->isHost2DeviceTensor());
}

char* Tensor::getHostMinData() const
{
    const Tensor* h2dTensor = m_twinHost2DeviceTensor ? m_twinHost2DeviceTensor.get() : this;
    HB_ASSERT(h2dTensor->isHostShapeTensor() || h2dTensor->isHost2DeviceTensor(),
              "Function called for a non-host tensor {}",
              m_name);
    // Min data goes immediately after max data
    return h2dTensor->getData() + h2dTensor->getHostDataSize();
}

uint64_t Tensor::getHostDataSize() const
{
    const Tensor* h2dTensor = m_twinHost2DeviceTensor ? m_twinHost2DeviceTensor.get() : this;
    HB_ASSERT(h2dTensor->isHostShapeTensor() || h2dTensor->isHost2DeviceTensor(),
              "Function called for a non-host tensor {}",
              m_name);
    uint64_t hostDataSize = h2dTensor->getTotalElements() * h2dTensor->getElementSizeInBytes();

    // There are two sets of host data, the max set and the min set.
    //
    // This function returns the size of just ONE such set.
    //
    // For a host shape tensor, this additional factor of 2 is
    // indicated by an extra outermost dimension of 2, so the total number
    // of elements includes both sets. To find out the size of one set,
    // we divide the total number of elements by 2.
    //
    // A host2device tensor buffer contains both sets, but there is no additional
    // dimension to indicate that, so the total number of elements includes
    // just one of the sets. Thus we do not divide it by 2.
    //
    // This is a discrepancy between how the shape of host shape tensors and
    // host2device tensors is handled. This probably needs to be unified.
    // TODO An API change is needed for that.

    if (h2dTensor->isHostShapeTensor())
    {
        hostDataSize /= 2;
    }

    return hostDataSize;
}

unsigned Tensor::getIndexOfMaxNonDegenerateStride() const
{
    unsigned tensorDim = getDim();

    if (tensorDim == 0) return 0; // W/A for graph_optimizer unit tests
    if (tensorDim == 1) return 0; // W/A for cguid tests - SW-116605

    const auto& sizes = getAllNSizesInElements();
    tensorDim--;

    if (GCFG_ENABLE_TPC_LAST_DIM_OPT.value())
    {
        // removing upper degenerate dims
        while ((sizes[tensorDim] == 1) && (tensorDim >= 1))
        {
            tensorDim--;
        }
        // finding biggest stride (could be that it's not the last one)
        const TStride* maxStrideInBytes = std::max_element(std::begin(m_strides), std::begin(m_strides) + tensorDim);
        if (m_strides[tensorDim] < *maxStrideInBytes)
        {
            tensorDim = std::distance(std::begin(m_strides), maxStrideInBytes);
        }
    }

    // per PRM - last_dim should be one of {1, 2, 3, 4}
    return std::clamp(tensorDim, 1u, (unsigned)(c_tensorMaxDim - 1));
}

std::string Tensor::getClonedTensorName(const Tensor& t, TensorNameClonePolicy namePolicy)
{
    switch (namePolicy)
    {
        case TensorNameClonePolicy::EMPTY_NAME:
            return "";
        case TensorNameClonePolicy::COPY_NAME:
            return t.m_name;
        case TensorNameClonePolicy::DEFUALT_NAME:
        default:
            return fmt::format("{}_c{}", t.m_name, m_id);
    }
}
