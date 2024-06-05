#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "infra/containers/slot_map_alloc.hpp"
#include "infra/type_utils.h"
#include "layout.h"
#include "quantization_data.h"
#include "section_handle.hpp"
#include "synapse_types.h"
#include "tensor_annotation.h"
#include "tensor_shape.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

#define GET_REAL_TENSOR_IF_NULL_CONTINUE(TENSOR)                                                                       \
    if (TENSOR == nullptr)                                                                                             \
    {                                                                                                                  \
        continue;                                                                                                      \
    }                                                                                                                  \
    TENSOR = Tensor::getRealTensor(TENSOR);

#define GET_REAL_TENSOR_IF_NULL_RETURN_VAL(TENSOR, RET)                                                                \
    if (TENSOR == nullptr)                                                                                             \
    {                                                                                                                  \
        return RET;                                                                                                    \
    }                                                                                                                  \
    TENSOR = Tensor::getRealTensor(TENSOR);

#define GET_REAL_TENSOR_IF_NULL_RETURN(TENSOR)                                                                         \
    if (TENSOR == nullptr)                                                                                             \
    {                                                                                                                  \
        return;                                                                                                        \
    }                                                                                                                  \
    TENSOR = Tensor::getRealTensor(TENSOR);


static constexpr unsigned int CONV_3D_TENSOR_DIM = 5;
// Crossing the following limit requires special slicing to fullfil HW requirements
static constexpr unsigned int HW_DENSE_TENSOR_LIMIT = (uint32_t)-1;

// the slowest changing dimension is in the highest index in the array.
// Indices always start at 'm_sizes' array index 0 and end according to the order of the tensor.
// e.g. sizes=[9,9, 0,0] represents a 9*9 matrix.

typedef enum
{
    DIM_C = 0,
    DIM_W,
    DIM_H,
    DIM_B,
    DIM_D_FOR_5D_TENSOR = 3,
    DIM_B_FOR_5D_TENSOR = 4,
    DIM_GEMM_BATCH = 2
} DimsIndex;

typedef enum
{
    WEIGHT_DIM_K = 0,
    WEIGHT_DIM_C,
    WEIGHT_DIM_S,
    WEIGHT_DIM_R,
    WEIGHT_DIM_Q
} WeightDimsIndex;

enum TensorSemanticType
{
    TYPE_ACTIVATION,
    TYPE_WEIGHTS,
    TYPE_BIAS
};

enum TensorUsage
{
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    INTERMEDIATE_TENSOR
};

enum class TensorNameClonePolicy : uint8_t
{
    EMPTY_NAME,
    COPY_NAME,
    DEFUALT_NAME
};

// Opaque member type used by the Graph
typedef std::shared_ptr<struct TensorGraphToken> TensorGraphToken_t;

struct synMemoryDescriptor
{
    synMemoryDescriptor() = default;
    synMemoryDescriptor(bool isPersistent) : m_isPersistent(isPersistent) {}
    bool m_isPersistent = false;
};

//Descriptor of a tensor object on the host.
//A tensor is a geometric wrapper around a piece of memory on the host.
//Tensors may have no physical representation on the host if they're not needed
//For example: tensors representing output of hidden layers

//Todo: decide on semantics of creating "virtual" tensors
//Is this inferred? Decided by framework back-ends? Driven by user?
class Tensor
{
public:
    using AddressRange = std::pair<uint64_t, uint64_t>;

    static const unsigned int c_tensorMaxDim = SYN_MAX_TENSOR_DIM;
    static const unsigned int c_tensorMaxNDim = HABANA_DIM_MAX;
    // FCD is the obvious 1
    static const unsigned int c_numOfStrides = c_tensorMaxDim + 1;
    static const unsigned int c_numOfNStrides = HABANA_DIM_MAX + 1;

    enum AliasTensorType
    {
        ALIAS_TENSOR_TYPE_NONE     = 0,
        ALIAS_TENSOR_TYPE_ALIAS,
        ALIAS_TENSOR_TYPE_CONCAT,
        ALIAS_TENSOR_TYPE_FLAT,
        ALIAS_TENSOR_TYPE_SLICE,
    };

    enum class IOClassification : uint8_t
    {
        ENFORCE,
        MASK,
        AUTO
    };

    enum class ControlEdgeType : uint8_t
    {
        MEM,       // maintained by graph editor using memory coherence
        SCHEDULE,  // internal scheduling control edge, is not maintained by memory coherence
        SYNC,      // force full sync between nodes (e.g. cache dependency)
        NONE,
    };

    // STATIC FUNCTIONS
    static const TensorPtr& getRealTensor(const TensorPtr& tensor);

    static std::optional<StrideArray> getCacheLineAlignedStrides(const SizeArray&   sizes,
                                                                 const StrideArray& initialStrides,
                                                                 unsigned           numOfDims,
                                                                 unsigned           dimToAlign = 0);

    // Tensor MEMBERS

    //Data and strides are optional in case there's already a buffer in the host space which this tensor should wrap
    Tensor(unsigned       dim,
           const TSize*   sizes,
           synDataType    type,
           char*          data       = nullptr,
           const TStride* strides    = nullptr,
           bool           isOutput   = false,
           bool           isInput    = false,
           unsigned       batchPos   = INVALID_BATCH_POS,
           const TSize*   minSizes   = nullptr,
           synTensorType  tensorType = DATA_TENSOR);

    Tensor(synTensorType tensorType, int32_t graphID, const char* tensorName);

    Tensor(unsigned dim, const TSize* sizes, synDataType type, const TSize* minSizes);

    Tensor(const TensorShape& shape, synDataType type);

    Tensor(const Tensor&         t,
           bool                  copyAddresses = false,
           bool                  copyData      = true,
           TensorNameClonePolicy namePolicy    = TensorNameClonePolicy::DEFUALT_NAME);

    Tensor(const Tensor &t, const TransposePermutationArray& permutation, const std::string& name);

    Tensor(synDataType   type = syn_type_na,
           const char*   tensorName = nullptr,
           synTensorType tensorType = DATA_TENSOR,
           int32_t       graphID = 0);

    // Ideally this should be set during Tensor creation, and not as a method, but for backward compatible...
    void               setName(std::string_view name, bool forceExact = false);
    const std::string& getName() const { return m_name; }

    bool compareGeometry(const Tensor& o) const;
    bool compareGeometryWithTranspose(const Tensor& o, const TransposePermutationArray& transposeParams) const;
    bool operator==(const Tensor& o) const;
    bool operator!=(const Tensor& o) const { return !(*this == o); }

    Tensor& operator= (const Tensor& other) = delete; // disable - use clone or c'tor instead

    //Creates a deep copy - another tensor with the same dimensions with a different backing storage
    TensorPtr clone(bool                  copyAddresses  = false,
                    bool                  copyData       = true,
                    bool                  keepPersistent = false,
                    TensorNameClonePolicy namePolicy     = TensorNameClonePolicy::DEFUALT_NAME) const;
    //Creates a shallow copy - another tensor with the same geometry but discard alias information and data
    TensorPtr cloneGeometry() const;

    void mergeTensorsAttributes(const TensorPtr& otherTensor);

    // Setting dimension and sizes according to geometry type
    void setGeometry(unsigned dim, const TSize* sizes, synGeometryType geometryType);

    //Reshape tensor dimensions. Data remains unchanged.
    void reshape(unsigned       dim,
                 const TSize*   maxSizes = nullptr,
                 const TStride* strides  = nullptr,
                 const TSize*   minSizes = nullptr);

    void setDenseStrides();

    bool resizeDims(unsigned dims);

    // Assume minSizes is an array with at least getDims() elements
    void setMinSize(const TSize* minSizes) { m_shape.setMinSize(minSizes); }

    bool isBound() const { return getData() != nullptr; }
    //Bind this tensor to the given pointer
    void bind(void* ptr, bool shouldFreeBuffer);
    //Unbind the tensor.
    void unbind();
    //Allocates new data with given size in bytes, unbinds old data and rebind new data
    void rebind(TSize size);
    //Set or copy a buffer and metadata to this tensor
    void setTensorBuffer(void* ptr, uint64_t size, synDataType bufferDataType, bool copyBuffer=true);

    //Returns a pointer accessible in host memory, allowing update of the tensor contents
    void* map();
    //Signals tensor contents have been updated and should be synchronized
    void unmap() {}

    //Todo: this whole section doesn't belong in an API header, this is a bunch of internal metadata customers shouldn't care about
    //Returns whether this Tensor needs to be treated as an output tensor
    bool isEnforcedOutput() const { return m_outputState == IOClassification::ENFORCE; }
    //Returns whether this Tensor not allowed to be treated as an output tensor
    bool isMaskedOutput() const { return m_outputState == IOClassification::MASK; }
    //Set this tensor as not output tensor (remained for internal needs).
    void maskOutput() { m_outputState = IOClassification::MASK; }
    //Set this tensor as an output tensor (upon user's input)
    void enforceOutput(bool enforce = true)
    {
        m_outputState = enforce ? IOClassification::ENFORCE : IOClassification::AUTO;
    }
    //Returns whether this Tensor is used as weights for a spatial convolution operations or not
    bool isWeights() const { return m_isWeights; }
    //Sets this Tensor as a weight tensor. This is a sticky bit and will assert if later assigned a producer
    void setAsWeights() { m_isWeights = true; }
    //Sets this Tensor as a bias tensor. This is a sticky bit and will assert if later assigned a producer
    void setAsBias() { m_isBias = true; }
    //Gets whether this Tensor is static (known at compile time).
    bool isStaticParam() const { return m_isParam; }
    //Sets this Tensor as a model param tensor. This is a sticky bit and will assert if later assigned a producer
    void setAsStaticParam(bool isStatic = true) { m_isParam = isStatic; }
    //Returns whether this Tensor is used as weights or bias for the model operations or not
    bool isModelParameter() const { return m_isBias || m_isWeights; }
    //Sets whether quantize tensor data per channel or not.
    void setPerChannelQuant(bool perChannel = false, bool force = false);
    //Gets whether this tensor is quantized per channel.
    bool isPerChannelQuant() const { return m_perChannelQuant; }
    //Returns whether this tensor has been packed (MME optimization)
    bool isPacked(unsigned dim) const { return m_annotation.dataInfo.packing[dim] != 1; }
    //Mark this tensor as packed for a given packing factor
    void setPacked(const std::array<TSize, MME_MAX_CONV_DIMS>& packing) { m_annotation.dataInfo.packing = packing; }
    //Returns whether this tensor has been lowered (MME optimization)
    bool isLowered() const { return m_annotation.dataInfo.isLowered; }
    //Mark this tensor as lowered
    void setLowered() { m_annotation.dataInfo.isLowered = true; }
    //Returns whether this tensor has been vectorized (part of MME lowering optimization)
    bool isVectorized() const { return m_annotation.dataInfo.isVectorized; }
    //Mark this tensor as vectorized
    void setVectorized() { m_annotation.dataInfo.isVectorized = true; }
    //Returns whether this weight-tensor's is swizzled
    bool isSwizzled() const { return m_annotation.dataInfo.isSwizzled; }
    //Marks this tensor as a weight-tensor that had been swizzled, and its swizzle-size
    void setSwizzled(uint64_t numOfSwizzleElements);
    //Returns whether tensor data is of same type as tensor data type
    bool isDataTypeMatchData() const { return m_bufferDataType == m_type; }
    //Sets this tensor as having data of same type as tensor data type
    void setAsDataTypeMatchData()
    {
        m_bufferDataType    = m_type;
        m_bufferSizeInBytes = m_deviceSizeInBytes;
    }
    //Returns whether tensor data is marked as sparsity weight data
    bool isSparsityWeights() const { return m_isSparsityWeights; }
    //Sets tensor as weight sparsity data, (used for calculating mme weights quantization info)
    void setAsSparsityWeights() { m_isSparsityWeights = true; }
    //Returns whether this tensor is strided on FCD
    bool isStridedOnFCD() const;
    //Sets this tensor as as aux tensor (created by the glue code and used by the TPC)
    void setAsAuxTensor(bool isScratchPad);
    //Returns whether this is an aux tensor (used by the TPC)
    bool isAuxTensor() const { return m_annotation.isAuxTensor; }
    //Returns whether this is a double store tensor
    bool isDoubleStoreTensor() const { return m_annotation.isDoubleStoreTensor; }
    // Returns whether this is a scratch-pad aux tensor (used by TPC)
    bool isScratchPadAuxTensor() const { return isAuxTensor() && m_annotation.isScratchPadAuxTensor; }
    // Return whether the tensor serves as a unit matrix.
    bool isUnitMatrix() const { return m_unitMatrix; }
    /* diagnose the tensor if it is zero-sized tensor. */
    bool isZeroSizedDataTensor() const { return !isShapeTensor() && m_shape.hasZeroSize(); }
    bool isNonScratchpadAuxTensor() const { return isAuxTensor() && !m_annotation.isScratchPadAuxTensor; }

    // Sets this tensor as a unit matrix
    void setUnitTensor()
    {
        m_unitMatrix = true;
        setAsStaticParam(true);
    }
    // Set data ptr - DON'T USE outside the deprecated tensor API!
    void setDataUnsafe(char* data) { m_data = data; }

    bool isDynamicShape() const { return m_shape.isDynamic(); }
    bool isDynamicDim(unsigned dim) const { return m_shape.isDynamicDim(dim); }

    std::optional<unsigned> getFirstDynamicDimIndex(unsigned startDim = 0) const
    {
        return m_shape.getFirstDynamicDimIndex(startDim);
    }

    void setElementType(synDataType type);
    void setElementTypeWithoutStrides(synDataType type) { m_type = type; }
    void setDeviceLayout(synDataType type, const uint32_t strides[HABANA_DIM_MAX]);

    // TODO [SW-35841] - Remove this function when m_type has syn_type_na as a valid default value
    // WORKAROUND - use only in data type selection passes
    void changeDefaultElementType(synDataType type, bool ignoreDataTypeMatch = false);

    void setUserContext(const TensorExUserContext& userContext);

    void resetStrides()
    {
        // Restoring original strides scheme (mostly required before additional execution of 'runLogicalOperation')
        memcpy(m_strides, m_origStrides, sizeof(TStride) * Tensor::c_numOfNStrides);
    }

    /**
     * Get all memory range for tensor use
     * Most of the time there will be one range
     * In concat node on an inner dim, the range will be split to several ranges
     */
    std::vector<AddressRange> getAddressRange() const;

    // dramOffset and sramOffset point to the actual data location.
    // No such thing as Local VS Alias data-location. They are the same.
    deviceAddrOffset getSramOffset() const;         // Returns tensor's SRAM offset.
    deviceAddrOffset getDramOffset() const;         // Returns tensor's DRAM offset.
    deviceAddrOffset getTensorOffset() const;       // Returns tensor's offset depending on its location.
    void setSramOffset(deviceAddrOffset offset);    // Sets tensor's SRAM offset.
    void             setDramOffset(deviceAddrOffset offset) { m_dramOffset.set(offset); }  // Sets tensor's DRAM offset.
    void setTensorInDram() { getTensorAnnotation().memory.location = TENSOR_IN_DRAM; }     // Sets tensor to be in Dram
    void setTensorInSram();                         // Sets tensor to be in Sram
    void setTensorInWorkspace();                    // Sets tensor to be in Dram workspace
    void setTensorAlignment(unsigned cacheLineSize) { getTensorAnnotation().memory.alignment = cacheLineSize; }
    void setTensorAsExternal(bool isExternal) {m_isExternal = isExternal;}// Sets tensor to be in external
    void unsetSramOffset();
    void unsetDramOffset();

    TensorLocation
    getTensorAllocatedLocation() const;  // Returns tensor's location which can be SRAM, DRAM or UNDEFINED.
    std::string_view getTensorLocationString() const;  // For logs
    TensorLocation   location() const;
    bool             inSram() const { return location() == TENSOR_IN_SRAM; }
    bool             inDram() const { return location() == TENSOR_IN_DRAM; }
    bool             tensorAllocatedInSram() const { return getTensorAllocatedLocation() == TENSOR_IN_SRAM; }
    bool             tensorAllocatedInDram() const { return getTensorAllocatedLocation() == TENSOR_IN_DRAM; }
    bool             tensorIsAllocated() const { return getTensorAllocatedLocation() != UNDEFINED_LOCATION; }
    bool             isDramOffsetSet() const { return m_dramOffset.is_set(); }

    const TensorPtr& getDramAllocatedTensor() const { return m_dramAllocatedTensor.m_parentTensor; }
    uint64_t         getDramAllocatedTensorOffset() const { return m_dramAllocatedTensor.m_byteOffset; }
    void             setDramAllocatedTensor(const TensorPtr& parentDramTensor, uint64_t byteOffset);
    bool      hasDramAllocatedTensor() const { return m_dramAllocatedTensor.m_parentTensor != nullptr; }

    // Returns reference to tensor annotation
    TensorAnnotation& getTensorAnnotation() { return m_annotation; }
    TensorAnnotation& getTensorAnnotation() const { return m_annotation; }
    void              setAsAliasSubTensor(const TensorPtr& src, uint64_t byteOffset = 0)
    {
        _updateAliasDeviceInfo(src, ALIAS_TENSOR_TYPE_ALIAS, byteOffset);
    }
    void setAsSliceSubTensor(const TensorPtr& aliasTensor, uint64_t byteOffset, const TStride* strides);
    void setAsConcatSubTensor(const TensorPtr& aliasTensor, uint64_t offset, unsigned concatDim);
    void setAsFlattenSubTensor(const TensorPtr& aliasTensor, bool isInput, unsigned axis = 1);
    void setAsHostAliasedSubTensor(const TensorPtr& aliasTensor, uint64_t byteOffset);

    // Used for being able to re-use a by-product Tensor in testing, instead of re-creating one.
    void resetAliasing();
    void resetHostAliasing() { m_hostAliasInfo.m_parentTensor = nullptr; }

    bool             isRealInLogical() const { return m_isRealInLogical; }
    void             setIsRealInLogical(bool isRealInLogical) { m_isRealInLogical = isRealInLogical; }
    bool             isRealInAliasing() const { return m_isRealInAliasing; }
    void             setIsRealInAliasing(bool isRealInAliasing) { m_isRealInAliasing = isRealInAliasing; }
    bool             isAliasedTensor() const { return _isAliasedTensor(); }
    bool             isHostAliasedTensor() const { return m_hostAliasInfo.m_parentTensor != nullptr; }
    const TensorPtr& getAliasTensor() const { return m_aliasInfo.pAliasedTensor; }
    const TensorPtr& getHostAliasTensor() const { return m_hostAliasInfo.m_parentTensor; }
    uint64_t         getHostAliasOffset() const { return m_hostAliasInfo.m_byteOffset; }
    void             updateAliasTensor(const TensorPtr& aliasTensor) { m_aliasInfo.pAliasedTensor = aliasTensor; }
    void             cloneAliasInfo(const TensorPtr& other, const TensorPtr& alias = nullptr);
    void             cloneHostAliasInfo(const TensorPtr& other, const TensorPtr& alias = nullptr);
    bool isDenseLayout() const;
    bool             isTrivialStrided() const { return isDenseLayout(getDim()); }
    // return the reduction info if any tensor in the alias chain has reduction enabled in annotation
    ReductionInfo getRealReductionInfo(bool checkSetOp = false) const;
    // return true if any tensor in the alias chain has reduction enabled in annotation
    bool isReductionEnabled(bool checkSetOp = false) const;
    // return the reduction operation if any tensor in the alias chain has reduction enabled in annotation
    ReductionOperation getReductionOperation(bool checkSetOp = false) const
    {
        return getRealReductionInfo(checkSetOp).reductionOperation;
    }

    const TensorShape& getShape() const { return m_shape; }

    unsigned          getDim() const { return m_shape.getDim(); }

    void              getAllSizesInElements(TSize* sizes, unsigned count) const;
    void              getAllNSizesInElements(TSize sizes[c_tensorMaxNDim]) const;
    void              getAllSizesInElementsCondensed(TSize* sizes, unsigned count) const;
    SizeArray         getAllSizesInElements() const { return m_shape.getSizes(); }
    const NSizeArray& getAllNSizesInElements() const { return m_shape.getNSizes(); }
    NSizeArray        getNSizesInElements() const { return m_shape.getNSizes(); }
    TSize             getSizeInElements(unsigned dim) const
    {
        HB_ASSERT(dim < c_tensorMaxNDim, "dimension is bigger than maximum dimensions");
        return m_shape.getSize(dim);
    }
    void getAllMinimalSizesInElements(TSize* sizes, unsigned count) const
    {
        HB_ASSERT(getDim() <= count, "This function is deprecated for dimension {}", getDim());
        memcpy(sizes, m_shape.getNMinSizes().data(), sizeof(TSize) * count);
    }
    void              getAllMinimalSizesInElementsCondensed(TSize* sizes, unsigned count) const;
    SizeArray         getAllMinimalSizesInElements() const { return m_shape.getMinSizes(); }
    const NSizeArray& getNMinimalSizesInElements() const { return m_shape.getNMinSizes(); }
    TSize             getMinimalSizeInElements(unsigned dim) const
    {
        HB_ASSERT(dim < c_tensorMaxNDim, "dimension is bigger than maximum dimensions");
        return m_shape.getMinSize(dim);
    }

    std::pair<SizeArray /*minSizes*/, SizeArray /*maxSizes*/> getAllMinMaxSizesInElements() const
    {
        return {m_shape.getMinSizes(), m_shape.getMaxSizes()};
    }
    std::pair<TSize /*minSize*/, TSize /*maxSize*/>           getMinMaxSizeInElements(unsigned dim) const;

    template<std::size_t count>
    void            getAllMinimalSizesInElementsCondensed(std::array<TSize, count>& data) const { getAllMinimalSizesInElementsCondensed(data.data(), count); }

    template<std::size_t count>
    void            getAllMinimalSizesInElements(std::array<TSize, count>& data) const { getAllMinimalSizesInElements(data.data(), count); }

    template<std::size_t count>
    void            getAllSizesInElementsCondensed(std::array<TSize, count>& data) const { getAllSizesInElementsCondensed(data.data(), count); }

    template<std::size_t count>
    void            getAllSizesInElements(std::array<TSize, count>& data) const { getAllSizesInElements(data.data(), count); }

    // Count how many dimensions have a value other than 1. Also could be regarded as the practical dim count.
    uint32_t        getNon1SizeDimsCount() const;

    uint64_t        getTotalElements() const;
    uint64_t        getMinimalElements() const;

    /* The following queries (with the prefix 'getDense') calculates their return values as if the tensor were dense */
    uint64_t getDenseSizeInElements() const
    {
        const NSizeArray& sizes = getAllNSizesInElements();
        return multiplyElements(std::begin(sizes), std::begin(sizes) + m_shape.getDim());
    }

    uint64_t        getDenseStrideInElements(int dim) const;
    uint64_t        getDenseSizeInBytes() const { return getDenseSizeInElements() * getElementSizeInBytes(); }

    void            getAllStridesInElements(TStride strides[c_numOfStrides]) const;

    TStride         getStrideInElements(unsigned dim) const;
    TStride         calcStrideInElements(TStride) const;

    NStrideArray    getNStridesInElements() const;
    StrideArray     getAllStridesInBytes() const;
    void            getNStridesInBytes(TStride bytesStrides[c_numOfNStrides]) const
    {
        std::copy(m_strides, m_strides + c_numOfNStrides, bytesStrides);
    }

    const TStride*  const getNStridesInBytes() const { return m_strides; }
    void                  getAllStridesInBytes(TStride* strides, unsigned count) const
    {
        HB_ASSERT(getDim() <= count, "This function is deprecated for dimension {}", getDim());
        std::copy(m_strides, m_strides + count, strides);
    }

    TStride getStrideInBytes(unsigned dim) const
    {
        HB_ASSERT(dim < c_numOfNStrides, "dimension is bigger than number of strides");
        return m_strides[dim];
    }

    TStride getMaxStride() const { return *std::max_element(m_strides, m_strides + getDim() + 1); }

    // Modify strides such that they will align to cache line, this may increase the tensor size
    void alignStridesToCacheLine();

    unsigned getElementSizeInBits() const { return dataTypeToSizeInBits(m_type); }
    unsigned getElementSizeInBytes() const { return dataTypeSizeInBytes(m_type, isCondensed4Bit()); }

    uint64_t getTotalSizeInBytes() const { return m_deviceSizeInBytes; }
    void     setDeviceSizeInBytes(uint64_t size)
    {
        HB_ASSERT(size >= m_deviceSizeInBytes, "decreasing deviceSize");
        m_deviceSizeInBytes = size;
    }

    TSize getSizeInBytes(unsigned dim) const
    {
        HB_ASSERT(dim < c_numOfNStrides, "dimension is bigger than maximum dimensions");
        return safeBitsToByte(m_shape.getSize(dim) * getElementSizeInBits());
    }

    uint64_t getMinimalSizeInBytes() const { return safeBitsToByte(getMinimalElements() * getElementSizeInBits()); }
    TSize    getMinimalSizeInBytes(unsigned dim) const
    {
        HB_ASSERT(dim < c_tensorMaxNDim, "dimension is bigger than maximum dimensions");
        return safeBitsToByte(m_shape.getMinSize(dim) * getElementSizeInBits());
    }

    uint64_t        getSampleSize() const;

    synDataType     getElementType() const { return m_type; }
    synDataType     getBufferDataType() const { return m_bufferDataType; }
    uint64_t        getBufferSizeInBytes() const { return m_bufferSizeInBytes; }

    // true if tensor data type is fp32 / fp16 /bf16
    bool isTypeFloat() const
    {
        return ::isTypeFloat(m_type);  // use the isTypeFloat function in utils.h
    }
    // true if tensor data type is int4 / uint4
    bool isType4Bit() const { return m_type == syn_type_int4 || m_type == syn_type_uint4; }
    // true if tensor strides were condensed to reflect 4bit arrangement
    bool isCondensed4Bit() const { return m_annotation.info4Bit.isCondensed; }
    bool needsToCondense4Bit() const { return isType4Bit() && !isCondensed4Bit(); }

    void*           getAddress() const { return static_cast<void*>(getData()); }
    void            debugPrint() const;
    char*           getData() const;

    void                                         setQuantizationParams(const synQuantMetadata& params);
    void                                         setQuantizationParams(const synFpQuantMetadata& params);
    void                                         setQuantizationParams(const synPerChannelQuantizationParams& params);
    void                                         setQuantizationParams(const synQuantizationParams& params);
    void                                         setQuantizationParams(const QuantizationData& params);
    void                                         setAllQuantizationParams(const QuantizationMap& params);
    bool                                         setDynamicRange(DynamicRange dynamicRange);
    bool                                         setPerChannelDynamicRange(const synPerChannelDynamicRange& perChannelDynamicRange);
    bool                                         setPerChannelDynamicRange(const PerChannelDynamicRange& perChannelDynamicRange);
    double                                       getZeroPoint(unsigned index = 0) const;
    double                                       getZeroPoint(synDataType type, unsigned index = 0) const;
    double                                       getScale(unsigned index = 0) const;
    double                                       getScale(synDataType type, unsigned index = 0) const;
    void                                         setScale(double newScale, unsigned index = 0);
    bool                                         setScaleForDtype(double newScale, synDataType dtype = syn_type_fp8_143);
    unsigned                                     getExpBias(unsigned index = 0) const;
    unsigned                                     getExpBias(synDataType type, unsigned index = 0) const;
    void                                         setExpBias(unsigned newExpBias, unsigned index = 0);
    bool                                         setExpBiasForDtype(unsigned newExpBias, synDataType dtype = syn_type_fp8_143);
    void                                         setZeroPoint(double newZeroPoint, unsigned index = 0);
    const QuantizationData&                      getQuantizationParams() const { return getQuantizationParams(m_type); }
    const QuantizationData&                      getQuantizationParams(synDataType dataType) const;
    const QuantizationData&                      getQuantizationParams(eQuantDataType dataType) const;
    bool                                         isQuantizationParamsExist(synDataType dataType) const;
    const QuantizationMap&                       getAllQuantizationParams() const { return m_quantizationParamsMap; }
    DynamicRange                                 getDynamicRange() const { return m_dynamicRange; }
    const PerChannelDynamicRange&                getPerChannelDynamicRange() const { return m_perChannelDynamicRange; }
    void                                         lockQuantization(NodePtr lockingNode);
    void                                         requantLock(NodePtr lockingNode);
    void                                         saveConflicts();
    void                                         saveConflicts(NodePtr node, QuantizationMap& quantizationMap);
    void                                         revertQuantization();
    bool                                         isLocked() const { return m_quantizationLockingNodes.size() > 0; }
    bool                                         isRequantLocked() const { return m_isRequantLocked; }
    void setMeasuredQuantizationParams() { m_measuredQuantizationMap = m_quantizationParamsMap; }
    void                                         setMeasuredQuantizationParams(QuantizationMap& measuredQuantizationMap);
    void                                         resetQuantization();
    const std::set<synNodeId>&                   getLockingNodes() { return m_quantizationLockingNodes; }
    void setLockingNodes(const std::set<synNodeId>& lockingNodes) { m_quantizationLockingNodes = lockingNodes; }
    const QuantizationConflictMap&               getConflictingQuants() const { return m_conflictQuants; }
    void                                         setInt8FixedPoint(bool value) { m_isInt8FixedPoint = value; }
    void                                         setInt16Limited(bool value) { m_isInt16Limited = value; }
    bool                                         isInt8FixedPoint() { return m_isInt8FixedPoint; }
    bool                                         isInt16Limited() { return m_isInt16Limited; }
    void                                         enforceInt8FixedPoint();
    void                                         enforceInt16Limited();

    AliasTensorType         getAliasedTensorType() const { return _getAliasedTensorType(); }
    uint64_t                getAliasedByteOffset() const { return _getAliasedByteOffset(); }
    std::string             getAliasedTensorTypeStr() const;

    void setShouldFreeBuffer(bool value) { m_shouldFreeBuffer = value; }
    bool getShouldFreeBuffer() const { return m_shouldFreeBuffer; }

    unsigned getBatchPos() const { return m_batchPos; }
    void     setBatchPos(unsigned batchPos) { m_batchPos = batchPos; }

    void updateTensorROISize(uint64_t sizeInBytes, bool force = false)
    {
        if ((!m_tensorROISizeInBytes.is_set()) || force)
        {
            m_tensorROISizeInBytes = sizeInBytes;
        }
    }

    uint64_t getTensorROISize()
    {
        HB_ASSERT(m_tensorROISizeInBytes.is_set(), "Trying to get a tensor ROI size that was not set");
        return m_tensorROISizeInBytes.value();
    }
                            ~Tensor();
    void                    setId(uint64_t id) { m_id = id; }
    uint64_t                getId() const { return m_id; }
    void                    setShapePlaneIndex(uint32_t index) { m_shapePlaneIndex = index; }
    uint32_t                getShapePlaneIndex() { return m_shapePlaneIndex; }

    // For debugging
    std::string             getDimSizesStr(bool isFCDLast = false, bool minSizes = false) const;
    std::string             getStridesStr(bool isFCDLast = false) const;

    bool                    isNotNeeded() const { return m_annotation.isNotNeeded; }
    void                    setIsNotNeeded(bool isNotNeeded = true) { m_annotation.isNotNeeded = isNotNeeded; }

    bool                    isControlEdge() const { return m_ctrlEdgeType != ControlEdgeType::NONE; }
    ControlEdgeType         getControlEdgeType() const { return m_ctrlEdgeType; }
    void                    setAsControlEdge(ControlEdgeType type = ControlEdgeType::MEM) { m_ctrlEdgeType = type; }
    synTensorType           getTensorType() const { return m_tensorType; }
    void                    setTensorType(synTensorType tensorType) { m_tensorType = tensorType; }
    bool                    isShapeTensor() const
    {
        return (m_tensorType == INPUT_DESCRIBING_SHAPE_TENSOR) || (m_tensorType == OUTPUT_DESCRIBING_SHAPE_TENSOR) ||
               (m_tensorType == HOST_SHAPE_TENSOR);
    }
    bool isTensorShapeOutput() const { return getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR; }

    bool isTensorAuxOrShapeOutput() const { return isAuxTensor() || isTensorShapeOutput() || isHostOnly(); }
    bool isHostTensor() const { return m_tensorType == HOST_SHAPE_TENSOR || m_tensorType == HOST_TO_DEVICE_TENSOR; }
    bool isHost2DeviceTensor() const { return m_tensorType == HOST_TO_DEVICE_TENSOR; }
    bool isHostShapeTensor() const { return m_tensorType == HOST_SHAPE_TENSOR; }
    bool isDataTensor() const { return (m_tensorType == DATA_TENSOR) || (m_tensorType == DATA_TENSOR_DYNAMIC); }
    void                    setShapeTensor(synTensorType shapeTensorType);
    void                        setMemoryDescriptor(const synMemoryDescriptor& memDesc) { m_memoryDescriptor = memDesc; }
    void                        setAsNonUserManaged();
    bool                        isPersistent() const                                    { return m_memoryDescriptor.m_isPersistent; }
    bool                        isAssignedToConstSection() const;
    bool                        isUserManagedDram() const;
    bool                        isPartOfRMWSection() const;
    bool                        inConstSection() const;
    bool                        isPartOfWorkspaceSection() const { return isUserManagedDram() && !isPersistent(); };

    void setSectionHandle(const SlotMapItemSptr<InternalSectionHandle>& sectionHandle);
    const SlotMapItemSptr<InternalSectionHandle>& getSectionHandle() const;

    void                        setMemorySectionID(uint64_t memSecId)                   { m_memSectionId = memSecId; }
    void                        setGraphID(uint32_t graphID)                            { m_graphID = graphID; }
    void                        setMemorySectionOffset(uint64_t offset)                 { m_memorySectionOffset = offset; }
    const synMemoryDescriptor&  getMemoryDescriptor() const                             { return m_memoryDescriptor; }
    const uint32_t              getGraphID() const                                      { return m_graphID; }
    uint64_t                    getMemorySectionOffset() const                          { return m_memorySectionOffset; }
    bool                        getTensorIsExternal() const                             { return m_isExternal;}

    uint64_t getMemorySectionID() const
    {
        return (_isAliasedTensor()) ? m_aliasInfo.pAliasedTensor->getMemorySectionID() : m_memSectionId;
    }
    void validateFlattenSubTensor(const TensorPtr& aliasTensor, unsigned axis);

    TensorGraphToken_t          m_graphToken; // Opaque member solely known to, and used by, the Graph

    void setAsDroppedTensor() { m_isDropped = true; }

    bool isDroppedTensor() { return m_isDropped; }

    void condenseStridesTo4Bit(CondenseDimIndex condensedDimIndex); // condense strides to match 2 elements per byte arrangement.
                                                                    // strides of the given dimension are the first to be condensed.
    bool is1DAndSameDepthAsOther(const TensorPtr& other) const;

    bool is64BitElementSize() const
    {
        constexpr auto mask64BitDatatype = (syn_type_int64 | syn_type_uint64);
        return (getElementType() & mask64BitDatatype) != 0;
    }

    void setProp(synTensorProperty prop) { m_propsSet |= prop; }
    bool isPropSet(synTensorProperty prop) const { return m_propsSet & prop; }
    bool isPropsValid();
    void lockPropsAndFinalizeTensor();
    bool isPropsLocked() const { return m_lockProps && !m_duplicatedTensor; }
    void setDuplicatedTensor()
    {
        m_duplicatedTensor = true;
        // clear shape specific information
        m_propsSet &= ~(synTensorPropDeviceLayout | synTensorPropGeometryMin | synTensorPropGeometryMax |
                        synTensorPropGeometryDim);
    }

    TStride  getPrefetchStride() { return m_prefStride; }
    void     setPrefetchStride(TStride prefStride) {m_prefStride = prefStride; }

    // Next 4 functions are for tensors of type HOST_TO_DEVICE and HOST_SIZE, and their device twins
    bool     hasHostData() const;
    char*    getHostMinData() const;
    char*    getHostMaxData() const;
    uint64_t getHostDataSize() const;

    void     setHostOnly() { m_isHostOnly = (m_tensorType == HOST_TO_DEVICE_TENSOR); }  //only set if tensor is H2D
    bool     isHostOnly() const { return m_isHostOnly; }

    // These 2 functions are for device twins of HOST_TO_DEVICE tensors
    const TensorPtr& getTwinHost2DeviceTensor() { return m_twinHost2DeviceTensor; }
    void             setTwinHost2DeviceTensor(const TensorPtr& twin) { m_twinHost2DeviceTensor = twin; }

    // get tensor permutation - will return empty if not set
    const std::optional<gc::Permutation>& getPermutation() const { return m_permutation; };
    // set tensor permutation - will return false if given permutation is invalid
    bool setPermutation(const gc::Permutation& permutation);
    void unsetPermutation() { m_permutation.reset(); }

    unsigned getIndexOfMaxNonDegenerateStride() const;
    void setConnectAtomicNodes();

protected:

    //Allocates host memory for this tensor if needed
    void                    AllocateHostMemory();

    void _updateAliasDeviceInfo(const TensorPtr& aliasTensor, AliasTensorType aliasType, uint64_t byteOffset = 0);
    AliasTensorType _getAliasedTensorType() const { return m_aliasInfo.aliasTensorType; }
    uint64_t        _getAliasedByteOffset() const { return m_aliasInfo.byteOffset; }

    void                    _debugPrintAliasInfo() const;

    bool _isAliasedTensor() const { return m_aliasInfo.aliasTensorType != ALIAS_TENSOR_TYPE_NONE; }

    uint64_t                _getDistanceToElementInBytes(uint64_t elementNum) const;
private:
    std::string getClonedTensorName(const Tensor& t, TensorNameClonePolicy namePolicy);

    template<int N, typename T> void getFirstNStridesInElements(T& strides) const;

    void matchSizesTo4Bit(TSize* sizes, unsigned count = c_tensorMaxDim) const; // helper function - match sizes to 4 bit arrangement

    // calculate the degenerate strides of a tensor (for strides larger than tensor dim)
    void calculateDegenerateStrides(unsigned dim, TStride maxStride = 0, uint64_t lastElementOffset = 0);

    void setDefaultStrides();

    bool isDenseLayout(unsigned lastDim) const;

    /*
     * *************************************************************************************************
     *
     *  @brief AliasTensor Struct holds alias information, mainly used for data-manipulation on Alias Tensors
     *
     *  The Tensor class includes, inherently, two types of data that are representing some kind of shape:
     *      1) Geometry of the Tensor
     *      2) Data location and order in the Tensor's data-host.
     *
     *  Currently only the Concatenate's Alias-Type represent manipulation over the data,
     *  where the data order is different than the geometry of the Tensor.
     *
     *  Due to the above, it would be better to refactor this struct and to use class-inheritance,
     *  from the main Tensor class.
     *
     *  The sizes is also a parameter of this struct, although it is not currently used.
     *  I believe that it may have real usage, when we will want to perform non-FCD concatenate operation.
     *
     * *************************************************************************************************
     */
    // REMARK 1 - There is a limitation in the alias-tensor feature,
    //            that limits a single alias per tensor.
    // REMARK 2 - The code also does not support consecutive alias Tensors, AFAIU.
    //            For example in the AllocateSram code.
    // REMARK 3 - We should hold here the alias information (dim, strides[] and size[]).
    //              * They should be used when we are looking on this class as an output.
    //                For example, to where the data needs to be located.
    //              * The same parameters that are "directly located" members should be used
    //                when we are looking on this class as input.
    //                For example, that the two conv. operands match in proper dimension's size.
    struct AliasTensor
    {
        TensorPtr       pAliasedTensor  = nullptr;
        uint64_t        byteOffset      = 0;
        AliasTensorType aliasTensorType = ALIAS_TENSOR_TYPE_NONE;
        // REMARK - In general, we should call the "proper get method" of the pAliasedTensor,
        //          for the alias-of-alias cases -> todo
    };

    struct HostAliasTensor
    {
        TensorPtr m_parentTensor = nullptr;
        uint64_t  m_byteOffset   = 0;
    };

    struct DramAllocatedTensor
    {
        TensorPtr m_parentTensor = nullptr;
        uint64_t  m_byteOffset   = 0;
    };

    uint64_t                 m_id;
    std::string              m_name;
    bool                     m_lockName = false;
    synDataType              m_type = syn_type_na;
    bool                     m_shouldFreeBuffer = false;
    IOClassification m_outputState = IOClassification::AUTO;  // Whether this Tensor is an output, not output or DC
    IOClassification m_inputState  = IOClassification::AUTO;  // Whether this Tensor is an input, not input or DC
    bool                     m_isWeights = false;                // Whether this Tensor contains spatial convolution weights or not
    bool                     m_isBias = false;                   // Whether this Tensor contains spatial convolution bias or not
    bool                     m_isParam = false;                  // Whether this Tensor is a static parameter or not
    bool                     m_unitMatrix = false;
    bool                     m_perChannelQuant = false;          // Whether this tensor should be quantized per channel or not
    char*                    m_data = nullptr;                     // Char for easy pointer arithmetic
    TensorShape              m_shape;
    TStride                  m_strides[c_numOfNStrides];  // Stride in bytes
    TStride                  m_origStrides[c_numOfNStrides];
    uint64_t                 m_bufferSizeInBytes = 0;
    uint64_t                 m_deviceSizeInBytes = 0;
    synDataType              m_bufferDataType = syn_type_float;

    Settable<uint64_t>       m_tensorROISizeInBytes;
    unsigned                 m_batchPos = INVALID_BATCH_POS;

    AliasTensor              m_aliasInfo;
    bool m_isRealInLogical  = false;  // whether this Tensor is aliased to or not (before handleLogicalOps)
    bool m_isRealInAliasing = false;  // whether this Tensor is aliased to or not (after handleLogicalOps)

    HostAliasTensor          m_hostAliasInfo;
    DramAllocatedTensor      m_dramAllocatedTensor;      // The original tensor that holds the shared DRAM address. Relevant for static tensors only.

    Settable<deviceAddrOffset>  m_sramOffset;            // If SRAM is set, tensor is in SRAM. otherwise, it is in DRAM or not allocated yet.
    Settable<deviceAddrOffset>  m_dramOffset;

    mutable TensorAnnotation    m_annotation;

    QuantizationMap             m_quantizationParamsMap;
    QuantizationMap             m_measuredQuantizationMap;

    static std::atomic<uint64_t>           m_nextId;
    uint32_t                               m_shapePlaneIndex = UINT32_MAX;
    uint64_t                               m_memSectionId;
    uint32_t                               m_graphID               = 0;
    uint64_t                               m_memorySectionOffset   = 0;
    SlotMapItemSptr<InternalSectionHandle> m_internalSectionHandle = nullptr;
    synMemoryDescriptor                    m_memoryDescriptor;
    synTensorType                          m_tensorType   = DATA_TENSOR;
    ControlEdgeType                        m_ctrlEdgeType = ControlEdgeType::NONE;
    std::set<synNodeId>                    m_quantizationLockingNodes;
    QuantizationConflictMap                m_conflictQuants;
    DynamicRange                           m_dynamicRange;
    PerChannelDynamicRange                 m_perChannelDynamicRange;
    unsigned                               m_propsSet              = 0;
    TStride                                m_prefStride            = 0;
    bool                                   m_isDropped             = false;
    bool                                   m_isRequantLocked       = false;
    bool                                   m_isInt8FixedPoint      = false;
    bool                                   m_isInt16Limited        = false;
    bool                                   m_isSparsityWeights     = false;
    bool                                   m_lockProps             = false;
    bool                                   m_duplicatedTensor      = false;
    bool                                   m_isExternal            = false;
    TensorPtr                              m_twinHost2DeviceTensor = nullptr;
    bool                                   m_isHostOnly            = false;

    // The permutation that needs to be applied on the tensor's strides, as a logical transpose,
    // for them to be dense (as in what's called a "permuted tensor").
    std::optional<gc::Permutation> m_permutation;
};

struct TensorHasher
{
    size_t operator()(const TensorPtr& t) const { return std::hash<uint64_t>()(t->getId()); }
};

struct TensorComparator
{
    bool operator()(const TensorPtr& tensor1, const TensorPtr& tensor2) const
    {
        if (tensor1 == nullptr) return false;
        if (tensor2 == nullptr) return true;
        return tensor1->getId() < tensor2->getId();
    }
};
typedef std::set<TensorPtr, TensorComparator> TensorSet;
typedef std::vector<TensorSet>                TensorSetVector;


#endif  // _TENSOR_H_
