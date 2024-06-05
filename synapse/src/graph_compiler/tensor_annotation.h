#ifndef _TENSOR_ANNOTATION_H_
#define _TENSOR_ANNOTATION_H_

#include <map>
#include "node_annotation.h"
#include "tensor_shape.h"
#include "cache_types.h"

static const unsigned UNDEFINED_CACHE_LINE_SIZE_IN_BYTES = 0xffffffff;

enum TensorLocation
{
    TENSOR_IN_SRAM = 0,
    TENSOR_IN_DRAM,
    UNDEFINED_LOCATION
};

enum ReductionOperation
{
    REDUCTION_ADD  = 0,
    REDUCTION_SUB  = 1,
    REDUCTION_MIN  = 2,
    REDUCTION_MAX  = 3,
    REDUCTION_SET  = 4,  // This operation is virtual and not supported by HW; each input overwrites the output data.
    REDUCTION_MAX0 = 5,
    REDUCTION_UNORDERED_SET = 6,  // same as reduction set, but without any ordering between the different writes
    ENUM_MAX_REDUCTION_OPERATIONS
};

struct ReductionInfo
{
    ReductionInfo() = default;
    ReductionInfo(bool isRedEnabled, ReductionOperation Op) : isReductionEnabled(isRedEnabled), reductionOperation(Op)
    {
    }

    static bool isReductionSet(ReductionOperation reductionOp)
    {
        return reductionOp == REDUCTION_SET || reductionOp == REDUCTION_UNORDERED_SET;
    }

    static bool isRMWReduction(ReductionOperation reductionOp)
    {
        return reductionOp < ENUM_MAX_REDUCTION_OPERATIONS && !isReductionSet(reductionOp);
    }

    bool     isReductionEnabled = false;
    ReductionOperation reductionOperation = REDUCTION_ADD;
};

struct TensorMemoryAlignment
{
    TensorMemoryAlignment();
    TensorMemoryAlignment(uint64_t cacheLineSizeInBytes) : alignment(cacheLineSizeInBytes) {}

    uint64_t alignment = UNDEFINED_CACHE_LINE_SIZE_IN_BYTES;
    uint64_t offset    = 0;
    bool     pinned =
        false;  // When set to true it indicates tensor is persistent in SRAM. Relevant only for static tensors.
    bool allowPermutation =
        false;  // When set to true, graph compiler may leave it as strided (in case of persistent tensor)
    TensorLocation location = UNDEFINED_LOCATION;  // Set where the tensor should be allocated.
};

struct DataInfo
{
    DataInfo() { packing.fill(1); }

    std::array<TSize, MME_MAX_CONV_DIMS> packing;
    bool                                 isVectorized  = false;
    bool                                 isLowered     = false;
    bool                                 isInterleaved = false;
    bool                                 isSwizzled    = false;
    bool                                 isBiasFixedUp = false;
    bool                                 mustBeDense   = false;
};

enum CondenseDimIndex
{
    ACTIVATION_CONDENSE_DIM = 0,
    MME_STATIC_WEIGHTS_CONDENSE_DIM = 1,
    UNINITIALIZED_CONDENSED_DIM = SYN_MAX_TENSOR_DIM + 1
};

struct Info4Bit
{
    bool             isCondensed = false;  // true if tensor strides were condensed to reflect 4bit arrangement
    CondenseDimIndex condensedDim =
        UNINITIALIZED_CONDENSED_DIM;  // the dimension index along which the Tensor strides will be condensed
};

// Holds data for a non-persistent memory section.
// Memory section can be defined in 2 ways:
// 1. ID + buffering-level
// 2. ID + offset
// Setting both buffering-level and offset is not allowed.
// In the first case offset + allocation size will be assigned later by setNonPersistentSectionInfo pass
// (according to buffering-level).
// In the second case offset already given and allocation size will be assigned later by this pass.
struct NonPersistentSectionInfo
{
    Settable<uint64_t> sectionId;       // Unique identifier for the memory section
    Settable<uint64_t> offsetFromBase;  // Offset from base allocation address
    Settable<uint32_t> bufferingLevel;  // Number of slots to hold the tensors in the section
};

// Temporary hack to try splitting to cores and using on-dcore caching in Gaudi3.
// These hints help set the tensor allocation according to the calculation of the slicing brain.
struct LitePerforationLocalityHints
{
    bool cached = false;
    bool sliced = false;
};

struct TensorAnnotation
{
    TensorAnnotation() = default;
    TensorAnnotation(uint64_t cacheLineSizeInBytes) : memory(cacheLineSizeInBytes) {}

    TensorMemoryAlignment memory;
    MemorySpaceInfo memorySpaceInfo;
    DataInfo dataInfo;
    Settable<unsigned int> sizeToAllocate;  // The requested buffer size in bytes for this tensor
    NonPersistentSectionInfo nonPersistentSectionInfo;
    TensorExUserContext      userContext = {};
    ReductionInfo            tensorReductionInfo;
    Info4Bit info4Bit;
    bool                     isAuxTensor   = false;
    bool                     isDoubleStoreTensor   = false;
    bool                     isScratchPadAuxTensor = false;
    bool sparseAccess = false;  // Tensor will be marked as sparseAccess when at least one node accesses it sparsely.
    bool      isNotNeeded  = false;  // Not persistent and no consumers
    TensorPtr origBigTensor; // If the tensor is sliced, this field indicates the original tensor pre-slicing.

    std::optional<LitePerforationLocalityHints> perforation;
    bool connectAtomicNodes = false;  // Atomic nodes are pairs of marked nodes that are required to be adjacent to each
                                      // other when compilation is finished. Tensor will be marked as connectAtomicNodes
                                      // when connecting two atomic nodes.
                                      //
    std::optional<SizeArray> m_preSlicingSize;  // for validation
};

#endif
