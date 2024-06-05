#ifndef MME__ACCESS_PATTERN_H
#define MME__ACCESS_PATTERN_H

#include <unordered_map>
#include "llvm/small_vector.h"
#include "mme_common/mme_common_enum.h"

namespace MmeCommon
{
// Estimations to balance small vector performance but keep the interface flexible in case something exceeds.
constexpr static unsigned EST_MAX_TENSOR_DIMS = 5;
constexpr static unsigned EST_MAX_INDEX_SPACE_DIMS = 10;
constexpr static unsigned EST_MAX_INPUTS = 6;  // A, B, Masks, Aux
constexpr static unsigned EST_MAX_OUTPUTS = 2;  // C, C_copy

template<class T, unsigned N>
using Vector = llvm_vecsmall::SmallVector<T, N>;

// Roles for each of the operands that can be attached to an MME operation
using RoleBase = uint8_t;
enum class OperandRole : RoleBase
{
    X,
    W,
    Y,
    OUTPUT_COPY,
    BIAS,
    SHAPE,
    MASK_A,
    MASK_B,
    SCRATCH_PAD,  // Intermediate computation buffer
    CONST,  // For transpose, memcpy, reduce-add in CD parallelism, ...

    INVALID
};

// This struct describes the semantic attributes of a problem solved by MME. It's meant to be HW independent and
// extensible in case new problems are added to the MME stack.
struct LayerSemantics
{
    struct TensorProperties
    {
        using TensorShape = Vector<uint64_t, EST_MAX_TENSOR_DIMS>;

        TensorShape shape;
    };

    // The type of the MME operation
    EMmeOpType op = e_mme_ab;

    // The shapes of the operands. Only operands that are involved in the problem should appear in this map.
    std::unordered_map<OperandRole, TensorProperties> operandShapes;

    // In case of fwd or bwd convolution - describe the parameters of the convolution. Must be set in case of
    // convolution family op type, must be nullopt otherwise.
    std::optional<MmeConv> convParams;
};

// This struct defines 2 things:
// 1. A semantic division of the node work into atomic units (index space). Each index elemnt represent a non-divisible
// part of the node's total work. This division is described using a multi-dimensional virtual geometry, where each
// element is an 'index'
// 2. A mapping from a multi-dimensional region in the index space geometry, that describe some part of the node's total
// work (a.k.a node ROI, or ISR), to the region in each tensor that's required for that part of the work (a.k.a tensor
// ROI or ISMR)
struct AccessPattern
{
    // Defines the mapping of a multi-dimensional region in the index space to a multi-dimensional region in a tensor.
    // The mapping is done by traversing the tensor dims and applying the DimAccessPattern formula on each.
    struct TensorAccessPattern
    {
        // Represents the access to a single dimension, given some region in the index space.
        // The given indices segment [N, M] in index space dimension 'indexSpaceDim', the tensor ROI in this dimension
        // would be the segment: [offset + (N * stride), offset + (N * stride) + (M - N + 1) * size - 1]
        struct DimAccessPattern
        {
            size_t indexSpaceDim;
            int64_t offset;
            uint64_t size;
            uint64_t stride;
        };

        // useful to name this for tests
        using DimsAPVector = Vector<DimAccessPattern, EST_MAX_TENSOR_DIMS>;

        DimsAPVector dimsAccessPattern;
    };

    // useful to name these for tests
    using IndexSpaceVector = Vector<uint64_t, EST_MAX_INDEX_SPACE_DIMS>;
    using InputsAPVector = Vector<TensorAccessPattern, EST_MAX_INPUTS>;
    using OutputsAPVector = Vector<TensorAccessPattern, EST_MAX_OUTPUTS>;
    using OperandAPMap = std::unordered_map<OperandRole, TensorAccessPattern>;

    IndexSpaceVector indexSpace;
    OperandAPMap operandAccessPatterns;

    // Convenience to get access pattern by input index or the output instead of by role
    OperandRole roleA = OperandRole::INVALID;
    OperandRole roleB = OperandRole::INVALID;
    OperandRole roleC = OperandRole::INVALID;
};

// Creates an AccessPattern object from any given layer params
struct AccessPatternFactory
{
    static AccessPattern createFrom(const MmeLayerParams*);
    static AccessPattern createFrom(const LayerSemantics*);

    // Modifies the given access pattern to reflect the given parallelization:
    // 1. Change the parallelized dimensions index space and access patterns
    // 2. Add scratch-pad and const access pattern to reflect the needed auxiliary buffers for deterministic
    //    parallelization.
    static void applyParallelism(AccessPattern*, size_t idxSpcDim, size_t parallelism);
};
}  // namespace MmeCommon

#endif //MME__ACCESS_PATTERN_H
