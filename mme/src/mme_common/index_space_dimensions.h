#ifndef MME__INDEX_SPACE_DIMENSIONS_H
#define MME__INDEX_SPACE_DIMENSIONS_H

#include <cstddef>

namespace MmeCommon::AccessPatternDetails
{
using Dim = size_t;
namespace Gemm
{
// These values order is aligned with GC defined index space dims designation. Until the MME is completely self relient
// and GC uses these only MME generated access pattern, the order here must remain aligned with GC designation for
// layered brain to function properly.
enum IndexSpaceDim : Dim
{
    DIM_OPERANDS_COMMON,
    DIM_MASKS_COMMON,

    DIM_OUT_FCD,
    DIM_OUT_HEIGHT,

    DIM_NUM_IDENTICAL_MASKS,  // The masks dims should be grouped together, but for now leave it here to be compatible
                              // with the legacy node dimensions.

    // TODO [SW-152867] this dimension makes no sense in the context of MME stack. It was
    // added for compatibility with the legacy implementation of MME access patterns in GC.
    DIM_BCAST,  // Special dimension for broadcasted inputs

    DIM_BATCH_0,
    DIM_BATCH_1,
    DIM_BATCH_2,

    MAX_INDEX_SPACE_DIM
};
}  // namespace Gemm

namespace Conv
{
// These values order is aligned with GC defined index space dims designation. Until the MME is completely self relient
// and GC uses these only MME generated access pattern, the order here must remain aligned with GC designation for
// layered brain to function properly.
enum IndexSpaceDim : Dim
{
    DIM_BATCH,
    DIM_DEPTH,
    DIM_HEIGHT,
    DIM_WIDTH,
    DIM_IN_CHANNELS,
    DIM_OUT_CHANNELS,
    DIM_FILTER_Q,
    DIM_FILTER_R,
    DIM_FILTER_S,

    MAX_INDEX_SPACE_DIM
};
}  // namespace Conv
}  // namespace MmeCommon::AccessPatternDetails

#endif //MME__INDEX_SPACE_DIMENSIONS_H
