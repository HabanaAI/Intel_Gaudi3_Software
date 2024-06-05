#include "transpose_utils.h"
#include "compilation_hal_reader.h"
#include "defs.h"
#include "dma_transpose_helper.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "transpose_permutation.h"
#include "types.h"
#include <cstdint>
#include <string>

std::tuple<uint32_t, TSize, TSize> getSizeAfterUtilizationProtection(const TSize* inputSizes, uint32_t dim, const DmaTransposeHelper& helper)
{
    uint64_t remainingTwoExponent = helper.getValidNumLinesRequiredTwoExponent();
    for (size_t i = 1; i < dim; i++)
    {
        TSize inputSize = inputSizes[i];
        if (inputSize == 0) break;  // we don't support 0 sized dims

        // __builtin_ffsll returns one plus the index of the least significant 1-bit of x,
        // or if x is zero, returns zero.
        size_t firstSetLsb = __builtin_ffsll(inputSize) - 1;

        if (remainingTwoExponent <= firstSetLsb)
        {
            return std::make_tuple(i, 1ULL << remainingTwoExponent, inputSize >> remainingTwoExponent);
        }
        else
        {
            remainingTwoExponent -= firstSetLsb;
        }
    }
    HB_ASSERT(false, "No dimension satisfies valid num lines");
    return {};
}


