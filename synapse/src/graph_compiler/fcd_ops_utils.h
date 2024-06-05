#pragma once

#include "hal_reader/hal_reader.h"
#include "llvm/small_vector.h"
#include "transpose_permutation.h"
#include "types.h"

#include <cstdint>

namespace FcdOpsUtils
{
uint64_t calculateExpectedCost(uint64_t optimalCost, unsigned cacheLineSizeInBytes, uint64_t fcdSize);
uint64_t calculateExpectedCost(const Tensor& tensor, unsigned cacheLineSizeInBytes, uint64_t fcdSize);
uint64_t getOptimalCost(const Tensor& tensor);

struct ShiftTransposesForFcdOpsResults
{
    unsigned     newFcdDim;  // contain the index of the dim that became dim 0 (FCD)
    NodeList     newNodes;
    TensorVector newInputs;
    TensorVector newOutputs;
    uint64_t     expectedCost = std::numeric_limits<uint64_t>::max();
};

/**
 * @brief Find the dimension with best utilization for opposite shift transposes optimization
 * @return res.newFcdDim    The best utilization dimension
 * @return res.firstTransposeUtilization    The first transpose utilization
 * @return res.secondTransposeUtilization   The second (opposite) transpose utilization
 */
void findBestDimForTranspose(const HalReader&                 hal,
                             const unsigned                   startFromDim,
                             const TensorVector&              inputs,
                             const TensorVector&              outputs,
                             ShiftTransposesForFcdOpsResults& res /*inplace modify */);
/**
 * @brief Create sequences of transpose nodes before and after  operation, and also for shape tensor if provided.
          The transpose is Shift transpose, that mean: (perm[i] + 1) % tensor dim = (perm[i + 1]) % tensor dim.
          We calculate the transposes utilization and create the transposes with highest utilization
 * @return ShiftTransposesForFcdResults
 */
ShiftTransposesForFcdOpsResults createOppositeShiftTransposes(const HalReader&    hal,
                                                              std::string_view    name,
                                                              const TensorVector& input,
                                                              const TensorVector& output,
                                                              const unsigned      startFromDim = 0);

llvm_vecsmall::SmallVector<NodePtr, 3>
         createFlattenByReshapeNode(const TensorPtr& tensor, const unsigned axis, std::string_view name);
NodePtr  createFlattenShapeNode(const TensorPtr& tensor, const unsigned axis);
NodePtr  createExtractShapeNode(const TensorPtr& tensor);
NodePtr  createTransposedShape(const TensorPtr& tensor, const TransposePermutationArray& perm);
NodeList aggregateFcdByReshapeNode(const TensorPtr& tensor, const unsigned axis, std::string_view name);

}  // namespace FcdOpsUtils
