#include "dma_transpose_cost_model.h"
#include "dma_transpose_node.h"
#include "fcd_ops_utils.h"
#include "compilation_hal_reader.h"
#include "transpose_node.h"

uint64_t DmaTransposeCostModel::getCost(const TensorPtr& input, const TransposePermutationArray& permutation) const
{
    if (TransposeNode::isPhysicalTranspose(input, permutation))
    {
        return getPhysicalTransposeCost(input, permutation);
    }
    // check if it will resolved with reshape
    if (isSameDataMemOrder(*input, permutation))
    {
        return 0;
    }
    return getLogicalTransposeCost(input, permutation);
}

uint64_t DmaTransposeCostModel::getCost(const NodeVector& extractedNodes) const
{
    uint64_t cost = 0;
    for (const auto& node : extractedNodes)
    {
        if (!node->isTranspose()) continue;
        if (node->isLogicalOperation())
        {
            const auto& logical = std::dynamic_pointer_cast<LogicalTransposeNode>(node);
            HB_ASSERT_PTR(logical);
            cost += getLogicalTransposeCost(node->getInput(0), logical->permutation());
        }
        else
        {
            const auto& physical = std::dynamic_pointer_cast<DMATransposeNode>(node);
            HB_ASSERT_PTR(physical);
            cost += getPhysicalTransposeCost(node->getInput(0), physical->permutation());
        }
    }
    return cost;
}

uint64_t DmaTransposeCostModel::getLogicalTransposeCost(const TensorPtr&                 input,
                                                        const TransposePermutationArray& permutation) const
{
    HB_ASSERT(permutation.at(0) == 0, "permutation[0] is {}, but in logical transpose it must be 0", permutation.at(0));
    unsigned    cacheLineSizeInBytes = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();

    uint64_t aggregatedFcdSize = input->getElementSizeInBytes();
    // multiply the sizes of the inner dimensions until the first dimension that change
    for (unsigned dim = 0; dim < input->getDim(); ++dim)
    {
        if (permutation.at(dim) != (TransposePermutationDim)dim) break;
        aggregatedFcdSize *= input->getSizeInElements(dim);
    }
    auto actualCost  = FcdOpsUtils::calculateExpectedCost(*input, cacheLineSizeInBytes, aggregatedFcdSize);
    auto optimalCost = FcdOpsUtils::getOptimalCost(*input);
    HB_ASSERT(actualCost >= optimalCost, "actualCost must be greater or equal to optimalCost");
    return actualCost - optimalCost;
}

bool DmaTransposeCostModel::isFullyUtilizedPermutation(const TransposePermutationArray& permutation,
                                                       const unsigned                   newFcdDim)
{
    if (permutation.size() == 2) return false;
    for (unsigned dim = 0; dim < newFcdDim; ++dim)
    {
        if (permutation[dim] != dim + 1) return false;
    }
    for (unsigned dim = newFcdDim + 1; dim < permutation.size(); ++dim)
    {
        if (permutation[dim] != dim) return false;
    }
    return true;
}

// Permutation [a,b,c,d,e] called cyclic if it can be written as [a, a + 1, a + 2, a + 3, a + 4] where each value is
// modulus permutation size.
// Example [2,3,4,0,1] is [2, 2 + 1, 2 + 2, (2 + 3) % 5, (2 + 4) % 5]
bool DmaTransposeCostModel::isCyclicPermutation(const TransposePermutationArray& permutation)
{
    for (unsigned dim = 0; dim < permutation.size(); ++dim)
    {
        if ((permutation[dim] + 1) % permutation.size() != permutation[(dim + 1) % permutation.size()]) return false;
    }
    return true;
}

// Two cases are supported:
// (1) Only dim 0 is moving (fully utilized).
// (2) Cyclic permutation (generic dma transpose).
uint64_t DmaTransposeCostModel::getPhysicalTransposeCost(const TensorPtr&                 input,
                                                         const TransposePermutationArray& permutation) const
{
    unsigned cacheLineSizeInBytes = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    unsigned newFcdDim = std::distance(permutation.begin(), std::find(permutation.begin(), permutation.end(), 0));

    if (isFullyUtilizedPermutation(permutation, newFcdDim))
    {
        // If transpose is fully utilized it mean that the write is always aligned to cache line,
        // so the bottleneck is the read cost.
        return FcdOpsUtils::calculateExpectedCost(*input, cacheLineSizeInBytes, input->getSizeInBytes(0));
    }
    HB_ASSERT(isCyclicPermutation(permutation), "Permutation must be fully utilized or cyclic");

    uint64_t outputFcdSize = input->getElementSizeInBytes();
    for (unsigned dim = 0; dim < newFcdDim; ++dim)
    {
        outputFcdSize *= input->getSizeInElements(permutation[dim]);
    }

    uint64_t inputFcdSize = input->getElementSizeInBytes();
    for (unsigned dim = newFcdDim; dim < permutation.size(); ++dim)
    {
        inputFcdSize *= input->getSizeInElements(permutation[dim]);
    }

    auto readCost  = FcdOpsUtils::calculateExpectedCost(*input, cacheLineSizeInBytes, inputFcdSize);
    auto writeCost = FcdOpsUtils::calculateExpectedCost(*input, cacheLineSizeInBytes, outputFcdSize);
    return std::max(readCost, writeCost);
}

uint64_t DmaTransposeCostModel::getDmaCost(const TensorPtr& input) const
{
    return FcdOpsUtils::getOptimalCost(*input);
}