#pragma once

#include "types.h"
#include "transpose_nodes_creator.h"

class DmaTransposeCostModel : public TransposeCostModel
{
public:
    DmaTransposeCostModel()  = default;
    ~DmaTransposeCostModel() = default;
    uint64_t getCost(const NodeVector& extractedNodes) const override;
    uint64_t getCost(const TensorPtr& input, const TransposePermutationArray& permutation) const override;
    uint64_t getDmaCost(const TensorPtr& input) const;

private:
    uint64_t    getLogicalTransposeCost(const TensorPtr& input, const TransposePermutationArray& permutation) const;
    uint64_t    getPhysicalTransposeCost(const TensorPtr& input, const TransposePermutationArray& permutation) const;
    static bool isCyclicPermutation(const TransposePermutationArray& permutation);
    static bool isFullyUtilizedPermutation(const TransposePermutationArray& permutation, const unsigned newFcdDim);
};
