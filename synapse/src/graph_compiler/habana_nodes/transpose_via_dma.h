#pragma once

#include "transpose_nodes_creator.h"
#include "transpose_strategies.h"
#include "dma_transpose_cost_model.h"

enum DmaTransposePriority : uint64_t
{
    UNUSED_PRIORITY,
    OPTIMIZED_ROTATION_PRIORITY,
    TRANSPOSE_UTILIZED_PHYSICAL_THEN_LOGICAL,
    DOUBLE_TRANSPOSE_PRIORITY,
    FULLY_UTILIZED_PRIORITY,
    GENERIC_DMA_PRIORITY,
    GENERIC_DMA_WITH_CASTS_PRIORITY,
    // keep last
    LAST_PRIOTIRY
};

class DmaTransposeStrategy
{
public:
    virtual bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const = 0;
    virtual NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const   = 0;
    virtual uint64_t   cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const      = 0;
    virtual std::string_view strategyName() const                                            = 0;
    // The lowest, the better
    virtual uint64_t priority(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const = 0;

protected:
    DmaTransposeCostModel m_costModel;

    static TransposePermutationArrayVec getSplittedPermutationsDma(const TransposePermutationArray& permutation,
                                                                   bool preferLogicalBeforePhysical);
    static uint32_t                     getFcdIndex(const TransposePermutationArray& permutation);
    static bool                         isRotation(const TransposePermutationArray& permutation, unsigned int dim);
    static TSize getRotationWriteSizeInElements(const Tensor* input, const TransposePermutationArray& p);
    static TSize getRotationReadSizeInElements(const Tensor* input, const TransposePermutationArray& p);
    static bool  isOptimizedRotation(const Tensor*                    input,
                                     const TransposePermutationArray& permutation,
                                     const DmaTransposeEngineParams&  params);
};

using StrategyPriorityPair = std::pair<const DmaTransposeStrategy*, uint64_t>;
using TransposeStrategyVec = llvm_vecsmall::SmallVector<StrategyPriorityPair, DmaTransposePriority::LAST_PRIOTIRY>;

class TransposeViaDMA : public TransposeViaPhysical
{
public:
    bool canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override
    {
        return true;
    };
    NodeVector       extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose by dma"; }
    // Can be called recursively
    static NodeVector skipPriorityCreateTransposeNodes(const TransposeNodeParams& transposeNodeParams,
                                                       const HalReaderPtr&        hal,
                                                       DmaTransposePriority       skipPriority = UNUSED_PRIORITY);
    virtual uint64_t  calculateCost(const TransposeNodeParams& transposeNodeParams,
                                    const HalReaderPtr&        hal) const override;

private:
    static TransposeStrategyVec initiateSubclasses(const TransposeNodeParams& transposeNodeParams,
                                                   const HalReaderPtr&        hal,
                                                   DmaTransposePriority       skipStaticPriority);
    NodeVector                  createTransposeNode(const TensorPtr&                 in,
                                                    const TensorPtr&                 out,
                                                    const TransposePermutationArray& permutation,
                                                    std::string_view                 name);
};
