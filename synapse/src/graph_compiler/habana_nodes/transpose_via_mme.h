#pragma once

#include "hal_reader/hal_reader.h"
#include "mme_transpose_cost_model.h"
#include "transpose_nodes_creator.h"
#include "transpose_permutation.h"
#include "transpose_strategies.h"
#include "types.h"

class TransposeViaNativeMME : public TransposeViaPhysical
{
public:
    bool             canBeUsed(const TransposeNodeParams& params, const HalReaderPtr& hal) const override;
    NodeVector       extract(const TransposeNodeParams& params, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose by native MME"; }
    uint64_t         calculateCost(const TransposeNodeParams& params, const HalReaderPtr& hal) const override
    {
        return m_costModel.getCost(extract(params, hal));
    }

    /**
     * @brief Returns the permuted part of input permutation disregarding
     *        the rightmost (slow changing) identity dims.
     */
    static TransposePermutationArray getEffectivePermutation(const TransposePermutationArray& perm);

    /**
     * @brief Returns whether a transpose defined by input permutation is supported.
     */
    static bool supported(const TransposePermutationArray& permutation);

private:
    using MaxMinSizes = std::pair<SizeVector, SizeVector>;

    /**
     * @brief Returns the permutation (1,0,2,3,...,rank-1).
     */
    TransposePermutationArray getTransposePermutation(unsigned rank) const;

    /**
     * @brief Creates the required reshapes for lowering the transpose such that
     *        permuted dims are aggregated into a 2D physical transpose.
     *        Before: [in]->transpose->[out]
     *        After : [in]->reshape->[]->transpose->[]->reshape->[out]
     *
     *        Lowering is performed by flattening dimensions transposed together around the
     *        newFcd dimension pivot while leaving the non permuted (identity) dims untouched.
     *        Resulting lowered shape (fcd left):
     *        [0*..*(newFcd-1)][newFcd*..*last permuted dim][last permuted dim + 1,..]
     * @return std::tuple<reshape nodes, new transpose input, new transpose output>
     */
    std::optional<std::tuple<NodeVector, TensorPtr, TensorPtr>>
    createLoweringSequence(const TensorPtr& in, const TensorPtr& out, const TransposePermutationArray& perm) const;

    /**
     * @brief Returns N max and min sizes of a lowered transpose given inputs.
     */
    MaxMinSizes getLoweredSizes(const TensorPtr& in, const TransposePermutationArray& perm) const;

    /**
     * @brief Create a concrete mme transpose node
     */
    NodePtr createMmeTranspose(const TensorPtr& in, const TensorPtr& out, const std::string& name) const;

    /**
     * @brief Create ExtractShape, ReshapeShape and TransposeShape nodes for lowered
     *        transpose reshapes with dynamic shape.
     *        Before: [in]->reshape->transpose->reshape->[out]
     *        After:  [in]--------------------------------->[]->reshape->transpose->[]->reshape->[out]
     *                   \               []->reshapeShape->[]---^                      /
     *                    \                                                           /
     *                      extractShape->[]->transposeShape->[]---------------------^
     *
     * @return std::tuple<ExtractShape, ReshapeShape, TransposeShape>
     */
    std::tuple<NodePtr, NodePtr, NodePtr> createTransposeLoweringShapes(const TensorPtr&                 transposeIn,
                                                                        MaxMinSizes                      loweredSizes,
                                                                        const TransposePermutationArray& perm) const;

    /**
     * @brief Returns the einsum equation string corresponding to the first transpose lowering reshape.
     *        Aggregate dims between [0,newFcd-1] and [newFcd, last permuted Dim] and the remaining
     *        dimensions are untouched.
     *        Examples:
     *        perm=(230145), newFcd=2, lastPermutedDim=3 -> equation: a,b,c,d,e,f->a*b,c*d,e,f
     *        perm=(12340), newFcd=1, lastPermutedDim=4  -> equation: a,b,c,d,e->a,b*c*d*e
     */
    std::string createTransposeLoweringEinsumEquation(unsigned dim, unsigned newFcd, unsigned lastPermutedDim) const;

    bool isHighRank(const TensorPtr& t) const;

    MmeTransposeCostModel m_costModel;
};

class TransposeViaMME : public TransposeViaPhysical
{
public:
    using PhysicalTransposeStrategy = TransposeViaNativeMME;
    TransposeViaMME(const LogicTransposeStrategies& logicStrategies) : m_logicStrategies(logicStrategies) {}
    bool             canBeUsed(const TransposeNodeParams& params, const HalReaderPtr& hal) const override;
    NodeVector       extract(const TransposeNodeParams& params, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose by MME and Logical"; }
    uint64_t         calculateCost(const TransposeNodeParams& params, const HalReaderPtr& hal) const override
    {
        return m_costModel.getCost(extract(params, hal));
    }

private:
    /**
     * @brief Get the first logical transpose strategy applicable for input params.
     *        Assuming logic transpose is applicable via at least one of the startegies.
     */
    const TransposeNodeStrategy* getLogicTransposeStrategy(const TransposeNodeParams& logicTransposeParams,
                                                           const HalReaderPtr&        hal) const;

    /**
     * @brief Create the appropriate transpose node given input params and hal.
     */
    NodeVector createTranspose(const TransposeNodeParams& params, const HalReaderPtr& hal) const;

    /**
     * @brief Given an original transpose params and a vector of split permutations,
     *        create the appropriate transpose for each permutation.
     */
    NodeVector createTransposes(const TransposeNodeParams&          origParams,
                                const TransposePermutationArrayVec& permutations,
                                const HalReaderPtr&                 hal) const;

    /**
     * @brief Split input permutation into a supported physical->logical or reversed permutation.
     */
    TransposePermutationArrayVec splitPermutation(const TransposePermutationArray& perm, bool reversed) const;

    MmeTransposeCostModel    m_costModel;
    LogicTransposeStrategies m_logicStrategies;
};
