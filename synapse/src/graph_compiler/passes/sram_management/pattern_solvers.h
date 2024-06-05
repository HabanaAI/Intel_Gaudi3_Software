#pragma once

#include "mme_slicing_strategy.h"
#include <queue>
#include "slicing_brain.h"
#include "cost_model_comparator.h"

// Comparator function which should return if SlicingStrategy A is better then SlicingStrategy B.
class StrategyComparator
{
public:
    bool operator()(const SlicingStrategyPtr& a, const SlicingStrategyPtr& b) const;

    struct NormalizedMetrics
    {
        float SRAMCapacity = 0;
        float MMEUtilization = 0;
        float HBMBandwidth = 0;
        float SBReuse = 0;
        float DoubleBuffered = 0;
        float walkingPattern = 0;
        bool valid = false;
    };
private:
    // normalized the metrics to be in the range of [0,1]
    NormalizedMetrics normalizeMetrics(const SlicingStrategyPtr& strategy) const;
    // calculate weighted average of all metrics.
    float getScore(const NormalizedMetrics& met) const;
};

class Solver
{
public:
    using Solution       = Bundle::Solution;
    using SlicedOperand  = Solution::SlicedOperand;
    using pSlicedOperand = std::shared_ptr<SlicedOperand>;
    using Operation      = Solution::Operation;
    using SliceReference = Operation::SliceReference;
    using SlicingData    = MmeSlicingStrategy::MmeSlicingData;

    explicit Solver(const HalReader& halReader, const pBundle& b) : m_halReader(halReader), m_strategies(), m_bundle(b)
    {
    }
    virtual ~Solver() = default;

    SlicingStrategyList& getStrategies() { return m_strategies; }
    SlicingStrategyList  getUniqueStrategies();
    SlicingStrategyList  getReducedStrategyList();
    virtual bool effectiveForBundle() = 0;
    virtual void createAllStrategies() = 0;
    void AddStrategiesForGraphSizeOptimization();

protected:
    void addStrategy(SlicingStrategyPtr s, bool printLog = true);
    const pBundle& getBundle() const { return m_bundle; }
    static pNode getFirstMMENodeFromBundle(const Bundle& bundle);
    static pNode getFirstTPCNodeFromBundle(const Bundle& bundle);
    void addStrategies(const SlicingStrategyList& newStrategies);
    void addSingleBufferStrategyIfFits(SlicingStrategyPtr strategy);
    void addStrategiesIfNotExist(const SlicingStrategyList& newStrategies);
    const HalReader& m_halReader;

private:
    bool validateStrategy(const SlicingStrategyPtr& s);
    SlicingStrategyList m_strategies;
    pBundle m_bundle;
};

class MmeBundleSolver : public Solver
{
public:
    explicit MmeBundleSolver(const HalReader& halReader, const pBundle& b)
    : Solver(halReader, b), m_mmeNode(getFirstMMENodeFromBundle(*getBundle())), m_dimController(m_mmeNode)
    {
    }

protected:
    void
    setFinalShapeInOperands(pSlicedOperand operandA, pSlicedOperand operandB, pSlicedOperand output, pNode mmeNode);
    // Relevant for convolutions only
    unsigned getNextSpatialSlicingDim(const DimVector& dims, const unsigned currentDim) const;

    const NodePtr         m_mmeNode;
    const MmeDimController m_dimController;
};

inline bool isBundleBatchGemm(const pBundle& bundle)
{
    const auto& nodes = bundle->getNodes();
    if (nodes.size() != 1) return false;
    return nodes.front()->isBatchGemm();
}

inline bool isBundleGemm(const pBundle& bundle)
{
    const auto& nodes = bundle->getNodes();
    if (nodes.size() != 1) return false;
    const auto t = nodes.front()->getNodeType();
    return t == Node::TYPE_GEMM || t == Node::TYPE_GEMM_DEDX || t == Node::TYPE_GEMM_DEDW;
}

inline const SlicingStrategyPtr& findWinningStrategy(const SlicingStrategyList& strategies,
                                                     const pBundle&             bundle,
                                                     const HabanaGraph&         graph,
                                                     const AllBrains&           slicingBrains,
                                                     bool                       resetCache = true)
{
    if (GCFG_SRAM_SLICER_COST_MODEL_ENABLED.value() && (bundle->type() == BundleType::MME))
    {
        // Cost model based comparison - the winning strategy is the one with the minimal cost.
        return *std::min_element(strategies.begin(),
                                 strategies.end(),
                                 StrategyCostModelComparator(bundle, graph, slicingBrains, resetCache));
    }
    // Metrics based comparison - the winning strategy is the one with the higher score.
    return *std::max_element(strategies.begin(), strategies.end(), StrategyComparator());
}

class TrivialSolver : public MmeBundleSolver
{
public:
    explicit TrivialSolver(const HalReader& halReader, const pBundle& b) : MmeBundleSolver(halReader, b) {}
    virtual ~TrivialSolver() = default;
    bool effectiveForBundle() override;
    virtual void createAllStrategies() override;
};

class TPCScalarPipeSolver : public Solver
{
public:
    explicit TPCScalarPipeSolver(const HalReader& halReader, const pBundle& b) : Solver(halReader, b) {}
    virtual ~TPCScalarPipeSolver() = default;
    bool effectiveForBundle() override;
    virtual void createAllStrategies() override;
};

class DMATransposeSolver : public Solver
{
public:
    explicit DMATransposeSolver(const HalReader& halReader, const pBundle& b) : Solver(halReader, b) {}
    virtual ~DMATransposeSolver() = default;
    bool         effectiveForBundle() override;
    virtual void createAllStrategies() override;
};
