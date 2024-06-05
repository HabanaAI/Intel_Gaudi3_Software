#pragma once

#include "bundle.h"
#include "bundlizer.h"
#include "hal_reader/hal_reader.h"
#include "log_manager.h"
#include "mme_shared_input.h"
#include "mme_slicing_strategy.h"

class HabanaGraph;

class Solver;

/* Class responsible for generating a plan (solution) for tensor slicing on a specific bundle.
 * will utilize Solvers to try different SlicingStrategies and eventually pick the best slicing for the node. */
class SlicingBrain
{
public:
    // TODO SW-7587 - slicing brain doesn't need a graph
    explicit SlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true);

    virtual ~SlicingBrain(){}

    // Control knobs used to tweak and optimize the SlicingBrain from the outside.
    thread_local static struct Knobs
    {
        // maximum SRAM  capacity to be taken into consideration when slicing.
        uint64_t maxSRAMCapInBytes;
        // maximum slice size on the wide input - will be aligned to the MME geometry by the solver.
        unsigned maxWideSliceSizeFactor_nonCommon2D;
        unsigned maxWideSliceSizeFactor_nonCommon4D;
        // maximum slice size on the narrow input - will be aligned to the MME geometry by the solver.
        unsigned maxNarrowSliceSize;
        // minimum CD slice size when partials are needed.
        unsigned minCDSizeForPartials;
        // the factor in which we multiply the slice sizes in the graph size optimization solver
        double graphSizeOptimizationMultiplicationFactor;
        // device frequency
        double   freqGHz;
        bool     snakeWalkingTraversal;
        // When multiple engines are working in parallel, this knob approximate the imperfection of their pipelining
        // i.e. the total processing time of the bundle will be the maximal processing time of one of the engines times
        // this factor
        double   aggProcessingTimePipeliningFactor;
        double   hbmAvailableBWGBps;
        // Take strategies from all effective solvers
        bool     allowMultipleSolvers;
        // After the strategies are created by the solvers, trying to eliminate strategies which have more output slices
        // than this threshold in case there is still enough room in SRAM for stitching and pre-fetching the next
        // bundle. This is done in order to improve the compilation time.
        unsigned numOfSlicesThreshold;
        // Used in cost-model comparator to determine by how many percent the difference between the HBM traffic of
        // the compared strategies is considered significant.
        double hbmTrafficDiffThreshold;
    } knobs;

protected:
    const HabanaGraph& m_graph;

    void initKnobsValues();
};

class MMESlicingBrain : public SlicingBrain
{
public:
    using SolverList = std::list<std::shared_ptr<Solver> >;

    explicit MMESlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true);

    // Return a list of solution strategies ordered by priority - from highest to lowest.
    virtual SlicingStrategyList getSolutionStrategies(const pBundle& bundle) const;

protected:
    SolverList getSolversForBundle(const pBundle& bundle) const;
};

class TPCSlicingBrain : public SlicingBrain
{
public:
    explicit TPCSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlicingBrain(graph, initKnobsFromHal)
    {
    }

    // Return a list of solution strategies ordered by priority - from highest to lowest.
    virtual SlicingStrategyList getSolutionStrategies(const pBundle& bundle) const;

protected:
    std::shared_ptr<Solver> getSolverForBundle(const pBundle& bundle) const;
};

// while the MME brain is a "master" brain that makes the initial decisions,
// the slave brain makes its decisions based on those decisions by the master brain.
class SlaveSlicingBrain : public SlicingBrain
{
public:
    explicit SlaveSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlicingBrain(graph, initKnobsFromHal)
    {}

    // update producer strategy and update expansion candidate at the end if needed
    // May update the candidate if needed.
    virtual bool addProducerToStrategy(pBundleExpansion& expansionCandidate, pMmeSlicingStrategy& strategy) const;

    // Update strategy with sliced consumer that reads the stitched operand from SRAM.
    // May update the candidate if needed.
    virtual bool addConsumerToStrategy(pBundleExpansion& expansionCandidate, pMmeSlicingStrategy& strategy) const;

protected:
    virtual NodePtr getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const = 0;

    virtual bool validateNodeToBeStitched(const NodePtr& node) const = 0;

    virtual pBackwardSliceMapping mapOutputToInputs(const NodePtr& node,
                                                    const std::list<pSlicedOperand>& inputs,
                                                    const pSlicedOperand& output) const;

    // Create mapping of a slice of the keyInput operand to slices in allInputs and allOutputs
    virtual pForwardSliceMapping mapSlicedOperandForward(const NodePtr& node,
                                                         const pSlicedOperand& keyInput,
                                                         const std::list<pSlicedOperand>& allInputs,
                                                         const std::list<pSlicedOperand>& allOutputs) const;

    virtual void updateProducerExpansionCandidateIfNeeded(pBundleExpansion& expansionCandidate,
                                                          const std::list<pSlicedOperand>& slicedInputs) const
    {
    }

    virtual void updateConsumerExpansionCandidateIfNeeded(pBundleExpansion& expansionCandidate,
                                                          const std::list<pSlicedOperand>& slicedOutputs) const
    {
    }

    virtual void setOperandMemoryLocation(pSlicedOperand slicedOperand, const pSlicedOperand& stitchedOperand) const
    { // default - do nothing = operand is in dram
    }

    // Return false if the candidate cannot be stitched into the bundle
    virtual bool validateConsumerCandidate(const pBundleExpansion&) const;

    virtual bool allowOperandReshape() const;

private:
    // Given a sliced operand of a node, create sliced operands that fit it - meaning
    // they either have the same chunk size or they are only sliced on the FCD
    std::list<pSlicedOperand> getSlicedOperandsByStitchedOperand(const TensorVector&   operands,
                                                                 const pSlicedOperand& stitchedOperand,
                                                                 bool                  allowReshape) const;

    // Adjust the input chunk size according to the output.
    static void setOperandChunkSizeByStitchedOperand(pSlicedOperand&       slicedOperand,
                                                     const pSlicedOperand& slicedStitchedOperand,
                                                     bool                  allowReshape);
};

/*
 * This class is in charge of stitching MME shared input consumer candidates.
 * There are 2 ways of stitching - same position and different position
 * Same position stitching - happens when the shared input is on the same input index (position)
 *                           regarding the master node and the slave node (not transposed)
 *                           or on different position but transposed regarding one of the nodes.
 *                           In this stitching method the slave node is sliced on the non-common dim.
 * Different position stitching - happens when the shared operand is on different input index (position)
 *                                regarding the master and the slave node, or on the same position but
 *                                transposed regarding one of the nodes.
 *                                In this stitching method there will be partials (slicing on the common dim) on the master or the slave node.
 *                                If the master is sliced on the common dim - the slave will be sliced on the non-common dim,
 *                                and vice-verse.
 * This class will find the optional candidates in the graph, figure out the stitching method needed,
 * the best slicing of the operands and creating the actual stitched operands.
 */
class MMESlaveBrain : public SlaveSlicingBrain
{
public:
    explicit MMESlaveBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlaveSlicingBrain(graph, initKnobsFromHal)
    {
    }

    /**
     * Add MME node which share the same input as the master MME node
     */
    void addSharedOperandMme(const pBundleExpansion& expansionCandidate, pMmeSlicingStrategy& strategy) const;

    bool validateNodeToBeStitched(const NodePtr& masterNode, const pBundle& masterBundle, const pBundle& slaveBundle);
    // will handle the slicing of the slave operands to fit the remaining SRAM capacity
    pBundleExpansion adjustCandidateToStrategy(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const;
private:
    virtual NodePtr getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const;

    virtual bool validateNodeToBeStitched(const NodePtr& node) const;

    void createStitchedOperands(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const;
    bool needToPerformSamePositionStitching(const pBundleExpansion& candidate) const;
    bool isSameInputIndex(const pTensor& sharedInput, const NodePtr& masterNode, const NodePtr& slaveNode) const;
    bool isSharedInputTransposed(const pTensor& sharedInput, const NodePtr& node) const;
    void doSamePositionStitching(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const;
    void doBatchGemmStitching(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const;
    void doDifferentPositionStitching(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const;
    void createSlicedOutputOperand(const pSlicedOperand& sharedOperand,
                                   const pSlicedOperand& nonSharedOperand,
                                   pSlicedOperand& slaveOutputOperand,
                                   const NodePtr& slaveNode) const;
    // resets the slaveOperands to initial state - not in sram and no slicing.
    void resetSlaveOperandsOfCandidate(const pBundleExpansion& candidate, const pMmeSlicingStrategy& strategy) const;

    SharedMMEInputCandidateHandler m_candidateHandler;
};

class TPCSlaveBrain : public SlaveSlicingBrain
{
public:
    explicit TPCSlaveBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlaveSlicingBrain(graph, initKnobsFromHal)
    {
    }

protected:
    virtual NodePtr getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const override;
    virtual bool validateNodeToBeStitched(const NodePtr& node) const override;
    virtual pBackwardSliceMapping mapOutputToInputs(const NodePtr& node,
                                                    const std::list<pSlicedOperand>& inputs,
                                                    const pSlicedOperand& output) const override;
};

class ReshapeSlicingBrain: public SlaveSlicingBrain
{
public:
    explicit ReshapeSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlaveSlicingBrain(graph, initKnobsFromHal)
    {}

protected:
    virtual NodePtr getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const override;
    virtual bool validateNodeToBeStitched(const NodePtr& node) const override;
    virtual pBackwardSliceMapping mapOutputToInputs(const NodePtr& node,
                                            const std::list<pSlicedOperand>& inputs,
                                            const pSlicedOperand& output) const override;
    void updateProducerExpansionCandidateIfNeeded(pBundleExpansion& expansionCandidate,
                                                  const std::list<pSlicedOperand>& slicedInputs) const override;
    void updateConsumerExpansionCandidateIfNeeded(pBundleExpansion& expansionCandidate,
                                                  const std::list<pSlicedOperand>& slicedOutputs) const override;
    virtual void                  setOperandMemoryLocation(pSlicedOperand        slicedOperand,
                                                           const pSlicedOperand& stitchedOperand) const override;

    virtual bool allowOperandReshape() const override;
};

class AccessPatternSlicingBrain : public SlaveSlicingBrain
{
public:
    explicit AccessPatternSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlaveSlicingBrain(graph, initKnobsFromHal)
    {
    }

    virtual bool addProducerToStrategy(pBundleExpansion&    expansionCandidate,
                                       pMmeSlicingStrategy& strategy) const override;
    virtual bool addConsumerToStrategy(pBundleExpansion&    expansionCandidate,
                                       pMmeSlicingStrategy& strategy) const override;
};

class TPCAccessPatternSlicingBrain : public AccessPatternSlicingBrain
{
public:
    explicit TPCAccessPatternSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : AccessPatternSlicingBrain(graph, initKnobsFromHal)
    {
    }

protected:
    NodePtr getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const override;
    bool    validateNodeToBeStitched(const NodePtr& node) const override;
};

class ReshapeAccessPatternSlicingBrain : public AccessPatternSlicingBrain
{
public:
    explicit ReshapeAccessPatternSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : AccessPatternSlicingBrain(graph, initKnobsFromHal)
    {
    }

    virtual bool addProducerToStrategy(pBundleExpansion&    expansionCandidate,
                                       pMmeSlicingStrategy& strategy) const override;
    virtual bool addConsumerToStrategy(pBundleExpansion&    expansionCandidate,
                                       pMmeSlicingStrategy& strategy) const override;

protected:
    NodePtr getNodeToBeStitched(const pBundleExpansion& expansionCandidate) const override;
    bool    validateNodeToBeStitched(const NodePtr& node) const override;
    void    updateNextStitchedOperand(pBundleExpansion&          expansionCandidate,
                                      const pMmeSlicingStrategy& strategy,
                                      const TensorPtr&           nextStitchedTensor) const;
    void    updateOperandMemoryLocation(pSlicedOperand slicedOperand, const pSlicedOperand& stitchedOperand) const;
};

class DmaTransposeSlicingBrain : public SlicingBrain
{
public:
    explicit DmaTransposeSlicingBrain(const HabanaGraph& graph, bool initKnobsFromHal = true)
    : SlicingBrain(graph, initKnobsFromHal)
    {
    }

    // Return a list of solution strategies ordered by priority - from highest to lowest.
    virtual SlicingStrategyList getSolutionStrategies(const pBundle& bundle) const;

protected:
    std::shared_ptr<Solver> getSolverForBundle(const pBundle& bundle) const;
};

class AllBrains
{
public:
    AllBrains(const HabanaGraph& graph)
    : m_mmeBrain(graph),
      m_tpcBrain(graph),
      m_mmeSlaveBrain(graph),
      m_dmaTransposeSlicingBrain(graph),
      m_tpcSlaveBrain(
          GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value()
              ? std::static_pointer_cast<SlaveSlicingBrain>(std::make_shared<TPCSlaveBrain>(graph))
              : std::static_pointer_cast<SlaveSlicingBrain>(std::make_shared<TPCAccessPatternSlicingBrain>(graph))),
      m_reshapeBrain(
          GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value()
              ? std::static_pointer_cast<SlaveSlicingBrain>(std::make_shared<ReshapeSlicingBrain>(graph))
              : std::static_pointer_cast<SlaveSlicingBrain>(std::make_shared<ReshapeAccessPatternSlicingBrain>(graph)))
    {}

    MMESlicingBrain     m_mmeBrain;
    TPCSlicingBrain     m_tpcBrain;
    MMESlaveBrain       m_mmeSlaveBrain;
    DmaTransposeSlicingBrain           m_dmaTransposeSlicingBrain;
    std::shared_ptr<SlaveSlicingBrain> m_tpcSlaveBrain;
    std::shared_ptr<SlaveSlicingBrain> m_reshapeBrain;
};
