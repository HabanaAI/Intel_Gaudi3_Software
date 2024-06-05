#pragma once

#include "habana_graph.h"
#include "bundle_solver.h"
#include "common_tile_size_calculator.h"

using BundleAndSolver      = std::pair<PipelineBundlePtr, BundleSolverPtr>;
using BundlesInfoContainer = std::list<BundleAndSolver>;

enum BundlingPolicy
{
    BUNDLE_BY_TRANSFORMER_PATTERNS,
    BUNDLE_BY_VISION_PATTERNS
};

class PipelineBundlizer
{
public:
    PipelineBundlizer(const HabanaGraph& graph) : m_graph(graph) {}
    virtual BundlesInfoContainer generateBundles() = 0;

protected:
    void addNodeToBundle(const NodePtr& node, PipelineBundlePtr& bundle);
    void removeNodeFromBundle(const NodePtr& node, PipelineBundlePtr& bundle);

    virtual bool validateBundles(const BundlesInfoContainer& bundles) const = 0;
    virtual bool isSupportedLogicalNode(const NodePtr& node) const          = 0;

    void removeBundle(PipelineBundlePtr bundle);

    TensorGranularity getMmeSharedInputGranularity(const MMENodeSet&        mmeNodes,
                                                   const TensorPtr&         input,
                                                   const std::vector<Dim>&  slicedDims,
                                                   const TensorGranularity& granularity) const;

    uint64_t getSlicedTensorMinSize(const TensorPtr&            t,
                                    const TensorTile::Geometry& granularity,
                                    const std::vector<Dim>&     slicedDims) const;

    bool isSupportedLogicalNodeType(const NodePtr& node) const;

    const HabanaGraph& m_graph;
};

class TPCExpansionsAndSingleMMEBundlizer : public PipelineBundlizer
{
public:
    explicit TPCExpansionsAndSingleMMEBundlizer(const HabanaGraph& graph) : PipelineBundlizer(graph) {}
    BundlesInfoContainer generateBundles() override;

protected:
    using PipelinedNode      = TPCExpansionsAndSingleMMEBundleSolver::PipelinedNode;
    using PipelineChain      = TPCExpansionsAndSingleMMEBundleSolver::PipelineChain;
    using PipelineMultiChain = TPCExpansionsAndSingleMMEBundleSolver::PipelineMultiChain;

    const size_t MAX_CHAIN_SIZE = 12;

    BundlesInfoContainer                   generateMmeWithTpcExpansionsBundles();
    virtual std::optional<BundleAndSolver> getMmeWithTpcExpansionsBundle(const NodePtr& mmeNode);

    PipelineChain
    getProducerChain(const NodePtr& mmeNode, const TensorPtr& input, const NodeSet& otherNodesInBundle) const;
    static bool   producerChainsIntersect(const PipelineMultiChain& producerChains);

    // Returns a chain of valid producers, MAX_CHAIN_SIZE at the most. If no valid chain found, returns an empty chain.
    PipelineChain expandProducerChain(const NodePtr&       finalConsumer,
                                      const PipelinedNode& nextCandidate,
                                      PipelineChain        currentChain,
                                      const NodeSet&       otherNodesInBundle) const;
    PipelinedNode createProducerCandidate(const TensorPtr& tensor, const std::vector<Dim>& slicingDims) const;
    void          clipProducerChain(PipelineChain& currentChain, unsigned maxTPCs = 0) const;

    // Returns whether the next producer to add to the chain is invalid
    bool producerIsChainBreaker(const PipelinedNode& nextCandidate,
                                const PipelineChain& currentChain,
                                const NodePtr&       finalConsumer,
                                const NodeSet&       otherNodesInBundle) const;
    bool        commonChainBreakerChecks(const NodePtr& candidateNode) const;
    static bool isTPCChainBreaker(const PipelinedNode& nextCandidate);
    bool isLogicalChainBreaker(const PipelinedNode& nextCandidate, const TensorPtr& nextTensor) const;
    bool producerCyclesChainBreaker(const PipelinedNode& nextCandidate,
                                    const PipelineChain& currentChain,
                                    const NodePtr&       finalConsumer,
                                    const NodeSet&       otherNodesInBundle) const;
    bool producerSharedInputChainBreaker(const PipelinedNode& nextCandidate,
                                         const PipelineChain& currentChain,
                                         const NodePtr&       finalConsumer,
                                         const NodeSet&       otherNodesInBundle) const;
    bool producerMultipleConnectingTensorsChainBreaker(const PipelinedNode& nextCandidate,
                                                       const PipelineChain& currentChain,
                                                       const NodePtr&       finalConsumer) const;
    bool areAssociatedWithDifferentFlashAttention(const NodePtr& nextCandidate, const NodePtr& finalConsumer) const;
    // Returns whether the next producer to add to the chain, should stop the chain expansion.
    virtual bool isProducerChainBoundary(const NodePtr& node) const;
    virtual bool isConsumerChainBoundary(const NodePtr& node) const;
    virtual bool consumerGranularityChainBreaker(const PipelinedNode& nextCandidate) const;

    unsigned selectSramInputIndexByBw(const NodePtr& mmeConsumer, bool& bothReadOnce) const;


    virtual NodeVector getNodesBundleCreationOrder(const NodeSet& nodes);
    bool               validateBundles(const BundlesInfoContainer& bundles) const override;
    bool               isSupportedLogicalNode(const NodePtr& node) const override;

    PipelineChain getConsumerChain(const NodePtr&          mmeNode,
                                   const TensorPtr&        output,
                                   const TensorPtr&        slicedInput,
                                   const std::vector<Dim>& inputSlicingDims,
                                   const NodeSet&          otherNodesInBundle) const;

    // Returns a chain of valid consumers, MAX_CHAIN_SIZE at the most. If no valid chain found, returns an empty chain.
    PipelineChain expandConsumerChain(const NodePtr&       firstProducer,
                                      const PipelinedNode& nextCandidate,
                                      PipelineChain        currentChain,
                                      const NodeSet&       otherNodesInBundle) const;
    PipelineChain createConsumerCandidates(const TensorPtr& tensor, const std::vector<Dim>& slicingDims) const;

    // Returns whether the next consumer to add to the chain is invalid
    bool consumerIsChainBreaker(const PipelinedNode& nextCandidate,
                                const PipelineChain& currentChain,
                                const NodePtr&       firstProducer,
                                const NodeSet&       otherNodesInBundle) const;
    bool consumerCyclesChainBreaker(const PipelinedNode& nextCandidate,
                                    const PipelineChain& currentChain,
                                    const NodePtr&       firstProducer,
                                    const NodeSet&       otherNodesInBundle) const;

    // Collect mme node and what can fit in SRAM from the producer chains and consumer chain into a bundle.
    virtual std::optional<BundleAndSolver> createBundle(const NodePtr&            mmeNode,
                                                        const PipelineMultiChain& producerChains,
                                                        const PipelineChain&      consumerChain,
                                                        const TensorPtr&          mmeInputToSlice);

    bool inputHasProducersChain(const TensorPtr& input, const PipelineMultiChain& producerChains);
    bool shouldSliceOnSpatialDim(const NodePtr& mmeNode, const TensorPtr& slicedInput, unsigned doubelBufFactor = 1);
    TensorGranularity getInputGranularityWithMinimalBundle(const NodePtr&            mmeNode,
                                                           const PipelineMultiChain& producerChains,
                                                           const TensorPtr&          inputToSlice,
                                                           const std::vector<Dim>&   slicedDims) const;
    TSize             predictMinSliceSizeWithMinimalBundle(const NodePtr&            mmeNode,
                                                           const PipelineMultiChain& producerChains,
                                                           const unsigned            inputIdx) const;

    std::optional<TensorPtr> getLargestInputThatFitsSram(const NodePtr& mmeNode, const unsigned largerInputIndex);

    TensorPtr selectMmeInputToSliceBySize(const NodePtr& mmeNode, const PipelineMultiChain& producerChains);
    std::optional<TensorPtr> tryPlaceBothOperandsInSram(const NodePtr&            mmeNode,
                                                        const PipelineMultiChain& producerChains);
    std::optional<TensorPtr> selectMmeInputToSliceByPerf(const NodePtr&            mmeNode,
                                                         const PipelineMultiChain& producerChains,
                                                         bool                      allowCopyInputWithoutProducers);
    std::optional<TensorPtr> selectMmeInputToSliceByCLAlignment(const NodePtr&          mmeNode,
                                                                std::array<unsigned, 2> fetchNr) const;
    std::optional<TensorPtr> selectMmeInputForSpatialSlicing(const NodePtr&            mmeNode,
                                                             const PipelineMultiChain& producerChains);
    TensorPtr                getMmeInputToSlice(const NodePtr&            mmeNode,
                                                const PipelineMultiChain& producerChains,
                                                bool                      allowCopyInputWithoutProducers);
    virtual bool             isPlacedInSramUnsliced(const NodePtr&            mmeNode,
                                                    const PipelineMultiChain& producerChains,
                                                    const unsigned            unslicedOpIndex);
    bool isUnbalancedBundle(const NodePtr& mmeNode, const PipelineMultiChain& producerChains, const unsigned inputIdx);
};

class TPCExpansionsAndSharedMMEBundlizer : public TPCExpansionsAndSingleMMEBundlizer
{
public:
    explicit TPCExpansionsAndSharedMMEBundlizer(HabanaGraph& graph) : TPCExpansionsAndSingleMMEBundlizer(graph) {}

protected:
    // Collect mme node and what can fit in SRAM from the producer chains and consumer chain into a bundle.
    std::optional<BundleAndSolver> createBundle(const NodePtr&            mmeNode,
                                                const PipelineMultiChain& producerChains,
                                                const PipelineChain&      consumerChain,
                                                const TensorPtr&          mmeInputToSlice) override;

    NodeVector getNodesBundleCreationOrder(const NodeSet& nodes) override;

    bool dedwPrefersToCacheX(const NodePtr& dedw);
    bool canCacheXInJointBundle(const NodePtr& dedw);
    bool isDedwBwBoundOnX(const NodePtr& dedw);
};
// This knows to detect Manta Ray patterns:
// Several MME nodes that share exactly one operand
// With TPC / logical chain leading to shared operand (with max length limitations)
// And optional TPC producers chains on non shared operands

class MantaRayBundlizer : public TPCExpansionsAndSingleMMEBundlizer
{
public:
    explicit MantaRayBundlizer(const HabanaGraph& graph) : TPCExpansionsAndSingleMMEBundlizer(graph) {}
    BundlesInfoContainer generateBundles() override;

protected:
    const size_t nMMEToBundle = 3;  // Limits how many MMEs to bundle together. For BERT, 3 catches KQV on FWD.
    const size_t nTPCForUnsharedOperand = 3;  // Controls max TPCs on a single producer chain for the non-shared operand

    BundlesInfoContainer generateMultiMmeWithTpcProducersBundles();

    bool isProducerChainBoundary(const NodePtr& node) const override;
    bool isConsumerChainBoundary(const NodePtr& node) const override;
    bool consumerGranularityChainBreaker(const PipelinedNode& nextCandidate) const override;

    std::optional<BundleAndSolver> createBundle(const NodePtr&            mmeNode,
                                                const PipelineMultiChain& producerChains,
                                                const PipelineChain&      consumerChain,
                                                const TensorPtr&          mmeInputToSlice) override;

    bool isSupportedLogicalNode(const NodePtr& node) const override;

    std::optional<BundleAndSolver>
    createMantaRayBundle(const MMENodeSet& mmeNodes, const std::vector<Dim>& slicingDims, const pTensor& masterOperand);

    std::vector<unsigned> getSlicingDimForSharedOperand(const TensorPtr& masterOperand, const MMENodeSet& consumers);

    std::vector<Dim> getMultiMmeCommonSlicingDims(const MMENodeSet& mmeConsumers,
                                                  const TensorPtr&  masterOperand,
                                                  Dim               selectedSlicingDim);

    bool isSharedMultiBufUseful(const PipelineChain& chain);

    void addSortedMmeNodesToBundle(const MMENodeSet&             mmeNodes,
                                   PipelineBundlePtr             bundle,
                                   MantaRaySolver::BundleParams& params);
    void addValidMmeNodesToBundle(MMENodeSet&        mmeConsumers,
                                  PipelineBundlePtr& bundle,
                                  const TensorPtr&   masterOperand,
                                  bool               partialsConsumers);
    bool isSlicedOnCommonDim(const MMENodePtr& n, const TensorPtr& t, const std::vector<Dim>& slicingDims) const;

    bool canBundleSharedInputMmeNode(const MMENodePtr& candidate,
                                     const MMENodeSet& allCandidates,
                                     const MMENodeSet& committedCandidates,
                                     const TensorPtr&  sharedInput);

    bool isBundleSupportedForConsumers(MMENodeSet         mmeNodes,
                                       PipelineChain      consumerChain,
                                       PipelineBundlePtr& bundle,
                                       unsigned int       doubleBufferFactor) const;

    bool     tensorsFitSramCapacity(uint64_t tensorsSizeInBytes, uint64_t availableSramBytes) const;
    // Calculates the minimal sliced chains SRAM usage in bytes, including buffer level factor
    uint64_t getSlicedTensorsMinSramUsage(const MantaRaySolver::BundleParams& params,
                                          const PipelineMultiChain&           slicedChains,
                                          bool                                sharedChainMaxTile) const;

    std::optional<TStride> getSharedOperandSliceAlignmentSize(const MantaRaySolver::BundleParams& params,
                                                              const TensorGranularity&            granularity) const;

    // Calculates the minimal size of each chain tensor.
    // If maxTile is false - returns the sum of all minimal tiles sizes.
    // If maxTile is true - returns the max tile size among the chain tensors.
    uint64_t getSlicedChainGranuleSize(const PipelineChain& chain, const TileSizePerTensor& tiles, bool maxTile) const;
    uint64_t getNumGranules(const TensorGranularity& sizes,
                            const TensorGranularity& granularity,
                            const std::vector<Dim>&  slicedDims) const;

    bool shouldCopyToSram(const MMENodePtr& n, const TensorPtr& t);
    bool blockNonSharedProducersChain(const MMENodePtr& mme, const TensorPtr& input, unsigned numMmesInBundle);

    bool tryPlaceSharedOperandInSram(MantaRaySolver::BundleParams& params);
    bool preferSlicingOnMultipleDims(bool fitsSram, const MantaRaySolver::BundleParams& params) const;
    bool addSharedInputFirstProducer(MantaRaySolver::BundleParams& params,
                                     PipelineBundlePtr&            bundle,
                                     const PipelineChain&          sharedProducerChain);
    void extendSharedInputProducers(MantaRaySolver::BundleParams& params,
                                    PipelineBundlePtr&            bundle,
                                    const PipelineChain&          sharedProducerChain);
    void trimSharedProducersChainToFitSram(MantaRaySolver::BundleParams& params,
                                           const PipelineChain&          candidateFullChain,
                                           uint64_t                      sramCap,
                                           bool                          allowMultiBuf);
    void addNonSharedProducers(MantaRaySolver::BundleParams& params, PipelineBundlePtr& bundle);

    void addConsumers(MantaRaySolver::BundleParams& params, PipelineBundlePtr& bundle);
    bool consumerCanBeProducer(const TensorPtr& consumerOutput, const NodePtr& originalConsumerNode) const;
    NodeSet getNextMmeConsumers(const TensorPtr& t) const;
    bool sliceNonSharedProducersChainInSramAllowed(const MantaRaySolver::BundleParams& params) const;
    void updateNonSharedProducers(MantaRaySolver::BundleParams& params, PipelineBundlePtr& bundle);
    bool sliceOnMoreThanOneDimIsAllowed(const std::vector<Dim>& slicingDims) const;
    bool extendSharedInputProducersIsAllowed(const MantaRaySolver::BundleParams& params) const;

    bool tryPlacePartialsOutputsInSram(MantaRaySolver::BundleParams& params) const;

    PipelineChain      getSharedInputCommonProducersChain(const MantaRaySolver::BundleParams& params,
                                                          const PipelineBundlePtr&            bundle);
    PipelineChain      getNonSharedProducersChain(const MMENodePtr&        n,
                                                  const TensorPtr&         input,
                                                  const PipelineBundlePtr& bundle,
                                                  const std::vector<Dim>&  slicingDims);
    PipelineMultiChain getBundledSlicedChains(const MantaRaySolver::BundleParams& params) const;

    void updateSlicingGranularityParams(MantaRaySolver::BundleParams& params) const;

    std::vector<Dim> getSlicingDimsIntersection(const std::vector<Dim>& n1SlicingDims,
                                                const std::vector<Dim>& slicingDimsIntersectionAcc,
                                                const NodePtr&          n1) const;

    bool isPlacedInSramUnsliced(const NodePtr&            mmeNode,
                                const PipelineMultiChain& producerChains,
                                const unsigned            unslicedOpIndex) override;
};
