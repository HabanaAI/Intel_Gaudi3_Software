#pragma once

#include "common_tile_size_calculator.h"
#include "sram_management/bundle.h"
#include "sram_management/slicing_strategy.h"
#include "node_solver.h"
#include "mme_node.h"
#include <unordered_set>

using PipelineBundle    = Bundle;
using PipelineBundlePtr = std::shared_ptr<PipelineBundle>;
using BundleStrategyPtr = SlicingStrategyPtr;

// Full bundle solver - handles a known bundle content pattern using single node solvers
class BundleSolver
{
public:
    explicit BundleSolver(PipelineBundlePtr& bundle, const HabanaGraph& graph) : m_bundle(bundle), m_graph(graph) {}
    virtual ~BundleSolver() = default;
    // Solve each node in the bundle - find the bundle constraints and apply them on the solution of each node using
    // node solvers.
    virtual BundleStrategyPtr solveBundle()                                         = 0;
    virtual void              fillBundleSolution(const BundleStrategyPtr& strategy) = 0;
    bool                      validateBundle() const;

protected:
    virtual std::unordered_set<TensorPtr> getConnectingTensors() const                             = 0;
    virtual bool                          isTensorSharedByMMEBundleNodes(const TensorPtr& t) const = 0;
    NodeVector getAdjacentBundleNodes(const TensorPtr& t, const std::unordered_set<NodePtr>& bundleNodes) const;
    bool       isBundlePersistentInputTensor(const TensorPtr& t, const std::unordered_set<NodePtr>& bundleNodes) const;
    bool       isConnectingTensor(const TensorPtr& t, const std::unordered_set<TensorPtr>& connectingTensors) const;

    PipelineBundlePtr  m_bundle;
    const HabanaGraph& m_graph;
};

using BundleSolverPtr   = std::shared_ptr<BundleSolver>;
using BundleSolversList = std::list<BundleSolverPtr>;

class TPCExpansionsAndSingleMMEBundleSolver : public BundleSolver
{
    using BaseClass = BundleSolver;

public:
    struct PipelinedNode
    {
        PipelinedNode(const NodePtr& n, const TensorPtr& t, const std::vector<Dim>& d)
        : node(n), connectingTensor(t), slicingDims(d)
        {
        }
        NodePtr   node;
        TensorPtr connectingTensor;
        std::vector<Dim> slicingDims;  // Empty in case the node is not sliced
    };
    // Holds a chain of producers to some tensor
    using PipelineChain = std::vector<PipelinedNode>;
    // Holds several chains of producers to the inputs of an operation.
    using PipelineMultiChain = std::vector<PipelineChain>;

    TPCExpansionsAndSingleMMEBundleSolver(PipelineBundlePtr& bundle,
                                          const HabanaGraph& graph,
                                          PipelineMultiChain producerChains,
                                          PipelineChain      consumerChain,
                                          const TensorPtr&   mmeInputToSlice)
    : BaseClass(bundle, graph),
      m_producerChains(std::move(producerChains)),
      m_consumerChain(std::move(consumerChain)),
      m_mmeInputToSlice(mmeInputToSlice)
    {
    }
    BundleStrategyPtr solveBundle() override;
    void              fillBundleSolution(const BundleStrategyPtr& strategy) override;

    static uint64_t getUnSlicedChainSize(const PipelineChain& chain, const HabanaGraph& g);

    static bool ignoreInSramCapacityCalculation(const NodePtr& n, const PipelineChain& chain, const HabanaGraph& g);

    static PipelineChain getMmeToFirstTpcSubChain(const PipelineChain& chain);

    static bool alignmentAllowedForTensorInBundle(const TensorPtr& t, const HabanaGraph& graph, unsigned bundleIndex);

protected:
    /**
     * @brief Encapsulates the logic of whether to ignore or not a given input node
     *        when calculating sram capacity.
     */
    class ShouldIgnoreNodeInSramCapacityCalculation
    {
    public:
        using PipelineChain = TPCExpansionsAndSingleMMEBundleSolver::PipelineChain;
        using IgnoreFunc    = std::function<bool(const NodePtr& n, const PipelineChain& chain, const HabanaGraph& g)>;

        template<typename T>
        explicit ShouldIgnoreNodeInSramCapacityCalculation(T ignoreFunc) : m_ignoreFunc(std::move(ignoreFunc))
        {
        }

        bool query(const NodePtr& n, const PipelineChain& chain, const HabanaGraph& g) const
        {
            return m_ignoreFunc(n, chain, g);
        }

        static std::unique_ptr<ShouldIgnoreNodeInSramCapacityCalculation> getIgnoreNodeMethod();

    protected:
        static bool     isNodeWithCtrlEdges(const NodePtr& n);
        static NodeList getLogicOpsSubChain(const NodePtr& n, const PipelineChain& chain);
        static bool     ignoreNonTransposeLogicals(const NodePtr& n, const PipelineChain& chain, const HabanaGraph& g);
        static bool
        ignoreNonTransposeLogicalsRelaxed(const NodePtr& n, const PipelineChain& chain, const HabanaGraph& g);

        IgnoreFunc m_ignoreFunc;
    };

    virtual BundleSolutionConstraints collectInitialConstraints();
    virtual std::map<Dim, TSize>      getSharedOperandSlicingDimsAlignmentConstraint() const;
    virtual std::vector<Dim>          getSharedOperandSlicingDims() const;
    virtual bool                      canAlignTensorToCL(const TensorPtr& tensorToAlign) const;
    virtual bool                      canSlicePrimeNodeOnMultipleDims() const;
    virtual TensorSet                 getInitialTensorsInSRAMConstraint() const;
    virtual GranularityPerTensor      getSlicingGranularityConstraint() const;
    virtual unsigned                  getSramBytesConstraint(const GranularityPerTensor& slicingGranularity);
    virtual void                      updateTensorsInSramConstraint(TensorSet& tensorsInSram) {};
    GranularityPerTensor getTensorsSlicingGranularity(const NodeSet& nodesInSram, const TensorSet& tensorsInSram) const;
    virtual bool         canSliceNonMasterOperand() const;

    virtual NodeStrategyPtr solvePrimeNode(const BundleSolutionConstraints& constraints) const;

    virtual void      addUnslicedChainToPrimeNodeStrategy(NodeStrategyPtr& primeNodeStrategy) const;
    virtual void      tryPlaceInSramForCLAlignment(NodeStrategyPtr& primeNodeStrategy) const;
    BundleStrategyPtr createInitialStrategyForBundle(const NodeStrategyPtr& nodeStrategy) const;
    NodeStrategyPtr               projectNodeStrategy(const NodePtr&           node,
                                                      const BundleStrategyPtr& bundleStrategy,
                                                      const TensorPtr&         connectingTensor) const;
    virtual void      expandInitialStrategy(BundleStrategyPtr& bundleStrategy);
    virtual void                  optimizeNumBuffers(BundleStrategyPtr& bundleStrategy) const {};
    void                          expandInitialStrategyWithConsumers(BundleStrategyPtr& bundleStrategy);
    void              addProducerNodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                              BundleStrategyPtr&     bundleStrategy,
                                                              const PipelinedNode&   producer) const;
    void              addConsumerNodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                              BundleStrategyPtr&     bundleStrategy,
                                                              const PipelinedNode&   consumer) const;
    virtual NodePtr   getBundlePrimeMmeNode() const;
    bool                          isTensorSharedByMMEBundleNodes(const TensorPtr& t) const override;
    std::unordered_set<TensorPtr> getConnectingTensors() const override;
    PipelineMultiChain            m_producerChains;
    PipelineChain                 m_consumerChain;
    const TensorPtr               m_mmeInputToSlice;

    static void gatherConnectingTensors(const PipelineMultiChain&      pipelineChains,
                                        std::unordered_set<TensorPtr>& connectingTensors);
};

class MantaRaySolver : public TPCExpansionsAndSingleMMEBundleSolver
{
    using BaseClass = TPCExpansionsAndSingleMMEBundleSolver;

public:
    struct BundleParams
    {
        MMENodeSet         mmeNodes;
        MMENodeSet         partialsConsumers;
        MMENodeSet         nonPartialsConsumers;
        PipelineMultiChain nonSharedProducers;
        PipelineChain      sharedOperandProducers;
        PipelineChain      mmeOutputConsumers;
        TensorSet          copyTensorsToSram;  // without producers chain

        TensorPtr            sharedOperand;
        std::vector<Dim>     sharedOperandSlicingDims;
        std::map<Dim, TSize> sharedOperandSlicingDimsAlignment;  // Unified util requirements for all bundled MME nodes

        TileSizePerTensor  tileSizes;
        uint64_t           sharedOperandMinSize;           // single buffer
        uint64_t           slicedOperandsSramMinSize = 0;  // multiplied by buffer level
        uint64_t           unslicedOperandsSramSize  = 0;  // unsliced chains, copied operands, partials outputs
        unsigned           doubleBufferFactor;
        bool               sharedOperandChainInSharedMultiBuf = false;
        // Enable slicing of the non-master operand when its slices can be placed concurrently in SRAM
        bool sliceNonSharedProducersChain = false;
    };

    MantaRaySolver(PipelineBundlePtr& bundle, const HabanaGraph& graph, const BundleParams& bundleParams)
    : TPCExpansionsAndSingleMMEBundleSolver(bundle,
                                            graph,
                                            !bundleParams.sharedOperandProducers.empty()
                                                ? PipelineMultiChain {bundleParams.sharedOperandProducers}
                                                : PipelineMultiChain {},
                                            bundleParams.mmeOutputConsumers,
                                            bundleParams.sharedOperand),
      m_mmeNodes(bundleParams.mmeNodes),
      m_partialsConsumers(bundleParams.partialsConsumers),
      m_nonpartialsConsumers(bundleParams.nonPartialsConsumers),
      m_nonMasterProducers(bundleParams.nonSharedProducers),
      m_masterOperandProducers(bundleParams.sharedOperandProducers),
      m_copyTensorsToSram(bundleParams.copyTensorsToSram),
      m_tileSizes(bundleParams.tileSizes),
      m_slicedOperandsSramMinSize(bundleParams.slicedOperandsSramMinSize),
      m_sharedOperandMinSize(bundleParams.sharedOperandMinSize),
      m_unslicedOperandsSramSize(bundleParams.unslicedOperandsSramSize),
      m_doubleBufferFactor(bundleParams.doubleBufferFactor),
      m_sharedOperandChainInSharedMultiBuf(bundleParams.sharedOperandChainInSharedMultiBuf),
      m_sharedOperandSlicingDimsAlignment(bundleParams.sharedOperandSlicingDimsAlignment),
      m_sliceNonSharedProducersChain(bundleParams.sliceNonSharedProducersChain),
      m_sharedOperandSlicingDims(bundleParams.sharedOperandSlicingDims)
    {
    }

    void fillBundleSolution(const BundleStrategyPtr& strategy) override;
    static void printTensorsTileGranularity(const TileSizePerTensor& tensorTileSizes);

protected:
    TensorSet            getInitialTensorsInSRAMConstraint() const override;
    GranularityPerTensor getSlicingGranularityConstraint() const override;
    unsigned             getSramBytesConstraint(const GranularityPerTensor& slicingGranularity) override;
    std::map<Dim, TSize> getSharedOperandSlicingDimsAlignmentConstraint() const override;
    std::vector<Dim>     getSharedOperandSlicingDims() const override;
    bool                 canSliceNonMasterOperand() const override;

    NodeStrategyPtr solvePrimeNode(const BundleSolutionConstraints& constraints) const override;
    static bool     effectiveSramUsage(const pSlicedOperand& slicedOperand, const PipelineChain& producers);

    void expandInitialStrategy(BundleStrategyPtr& bundleStrategy) override;
    void  optimizeNumBuffers(BundleStrategyPtr& bundleStrategy) const override;
    void expandStrategyWithSharedProducersChain(BundleStrategyPtr& bundleStrategy);
    void expandStrategyWithSharedMmeNodes(BundleStrategyPtr& bundleStrategy) const;
    void  expandStrategyWithConsumersChain(BundleStrategyPtr& bundleStrategy);
    void expandStrategyWithNonSharedProducersChains(BundleStrategyPtr& bundleStrategy);
    void updatePartialsSharedMmeOutputs(BundleStrategyPtr& bundleStrategy);
    void updateCopyToSramMmeInputs(BundleStrategyPtr& bundleStrategy);
    void tryAlignMmeInputsToCacheline(BundleStrategyPtr& bundleStrategy);
    TSize tryAlignSingleOperandToCacheline(BundleStrategyPtr& bundleStrategy, const pSlicedOperand& slicedOperand);
    bool isAlreadyAligned(const pSlicedOperand& slicedOperand);
    bool isValidForAlignment(const pSlicedOperand& slicedOperand);

    void addUnslicedChainToPrimeNodeStrategy(NodeStrategyPtr& primeNodeStrategy) const override {}
    void tryPlaceInSramForCLAlignment(NodeStrategyPtr& primeNodeStrategy) const override {}
    void addFwdMappedNodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                  BundleStrategyPtr&     bundleStrategy,
                                                  const NodePtr&         node,
                                                  const TensorPtr&       connectingTensor) const;
    uint64_t calcEffectiveSliceSize(const pSlicedOperand slicedOp, unsigned numBuffers) const;

    // Returns one of the MME nodes that will dictate the slice size for the master operand
    NodePtr                       getBundlePrimeMmeNode() const override;
    bool                          isTensorSharedByMMEBundleNodes(const TensorPtr& t) const override;
    std::unordered_set<TensorPtr> getConnectingTensors() const override;

    MMENodeSet              m_mmeNodes;
    MMENodeSet              m_partialsConsumers;
    MMENodeSet              m_nonpartialsConsumers;
    PipelineMultiChain      m_nonMasterProducers;
    PipelineChain           m_masterOperandProducers;
    TensorSet               m_copyTensorsToSram;  // without producers chain
    uint64_t                m_allocatedSram = 0;
    const TileSizePerTensor m_tileSizes;
    uint64_t                m_slicedOperandsSramMinSize = 0;
    uint64_t                m_sharedOperandMinSize = 0;
    uint64_t                m_unslicedOperandsSramSize = 0;
    unsigned                m_doubleBufferFactor;
    bool                    m_sharedOperandChainInSharedMultiBuf;
    std::map<Dim, TSize>    m_sharedOperandSlicingDimsAlignment;
    bool                    m_sliceNonSharedProducersChain;
    std::vector<Dim>        m_sharedOperandSlicingDims;
};
class SharedMmeProducerChainBundleSolver : public TPCExpansionsAndSingleMMEBundleSolver
{
    using BaseClass = TPCExpansionsAndSingleMMEBundleSolver;

public:
    explicit SharedMmeProducerChainBundleSolver(PipelineBundlePtr& bundle,
                                                const HabanaGraph& graph,
                                                PipelineMultiChain producers,
                                                PipelineChain      consumers,
                                                NodePtr            primeNode,
                                                NodePtr            slaveNode,
                                                const TensorPtr&   mmeInputToSlice)
    : TPCExpansionsAndSingleMMEBundleSolver(bundle, graph, producers, consumers, mmeInputToSlice),
      m_primeNode(primeNode),
      m_slaveNode(slaveNode)
    {
    }

protected:
    unsigned getSramBytesConstraint(const GranularityPerTensor& slicingGranularity) override;
    void     expandInitialStrategy(BundleStrategyPtr& bundleStrategy) override;
    void     addSharedMMENodeSolutionToBundleStrategy(const NodeStrategyPtr& nodeStrategy,
                                                      BundleStrategyPtr&     bundleStrategy,
                                                      const NodePtr&         node) const;
    void     updateTensorsInSramConstraint(TensorSet& tensorsInSram) override;
    bool     isTensorSharedByMMEBundleNodes(const TensorPtr& t) const override;
    void     tryPlaceInSramForCLAlignment(NodeStrategyPtr& primeNodeStrategy) const override {}

    bool          m_isPrimeNonSharedInSram = false;
    bool          m_isPrimeOutputInSram    = false;
    bool          m_isSlaveNonSharedInSram = false;
    bool          m_isSlaveOutputInSram    = false;
    const NodePtr m_primeNode;
    const NodePtr m_slaveNode;
};
