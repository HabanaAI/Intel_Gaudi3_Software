#pragma once
#include "mme_slicing_strategy.h"
#include <vector>
#include "bundle.h"

class HabanaGraph;

using BundleList = std::list<pBundle>;
struct BundleSolvingData
{
    SlicingStrategyList strategies {};
};

using BundleSolvingDataMap = std::unordered_map<pBundle, BundleSolvingData>;
using ExpansionCandidatesSet = std::unordered_map<pNode, pBundleExpansion>;
using namespace gc::access_pattern;
class Bundlizer
{
public:
    explicit Bundlizer(HabanaGraph& graph);

    // Generate bundles in order
    void generateBundles(BundleList& mmeBundles,
                         BundleList& scalarPipeBundles,
                         BundleList& tpcBundles,
                         BundleList& rmwSectionBundles,
                         BundleList& dmaTransposeBundles);

    /**
     * Get bundles functions use for testing
     */
    // Generate a list of bundles with MME nodes
    BundleList getMMEBundles();

    // Generate a list of bundles with TPC nodes
    BundleList getTPCScalarPipeBundles();

    BundleList  getTPCBundles();

    pBundle makeBundle(NodeList& bundleNodes, BundleType type);

    bool addCandidateToBundle(pBundle& bundle, const pBundleExpansion& expansionCandidate);

    // Find MME consumer to the bundle MME node input that may extend the current strategy,
    // That is not already in existingCandidates (optional optimization - NOT IMPLEMENTED YET)
    pBundleExpansion
    findMmeInputSharingCandidate(const pMmeSlicingStrategy&    strategy,
                                 const BundleSolvingDataMap&   solvingDataPerBundle,
                                 const ExpansionCandidatesSet& existingCandidates = ExpansionCandidatesSet()) const;

    // Find a TPC producing one of the MME inputs that can be added to the bundle, according to the solution strategy,
    // That is not already in existingCandidates (optional optimization)
    pBundleExpansion findWideTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                           const ExpansionCandidatesSet& existingCandidates = ExpansionCandidatesSet()) const;

    pBundleExpansion findNarrowTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                             const ExpansionCandidatesSet& existingCandidates = ExpansionCandidatesSet()) const;

    pBundleExpansion findSlaveTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                            const ExpansionCandidatesSet& existingCandidates = ExpansionCandidatesSet()) const;


    // Find a TPC consuming the MME output that can be added to the bundle, according to the solution strategy,
    // That is not already in existingCandidates (optional optimization)
    pBundleExpansion findTpcConsumerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                       const ExpansionCandidatesSet& existingCandidates = ExpansionCandidatesSet(),
                                                       BundleExpansion::Role role = BundleExpansion::OutputConsumer) const;

    void logGraphBundlingStatus(const BundleSolvingDataMap& strategiesMap); // Not const in order to get updated execution order

    pBundle removeBundle(const pNode& node);

    bool canMMEInputConsumerBeAddedToBundle(const pBundleExpansion& candidate,
                                            const pMmeSlicingStrategy& strategy) const;

    // Check if a node can be stitched to the sliced operand
    bool isNodeEligibleForStitching(const NodePtr&        node,
                                    const pSlicedOperand& stitchedOperand,
                                    const NodePtr&        reshapeNode) const;

    pBundle findBundleByNode(const pNode& node) const;

    // Find the connecting tensor between 2 nodes: producer and consumer.
    TensorPtr getConnectingTensor(const NodePtr& producer, const NodePtr& consumer) const;

    // Checks if the operand slicing is valid according to the given node access-pattern.
    static bool isOperandSlicingValid(const NodePtr& node, const pSlicedOperand& operand);
    // Mark the nodes with RMW tensors with bundle ID. Used also by Pipeline Manager temporarily
    void bundleNodesWithRMWSectionOperand();

protected:
    pBundleExpansion findTpcProducerExpansionCandidate(const pMmeSlicingStrategy& strategy,
                                                       const pSlicedOperand& mmeSlicedInput,
                                                       const BundleExpansion::Role& role,
                                                       const ExpansionCandidatesSet& existingCandidates) const;

    HabanaGraph&                       m_graph;
    std::unordered_map<pNode, pBundle> m_nodeBundleMap;

    // Finds the producer of a tensor. If the producer is a reshape node, return it's input producer,
    // and return the reshape node as the referenced parameter.
    pNode findNonReshapeProducer(const pTensor& tensor, pNode& reshapeNode) const;

    // Check if the strategy can support TPC consumer stitching
    bool strategyIsConsumerStitchingCapable(const pMmeSlicingStrategy& strategy, const pSlicedOperand& consumed) const;

    // Check if the TPC node should be bundled in a scalar-pipe bundle and get SRAM for its scalar inputs.
    // For inputs up to 1 CL (128 bytes) the performance gain should be minimal since after the first load the data is
    // cached.
    bool shouldEnableScalarPipeBundle(const TPCNodePtr& tpcNode, tpc_lib_api::DeviceId deviceId) const;

private:
    void      generateRMWSectionBundles(BundleList& complexGuidBundles);
    void      registerNodeToBundle(pBundle& bundle, const BundleInfo& info, const NodePtr& node);
    void      addNodeToBundle(pBundle& bundle, const NodePtr& node);
    pBundle   createBundleFromNodes(const NodeVector& nodes, BundleType type);
    pBundle   getTPCBundle(const pNode& node);
    bool      isValidNodeForTPCBundle(const NodeList& bundleNodes, pNode& candidate, TensorPtr& connectingTensor);
    bool canConsumeTpcBundle(const pNode& candidate, const TensorPtr& connectingTensor, const pNode& producer) const;
    bool canAddParallelTpc(const TPCNodePtr& tpcCandidate, const NodeSet& parallelTpcs) const;
    void      removeLastNodesFromBundle(NodeList& bundleNodes, int counter);
    bool      isNodeAllowedForSlicing(const TPCNodePtr&     node,
                                      const pSlicedOperand& stitchedOperand,
                                      const NodePtr&        reshapeNode) const;
    bool      isNodeEligibleForStitchingInternal(const TPCNodePtr&     node,
                                                 const pSlicedOperand& stitchedOperand,
                                                 const NodePtr&        reshapeNode) const;
    bool      isBnStitchingAllowed(const TPCNodePtr&     node,
                                   const pSlicedOperand& stitchedOperand,
                                   const NodePtr&        reshapeNode) const;
    pSlicedOperand getNextStitchedOperand(const NodePtr& reshapeNode, const pSlicedOperand& stitchedOperand) const;
};
