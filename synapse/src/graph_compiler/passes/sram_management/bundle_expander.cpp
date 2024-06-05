#include "slicing_brain.h"
#include "habana_graph.h"
#include "pattern_solvers.h"
#include "slicing_utils.h"
#include "graph_size_optimization_solver.h"
#include "bundle_expander.h"
#include "graph_editor.h"
#include "bundle_plane_graph.h"
#include <algorithm>
#include "defs.h"
#include "sram_management/bundle_paths_validation.h"
#include "types.h"

std::set<pNode> mergeBundleNodesAndStrategyCandidates(const pMmeSlicingStrategy& strategy,
                                                      const pBundle& bundle)
{
    std::set<pNode> nodes(bundle->getNodes().begin(), bundle->getNodes().end());

    for(const pBundleExpansion& candidate : strategy->getMmeSlicingData().getRoleCandidates())
    {
        if(candidate)
        {
            nodes.insert(candidate->nodeToStitch);
        }
    }

    return nodes;
}

SlicingStrategyList BundleExpander::generateExpandedStrategies(const pBundle& bundle) const
{
    std::list<BundleExpansion::Role> firstDependancyRoles;
    for (BundleExpansion::Role role = BundleExpansion::FirstRole;
         role < BundleExpansion::NumOfRoles;
         role = BundleExpansion::Role((unsigned)role + 1))
    {
        if (BundleExpansion::isExpansionEnabledForRole(role) && !BundleExpansion::isDependentRole(role) &&
                isExpansionRoleEnabledForBundle(bundle, role))
        {
            firstDependancyRoles.emplace_back(role);
        }
    }
    return generateExpandedStrategies(bundle, firstDependancyRoles);
}

SlicingStrategyList BundleExpander::generateExpandedStrategies(const pBundle& bundle,
                                                               const std::list<BundleExpansion::Role>& roles) const
{
    std::list<pBundleExpansion> allCandidates = discoverExpansionCandidatesForBundle(bundle, roles);

    /* The initial expanded strategies list is the original strategies */
    SlicingStrategyList expandedStrategies = m_solvingDataPerBundle.at(bundle).strategies;

    /* For each valid subset of candidates a new strategy will be created. */
    for (const auto& nodeCandidate : allCandidates)
    {
        expandStrategiesWithCandidate(expandedStrategies, nodeCandidate, bundle);
    }

    return expandedStrategies;
}

std::list<pBundleExpansion> BundleExpander::discoverExpansionCandidatesForBundle(const pBundle& bundle,
                                                                                 const std::list<BundleExpansion::Role>& roles) const
{
    std::list<pBundleExpansion> allCandidates;

    for (BundleExpansion::Role role : roles)
    {
        ExpansionCandidatesSet roleCandidates = findBundleExpansionCandidatesForRole(bundle, role);

        for (const auto& nodeCandidate : roleCandidates)
        {
            allCandidates.push_back(nodeCandidate.second);
        }
    }

    // remove candidates that will be invalid in all strategies
    allCandidates.remove_if([&](const pBundleExpansion& cand) {
        return !validateCandidatePaths(m_graph,
                                       cand,
                                       NodeSet(bundle->getNodes().begin(), bundle->getNodes().end()),
                                       {});
    });

    for (const pBundleExpansion& candidate : allCandidates)
    {
        findDependantCandidates(candidate);
    }

    return allCandidates;
}

ExpansionCandidatesSet BundleExpander::findBundleExpansionCandidatesForRole(const pBundle& bundle,
                                                                            BundleExpansion::Role role) const
{
    ExpansionCandidatesSet candidates;

    for (const SlicingStrategyPtr& s : m_solvingDataPerBundle.at(bundle).strategies)
    {
        pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        pBundleExpansion candidate = nullptr;

        if (strategy->getMmeSlicingData().blockExpansionForRole(role))
        {
            continue;
        }

        switch (role)
        {
            case BundleExpansion::WideInputProducer:
                candidate = m_bundlizer.findWideTpcProducerExpansionCandidate(strategy, candidates);
                break;
            case BundleExpansion::NarrowInputProducer:
                candidate = m_bundlizer.findNarrowTpcProducerExpansionCandidate(strategy, candidates);
                break;
            case BundleExpansion::SlaveOutputConsumer:
            case BundleExpansion::OutputConsumer:
                candidate = m_bundlizer.findTpcConsumerExpansionCandidate(strategy, candidates, role);
                break;
            case BundleExpansion::SharedInputConsumer:
                candidate = m_bundlizer.findMmeInputSharingCandidate(strategy, m_solvingDataPerBundle, candidates);
                break;
            case BundleExpansion::SlaveInputProducer:
                candidate = m_bundlizer.findSlaveTpcProducerExpansionCandidate(strategy, candidates);
                break;
            default:
                HB_ASSERT(false, "Unexpected role to find expansion candidate for.");
                return ExpansionCandidatesSet();
        }

        if (candidate && candidate->nodeToStitch)
        {
            auto iterAndIsInserted = candidates.insert({candidate->nodeToStitch, candidate});
            if (iterAndIsInserted.second)
            {
                SLC_TRACE("{}: found candidate for the role of {}: {}.",
                          HLLOG_FUNC,
                          BundleExpansion::role2String(role),
                          candidate->nodeToStitch->getNodeName());
            }
        }

    }

    return candidates;
}

void BundleExpander::findWinningStrategyForSlaveBundle(pBundleExpansion slaveCandidate,
                                                       const pBundle&   slaveBundle) const
{
    HB_ASSERT(slaveCandidate->role == BundleExpansion::SharedInputConsumer, "Invalid candidate role");
    HB_ASSERT_PTR(slaveBundle);

    // Find initial strategies for slave bundle.
    // We can't use m_solvingDataPerBundle.at(slaveBundle).strategies since those strategies might have been already
    // expanded.
    SlicingStrategyList expandedSlaveStrategies = m_brains.m_mmeBrain.getSolutionStrategies(slaveBundle);

    // We can assume the slave MME has at least one valid solution, otherwise it will not be a valid candidate
    // (see findMmeInputSharingCandidate).
    HB_ASSERT(!expandedSlaveStrategies.empty(), "No strategies found for slave candidate");

    // Expand the initial strategies with the slave candidates.
    for (const auto& candidate : slaveCandidate->dependentCandidates)
    {
        expandStrategiesWithCandidate(expandedSlaveStrategies, candidate, slaveBundle);
    }

    // WA to force score calculation in the strategy cost model comparator.
    if (expandedSlaveStrategies.size() == 1)
    {
        expandedSlaveStrategies.emplace_back(expandedSlaveStrategies.front()->clone(false));
    }

    slaveCandidate->winningStrategyForSlaveBundle = std::static_pointer_cast<MmeSlicingStrategy>(
        findWinningStrategy(expandedSlaveStrategies, slaveBundle, m_graph, m_brains, false));
}

void BundleExpander::findDependantCandidates(pBundleExpansion candidate) const
{
    if (!candidate) return;

    if (candidate->role != BundleExpansion::SharedInputConsumer) return;

    // Detect the bundle of the MME slave
    pBundle slaveBundle = m_bundlizer.findBundleByNode(candidate->nodeToStitch);

    LOG_DEBUG(SRAM_SLICE, "{}: looking for dependant candidates for: {} .", HLLOG_FUNC, candidate->nodeToStitch->getNodeName());

    // Collect the relevant master roles for expansion
    std::list<BundleExpansion::Role> candidateExpansionRoles;
    for (BundleExpansion::Role role = BundleExpansion::FirstRole;
         role < BundleExpansion::NumOfRoles;
         role = BundleExpansion::Role((unsigned)role + 1))
    {
        // For the slave original bundle, we want to expand with roles that are not dependent nor the slave role itself
        if (!BundleExpansion::isDependentRole(role) && (role != BundleExpansion::Role::SharedInputConsumer) &&
            BundleExpansion::isExpansionEnabledForRole(BundleExpansion::masterToSlaveEquivalentRole(role)) &&
            isExpansionRoleEnabledForBundle(slaveBundle, role))
        {
            candidateExpansionRoles.emplace_back(role);
        }
    }

    // This is a recursion of sorts, since discoverExpansionCandidatesForBundle calls the current method.
    // But assuming candidateExpansionRoles do not contain SharedInputConsumer, the breakout if above would
    // stop the recursion from deepening further.
    candidate->dependentCandidates = discoverExpansionCandidatesForBundle(slaveBundle, candidateExpansionRoles);

    if (GCFG_SRAM_SLICER_COST_MODEL_ENABLED.value())
    {
        // In case we use the cost-model for strategy comparison, we need to find
        // the winning strategy for the slave bundle with possibly TPC producer and consumer.
        // This strategy will be used to evalute the invalid slave candidates cost.
        findWinningStrategyForSlaveBundle(candidate, slaveBundle);
    }

    // Swap the master roles with slave roles
    for (pBundleExpansion& dependentCandidate : candidate->dependentCandidates)
    {
        dependentCandidate->role = BundleExpansion::masterToSlaveEquivalentRole(dependentCandidate->role);
    }
}

bool BundleExpander::validateCandidateDependency(const pMmeSlicingStrategy& strategy, const pBundleExpansion& candidate)
{
    if (!BundleExpansion::isDependentRole(candidate->role))
    {
        return true;
    }

    /*Check pre-condition*/
    CHECK_RET_FALSE(candidate->role == BundleExpansion::SlaveOutputConsumer ||
                    candidate->role == BundleExpansion::SlaveInputProducer,
                    "{}: role {} validation is not supported!",
                    HLLOG_FUNC,
                    BundleExpansion::role2String(candidate->role));

    pBundleExpansion slave = strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer];
    bool candidateDependencyFulfilled = false;

    if (slave && candidate->role ==  BundleExpansion::SlaveOutputConsumer)
    {
        /* Check if this candidate consumes the output of the salve. */
        candidateDependencyFulfilled =
                candidate->stitchedOperand->originalTensor == slave->nodeToStitch->getOutputs().front();
    }
    else if (slave && candidate->role ==  BundleExpansion::SlaveInputProducer)
    {
        /* Check if this candidate produces slave node's non-shared input*/
        candidateDependencyFulfilled =
                candidate->stitchedOperand->originalTensor == slave->slaveOperands.getInput()->originalTensor;
    }
    return candidateDependencyFulfilled;
}

void BundleExpander::expandStrategiesWithCandidate(SlicingStrategyList&    strategies,
                                                   const pBundleExpansion& candidate,
                                                   const pBundle&          bundle) const
{
    std::list<pMmeSlicingStrategy> newStrategies;

    for (SlicingStrategyPtr& s : strategies)
    {
        pMmeSlicingStrategy strategy = std::static_pointer_cast<MmeSlicingStrategy>(s);
        if (!validateCandidateDependency(strategy, candidate))
        {
            /* Change the existing strategy to not accept the candidate - will be used for strategies comparison*/
            strategy->getMmeSlicingData().addInvalidCandidate(candidate);
            strategy->calculateMetrics();
            continue;
        }

        pBundleExpansion adjustedCandidate;
        if (candidate->role == BundleExpansion::SharedInputConsumer)
        {
            adjustedCandidate = m_brains.m_mmeSlaveBrain.adjustCandidateToStrategy(candidate, strategy);
        }
        else
        {
            adjustedCandidate = strategy->getMmeSlicingData().getAdjustedCandidate(candidate);
        }

        if (isCandidateValidForStrategy(adjustedCandidate, strategy, bundle))
        {
            /* Add a new strategy with the candidate accepted. */
            // reset alignment in order not to miss an opportunity to add a candidate to the bundle (when the new
            // strategy is added to the list, we will try again to align the tensors if possible given sram restrictions)
            pMmeSlicingStrategy strategyWithCandidate = std::static_pointer_cast<MmeSlicingStrategy>(strategy->clone(true /* reset alignment */));

            strategyWithCandidate->getMmeSlicingData().addValidCandidate(adjustedCandidate, true);
            if (strategyWithCandidate->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
            {
                strategyWithCandidate->tryAlignToCacheLine();
                newStrategies.push_back(strategyWithCandidate);
            }
        }
        /* Change the existing strategy to not accept the candidate - will be used for strategies comparison*/
        strategy->getMmeSlicingData().addInvalidCandidate(adjustedCandidate, false);
        strategy->calculateMetrics();
    }

    SLC_TRACE("Candidate role - {}, {} expanded strategies added", BundleExpansion::role2String(candidate->role), newStrategies.size());

    strategies.insert(strategies.end(), newStrategies.begin(), newStrategies.end());

    for (const pBundleExpansion& dependantCandidate: candidate->dependentCandidates)
    {
        BundleExpander::expandStrategiesWithCandidate(strategies, dependantCandidate, bundle);
    }
}

bool BundleExpander::isCandidateProducer(const pBundleExpansion& candidate)
{
    return BundleExpansion::isProducer(candidate->role);
}

bool BundleExpander::validateCandidatePaths(const HabanaGraph&      g,
                                            const pBundleExpansion& candidate,
                                            const NodeSet&          acceptedNodes,
                                            const NodeSet&          acceptedProducers)
{
    /**
    BundlePathsValidation implements the design from SW-23360 - see details in class
    **/
    bool             res            = true;
    const NodePtr&   candidateNode  = candidate->nodeToStitch;
    const TensorPtr& stitchedTensor = candidate->stitchedOperand->originalTensor;
    BundlePathsValidation pathsValidation(g);

    // this is a producer candidate
    if (isCandidateProducer(candidate))
    {
        res = pathsValidation.validateProducerPaths(candidateNode, stitchedTensor, acceptedNodes, acceptedProducers);
    }
    else  // consumer candidate
    {
        res = pathsValidation.validateConsumerPaths(candidateNode, stitchedTensor, acceptedNodes);
    }
    return res;
}

bool BundleExpander::validateCandidateOperands(const pBundleExpansion& candidate, const NodeSet& strategyNodes) const
{
    if (GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value())
    {
        return true;
    }

    for (const TensorPtr& candidateInput : candidate->nodeToStitch->getInputs())
    {
        if (!candidateInput) continue;
        // Aux tensors that are not scratch-pad can be shared
        if (candidateInput->isNonScratchpadAuxTensor()) continue;

        const NodePtr& inputProducer        = m_graph.getTensorProducer(candidateInput);
        bool           isBundleIntermediate = strategyNodes.find(inputProducer) != strategyNodes.end();
        for (const NodePtr& inputConsumer : m_graph.getTensorConsumers(candidateInput))
        {
            // Block only intermediate tensors, as double creation of the tensor in the bundle for
            // them isn't handled
            auto consumerInBundle = strategyNodes.find(inputConsumer);
            if (isBundleIntermediate && (consumerInBundle != strategyNodes.end()))
            {
                // Allow intermediate tensor sharing for 2 MME nodes in case the stitched operand is the shared operand.
                // This will prevent stitching of TPC producer to the slave MME in case it produces one of the inputs
                // of the master MME node, this case is not supported by solution-generator.
                bool isSharedMMEInput = (candidate->role == BundleExpansion::SharedInputConsumer) &&
                                        m_graph.runsOnMME(*consumerInBundle) &&
                                        (candidate->stitchedOperand->originalTensor == candidateInput);
                if (!isSharedMMEInput)
                {
                    SLC_DEBUG("Can't add candidate node {} to strategy - shares an input {} with {} which is already "
                              "in the bundle",
                              candidate->nodeToStitch->getNodeName(),
                              candidateInput->getName(),
                              inputConsumer->getNodeName());
                    return false;
                }
            }
        }
    }

    return true;
}

bool BundleExpander::isCandidateValidForStrategy(const pBundleExpansion& candidate,
                                                 const pMmeSlicingStrategy& strategy,
                                                 const pBundle& bundle) const
{
    /* Sanity */
    if (!candidate || !candidate->nodeToStitch) return false;

    auto& slicingData = strategy->getMmeSlicingData();

    if (slicingData.getRoleCandidates()[candidate->role])
    {
        /* The role is already occupied in this strategy */
        return false;
    }

    // Make sure the strategy can be expanded with this role
    if (strategy->getMmeSlicingData().blockExpansionForRole(candidate->role))
    {
        return false;
    }

    // remove WA as part of [SW-82260]
    if (!candidate->slaveOperands.empty() &&
        (SlicedOperandUtils::nofSlices(candidate->slaveOperands.getInput()) > 10000 ||
         SlicedOperandUtils::nofSlices(candidate->slaveOperands.getOutput()) > 10000))
    {
        SLC_WARN("{}, detected {}/{} number of slices for candidate",
                 HLLOG_FUNC,
                 SlicedOperandUtils::nofSlices(candidate->slaveOperands.getInput()),
                 SlicedOperandUtils::nofSlices(candidate->slaveOperands.getOutput()));
        return false;
    }

    // Make sure the candidate role matches the strategy to avoid same producer to both operands.
    if ((candidate->role == BundleExpansion::Role::WideInputProducer) &&
        (candidate->stitchedOperand->originalTensor != strategy->getMmeSlicingData().getWide()->originalTensor))
    {
        return false;
    }
    if ((candidate->role == BundleExpansion::Role::NarrowInputProducer) &&
        (candidate->stitchedOperand->originalTensor != strategy->getMmeSlicingData().getNarrow()->originalTensor))
    {
        return false;
    }
    if ((candidate->role == BundleExpansion::Role::SlaveInputProducer) &&
        (candidate->stitchedOperand->originalTensor == strategy->getMmeSlicingData().getWide()->originalTensor ||
         candidate->stitchedOperand->originalTensor == strategy->getMmeSlicingData().getNarrow()->originalTensor))
    {
        return false;
    }

    bool res = validateCandidatePaths(m_graph,
                                      candidate,
                                      strategy->getMmeSlicingData().getStrategyNodes(bundle),
                                      strategy->getMmeSlicingData().getStrategyProducers());

    res &= validateCandidateOperands(candidate, strategy->getMmeSlicingData().getStrategyNodes(bundle));

    if (HabanaGraph::runsOnTPC(candidate->nodeToStitch))
    {
        res &= m_bundlizer.isNodeEligibleForStitching(candidate->nodeToStitch,
                                                      candidate->stitchedOperand,
                                                      candidate->reshapeNode);
    }

    if (candidate->role == BundleExpansion::SharedInputConsumer)
    {
        res &= m_bundlizer.canMMEInputConsumerBeAddedToBundle(candidate, strategy);
    }

    res &= !isCandidateCreatesInvalidDependencies(candidate, strategy, bundle, true);

    SLC_DEBUG("candidate {}, valid for strategy = {}, (role = {})", candidate->nodeToStitch->getNodeName(), res,
              BundleExpansion::role2String(candidate->role));

    return res;
}

bool BundleExpander::isCandidateCreatesInvalidDependencies(const pBundleExpansion& candidate,
                                                           const pMmeSlicingStrategy& strategy,
                                                           const pBundle& bundle,
                                                           bool  useCache) const
{
    HashableNodeList nodeList(bundle->index());

    // adding the new nodes:
    if (candidate->nodeToStitch != nullptr)
    {
        nodeList.push_back(candidate->nodeToStitch);
    }
    if (candidate->reshapeNode != nullptr)
    {
        nodeList.push_back(candidate->reshapeNode);
    }

    // adding the existing nodes in the strategy:
    for (pBundleExpansion expansion : strategy->getMmeSlicingData().getRoleCandidates())
    {
        if (expansion == nullptr) continue;
        if (expansion->nodeToStitch != nullptr)
        {
            nodeList.push_back(expansion->nodeToStitch);
        }
        if (expansion->reshapeNode != nullptr)
        {
            nodeList.push_back(expansion->reshapeNode);
        }
    }

    // adding the bundle nodes:
    nodeList.insert(nodeList.end(), bundle->getNodes().begin(), bundle->getNodes().end());

    bool res = true;
    if (useCache)
    {
        if (!m_mapNodeListToDepCheck.empty() && m_mapNodeListToDepCheck.begin()->first.getBundleId() != bundle->index())
        {   // prevent the map from getting too big - having the cache "per bundle" is good enough
            m_mapNodeListToDepCheck.clear();
        }
        const auto& iterFound =  m_mapNodeListToDepCheck.find(nodeList);
        if (iterFound != m_mapNodeListToDepCheck.end())
        {
            res = iterFound->second;
        }
        else
        {
            res                               = GraphEditor::isInGroupDependencies<NodeList>(m_graph, nodeList);
            m_mapNodeListToDepCheck[nodeList] = res;
        }
    }
    else
    {
        res = GraphEditor::isInGroupDependencies<NodeList>(m_graph, nodeList);
    }
    return res;
}

bool BundleExpander::isBundleDependentOnNodes(const NodeList& prevNodes,
                                              const NodeList& postNodes,
                                              const NodeList& midBundleNodes) const
{
    return m_graph.isAncestor(prevNodes, midBundleNodes) && m_graph.isAncestor(midBundleNodes, postNodes);
}

bool BundleExpander::isExpansionRoleEnabledForBundle(const pBundle& bundle, BundleExpansion::Role role) const
{
    if (isBundleBatchGemm(bundle) && role != BundleExpansion::SharedInputConsumer &&
        !GCFG_ENABLE_TPC_STITCHING_TO_BGEMM.value())
    {
        SLC_WARN("Disable {} for BatchGemm", BundleExpansion::role2String(role));
        return false;
    }
    if (role == BundleExpansion::OutputConsumer &&
        bundle->type() == BundleType::MME)
    {
        // If the mme node has TPC bundle as producer, prevent consumer expansion.
        // Make sure there is only one path between the producer to the MME node,
        // to avoid blocking consumer stitching for BWD MME produced by a FWD TPC sequence.
        const NodePtr& mmeNode = bundle->getNodes().front();
        HB_ASSERT(HabanaGraph::runsOnMME(mmeNode), "Expected MME node in MME bundle to be first and only one before expansion");
        for (const TensorPtr& input : mmeNode->getInputs())
        {
            const NodePtr& producer = m_graph.getTensorProducer(input);
            const pBundle& producerBundle = m_bundlizer.findBundleByNode(producer);
            if ((producerBundle != nullptr) && (producerBundle->type() == BundleType::TPC) &&
                (m_graph.getBPGraph()->getNumberOfPaths(producer, mmeNode) == 1))
            {
                SLC_DEBUG("Disable TPC consumer stitching for bundle {}, due to TPC bundle producer {}",
                          bundle->getName(),
                          producerBundle->getName());
                return false;
            }
        }
    }
    return true;
}

/***********************************************************************************************************************/
/**********************************************************HashableNodeList*********************************************/
/***********************************************************************************************************************/
BundleExpander::HashableNodeList::HashableNodeList(unsigned bundleId): m_bundleId(bundleId)
{
}

bool BundleExpander::HashableNodeList::operator==(const HashableNodeList& other) const
{
    return (getBundleId() == other.getBundleId() &&
    size() == other.size() &&
    std::equal(begin(),end(), other.begin()));
}

unsigned BundleExpander::HashableNodeList::getBundleId() const
{
    return m_bundleId;
}

std::size_t BundleExpander::Hasher::operator()(const HashableNodeList& hashableNodeList) const
{
    std::stringstream ss;
    ss << hashableNodeList.getBundleId() << ':';
    for (const auto& n : hashableNodeList)
    {
        ss << n->getId() << '_';
    }
    return std::hash<std::string>()(ss.str());
}
