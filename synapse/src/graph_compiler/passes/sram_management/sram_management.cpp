#include "backward_operand_chunk_setter.h"
#include "defs.h"
#include "sram_management.h"

#include <unordered_set>
#include "bundlizer.h"
#include "bundle_slicer.h"
#include "habana_global_conf.h"
#include "solution_generator.h"
#include "slicing_utils.h"
#include "sram_management/bundle.h"
#include "synapse_common_types.h"
#include "tpc_bundle_solver.h"
#include "pattern_solvers.h"
#include "mme_shared_input.h"
#include "bundle_expander.h"
#include "flatten_mme.h"
#include "metrics_calculator.h"
#include "passes/contiguous_reshape_remover.h"
#include "bundle_plane_graph.h"
#include "batch_norm_eviction_fuser.h"
#include "handle_duplicate_mme_inputs.h"
#include "snapshot_pre_slicing_sizes.h"

bool SRAMSlicingManager::sliceGraph()
{
    if (GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.value() == 0UL)
    {
        return true;
    }

    // snapshot pre-slice tensor sizes for later validation
    snapshotPreSlicingSizes(m_graph);

    // sram managment pre-step. when calculating strategies we assume that each operand is related to unique tensor.
    // however, it is possile for a mme node to have multiple inputs with the same tensor.
    // to avoid issues realted to duplicate inputs we remove duplications with a pre-step
    DuplicateMmeInputsHandler::handleDuplicateMmeInputs(m_graph);

    m_normsHandler.findAndRemoveSliceNormNodes();
    generateInitialBundles();
    generateInitialStrategies();
    expandAllBundles();
    preSlicingOptimizations();
    sliceAllBundles();
    return m_normsHandler.handleRemovedSliceNormNodes();
}

void SRAMSlicingManager::generateInitialBundles()
{
    m_graph.constructBPGraph();
    m_bundlizer.generateBundles(m_mmeBundles,
                                m_tpcScalarPipeBundles,
                                m_tpcBundles,
                                m_rmwSectionBundles,
                                m_dmaTransposeBundles);
}

void SRAMSlicingManager::generateInitialStrategies()
{
    SLC_DEBUG("{}: Generate single MME node bundles strategies", HLLOG_FUNC);
    for (pBundle& bundle : m_mmeBundles)
    {
        m_solvingDataPerBundle[bundle].strategies = m_brains.m_mmeBrain.getSolutionStrategies(bundle);
        if (m_solvingDataPerBundle[bundle].strategies.empty())
        {
            SLC_WARN("{}: did not find strategies for bundle: {}, MME node: {}",
                     HLLOG_FUNC,
                     bundle->getName(),
                     bundle->getNodes().front()->getNodeName());
        }
    }
    for (pBundle& bundle : m_tpcScalarPipeBundles)
    {
        m_solvingDataPerBundle[bundle].strategies = m_brains.m_tpcBrain.getSolutionStrategies(bundle);
    }
    for (pBundle& bundle : m_dmaTransposeBundles)
    {
        m_solvingDataPerBundle[bundle].strategies = m_brains.m_dmaTransposeSlicingBrain.getSolutionStrategies(bundle);
    }
    for (pBundle& bundle : m_tpcBundles)
    {
        m_solvingDataPerBundle[bundle].strategies = m_brains.m_tpcBrain.getSolutionStrategies(bundle);
    }

    for (pBundle& bundle : m_rmwSectionBundles)
    {
        m_solvingDataPerBundle[bundle].strategies = {}; // complex-guid bundles are not sliced
    }

    logGraphSizeOptimizationStats(true);
}

void SRAMSlicingManager::expandAllBundles()
{
    if (GCFG_SRAM_SLICER_BUNDLE_EXPANSION_ENABLED.value())
    {
        SLC_DEBUG("{}: Try to expand all bundles", HLLOG_FUNC);
        for (pBundle& bundle : getBundleExpansionOrder())
        {
            if (bundle->getNodes().empty()) continue;
            expandBundle(bundle);
        }
        logGraphSizeOptimizationStats(false);
    }
    m_graph.discardBPGraph();
}

void SRAMSlicingManager::preSlicingOptimizations()
{
    setTpcBundleSlicingChunk();
    fuseEvictions();
}

bool SRAMSlicingManager::isTpcBundleSlicingValid(const pBundle& tpcBundle, const StrategySlicingData& slicingData) const
{
    if (GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value())
    {
        return true;
    }

    for (const auto& node : tpcBundle->getNodes())
    {
        for (const auto& operand : node->getOperands())
        {
            if (!operand) continue;
            const auto& slicedOperand = slicingData.getSlicedOperand(operand);
            HB_ASSERT(slicedOperand,
                      "Node {} : Missing sliced operand for tensor {} in TPC bundle {}",
                      node->getNodeName(),
                      operand->getName(),
                      tpcBundle->index());
            if (!Bundlizer::isOperandSlicingValid(node, slicedOperand))
            {
                SLC_DEBUG("Node {}: slicing of tensor {} is not according to index-space",
                          node->getNodeName(),
                          operand->getName());
                return false;
            }
        }
    }

    return true;
}

void SRAMSlicingManager::setTpcBundleSlicingChunk()
{
    for (const auto& bundle : m_tpcBundles)
    {
        auto& bundleSlicingData = m_solvingDataPerBundle[bundle].strategies.front()->getSlicingData();
        pSlicedOperand masterOperand = bundleSlicingData.masterOperand;
        pSlicedOperand slaveOperand;
        if (!bundleSlicingData.getSlavesPatterns().empty())
        {
            slaveOperand = bundleSlicingData.getSlavesPatterns().front().getOperand();
        }

        // Find the connecting node between the TPC bundle and its MME consumer
        const auto& connectingNode = m_graph.getTensorProducer(masterOperand->originalTensor);
        HB_ASSERT_PTR(connectingNode);
        const auto& masterConsumers = m_graph.getNodeConsumers(connectingNode);
        pBundle mmeBundle;
        TensorPtr   connectingTensor;
        for (const auto& node : masterConsumers)
        {
            if (! HabanaGraph::runsOnMME(node)) continue;
            if (slaveOperand)
            {
                const TensorVector& nodeInputs = node->getInputs();
                // Expect to both master and slave operand to connect to the mme node
                if (std::find(nodeInputs.begin(), nodeInputs.end(), slaveOperand->originalTensor) == nodeInputs.end()) continue;
            }
            mmeBundle = m_bundlizer.findBundleByNode(node);
            if (mmeBundle)
            {
                connectingTensor = m_bundlizer.getConnectingTensor(connectingNode, node);
                HB_ASSERT_PTR(connectingTensor);
                break;
            }
        }
        if (!mmeBundle)
        {
            SLC_WARN("Can't find divider for tpc bundle {}, remain one piece", bundle->getName());
            continue;
        }

        if (m_solvingDataPerBundle[mmeBundle].strategies.empty())
        {
            SLC_WARN("Can't slice TPC bundle {}, as consumer MME bundle, {}, has no strategies", bundle->getName(), mmeBundle->getName());
            continue;
        }
        const auto& winningStrategy =
            findWinningStrategy(m_solvingDataPerBundle[mmeBundle].strategies, mmeBundle, m_graph, m_brains);
        const auto& slicingData = winningStrategy->getSlicingData();

        // [SW-76800] TPC bundle assumption is that master operand is sliced on outdim. In case master operand (
        // consumer MME input) isn't sliced on outer dim avoid slicing.
        if (!SlicedOperandUtils::isSlicedOnDimension(slicingData.getSlicedOperand(connectingTensor),
                                                     connectingTensor->getDim() - 1))
        {
            SLC_WARN("Can't slice TPC bundle {}, as consumer MME bundle {} is not sliced on batch (outer dim)",
                     bundle->getName(),
                     mmeBundle->getName());
            continue;
        }

        // Slice main TPC chain
        auto                       opDimAndChunk = TpcBundleSolver::getDimAndChunk(connectingTensor, slicingData);
        BackwardOperandChunkSetter operandSetter(m_graph, bundleSlicingData);
        if (opDimAndChunk.is_set())
        {
            bundleSlicingData.traversalPattern = {opDimAndChunk->first};
            operandSetter.setDimensionChunk(masterOperand, opDimAndChunk->first, opDimAndChunk->second);
        }

        // Slice parallel TPC nodes
        for (auto& slavePattern : bundleSlicingData.getSlavesPatterns())
        {
            opDimAndChunk = TpcBundleSolver::getDimAndChunk(slavePattern.getOperand()->originalTensor, slicingData);
            if (opDimAndChunk.is_set())
            {
                slavePattern.setDimOrder({opDimAndChunk->first});
                operandSetter.setDimensionChunk(slavePattern.getOperand(), opDimAndChunk->first, opDimAndChunk->second);
            }
        }

        if (!isTpcBundleSlicingValid(bundle, bundleSlicingData))
        {
            SLC_WARN("Can't slice TPC bundle {} - invalid slicing", bundle->getName());
            m_solvingDataPerBundle[bundle].strategies = {};  // Reset slicing for this bundle
        }
    }
}

void SRAMSlicingManager::fuseEvictions()
{
    // Fuse eviction of an intermediate tensor in the bundle which is generated slice by slice in SRAM and is also
    // needed in HBM (because it is persistent or it is read by an external consumer). This method tries to have the
    // bundled producer generate the same output in 2 places: one as intermediate tensor for the bundled consumer(s) and
    // one for the rest of the graph or the user to read.

    if (GCFG_ENABLE_BUNDLE_EVICTION_FUSING.value())
    {
        for (auto& bundle : m_mmeBundles)
        {
            if (m_solvingDataPerBundle[bundle].strategies.size() != 1)
            {
                SLC_DEBUG("Not fusing eviction of bundle {}. Require a single strategy to evict, but found {}",
                          bundle->index(),
                          m_solvingDataPerBundle[bundle].strategies.size());
                continue;
            }

            fuseMMEEvictions(bundle);
            fuseTPCEvictions(bundle);
        }
    }
}

void SRAMSlicingManager::fuseMMEEvictions(pBundle& bundle)
{
    if (m_graph.getHALReader()->isDuplicateMmeOutputSupported())
    {
        // TODO [SW-52941]: In Gaudi2, when an MME output is generated in SRAM and is required outside the bundle (or is
        // persistent), add another output to the MME that generates the slices directly in HBM (subject to the MME
        // limitations. See ticket).
    }
}

void SRAMSlicingManager::fuseTPCEvictions(pBundle& bundle)
{
    // TODO - In the future, we may want to use the TPC fuser to fuse memcpy where an eviction from SRAM to HBM is
    // needed. Currently only batch norm is supported, via its optional output, not TPC fuser.
    fuseBNEvictions(bundle);
}

void SRAMSlicingManager::fuseBNEvictions(pBundle& bundle)
{
    auto& strategy = m_solvingDataPerBundle[bundle].strategies.front();

    BatchNormStagesEvictionFuser bnsef(m_graph, bundle, strategy);
    bnsef.fuseEvictions(true, true, false /*BN1_BWD eviction fusion not implemented yet in new pass*/);
}

// Create an order in which to greedily expand bundles.
// Using only the MME bundles and assuming they were not expanded yet (i.e each bundle has a single MME node).
std::vector<pBundle> SRAMSlicingManager::getBundleExpansionOrder() const
{
    // Initially insert bundles in reverse order to give higher prioirity to producers
    std::vector<pBundle> bundles(m_mmeBundles.rbegin(), m_mmeBundles.rend());

    // Preferring bundles with dedx, because heuristically in RN50, it improves the BWD pipelining.
    std::stable_sort(bundles.begin(), bundles.end(), [](const pBundle& lhs, const pBundle& rhs) {
        return lhs->getNodes().front()->getNodeType() == Node::TYPE_DEDX &&
               rhs->getNodes().front()->getNodeType() != Node::TYPE_DEDX;
    });

    return bundles;
}

void SRAMSlicingManager::expandBundle(pBundle& bundle)
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Bundle#{}_expansion", bundle->index()));

    BundleExpander      expander(m_graph, m_brains, m_bundlizer, m_solvingDataPerBundle);
    SlicingStrategyList expandedStrategies = expander.generateExpandedStrategies(bundle);

    if (!expandedStrategies.empty())
    {
        SlicingStrategyPtr winningStrategy = findWinningStrategy(expandedStrategies, bundle, m_graph, m_brains);
        applyWinningStrategyToBundle(bundle, std::static_pointer_cast<MmeSlicingStrategy>(winningStrategy));
    }
}

void SRAMSlicingManager::logGraphSizeOptimizationStats(bool beforeBundleExpansion)
{
    if (LOG_LEVEL_AT_LEAST_INFO(SRAM_SLICE))
    {
        unsigned totalStrategies = 0;
        unsigned graphSizeOptimized = 0;
        for (const pBundle& bundle : m_mmeBundles)
        {
            totalStrategies += m_solvingDataPerBundle[bundle].strategies.size();
            graphSizeOptimized += std::count_if(m_solvingDataPerBundle[bundle].strategies.begin(),
                                                m_solvingDataPerBundle[bundle].strategies.end(),
                                                [&](const SlicingStrategyPtr& strategy)
                                                {
                                                    return (strategy->getGraphSizeOptimized());
                                                });
        }

        LOG_INFO(SRAM_SLICE,
                 "({}) total bundles = {}, total strategies = {}, original strategies = {}, graph size optimized = {} ",
                 beforeBundleExpansion ? "initial strategies" : "after bundle expansion",
                 m_mmeBundles.size() + m_tpcScalarPipeBundles.size(),
                 totalStrategies,
                 totalStrategies - graphSizeOptimized,
                 graphSizeOptimized);
    }
}

void SRAMSlicingManager::flattenBundleNodes(pBundle& bundle, pMmeSlicingStrategy winningStrategy)
{
    bool shouldFlatten = SlicedOperandUtils::shouldAnyOperandBeFlattened(winningStrategy->getMmeSlicingData().getSlicedOperands());
    if (shouldFlatten)
    {
        for (const NodePtr& node : bundle->getNodes())
        {
            flattenMmeNode(node, winningStrategy->getMmeSlicingData().getSlicedOperands());
        }
        //When flattenning a bundle we must flatten both master and slave nodes
        pBundleExpansion sharedInputConsumer =
                winningStrategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer];

        if (sharedInputConsumer != nullptr)
        {
            flattenMmeNode(sharedInputConsumer->nodeToStitch, sharedInputConsumer->slaveOperands.getSlaveOperands());
        }
    }
}

void SRAMSlicingManager::applyWinningStrategyToBundle(pBundle& bundle, pMmeSlicingStrategy winningStrategy)
{
    // Forget the rest
    m_solvingDataPerBundle.at(bundle).strategies = {winningStrategy};
    flattenBundleNodes(bundle, winningStrategy);
    // stitch each candidate
    for (pBundleExpansion& candidate : winningStrategy->getMmeSlicingData().getRoleCandidates())
    {
        if (candidate && candidate->nodeToStitch)
        {
            addCandidateToBundle(bundle, candidate, winningStrategy);
        }
    }
    HB_ASSERT(m_graph.getBPGraph()->getBundlePlaneGraph()->isAcyclicGraph(),
              "circularity was found when expanding bundle {}!",
              bundle->index());
    m_graph.invalidateExecutionSchedule();
}

void SRAMSlicingManager::addCandidateToBundle(pBundle& bundle,
                                              pBundleExpansion& candidate,
                                              pMmeSlicingStrategy& strategy)
{
    SLC_DEBUG("Adding {} candidate node {}", BundleExpansion::role2String(candidate->role),
                                             candidate->nodeToStitch->getNodeName());
    switch (candidate->role)
    {
    case BundleExpansion::WideInputProducer:
    case BundleExpansion::NarrowInputProducer:
    case BundleExpansion::SlaveInputProducer:
        addTpcProducer(bundle, candidate, strategy);
        break;
    case BundleExpansion::SlaveOutputConsumer:
    case BundleExpansion::OutputConsumer:
        addTpcConsumer(bundle, candidate, strategy);
        break;
    case BundleExpansion::SharedInputConsumer:
        addSharedInput(bundle, candidate, strategy);
        break;
    default:
        HB_ASSERT(false, "Unexpected role to add candidate for");
        break;
    }
    /* make sure this candidate (after being added to the bundle) will not be calculated twice in the metrics. */
    strategy->getMmeSlicingData().setCandidateAsBundled(candidate);
}

void SRAMSlicingManager::addSharedInput(pBundle& bundle,
                                        pBundleExpansion& candidate,
                                        pMmeSlicingStrategy& strategy)
{
    // recheck that the candidate is valid - in case some other expansion was added from the last check
    if (strategy->calculateMetrics().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes)
    {
        // check if master node is already stitched to the slave
        pBundle slaveBundle = m_bundlizer.findBundleByNode(candidate->nodeToStitch);
        const pNode& mmeNodeToAdd = candidate->nodeToStitch;
        LOG_DEBUG(SRAM_SLICE,
                  "MME node {} is added to bundle {}, index {}, has shared operand with master MME node",
                  mmeNodeToAdd->getNodeName(),
                  bundle->getName(),
                  bundle->index());

        // Remove the mme node bundle as it joins a different bundle
        pBundle removedBundle = m_bundlizer.removeBundle(mmeNodeToAdd);
        m_mmeBundles.remove(removedBundle);
        m_brains.m_mmeSlaveBrain.addSharedOperandMme(candidate, strategy);
        m_bundlizer.addCandidateToBundle(bundle, candidate);
    }
}

void SRAMSlicingManager::addTpcProducer(pBundle& bundle,
                                        pBundleExpansion& candidate,
                                        pMmeSlicingStrategy& strategy)
{
    HB_ASSERT(candidate->role == BundleExpansion::Role::SlaveInputProducer ||
              candidate->role == BundleExpansion::Role::NarrowInputProducer ||
              candidate->role == BundleExpansion::Role::WideInputProducer,
              "role mismatch");

    pNode consumer;
    bool canAddCandidate = true;
    if (candidate->role == BundleExpansion::Role::SlaveInputProducer)
    {
        pBundleExpansion slave = strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer];
        consumer = slave->nodeToStitch;
        HB_ASSERT(consumer->getNodeAnnotation().bundleInfo.is_set() &&
                      consumer->getNodeAnnotation().bundleInfo->bundleIndex == bundle->index(),
                  "{}: Trying to add slave input producer candidate {} but it's consumer is not in the bundle",
                  HLLOG_FUNC,
                  candidate->nodeToStitch->getNodeName());
    }
    else
    {
        consumer = bundle->getNodes().front();
    }
    pBundleExpansion candidateToCheck = std::make_shared<BundleExpansion>(*candidate);
    candidateToCheck->bundleNode = consumer;

    pNode producer = m_graph.getTensorProducer(candidate->stitchedOperand->originalTensor);
    pNode producerReshape = nullptr;
    if (producer && producer->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
    {
        producerReshape = producer;
    }
    if (candidate->reshapeNode)
    {
        //Reshape node is the original user reshape we found when adding candidate
        if (producerReshape != nullptr)
        {
            if (candidate->reshapeNode != producerReshape)
            {
                //If we have a different producer for the stitched operand and it is reshape
                //Then it is the reshape we created when flattenning.
                //Fuse it with the original reshape.
               ContiguousReshapeRemover remover(m_graph);
               if (remover.fuseProducerAndConsumerReshape(candidate->reshapeNode, producerReshape) != true)
               {
                   LOG_ERR(SRAM_SLICE, "Failed fusing reshapes for node {}",
                           candidate->reshapeNode->getNodeName());
                   producerReshape = nullptr;
                   canAddCandidate = false;
               }
            }
        }
    }
    candidate->reshapeNode = producerReshape;
    // No need to align reshape when slicing according to index-space
    if (GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value() && candidate->reshapeNode)
    {
        if (!SlicedOperandUtils::isTriviallySliced(candidate->stitchedOperand))
        { // we only want to move reshapes out of the bundle when the slicing is not trivial
            if (m_reshapeAligner.alignProducerReshape(candidate))
            {
                candidate->reshapeNode = nullptr;
            }
            else
            {
                canAddCandidate = false;
            }
        }
    }
    HB_ASSERT(canAddCandidate, "failed to align producer reshape");
    if (canAddCandidate)
    {
        if (m_bundlizer.addCandidateToBundle(bundle, candidate))
        {
            m_brains.m_reshapeBrain->addProducerToStrategy(candidate, strategy);
            m_brains.m_tpcSlaveBrain->addProducerToStrategy(candidate, strategy);
        }
    }
}

void SRAMSlicingManager::addTpcConsumer(pBundle& bundle, pBundleExpansion& candidate, pMmeSlicingStrategy& strategy)
{
    HB_ASSERT(candidate->role == BundleExpansion::Role::SlaveOutputConsumer ||
              candidate->role == BundleExpansion::Role::OutputConsumer,
              "role mismatch");

    pNode producer;

    if (candidate->role == BundleExpansion::Role::SlaveOutputConsumer)
    {
        producer = strategy->getMmeSlicingData().getRoleCandidates()[BundleExpansion::SharedInputConsumer]->nodeToStitch;
        HB_ASSERT(producer->getNodeAnnotation().bundleInfo.is_set() &&
                      producer->getNodeAnnotation().bundleInfo->bundleIndex == bundle->index(),
                  "{}: Trying to add slave output consumer candidate {} but it's producer is not in the bundle",
                  HLLOG_FUNC,
                  candidate->nodeToStitch->getNodeName(),
                  bundle->index(),
                  bundle->getName());
    }
    else
    {
        producer = bundle->getNodes().front();
    }

    const pSlicedOperand& mmeOut = candidate->stitchedOperand;

    bool canAddCandidate = true;
    pNode consumerReshape = nullptr;
    for (const pNode& mmeOutConsumer : m_graph.getTensorConsumers(mmeOut->originalTensor))
    {
        if (mmeOutConsumer->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            for (const pNode& reshapeOutputConsumer : m_graph.getRealConsumers(mmeOutConsumer->getOutput(0)))
            {
                if (reshapeOutputConsumer == candidate->nodeToStitch)
                {
                    consumerReshape = mmeOutConsumer;
                }
            }
        }
    }
    if (candidate->reshapeNode)
    {
        //Reshape node is the original user reshape we found when adding candidate
        if (consumerReshape != nullptr)
        {
            if (candidate->reshapeNode != consumerReshape)
            {
                //If we have a different consumer for the stitched operand and it is reshape
                //Then it is the reshape we created when flattenning.
                //Fuse it with the original reshape.
                ContiguousReshapeRemover remover(m_graph);
                if (remover.fuseProducerAndConsumerReshape(consumerReshape, candidate->reshapeNode) != true)
                {
                    LOG_ERR(SRAM_SLICE, "Failed fusing reshapes for node {}",
                            candidate->reshapeNode->getNodeName());
                    canAddCandidate = false;
                }
            }
        }
    }
    //If we only added reshape as a result of flattenning
    if (candidate->reshapeNode == nullptr && consumerReshape != nullptr)
    {
        candidate->reshapeNode = consumerReshape;
    }
    // No need to align reshape when slicing according to index-space
    if (GCFG_IGNORE_INDEX_SPACE_FOR_SLICING.value() && candidate->reshapeNode)
    {
        if (!SlicedOperandUtils::isTriviallySliced(candidate->stitchedOperand))
        {
            // we only want to move reshapes out of the bundle when the slicing is not trivial
            if (m_reshapeAligner.alignConsumerReshape(candidate))
            {
                candidate->reshapeNode = nullptr;
            }
            else
            {
                canAddCandidate = false;
            }
        }
    }

    HB_ASSERT(canAddCandidate, "failed to align consumer reshape");

    if (canAddCandidate)
    {
        if (m_bundlizer.addCandidateToBundle(bundle, candidate))
        {
            m_brains.m_reshapeBrain->addConsumerToStrategy(candidate, strategy);
            m_brains.m_tpcSlaveBrain->addConsumerToStrategy(candidate, strategy);
        }
    }
}

void SRAMSlicingManager::sliceAllBundles()
{
    m_bundlizer.logGraphBundlingStatus(m_solvingDataPerBundle);

    SLC_DEBUG("{}: Slice all MME bundles", HLLOG_FUNC);
    sliceBundles(m_mmeBundles);
    SLC_DEBUG("{}: Slice all TPC scalar pipe bundles", HLLOG_FUNC);
    sliceBundles(m_tpcScalarPipeBundles);
    SLC_DEBUG("{}: Slice all TPC bundles", HLLOG_FUNC);
    sliceBundles(m_tpcBundles);
    SLC_DEBUG("{}: Slice all DMA Transpose bundles", HLLOG_FUNC);
    sliceBundles(m_dmaTransposeBundles);
}

void SRAMSlicingManager::sliceBundles(BundleList& bundles)
{
    while (!bundles.empty())
    {
        pBundle& bundle = bundles.front();
        if (!sliceBundle(bundle))
        {
            // No slicing strategy that fits in SRAM - work from HBM.
            LOG_WARN(SRAM_SLICE, "No solution found for bundle {}", bundle->getName());
        }
        bundles.remove(bundle);
    }
}

bool SRAMSlicingManager::sliceBundle(pBundle& bundle)
{
    if (m_solvingDataPerBundle.at(bundle).strategies.empty())
    {
        return false;
    }

    const SlicingStrategyPtr& strategy =
        findWinningStrategy(m_solvingDataPerBundle.at(bundle).strategies, bundle, m_graph, m_brains);
    SolutionGenerator generator(m_graph, bundle, strategy);
    if (generator.fillSolution())
    {
        BundleSlicer::sliceBundle(*bundle, m_graph);
    }
    else
    {
        // No slicing strategy that fits in SRAM - work from HBM.
        LOG_WARN(SRAM_SLICE, "No solution found for bundle {}", bundle->getName());
    }

    return true;
}

void SRAMSlicingManager::flattenMmeNode(pNode node, std::vector<pSlicedOperand> slicedOperands)
{
    if(!node || slicedOperands.empty())
    {
        return;
    }
    if (MMENodeFlattener::canFlattenMMENode(node))
    {
        MMENodeFlattener flattener(m_graph);
        flattener.doFlatten(node, slicedOperands);
    }
}

// SRAM slicing pass "main"
bool sliceGraphToSRAMCapacity(HabanaGraph& g)
{
    SRAMSlicingManager sramSlicingManager(g);
    return sramSlicingManager.sliceGraph();
}
