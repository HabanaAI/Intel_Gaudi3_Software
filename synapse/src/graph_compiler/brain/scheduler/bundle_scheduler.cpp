#include "bundle_scheduler.h"
#include "defs.h"
#include "habana_graph.h"
#include "bundle_scheduler_routes_creator.h"
#include "bundle_scheduler_bvd_traversal_sorter.h"
#include "bundle_scheduler_slices_sequencer.h"
#include "bundle_scheduler_threads_creator.h"
#include "bundle_scheduler_threads_scheduler.h"
#include "brain_data.h"
#include "layered_brain.h"
#include "node.h"

using namespace gc::layered_brain;

enum class RoutesCreationAlog
{
    DFS = 0,
    BFS,
};

BundleScheduler::BundleScheduler(HabanaGraph& graph, const BundleNodes& bundleNodes)
: m_graph(graph), m_bundle(bundleNodes)
{
    HB_ASSERT(!m_bundle.empty(), "No bundle nodes");
    auto bundleIdx = getBundleIndex(m_bundle.front());
    HB_ASSERT(bundleIdx, "bundle nodes expected to have bundle index");
    m_bundleIdx = *bundleIdx;
}

bool BundleScheduler::setBundleNodesSchedule(bool dryRun)
{
    SET_TEMP_LOG_CONTEXT(fmt::format("Scheduler Bundle#{}", m_bundleIdx));
    validateCtrlDepForMemsets();

    const auto* lbData = m_graph.getLayeredBrainData();
    HB_ASSERT_PTR(lbData);
    HB_ASSERT(lbData->isLayeredBrainBundle(m_bundleIdx),
              "valid bundle for layered brain expected, bundle index {}",
              m_bundleIdx);
    const BundleData& bundleData       = lbData->m_bundleData.at(m_bundleIdx);
    const auto        routeEndNodes(getRouteEndNodes(bundleData));
    auto              routesPerSlice   = getRoutesCreator(m_graph, m_bundle)->getRoutesPerSlice(routeEndNodes);
    auto traversalPattern = getBvdsSorter()->getBundleViewsByTraversalOrder(bundleData);

    const auto sliceToReductionInfo = createSliceToReductionInfoMap(bundleData, routesPerSlice);
    auto       sliceSetsSequence =
        getSlicesSequencer()->getSliceSetsSequence(traversalPattern, bundleData, sliceToReductionInfo, routeEndNodes);
    auto threadsSequence = getThreadsCreator()->getThreadsSequeceBySliceSets(sliceSetsSequence, routesPerSlice);

    getThreadsScheduler(m_graph, m_bundle)
        ->scheduleNodes(threadsSequence, routeEndNodes, bundleData.getPipelineDepth());
    return validateAllNodesScheduled();
}

// The following can be replaced by factory/abstract-factory call

std::unique_ptr<RoutesCreator> BundleScheduler::getRoutesCreator(HabanaGraph& graph, const BundleNodes& bundle)
{
    const auto algo = static_cast<RoutesCreationAlog>(GCFG_LAYERED_BRAIN_SCHEDULE_ROUTES_CREATION_ALGO.value());
    switch (algo)
    {
        case RoutesCreationAlog::DFS:
            return std::make_unique<DfsRoutesCreator>(graph, bundle);
        case RoutesCreationAlog::BFS:
            return std::make_unique<BfsRoutesCreator>(graph, bundle);
        default:
            HB_ASSERT(false, "invalid LAYERED_BRAIN_SCHEDULE_ROUTES_CREATION_ALGO value {}", algo);
            return nullptr;
    }
}

std::unique_ptr<BVDsTraversalSorter> BundleScheduler::getBvdsSorter()
{
    return std::make_unique<SlicedBVDsTraversalSorter>();
}

std::unique_ptr<OutputSlicesSequencer> BundleScheduler::getSlicesSequencer()
{
    return std::make_unique<SimpleSlicesSequencer>();
}

std::unique_ptr<ThreadsCreator> BundleScheduler::getThreadsCreator()
{
    return std::make_unique<SimpleThreadsCreator>();
}

std::unique_ptr<ThreadsScheduler> BundleScheduler::getThreadsScheduler(HabanaGraph& graph, const BundleNodes& bundle)
{
    return std::make_unique<MultiThreadScheduler>(graph, bundle);
}

bool BundleScheduler::validateAllNodesScheduled() const
{
    for (const auto& node : m_bundle)
    {
        auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
        HB_ASSERT(bundleInfo.is_set(), "Bundle node without bundle info");
        HB_ASSERT(bundleInfo->operationIndex != 0, "Unexpectedly found unscheduled node {}", node->getNodeName());
        if (bundleInfo->operationIndex == 0) return false;
    }
    return true;
}

std::vector<NodePtr> BundleScheduler::getRouteEndNodes(const BundleData& bundleData) const
{
    std::vector<NodePtr> routeEndNodes;
    std::copy_if(m_bundle.begin(), m_bundle.end(), std::back_inserter(routeEndNodes), [&bundleData](const NodePtr& n) {
        return BundleScheduler::isRouteEnd(n, bundleData);
    });
    std::sort(routeEndNodes.begin(), routeEndNodes.end(), NodeComparator());
    return routeEndNodes;
}

/**
 * @brief Find reduced BVDs by comparing the bvd coord of each of the reduction inputs.
 *        A reduced BVD is expected to change between reduction inputs.
 */
std::unordered_set<BundleViewId> BundleScheduler::getReducedBvdIds(const NodePtr&    node,
                                                                   const BundleData& bundleData) const
{
    std::unordered_set<BundleViewId> reducedBvds;
    HB_ASSERT(bundleData.isSlicerReduction(node),
              "Expected {} to be of reduction node type, actual: {}",
              node->getNodeName(),
              node->getNodeTypeStr());
    for (BundleViewId bvdId = 0; bvdId < bundleData.getBundleViews()->getNumOfBundleViews(); ++bvdId)
    {
        std::optional<BVDCoord> refCoord;
        for (auto inputIdx = 0; inputIdx < node->getInputs().size(); ++inputIdx)
        {
            const auto& slice = node->getInput(inputIdx);
            const auto  it    = bundleData.getRouteEndInputsCoords().find(node);
            HB_ASSERT(it != bundleData.getRouteEndInputsCoords().end(),
                      "Expecting inputs coords for reduction {}",
                      node->getNodeName());
            const auto& sliceCoords = it->second.at(inputIdx);
            if (sliceCoords.empty()) continue;  // skip input without bvd coords
            if (!refCoord.has_value())
            {
                refCoord = sliceCoords;
            }
            else
            {
                if (refCoord.value().at(bvdId) != sliceCoords.at(bvdId))
                {
                    reducedBvds.insert(bvdId);
                    break;
                }
            }
        }
    }
    return reducedBvds;
}

SliceToReductionInfo BundleScheduler::createSliceToReductionInfoMap(const BundleData&  bundleData,
                                                                    const RoutesTable& routesTable) const
{
    // Gather reduced BVD IDs in route for each slice
    SliceToReductionInfo sliceToReductionInfo;
    for (const auto& [slice, routes] : routesTable)
    {
        for (const auto& route : routes)
        {
            for (const auto& node : route)
            {
                if (!bundleData.isSlicerReduction(node)) continue;
                const auto& reducedBVDs = getReducedBvdIds(node, bundleData);
                HB_ASSERT(!reducedBVDs.empty(),
                          "Expecting slice {} route node {}[{}] to have reduced BVDs",
                          slice->getName(),
                          node->getNodeName(),
                          node->getNodeTypeStr());
                sliceToReductionInfo[slice].reducedBvdIds.insert(reducedBVDs.begin(), reducedBVDs.end());
            }
        }

        if (auto it = sliceToReductionInfo.find(slice); it != sliceToReductionInfo.end())
        {
            // save route end node corresponding to slice
            const auto& consumers = m_graph.getTensorConsumers(slice);
            std::vector<NodePtr> routeEnds;
            std::copy_if(consumers.begin(),
                         consumers.end(),
                         std::back_inserter(routeEnds),
                         [&bundleData](const NodePtr& n) { return BundleScheduler::isRouteEnd(n, bundleData); });
            HB_ASSERT(routeEnds.size() == 1,
                      "Expecting exactly 1 routeEnd consumer for slice {}, found : {}",
                      slice->getName(),
                      routeEnds.size());
            it->second.sliceConsumer = routeEnds.front();
        }
    }
    return sliceToReductionInfo;
}

void BundleScheduler::validateCtrlDepForMemsets()
{
    if (!GCFG_ENABLE_LAYERED_BRAIN_SCHEDULER_OPTIMIZE_MEMSET.value()) return;
    for (const auto& n : m_bundle)
    {
        // Skip nodes that aren't reduction
        if (n->getNodeType() != Node::TYPE_INTERNAL_REDUCTION) continue;
        const auto& producers = m_graph.getNodeRealProducers(n, Node::TENSOR_TYPE_DATA);
        NodeSet     blockingNodes;
        NodeSet     blockedNodes;
        for (const auto& producer : producers)
        {
            if (producer->isMemset())
            {
                blockingNodes.insert(producer);
            }
            else
            {
                blockedNodes.insert(producer);
            }
        }
        HB_ASSERT(blockingNodes.size() <= 2,
                  "expecting a single memset and possibly additional CL aware memget for {}",
                  n->getNodeName());
        // Schedule the reduction producers to be executed after the memset(s)
        for (const auto& blocking : blockingNodes)
        {
            for (const auto& blocked : blockedNodes)
            {
                HB_ASSERT(m_graph.isControlDependencyBetweenNodes(blocking, blocked),
                          "Expected a control dependency between {} and {}",
                          blocking->getNodeName(),
                          blocked->getNodeName());
            }
        }
    }
}

bool BundleScheduler::isRouteEnd(const NodePtr& n, const BundleData& bundleData)
{
    return Node::isJoinNode(n) || bundleData.isSlicerReduction(n);
}