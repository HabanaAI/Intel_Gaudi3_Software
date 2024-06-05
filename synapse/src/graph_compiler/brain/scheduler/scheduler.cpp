#include "scheduler.h"
#include "bundle_collector.h"
#include "habana_graph.h"
#include "bundle_scheduler.h"
#include "brain_data.h"

using namespace gc::layered_brain;

bool SlicedNodesScheduler::handleAllBundles(bool dryRun)
{
    for (const auto& bundleIdxAndNodes : getBundles())
    {
        const auto* lbData = m_graph.getLayeredBrainData();
        HB_ASSERT_PTR(lbData);

        if (!lbData->isLayeredBrainBundle(bundleIdxAndNodes.first)) continue;
        if (alreadyScheduled(bundleIdxAndNodes.second)) continue;

        bool res = scheduleBundle(bundleIdxAndNodes.second, dryRun);
        CHECK_RET_FALSE(res, "Failed to schedule bundle {} nodes.", bundleIdxAndNodes.first);
    }
    return true;
}

Bundles SlicedNodesScheduler::getBundles() const
{
    BundleCollector bc {m_graph};
    return bc.getAllBundles();
}

bool SlicedNodesScheduler::alreadyScheduled(const BundleNodes& bundleNodes) const
{
    for (const NodePtr& n : bundleNodes)
    {
        HB_ASSERT(n->getNodeAnnotation().bundleInfo.is_set(),
                  "Unexpecte node without bundle info: {}",
                  n->getNodeName());
        if (n->getNodeAnnotation().bundleInfo->operationIndex > 0)
        {
            // Bundle already scheduled
            return true;
        }
    }
    return false;
}

bool SlicedNodesScheduler::scheduleBundle(const BundleNodes& bundleNodes, bool dryRun)
{
    LOG_TRACE(LB_SCHEDULER, "{}", HLLOG_FUNC);

    BundleScheduler scheduler {m_graph, bundleNodes};
    bool            scheduled = scheduler.setBundleNodesSchedule(dryRun);
    if (scheduled)
    {
        scheduleMemsetNodes(bundleNodes);
        m_graph.invalidateExecutionSchedule();
    }
    return scheduled;
}

// Schedule memset operation as late as possible.
// This flow is needed for compatability between the node scheduling by the Gaudi Scheduler and the Bundle Scheduler.
// It's not needed for scheduling correctness.
// When Gaudi Scheduler and Bundle Scheduler are fully compatible, this part can be removed
void SlicedNodesScheduler::setOpIdxForMemset(const NodePtr& memsetNode) const
{
    NodeSet memsetBlocked = m_graph.getNodeConsumers(memsetNode, Node::TENSOR_TYPE_ALL);
    HB_ASSERT(!memsetBlocked.empty(), "Memset node: {} has no consumers", memsetNode->getNodeName());

    bool allBlockedInBundle = std::all_of(memsetBlocked.begin(), memsetBlocked.end(), [&memsetNode](const NodePtr& n) {
        return n->getNodeAnnotation().bundleInfo.is_set() &&
               n->getNodeAnnotation().bundleInfo->bundleIndex ==
                   memsetNode->getNodeAnnotation().bundleInfo->bundleIndex;
    });

    // For the case handled in this flow, we assume all the consumers are in the bundle
    HB_ASSERT(allBlockedInBundle, "Memset node {} has consumer out of bundle", memsetNode->getNodeName());

    const auto& blockedWithMinOpIndexIt =
        std::min_element(memsetBlocked.begin(), memsetBlocked.end(), [](const NodePtr& a, const NodePtr& b) {
            return a->getNodeAnnotation().bundleInfo->operationIndex <
                   b->getNodeAnnotation().bundleInfo->operationIndex;
        });

    memsetNode->getNodeAnnotation().bundleInfo->operationIndex =
        (*blockedWithMinOpIndexIt)->getNodeAnnotation().bundleInfo->operationIndex;
}

void SlicedNodesScheduler::scheduleMemsetNodes(const BundleNodes& bundleNodes) const
{
    // If the bundle scheduler handles memset - don't change their scheduling order
    if (GCFG_ENABLE_LAYERED_BRAIN_SCHEDULER_OPTIMIZE_MEMSET.value()) return;
    // TODO: temp WA until SW-160814 is solved - schedule memset nodes in the beginning of the bundle
    for (const auto& n : bundleNodes)
    {
        if (n->isMemset() && (n->getNumInputs(Node::TENSOR_TYPE_DATA) == 0))
        {
            HB_ASSERT(n->getNodeAnnotation().bundleInfo.is_set(),
                      "Bundle info was not set for node {}",
                      n->getNodeName());
            setOpIdxForMemset(n);
        }
    }
}

bool bundleNodesSchedule(HabanaGraph& g)
{
    if (BundleCollector::nofLBBundles(g) == 0)
    {
        // nothing to do
        return true;
    }
    bool scheduled = SlicedNodesScheduler(g).handleAllBundles(false /* dryRun */);
    if (scheduled)
    {
        g.invalidateExecutionSchedule();
    }
    return scheduled;
}
