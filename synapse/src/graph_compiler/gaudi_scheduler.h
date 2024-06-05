#pragma once

#include "node.h"
#include "scheduler.h"

class GaudiScheduler : public Scheduler
{
    using BundleIdx         = unsigned;

public:
    explicit GaudiScheduler(const Graph* graph) : Scheduler(graph) {}

    NodeList scheduleNodes() override;

    static std::map<BundleIdx, NodePtr> findFirstNodePerBundle(const Graph& g);

protected:
    void        optimizeMemsetsLocation(NodeList& schedule) const;
    void        generateExecutionSchedule(NodeList& schedule) const;
    NodeSet     getBlockingNodes(const NodePtr& node) const;
    NodeSet     getBlockedNodes(const NodePtr& node) const;
    void        createFirstNodePerBundleMapping();
    bool        isValidGraphForMaxPath() const;
    NodePtr     replaceBlockedNodeWithFirstInBundle(const NodePtr& blocked, const NodePtr& memsetNode) const;
    bool        shouldOptimizeMemset(const NodePtr& memset) const;
    static void fixupMemsetBundleOpIndex(const NodePtr& memset, const NodePtr& reductionProducer);

    std::map<BundleIdx, NodePtr> m_bundleIdxToFirstNode;
};

namespace gaudi
{
    struct NodeScheduleComparator
    {
        std::optional<bool> compareBundlePriority(const NodePtr& n1, const NodePtr& n2) const
        {
            //ensure everything comes before null
            if (n1 == nullptr) return false;
            if (n2 == nullptr) return true;

            if (n1->getNodeAnnotation().bundleInfo.is_set() && !n2->getNodeAnnotation().bundleInfo.is_set())
                return false;

            if (!n1->getNodeAnnotation().bundleInfo.is_set() && n2->getNodeAnnotation().bundleInfo.is_set())
                return true;

            if (n1->getNodeAnnotation().bundleInfo.is_set() && n2->getNodeAnnotation().bundleInfo.is_set())
            {
                // TODO: [SW-6241] Support interleaving of bundles
                if (n1->getNodeAnnotation().bundleInfo->bundleType != BundleType::SCALAR_PIPE && n2->getNodeAnnotation().bundleInfo->bundleType == BundleType::SCALAR_PIPE)
                    return true;
                if (n1->getNodeAnnotation().bundleInfo->bundleType == BundleType::SCALAR_PIPE && n2->getNodeAnnotation().bundleInfo->bundleType != BundleType::SCALAR_PIPE)
                    return false;
                if (n1->getNodeAnnotation().bundleInfo->bundleIndex < n2->getNodeAnnotation().bundleInfo->bundleIndex)
                    return true;
                if (n2->getNodeAnnotation().bundleInfo->bundleIndex < n1->getNodeAnnotation().bundleInfo->bundleIndex)
                    return false;
                if (n1->getNodeAnnotation().bundleInfo->bundleIndex == n2->getNodeAnnotation().bundleInfo->bundleIndex)
                {
                    if (n1->getNodeAnnotation().bundleInfo->operationIndex <
                        n2->getNodeAnnotation().bundleInfo->operationIndex)
                        return true;
                    if (n2->getNodeAnnotation().bundleInfo->operationIndex <
                        n1->getNodeAnnotation().bundleInfo->operationIndex)
                        return false;
                }
            }
            return std::nullopt;
        }

        bool compareNodeId(const NodePtr& n1, const NodePtr& n2) const
        {
            if (GCFG_ENABLE_PARENT_ID_SCHEDULE.value())
            {
                if (n1->getParentId() < n2->getParentId()) return true;
                if (n1->getParentId() > n2->getParentId()) return false;
            }

            //fall-through path, just break ties by order of creation for consistency between compilations
            return n1->getId() < n2->getId();
        }

        bool operator()(const NodePtr& n1, const NodePtr& n2) const
        {
            return compareBundlePriority(n1,n2).value_or(compareNodeId(n1, n2));
        }
    }; //NodeScheduleComparator
    }  // namespace gaudi
