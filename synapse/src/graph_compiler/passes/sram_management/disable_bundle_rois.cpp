#include "habana_pass.h"
#include "log_manager.h"
#include "habana_graph.h"
#include "habana_nodes.h"

struct BundleEncounters
{
    bool firstMme = true;
    bool firstTpc = true;
    bool firstDma = true;
    pNode lastMme = nullptr;
    pNode lastTpc = nullptr;
    pNode lastDma = nullptr;
};

bool disableBundleRois(HabanaGraph& g)
{
    LOG_INFO(GC, "{}: Disable ROIs for bundle intermediate nodes", HLLOG_FUNC);

    if (!GCFG_PIPELINE_BUNDLE_EDGES_ENABLED.value())
    {
        // Backwards compatibility mode
        for (pNode n : g.getExeSortedNodes())
        {
            NodeAnnotation& annotations = n->getNodeAnnotation();
            // Dma transpose must split.
            if (annotations.bundleInfo.is_set() && annotations.bundleInfo->operationIndex > 0 && annotations.bundleInfo->bundleType != BundleType::SCALAR_PIPE)
            {
                annotations.updateSplitToLogicalROIs(false);
            }
        }
        return true;
    }

    using BundleID = unsigned;
    std::unordered_map<BundleID, BundleEncounters> bundleEncounters;
    for (pNode n : g.getExeSortedNodes())
    {
        NodeAnnotation& annotations = n->getNodeAnnotation();
        if (annotations.bundleInfo.is_set() && !annotations.isPerforated() &&
            (annotations.bundleInfo->bundleType != BundleType::SCALAR_PIPE))
        {
            // Note: Does not insert if there is already an encounter in the map.
            auto iter = bundleEncounters.insert({annotations.bundleInfo->bundleIndex, BundleEncounters{}}).first;
            BundleEncounters& encounters = iter->second;

            if (HabanaGraph::runsOnMME(n))
            {
                annotations.updateSplitToLogicalROIs(encounters.firstMme);
                encounters.firstMme = false;
                encounters.lastMme = n;
            }
            if (HabanaGraph::runsOnTPC(n))
            {
                annotations.updateSplitToLogicalROIs(encounters.firstTpc);
                encounters.firstTpc = false;
                encounters.lastTpc = n;
            }
            if (std::dynamic_pointer_cast<DMANode>(n) != nullptr)
            {
                // Dma transpose must split.
                annotations.updateSplitToLogicalROIs(encounters.firstDma);
                encounters.firstDma = false;
                encounters.lastDma = n;
            }
        }
    }

    for (auto& bundleIdxAndEncounters : bundleEncounters)
    {
        BundleEncounters& encounters = bundleIdxAndEncounters.second;
        if (encounters.lastMme)
        {
            encounters.lastMme->getNodeAnnotation().updateSplitToLogicalROIs(true);
        }
        if (encounters.lastTpc)
        {
            encounters.lastTpc->getNodeAnnotation().updateSplitToLogicalROIs(true);
        }
        if (encounters.lastDma)
        {
            encounters.lastDma->getNodeAnnotation().updateSplitToLogicalROIs(true);
        }
    }

    return true;
}
