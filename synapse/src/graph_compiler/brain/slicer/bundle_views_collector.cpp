#include "bundle_views_collector.h"
#include "access_pattern.h"

using namespace gc::layered_brain;

void BundleViewsCollector::createInitialBundleViews()
{
    for (const auto& node : m_bundleNodes)
    {
        const auto& nodeAP = node->getNodeAccessPattern();
        HB_ASSERT_PTR(nodeAP);
        for (const auto& tensor : node->getOperands())
        {
            if (!tensor) continue;
            for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
            {
                const Dim    indexSpaceDim = nodeAP->getIndexSpaceDim(tensor, tensorDim);
                auto         it            = m_nodeDimToBVD.find({node, indexSpaceDim});
                BundleViewId bvdId         = (it != m_nodeDimToBVD.end()) ? it->second : m_numOfInitialBVDs++;
                if (it == m_nodeDimToBVD.end())
                {
                    m_nodeDimToBVD[{node, indexSpaceDim}] = bvdId;
                }
                m_tensorDimToBVDSet[{tensor, tensorDim}].insert(bvdId);
                LOG_DEBUG(LB_SLICER,
                          "\t Map node {}, index-space dim {}, tensor {}, tensor dim {} to initial BVD id {}",
                          node->getNodeName(),
                          indexSpaceDim,
                          tensor->getName(),
                          tensorDim,
                          bvdId);
            }
        }
    }
}

bool BundleViewsCollector::isIntersect(const std::set<BundleViewId>& s1, const std::set<BundleViewId>& s2) const
{
    std::vector<BundleViewId> intersection;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(intersection));
    return !intersection.empty();
}

std::optional<std::pair<size_t, size_t>> BundleViewsCollector::findIntersectingPair() const
{
    for (auto i = 0; i < m_mergedBVDs.size(); i++)
    {
        for (auto j = 0; j < m_mergedBVDs.size(); j++)
        {
            if (i == j) continue;
            if (isIntersect(m_mergedBVDs[i], m_mergedBVDs[j]))
            {
                return std::make_pair(i, j);
            }
        }
    }
    return {};
}

void BundleViewsCollector::mergeInitialBundleViews()
{
    // Start with sets of initial BVD ids of each tensor dim.
    for (const auto& tensorDimToBVDSet : m_tensorDimToBVDSet)
    {
        m_mergedBVDs.push_back(tensorDimToBVDSet.second);
    }

    // Merge all sets that intersect, so we end up with a single merged BVD
    // for each tensor dim.
    while (auto intersectPair = findIntersectingPair())
    {
        LOG_DEBUG(
            LB_SLICER,
            "\t Merged bundle views : [{}] and [{}]",
            toString(m_mergedBVDs[intersectPair->first].begin(), m_mergedBVDs[intersectPair->first].end(), ','),
            toString(m_mergedBVDs[intersectPair->second].begin(), m_mergedBVDs[intersectPair->second].end(), ','));
        // Merge the second set into the first one
        m_mergedBVDs[intersectPair->first].merge(m_mergedBVDs[intersectPair->second]);
        m_mergedBVDs.erase(m_mergedBVDs.begin() + intersectPair->second);
    }
}

void BundleViewsCollector::reallocateBundleViewIds()
{
    m_initialBVDToFinalBVD.resize(m_numOfInitialBVDs);
    uint32_t bvdIdsCount = 0;
    std::sort(m_mergedBVDs.begin(),
              m_mergedBVDs.end(),
              [](const std::set<BundleViewId>& bvdSet1, const std::set<BundleViewId>& bvdSet2) {
                  // Sort by minimal BVD ID in the set, to keep the BVDs consistent.
                  return *bvdSet1.begin() < *bvdSet2.begin();
              });
    for (const auto& mergedBVD : m_mergedBVDs)
    {
        LOG_DEBUG(LB_SLICER,
                  "\t Map merged bundle-view ids {} to final bundle view id {}",
                  toString(mergedBVD.begin(), mergedBVD.end(), ','),
                  m_numOfFinalBVDs);
        for (const auto& bvdId : mergedBVD)
        {
            m_initialBVDToFinalBVD[bvdId] = m_numOfFinalBVDs;
        }
        m_numOfFinalBVDs++;
        bvdIdsCount += mergedBVD.size();
    }
    HB_ASSERT(m_numOfInitialBVDs == bvdIdsCount, "Each initial BVD id should appear in one merged BVD");
}

BundleViewContainerPtr BundleViewsCollector::createFinalBundleViews(const TileSizePerTensor& granularityPerTensor,
                                                                    const TileSizePerNode&   granularityPerNode)
{
    reallocateBundleViewIds();

    auto bundleViews = std::make_shared<BundleViewContainer>(m_numOfFinalBVDs);

    for (const auto& [tensorDim, bvdId] : m_tensorDimToBVDSet)
    {
        HB_ASSERT(*bvdId.begin() < m_initialBVDToFinalBVD.size(), "Invalid initial BVD id {}", *bvdId.begin());
        const BundleViewId finalBVDId = m_initialBVDToFinalBVD[*bvdId.begin()];
        HB_ASSERT(finalBVDId < m_numOfFinalBVDs, "Invalid final BVD id {}", finalBVDId);
        const auto& it = granularityPerTensor.find(tensorDim.first);
        HB_ASSERT(it != granularityPerTensor.end(),
                  "Failed to find granularity for tensor {}",
                  tensorDim.first->getName());
        bundleViews->mapTensorDimToBVD(tensorDim.first, tensorDim.second, finalBVDId, it->second[tensorDim.second]);
    }

    for (const auto& [nodeDim, bvdId] : m_nodeDimToBVD)
    {
        HB_ASSERT(bvdId < m_initialBVDToFinalBVD.size(), "Invalid initial BVD id {}", bvdId);
        const BundleViewId finalBVDId = m_initialBVDToFinalBVD[bvdId];
        HB_ASSERT(finalBVDId < m_numOfFinalBVDs, "Invalid final BVD id {}", finalBVDId);
        const auto& it = granularityPerNode.find(nodeDim.first);
        HB_ASSERT(it != granularityPerNode.end(),
                  "Failed to find granularity for node {}",
                  nodeDim.first->getNodeName());
        bundleViews->mapNodeDimToBVD(nodeDim.first, nodeDim.second, finalBVDId, it->second[nodeDim.second]);
    }

    return bundleViews;
}

BundleViewContainerPtr BundleViewsCollector::getAllBundleViews(const TileSizePerTensor& granularityPerTensor,
                                                               const TileSizePerNode&   granularityPerNode)
{
    SET_TEMP_LOG_CONTEXT("BundleViewsCollector");

    LOG_TRACE(LB_SLICER, "Collecting bundle views for bundle");

    LOG_DEBUG(LB_SLICER, "Stage 1: Iterate over the bundle nodes and tensors to create initial bundle view list");
    createInitialBundleViews();

    LOG_DEBUG(LB_SLICER, "Stage 2: Merge initial bundle views");
    mergeInitialBundleViews();

    LOG_DEBUG(LB_SLICER, "Stage 3: Create final bundle views, update granularity per each tensor/node dim");
    const auto& bundleViews = createFinalBundleViews(granularityPerTensor, granularityPerNode);

    bundleViews->logBundleViews();

    LOG_TRACE(LB_SLICER, "{} bundle views created for bundle", bundleViews->getNumOfBundleViews());

    return bundleViews;
}