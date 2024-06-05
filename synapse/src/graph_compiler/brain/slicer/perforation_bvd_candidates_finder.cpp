#include "perforation_bvd_candidates_finder.h"
#include "habana_graph.h"
#include "habana_global_conf.h"
#include "math_utils.h"
#include "brain_conf.h"
#include "node.h"

using namespace gc::layered_brain;

std::map<NodePtr, PerforationCandidates>
PerforationBVDCandidatesFinder::findPerforationCandidates(const StrategyPtr&        strategy,
                                                          const ReducedBVDsPerNode& reducedBVDsPerNode) const
{
    LOG_DEBUG(LB_SLICER, "Find perforation candidates for strategy {}", strategy->index());

    const auto& bundlePreferredCandidates = getBundlePreferredCandidates(strategy, reducedBVDsPerNode);
    const auto& bundleValidCandidates     = getBundleValidCandidates(strategy);

    LOG_DEBUG(LB_SLICER,
              "Bundle preferred candidates: [{}], Bundle valid candidates: [{}]",
              toString(bundlePreferredCandidates, ','),
              toString(bundleValidCandidates, ','));

    // Update candidates per node
    std::map<NodePtr, PerforationCandidates> candidates;
    for (const auto& node : m_bundleNodes)
    {
        candidates[node] = getNodeCandidates(node,
                                             strategy,
                                             bundlePreferredCandidates,
                                             bundleValidCandidates,
                                             getNodeReducedBVDs(node, reducedBVDsPerNode));
        LOG_DEBUG(LB_SLICER,
                  "Perforation candidates for node {} : MME candidate = [{}], Preferred candidates = [{}], Valid "
                  "candidates = [{}]",
                  node->getNodeName(),
                  candidates.at(node).mmeCandidate.has_value()
                      ? std::to_string(candidates.at(node).mmeCandidate.value())
                      : "",
                  toString(candidates.at(node).preferredCandidates, ','),
                  toString(candidates.at(node).validCandidates, ','));
    }
    return candidates;
}

std::vector<BundleViewId>
PerforationBVDCandidatesFinder::getBundlePreferredCandidates(const StrategyPtr&        strategy,
                                                             const ReducedBVDsPerNode& reducedBVDsPerNode) const
{
    std::vector<BundleViewId> candidates;
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        const auto bvdResolution   = m_bundleViews->getBundleView(bvd).resolution;
        float      perforationUtil = (float)bvdResolution / (float)round_to_multiple(bvdResolution, m_numDcores);
        // Prefer candidates with high utilization to minimize the utilization loss as a result of perforation.
        // For example: BVD with 7 granules will have 7/8 utilization (assuming 4 DCOREs).
        if ((bvdResolution >= m_numDcores) && (perforationUtil >= GCFG_PERFORATION_UTILIZATION_THRESHOLD.value()))
        {
            candidates.push_back(bvd);
        }
    }

    BVDSet bundleReducedBVDs {};
    for (const auto& nodeReducedBVDs : reducedBVDsPerNode)
    {
        bundleReducedBVDs.insert(nodeReducedBVDs.second.begin(), nodeReducedBVDs.second.end());
    }
    LOG_DEBUG(LB_SLICER,
              "Filter out bundle reduced BVDs from preferred candidates: {}",
              toString(bundleReducedBVDs, ','));

    // Remove reduction dimensions from preferred candidates
    candidates.erase(std::remove_if(candidates.begin(),
                                    candidates.end(),
                                    [&bundleReducedBVDs](const BundleViewId& bvd) {
                                        return bundleReducedBVDs.find(bvd) != bundleReducedBVDs.end();
                                    }),
                     candidates.end());

    // Sort the preferred candidates based on number of occurrences in bundle nodes
    std::sort(candidates.begin(), candidates.end(), [&](const BundleViewId& bvd1, const BundleViewId& bvd2) {
        auto numNodeDimsInBVD1 = m_bundleViews->getBundleView(bvd1).nodeDimsGranularity.size();
        auto numNodeDimsInBVD2 = m_bundleViews->getBundleView(bvd2).nodeDimsGranularity.size();
        if (numNodeDimsInBVD1 != numNodeDimsInBVD2)
        {
            return numNodeDimsInBVD1 > numNodeDimsInBVD2;
        }
        return m_bundleViews->getBundleView(bvd1).id < m_bundleViews->getBundleView(bvd2).id;
    });

    return candidates;
}

std::vector<BundleViewId> PerforationBVDCandidatesFinder::getBundleValidCandidates(const StrategyPtr& strategy) const
{
    std::vector<BundleViewId> candidates;
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        if (m_bundleViews->getBundleView(bvd).resolution >= m_numDcores)
        {
            candidates.push_back(bvd);
        }
    }
    return candidates;
}

bool PerforationBVDCandidatesFinder::isCDPerforationAllowed(const NodePtr&     mmeNode,
                                                            const StrategyPtr& strategy,
                                                            BundleViewId       cdBVD) const
{
    const auto& perforationBVDs = strategy->getMMEPreferredPerforationBVDs(mmeNode);
    bool        isMMEPreferredPerforationBVD =
        std::find(perforationBVDs.begin(), perforationBVDs.end(), cdBVD) != perforationBVDs.end();
    if (!isMMEPreferredPerforationBVD)
    {
        return false;
    }

    bool requiresMemset = strategy->getNodeQORs(mmeNode)->solutionRequirements.requiresMemset;
    if (!GCFG_ENABLE_CD_PARALLEL.value() && !requiresMemset)
    {
        return false;  // both cd-parallel and non-deterministic cd-concurrency (requires memset) aren't allowed
    }

    return true;
}

bool PerforationBVDCandidatesFinder::isValidNodeBVD(const NodePtr&     node,
                                                    BundleViewId       bvd,
                                                    const StrategyPtr& strategy,
                                                    const BVDSet&      tpcReducedBVDs) const
{
    if (HabanaGraph::runsOnMME(node))
    {
        const auto& commonDims = strategy->getMMECommonDims(node);
        const bool  isCDBVD    = std::find(commonDims.begin(), commonDims.end(), bvd) != commonDims.end();
        if (isCDBVD && !isCDPerforationAllowed(node, strategy, bvd))
        {
            return false;  // Block perforation on common dims for solutions without cd-concurrency (memset required)
                           // or without cd-parallel
        }
    }
    else if (tpcReducedBVDs.find(bvd) != tpcReducedBVDs.end())
    {
        return false;  // Block perforation on reduced dims
    }
    // Block perforation in case the node has multiple dims in the BVD
    return m_bundleViews->getNodeDimsInBVD(bvd, node).size() == 1;
}

// Returns the most external tensor dim of the given node in the BVD and the number of occurrences of this dim
std::pair<Dim, unsigned> PerforationBVDCandidatesFinder::getNodeExternalTensorDimInBVD(const NodePtr& node,
                                                                                       BundleViewId   bvd) const
{
    const auto&        nodeOperands = node->getOperands();
    std::multiset<Dim> nodeTensorDims;
    for (const auto& tensorDimGranularity : m_bundleViews->getBundleView(bvd).tensorDimsGranularity)
    {
        if (std::find(nodeOperands.begin(), nodeOperands.end(), tensorDimGranularity.first.first) != nodeOperands.end())
        {
            nodeTensorDims.insert(tensorDimGranularity.first.second);
        }
    }
    HB_ASSERT(!nodeTensorDims.empty(), "BVD {} doesn't contain tensor dims for node {}", bvd, node->getNodeName());
    Dim nodeExternalTensorDimInBVD = *nodeTensorDims.rbegin();  // last element
    return {nodeExternalTensorDimInBVD, nodeTensorDims.count(nodeExternalTensorDimInBVD)};
}

PerforationCandidates
PerforationBVDCandidatesFinder::getNodeCandidates(const NodePtr&                   node,
                                                  const StrategyPtr&               strategy,
                                                  const std::vector<BundleViewId>& bundlePreferredCandidates,
                                                  const std::vector<BundleViewId>& bundleValidCandidates,
                                                  const BVDSet&                    reducedBVDs) const
{
    PerforationCandidates nodeCandidates;
    for (const auto& bvd : bundlePreferredCandidates)
    {
        if (isValidNodeBVD(node, bvd, strategy, reducedBVDs))
        {
            nodeCandidates.preferredCandidates.push_back(bvd);
        }
    }
    for (const auto& bvd : bundleValidCandidates)
    {
        if (isValidNodeBVD(node, bvd, strategy, reducedBVDs))
        {
            nodeCandidates.validCandidates.push_back(bvd);
        }
    }

    // Sort valid candidates, preferred candidates are already sorted
    std::sort(nodeCandidates.validCandidates.begin(),
              nodeCandidates.validCandidates.end(),
              [&](const BundleViewId& bvd1, const BundleViewId& bvd2) {
                  auto externalDimInBVD1 = getNodeExternalTensorDimInBVD(node, bvd1);
                  auto externalDimInBVD2 = getNodeExternalTensorDimInBVD(node, bvd2);
                  if (externalDimInBVD1.first != externalDimInBVD2.first)
                  {
                      return externalDimInBVD1.first > externalDimInBVD2.first;
                  }
                  else if (externalDimInBVD1.second != externalDimInBVD2.second)
                  {
                      return externalDimInBVD1.second > externalDimInBVD2.second;
                  }
                  return m_bundleViews->getBundleView(bvd1).id < m_bundleViews->getBundleView(bvd2).id;
              });

    if (HabanaGraph::runsOnMME(node))
    {
        const auto& mmePreferredPerforationBVDs = strategy->getMMEPreferredPerforationBVDs(node);
        if (!mmePreferredPerforationBVDs.empty() &&
            isValidNodeBVD(node, mmePreferredPerforationBVDs.front(), strategy, reducedBVDs))
        {
            nodeCandidates.mmeCandidate = mmePreferredPerforationBVDs.front();
        }
    }
    return nodeCandidates;
}

BVDSet PerforationBVDCandidatesFinder::getNodeReducedBVDs(const NodePtr&            node,
                                                          const ReducedBVDsPerNode& reducedBVDsPerNode) const
{
    if (reducedBVDsPerNode.find(node) != reducedBVDsPerNode.end())
    {
        return reducedBVDsPerNode.at(node);
    }
    return {};
}