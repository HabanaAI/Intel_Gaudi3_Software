#include "strategy.h"
#include "mme_brain_ifc.h"
#include "node.h"

using namespace gc::layered_brain;

std::atomic<uint64_t> Strategy::m_nextIdx {0};

Strategy::Strategy() : m_idx(m_nextIdx++), m_mmeSolution(nullptr) {}
Strategy::Strategy(const MmeSolutionPtr& mmeSolution) : m_idx(m_nextIdx++), m_mmeSolution(mmeSolution) {}
Strategy::Strategy(const Strategy& other)
: m_idx(other.m_idx),
  m_granularityMultiplier(other.m_granularityMultiplier),
  m_mmeSolution(other.m_mmeSolution ? std::make_shared<MmeSolution>(*other.m_mmeSolution) : nullptr),
  m_perforation(other.m_perforation),
  m_pipelineDepth(other.m_pipelineDepth)
{
}

StrategyPtr Strategy::clone() const
{
    return std::make_shared<Strategy>(*this);
}

uint64_t Strategy::index() const
{
    return m_idx;
}

BVDMultiplier Strategy::getBVDMultiplier(BundleViewId bvd) const
{
    auto it = m_granularityMultiplier.find(bvd);
    HB_ASSERT(it != m_granularityMultiplier.end(), "Missing multiplier for BVD id {} in strategy {}", bvd, m_idx);
    return it->second;
}

void Strategy::setBVDMultiplier(BundleViewId bvd, const BVDMultiplier& multiplier)
{
    m_granularityMultiplier[bvd] = multiplier;
}

void Strategy::inflateBVD(BundleViewId bvd, uint64_t resolution)
{
    auto it = m_granularityMultiplier.find(bvd);
    HB_ASSERT(it != m_granularityMultiplier.end(), "Missing multiplier for BVD id {} in strategy {}", bvd, m_idx);
    HB_ASSERT(it->second.isSliced(), "Expected a sliced BVD");
    it->second.inflateOneStep();
    if (it->second.getMultiplier() >= resolution)
    {
        // If the new multiplier after inflation >= BVD resolution - mark it as unsliced.
        it->second.unslice();
    }
}

void Strategy::fillMissingMultipliers(const BundleViewContainerPtr& bundleViews, uint64_t value)
{
    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
    {
        if (m_granularityMultiplier.find(bvdId) == m_granularityMultiplier.end())
        {
            LOG_DEBUG(LB_SLICER, "Missing multiplier for BVD {}, init multiplier to {}", bvdId, value);
            m_granularityMultiplier[bvdId] = BVDMultiplier(value);
        }
    }
}

uint64_t Strategy::getNumOfSlicesForBVD(BundleViewId bvd, const BundleViewContainerPtr& bundleViews) const
{
    const auto& bvdMultiplier = getBVDMultiplier(bvd);
    auto        bvdResolution = bundleViews->getBundleView(bvd).resolution;
    return bvdMultiplier.isSliced() ? div_round_up(bvdResolution, bvdMultiplier.getMultiplier()) : 1UL;
}

MmeSolutionPtr Strategy::getMmeSolution() const
{
    return m_mmeSolution;
}

void Strategy::updateMmeSolution(const MmeSolutionPtr& mmeSolution)
{
    HB_ASSERT_PTR(mmeSolution);
    m_mmeSolution = mmeSolution;
}

SolutionParamsPtr Strategy::getNodeQORs(const NodePtr& node) const
{
    HB_ASSERT_PTR(m_mmeSolution);
    const auto& it = m_mmeSolution->QORs.find(node);
    HB_ASSERT(it != m_mmeSolution->QORs.end(), "Invalid node {} for strategy {}", node->getNodeName(), m_idx);
    return it->second;
}

std::vector<BundleViewId> Strategy::getMMEInflateForUtilizationBVDs(const NodePtr& node) const
{
    const auto& nodeQORs    = getNodeQORs(node);
    const auto& inflateDims = nodeQORs->solutionRequirements.utilizationInflationDims;

    validateMmeSolutionBundleViewDims(inflateDims, node);

    std::vector<BundleViewId> inflateForUtilizationBVDs {inflateDims.begin(), inflateDims.end()};
    return inflateForUtilizationBVDs;
}

std::vector<BundleViewId> Strategy::getMMEInflateForBwBVDs(const NodePtr& node) const
{
    std::vector<BundleViewId> inflateForBwBVDs;
    const auto&               nodeQORs = getNodeQORs(node);
    if (nodeQORs->solutionRequirements.bwInflationDim.has_value())
    {
        BundleViewId inflateDim = nodeQORs->solutionRequirements.bwInflationDim.value();
        validateMmeSolutionBundleViewDims({inflateDim}, node);
        inflateForBwBVDs.push_back(inflateDim);
    }
    return inflateForBwBVDs;
}

std::vector<BundleViewId> Strategy::getMMEPreferredPerforationBVDs(const NodePtr& node) const
{
    const auto& nodeQORs        = getNodeQORs(node);
    const auto& perforationDims = nodeQORs->solutionRequirements.perforationDimVec;

    validateMmeSolutionBundleViewDims(perforationDims, node);

    std::vector<BundleViewId> perforationBVDs {perforationDims.begin(), perforationDims.end()};
    return perforationBVDs;
}

std::vector<BundleViewId> Strategy::getMMECommonDims(const NodePtr& node) const
{
    const auto& nodeQORs   = getNodeQORs(node);
    const auto& commonDims = nodeQORs->solutionRequirements.cdDims;

    validateMmeSolutionBundleViewDims(commonDims, node);

    std::vector<BundleViewId> commonDimsBVDs {commonDims.begin(), commonDims.end()};
    return commonDimsBVDs;
}

bool Strategy::isSlicedOnCommonDim() const
{
    if (!m_mmeSolution) return false;

    for (const auto& nodeSolution : m_mmeSolution->QORs)
    {
        if (nodeSolution.second->solutionRequirements.cdSliced)
        {
            return true;
        }
    }
    return false;
}

std::vector<BundleViewId> Strategy::getCommonDimsOfAllMMEs() const
{
    std::vector<BundleViewId> commonDimBVDs;
    if (m_mmeSolution)
    {
        for (const auto& [node, solutionParams] : m_mmeSolution->QORs)
        {
            const auto& commonDims = solutionParams->solutionRequirements.cdDims;
            validateMmeSolutionBundleViewDims(commonDims, node);
            commonDimBVDs.insert(commonDimBVDs.end(), commonDims.begin(), commonDims.end());
        }
    }
    return commonDimBVDs;
}

NodeToItemOrderedMap<std::vector<BundleViewId>> Strategy::getWalkPatternPerMmeNode() const
{
    NodeToItemOrderedMap<std::vector<BundleViewId>> walkDimsPerMmeNode;
    if (m_mmeSolution)
    {
        for (const auto& [node, solutionParams] : m_mmeSolution->QORs)
        {
            validateMmeSolutionBundleViewDims(solutionParams->solutionRequirements.walkDims, node);
            walkDimsPerMmeNode.emplace(node, solutionParams->solutionRequirements.walkDims);
        }
    }
    return walkDimsPerMmeNode;
}

void Strategy::validateMmeSolutionBundleViewDims(const std::vector<unsigned int>& dims, const NodePtr& node) const
{
    for (auto dim : dims)
    {
        HB_ASSERT(m_granularityMultiplier.find(dim) != m_granularityMultiplier.end(),
                  "Invalid BVD {} for node {}",
                  dim,
                  node->getNodeName());
    }
}

void Strategy::setPerforationData(const PerforationPerNode& perforationPerNode)
{
    m_perforation = perforationPerNode;
}

const PerforationPerNode& Strategy::getPerforationData() const
{
    return m_perforation;
}

std::optional<BundleViewId> Strategy::getPerforationBVDForNode(const NodePtr& node) const
{
    auto it = m_perforation.find(node);
    HB_ASSERT(it != m_perforation.end(),
              "Missing perforation dim for node {} in strategy {}",
              node->getNodeName(),
              m_idx);
    // Each node should have a perforation decision - can be empty if no available
    // candidates found (For example: all dims are all-required, low resolution).
    return it->second;
}

unsigned Strategy::getNumPerforatedNodes() const
{
    return std::count_if(m_perforation.begin(), m_perforation.end(), [](const auto& perforationPerNode) {
        return perforationPerNode.second.has_value();
    });
}

unsigned Strategy::getPipelineDepth() const
{
    return m_pipelineDepth;
}

void Strategy::setPipelineDepth(unsigned pipelineDepth)
{
    m_pipelineDepth = pipelineDepth;
}

void Strategy::log() const
{
    LOG_DEBUG(LB_SLICER, "###### Strategy {} #####", m_idx);
    LOG_DEBUG(LB_SLICER, "\t Pipeline depth: {}", m_pipelineDepth);
    LOG_DEBUG(LB_SLICER, "\t Granularity multiplier per bundle-view: ");
    for (const auto& [bvdId, granularityMultiplier] : m_granularityMultiplier)
    {
        LOG_DEBUG(
            LB_SLICER,
            "\t\t BVD : {} -> multiplier : {}",
            bvdId,
            ((granularityMultiplier.isSliced()) ? std::to_string(granularityMultiplier.getMultiplier()) : "UNSLICED"));
    }
    if (m_mmeSolution)
    {
        LOG_DEBUG(LB_SLICER, "\t Quality and expansion information per MME node: ");
        for (const auto& [node, info] : m_mmeSolution->QORs)
        {
            LOG_DEBUG(LB_SLICER, "\t\t Node {}:", node->getNodeName());
            LOG_DEBUG(LB_SLICER,
                      "\t\t\t Perf attributes: MaxUtil {}, Utilization {}, A BW {} Access {}/{}[D/C], B BW {} Access "
                      "{}/{}[D/C], Out BW {}",
                      info->perfAttr.maxUtilization,
                      info->perfAttr.mmeUtilization,
                      info->perfAttr.memoryAttrA.accessBW,
                      info->perfAttr.memoryAttrA.accessesPerDcore,
                      info->perfAttr.memoryAttrA.accessesPerChip,
                      info->perfAttr.memoryAttrB.accessBW,
                      info->perfAttr.memoryAttrB.accessesPerDcore,
                      info->perfAttr.memoryAttrB.accessesPerChip,
                      info->perfAttr.memoryAttrC.accessBW);
            LOG_DEBUG(LB_SLICER,
                      "\t\t\t Requirements (BVDs): Perforation: [{}], Inflation for BW: [{}], Inflation for "
                      "utilization: [{}], Walk: [{}], Common-dims: [{}], Memset: {}, Cast: {}, Reduction: {}",
                      toString(info->solutionRequirements.perforationDimVec, ','),
                      (info->solutionRequirements.bwInflationDim.has_value()
                           ? std::to_string(info->solutionRequirements.bwInflationDim.value())
                           : ""),
                      toString(info->solutionRequirements.utilizationInflationDims, ','),
                      toString(info->solutionRequirements.walkDims, ','),
                      toString(info->solutionRequirements.cdDims, ','),
                      info->solutionRequirements.requiresMemset,
                      info->solutionRequirements.requiresCast,
                      info->solutionRequirements.performsReduction);
        }
    }
}