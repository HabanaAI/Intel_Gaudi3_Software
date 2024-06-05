#include "strategy_inflator_logic.h"
#include "compilation_hal_reader.h"
#include "brain_conf.h"

using namespace gc::layered_brain;

bool StrategyInflatorLogic::inflateOneStep(const StrategyPtr& strategy, const NodePtr& node) const
{
    const std::vector<BundleViewId>& inflationCandidates = getCandidatesForInflation(strategy, node);
    if (inflationCandidates.empty())
    {
        LOG_DEBUG(LB_SLICER, "Inflation failed, no candidates to inflate");
        return false;
    }
    return inflateCandidates(strategy, node, inflationCandidates);
}

bool StrategyInflatorLogic::inflateCandidates(const StrategyPtr&               strategy,
                                              const NodePtr&                   node,
                                              const std::vector<BundleViewId>& inflationCandidates) const
{
    LOG_DEBUG(LB_SLICER,
              "Inflation candidates for strategy {}: [{}]",
              strategy->index(),
              toString(inflationCandidates, ','));
    HB_ASSERT(inflationCandidates.size() == 1, "Expected a single inflation candidate");
    BundleViewId bvdToInflate = inflationCandidates.front();
    strategy->inflateBVD(bvdToInflate, m_bundleViews->getBundleView(bvdToInflate).resolution);
    return true;
}

bool StrategyInflatorLogic::canInflateBVD(const StrategyPtr& strategy, BundleViewId bvd) const
{
    const auto& currentMultiplier = strategy->getBVDMultiplier(bvd);
    if (!currentMultiplier.isSliced())
    {
        return false;
    }
    const auto&    commonDims          = strategy->getCommonDimsOfAllMMEs();
    bool           isSlicedOnCommonDim = std::find(commonDims.begin(), commonDims.end(), bvd) != commonDims.end();
    const uint64_t maxMultiplier       = m_bundleViews->getBundleView(bvd).resolution;
    if (isSlicedOnCommonDim)
    {
        // When inflating on a BVD which is a sliced common dimension, inflator should not inflate all the way,
        // as this changes the type of solution.
        return currentMultiplier.getMultiplier() < div_round_up(maxMultiplier, 2UL);
    }
    return currentMultiplier.getMultiplier() < maxMultiplier;
}

bool StrategyInflatorForUtilization::inflateCandidates(const StrategyPtr&               strategy,
                                                       const NodePtr&                   node,
                                                       const std::vector<BundleViewId>& inflationCandidates) const
{
    LOG_DEBUG(LB_SLICER,
              "Inflation candidates for strategy {}: [{}]",
              strategy->index(),
              toString(inflationCandidates, ','));
    auto mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    HB_ASSERT_PTR(mmeNode);
    HB_ASSERT_PTR(strategy->getMmeSolution());
    MmeSolutionPtr newSolution =
        mmeNode->getMmeBrainIfc()->inflateForUtilization(strategy->getMmeSolution(), node, m_bundleViews, std::nullopt);
    if (!newSolution)
    {
        LOG_DEBUG(LB_SLICER, "failed to inflate solution");
        return false;  // No solution found, inflation failed
    }
    for (const auto& [bvd, newMult] : newSolution->bvdMultipliers)
    {
        auto maxMult = m_bundleViews->getBundleView(bvd).resolution;
        HB_ASSERT((newMult <= maxMult) && (newMult > 0),
                  "Invalid multiplier {} for BVD {} (max multiplier={})",
                  newMult,
                  bvd,
                  maxMult);
        HB_ASSERT(strategy->getMmeSolution()->bvdMultipliers.find(bvd) !=
                      strategy->getMmeSolution()->bvdMultipliers.end(),
                  "BVD {} doesn't exist in solution",
                  bvd);
        auto oldMult = strategy->getMmeSolution()->bvdMultipliers.at(bvd);
        if (oldMult != newMult)
        {
            HB_ASSERT(std::find(inflationCandidates.begin(), inflationCandidates.end(), bvd) !=
                          inflationCandidates.end(),
                      "BVD {} is not a candidate for inflation",
                      bvd);
        }
        strategy->setBVDMultiplier(bvd, (newMult == maxMult) ? BVDMultiplier() : BVDMultiplier(newMult));
    }
    HB_ASSERT(newSolution->QORs.find(node) != newSolution->QORs.end(), "Missing QORs for node {}", node->getNodeName());
    HB_ASSERT(newSolution->QORs.at(node)->perfAttr.mmeUtilization >
                  strategy->getNodeQORs(node)->perfAttr.mmeUtilization,
              "MME utilization for node {} was not improved after inflation: old utilization = {} new utilization = {}",
              node->getNodeName(),
              strategy->getNodeQORs(node)->perfAttr.mmeUtilization,
              newSolution->QORs.at(node)->perfAttr.mmeUtilization);
    LOG_DEBUG(LB_SLICER,
              "Update MME solution for node {} strategy {} after inflation: old utilization = {} new utilization = {}",
              node->getNodeName(),
              strategy->index(),
              strategy->getNodeQORs(node)->perfAttr.mmeUtilization,
              newSolution->QORs.at(node)->perfAttr.mmeUtilization);
    strategy->updateMmeSolution(newSolution);
    return true;
}

std::vector<BundleViewId> StrategyInflatorForUtilization::getCandidatesForInflation(const StrategyPtr& strategy,
                                                                                    const NodePtr&     node) const
{
    HB_ASSERT_PTR(node);
    const auto& candidates = strategy->getMMEInflateForUtilizationBVDs(node);
    if (candidates.empty())
    {
        return {};
    }
    HB_ASSERT(strategy->getNodeQORs(node)->perfAttr.maxUtilization > 0,
              "Invalid max util for node {}",
              node->getNodeName());
    if ((strategy->getNodeQORs(node)->perfAttr.mmeUtilization / strategy->getNodeQORs(node)->perfAttr.maxUtilization) >
        GCFG_LAYERED_BRAIN_MAX_MME_UTIL_RATIO_FOR_INFLATION.value())
    {
        return {};  // Don't try to inflate if the gain is low
    }
    return candidates;
}

std::vector<BundleViewId> StrategyInflatorForBW::getCandidatesForInflation(const StrategyPtr& strategy,
                                                                           const NodePtr&     node) const
{
    HB_ASSERT_PTR(node);
    for (BundleViewId bvd : strategy->getMMEInflateForBwBVDs(node))
    {
        if (canInflateBVD(strategy, bvd))
        {
            return {bvd};
        }
    }
    return {};
}

std::vector<BundleViewId> StrategyInflatorForPerforation::getCandidatesForInflation(const StrategyPtr& strategy,
                                                                                    const NodePtr&     node) const
{
    HB_ASSERT_PTR(node);
    std::optional<BundleViewId> perforatedBVD = strategy->getPerforationBVDForNode(node);
    if (!perforatedBVD.has_value() || !canInflateBVD(strategy, perforatedBVD.value()) ||
        (strategy->getBVDMultiplier(perforatedBVD.value()).getMultiplier() %
             CompilationHalReader::getHalReader()->getNumDcores() ==
         0))
    {
        return {};
    }
    return {perforatedBVD.value()};
}

bool StrategyInflatorForPerforation::inflateCandidates(const StrategyPtr&               strategy,
                                                       const NodePtr&                   node,
                                                       const std::vector<BundleViewId>& inflationCandidates) const
{
    HB_ASSERT(inflationCandidates.size() == 1, "Expected a single inflation candidate");
    BundleViewId bvdToInflate = inflationCandidates.front();
    HB_ASSERT(strategy->getBVDMultiplier(bvdToInflate).isSliced(), "Expected a sliced BVD");
    const auto numDcores = CompilationHalReader::getHalReader()->getNumDcores();
    StrategyInflatorLogic::inflateCandidates(strategy, node, inflationCandidates);
    const auto& newMultiplier = strategy->getBVDMultiplier(bvdToInflate);
    if (newMultiplier.isSliced() && (newMultiplier.getMultiplier() % numDcores == 0))
    {
        // After successful inflation (multiplier % numDcores == 0) the old multiplier is replaced by the new one.
        // This will guarantee that later inflations of perforation BVDs, will keep the multiplier divided by num
        // dcores.
        strategy->setBVDMultiplier(bvdToInflate, BVDMultiplier(newMultiplier.getMultiplier()));
    }
    return true;
}

std::vector<BundleViewId> StrategyInflatorForNumSlices::getCandidatesForInflation(const StrategyPtr& strategy,
                                                                                  const NodePtr&     node) const
{
    HB_ASSERT(node == nullptr, "Inflate for num slices doesn't require node");
    std::vector<BundleViewId> inflationCandidates;
    for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
    {
        if (canInflateBVD(strategy, bvd))
        {
            inflationCandidates.push_back(bvd);
        }
    }
    if (inflationCandidates.empty())
    {
        return {};
    }
    BundleViewId inflateBVD = *std::min_element(
        inflationCandidates.begin(),
        inflationCandidates.end(),
        [&](const BundleViewId& bvd1, const BundleViewId& bvd2) {
            Dim minTensorDimInBVD1 = getSmallestTensorDimInBVD(bvd1);
            Dim minTensorDimInBVD2 = getSmallestTensorDimInBVD(bvd2);
            if (minTensorDimInBVD1 == minTensorDimInBVD2)
            {
                auto numOccurrencesOfTensorDimInBVD1 = getNumOccurrencesOfTensorDimInBVD(bvd1, minTensorDimInBVD1);
                auto numOccurrencesOfTensorDimInBVD2 = getNumOccurrencesOfTensorDimInBVD(bvd2, minTensorDimInBVD2);
                if (numOccurrencesOfTensorDimInBVD1 == numOccurrencesOfTensorDimInBVD2)
                {
                    return m_bundleViews->getBundleView(bvd1).id < m_bundleViews->getBundleView(bvd2).id;
                }
                return numOccurrencesOfTensorDimInBVD1 > numOccurrencesOfTensorDimInBVD2;
            }
            return minTensorDimInBVD1 < minTensorDimInBVD2;
        });
    return {inflateBVD};
}

Dim StrategyInflatorForNumSlices::getSmallestTensorDimInBVD(BundleViewId bvd) const
{
    const auto& bundleView = m_bundleViews->getBundleView(bvd);
    HB_ASSERT(!bundleView.tensorDimsGranularity.empty(), "Expected at least a single tensor dim in BVD {}", bvd);
    Dim minTensorDim = bundleView.tensorDimsGranularity.begin()->first.second;
    for (const auto& tensorDimGranularity : bundleView.tensorDimsGranularity)
    {
        minTensorDim = std::min(tensorDimGranularity.first.second, minTensorDim);
    }
    return minTensorDim;
}

unsigned StrategyInflatorForNumSlices::getNumOccurrencesOfTensorDimInBVD(BundleViewId bvd, Dim dim) const
{
    unsigned    numOccurrences = 0;
    const auto& bundleView     = m_bundleViews->getBundleView(bvd);
    for (const auto& tensorDimGranularity : bundleView.tensorDimsGranularity)
    {
        if (tensorDimGranularity.first.second == dim)
        {
            numOccurrences++;
        }
    }
    return numOccurrences;
}

bool StrategyInflatorForNumSlices::inflateCandidates(const StrategyPtr&               strategy,
                                                     const NodePtr&                   node,
                                                     const std::vector<BundleViewId>& inflationCandidates) const
{
    HB_ASSERT(inflationCandidates.size() == 1, "Expected a single inflation candidate");
    BundleViewId bvdToInflate               = inflationCandidates.front();
    uint64_t     numOfSlicesBeforeInflation = strategy->getNumOfSlicesForBVD(bvdToInflate, m_bundleViews);
    HB_ASSERT(numOfSlicesBeforeInflation > 1,
              "Invalid BVD candidate for inflation ({}), num of slices = {}",
              bvdToInflate,
              numOfSlicesBeforeInflation);
    do
    {
        // Continue to inflate until number of slices is changed
        StrategyInflatorLogic::inflateCandidates(strategy, node, inflationCandidates);
    } while ((numOfSlicesBeforeInflation == strategy->getNumOfSlicesForBVD(bvdToInflate, m_bundleViews)) &&
             canInflateBVD(strategy, bvdToInflate));
    return true;
}