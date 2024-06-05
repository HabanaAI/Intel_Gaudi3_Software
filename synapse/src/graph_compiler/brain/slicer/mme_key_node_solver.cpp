#include "mme_key_node_solver.h"
#include "mme_brain_ifc.h"

using namespace gc::layered_brain;

StrategyContainer MMEKeyNodeSolver::getSlicingStrategies(const BundleViewContainerPtr& bundleViews,
                                                         const StrategyContainer&      existingStrategies)
{
    SET_TEMP_LOG_CONTEXT("MMEKeyNodeSolver");
    LOG_TRACE(LB_SLICER, "Get slicing strategies for node {}", m_keyNode->getNodeName());

    HB_ASSERT(HabanaGraph::runsOnMME(m_keyNode), "Expected {} to be a MME node", m_keyNode->getNodeName());
    HB_ASSERT(existingStrategies.strategies.size() == existingStrategies.mmeSolutions.size(),
              "Invalid previous strategies container for MME node {}",
              m_keyNode->getNodeName());

    auto mmeNode      = std::dynamic_pointer_cast<MmeNode>(m_keyNode);
    auto mmeSolutions = mmeNode->getMmeBrainIfc()->generateLayeredBrainStrategies(m_keyNode,
                                                                                  bundleViews,
                                                                                  existingStrategies.mmeSolutions);

    if (mmeSolutions.empty())
    {
        HB_ASSERT(!existingStrategies.nodes.empty(),
                  "Expected at least a single solution from MME brain for node {} (no constraints from previous MMEs)",
                  m_keyNode->getNodeName());
        LOG_WARN(LB_SLICER, "No strategies found for node {}", m_keyNode->getNodeName());
        return {};
    }

    if (GCFG_HARD_TRIM_MME_BRAIN_STRATEGIES.value() > 0 &&
        mmeSolutions.size() > GCFG_HARD_TRIM_MME_BRAIN_STRATEGIES.value())
    {
        LOG_DEBUG(LB_SLICER,
                  "Trimming number of solutions from MME brain to {}",
                  GCFG_HARD_TRIM_MME_BRAIN_STRATEGIES.value());
        mmeSolutions.resize(GCFG_HARD_TRIM_MME_BRAIN_STRATEGIES.value());
    }

    LOG_DEBUG(LB_SLICER,
              "Create layered-brain strategies for node {} from {} MME brain solutions",
              m_keyNode->getNodeName(),
              mmeSolutions.size());

    const auto& strategies = createStrategiesFromMmeSolutions(bundleViews, mmeSolutions, existingStrategies);

    logStrategies(strategies, bundleViews);

    LOG_TRACE(LB_SLICER, "{} strategies created for node {}", strategies.strategies.size(), m_keyNode->getNodeName());

    return strategies;
}

StrategyContainer MMEKeyNodeSolver::createStrategiesFromMmeSolutions(const BundleViewContainerPtr& bundleViews,
                                                                     const MmeSolutionContainer&   mmeSolutions,
                                                                     const StrategyContainer& existingStrategies) const
{
    StrategyContainer strategies;
    strategies.nodes = existingStrategies.nodes;
    strategies.nodes.push_back(m_keyNode);
    strategies.mmeSolutions = mmeSolutions;
    for (const auto& mmeSolution : mmeSolutions)
    {
        StrategyPtr strategy = std::make_shared<Strategy>(mmeSolution);
        for (const auto& bvd : bundleViews->getNodesBVDs(strategies.nodes))
        {
            HB_ASSERT(mmeSolution->bvdMultipliers.find(bvd) != mmeSolution->bvdMultipliers.end(),
                      "Missing multiplier for BVD {} in MME solution",
                      bvd);
            auto bvdMultiplier = mmeSolution->bvdMultipliers.at(bvd);
            HB_ASSERT((bvdMultiplier <= bundleViews->getBundleView(bvd).resolution) && (bvdMultiplier > 0),
                      "Invalid multiplier {} for BVD {} (max multiplier={})",
                      bvdMultiplier,
                      bvd,
                      bundleViews->getBundleView(bvd).resolution);
            strategy->setBVDMultiplier(bvd,
                                       (bvdMultiplier == bundleViews->getBundleView(bvd).resolution)
                                           ? BVDMultiplier()
                                           : BVDMultiplier(bvdMultiplier));
        }

        HB_ASSERT(mmeSolution->brainSolution.size() == strategies.nodes.size(),
                  "Expected a MME brain solution for {} nodes",
                  strategies.nodes.size());
        HB_ASSERT(mmeSolution->QORs.size() == strategies.nodes.size(),
                  "Expected quality and expansion information for {} node",
                  strategies.nodes.size());
        for (const auto& node : strategies.nodes)
        {
            HB_ASSERT(mmeSolution->brainSolution.find(node) != mmeSolution->brainSolution.end(),
                      "Missing MME brain solution for MME node {}",
                      node->getNodeName());
            HB_ASSERT(mmeSolution->QORs.find(node) != mmeSolution->QORs.end(),
                      "Missing quality and expansion information for MME node {}",
                      node->getNodeName());
        }

        strategies.strategies.push_back(strategy);
    }
    return strategies;
}