#include "legacy_mme_key_node_solver.h"
#include "slicing_brain.h"

using namespace gc::layered_brain;

BundleSolutionConstraints
LegacyMMEKeyNodeSolver::prepareMmeBrainRequest(const BundleViewContainerPtr& bundleViews) const
{
    BundleSolutionConstraints constraints {};

    // Temp implementation - give the node solver full SRAM budget and let it do the inflation.
    // m_maxTileSize and m_minCommonDimForPartials are ignored.
    constraints.availableSramBytes = SlicingBrain::knobs.maxSRAMCapInBytes;
    constraints.canSliceMultipleDims = false;
    constraints.canAlignInputToCL  = false;
    // Slice input 0 and reserve SRAM for output - can be improved later if needed.
    constraints.tensorsInSram                    = {m_keyNode->getInput(0), m_keyNode->getOutput(0)};
    constraints.sharedOperandSlicingDimsAlignment = {};

    // Update slicing granularity constraints
    for (const auto& tensor : m_keyNode->getOperands())
    {
        if (tensor)
        {
            constraints.slicingGranularityPerTensor[tensor] = TensorTile::Geometry(tensor->getDim(), 1);
            for (auto tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
            {
                constraints.slicingGranularityPerTensor[tensor][tensorDim] =
                    bundleViews->getGranularityForTensorDim(tensor, tensorDim);
            }
        }
    }

    return constraints;
}

NodeStrategyPtr LegacyMMEKeyNodeSolver::sendRequestToMmeBrain(const BundleSolutionConstraints& constraints) const
{
    return NodeSolver::solveNode(m_keyNode, constraints);
}

StrategyContainer LegacyMMEKeyNodeSolver::processMmeBrainResponse(const BundleViewContainerPtr& bundleViews,
                                                                  const NodeStrategyPtr&        nodeStrategy) const
{
    HB_ASSERT_PTR(nodeStrategy);

    auto mmeSolution             = std::make_shared<MmeSolution>();
    mmeSolution->QORs[m_keyNode] = std::make_shared<SolutionParams>();
    StrategyPtr strategy         = std::make_shared<Strategy>(mmeSolution);

    MultiplierPerBVD existingMultipliers;
    for (const auto& slicedOperand : nodeStrategy->getSlicingData().getSlicedOperands())
    {
        for (Dim tensorDim = 0; tensorDim < slicedOperand->originalTensor->getDim(); tensorDim++)
        {
            BundleViewId bvdId    = bundleViews->getBVDForTensorDim(slicedOperand->originalTensor, tensorDim);
            TSize        tileSize = slicedOperand->chunkDimensions[tensorDim];
            TSize granularitySize = bundleViews->getGranularityForTensorDim(slicedOperand->originalTensor, tensorDim);
            bool  isDimSliced     = (tileSize != slicedOperand->originalTensor->getSizeInElements(tensorDim));
            // In case the tensor dim is not sliced, tile size doesn't have to be a multiple of the granularity size.
            HB_ASSERT(!isDimSliced || (tileSize % granularitySize == 0),
                      "Expect the tile size {} to be a multiple of the granularity size {}",
                      tileSize,
                      granularitySize);
            BVDMultiplier multiplier = isDimSliced ? BVDMultiplier(tileSize / granularitySize) : BVDMultiplier();
            LOG_DEBUG(LB_SLICER,
                      "\t Granularity multiplier for tensor {} dim {} (BVD id {}) is set to {}",
                      slicedOperand->originalTensor->getName(),
                      tensorDim,
                      bvdId,
                      (multiplier.isSliced()) ? std::to_string(multiplier.getMultiplier()) : "UNSLICED");
            auto it = existingMultipliers.find(bvdId);
            if (it != existingMultipliers.end())
            {
                HB_ASSERT((!it->second.isSliced() && !multiplier.isSliced()) ||
                              ((it->second.isSliced() == multiplier.isSliced()) &&
                               (it->second.getMultiplier() == multiplier.getMultiplier())),
                          "Expected the same granularity multiplier for all tensor dims in BVD {}",
                          bvdId);
            }
            else
            {
                HB_ASSERT(!multiplier.isSliced() ||
                              (multiplier.getMultiplier() <= bundleViews->getBundleView(bvdId).resolution),
                          "Invalid multiplier for BVD {}",
                          bvdId);
                strategy->setBVDMultiplier(bvdId, multiplier);
                existingMultipliers[bvdId] = multiplier;
            }
        }
    }

    if (nodeStrategy->getSlicingData().getSlicedOperand(m_keyNode->getOutput(0))->finalElementType !=
        m_keyNode->getOutput(0)->getElementType())
    {
        strategy->getMmeSolution()->QORs.at(m_keyNode)->solutionRequirements.requiresCast = true;
    }

    MmeDimController dimController(m_keyNode);
    auto&            solutionCommonDims = strategy->getMmeSolution()->QORs.at(m_keyNode)->solutionRequirements.cdDims;
    for (auto inputIdx : {0, 1})
    {
        for (auto dim : dimController.commonDimsForOperand(inputIdx))
        {
            BundleViewId cdBVD = bundleViews->getBVDForTensorDim(m_keyNode->getInput(inputIdx), dim);
            if (std::find(solutionCommonDims.begin(), solutionCommonDims.end(), cdBVD) == solutionCommonDims.end())
            {
                solutionCommonDims.push_back(cdBVD);
            }
        }
    }

    return StrategyContainer(strategy, m_keyNode, mmeSolution);
}

StrategyContainer LegacyMMEKeyNodeSolver::handleNoStrategy(const BundleViewContainerPtr& bundleViews) const
{
    LOG_WARN(LB_SLICER, "No strategies found for node {}, init all BVDs as unsliced", m_keyNode->getNodeName());

    auto mmeSolution             = std::make_shared<MmeSolution>();
    mmeSolution->QORs[m_keyNode] = std::make_shared<SolutionParams>();
    StrategyPtr strategy         = std::make_shared<Strategy>(mmeSolution);

    const auto& mmeNodeAP = m_keyNode->getNodeAccessPattern();
    HB_ASSERT_PTR(mmeNodeAP);
    for (Dim nodeDim = 0; nodeDim < mmeNodeAP->getNodeResolution().size(); nodeDim++)
    {
        if (bundleViews->isNodeDimMappedToBVD(m_keyNode, nodeDim))
        {
            BundleViewId bvd = bundleViews->getBVDForNodeDim(m_keyNode, nodeDim);
            strategy->setBVDMultiplier(bvd, BVDMultiplier());
        }
    }

    return StrategyContainer(strategy, m_keyNode, mmeSolution);
}

StrategyContainer LegacyMMEKeyNodeSolver::getSlicingStrategies(const BundleViewContainerPtr& bundleViews,
                                                               const StrategyContainer&      existingStrategies)
{
    SET_TEMP_LOG_CONTEXT("LegacyMMEKeyNodeSolver");

    LOG_TRACE(LB_SLICER, "Get slicing strategies for node {}", m_keyNode->getNodeName());

    // Assume a single MME key node in bundle
    HB_ASSERT(HabanaGraph::runsOnMME(m_keyNode), "Expected {} to be a MME node", m_keyNode->getNodeName());
    HB_ASSERT(existingStrategies.strategies.empty(), "Multi MME nodes bundle is currently not supported");

    const BundleSolutionConstraints& constraints = prepareMmeBrainRequest(bundleViews);

    const NodeStrategyPtr& nodeStrategy = sendRequestToMmeBrain(constraints);
    if (!nodeStrategy || !nodeStrategy->getSlicingData().getSlicedOperand(m_keyNode->getInput(0))->resideInSRAM)
    {
        return handleNoStrategy(bundleViews);
    }

    const StrategyContainer& strategies = processMmeBrainResponse(bundleViews, nodeStrategy);

    logStrategies(strategies, bundleViews);

    LOG_TRACE(LB_SLICER, "{} strategies created for node {}", strategies.strategies.size(), m_keyNode->getNodeName());

    return strategies;
}