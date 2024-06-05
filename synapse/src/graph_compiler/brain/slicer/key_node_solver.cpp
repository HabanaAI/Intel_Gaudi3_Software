#include "key_node_solver.h"
#include "slicing_brain.h"

using namespace gc::layered_brain;

void KeyNodeSolver::logStrategies(const StrategyContainer& strategies, const BundleViewContainerPtr& bundleViews) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(LB_SLICER)) return;

    for (const auto& strategy : strategies.strategies)
    {
        strategy->log();
        for (const auto& node : strategies.nodes)
        {
            LOG_DEBUG(LB_SLICER, "\t Sliced operands for node {}: ", node->getNodeName());
            for (const auto& tensor : node->getOperands())
            {
                if (!tensor) continue;
                const auto& fullSize  = tensor->getAllSizesInElements();
                SizeArray   sliceSize = tensor->getAllSizesInElements();
                for (uint32_t tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
                {
                    BundleViewId bvdId      = bundleViews->getBVDForTensorDim(tensor, tensorDim);
                    const auto&  multiplier = strategy->getBVDMultiplier(bvdId);
                    if (multiplier.isSliced())  // tensorDim is sliced
                    {
                        TSize slicedDimSize =
                            bundleViews->getGranularityForTensorDim(tensor, tensorDim) * multiplier.getMultiplier();
                        HB_ASSERT(slicedDimSize <= fullSize[tensorDim],
                                  "Invalid slice size ({}) for tensor {} dim {} (full dim size {})",
                                  slicedDimSize,
                                  tensor->getName(),
                                  tensorDim,
                                  fullSize[tensorDim]);
                        sliceSize[tensorDim] = slicedDimSize;
                    }
                }
                LOG_DEBUG(LB_SLICER,
                          "\t\t Tensor {}: Full size: [{}] Slice size: [{}]",
                          tensor->getName(),
                          toString(fullSize.begin(), fullSize.begin() + tensor->getDim(), 'x'),
                          toString(sliceSize.begin(), sliceSize.begin() + tensor->getDim(), 'x'));
            }
        }
    }
}