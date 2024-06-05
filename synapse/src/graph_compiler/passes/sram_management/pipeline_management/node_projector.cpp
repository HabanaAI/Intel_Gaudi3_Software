#include "node_projector.h"
#include "access_pattern.h"
#include "compilation_hal_reader.h"
#include <algorithm>

NodeSolutionProjector::SlicedOperands
NodeSolutionProjector::findSlicedOperandsInBundleStrategy(const BundleStrategyPtr& bundleStrategy) const
{
    SlicedOperands ret {};

    std::unordered_set<pSlicedOperand> slicedOperandSet {};
    const auto&    bundleSlicedData = bundleStrategy->getSlicingData();
    for (const auto& tensor : m_node->getOperands())
    {
        if (!tensor) continue;
        if (tensor->isAuxTensor()) continue;  // Aux tensors may be shared, but they are not sliced.

        auto slicedTensor = bundleSlicedData.getSlicedOperand(tensor);
        if (slicedTensor)
        {
            // Only a single sliced operand need be maintained, especially for operands "used"
            // more than once by the projected node
            const auto isUniqueTensorSlice = slicedOperandSet.insert(slicedTensor).second;
            if (isUniqueTensorSlice)
            {
                LOG_DEBUG(SRAM_SLICE, "{}: Found first sliced operand for tensor {}", HLLOG_FUNC, tensor->getName());
                ret.push_back(slicedTensor);
            }
            else
            {
                LOG_DEBUG(SRAM_SLICE, "{}: Found another sliced operand for tensor {}", HLLOG_FUNC, tensor->getName());
            }
        }
    }

    HB_ASSERT(!ret.empty(),
              "Something went wrong - trying to project a node that has no sliced operand in the bundle strategy");
    return ret;
}

AccessPatternNodeSolutionProjector::AccessPatternNodeSolutionProjector(const NodePtr& node)
: NodeSolutionProjector(node)
{
    HB_ASSERT_PTR(node->getNodeAccessPattern());
}

NodeStrategyPtr AccessPatternNodeSolutionProjector::getNodeStrategy(const SlicedOperands& slicedOperands,
                                                                    const TensorPtr&      sliceDefiningTensor) const
{
    const NodeTile& nodeTile     = getNodeTileFromSlicedOperands(slicedOperands, sliceDefiningTensor);
    NodeStrategyPtr nodeStrategy = createStrategyFromNodeTile(nodeTile, slicedOperands);
    return nodeStrategy;
}

NodeStrategyPtr AccessPatternNodeSolutionProjector::getNodeStrategy(const BundleStrategyPtr& bundleStrategy,
                                                                    const TensorPtr&         sliceDefiningTensor) const
{
    const auto& slicedOperands = findSlicedOperandsInBundleStrategy(bundleStrategy);
    return getNodeStrategy(slicedOperands, sliceDefiningTensor);
}

NodeTile AccessPatternNodeSolutionProjector::getNodeTileFromSlicedOperands(const SlicedOperands& slicedOperands,
                                                                           const TensorPtr& sliceDefiningTensor) const
{
    HB_ASSERT_PTR(sliceDefiningTensor);
    HB_ASSERT(!slicedOperands.empty(),
              "Unexpectedly empty slicedOperands container, sliceDefiningTensor={}",
              sliceDefiningTensor->getName());

    const auto it =
        std::find_if(slicedOperands.begin(), slicedOperands.end(), [&sliceDefiningTensor](const auto& slicedOperand) {
            return slicedOperand && slicedOperand->originalTensor == sliceDefiningTensor;
        });
    HB_ASSERT(it != slicedOperands.end(),
              "slicedOperand corresponding to sliceDefiningTensor {} not found",
              sliceDefiningTensor->getName());
    const pSlicedOperand slicedOp(*it);
    LOG_DEBUG(SRAM_SLICE, "Found sliced operand matching sliceDefiningTensor {}", sliceDefiningTensor->getName());
    TensorTile tensorTile(sliceDefiningTensor->getDim(),
                          slicedOp->chunkDimensions,
                          TensorTile::Offset(sliceDefiningTensor->getDim(), 0));

    return m_node->getNodeAccessPattern()->getNodeTile(sliceDefiningTensor, tensorTile);
}

NodeStrategyPtr
AccessPatternNodeSolutionProjector::createStrategyFromNodeTile(const NodeTile&       nodeTile,
                                                               const SlicedOperands& slicedOperands) const
{
    SlicingStrategyPtr   nodeStrategy = SlicingStrategy::createStrategy(*CompilationHalReader::getHalReader(), m_node);
    StrategySlicingData& nodeSlicingData = nodeStrategy->getSlicingData();

    NodeAccessPatternPtr nodeAp = m_node->getNodeAccessPattern();

    // A tensor may not be fully covered by the node resolution (some tensor elements may not be mapped to any node
    // tile). In this case, mapping the nodeTile to a tensor tile may produce slicing in dimensions that are not really
    // sliced, simply not fully used.
    // For example, down-sampling a 100x100 sample with a pooling window of 1x1 and stride of 2x2, means that the last
    // row and the last column are not read and so they may not be mapped to any node tile. In this case, even if the
    // node tile covers the entire node resolution, the mapping would return a partial tensor tile which in this context
    // would look like slicing.
    // To mitigate this, the nodeTile is only mapped to dimensions that are known to be sliced according to the sliced
    // dims mapping provided by the access pattern.
    SlicedDimsProjector slicedDimsProjector(nodeAp, slicedOperands);

    for (const TensorPtr& tensor : m_node->getOperands())
    {
        if (!tensor) continue;
        pSlicedOperand slicedTensor = nodeSlicingData.getSlicedOperand(tensor);
        HB_ASSERT_PTR(slicedTensor);
        projectSlicingOnTensor(slicedTensor, nodeAp, nodeTile, slicedDimsProjector);
    }
    return nodeStrategy;
}

void AccessPatternNodeSolutionProjector::projectSlicingOnTensor(pSlicedOperand              tensorSlicedOperand,
                                                                const NodeAccessPatternPtr& nodeAP,
                                                                const NodeTile&             nodeTile,
                                                                const SlicedDimsProjector&  slicedDimsProjector) const
{
    const TensorPtr& tensor        = tensorSlicedOperand->originalTensor;
    const TensorTile&       tensorTile    = nodeAP->getTensorTile(tensor, nodeTile);
    const IntersectionTile& tensorOverlap = nodeAP->getTensorOverlap(tensor);

    HB_ASSERT(tensorTile.geometry.size() <= tensorSlicedOperand->chunkDimensions.size(),
              "Tensor {} tile rank exceeds sliced operand chunk dimensions. Tile rank: {}, chunk dimensions size: {}",
              tensor->getName(),
              tensorTile.geometry.size(),
              tensorSlicedOperand->chunkDimensions.size());

    for (Dim slicedDim : slicedDimsProjector.getSlicedDims(tensor))
    {
        tensorSlicedOperand->chunkDimensions[slicedDim]      = tensorTile.geometry[slicedDim];
        tensorSlicedOperand->overlapElementsCount[slicedDim] = tensorOverlap.geometry[slicedDim];
    }
}

// Returns the node strategy according to the bundle strategy
// This code is intended only for Resnet shared input bundles of Dedx as master, and dedw as
// slave (shared operand is dY).
NodeStrategyPtr SharedMMENodeSolutionProjector::getNodeStrategy(const BundleStrategyPtr& bundleStrategy,
                                                                const TensorPtr&         sliceDefiningTensor) const
{
    LOG_DEBUG(SRAM_SLICE, "SharedMMENodeSolutionProjector: projecting slicing to slave node {}", m_node->getNodeName());
    StrategySlicingData& bundleSlicingData   = bundleStrategy->getSlicingData();
    TensorPtr            sharedInputTensor   = m_node->getInput(0);
    pSlicedOperand       masterSharedOperand = bundleSlicingData.getSlicedOperand(sharedInputTensor);

    // Assert all assumptions:
    auto convNode = std::static_pointer_cast<ConvBaseNode>(m_node);
    HB_ASSERT(m_node->getNodeType() == Node::TYPE_DEDW, "Only DEDW stitching is supported");
    HB_ASSERT(sharedInputTensor == convNode->getYOperand(), "Input at index 0 is expected to be Y operand (dY)");
    HB_ASSERT_PTR(masterSharedOperand);

    // Create strategy
    SlicingStrategyPtr   nodeStrategy = SlicingStrategy::createStrategy(*CompilationHalReader::getHalReader(), m_node);
    StrategySlicingData& nodeSlicingData            = nodeStrategy->getSlicingData();
    // Reduction is performed in high precision data type.
    nodeSlicingData.masterOperand->finalElementType = SlicedOperandUtils::getTypeForPartials(nodeSlicingData.masterOperand->finalElementType);
    // Project slicing to dedw non shared operand
    const DimVector& sharedOpSlicedDims = SlicedOperandUtils::getSlicedDims(masterSharedOperand);
    // if dedx is sliced trivially, no slicing is needed for dedw
    if (!sharedOpSlicedDims.empty())
    {
        HB_ASSERT(sharedOpSlicedDims.size() == 1, "Only support slicing single dimension");
        HB_ASSERT((!convNode->is3DConvolution() && sharedOpSlicedDims.front() == DIM_B) ||
                      (convNode->is3DConvolution() && sharedOpSlicedDims.front() == DIM_B_FOR_5D_TENSOR),
                  "expecting dY to be sliced on batch");
        uint32_t        slicedDim                         = sharedOpSlicedDims.front();
        pSlicedOperand& slaveNonSharedOperand             = nodeSlicingData.bundleTensors[1];
        slaveNonSharedOperand->chunkDimensions[slicedDim] = masterSharedOperand->chunkDimensions[slicedDim];
        LOG_DEBUG(SRAM_SLICE,
                  "Slicing slave operand on common dim: operandB on dim: {}, to chunk size  {}",
                  slicedDim,
                  slaveNonSharedOperand->chunkDimensions[slicedDim]);
    }
    return nodeStrategy;
}