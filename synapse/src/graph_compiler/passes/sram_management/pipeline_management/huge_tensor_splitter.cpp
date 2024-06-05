#include "huge_tensor_splitter.h"
#include "access_pattern.h"
#include "bundle.h"
#include "bundle_plane_graph.h"
#include "bundle_slicer.h"
#include "slicer/bundle_views_collector.h"
#include "compilation_hal_reader.h"
#include "log_manager.h"
#include "mme_node.h"
#include "node.h"
#include "node_projector.h"
#include "pipeline_bundlizer.h"
#include "habana_graph.h"
#include "slice_mapping.h"
#include "synapse_common_types.h"
#include "types.h"
#include "solution_generator.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_set>
#include "slicer/sliced_node_graph_generator.h"

using namespace gc::layered_brain;
MmeHugeTensorSplitter::MmeHugeTensorSplitter(const NodePtr& nodeToSplit)
: m_nodeToSplit(nodeToSplit), m_dimController(nodeToSplit) {};

BundleViewContainerPtr MmeHugeTensorSplitter::createBundleViews() const
{
    TileSizePerTensor granularityPerTensor;
    TileSizePerNode   granularityPerNode;
    granularityPerNode[m_nodeToSplit] =
        NodeTile::Geometry(m_nodeToSplit->getNodeAccessPattern()->getNodeResolution().size(), 1);
    for (const auto& tensor : m_nodeToSplit->getOperands())
    {
        if (tensor)
        {
            granularityPerTensor[tensor] = m_nodeToSplit->getNodeAccessPattern()->getTensorGranularity(tensor).geometry;
        }
    }
    BundleViewsCollector bundleViewsCollector({m_nodeToSplit});
    return bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
}

gc::layered_brain::StrategyPtr MmeHugeTensorSplitter::getBvdNodeStrategy(const BundleViewContainerPtr& bundleViews,
                                                                         const NodeTile&               nodeTile) const
{
    StrategyPtr retStrategy   = std::make_shared<Strategy>();
    const auto& accessPattern = m_nodeToSplit->getNodeAccessPattern();
    auto        nodeDimsNr    = accessPattern->getNodeResolution().size();

    for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
    {
        if (bundleViews->isNodeDimMappedToBVD(m_nodeToSplit, nodeDim))
        {
            BundleViewId bvdId           = bundleViews->getBVDForNodeDim(m_nodeToSplit, nodeDim);
            auto         tileSize        = nodeTile.geometry[nodeDim];
            auto         granularitySize = bundleViews->getGranularityForNodeDim(m_nodeToSplit, nodeDim);
            bool         isDimSliced     = (tileSize != accessPattern->getNodeResolution()[nodeDim]);
            // In case the node dim is not sliced, tile size doesn't have to be a multiple of the granularity size.
            HB_ASSERT(!isDimSliced || (tileSize % granularitySize == 0),
                      "Expect the tile size {} to be a multiple of the granularity size {}",
                      tileSize,
                      granularitySize);
            BVDMultiplier multiplier = isDimSliced ? BVDMultiplier(tileSize / granularitySize) : BVDMultiplier();
            LOG_DEBUG(HUGE_TENSOR_SLICE,
                      "Granularity multiplier for node {} (BVD id {}) is set to {}",
                      m_nodeToSplit->getNodeName(),
                      bvdId,
                      (multiplier.isSliced()) ? std::to_string(multiplier.getMultiplier()) : "UNSLICED");
            HB_ASSERT(!multiplier.isSliced() ||
                          (multiplier.getMultiplier() <= bundleViews->getBundleView(bvdId).resolution),
                      "Invalid multiplier for BVD {}",
                      bvdId);
            retStrategy->setBVDMultiplier(bvdId, multiplier);
        }
    }
    return {retStrategy};
}

NodeTile MmeHugeTensorSplitter::getNodeTileFromStrategy(const SlicingStrategyPtr& strategy) const
{
    NodeAccessPattern::TilePerTensor tensorTiles;
    for (auto slicedOp : strategy->getSlicingData().getSlicedOperands())
    {
        TensorTile tensorTile(slicedOp->originalTensor->getDim(),
                              slicedOp->chunkDimensions,
                              TensorTile::Offset(slicedOp->originalTensor->getDim(), 0));
        tensorTiles.emplace(slicedOp->originalTensor, tensorTile);
    }
    return m_nodeToSplit->getNodeAccessPattern()->getLcmNodeTile(tensorTiles);
}

/* This function is responsible for handling sliced nodes with bias input in case they have reduction consumer.
For each reduction node, it leaves only one producer with bias input by removing the bias from the other
producers.
*/
void MmeHugeTensorSplitter::handleBiasedNodesWithReduction(NodeSet& slicedNodes) const
{
    std::unordered_set<TensorPtr> reductionNodesProducersSet =
        {};  // set to hold the reduction producers that may be handled.
    for (const NodePtr& node : slicedNodes)
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            TensorVector reductionInputs = node->getInputs();
            // For each reduction node, we may need to handle all its input producers except for one.
            reductionNodesProducersSet.insert(reductionInputs.begin() + 1, reductionInputs.end());
        }
    }

    if (!reductionNodesProducersSet.empty())
    {
        LOG_DEBUG(HUGE_TENSOR_SLICE,
                  "Handling slices with bias input and reduction consumer for original huge node {}",
                  m_nodeToSplit->getNodeName());
        for (const NodePtr& node : slicedNodes)
        {
            if (HabanaGraph::runsOnMME(node))
            {
                MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
                HB_ASSERT_PTR(mmeNode);
                // If the current mmeNode has a bias input and it is a reduction producer that needs to be handled,
                // remove its bias input.
                if (mmeNode->hasBias() &&
                    reductionNodesProducersSet.find(mmeNode->getOutput(0)) != reductionNodesProducersSet.end())
                {
                    mmeNode->removeInput(mmeNode->getInput(TENSOR_BIAS));
                    LOG_DEBUG(HUGE_TENSOR_SLICE, "Removed bias input of slice {}", mmeNode->getNodeName());
                }
            }
        }
    }
}

NodeList MmeHugeTensorSplitter::sliceHugeNode(const NodeTile& hugeNodeTile) const
{
    BundleViewContainerPtr   bvdContainer    = createBundleViews();
    auto                     bvdNodeStrategy = getBvdNodeStrategy(bvdContainer, hugeNodeTile);
    SlicedNodeGraphGenerator slicedGraphGenerator(m_nodeToSplit, bvdContainer, bvdNodeStrategy, {m_nodeToSplit});
    auto                     retSlicedNodes = slicedGraphGenerator.createSlicedNode();
    MMENodePtr               originalMmeNode = std::dynamic_pointer_cast<MmeNode>(m_nodeToSplit);
    HB_ASSERT_PTR(originalMmeNode);
    if (originalMmeNode->hasBias())
    {
        handleBiasedNodesWithReduction(retSlicedNodes);
    }
    return NodeList(retSlicedNodes.begin(), retSlicedNodes.end());
}

NodeList MmeHugeTensorSplitter::splitNodeWithHugeTensor() const
{
    // Solver
    SlicingStrategyPtr nodeStrategy = createInitialStrategy();
    bool               isFitHw      = this->sliceNodeForFunctionality(nodeStrategy);
    HB_ASSERT(isFitHw, "not able to fit node {} to HW restrictions", m_nodeToSplit->getNodeName());

    auto hugeNodeTile = getNodeTileFromStrategy(nodeStrategy);
    // Slicer
    return sliceHugeNode(hugeNodeTile);
}

bool MmeHugeTensorSplitter::isSliceSizeFitsHw(const pSlicedOperand& slicedOperand) const
{
    return SlicedOperandUtils::getSliceSizeInBytes(slicedOperand) < HW_DENSE_TENSOR_LIMIT;
}

bool MmeHugeTensorSplitter::doesMmeNodeRequireSlicing() const
{
    if (!m_nodeToSplit) return false;
    for (const TensorVector* tensors : {&m_nodeToSplit->getInputs(), &m_nodeToSplit->getOutputs()})
    {
        for (const auto& tensor : *tensors)
        {
            if (!tensor) continue;
            if (tensor->getDenseSizeInBytes() > HW_DENSE_TENSOR_LIMIT) return true;
        }
    }
    return false;
}

SlicingStrategyPtr MmeHugeTensorSplitter::createInitialStrategy() const
{
    SlicingStrategyPtr strategy = SlicingStrategy::createStrategy(*CompilationHalReader::getHalReader(), m_nodeToSplit);
    StrategySlicingData data = strategy->getSlicingData();
    strategy->setDoubleBuffer(false);
    strategy->setOutputIsInSRAM(false);

    int inputIdx = 0;
    for (auto input : strategy->getSlicingData().bundleTensors)
    {
        strategy->setInputIsInSRAM(inputIdx, false);
        inputIdx += 1;
    }

    DimVector nodeDimVector;
    for (auto i = 0; i < strategy->getSlicingData().masterOperand->originalTensor->getDim(); ++i)
    {
        nodeDimVector.push_back(i);
    }
    // set traversalPattern to include all output dims
    strategy->getSlicingData().traversalPattern = nodeDimVector;

    return strategy;
}

bool MmeHugeTensorSplitter::doesNodeSupportSlicing() const
{
    auto mmeNode = std::dynamic_pointer_cast<MmeNode>(m_nodeToSplit);
    HB_ASSERT_PTR(mmeNode);
    // [SW-141874] In such a case actually the common dimension and the non common dimention are the same and need to be
    // sliced the same, also need to support mme nodes to be sliced on common dimention
    return m_nodeToSplit->getInput(0) != m_nodeToSplit->getInput(1);
}

BgemmHugeTensorSplitter::BgemmHugeTensorSplitter(const NodePtr& m_nodeToSplit)
: GemmHugeTensorSplitter(m_nodeToSplit) {};

bool MmeHugeTensorSplitter::sliceNodeForFunctionality(SlicingStrategyPtr& strategy) const
{
    return false;
}

bool BgemmHugeTensorSplitter::doesNodeSupportSlicing() const
{
    if (m_nodeToSplit->getNodeType() != Node::TYPE_BATCH_GEMM &&
        m_nodeToSplit->getNodeType() != Node::TYPE_MASKED_BATCH_GEMM)
        return false;
    return MmeHugeTensorSplitter::doesNodeSupportSlicing();
}

GemmHugeTensorSplitter::GemmHugeTensorSplitter(const NodePtr& m_nodeToSplit) : MmeHugeTensorSplitter(m_nodeToSplit) {};

bool GemmHugeTensorSplitter::sliceNodeForFunctionality(SlicingStrategyPtr& strategy) const
{
    pSlicedOperand slicedInputOperandA = strategy->getSlicingData().getSlicedOperand(m_nodeToSplit->getInput(0));
    pSlicedOperand slicedInputOperandB = strategy->getSlicingData().getSlicedOperand(m_nodeToSplit->getInput(1));
    pSlicedOperand slicedOutputOperand = strategy->getSlicingData().masterOperand;
    Dim            batchDim            = m_dimController.batchDim().back();

    for (auto inputOperand : {slicedInputOperandA, slicedInputOperandB, slicedOutputOperand})
    {
        if (!inputOperand) continue;
        if (isSliceSizeFitsHw(inputOperand)) continue;
        for (auto iter = strategy->getSlicingData().traversalPattern.rbegin();
             iter != strategy->getSlicingData().traversalPattern.rend();
             ++iter)
        {
            Dim   dimToSlice    = *iter;
            // slice huge node on common dim is not supported on gaudi since need to modify reduction output to be in
            // SRAM
            if (((inputOperand == slicedInputOperandA && m_dimController.isCommonDimOperandA(dimToSlice)) ||
                 (inputOperand == slicedInputOperandB && m_dimController.isCommonDimOperandB(dimToSlice))) &&
                CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi)
                continue;
            TSize slicingFactor =
                div_round_up(SlicedOperandUtils::getSliceSizeInBytes(inputOperand), HW_DENSE_TENSOR_LIMIT);
            auto  elementsBeforeSlicing               = inputOperand->chunkDimensions[dimToSlice];
            TSize elementsAfterSlicing                = div_round_up(elementsBeforeSlicing, slicingFactor);
            inputOperand->chunkDimensions[dimToSlice] = elementsAfterSlicing;
            projectSingleDimSlice(strategy, inputOperand);

            if (isSliceSizeFitsHw(inputOperand)) break;
        }
        if (!isSliceSizeFitsHw(inputOperand)) return false;
    }
    return true;
}

void GemmHugeTensorSplitter::projectSingleDimSlice(SlicingStrategyPtr&   strategy,
                                                   const pSlicedOperand& slicedOperand) const
{
    pSlicedOperand                     slicedInputOperand = slicedOperand;
    AccessPatternNodeSolutionProjector projector {m_nodeToSplit};

    NodeStrategyPtr projectedStrategy =
        projector.getNodeStrategy({slicedInputOperand}, slicedInputOperand->originalTensor);

    for (auto& t : m_nodeToSplit->getOperands())
    {
        // Avoid projecting on the same operand in case both inputs are the same tensor
        if (!t || t == slicedInputOperand->originalTensor) continue;

        pSlicedOperand operand = strategy->getSlicingData().getSlicedOperand(t);
        HB_ASSERT_PTR(operand);
        pSlicedOperand projectedOperand = projectedStrategy->getSlicingData().getSlicedOperand(t);
        HB_ASSERT_PTR(projectedOperand);
        for (auto dim = 0; dim < projectedOperand->originalTensor->getDim(); dim++)
        {
            operand->chunkDimensions[dim] =
                std::min(operand->chunkDimensions[dim], projectedOperand->chunkDimensions[dim]);
        }
    }
}

bool GemmHugeTensorSplitter::doesNodeSupportSlicing() const
{
    if (m_nodeToSplit->getNodeType() != Node::TYPE_GEMM) return false;
    return MmeHugeTensorSplitter::doesNodeSupportSlicing();
}