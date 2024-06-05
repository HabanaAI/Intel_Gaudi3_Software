#include "sliced_node_generator.h"
#include "compilation_hal_reader.h"
#include "node_dcore_rois_setter.h"
#include "operation_slice.h"
#include "types.h"
#include "brain_conf.h"

using namespace gc::layered_brain;

NodePtr SlicedNodeGenerator::getSlicedNode(const NodePtr& origNode, const BVDCoord& bvdCoord) const
{
    HB_ASSERT_PTR(m_bundleViews);
    HB_ASSERT_PTR(m_strategy);

    NodePtr slicedNode = origNode->getSlice();
    slicedNode->setName(fmt::format("{}{}/{}/op_{}",
                                    m_bundleIdx.has_value() ? "lb_bundle_" : "",
                                    m_bundleIdx.has_value() ? std::to_string(*m_bundleIdx) : "",
                                    origNode->getNodeName(),
                                    toString(bvdCoord.begin(), bvdCoord.end(), '_')));

    generateSlicedNodeROI(origNode, bvdCoord, slicedNode);
    if (m_bundleIdx.has_value())
    {
        slicedNode->getNodeAnnotation().bundleInfo.set(BundleInfo(*m_bundleIdx, BundleType::UNDEFINED, 0));
        slicedNode->getNodeAnnotation().origBigNode = origNode;

        splitToDcoreROIs(origNode, slicedNode);
    }

    LOG_DEBUG(LB_SLICER,
              "Create sliced node {} from original node {} : BVD coord [{}]",
              slicedNode->getNodeName(),
              origNode->getNodeName(),
              toString(bvdCoord.begin(), bvdCoord.end(), ','));

    return slicedNode;
}

void SlicedNodeGenerator::addAuxTensors(const NodePtr& origNode, const NodePtr& slicedNode)
{
    auto mmeSlice = std::dynamic_pointer_cast<MmeNode>(slicedNode);
    if (mmeSlice && m_strategy->getMmeSolution())
    {
        const auto& mmeBrainSolIt = m_strategy->getMmeSolution()->brainSolution.find(origNode);
        if (mmeBrainSolIt != m_strategy->getMmeSolution()->brainSolution.end())
        {
            HB_ASSERT_PTR(mmeBrainSolIt->second);
            m_mmeServices.addAuxTensorToNode(mmeSlice, mmeBrainSolIt->second->strategy);
            m_mmeServices.adjustDcoreRoisForCdParallel(mmeSlice, mmeBrainSolIt->second->strategy);
        }
    }
}

void SlicedNodeGenerator::updateTensorSliceOffset(const NodePtr&     slicedNode,
                                                  const TensorPtr&   origTensor,
                                                  const TensorPtr&   slicedTensor,
                                                  const OffsetArray& slicedTensorOffset)
{
    auto operationSlicePtr = std::dynamic_pointer_cast<OperationSlice>(slicedNode);
    if (operationSlicePtr)
    {
        operationSlicePtr->addTensorSliceOffset(slicedTensor, origTensor, slicedTensorOffset);
    }
}

void SlicedNodeGenerator::generateSlicedNodeROI(const NodePtr&  origNode,
                                                const BVDCoord& bvdCoord,
                                                const NodePtr&  slicedNode) const
{
    HB_ASSERT_PTR(origNode->getNodeAccessPattern());
    const auto& fullTile = origNode->getNodeAccessPattern()->getNodeResolution();
    NodeTile    sliceROI(fullTile);

    for (Dim nodeDim = 0; nodeDim < fullTile.size(); nodeDim++)
    {
        if (m_bundleViews->isNodeDimMappedToBVD(origNode, nodeDim))
        {
            BundleViewId bvd = m_bundleViews->getBVDForNodeDim(origNode, nodeDim);
            HB_ASSERT(bvd < bvdCoord.size(),
                      "Invalid BVD coord for node {} dim {} (BVD {})",
                      origNode->getNodeName(),
                      nodeDim,
                      bvd);
            const auto& multiplier = m_strategy->getBVDMultiplier(bvd);
            if (multiplier.isSliced())
            {
                TSize slicedDimSize =
                    multiplier.getMultiplier() * m_bundleViews->getGranularityForNodeDim(origNode, nodeDim);
                sliceROI.offset[nodeDim] = slicedDimSize * bvdCoord.at(bvd);

                // Offset must fall inside the dimension unless that dimension size is 0.
                HB_ASSERT((fullTile.at(nodeDim) > sliceROI.offset.at(nodeDim)) ||
                              (fullTile.at(nodeDim) == 0 && sliceROI.offset.at(nodeDim) == 0),
                          "Invalid offset ({}) for node {} dim {} BVD {} BVD coord {}",
                          sliceROI.offset.at(nodeDim),
                          origNode->getNodeName(),
                          nodeDim,
                          bvd,
                          bvdCoord.at(bvd));

                // Clip the size of the last slice according to node resolution bounds.
                sliceROI.geometry[nodeDim] =
                    std::min(fullTile.at(nodeDim) - sliceROI.offset.at(nodeDim), slicedDimSize);
            }
            else
            {
                HB_ASSERT(bvdCoord[bvd] == 0,
                          "Node {} dim {} (BVD {}) is not sliced - expected zero coord",
                          origNode->getNodeName(),
                          nodeDim,
                          bvd);
            }
        }
    }

    slicedNode->getNodeAnnotation().sliceROI = sliceROI;

    LOG_DEBUG(LB_SLICER,
              "ROI for sliced node {}: size = {} offset = {} (full size = {})",
              slicedNode->getNodeName(),
              toString(sliceROI.geometry, ','),
              toString(sliceROI.offset, ','),
              toString(fullTile, ','));
}

void SlicedNodeGenerator::splitToDcoreROIs(const NodePtr& origNode, const NodePtr& slicedNode) const
{
    if (!GCFG_ENABLE_LAYERED_BRAIN_PERFORATION.value()) return;
    if (slicedNode->isLogicalOperation()) return;

    HB_ASSERT(slicedNode->getNodeAnnotation().sliceROI.has_value(),
              "Missing sliced node ROI for node {}",
              slicedNode->getNodeName());

    auto perforationBVD = m_strategy->getPerforationBVDForNode(origNode);
    if (!perforationBVD.has_value())
    {
        return;
    }

    const auto& nodeDimsInBVD = m_bundleViews->getNodeDimsInBVD(perforationBVD.value(), origNode);
    HB_ASSERT(nodeDimsInBVD.size() == 1,
              "Expected a single node dim in perforation BVD ({}) for node {}",
              perforationBVD.value(),
              origNode->getNodeName());

    unsigned perforationNodeDim = nodeDimsInBVD.front();
    LOG_DEBUG(LB_SLICER,
              "\t Split sliced node {} to DCORE ROIs on dim {} (BVD {})",
              slicedNode->getNodeName(),
              perforationNodeDim,
              perforationBVD.value());

    NodeDcoreROIsSetter(slicedNode, CompilationHalReader::getHalReader()->getNumDcores())
        .splitToDcoreROIs(perforationNodeDim,
                          m_bundleViews->getGranularityForNodeDim(origNode, perforationNodeDim),
                          *perforationBVD);
}