
#include "sliced_tensor_generator.h"
#include "utils.h"

using namespace gc::layered_brain;

std::pair<SizeArray, OffsetArray> SlicedTensorGenerator::calcSliceSizeAndOffset(const NodePtr&   origNode,
                                                                                const NodeTile&  slicedNodeTile,
                                                                                const TensorPtr& origTensor) const
{
    const auto& nodeAP = origNode->getNodeAccessPattern();
    HB_ASSERT_PTR(nodeAP);
    const auto& nodeResolution   = nodeAP->getNodeResolution();
    const auto& slicedTensorTile = nodeAP->getTensorTile(origTensor, slicedNodeTile);

    SizeArray   sliceSize = origTensor->getAllSizesInElements();
    OffsetArray sliceOffset {};
    sliceOffset.fill(0);
    for (Dim tensorDim = 0; tensorDim < origTensor->getDim(); tensorDim++)
    {
        const auto nodeDim = nodeAP->getIndexSpaceDim(origTensor, tensorDim);
        if (slicedNodeTile.geometry.at(nodeDim) < nodeResolution.at(nodeDim))  // Tensor dim is sliced
        {
            sliceSize.at(tensorDim)   = slicedTensorTile.geometry.at(tensorDim);
            sliceOffset.at(tensorDim) = slicedTensorTile.offset.at(tensorDim);
            if (sliceOffset.at(tensorDim) < 0)
            {
                HB_ASSERT(sliceSize.at(tensorDim) > std::abs(sliceOffset.at(tensorDim)),
                          "Expected slice size ({}) > slice offset ({}) for tensor dim {}",
                          sliceSize.at(tensorDim),
                          std::abs(sliceOffset.at(tensorDim)),
                          tensorDim);
                sliceSize.at(tensorDim) += sliceOffset.at(tensorDim);
                sliceOffset.at(tensorDim) = 0;
            }
            // Offset must fall inside the dimension unless that dimension size is 0.
            HB_ASSERT((origTensor->getSizeInElements(tensorDim) > sliceOffset.at(tensorDim)) ||
                          (origTensor->getSizeInElements(tensorDim) == 0 && sliceOffset.at(tensorDim) == 0),
                      "Invalid offset ({}) for tensor {} dim {}",
                      sliceOffset.at(tensorDim),
                      origTensor->getName(),
                      tensorDim);
            // Clip the size of the last slice according to tensor bounds.
            sliceSize.at(tensorDim) =
                std::min(origTensor->getSizeInElements(tensorDim) - sliceOffset.at(tensorDim), sliceSize.at(tensorDim));
        }
    }

    return {sliceSize, sliceOffset};
}

std::pair<TensorPtr, OffsetArray> SlicedTensorGenerator::getSlicedTensor(const NodePtr&   origNode,
                                                                         const NodeTile&  slicedNodeTile,
                                                                         const TensorPtr& origTensor,
                                                                         const BVDCoord&  bvdCoord)
{
    const auto& [sliceSize, sliceOffset] = calcSliceSizeAndOffset(origNode, slicedNodeTile, origTensor);

    auto it = m_slicedTensorsDB.find({origTensor, bvdCoord});
    if (it != m_slicedTensorsDB.end())
    {
        const auto& [cachedSlice, cachedSliceOffset] = it->second;
        LOG_DEBUG(LB_SLICER,
                  "Use cached slice {} for original tensor {} : BVD coord [{}]",
                  cachedSlice->getName(),
                  origTensor->getName(),
                  toString(bvdCoord.begin(), bvdCoord.end(), ','));
        HB_ASSERT(cachedSlice->getAllSizesInElements() == sliceSize,
                  "Mismatch in slice size ({}) for tensor {} node {} (cached size {})",
                  toString(sliceSize.begin(), sliceSize.begin() + origTensor->getDim(), ','),
                  origTensor->getName(),
                  origNode->getNodeName(),
                  cachedSlice->getDimSizesStr());
        HB_ASSERT(cachedSliceOffset == sliceOffset,
                  "Mismatch in slice offset ({}) for tensor {} node {} (cached offset {})",
                  toString(sliceOffset.begin(), sliceOffset.begin() + origTensor->getDim(), ','),
                  origTensor->getName(),
                  origNode->getNodeName(),
                  toString(cachedSliceOffset.begin(), cachedSliceOffset.begin() + origTensor->getDim(), ','));
        return it->second;
    }

    TensorPtr slicedTensor = origTensor->clone(false, false, false);
    slicedTensor->reshape(slicedTensor->getDim(), sliceSize.data(), nullptr);
    slicedTensor->setName(fmt::format("{}{}{}/slice_{}",
                                      origTensor->getName(),
                                      m_bundleIdx.has_value() ? "_bundle_" : "",
                                      m_bundleIdx.has_value() ? std::to_string(*m_bundleIdx) : "",
                                      toString(bvdCoord.begin(), bvdCoord.end(), '_')),
                          true);
    slicedTensor->getTensorAnnotation().origBigTensor = origTensor;
    LOG_DEBUG(LB_SLICER,
              "Create sliced tensor {} from original tensor {} : BVD coord [{}], size [{}], offset [{}]",
              slicedTensor->getName(),
              origTensor->getName(),
              toString(bvdCoord.begin(), bvdCoord.end(), ','),
              toString(sliceSize.begin(), sliceSize.begin() + slicedTensor->getDim(), ','),
              toString(sliceOffset.begin(), sliceOffset.begin() + slicedTensor->getDim(), ','));

    m_slicedTensorsDB[{origTensor, bvdCoord}] = {slicedTensor, sliceOffset};
    return {slicedTensor, sliceOffset};
}