#include "node_dcore_rois_setter.h"
#include "habana_graph.h"
#include "node.h"
#include "synapse_common_types.h"
#include <memory>
#include "mme_brain_ifc.h"

using namespace gc::layered_brain;
using namespace gc::access_pattern;

bool NodeDcoreROIsSetter::isPerforationSupported(const std::vector<TSize>& dcoreSizes, unsigned dim) const
{
    if (dcoreSizes.empty()) return false;

    if (HabanaGraph::runsOnMME(m_node))  // TODO [SW-171735]: support uneven CD perforation for MME nodes
    {
        MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(m_node);
        HB_ASSERT_PTR(mmeNode);
        if (!mmeNode->getMmeBrainIfc()->isCdDim(dim))
        {
            return true;
        }
        if (std::any_of(dcoreSizes.begin() + 1, dcoreSizes.end(), [&](const TSize& s) {
                return s != dcoreSizes.at(0);
            }))
        {
            LOG_DEBUG(LB_SLICER, "Aborting split - uneven CD perforation is currently not supported for MME nodes");
            return false;
        }
    }
    return true;
}

std::vector<TSize> NodeDcoreROIsSetter::split(TSize size) const
{
    HB_ASSERT(m_numDcores > 0, "Num DCOREs should be > 0");
    std::vector<TSize> dcoreSizes(m_numDcores, size / m_numDcores);
    TSize              remainder = size % m_numDcores;
    HB_ASSERT(remainder < m_numDcores, "Expected remainder ({}) < num DCOREs ({})", remainder, m_numDcores);
    for (auto dcore = 0; dcore < remainder; dcore++)
    {
        dcoreSizes[dcore] += 1;
    }
    return dcoreSizes;
}

std::vector<TSize> NodeDcoreROIsSetter::splitByGranularity(TSize size, TSize granularity) const
{
    std::vector<TSize> dcoreSizes = split(size / granularity);
    HB_ASSERT(dcoreSizes.size() == m_numDcores, "Failed to split to DCORE ROIs node {}", m_node->getNodeName());
    for (auto dcore = 0; dcore < m_numDcores; dcore++)
    {
        dcoreSizes[dcore] *= granularity;
    }
    // The last DCORE gets the remainder (the initial split divides the remainder between the first
    // DCOREs so it's guaranteed that the last DCORE has the minimal amount of work).
    dcoreSizes.back() += (size % granularity);
    HB_ASSERT(std::accumulate(dcoreSizes.begin(), dcoreSizes.end(), 0UL) == size,
              "Failed to split to DCORE ROIs node {}",
              m_node->getNodeName());
    return dcoreSizes;
}

void NodeDcoreROIsSetter::splitToDcoreROIs(unsigned dim, TSize granularity, const std::optional<unsigned>& perforationGroup) const
{
    SET_TEMP_LOG_CONTEXT("NodeDcoreROIsSetter");

    HB_ASSERT(m_node->getNodeAnnotation().sliceROI.has_value(),
              "Missing sliced node ROI for node {}",
              m_node->getNodeName());

    const auto& fullSliceROI = m_node->getNodeAnnotation().sliceROI.value();
    HB_ASSERT(dim < fullSliceROI.geometry.size(),
              "Invalid dim to split node {} ({}), node resolution rank is {}",
              m_node->getNodeName(),
              dim,
              fullSliceROI.geometry.size());
    HB_ASSERT(granularity != 0, "Invalid slicing granularity for node {}", m_node->getNodeName());
    TSize sizeToSplit = fullSliceROI.geometry.at(dim);

    const auto& dcoreSizes = splitByGranularity(sizeToSplit, granularity);
    if (!isPerforationSupported(dcoreSizes, dim))
    {
        return;
    }

    LOG_DEBUG(LB_SLICER,
              "Split node {} to DCORE ROIs on dim {} (granulrity={} size={}) : size per DCORE {}",
              m_node->getNodeName(),
              dim,
              granularity,
              sizeToSplit,
              toString(dcoreSizes.begin(), dcoreSizes.end(), ','));

    auto& dcoreROIs = m_node->getNodeAnnotation().m_dcoreROIs;
    HB_ASSERT(dcoreROIs.empty(), "DCORE ROIs for node {} should be empty before the split", m_node->getNodeName());
    dcoreROIs.resize(m_numDcores);
    TOffset dcoreOffset = 0;
    LOG_DEBUG(LB_SLICER, "{} DCORE ROIs generated for node {}:", m_numDcores, m_node->getNodeName());
    for (auto dcore = 0; dcore < m_numDcores; dcore++)
    {
        std::copy(fullSliceROI.geometry.begin(), fullSliceROI.geometry.end(), dcoreROIs[dcore].size);
        // ROI size may have more dimensions than the index space geometry. The un-used dimension sizes should be set
        // to 1.
        std::fill(dcoreROIs[dcore].size + fullSliceROI.geometry.size(),
                  dcoreROIs[dcore].size + ARRAY_SIZE(dcoreROIs[dcore].size),
                  1);
        std::copy(fullSliceROI.offset.begin(), fullSliceROI.offset.end(), dcoreROIs[dcore].baseOffset);

        dcoreROIs[dcore].size[dim] = dcoreSizes.at(dcore);
        dcoreROIs[dcore].baseOffset[dim] += dcoreOffset;
        dcoreOffset += dcoreSizes.at(dcore);

        LOG_DEBUG(LB_SLICER,
                  "\t DCORE[{}]: ROI size = {} ROI offset {}",
                  dcore,
                  toString(std::begin(dcoreROIs[dcore].size), std::end(dcoreROIs[dcore].size), ','),
                  toString(std::begin(dcoreROIs[dcore].baseOffset), std::end(dcoreROIs[dcore].baseOffset), ','));
    }

    setNodePerforationData(dim, perforationGroup);
}

void NodeDcoreROIsSetter::setDcoreROIs(const std::vector<NodeTile>&   dcoreNodeTiles,
                                       const std::optional<unsigned>& perforationDim,
                                       const std::optional<unsigned>& perforationGroup) const
{
    SET_TEMP_LOG_CONTEXT("NodeDcoreROIsSetter");
    HB_ASSERT(dcoreNodeTiles.size() == m_numDcores, "not enough node tiles provided for {}", m_node->getNodeName());
    auto& dcoreROIs = m_node->getNodeAnnotation().m_dcoreROIs;
    HB_ASSERT(dcoreROIs.empty(), "DCORE ROIs for node {} should be empty before they are set", m_node->getNodeName());
    dcoreROIs.resize(m_numDcores);
    LOG_DEBUG(LB_SLICER, "{} DCORE ROIs generated for node {}:", m_numDcores, m_node->getNodeName());
    for (auto dcore = 0; dcore < m_numDcores; dcore++)
    {
        const auto& dcoreNodeTile = dcoreNodeTiles.at(dcore);

        // init node dcore roi size values for unset dims. offset is 0 by default
        unsigned initVal = dcoreNodeTile.geometry == NodeTile::Geometry(dcoreNodeTile.geometry.size(), 0) ? 0 : 1;
        std::fill(std::begin(dcoreROIs[dcore].size), std::end(dcoreROIs[dcore].size), initVal);
        // copy the node tile sizes and offset to the node dcore roi
        std::copy(dcoreNodeTile.geometry.begin(), dcoreNodeTile.geometry.end(), std::begin(dcoreROIs[dcore].size));
        std::copy(dcoreNodeTile.offset.begin(), dcoreNodeTile.offset.end(), std::begin(dcoreROIs[dcore].baseOffset));
        LOG_DEBUG(LB_SLICER,
                  "\t DCORE[{}]: ROI size = {} ROI offset {}",
                  dcore,
                  toString(std::begin(dcoreROIs[dcore].size), std::end(dcoreROIs[dcore].size), ','),
                  toString(std::begin(dcoreROIs[dcore].baseOffset), std::end(dcoreROIs[dcore].baseOffset), ','));
    }
    setNodePerforationData(perforationDim, perforationGroup);
    validatePerforation();
}

void NodeDcoreROIsSetter::setNodePerforationData(const std::optional<unsigned>& perforationDim,
                                                 const std::optional<unsigned>& perforationGroup) const
{
    m_node->getNodeAnnotation().splitToLogicalROIs = false;  // logical and DCORE split aren't supported together
    if (perforationDim.has_value())
    {
        m_node->getNodeAnnotation().perforationDim = perforationDim;
        if (m_node->getNodeAnnotation().bundleInfo.is_set())
        {
            m_node->getNodeAnnotation().bundleInfo->perforationGroup = perforationGroup;
        }
    }
}

void NodeDcoreROIsSetter::validatePerforation() const
{
    HB_ASSERT(m_node->getNodeAnnotation().sliceROI.has_value(),
              "Missing sliced node ROI for node {}",
              m_node->getNodeName());
    const NodeTile&     fullROI           = m_node->getNodeAnnotation().sliceROI.value();
    const DcoreRoisVec& dcoreROIs         = m_node->getNodeAnnotation().m_dcoreROIs;
    TSize               dcorRoisTotalSize = 0;
    for (const auto& roi : dcoreROIs)
    {
        dcorRoisTotalSize += multiplyElements(roi.size, roi.size + HABANA_DIM_MAX);
    }
    TSize fullRoiSize = multiplyElements(fullROI.geometry);
    HB_ASSERT(dcorRoisTotalSize == fullRoiSize, "Dcore Rois don't add up to full Roi");
}
