#include "dma_transpose_node.h"
#include "habana_graph.h"
#include "node.h"
#include "habana_nodes.h"
#include "defs.h"
#include "split_strategies.h"
#include "project_node_rois_to_tensor_rois.h"


// Assumes all strides are equal
RoiShapeType getRoiShapeType(uint32_t dimCount, const TSize* minSizes, const TOffset* roiOffset, const TSize* roiSize)
{
    for (size_t dim = 0; dim < dimCount; dim++)
    {
        if (roiOffset[dim] + roiSize[dim] > minSizes[dim])
        {
            return RoiShapeType::DYNAMIC_ROI;
        }
    }
    return RoiShapeType::FIXED_ROI;
}


static RoiShapeType getRoiShapeTypeForNode(const Node& node, const NodeROI& nodeRoi)
{
    if (!node.isDynamicShape())
    {
        return RoiShapeType::FIXED_ROI;
    }

    // ATTN dynamic base address node will fail the next check
    if (node.isDma() && dynamic_cast<const DMANode&>(node).getDynamicMemoryOpType() == DMA_OP_DYNAMIC_BASE)
    {
        return RoiShapeType::DYNAMIC_ROI;
    }
    for (const auto& inputRoi : nodeRoi.inputRois)
    {
        const auto& input = inputRoi.m_parentTensor;
        auto roiLayout = inputRoi.getLayout();
        auto        intersectResult = getRoiShapeType(input->getDim(),
                                               input->getAllMinimalSizesInElements().data(),
                                               roiLayout.m_baseOffset,
                                               roiLayout.m_size.data());
        if (intersectResult == RoiShapeType::DYNAMIC_ROI)
        {
            return RoiShapeType::DYNAMIC_ROI;
        }
    }
    for (const auto& outputRoi : nodeRoi.outputRois)
    {
        const auto& output = outputRoi.m_parentTensor;
        auto roiLayout = outputRoi.getLayout();
        auto        intersectResult = getRoiShapeType(output->getDim(),
                                               output->getAllMinimalSizesInElements().data(),
                                               roiLayout.m_baseOffset,
                                               roiLayout.m_size.data());
        if (intersectResult == RoiShapeType::DYNAMIC_ROI)
        {
            return RoiShapeType::DYNAMIC_ROI;
        }
    }
    return RoiShapeType::FIXED_ROI;
}

static void correctTransposeRoi(const Node& n, NodeROI& transposeRoi)
{
    projectDmaRoi(transposeRoi, n);
}

static RoiShapeType getRoiShapeTypeWithCorrections(const Node& n, const NodeROI& roi)
{
    if (n.isDma() && static_cast<const DMANode&>(n).isTranspose())
    {
        auto copyRoi = roi;
        correctTransposeRoi(n, copyRoi);
        return getRoiShapeTypeForNode(n, copyRoi);
    }
    return getRoiShapeTypeForNode(n, roi);
}

bool setRoiShapeType(HabanaGraph& g)
{
    for (const pNode& n : g.getExeSortedNodes())
    {
        for (auto& roi : g.getCodeGenerator()->getPhysicalRois(n))
        {
            roi.roiDsdType = getRoiShapeTypeWithCorrections(*n, roi);
        }
    }

    return true;
}
