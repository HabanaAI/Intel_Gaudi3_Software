#include "roi_splitter.h"
#include "gaudi_graph.h"
#include "dma_transpose_helper.h"

NodeROI GaudiROISplitter::createMemsetROI(NodeROI& nodeRoi, unsigned baseOffset, unsigned roiSize) const
{
    NodeROI newRoi(nodeRoi);

    memset(newRoi.baseOffset, 0, sizeof(TOffset) * Tensor::c_tensorMaxNDim);
    std::fill_n(newRoi.size, Tensor::c_tensorMaxDim, 1);
    newRoi.baseOffset[0] = baseOffset;
    newRoi.size[0] = roiSize;

    return newRoi;
}

std::list<NodeROI> GaudiROISplitter::splitDMA(pNode node, HabanaGraph& g) const
{
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(node);

    if (dmaNode->isMemset() && (dmaNode->getOpType() == DMA_OP_TYPE::DMA_OP_COPY))
    {
        return splitMemsetDMA(node, g);
    }
    else
    {
        return ROISplitter::splitDMA(node, g);
    }
}

std::list<NodeROI> GaudiROISplitter::splitMemsetDMA(NodePtr node, HabanaGraph& g) const
{
    std::list<NodeROI> ret;
    TensorPtr tensor =  node->getOutput(0);

    HB_ASSERT(tensor->isDenseLayout(), "DMA node {} expecting dense tensor ({})", node->getNodeName(), tensor->getName());
    std::unique_ptr<CodeGenerator>& codeGenerator      = g.getCodeGenerator();
    NodeROI&       roi                = g.GetNodeROIs(node)->front();
    unsigned       elementSizeInBytes = tensor->getElementSizeInBytes();
    uint64_t       numElem            = tensor->getTotalSizeInBytes() / elementSizeInBytes;
    DMANode*       dmaNode            = reinterpret_cast<DMANode*>(node.get());
    uint64_t       roiSize            = dmaNode->chunkSizeInBytes() * dmaNode->parallelLevel() / elementSizeInBytes;
    unsigned       maxNumChunks       = codeGenerator->getMaxDMAChunks();
    auto           cacheLineSize      = g.getHALReader()->getCacheLineSizeInBytes();
    auto           numElemInCacheLine = cacheLineSize / elementSizeInBytes;

    LOG_DEBUG(ROI_SPLITTER, "Splitting memset DMA node {} to logical ROIs. Original size: [{}, {}, {}, {}]",
        node->getNodeName(), roi.size[0], roi.size[1], roi.size[2], roi.size[3]);

    // memeset is replaced with DMA copy because of HW bug https://jira.habana-labs.com/browse/SIV-23
    // Number of elements and roi size should be aligned to 128 bytes. This is required in order to set
    // src size_0  the DMA descriptor to 128 to support effecient copy
    uint64_t numElemAlignedtoCacheLine = alignSizeDown(numElem,numElemInCacheLine * dmaNode->parallelLevel());
    unsigned numElemReminder           = numElem - numElemAlignedtoCacheLine;

    //Adjust ROI size to be aligned to cache line
    if (numElem / roiSize > maxNumChunks)
    {
        LOG_TRACE(ROI_SPLITTER, "Increasing roiSize so we won't have more than {} rois", maxNumChunks);
        roiSize = div_round_up(numElem * dmaNode->parallelLevel(), maxNumChunks);
    }
    roiSize = alignSizeUp(roiSize,cacheLineSize * dmaNode->parallelLevel());

    for (unsigned elem = 0; elem < numElemAlignedtoCacheLine; elem += roiSize)
    {
        unsigned size = std::min<unsigned>(roiSize, numElemAlignedtoCacheLine - elem);

        auto newRoi = createMemsetROI(roi, elem,size);
        ret.push_back(newRoi);
    }

    if (numElemReminder)
    {
        uint64_t numElemReminderAlignedToCacheLine = alignSizeDown(numElemReminder, numElemInCacheLine);
        unsigned numElemCacheLineReminder          = numElemReminder - numElemReminderAlignedToCacheLine;

        if (numElemReminderAlignedToCacheLine)
        {
            auto newRoi = createMemsetROI(roi, numElemAlignedtoCacheLine , numElemReminderAlignedToCacheLine);
            ret.push_back(newRoi);
        }

        if (numElemCacheLineReminder)
        {
            auto baseOffset = numElemAlignedtoCacheLine + numElemReminderAlignedToCacheLine;
            auto newRoi = createMemsetROI(roi, baseOffset, numElemCacheLineReminder);
            ret.push_back(newRoi);
        }
    }

    LOG_TRACE(ROI_SPLITTER, "Node {} with tensor {} of size {} was split to {} ROIs",
              node->getNodeName(), tensor->getName(), tensor->getTotalSizeInBytes(), ret.size());

    return ret;
}
