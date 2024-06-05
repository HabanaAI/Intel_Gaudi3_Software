#include "dma_inserter.h"

#include "habana_graph.h"

std::string dmaTypeToString(DMA_TYPE dmaType)
{
    switch (dmaType)
    {
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_INVALID)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_UPSTREAM)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_DOWNSTREAM)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_INTERMEDIATES)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_PREFETCH_STATIC_TENSORS)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_PREFETCH_ACTIVATIONS)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_SPILL)
        TRANSLATE_ENUM_TO_STRING(DMA_TYPE_INTERNAL)
    };
    return "ERROR - not handled";
}

void addDmaNodes(HabanaGraph& g, std::vector<DmaInsertionPoint> points, bool isSetup, const std::string& name)
{
    auto namPrefix = name;
    if (!name.empty()) namPrefix += "_";
    for (auto p : points)
    {
        pNode dmaNode =
            pNode(new DMANode(p.t, namPrefix + dmaTypeToString(p.dmaType) + '_' + p.t->getName(), p.dmaType));

        // Graph execution order cache will be invalidated when the node will be added on the next instruction
        NodeAnnotation& nodeAnnotation = dmaNode->getNodeAnnotation();
        nodeAnnotation.memorySpaceInfo = p.t->getTensorAnnotation().memorySpaceInfo;

        if(p.dmaType == DMA_TYPE_DOWNSTREAM)
        {
            std::list<pNode> consumers = g.getTensorConsumers(p.t);
            const NodeAnnotation& consumerAnn = consumers.front()->getNodeAnnotation();
            nodeAnnotation.sliceIndex = consumerAnn.sliceIndex;
            nodeAnnotation.rangeIndex = consumerAnn.rangeIndex;
        }

        if(p.dmaType == DMA_TYPE_UPSTREAM)
        {
            const NodeAnnotation& producerAnn = g.getTensorProducer(p.t)->getNodeAnnotation();
            nodeAnnotation.sliceIndex = producerAnn.sliceIndex;
            nodeAnnotation.rangeIndex = producerAnn.rangeIndex;
        }

        if (isSetup)
        {
            g.addSetupNode(dmaNode);
        }
        else
        {
            GraphEditor::addNode(g, dmaNode);
        }
        std::list<NodeROI>* rois = g.GetNodeROIs(dmaNode);
        NodeROI fullRoi = dmaNode->generateRoi();
        rois->push_back(fullRoi);
    }
}
