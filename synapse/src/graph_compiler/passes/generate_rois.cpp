#include <memory>

#include "compilation_hal_reader.h"
#include "habana_graph.h"
#include "node.h"
#include "habana_pass.h"

void CopyCacheData(NodeROI& roi, const NodePtr& node);

//Generate full ROI for every node
bool generateROIs(HabanaGraph& g)
{
    const NodeVector& nodes      = g.getExeSortedNodes();
    const NodeSet&    setupNodes = g.getSetupNodes();

    for (const NodePtr& n : setupNodes)
    {
        std::list<NodeROI>* rois = g.GetNodeROIs(n);
        NodeROI fullRoi = n->generateRoi();
        rois->push_back(fullRoi);
    }

    for (const NodePtr& n : nodes)
    {
        if (n->isLogicalOperation()) continue;
        std::list<NodeROI>* rois = g.GetNodeROIs(n);
        NodeROI fullRoi = n->generateRoi();
        CopyCacheData(fullRoi, n);
        rois->push_back(fullRoi);
        n->setLogicalRois(*rois);
    }

    return true;
};

void CopyCacheData(NodeROI& roi, const NodePtr& node)
{
    if (CompilationHalReader::isHalReaderSet() && CompilationHalReader::getHalReader()->isCacheSupported())
    {
        HB_ASSERT(node->getNumInputs() == node->getNodeAnnotation().inputsCacheMetaData.size(),
                  "{}: Unexpected number of cache metadata entries ({}) for node with {} inputs",
                  node->getNodeName(),
                  node->getNodeAnnotation().inputsCacheMetaData.size(),
                  node->getNumInputs());

        HB_ASSERT(node->getNumOutputs() == node->getNodeAnnotation().outputsCacheMetaData.size(),
                  "{}: Unexpected number of cache metadata entries ({}) for node with {} outputs",
                  node->getNodeName(),
                  node->getNodeAnnotation().outputsCacheMetaData.size(),
                  node->getNumOutputs());
    }

    roi.dcoreROIs            = node->getNodeAnnotation().m_dcoreROIs;
    roi.inputsCacheMetaData  = node->getNodeAnnotation().inputsCacheMetaData;
    roi.outputsCacheMetaData = node->getNodeAnnotation().outputsCacheMetaData;
}