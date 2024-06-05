#include "sync_utils.h"
#include "habana_graph.h"
#include "include/sync/overlap.h"
#include "tensor_roi.h"
#include "node_roi.h"
#include "graph_compiler/habana_nodes/habana_nodes.h"

void generateOverlapRois(TensorROIVector& tensorRois, std::list<OverlapRoi>& roiList)
{
    for (TensorROI& tRoi : tensorRois)
    {
        tRoi.m_overlapRoi.isSram      = tRoi.m_layout.inSram;
        tRoi.m_overlapRoi.isReduction = tRoi.m_layout.isReduction;
        roiList.push_back(tRoi.m_overlapRoi);
    }
}

bool isNodeHandlingInternalDependencies(const NodePtr& node)
{
    return dynamic_cast<TPCNode*>(node.get()) != nullptr;
}

bool canSignal(const NodeROI& roi)
{
    if (roi.numSignals == 0) return false;

    bool ret = true;

    if (roi.mmePartial.type == MMEPartial::PARTIAL_FILTER || roi.mmePartial.type == MMEPartial::PARTIAL_IFM)
    {
        ret = roi.mmePartial.canSignal;
    }

    return ret;
}

bool shouldBlockOnControlEdges(const NodePtr& node, const HabanaGraph& g)
{
    // sync objects added only if control edge are "Hard".
    if (!GCFG_MAKE_CTRL_DEP_SOFT.value()) return true;
    return false;
}
