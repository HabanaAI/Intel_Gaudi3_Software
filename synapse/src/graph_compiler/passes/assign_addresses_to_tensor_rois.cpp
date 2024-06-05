
#include "node_roi.h"
#include "tensor_roi.h"
#include "habana_nodes.h"
#include "habana_pass.h"
#include "habana_graph.h"

static bool isPrefetchNode(const Node& n)
{
    const auto* dmaNode = dynamic_cast<const DMANode*>(&n);
    if (dmaNode)
    {
        return dmaNode->isPrefetch();
    }
    return false;
}

static bool isTensorInSram(const TensorROI& roi, const Node& n, bool isOutput)
{
    if (isPrefetchNode(n))
    {
        /*
         * In prefetch, both rois use same output tensor, so cannot use it for determining location.
         * However for such a node output roi is always in sram and input roi is always in dram.
         */

        return isOutput;
    }
    return roi.m_parentTensor->tensorAllocatedInSram();
}

static void setRoiAddress(const Tensor& t,
                          const Node&   n,
                          bool          isOutput,
                          TensorROI&    roi)
{
    // the isReduction in the roi layout is used to determine write after write dependencies in the overlap mechanism.
    // Hence, for any reduction operation that is not REDUCTION_SET (including UNORDERED_SET), we want to set it.
    // This way we allow write after write without additional syncs
    auto reductionInfo          = t.getRealReductionInfo(/* check set op*/ true);
    roi.getLayout().isReduction = reductionInfo.isReductionEnabled;

    const bool isInSram         = isTensorInSram(roi, n, isOutput);
    roi.getLayout().inSram      = isInSram;
    roi.getLayout().baseAddress = isInSram ? t.getSramOffset() : t.getDramOffset();
}

bool assignAddressesToTensorROIs(HabanaGraph& g)
{
    for (const std::shared_ptr<Node>& n : g.getExeSortedNodes())
    {
        if (n->isLogicalOperation()) continue;
        for (NodeROI& roi : *g.GetNodeROIs(n))
        {
            for (TensorROI& tRoi : roi.inputRois)
            {
                if (tRoi.m_parentTensor == nullptr || tRoi.m_parentTensor->isShapeTensor()) continue;
                setRoiAddress(*tRoi.m_parentTensor, *n, false, tRoi);
            }
            for (TensorROI& tRoi : roi.outputRois)
            {
                if (tRoi.m_parentTensor == nullptr || tRoi.m_parentTensor->isShapeTensor()) continue;
                setRoiAddress(*tRoi.m_parentTensor, *n, true, tRoi);
            }
        }
    }

    return true;
};
