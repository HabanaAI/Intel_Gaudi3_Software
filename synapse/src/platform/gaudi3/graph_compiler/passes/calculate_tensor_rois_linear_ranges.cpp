#include "calculate_tensor_rois_linear_ranges.h"
#include "gaudi3_graph.h"
#include "gaudi3_types.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "platform/gaudi3/graph_compiler/descriptor_generator.h"
#include "generate_mme_descriptors.h"
#include "mme_desc_gen_utils.h"

// using namespace Gaudi3;

namespace gaudi3
{
void CalculateTensorROIsLinearRanges::calculateMmeLinearRanges(HabanaGraph& g, const NodePtr& node) const
{
    const MMENodePtr& mmeNode    = std::static_pointer_cast<MmeNode>(node);
    NodePtr        nodeShared = g.getNodeSharedPtr(*mmeNode);
    HB_ASSERT_PTR(nodeShared);

    std::list<NodeROI>& nodeRois = *g.GetNodeROIs(node);

    // Currently support only one ROI
    HB_ASSERT(nodeRois.size() == 1, "Support only on ROI");

    NodeROI& origROI = nodeRois.front();
    resizeOrigRoi(origROI);

    Gaudi3Graph*       gaudi3Graph = downcaster<Gaudi3Graph>(&g);
    unsigned           pipeLevel   = 0;
    std::list<NodeROI> newRois;

    for (const auto& activation : gaudi3Graph->getMmeNodeDescriptorGenerator(nodeShared).getMmeActivations())
    {
        ActivationOverlapRoi overlapRoi = {activation.roiX,
                                           activation.roiY,
                                           activation.roiW,
                                           activation.roiO,
                                           activation.numSignals,
                                           activation.operandRoles};

        addRoi(origROI, overlapRoi, newRois, pipeLevel, activation.isMask, *mmeNode);
    }
    nodeRois.swap(newRois);
    node->setPhysicalRois(nodeRois);
}

MmeCommon::EMmeOpType CalculateTensorROIsLinearRanges::getMmeNodeOpType(const MmeNode&(mmeNode)) const
{
    return getOperationTypeCommon(MmeCommon::e_mme_Gaudi3, mmeNode);
}

bool calculateTensorROIsLinearRanges(Gaudi3Graph& g)
{
    CalculateTensorROIsLinearRanges calculator;
    return calculator.apply(g);
}

}  // namespace gaudi3
