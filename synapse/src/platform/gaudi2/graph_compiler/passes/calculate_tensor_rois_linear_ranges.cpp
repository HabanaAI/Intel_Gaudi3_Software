#include "calculate_tensor_rois_linear_ranges.h"
#include "gaudi2_graph.h"
#include "gaudi2_code_generator.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "mme_desc_gen_utils.h"

using namespace Gaudi2;

namespace gaudi2
{

void CalculateTensorROIsLinearRanges::calculateMmeLinearRanges(HabanaGraph& g, const NodePtr& node) const
{
    const MmeNode &mmeNode = *static_cast<MmeNode*>(node.get());
    NodePtr nodeShared = g.getNodeSharedPtr(mmeNode);
    HB_ASSERT_PTR(nodeShared);

    std::list<NodeROI>& nodeRois = *g.GetNodeROIs(node);

    // Currently support only one ROI
    HB_ASSERT(nodeRois.size() == 1, "Only one ROI is supported");

    NodeROI& origROI = nodeRois.front();
    resizeOrigRoi(origROI);

    Gaudi2CodeGenerator* gaudi2CodeGenerator = downcaster<Gaudi2CodeGenerator>(g.getCodeGenerator().get());
    unsigned pipeLevel = 0;
    std::list<NodeROI> newRois;

    for (const auto& activation : gaudi2CodeGenerator->getMmeNodeDescriptorGenerator(nodeShared).getMmeActivations())
    {
        ActivationOverlapRoi overlapRoi = {activation.roiX,
                                           activation.roiY,
                                           activation.roiW,
                                           activation.roiO,
                                           activation.numSignals,
                                           activation.operandRoles};

        addRoi(origROI, overlapRoi, newRois, pipeLevel, activation.isMask, mmeNode);
    }
    nodeRois.swap(newRois);

    std::list<NodeROI>::iterator newRoi = nodeRois.begin();
    for (const auto& activation : gaudi2CodeGenerator->getMmeNodeDescriptorGenerator(nodeShared).getMmeActivations())
    {
        const MmeNode& constNode = *static_cast<MmeNode*>(nodeShared.get());
        for (unsigned mmeIdx = 0; mmeIdx < gaudi2CodeGenerator->getHALReader()->getNumMmeEngines(); mmeIdx++)
        {
            gaudi2CodeGenerator->updateMmeNodeDescriptorWrapper(constNode, activation.getDesc(mmeIdx), *newRoi);
        }
        newRoi++;
    }
    node->setPhysicalRois(nodeRois);
}

MmeCommon::EMmeOpType CalculateTensorROIsLinearRanges::getMmeNodeOpType(const MmeNode&(mmeNode)) const
{
    return getOperationTypeCommon(MmeCommon::e_mme_Gaudi2, mmeNode);
}

bool calculateTensorROIsLinearRanges(Gaudi2Graph& g)
{
    CalculateTensorROIsLinearRanges calculator;
    return calculator.apply(g);
}

} // namespace gaudi2