#include "../descriptor_generator.h"
#include "gaudi_graph.h"
#include "gaudi_types.h"
#include "mme_node.h"
#include "transpose_node.h"
#include "node_roi.h"
#include "gaudi_code_generator.h"

#include "platform/gaudi/graph_compiler/passes.h"
#include "calculate_tensor_rois_linear_ranges.h"
#include "mme_desc_gen_utils.h"

static void addTensorRoiLinearRanges(const OverlapRoi& overlapRoi,
                                     const pTensor& tensor,
                                     TensorROI& tensorRoi)
{
    tensorRoi.m_layout.inSram = tensor->tensorAllocatedInSram();
    tensorRoi.m_overlapRoi = overlapRoi;
    tensorRoi.m_overlapRoi.offset = tensor->getTensorOffset();
}

static void calculateForActivationGaudi1(const gaudi::MmeActivation& activation, const MmeNode& mmeNode, NodeROI& roi)
{
    const OverlapRoi* firstInputRoi  = nullptr;
    const OverlapRoi* secondInputRoi = nullptr;
    const OverlapRoi* outputRoi      = nullptr;

    MmeCommon::EMmeOpType opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, mmeNode);
    switch (opType )
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
            firstInputRoi  = &activation.roiX;
            secondInputRoi = &activation.roiW;
            outputRoi      = &activation.roiY;
            break;
        case MmeCommon::e_mme_dedw:
            firstInputRoi = &activation.roiY;
            secondInputRoi = &activation.roiX;
            outputRoi      = &activation.roiW;
            if (mmeNode.getNodeType() != Node::TYPE_DEDW)
            {
                std::swap(secondInputRoi, firstInputRoi);
            }
            break;
        case MmeCommon::e_mme_dedx:
            firstInputRoi = &activation.roiY;
            outputRoi       = &activation.roiX;
            secondInputRoi  = &activation.roiW;
            break;
        default:
            HB_ASSERT( false, "unsupported operation type ");
    }

    addTensorRoiLinearRanges(*firstInputRoi, mmeNode.getInput(TENSOR_IFM), roi.inputRois[TENSOR_IFM]);
    addTensorRoiLinearRanges(*secondInputRoi, mmeNode.getInput(TENSOR_WEIGHT), roi.inputRois[TENSOR_WEIGHT]);
    addTensorRoiLinearRanges(*outputRoi, mmeNode.getOutput(TENSOR_OFM), roi.outputRois[TENSOR_OFM]);
}


//
// gaudi::CalculateTensorROIsLinearRanges
//

bool gaudi::CalculateTensorROIsLinearRanges::apply(HabanaGraph& g) const
{
    bool res = ::CalculateTensorROIsLinearRanges::apply(g);

    DES_CACHE.printDesCacheStats();
    return res;
}

void gaudi::CalculateTensorROIsLinearRanges::calculateMmeLinearRanges(HabanaGraph& g, const NodePtr& node) const
{
    std::list<NodeROI>& nodeRois = *g.GetNodeROIs(node);
    std::list<MmeActivation> activations;
    static const uint32_t CONV_NUM_OF_INPUTS = 2;
    static const uint32_t CONV_NUM_OF_OUTPUTS = 1;

    HB_ASSERT(std::dynamic_pointer_cast<MmeNode>(node) != nullptr, "Impropper node type");
    // Currently support only one ROI
    HB_ASSERT(nodeRois.size() == 1, "Currently only one ROI is supported");

    const MmeNode& mmeNode = *static_cast<MmeNode*>(node.get());
    gaudi::DescriptorGenerator::generateMmeDescriptor(mmeNode, activations);

    NodeROI origROI = nodeRois.front();
    origROI.inputRois.resize(CONV_NUM_OF_INPUTS);
    origROI.outputRois.resize(CONV_NUM_OF_OUTPUTS);

    GaudiGraph* gaudiGraph = downcaster<GaudiGraph>(&g);
    unsigned pipeLevel = 0;
    std::list<NodeROI>& newRois    = g.getCodeGenerator()->getPhysicalRois(node);
    newRois.resize(activations.size());
    nodeRois.clear();

    std::list<NodeROI>::iterator newRoi = newRois.begin();
    for (const MmeActivation& activation : activations)
    {
        *newRoi = origROI;
        downcaster<GaudiCodeGenerator>(gaudiGraph->getCodeGenerator().get())
            ->updateMmeNodeDescriptorWrapper(mmeNode, activation.getDesc(0), *newRoi);
        downcaster<GaudiCodeGenerator>(gaudiGraph->getCodeGenerator().get())
            ->updateMmeNodeDescriptorWrapper(mmeNode, activation.getDesc(1), *newRoi);
        gaudiGraph->updateMmeRollupsArray(mmeNode, activation.numRollups);

        calculateForActivationGaudi1(activation, mmeNode, *newRoi);
        HB_ASSERT(g.getHALReader()->getSyncObjectMaxValue() > activation.numSignals,
                  "Num signals {} is used to set the value of the sync object and cannot ecxeed {}",
                  activation.numSignals,
                  g.getHALReader()->getSyncObjectMaxValue());
        newRoi->numSignals = activation.numSignals;
        newRoi->pipelineLevel = pipeLevel++;
        nodeRois.push_back(*newRoi);

        printLinearRanges(newRoi->inputRois[0], true);
        printLinearRanges(newRoi->inputRois[1], true);
        printLinearRanges(newRoi->outputRois[0], false);
        newRoi++;
    }
    node->setPhysicalRois(newRois);
    DES_CACHE.printDesCacheStats();
}

bool gaudi::calculateTensorROIsLinearRanges(GaudiGraph& g)
{
    gaudi::CalculateTensorROIsLinearRanges pass;
    return pass.apply(g);
}
