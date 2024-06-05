#include "descriptor_generator.h"

#include "code_generator.h"
#include "compilation_hal_reader.h"
#include "dma_cost_model.h"
#include "gaudi2_code_generator.h"
#include "gaudi2_graph.h"
#include "habana_global_conf.h"
#include "hal_reader/hal_reader.h"
#include "include/mme_common/mme_common_enum.h"
#include "mme_desc_gen_utils.h"
#include "node_utils.h"
#include "tpc_descriptor_generator.h"
#include "tpc_slice_desc_update.h"

#include <memory>

using namespace Gaudi2;

namespace gaudi2
{
DescriptorGenerator::DescriptorGenerator(Gaudi2CodeGenerator* codeGenerator)
: m_codeGenerator(codeGenerator), m_syncSchemeFwContext(codeGenerator->getNodeUtility())
{
}

void DescriptorGenerator::visit(TPCNode* node)
{
    addTpcDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(TPCSlice* node)
{
    TPCSliceDescUpdate updater(node);
    addTpcDescriptorsToGraph(*static_cast<TPCNode*>(node), &updater);
}

void DescriptorGenerator::addDmaCostInfo(DMANode* node)
{
    DmaCostModel costModel(*Gaudi2HalReader::instance());
    node->getNodeAnnotation().dmaCost = costModel.getCostModelResult(*node);
}

void DescriptorGenerator::visit(DMANode* node)
{
    addDmaDescriptorsToGraph(*node);
    addDmaCostInfo(node);
}

void DescriptorGenerator::visit(MmeNode* node)
{
    updateMmeDescriptorWrappers(*node);
}

void DescriptorGenerator::visit(RotateNode* node)
{
    addRotatorDescriptorsToGraph(*node);
}

void DescriptorGenerator::addTpcDescriptorsToGraph(const TPCNode& node, const TPCSliceDescUpdate* updater)
{
    TpcDescriptorGenerator::DescriptorsVector descs;
    std::list<NodeROI>*                       rois;
    const NodePtr&                            nodePtr    = m_codeGenerator->getNodeSharedPtr(node);
    deviceAddrOffset                          kernelAddr = m_codeGenerator->getKernelAddress(node.getUniqueID());

    rois = m_codeGenerator->getNodeROIs(nodePtr);
    TpcDescriptorGenerator::generateTpcWdDescriptors(node, *rois, kernelAddr, descs);

    if (updater)
    {
        updater->update(descs);
    }

    auto descIt = descs.begin();
    auto roisIt = rois->begin();
    HB_ASSERT(descs.size() == rois->size(), "Number of ROIs does not match the number of descriptors");

    for (; descIt != descs.end() && roisIt != rois->end(); ++descIt, ++roisIt)
    {
        m_syncSchemeFwContext.fillArcSyncScheme<tpc_wd_ctxt_t>(nodePtr, roisIt->pipelineLevel, descIt->fwCtx);

        m_codeGenerator->updateTPCDescriptorWrapper(node, descIt->desc, descIt->mask, descIt->fwCtx, *roisIt);
    }
}

void DescriptorGenerator::addDmaDescriptorsToGraph(const DMANode& node)
{
    if (node.getDmaType() != DMA_TYPE_INTERNAL) return;

    DmaDescriptorsList  descs;
    const NodePtr&      nodePtr = m_codeGenerator->getNodeSharedPtr(node);
    std::list<NodeROI>& rois    = m_codeGenerator->getPhysicalRois(nodePtr);

    generateDmaDescriptors(node, rois, descs);

    auto descIt = descs.begin();
    auto roisIt = rois.begin();
    HB_ASSERT(descs.size() == rois.size(), "Number of ROIs does not match the number of descriptors");

    unsigned prevPipelineLevel = 0xFFFFFFFF;

    for (; descIt != descs.end() && roisIt != rois.end(); ++descIt, ++roisIt)
    {
        // Populate FW Context sync scheme only on the first descriptor of each pipeline
        if (prevPipelineLevel != roisIt->pipelineLevel)
        {
            m_syncSchemeFwContext.fillArcSyncScheme<edma_wd_ctxt_t>(nodePtr, roisIt->pipelineLevel, descIt->fwCtx);
        }

        prevPipelineLevel = roisIt->pipelineLevel;

        m_codeGenerator->updateDMADescriptorWrapper(node, descIt->desc, descIt->mask, descIt->fwCtx, *roisIt);
    }
}

void DescriptorGenerator::updateMmeDescriptorWrappers(const MmeNode& node)
{
    const NodePtr&          nodePtr             = m_codeGenerator->getNodeSharedPtr(node);
    MmeDescriptorsWrappers& descriptorsWrappers = m_codeGenerator->getMmeNodeDescriptorsWrappers(nodePtr);
    std::list<NodeROI>&     logicalRois         = *(m_codeGenerator->getNodeROIs(nodePtr));
    unsigned                descIdx             = 0;

    HB_ASSERT(logicalRois.size() * m_codeGenerator->getHALReader()->getNumMmeEngines() == descriptorsWrappers.size(),
              "mismatch between number of descriptors and number of ROIs in node {}",
              node.getNodeName());

    for (auto roi : logicalRois)
    {
        mme_wd_ctxt_t mmeFwCtx = {0};
        m_syncSchemeFwContext.fillArcSyncScheme<mme_wd_ctxt_t>(nodePtr, roi.pipelineLevel, mmeFwCtx);

        for (unsigned mmeIdx = 0; mmeIdx < m_codeGenerator->getHALReader()->getNumMmeEngines(); mmeIdx++)
        {
            descriptorsWrappers[descIdx++].setFwCtx(mmeFwCtx);
        }
    }
}

void DescriptorGenerator::addRotatorDescriptorsToGraph(const RotateNode& node)
{
    RotDescriptorsList  descs;
    const NodePtr&      nodePtr = m_codeGenerator->getNodeSharedPtr(node);
    std::list<NodeROI>& rois    = m_codeGenerator->getPhysicalRois(nodePtr);

    generateRotatorDescriptors(node, rois, m_codeGenerator->getSramBaseAddr(), descs);

    auto descIt = descs.begin();
    auto roisIt = rois.begin();

    HB_ASSERT(descs.size() == rois.size(), "Number of ROIs does not match the number of descriptors");

    unsigned prevPipelineLevel = 0xFFFFFFFF;

    for (; descIt != descs.end() && roisIt != rois.end(); ++descIt, ++roisIt)
    {
        // Populate FW Context sync scheme only on the first descriptor of each pipeline
        if (prevPipelineLevel != roisIt->pipelineLevel)
        {
            m_syncSchemeFwContext.fillArcSyncScheme<rot_wd_ctxt_t>(nodePtr, roisIt->pipelineLevel, descIt->fwCtx);
        }

        prevPipelineLevel = roisIt->pipelineLevel;

        m_codeGenerator->updateRotatorDescriptorWrapper(node, descIt->desc, descIt->mask, descIt->fwCtx, *roisIt);
    }
}

void DescriptorGenerator::getTensorRoles(const MmeNode&        node,
                                         MmeCommon::EMmeOpType opType,
                                         TensorPtr&            xTensor,
                                         TensorPtr&            wTensor,
                                         TensorPtr&            yTensor,
                                         TensorPtr&            oTensor,
                                         TensorPtr&            aMaskTensor,
                                         TensorPtr&            bMaskTensor)
{
    if (node.getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        aMaskTensor = node.getInput(TENSOR_AUX_BGEMM_MASK_A);
        bMaskTensor = node.getInput(TENSOR_AUX_BGEMM_MASK_B);
    }
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);
}

void DescriptorGenerator::getInputOutputTensors(const MmeNode& node,
                                                TensorPtr&     aTensor,
                                                TensorPtr&     bTensor,
                                                TensorPtr&     cTensor,
                                                TensorPtr&     oTensor,
                                                AuxTensorArray& auxTensors)
{
    const EMmeOpType      opType   = getOperationTypeCommon(MmeCommon::e_mme_Gaudi2, node);
    const Node::eNodeType nodeType = node.getNodeType();
    oTensor                        = node.getOutput(TENSOR_SECONDARY_OFM);
    auxTensors = {};

    switch (opType)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
            if (node.getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
            {
                auxTensors[MmeCommon::MmeAuxTensorIdx::MASKED_BGEMM_A] = node.getInput(TENSOR_AUX_BGEMM_MASK_A);
                auxTensors[MmeCommon::MmeAuxTensorIdx::MASKED_BGEMM_B] = node.getInput(TENSOR_AUX_BGEMM_MASK_B);
            }
        case MmeCommon::e_mme_fwd:
            if (nodeType == Node::TYPE_BATCH_GEMM_DEDW || nodeType == Node::TYPE_GEMM_DEDW)
            {
                aTensor = node.getInput(TENSOR_X_BWD);
                bTensor = node.getInput(TENSOR_DEDY);
                cTensor = node.getOutput(TENSOR_DEDW);
            }
            else if (nodeType == Node::TYPE_BATCH_GEMM_DEDX || nodeType == Node::TYPE_GEMM_DEDX)
            {
                aTensor = node.getInput(TENSOR_DEDY);
                bTensor = node.getInput(TENSOR_WEIGHT);
                cTensor = node.getOutput(TENSOR_DEDX);
            }
            else
            {
                aTensor = node.getInput(TENSOR_IFM);
                bTensor = node.getInput(TENSOR_WEIGHT);
                cTensor = node.getOutput(TENSOR_OFM);
            }
            break;
        case MmeCommon::e_mme_dedw:
            aTensor = node.getInput(TENSOR_X_BWD);
            bTensor = node.getInput(TENSOR_DEDY);
            cTensor = node.getOutput(TENSOR_DEDW);
            break;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            aTensor = node.getInput(TENSOR_DEDY);
            bTensor = node.getInput(TENSOR_WEIGHT);
            cTensor = node.getOutput(TENSOR_DEDX);
            break;
        default:
            HB_ASSERT(false, "Unsupported Gaudi2 MME operation type {}", opType);
    }

    // Set the aux tensors for deterministic cd concurrency
    if (node.getNodeAnnotation().mmeMetaData.mmeStrategy.isDeterministic &&
        node.getNodeAnnotation().mmeMetaData.mmeStrategy.reductionLevel > 1)
    {
        auxTensors[MmeCommon::MmeAuxTensorIdx::CD_SCRATCHPAD] = node.getInput(TENSOR_AUX_CD_SCRATCHPAD);
        auxTensors[MmeCommon::MmeAuxTensorIdx::CD_REDUCTION] = node.getInput(TENSOR_AUX_CD_REDUCTION);
    }
}

EMmeOperand DescriptorGenerator::inputPort2Operand(const EMmeOpType operation, const bool isA)
{
    switch(operation)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_fwd:
            return isA ? MmeCommon::e_mme_op_x : MmeCommon::e_mme_op_w;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            return isA ? MmeCommon::e_mme_op_y : MmeCommon::e_mme_op_w;
        case MmeCommon::e_mme_dedw:
            return isA ? MmeCommon::e_mme_op_x : MmeCommon::e_mme_op_y;
        default:
            HB_ASSERT(false, "Unsupported Gaudi2 MME operation type {}", operation);
    }
    return MmeCommon::e_mme_op_x;
}

EMmeOperand DescriptorGenerator::outputPort2Operand(const EMmeOpType operation)
{
    switch(operation)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_fwd:
            return MmeCommon::e_mme_op_y;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            return MmeCommon::e_mme_op_x;
        case MmeCommon::e_mme_dedw:
            return MmeCommon::e_mme_op_w;
        default:
            HB_ASSERT(false, "Unsupported Gaudi2 MME operation type {}", operation);
    }

    return MmeCommon::e_mme_op_y;
}

}
