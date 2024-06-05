#include "gaudi3_graph.h"
#include "mcid_converter.h"
#include "include/mme_common/mme_common_enum.h"
#include "descriptor_generator.h"
#include "habana_global_conf.h"
#include "platform/common/tpc_slice_desc_update.h"
#include "tpc_descriptor_generator.h"
#include "gaudi3_code_generator.h"
#include "mme_desc_gen_utils.h"

namespace gaudi3
{
DescriptorGenerator::DescriptorGenerator(Gaudi3Graph* graph) : m_graph(graph), m_syncSchemeFwContext(*graph) {}

void DescriptorGenerator::visit(TPCNode* node)
{
    addTpcDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(MmeNode* node)
{
    updateMmeDescriptorWrappers(*node);
}

void DescriptorGenerator::visit(TPCSlice* node)
{
    TPCSliceDescUpdate updater(node);
    addTpcDescriptorsToGraph(*static_cast<TPCNode*>(node), &updater);
}

void DescriptorGenerator::visit(RotateNode* node)
{
    addRotatorDescriptorsToGraph(*node);
}

void DescriptorGenerator::addTpcDescriptorsToGraph(const TPCNode& node, const TPCSliceDescUpdate* updater)
{
    TpcDescriptorGenerator::DescriptorsVector descs;
    std::list<NodeROI>*                     rois;
    const NodePtr&                          nodePtr = m_graph->getNodeSharedPtr(node);
    deviceAddrOffset kernelAddr = m_graph->getCodeGenerator()->getKernelAddress(node.getUniqueID());

    rois = m_graph->GetNodeROIs(nodePtr);
    TpcDescriptorGenerator::McidTpcUsage mcidTpcUsage;
    TpcDescriptorGenerator::generateTpcWdDescriptors(node,
                                                     *rois,
                                                     kernelAddr,
                                                     descs,
                                                     mcidTpcUsage,
                                                     m_graph->getCodeGenerator()->getMcidConverter());

    if (updater)
    {
        updater->update(descs);
    }

    auto descIt = descs.begin();
    auto roisIt = rois->begin();
    HB_ASSERT(descs.size() == rois->size(), "Number of ROIs does not match the number of descriptors");

    for (; descIt != descs.end() && roisIt != rois->end(); ++descIt, ++roisIt)
    {
        m_syncSchemeFwContext.fillArcSyncScheme<tpc_wd_ctxt_t>(nodePtr, roisIt->pipelineLevel, descIt->fwCtxs.front());
        for (unsigned i = 1; i < descIt->fwCtxs.size(); i++)
        {
            // sync scheme is identical in all contexts
            m_syncSchemeFwContext.copyArcSyncScheme<tpc_wd_ctxt_t>(descIt->fwCtxs.front(), descIt->fwCtxs[i]);
        }
        m_graph->updateTPCDescriptorWrapper(node, descIt->desc, descIt->mask, descIt->fwCtxs, *roisIt, mcidTpcUsage);
    }
}

void DescriptorGenerator::updateMmeDescriptorWrappers(const MmeNode& node)
{
    const NodePtr&          nodePtr             = m_graph->getNodeSharedPtr(node);
    MmeDescriptorsWrappers& descriptorsWrappers = m_graph->getMmeNodeDescriptorsWrappers(nodePtr);
    std::list<NodeROI>&     logicalRois         = *(m_graph->GetNodeROIs(nodePtr));
    unsigned                descIdx             = 0;

    HB_ASSERT(logicalRois.size() * m_graph->getHALReader()->getNumMmeEngines() == descriptorsWrappers.size(),
              "mismatch between number of descriptors and number of ROIs in node {}",
              node.getNodeName());

    for (auto roi : logicalRois)
    {
        mme_wd_ctxt_t mmeFwCtx = {0};
        m_syncSchemeFwContext.fillArcSyncScheme<mme_wd_ctxt_t>(nodePtr, roi.pipelineLevel, mmeFwCtx);

        for (unsigned mmeIdx = 0; mmeIdx < m_graph->getHALReader()->getNumMmeEngines(); mmeIdx++)
        {
            descriptorsWrappers[descIdx++].setFwCtx(mmeFwCtx);
        }
    }
}

void DescriptorGenerator::addRotatorDescriptorsToGraph(const RotateNode& node)
{
    RotDescriptorsList  descs;
    std::unique_ptr<CodeGenerator>& codeGenerator = m_graph->getCodeGenerator();
    std::list<NodeROI>& rois          = codeGenerator->getPhysicalRois(m_graph->getNodeSharedPtr(node));
    const NodePtr&      nodePtr = m_graph->getNodeSharedPtr(node);

    generateRotatorDescriptors(node, rois, codeGenerator->getSramBaseAddr(), descs);

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

        m_graph->updateRotatorDescriptorWrapper(node, descIt->desc, descIt->mask, descIt->fwCtx, *roisIt);
    }
}

MmeCommon::EMmeOpType DescriptorGenerator::getOperationType(const MmeNode& node)
{
    return getOperationTypeCommon(MmeCommon::e_mme_Gaudi3, node);
}

void DescriptorGenerator::getTensorRoles(const MmeNode&        node,
                                         MmeCommon::EMmeOpType opType,
                                         TensorPtr&            xTensor,
                                         TensorPtr&            wTensor,
                                         TensorPtr&            yTensor,
                                         TensorPtr&            oTensor)
{
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);
}

void DescriptorGenerator::getTensorCacheMetaDataFromRoi(const MMENodePtr& node,
                                                        const NodeROI&    roi,
                                                        CacheMetaData&    aTensorCacheMetaData,
                                                        CacheMetaData&    bTensorCacheMetaData,
                                                        CacheMetaData&    cTensorCacheMetaData)
{
    const MmeCommon::EMmeOpType opType   = getOperationTypeCommon(MmeCommon::e_mme_Gaudi3, *node);
    const Node::eNodeType       nodeType = node->getNodeType();

    if (opType == MmeCommon::e_mme_fwd || opType == MmeCommon::e_mme_ab || opType == MmeCommon::e_mme_atb ||
        opType == MmeCommon::e_mme_abt || opType == MmeCommon::e_mme_atbt)
    {
        if (nodeType == Node::TYPE_BATCH_GEMM_DEDW || nodeType == Node::TYPE_GEMM_DEDW)
        {
            aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_X_BWD];
            bTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_DEDY];
            cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_DEDW];
        }
        else if (nodeType == Node::TYPE_BATCH_GEMM_DEDX || nodeType == Node::TYPE_GEMM_DEDX)
        {
            aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_DEDY];
            bTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_WEIGHT];
            cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_DEDX];
        }
        else
        {
            aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_IFM];
            bTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_WEIGHT];
            cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_OFM];
        }
    }
    else if (opType == MmeCommon::e_mme_dedw)
    {
        aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_X_BWD];
        bTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_DEDY];
        cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_DEDW];
    }
    else if (opType == MmeCommon::e_mme_dedx || opType == MmeCommon::e_mme_transposed_dedx)
    {
        aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_DEDY];
        bTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_WEIGHT];
        cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_DEDX];
    }
    else if (opType == MmeCommon::e_mme_memcpy || opType == MmeCommon::e_mme_trans)
    {
        aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_IFM];
        bTensorCacheMetaData = CacheMetaData();
        cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_OFM];
    }
    else if (opType == MmeCommon::e_mme_gemm_transpose)
    {
        aTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_IFM];
        bTensorCacheMetaData = roi.inputsCacheMetaData[TENSOR_UNIT_MATRIX];
        cTensorCacheMetaData = roi.outputsCacheMetaData[TENSOR_OFM];

        // The unit tensor should be fetched in the highest class with allocDH policy
        bTensorCacheMetaData.cacheDirective = SharedAllocate;
        bTensorCacheMetaData.cacheClass = Top;

        // Check if the original allocation policy of the transpose output is suitable
        if (!isAllocPolicySuitableForTensor(cTensorCacheMetaData.cacheDirective, node->getOutput(TENSOR_OFM)))
        {
            LOG_INFO(GC,
                     "Original allocation policy {} for transpose output isn't suitable, changing it to allocH",
                     cTensorCacheMetaData.cacheDirective);

            cTensorCacheMetaData.cacheDirective = HomeAllocate;
        }
    }
}

void DescriptorGenerator::getTensorCacheMetaDataForCDParallel(const MMENodePtr&   mmeNode,
                                                              CacheMetaDataArray& cacheMetaDataVec)
{
    // firstRoi is compute roi
    const NodeROI& firstRoi = mmeNode->getLogicalRois()->front();
    // lastRoi is reduction roi
    const NodeROI& lastRoi = mmeNode->getLogicalRois()->back();

    // Get cache metadata from compute roi - inputA, inputB , outAuxSchratchpad
    std::array<CacheMetaData, 3> firstRoiCacheMetaData;
    getTensorCacheMetaDataFromRoi(mmeNode,
                                  firstRoi,
                                  firstRoiCacheMetaData[MmeCommon::INPUT_TENSOR_A],
                                  firstRoiCacheMetaData[MmeCommon::INPUT_TENSOR_B],
                                  firstRoiCacheMetaData[MmeCommon::OUTPUT_TENSOR_C]);

    HB_ASSERT(firstRoiCacheMetaData[2] == lastRoi.inputsCacheMetaData[1],
              "Aux scratchpad must be the compute roi output and the reductionAdd roi input");

    // Fill cacheMetadData vector
    cacheMetaDataVec[MmeCommon::INPUT_TENSOR_A]  = firstRoiCacheMetaData[MmeCommon::INPUT_TENSOR_A];
    cacheMetaDataVec[MmeCommon::INPUT_TENSOR_B]  = firstRoiCacheMetaData[MmeCommon::INPUT_TENSOR_B];
    cacheMetaDataVec[MmeCommon::OUTPUT_TENSOR_C] = lastRoi.outputsCacheMetaData[0];

    // In case of CD Parallel - get also aux tensors cache metadata from reduction roi
    cacheMetaDataVec[MmeCommon::AUX_TENSOR_SCRATCHPAD] = lastRoi.inputsCacheMetaData[1];
    cacheMetaDataVec[MmeCommon::AUX_TENSOR_REDUCTION]  = lastRoi.inputsCacheMetaData[0];
}

void DescriptorGenerator::getTensorCacheMetaData(const MMENodePtr& node, CacheMetaDataArray& cacheMetaDataVec)
{
    CacheMetaData             aTensorCacheMetaData, bTensorCacheMetaData, cTensorCacheMetaData;
    const std::list<NodeROI>* rois = node->getLogicalRois();
    HB_ASSERT_PTR(rois);

    // We're assuming all ROIs contain the same cache metadata.
    for (const NodeROI& roi : *rois)
    {
        HB_ASSERT(rois->front().inputsCacheMetaData.size() == roi.inputsCacheMetaData.size() &&
                      rois->front().outputsCacheMetaData.size() == roi.outputsCacheMetaData.size(),
                  "All Logical ROIs should have the same cache metadata");

        for (size_t i = 0; i < roi.inputsCacheMetaData.size(); i++)
        {
            HB_ASSERT(rois->front().inputsCacheMetaData[i] == roi.inputsCacheMetaData[i],
                      "All Logical ROIs should have the same cache metadata");
        }
        for (size_t i = 0; i < roi.outputsCacheMetaData.size(); i++)
        {
            HB_ASSERT(rois->front().outputsCacheMetaData[i] == roi.outputsCacheMetaData[i],
                      "All Logical ROIs should have the same cache metadata");
        }
    }

    const NodeROI roi = rois->front();

    getTensorCacheMetaDataFromRoi(node, roi, aTensorCacheMetaData, bTensorCacheMetaData, cTensorCacheMetaData);

    cacheMetaDataVec[MmeCommon::INPUT_TENSOR_A]  = aTensorCacheMetaData;
    cacheMetaDataVec[MmeCommon::INPUT_TENSOR_B]  = bTensorCacheMetaData;
    cacheMetaDataVec[MmeCommon::OUTPUT_TENSOR_C] = cTensorCacheMetaData;

    // In case of CD Parallel - get also aux tensors cache metadata
    if (node->isCdPerforated())
    {
        cacheMetaDataVec[MmeCommon::AUX_TENSOR_SCRATCHPAD] = roi.inputsCacheMetaData[TENSOR_AUX_CD_SCRATCHPAD];
        cacheMetaDataVec[MmeCommon::AUX_TENSOR_REDUCTION]  = roi.inputsCacheMetaData[TENSOR_AUX_CD_REDUCTION];
    }
    return;
}

void DescriptorGenerator::getInputOutputTensors(const MmeNode& node,
                                                TensorPtr&     aTensor,
                                                TensorPtr&     bTensor,
                                                TensorPtr&     cTensor,
                                                TensorPtr&     oTensor)
{
    const EMmeOpType      opType   = getOperationType(node);
    const Node::eNodeType nodeType = node.getNodeType();
    oTensor                        = node.getOutput(TENSOR_SECONDARY_OFM);

    switch (opType)
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
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
        case MmeCommon::e_mme_memcpy:
        case MmeCommon::e_mme_trans:
            //  can be combined with bgemm/fwd since it would return nullptr for w by default
            aTensor = node.getInput(TENSOR_IFM);
            bTensor = nullptr;
            cTensor = node.getOutput(TENSOR_OFM);
            break;
        case MmeCommon::e_mme_gemm_transpose:
            aTensor = node.getInput(TENSOR_IFM);
            bTensor = node.getInput(TENSOR_UNIT_MATRIX);
            cTensor = node.getOutput(TENSOR_OFM);
            break;
        default:
            HB_ASSERT(false, "Unsupported Gaudi3 MME operation type {}", opType);
    }
}

EMmeDataType getMmeElementType(synDataType elementType)
{
    switch (elementType)
    {
        case syn_type_fp16:
            return MmeCommon::e_type_fp16;
        case syn_type_bf16:
            return MmeCommon::e_type_bf16;
        case syn_type_single:
            return MmeCommon::e_type_fp32;
        case syn_type_tf32:
            return MmeCommon::e_type_tf32;
        case syn_type_hb_float:
            return MmeCommon::e_type_fp32;
        case syn_type_fp8_143:
            return MmeCommon::e_type_fp8_143;
        case syn_type_fp8_152:
            return MmeCommon::e_type_fp8_152;
        default:
            HB_ASSERT(false, "Unsupported Gaudi3 MME data type {}", elementType);
    }
    return MmeCommon::e_type_fp16;
}

EMmeOperand DescriptorGenerator::inputPort2Operand(const EMmeOpType operation, const bool isA)
{
    switch (operation)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_gemm_transpose:
            return isA ? MmeCommon::e_mme_op_x : MmeCommon::e_mme_op_w;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            return isA ? MmeCommon::e_mme_op_y : MmeCommon::e_mme_op_w;
        case MmeCommon::e_mme_dedw:
            return isA ? MmeCommon::e_mme_op_x : MmeCommon::e_mme_op_y;
        case MmeCommon::e_mme_memcpy:
        case MmeCommon::e_mme_trans:
            return MmeCommon::e_mme_op_x;
        default:
            HB_ASSERT(false, "Unsupported Gaudi3 MME operation type {}", operation);
    }
    return MmeCommon::e_mme_op_x;
}

EMmeOperand DescriptorGenerator::outputPort2Operand(const EMmeOpType operation)
{
    switch (operation)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_memcpy:
        case MmeCommon::e_mme_trans:
        case MmeCommon::e_mme_gemm_transpose:
            return MmeCommon::e_mme_op_y;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            return MmeCommon::e_mme_op_x;
        case MmeCommon::e_mme_dedw:
            return MmeCommon::e_mme_op_w;
        default:
            HB_ASSERT(false, "Unsupported Gaudi3 MME operation type {}", operation);
    }

    return MmeCommon::e_mme_op_y;
}

}  // namespace gaudi3
