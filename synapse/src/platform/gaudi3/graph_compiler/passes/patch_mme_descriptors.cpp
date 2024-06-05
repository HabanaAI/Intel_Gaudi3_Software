#include "patch_mme_descriptors.h"
#include "gaudi3_graph.h"
#include "../descriptor_generator.h"
#include "mme_desc_gen_utils.h"

using namespace MmeCommon;

namespace gaudi3
{
bool patchMmeDescriptors(Gaudi3Graph& g)
{
    const NodeVector& sortedNodes = g.getExeSortedNodes();
    for (NodePtr node : sortedNodes)
    {
        if (g.runsOnMME(node))
        {
            const MmeNode& mmeNode    = *static_cast<MmeNode*>(node.get());
            NodePtr        nodeShared = g.getNodeSharedPtr(mmeNode);
            HB_ASSERT_PTR(nodeShared);

            MmeDescriptorGenerator& descGenerator = g.getMmeNodeDescriptorGenerator(nodeShared);
            MMETensorPatcher::patchTensors(mmeNode, descGenerator);
        }
    }
    return true;
}

void MMETensorPatcher::patchTensors(const MmeNode& mmeNode, MmeDescriptorGenerator& descGenerator)
{
    TensorPtr aTensor;
    TensorPtr bTensor;
    TensorPtr cTensor;
    TensorPtr oTensor;
    DescriptorGenerator::getInputOutputTensors(mmeNode, aTensor, bTensor, cTensor, oTensor);
    const EMmeOpType opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi3, mmeNode);

    // Precaution measure: the output of transpose via gemm must be aligned
    HB_ASSERT(opType != MmeCommon::e_mme_gemm_transpose || isTensorAddressCacheLineAligned(cTensor, false),
              "The output of transpose via gemm must be aligned");

    // Record all tensor addresses
    // patchMetaData struct holds for each logical tensor its actual address and location (isSram).
    // It always holds a and c. o, aux tensors and secondary output are optional.
    MmePatchMetaData patchMetaData;

    patchMetaData.bOperandUsed = (opType != e_mme_memcpy && opType != e_mme_trans);
    patchMetaData.oOperandUsed = (oTensor ? true : false);

    patchMetaData.tensorMetaData[INPUT_TENSOR_A] = {aTensor->getTensorOffset(), aTensor->tensorAllocatedInSram()};
    if (patchMetaData.bOperandUsed)
    {
        patchMetaData.tensorMetaData[INPUT_TENSOR_B] = {bTensor->getTensorOffset(), bTensor->tensorAllocatedInSram()};
    }
    patchMetaData.tensorMetaData[OUTPUT_TENSOR_C] = {cTensor->getTensorOffset(), cTensor->tensorAllocatedInSram()};
    if (patchMetaData.oOperandUsed)
    {
        patchMetaData.tensorMetaData[OUTPUT_TENSOR_O] = {oTensor->getTensorOffset(), oTensor->tensorAllocatedInSram()};
    }

    // Aux tensors, currently not in use in gaudi3
    patchMetaData.tensorMetaData[AUX_TENSOR_0] = std::nullopt;
    patchMetaData.tensorMetaData[AUX_TENSOR_1] = std::nullopt;

    // Patch CD Parallel optimization aux tensors
    if (mmeNode.getNumInputs() > TENSOR_AUX_CD_SCRATCHPAD && mmeNode.getInput(TENSOR_AUX_CD_SCRATCHPAD) != nullptr)
    {
        TensorPtr auxScratchpad                                = mmeNode.getInput(TENSOR_AUX_CD_SCRATCHPAD);
        patchMetaData.tensorMetaData[AUX_ROLE_CD_SCRATCHPAD]   = {auxScratchpad->getTensorOffset(),
                                                                auxScratchpad->tensorAllocatedInSram()};
    }

    if (mmeNode.getNumInputs() > TENSOR_AUX_CD_REDUCTION && mmeNode.getInput(TENSOR_AUX_CD_REDUCTION) != nullptr)
    {
        TensorPtr auxReduction                                = mmeNode.getInput(TENSOR_AUX_CD_REDUCTION);
        patchMetaData.tensorMetaData[AUX_ROLE_CD_REDUCTION]   = {auxReduction->getTensorOffset(),
                                                               auxReduction->tensorAllocatedInSram()};
    }

    // Patch the mme tensors
    bool calcRoi = mmeNode.getGraphTraits()->getCompilationMode() != CompilationMode::Eager;
    descGenerator.patchMmeDescriptors(patchMetaData, calcRoi);

    const auto& layerParams = descGenerator.getParams();
    if (layerParams.tracing.traceMode != e_mme_trace_mode_none)
    {
        descGenerator.patchContextId(mmeNode.getContextId());
        LOG_TRACE(GC, "MME node {} got context id {}", mmeNode.getNodeName(), mmeNode.getContextId());
    }
}
}  // namespace gaudi3
