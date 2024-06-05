#include "platform/gaudi3/graph_compiler/patch_point_generator.h"

#include "gaudi3/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi3/gaudi3_tpc_descriptor.h"
#include "gaudi3_dynamic_tpc_pp_generator.h"
#include "gaudi3_types.h"
#include "hal_reader/hal_reader.h"
#include "mme_desc_gen_utils.h"
#include "platform/gaudi3/graph_compiler/descriptor_generator.h"
#include "platform/gaudi3/graph_compiler/gaudi3_dynamic_mme_pp_generator.h"

#include <cstdint>

namespace gaudi3
{
void Gaudi3MMEPatchPointGenerator::generateMmePatchPoints(const MmeNode&                node,
                                                          const MmeDescriptorGenerator& descGenerator,
                                                          DescriptorWrapper<MmeDesc>&   descWrapper,
                                                          const McidMmeUsage&           mcidMmeUsage,
                                                          unsigned                      engineIdx)
{
    TensorPtr xTensor, wTensor, yTensor, oTensor;
    auto      opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi3, node);
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);

    createPatchPointIfNotNull(xTensor, MmeCommon::e_mme_op_x, descGenerator, descWrapper);
    createPatchPointIfNotNull(wTensor, MmeCommon::e_mme_op_w, descGenerator, descWrapper);
    createPatchPointIfNotNull(yTensor, MmeCommon::e_mme_op_y, descGenerator, descWrapper);
    createPatchPointIfNotNull(oTensor, MmeCommon::e_mme_op_o, descGenerator, descWrapper);

    generateMcidMmePatchPoints(node, descWrapper, mcidMmeUsage);
    if (!node.isDynamicShape())
    {
        return;
    }

    DynamicMMEPatchPointGenerator dynGen;
    dynGen.generateDynamicShapesPatchPoints(node, descGenerator, descWrapper, engineIdx);
}

void Gaudi3MMEPatchPointGenerator::createPatchPointIfNotNull(const TensorPtr&              tensor,
                                                             EMmeOperand                   operand,
                                                             const MmeDescriptorGenerator& descGenerator,
                                                             DescriptorWrapper<MmeDesc>&   descWrapper)
{
    if (tensor == nullptr)
    {
        return;
    }

    BasicFieldsContainerInfo& basicFieldsContainerInfo = descWrapper.getBasicFieldsContainerInfo();

    const uint64_t* addr = descGenerator.mmeGetTensorAddressFields(operand, descWrapper.getDescriptor());
    if (addr)
    {
        const uint32_t* lowAddr  = (const uint32_t*)addr;
        const uint32_t* highAddr = ((const uint32_t*)addr) + 1;
        createDescriptorPatchPoint(tensor,
                                   lowAddr,
                                   highAddr,
                                   basicFieldsContainerInfo,
                                   (uint64_t)&descWrapper.getDescriptor());
    }
}

void Gaudi3MMEPatchPointGenerator::generateMcidMmePatchPoints(const MmeNode&              node,
                                                              DescriptorWrapper<MmeDesc>& descWrapper,
                                                              const McidMmeUsage&         mcidMmeUsage)
{
    BasicFieldsContainerInfo& basicFieldsContainerInfo = descWrapper.getBasicFieldsContainerInfo();
    NodePtr nodePtr = const_cast<MmeNode&>(node).shared_from_this();

    if (mcidMmeUsage.operandA != NOP)
    {
        uint64_t offset = offsetof(MmeDesc, axiUserDataA) / sizeof(uint32_t);
        auto fieldInfo = std::make_shared<McidFieldInfo>(offset, nodePtr, mcidMmeUsage.operandA);
        basicFieldsContainerInfo.add({offset, fieldInfo});
    }
    if (mcidMmeUsage.operandB != NOP)
    {
        uint64_t offset = offsetof(MmeDesc, axiUserDataB) / sizeof(uint32_t);
        auto fieldInfo = std::make_shared<McidFieldInfo>(offset, nodePtr, mcidMmeUsage.operandB);
        basicFieldsContainerInfo.add({offset, fieldInfo});
    }
    if (mcidMmeUsage.operandC != NOP)
    {
        uint64_t offset = offsetof(MmeDesc, axiUserDataCout) / sizeof(uint32_t);
        auto fieldInfo = std::make_shared<McidFieldInfo>(offset, nodePtr, mcidMmeUsage.operandC);
        basicFieldsContainerInfo.add({offset, fieldInfo});
    }
}

uint32_t* Gaudi3TPCPatchPointGenerator::getBaseAddrHigh(gaudi3::TpcDesc& desc)
{
    return &desc.m_tensors[m_tensorIdx].base.base_addr_high._raw;
}

uint32_t* Gaudi3TPCPatchPointGenerator::getBaseAddrLow(gaudi3::TpcDesc& desc)
{
    return &desc.m_tensors[m_tensorIdx].base.base_addr_low._raw;
}

void Gaudi3TPCPatchPointGenerator::generateTpcMcidPatchPoints(const TPCNode&                      node,
                                                              DescriptorWrapper<gaudi3::TpcDesc>& descWrapper)
{
    BasicFieldsContainerInfo& basicFieldsContainerInfo = descWrapper.getBasicFieldsContainerInfo();

    NodePtr nodePtr = const_cast<TPCNode&>(node).shared_from_this();

    /****************************************************************************************
     * Loop over McidTpcUsage and create patch points for non NOP cache maintenance actions *
     ****************************************************************************************/

    // FAST_CFG
    if (m_mcidTpcUsage.fastCfg != NOP)
    {
        uint64_t offsetDWord = MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, mcid_fast_cfg);
        auto fieldInfo = std::make_shared<McidFieldInfo>(offsetDWord, nodePtr, m_mcidTpcUsage.fastCfg);
        basicFieldsContainerInfo.add({offsetDWord, fieldInfo});
    }

    // SRF
    unsigned numSRFs = Gaudi3HalReader::instance()->getNumFastConfigMcidSRFs();
    unsigned baseSrfId = Gaudi3HalReader::instance()->getNumSRFs() - numSRFs;

    for (unsigned i = 0; i < m_mcidTpcUsage.srf.size(); i++)
    {
        if (m_mcidTpcUsage.srf[i] != NOP)
        {
            uint64_t offsetDWord = MASK_OFFSET(TpcDesc, m_desc) +
                                   MASK_OFFSET(block_tpc_non_tensor_descriptor, srf) +
                                   (baseSrfId + i);

            auto fieldInfo = std::make_shared<McidFieldInfo>(offsetDWord, nodePtr, (m_mcidTpcUsage.srf[i]));
            basicFieldsContainerInfo.add({offsetDWord, fieldInfo});
        }
    }

    // Private tensor's AXI_CFG
    for (auto& tensorIdxAndAction : m_mcidTpcUsage.tensorPrivateCfg)
    {
        if (tensorIdxAndAction.second != NOP)
        {
            uint64_t offsetDWord =
                MASK_OFFSET(TpcDesc, m_tensors) +                                          // tensors array in desc
                (sizeof(TensorDescGaudi3) / sizeof(uint32_t)) * tensorIdxAndAction.first + // the tensor in the array
                MASK_OFFSET(TensorDescGaudi3, shared) +                                    // shared sub-section
                MASK_OFFSET(block_tpc_tensor_shared, hbw_axi_cfg);                         // axi_cfg in sub-section

            auto fieldInfo = std::make_shared<McidFieldInfo>(offsetDWord, nodePtr, tensorIdxAndAction.second);
            basicFieldsContainerInfo.add({offsetDWord, fieldInfo});
        }
    }
}

void Gaudi3TPCPatchPointGenerator::generateTpcPatchPoints(const TPCNode&                      node,
                                                          DescriptorWrapper<gaudi3::TpcDesc>& descWrapper)
{
    TPCPatchPointGenerator<gaudi3::TpcDesc>::generateTpcPatchPoints(node, descWrapper);
    generatePrintfPatchPoints(node, descWrapper);
    generateTpcMcidPatchPoints(node, descWrapper);

    DynamicTPCPatchPointGenerator dynamicGenerator(descWrapper);
    dynamicGenerator.generatePatchPoints(node);
}

}  // namespace gaudi3
