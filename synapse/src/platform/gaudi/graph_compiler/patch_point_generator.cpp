#include "patch_point_generator.h"

#include "defs.h"
#include "gaudi_dynamic_dma_pp_generator.h"
#include "habana_nodes.h"
#include "physical_concat_split_subnode.h"
#include "physical_memory_ops_nodes.h"
#include "recipe.h"
#include "utils.h"

#include "platform/gaudi/graph_compiler/descriptor_generator.h"
#include "platform/gaudi/graph_compiler/smf/smf.h"

#include "recipe_metadata.h"

#include "gaudi/asic_reg_structs/dma_core_regs.h"
#include "gaudi_dynamic_dma_pp_generator.h"
#include "gaudi_dynamic_mme_pp_generator.h"
#include "gaudi_dynamic_tpc_pp_generator.h"
#include "mme_desc_gen_utils.h"

namespace gaudi
{
uint32_t* GaudiTPCPatchPointGenerator::getBaseAddrHigh(gaudi::TpcDesc& desc)
{
    return &desc.m_tensors[m_tensorIdx].base_addr_high._raw;
}

uint32_t* GaudiTPCPatchPointGenerator::getBaseAddrLow(gaudi::TpcDesc& desc)
{
    return &desc.m_tensors[m_tensorIdx].base_addr_low._raw;
}

// Section 1: TPC patch point generation
void GaudiTPCPatchPointGenerator::generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<gaudi::TpcDesc>& descWrapper)
{
    TPCPatchPointGenerator<gaudi::TpcDesc>::generateTpcPatchPoints(node, descWrapper);
    generatePrintfPatchPoints(node, descWrapper);

    gaudi::DynamicTPCPatchPointGenerator dynamicGenerator(descWrapper);

    dynamicGenerator.generatePatchPoints(node);
}

// Section 2: MME patch point generation

void GaudiMMEPatchPointGenerator::generateMmePatchPoints(const MmeNode& node, DescriptorWrapper<gaudi::MmeDesc>& descWrapper)
{
    pTensor xTensor, wTensor, yTensor, oTensor;
    auto opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);
    generatePatchPoint(xTensor, MmeCommon::e_mme_op_x, descWrapper, node);
    generatePatchPoint(wTensor, MmeCommon::e_mme_op_w, descWrapper, node);
    generatePatchPoint(yTensor, MmeCommon::e_mme_op_y, descWrapper, node);

    if (!node.isDynamicShape())
    {
        return;
    }

    DynamicMMEPatchPointGenerator dynGen;
    dynGen.generateDynamicShapesPatchPoints(node, descWrapper);
}

void GaudiMMEPatchPointGenerator::generatePatchPoint(const pTensor&                     tensor,
                                                     MmeCommon::EMmeOperand             op,
                                                     DescriptorWrapper<gaudi::MmeDesc>& descWrapper,
                                                     const MmeNode&                     node)
{
    MmeDesc& descriptor = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& basicFieldsContainerInfo = descWrapper.getBasicFieldsContainerInfo();
    uint32_t* highAddr = nullptr;
    uint32_t* lowAddr = nullptr;
    MmeCommon::MmeLayerParams params                   = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi);
    params.opType                                      = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    gaudi::getTensorAddressFields(op, descriptor, &highAddr, &lowAddr, !params.isConvOperation());
    if (highAddr != nullptr && lowAddr != nullptr)
    {
        createDescriptorPatchPoint(tensor, lowAddr, highAddr,basicFieldsContainerInfo, (uint64_t)&descriptor, const_cast<MmeNode&>(node).shared_from_this());
    }
}

// Section 3: DMA patch point generation

void GaudiDMAPatchPointGenerator::generateDmaPatchPoints(const DMANode& node, DescWrapper& descWrapper)
{
    // first generate generic patch points (call parent method)
    DMAPatchPointGenerator<gaudi::DmaDesc>::generateDmaPatchPoints(node, descWrapper);

    if (!node.isDynamicShape() || node.isTranspose() || node.isMemset())
    {
        return;
    }

    gaudi::DynamicDMAPatchPointGenerator dynamicGenerator(descWrapper);

    dynamicGenerator.generatePatchPoints(node);
}

}
