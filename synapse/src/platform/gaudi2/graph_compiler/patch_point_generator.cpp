#include "platform/gaudi2/graph_compiler/patch_point_generator.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_dynamic_dma_pp_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_dynamic_mme_pp_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_dynamic_tpc_pp_generator.h"
#include "mme_desc_gen_utils.h"

using namespace Gaudi2;

namespace gaudi2
{

void Gaudi2MMEPatchPointGenerator::generateMmePatchPoints(const MmeNode& node,
                                                          const MmeDescriptorGenerator& descGenerator,
                                                          DescriptorWrapper<MmeDesc>& descWrapper)
{
    TensorPtr xTensor, wTensor, yTensor, oTensor, aMaskTensor, bMaskTensor;
    auto      opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi2, node);
    DescriptorGenerator::getTensorRoles(node, opType, xTensor, wTensor, yTensor, oTensor, aMaskTensor, bMaskTensor);
    bool useAuxInputs = false;
    if (descWrapper.getBasicFieldsContainerInfo().getRoi() && descWrapper.getBasicFieldsContainerInfo().getRoi()->isAux)
    {
        useAuxInputs = true;
    }
    createPatchPointIfNotNull(useAuxInputs ? aMaskTensor : xTensor, MmeCommon::e_mme_op_x, descGenerator, descWrapper);
    createPatchPointIfNotNull(useAuxInputs ? bMaskTensor : wTensor, MmeCommon::e_mme_op_w, descGenerator, descWrapper);
    createPatchPointIfNotNull(yTensor, MmeCommon::e_mme_op_y, descGenerator, descWrapper);
    createPatchPointIfNotNull(oTensor, MmeCommon::e_mme_op_o, descGenerator, descWrapper);

    if (!node.isDynamicShape())
    {
        return;
    }

    DynamicMMEPatchPointGenerator dynGen;
    dynGen.generateDynamicShapesPatchPoints(node, descGenerator, descWrapper, 0);
}

void Gaudi2MMEPatchPointGenerator::createPatchPointIfNotNull(const TensorPtr& tensor, EMmeOperand operand,
                                                             const MmeDescriptorGenerator& descGenerator,
                                                             DescriptorWrapper<MmeDesc>& descWrapper)
{
    if (tensor == nullptr)
    {
        return;
    }

    BasicFieldsContainerInfo& basicFieldsContainerInfo = descWrapper.getBasicFieldsContainerInfo();

    const uint64_t* addr = descGenerator.mmeGetTensorAddressFields(operand, descWrapper.getDescriptor());
    if (addr)
    {
        const uint32_t * lowAddr = (const uint32_t*)addr;
        const uint32_t * highAddr = ((const uint32_t*)addr)+1;
        createDescriptorPatchPoint(tensor, lowAddr, highAddr, basicFieldsContainerInfo, (uint64_t) &descWrapper.getDescriptor());
    }
}

void Gaudi2DMAPatchPointGenerator::generateDmaPatchPoints(const DMANode&                      node,
                                                          DescriptorWrapper<gaudi2::DmaDesc>& descWrapper)
{
    gaudi2::DmaDesc&          descriptor               = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& basicFieldsConatinerInfo = descWrapper.getBasicFieldsContainerInfo();

    // handle source address if this isn't memset node (in memset the source address functions as the memset value)
    if (!node.isMemset())
    {
        createDescriptorPatchPoint(node.getInput(0),
                                   &descriptor.ctx.src_base_lo._raw,
                                   &descriptor.ctx.src_base_hi._raw,
                                   basicFieldsConatinerInfo,
                                   (uint64_t)&descriptor,
                                   const_cast<DMANode&>(node).shared_from_this());
    }

    // handle destination address
    createDescriptorPatchPoint(node.getOutput(0),
                               &descriptor.ctx.dst_base_lo._raw,
                               &descriptor.ctx.dst_base_hi._raw,
                               basicFieldsConatinerInfo,
                               (uint64_t)&descriptor,
                               const_cast<DMANode&>(node).shared_from_this());

    if (!node.isDynamicShape() || node.isTranspose())
    {
        return;
    }

    DynamicDMAPatchPointGenerator dynamicGenerator(descWrapper);
    dynamicGenerator.generatePatchPoints(node);
}

uint32_t* Gaudi2TPCPatchPointGenerator::getBaseAddrHigh(gaudi2::TpcDesc& desc)
{
    return &desc.m_tensors[m_tensorIdx].base_addr_high._raw;
}

uint32_t* Gaudi2TPCPatchPointGenerator::getBaseAddrLow(gaudi2::TpcDesc& desc)
{
    return &desc.m_tensors[m_tensorIdx].base_addr_low._raw;
}

void Gaudi2TPCPatchPointGenerator::generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<gaudi2::TpcDesc>& descWrapper)
{
    TPCPatchPointGenerator<gaudi2::TpcDesc>::generateTpcPatchPoints(node, descWrapper);
    generatePrintfPatchPoints(node, descWrapper);

    DynamicTPCPatchPointGenerator dynamicGenerator(descWrapper);
    dynamicGenerator.generatePatchPoints(node);
}

}
