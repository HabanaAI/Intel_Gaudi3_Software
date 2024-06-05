#pragma once

#include "math_utils.h"
#include "patch_point_generator.h"
#include "tensor.h"
#include "synapse_common_types.h"
#include "dma_node.h"


template<class Descriptor>
void TPCPatchPointGenerator<Descriptor>::generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<Descriptor>& descWrapper)
{
    Descriptor& descriptor = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& basicFieldsConatinerInfo = descWrapper.getBasicFieldsContainerInfo();

    const auto& nodeInputs = node.getInputs();
    const auto& nodeOutputs = node.getOutputs();
    NodePtr nodePtr = const_cast<TPCNode&>(node).shared_from_this();
    // Writing the input & outputs (but no aux) tensors to the TPCDescriptor
    for(const auto& nodeOperands : {nodeInputs, nodeOutputs})
    {
        for (const TensorPtr& t : nodeOperands)
        {
            // OUTPUT_DESCRIBING_SHAPE_TENSORs are not in the descriptor
            // so they should be skipped (m_tensorIdx not incremented for them)
            // INPUT_DESCRIBING_SHAPE_TENSORs are in the descriptor but
            // they do not need patch points, so m_tensorIdx is incremented
            // but no patch points created.
            // Aux tensors are always at the end of the descriptor list,
            // they are handled in the next loop
            if (t->isAuxTensor()) continue;
            if (t->getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR) continue;
            if (!t->isShapeTensor())
            {
                uint32_t* highAddr = getBaseAddrHigh(descriptor);
                uint32_t* lowAddr  = getBaseAddrLow(descriptor);
                createDescriptorPatchPoint(t,
                        lowAddr,
                        highAddr,
                        basicFieldsConatinerInfo,
                        (uint64_t) &descriptor,
                        nodePtr);
            }

            m_tensorIdx += div_round_up(t->getDim(), Descriptor::c_max_tensor_dims);
        }
    }

    // Writing the aux tensors to the TPCDescriptor
    for (const TensorPtr& t : nodeInputs)
    {
        if (!t->isAuxTensor()) continue;

        uint32_t* highAddr = getBaseAddrHigh(descriptor);
        uint32_t* lowAddr  = getBaseAddrLow(descriptor);
        createDescriptorPatchPoint(t,
                lowAddr,
                highAddr,
                basicFieldsConatinerInfo,
                (uint64_t) &descriptor,
                nodePtr);

        m_tensorIdx += div_round_up(t->getDim(), Descriptor::c_max_tensor_dims);
    }

    // Generate patch-point for the kernel address which is embedded in the descriptor
    createDescriptorPatchPoint(nullptr, // no tensor
                               &descriptor.m_desc.kernel_base_address_low._raw,
                               &descriptor.m_desc.kernel_base_address_high._raw,
                               basicFieldsConatinerInfo,
                               (uint64_t) &descriptor,
                               nodePtr);
}


template<class Descriptor>
void TPCPatchPointGenerator<Descriptor>::generatePrintfPatchPoints(const TPCNode& node, DescriptorWrapper<Descriptor>& descWrapper)
{
    Descriptor& descriptor = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& basicFieldsConatinerInfo = descWrapper.getBasicFieldsContainerInfo();

    // handle printf tensor
    if (node.isPrintfUsed())
    {
        pTensor t = node.getPrintfTensor();
        if (t != nullptr)
        {
            // The printf tensor index is determined by the elf header. It is not necessarily positioned right after
            // the input/output tensors
            m_tensorIdx = node.getPrintfPosition(m_tensorIdx);

            uint32_t* highAddr = getBaseAddrHigh(descriptor);
            uint32_t* lowAddr  = getBaseAddrLow(descriptor);
            createDescriptorPatchPoint(t,
                                       lowAddr,
                                       highAddr,
                                       basicFieldsConatinerInfo,
                                       (uint64_t) &descriptor,
                                       const_cast<TPCNode&>(node).shared_from_this());
            // Set to invalid index so no one will use it beyond this point
            m_tensorIdx = 0xFF;
        }
    }
}

template<class Descriptor>
void DMAPatchPointGenerator<Descriptor>::generateDmaPatchPoints(const DMANode& node, DescriptorWrapper<Descriptor>& descWrapper)
{
    Descriptor& descriptor = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& basicFieldsConatinerInfo = descWrapper.getBasicFieldsContainerInfo();

    // handle source address if this isn't memset node (in memset the source address functions as the memset value)
    if (!node.isMemset())
    {
        createDescriptorPatchPoint(node.getInput(0),
                                   &descriptor.src_base_lo._raw,
                                   &descriptor.src_base_hi._raw,
                                   basicFieldsConatinerInfo,
                                   (uint64_t) &descriptor,
                                   const_cast<DMANode&>(node).shared_from_this());
    }

    // handle destination address
    createDescriptorPatchPoint(node.getOutput(0),
                               &descriptor.dst_base_lo._raw,
                               &descriptor.dst_base_hi._raw,
                               basicFieldsConatinerInfo,
                               (uint64_t) &descriptor,
                               const_cast<DMANode&>(node).shared_from_this());
}

template<class RotatorDescriptor>
void RotatorPatchPointGenerator<RotatorDescriptor>::generateRotatorPatchPoints(const RotateNode& node, DescriptorWrapper<RotatorDescriptor>& descWrapper)
{
    RotatorDescriptor& descriptor = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& descAddressFieldsInfo = descWrapper.getBasicFieldsContainerInfo();

    // In each Rotate descriptor there are two addresses that need to be patched: input address and output address
    uint32_t *lowAddr, *highAddr;
    // Set patch point for the beginning of the input tensor
    lowAddr  = (uint32_t*)(&descriptor.in_img_start_addr_l);
    highAddr = (uint32_t*)(&descriptor.in_img_start_addr_h);
    createDescriptorPatchPoint(node.getInput(0), lowAddr, highAddr, descAddressFieldsInfo, (uint64_t) &descriptor, const_cast<RotateNode&>(node).shared_from_this());

    // Set patch point for the beginning of the output tensor
    lowAddr  = (uint32_t*)(&descriptor.out_img_start_addr_l);
    highAddr = (uint32_t*)(&descriptor.out_img_start_addr_h);
    createDescriptorPatchPoint(node.getOutput(0), lowAddr, highAddr, descAddressFieldsInfo, (uint64_t) &descriptor, const_cast<RotateNode&>(node).shared_from_this());
}

template<class RotatorDescriptor>
void RotatorPatchPointGenerator<RotatorDescriptor>::generateEmptyJobRotatorPatchPoints(DescriptorWrapper<RotatorDescriptor>& descWrapper)
{
    RotatorDescriptor& descriptor = descWrapper.getDescriptor();
    BasicFieldsContainerInfo& descAddressFieldsInfo = descWrapper.getBasicFieldsContainerInfo();

    // In each Rotate descriptor there are two addresses that need to be patched: input address and output address
    uint32_t *lowAddr, *highAddr;
    // Set patch point for the beginning of the input tensor
    lowAddr  = (uint32_t*)(&descriptor.in_img_start_addr_l);
    highAddr = (uint32_t*)(&descriptor.in_img_start_addr_h);
    createDescriptorPatchPoint(nullptr, lowAddr, highAddr, descAddressFieldsInfo, (uint64_t) &descriptor, nullptr);

    // Set patch point for the beginning of the output tensor
    lowAddr  = (uint32_t*)(&descriptor.out_img_start_addr_l);
    highAddr = (uint32_t*)(&descriptor.out_img_start_addr_h);
    createDescriptorPatchPoint(nullptr, lowAddr, highAddr, descAddressFieldsInfo, (uint64_t) &descriptor, nullptr);

}
