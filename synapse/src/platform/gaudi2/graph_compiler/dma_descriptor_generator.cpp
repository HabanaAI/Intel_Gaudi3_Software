#include "syn_logging.h"
#include "descriptor_generator.h"
#include "habana_nodes.h"
#include "tensor.h"
#include "utils.h"
#include "defs.h"
#include "gaudi2_graph.h"
#include "queue_command.h"
#include "habana_global_conf.h"

using namespace gaudi2;

static int getReductionDataType(pTensor tensor)
{
    switch (tensor->getElementType())
    {  // per GAUDI2 Interconnect doc, paragraph 1.26.1
        case syn_type_int8:
            return 0;
        case syn_type_int16:
            return 1;
        case syn_type_int32:
            return 2;
        case syn_type_uint8:
            return 3;
        case syn_type_uint16:
            return 4;
        case syn_type_uint32:
            return 5;
        case syn_type_bf16:
            return 6;
        case syn_type_tf32:
        case syn_type_float:
        case syn_type_hb_float:
            return 7;
        case syn_type_fp16:
            return 8;
        case syn_type_fp8_152:
            return 9;
        default:
            break;
    }

    HB_ASSERT(0, "{}: getReductionDataType unsupported", tensor->getName());
    return 0;
}

static void updateSetHbWrReduction(DmaDesc& desc, pTensor tensor)
{
    axuser::reg_hb_wr_reduction& reg = desc.axuser.hb_wr_reduction;

    reg.ind = tensor->isReductionEnabled();

    if (reg.ind)
    {
        reg.dtype = getReductionDataType(tensor);
        reg.op    = tensor->getReductionOperation() & 0x3;
        reg.max   = (tensor->getReductionOperation() == REDUCTION_MAX0) ? 1U : 0U;  // reg.max overrides reg.op
    }

    return;
}

void DescriptorGenerator::generateDmaDescriptors(const DMANode&            node,
                                                 const std::list<NodeROI>& physicalRois,
                                                 DmaDescriptorsList&       descriptors)
{
    bool             isMemset             = node.isMemset();
    bool             isTranspose          = node.isTranspose();
    const TensorPtr& src                  = (node.isPrefetch() || isMemset) ? node.getOutput(0) : node.getInput(0);
    const TensorPtr& dst                  = node.getOutput(0);
    auto             srcElementSizeInBits = src->getElementSizeInBits();
    auto             dstElementSizeInBits = src->getElementSizeInBits();

    HB_ASSERT(src->tensorIsAllocated(), "{}: source tensor has no address, cannot DMA it", node.getNodeName());
    HB_ASSERT(dst->tensorIsAllocated(), "{}: destination tensor has no address, cannot DMA it", node.getNodeName());

    deviceAddrOffset srcOffset = src->tensorAllocatedInSram() ? src->getSramOffset() : src->getDramOffset();
    deviceAddrOffset dstOffset = dst->tensorAllocatedInSram() ? dst->getSramOffset() : dst->getDramOffset();

    DmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    ValidityMask<DmaDesc> descMask {false};

    // Enable/Disable Reduction
    updateSetHbWrReduction(desc, dst);
    SET_MASK_REG_ON(descMask, MASK_OFFSET(DmaDesc, axuser.hb_wr_reduction));

    desc.ctx.idx.val     = node.getContextId();
    desc.ctx.idx_inc.val = 0;
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(DmaDesc, ctx.idx), MASK_OFFSET(DmaDesc, ctx.src_tsize_0));
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(DmaDesc, ctx.wr_comp_addr_hi), MASK_OFFSET(DmaDesc, ctx.commit));

    if (isTranspose)
    {
        SET_MASK_REG_ON(descMask, MASK_OFFSET(DmaDesc, ctx.te_numrows));
    }

    // In case we do linear DMA using descriptor (and not using LinDma packet), the hardware doesn't care about:
    //     1) src_tsize[1..4]
    //     2) dst_tsize[1..4]
    //     3) all strides
    // In addition, the hardware uses dst_tsize_0 also as the source size.
    if (!node.isLinearDma())
    {
        SET_MASK_BULK_ON(std::begin(descMask),
                         MASK_OFFSET(DmaDesc, ctx.dst_tsize_1),
                         MASK_OFFSET(DmaDesc, ctx.wr_comp_addr_hi));

        // In gaudi2, the memset operation does not need source sizes and strides.
        // The hardware ignores them. This is not the case in gaudi.
        // Compare to src/platform/gaudi/graph_compiler/dma_descriptor_generator.cpp
        if (!isMemset)
        {
            SET_MASK_BULK_ON(std::begin(descMask),
                             MASK_OFFSET(DmaDesc, ctx.src_tsize_0),
                             MASK_OFFSET(DmaDesc, ctx.dst_tsize_1));
        }
    }
    for (const auto& roi : physicalRois)
    {
        ptrToInt  srcRoiOffset;
        ptrToInt  dstRoiOffset;
        ptrToInt  srcTensorAddr;
        ptrToInt  dstTensorAddr;

        if (roi.inputRois.size() > 0)
        {
            const TensorROILayout& srcLayout = roi.inputRois[0].getLayout();
            const TensorPtr&       roiSrc    = roi.inputRois[0].m_parentTensor;

            // size of dim 0 is specified in bytes whereas size of dims 1..4 are specified in elements
            desc.ctx.src_tsize_0.val = safeBitsToByte(srcLayout.m_size[0] * srcElementSizeInBits);
            desc.ctx.src_tsize_1.val = srcLayout.m_size[1];
            desc.ctx.src_tsize_2.val = srcLayout.m_size[2];
            desc.ctx.src_tsize_3.val = srcLayout.m_size[3];
            desc.ctx.src_tsize_4.val = srcLayout.m_size[4];

            // all strides are specified in bytes
            desc.ctx.src_stride_1.val = srcLayout.spatialStrides[0];
            desc.ctx.src_stride_2.val = srcLayout.spatialStrides[1];
            desc.ctx.src_stride_3.val = srcLayout.spatialStrides[2];
            desc.ctx.src_stride_4.val = srcLayout.spatialStrides[3];

            if (src == roiSrc)
            {
                srcTensorAddr.u64 = srcOffset;
            }
            else
            {
                srcTensorAddr.u64 = roiSrc->tensorAllocatedInSram() ? roiSrc->getSramOffset() : roiSrc->getDramOffset();
            }
            srcRoiOffset.u64 = srcLayout.baseAddress - srcTensorAddr.u64;
        }
        else
        {
            HB_ASSERT(isMemset, "{}: missing src info", node.getNodeName());
            // src address is used as the memset data
            srcTensorAddr.u64 = 0;
            srcRoiOffset.u64 = 0;
        }

        HB_ASSERT(!roi.outputRois.empty(), "{}: missing dst info", node.getNodeName());
        const TensorROILayout& dstLayout = roi.outputRois[0].getLayout();

        // in dma broadcast, parent tensor may change between different logical rois
        const TensorPtr& roiDst = roi.outputRois[0].m_parentTensor;

        // size of dim 0 is specified in bytes whereas size of dims 1..4 are specified in elements
        desc.ctx.dst_tsize_0.val = safeBitsToByte(dstLayout.m_size[0] * dstElementSizeInBits);
        desc.ctx.dst_tsize_1.val = dstLayout.m_size[1];
        desc.ctx.dst_tsize_2.val = dstLayout.m_size[2];
        desc.ctx.dst_tsize_3.val = dstLayout.m_size[3];
        desc.ctx.dst_tsize_4.val = dstLayout.m_size[4];

        // all strides are specified in bytes
        desc.ctx.dst_stride_1.val = dstLayout.spatialStrides[0];
        desc.ctx.dst_stride_2.val = dstLayout.spatialStrides[1];
        desc.ctx.dst_stride_3.val = dstLayout.spatialStrides[2];
        desc.ctx.dst_stride_4.val = dstLayout.spatialStrides[3];

        if (dst == roiDst)
        {
            dstTensorAddr.u64 = dstOffset;
        }
        else
        {
            dstTensorAddr.u64 = roiDst->tensorAllocatedInSram() ? roiDst->getSramOffset() : roiDst->getDramOffset();
        }
        dstRoiOffset.u64 = dstLayout.baseAddress - dstTensorAddr.u64;

        desc.ctx.src_offset_lo.val = srcRoiOffset.u32[0];
        desc.ctx.src_offset_hi.val = srcRoiOffset.u32[1];
        desc.ctx.src_base_lo.val   = srcTensorAddr.u32[0];
        desc.ctx.src_base_hi.val   = srcTensorAddr.u32[1];
        desc.ctx.dst_offset_lo.val = dstRoiOffset.u32[0];
        desc.ctx.dst_offset_hi.val = dstRoiOffset.u32[1];
        desc.ctx.dst_base_lo.val   = dstTensorAddr.u32[0];
        desc.ctx.dst_base_hi.val   = dstTensorAddr.u32[1];

        if ((desc.ctx.src_base_lo.val & 0x7F) != 0)
        {
            LOG_DEBUG(GC, "src address is not aligned to 128-bytes cache line size");
        }
        if ((desc.ctx.dst_base_lo.val & 0x7F) != 0)
        {
            LOG_DEBUG(GC, "dst address is not aligned to 128-bytes cache line size");
        }

        if (isTranspose)
        {
            desc.ctx.te_numrows.val = desc.ctx.src_tsize_1.val * desc.ctx.src_tsize_2.val * desc.ctx.src_tsize_3.val *
                                      desc.ctx.src_tsize_4.val;
            HB_ASSERT(desc.ctx.te_numrows.val * srcElementSizeInBits % (128 * 8) == 0 ||
                          desc.ctx.te_numrows.val * srcElementSizeInBits < 128 * 8,
                      "num rows must be lower than 128B or dividable by 128B");
            desc.ctx.ctrl.transpose = 1;
            unsigned dtype;
            switch (srcElementSizeInBits)
            {
                case 32: { dtype = 3; break;}
                case 16: { dtype = 2; break;}
                case 8:  { dtype = 1; break;}
                case 4:  { dtype = 0; break;}
                default: { dtype = 0; break;}
            }
            desc.ctx.ctrl.dtype = dtype;
        }
        else
        {
            desc.ctx.ctrl.transpose = 0;
        }

        descriptors.push_back({desc, descMask, edma_wd_ctxt_t {0}});
    }
    LOG_DEBUG(GC, "total DMA descriptors for node {}: {} ", node.getNodeName(), physicalRois.size());
}
