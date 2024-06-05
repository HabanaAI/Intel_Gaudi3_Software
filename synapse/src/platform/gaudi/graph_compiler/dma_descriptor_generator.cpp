#include "descriptor_wrapper.h"
#include "gaudi/gaudi.h"
#include "syn_logging.h"
#include "descriptor_generator.h"
#include "habana_nodes.h"
#include "sync_object_manager.h"
#include "tensor.h"
#include "utils.h"
#include "defs.h"

#include "queue_command.h"
#include "habana_global_conf.h"
#include "compilation_hal_reader.h"

#include <limits>
#include <habana_global_conf.h>

using namespace gaudi;

#pragma pack(push, 1)
struct wr_awuser_31_11_spec
{
    union
    {
        struct {
            uint32_t
                reserved2  : 2,
                indication : 1,
                data_type  : 3,
                reserved17 : 1,
                operation  : 2,
                round_mode : 2,
                reserved21 : 21;
        };
        uint32_t _raw;
    };
};
#pragma pack(pop)

static int getReductionDataType(pTensor tensor)
{
    // TODO: In future syn_type_uint16 should be 4 and syn_type_uint32 should be 5
    switch(tensor->getElementType())
    {
        case syn_type_int8:     return 0;
        case syn_type_int16:    return 1;
        case syn_type_int32:    return 2;
        case syn_type_uint8:    return 3;
        case syn_type_bf16:     return 6;
        case syn_type_float:    return 7;
        default:                break;
    }

    HB_ASSERT(0, "{}: getReductionDataType unsupported", tensor->getName());
    return 0;
}

static void updateSetWrAwuser31_11(DmaDesc& desc, pTensor tensor)
{
    wr_awuser_31_11_spec* reg = reinterpret_cast<wr_awuser_31_11_spec*>(&desc.wr_awuser_31_11);

    reg->indication = tensor->isReductionEnabled();

    if (reg->indication)
    {
        reg->data_type = getReductionDataType(tensor);
        reg->operation = tensor->getReductionOperation();
    }
}


static void setDescDstAddress(gaudi::DmaDesc& desc, uint64_t address)
{
    ptrToInt dstRoiAddress = {.u64 = address};
    desc.dst_base_lo.val = dstRoiAddress.u32[0];
    desc.dst_base_hi.val = dstRoiAddress.u32[1];

    if ((desc.dst_base_lo.val & 0x7F) != 0)
    {
        LOG_DEBUG(GC, "dst address is not aligned to 128-bytes cache line size");
    }
}

static void setDescSrcAddress(gaudi::DmaDesc& desc, uint64_t address)
{
    ptrToInt srcRoiAddress = {.u64 = address};
    desc.src_base_lo.val = srcRoiAddress.u32[0];
    desc.src_base_hi.val = srcRoiAddress.u32[1];

    if ((desc.src_base_lo.val & 0x7F) != 0)
    {
        LOG_DEBUG(GC, "src address is not aligned to 128-bytes cache line size");
    }
}

static void setDescriptorForMemset(DmaDesc& desc, bool isLinear, DMA_OP_TYPE opType)
{
    const unsigned int cacheLineSize = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    uint64_t fullRoiSize = (uint64_t) desc.dst_tsize_0.val * desc.dst_tsize_1.val * desc.dst_tsize_2.val * desc.dst_tsize_3.val *
                           desc.dst_tsize_4.val;

    HB_ASSERT(fullRoiSize < (uint64_t)std::numeric_limits<uint32_t>::max(),
          "ROI too large, ROI size {} max size {}",
          fullRoiSize,
          std::numeric_limits<uint32_t>::max());

    // Linear memset is implemented with DMA memset (special handling isn't required)
    if (isLinear) return;

    desc.src_stride_1.val = 0;
    desc.src_stride_2.val = 0;
    desc.src_stride_3.val = 0;
    desc.src_stride_4.val = 0;

    // Empty job
    if (fullRoiSize == 0)
    {
        desc.src_tsize_0.val = 0;
        desc.src_tsize_1.val = 0;
    }
    // DMA copy is currently only when the semantic memset node was linear.
    else if ((opType == DMA_OP_TYPE::DMA_OP_COPY) && ((fullRoiSize % cacheLineSize) == 0))
    {
       // ROI size is expected to be aligned to cache line 128 bytes
       desc.src_tsize_0.val = static_cast<uint64_t>(cacheLineSize);
       desc.src_tsize_1.val = fullRoiSize / static_cast<uint64_t>(cacheLineSize);
    }
    else // strided memset or when the ROI is not aligned to cache line
    {
        desc.src_tsize_0.val = 1;
        desc.src_tsize_1.val = fullRoiSize;
    }

    desc.src_tsize_2.val = 1;
    desc.src_tsize_3.val = 1;
    desc.src_tsize_4.val = 1;

    setDescSrcAddress(desc, SRAM_BASE_ADDR);
}

void setDescSrcSizes(DmaDesc& desc, const unsigned* sizes, unsigned elementSize)
{
    // size of dim 0 is specified in bytes whereas size of dims 1..4 are specified in elements
    desc.src_tsize_0.val = sizes[0] * elementSize;
    desc.src_tsize_1.val = sizes[1];
    desc.src_tsize_2.val = sizes[2];
    desc.src_tsize_3.val = sizes[3];
    desc.src_tsize_4.val = sizes[4];
}

void setDescSrcStrides(DmaDesc& desc, const uint64_t* strides)
{
    // all strides are specified in bytes
    desc.src_stride_1.val = strides[0];
    desc.src_stride_2.val = strides[1];
    desc.src_stride_3.val = strides[2];
    desc.src_stride_4.val = strides[3];
}


void setDescDstSizes(DmaDesc& desc, const unsigned* sizes, unsigned elementSize)
{
    // size of dim 0 is specified in bytes whereas size of dims 1..4 are specified in elements
    desc.dst_tsize_0.val = sizes[0] * elementSize;
    desc.dst_tsize_1.val = sizes[1];
    desc.dst_tsize_2.val = sizes[2];
    desc.dst_tsize_3.val = sizes[3];
    desc.dst_tsize_4.val = sizes[4];
}

void setDescDstStrides(DmaDesc& desc, const uint64_t* strides)
{
    // all strides are specified in bytes
    desc.dst_stride_1.val = strides[0];
    desc.dst_stride_2.val = strides[1];
    desc.dst_stride_3.val = strides[2];
    desc.dst_stride_4.val = strides[3];
}


void setTransposeParameters(DmaDesc& desc, unsigned elementSize)
{
    desc.te_numrows.val =
        desc.src_tsize_1.val * desc.src_tsize_2.val * desc.src_tsize_3.val * desc.src_tsize_4.val;
    HB_ASSERT(desc.te_numrows.val * elementSize % 128 == 0 || desc.te_numrows.val * elementSize < 128,
          "num rows must be lower than 128 or dividable by 128");
}

// Due to [H3-2116] we set the address of the sync object to a dummy since we don't want to
// disrupt the sync scheme
static void noSignaling(gaudi::DmaDesc& desc, SyncObjectManager::SyncId dummySyncId)
{
    ptrToInt soAddress;
    soAddress.u64 = getSyncObjectAddress(dummySyncId);

    desc.wr_comp_addr_lo.val = soAddress.u32[0];
    desc.wr_comp_addr_hi.val = soAddress.u32[1];
}

void DescriptorGenerator::generateDmaDescriptors(
    const DMANode&                                                      node,
    const std::list<NodeROI>&                                           physicalRois,
    std::list<std::pair<gaudi::DmaDesc, ValidityMask<gaudi::DmaDesc>>>& descriptors,
    SyncObjectManager::SyncId                                           dummySyncObj)
{
    LOG_DEBUG(GC, "Dma {}", node.getNodeName());
    pTensor src      = node.isMemset() ? node.getOutput(0) : node.getInput(0);
    pTensor dst      = node.getOutput(0);
    bool    isLinear = node.isLinearDma();
    bool    isMemset = node.isMemset();
    DMA_OP_TYPE opType = node.getOpType();

    HB_ASSERT(src->tensorIsAllocated(), "{}: source tensor has no address, cannot DMA it", node.getNodeName());
    HB_ASSERT(dst->tensorIsAllocated(), "{}: destination tensor has no address, cannot DMA it", node.getNodeName());

    DmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    ValidityMask<DmaDesc> descMask {false};

    // Enable/Disable Reduction
    updateSetWrAwuser31_11(desc, dst);

    SET_MASK_REG_ON(descMask, MASK_OFFSET(DmaDesc, wr_awuser_31_11));
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(DmaDesc, src_base_lo), MASK_OFFSET(DmaDesc, _pad36));
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(DmaDesc, wr_comp_wdata), MASK_OFFSET(DmaDesc, wr_comp_awuser_31_11));
    SET_MASK_REG_ON(descMask, MASK_OFFSET(DmaDesc, dst_tsize_0));
    if (node.isTranspose())
    {
        SET_MASK_REG_ON(descMask, MASK_OFFSET(DmaDesc, te_numrows));
    }

    // In case we do linear DMA using descriptor (and not using LinDma packet), the hardware doesn't care about:
    //     1) src_tsize[1..4]
    //     2) dst_tsize[1..4]
    //     3) all strides
    // In addition, the hardware uses dst_tsize_0 also as the source size.
    // Linear memset is implemented as strided DMA
    if (!isLinear)
    {
        SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(DmaDesc, dst_tsize_1), MASK_OFFSET(DmaDesc, dst_tsize_0));

        // The following line is NOT protected by "if (!node.isMemset())"
        // It is always needed (for memset and non-memset operations) because of the HW problem
        // with strided memset. We implement strided memset as strided memcpy
        // so source sizes and strides need to be get transferred. They get set up a
        // below in setDescriptorForStridedMemset(desc).
        SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(DmaDesc, src_tsize_1), MASK_OFFSET(DmaDesc, _pad80));
    }
    for (auto& roi : physicalRois)
    {
        LOG_DEBUG(GC, " #Pipe {} Engine {} Signals {}", roi.pipelineLevel, roi.engineIndex, roi.numSignals);
        if (roi.inputRois.size() > 0)
        {
            const TensorROILayout& srcLayout = roi.inputRois[0].getLayout();
            unsigned sizes[Tensor::c_tensorMaxNDim];
            castNcopy(sizes, srcLayout.m_size.data(), Tensor::c_tensorMaxNDim);
            setDescSrcSizes(desc, sizes, src->getElementSizeInBytes());
            setDescSrcStrides(desc, srcLayout.spatialStrides);
            setDescSrcAddress(desc, srcLayout.baseAddress);
            LOG_DEBUG(GC, "Dma Input sizes: {}", toString(sizes, sizes + 5, ','));
            LOG_DEBUG(GC,
                      "Dma Input strides: {}",
                      toString(srcLayout.spatialStrides, srcLayout.spatialStrides + 4, ','));
            LOG_DEBUG(GC, "Dma Input base address: {:x} {}", srcLayout.baseAddress, src->inDram() ? "dram" : "sram");

            if (node.isTranspose())
            {
                setTransposeParameters(desc, src->getElementSizeInBytes());
            }
        }
        else
        {
            HB_ASSERT(node.isMemset(), "{}: missing src info", node.getNodeName());
            setDescSrcAddress(desc, 0);
        }

        if (roi.outputRois.size() > 0)
        {
            const TensorROILayout& dstLayout = roi.outputRois[0].getLayout();

            unsigned sizes[Tensor::c_tensorMaxNDim];
            castNcopy(sizes, dstLayout.m_size.data(), Tensor::c_tensorMaxNDim);

            setDescDstSizes(desc, sizes, dst->getElementSizeInBytes());
            setDescDstStrides(desc, dstLayout.spatialStrides);
            setDescDstAddress(desc, dstLayout.baseAddress);
            LOG_DEBUG(GC, "Dma Output sizes: {}", toString(sizes, sizes + 5, ','));
            LOG_DEBUG(GC,
                      "Dma Output strides: {}",
                      toString(dstLayout.spatialStrides, dstLayout.spatialStrides + 4, ','));
            LOG_DEBUG(GC, "Dma Output base address: {:x} {}", dstLayout.baseAddress, dst->inDram() ? "dram" : "sram");

            // A WA for a HW bug:
            // 1. Strided memset
            // 2. https://jira.habana-labs.com/browse/SIV-23 and strided memset
            if (isMemset)
            {
                setDescriptorForMemset(desc, isLinear, opType);
            }
        }
        else
        {
            HB_ASSERT(roi.outputRois.size() > 0, "{}: missing dst info", node.getNodeName());
        }
        desc.dst_base_hi.ctx_id_hi = (uint32_t)((node.getContextId() & 0xFF00) >> 8);

        if (roi.numSignals == 0)
        {
            noSignaling(desc, dummySyncObj);
        }

        descriptors.emplace_back(desc, descMask);
    }
}
