#include "tpc_descriptor_generator.h"

#include "block_data.h"
#include "descriptor_generator.h"
#include "gaudi2_graph.h"
#include "gaudi2_types.h"
#include "habana_global_conf.h"
#include "hal_reader/hal_reader.h"
#include "queue_command.h"
#include "syn_logging.h"
#include "synapse_common_types.h"
#include "sync/sync_conventions.h"
#include "tpc_node.h"
#include "utils.h"

namespace gaudi2
{
enum RMW_DATA_TYPE
{
    RMW_INT8   = 0x0,
    RMW_INT16  = 0x1,
    RMW_INT32  = 0x2,
    RMW_UINT8  = 0x3,
    RMW_UINT16 = 0x4,
    RMW_UINT32 = 0x5,
    RMW_BF16   = 0x6,
    RMW_FP32   = 0x7,
    RMW_FP16   = 0x8,
    RMW_FP8_152 = 0x9,
    RMW_FP8_143 = 0xA
};

static RMW_DATA_TYPE getTensorDatatype(const TensorPtr& gcTensor)
{
    switch (gcTensor->getElementType())
    {
    case syn_type_bf16:
        return RMW_BF16;

    case syn_type_single:
        return RMW_FP32;

    case syn_type_int32:
    case syn_type_int64:
        return RMW_INT32;

    case syn_type_int16:
        return RMW_INT16;

    case syn_type_uint16:
        return RMW_UINT16;

    case syn_type_uint8:
        return RMW_UINT8;

    case syn_type_fixed:
        return RMW_INT8;

    case syn_type_fp16:
        return RMW_FP16;

    case syn_type_uint32:
    case syn_type_uint64:
        return RMW_UINT32;

    case syn_type_fp8_152:
        return RMW_FP8_152;

    case syn_type_fp8_143:
        return RMW_FP8_143;

    default:
        LOG_ERR(GC, "{}: Got unknown data type {}", HLLOG_FUNC, gcTensor->getElementType());
        HB_ASSERT(0, "Unknown data");
        break;
    }
    return RMW_INT8;
}

static void fillTPCTensorDesc(const TensorPtr&                gcTensor,
                              gaudi2::block_tpc_tensor*       tpcTensor,
                              ValidityMask<TpcDesc>::iterator tpcMask,
                              uint32_t                        paddingValue,
                              bool                            isIRF44)
{
    static const auto MAX_SUPPORTED_DIM_FOR_TENSOR_CONFIG = 5u;

    unsigned tensorDim = gcTensor->getDim();

    HB_ASSERT(!gcTensor->isStridedOnFCD(), "stride on fcd isn't supported");

    tpcTensor->padding_value.v              = paddingValue;
    tpcTensor->tensor_config.data_type      = getTensorDatatype(gcTensor);
    tpcTensor->tensor_config.last_dim       = gcTensor->getIndexOfMaxNonDegenerateStride();
    tpcTensor->tensor_config.valid_dim_mask = 0x1f;
    tpcTensor->tensor_config.last_dim64     = 0;     //@TODO should be updated, currently set with default value
    tpcTensor->tensor_config.t_pref_dis     = GCFG_ENABLE_TPC_DCACHE_PREFETCH.value()? 0 : 1;     // Enables the new D$ prefetch
    tpcTensor->pref_stride.val              = gcTensor->getPrefetchStride();

    // Enable/Disable RMW (reduction)
    const bool isReductionEnabled    = gcTensor->isReductionEnabled();
    tpcTensor->tensor_config.rmw_set = isReductionEnabled;
    tpcTensor->tensor_config.rmw_op  = isReductionEnabled ? gcTensor->getReductionOperation() : REDUCTION_ADD;

    tpcTensor->tensor_config.dup_oob = 0; //@TODO should be updated, currently set with default value
    tpcTensor->tensor_config.l0cd    = 0; //@TODO should be updated, currently set with default value

    NSizeArray   sizes        = getTpcDescNSizesInElements(*gcTensor);
    NStrideArray elemNStrides = getTpcDescNStridesInElements(*gcTensor);

    ptrToInt tensorOffset;
    tensorOffset.u64   = gcTensor->getTensorOffset();
    auto numDimBundles = div_round_up(tensorDim, gaudi2::TpcDesc::c_max_tensor_dims);
    for (uint32_t dimBundle = 0; dimBundle < numDimBundles; dimBundle++)
    {
        SET_MASK_BULK_ON(
            tpcMask,
            MASK_SIZE(gaudi2::block_tpc_tensor) * dimBundle + MASK_OFFSET(gaudi2::block_tpc_tensor, dim_0_size),
            MASK_SIZE(gaudi2::block_tpc_tensor) * dimBundle + MASK_OFFSET(gaudi2::block_tpc_tensor, dim_4_stride) + 1);

        // The below fields are in structure of array format in the HW regs, so they are strided
        uint32_t* dimSize   = &(tpcTensor[dimBundle].dim_0_size._raw);
        uint32_t* dimStride = &(tpcTensor[dimBundle].dim_0_stride._raw);

        auto dimSizeStrideH = (tpc_tensor::reg_dim_0_size_stride_high*) &(tpcTensor[dimBundle].dim_0_size_stride_high._raw);

        for (unsigned dimOffset = 0; dimOffset < gaudi2::TpcDesc::c_max_tensor_dims; ++dimOffset)
        {
            unsigned dim = dimBundle * gaudi2::TpcDesc::c_max_tensor_dims + dimOffset;
            if (dim >= tensorDim)
            {
                // The tensor AGU will always calculate address using all 5 dimensions even when
                // valid_dim_mask represents less than 5. In case the user accidentally placed coordinates beyond
                // valid_dim_mask we program the descriptor to ignore them.
                // [CID: 48078, CID:48081] Intentional - dimSize and dimArray are pointers to struct members of a struct
                // of type block_tpc_tensor_shared, which is in structure of array format, thus accessed as an array.
                dimSize[2 * dimOffset]   = 1;
                dimStride[2 * dimOffset] = 0;
                if (isIRF44)
                {
                    dimSizeStrideH[dimOffset].size_high   = 0;
                    dimSizeStrideH[dimOffset].stride_high = 0;
                }
            }
            else
            {
                dimSize[2 * dimOffset]   = sizes[dim] & 0xffffffff;
                dimStride[2 * dimOffset] = elemNStrides[dim] & 0xffffffff;
                if (isIRF44)
                {
                    dimSizeStrideH[dimOffset].size_high   = sizes[dim] >> 32;
                    dimSizeStrideH[dimOffset].stride_high = elemNStrides[dim] >> 32;
                }
            }
        }

        if (!gcTensor->isShapeTensor() && (dimBundle == 0))
        {
            // Update the address for first tensor descriptor only (lower dims). There is no need to update the next
            // tensor descriptors (upper dims) since they are used to hold the sizes/strides only.
            tpcTensor[dimBundle].base_addr_low.v  = tensorOffset.u32[0];
            tpcTensor[dimBundle].base_addr_high.v = tensorOffset.u32[1];
        }
    }

    SET_MASK_BULK_ON(tpcMask, MASK_OFFSET(gaudi2::block_tpc_tensor, base_addr_low), MASK_OFFSET(gaudi2::block_tpc_tensor, dim_4_size_stride_high) + 1);
}

static void updateTPCPrintfTensorDesc(gaudi2::block_tpc_tensor* tpcPrintfTensor,
                                      uint64_t printfBaseAddr,
                                      int roiBlockSize,
                                      int idx)
{
    if (tpcPrintfTensor != nullptr)
    {
        ptrToInt pPrintf;

        pPrintf.u64 = printfBaseAddr + idx * roiBlockSize;
        tpcPrintfTensor->base_addr_low.v = pPrintf.u32[0];
        tpcPrintfTensor->base_addr_high.v = pPrintf.u32[1];
    }
}

void TpcDescriptorGenerator::generateCommonTpcDescriptorSection(const TPCNode&                     tpcNode,
                                                                const std::list<NodeROI>&          rois,
                                                                TpcDesc&                           desc,
                                                                ValidityMask<TpcDesc>&             descMask,
                                                                const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                                gaudi2::block_tpc_tensor**         tpcPrintfTensor,
                                                                int&                               roiBlockSize)
{
    unsigned nSRF = tpcNode.getNumParams();

    memset(desc.m_desc.srf, 0, sizeof(desc.m_desc.srf));
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, srf), 0, nSRF + 1);
    memcpy(desc.m_desc.srf, instance.kernel.scalarParams, sizeof(uint32_t) * nSRF);

    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_config),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, power_loop));
    SET_MASK_REG_ON(std::begin(descMask), MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_id_inc));
    desc.m_desc.kernel_config.aso_evict_l0            = 1;
    desc.m_desc.kernel_config.small_vlm               = tpcNode.isSmallVLMRequired();
    desc.m_desc.kernel_config.num_valid_srfs          = nSRF;
    desc.m_desc.kernel_config.rd_rate_limit_rst_token = 0x20;
    desc.m_desc.kernel_config.wr_rate_limit_rst_token = 0x6; //Note: WR rate limit is currently disabled. Keeping at reset val

    // 1 is the reset value and indicates I32 index-mode; meaning, the kernel's tensor index is 32bit.
    const bool isIRF44 = tpcNode.is44bitMode();
    desc.m_desc.kernel_config.irf_32bit_compatibility = isIRF44 ? 0 : 1;

    //Todo: do we want consecutive context IDs for TPCs? probably not.
    desc.m_desc.kernel_id.v = tpcNode.getContextId();
    LOG_TRACE(GC, "TPC node {} got context id {}", tpcNode.getNodeName(), tpcNode.getContextId());

    unsigned       descTensorCount = 0;
    auto           descTensorMask  = std::begin(descMask) + MASK_OFFSET(TpcDesc, m_tensors);
    const unsigned tensorMaskSize  = MASK_SIZE(block_tpc_tensor);
    // Writing the input only tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getInputs())
    {
        if (t->getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR) continue;
        if (t->isAuxTensor()) continue;

        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t), isIRF44);
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    // Writing the output tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getOutputs())
    {
        if (t->isDoubleStoreTensor()) continue;
        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t), isIRF44);
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    // Writing the aux tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getInputs())
    {
        if (!t->isAuxTensor()) continue;

        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t), isIRF44);
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    // Writing the double store output tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getOutputs())
    {
        if (!t->isDoubleStoreTensor()) continue;
        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t), isIRF44);
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    TensorPtr printfTensor = tpcNode.getPrintfTensor();

    if (printfTensor)
    {
        // The printf tensor index is determined by the elf header. It is not necessarily positioned right after
        // the input/output tensors
        unsigned currentPrintfPosition = tpcNode.getPrintfPosition(descTensorCount);
        descTensorMask += tensorMaskSize * (currentPrintfPosition - descTensorCount);

        fillTPCTensorDesc(printfTensor,
                          desc.m_tensors + currentPrintfPosition,
                          descTensorMask,
                          tpcNode.getPaddingValue(printfTensor),
                          isIRF44);

        *tpcPrintfTensor = &desc.m_tensors[currentPrintfPosition];

        roiBlockSize = alignSizeDown(GCFG_TPC_PRINTF_TENSOR_SIZE.value() / rois.size(), Gaudi2HalReader::instance()->getCacheLineSizeInBytes());

        LOG_DEBUG(GC, "Splitting TPC printf tensor into {} parts - each of size: {}", rois.size(), roiBlockSize);

        unsigned usedDescs = div_round_up(printfTensor->getDim(), TpcDesc::c_max_tensor_dims);

        descTensorCount += usedDescs;
        descTensorMask += tensorMaskSize * usedDescs;
    }

    HB_ASSERT(descTensorCount <= TpcDesc::c_max_tensors_nr, "Tensor overflow");
}

void TpcDescriptorGenerator::generateTpcWdDescriptors(const TPCNode&            tpcNode,
                                                      const std::list<NodeROI>& rois,
                                                      deviceAddrOffset          kernelAddr,
                                                      DescriptorsVector&        descriptors)
{
    const auto& instance = tpcNode.getInstance();

    descriptors.resize(rois.size(), DescEntry());
    auto& descEntry = descriptors.front();
    auto& desc      = descEntry.desc;
    auto& descMask  = descEntry.mask;

    // We only program the sync operation and value, but not the SOB address.
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_so),
                        MASK_OFFSET(block_sync_object, message),
                        MASK_OFFSET(block_sync_object, addr));

    desc.m_so.message.so_operation   = SyncObjOp::SYNC_OP_ADD;  // increment
    desc.m_so.message.so_write_value = 1;                       // by 1
    if (GCFG_TPC_SYNC_TRACE_EN_MASK.value()) gaudi2::setSendSyncEvents(desc.m_so.message._raw);

    ptrToInt p;
    p.u64 = kernelAddr;
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_base_address_low),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, tid_base_dim_0));
    desc.m_desc.kernel_base_address_low.v  = p.u32[0];
    desc.m_desc.kernel_base_address_high.v = p.u32[1];

    gaudi2::block_tpc_tensor* tpcPrintfTensor = nullptr;
    int                       roiBlockSize    = 0;

    generateCommonTpcDescriptorSection(tpcNode, rois, desc, descMask, instance, &tpcPrintfTensor, roiBlockSize);

    ptrToInt pPrintf {0};

    if (tpcPrintfTensor != nullptr)
    {
        pPrintf.u32[0] = tpcPrintfTensor->base_addr_low.v;
        pPrintf.u32[1] = tpcPrintfTensor->base_addr_high.v;
    }

    uint64_t printfBaseAddr = pPrintf.u64;
    int      idx            = 0;

    // For ARCs mode we skip splitToPhysicalROIs pass. Therefor we loop here on the logical ROIs
    for (const NodeROI& roi : rois)
    {
        HB_ASSERT(roi.tpcWdCtx.size() == 1, "Locality mode not supported in Gaudi2");
        auto& currentDescEntry = descriptors[idx];
        if (idx > 0)
        {
            currentDescEntry.desc = desc;
            currentDescEntry.mask = descMask;
        }
        tpc_wd_ctxt_t& tpcFwCtx = currentDescEntry.fwCtx;

        castNcopy(tpcFwCtx.ist.base_cord, roi.tpcWdCtx[0].baseCord, MAX_DIMENSIONS_NUM);
        castNcopy(tpcFwCtx.ist.box_size, roi.tpcWdCtx[0].boxSize, MAX_DIMENSIONS_NUM);
        castNcopy(tpcFwCtx.ist.grid_size, roi.tpcWdCtx[0].gridSize, MAX_DIMENSIONS_NUM);
        tpcFwCtx.shuffle_index = roi.tpcWdCtx[0].shuffleIndex;

        // for each ROI - update the printf offset within the printf tensor
        updateTPCPrintfTensorDesc(tpcPrintfTensor, printfBaseAddr, roiBlockSize, idx++);
    }
}

}  // namespace gaudi2