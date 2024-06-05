#include <habana_global_conf.h>
#include <algorithm>
#include <numeric>

#include "gaudi_types.h"
#include "descriptor_generator.h"
#include "synapse_common_types.h"
#include "tpc_node.h"
#include "types.h"
#include "utils.h"

#include "graph_compiler/compilation_hal_reader.h"

using namespace gaudi;

enum RMW_DATA_TYPE
{
    RMW_INT8   = 0x0,
    RMW_INT16  = 0x1,
    RMW_INT32  = 0x2,
    RMW_UINT8  = 0x3,
    RMW_UINT16 = 0x4,
    RMW_UINT32 = 0x5,
    RMW_BF16   = 0x6,
    RMW_FP32   = 0x7
};

static RMW_DATA_TYPE getTensorDatatype(const pTensor& gcTensor)
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

        case syn_type_uint8:
            return RMW_UINT8;

        case syn_type_fixed:
            return RMW_INT8;

        case syn_type_uint16:
            return RMW_UINT16;

        case syn_type_uint32:
        case syn_type_uint64:
            return RMW_UINT32;

        case syn_type_fp16:
            return RMW_UINT16;

        default:
            HB_ASSERT(false, "{}: Got unknown data type", __func__);
            break;
    }
    return RMW_INT8;
}

static void fillTPCTensorDesc(const pTensor&         gcTensor,
                              block_tpc_tensor*      tpcTensor,
                              ValidityMask<TpcDesc>::iterator tensorMask,
                              uint32_t               paddingValue)
{
    static const auto MAX_SUPPORTED_DIM_FOR_TENSOR_CONFIG = 5u;

    unsigned tensorDim     = gcTensor->getDim();
    unsigned configLastDim = gcTensor->getIndexOfMaxNonDegenerateStride();

    tpcTensor->padding_value.v              = paddingValue;
    tpcTensor->tensor_config.data_type      = getTensorDatatype(gcTensor);
    tpcTensor->tensor_config.last_dim       = configLastDim;
    tpcTensor->tensor_config.valid_dim_mask = 0x1f;

    // Enable/Disable RMW (SRAM reduction)
    tpcTensor->tensor_config.rmw_set = gcTensor->isReductionEnabled();
    tpcTensor->tensor_config.rmw_op  = gcTensor->getReductionOperation();

    HB_ASSERT(!gcTensor->isStridedOnFCD(), "stride on fcd isn't supported");

    NSizeArray   sizes        = getTpcDescNSizesInElements(*gcTensor);
    NStrideArray elemNStrides = getTpcDescNStridesInElements(*gcTensor);

    for (uint32_t dimBundle = 0; dimBundle < div_round_up(tensorDim, TpcDesc::c_max_tensor_dims); dimBundle++)
    {
        SET_MASK_BULK_ON(tensorMask,
                         MASK_SIZE(block_tpc_tensor) * dimBundle + MASK_OFFSET(block_tpc_tensor, dim_0_size),
                         MASK_SIZE(block_tpc_tensor) * dimBundle + MASK_OFFSET(block_tpc_tensor, dim_4_stride) + 1);
        // TPC expects element stride between dimensions
        // The below fields are in structure of array format in the HW regs, so they are strided
        uint32_t* dimSize   = &(tpcTensor[dimBundle].dim_0_size._raw);
        uint32_t* dimStride = &(tpcTensor[dimBundle].dim_0_stride._raw);

        for (unsigned dimOffset = 0; dimOffset < TpcDesc::c_max_tensor_dims; ++dimOffset)
        {
            unsigned dim = dimBundle * TpcDesc::c_max_tensor_dims + dimOffset;
            // The tensor AGU will always calculate address using all 5 dimensions even when
            // valid_dim_mask represents less than 5. In case the user accidentally placed coordinates beyond
            // valid_dim_mask we program the descriptor to ignore them.
            if (dim >= tensorDim)
            {
                dimSize[2 * dimOffset]   = 1;
                dimStride[2 * dimOffset] = 0;
            }
            else
            {
                dimSize[2 * dimOffset]   = sizes[dim];
                dimStride[2 * dimOffset] = elemNStrides[dim];
            }
        }

        if (!gcTensor->isShapeTensor() && (dimBundle == 0))
        {
            // Update the address for first tensor descriptor only (lower dims). There is no need to update the next
            // tensor descriptors (upper dims) since they are used to hold the sizes/strides only.
            ptrToInt p;
            p.u64 = gcTensor->getTensorOffset();
            tpcTensor[dimBundle].base_addr_low.v  = p.u32[0];
            tpcTensor[dimBundle].base_addr_high.v = p.u32[1];
        }
    }

    SET_MASK_BULK_ON(tensorMask,
                     MASK_OFFSET(block_tpc_tensor, base_addr_low),
                     MASK_OFFSET(block_tpc_tensor, dim_4_stride) + 1);
}

static void updateTPCPrintfTensorDesc(block_tpc_tensor* tpcPrintfTensor,
                                      ptrToInt&         pPrintf,
                                      uint64_t          printfBaseAddr,
                                      int               roiBlockSize,
                                      int               idx)
{
    if (tpcPrintfTensor != nullptr && pPrintf.u64 > 0)
    {
        pPrintf.u64                       = printfBaseAddr + idx * roiBlockSize;
        tpcPrintfTensor->base_addr_low.v  = pPrintf.u32[0];
        tpcPrintfTensor->base_addr_high.v = pPrintf.u32[1];
    }
}

void DescriptorGenerator::generateTpcDescriptors(const TPCNode&                          tpcNode,
                                                 const std::list<NodeROI>&               rois,
                                                 deviceAddrOffset                        kernelAddr,
                                                 std::list<DescAndMask<gaudi::TpcDesc>>& descriptors)
{
    const auto& instance = tpcNode.getInstance();

    TpcDesc desc;
    memset(&desc, 0, sizeof(desc));
    ValidityMask<TpcDesc> descMask {false};

    // Todo: no sync currently, we just set signalEn to 0

    // Patch the kernel address in the descriptor
    ptrToInt p;
    p.u64 = kernelAddr;
    SET_MASK_BULK_ON(std::begin(descMask), MASK_OFFSET(TpcDesc, m_so), MASK_OFFSET(TpcDesc, m_desc));
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_base_address_low),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, tid_base_dim_0));
    desc.m_desc.kernel_base_address_low.v  = p.u32[0];
    desc.m_desc.kernel_base_address_high.v = p.u32[1];

    memset((uint8_t*)&desc + offsetof(TpcDesc, m_desc.tid_base_dim_0),
           0,
           TpcDesc::c_max_tensor_dims * sizeof(uint32_t) * 2);

    // The below fields are in structure of array format in the HW regs, so their stride is 2x the expected
    uint32_t* tidOffset = &(desc.m_desc.tid_base_dim_0._raw);
    uint32_t* tidSize   = &(desc.m_desc.tid_size_dim_0._raw);

    unsigned nSRF = tpcNode.getNumParams();

    HB_ASSERT(nSRF * sizeof(uint32_t) <= sizeof(desc.m_desc.srf), "Scalar param overrun");

    memset(desc.m_desc.srf, 0, sizeof(desc.m_desc.srf));
    memcpy(desc.m_desc.srf, instance.kernel.scalarParams, sizeof(uint32_t) * nSRF);

    desc.m_desc.kernel_config.aso_evict_l0 = 1;
    desc.m_desc.kernel_config.small_vlm    = tpcNode.isSmallVLMRequired();
    desc.m_desc.kernel_config.num_valid_srfs =
        std::max(nSRF, CompilationHalReader::getHalReader()->getTPCMinSRF());  // workaround for TPC trace CDC issue
    desc.m_desc.kernel_config.rd_rate_limit_rst_token = 0x20;
    desc.m_desc.kernel_config.wr_rate_limit_rst_token =
        0x6;  // Note: WR rate limit is currently disabled. Keeping at reset val
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_config),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, srf) + nSRF + 1);

    // Todo: do we want consecutive context IDs for TPCs? probably not.
    desc.m_desc.kernel_id.v = tpcNode.getContextId();

    unsigned     descTensorCount    = 0;
    auto         descTensorMask     = std::begin(descMask) + MASK_OFFSET(TpcDesc, m_tensors);
    const size_t descTensorMaskSize = MASK_SIZE(block_tpc_tensor);

    // Writing the input only tensors to the TPCDescriptor
    int nShapeTensors = 0;

    for (const pTensor& t : tpcNode.getInputs())
    {
        if (t->getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR)
        {
            ++nShapeTensors;
            continue;
        }
        if (t->isAuxTensor()) continue;

        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t));
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorMask += descTensorMaskSize * usedDescs;
        descTensorCount += usedDescs;
    }

    // Writing the output tensors to the TPCDescriptor
    for (const pTensor& t : tpcNode.getOutputs())
    {
        if (t->isDoubleStoreTensor()) continue;

        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t));
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorMask += descTensorMaskSize * usedDescs;
        descTensorCount += usedDescs;
    }

    // Writing the aux tensors to the TPCDescriptor
    for (const pTensor& t : tpcNode.getInputs())
    {
        if (!t->isAuxTensor()) continue;

        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t));
        descTensorMask += descTensorMaskSize;
        ++descTensorCount;
    }

    // Writing the double store output tensors to the TPCDescriptor
    for (const pTensor& t : tpcNode.getOutputs())
    {
        if (!t->isDoubleStoreTensor()) continue;

        fillTPCTensorDesc(t, desc.m_tensors + descTensorCount, descTensorMask, tpcNode.getPaddingValue(t));
        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorMask += descTensorMaskSize * usedDescs;
        descTensorCount += usedDescs;
    }

    ptrToInt pPrintf;
    pPrintf.u64                       = 0;
    block_tpc_tensor* tpcPrintfTensor = nullptr;
    int               roiBlockSize    = 0;

    int printfTensorCount = 0;
    UNUSED(printfTensorCount);
    std::shared_ptr<Tensor> gcTensor = tpcNode.getPrintfTensor();
    if (gcTensor)
    {
        // The printf tensor index is determined by the elf header. It is not necessarily positioned right after
        // the input/output tensors
        unsigned currentPrintfPosition = tpcNode.getPrintfPosition(descTensorCount);
        descTensorMask += descTensorMaskSize * (currentPrintfPosition - descTensorCount);

        fillTPCTensorDesc(gcTensor,
                          desc.m_tensors + currentPrintfPosition,
                          descTensorMask,
                          tpcNode.getPaddingValue(gcTensor));

        tpcPrintfTensor = desc.m_tensors + currentPrintfPosition;
        pPrintf.u64     = getAddressFromSplitParts(tpcPrintfTensor->base_addr_high.v, tpcPrintfTensor->base_addr_low.v);
        roiBlockSize    = alignSizeDown(GCFG_TPC_PRINTF_TENSOR_SIZE.value() / rois.size(),
                                     CompilationHalReader::getHalReader()->getCacheLineSizeInBytes());
        LOG_DEBUG(GC, "Splitting TPC printf tensor into {} parts - each of size: {}", rois.size(), roiBlockSize);

        descTensorMask += descTensorMaskSize;
        ++descTensorCount;
        printfTensorCount = 1;
    }
    auto inputs      = tpcNode.getInputs();
    auto outputs     = tpcNode.getOutputs();
    auto inputCount  = std::accumulate(inputs.begin(), inputs.end(), 0, [](uint32_t x, TensorPtr t) {
        return x + div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
    });
    auto outputCount = std::accumulate(outputs.begin(), outputs.end(), 0, [](uint32_t x, TensorPtr t) {
        return x + div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
    });
    HB_ASSERT(descTensorCount + nShapeTensors == inputCount + outputCount + printfTensorCount, "Size mismatch");
    HB_ASSERT(descTensorCount <= TpcDesc::c_max_tensors_nr, "Tensor overflow");

    uint64_t printfBaseAddr = pPrintf.u64;

    int idx = 0;
    // Signaling that all the dimensions are data.
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, tid_base_dim_0),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_config));
    for (const NodeROI& roi : rois)
    {
        for (unsigned dim = 0; dim < instance.indexSpaceRank; ++dim)
        {
            tidOffset[2 * dim] = tpcNode.getNodeAnnotation().baseOffset[dim] + roi.baseOffset[dim];
            tidSize[2 * dim]   = roi.size[dim];
        }

        // for each ROI - update the printf offset within the printf tensor
        updateTPCPrintfTensorDesc(tpcPrintfTensor, pPrintf, printfBaseAddr, roiBlockSize, idx);
        idx++;

        descriptors.push_back({desc, descMask});
    }
}
