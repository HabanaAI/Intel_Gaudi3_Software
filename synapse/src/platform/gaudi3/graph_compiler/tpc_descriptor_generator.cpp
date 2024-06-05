#include "tpc_descriptor_generator.h"

#include "cache_types.h"
#include "descriptor_generator.h"
#include "gaudi3_graph.h"
#include "gaudi3_types.h"
#include "habana_global_conf.h"
#include "hal_reader/hal_reader.h"
#include "queue_command.h"
#include "syn_logging.h"
#include "tpc_node.h"
#include "utils.h"

namespace gaudi3
{
enum RMW_DATA_TYPE
{
    RMW_INT8    = 0x0,
    RMW_INT16   = 0x1,
    RMW_INT32   = 0x2,
    RMW_UINT8   = 0x3,
    RMW_UINT16  = 0x4,
    RMW_UINT32  = 0x5,
    RMW_BF16    = 0x6,
    RMW_FP32    = 0x7,
    RMW_FP16    = 0x8,
    RMW_FP8_152 = 0x9,
    RMW_FP8_143 = 0xA,
    RMW_INT64   = 0xB
};

static RMW_DATA_TYPE getTensorDatatype(const TensorPtr& gcTensor)
{
    switch (gcTensor->getElementType())
    {
        case syn_type_bf16:
            return RMW_BF16;

        case syn_type_single:
        case syn_type_hb_float:
        case syn_type_tf32:
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

static void getPhysicalMcid(McidConverter& mcidConverter, CacheMetaData cacheMD, uint64_t &physicalMcid)
{
    physicalMcid = 0;

    switch (cacheMD.cmAction)
    {
        case NOP:
            break;
        case DEGRADE:
            mcidConverter.convertDegrade(cacheMD.mcid, (uint16_t&)physicalMcid);
            break;
        case DISCARD:
            unsigned dummyRolloverIndication;
            mcidConverter.convertDiscard(cacheMD.mcid, (uint16_t&)physicalMcid, dummyRolloverIndication);
            break;
        default:
            HB_ASSERT(false, "Cache Maintenance Action not supportted");
    }
}

static void fillTPCTensorDesc(const TensorPtr&                gcTensor,
                              TensorDescGaudi3*               tpcTensor,
                              ValidityMask<TpcDesc>::iterator tpcMask,
                              uint32_t                        paddingValue,
                              bool                            isIRF44)
{
    static const auto MAX_SUPPORTED_DIM_FOR_TENSOR_CONFIG = 5u;
    static const unsigned blockTensorSharedOffset = MASK_OFFSET(TensorDescGaudi3, shared);

    unsigned tensorDim = gcTensor->getDim();

    HB_ASSERT(!gcTensor->isStridedOnFCD(), "stride on fcd isn't supported");

    tpcTensor->shared.padding_value.v              = paddingValue;
    tpcTensor->shared.tensor_config.data_type      = getTensorDatatype(gcTensor);
    tpcTensor->shared.tensor_config.last_dim       = gcTensor->getIndexOfMaxNonDegenerateStride();
    tpcTensor->shared.tensor_config.valid_dim_mask = 0x1f;
    tpcTensor->shared.tensor_config.t_pref_dis     = 0;  // Enables the new D$ prefetch
    tpcTensor->base.pref_stride.val                = gcTensor->getPrefetchStride();

    // Enable/Disable RMW (reduction)
    tpcTensor->shared.tensor_config.rmw_set = gcTensor->isReductionEnabled();
    tpcTensor->shared.tensor_config.rmw_op  = gcTensor->getReductionOperation();

    tpcTensor->shared.tensor_config.dup_oob = 0;  //@TODO should be updated, currently set with default value
    tpcTensor->shared.tensor_config.l0cd    = 0;  //@TODO should be updated, currently set with default value

    NSizeArray   sizes        = getTpcDescNSizesInElements(*gcTensor);
    NStrideArray elemNStrides = getTpcDescNStridesInElements(*gcTensor);

    for (uint32_t dimBundle = 0; dimBundle < div_round_up(tensorDim, gaudi3::TpcDesc::c_max_tensor_dims); dimBundle++)
    {
        SET_MASK_BULK_ON(
            tpcMask,
            MASK_SIZE(TensorDescGaudi3) * dimBundle +
                blockTensorSharedOffset + MASK_OFFSET(gaudi3::block_tpc_tensor_shared, dim_0_size),
            MASK_SIZE(TensorDescGaudi3) * dimBundle +
                blockTensorSharedOffset + MASK_OFFSET(gaudi3::block_tpc_tensor_shared, dim_4_stride) + 1);

        // The below fields are in structure of array format in the HW regs, so they are strided
        uint32_t* dimSize   = &(tpcTensor[dimBundle].shared.dim_0_size._raw);
        uint32_t* dimStride = &(tpcTensor[dimBundle].shared.dim_0_stride._raw);

        auto dimSizeStrideH = (tpc_tensor_shared::reg_dim_0_size_stride_h*) &(tpcTensor[dimBundle].shared.dim_0_size_stride_h._raw);

        for (unsigned dimOffset = 0; dimOffset < gaudi3::TpcDesc::c_max_tensor_dims; ++dimOffset)
        {
            unsigned dim = dimBundle * gaudi3::TpcDesc::c_max_tensor_dims + dimOffset;
            if (dim >= tensorDim)
            {
                // The tensor AGU will always calculate address using all 5 dimensions even when
                // valid_dim_mask represents less than 5. In case the user accidentally placed coordinates beyond
                // valid_dim_mask we program the descriptor to ignore them.
                // [CID: 48072, CID:48105] Intentional - dimSize and dimArray are pointers to struct members of a struct
                // of type block_tpc_tensor_shared, which is in structure of array format, thus accessed as an array.
                dimSize[2 * dimOffset]   = 1;
                dimStride[2 * dimOffset] = 0;
                if (isIRF44)
                {
                    dimSizeStrideH[dimOffset].size_h   = 0;
                    dimSizeStrideH[dimOffset].stride_h = 0;
                }
            }
            else
            {
                dimSize[2 * dimOffset]   = sizes[dim] & 0xffffffff;
                dimStride[2 * dimOffset] = elemNStrides[dim] & 0xffffffff;
                if (isIRF44)
                {
                    dimSizeStrideH[dimOffset].size_h   = sizes[dim] >> 32;
                    dimSizeStrideH[dimOffset].stride_h = elemNStrides[dim] >> 32;
                }
            }
        }

        if (!gcTensor->isShapeTensor() && (dimBundle == 0))
        {
            // Update the address for first tensor descriptor only (lower dims). There is no need to update the next
            // tensor descriptors (upper dims) since they are used to hold the sizes/strides only.
            ptrToInt p;
            p.u64 = gcTensor->getTensorOffset();
            tpcTensor[dimBundle].base.base_addr_low.v  = p.u32[0];
            tpcTensor[dimBundle].base.base_addr_high.v = p.u32[1];
        }

    }

    SET_MASK_BULK_ON(tpcMask,
                     MASK_OFFSET(gaudi3::block_tpc_tensor_base, base_addr_low),
                     MASK_OFFSET(gaudi3::block_tpc_tensor_base, pref_stride) + 1);
    SET_MASK_BULK_ON(tpcMask,
                     blockTensorSharedOffset + MASK_OFFSET(gaudi3::block_tpc_tensor_shared, padding_value),
                     blockTensorSharedOffset + MASK_OFFSET(gaudi3::block_tpc_tensor_shared, hbw_axi_cfg) + 1);
}

static void updateTPCPrintfTensorDesc(TensorDescGaudi3* tpcPrintfTensor,
                                      ptrToInt&         pPrintf,
                                      uint64_t          printfBaseAddr,
                                      int               roiBlockSize,
                                      int               idx)
{
    if (tpcPrintfTensor != nullptr && pPrintf.u64 > 0)
    {
        pPrintf.u64                            = printfBaseAddr + idx * roiBlockSize;
        tpcPrintfTensor->base.base_addr_low.v  = pPrintf.u32[0];
        tpcPrintfTensor->base.base_addr_high.v = pPrintf.u32[1];
    }
}

void TpcDescriptorGenerator::fillMcidTpcConfiguration(const TPCNode&            tpcNode,
                                                      TpcDesc&                  desc,
                                                      ValidityMask<TpcDesc>&    descMask,
                                                      std::vector<McidInfo>&    mcidTpcInfo,
                                                      McidTpcUsage&             mcidTpcUsage)
{
    /************************************************************************************
     *                    Sort all MCIDs according to frequency                         *
     ************************************************************************************/

    // Keep in map all physical MCIDs and the number of their appearances in the descriptor
    std::unordered_map<uint64_t, uint8_t> mcidsMap;

    for (unsigned i = 0; i < mcidTpcInfo.size(); i++)
    {
        if (mcidTpcInfo[i].cacheMD.cmAction == DISCARD)
        {
            // Since we can get same physical mcid for both DISCARD and DEGRADE ops we need to distinguish them in the mcid sort
            mcidTpcInfo[i].physicalMcid += UINT32_MAX;
        }
        if (mcidsMap.find(mcidTpcInfo[i].physicalMcid) == mcidsMap.end())
        {
            mcidsMap[mcidTpcInfo[i].physicalMcid] = 1; // First time we see this physical mcid
        }
        else
        {
            mcidsMap[mcidTpcInfo[i].physicalMcid]++; // This physical MCID was already seen before
        }
    }

    // Sort the array according to the map info. Most frequent MCID will be at the beginning
    sort(mcidTpcInfo.begin(), mcidTpcInfo.end() ,[&mcidsMap](McidInfo mcid1, McidInfo mcid2)
    {
        if (mcidsMap[mcid1.physicalMcid] != mcidsMap[mcid2.physicalMcid])
        {
            return mcidsMap[mcid1.physicalMcid] > mcidsMap[mcid2.physicalMcid];
        }
        return mcid1.physicalMcid > mcid2.physicalMcid;
    });

    // Move 0 to the end (we want to configure mcid=0 therefore we do not remove it from array,
    // BUT we want to use the fast cfg and SRFs for non zero mcid first)
    for (unsigned i = 0; i < mcidTpcInfo.size(); i++)
    {
        if (mcidTpcInfo[i].physicalMcid == 0)
        {
            auto it = mcidTpcInfo.begin() + i;
            std::rotate(it, it + mcidsMap[0], mcidTpcInfo.end());
            break;
        }
    }

    /************************************************************************************
     *                             Configure FAST_CFG first                             *
     ************************************************************************************/

    // This is a running index on mcidInfo vector after it was sorted by physical MCID frequency
    unsigned mcidIndex = 0;

    if (GCFG_TPC_MCID_CONFIG_MASK.value() & 0x1)
    {
        CacheMaintenanceAction cmAction = mcidTpcInfo[mcidIndex].cacheMD.cmAction;
        uint64_t physicalMcid = mcidTpcInfo[mcidIndex].physicalMcid;

        // TPC usage
        mcidTpcUsage.fastCfg = cmAction;

        // MCID
        desc.m_desc.mcid_fast_cfg.mcid = cmAction == DISCARD ? physicalMcid - UINT32_MAX : physicalMcid;

        // MASK
        uint16_t mask = 0;
        unsigned nextMcidIndex = mcidsMap[mcidTpcInfo[mcidIndex].physicalMcid];
        for (; mcidIndex < nextMcidIndex; mcidIndex++)
        {
            mask |= 1 << (mcidTpcInfo[mcidIndex].tensorIndex);
        }
        desc.m_desc.mcid_fast_cfg.mask = mask;

        // Set validity mask
        SET_MASK_REG_ON(std::begin(descMask),
                    MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, mcid_fast_cfg));

        LOG_DEBUG(CACHE_MAINT, "TPC node {} FAST_CFG CM configuration: mcid = {}, mask = 0x{:x}, cmAction {}",
                  tpcNode.getNodeName(),
                  desc.m_desc.mcid_fast_cfg.mcid,
                  desc.m_desc.mcid_fast_cfg.mask,
                  mcidTpcUsage.fastCfg);
    }

    /************************************************************************************
     *                                    Configure SRFs                                *
     ************************************************************************************/

    if (GCFG_TPC_MCID_CONFIG_MASK.value() & 0x2)
    {
        // Get SRF base index
        unsigned numSRFs = Gaudi3HalReader::instance()->getNumFastConfigMcidSRFs();
        unsigned baseSrfId = Gaudi3HalReader::instance()->getNumSRFs() - numSRFs;

        for (unsigned srfId = baseSrfId; srfId < (baseSrfId + numSRFs) && mcidIndex < mcidTpcInfo.size(); srfId++)
        {
            CacheMaintenanceAction cmAction = mcidTpcInfo[mcidIndex].cacheMD.cmAction;
            uint64_t physicalMcid = mcidTpcInfo[mcidIndex].physicalMcid;

            // TPC usage
            mcidTpcUsage.srf.push_back(cmAction);

            // MCID
            uint32_t mcid = cmAction == DISCARD ? physicalMcid - UINT32_MAX : physicalMcid;

            // MASK
            uint32_t mask = 0;
            unsigned nextMcidIndex = mcidIndex + mcidsMap[physicalMcid];
            for (; mcidIndex < nextMcidIndex; mcidIndex++)
            {
                mask |= 1 << (mcidTpcInfo[mcidIndex].tensorIndex);
            }
            mask = mask << 16;

            desc.m_desc.srf[srfId].v = mcid + mask;

            LOG_DEBUG(CACHE_MAINT, "TPC node {} SRF[{}] CM configuration: mcid = {}, mcid+mask = 0x{:x}, cmAction {}",
                      tpcNode.getNodeName(),
                      srfId,
                      mcid,
                      desc.m_desc.srf[srfId].v,
                      mcidTpcUsage.srf[srfId - baseSrfId]);
        }

        // Set validity mask
        SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc) +
                         MASK_OFFSET(block_tpc_non_tensor_descriptor, srf),
                         baseSrfId,
                         baseSrfId + numSRFs + 1);
    }

    /************************************************************************************
     *                        Configure Private Tensor's MCIDs                          *
     ************************************************************************************/

    for (; mcidIndex < mcidTpcInfo.size(); mcidIndex++)
    {
        unsigned tensorIdInDesc   = mcidTpcInfo[mcidIndex].tensorIndex;
        TensorDescGaudi3* tpcDesc = desc.m_tensors + tensorIdInDesc;

        CacheMaintenanceAction cmAction = mcidTpcInfo[mcidIndex].cacheMD.cmAction;
        uint64_t physicalMcid           = mcidTpcInfo[mcidIndex].physicalMcid;

        // MCID
        tpcDesc->shared.hbw_axi_cfg.mcid = cmAction == DISCARD ? physicalMcid - UINT32_MAX : physicalMcid;

        // TPC usage
        mcidTpcUsage.tensorPrivateCfg[tensorIdInDesc] = cmAction;

        // Validity mask is set at the end of fillTPCTensorDesc

        LOG_DEBUG(CACHE_MAINT, "TPC node {} Private tensor [{}] CM configuration: cmAction = {}",
                  tpcNode.getNodeName(),
                  tensorIdInDesc,
                  cmAction);
    }

    // Set alloc policy and class for all tensors (regardless of fast config and srf use)
    for (unsigned i = 0; i < mcidTpcInfo.size(); i++)
    {
        unsigned tensorIdInDesc   = mcidTpcInfo[i].tensorIndex;
        TensorDescGaudi3* tpcDesc = desc.m_tensors + tensorIdInDesc;

        // Alloc policy
        const auto  cacheDirective = Gaudi3HalReader::instance()->getCacheDirectiveBits(mcidTpcInfo[i].cacheMD.cacheDirective);
        tpcDesc->shared.hbw_axi_cfg.arcache = cacheDirective;
        tpcDesc->shared.hbw_axi_cfg.awcache = cacheDirective;

        // CALSS
        tpcDesc->shared.hbw_axi_cfg.clas    = mcidTpcInfo[i].cacheMD.cacheClass;

        // Validity mask is set at the end of fillTPCTensorDesc

        LOG_DEBUG(CACHE_MAINT, "TPC node {} Private tensor [{}] CM configuration: arcache = {}, awcache = {}, class = {}, mcid = {}",
                  tpcNode.getNodeName(),
                  tensorIdInDesc,
                  tpcDesc->shared.hbw_axi_cfg.arcache,
                  tpcDesc->shared.hbw_axi_cfg.awcache,
                  tpcDesc->shared.hbw_axi_cfg.clas,
                  tpcDesc->shared.hbw_axi_cfg.mcid);
    }
}

void TpcDescriptorGenerator::generateCommonTpcDescriptorSection(const TPCNode&                                tpcNode,
                                                                const std::list<NodeROI>&                     rois,
                                                                TpcDesc&                                      desc,
                                                                ValidityMask<TpcDesc>&                        descMask,
                                                                const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                                ptrToInt                                      pPrintf,
                                                                TensorDescGaudi3** tpcPrintfTensor,
                                                                int&               roiBlockSize,
                                                                McidTpcUsage&      mcidTpcUsage,
                                                                McidConverter&     mcidConverter,
                                                                bool               eagerMode)
{
    unsigned nSRF = tpcNode.getNumParams();

    memset(desc.m_desc.srf, 0, sizeof(desc.m_desc.srf));
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc) +
                         MASK_OFFSET(block_tpc_non_tensor_descriptor, srf),
                     0,
                     nSRF + 1);
    memcpy(desc.m_desc.srf, instance.kernel.scalarParams, sizeof(uint32_t) * nSRF);

    HB_ASSERT(nSRF < (Gaudi3HalReader::instance()->getNumSRFs() - Gaudi3HalReader::instance()->getNumFastConfigMcidSRFs()), "Kernel uses dedicated MCID SRFs");

    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_id),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, srf));
    SET_MASK_REG_ON(std::begin(descMask),
                    MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_config));
    desc.m_desc.kernel_config.aso_evict_l0            = 1;
    desc.m_desc.kernel_config.small_vlm               = tpcNode.isSmallVLMRequired();
    desc.m_desc.kernel_config.num_valid_srfs          =
        std::max(nSRF, Gaudi3HalReader::instance()->getTPCMinSRF());  // workaround for TPC trace CDC issue
    desc.m_desc.kernel_config.rd_rate_limit_rst_token = 0x20;
    // Note: WR rate limit is currently disabled. Keeping at reset val
    desc.m_desc.kernel_config.wr_rate_limit_rst_token = 0x6;

    // 1 is the reset value and indicates I32 index-mode; meaning, the kernel's tensor index is 32bit.
    // Set to 0 to switch to I44 index-mode when a dimension in the tensor is larger than 4GB.
    const bool isIRF44 = tpcNode.is44bitMode();
    desc.m_desc.kernel_config.irf_32bit_compatibilty = isIRF44 ? 0 : 1;

    // active thrd should be 1 even when SMT mode is not active. when activating SMT mode, need to update this value
    desc.m_desc.active_thrd.active_thrd = 1;
    // Todo: do we want consecutive context IDs for TPCs? probably not.
    desc.m_desc.kernel_id.v = tpcNode.getContextId();
    LOG_TRACE(GC, "TPC node {} got context id {}", tpcNode.getNodeName(), tpcNode.getContextId());

    SET_MASK_REG_ON(std::begin(descMask),
                    MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, dcache_axi_cfg));
    SET_MASK_REG_ON(std::begin(descMask),
                    MASK_OFFSET(TpcDesc, m_desc) + MASK_OFFSET(block_tpc_non_tensor_descriptor, icache_axi_cfg));
    const auto nonTensorCacheDirective = GCFG_DEFAULT_CACHE_DIRECTIVE.value() == SkipCache ?
        Gaudi3HalReader::instance()->getCacheDirectiveBits(SkipCache) :
        Gaudi3HalReader::instance()->getCacheDirectiveBits(NoAllocate);
    desc.m_desc.dcache_axi_cfg.awcache = nonTensorCacheDirective;
    desc.m_desc.icache_axi_cfg.arcach  = nonTensorCacheDirective;

    unsigned       descTensorCount = 0;
    auto           descTensorMask  = std::begin(descMask) + MASK_OFFSET(TpcDesc, m_tensors);
    const unsigned tensorMaskSize  = MASK_SIZE(TensorDescGaudi3);

    std::vector<McidInfo> mcidTpcInfo;

    int descTensorIndex = -1;

    auto addMcidInfo = [&mcidTpcInfo, &mcidConverter](const CacheMetaData& cacheMD, unsigned tensorIndex) {
        McidInfo info;
        info.tensorIndex = tensorIndex;
        info.cacheMD     = cacheMD;
        getPhysicalMcid(mcidConverter, cacheMD, info.physicalMcid);
        mcidTpcInfo.push_back(info);
    };

    // TODO need to properly set the cache directive for Eager
    // currently using the default.
    static const CacheMetaData defaultCacheDirective;
    // Writing the input only tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getInputs())
    {
        descTensorIndex++;
        if (t->isTensorAuxOrShapeOutput()) continue;

        const CacheMetaData& currentCacheMD =
            eagerMode ? defaultCacheDirective : rois.front().inputsCacheMetaData[descTensorIndex];

        fillTPCTensorDesc(t,
                          desc.m_tensors + descTensorCount,
                          descTensorMask,
                          tpcNode.getPaddingValue(t),
                          isIRF44);

        addMcidInfo(currentCacheMD, descTensorCount);

        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    descTensorIndex = 0;
    // Writing the output tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getOutputs())
    {
        const CacheMetaData& currentCacheMD =
            eagerMode ? defaultCacheDirective : rois.front().outputsCacheMetaData[descTensorIndex];

        fillTPCTensorDesc(t,
                          desc.m_tensors + descTensorCount,
                          descTensorMask,
                          tpcNode.getPaddingValue(t),
                          isIRF44);

        addMcidInfo(currentCacheMD, descTensorCount);
        descTensorIndex++;

        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    descTensorIndex = -1;
    // Writing the aux tensors to the TPCDescriptor
    for (const TensorPtr& t : tpcNode.getInputs())
    {
        descTensorIndex++;
        if (!t->isAuxTensor()) continue;

        const CacheMetaData& currentCacheMD =
            eagerMode ? defaultCacheDirective : rois.front().inputsCacheMetaData[descTensorIndex];

        fillTPCTensorDesc(t,
                          desc.m_tensors + descTensorCount,
                          descTensorMask,
                          tpcNode.getPaddingValue(t),
                          isIRF44);

        addMcidInfo(currentCacheMD, descTensorCount);

        unsigned usedDescs = div_round_up(t->getDim(), TpcDesc::c_max_tensor_dims);
        descTensorCount += usedDescs;
        descTensorMask += (tensorMaskSize * usedDescs);
    }

    TensorPtr gcTensor = tpcNode.getPrintfTensor();

    if (gcTensor)
    {
        // The printf tensor index is determined by the elf header. It is not necessarily positioned right after
        // the input/output tensors
        unsigned currentPrintfPosition = tpcNode.getPrintfPosition(descTensorCount);
        descTensorMask += tensorMaskSize * (currentPrintfPosition - descTensorCount);

        fillTPCTensorDesc(gcTensor,
                          desc.m_tensors + currentPrintfPosition,
                          descTensorMask,
                          tpcNode.getPaddingValue(gcTensor),
                          isIRF44);

        *tpcPrintfTensor = &desc.m_tensors[currentPrintfPosition];

        roiBlockSize    = alignSizeDown(GCFG_TPC_PRINTF_TENSOR_SIZE.value() / rois.size(),
                                     Gaudi3HalReader::instance()->getCacheLineSizeInBytes());
        LOG_DEBUG(GC, "Splitting TPC printf tensor into {} parts - each of size: {}", rois.size(), roiBlockSize);
        unsigned usedDescs = div_round_up(gcTensor->getDim(), TpcDesc::c_max_tensor_dims);

        descTensorCount += usedDescs;
        descTensorMask += tensorMaskSize * usedDescs;
    }

    fillMcidTpcConfiguration(tpcNode, desc, descMask, mcidTpcInfo, mcidTpcUsage);

    HB_ASSERT(descTensorCount <= TpcDesc::c_max_tensors_nr, "Tensor overflow");
}

void TpcDescriptorGenerator::generateTpcWdDescriptors(const TPCNode&            tpcNode,
                                                      const std::list<NodeROI>& rois,
                                                      deviceAddrOffset          kernelAddr,
                                                      DescriptorsVector&        descriptors,
                                                      McidTpcUsage&             mcidTpcUsage,
                                                      McidConverter&            mcidConverter,
                                                      bool                      eagerMode)
{
    const auto& instance = tpcNode.getInstance();

    descriptors.resize(rois.size(), DescEntry());
    auto& descEntry = descriptors.front();
    auto& desc      = descEntry.desc;
    auto& descMask  = descEntry.mask;

    // In ARC mode 3 (WD + new Sync Scheme) we only program the sync operation and value, but not the SOB address.
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_so_th0),
                     MASK_OFFSET(block_sync_object, message),
                     MASK_OFFSET(block_sync_object, addr));

    desc.m_so_th0.message.so_operation   = SyncObjOp::SYNC_OP_ADD;  // increment
    desc.m_so_th0.message.so_write_value = 1;                       // by 1
    if (GCFG_TPC_SYNC_TRACE_EN_MASK.value()) gaudi3::setSendSyncEvents(desc.m_so_th0.message._raw);

    ptrToInt p;
    p.u64 = kernelAddr;
    SET_MASK_BULK_ON(std::begin(descMask) + MASK_OFFSET(TpcDesc, m_desc),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_base_address_low),
                     MASK_OFFSET(block_tpc_non_tensor_descriptor, kernel_config));
    desc.m_desc.kernel_base_address_low.v  = p.u32[0];
    desc.m_desc.kernel_base_address_high.v = p.u32[1];

    ptrToInt pPrintf;
    pPrintf.u64                       = 0;
    TensorDescGaudi3* tpcPrintfTensor = nullptr;
    int               roiBlockSize    = 0;

    generateCommonTpcDescriptorSection(tpcNode,
                                       rois,
                                       desc,
                                       descMask,
                                       instance,
                                       pPrintf,
                                       &tpcPrintfTensor,
                                       roiBlockSize,
                                       mcidTpcUsage,
                                       mcidConverter,
                                       eagerMode);

    uint64_t printfBaseAddr = pPrintf.u64;
    int      idx            = 0;

    // For ARCs mode we skip splitToPhysicalROIs pass. Therefor we loop here on the logical ROIs
    for (const NodeROI& roi : rois)
    {
        auto& currentDescEntry = descriptors[idx];
        if (idx > 0)
        {
            currentDescEntry.desc         = desc;
            currentDescEntry.mask         = descMask;
            currentDescEntry.mcidTpcUsage = mcidTpcUsage;
        }
        TpcFwCtxVector& tpcFwCtxs = currentDescEntry.fwCtxs;
        tpcFwCtxs.resize(roi.tpcWdCtx.size());
        for (size_t i = 0; i < roi.tpcWdCtx.size(); ++i)
        {
            tpc_wd_ctxt_t& tpcFwCtx = tpcFwCtxs[i];
            castNcopy(tpcFwCtx.ist.base_cord, roi.tpcWdCtx[i].baseCord, MAX_DIMENSIONS_NUM);
            castNcopy(tpcFwCtx.ist.box_size, roi.tpcWdCtx[i].boxSize, MAX_DIMENSIONS_NUM);
            castNcopy(tpcFwCtx.ist.grid_size, roi.tpcWdCtx[i].gridSize, MAX_DIMENSIONS_NUM);
            castNcopy(tpcFwCtx.ist.dim_slices, roi.tpcWdCtx[i].dimSlices, MAX_DIMENSIONS_NUM);
            tpcFwCtx.shuffle_index = roi.tpcWdCtx[i].shuffleIndex;
            tpcFwCtx.tpc_wd_mode   = TPC_WD_ST_MODE;  // until we support SMT4 we must use the ST (single thread) mode
            tpcFwCtx.wd_type       = roi.tpcWdCtx.size() == 1 ? TPC_WD_USING_GLBL_INDEX : TPC_WD_USING_DCORE_INDEX;
        }

        // for each ROI - update the printf offset within the printf tensor
        updateTPCPrintfTensorDesc(tpcPrintfTensor, pPrintf, printfBaseAddr, roiBlockSize, idx++);

        // we send to fillTPCTensorDesc cache meta data for one roi, assuming it's the same for all rois
        // this assumtion is wrong if we decide to split to Dcores and Logical Rois, that's why we assert
        HB_ASSERT(rois.front().inputsCacheMetaData.size() == roi.inputsCacheMetaData.size() &&
                  rois.front().outputsCacheMetaData.size() == roi.outputsCacheMetaData.size(),
                  "when using locality can't split to Rois");
        for (size_t i = 0; i < roi.inputsCacheMetaData.size(); i++)
        {
            HB_ASSERT(rois.front().inputsCacheMetaData[i] == roi.inputsCacheMetaData[i],
                      "when using locality can't split to Rois");
        }
        for (size_t i = 0; i < roi.outputsCacheMetaData.size(); i++)
        {
            HB_ASSERT(rois.front().outputsCacheMetaData[i] == roi.outputsCacheMetaData[i],
                      "when using locality can't split to Rois");
        }
    }
}

}  // namespace gaudi3