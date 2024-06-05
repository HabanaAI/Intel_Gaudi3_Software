#pragma once

#include "types.h"
#include "gaudi3_types.h"
#include "node_roi.h"
#include "habana_nodes/tpc_node.h"
#include "utils.h"
#include "gaudi3/gaudi3_tpc_descriptor.h"
#include "mcid_converter.h"

namespace gaudi3
{
class TpcDescriptorGenerator
{
public:
    struct McidTpcUsage
    {
        CacheMaintenanceAction                     fastCfg = NOP;
        std::vector<CacheMaintenanceAction>        srf;
        std::map<unsigned, CacheMaintenanceAction> tensorPrivateCfg; // key is tensor index in descriptor
    };

    struct McidInfo
    {
        // Use uint64_t for physical mcid and not uint16_t since this is a helper (temp) struct for MCID configuration
        // and we need to add UINT32_MAX for some operation for mid-calc alogorithm. Please see fillMcidTpcConfiguration.
        uint64_t      physicalMcid;
        unsigned      tensorIndex;
        CacheMetaData cacheMD;
    };

    struct DescEntry
    {
        gaudi3::TpcDesc               desc = {0};
        ValidityMask<gaudi3::TpcDesc> mask = {0};
        TpcFwCtxVector                fwCtxs;
        McidTpcUsage                  mcidTpcUsage;
    };
    using DescriptorsVector = llvm_vecsmall::SmallVector<DescEntry, 1>;

    static void generateTpcWdDescriptors(const TPCNode&            tpcNode,
                                         const std::list<NodeROI>& rois,
                                         deviceAddrOffset          kernelAddr,
                                         DescriptorsVector&        descriptors,
                                         McidTpcUsage&             mcidTpcUsage,
                                         McidConverter&            mcidConverter,
                                         bool                      eagerMode = false);

    static void generateCommonTpcDescriptorSection(const TPCNode&                                tpcNode,
                                                   const std::list<NodeROI>&                     rois,
                                                   gaudi3::TpcDesc&                              desc,
                                                   ValidityMask<gaudi3::TpcDesc>&                descMask,
                                                   const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                   ptrToInt                                      pPrintf,
                                                   TensorDescGaudi3**                            tpcPrintfTensor,
                                                   int&                                          roiBlockSize,
                                                   McidTpcUsage&                                 mcidTpcUsage,
                                                   McidConverter&                                mcidConverter,
                                                   bool                                          eagerMode);

    static void fillMcidTpcConfiguration(const TPCNode&         tpcNode,
                                         TpcDesc&               desc,
                                         ValidityMask<TpcDesc>& descMask,
                                         std::vector<McidInfo>& mcidTpcInfo,
                                         McidTpcUsage&          mcidTpcUsage);
};

}  // namespace gaudi3
