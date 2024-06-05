#pragma once

#include "types.h"
#include "gaudi2_types.h"
#include "node_roi.h"
#include "habana_nodes/tpc_node.h"
#include "utils.h"

namespace gaudi2
{
class TpcDescriptorGenerator
{
public:
    struct DescEntry
    {
        gaudi2::TpcDesc               desc  = {0};
        ValidityMask<gaudi2::TpcDesc> mask  = {0};
        tpc_wd_ctxt_t                 fwCtx = {0};
    };
    using DescriptorsVector = llvm_vecsmall::SmallVector<DescEntry, 1>;

    static void generateTpcWdDescriptors(const TPCNode&            node,
                                         const std::list<NodeROI>& rois,
                                         deviceAddrOffset          kernelAddr,
                                         DescriptorsVector&        descriptors);

    static void generateCommonTpcDescriptorSection(const TPCNode&                                tpcNode,
                                                   const std::list<NodeROI>&                     rois,
                                                   gaudi2::TpcDesc&                              desc,
                                                   ValidityMask<gaudi2::TpcDesc>&                descMask,
                                                   const tpc_lib_api::HabanaKernelInstantiation& instance,
                                                   gaudi2::block_tpc_tensor**                    tpcPrintfTensor,
                                                   int&                                          roiBlockSize);
};

}  // namespace gaudi2
