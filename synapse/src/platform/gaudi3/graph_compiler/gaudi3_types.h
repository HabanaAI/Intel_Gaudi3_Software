#pragma once

#include "descriptor_wrapper.h"
#include "gaudi3/gaudi3_tpc_descriptor.h"
#include "gaudi3/asic_reg_structs/rot_desc_regs.h"
#include "gaudi3/mme.h"
#include "include/gaudi3/mme_descriptor_generator.h"
#include "include/gaudi3/gaudi3_defs.h"
#include "hal_conventions.h"
#include "gaudi3/gaudi3_arc_eng_packets.h"

namespace gaudi3
{
using MmeDesc     = gaudi3::Mme::Desc;
using TpcDesc     = struct Gaudi3TpcDesc;
using RotatorDesc = struct block_rot_desc;
}  // namespace gaudi3

// Template specialization that couples the Gaudi3's descriptor type to the FW Context type. In must come
// after the declaration of the descriptor type and before the declaration of the descriptor wrapper type.
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi3::MmeDesc, mme_wd_ctxt_t);
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi3::TpcDesc, tpc_wd_ctxt_t);
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi3::RotatorDesc, rot_wd_ctxt_t);

namespace gaudi3
{
using EMmeOpType             = MmeCommon::EMmeOpType;
using MmeConv                = MmeCommon::MmeConv;
using MmeTensorView          = MmeCommon::MmeTensorView;
using EMmeOperand            = MmeCommon::EMmeOperand;
using EMmeDataType           = MmeCommon::EMmeDataType;
using EMmeGeometry           = MmeCommon::EMmeGeometry;
using EMmePattern            = MmeCommon::EMmePattern;
using MmeLayerParams         = MmeCommon::MmeLayerParams;
using MmeActivation          = gaudi3::MmeActivation;
using Activations            = gaudi3::ActivationVec;
using MmeDescriptorGenerator = gaudi3::MmeDescriptorGenerator;

using MmeDescWrapper         = DescriptorWrapper<MmeDesc>;
using MmeDescriptorsWrappers = llvm_vecsmall::SmallVector<MmeDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using TpcDescWrapper         = DescriptorWrapper<TpcDesc>;
using TpcDescriptorsWrappers = llvm_vecsmall::SmallVector<TpcDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using RotatorDescWrapper         = DescriptorWrapper<RotatorDesc>;
using RotatorDescriptorsWrappers = llvm_vecsmall::SmallVector<RotatorDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using MmeLayerParamsPtr         = std::shared_ptr<MmeLayerParams>;
using MmeDescriptorGeneratorPtr = std::shared_ptr<MmeDescriptorGenerator>;

using TpcFwCtxVector = llvm_vecsmall::SmallVector<tpc_wd_ctxt_t, MAX_NUM_DCORES>;

using Overlap = Overlap<gaudi3::LOGICAL_QUEUE_MAX_ID>;

struct McidMmeUsage
{
   CacheMaintenanceAction operandA = NOP;
   CacheMaintenanceAction operandB = NOP;
   CacheMaintenanceAction operandC = NOP;
};
}  // namespace gaudi3
