#pragma once

#include "descriptor_wrapper.h"
#include "gaudi2/gaudi2_tpc_descriptor.h"
#include "gaudi2/asic_reg_structs/dma_core_ctx_regs.h"
#include "gaudi2/asic_reg_structs/rot_desc_regs.h"
#include "gaudi2/asic_reg_structs/axuser_regs.h"
#include "gaudi2/mme.h"
#include "include/gaudi2/mme_descriptor_generator.h"
#include "include/gaudi2/gaudi2_defs.h"
#include "hal_conventions.h"
#include "gaudi2_arc_eng_packets.h"
#include "llvm/small_vector.h"

namespace gaudi2
{
struct block_axuser_dma_core_ctx
{
    struct block_axuser       axuser;
    uint32_t                  _pad2128[4];
    struct block_dma_core_ctx ctx;
};

using MmeDesc                    = Gaudi2::Mme::Desc;
using TpcDesc                    = struct Gaudi2TpcDesc;
using DmaDesc                    = struct block_axuser_dma_core_ctx;
using RotatorDesc                = struct block_rot_desc;
}  // namespace gaudi2

// Template specialization that couples the Gaudi2's descriptor type to the FW Context type. In must come
// after the declaration of the descriptor type and before the declaration of the descriptor wrapper type.
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi2::MmeDesc, mme_wd_ctxt_t);
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi2::TpcDesc, tpc_wd_ctxt_t);
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi2::DmaDesc, edma_wd_ctxt_t);
COUPLE_DESCRIPTOR_TO_FW_CTX(gaudi2::RotatorDesc, rot_wd_ctxt_t);

namespace gaudi2
{
using EMmeOpType                 = MmeCommon::EMmeOpType;
using MmeConv                    = MmeCommon::MmeConv;
using MmeTensorView              = MmeCommon::MmeTensorView;
using EMmeOperand                = MmeCommon::EMmeOperand;
using EMmeDataType               = MmeCommon::EMmeDataType;
using EMmeGeometry               = MmeCommon::EMmeGeometry;
using EMmePattern                = MmeCommon::EMmePattern;
using MmeLayerParams             = MmeCommon::MmeLayerParams;
using MmeActivation              = Gaudi2::MmeActivation;
using MmeActivations             = Gaudi2::ActivationVec;
using MmeDescriptorGenerator     = Gaudi2::MmeDescriptorGenerator;

using MmeDescWrapper             = DescriptorWrapper<MmeDesc>;
using MmeDescriptorsWrappers     = llvm_vecsmall::SmallVector<MmeDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using TpcDescWrapper             = DescriptorWrapper<TpcDesc>;
using TpcDescriptorsWrappers     = llvm_vecsmall::SmallVector<TpcDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using DmaDescWrapper             = DescriptorWrapper<DmaDesc>;
using DmaDescriptorsWrappers     = llvm_vecsmall::SmallVector<DmaDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using RotatorDescWrapper         = DescriptorWrapper<RotatorDesc>;
using RotatorDescriptorsWrappers = llvm_vecsmall::SmallVector<RotatorDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE>;

using MmeLayerParamsPtr          = std::shared_ptr<MmeLayerParams>;
using MmeDescriptorGeneratorPtr  = std::shared_ptr<MmeDescriptorGenerator>;
}  // namespace gaudi2
