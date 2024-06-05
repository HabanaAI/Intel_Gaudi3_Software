#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "platform/gaudi2/graph_compiler/block_data.h"

namespace eager_mode::gaudi2_spec_info
{
// some code needs this information at compile time.
// getMaxTpcTensorsNr enforces the correctness of the value.
static constexpr TensorsNrType maxTpcTensorsNr = 15;

namespace qman_regs
{
using namespace gaudi2;
static constexpr AsicRegType cacheBaseRegBase = GET_ADDR_OF_QMAN_BLOCK_FIELD(qman_wr64_base_addr0);
}  // namespace qman_regs

namespace mme_regs
{
using namespace gaudi2;

// Define valid range of MME descriptor. Don't include tensors addresses.
static constexpr AsicRegType inclusiveFirst = GET_ADDR_OF_MME_BLOCK_FIELD(arch_non_tensor_start);
static constexpr AsicRegType exclusiveLast =
    GET_ADDR_OF_MME_BLOCK_FIELD(arch_non_tensor_end) + offsetof(block_mme_non_tensor_descriptor, pcu);
static constexpr BlobSizeType descSize = exclusiveLast - inclusiveFirst;

// Address of first MME tensor
static constexpr AsicRegType firstTensorAddr =
    GET_ADDR_OF_MME_BLOCK_FIELD(arch_base_addr) + offsetof(block_mme_address_descriptor, cout0_low);
}  // namespace mme_regs

namespace tpc_regs
{
using namespace gaudi2;
// Define valid range of TPC descriptor. Don't include tensors addresses.
static constexpr AsicRegType inclusiveFirst =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, kernel_config);
static constexpr AsicRegType exclusiveLast =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, tid_base_size_high_dim_0);
static constexpr BlobSizeType descSize = exclusiveLast - inclusiveFirst;
// TPC tensors descriptor
static constexpr AsicRegType  tensorDescAddr       = GET_ADDR_OF_TPC_BLOCK_FIELD(qm_tensor_0);
static constexpr AsicRegType  tensorBaseAddr       = tensorDescAddr + offsetof(block_tpc_tensor, base_addr_low);
static constexpr AsicRegType  subFirstTensorOffset = offsetof(block_tpc_tensor, padding_value);
static constexpr BlobSizeType tensorDescSize       = sizeof(block_tpc_tensor);
// QM_KERNEL_BASE_ADDRESS_LOW
static constexpr AsicRegType kernelBaseAddr =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, kernel_base_address_low);
// For cache invalidation
static constexpr AsicRegType tpcCmdRegAddr = GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd);
// QM_SYNC_OBJECT_MESSAGE
static constexpr AsicRegType synObjMsgRegAddr =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm_sync_object) + offsetof(block_sync_object, message);
}  // namespace tpc_regs

namespace dma_regs
{
using namespace gaudi2;
static constexpr AsicRegType ctxBase    = GET_ADDR_OF_DMA_BLOCK_FIELD(ctx);
static constexpr AsicRegType axuserBase = GET_ADDR_OF_DMA_BLOCK_FIELD(axuser);
// Define valid range of DMA descriptor. Don't include tensors addresses.
static constexpr AsicRegType inclusiveFirst = ctxBase + offsetof(block_dma_core_ctx, te_numrows);
static constexpr AsicRegType exclusiveLast  = ctxBase + offsetof(block_dma_core_ctx, src_base_lo);
// Other registers required for descriptor
static constexpr AsicRegType hbWrReduction = axuserBase + offsetof(block_axuser, hb_wr_reduction);
static constexpr AsicRegType dstTSize0     = ctxBase + offsetof(block_dma_core_ctx, dst_tsize_0);
// DMA tensors addresses
static constexpr AsicRegType srcTensorAddr = ctxBase + offsetof(block_dma_core_ctx, src_base_lo);
static constexpr AsicRegType dstTensorAddr = ctxBase + offsetof(block_dma_core_ctx, dst_base_lo);
// Regs required for NOP
static constexpr AsicRegType compWdataRegAddr = ctxBase + offsetof(block_dma_core_ctx, wr_comp_wdata);
}  // namespace dma_regs

}  // namespace eager_mode::gaudi2_spec_info