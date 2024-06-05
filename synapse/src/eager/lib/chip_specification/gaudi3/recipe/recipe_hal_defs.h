#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"

// synapse-internal gaudi3-specific includes (relative to src/)
#include "platform/gaudi3/graph_compiler/block_data.h"

// relative to <specs_h9>/
#include "gaudi3/gaudi3_tpc_descriptor.h"

namespace eager_mode::gaudi3_spec_info
{
// some code needs this information at compile time.
// getMaxTpcTensorsNr enforces the correctness of the value.
static constexpr TensorsNrType maxTpcTensorsNr = 31;
static constexpr TensorsNrType maxMmeTransposeTensorsNr = 2;

constexpr bool isMmeTranspose(TensorsNrType tensorsNr)
{
    return tensorsNr == 2;
}

namespace qman_regs
{
using namespace gaudi3;
static constexpr AsicRegType cacheBaseRegBase = GET_ADDR_OF_QMAN_BLOCK_FIELD(qman_wr64_base_addr0);
}  // namespace qman_regs

namespace mme_regs
{
using namespace gaudi3;
// Define valid range of MME descriptor. Don't include tensors addresses.

static constexpr AsicRegType getInclusiveFirst(bool isTranspose)
{
    return isTranspose ? GET_ADDR_OF_MME_BLOCK_FIELD(arch_dma_n_ten_st) : GET_ADDR_OF_MME_BLOCK_FIELD(arch_n_ten_st);
}

static constexpr AsicRegType getExclusiveLast(bool isTranspose)
{
    AsicRegType baseAddress =
        isTranspose ? GET_ADDR_OF_MME_BLOCK_FIELD(arch_dma_n_ten) : GET_ADDR_OF_MME_BLOCK_FIELD(arch_n_ten);
    return baseAddress + offsetof(block_mme_non_tensor_descriptor, rate_limiter);
}

static constexpr AsicRegType getDescSize(bool isTranspose)
{
    return getExclusiveLast(isTranspose) - getInclusiveFirst(isTranspose);
}

// Address of first MME tensor
static constexpr AsicRegType getFirstTensorAddr(bool isTranspose)
{
    AsicRegType baseAddress =
        isTranspose ? GET_ADDR_OF_MME_BLOCK_FIELD(arch_dma_base_addr) : GET_ADDR_OF_MME_BLOCK_FIELD(arch_base_addr);
    return baseAddress + offsetof(block_mme_address_descriptor, cout0_lo);
}

static constexpr AsicRegType getBlackListInclusiveFirst(bool isTranspose)
{
    AsicRegType baseAddress =
        isTranspose ? GET_ADDR_OF_MME_BLOCK_FIELD(arch_dma_n_ten) : GET_ADDR_OF_MME_BLOCK_FIELD(arch_n_ten);
    return baseAddress + offsetof(block_mme_non_tensor_descriptor, rate_limiter);
}

static constexpr AsicRegType getBlackListInclusiveLast(bool isTranspose)
{
    AsicRegType baseAddress =
        isTranspose ? GET_ADDR_OF_MME_BLOCK_FIELD(arch_dma_n_ten) : GET_ADDR_OF_MME_BLOCK_FIELD(arch_n_ten);
    return baseAddress + offsetof(block_mme_non_tensor_descriptor, wkl_id);
}

}  // namespace mme_regs

namespace tpc_regs
{
using namespace gaudi3;
// Define valid range of TPC descriptor. Don't include tensors addresses.
static constexpr AsicRegType inclusiveFirst =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, kernel_config);
static constexpr AsicRegType exclusiveLast =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, icache_axi_cfg);
static constexpr BlobSizeType descSize = exclusiveLast - inclusiveFirst;

static constexpr AsicRegType ICACHE_AXI_CFG =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, icache_axi_cfg);
static constexpr AsicRegType DCACHE_AXI_CFG =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, dcache_axi_cfg);

// TPC tensors descriptor
static constexpr AsicRegType  tensorDescAddr       = GET_ADDR_OF_TPC_BLOCK_FIELD(qm_tensor_0);
static constexpr AsicRegType  tensorBaseAddr       = tensorDescAddr + offsetof(block_tpc_tensor_base, base_addr_low);
static constexpr AsicRegType  subFirstTensorOffset = offsetof(block_tpc_tensor_base, pref_stride);
static constexpr BlobSizeType tensorDescSize       = sizeof(TensorDescGaudi3);
// QM_KERNEL_BASE_ADDRESS_LOW
static constexpr AsicRegType kernelBaseAddr =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm) + offsetof(block_tpc_non_tensor_descriptor, kernel_base_address_low);
// For cache invalidation
static constexpr AsicRegType tpcCmdRegAddr = GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd);
// QM_SYNC_OBJECT_MESSAGE
static constexpr AsicRegType synObjMsgRegAddr =
    GET_ADDR_OF_TPC_BLOCK_FIELD(qm_sync_object_th0) + offsetof(block_sync_object, message);
}  // namespace tpc_regs

}  // namespace eager_mode::gaudi3_spec_info