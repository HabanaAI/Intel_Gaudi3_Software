#pragma once

#include "include/sync/overlap.h"
#include "descriptor_wrapper.h"
#include "gaudi/gaudi_tpc_descriptor.h"
#include "gaudi/mme.h"
#include "hal_conventions.h"

namespace gaudi
{
#include "gaudi/asic_reg_structs/dma_core_regs.h"

typedef Mme::Desc                       MmeDesc;
typedef struct GaudiTpcDesc             TpcDesc;
typedef struct block_dma_core           DmaDesc;

typedef DescriptorWrapper<MmeDesc>      MmeDescWrapper;
typedef llvm_vecsmall::SmallVector<MmeDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE> MmeDescriptorsWrappers;

typedef DescriptorWrapper<TpcDesc>      TPCDescWrapper;
typedef llvm_vecsmall::SmallVector<TPCDescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE> TPCDescriptorsWrappers;

typedef DescriptorWrapper<DmaDesc>      DMADescWrapper;
typedef llvm_vecsmall::SmallVector<DMADescWrapper, MAX_ESTIMATE_AMOUNT_DESC_PER_NODE> DMADescriptorsWrappers;
}

typedef Overlap<gaudi::LOGICAL_QUEUE_MAX_ID> GaudiOverlap;
