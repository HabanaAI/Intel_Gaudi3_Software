#pragma once

#include "gaudi2/asic_reg_structs/tpc_tensor_regs.h"
#include "gaudi2_types.h"
#include "tpc_descriptor_generator.h"
#include "platform/common/tpc_slice_desc_update.h"

namespace gaudi2
{
class TPCSliceDescUpdate : public ::TPCSliceDescUpdate<gaudi2::TpcDesc, block_tpc_tensor>
{
public:
    TPCSliceDescUpdate(const TPCSlice* slice) : ::TPCSliceDescUpdate<gaudi2::TpcDesc, block_tpc_tensor>(slice) {}

    void update(TpcDescriptorGenerator::DescriptorsVector& descs) const;
};
}  // namespace gaudi2