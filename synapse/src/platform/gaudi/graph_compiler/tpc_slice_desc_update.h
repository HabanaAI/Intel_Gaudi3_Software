#pragma once

#include "gaudi/asic_reg_structs/tpc_tensor_regs.h"
#include "gaudi_types.h"
#include "platform/common/tpc_slice_desc_update.h"

namespace gaudi
{
class TPCSliceDescUpdate : public ::TPCSliceDescUpdate<gaudi::TpcDesc, block_tpc_tensor>
{
public:
    TPCSliceDescUpdate(const TPCSlice* slice) : ::TPCSliceDescUpdate<gaudi::TpcDesc, block_tpc_tensor>(slice) {}

    void update(std::list<DescAndMask<TpcDesc>>& descs) const;
};
}  // namespace gaudi