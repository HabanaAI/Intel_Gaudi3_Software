#include "tpc_slice_desc_update.h"

namespace gaudi
{
void TPCSliceDescUpdate::update(std::list<DescAndMask<TpcDesc>>& descs) const
{
    for (auto& descTuple : descs)
    {
        auto& tpcDesc = std::get<0>(descTuple);
        ::TPCSliceDescUpdate<gaudi::TpcDesc, block_tpc_tensor>::update(tpcDesc);
    }
}

}  // namespace gaudi