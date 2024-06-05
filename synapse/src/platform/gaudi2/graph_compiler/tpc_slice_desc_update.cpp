#include "tpc_slice_desc_update.h"

namespace gaudi2
{
void TPCSliceDescUpdate::update(TpcDescriptorGenerator::DescriptorsVector& descs) const
{
    for (auto& descTuple : descs)
    {
        auto& tpcDesc = descTuple.desc;
        ::TPCSliceDescUpdate<gaudi2::TpcDesc, block_tpc_tensor>::update(tpcDesc);
    }
}

}  // namespace gaudi2