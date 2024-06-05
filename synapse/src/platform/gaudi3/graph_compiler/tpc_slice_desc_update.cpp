#include "tpc_slice_desc_update.h"

namespace gaudi3
{
void TPCSliceDescUpdate::update(TpcDescriptorGenerator::DescriptorsVector& descs) const
{
    for (auto& descTuple : descs)
    {
        auto& tpcDesc = descTuple.desc;
        ::TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>::update(tpcDesc);
    }
}

}  // namespace gaudi3