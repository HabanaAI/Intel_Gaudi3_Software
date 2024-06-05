#pragma once

#include "tpc_descriptor_generator.h"
#include "platform/common/tpc_slice_desc_update.h"

// specialize tensor descriptor access methods due to change in descriptor layout
// in gaudi3.
template<>
inline ptrToInt
TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>::getTpcDescBaseAddress(const TensorDescGaudi3& tensorDesc)
{
    ptrToInt p;
    p.u32[0] = tensorDesc.base.base_addr_low.v;
    p.u32[1] = tensorDesc.base.base_addr_high.v;
    return p;
}

template<>
inline void TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>::setTpcDescBaseAddress(TensorDescGaudi3& tensorDesc,
                                                                                       ptrToInt          newAddress)
{
    tensorDesc.base.base_addr_low.v  = newAddress.u32[0];
    tensorDesc.base.base_addr_high.v = newAddress.u32[1];
}

template<>
inline void
TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>::setTpcDescLastDim(TensorDescGaudi3& tensorDesc,
                                                                       unsigned          indexOfMaxNonDegenerateStride)
{
    tensorDesc.shared.tensor_config.last_dim = indexOfMaxNonDegenerateStride;
}

template<>
inline uint32_t* TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>::getTpcDescDimSize(TensorDescGaudi3& tensorDesc,
                                                                                        unsigned          dim)
{
    // The dim size fields are interleaved with the dim stride fields in the descriptor, so to access the size
    // of the next dimension, need to skip the previous dimension's size and stride (2 x uint32_t).
    return &(tensorDesc.shared.dim_0_size._raw) + static_cast<size_t>(2 * dim);
}

namespace gaudi3
{
class TPCSliceDescUpdate : public ::TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>
{
public:
    TPCSliceDescUpdate(const TPCSlice* slice) : ::TPCSliceDescUpdate<Gaudi3TpcDesc, TensorDescGaudi3>(slice) {}

    void update(TpcDescriptorGenerator::DescriptorsVector& descs) const;
};
}  // namespace gaudi3