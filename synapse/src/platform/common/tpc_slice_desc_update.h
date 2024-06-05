#pragma once

#include "tpc_slice.h"
#include "types.h"

// The methods bellow are used for all gaudi devices descriptor generation and
// tpc slice descriptor update *for the case of sizes).
// single 64b elements will be treated as 2 32b elements (32b low, 32b high)
// hence FCD of size is doubled.
inline NSizeArray getTpcDescNSizesInElements(const Tensor& tensor)
{
    NSizeArray sizes = tensor.getNSizesInElements();
    if (tensor.is64BitElementSize())
    {
        sizes[0] *= 2;
    }
    return sizes;
}

inline NStrideArray getTpcDescNStridesInElements(const Tensor& tensor)
{
    NStrideArray elemNStrides = tensor.getNStridesInElements();
    if (tensor.is64BitElementSize())
    {
        for (auto dim = 1; dim < tensor.getDim(); ++dim)
        {
            elemNStrides[dim] *= 2;
        }
    }
    return elemNStrides;
}

// Objects of this type take descriptors generated for a TPC slice as if it's a TPCNode and update them for the specific
// slice
template<class TpcDesc, class TpcTensorDesc>
class TPCSliceDescUpdate
{
public:
    TPCSliceDescUpdate(const TPCSlice* slice) : m_sliceNode(slice) {}

    void update(TpcDesc& tpcDesc) const;

protected:
    unsigned updateTensorDesc(TpcTensorDesc* tensorDesc, const TensorPtr& tensor) const;
    void     updateTensorOffset(TpcTensorDesc* tensorDesc, const TensorPtr& tensor) const;
    void     updateTensorShape(TpcTensorDesc* tensorDesc, const TensorPtr& tensor) const;
    // methods to encapsulate access to tensor descriptor fields since descriptor layout
    // has changed in gaudi3 so we need to diverge while keeping the logic common.
    static ptrToInt  getTpcDescBaseAddress(const TpcTensorDesc& tensorDesc);
    static void      setTpcDescBaseAddress(TpcTensorDesc& tensorDesc, ptrToInt newAddress);
    static void      setTpcDescLastDim(TpcTensorDesc& tensorDesc, unsigned indexOfMaxNonDegenerateStride);
    static uint32_t* getTpcDescDimSize(TpcTensorDesc& tensorDesc, unsigned dim);

private:
    const TPCSlice* m_sliceNode;
};

template<class TpcDesc, class TpcTensorDesc>
void TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::update(TpcDesc& tpcDesc) const
{
    unsigned tensorDescIdx = 0;
    for (const auto& input : m_sliceNode->getInputs())
    {
        if ((input->getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR) || input->isAuxTensor())
        {
            continue;
        }
        auto numDescUpdated = updateTensorDesc(tpcDesc.m_tensors + tensorDescIdx, input);
        tensorDescIdx += numDescUpdated;
    }
    for (const auto& output : m_sliceNode->getOutputs())
    {
        auto numDescUpdated = updateTensorDesc(tpcDesc.m_tensors + tensorDescIdx, output);
        tensorDescIdx += numDescUpdated;
    }
}

template<class TpcDesc, class TpcTensorDesc>
unsigned TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::updateTensorDesc(TpcTensorDesc*   tensorDesc,
                                                                      const TensorPtr& tensor) const
{
    updateTensorOffset(tensorDesc, tensor);
    updateTensorShape(tensorDesc, tensor);
    return TPCNode::numTensorDescriptors(*tensor);
}

template<class TpcDesc, class TpcTensorDesc>
void TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::updateTensorOffset(TpcTensorDesc*   tensorDesc,
                                                                    const TensorPtr& tensor) const
{
    // Slice operands offset need to be updated to a virtual address, such that starting from it and advancing using
    // the index space, access pattern and tensor strides, the AGU will generate the real tensor ROI address.
    uint64_t offset = 0;
    for (unsigned dim = 0; dim < tensor->getDim(); dim++)
    {
        offset += m_sliceNode->getTensorSliceOffsetInDim(tensor, dim) * tensor->getStrideInBytes(dim);
    }
    if (offset)
    {
        auto numDimBundles = TPCNode::numTensorDescriptors(*tensor);
        for (uint32_t dimBundle = 0; dimBundle < numDimBundles; dimBundle++)
        {
            ptrToInt p = getTpcDescBaseAddress(tensorDesc[dimBundle]);
            if (p.u64 < offset)
            {
                LOG_DEBUG(TPC_SLICE,
                          "Big-Tensor slice offset ({}) is bigger then the slice address in memory (0x{:x}).",
                          offset,
                          p.u64);
            }

            LOG_DEBUG(TPC_SLICE,
                      "Updating tensor {} : orig address: 0x{:x}, new address: 0x{:x}",
                      tensor->getName(),
                      p.u64,
                      p.u64 - offset);

            p.u64 -= offset;
            setTpcDescBaseAddress(tensorDesc[dimBundle], p);
        }
    }
}

template<class TpcDesc, class TpcTensorDesc>
void TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::updateTensorShape(TpcTensorDesc*   tensorDesc,
                                                                   const TensorPtr& tensor) const
{
    auto        tensorDim     = tensor->getDim();
    const auto& sizeTensor    = m_sliceNode->getOriginalTensor(tensor);
    auto        numDimBundles = TPCNode::numTensorDescriptors(*tensor);

    // The last dim field holds the highest (slowest) non-degenerated dimension. It is used for Huge tensors handling in
    // TPC. In case the shape updates, it should be updated as well. For example, slicing the SCD to size=1 would make
    // it seem degenerated, but when the sizes are taken from the original tensor, the SCD size may be > 1.
    setTpcDescLastDim(*tensorDesc, sizeTensor->getIndexOfMaxNonDegenerateStride());
    NSizeArray curSizes = getTpcDescNSizesInElements(*sizeTensor);
    for (uint32_t dimBundle = 0; dimBundle < numDimBundles; dimBundle++)
    {
        for (unsigned dimOffset = 0; dimOffset < TpcDesc::c_max_tensor_dims; ++dimOffset)
        {
            unsigned dim = dimBundle * TpcDesc::c_max_tensor_dims + dimOffset;
            if (dim >= tensorDim) break;
            uint32_t* pDimSize = getTpcDescDimSize(tensorDesc[dimBundle], dimOffset);
            if (*pDimSize != curSizes[dim])
            {
                LOG_DEBUG(TPC_SLICE,
                          "Updating tensor {} size in dim: {} from: {} to {}",
                          tensor->getName(),
                          dim,
                          *pDimSize,
                          curSizes[dim]);

                *pDimSize = curSizes[dim];
            }
        }
    }
}

template<class TpcDesc, class TpcTensorDesc>
ptrToInt TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::getTpcDescBaseAddress(const TpcTensorDesc& tensorDesc)
{
    ptrToInt p;
    p.u32[0] = tensorDesc.base_addr_low.v;
    p.u32[1] = tensorDesc.base_addr_high.v;
    return p;
}

template<class TpcDesc, class TpcTensorDesc>
void TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::setTpcDescBaseAddress(TpcTensorDesc& tensorDesc, ptrToInt newAddress)
{
    tensorDesc.base_addr_low.v  = newAddress.u32[0];
    tensorDesc.base_addr_high.v = newAddress.u32[1];
}

template<class TpcDesc, class TpcTensorDesc>
void TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::setTpcDescLastDim(TpcTensorDesc& tensorDesc,
                                                                   unsigned       indexOfMaxNonDegenerateStride)
{
    tensorDesc.tensor_config.last_dim = indexOfMaxNonDegenerateStride;
}

template<class TpcDesc, class TpcTensorDesc>
uint32_t* TPCSliceDescUpdate<TpcDesc, TpcTensorDesc>::getTpcDescDimSize(TpcTensorDesc& tensorDesc, unsigned dim)
{
    // The dim size fields are interleaved with the dim stride fields in the descriptor, so to access the size
    // of the next dimension, need to skip the previous dimension's size and stride (2 x uint32_t).
    return &(tensorDesc.dim_0_size._raw) + static_cast<size_t>(2 * dim);
}