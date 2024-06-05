#pragma once
#include "tensor.h"
#include "include/sync/overlap.h"

struct TensorROILayout
{
    uint64_t getByteOffset(TOffset coord[Tensor::c_tensorMaxNDim], unsigned elementSizeInBits);
    void     getStartIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const;
    void     getEndIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const;

    // base address of this chunk
    uint64_t baseAddress = 0;
    bool     inSram      = false;  // false = in DRAM
    bool     isReduction = false;

    unsigned int tensorDim = 0;

    // For well-formed tensors, such as DMA or TPC
    TOffset    m_baseOffset[Tensor::c_tensorMaxNDim] = {};
    NSizeArray m_size                                = {};

    // Doesn't necessarily match parent tensor
    uint64_t spatialStrides[Tensor::c_numOfNStrides] = {};
};

class TensorROI
{
public:
    void print() const;

    void getStartIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const;
    void getEndIndex(TOffset idx[Tensor::c_tensorMaxNDim]) const;

    TensorROILayout&       getLayout();
    const TensorROILayout& getLayout() const;

    OverlapRoi&       getOverlapROI();
    const OverlapRoi& getOverlapROI() const;

    TensorSizesVector   getDimSizesInElements() const;
    TensorStridesVector getStridesNoFcdDim() const;
    TensorStridesVector getStridesWithFcdDim() const;

    TensorROILayout m_layout;
    OverlapRoi      m_overlapRoi;
    // So users can query things that are common to all tensor ROIs
    std::shared_ptr<Tensor> m_parentTensor;
};

// TensorROI is a big object so we do not wish to have too many
// elements in the small vector local storage and one is sufficient
// to avoid memory allocations for many of the cases.
using TensorROIVector = llvm_vecsmall::SmallVector<TensorROI, 1>;