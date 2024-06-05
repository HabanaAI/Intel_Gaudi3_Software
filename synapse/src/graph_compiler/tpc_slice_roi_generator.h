#pragma once

#include "tpc_slice.h"

// Slice ROI is the index space elements which the slice is computing in each dimension.
// This information is obtained from the tensor slices, by backward mapping their size to an index space region size.
// Since some tensors may be broadcasted or all required, which means their size may reflect the full index space range,
// the minimal range size accross all tensors should be taken.
// This class is responsible to go through the input and output slices and deduce the node's ROI.
class TPCSliceROIGenerator
{
public:
    explicit TPCSliceROIGenerator(const TPCSlice& sliceNode);

    NodeROI generateROI();

private:
    using Instance               = tpc_lib_api::HabanaKernelInstantiation;
    using Size                   = unsigned;
    using ROISize                = std::vector<Size>;
    using Offset                 = OperationSlice::OffsetArray::value_type;
    using ROIOffset              = std::vector<Offset>;
    using TensorAccessPattern    = tpc_lib_api::TensorAccessPattern;
    using TensorDimAccessPattern = tpc_lib_api::DimIndexSpaceMapping;

    const TPCSlice& m_sliceNode;
    const Instance& m_instance;
    ROISize         m_roiSize;
    ROIOffset       m_roiOffset;

    inline void init();
    inline void initROISize();
    inline void initROIOffset();

    inline void analyzeInputs();
    inline void analyzeOutputs();

    template<typename Container>
    void analyzeTensors(const Container& sliceTensor, const TensorAccessPattern* tensorAPs);
    void analyzeTensor(const TensorPtr& sliceTensor, const TensorAccessPattern* tensorAPs);
    void analyzeDimSize(const TensorPtr& sliceTensor, unsigned tensorDim, const TensorDimAccessPattern& tensorDimAP);
    void analyzeDimOffset(const TensorPtr& sliceTensor, unsigned tensorDim, const TensorDimAccessPattern& tensorDimAP);

    const TensorAccessPattern* findTensorAccessPattern(const TensorPtr&           sliceTensor,
                                                       const TensorAccessPattern* tensorAPs) const;

    NodeROI convertSizeOffsetToNodeROI() const;

    static Size   getOperandIdxSpaceRegionSize(Size tensorDimSize, const TensorDimAccessPattern& tensorDimAP);
    static Offset getOperandIdxSpaceOffset(Offset offsetInElements, const TensorDimAccessPattern& tensorDimAP);

    bool shouldSkipTensorForDimIdxSpaceAnalysis(const TensorPtr& tensor, unsigned dim) const;
};
