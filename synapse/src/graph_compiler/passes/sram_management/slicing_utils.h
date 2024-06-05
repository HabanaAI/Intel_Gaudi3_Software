#pragma once

#include "mme_slicing_strategy.h"
#include "bundle.h"
#include "types.h"

using SlicingData = MmeSlicingStrategy::MmeSlicingData;

class SlicedOperandUtils
{
public:
    using ReshapeOutputToInputMapping = llvm_vecsmall::SmallVector<DimVector, HABANA_DIM_MAX>;

    // return the tensor coordinate offset for the slice coordinate, for a single dimension
    static unsigned getSliceCoordOffset(unsigned coordIdx, unsigned chunkSize, int overlap, int tensoroffsetBefore);

    // Return the offset in elements of the slice in each dimension
    static OffsetArray getSliceOffsets(const pSliceReference& sliceRef);

    // calculate final slice size according to the given inputs
    static SizeArray calcSliceSizesFromCoordinate(const SizeArray&      originalTensorSizes,
                                                  const pSlicedOperand& operand,
                                                  const CoordArray&     coord);

    // calculate final slice size according to the given slice reference
    static SizeArray calcSliceSizesFromSliceRef(const pSliceReference& sliceRef);

    // return the aggregated size in elements that the given slice reference represents in the given dimensions
    static TSize getAggregatedDimSizeInElements(const pSliceReference& sliceRef, const DimVector& dims);

    // Size of given slice ref in elements
    static TSize getSliceSizeInElements(const pSliceReference& sliceRef);

    // Size of given slice ref in bytes
    static TStride getSliceSizeInBytes(const pSliceReference& sliceRef,
                                       bool                   useOriginalTensorElementType = false);

    // Size of a single slice in elements
    static TSize getSliceSizeInElements(const pSlicedOperand& operand);

    // Size of a single slice in bytes
    static TStride getSliceSizeInBytes(const pSlicedOperand& operand);

    // given final sizes of a slice, return the aligned strides for it or {} if no alignment is needed
    static std::optional<StrideArray> getCacheLineAlignedStrides(const SizeArray& finalSizes, const pSlicedOperand& operand);

    //  return the aligned strides for given inputs or {} if no alignment is needed
    static std::optional<StrideArray> getCacheLineAlignedStrides(const SizeArray&   sizes,
                                                                 const StrideArray& initialStrides,
                                                                 unsigned           numOfDims,
                                                                 unsigned           dimToAlign = 0);

    // fill given strides according to other input parameters
    static void fillStrides(StrideArray& strides,
                            const SizeArray& sizes,
                            unsigned fromDim,
                            unsigned fromDimValue,
                            unsigned ToDim);

    // return the full size of the narrow axis (without slicing)
    static TSize getNarrowFullAxisSize(const MmeSlicingStrategy& strategy);

    // return the full size of the wide axis (without slicing)
    static TSize getWideFullAxisSize(const MmeSlicingStrategy& strategy);

    // return true if the operand is sliced to a single slice (chunk shape is the same as the tensor)
    static bool isTriviallySliced(const pSlicedOperand& operand);

    // return true if all operands are trivially sliced
    static bool isTriviallySliced(const SlicingStrategy& strategy);

    // return true if the specified dimension is sliced in for the operand
    static bool isSlicedOnDimension(const pSlicedOperand& operand, unsigned dimIdx);

    // checks if a tensor is 2D or flattened 4D tensor.
    static bool isTensor2D(const pTensor& tensor);

    //check if final shape is 2D
    static bool isFinalShape2D(const pSlicedOperand& op);

    // checks if operand is flattened 4D tensor.
    static bool shouldOperandBeFlattened(const pSlicedOperand& op);

    //check if any operand in list is flattened
    static bool shouldAnyOperandBeFlattened(const std::vector<pSlicedOperand>& operands);

    // return the number of slices for a sliced operand in a given list of dimensions
    static unsigned nofSlices(const pSlicedOperand& op, const DimVector& dims);

    // return the number of slices for a sliced operand in a given dimension
    static unsigned nofSlices(const pSlicedOperand& op, unsigned dim);

    // return the total number of slices for a sliced operand
    static unsigned nofSlices(pSlicedOperand op);

    // return the total number of slices for a sliced operand
    static CoordArray totalNumberOfSlices(pSlicedOperand op);

    // count how many dimensions are sliced in the given operand.
    static unsigned getNumOfSlicedDims(const pSlicedOperand& operand);

    // given the stitched operand, return the non-stitched operand of SlicingData
    static const pSlicedOperand& getNonStitchedOperand(const SlicingData& slicingData,
                                                       const pSlicedOperand& stitchedOperand);

    static pSlicedOperand createNonSharedOperandOfNode(const pSlicedOperand& sharedOperand,
                                                       const pNode& node);

    static bool isSlicedOnCommonDim(const pSlicedOperand& operand, const pNode& node);

    static Settable<unsigned> getFirstNonDegeneratedDim(const DimVector& dims, const pSlicedOperand& operand);
    // return true if all dims in the input range (that are inside the actual tensor dims) equal 1.
    static bool areDimsDegenerated(const pTensor& tensor, unsigned fromDim, unsigned toDim);
    // given a node where the number of dims in the 1st input is different than the number of dims in the 1st output,
    // checks if all dims that this difference is consist of, are degenerated (meaning their size = 1)
    static bool areDegeneratedDimsReshaped(const pNode& reshapeNode);

    // return an array of bool per dimension, set to true if this is the first slice in the dimension, or false if it's not
    static std::array<bool, SYN_MAX_TENSOR_DIM> isFirstSlice(const pSliceReference& sliceRef);
    // return an array of bool per dimension, set to true if this is the last slice in the dimension, or false if it's not
    static std::array<bool, SYN_MAX_TENSOR_DIM> isLastSlice(const pSliceReference& sliceRef);
    // returns a mapping from reshape output to input - for each dim in output, what dims in input are required.
    static ReshapeOutputToInputMapping getReshapeOutputToInputMapping(const TensorShape& input, const TensorShape& output);
    // returns true if the tpcNode uses broadcast (supported + needed by a input)
    static bool isBroadcast(const NodePtr& tpcNode);
    // return true is a resahpe with map reshapeMap can also reshape a broadcasted input.
    static bool canReshapeBroadcast(const TensorPtr& input, const TensorPtr& output, const TensorPtr& reshapeOutput);
    // Return all sliced dims from the inner dimension to the outer one
    static DimVector getSlicedDims(const pSlicedOperand& operand);
    // check if producer reshape can be aligned
    static bool canAlignProducerReshape(const pBundleExpansion& candidate);

    // Checks if final shape is different than original tensor sizes.
    static bool isOperandFlattened(const pSlicedOperand& operand);

    // Returns high precision data type for partials summation.
    static synDataType getTypeForPartials(synDataType currentMmeOutputDataType);

    // Returns operand slice size in bytes.
    static TStride getTensorSliceSizeInBytes(const TensorPtr& t, const std::map<unsigned, TSize>& sizePerSlicedDim);

    // Checks whether mme input is aligned to mme cache line.
    static bool isTensorAlignedToMmeCL(const NodePtr& n, unsigned inputIdx);

    // Returns the aligment size in bytes for a slice of big tensor t, which is sliced on slicedDim to size
    // slicedDimSize. The alignment size is the delta between the aligned sliced tensor to the dense sliced tensor.
    static TStride getSliceAlignmentSize(const TensorPtr& t, const std::map<unsigned, TSize>& sizePerSlicedDim);

private:
    static bool isTensorShape2D(const pTensor& tensor, const SizeArray& shape);
};
