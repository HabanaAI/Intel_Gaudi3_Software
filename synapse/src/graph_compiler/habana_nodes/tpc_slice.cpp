#include "tpc_slice.h"
#include "tpc_slice_roi_generator.h"
#include "utils.h"

NodePtr TPCSlice::clone() const
{
    HB_ASSERT(false, "TPC Slices should not be cloned");
    return nullptr;
}

tpc_lib_api::GlueCodeReturn TPCSlice::init(tpc_lib_api::DeviceId   deviceId,
                                           AuxiliaryTensors*       cachedAuxiliaryTensors,
                                           std::optional<uint32_t> kernelUniqueId)
{
    HB_ASSERT(false, "TPC Slices should not be instantiated");
    return tpc_lib_api::GLUE_FAILED;
}

void TPCSlice::resetInstantiated()
{
    HB_ASSERT(false, "TPC Slices should not be re-instantiated");
}

bool TPCSlice::isSuggestedOptimizationDone() const
{
    // TPC slices should not be optimized on their own. The optimization pass may be called anyway, so this makes sure
    // they don't get optimized by it.
    return true;
}

NodeROI TPCSlice::generateRoi() const
{
    TPCSliceROIGenerator roiGenerator(*this);
    return roiGenerator.generateROI();
}

// TPC slices have tensors that fits the slice, but access pattern that fit the full operation with the big tensors.
// getWorkROI generates the size and offset in the slice tensor from the access pattern and roi that are in the big
// tensors dimensions.
// For example, suppose the full tensor has 1024 elements in some dimension, the slice tensor represents elements
// 512-1023 (meaning that it's size is 512) and the slice node is split to ROIs, so that the given ROI relates to
// elements 544-607 (64 elements, 32 elements from the slice start).
//                                                                                                      slice and big
//                     big tensor start              slice start    ROI start    ROI end                   tensor end
//                     |___________________________________|___________|____________|_______________________________|
// Big tensor idx:     0                                    512         544      607                            1023
// Slice tensor idx:                                        0           32        95                             511
//
// The point of these methods is to return offset=32 and size=64 for this ROI, by setting -512 offset to the slice start
// and end.
TOffset TPCSlice::getSliceOffset(const TensorPtr&                         tensor,
                                 unsigned                                 dim,
                                 const tpc_lib_api::DimIndexSpaceMapping& dimAccessPattern) const
{
    return -1 * getTensorSliceOffsetInDim(tensor, dim);
}