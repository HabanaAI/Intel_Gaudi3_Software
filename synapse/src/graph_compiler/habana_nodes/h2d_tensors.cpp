#include "h2d_tensors.h"

bool isDynamicStridedDmaH2DTensorFcdStrided(const TensorPtr& t)
{
    if (!t->isHost2DeviceTensor()) return false;

    synDynamicStridedDmaH2dTensor* dynStridesMaxData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(t->getHostMaxData());
    synDynamicStridedDmaH2dTensor* dynStridesMinData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(t->getHostMinData());

    return dynStridesMaxData->strides[0] != 1 || dynStridesMinData->strides[0] != 1;
}

bool isDynamicStridedDmaH2DTensorDynamic(const TensorPtr& t)
{
    if (!t->isHost2DeviceTensor()) return false;

    synDynamicStridedDmaH2dTensor* dynStridesMaxData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(t->getHostMaxData());
    synDynamicStridedDmaH2dTensor* dynStridesMinData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(t->getHostMinData());

    if (dynStridesMaxData->offset != dynStridesMinData->offset) return true;

    for (int i = 0; i < dynStridesMaxData->num_strides; i++)
    {
        if (dynStridesMaxData->strides[i] != dynStridesMinData->strides[i])
        {
            return true;
        }
    }

    return false;
}

bool isDynamicSliceDmaH2DTensorDynamic(const TensorPtr& t)
{
    if (!t->isHost2DeviceTensor()) return false;

    synDynamicSliceDmaH2dTensor* dynSliceMaxData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(t->getHostMaxData());
    synDynamicSliceDmaH2dTensor* dynSliceMinData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(t->getHostMinData());

    for (int i = 0; i < dynSliceMaxData->dims; i++)
    {
        if (dynSliceMaxData->starts[i] != dynSliceMinData->starts[i] ||
            dynSliceMaxData->steps[i]  != dynSliceMinData->steps[i])
        {
            return true;
        }
    }

    return false;
}