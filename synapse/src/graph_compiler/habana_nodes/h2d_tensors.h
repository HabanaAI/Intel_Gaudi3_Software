#pragma once

#include "types.h"
#include "tensor.h"

bool isDynamicStridedDmaH2DTensorFcdStrided(const TensorPtr& t);

bool isDynamicStridedDmaH2DTensorDynamic(const TensorPtr& t);

bool isDynamicSliceDmaH2DTensorDynamic(const TensorPtr& t);