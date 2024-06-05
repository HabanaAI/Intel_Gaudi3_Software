#pragma once

#include "synapse_common_types.h"
#include <array>
#include <map>

typedef std::array<TSize, SYN_MAX_TENSOR_DIM> TensorDimensions;

class TestTensorsDimensions
{
public:
    TestTensorsDimensions() = default;

    void setDimensions(bool isInput, unsigned tensorIndex, const TensorDimensions& rTensorDimensions);

    const TensorDimensions* getDimensions(bool isInput, unsigned tensorIndex) const;

private:
    using TensorKey               = std::pair<bool, unsigned>;
    using TestTensorDimensionsMap = std::map<TensorKey, TensorDimensions>;
    TestTensorDimensionsMap m_tensorsDimensions;
};
