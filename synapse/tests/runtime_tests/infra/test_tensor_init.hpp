#pragma once

#include "synapse_common_types.h"

#include "../infra/test_types.hpp"

#include <cstdint>

enum class TensorInitOp
{
    RANDOM_WITH_NEGATIVE,
    RANDOM_POSITIVE,
    ALL_ONES,
    ALL_ZERO,
    CONST,
    NONE,
};

struct OneTensorInitInfo
{
    TensorInitOp m_tensorInitOp       = TensorInitOp::NONE;
    uint64_t     m_initializer        = 0;
    bool         m_isDefaultGenerator = true;
};

struct TensorInitInfo
{
    OneTensorInitInfo input;
    OneTensorInitInfo output;
};

bool initBufferValues(OneTensorInitInfo oneTensorInitInfo, synDataType dataType, uint64_t numElements, void* output);
