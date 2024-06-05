#pragma once

// std includes
#include <cstdint>

namespace eager_mode::gaudi2_spec_info
{
static constexpr uint64_t TENSOR_SIZE_THRESHOLD_FOR_EAGER_PARALLEL_EXECUTION = 1 << 16;
}