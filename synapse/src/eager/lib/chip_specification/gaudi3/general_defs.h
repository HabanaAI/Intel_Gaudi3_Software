#pragma once

// std includes
#include <cstdint>

namespace eager_mode::gaudi3_spec_info
{
// TODO:: need to calibrate once silicon arrives
static constexpr uint64_t TENSOR_SIZE_THRESHOLD_FOR_EAGER_PARALLEL_EXECUTION = 1 << 16;
}  // namespace eager_mode::gaudi3_spec_info