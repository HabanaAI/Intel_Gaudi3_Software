#pragma once

#include <cstdint>

static constexpr std::size_t DYNAMIC_EXECUTE_METADATA_COMMANDS_LEN   = 7;
static constexpr std::size_t DYNAMIC_EXECUTE_MAX_TPC_DIM_PROJECTIONS = 16;
static constexpr std::size_t DYNAMIC_EXECUTE_MAX_DIM_PROJECTIONS     = 5;
static constexpr std::size_t DYNAMIC_EXECUTE_MAX_DIMENSIONS          = 5;

// For Gaudi
struct tpc_sm_params_t
{
    uint32_t this_dim        : 6;
    uint32_t num_projections : 6;
    uint32_t reserved        : 20;
    struct
    {
        uint32_t tensor_idx : 6;
        uint32_t tensor_dim : 6;
        uint32_t is_output  : 1;
        uint32_t reserved   : 19;
        uint32_t size;
        float    a;
    } projections[DYNAMIC_EXECUTE_MAX_TPC_DIM_PROJECTIONS];
};
