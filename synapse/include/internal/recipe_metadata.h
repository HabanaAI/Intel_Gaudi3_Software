#pragma once

#include <cstdint>
#include "include/mme_common/mme_common_enum.h"
#include "synapse_common_types.h"
#include "common_tpc_tid_metadata.h"

struct tpc_size_sm_params_t
{
    uint32_t this_dim;
    uint32_t tensor_index;
    uint32_t is_output;
    int64_t  offset;  // Offset in big tensor (in elements)
};

struct tpc_stride_sm_params_t
{
    uint32_t is_output         : 1;
    uint32_t this_dim          : 5;
    uint32_t first_dynamic_dim : 5;
    uint32_t element_size      : 5;
    uint32_t tensor_index      : 6;
    uint32_t reserved          : 10;
};

struct dynamic_execution_sm_params_t
{
    uint16_t cmd_len;
    uint16_t should_bypass;
    uint32_t commands[DYNAMIC_EXECUTE_METADATA_COMMANDS_LEN];
    uint16_t num_projections : 6;
    uint32_t reserved        : 10;
    struct
    {
        uint16_t tensor_idx : 6;
        uint16_t tensor_dim : 6;
        uint16_t is_output  : 1;
        uint16_t reserved   : 3;
    } projections[DYNAMIC_EXECUTE_MAX_DIM_PROJECTIONS];
};

struct mme_sm_params_t
{
    uint8_t  dim;
    int8_t   tensor_input_index;
    int8_t   tensor_output_index;
    uint32_t multiply_factor;
};

struct mme_multi_dcore_sm_params_t
{
    uint8_t  dim;
    int8_t   tensor_input_index;
    int8_t   tensor_output_index;
    uint64_t dcore_roi_offset;
    uint64_t dcore_roi_size;
    uint32_t multiply_factor;
};

struct mme_sync_sm_params_t
{
    uint32_t num_signals;
};

struct mme_padding_sm_params_t
{
    uint32_t old_padding[MAX_CONV_DIMS * 2];
    uint32_t conv_stride[MAX_CONV_DIMS];
    uint32_t conv_dilation[MAX_CONV_DIMS];
    uint32_t conv_kernel[MAX_CONV_DIMS];

    uint32_t tensor_strides[MAX_DIMENSIONS_NUM];
    int32_t  old_offsets[MAX_DIMENSIONS_NUM];

    uint32_t this_dim;

    MmeCommon::EMmeOpType opType;
};

struct dma_sm_params_t
{
    uint32_t this_dim       : 3;
    uint32_t is_destination : 1;
    uint32_t is_memset      : 1;
    uint32_t is_total       : 1;
    uint32_t element_size   : 5;
    uint32_t reserved       : 20;
};

struct bulk_size_stride_sm_params_t
{
    uint32_t first_dynamic_dim : 3;
    uint32_t element_size      : 5;
    uint32_t affected_fields   : 4;
    uint32_t is_src            : 1;
    uint32_t reserved          : 19;
};

struct last_stride_sm_params_t
{
    uint32_t element_size : 5;
    uint32_t is_src       : 1;
    uint32_t reserved     : 26;
};

struct view_stride_sm_params_t
{
    uint64_t num_real_elements : 64;
    uint32_t this_dim          : 3;
    uint32_t element_size      : 5;
    uint32_t is_src            : 1;
    uint32_t reserved          : 23;
};

struct address_sm_params_t
{
    uint64_t base_address : 64;
    uint32_t element_size : 5;
    uint32_t is_src       : 1;
};

struct physical_concat_split_sm_params_t
{
    uint64_t roi_base_address       : 64;
    uint32_t element_size           : 5;
    uint32_t concat_split_dim       : 3;
    uint32_t number_in_concat_split : 24;
    uint64_t output_strides[MAX_DIMENSIONS_NUM + 1];
};
struct slice_stride_sm_params_t
{
    uint32_t dim          : 3;
    uint32_t element_size : 5;
    uint32_t is_src       : 1;
};

struct slice_address_sm_params_t
{
    uint64_t base_address      : 64;
    uint64_t num_real_elements : 64;
    uint32_t element_size : 5;
    uint32_t is_src       : 1;
};

struct view_address_sm_params_t
{
    uint64_t base_address : 64;
    uint64_t max_offset   : 64;
    uint32_t element_size : 5;
    uint32_t is_src       : 1;
    uint64_t max_strides[MAX_DIMENSIONS_NUM + 1];
};
