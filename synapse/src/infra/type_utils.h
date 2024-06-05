#pragma once

#include "synapse_common_types.h"
#include "defs.h"
// TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

inline unsigned dataTypeSizeInBytes(synDataType type, bool packed = false)
{
    switch (type)
    {
        case syn_type_int4:
        case syn_type_uint4:
            HB_ASSERT(packed, "4 bit data type {} is smaller than byte when not condensed", type);
            return 1;
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
            return 1;

        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_fp16:
            return 2;

        case syn_type_single:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_hb_float:
        case syn_type_tf32:
            return 4;

        case syn_type_int64:
        case syn_type_uint64:
            return 8;

        default:
            HB_ASSERT(false, "Invalid data type {}", type);
            return 0;
    }
}

inline uint32_t dataTypeToSizeInBits(synDataType dtype)
{
    switch (dtype)
    {
        case syn_type_int8:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
            return 8;
        case syn_type_bf16:
        case syn_type_fp16:
        case syn_type_int16:
        case syn_type_uint16:
            return 16;
        case syn_type_float:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_hb_float:
        case syn_type_tf32:
            return 32;
        case syn_type_int64:
        case syn_type_uint64:
            return 64;
        case syn_type_int4:
        case syn_type_uint4:
            return 4;
        default:
            HB_ASSERT(false, "unreachable {}", dtype);
            return 0;
    }
}

inline bool isHighPrecisionFloat(synDataType dtype)
{
    switch (dtype)
    {
        case syn_type_float:
        case syn_type_hb_float:
            return true;
        default:
            return false;
    }
}

template<typename T>
uint64_t getActualTensorSize(const uint32_t dims, const T sizes[], uint32_t elementType)
{
    if (dims == 0) return 0;
    uint64_t elements = 1;
    for (int i = 0; i < dims; i++)
    {
        elements *= sizes[i];
    }
    return elements * dataTypeToSizeInBits((synDataType)elementType) / 8;
}

// TODO: [SW-166081] Remove when fuser is moved to protocolIR
inline gcapi::TensorDataType_t toGlueCodeDataType(synDataType type)
{
    switch (type)
    {
        case syn_type_int4:
            return gcapi::TensorDataType_t::DATA_I4;

        case syn_type_uint4:
            return gcapi::TensorDataType_t::DATA_U4;

        case syn_type_fixed:
            return gcapi::TensorDataType_t::DATA_I8;

        case syn_type_uint8:
            return gcapi::TensorDataType_t::DATA_U8;

        case syn_type_int16:
            return gcapi::TensorDataType_t::DATA_I16;

        case syn_type_uint16:
            return gcapi::TensorDataType_t::DATA_U16;

        case syn_type_int32:
            return gcapi::TensorDataType_t::DATA_I32;

        case syn_type_uint32:
            return gcapi::TensorDataType_t::DATA_U32;

        case syn_type_int64:
            return gcapi::TensorDataType_t::DATA_I64;

        case syn_type_uint64:
            return gcapi::TensorDataType_t::DATA_U64;

        case syn_type_bf16:
            return gcapi::TensorDataType_t::DATA_BF16;

        case syn_type_fp16:
            return gcapi::TensorDataType_t::DATA_F16;

        case syn_type_fp8_152:
            return gcapi::TensorDataType_t::DATA_F8_152;

        case syn_type_fp8_143:
            return gcapi::TensorDataType_t::DATA_F8_143;

        case syn_type_single:
        case syn_type_hb_float:
        case syn_type_tf32:
            return gcapi::TensorDataType_t::DATA_F32;

        case syn_type_na:
        default:
            HB_ASSERT(false, "Unsupported data type");
            return gcapi::TensorDataType_t::NUM_DATATYPES;
    }
}

inline tpc_lib_api::TensorDataType toTpcLibDataType(synDataType type)
{
    switch (type)
    {
        case syn_type_int4:
            return tpc_lib_api::TensorDataType::DATA_I4;

        case syn_type_uint4:
            return tpc_lib_api::TensorDataType::DATA_U4;

        case syn_type_fixed:
            return tpc_lib_api::TensorDataType::DATA_I8;

        case syn_type_uint8:
            return tpc_lib_api::TensorDataType::DATA_U8;

        case syn_type_int16:
            return tpc_lib_api::TensorDataType::DATA_I16;

        case syn_type_uint16:
            return tpc_lib_api::TensorDataType::DATA_U16;

        case syn_type_int32:
            return tpc_lib_api::TensorDataType::DATA_I32;

        case syn_type_uint32:
            return tpc_lib_api::TensorDataType::DATA_U32;

        case syn_type_int64:
            return tpc_lib_api::TensorDataType::DATA_I64;

        case syn_type_uint64:
            return tpc_lib_api::TensorDataType::DATA_U64;

        case syn_type_bf16:
            return tpc_lib_api::TensorDataType::DATA_BF16;

        case syn_type_fp16:
            return tpc_lib_api::TensorDataType::DATA_F16;

        case syn_type_fp8_152:
            return tpc_lib_api::TensorDataType::DATA_F8_152;

        case syn_type_fp8_143:
            return tpc_lib_api::TensorDataType::DATA_F8_143;

        case syn_type_single:
        case syn_type_hb_float:
        case syn_type_tf32:
            return tpc_lib_api::TensorDataType::DATA_F32;

        case syn_type_na:
        default:
            HB_ASSERT(false, "Unsupported data type");
            return tpc_lib_api::TensorDataType::NUM_DATATYPES;
    }
}