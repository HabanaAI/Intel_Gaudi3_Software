#pragma once

#include "mme_reference/data_types/non_standard_dtypes.h"
#include "synapse_common_types.h"

namespace detail
{

// This functionality enables us to convert an enum `synDataType` to an actual C++ type.
template<synDataType T>
struct synDataTypeToType
{
};

template<typename T>
struct TypeToSynDataType
{
};

#define MAP_SYN_DATA(SYN, TYPE)                                                                                        \
    template<>                                                                                                         \
    struct synDataTypeToType<SYN>                                                                                      \
    {                                                                                                                  \
        using type = TYPE;                                                                                             \
    };                                                                                                                 \
    template<>                                                                                                         \
    struct TypeToSynDataType<TYPE> : std::integral_constant<synDataType, SYN>                                          \
    {                                                                                                                  \
    }

MAP_SYN_DATA(syn_type_fixed,  int8_t);
MAP_SYN_DATA(syn_type_uint8,  uint8_t);
MAP_SYN_DATA(syn_type_bf16,   bfloat16);
MAP_SYN_DATA(syn_type_fp16, fp16_t);
MAP_SYN_DATA(syn_type_fp8_152, fp8_152_t);
MAP_SYN_DATA(syn_type_fp8_143, fp8_143_t);
MAP_SYN_DATA(syn_type_float, float);
MAP_SYN_DATA(syn_type_int16,  int16_t);
MAP_SYN_DATA(syn_type_uint16, uint16_t);
MAP_SYN_DATA(syn_type_int32,  int32_t);
MAP_SYN_DATA(syn_type_uint32, uint32_t);
MAP_SYN_DATA(syn_type_int64, int64_t);
MAP_SYN_DATA(syn_type_uint64, uint64_t);
} // namespace detail

template<synDataType T>
using AsCppType = typename detail::synDataTypeToType<T>::type;

template<typename T>
constexpr synDataType asSynType()
{
    return detail::TypeToSynDataType<T>::value;
}
