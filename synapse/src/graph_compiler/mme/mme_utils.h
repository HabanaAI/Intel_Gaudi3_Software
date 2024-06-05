#pragma once

#include "defs.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe.h"

#include <string>
#include <map>

namespace mme_utils
{
inline std::string toString(const MmeCommon::EMmeOpType opType)
{
    switch (opType)
    {
        case MmeCommon::e_mme_fwd:
            return "e_mme_fwd";
        case MmeCommon::e_mme_dedx:
            return "e_mme_dedx";
        case MmeCommon::e_mme_transposed_dedx:
            return "e_mme_transposed_dedx";
        case MmeCommon::e_mme_dedw:
            return "e_mme_dedw";
        case MmeCommon::e_mme_ab:
            return "e_mme_ab";
        case MmeCommon::e_mme_atb:
            return "e_mme_atb";
        case MmeCommon::e_mme_abt:
            return "e_mme_abt";
        case MmeCommon::e_mme_atbt:
            return "e_mme_atbt";
        case MmeCommon::e_mme_memcpy:
            return "e_mme_memcpy";
        case MmeCommon::e_mme_trans:
            return "e_mme_trans";
        case MmeCommon::e_mme_reductionAdd:
            return "e_mme_reductionAdd";
        case MmeCommon::e_mme_gemm_transpose:
            return "e_mme_gemm_transpose";
        default:
            HB_ASSERT(0, "opType is not supported");
            return "";
    }
    return {};
}

inline std::string toString(const MmeCommon::RoundingMode rounding)
{
    switch (rounding)
    {
        case MmeCommon::RoundingMode::RoundToNearest:
            return "RoundNearest";
        case MmeCommon::RoundingMode::RoundToZero:
            return " RoundToZero";
        case MmeCommon::RoundingMode::RoundUp:
            return "RoundUp";
        case MmeCommon::RoundingMode::RoundDown:
            return "RoundDown";
        case MmeCommon::RoundingMode::StochasticRounding:
            return "StochasticRounding";
        case MmeCommon::RoundingMode::RoundAwayFromZero:
            return "RoundHalfAwayFromZero";
        case MmeCommon::RoundingMode::StochasticRoundingAndNearest:
            return "StochasticRoundWithRoundNearest";
        default:
            HB_ASSERT(0, "invalid rounding mode");
    }
    return {};
}

inline std::string toString(const MmeCommon::EMmeSignalingMode signaling)
{
    switch (signaling)
    {
        case MmeCommon::e_mme_signaling_none:
            return "None";
        case MmeCommon::e_mme_signaling_once:
            return "Once";
        case MmeCommon::e_mme_signaling_desc:
            return "Desc";
        case MmeCommon::e_mme_signaling_desc_with_store:
            return "DescWithStore";
        case MmeCommon::e_mme_signaling_chunk:
            return "Chunk";
        case MmeCommon::e_mme_signaling_output:
            return "Output";
        case MmeCommon::e_mme_signaling_partial:
            return "Partial";
        default:
            HB_ASSERT(0, "invalid signaling mode");
    }
    return {};
}

inline std::string toString(const MmeCommon::EMmeGeometry geometry)
{
    switch (geometry)
    {
        case MmeCommon::e_mme_geometry_4wx1h:
            return "4Wx1H";
        case MmeCommon::e_mme_geometry_2wx2h:
            return "2Wx2H";
        case MmeCommon::e_mme_geometry_1wx4h:
            return "1Wx4H";
        case MmeCommon::e_mme_geometry_4xw:
            return "4xW";
        case MmeCommon::e_mme_geometry_2xw:
            return "2xW";
        case MmeCommon::e_mme_geometry_2xh:
            return "2xH";
        case MmeCommon::e_mme_geometry_4xh:
            return "4xH";
        default:
            HB_ASSERT(0, "invalid geometry");
    }
    return {};
}

inline std::string toString(MmeCommon::EMmePattern pattern)
{
    switch (pattern)
    {
        case MmeCommon::e_mme_sp_reduction_kfc:
            return "SP_KFC";
        case MmeCommon::e_mme_sp_reduction_fkc:
            return "SP_FKC";
        case MmeCommon::e_mme_sp_reduction_fck:
            return "SP_FCK";
        case MmeCommon::e_mme_sp_reduction_cfk:
            return "SP_CFK";
        case MmeCommon::e_mme_sp_reduction_kcf:
            return "SP_KCF";
        case MmeCommon::e_mme_sp_reduction_ckf:
            return "SP_CKF";
        case MmeCommon::e_mme_z_reduction_ksf:
            return "Z_KSF";
        case MmeCommon::e_mme_z_reduction_skf:
            return "Z_SKF";
        default:
            HB_ASSERT_DEBUG_ONLY(false, "undefined mme pattern");
            return "N\\A";
    }
}

inline std::string toString(const MmeCommon::EMmeDataType dtype)
{
    switch (dtype)
    {
        case MmeCommon::e_type_fp16:
            return "fp16";
        case MmeCommon::e_type_bf16:
            return "bf16";
        case MmeCommon::e_type_fp32:
            return "fp32";
        case MmeCommon::e_type_tf32:
            return "tf32";
        case MmeCommon::e_type_fp8_143:
            return "fp8_143";
        case MmeCommon::e_type_fp8_152:
            return "fp8_152";
        case MmeCommon::e_type_fp32_ieee:
            return "fp32ieee";
        case MmeCommon::e_type_int4:
            return "int4";
        case MmeCommon::e_type_uint4:
            return "uint4";
        case MmeCommon::e_type_int8:
            return "int8";
        case MmeCommon::e_type_uint8:
            return "uint8";
        case MmeCommon::e_type_int16:
            return "int16";
        case MmeCommon::e_type_uint16:
            return "uint16";
        case MmeCommon::e_type_int32:
            return "int32";
        case MmeCommon::e_type_int32_26:
            return "int32_26";
        case MmeCommon::e_type_int32_16:
            return "int32_16";
        default:
            HB_ASSERT(0, "invalid data type");
    }
    return {};
}

inline std::string toString(MmeCommon::EMmeReuseType reuseType)
{
    switch (reuseType)
    {
        case MmeCommon::EMmeReuseType::e_mme_no_reuse:
            return "N/A";
        case MmeCommon::EMmeReuseType::e_mme_1d_reuse_a:
            return "A";
        case MmeCommon::EMmeReuseType::e_mme_1d_reuse_b:
            return "B";
        case MmeCommon::EMmeReuseType::e_mme_2d_reuse_ab:
            return "AB";
        case MmeCommon::EMmeReuseType::e_mme_2d_reuse_ba:
            return "BA";
        default:
            HB_ASSERT(0, "Unsupported reuse type");
    }
    return "N/A";
}

inline MmeCommon::EMmeGeometry strToGeometry(const std::string& str)
{
    static std::map<std::string, MmeCommon::EMmeGeometry> string_to_enum_map = {
        {"4Wx1H", MmeCommon::EMmeGeometry::e_mme_geometry_4wx1h},
        {"2Wx2H", MmeCommon::EMmeGeometry::e_mme_geometry_2wx2h},
        {"1Wx4H", MmeCommon::EMmeGeometry::e_mme_geometry_1wx4h},
        {"4xW", MmeCommon::EMmeGeometry::e_mme_geometry_4xw},
        {"2xW", MmeCommon::EMmeGeometry::e_mme_geometry_2xw},
        {"2xH", MmeCommon::EMmeGeometry::e_mme_geometry_2xh},
        {"4xH", MmeCommon::EMmeGeometry::e_mme_geometry_4xh}};
    auto it = string_to_enum_map.find(str);
    if (it == string_to_enum_map.end())
    {
        HB_ASSERT(0, "Invalid geometry");
        return MmeCommon::EMmeGeometry::e_mme_geometry_nr;
    }
    return it->second;
}

}  // namespace mme_utils
