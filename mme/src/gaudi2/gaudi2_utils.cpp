#include "include/gaudi2/gaudi2_utils.h"
#include "mme_assert.h"

using namespace MmeCommon;

namespace Gaudi2
{

Mme::EMmeDataType ConvertDataTypeToGaudi2(EMmeDataType dt)
{
    switch (dt)
    {
        case e_type_fp16:
            return Mme::EMmeDataType::e_mme_dt_fp16;
        case e_type_bf16:
            return Mme::EMmeDataType::e_mme_dt_bf16;
        case e_type_fp32_ieee:
            return Mme::EMmeDataType::e_mme_dt_fp32ieee;
        case e_type_fp8_143:
            return Mme::EMmeDataType::e_mme_dt_fp8_143;
        case e_type_fp8_152:
            return Mme::EMmeDataType::e_mme_dt_fp8_152;
        case e_type_fp32:
            return Mme::EMmeDataType::e_mme_dt_fp32;
        case e_type_tf32:
            return Mme::EMmeDataType::e_mme_dt_tf32;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return Mme::EMmeDataType::e_mme_dt_fp32;
}

EMmeDataType ConvertDataTypeFromGaudi2(Mme::EMmeDataType dt)
{
    switch (dt)
    {
        case Mme::EMmeDataType::e_mme_dt_fp16:
            return e_type_fp16;
        case Mme::EMmeDataType::e_mme_dt_bf16:
            return e_type_bf16;
        case Mme::EMmeDataType::e_mme_dt_fp32ieee:
            return e_type_fp32_ieee;
        case Mme::EMmeDataType::e_mme_dt_fp8_143:
            return e_type_fp8_143;
        case Mme::EMmeDataType::e_mme_dt_fp8_152:
            return e_type_fp8_152;
        case Mme::EMmeDataType::e_mme_dt_fp32:
            return e_type_fp32;
        case Mme::EMmeDataType::e_mme_dt_tf32:
            return e_type_tf32;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return e_type_fp32;
}
}
