#include "gaudi3_utils.h"
#include "mme_assert.h"

using namespace MmeCommon;

namespace gaudi3
{


EMmeDataType ConvertDataTypeFromGaudi3(Mme::EMmeDataType dt)
{
    switch (dt)
    {
        case Mme::EMmeDataType::e_mme_dt_fp16:
            return e_type_fp16;
        case Mme::EMmeDataType::e_mme_dt_bf16:
            return e_type_bf16;
        case Mme::EMmeDataType::e_mme_dt_fp8:
            return e_type_fp8_143;
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
