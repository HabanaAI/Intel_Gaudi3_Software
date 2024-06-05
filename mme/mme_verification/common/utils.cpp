#include "utils.h"
#include "mme_assert.h"
namespace MmeCommon
{

EMmeOperand getInputFromOperation( EMmeOpType operation, bool isA)
{
    switch (operation)
    {
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
        case e_mme_fwd:
            return isA ? e_mme_op_x : e_mme_op_w;
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            return isA ? e_mme_op_y : e_mme_op_w;
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            return isA ? e_mme_op_x : e_mme_op_y;
        default:
            MME_ASSERT(0, "invalid operation");
            break;
    }
    return EMmeOperand::e_mme_op_x;
}

EMmeOperand getOutputFromOperation(EMmeOpType operation)
{
    switch (operation)
    {
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_gemm_transpose:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
        case e_mme_fwd:
            return e_mme_op_y;
            break;
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            return e_mme_op_x;
            break;
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            return e_mme_op_w;
            break;
        default:
            MME_ASSERT(0, "invalid operation");
            break;
    }
    return EMmeOperand::e_mme_op_y;
}

} // namespace MmeCommon