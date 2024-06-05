#include "gaudi2_fma.h"
#include "fs_fma_gaudi2.h"

#include "data_types/fp32.h"
#include "data_types/fp8.h"
#include "data_types/fp16.h"
#include "data_types/bfloat16.h"
#include "data_types/tf32.h"

namespace Gaudi2
{
std::unique_ptr<Gaudi2Fma> Gaudi2Fma::getGaudi2Fma(MmeCommon::EMmeDataType inputA,
                                                   MmeCommon::EMmeDataType inputB,
                                                   uint8_t expBiasA,
                                                   uint8_t expBiasB,
                                                   bool clipFp)
{
    switch (inputA)
    {
        default:
            MME_ASSERT(0, "data type not supported by Gaudi2");
        case MmeCommon::e_type_fp8_143:
        case MmeCommon::e_type_fp8_152:
            return std::make_unique<Gaudi2Fp8Fma>(inputA, inputB, expBiasA, expBiasB);
        case MmeCommon::e_type_fp16:
            return std::make_unique<Gaudi2Fp16Fma>(inputA, inputB, expBiasA, expBiasB);
        case MmeCommon::e_type_bf16:
            return std::make_unique<Gaudi2Bf16Fma>(inputA, inputB, expBiasA, expBiasB);
        case MmeCommon::e_type_tf32:
            return std::make_unique<Gaudi2Tf32Fma>(inputA, inputB, expBiasA, expBiasB, clipFp);
        case MmeCommon::e_type_fp32:
            return std::make_unique<Gaudi2Fp32Fma>(inputA, inputB, expBiasA, expBiasB);
        case MmeCommon::e_type_fp32_ieee:
            return std::make_unique<Gaudi2Fp32IEEEFma>(inputA, inputB, expBiasA, expBiasB);
    }
}

Gaudi2Fma::Gaudi2Fma(MmeCommon::EMmeDataType inputA,
                     MmeCommon::EMmeDataType inputB,
                     uint8_t expBiasA,
                     uint8_t expBiasB,
                     bool clipFp)
//  Gaudi2 EU and ACC rounding mode is set by HW to round to nearest.
: ChipFma(inputA, inputB, expBiasA, expBiasB, MmeCommon::RoundToNearest, MmeCommon::RoundToNearest, clipFp)
{
    MME_ASSERT(inputA == inputB, "Gaudi2 doesnt support mixed input type flavors");
}

//  FMA functions
float Gaudi2Fp8Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    unsigned fp8_exp, fp8_man;
    if (m_typeA == MmeCommon::e_type_fp8_143)
    {
        fp8_exp = FP8_MODE143_EXP;
        fp8_man = FP8_MODE143_MAN;
    }
    else
    {
        fp8_exp = FP8_MODE152_EXP;
        fp8_man = FP8_MODE152_MAN;
    }

    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint8_t* a = (uint8_t*) inputA;
    uint8_t* b = (uint8_t*) inputB;
    uint16_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
    for (int cd = 0; cd < cdSize; cd += c_fma_tree_addr_width)
    {
        for (int i = 0; i < c_fma_tree_addr_width; i++)
        {
            if (cd + i < cdSize)
            {
                a_tree[i] = fp8_to_fp16(a[cd + i], fp8_exp, fp8_man, m_expBiasA, 0);
                b_tree[i] = fp8_to_fp16(b[cd + i], fp8_exp, fp8_man, m_expBiasB, 0);
            }
            else
            {
                a_tree[i] = 0;
                b_tree[i] = 0;
            }
        }

        // MAC operation
        ui32c = fma_mul_add_tree_N16_K4_add_C_before_norm(a_tree[0],
                                                          b_tree[0],
                                                          a_tree[1],
                                                          b_tree[1],
                                                          a_tree[2],
                                                          b_tree[2],
                                                          a_tree[3],
                                                          b_tree[3],
                                                          a_tree[4],
                                                          b_tree[4],
                                                          a_tree[5],
                                                          b_tree[5],
                                                          a_tree[6],
                                                          b_tree[6],
                                                          a_tree[7],
                                                          b_tree[7],
                                                          a_tree[8],
                                                          b_tree[8],
                                                          a_tree[9],
                                                          b_tree[9],
                                                          a_tree[10],
                                                          b_tree[10],
                                                          a_tree[11],
                                                          b_tree[11],
                                                          a_tree[12],
                                                          b_tree[12],
                                                          a_tree[13],
                                                          b_tree[13],
                                                          a_tree[14],
                                                          b_tree[14],
                                                          a_tree[15],
                                                          b_tree[15],
                                                          ui32c,
                                                          (uint8_t) m_roundingModeEU);
    }
    return (float) c;
}

float Gaudi2Fp16Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint16_t* a = (uint16_t*) inputA;
    uint16_t* b = (uint16_t*) inputB;
    uint32_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
    for (int cd = 0; cd < cdSize; cd += c_fma_tree_addr_width)
    {
        for (int i = 0; i < c_fma_tree_addr_width; i++)
        {
            if (cd + i < cdSize)
            {
                a_tree[i] = fp16_to_fp32(a[cd + i], m_clipFp);
                b_tree[i] = fp16_to_fp32(b[cd + i], m_clipFp);
            }
            else
            {
                a_tree[i] = 0;
                b_tree[i] = 0;
            }
        }

        // MAC operation
        ui32c = fma_mul_add_tree_tf32_N8_K26_add_C_before_norm(a_tree[0],
                                                               b_tree[0],
                                                               a_tree[1],
                                                               b_tree[1],
                                                               a_tree[2],
                                                               b_tree[2],
                                                               a_tree[3],
                                                               b_tree[3],
                                                               a_tree[4],
                                                               b_tree[4],
                                                               a_tree[5],
                                                               b_tree[5],
                                                               a_tree[6],
                                                               b_tree[6],
                                                               a_tree[7],
                                                               b_tree[7],
                                                               ui32c,
                                                               (uint8_t) m_roundingModeEU);
    }

    return (float) c;
}

float Gaudi2Bf16Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint16_t* a = (uint16_t*) inputA;
    uint16_t* b = (uint16_t*) inputB;
    uint32_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
    for (int cd = 0; cd < cdSize; cd += c_fma_tree_addr_width)
    {
        for (int i = 0; i < c_fma_tree_addr_width; i++)
        {
            if (cd + i < cdSize)
            {
                a_tree[i] = bf16_to_fp32(a[cd + i], false);
                b_tree[i] = bf16_to_fp32(b[cd + i], false);
            }
            else
            {
                a_tree[i] = 0;
                b_tree[i] = 0;
            }
        }

        // MAC operation
        ui32c = fma_mul_add_tree_tf32_N8_K26_add_C_before_norm(a_tree[0],
                                                               b_tree[0],
                                                               a_tree[1],
                                                               b_tree[1],
                                                               a_tree[2],
                                                               b_tree[2],
                                                               a_tree[3],
                                                               b_tree[3],
                                                               a_tree[4],
                                                               b_tree[4],
                                                               a_tree[5],
                                                               b_tree[5],
                                                               a_tree[6],
                                                               b_tree[6],
                                                               a_tree[7],
                                                               b_tree[7],
                                                               ui32c,
                                                               (uint8_t) m_roundingModeEU);
    }

    return (float) c;
}

float Gaudi2Tf32Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint32_t* a = (uint32_t*) inputA;
    uint32_t* b = (uint32_t*) inputB;
    uint32_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
    for (int cd = 0; cd < cdSize; cd += c_fma_tree_addr_width)
    {
        for (int i = 0; i < c_fma_tree_addr_width; i++)
        {
            if (cd + i < cdSize)
            {
                a_tree[i] = a[cd + i];
                b_tree[i] = b[cd + i];
            }
            else
            {
                a_tree[i] = 0;
                b_tree[i] = 0;
            }
        }

        // MAC operation
        ui32c =
            fma_mul_add_tree_tf32_N4_K26_add_C_before_norm(a_tree[0],
                                                           b_tree[0],
                                                           a_tree[1],
                                                           b_tree[1],
                                                           a_tree[2],
                                                           b_tree[2],
                                                           a_tree[3],
                                                           b_tree[3],
                                                           ui32c,
                                                           (uint8_t) m_roundingModeEU,
                                                           0 /* make sure to change to value from json in future */);
    }

    return (float) c;
}

float Gaudi2Fp32Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint32_t* a = (uint32_t*) inputA;
    uint32_t* b = (uint32_t*) inputB;
    uint32_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
    for (int cd = 0; cd < cdSize; cd += c_fma_tree_addr_width)
    {
        for (int i = 0; i < c_fma_tree_addr_width; i++)
        {
            if (cd + i < cdSize)
            {
                a_tree[i] = a[cd + i];
                b_tree[i] = b[cd + i];
            }
            else
            {
                a_tree[i] = 0;
                b_tree[i] = 0;
            }
        }

        // MAC operation
        ui32c = fma_mul_add_tree_fp32_N2_K26_add_C_before_norm_fp32_emul(a_tree[0],
                                                                         b_tree[0],
                                                                         a_tree[1],
                                                                         b_tree[1],
                                                                         ui32c,
                                                                         (uint8_t) m_roundingModeEU);
    }
    return (float) c;
}

float Gaudi2Fp32IEEEFma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();
    uint32_t* a = (uint32_t*) inputA;
    uint32_t* b = (uint32_t*) inputB;

    for (int i = 0; i < cdSize; i++)
    {
        ui32c = fma_fp32(a[i], b[i], ui32c, (uint8_t) m_roundingModeEU);
    }
    return (float) c;
}

}  // namespace Gaudi2
