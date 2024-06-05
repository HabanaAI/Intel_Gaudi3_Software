#include "gaudi3_fma.h"
#include "fs_fma_gaudi3.h"

#include "data_types/fp32.h"
#include "data_types/fp8.h"
#include "data_types/fp16.h"
#include "data_types/bfloat16.h"
#include "data_types/tf32.h"

namespace gaudi3
{
std::unique_ptr<Gaudi3Fma> Gaudi3Fma::getGaudi3Fma(MmeCommon::EMmeDataType inputA,
                                                   MmeCommon::EMmeDataType inputB,
                                                   uint8_t expBiasA,
                                                   uint8_t expBiasB,
                                                   MmeCommon::RoundingMode rmACC,
                                                   bool clipFp,
                                                   bool clipFpInfIn,
                                                   MmeCommon::InfNanMode infNanModeA,
                                                   MmeCommon::InfNanMode infNanModeB)
{
    switch (inputA)
    {
        default:
            MME_ASSERT(0, "data type not supported by Gaudi3");
        case MmeCommon::e_type_fp8_143:
        case MmeCommon::e_type_fp8_152:
            return std::make_unique<Gaudi3Fp8Fma>(inputA, inputB, expBiasA, expBiasB, rmACC, infNanModeA, infNanModeB);
        case MmeCommon::e_type_ufp16:
        case MmeCommon::e_type_fp16:
            return std::make_unique<Gaudi3Fp16Fma>(inputA, inputB, expBiasA, expBiasB, rmACC, infNanModeA, infNanModeB);
        case MmeCommon::e_type_bf16:
            return std::make_unique<Gaudi3Bf16Fma>(inputA, inputB, expBiasA, expBiasB, rmACC);
        case MmeCommon::e_type_tf32:
            return std::make_unique<Gaudi3Tf32Fma>(inputA, inputB, expBiasA, expBiasB, rmACC, clipFp, clipFpInfIn);
        case MmeCommon::e_type_fp32:
            return std::make_unique<Gaudi3Fp32Fma>(inputA, inputB, expBiasA, expBiasB, rmACC);
    }
}

Gaudi3Fma::Gaudi3Fma(MmeCommon::EMmeDataType inputA,
                     MmeCommon::EMmeDataType inputB,
                     uint8_t expBiasA,
                     uint8_t expBiasB,
                     MmeCommon::RoundingMode rmACC,
                     bool clipFp,
                     bool clipFpInfIn,
                     MmeCommon::InfNanMode infNanModeA,
                     MmeCommon::InfNanMode infNanModeB)
// Gaudi3 EU rounding mode is set by HW to round to zero
: ChipFma(inputA,
          inputB,
          expBiasA,
          expBiasB,
          MmeCommon::RoundToZero,
          rmACC,
          clipFp,
          clipFpInfIn,
          infNanModeA,
          infNanModeB)
{
    if ((!isTypeFp8(inputA) || !isTypeFp8(inputB)) && (!isTypeFp16(inputA) || !isTypeFp16(inputB)))
    {
        // only fp8 flavors are supported now
        MME_ASSERT(inputA == inputB, "Gaudi3 doesnt support mixed input type flavors");
    }
}

//  FMA functions
float Gaudi3Fp8Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    unsigned fp8_expA, fp8_manA, fp8_expB, fp8_manB;
    if (m_typeA == MmeCommon::e_type_fp8_143)
    {
        fp8_expA = FP8_MODE143_EXP;
        fp8_manA = FP8_MODE143_MAN;
    }
    else
    {
        fp8_expA = FP8_MODE152_EXP;
        fp8_manA = FP8_MODE152_MAN;
    }
    if (m_typeB == MmeCommon::e_type_fp8_143)
    {
        fp8_expB = FP8_MODE143_EXP;
        fp8_manB = FP8_MODE143_MAN;
    }
    else
    {
        fp8_expB = FP8_MODE152_EXP;
        fp8_manB = FP8_MODE152_MAN;
    }

    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint8_t* a = (uint8_t*) inputA;
    uint8_t* b = (uint8_t*) inputB;
    uint8_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
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
        ui32c = fma_mul_add_tree_fp8_N8_K4_add_C_in_tree_no_ftz(a_tree[0],
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
                                                                fp8_expA,
                                                                fp8_expB,
                                                                fp8_manA,
                                                                fp8_manB,
                                                                m_expBiasA,
                                                                m_expBiasB,
                                                                m_infNanModeA,
                                                                m_infNanModeB);
    }
    return (float) c;
}

float Gaudi3Fp16Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    bool isUnsignedA = m_typeA == MmeCommon::e_type_ufp16;
    bool isUnsignedB = m_typeB == MmeCommon::e_type_ufp16;

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
        ui32c = fma_mul_add_tree_cfp16_emul_N2_K4_add_C_in_tree_no_ftz(a_tree[0],
                                                                       b_tree[0],
                                                                       a_tree[1],
                                                                       b_tree[1],
                                                                       ui32c,
                                                                       m_expBiasA,
                                                                       m_expBiasB,
                                                                       isUnsignedA,
                                                                       isUnsignedB,
                                                                       m_infNanModeA,
                                                                       m_infNanModeB);
    }

    return (float) c;
}

float Gaudi3Bf16Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();

    uint16_t* a = (uint16_t*) inputA;
    uint16_t* b = (uint16_t*) inputB;
    uint16_t a_tree[c_fma_tree_addr_width], b_tree[c_fma_tree_addr_width];
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
        ui32c = fma_mul_add_tree_bf16_N8_K4_add_C_in_tree_no_ftz(a_tree[0],
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
                                                                 ui32c);
    }

    return (float) c;
}

float Gaudi3Tf32Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
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
        ui32c = fma_mul_add_tree_tf32_emul_N2_K4_add_C_in_tree_no_ftz(a_tree[0],
                                                                      b_tree[0],
                                                                      a_tree[1],
                                                                      b_tree[1],
                                                                      ui32c,
                                                                      m_clipFp,
                                                                      m_clipFpInfIn);
    }

    return (float) c;
}

float Gaudi3Fp32Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    float32 c(0.0f);
    uint32_t& ui32c = c.value();
    uint32_t* a = (uint32_t*) inputA;
    uint32_t* b = (uint32_t*) inputB;
    for (int i = 0; i < cdSize; i++)
    {
        ui32c = fma_mul_add_tree_fp32_N1_K4_add_C_in_tree_no_ftz_fp32_emul(a[i], b[i], ui32c);
    }

    return (float) c;
}

}  // namespace gaudi3
