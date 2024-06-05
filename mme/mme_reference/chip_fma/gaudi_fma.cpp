#include "gaudi_fma.h"

#include "data_types/fp32.h"
#include "data_types/fp8.h"
#include "data_types/fp16.h"
#include "data_types/bfloat16.h"
#include "data_types/tf32.h"

//  Gaudi still uses Gaudi2 fma functions, consider move to gaudi implementaiton
#include "fs_fma_gaudi2.h"
using namespace Gaudi2;

namespace gaudi
{
std::unique_ptr<GaudiFma> GaudiFma::getGaudiFma(MmeCommon::EMmeDataType inputA,
                                                MmeCommon::EMmeDataType inputB,
                                                uint8_t expBiasA,
                                                uint8_t expBiasB,
                                                MmeCommon::RoundingMode rmEU)
{
    switch (inputA)
    {
        default:
            MME_ASSERT(0, "data type not supported by Gaudi");
        case MmeCommon::e_type_fp16:
            return std::make_unique<GaudiFp16Fma>(inputA, inputB, expBiasA, expBiasB, rmEU);
        case MmeCommon::e_type_bf16:
            return std::make_unique<GaudiBf16Fma>(inputA, inputB, expBiasA, expBiasB, rmEU);
        case MmeCommon::e_type_fp32:
            return std::make_unique<GaudiFp32Fma>(inputA, inputB, expBiasA, expBiasB, rmEU);
    }
}

GaudiFma::GaudiFma(MmeCommon::EMmeDataType inputA,
                   MmeCommon::EMmeDataType inputB,
                   uint8_t expBiasA,
                   uint8_t expBiasB,
                   MmeCommon::RoundingMode rm)
//  Gaudi ACC rounding mode is the same as EU rounding mode.
: ChipFma(inputA, inputB, expBiasA, expBiasB, rm, rm)
{
    MME_ASSERT(inputA == inputB, "Gaudi doesnt support mixed input type flavors");
    MME_ASSERT(expBiasA == expBiasB, "Gaudi doesnt support different input biases");
}

//  FMA functions
float GaudiFp16Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    uint16_t* a = (uint16_t*) inputA;
    uint16_t* b = (uint16_t*) inputB;
    uint32_t c[c_fma_pipline_depth] = {0};

    for (int cd = 0; cd < cdSize; cd++)
    {
        unsigned idx = cd % c_fma_pipline_depth;

        // MAC operation
        c[idx] = fma_fp16_fp32((uint16_t) a[cd], (uint16_t) b[cd], c[idx], (uint8_t) m_roundingModeEU);
    }

    uint32_t Idx0 = (cdSize + 0) % c_fma_pipline_depth;
    uint32_t Idx1 = (cdSize + 1) % c_fma_pipline_depth;
    uint32_t Idx2 = (cdSize + 2) % c_fma_pipline_depth;
    uint32_t Idx3 = (cdSize + 3) % c_fma_pipline_depth;

    uint32_t rollup0 = add_fp32(c[Idx0], c[Idx1], (uint8_t) m_roundingModeEU);
    uint32_t rollup1 = add_fp32(c[Idx2], c[Idx3], (uint8_t) m_roundingModeEU);
    uint32_t res = add_fp32(rollup0, rollup1, (uint8_t) m_roundingModeEU);
    return reinterpret_ptr<float>(&res);
}

float GaudiBf16Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    uint16_t* a = (uint16_t*) inputA;
    uint16_t* b = (uint16_t*) inputB;
    uint32_t c[c_fma_pipline_depth] = {0};

    for (int cd = 0; cd < cdSize; cd++)
    {
        unsigned idx = cd % c_fma_pipline_depth;

        // MAC operation
        c[idx] = fma_bfp16_fp32((uint16_t) a[cd], (uint16_t) b[cd], c[idx], (uint8_t) m_roundingModeEU);
    }

    uint32_t Idx0 = (cdSize + 0) % c_fma_pipline_depth;
    uint32_t Idx1 = (cdSize + 1) % c_fma_pipline_depth;
    uint32_t Idx2 = (cdSize + 2) % c_fma_pipline_depth;
    uint32_t Idx3 = (cdSize + 3) % c_fma_pipline_depth;

    uint32_t rollup0 = add_fp32(c[Idx0], c[Idx1], (uint8_t) m_roundingModeEU);
    uint32_t rollup1 = add_fp32(c[Idx2], c[Idx3], (uint8_t) m_roundingModeEU);
    uint32_t res = add_fp32(rollup0, rollup1, (uint8_t) m_roundingModeEU);
    return reinterpret_ptr<float>(&res);
}

float GaudiFp32Fma::fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const
{
    uint32_t* a = (uint32_t*) inputA;
    uint32_t* b = (uint32_t*) inputB;
    uint32_t c[c_fma_pipline_depth] = {0};

    for (int cd = 0; cd < cdSize; cd++)
    {
        unsigned idx = cd % c_fma_pipline_depth;

        // MAC operation
        c[idx] = fma_fp32((uint32_t) a[cd], (uint32_t) b[cd], c[idx], (int) m_roundingModeEU);
    }

    uint32_t Idx0 = (cdSize + 0) % c_fma_pipline_depth;
    uint32_t Idx1 = (cdSize + 1) % c_fma_pipline_depth;
    uint32_t Idx2 = (cdSize + 2) % c_fma_pipline_depth;
    uint32_t Idx3 = (cdSize + 3) % c_fma_pipline_depth;

    uint32_t rollup0 = add_fp32(c[Idx0], c[Idx1], (uint8_t) m_roundingModeEU);
    uint32_t rollup1 = add_fp32(c[Idx2], c[Idx3], (uint8_t) m_roundingModeEU);
    uint32_t res = add_fp32(rollup0, rollup1, (uint8_t) m_roundingModeEU);
    return reinterpret_ptr<float>(&res);
}

}  // namespace gaudi
