#pragma once

#include "chip_fma.h"

namespace gaudi3
{
class Gaudi3Fma : public MmeCommon::ChipFma
{
public:
    static std::unique_ptr<Gaudi3Fma> getGaudi3Fma(MmeCommon::EMmeDataType inputA,
                                                   MmeCommon::EMmeDataType inputB,
                                                   uint8_t expBiasA,
                                                   uint8_t expBiasB,
                                                   MmeCommon::RoundingMode rmACC = MmeCommon::RoundToZero,
                                                   bool clipFp = false,
                                                   bool clipFpInfIn = false,
                                                   MmeCommon::InfNanMode infNanModeA = MmeCommon::e_mme_full_inf_nan,
                                                   MmeCommon::InfNanMode infNanModeB = MmeCommon::e_mme_full_inf_nan);

    Gaudi3Fma(MmeCommon::EMmeDataType inputA,
              MmeCommon::EMmeDataType inputB,
              uint8_t expBiasA,
              uint8_t expBiasB,
              MmeCommon::RoundingMode rmACC = MmeCommon::RoundToZero,
              bool clipFp = false,
              bool clipFpInfIn = false,
              MmeCommon::InfNanMode infNanModeA = MmeCommon::e_mme_full_inf_nan,
              MmeCommon::InfNanMode infNanModeB = MmeCommon::e_mme_full_inf_nan);

    ~Gaudi3Fma() = default;
};

class Gaudi3Fp8Fma : public Gaudi3Fma
{
public:
    Gaudi3Fp8Fma(MmeCommon::EMmeDataType inputA,
                 MmeCommon::EMmeDataType inputB,
                 uint8_t expBiasA,
                 uint8_t expBiasB,
                 MmeCommon::RoundingMode rmACC,
                 MmeCommon::InfNanMode infNanModeA,
                 MmeCommon::InfNanMode infNanModeB)
    : Gaudi3Fma(inputA, inputB, expBiasA, expBiasB, rmACC, false, false, infNanModeA, infNanModeB) {};
    ~Gaudi3Fp8Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 8;
};

class Gaudi3Fp16Fma : public Gaudi3Fma
{
public:
    Gaudi3Fp16Fma(MmeCommon::EMmeDataType inputA,
                  MmeCommon::EMmeDataType inputB,
                  uint8_t expBiasA,
                  uint8_t expBiasB,
                  MmeCommon::RoundingMode rmACC,
                  MmeCommon::InfNanMode infNanModeA,
                  MmeCommon::InfNanMode infNanModeB)
    : Gaudi3Fma(inputA, inputB, expBiasA, expBiasB, rmACC, false, false, infNanModeA, infNanModeB) {};
    ~Gaudi3Fp16Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 2;
};

class Gaudi3Bf16Fma : public Gaudi3Fma
{
public:
    Gaudi3Bf16Fma(MmeCommon::EMmeDataType inputA,
                  MmeCommon::EMmeDataType inputB,
                  uint8_t expBiasA,
                  uint8_t expBiasB,
                  MmeCommon::RoundingMode rmACC)
    : Gaudi3Fma(inputA, inputB, expBiasA, expBiasB, rmACC) {};
    ~Gaudi3Bf16Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 8;
};

class Gaudi3Tf32Fma : public Gaudi3Fma
{
public:
    Gaudi3Tf32Fma(MmeCommon::EMmeDataType inputA,
                  MmeCommon::EMmeDataType inputB,
                  uint8_t expBiasA,
                  uint8_t expBiasB,
                  MmeCommon::RoundingMode rmACC,
                  bool clipFp,
                  bool clipFpInfIn)
    : Gaudi3Fma(inputA, inputB, expBiasA, expBiasB, rmACC, clipFp, clipFpInfIn) {};
    ~Gaudi3Tf32Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 2;
};

class Gaudi3Fp32Fma : public Gaudi3Fma
{
public:
    Gaudi3Fp32Fma(MmeCommon::EMmeDataType inputA,
                  MmeCommon::EMmeDataType inputB,
                  uint8_t expBiasA,
                  uint8_t expBiasB,
                  MmeCommon::RoundingMode rmACC)
    : Gaudi3Fma(inputA, inputB, expBiasA, expBiasB, rmACC) {};
    ~Gaudi3Fp32Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 1;
};
}  // namespace gaudi3