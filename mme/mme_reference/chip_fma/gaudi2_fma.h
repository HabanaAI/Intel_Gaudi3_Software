#pragma once

#include "chip_fma.h"

namespace Gaudi2
{
class Gaudi2Fma : public MmeCommon::ChipFma
{
public:
    static std::unique_ptr<Gaudi2Fma> getGaudi2Fma(MmeCommon::EMmeDataType inputA,
                                                   MmeCommon::EMmeDataType inputB,
                                                   uint8_t expBiasA,
                                                   uint8_t expBiasB,
                                                   bool clipFp = false);

    Gaudi2Fma(MmeCommon::EMmeDataType inputA,
              MmeCommon::EMmeDataType inputB,
              uint8_t expBiasA,
              uint8_t expBiasB,
              bool clipFp = false);

    ~Gaudi2Fma() = default;
};

class Gaudi2Fp8Fma : public Gaudi2Fma
{
public:
    Gaudi2Fp8Fma(MmeCommon::EMmeDataType inputA, MmeCommon::EMmeDataType inputB, uint8_t expBiasA, uint8_t expBiasB)
    : Gaudi2Fma(inputA, inputB, expBiasA, expBiasB) {};
    ~Gaudi2Fp8Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 16;
};

class Gaudi2Fp16Fma : public Gaudi2Fma
{
public:
    Gaudi2Fp16Fma(MmeCommon::EMmeDataType inputA, MmeCommon::EMmeDataType inputB, uint8_t expBiasA, uint8_t expBiasB)
    : Gaudi2Fma(inputA, inputB, expBiasA, expBiasB) {};
    ~Gaudi2Fp16Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 8;
};

class Gaudi2Bf16Fma : public Gaudi2Fma
{
public:
    Gaudi2Bf16Fma(MmeCommon::EMmeDataType inputA, MmeCommon::EMmeDataType inputB, uint8_t expBiasA, uint8_t expBiasB)
    : Gaudi2Fma(inputA, inputB, expBiasA, expBiasB) {};
    ~Gaudi2Bf16Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 8;
};

class Gaudi2Tf32Fma : public Gaudi2Fma
{
public:
    Gaudi2Tf32Fma(MmeCommon::EMmeDataType inputA,
                  MmeCommon::EMmeDataType inputB,
                  uint8_t expBiasA,
                  uint8_t expBiasB,
                  bool clipFp)
    : Gaudi2Fma(inputA, inputB, expBiasA, expBiasB, clipFp) {};
    ~Gaudi2Tf32Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 4;
};

class Gaudi2Fp32Fma : public Gaudi2Fma
{
public:
    Gaudi2Fp32Fma(MmeCommon::EMmeDataType inputA, MmeCommon::EMmeDataType inputB, uint8_t expBiasA, uint8_t expBiasB)
    : Gaudi2Fma(inputA, inputB, expBiasA, expBiasB) {};
    ~Gaudi2Fp32Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 2;
};

class Gaudi2Fp32IEEEFma : public Gaudi2Fma
{
public:
    Gaudi2Fp32IEEEFma(MmeCommon::EMmeDataType inputA,
                      MmeCommon::EMmeDataType inputB,
                      uint8_t expBiasA,
                      uint8_t expBiasB)
    : Gaudi2Fma(inputA, inputB, expBiasA, expBiasB) {};
    ~Gaudi2Fp32IEEEFma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;

private:
    static const unsigned c_fma_tree_addr_width = 1;
};
}  // namespace Gaudi2