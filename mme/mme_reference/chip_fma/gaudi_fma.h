#pragma once

#include "chip_fma.h"

namespace gaudi
{
class GaudiFma : public MmeCommon::ChipFma
{
public:
    static std::unique_ptr<GaudiFma> getGaudiFma(MmeCommon::EMmeDataType inputA,
                                                 MmeCommon::EMmeDataType inputB,
                                                 uint8_t expBiasA,
                                                 uint8_t expBiasB,
                                                 MmeCommon::RoundingMode rmEU = MmeCommon::RoundToNearest);

    GaudiFma(MmeCommon::EMmeDataType inputA,
             MmeCommon::EMmeDataType inputB,
             uint8_t expBiasA,
             uint8_t expBiasB,
             MmeCommon::RoundingMode rmEU = MmeCommon::RoundToNearest);

    ~GaudiFma() = default;

protected:
    static const unsigned c_fma_pipline_depth = 4;
};

class GaudiFp16Fma : public GaudiFma
{
public:
    GaudiFp16Fma(MmeCommon::EMmeDataType inputA,
                 MmeCommon::EMmeDataType inputB,
                 uint8_t expBiasA,
                 uint8_t expBiasB,
                 MmeCommon::RoundingMode rm)
    : GaudiFma(inputA, inputB, expBiasA, expBiasB, rm) {};
    ~GaudiFp16Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;
};

class GaudiBf16Fma : public GaudiFma
{
public:
    GaudiBf16Fma(MmeCommon::EMmeDataType inputA,
                 MmeCommon::EMmeDataType inputB,
                 uint8_t expBiasA,
                 uint8_t expBiasB,
                 MmeCommon::RoundingMode rm)
    : GaudiFma(inputA, inputB, expBiasA, expBiasB, rm) {};
    ~GaudiBf16Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;
};

class GaudiFp32Fma : public GaudiFma
{
public:
    GaudiFp32Fma(MmeCommon::EMmeDataType inputA,
                 MmeCommon::EMmeDataType inputB,
                 uint8_t expBiasA,
                 uint8_t expBiasB,
                 MmeCommon::RoundingMode rm)
    : GaudiFma(inputA, inputB, expBiasA, expBiasB, rm) {};
    ~GaudiFp32Fma() = default;

    virtual float fma_vec(const void* inputA, const void* inputB, unsigned cdSize) const override;
};
}  // namespace gaudi