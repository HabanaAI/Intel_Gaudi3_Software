#pragma once
#include <cstdint>
#include <array>

#include "include/mme_common/mme_common_enum.h"

struct ConvolutionParams
{
public:
    constexpr static unsigned maxConvDim = 3;
    int dim = maxConvDim;
    union
    {
        int16_t bf16;
        int32_t f32;
        uint32_t int32 = 0;
    } paddingValue;
    std::array<int, maxConvDim + 1> padding;
    std::array<int, maxConvDim + 1> convStride;
    std::array<int, maxConvDim + 1> dilation;
    bool relu = false;

    ConvolutionParams()
    {
        dilation.fill(1);
        convStride.fill(1);
        padding.fill(0);
    }
};

struct BGemm
{
    bool relu;
};
