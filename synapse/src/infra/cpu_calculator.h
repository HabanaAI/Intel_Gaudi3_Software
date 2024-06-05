#pragma once

#include "synapse_types.h"
#include "quantization_data.h"

/*
 * Forward declaration of custom float types
 */
class Bfloat16;

/**
 * Convolution on CPU
 */
template<typename InputData,
         typename WeightType        = InputData,
         typename OutputData        = InputData,
         typename StorageFormat     = int64_t,
         typename IntermediateClamp = int32_t>
void DoConvolution(const InputData* ifm, const WeightType* weights, OutputData* bias, const InputData* cin, OutputData* ofm,
                   unsigned wIFM, unsigned hIFM, unsigned dIFM, unsigned nIFM,
                   unsigned wOFM, unsigned hOFM, unsigned dOFM, unsigned nOFM,
                   unsigned padded_width, unsigned padded_height, unsigned padded_depth,
                   int padL, int padT, int padF,
                   unsigned kW, unsigned kH, unsigned kD,
                   unsigned dW, unsigned dH, unsigned dD,
                   unsigned dilationHeight, unsigned dilationWidth, unsigned dilationDepth,
                   unsigned batch,
                   synActivationParams* activationParams,
                   const ConvQuantizationParams* qParams = nullptr);

/**
 * 2D Convolution on CPU
 */
template<typename InputData,
         typename WeightType        = InputData,
         typename OutputData        = InputData,
         typename StorageFormat     = int64_t,
         typename IntermediateClamp = int32_t>
void DoConvolution2D(const InputData* ifm, const WeightType* weights, OutputData* bias, const InputData* cin, OutputData* ofm,
                   unsigned wIFM, unsigned hIFM, unsigned nIFM,
                   unsigned wOFM, unsigned hOFM, unsigned nOFM,
                   unsigned padded_width, unsigned padded_height,
                   int padL, int padT,
                   unsigned kW, unsigned kH,
                   unsigned dW, unsigned dH,
                   unsigned dilationHeight, unsigned dilationWidth,
                   unsigned batch,
                   synActivationParams* activationParams,
                   const ConvQuantizationParams* qParams = nullptr);


/**
 * Gemm on CPU
 */
template<typename InputData, typename OutputData, typename StorageFormat = int64_t, typename IntermediateClamp = int32_t>
void DoGEMM_typed(InputData* pA, unsigned aW, unsigned aH,
                  InputData* pB, unsigned bW, unsigned bH,
                  OutputData* pC,
                  double zpA, double zpB, double zpC, double scaleA, double scaleB, double scaleC,
                  OutputData* pbias, OutputData* pCin,
                  bool transposeA = false, bool transposeB = false);

/**
 * Batch gemm on CPU
 */
template<typename InputData, typename OutputData, typename StorageFormat = int64_t, typename IntermediateClamp = int32_t>
void DoBatchGEMM_typed(InputData* pA, unsigned aW, unsigned aH,
                       InputData* pB, unsigned bW, unsigned bH,
                       OutputData* pC,
                       unsigned int batch,
                       double zpA, double zpB, double zpC, double scaleA, double scaleB, double scaleC,
                       OutputData* pbias, OutputData* pCin,
                       bool transposeA = false, bool transposeB = false);

template<typename OutFormat = int32_t, typename InFormat = int64_t>
OutFormat saturate(InFormat in);

template<>
Bfloat16 saturate<Bfloat16, long int>(long int in);

template<>
float saturate<float>(float in);

template<typename InputData>
void copy_into_padded(const InputData* ifm, InputData* padded_ifm, unsigned width, unsigned height, unsigned depth, unsigned channel,
                      unsigned padded_width, unsigned padded_height, int padLeft, int padTop, int padFront);

template<typename InputData>
void copy_into_padded(const InputData* ifm, InputData* padded_ifm, unsigned width, unsigned height, unsigned channel, unsigned padded_width, int padL, int padT);


template<class T>
T scaleOutput(T val, double scaleX, double scaleW, double scaleOutput);

template<class T>
T shiftAndScale(T value, double scaleFactor);

template<class T>
int64_t getIntRep(T value);

template<>
int64_t getIntRep(float value);

template<class T>
T getRepFromInt(int64_t);

template<>
float getRepFromInt(int64_t);

int32_t satRndDblHighMul(int32_t a, int32_t b);

int32_t rndDivByPOT(int32_t x, int exponent);

int32_t satRndMulByPOT(int32_t x, int exponent);

int64_t scaleCIn(int64_t val, double scaleX, double scaleW, double scaleCIn);

#include "cpu_calculator.inl"
