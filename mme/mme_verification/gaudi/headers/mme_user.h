#pragma once
#ifndef _MME_USER_H_
#define _MME_USER_H_

#include "convolution_params.h"
#include "gaudi/mme.h"
#include "mme_reg_write_cmd.h"
#include "sim_tensor.h"
#include <list>

typedef struct _MmeTestParams
{
    bool randomMD;
    bool wkldIdMD;
    unsigned repeats;
    bool sbResuseInStripes;
    bool incDec;
    bool maskSignals;
} MmeTestParams;

void executeConvFWD(const bool verifMode,
                    unsigned wkldId,
                    const ConvolutionParams* conv,
                    const uint32_t* soAddrLow,
                    const uint32_t soAddrHigh,
                    const MmeSimTensor* x,
                    const MmeSimTensor* w,
                    const MmeSimTensor* y,
                    const MmeCommon::RoundingMode roundingMode,
                    const MmeCommon::RoundingMode conversionRoundingMode,
                    const MmeCommon::MmeStrategy* strategy,
                    const MmeTestParams* testParams,
                    int* targetSoValue,
                    std::list<MmeRegWriteCmd>* cmds,
                    const Mme::Desc* prevDesc,
                    Mme::Desc* lastDesc);

void executeConvDEDX(const bool verifMode,
                     unsigned wkldId,
                     const ConvolutionParams* conv,
                     const uint32_t* soAddrLow,
                     const uint32_t soAddrHigh,
                     const MmeSimTensor* x,
                     const MmeSimTensor* w,
                     const MmeSimTensor* y,
                     const MmeCommon::RoundingMode roundingMode,
                     const MmeCommon::RoundingMode conversionRoundingMode,
                     const MmeCommon::MmeStrategy* strategy,
                     const MmeTestParams* testParams,
                     int* targetSoValue,
                     std::list<MmeRegWriteCmd>* cmds,
                     const Mme::Desc* prevDesc,
                     Mme::Desc* lastDesc);

void executeConvDEDW(const bool verifMode,
                     unsigned wkldId,
                     const ConvolutionParams* conv,
                     const uint32_t* soAddrLow,
                     const uint32_t soAddrHigh,
                     const MmeSimTensor* x,
                     const MmeSimTensor* w,
                     const MmeSimTensor* y,
                     const MmeCommon::RoundingMode roundingMode,
                     const MmeCommon::RoundingMode conversionRoundingMode,
                     const MmeCommon::MmeStrategy* strategy,
                     const MmeTestParams* testParams,
                     int* targetSoValue,
                     std::list<MmeRegWriteCmd>* cmds,
                     const Mme::Desc* prevDesc,
                     Mme::Desc* lastDesc);

void executeBGemm(MmeCommon::EMmeOpType opType,
                  const bool verifMode,
                  unsigned wkldId,
                  const ConvolutionParams* conv,
                  const uint32_t* soAddrLow,
                  const uint32_t soAddrHigh,
                  const MmeSimTensor* x,
                  const MmeSimTensor* w,
                  const MmeSimTensor* y,
                  const MmeCommon::RoundingMode roundingMode,
                  const MmeCommon::RoundingMode conversionRoundingMode,
                  const MmeCommon::MmeStrategy* strategy,
                  const MmeTestParams* testParams,
                  int* targetSoValue,
                  std::list<MmeRegWriteCmd>* cmds,
                  const Mme::Desc* prevDesc,
                  Mme::Desc* lastDesc);

#endif