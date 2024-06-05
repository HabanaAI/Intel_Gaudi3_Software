#pragma once

#include <cstddef>

#include "cpu_calculator.h"
#include "defs.h"
#include "quantization_data.h"
#include "utils.h"
// #include "utils.h"

template<typename InputData,
         typename WeightType,
         typename OutputData,
         typename StorageFormat,
         typename IntermediateClamp>
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
                   const ConvQuantizationParams* qParams)
{
    const InputData* pIFM      = ifm;
    const InputData* pCIn      = cin;
    OutputData*      pOFM      = ofm;
    InputData*       paddedIFM = nullptr;
    StorageFormat*   newOFM    = new StorageFormat[static_cast<unsigned long>(nOFM) * dOFM * hOFM * wOFM];
    paddedIFM = new InputData[static_cast<unsigned long>(padded_depth) * padded_height * padded_width * nIFM];
    InputData* curIFM;

    //Todo: why are zero points doubles?

    double zpA      = qParams ? qParams->x.zp()           : 0;
    double zpB      = qParams ? qParams->w.zp()           : 0;
    double zpC      = qParams ? qParams->out.zp()         : 0;
    double zpCin    = qParams ? qParams->residual.zp()    : 0;
    double scaleA   = qParams ? qParams->x.scale()        : 1.;
    double scaleB   = qParams ? qParams->w.scale()        : 1.;
    double scaleC   = qParams ? qParams->out.scale()      : 1.;
    double scaleCin = qParams ? qParams->residual.scale() : 1.;

    //Todo: non-zero padding
    for (unsigned d = 0; d < padded_depth; ++d)
    {
        for (unsigned h = 0; h < padded_height; ++h)
        {
            for (unsigned w = 0; w < padded_width; ++w)
            {
                for (unsigned z = 0; z < nIFM; ++z)
                {
                    paddedIFM[z + w * nIFM +
                              h * padded_width * nIFM +
                              d * padded_width * padded_height * nIFM] = (InputData)zpA;
                }
            }
        }
    }

    curIFM = paddedIFM +
             (kD / 2) * padded_width * padded_height * nIFM +
             (kH / 2) * padded_width * nIFM +
             (kW / 2) * nIFM;

    for (unsigned b = 0; b < batch; ++b)
    {
        //Init OFM
        for (unsigned n = 0; n < nOFM; ++n)
        {
            for (unsigned d = 0; d < dOFM; ++d)
            {
                for (unsigned h = 0; h < hOFM; ++h)
                {
                    for (unsigned w = 0; w < wOFM; ++w)
                    {
                        StorageFormat cinVal = pCIn ? scaleCIn((double)(pCIn[n + w * nOFM +
                                                                             h * nOFM * wOFM +
                                                                             d * nOFM * wOFM * hOFM]) - zpCin, scaleA, scaleB, scaleCin) : 0;
                        newOFM[n + w * nOFM + h * nOFM * wOFM + (d * nOFM * wOFM* hOFM)] = cinVal;

                    }
                }
            }
        }
        //Init IFM
        copy_into_padded<InputData>(pIFM, paddedIFM, wIFM, hIFM, dIFM, nIFM, padded_width, padded_height, padL, padT, padF);

        for (unsigned nOFMitr = 0; nOFMitr < nOFM; ++nOFMitr)
        {
            OutputData biasVal = 0;
            if (bias != nullptr)
            {
                biasVal = bias[nOFMitr];
            }

            //Init bias
            for (unsigned i = 0; i < hOFM * wOFM * dOFM; ++i)
            {
                /* OFM was previously zeroed */
                newOFM[i * nOFM + nOFMitr] += (StorageFormat)(double)(biasVal);
            }

            for (unsigned nIFMitr = 0; nIFMitr < nIFM; ++nIFMitr)
            {
                const WeightType* wBase = weights + (nIFMitr * nOFM + nOFMitr);
                for (unsigned curD = 0; curD < dOFM; ++curD)
                {
                    for (unsigned curH = 0; curH < hOFM; ++curH)
                    {
                        for (unsigned curW = 0; curW < wOFM; ++curW)
                        {
                            int newOFM_idx = curD * hOFM * wOFM * nOFM +
                                             curH * wOFM * nOFM +
                                             curW * nOFM +
                                             nOFMitr;
                            //Calculate the contribution of the ith feature map to the oth feature map in pixel [curW, curH]
                            for (unsigned ld = 0; ld < kD; ++ld)
                            {
                                int ifm_dz = (int) (ld * dilationDepth) - (int) (kD / 2);
                                for (unsigned lh = 0; lh < kH; ++lh)
                                {
                                    int ifm_dy = (int) (lh * dilationHeight) - (int) (kH / 2);
                                    for (unsigned lw = 0; lw < kW; ++lw)
                                    {
                                        int ifm_dx = (int) (lw * dilationWidth) - (int) (kW / 2);
                                        int ifm_row = (int) curH * dH + ifm_dy;
                                        int ifm_col = (int) curW * dW + ifm_dx;
                                        int ifm_depth = (int) curD * dD + ifm_dz;

                                        StorageFormat aVal = (StorageFormat) (
                                                (double) (curIFM[ifm_depth * (int) padded_width * (int) padded_height * (int) nIFM +
                                                                 ifm_row * (int) padded_width * (int) nIFM +
                                                                 ifm_col * (int) nIFM +
                                                                 (int) nIFMitr]) - zpA);
                                        StorageFormat bVal = (StorageFormat) (
                                                (double) (wBase[ld * kH * kW * nIFM * nOFM +
                                                                lh * kW * nIFM * nOFM +
                                                                lw * nIFM * nOFM]) - zpB);
                                        newOFM[newOFM_idx] += aVal * bVal;
                                        newOFM[newOFM_idx] = saturate<IntermediateClamp, StorageFormat>(
                                                newOFM[newOFM_idx]);
                                    }
                                }
                            }
                        }
                    }
                }//DEPTH
            }
        }
        //Clamp temp storage into pOFM
        for (unsigned p = 0; p < nOFM * dOFM * hOFM * wOFM; ++p)
        {
            if (activationParams->reluEnable)
            {
                newOFM[p] = std::max(newOFM[p], (StorageFormat)0);
            }
            StorageFormat oFinal = scaleOutput<StorageFormat>(newOFM[p], scaleA, scaleB, scaleC);
            pOFM[p] = saturate<OutputData, StorageFormat>(oFinal + (StorageFormat)zpC);
        }
        pIFM += wIFM * hIFM * dIFM * nIFM;
        pOFM += wOFM * hOFM * dOFM * nOFM;
        if (pCIn)
        {
            pCIn += wOFM * hOFM * nOFM * dOFM;
        }
    }
    if (paddedIFM != nullptr)
    {
        delete[] paddedIFM;
    }
    delete[] newOFM;

}


template<typename InputData,
         typename WeightType,
         typename OutputData,
         typename StorageFormat,
         typename IntermediateClamp>
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
                     const ConvQuantizationParams* qParams)
{
    unsigned dIFM = 1;
    unsigned dOFM = 1;
    unsigned padded_depth = 1;
    unsigned padF = 0;
    unsigned kD = 1;
    unsigned dD = 1;
    unsigned dilationDepth = 1;
    DoConvolution<InputData,
                  WeightType,
                  OutputData,
                  StorageFormat,
                  IntermediateClamp>(ifm, weights, bias, cin, ofm,
                  wIFM, hIFM, dIFM, nIFM,
                  wOFM, hOFM, dOFM, nOFM,
                  padded_width, padded_height, padded_depth,
                  padL, padT, padF,
                  kW, kH, kD,
                  dW, dH, dD,
                  dilationHeight, dilationWidth, dilationDepth,
                  batch,
                  activationParams,
                  qParams);
}


template<typename InputData, typename OutputData, typename StorageFormat, typename IntermediateClamp>
void DoGEMM_typed(InputData* pA, unsigned aW, unsigned aH,
                  InputData* pB, unsigned bW, unsigned bH,
                  OutputData* pC,
                  double zpA, double zpB, double zpC, double scaleA, double scaleB, double scaleC,
                  OutputData* pbias, OutputData* pCin,
                  bool transposeA, bool transposeB)
{

    unsigned aRows = transposeA ? aW : aH;

    unsigned bCol = transposeB ? bH : bW;

    unsigned cd = transposeA? aH : aW;
    assert(transposeB ? bW == cd : bH == cd);

    StorageFormat* newC = new StorageFormat[aRows * bCol];

    for (unsigned c_row = 0; c_row < aRows; ++c_row)
    {
        for (unsigned c_col = 0; c_col < bCol; ++c_col)
        {
            StorageFormat* curC = newC + (c_row * bCol + c_col);
            StorageFormat acc = 0;
            if (pbias != nullptr)
            {
                acc += pbias[c_col]; //Todo: no zero point shift here?
            }

            StorageFormat rowSum = 0; //For GEMMLOWP
            for (unsigned e = 0; e < cd; ++e)
            {
                unsigned aIdx = transposeA? c_row + aW * e : c_row * aW + e;
                unsigned bIdx = transposeB? e + bW * c_col : e * bW + c_col;
                StorageFormat aVal = (StorageFormat)((double)(pA[aIdx]) - zpA);
                StorageFormat bVal = (StorageFormat)((double)(pB[bIdx]) - zpB);
                StorageFormat el = aVal * bVal;
                acc += el;
                acc = saturate<IntermediateClamp, StorageFormat>(acc);
                rowSum += aVal;
            }
            acc = scaleOutput<StorageFormat>(acc, scaleA, scaleB, scaleC);
            *curC = saturate<IntermediateClamp, StorageFormat>(acc + zpC);
        }
    }
    for (unsigned c_row = 0; c_row < aRows; ++c_row)
    {
        for (unsigned c_col = 0; c_col < bCol; ++c_col)
        {
            OutputData* curC = pC + (c_row * bCol + c_col);
            StorageFormat cinVal = (StorageFormat)(pCin ? (double)(pCin[c_row * bCol + c_col]) : 0);
            *curC = saturate<OutputData, IntermediateClamp>(newC[c_row * bCol + c_col] + cinVal);
        }
    }

    if (newC != nullptr)
    {
        delete[] newC;
    }
}

template<typename InputData, typename OutputData, typename StorageFormat, typename IntermediateClamp>
void DoBatchGEMM_typed(InputData* pA, unsigned aW, unsigned aH,
                       InputData* pB, unsigned bW, unsigned bH,
                       OutputData* pC,
                       unsigned int batch,
                       double zpA, double zpB, double zpC, double scaleA, double scaleB, double scaleC,
                       OutputData* pbias, OutputData* pCin,
                       bool transposeA, bool transposeB)
{
    unsigned cW = transposeB ? bH : bW;
    unsigned cH = transposeA ? aW : aH;

    for (unsigned int b = 0; b < batch; ++b)
    {
        unsigned int nextA   = b * aW * aH;
        unsigned int nextB   = b * bW * bH;
        unsigned int nextC   = b * cW * cH;
        unsigned int nextCin = pCin ? nextC : 0;

        DoGEMM_typed<InputData, OutputData, StorageFormat, IntermediateClamp>(pA + nextA, aW, aH, pB + nextB, bW, bH, pC + nextC, zpA, zpB, zpC, scaleA, scaleB, scaleC, pbias, pCin + nextCin, transposeA, transposeB);
    }
}

template<typename OutFormat, typename InFormat>
OutFormat saturate(InFormat in)
{
    if (in > std::numeric_limits<OutFormat>::max())
    {
        return std::numeric_limits<OutFormat>::max();
    }

    if (in < std::numeric_limits<OutFormat>::lowest())
    {
        return std::numeric_limits<OutFormat>::lowest();
    }

    return (OutFormat)(in & ((1UL<<(sizeof(OutFormat)*8)) - 1));
}

template<typename InputData>
void copy_into_padded(const InputData* ifm, InputData* padded_ifm, unsigned width, unsigned height, unsigned depth, unsigned channel,
                      unsigned padded_width, unsigned padded_height, int padLeft, int padTop, int padFront)
{
    for (unsigned d = 0; d < depth; ++d)
    {
        for (unsigned r = 0; r < height; ++r)
        {
            for (unsigned c = 0; c < width; ++c)
            {
                for (unsigned z = 0; z < channel; ++z)
                {
                    padded_ifm[ (d + padFront) * padded_height * padded_width * channel +
                                (r + padTop) * padded_width * channel +
                                (c + padLeft) * channel + z] =
                                ifm[(d * height * width * channel) +  r * width * channel + c * channel + z];
                }
            }
        }
    }
}


template<typename InputData>
void copy_into_padded(const InputData* ifm, InputData* padded_ifm, unsigned width, unsigned height, unsigned channel, unsigned padded_width, int padL, int padT)
{
    unsigned depth = 1;
    unsigned padFront = 0;
    unsigned padded_height=1;
    copy_into_padded(ifm, padded_ifm, width, height, depth, channel, padded_width, padded_height, padL, padT, padFront);
}


template<class T>
T scaleOutput(T val, double scaleX, double scaleW, double scaleOutput)
{
    HB_ASSERT(scaleOutput != 0, "cannot scale the output with zero scale");
    double scaleFactor = scaleX * scaleW / scaleOutput;
    return shiftAndScale(val, scaleFactor);
}

template<class T>
T shiftAndScale(T value, double scaleFactor)
{
    int64_t val = getIntRep(value);
    int32_t scale, exp;

    realToFixedPoint(scaleFactor, scale, exp);

    if (exp < 0)
    {
        if (scale != 0)
        {
            val = satRndDblHighMul((int32_t)val, scale);
        }
        val = rndDivByPOT((int32_t)val, -exp);
    }
    else
    {
        val = satRndMulByPOT((int32_t)val, exp);
        if (scale != 0)
        {
            val = satRndDblHighMul((int32_t)val, scale);
        }
    }

    return getRepFromInt<T>(val);
}

template<class T>
int64_t getIntRep(T value)
{
    return (int64_t)value;
}

template<class T>
T getRepFromInt(int64_t value)
{
    return (T)value;
}
