#pragma once
#include <cstdlib>
#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

template <typename KERNEL_FLAVOR>
void calcBatchNormMeanAndVarianceRef(float&                mean,
                                     float&                variance,
                                     float&                inverseStd,
                                     float&                bias,
                                     float&                scale,
                                     const KERNEL_FLAVOR*  pCurrInputBuffer,
                                     float                 beta,
                                     float                 gamma,
                                     unsigned              spatialSize,
                                     unsigned              spatialStride)
{
    static const float epsilon = 1e-5f;

    mean     = 0;
    variance = 0;

    const KERNEL_FLAVOR *pElement = pCurrInputBuffer;
    for( unsigned j = 0; j < spatialSize; j++, pElement += spatialStride )
    {
        KERNEL_FLAVOR xi = *pElement;
        mean     += ((float)xi) / spatialSize;
        variance += ((float)xi * (float)xi) / spatialSize;
    }
    variance -= mean * mean;

    inverseStd = 1.0f / std::sqrt(variance + epsilon);

    scale      = gamma * inverseStd;
    bias       = beta - scale * mean;
}

void calcBatchNormMovingValuesRef(float&    movingMean,
                                  float&    movingVariance,
                                  float     mean,
                                  float     variance,
                                  float     momentum)
{
    movingMean     = (momentum * mean)     + ((1 - momentum) * movingMean);
    movingVariance = (momentum * variance) + ((1 - momentum) * movingVariance);
}

template <typename KERNEL_FLAVOR>
void calcBatchNormOutputRef(KERNEL_FLAVOR*        pCurrOutputBuffer,
                            const KERNEL_FLAVOR*  pCurrInputBuffer,
                            float                 bias,
                            float                 scale,
                            unsigned              spatialSize,
                            unsigned              spatialStride)
{
    for( unsigned j = 0; j < spatialSize; j++ )
    {
        *pCurrOutputBuffer = std::fma((float)*pCurrInputBuffer, scale, bias);

        pCurrInputBuffer  += spatialStride;
        pCurrOutputBuffer += spatialStride;
    }
}

template <typename KERNEL_FLAVOR>
void calcBatchNormForwardRef(float*                 pMeanBuffer,
                             float*                 pInverseStdBuffer,
                             float*                 pMovingMean,
                             float*                 pMovingVariance,
                             KERNEL_FLAVOR*         pOutputBuffer,
                             float                  momentum,
                             const KERNEL_FLAVOR*   pInputBuffer,
                             const float*           pBeta,
                             const float*           pGamma,
                             const unsigned         dataSizes[])
{
    // In this case - stride to the next element on the same channel
    const unsigned spatialStride = dataSizes[0];

    const unsigned dimSize1      = dataSizes[1];
    const unsigned dimSize2      = dataSizes[2];
    const unsigned dimSize3      = dataSizes[3];

    const unsigned spatialSize   = dimSize1 * dimSize2 * dimSize3;

    const KERNEL_FLAVOR* pCurrInputBuffer  = pInputBuffer;
    KERNEL_FLAVOR*       pCurrOutputBuffer = pOutputBuffer;
    for ( unsigned i = 0; i < spatialStride; i++ )
    {
        const float beta  = *pBeta;
        const float gamma = *pGamma;

        float& mean           = *pMeanBuffer;
        float& inverseStd     = *pInverseStdBuffer;

        float variance = 0;
        float bias     = 0;
        float scale    = 0;

        mean       = 0;
        inverseStd = 0;

        calcBatchNormMeanAndVarianceRef(mean,
                                        variance,
                                        inverseStd,
                                        bias,
                                        scale,
                                        pCurrInputBuffer,
                                        beta,
                                        gamma,
                                        spatialSize,
                                        spatialStride);

        if (pMovingMean != nullptr && pMovingVariance != nullptr)
        {
            float& movingMean     = *pMovingMean;
            float& movingVariance = *pMovingVariance;
            calcBatchNormMovingValuesRef(movingMean,
                                         movingVariance,
                                         mean,
                                         variance,
                                         momentum);
        }

        if (pCurrOutputBuffer != nullptr)
        {
            calcBatchNormOutputRef<KERNEL_FLAVOR>(pCurrOutputBuffer,
                                                  pCurrInputBuffer,
                                                  bias,
                                                  scale,
                                                  spatialSize,
                                                  spatialStride);
        }

        pMeanBuffer++;
        pInverseStdBuffer++;

        if (pMovingMean != nullptr && pMovingVariance != nullptr)
        {
            pMovingVariance++;
            pMovingMean++;
        }

        pBeta++;
        pGamma++;

        pInputBuffer++;

        if (pOutputBuffer != nullptr)
        {
            pOutputBuffer++;
            pCurrOutputBuffer  = pOutputBuffer;
        }

        pCurrInputBuffer   = pInputBuffer;
    }
}

// returns 0.1 * StdDev(exp) - Sum(|exp - res|)/len
template<typename KernelFlavor>
double assesBufferError(float* exp, KernelFlavor *res, unsigned len)
{
    // StdDev(exp)
    double mean = 0.0;
    double var = 0.0;
    for (int i = 0; i < len; i++)
    {
        mean += exp[i] / len;
        var += (exp[i] * exp[i]) / len;
    }
    var -= mean * mean;
    double stddev = std::sqrt(var);

    double resultEstimate = 0.0;
    for (int i = 0; i < len; i++)
    {
        resultEstimate += std::abs(exp[i] - ((float) res[i])) / len;
    }

    return 0.1 * stddev - resultEstimate;
}

template <typename KERNEL_FLAVOR>
void calcBatchNormGradBetaGradGammaRef(float&                gradBeta,
                                       float&                gradGamma,
                                       const KERNEL_FLAVOR*  pCurrInputBuffer,
                                       const KERNEL_FLAVOR*  pCurrGradOutputBuffer,
                                       float                 mean,
                                       float                 inverseStd,
                                       unsigned              spatialSize,
                                       unsigned              spatialStride)
{
    gradBeta  = 0;
    gradGamma = 0;

    // gradBeta  = sum( gradOutput )
    // gradGamma = sum( gradOutput * (x - mean) ) * inverseStd

    for( unsigned j = 0; j < spatialSize; j++ )
    {
        KERNEL_FLAVOR currInput      = *pCurrInputBuffer;
        KERNEL_FLAVOR currGradOutput = *pCurrGradOutputBuffer;

        gradBeta  += (float)currGradOutput;
        gradGamma += (float)currGradOutput * ((float)currInput - mean);

        pCurrInputBuffer      += spatialStride;
        pCurrGradOutputBuffer += spatialStride;
    }

    gradGamma *= inverseStd;
}

template <typename KERNEL_FLAVOR>
void calcBatchNormGradNormalizedInput(float&                sumGradNormInput,
                                      float&                sumGradNormInputMulByNormInput,
                                      const KERNEL_FLAVOR*  pCurrInputBuffer,
                                      const KERNEL_FLAVOR*  pCurrGradOutputBuffer,
                                      float                 mean,
                                      float                 inverseStd,
                                      float                 gamma,
                                      unsigned              spatialSize,
                                      unsigned              spatialStride)
{
    sumGradNormInput               = 0;
    sumGradNormInputMulByNormInput = 0;

    // sumGradNormInput               = sum( gradOutput * gamma )
    // sumGradNormInputMulByNormInput = sum( ( gradOutput * gamma ) * ( x - mean ) * inverseStd )

    for( unsigned j = 0; j < spatialSize; j++ )
    {
        KERNEL_FLAVOR currInput = *pCurrInputBuffer;
        float currGradNormInput = (float)(*pCurrGradOutputBuffer) * gamma;

        sumGradNormInput               += currGradNormInput;
        sumGradNormInputMulByNormInput += (float)currGradNormInput * ((float)currInput - mean) * inverseStd;

        pCurrInputBuffer      += spatialStride;
        pCurrGradOutputBuffer += spatialStride;
    }
}

template <typename KERNEL_FLAVOR>
void calcBatchNormGradInput(KERNEL_FLAVOR*        pCurrGradInput,
                            float                 sumGradNormInput,
                            float                 sumGradNormInputMulByNormInput,
                            const KERNEL_FLAVOR*  pCurrGradOutputBuffer,
                            const KERNEL_FLAVOR*  pCurrInputBuffer,
                            float                 mean,
                            float                 inverseStd,
                            float                 gamma,
                            unsigned              spatialSize,
                            unsigned              spatialStride)
{
    for( unsigned j = 0; j < spatialSize; j++ )
    {
        float currGradNormInput = (float)(*pCurrGradOutputBuffer) * gamma;
        float gradInput         = 0;

        // de/dx = ( de/dx_norm - (sumDeDx_norm + x_hat * sumDeDx_norm_mul_dx_norm) / spatialSize ) * invStd
        KERNEL_FLAVOR currInput = *pCurrInputBuffer;
        float x_hat = ((float)currInput - mean) * inverseStd;
        gradInput =  (1 / (float)spatialSize) * inverseStd * (spatialSize * currGradNormInput - sumGradNormInput - x_hat * sumGradNormInputMulByNormInput);

        *pCurrGradInput = gradInput;

        pCurrGradOutputBuffer += spatialStride;
        pCurrGradInput        += spatialStride;
        pCurrInputBuffer      += spatialStride;
    }
}

template <typename KERNEL_FLAVOR>
void calcBatchNormBackwardRef(float*                 pGradBetaBufferRef,
                              float*                 pGradGammaBufferRef,
                              KERNEL_FLAVOR*         pGradInputBufferRef,
                              const KERNEL_FLAVOR*   pInputBuffer,
                              const KERNEL_FLAVOR*   pGradOutputBuffer,
                              const float*           pMeanBuffer,
                              const float*           pInverseStdBuffer,
                              const float*           pGammaBuffer,
                              const unsigned         dataDimSizes[])
{
    const unsigned spatialStride = dataDimSizes[0];

    const unsigned dimSize1      = dataDimSizes[1];
    const unsigned dimSize2      = dataDimSizes[2];
    const unsigned dimSize3      = dataDimSizes[3];

    const unsigned spatialSize   = dimSize1 * dimSize2 * dimSize3;

    const KERNEL_FLAVOR* pCurrInputBuffer      = pInputBuffer;
    const KERNEL_FLAVOR* pCurrGradOutputBuffer = pGradOutputBuffer;

    KERNEL_FLAVOR* pCurrGradInputBuffer = pGradInputBufferRef;

    for ( unsigned i = 0; i < spatialStride; i++ )
    {
        const float mean       = *pMeanBuffer;
        const float inverseStd = *pInverseStdBuffer;
        const float gamma      = *pGammaBuffer;

        float& gradBeta  = *pGradBetaBufferRef;
        float& gradGamma = *pGradGammaBufferRef;

        float sumGradNormInput               = 0;
        float sumGradNormInputMulByNormInput = 0;


        calcBatchNormGradBetaGradGammaRef(gradBeta,
                                          gradGamma,
                                          pCurrInputBuffer,
                                          pCurrGradOutputBuffer,
                                          mean,
                                          inverseStd,
                                          spatialSize,
                                          spatialStride);

        calcBatchNormGradNormalizedInput(sumGradNormInput,
                                         sumGradNormInputMulByNormInput,
                                         pCurrInputBuffer,
                                         pCurrGradOutputBuffer,
                                         mean,
                                         inverseStd,
                                         gamma,
                                         spatialSize,
                                         spatialStride);

        calcBatchNormGradInput(pCurrGradInputBuffer,
                               sumGradNormInput,
                               sumGradNormInputMulByNormInput,
                               pCurrGradOutputBuffer,
                               pCurrInputBuffer,
                               mean,
                               inverseStd,
                               gamma,
                               spatialSize,
                               spatialStride);
        pMeanBuffer++;
        pInverseStdBuffer++;

        pGradBetaBufferRef++;
        pGradGammaBufferRef++;

        pInputBuffer++;
        pGradOutputBuffer++;
        pGradInputBufferRef++;

        pCurrInputBuffer      = pInputBuffer;
        pCurrGradOutputBuffer = pGradOutputBuffer;
        pCurrGradInputBuffer  = pGradInputBufferRef;
    }
}

struct BackwardsRefInputs
{
    const float* x;
    const float* dy;
    const float* invStd;
    const float* mean;
    const float* gamma;

    unsigned spatialSize;
    unsigned numberOfchannels;
};

struct BackwardsRefOutputs
{
    float* dx;
    float* dbeta;
    float* dgamma;
};

class BackwardsIntermediateResults
{
public:
    float* xnorm;
    float* xmu;
    float* dxnorm;
    float* dvar;
    float* dmu;

    BackwardsIntermediateResults(const BackwardsRefInputs& in) :
    xnorm   (new float[in.spatialSize * in.numberOfchannels]),
    xmu     (new float[in.spatialSize * in.numberOfchannels]),
    dxnorm  (new float[in.spatialSize * in.numberOfchannels]),
    dvar    (new float[in.numberOfchannels]),
    dmu     (new float[in.numberOfchannels]),
    m_in    (in)
    {
        calcIntermediates();
    }

    ~BackwardsIntermediateResults()
    {
        delete[] xnorm;
        delete[] xmu;
        delete[] dxnorm;
        delete[] dvar;
        delete[] dmu;
    }
private:
    const BackwardsRefInputs& m_in;

    void calcIntermediates()
    {
        calcXNormXMu();
        calcDVarDMu();
    }
    void calcXNormXMu();
    void calcDVarDMu();
};

void calcDBetaDGamma(BackwardsRefInputs& in,
                            BackwardsIntermediateResults &imdRes,
                            BackwardsRefOutputs &out);

void calcDX(BackwardsRefInputs& in,
                   BackwardsIntermediateResults &imdRes,
                   BackwardsRefOutputs &out);

void calcBackwardsRef(BackwardsRefInputs& in, BackwardsRefOutputs& out);
