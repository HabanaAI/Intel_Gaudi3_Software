#pragma once

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
                                     unsigned              spatialStride);

void calcBatchNormMovingValuesRef(float&    movingMean,
                                  float&    movingVariance,
                                  float     mean,
                                  float     variance,
                                  float     momentum);

template <typename KERNEL_FLAVOR>
void calcBatchNormOutputRef(KERNEL_FLAVOR*        pCurrOutputBuffer,
                            const KERNEL_FLAVOR*  pCurrInputBuffer,
                            float                 bias,
                            float                 scale,
                            unsigned              spatialSize,
                            unsigned              spatialStride);

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
                             const unsigned         dataSizes[]);

template<typename KernelFlavor>
double assesBufferError(float* exp, KernelFlavor *res, unsigned len);


template <typename KERNEL_FLAVOR>
void calcBatchNormGradBetaGradGammaRef(float&                gradBeta,
                                       float&                gradGamma,
                                       const KERNEL_FLAVOR*  pCurrInputBuffer,
                                       const KERNEL_FLAVOR*  pCurrGradOutputBuffer,
                                       float                 mean,
                                       float                 inverseStd,
                                       unsigned              spatialSize,
                                       unsigned              spatialStride);


template <typename KERNEL_FLAVOR>
void calcBatchNormGradNormalizedInput(float&                sumGradNormInput,
                                      float&                sumGradNormInputMulByNormInput,
                                      const KERNEL_FLAVOR*  pCurrInputBuffer,
                                      const KERNEL_FLAVOR*  pCurrGradOutputBuffer,
                                      float                 mean,
                                      float                 inverseStd,
                                      float                 gamma,
                                      unsigned              spatialSize,
                                      unsigned              spatialStride);

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
                            unsigned              spatialStride);


template <typename KERNEL_FLAVOR>
void calcBatchNormBackwardRef(float*                 pGradBetaBufferRef,
                              float*                 pGradGammaBufferRef,
                              KERNEL_FLAVOR*         pGradInputBufferRef,
                              const KERNEL_FLAVOR*   pInputBuffer,
                              const KERNEL_FLAVOR*   pGradOutputBuffer,
                              const float*           pMeanBuffer,
                              const float*           pInverseStdBuffer,
                              const float*           pGammaBuffer,
                              const unsigned         dataDimSizes[]);
