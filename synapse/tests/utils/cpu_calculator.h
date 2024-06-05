#pragma once

#include "graph_compiler/types.h"
#include "include/mme_common/mme_common_enum.h"
#include "synapse_api_types.h"

typedef enum
{
    REFERENCE_OP_FWD,
    REFERENCE_OP_DEDX,
    REFERENCE_OP_DEDW,
    REFERENCE_OP_AB,
    REFERENCE_OP_ABT,
    REFERENCE_OP_ATB,
    REFERENCE_OP_ATBT,
    REFERENCE_OP_TRANSPOSED_DEDX,
} ERepefenceOp;

float getIndexValue(const TSize* sizes, const CoordArray& wrongIdx, synDataType type, void* buffer);

/**
 * Check results between two buffers with tolerance
 *
 * @param desc        IN  Descriptor of first and second data
 * @param firstData   IN  First tensor data
 * @param secondData  IN  Second tensor data
 * @param outIndex    OUT Index to the differ element - endIndex if the return value is true
 *
 * @RET   True if check has passed
 */
bool checkResults(const synTensorDescriptor& desc, char* firstData, char* secondData, CoordArray& outIndex);

bool checkMmeOp(const synTensorDescriptor&    xDesc,
                char*                         xData,
                const synTensorDescriptor&    wDesc,
                char*                         wData,
                const synTensorDescriptor&    yDesc,
                char*                         yData,
                const synConvolution3DParams& convParams,
                ERepefenceOp                  op,
                CoordArray&                   outIndex,
                synDeviceType                 deviceType,
                float*                        expectedResult       = nullptr,
                bool                          usePearsonComparison = false,
                MmeCommon::RoundingMode       roundingMode         = MmeCommon::RoundingMode::RoundToNearest);

bool checkBatchGemmOp(const synTensorDescriptor& xDesc,
                      char*                      xData,
                      const synTensorDescriptor& wDesc,
                      char*                      wData,
                      const synTensorDescriptor& yDesc,
                      char*                      yData,
                      ERepefenceOp               op,
                      CoordArray&                outIndex,
                      float*                     expectedResult,
                      synDeviceType              deviceType,
                      MmeCommon::RoundingMode    roundingMode = MmeCommon::RoundingMode::RoundToNearest);

bool checkMaskedBatchGemmOp(const synTensorDescriptor& xDesc,
                            char*                      xData,
                            const synTensorDescriptor& wDesc,
                            char*                      wData,
                            const synTensorDescriptor& xMaskDesc,
                            char*                      xMaskData,
                            const synTensorDescriptor& wMaskDesc,
                            char*                      wMaskData,
                            const synTensorDescriptor& yDesc,
                            char*                      yData,
                            ERepefenceOp               op,
                            CoordArray&                outIndex,
                            float*                     expectedResult,
                            synDeviceType              deviceType,
                            MmeCommon::RoundingMode    roundingMode = MmeCommon::RoundingMode::RoundToNearest);
/**
 * Check result of fwd convolution with tolerance
 *
 * @param x           IN  Descriptor of x input
 * @param xData       IN  Data of x tensor
 * @param w           IN  Descriptor of w input
 * @param wData       IN  Data of w tensor
 * @param y           IN  Descriptor of y output
 * @param yData       IN  Data of y tensor
 * @param convParams  IN The convolution params
 * @param roundingMode  IN The Rounding Mode value
 * @param outIndex    OUT Index to the differ element - endIndex if the return value is true
 *
 * @RET   True if check has passed
 */
bool checkFwdConvolution(const synTensorDescriptor&    x,
                         char*                         xData,
                         const synTensorDescriptor&    w,
                         char*                         wData,
                         const synTensorDescriptor&    y,
                         char*                         yData,
                         const synConvolution3DParams& convParams,
                         CoordArray&                   outIndex,
                         synDeviceType                 deviceType,
                         MmeCommon::RoundingMode       roundingMode = MmeCommon::RoundingMode::RoundToNearest);

/**
 * Check result of DEDX with tolerance
 *
 * @param y           IN  Descriptor of y input
 * @param yData       IN  Data of y tensor
 * @param w           IN  Descriptor of w input
 * @param wData       IN  Data of w tensor
 * @param x           IN  Descriptor of x output
 * @param xData       IN  Data of x tensor
 * @param convParams  IN The convolution params
 * @param roundingMode  IN The Rounding Mode value
 * @param outIndex    OUT Index to the differ element - endIndex if the return value is true
 *
 * @RET   True if check has passed
 */
bool checkDEDX(const synTensorDescriptor&    y,
               char*                         yData,
               const synTensorDescriptor&    w,
               char*                         wData,
               const synTensorDescriptor&    x,
               char*                         xData,
               const synConvolution3DParams& convParams,
               CoordArray&                   outIndex,
               synDeviceType                 deviceType,
               MmeCommon::RoundingMode       roundingMode = MmeCommon::RoundingMode::RoundToNearest);

/**
 * Check result of DEDW with tolerance
 *
 * @param y           IN  Descriptor of y input
 * @param yData       IN  Data of y tensor
 * @param x           IN  Descriptor of x input
 * @param xData       IN  Data of x tensor
 * @param w           IN  Descriptor of w output
 * @param wData       IN  Data of w tensor
 * @param convParams  IN The convolution params
 * @param roundingMode  IN The Rounding Mode value
 * @param outIndex    OUT Index to the differ element - endIndex if the return value is true
 *
 * @RET   True if check has passed
 */
bool checkDEDW(const synTensorDescriptor&    y,
               char*                         yData,
               const synTensorDescriptor&    x,
               char*                         xData,
               const synTensorDescriptor&    w,
               char*                         wData,
               const synConvolution3DParams& convParams,
               CoordArray&                   outIndex,
               synDeviceType                 deviceType,
               bool                          usePearsonCompare = false,
               MmeCommon::RoundingMode       roundingMode = MmeCommon::RoundingMode::RoundToNearest);

/**
 * Calculate fwd convolution
 *
 * @param x           IN  Descriptor of x input
 * @param xData       IN  Data of x tensor
 * @param w           IN  Descriptor of w input
 * @param wData       IN  Data of w tensor
 * @param y           IN  Descriptor of y output
 * @param result      OUT Result
 * @param convParams  IN The convolution params
 * @param roundingMode  IN The Rounding Mode value
 */
void calculateFwdConvolution(const synTensorDescriptor&    x,
                             char*                         xData,
                             const synTensorDescriptor&    w,
                             char*                         wData,
                             const synTensorDescriptor&    y,
                             char*                         result,
                             const synConvolution3DParams& convParams,
                             synDeviceType                 deviceType,
                             MmeCommon::RoundingMode       roundingMode = MmeCommon::RoundingMode::RoundToNearest);

/**
 * calculate DEDX
 *
 * @param y           IN  Descriptor of y input
 * @param yData       IN  Data of y tensor
 * @param w           IN  Descriptor of w input
 * @param wData       IN  Data of w tensor
 * @param x           IN  Descriptor of x output
 * @param result      OUT Result
 * @param convParams  IN The convolution params
 * @param roundingMode  IN The Rounding Mode value
 */
void calculateDEDX(const synTensorDescriptor&    y,
                   char*                         yData,
                   const synTensorDescriptor&    w,
                   char*                         wData,
                   const synTensorDescriptor&    x,
                   char*                         result,
                   const synConvolution3DParams& convParams,
                   synDeviceType                 deviceType,
                   MmeCommon::RoundingMode       roundingMode = MmeCommon::RoundingMode::RoundToNearest);

/**
 * calculate DEDW
 *
 * @param y           IN  Descriptor of y input
 * @param yData       IN  Data of y tensor
 * @param x           IN  Descriptor of x input
 * @param xData       IN  Data of x tensor
 * @param w           IN  Descriptor of w output
 * @param result      OUT Result
 * @param convParams  IN The convolution params
 * @param roundingMode  IN The Rounding Mode value
 */
void calculateDEDW(const synTensorDescriptor&    y,
                   char*                         yData,
                   const synTensorDescriptor&    x,
                   char*                         xData,
                   const synTensorDescriptor&    w,
                   char*                         result,
                   const synConvolution3DParams& convParams,
                   synDeviceType                 deviceType,
                   MmeCommon::RoundingMode       roundingMode = MmeCommon::RoundingMode::RoundToNearest);

/**
 * calculate Relu
 *
 * @param y           IN  Descriptor of input
 * @param yData       IN  Data of input tensor
 * @param x           IN  Descriptor of output
 * @param result      OUT Result
 */
void calculateRelu(const synTensorDescriptor& inDesc, void* inData, const synTensorDescriptor& outDesc, void* result);

template<class T>
void calculateRelu(T* arr, unsigned size)
{
    for (unsigned i = 0; i < size; i++)
    {
        arr[i] = arr[i] > T(0.0f) ? arr[i] : T(0.0f);
    }
}

void calculateGemm(const synTensorDescriptor& xDesc,
                   char*                      xData,
                   const synTensorDescriptor& wDesc,
                   char*                      wData,
                   const synTensorDescriptor& yDesc,
                   char*                      yData,
                   const synGEMMParams&       params,
                   ERepefenceOp               op,
                   synDeviceType              deviceType,
                   MmeCommon::RoundingMode    roundingMode = MmeCommon::RoundingMode::RoundToNearest);

void calculateAdd(const synTensorDescriptor& desc, char* inA, char* inB, char* out);
