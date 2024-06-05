#pragma once
#include <cstdint>
#include <cstring>
#include <random>
#include "convolution_params.h"
#include "include/mme_common/mme_common_enum.h"
#include "sim_tensor.h"

struct ActivationParams
{
    bool resFirst;
    int* pwlNegExp = nullptr;
    void* pwlNegScale = nullptr;
    int* pwlPosExp = nullptr;
    void* pwlPosScale = nullptr;
    int* glpResExp = nullptr;
    int* glpResScale = nullptr;
    int glpResZp = 0;
    int glpOutZp = 0;
    int* glpOutExp = nullptr;
    int* glpOutScale = nullptr;
    uint8_t* lut = nullptr;
    int32_t lfsrVal;

    ActivationParams(bool resFirst,
                     int* pwlNegExp,
                     void* pwlNegScale,
                     int* pwlPosExp,
                     void* pwlPosScale,
                     int* glpResExp,
                     int* glpResScale,
                     int glpResZp,
                     int glpOutZp,
                     int* glpOutExp,
                     int* glpOutScale,
                     uint8_t* lut,
                     int seed)
    : resFirst(resFirst),
      pwlNegExp(pwlNegExp),
      pwlNegScale(pwlNegScale),
      pwlPosExp(pwlPosExp),
      pwlPosScale(pwlPosScale),
      glpResExp(glpResExp),
      glpResScale(glpResScale),
      glpResZp(glpResZp),
      glpOutZp(glpOutZp),
      glpOutExp(glpOutExp),
      glpOutScale(glpOutScale),
      lut(lut)
    {
        std::mt19937 gen(seed + 109);
        std::uniform_int_distribution<int32_t> uniformInt(std::numeric_limits<int32_t>::min(),
                                                          std::numeric_limits<int32_t>::max());
        lfsrVal = uniformInt(gen);
    };
    ActivationParams() { memset(this, 0, sizeof(ActivationParams)); }
};


class Matrix
{
public:
    Matrix(MmeCommon::EMmeDataType type,
           uint64_t height,
           uint64_t width,
           byte* data = nullptr,
           const bool shouldCopyData = true);
    Matrix(MmeCommon::EMmeDataType type,
        uint64_t height,
        uint64_t width,
        uint64_t inputElementStride,
        const byte* inputDataPtr);
    Matrix(const Matrix& other)
    {
        m_type = other.m_type;
        m_matrixDataPtr = nullptr;
        m_matrix = other.m_matrix;
    }

    ~Matrix();

    CommonRefMatrix& getMatrix() { return m_matrix; }
    const CommonRefMatrix& getMatrix() const { return m_matrix; }
    const MmeCommon::EMmeDataType getElementType() const { return m_type; }
    void doTranspose();
    byte* data() { return getMatrix().data.get(); }
    void getDataWithStride(uint64_t height, uint64_t width, byte* outputDataPtr, int outputElementSize, uint64_t outputStride);
    const byte* data() const { return getMatrix().data.get(); }
    uint64_t getSizeInElements() const { return (getMatrix().getMemorySize() / getMatrix().sizeOfDataType); }
    MmeCommon::SizeArray getOffsetOfIndex(uint64_t index) const
    {
        unsigned heightOffset = index / getMatrix().getWidth();  // round down
        unsigned widthOffset = index % getMatrix().getWidth();
        return {heightOffset, widthOffset, 0, 0, 0};
    }
    byte* getElementAt(const MmeCommon::SizeArray offset) { return getMatrix().getElementAt(offset[0], offset[1]); }

private:
    void setInternalMatrixSizes(MmeCommon::EMmeDataType type, uint64_t height, uint64_t width);

    MmeCommon::EMmeDataType m_type;
    byte* m_matrixDataPtr;
    CommonRefMatrix m_matrix;
};
using pCommonMatrix = std::shared_ptr<Matrix>;

class CPUCalculatorImpl;
class CPUCalculator
{
public:
    CPUCalculator(MmeCommon::ChipType chip,
                  int max_tensor_dims = MME_MAX_TENSOR_DIMS,
                  int max_conv_dims = MME_MAX_CONV_DIMS,
                  unsigned seed = 0,
                  uint32_t lfsrNumRegs = 256,
                  uint32_t* lfsr = nullptr,
                  uint32_t polynomial = 0);
    ~CPUCalculator();
    // let's not move this class around.
    CPUCalculator(const CPUCalculator& other) = delete;
    CPUCalculator& operator=(CPUCalculator rhs) = delete;
    void limitNumOfThreads(unsigned numOfThreads);


    // do simple GEMM operation - 2 dim tensors.
    void doGemm(Matrix& output,
                const Matrix& inputA,
                const Matrix& inputB,
                bool transposeA,
                bool transposeB,
                MmeCommon::RoundingMode rm = MmeCommon::RoundToNearest,
                unsigned fpBiasA = 0,
                unsigned fpBiasB = 0,
                bool clipFp = false,
                bool clipFpInfIn = false,
                MmeCommon::InfNanMode infNanModeA = MmeCommon::e_mme_full_inf_nan,
                MmeCommon::InfNanMode infNanModeB = MmeCommon::e_mme_full_inf_nan,
                int* zeroPoints = nullptr);

    void doDma(MmeSimTensor& outputTensor, const MmeSimTensor& xTensor, MmeCommon::EMmeOpType op);

    void doBatchGemm(MmeSimTensor& outputTensor,
                     const MmeSimTensor& xTensor,
                     const MmeSimTensor& wTensor,
                     MmeCommon::EMmeOpType op,
                     MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
                     bool clipFp = false,
                     bool clipFpInfIn = false,
                     int* zeroPoints = nullptr);

    void doConvolution(MmeSimTensor& outputTensor,
                       const MmeSimTensor& xTensor,
                       const MmeSimTensor& wTensor,
                       const MmeSimTensor& yTensor,
                       const ConvolutionParams& params,
                       MmeCommon::EMmeOpType op,
                       MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
                       bool clipFp = false,
                       bool clipFpInfIn = false,
                       int* zeroPoints = nullptr);

    void doActivation(pMMESimTensor output,
                      const pMMESimTensor acc,  // input
                      const pMMESimTensor res,  // residual (cin)
                      const pMMESimTensor bias,  // 1d bias tensor
                      bool relu,
                      MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest,
                      const ActivationParams* actParams = nullptr,
                      bool clipFp = false,
                      bool clipInfIn = false,
                      bool flushDenorms = false,
                      bool stochasticFTZfp8 = false);

    void doMemoryWrite(pMMESimTensor memory,
                       const pMMESimTensor mmeOut,
                       MmeCommon::EMmeReductionOp reductionOp = MmeCommon::e_mme_reduction_none,
                       MmeCommon::EMmeReductionRm reductionRm = MmeCommon::e_mme_reduction_round_half_to_nearest_even,
                       bool clipFp = false);

    void activationGemmlowp(pMMESimTensor out,  // the output tensor.
                            pMMESimTensor acc,  // the input tensor (the output of the conv stage).
                            pMMESimTensor res,  // residual tensor (cin). nullptr when there's no residual.
                            pMMESimTensor bias,  // 1D bias tensor with K elements. (either int32_t or
                                                 // float). nullptr when there's no bias.
                            bool reluEn,  // enable relu.
                            MmeCommon::RoundingMode roundingMode,  // rounding mode for type conversion
                            const ActivationParams* actParams);

private:
  std::unique_ptr<CPUCalculatorImpl> m_impl;
};
