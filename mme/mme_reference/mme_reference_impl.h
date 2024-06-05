#pragma once

#include "mme_reference.h"
#include "chip_fma/chip_fma.h"
#include "convolution_params.h"
#include "include/general_utils.h"
#include "sim_tensor.h"
#include <functional>
#include <iostream>
#include <map>
#include <type_traits>


// Responsible for calculating operations on the CPU,
// in order to check validity of the simulator\chip results.
// results are not bit-accurate as running simulator\chip as the accumulation is
// in different order.
class CPUCalculatorImpl
{
public:
    using InputOutputTypePair = std::pair<MmeCommon::EMmeDataType, MmeCommon::EMmeDataType>;
    using doGemmFunction = std::function<void(CPUCalculatorImpl*,
                                              CommonRefMatrix&,
                                              const CommonRefMatrix&,
                                              const CommonRefMatrix&,
                                              MmeCommon::RoundingMode,
                                              int* zeroPoints)>;
    using doApplyZPFunction =
        std::function<void(CPUCalculatorImpl*, CommonRefMatrix& output, const CommonRefMatrix& inputA, int* zeroPoints)>;

    CPUCalculatorImpl(MmeCommon::ChipType chip,
                  int max_tensor_dims = MME_MAX_TENSOR_DIMS,
                  int max_conv_dims = MME_MAX_CONV_DIMS,
                  unsigned seed = 0,
                  uint32_t lfsrNumRegs = 256,
                  uint32_t* lfsr = nullptr,
                  uint32_t polynomial = 0);
    ~CPUCalculatorImpl() = default;
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

    // perform ReLu on the input tensor - max(0, element) for all elements.
    template<typename T>
    static void doRelu(pMMESimTensor acc);
    static ConvolutionParams createGemmParams();
private:
    static int32_t saturatingRoundingDoublingHighMul(int32_t a, int32_t b, bool* r_bit, bool* s_bit);
    static int32_t saturatingRoundingMultiplyByPot(int32_t x, int exponent);
    static int32_t roundingDivideByPot(int32_t x, int exponent, bool r_bit, bool s_bit);
    static float singleElemPreluFloat(float input, float negScale, float posScale);
    static int32_t singleElemPrelu(int32_t input, int negExp, unsigned negScale, int posExp, unsigned posScale);
    static constexpr int MaxTensorDims = 5;
    static constexpr int MaxConvDims = 4;  // This is how it defined in the specs of all chips: gaudi, gaudi2

    void transposeTensor(MmeSimTensor& outputTensor, const MmeSimTensor& xTensor);
    void reverseWeightsDimS(MmeSimTensor& outputTensor, const MmeSimTensor& inputTensor);

    void doSingleGemm(Matrix& output,
                      const Matrix& inputA,
                      const Matrix& inputB,
                      bool transposeA,
                      bool transposeB,
                      MmeCommon::RoundingMode rm,
                      int* zeroPoints = nullptr);

    void doGemmInternal(MmeSimTensor& output,
                        const MmeSimTensor& inputA,
                        const MmeSimTensor& inputB,
                        bool transposeA,
                        bool transposeB,
                        const ConvolutionParams& params,
                        MmeCommon::EMmeOpType op,
                        MmeCommon::RoundingMode rm,
                        bool clipFp = false,
                        bool clipFpInfIn = false,
                        int* zeroPoints = nullptr);
    template<typename InputT, typename OutputT>
    void doMatrixMultiplicationInt(CommonRefMatrix& output,
                                   const CommonRefMatrix& inputA,
                                   const CommonRefMatrix& inputB,
                                   MmeCommon::RoundingMode rm,
                                   int* zeroPoints);
    template<typename InputT>
    void doMatrixMultiplicationFloat(CommonRefMatrix& output,
                                     const CommonRefMatrix& inputA,
                                     const CommonRefMatrix& inputB,
                                     MmeCommon::RoundingMode rm,
                                     int* zeroPoints);
    template<typename InputT, typename OutputT>
    void applyZeroPointsToResult(CommonRefMatrix& output, const CommonRefMatrix& inputA, int* zeroPoints);
    void apply_H3_2134(CommonRefMatrix& output);

    template<typename InputT, typename OutputT>
    void doGemmWorkerInt(CommonRefMatrix& output,
                         const CommonRefMatrix& inputA,
                         const CommonRefMatrix& inputB,
                         uint64_t startIdx,
                         uint64_t chunkSize,
                         MmeCommon::RoundingMode rm);
    template<typename InputT>
    void doGemmWorkerFloat(CommonRefMatrix& output,
                           const CommonRefMatrix& inputA,
                           const CommonRefMatrix& inputB,
                           uint64_t startIdx,
                           uint64_t chunkSize,
                           MmeCommon::RoundingMode rm);

    void getMatricesDim(const MmeCommon::SizeArray& sizeA,
                        const MmeCommon::SizeArray& sizeB,
                        const MmeCommon::SizeArray& sizeOut,
                        const MmeCommon::EMmeOpType op,
                        const ConvolutionParams& params,
                        CommonRefMatrix& inputA,
                        CommonRefMatrix& inputB,
                        CommonRefMatrix& output);

    // data conversion methods
    void convertToOutputDataType(pMMESimTensor output,
                                 const pMMESimTensor input,
                                 MmeCommon::RoundingMode rm,
                                 bool clipFp,
                                 bool clipInfIn,
                                 bool flushDenorms,
                                 bool stochasticFTZfp8);
    // Dedicated treatment for the zero-CD case
    bool handleZeroCdCase(MmeSimTensor& output, const MmeSimTensor& inputA, const MmeSimTensor& inputB);

    // goya2 reduction related methods
    enum GlpDirection
    {
        GLP_IN = 0,
        GLP_OUT = 1,
    };

    void internalPrelu(pMMESimTensor tensor,
                       const int* negExp,
                       const void* negScale,
                       const int* posExp,
                       const void* posScale);
    void
    reluPrelu(pMMESimTensor acc, bool reluEn, int* pwlNegExp, void* pwlNegScale, int* pwlPosExp, void* pwlPosScale);
    static int64_t internalGemmlowpOut(int32_t in, int glpExp, uint32_t glpScale, int32_t glpZp);
    void
    externalGemmlowpOut(pMMESimTensor in, pMMESimTensor out, int index, int glpExp, uint32_t glpScale, int32_t glpZp);
    void externalGemmlowpIn(pMMESimTensor res,
                            const pMMESimTensor in,
                            const int index,
                            const int* glpExp,
                            const int32_t* glpScale,
                            const int32_t glpZp);
    int32_t internalGemmlowpIn(int32_t in, int glpExp, uint32_t glpScale, int32_t glpZp);
    template<bool isFloat, GlpDirection DIRECTION>
    void externalGemmlowp(pMMESimTensor in,
                          pMMESimTensor out,
                          int* glpExp,
                          int* glpScale,
                          int glpZp,
                          int roundingMode,
                          const int seed);
    template<typename T>
    void addBias(pMMESimTensor acc, pMMESimTensor bias);
    void referenceLUT(pMMESimTensor tensor, uint8_t* lut);
    void restoreRounding(int prevRoundingMode, unsigned prevFlushZeroMode, unsigned prevDenormalsZeroMode);

    // Define a template for FMA operation
    template<typename InputT, typename OutputT>
    OutputT fma(const InputT& a, const InputT& b, const OutputT& c, MmeCommon::RoundingMode rm);

    void setMulFunc(MmeCommon::EMmeDataType typeA,
                    MmeCommon::EMmeDataType typeB,
                    MmeCommon::RoundingMode rm,
                    unsigned fpBiasA,
                    unsigned fpBiasB,
                    bool clipFp,
                    bool clipFpInfIn,
                    MmeCommon::InfNanMode infNanModeA,
                    MmeCommon::InfNanMode infNanModeB);

    static byte* createBufferWithoutStrides(const MmeSimTensor& inputTensor);
    static byte* createBufferWithStrides(const byte* inputBuffer, int elementSize, uint64_t requiredMemorySize,
        const MmeCommon::SizeArray& sizes, const MmeCommon::SizeArray& strides);

    const MmeCommon::ChipType m_chipType;
    const int m_mme_max_tensor_dims;
    const int m_mme_max_conv_dims;
    unsigned m_numOfThreads = 1;
    std::map<InputOutputTypePair, doGemmFunction> m_functionMap;
    std::unique_ptr<MmeCommon::ChipFma> chipFma;
    uint32_t m_numLfsrRegs;
    std::unique_ptr<uint32_t[]> m_lfsr;
    uint16_t m_polynomial;
};
