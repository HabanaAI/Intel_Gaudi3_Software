#include "mme_reference_impl.h"
#include "data_types/non_standard_dtypes.h"
#include "general_utils.h"
#include "im2col.h"
#include "include/mme_common/mme_common_enum.h"
#include <cstdint>
#include <include/general_utils.h>
#include <list>
#include <mme_reference/sim_tensor.h>
#include <mme_reference/sim_tensor_base.h>
#include <numeric>
#include <thread>

using namespace MmeCommon;  // for enums
using namespace gaudi3;  // for numeric operations

Matrix::Matrix(EMmeDataType type, uint64_t height, uint64_t width, byte* data, const bool shouldCopyData) :
    m_type(type), m_matrixDataPtr(nullptr)
{
    setInternalMatrixSizes(type, height, width);

    uint64_t sizeOfDataInBytes = height * width * m_matrix.sizeOfDataType;
    if (data) m_matrix.data = DataBuffer(data, sizeOfDataInBytes, shouldCopyData);
    else
        m_matrix.data = DataBuffer(sizeOfDataInBytes);
}

Matrix::Matrix(EMmeDataType type, uint64_t height, uint64_t width, uint64_t inputStride, const byte* inputDataPtr) :
    m_type(type)
{
    setInternalMatrixSizes(type, height, width);

    uint64_t inputShift = 0;
    m_matrixDataPtr = new byte[height * width * m_matrix.sizeOfDataType];
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            const byte* const inputPtr = inputDataPtr + inputShift++ * m_matrix.sizeOfDataType;
            byte* const outputPtr = m_matrixDataPtr + (i * width + j) * m_matrix.sizeOfDataType;
            std::memcpy(outputPtr, inputPtr, m_matrix.sizeOfDataType);
        }
        inputShift += inputStride;
    }

    const uint64_t sizeOfDataInBytes = height * width * m_matrix.sizeOfDataType;
    m_matrix.data = DataBuffer(m_matrixDataPtr, sizeOfDataInBytes, false);
}

Matrix::~Matrix()
{
    if (m_matrixDataPtr)
    {
        delete[] m_matrixDataPtr;
    }
}

void Matrix::setInternalMatrixSizes(EMmeDataType type, uint64_t height, uint64_t width)
{
    m_matrix.setShape(height, width);
    switch (type)
    {
        case EMmeDataType::e_type_fp16:
        case EMmeDataType::e_type_ufp16:
        case EMmeDataType::e_type_bf16:
            m_matrix.sizeOfDataType = sizeof(bf16_t);
            break;
        case EMmeDataType::e_type_fp8_143:
        case EMmeDataType::e_type_fp8_152:
            m_matrix.sizeOfDataType = sizeof(fp8_152_t);
            break;
        case EMmeDataType::e_type_fp32_ieee:
        case EMmeDataType::e_type_fp32:
            m_matrix.sizeOfDataType = sizeof(float);
            break;
        case EMmeDataType::e_type_tf32:
            m_matrix.sizeOfDataType = sizeof(tf32_t);
            break;
        case EMmeDataType::e_type_uint16:
        case EMmeDataType::e_type_int16:
            m_matrix.sizeOfDataType = sizeof(uint16_t);
            break;
        case EMmeDataType::e_type_uint8:
        case EMmeDataType::e_type_int8:
            m_matrix.sizeOfDataType = sizeof(uint8_t);
            break;
        case EMmeDataType::e_type_uint4:
        case EMmeDataType::e_type_int4:
            m_matrix.sizeOfDataType = sizeof(uint8_t);
            break;
        case EMmeDataType::e_type_int32:
        case EMmeDataType::e_type_int32_26:
        case EMmeDataType::e_type_int32_16:
            m_matrix.sizeOfDataType = sizeof(int32_t);
            break;
        default:
            MME_ASSERT(0, "type not supported");
    }
}

void Matrix::doTranspose()
{
    getMatrix().doTranspose();
}

void Matrix::getDataWithStride(uint64_t height, uint64_t width, byte* outputDataPtr, int outputElementSize,
    uint64_t outputStride)
{
    int outputShift = 0;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            const byte* const inputPtr = m_matrix.data.get() + (i * width + j) * outputElementSize;
            byte* const outputPtr = outputDataPtr + outputShift++ * outputElementSize;
            std::memcpy(outputPtr, inputPtr, outputElementSize);
        }
        outputShift += outputStride;
    }
}

#define REGISTER_TYPE_CALC_INT(inEnumType, outEnumType, inRealType, outRealType)                                       \
    m_functionMap[{inEnumType, outEnumType}] =                                                                         \
        std::mem_fn(&CPUCalculatorImpl::doMatrixMultiplicationInt<inRealType, outRealType>);

#define REGISTER_TYPE_CALC_FLOAT(inEnumType, outEnumType, inRealType)                                                  \
    m_functionMap[{inEnumType, outEnumType}] = std::mem_fn(&CPUCalculatorImpl::doMatrixMultiplicationFloat<inRealType>);

CPUCalculatorImpl::CPUCalculatorImpl(ChipType chip,
                             int max_tensor_dims,
                             int max_conv_dims,
                             unsigned seed,
                             uint32_t lfsrNumRegs,
                             uint32_t* lfsr,
                             uint32_t polynomial)
: m_chipType(chip),
  m_mme_max_tensor_dims(max_tensor_dims),
  m_mme_max_conv_dims(max_conv_dims),
  m_numLfsrRegs(lfsrNumRegs),
  m_lfsr(new uint32_t[m_numLfsrRegs])
{
    // Verify that the dynamic values are equal to the static consts
    MME_ASSERT(m_mme_max_tensor_dims == MaxTensorDims, "dynamic value different than static value");
    MME_ASSERT(m_mme_max_conv_dims == MaxConvDims, "dynamic value different than static value");

    if (lfsr == nullptr)  // lfsr values are not provided, so we generate the
                          // sr_registers to some values
    {
        for (int i = 0; i < m_numLfsrRegs; i++)
        {
            m_lfsr[i] = i + 2;
        }
    }
    else
    {
        for (int i = 0; i < m_numLfsrRegs; i++)
        {
            m_lfsr[i] = lfsr[i];
        }
    }
    if (polynomial == 0)  // polynomial is not provided, so we set it randomly
    {
        std::mt19937 gen(seed);  // seed the generator
        std::uniform_int_distribution<> distr(25, 63);  // define the range
        m_polynomial = distr(gen);
    }
    else
    {
        m_polynomial = polynomial;
    }

    // this map will ease the way we call the internal doGemm function.
    switch (m_chipType)
    {
        case e_mme_Gaudi:
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp32, EMmeDataType::e_type_fp32, uint32_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_bf16, EMmeDataType::e_type_fp32, uint16_t);
            break;
        case e_mme_Gaudi2:
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp32, EMmeDataType::e_type_fp32, uint32_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp32_ieee, EMmeDataType::e_type_fp32, uint32_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_bf16, EMmeDataType::e_type_fp32, uint16_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp16, EMmeDataType::e_type_fp32, uint16_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_tf32, EMmeDataType::e_type_fp32, uint32_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp8_143, EMmeDataType::e_type_fp32, uint8_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp8_152, EMmeDataType::e_type_fp32, uint8_t);
            break;
        case e_mme_Gaudi3:
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp32, EMmeDataType::e_type_fp32, uint32_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_bf16, EMmeDataType::e_type_fp32, uint16_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp16, EMmeDataType::e_type_fp32, uint16_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_ufp16, EMmeDataType::e_type_fp32, uint16_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_tf32, EMmeDataType::e_type_fp32, uint32_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp8_143, EMmeDataType::e_type_fp32, uint8_t);
            REGISTER_TYPE_CALC_FLOAT(EMmeDataType::e_type_fp8_152, EMmeDataType::e_type_fp32, uint8_t);
            break;
        default:
            MME_ASSERT(0, "unknown chip type");
    }
}

// Handle the undefined case of zero CD. Following TF behavior, we also produce an
// output tensor of Zeros
bool CPUCalculatorImpl::handleZeroCdCase(MmeSimTensor& output, const MmeSimTensor& inputA, const MmeSimTensor& inputB)
{
    if ((inputA.getSizeInElements() == 0) || (inputB.getSizeInElements() == 0))
    {
        MME_ASSERT(inputA.getSizeInElements() == inputB.getSizeInElements(),
                   "One of the input tensors is zero-size, while the other is not");

        memset(output.data(), 0, output.getMemorySize());

        return true;
    }
    return false;  // no zero size CD
}

void CPUCalculatorImpl::setMulFunc(EMmeDataType typeA,
                               EMmeDataType typeB,
                               MmeCommon::RoundingMode rm,
                               unsigned fpBiasA,
                               unsigned fpBiasB,
                               bool clipFp,
                               bool clipFpInfIn,
                               InfNanMode infNanModeA,
                               InfNanMode infNanModeB)
{
    chipFma.reset();
    if (!isTypeInteger(typeA))
    {
        chipFma = ChipFma::getChipFma(m_chipType,
                                      typeA,
                                      typeB,
                                      fpBiasA,
                                      fpBiasB,
                                      rm,
                                      rm,
                                      clipFp,
                                      clipFpInfIn,
                                      infNanModeA,
                                      infNanModeB);
    }
}

byte* CPUCalculatorImpl::createBufferWithoutStrides(const MmeSimTensor& inputTensor)
{
    const int elementSize = inputTensor.getElementSize();
    const uint64_t inputWidth = inputTensor.getSize(0);
    const uint64_t inputHeight = inputTensor.getSize(1);
    const uint64_t inputFirstStrideShift = inputTensor.getStride(1) - inputTensor.getSize(0);
    byte* const outputBuffer = new byte[inputTensor.getSize(0) * inputTensor.getSize(1) * inputTensor.getSize(2) *
        inputTensor.getSize(3) * inputTensor.getSize(4) * elementSize];
    uint64_t currentOutputBufferPos = 0;
    for (unsigned b2 = 0; b2 < inputTensor.getSize(4); ++b2)
    {
        const unsigned b2t = inputTensor.getSize(4) == 1 ? 0 : b2;
        for (unsigned b1 = 0; b1 < inputTensor.getSize(3); ++b1)
        {
            const unsigned b1t = inputTensor.getSize(3) == 1 ? 0 : b1;
            for (unsigned b0 = 0; b0 < inputTensor.getSize(2); ++b0)
            {
                const unsigned b0t = inputTensor.getSize(2) == 1 ? 0 : b0;
                const uint64_t currentInputBufferPos = b2 * inputTensor.getStride(4) + b1 * inputTensor.getStride(3) +
                    b0 * inputTensor.getStride(2);
                const byte* const currentInputBuffer = &inputTensor.data()[currentInputBufferPos * elementSize];
                uint64_t inputShift = 0;
                for (int i = 0; i < inputHeight; ++i)
                {
                    for (int j = 0; j < inputWidth; ++j)
                    {
                        const byte* const inputPtr = currentInputBuffer + inputShift++ * elementSize;
                        byte* const outputPtr = outputBuffer + currentOutputBufferPos++ * elementSize;
                        std::memcpy(outputPtr, inputPtr, elementSize);
                    }
                    inputShift += inputFirstStrideShift;
                }
            }
        }
    }

    return outputBuffer;
}

byte* CPUCalculatorImpl::createBufferWithStrides(const byte* inputBuffer, int elementSize, uint64_t requiredMemorySize,
    const MmeCommon::SizeArray& sizes, const MmeCommon::SizeArray& strides)
{
    const uint64_t outputWidth = sizes[0];
    const uint64_t outputHeight = sizes[1];
    const uint64_t outputFirstStrideShift = strides[1] - sizes[0];
    byte* const outputBuffer = new byte[requiredMemorySize];
    uint64_t currentInputBufferPos = 0;
    for (unsigned b2 = 0; b2 < sizes[4]; ++b2)
    {
        const unsigned b2t = sizes[4] == 1 ? 0 : b2;
        for (unsigned b1 = 0; b1 < sizes[3]; ++b1)
        {
            const unsigned b1t = sizes[3] == 1 ? 0 : b1;
            for (unsigned b0 = 0; b0 < sizes[2]; ++b0)
            {
                const unsigned b0t = sizes[2] == 1 ? 0 : b0;
                const uint64_t currentOutputBufferPos = b2* strides[4] + b1 * strides[3] + b0 * strides[2];
                byte* const currentOutputBuffer = &outputBuffer[currentOutputBufferPos * elementSize];
                int outputShift = 0;
                for (int i = 0; i < outputHeight; ++i)
                {
                    for (int j = 0; j < outputWidth; ++j)
                    {
                        const byte* const inputPtr = inputBuffer + currentInputBufferPos++ * elementSize;
                        byte* const outputPtr = currentOutputBuffer + outputShift++ * elementSize;
                        std::memcpy(outputPtr, inputPtr, elementSize);
                    }
                    outputShift += outputFirstStrideShift;
                }
            }
        }
    }

    return outputBuffer;
}

/* Reverse the s dim of the weights tensor as follows - outputTensor[c][k][S-1 - s][r][q] = inputTensor[c][k][s][r][q]
   Used specifically for transposed DEDX operation - in which specifically S dim is reversed by GC
 */
void CPUCalculatorImpl::reverseWeightsDimS(MmeSimTensor& outputTensor, const MmeSimTensor& inputTensor)
{
    // create new tensors
    EMmeDataType dt = outputTensor.getElementType();
    InfNanMode infNanMode = outputTensor.getInfNanMode();
    unsigned fpBias = outputTensor.getFpBias();
    MmeCommon::SizeArray outputSizes = outputTensor.getSizes();
    MmeCommon::SizeArray inputSizes = inputTensor.getSizes();

    MmeSimTensor newOutputTensor(outputSizes, MAX_DIMENSION, dt, outputTensor.data(), fpBias, infNanMode);
    MmeSimTensor newInputTensor(inputSizes, MAX_DIMENSION, dt, inputTensor.data(), fpBias, infNanMode);

    for (unsigned q = 0; q < newOutputTensor.getSize(4); ++q)
    {
        for (unsigned r = 0; r < newOutputTensor.getSize(3); ++r)
        {
            for (unsigned s = 0; s < newOutputTensor.getSize(2); ++s)
            {
                unsigned sDimSize = newInputTensor.getSize(2);
                uint64_t currentInputPos = q * newInputTensor.getStride(4) + r * newInputTensor.getStride(3) +
                                           (sDimSize - 1 - s) * newInputTensor.getStride(2);
                char* currentGemmInputPtr = &newInputTensor.data()[newInputTensor.getElementSize() * currentInputPos];
                uint64_t tensorSingleGemmWidth = newInputTensor.getSize(0);
                uint64_t tensorSingleGemmHeight = newInputTensor.getSize(1);
                Matrix currentInputA(newInputTensor.getElementType(),
                                     tensorSingleGemmHeight,
                                     tensorSingleGemmWidth,
                                     currentGemmInputPtr,
                                     false);

                uint64_t currentOutputPos = q * newOutputTensor.getStride(4) + r * newOutputTensor.getStride(3) +
                                            s * newOutputTensor.getStride(2);
                char* currentGemmOutputPtr =
                    &newOutputTensor.data()[newOutputTensor.getElementSize() * currentOutputPos];
                memcpy(currentGemmOutputPtr,
                       currentInputA.data(),
                       tensorSingleGemmWidth * tensorSingleGemmHeight * newOutputTensor.getElementSize());
            }
        }
    }
}

void CPUCalculatorImpl::transposeTensor(MmeSimTensor& outputTensor, const MmeSimTensor& xTensor)
{
    // Validate inputs
    MME_ASSERT(validateTranspose(xTensor.getSizes(), outputTensor.getSizes()), "transpose dimensions mismatch");

    // Create dense transpose operands
    DataBuffer inDenseBuf(createBufferWithoutStrides(xTensor), xTensor.getMemorySize(), false);
    inDenseBuf.setShoudFree(true);
    MmeSimTensor denseIn(xTensor.getSizes(),
                                MAX_DIMENSION,
                                outputTensor.getElementType(),
                                inDenseBuf.get(),
                                outputTensor.getFpBias(),
                                outputTensor.getInfNanMode(),
                                nullptr);
    MmeSimTensor denseOut(outputTensor.getSizes(),
                          MAX_DIMENSION,
                          outputTensor.getElementType(),
                          nullptr,
                          outputTensor.getFpBias(),
                          outputTensor.getInfNanMode(),
                          nullptr);

    // Iterate over batch dims and perform a 2D transpose in each iteration
    for (unsigned b2 = 0; b2 < denseOut.getSize(4); ++b2)
    {
        for (unsigned b1 = 0; b1 < denseOut.getSize(3); ++b1)
        {
            for (unsigned b0 = 0; b0 < denseOut.getSize(2); ++b0)
            {
                const MmeCommon::SizeArray batchCoords = {b0,b1,b2};
                // Transpose2D offset is the same for dense input and output
                // offset = (batchCoord[0]*stride[2] + ... + batchCoord[2]*stride[4])*elementSize
                const uint64_t transpose2DByteOffset = std::inner_product(std::next(denseIn.getStrides().begin(), 2),
                                                                          denseIn.getStrides().end(),
                                                                          batchCoords.begin(),
                                                                          0) *
                                                       denseIn.getElementSize();
                Matrix dense2DInputMatrix(denseIn.getElementType(),
                                     denseIn.getSize(1),
                                     denseIn.getSize(0),
                                     &denseIn.data()[transpose2DByteOffset],
                                     false);

                dense2DInputMatrix.getMatrix().doTranspose();
                const auto transposedSize = denseIn.getStride(2) * denseIn.getElementSize();
                std::memcpy(&denseOut.data()[transpose2DByteOffset], dense2DInputMatrix.data(), transposedSize);
            }
        }
    }

    // Write dense output to a sparse buffer
    DataBuffer sparseOutputBuf(createBufferWithStrides(denseOut.data(),
                                                        outputTensor.getElementSize(),
                                                        outputTensor.getMemorySize(),
                                                        outputTensor.getSizes(),
                                                        outputTensor.getStrides()),
                                outputTensor.getMemorySize(),
                                false);
    sparseOutputBuf.setShoudFree(true);
    outputTensor.setData(sparseOutputBuf);
}

void CPUCalculatorImpl::doDma(MmeSimTensor& outputTensor, const MmeSimTensor& xTensor, MmeCommon::EMmeOpType op)
{
    switch (op)
    {
        case e_mme_memcpy:
            outputTensor = xTensor;
            return;
        case e_mme_gemm_transpose:
        case e_mme_trans:
            transposeTensor(outputTensor, xTensor);
            return;
        default:
            MME_ASSERT(0, "invalid dma op");
    }
}

void CPUCalculatorImpl::doBatchGemm(MmeSimTensor& outputTensor,
                                const MmeSimTensor& xTensor,
                                const MmeSimTensor& wTensor,
                                EMmeOpType op,
                                RoundingMode rm,
                                bool clipFp,
                                bool clipFpInfIn,
                                int* zeroPoints)
{
    setMulFunc(xTensor.getElementType(),
               wTensor.getElementType(),
               rm,
               xTensor.getFpBias(),
               wTensor.getFpBias(),
               clipFp,
               clipFpInfIn,
               xTensor.getInfNanMode(),
               wTensor.getInfNanMode());
    for (unsigned dim = 2; dim < 5; ++dim)
    {
        MME_ASSERT((xTensor.getSize(dim) == outputTensor.getSize(dim) || xTensor.getSize(dim) == 1) &&
                       (wTensor.getSize(dim) == outputTensor.getSize(dim) || wTensor.getSize(dim) == 1),
                   "batch dims doesnt match !");
    }

    if (handleZeroCdCase(outputTensor, xTensor, wTensor))
    {
        return;
    }

    uint64_t xTensorSingleGemmWidth = xTensor.getSize(0);
    uint64_t xTensorSingleGemmHeight = xTensor.getSize(1);

    uint64_t wTensorSingleGemmWidth = wTensor.getSize(0);
    uint64_t wTensorSingleGemmHeight = wTensor.getSize(1);

    uint64_t outputTensorSingleGemmWidth = outputTensor.getSize(0);
    uint64_t outputTensorSingleGemmHeight = outputTensor.getSize(1);

    bool transA = (op == EMmeOpType::e_mme_atb || op == EMmeOpType::e_mme_atbt);
    bool transB = (op == EMmeOpType::e_mme_abt || op == EMmeOpType::e_mme_atbt);

    const int xElementSize = xTensor.getElementSize();
    const uint64_t xTensorFirstStrideShift = xTensor.getStride(1) - xTensor.getSize(0);
    const int wElementSize = wTensor.getElementSize();
    const uint64_t wTensorFirstStrideShift = wTensor.getStride(1) - wTensor.getSize(0);
    const int outputElementSize = outputTensor.getElementSize();
    const uint64_t outputTensorFirstStrideShift = outputTensor.getStride(1) - outputTensor.getSize(0);
    for (unsigned b2 = 0; b2 < outputTensor.getSize(4); ++b2)
    {
        unsigned b2w = wTensor.getSize(4) == 1 ? 0 : b2;
        unsigned b2x = xTensor.getSize(4) == 1 ? 0 : b2;
        for (unsigned b1 = 0; b1 < outputTensor.getSize(3); ++b1)
        {
            unsigned b1w = wTensor.getSize(3) == 1 ? 0 : b1;
            unsigned b1x = xTensor.getSize(3) == 1 ? 0 : b1;
            for (unsigned b0 = 0; b0 < outputTensor.getSize(2); ++b0)
            {
                unsigned b0w = wTensor.getSize(2) == 1 ? 0 : b0;
                unsigned b0x = xTensor.getSize(2) == 1 ? 0 : b0;
                uint64_t currentXPos =
                    b2x * xTensor.getStride(4) + b1x * xTensor.getStride(3) + b0x * xTensor.getStride(2);
                byte* currentGemmXPtr = &xTensor.data()[xElementSize * currentXPos];
                Matrix currentInputA(xTensor.getElementType(),
                                     xTensorSingleGemmHeight,
                                     xTensorSingleGemmWidth,
                                     xTensorFirstStrideShift,
                                     currentGemmXPtr);

                uint64_t currentWPos =
                    b2w * wTensor.getStride(4) + b1w * wTensor.getStride(3) + b0w * wTensor.getStride(2);
                char* currentGemmWPtr = &wTensor.data()[wTensor.getElementSize() * currentWPos];
                Matrix currentInputB(wTensor.getElementType(),
                                     wTensorSingleGemmHeight,
                                     wTensorSingleGemmWidth,
                                     wTensorFirstStrideShift,
                                     currentGemmWPtr);

                Matrix currentOutput(outputTensor.getElementType(),
                                     outputTensorSingleGemmHeight,
                                     outputTensorSingleGemmWidth,
                                     nullptr,
                                     false);

                doSingleGemm(currentOutput, currentInputA, currentInputB, transA, transB, rm, zeroPoints);

                const uint64_t currentOutputPos =
                    b2 * outputTensor.getStride(4) + b1 * outputTensor.getStride(3) + b0 * outputTensor.getStride(2);
                byte* const currentGemmOutputPtr = &outputTensor.data()[outputElementSize * currentOutputPos];
                currentOutput.getDataWithStride(outputTensorSingleGemmHeight, outputTensorSingleGemmWidth,
                    currentGemmOutputPtr, outputElementSize, outputTensorFirstStrideShift);
            }
        }
    }
}

void CPUCalculatorImpl::doGemm(Matrix& output,
                           const Matrix& inputA,
                           const Matrix& inputB,
                           bool transposeA,
                           bool transposeB,
                           RoundingMode rm,
                           unsigned fpBiasA,
                           unsigned fpBiasB,
                           bool clipFp,
                           bool clipFpInfIn,
                           InfNanMode infNanModeA,
                           InfNanMode infNanModeB,
                           int* zeroPoints)
{
    setMulFunc(inputA.getElementType(),
               inputB.getElementType(),
               rm,
               fpBiasA,
               fpBiasB,
               clipFp,
               clipFpInfIn,
               infNanModeA,
               infNanModeB);
    doSingleGemm(output, inputA, inputB, transposeA, transposeB, rm, zeroPoints);
}

void CPUCalculatorImpl::doSingleGemm(Matrix& output,
                                 const Matrix& inputA,
                                 const Matrix& inputB,
                                 bool transposeA,
                                 bool transposeB,
                                 RoundingMode rm,
                                 int* zeroPoints)
{
    CommonRefMatrix& outputMat = output.getMatrix();
    CommonRefMatrix inputAMat = inputA.getMatrix();
    CommonRefMatrix inputBMat = inputB.getMatrix();
    if (transposeA)
    {
        Matrix inputAT = inputA;
        inputAT.doTranspose();
        inputAMat = inputAT.getMatrix();
    }
    if (transposeB)
    {
        Matrix inputBT = inputB;
        inputBT.doTranspose();
        inputBMat = inputBT.getMatrix();
    }

    MME_ASSERT(m_functionMap.count({inputA.getElementType(), output.getElementType()}),
               "type is not registered in function map !");
    // call CPUCalculatorImpl::doMatrixMultiplication with the correct data types
    auto doGemmFunc = m_functionMap.at({inputA.getElementType(), output.getElementType()});
    doGemmFunc(this, outputMat, inputAMat, inputBMat, rm, zeroPoints);
}

void CPUCalculatorImpl::doConvolution(MmeSimTensor& outputTensor,
                                  const MmeSimTensor& xTensor,
                                  const MmeSimTensor& wTensor,
                                  const MmeSimTensor& yTensor,
                                  const ConvolutionParams& params,
                                  EMmeOpType op,
                                  RoundingMode rm,
                                  bool clipFp,
                                  bool clipFpInfIn,
                                  int* zeroPoints)
{
    switch (op)
    {
        case EMmeOpType::e_mme_fwd:
            // X*W = Y
            doGemmInternal(outputTensor,
                           xTensor,
                           wTensor,
                           false,
                           false,
                           params,
                           op,
                           rm,
                           clipFp,
                           clipFpInfIn,
                           zeroPoints);
            break;
        case EMmeOpType::e_mme_transposed_dedx:
        {
            SizeArray wSizesNew = wTensor.getSizes();
            wSizesNew[0] = wTensor.getSize(1);
            wSizesNew[1] = wTensor.getSize(0);
            MmeSimTensor wTensorTransposed = MmeSimTensor(wSizesNew,
                                                          wTensor.getDim(),
                                                          wTensor.getElementType(),
                                                          nullptr,
                                                          wTensor.getFpBias(),
                                                          wTensor.getInfNanMode());
            transposeTensor(wTensorTransposed, wTensor);
            MmeSimTensor wTensorReversedS = MmeSimTensor(wSizesNew,
                                                         wTensorTransposed.getDim(),
                                                         wTensorTransposed.getElementType(),
                                                         nullptr,
                                                         wTensorTransposed.getFpBias(),
                                                         wTensorTransposed.getInfNanMode());
            reverseWeightsDimS(wTensorReversedS, wTensorTransposed);
            // dY*Wt = dX
            doGemmInternal(outputTensor, yTensor, wTensorReversedS, false, true, params, op, rm, clipFp, clipFpInfIn);
            break;
        }
        case EMmeOpType::e_mme_dedx:
            // dY*Wt = dX
            doGemmInternal(outputTensor, yTensor, wTensor, false, true, params, op, rm, clipFp, clipFpInfIn);
            break;
        case EMmeOpType::e_mme_dedw:
            // Xt * dY = dW
            doGemmInternal(outputTensor, xTensor, yTensor, true, false, params, op, rm, clipFp, clipFpInfIn);
            break;
        default:
            MME_ASSERT(0, "invalid operation");
    }
}

void CPUCalculatorImpl::doActivation(pMMESimTensor output,
                                 const pMMESimTensor acc,  // input
                                 const pMMESimTensor res,  // residual (cin)
                                 const pMMESimTensor bias,  // 1d bias tensor
                                 bool relu,
                                 RoundingMode rm,
                                 const ActivationParams* actParams,
                                 bool clipFp,
                                 bool clipInfIn,
                                 bool flushDenorms,
                                 bool stochasticFTZfp8)
{
    if (actParams != nullptr)
    {
        activationGemmlowp(output, acc, res, bias, relu, rm, actParams);
        return;
    }

    MME_ASSERT(acc->getElementType() == EMmeDataType::e_type_fp32, "acc element type should be fp32");
    MME_ASSERT(acc->getDim() == output->getDim(), "dims number of outputTensor doesnt match acc output");
    MME_ASSERT(acc->getSizes() == output->getSizes(), "sizes of output Tensot doesnt match acc output");

    if (relu)
    {
        doRelu<float>(acc);
    }

    convertToOutputDataType(output, acc, rm, clipFp, clipInfIn, flushDenorms, stochasticFTZfp8);
}

uint8_t convertDataTypeToDataTypeRMW(EMmeDataType dataType)
{
    //  this check should be done per chip.
    //  Gaudi2 supports fp8 reduction but gaudi3 doesnt.
    switch (dataType)
    {
        case EMmeDataType::e_type_bf16:
            return ST_TNSR_RMW_BF16;
        case EMmeDataType::e_type_ufp16:
            MME_ASSERT(0, "ufp16 is not supported in reduction");
            break;
        case EMmeDataType::e_type_fp16:
            return ST_TNSR_RMW_FP16;
        case EMmeDataType::e_type_fp32:
            return ST_TNSR_RMW_FP32;
        case EMmeDataType::e_type_fp8_143:
            MME_ASSERT(0, "fp8_143 is not supported in reduction");
            break;
        case EMmeDataType::e_type_fp8_152:
            return ST_TNSR_RMW_FP8;
        default:
            MME_ASSERT(0, "Unsupported data type when converting to reduction RMW data type");
            break;
    }
    return -1;
}

void CPUCalculatorImpl::doMemoryWrite(pMMESimTensor memory,
                                  const pMMESimTensor mmeOut,
                                  EMmeReductionOp reductionOp,
                                  EMmeReductionRm reductionRm,
                                  bool clipFp)
{
    uint64_t numTensorBytes = memory->getMemorySize();
    if (reductionOp == e_mme_reduction_none)
    {
        memcpy(memory->data(), mmeOut->data(), numTensorBytes);
        return;
    }

    EMmeDataType dataType = memory->getElementType();
    uint8_t dataTypeRMW = convertDataTypeToDataTypeRMW(dataType);

    // executeOp is taken from func-sim6.
    // The function works on 4 bytes in every call
    uint32_t numIterations = numTensorBytes / 4;
    uint32_t leftover = numTensorBytes % 4;

    auto outBuf = memory->data();
    auto inBuf = mmeOut->data();
    for (int i = 0; i < numIterations; i++)
    {
        uint32_t res = executeOp(*reinterpret_cast<uint32_t*>(outBuf + 4 * i),
                                 *reinterpret_cast<uint32_t*>(inBuf + 4 * i),
                                 reductionOp,
                                 dataTypeRMW,
                                 reductionRm,
                                 clipFp,
                                 false /*clipInfIn*/);
        *reinterpret_cast<uint32_t*>(outBuf + 4 * i) = res;
    }

    if (leftover)
    {
        uint32_t res = executeOp(*reinterpret_cast<uint32_t*>(outBuf + 4 * numIterations),
                                 *reinterpret_cast<uint32_t*>(inBuf + 4 * numIterations),
                                 reductionOp,
                                 dataTypeRMW,
                                 reductionRm,
                                 clipFp,
                                 false /*clipInfIn*/);
        uint8_t * res_p = (uint8_t *) &res;
        for (int i=0; i < leftover; i++)
        {
            outBuf[4 * numIterations + i] = res_p[i];
        }
    }
}

void CPUCalculatorImpl::doGemmInternal(MmeSimTensor& output,
                                   const MmeSimTensor& inputA,
                                   const MmeSimTensor& inputB,
                                   bool transposeA,
                                   bool transposeB,
                                   const ConvolutionParams& convParams,
                                   EMmeOpType op,
                                   RoundingMode rm,
                                   bool clipFp,
                                   bool clipFpInfIn,
                                   int* zeroPoints)
{
    setMulFunc(inputA.getElementType(),
               inputB.getElementType(),
               rm,
               inputA.getFpBias(),
               inputB.getFpBias(),
               clipFp,
               clipFpInfIn,
               inputA.getInfNanMode(),
               inputB.getInfNanMode());
    CommonRefMatrix outputMat(output.getElementSize());
    CommonRefMatrix inputAMat(inputA.getElementSize());
    CommonRefMatrix inputBMat(inputB.getElementSize());
    RefIm2Col im2Col(m_mme_max_tensor_dims, rm);

    getMatricesDim(inputA.getSizes(),
                   inputB.getSizes(),
                   output.getSizes(),
                   op,
                   convParams,
                   inputAMat,
                   inputBMat,
                   outputMat);

    DataBuffer inputBMatDataBuffer(createBufferWithoutStrides(inputB), inputBMat.getMemorySize(), false);
    inputBMatDataBuffer.setShoudFree(true);
    inputBMat.data = inputBMatDataBuffer;
    outputMat.data = DataBuffer(outputMat.getHeight() * outputMat.getWidth() * outputMat.sizeOfDataType);

    // Fix the padding value
    ConvolutionParams convParamsCopy = convParams;
    if (op != EMmeOpType::e_mme_fwd)
    {
        convParamsCopy.paddingValue.int32 = 0;
    }
    else if (!inputA.isInt())
    {
        convParamsCopy.paddingValue.int32 = 0;
    }
    else if (inputA.is4Bits())
    {
        unsigned paddingValue4Bits = ((unsigned) convParamsCopy.paddingValue.int32) & 0xf;
        convParamsCopy.paddingValue.int32 = (int) ((paddingValue4Bits << 4) | paddingValue4Bits);
    }

    //  im2col for convolutions - FWD& DEDW.
    if (op == EMmeOpType::e_mme_fwd || op == EMmeOpType::e_mme_dedw)
    {
        inputAMat.data = DataBuffer(inputAMat.getMemorySize(), true);
        im2Col.doIm2Col(inputA,
                        convParamsCopy,
                        inputAMat,
                        op == EMmeOpType::e_mme_fwd ? inputB.getSizes() : output.getSizes(),
                        op == EMmeOpType::e_mme_fwd ? output.getSizes() : inputB.getSizes());
    }
    else
    {
        DataBuffer inputAMatDataBuffer(createBufferWithoutStrides(inputA), inputAMat.getMemorySize(), false);
        inputAMatDataBuffer.setShoudFree(true);
        inputAMat.data = inputAMatDataBuffer;
    }
    // transpose if needed
    if (transposeA)
    {
        inputAMat.doTranspose();
    }
    if (transposeB)
    {
        inputBMat.doTranspose();
    }

    MME_ASSERT(m_functionMap.count({inputA.getElementType(), output.getElementType()}),
               "type is not registered in function map !");
    // call CPUCalculatorImpl::doMatrixMultiplication with the correct data types
    auto doGemmFunc = m_functionMap.at({inputA.getElementType(), output.getElementType()});
    doGemmFunc(this, outputMat, inputAMat, inputBMat, rm, zeroPoints);

    if (op == EMmeOpType::e_mme_dedx || op == MmeCommon::e_mme_transposed_dedx)
    {
        im2Col.doCol2Im(output, convParams, outputMat, inputB.getSizes(), inputA.getSizes());
    }
    else
    {
        DataBuffer outputDataBuffer(createBufferWithStrides(outputMat.data.get(), output.getElementSize(), output.getMemorySize(),
            output.getSizes(), output.getStrides()), output.getMemorySize(), false);
        outputDataBuffer.setShoudFree(true);
        output.setData(outputDataBuffer);
    }
}

template<typename InputT, typename OutputT>
void CPUCalculatorImpl::applyZeroPointsToResult(CommonRefMatrix& output, const CommonRefMatrix& inputA, int* zeroPoints)
{
    if (zeroPoints != nullptr)  // FWD, int
    {
        std::vector<int64_t> lineSums(inputA.getHeight());
        for (uint64_t i = 0; i < inputA.getHeight(); ++i)
        {
            InputT* xData = &((InputT*) inputA.data)[i * inputA.getWidth()];
            int64_t sum = 0;
            for (uint64_t j = 0; j < inputA.getWidth(); ++j)
            {
                sum = xData[j] + sum;
            }
            lineSums[i] = sum;
        }

        for (uint64_t i = 0; i < output.getHeight(); i++)
        {
            iacc32_32_t* yData = &((iacc32_32_t*) output.data)[i * output.getWidth()];
            int64_t currentLineSum = lineSums[i];
            for (uint64_t j = 0; j < output.getWidth(); ++j)
            {
                yData[j] -= ((int64_t) zeroPoints[j]) * currentLineSum;
            }
        }
    }
}

/*Align to NTBF bug H3-2134*/
void CPUCalculatorImpl::apply_H3_2134(CommonRefMatrix& output)
{
    for (uint64_t i = 0; i < output.getHeight(); i++)  // Height
    {
        for (uint64_t j = 0; j < output.getWidth(); j++)  // Width
        {
            uint32_t& val = *(uint32_t*)output.getElementAt(i, j);
            val = add_fp32(val, 0, 0);
        }
    }
}

// internal function - uses CommonRefMatrix struct
template<typename InputT>
void CPUCalculatorImpl::doMatrixMultiplicationFloat(CommonRefMatrix& output,
                                                const CommonRefMatrix& inputA,
                                                const CommonRefMatrix& inputB,
                                                RoundingMode rm,
                                                int* zeroPoints)
{
    // validate node input shapes
    MME_ASSERT(inputA.getWidth() == inputB.getHeight(), "Common dim doesnt match between input A\\B");
    MME_ASSERT(output.getWidth() == inputB.getWidth(), "output width doesnt match to inputB width");

    // call job in threads - each thread will calculate a chunk out of the total
    // gemm.
    std::list<std::thread> threadList;
    uint64_t threadStart = 0;
    uint64_t chunkSize = div_round_up(output.getHeight(), m_numOfThreads);
    for (unsigned i = 0; i < m_numOfThreads; i++)
    {
        threadList.emplace_back(&CPUCalculatorImpl::doGemmWorkerFloat<InputT>,
                                this,
                                std::ref(output),
                                std::ref(inputA),
                                std::ref(inputB),
                                threadStart,
                                chunkSize,
                                rm);
        threadStart += chunkSize;
    }
    for (auto& t : threadList)
    {
        t.join();
    }

    switch (m_chipType)
    {
        case e_mme_Gaudi:
        case e_mme_Gaudi2:
            apply_H3_2134(output);
            break;
        default:
            break;
    }
}

template<typename InputT, typename OutputT>
void CPUCalculatorImpl::doMatrixMultiplicationInt(CommonRefMatrix& output,
                                           const CommonRefMatrix& inputA,
                                           const CommonRefMatrix& inputB,
                                           RoundingMode rm,
                                           int* zeroPoints)
{
    // validate node input shapes
    MME_ASSERT(inputA.getWidth() == inputB.getHeight(), "common dim doesnt match between inputs A\\B");
    MME_ASSERT(output.getWidth() == inputB.getWidth(), "output width doesnt match to inputB width");

    // call job in threads - each thread will calculate a chunk out of the total
    // gemm.
    std::list<std::thread> threadList;
    uint64_t threadStart = 0;
    uint64_t chunkSize = div_round_up(output.getHeight(), m_numOfThreads);
    for (unsigned i = 0; i < m_numOfThreads; i++)
    {
        threadList.emplace_back(&CPUCalculatorImpl::doGemmWorkerInt<InputT, OutputT>,
                                this,
                                std::ref(output),
                                std::ref(inputA),
                                std::ref(inputB),
                                threadStart,
                                chunkSize,
                                rm);
        threadStart += chunkSize;
    }
    for (auto& t : threadList)
    {
        t.join();
    }

    applyZeroPointsToResult<InputT, OutputT>(output, inputA, zeroPoints);
}

template<typename InputT, typename OutputT>
void CPUCalculatorImpl::doGemmWorkerInt(CommonRefMatrix& output,
                                    const CommonRefMatrix& inputA,
                                    const CommonRefMatrix& inputB,
                                    uint64_t startIdx,
                                    uint64_t chunkSize,
                                    RoundingMode rm)
{
    uint64_t endIdx = std::min(output.getHeight(), startIdx + chunkSize);
    for (uint64_t i = startIdx; i < endIdx; i++)  // Height
    {
        for (uint64_t j = 0; j < output.getWidth(); j++)  // Width
        {
            OutputT val = OutputT(0.0f);
            for (uint64_t k = 0; k < inputA.getWidth(); k++)  // Common Dim
            {
                auto inputAVal = *(InputT*) inputA.getElementAt(i, k);
                auto inputBVal = *(InputT*) inputB.getElementAt(k, j);
                // do multiply accumulate operation
                val = fma(inputAVal, inputBVal, val, rm);
            }
            uint64_t outputIndex = i * output.getWidth() + j;
            OutputT* outputPtr = (OutputT*) (output.data.get()) + outputIndex;
            memcpy((void*) outputPtr, &val, sizeof(OutputT));
        }
    }
}

template<typename InputT>
void CPUCalculatorImpl::doGemmWorkerFloat(CommonRefMatrix& output,
                                      const CommonRefMatrix& inputA,
                                      const CommonRefMatrix& inputB,
                                      uint64_t startIdx,
                                      uint64_t chunkSize,
                                      RoundingMode rm)
{
    uint64_t endIdx = std::min(output.getHeight(), startIdx + chunkSize);
    uint64_t cd = inputA.getWidth();
    InputT* a = new InputT[cd];
    InputT* b = new InputT[cd];
    for (uint64_t i = startIdx; i < endIdx; i++)  // Height
    {
        for (uint64_t j = 0; j < output.getWidth(); j++)  // Width
        {
            for (uint64_t k = 0; k < cd; k++)
            {
                a[k] = *(InputT*)inputA.getElementAt(i, k);
                b[k] = *(InputT*)inputB.getElementAt(k, j);
            }

            float val = chipFma->fma_vec(a, b, cd);

            uint64_t outputIndex = i * output.getWidth() + j;
            float* outputPtr = (float*) (output.data.get()) + outputIndex;
            memcpy((void*) outputPtr, &val, sizeof(float));
        }
    }
    delete[] a;
    delete[] b;
}

void CPUCalculatorImpl::getMatricesDim(const SizeArray& sizeA,
                                   const SizeArray& sizeB,
                                   const SizeArray& sizeOut,
                                   const EMmeOpType op,
                                   const ConvolutionParams& params,
                                   CommonRefMatrix& inputA,
                                   CommonRefMatrix& inputB,
                                   CommonRefMatrix& output)
{
    switch (op)
    {
        case EMmeOpType::e_mme_fwd:
        {
            uint64_t RSCSize = multiplyElements(std::next(sizeB.begin()), sizeB.end());
            uint64_t BHWtensorYSize = multiplyElements(std::next(sizeOut.begin()), sizeOut.end());

            // matrix A size is BHW x RSC
            inputA.setShape(BHWtensorYSize, RSCSize);
            // matrix B size is RSC x K
            inputB.setShape(RSCSize, sizeB[0]);
            // matrix Out size is BHW x K
            output.setShape(BHWtensorYSize, sizeB[0]);
            break;
        }
        case EMmeOpType::e_mme_transposed_dedx:
        case EMmeOpType::e_mme_dedx:
        {
            uint64_t RSCSize = multiplyElements(std::next(sizeB.begin()), sizeB.end());
            uint64_t BHWtensorYSize = multiplyElements(std::next(sizeA.begin()), sizeA.end());

            // Height: BHW, Width: K
            inputA.setShape(BHWtensorYSize, sizeA[0]);

            // Height: RSC, Width: K
            inputB.setShape(RSCSize, sizeA[0]);  // To be transposed

            // Height: BHW, Width: RSC
            output.setShape(BHWtensorYSize, RSCSize);  // To be col2im'd later
            break;
        }
        case EMmeOpType::e_mme_dedw:
        {
            uint64_t RSCSize = multiplyElements(std::next(sizeOut.begin()), sizeOut.end());
            uint64_t BHWtensorYSize = multiplyElements(std::next(sizeB.begin()), sizeB.end());

            // Height: RSC, Width: BHW
            inputA.setShape(BHWtensorYSize, RSCSize);  // To be transposed

            // Height: BHW, Width: k
            inputB.setShape(BHWtensorYSize, sizeOut[0]);

            // Height: RSC, Width: k
            output.setShape(RSCSize, sizeOut[0]);
            break;
        }
        case EMmeOpType::e_mme_atbt:
        {
            inputA.setShape(sizeA[1], sizeA[0]);
            inputB.setShape(sizeB[1], sizeB[0]);
            output.setShape(sizeOut[1], sizeOut[0]);
            break;
        }
        default:
            MME_ASSERT(0, "invalid op type");
    }
}

ConvolutionParams CPUCalculatorImpl::createGemmParams()
{
    ConvolutionParams params;
    params.paddingValue.int32 = 0;
    params.convStride.fill(1);
    params.dilation.fill(1);
    params.padding.fill(0);
    return params;
}

template<typename T>
void CPUCalculatorImpl::doRelu(pMMESimTensor acc)
{
    MME_ASSERT(acc->isContiguous(), "input and output should be flattened");
    for (uint64_t currentElement = 0; currentElement < acc->getSizeInElements(); currentElement++)
    {
        T& element = reinterpret_ptr_with_index<T>(acc->data(), currentElement);
        element = std::max((T) 0, element);
    }
}

void CPUCalculatorImpl::convertToOutputDataType(pMMESimTensor output,
                                            const pMMESimTensor input,
                                            RoundingMode rm,
                                            bool clipFp,
                                            bool clipInfIn,
                                            bool flushDenorms,
                                            bool stochasticFTZfp8)
{
    auto inputData = (float*) input->data();
    auto convertedData = output->data();
    bool reseedLfsr = rm == StochasticRoundingAndNearest || rm == StochasticRounding || stochasticFTZfp8;
    // Step 1: Convert to the target data type
    for (uint64_t i = 0; i < input->getSizeInElements(); i++)
    {
        switch (output->getElementType())
        {
            case EMmeDataType::e_type_fp16:
            {
                auto tempPtr = (fp16_t*) convertedData;
                int lfsrIdx = i % 64;
                fp16_t convertedVal(inputData[i],
                                    rm,
                                    output->getFpBias(),
                                    m_lfsr[lfsrIdx],
                                    flushDenorms,
                                    clipFp,
                                    clipInfIn,
                                    output->getInfNanMode());
                tempPtr[i] = convertedVal;
                if (reseedLfsr) m_lfsr[lfsrIdx] = mme_lfsr32(m_lfsr[lfsrIdx], m_polynomial);
                break;
            }
            case EMmeDataType::e_type_ufp16:
            {
                auto tempPtr = (ufp16_t*) convertedData;
                int lfsrIdx = i % 64;
                ufp16_t convertedVal(inputData[i],
                                     rm,
                                     output->getFpBias(),
                                     m_lfsr[lfsrIdx],
                                     flushDenorms,
                                     clipFp,
                                     clipInfIn,
                                     output->getInfNanMode());
                tempPtr[i] = convertedVal;
                if (reseedLfsr) m_lfsr[lfsrIdx] = mme_lfsr32(m_lfsr[lfsrIdx], m_polynomial);
                break;
            }
            case EMmeDataType::e_type_bf16:
            {
                auto tempPtr = (bf16_t*) convertedData;
                int lfsrIdx = i % 64;
                bf16_t convertedVal(inputData[i], rm, m_lfsr[lfsrIdx], clipFp, flushDenorms, clipInfIn);
                tempPtr[i] = convertedVal;
                if (reseedLfsr) m_lfsr[lfsrIdx] = mme_lfsr32(m_lfsr[lfsrIdx], m_polynomial);
                break;
            }
            case EMmeDataType::e_type_fp32:
            {
                auto tempPtr = (fp32_t*) convertedData;
                fp32_t convertedVal(inputData[i], rm, clipFp, flushDenorms, clipInfIn);
                tempPtr[i] = convertedVal;
                break;
            }
            case EMmeDataType::e_type_fp8_143:
            {
                auto tempPtr = (fp8_143_t*) convertedData;
                int lfsrIdx = i % 128;
                fp8_143_t convertedVal(inputData[i],
                                       rm,
                                       output->getFpBias(),
                                       m_lfsr[lfsrIdx],
                                       flushDenorms,
                                       clipFp,
                                       clipInfIn,
                                       stochasticFTZfp8,
                                       output->getInfNanMode());
                tempPtr[i] = convertedVal;
                if (reseedLfsr) m_lfsr[lfsrIdx] = mme_lfsr32(m_lfsr[lfsrIdx], m_polynomial);
                break;
            }
            case EMmeDataType::e_type_fp8_152:
            {
                auto tempPtr = (fp8_152_t*) convertedData;
                int lfsrIdx = i % 128;
                fp8_152_t convertedVal(inputData[i],
                                       rm,
                                       output->getFpBias(),
                                       m_lfsr[lfsrIdx],
                                       flushDenorms,
                                       clipFp,
                                       clipInfIn,
                                       stochasticFTZfp8,
                                       output->getInfNanMode());
                tempPtr[i] = convertedVal;
                if (reseedLfsr) m_lfsr[lfsrIdx] = mme_lfsr32(m_lfsr[lfsrIdx], m_polynomial);
                break;
            }
            default:
                MME_ASSERT(0, "Unsupported type");
        }  // end switch
    }
}

//=================================================================
#define USE_CPP11_MATH

#define REF_SIGN_OFFSET_FP32     31
#define REF_EXPONENT_OFFSET_FP32 23
#define REF_EXPONENT_BIAS_FP32   127

#define REF_SIGN_OFFSET_FP16      15
#define REF_SIGN_MASK_FP16        0x8000
#define REF_EXPONENT_OFFSET_FP16  10
#define REF_EXPONENT_MASK_FP16    0x7C00
#define REF_EXPONENT_BIAS_FP16    15
#define REF_SIGNIFICAND_MASK_FP16 0x03FF

static const int64_t REF_INT4_MAX = (1 << 3) - 1;
static const int64_t REF_INT4_MIN = -1 * (1 << 3);
static const int64_t REF_UINT4_MAX = (1 << 4) - 1;

static const uint32_t c_ref_default_nan_fp32 = 0x7FFFFFFF;

static const int MIN_GLP_EXP = -32;
static const int MAX_GLP_EXP = 31;
static const int32_t GLP_POWER_2_SCALE = std::numeric_limits<int32_t>::max();

//============

// todo AlonG: unify this function
int32_t CPUCalculatorImpl::saturatingRoundingDoublingHighMul(int32_t a, int32_t b, bool* r_bit, bool* s_bit)
{
    int64_t a_64(a);
    int64_t b_64(b);
    int64_t ab_64 = a_64 * b_64;
    *r_bit = (ab_64 >> 30) & 0x1;
    int64_t mask = 0x3FFFFFFFll;  //(1ll << 29) - 1;
    *s_bit = ((ab_64 & mask) != 0);
    return (int32_t)(ab_64 >> 31);
}

int32_t CPUCalculatorImpl::saturatingRoundingMultiplyByPot(int32_t x, int exponent)
{
    const int32_t min = std::numeric_limits<int32_t>::min();
    const int32_t max = std::numeric_limits<int32_t>::max();

    const int32_t threshold = ((1 << (31 - exponent)) - 1);
    int32_t result;
    if (x > threshold)
    {
        result = max;
    }
    else if (x < -threshold)
    {
        result = min;
    }
    else
    {
        // arithmetic shift left. Keeps sign of x and inserts 0 from LSB
        result = x * (1 << exponent);
    }
    return result;
};

int32_t CPUCalculatorImpl::roundingDivideByPot(int32_t x, int exponent, bool r_bit, bool s_bit)
{
    MME_ASSERT(exponent >= 0 , "exponent out of bounds");
    MME_ASSERT(exponent <= 31, "exponent out of bounds");
    const int32_t mask = (1ll << exponent) - 1;
    const int32_t remainder = x & mask;
    int32_t threshold = mask >> 1;
    threshold += (x < 0 && (r_bit | s_bit) == 0) ? 1 : 0;
    int32_t res = x >> exponent;
    res += (remainder > threshold) ? 1 : 0;
    return res;
}

float CPUCalculatorImpl::singleElemPreluFloat(float input, float negScale, float posScale)
{
    return input > 0 ? posScale * input : negScale * input;
}

int32_t CPUCalculatorImpl::singleElemPrelu(int32_t input, int negExp, unsigned negScale, int posExp, unsigned posScale)
{
    int32_t output;
    int effectiveExp = input > 0 ? posExp : negExp;
    unsigned effectiveScale = input > 0 ? posScale : negScale;
    bool r_bit;
    bool s_bit;

    if (effectiveExp < 0)
    {
        // First, scale: Mul by int32, rounding exponent right by 31
        int32_t scaledInput = saturatingRoundingDoublingHighMul(input, (int32_t) effectiveScale, &r_bit, &s_bit);

        // Then, rounding shift right by exponent which is a POT
        output = roundingDivideByPot(scaledInput, -effectiveExp, r_bit, s_bit);
    }
    else
    {
        // First, rounding shift left by exponent
        int32_t shiftedInput = saturatingRoundingMultiplyByPot(input, effectiveExp);

        // Then, scale: Mul by int32, rounding shifting right by 31
        output = saturatingRoundingDoublingHighMul(shiftedInput, (int32_t) effectiveScale, &r_bit, &s_bit);

        if (output >= 0) output = output + r_bit;
        else
            output = output + (r_bit & s_bit);
    }

    return output;
}

void CPUCalculatorImpl::internalPrelu(pMMESimTensor tensor,
                                  const int* negExp,
                                  const void* negScale,
                                  const int* posExp,
                                  const void* posScale)
{
    MME_ASSERT(tensor->isContiguous(), "tensor should be contiguous");
    const int numChannels = tensor->getSize(0);

    if (EMmeDataType::e_type_int32 == tensor->getElementType())
    {
        auto tensorData = (int32_t*) tensor->data();
        const auto negScaleInt = reinterpret_cast<const unsigned*>(negScale);
        const auto posScaleInt = reinterpret_cast<const unsigned*>(posScale);

        for (int i = 0; i < tensor->getElementsCount(); ++i)
        {
            int currentChannel = i % numChannels;
            *tensorData = singleElemPrelu(*tensorData,
                                          negExp[currentChannel],
                                          negScaleInt[currentChannel],
                                          posExp[currentChannel],
                                          posScaleInt[currentChannel]);
            tensorData++;
        }
    }
    else
    {
        auto tensorData = (float*) tensor->data();
        const auto negScaleFloat = reinterpret_cast<const float*>(negScale);
        const auto posScaleFloat = reinterpret_cast<const float*>(posScale);

        for (int i = 0; i < tensor->getElementsCount(); ++i)
        {
            int currentChannel = i % numChannels;
            *tensorData =
                singleElemPreluFloat(*tensorData, negScaleFloat[currentChannel], posScaleFloat[currentChannel]);
            tensorData++;
        }
    }
}

template<typename T>
static void reference_relu(pMMESimTensor tensor)
{
    MME_ASSERT(tensor->isContiguous(), "tensor should be contiguous");

    T* tensorData = (T*) tensor->data();

    for (int i = 0; i < tensor->getElementsCount(); ++i)
    {
        tensorData[i] = tensorData[i] > 0 ? tensorData[i] : 0;
    }
}

void CPUCalculatorImpl::reluPrelu(pMMESimTensor acc,
                              bool reluEn,
                              int* pwlNegExp,
                              void* pwlNegScale,
                              int* pwlPosExp,
                              void* pwlPosScale)
{
    MME_ASSERT(acc->isContiguous(), "acc tensor should be contiguous");
    if (reluEn)
    {
        MME_ASSERT(nullptr == pwlNegExp &&
                  nullptr == pwlNegScale &&
                  nullptr == pwlPosExp &&
                  nullptr == pwlPosScale,
                  "pwl exp\\scale should ne nullptr");

        if (EMmeDataType::e_type_fp32 == acc->getElementType())
        {
            doRelu<float>(acc);
        }
        else if (EMmeDataType::e_type_int32 == acc->getElementType())
        {
            doRelu<int32_t>(acc);
        }
        else
        {
            MME_ASSERT(0, "invalid element type");
        }
    }
    else if (nullptr != pwlNegScale)
    {
        MME_ASSERT((EMmeDataType::e_type_int32 == acc->getElementType() &&
                  (nullptr != pwlNegScale && nullptr != pwlPosExp && nullptr != pwlPosScale)) ||
                  (EMmeDataType::e_type_fp32 == acc->getElementType() &&
                  (nullptr != pwlPosScale && nullptr == pwlNegExp && nullptr == pwlPosExp)),
                  "invalid pwl scale\\exp");
        internalPrelu(acc, pwlNegExp, pwlNegScale, pwlPosExp, pwlPosScale);
    }
}

void CPUCalculatorImpl::referenceLUT(pMMESimTensor tensor, uint8_t* lut)
{
    uint8_t localLut[256];

    if (tensor->is4Bits())
    {
        for (unsigned i = 0; i < 16; i++)
        {
            MME_ASSERT(lut[i] < 16, "4-bit lut value overflow");
        }

        for (unsigned i = 0; i < sizeof(localLut); i++)
        {
            localLut[i] = lut[i & 0xf] + (lut[i >> 4] << 4);
        }
    }
    else
    {
        memcpy(localLut, lut, sizeof(localLut));
    }

    MME_ASSERT(tensor->isContiguous(), "tensor should be contiguous");
    uint8_t* tensorData = (uint8_t*) tensor->data();

    for (int i = 0; i < tensor->getElementsCount(); ++i)
    {
        tensorData[i] = localLut[tensorData[i]];
    }
}

int32_t CPUCalculatorImpl::internalGemmlowpIn(int32_t in, int glpExp, uint32_t glpScale, int32_t glpZp)
{
    // Perform GEMMLOWP according to spec
    int32_t CinWithZeroPointFixup = in - glpZp;
    int32_t scaledAndShiftedCin;
    bool r_bit;
    bool s_bit;

    if (glpExp < 0)
    {
        // First, scale: Mul by int32, rounding exponent right by 31
        int32_t scaledCin =
            saturatingRoundingDoublingHighMul(CinWithZeroPointFixup, (int32_t) glpScale, &r_bit, &s_bit);

        // Then, rounding shift right by exponent which is a POT
        scaledAndShiftedCin = roundingDivideByPot(scaledCin, -glpExp, r_bit, s_bit);
    }
    else
    {
        // First, rounding shift left by exponent
        int32_t shiftedCin = saturatingRoundingMultiplyByPot(CinWithZeroPointFixup, glpExp);

        // Then, scale: Mul by int32, rounding shifting right by 31
        scaledAndShiftedCin = saturatingRoundingDoublingHighMul(shiftedCin, (int32_t) glpScale, &r_bit, &s_bit);

        if (scaledAndShiftedCin >= 0) scaledAndShiftedCin = scaledAndShiftedCin + r_bit;
        else
            scaledAndShiftedCin = scaledAndShiftedCin + (r_bit & s_bit);
    }

    return scaledAndShiftedCin;
}

void CPUCalculatorImpl::externalGemmlowpIn(pMMESimTensor res,
                                       const pMMESimTensor in,
                                       const int index,
                                       const int* glpExp,
                                       const int32_t* glpScale,
                                       const int32_t glpZp)
{
    int64_t glpOutput0 = 0;
    int64_t glpOutput = 0;
    char* inputPtr;
    int32_t* outputPtr;
    EMmeDataType type = res->getElementType();
    switch (type)
    {
        case EMmeDataType::e_type_int4:
            inputPtr = (char*) (&((int8_t*) res->data())[index]);
            glpOutput0 = internalGemmlowpIn(((int4x2_t*) inputPtr)->i0, glpExp[0], glpScale[0], glpZp);
            glpOutput = internalGemmlowpIn(((int4x2_t*) inputPtr)->i1, glpExp[1], glpScale[1], glpZp);
            break;
        case EMmeDataType::e_type_uint4:
            inputPtr = (char*) (&((uint8_t*) res->data())[index]);
            glpOutput0 = internalGemmlowpIn(((uint4x2_t*) inputPtr)->i0, glpExp[0], glpScale[0], glpZp);
            glpOutput = internalGemmlowpIn(((uint4x2_t*) inputPtr)->i1, glpExp[1], glpScale[1], glpZp);
            break;
        case EMmeDataType::e_type_int8:
            inputPtr = (char*) (&((int8_t*) res->data())[index]);
            glpOutput = internalGemmlowpIn((int32_t) * (int8_t*) inputPtr, *glpExp, *glpScale, glpZp);
            break;
        case EMmeDataType::e_type_uint8:
            inputPtr = (char*) (&((uint8_t*) res->data())[index]);
            glpOutput = internalGemmlowpIn((int32_t) * (uint8_t*) inputPtr, *glpExp, *glpScale, glpZp);
            break;
        case EMmeDataType::e_type_int16:
            MME_ASSERT(0 == glpZp, "zeroPoint should be 0");
            MME_ASSERT(GLP_POWER_2_SCALE == *glpScale, "glp scale should be power of 2");
            inputPtr = (char*) (&((int16_t*) res->data())[index]);
            glpOutput = internalGemmlowpIn((int32_t) * (int16_t*) inputPtr, *glpExp, *glpScale, glpZp);
            break;
        case EMmeDataType::e_type_uint16:
            MME_ASSERT(0 == glpZp, "zeroPoint should be 0");
            MME_ASSERT(GLP_POWER_2_SCALE == *glpScale, "glp scale should be power of 2");
            inputPtr = (char*) (&((uint16_t*) res->data())[index]);
            glpOutput = internalGemmlowpIn((int32_t) * (uint16_t*) inputPtr, *glpExp, *glpScale, glpZp);
            break;
        case EMmeDataType::e_type_int32:
            MME_ASSERT(0 == glpZp, "zeroPoint should be 0");
            MME_ASSERT(GLP_POWER_2_SCALE == *glpScale, "glp scale should be power of 2");
            inputPtr = (char*) (&((int32_t*) res->data())[index]);
            glpOutput = internalGemmlowpIn(*(int32_t*) inputPtr, *glpExp, *glpScale, glpZp);
            break;
        default:
            MME_ASSERT(0, "invalid data type");
    }

    int64_t sum0 = 0;
    if (EMmeDataType::e_type_int4 == type || EMmeDataType::e_type_uint4 == type)
    {
        outputPtr = &(((int32_t*) in->data())[2 * index]);

        sum0 = glpOutput0 + *outputPtr;
        if (sum0 > std::numeric_limits<int32_t>::max()) sum0 = std::numeric_limits<int32_t>::max();
        else if (sum0 < std::numeric_limits<int32_t>::min())
            sum0 = std::numeric_limits<int32_t>::min();

        *outputPtr = (int32_t) sum0;
        outputPtr++;
    }
    else
    {
        outputPtr = &(((int32_t*) in->data())[index]);
    }

    int64_t sum = glpOutput + *outputPtr;
    if (sum > std::numeric_limits<int32_t>::max()) sum = std::numeric_limits<int32_t>::max();
    else if (sum < std::numeric_limits<int32_t>::min())
        sum = std::numeric_limits<int32_t>::min();

    *outputPtr = (int32_t) sum;
}

int64_t CPUCalculatorImpl::internalGemmlowpOut(int32_t in, int glpExp, uint32_t glpScale, int32_t glpZp)
{
    // Rescale to scaleC
    int32_t scaledAndShiftedCout;
    bool r_bit = 0;
    bool s_bit = 0;

    MME_ASSERT(glpExp >= MIN_GLP_EXP && glpExp <= MAX_GLP_EXP, "out of bound glp exponent");

    if (glpExp < 0)
    {
        // First, scale: Mul by int32, rounding exponent right by 31
        int32_t scaledCout = saturatingRoundingDoublingHighMul(in, glpScale, &r_bit, &s_bit);
        // Then, rounding shift right by exponent which is a POT
        scaledAndShiftedCout = roundingDivideByPot(scaledCout, -glpExp, r_bit, s_bit);
    }
    else
    {
        // First, rounding shift left by exponent
        int32_t shiftedCout = saturatingRoundingMultiplyByPot(in, glpExp);
        // Then, scale: Mul by int32, rounding shifting right by 31

        scaledAndShiftedCout = saturatingRoundingDoublingHighMul(shiftedCout, glpScale, &r_bit, &s_bit);
        if (scaledAndShiftedCout >= 0) scaledAndShiftedCout = scaledAndShiftedCout + r_bit;
        else
            scaledAndShiftedCout = scaledAndShiftedCout + (r_bit & s_bit);
    }

    // Add zero_point_C
    scaledAndShiftedCout += ((int32_t) glpZp);

    return (int64_t) scaledAndShiftedCout;
}

void CPUCalculatorImpl::externalGemmlowpOut(pMMESimTensor in,
                                        pMMESimTensor out,
                                        int index,
                                        int glpExp,
                                        uint32_t glpScale,
                                        int32_t glpZp)
{
    int64_t glpOutput = 0;
    int32_t* inputPtr = &(((int32_t*) in->data())[index]);
    char* outputPtr;
    EMmeDataType type = out->getElementType();
    switch (type)
    {
        case EMmeDataType::e_type_int4:
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*(int32_t*) inputPtr, glpExp, glpScale, glpZp);
            if (glpOutput > REF_INT4_MAX)
            {
                glpOutput = REF_INT4_MAX;
            }
            else if (glpOutput < REF_INT4_MIN)
            {
                glpOutput = REF_INT4_MIN;
            }
            out->setFourBitsValueAt<int4x2_t>(index, glpOutput);
            break;
        case EMmeDataType::e_type_uint4:
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*inputPtr, glpExp, glpScale, glpZp);
            if (glpOutput > REF_UINT4_MAX)
            {
                glpOutput = REF_UINT4_MAX;
            }
            else if (glpOutput < 0)
            {
                glpOutput = 0;
            }
            out->setFourBitsValueAt<uint4x2_t>(index, glpOutput);
            break;
        case EMmeDataType::e_type_int8:
            outputPtr = (char*) &(((int8_t*) out->data())[index]);
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*inputPtr, glpExp, glpScale, glpZp);
            if (glpOutput > std::numeric_limits<int8_t>::max()) glpOutput = std::numeric_limits<int8_t>::max();
            else if (glpOutput < std::numeric_limits<int8_t>::min())
                glpOutput = std::numeric_limits<int8_t>::min();
            *(int8_t*) outputPtr = (int8_t) glpOutput;
            break;
        case EMmeDataType::e_type_uint8:
            outputPtr = (char*) &(((uint8_t*) out->data())[index]);
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*inputPtr, glpExp, glpScale, glpZp);
            if (glpOutput > std::numeric_limits<uint8_t>::max()) glpOutput = std::numeric_limits<uint8_t>::max();
            else if (glpOutput < std::numeric_limits<uint8_t>::min())
                glpOutput = std::numeric_limits<uint8_t>::min();
            *(uint8_t*) outputPtr = (uint8_t) glpOutput;
            break;
        case EMmeDataType::e_type_int16:
            MME_ASSERT(0 == glpZp, "zeroPoint should be 0");
            MME_ASSERT(GLP_POWER_2_SCALE == glpScale, "glp scale should be power of 2");
            outputPtr = (char*) &(((int16_t*) out->data())[index]);
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*inputPtr, glpExp, glpScale, glpZp);
            if (glpOutput > std::numeric_limits<int16_t>::max()) glpOutput = std::numeric_limits<int16_t>::max();
            else if (glpOutput < std::numeric_limits<int16_t>::min())
                glpOutput = std::numeric_limits<int16_t>::min();
            *(int16_t*) outputPtr = (int16_t) glpOutput;
            break;
        case EMmeDataType::e_type_uint16:
            MME_ASSERT(0 == glpZp, "zeroPoint should be 0");
            MME_ASSERT(GLP_POWER_2_SCALE == glpScale, "glp scale should be power of 2");
            outputPtr = (char*) &(((uint16_t*) out->data())[index]);
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*inputPtr, glpExp, glpScale, glpZp);
            if (glpOutput > std::numeric_limits<uint16_t>::max()) glpOutput = std::numeric_limits<uint16_t>::max();
            else if (glpOutput < std::numeric_limits<uint16_t>::min())
                glpOutput = std::numeric_limits<uint16_t>::min();
            *(uint16_t*) outputPtr = (uint16_t) glpOutput;
            break;
        case EMmeDataType::e_type_int32:
            MME_ASSERT(0 == glpZp, "zeroPoint should be 0");
            MME_ASSERT(GLP_POWER_2_SCALE == glpScale, "glp scale should be power of 2");
            outputPtr = (char*) &(((int32_t*) out->data())[index]);
            glpOutput = CPUCalculatorImpl::internalGemmlowpOut(*inputPtr, glpExp, glpScale, glpZp);
            *(int32_t*) outputPtr = (int32_t) glpOutput;
            break;
        default:
            MME_ASSERT(0, "invalid data type");
    }
}

template<bool isFloat, CPUCalculatorImpl::GlpDirection DIRECTION>
void CPUCalculatorImpl::externalGemmlowp(pMMESimTensor in,
                                     pMMESimTensor out,
                                     int* glpExp,
                                     int* glpScale,
                                     int glpZp,
                                     int roundingMode,
                                     int32_t lfsrVal)
{
    MME_ASSERT(in->isContiguous() && out->isContiguous(), "input and output tensor should be contiguous");

    const int channelsCount = in->getSize(0);
    int elementsCount = DIRECTION == GLP_IN ? out->getSizeInElements() : in->getSizeInElements();
    int loops = DIRECTION == GLP_IN && in->is4Bits() ? elementsCount / 2 : elementsCount;

    if (!isFloat)
    {
        MME_ASSERT((nullptr != glpExp && nullptr != glpScale) || (nullptr == glpExp && nullptr == glpScale), "invalid glp scale\\exp");
    }
    else
    {
        MME_ASSERT(nullptr == glpExp && nullptr == glpScale, "glpExp\\glpScale should be null");
    }

    for (int i = 0; i < loops; ++i)
    {
        if (GLP_IN == DIRECTION)
        {
            bool is4Bit = in->is4Bits();
            int currentChannel = (i % channelsCount) * (is4Bit ? 2 : 1);

            if (!isFloat)
            {
                if (glpExp)
                {
                    externalGemmlowpIn(in, out, i, &glpExp[currentChannel], &glpScale[currentChannel], glpZp);
                }
                else
                {
                    int zeroExps[2] = {0};
                    int32_t scales[2] = {GLP_POWER_2_SCALE, GLP_POWER_2_SCALE};
                    externalGemmlowpIn(in, out, i, zeroExps, scales, 0);
                }
            }
            else
            {
                MME_ASSERT(EMmeDataType::e_type_fp32 == out->getElementType(), "output type should be fp32");

                switch (in->getElementType())
                {
                    case EMmeDataType::e_type_fp32:
                        ((float*) out->data())[i] += ((float*) in->data())[i];
                        break;
                    case EMmeDataType::e_type_fp16:
                    {
                        fp16_t inVal = ((fp16_t*) in->data())[i];
                        ((float*) out->data())[i] += inVal.toFloat();
                        break;
                    }
                    case EMmeDataType::e_type_bf16:
                    {
                        bf16_t inVal = ((bf16_t*) in->data())[i];
                        ((float*) out->data())[i] += inVal.toFloat();
                        break;
                    }
                    default:
                        MME_ASSERT(0, "invalid data type");
                }
            }
        }
        else
        {
            int currentChannel = i % channelsCount;
            if (!isFloat)
            {
                if (glpExp)
                {
                    externalGemmlowpOut(in, out, i, glpExp[currentChannel], glpScale[currentChannel], glpZp);
                }
                else
                {
                    externalGemmlowpOut(in, out, i, 0, GLP_POWER_2_SCALE, 0);
                }
            }
            else
            {
                if (EMmeDataType::e_type_fp16 == out->getElementType())
                {
                    // uint16_t converted = ref_fp32_to_fp16(((float*) in->data())[i],
                    // roundingMode, lfsrVal);

                    fp16_t converted(((float*) in->data())[i],
                                     (RoundingMode) roundingMode,
                                     REF_EXPONENT_BIAS_FP16,
                                     lfsrVal);
                    ((fp16_t*) out->data())[i] = *(fp16_t*) &converted;
                }
                else if (EMmeDataType::e_type_bf16 == out->getElementType())
                {
                    bf16_t converted(((float*) in->data())[i], (RoundingMode) roundingMode, lfsrVal);
                    ((bf16_t*) out->data())[i] = *(bf16_t*) &converted;
                }
                else
                {
                    ((float*) out->data())[i] = ((float*) in->data())[i];
                }
            }
        }
    }
}

void initRounding(RoundingMode roundingMode,
                  int& prevRoundingMode,
                  unsigned& prevFlushZeroMode,
                  unsigned& prevDenormalsZeroMode,
                  int& currentRoundingMode)
{
    prevRoundingMode = fegetround();
    prevFlushZeroMode = _MM_GET_FLUSH_ZERO_MODE();
    prevDenormalsZeroMode = _MM_GET_DENORMALS_ZERO_MODE();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    switch (roundingMode)
    {
        case RoundingMode::RoundToZero:
            currentRoundingMode = FE_TOWARDZERO;
            break;
        case RoundingMode::RoundToNearest:
            currentRoundingMode = FE_TONEAREST;
            break;
        case RoundingMode::RoundUp:
            currentRoundingMode = FE_UPWARD;
            break;
        case RoundingMode::RoundDown:
            currentRoundingMode = FE_DOWNWARD;
            break;
        default:
            MME_ASSERT(0, "invalid rounding mode");

            fesetround(currentRoundingMode);
    }
}

// todo AlonG: unify this function
void CPUCalculatorImpl::restoreRounding(int prevRoundingMode, unsigned prevFlushZeroMode, unsigned prevDenormalsZeroMode)
{
    fesetround(prevRoundingMode);
    _MM_SET_FLUSH_ZERO_MODE(prevFlushZeroMode);
    _MM_SET_DENORMALS_ZERO_MODE(prevDenormalsZeroMode);
}

template<typename T>
void CPUCalculatorImpl::addBias(pMMESimTensor acc, pMMESimTensor bias)
{
    MME_ASSERT(acc->getElementType() == bias->getElementType(), "element type of output tesnro and bias doesnt match");
    MME_ASSERT(acc->isContiguous(), "acc should be contiguous");
    MME_ASSERT(1 == bias->getDim(), "bias should be a scalar (1 dim)");

    const int channelsCount = bias->getElementsCount();
    MME_ASSERT(channelsCount == acc->getSize(0), "channel count of bias and output tensor should match");

    T* accData = (T*) acc->data();
    T* biasData = (T*) bias->data();

    for (uint64_t i = 0; i < acc->getElementsCount(); ++i)
    {
        accData[i] += biasData[i % channelsCount];
    }
}

void CPUCalculatorImpl::activationGemmlowp(pMMESimTensor out,  // the output tensor.
                                       pMMESimTensor acc,  // the input tensor (the output of the conv stage).
                                       pMMESimTensor res,  // residual tensor (cin). nullptr when there's no residual.
                                       pMMESimTensor bias,  // 1D bias tensor with K elements. (either int32_t or
                                                            // float). nullptr when there's no bias.
                                       bool reluEn,  // enable relu.
                                       RoundingMode roundingMode,  // rounding mode for type conversion
                                       const ActivationParams* actParams)
{
    bool resFirst = actParams->resFirst;
    int* pwlNegExp = actParams->pwlNegExp;
    void* pwlNegScale = actParams->pwlNegScale;
    int* pwlPosExp = actParams->pwlPosExp;
    void* pwlPosScale = actParams->pwlPosScale;
    int* glpResExp = actParams->glpResExp;
    int* glpResScale = actParams->glpResScale;
    int glpResZp = actParams->glpResZp;
    int* glpOutExp = actParams->glpOutExp;
    int* glpOutScale = actParams->glpOutScale;
    int glpOutZp = actParams->glpOutZp;
    uint8_t* lut = actParams->lut;
    int32_t lfsrVal = actParams->lfsrVal;

    int prevRoundingMode;
    int currentRoundingMode;
    unsigned flushZeroMode;
    unsigned denormalsZeroMode;

    initRounding(roundingMode, prevRoundingMode, flushZeroMode, denormalsZeroMode, currentRoundingMode);
    if (nullptr != bias)
    {
        if (EMmeDataType::e_type_int32 == bias->getElementType())
        {
            addBias<iacc32_32_saturate_t>(acc, bias);
        }
        else if (EMmeDataType::e_type_fp32 == bias->getElementType())
        {
            addBias<float>(acc, bias);
        }
        else
        {
            MME_ASSERT(0, "invalid data type");
        }
    }

    if (resFirst)
    {
        if (nullptr != res)
        {
            if (!res->isInt())
            {
                externalGemmlowp<true, GLP_IN>(res,
                                               acc,
                                               glpResExp,
                                               glpResScale,
                                               glpResZp,
                                               currentRoundingMode,
                                               lfsrVal);
            }
            else
            {
                externalGemmlowp<false, GLP_IN>(res,
                                                acc,
                                                glpResExp,
                                                glpResScale,
                                                glpResZp,
                                                currentRoundingMode,
                                                lfsrVal);
            }
        }

        reluPrelu(acc, reluEn, pwlNegExp, pwlNegScale, pwlPosExp, pwlPosScale);
    }
    else
    {
        reluPrelu(acc, reluEn, pwlNegExp, pwlNegScale, pwlPosExp, pwlPosScale);
        if (nullptr != res)
        {
            if (!res->isInt())
            {
                externalGemmlowp<true, GLP_IN>(res,
                                               acc,
                                               glpResExp,
                                               glpResScale,
                                               glpResZp,
                                               currentRoundingMode,
                                               lfsrVal);
            }
            else
            {
                externalGemmlowp<false, GLP_IN>(res,
                                                acc,
                                                glpResExp,
                                                glpResScale,
                                                glpResZp,
                                                currentRoundingMode,
                                                lfsrVal);
            }
        }
    }

    if (!acc->isInt())
    {
        externalGemmlowp<true, GLP_OUT>(acc, out, glpOutExp, glpOutScale, glpOutZp, currentRoundingMode, lfsrVal);
    }
    else
    {
        externalGemmlowp<false, GLP_OUT>(acc, out, glpOutExp, glpOutScale, glpOutZp, currentRoundingMode, lfsrVal);
    }

    if (nullptr != lut) referenceLUT(out, lut);

    restoreRounding(prevRoundingMode, flushZeroMode, denormalsZeroMode);
}

void CPUCalculatorImpl::limitNumOfThreads(unsigned numOfThreads)
{
    if (numOfThreads == 0)
    {
        m_numOfThreads = std::thread::hardware_concurrency();
    }
    else
    {
        m_numOfThreads = numOfThreads;
    }
}


//=================================================================
template<typename InputT, typename OutputT>
OutputT CPUCalculatorImpl::fma(const InputT& a, const InputT& b, const OutputT& c, RoundingMode rm)
{
    return InputT::fma(a, b, c, rm);
}

template<>
uint32_t CPUCalculatorImpl::fma<int8_t, uint32_t>(const int8_t& a, const int8_t& b, const uint32_t& c, RoundingMode rm)
{
    uint32_t res = (uint32_t) std::fma(a, b, c);
    return res;
}

template<>
uint32_t CPUCalculatorImpl::fma<uint8_t, uint32_t>(const uint8_t& a, const uint8_t& b, const uint32_t& c, RoundingMode rm)
{
    uint32_t prod = (uint32_t) a * (uint32_t) b;
    uint32_t res = prod + c;
    return res;
}

// iacc variants are: iacc32_32_t, iacc32_26_t, iacc32_16_t,
// iacc32_32_saturate_t

template<>
iacc32_32_t
CPUCalculatorImpl::fma<int8_t, iacc32_32_t>(const int8_t& a, const int8_t& b, const iacc32_32_t& c, RoundingMode rm)
{
    iacc32_32_t ret = c + ((int64_t) a * (int64_t) b);
    return ret;
}

template<>
iacc32_32_t
CPUCalculatorImpl::fma<uint8_t, iacc32_32_t>(const uint8_t& a, const uint8_t& b, const iacc32_32_t& c, RoundingMode rm)
{
    iacc32_32_t ret = c + ((int64_t) a * (int64_t) b);
    return ret;
}

template<>
iacc32_26_t
CPUCalculatorImpl::fma<int8_t, iacc32_26_t>(const int8_t& a, const int8_t& b, const iacc32_26_t& c, RoundingMode rm)
{
    iacc32_26_t ret = c + ((int64_t) a * (int64_t) b);
    return ret;
}

template<>
iacc32_26_t
CPUCalculatorImpl::fma<uint8_t, iacc32_26_t>(const uint8_t& a, const uint8_t& b, const iacc32_26_t& c, RoundingMode rm)
{
    iacc32_26_t ret = c + ((int64_t) a * (int64_t) b);
    return ret;
}

template<>
iacc32_16_t
CPUCalculatorImpl::fma<int8_t, iacc32_16_t>(const int8_t& a, const int8_t& b, const iacc32_16_t& c, RoundingMode rm)
{
    iacc32_16_t ret = c + ((int64_t) a * (int64_t) b);
    return ret;
}

template<>
iacc32_16_t
CPUCalculatorImpl::fma<uint8_t, iacc32_16_t>(const uint8_t& a, const uint8_t& b, const iacc32_16_t& c, RoundingMode rm)
{
    iacc32_16_t ret = c + ((int64_t) a * (int64_t) b);
    return ret;
}

template<>
iacc32_32_saturate_t CPUCalculatorImpl::fma<uint8_t, iacc32_32_saturate_t>(const uint8_t& a,
                                                                       const uint8_t& b,
                                                                       const iacc32_32_saturate_t& c,
                                                                       RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_32_saturate_t CPUCalculatorImpl::fma<int8_t, iacc32_32_saturate_t>(const int8_t& a,
                                                                      const int8_t& b,
                                                                      const iacc32_32_saturate_t& c,
                                                                      RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}

template<>
iacc32_32_t
CPUCalculatorImpl::fma<int4x2_t, iacc32_32_t>(const int4x2_t& a, const int4x2_t& b, const iacc32_32_t& c, RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_26_t
CPUCalculatorImpl::fma<int4x2_t, iacc32_26_t>(const int4x2_t& a, const int4x2_t& b, const iacc32_26_t& c, RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_16_t
CPUCalculatorImpl::fma<int4x2_t, iacc32_16_t>(const int4x2_t& a, const int4x2_t& b, const iacc32_16_t& c, RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_32_saturate_t CPUCalculatorImpl::fma<int4x2_t, iacc32_32_saturate_t>(const int4x2_t& a,
                                                                        const int4x2_t& b,
                                                                        const iacc32_32_saturate_t& c,
                                                                        RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_32_t CPUCalculatorImpl::fma<uint4x2_t, iacc32_32_t>(const uint4x2_t& a,
                                                       const uint4x2_t& b,
                                                       const iacc32_32_t& c,
                                                       RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_26_t CPUCalculatorImpl::fma<uint4x2_t, iacc32_26_t>(const uint4x2_t& a,
                                                       const uint4x2_t& b,
                                                       const iacc32_26_t& c,
                                                       RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_16_t CPUCalculatorImpl::fma<uint4x2_t, iacc32_16_t>(const uint4x2_t& a,
                                                       const uint4x2_t& b,
                                                       const iacc32_16_t& c,
                                                       RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}
template<>
iacc32_32_saturate_t CPUCalculatorImpl::fma<uint4x2_t, iacc32_32_saturate_t>(const uint4x2_t& a,
                                                                         const uint4x2_t& b,
                                                                         const iacc32_32_saturate_t& c,
                                                                         RoundingMode rm)
{
    MME_ASSERT(0, "this combination is not supported yet");
    return 0;
}

CPUCalculator::CPUCalculator(MmeCommon::ChipType chip,
                              int max_tensor_dims,
                              int max_conv_dims,
                              unsigned seed,
                              uint32_t lfsrNumRegs,
                              uint32_t* lfsr,
                              uint32_t polynomial)
{
    m_impl = std::make_unique<CPUCalculatorImpl>(chip,
                                                 max_tensor_dims,
                                                 max_conv_dims,
                                                 seed,
                                                 lfsrNumRegs,
                                                 lfsr,
                                                 polynomial);
}

CPUCalculator::~CPUCalculator() = default;

void CPUCalculator::limitNumOfThreads(unsigned numOfThreads)
{
    m_impl->limitNumOfThreads(numOfThreads);
}
    // do simple GEMM operation - 2 dim tensors.
void CPUCalculator::doGemm(Matrix& output,
            const Matrix& inputA,
            const Matrix& inputB,
            bool transposeA,
            bool transposeB,
            MmeCommon::RoundingMode rm,
            unsigned fpBiasA,
            unsigned fpBiasB,
            bool clipFp,
            bool clipFpInfIn,
            MmeCommon::InfNanMode infNanModeA,
            MmeCommon::InfNanMode infNanModeB,
            int* zeroPoints)
{
    m_impl->doGemm(output,
                   inputA,
                   inputB,
                   transposeA,
                   transposeB,
                   rm,
                   fpBiasA,
                   fpBiasB,
                   clipFp,
                   clipFpInfIn,
                   infNanModeA,
                   infNanModeB,
                   zeroPoints);
}

void CPUCalculator::doDma(MmeSimTensor& outputTensor, const MmeSimTensor& xTensor, MmeCommon::EMmeOpType op)
{
    m_impl->doDma(outputTensor, xTensor, op);
}

void CPUCalculator::doBatchGemm(MmeSimTensor& outputTensor,
                    const MmeSimTensor& xTensor,
                    const MmeSimTensor& wTensor,
                    MmeCommon::EMmeOpType op,
                    MmeCommon::RoundingMode rm,
                    bool clipFp,
                    bool clipFpInfIn,
                    int* zeroPoints)
{
    m_impl->doBatchGemm(outputTensor,
                        xTensor,
                        wTensor,
                        op,
                        rm,
                        clipFp,
                        clipFpInfIn,
                        zeroPoints);
}

void CPUCalculator::doConvolution(MmeSimTensor& outputTensor,
                    const MmeSimTensor& xTensor,
                    const MmeSimTensor& wTensor,
                    const MmeSimTensor& yTensor,
                    const ConvolutionParams& params,
                    MmeCommon::EMmeOpType op,
                    MmeCommon::RoundingMode rm,
                    bool clipFp,
                    bool clipFpInfIn,
                    int* zeroPoints)
{
    m_impl->doConvolution(outputTensor,
                          xTensor,
                          wTensor,
                          yTensor,
                          params,
                          op,
                          rm,
                          clipFp,
                          clipFpInfIn,
                          zeroPoints);
}

void CPUCalculator::doActivation(pMMESimTensor output,
                    const pMMESimTensor acc,  // input
                    const pMMESimTensor res,  // residual (cin)
                    const pMMESimTensor bias,  // 1d bias tensor
                    bool relu,
                    MmeCommon::RoundingMode rm,
                    const ActivationParams* actParams,
                    bool clipFp,
                    bool clipInfIn,
                    bool flushDenorms,
                    bool stochasticFTZfp8)
{
    m_impl->doActivation(std::move(output),
                         acc,
                         res,
                         bias,
                         relu,
                         rm,
                         actParams,
                         clipFp,
                         clipInfIn,
                         flushDenorms,
                         stochasticFTZfp8);
}

void CPUCalculator::doMemoryWrite(pMMESimTensor memory,
                    const pMMESimTensor mmeOut,
                    MmeCommon::EMmeReductionOp reductionOp,
                    MmeCommon::EMmeReductionRm reductionRm,
                    bool clipFp)
{
    m_impl->doMemoryWrite(std::move(memory),
                          mmeOut,
                          reductionOp,
                          reductionRm,
                          clipFp);
}

void CPUCalculator::activationGemmlowp(pMMESimTensor out,  // the output tensor.
                                        pMMESimTensor acc,  // the input tensor (the output of the conv stage).
                                        pMMESimTensor res,  // residual tensor (cin). nullptr when there's no residual.
                                        pMMESimTensor bias,  // 1D bias tensor with K elements. (either int32_t or
                                                            // float). nullptr when there's no bias.
                                        bool reluEn,  // enable relu.
                                        MmeCommon::RoundingMode roundingMode,  // rounding mode for type conversion
                                        const ActivationParams* actParams)
{
    m_impl->activationGemmlowp(out,
                               acc,
                               res,
                               bias,
                               reluEn,
                               roundingMode,
                               actParams);
}
