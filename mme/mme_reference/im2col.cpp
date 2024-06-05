#include "im2col.h"
#include "data_types/non_standard_dtypes.h"

using namespace MmeCommon;

#define REGISTER_TYPE_IM2COL(enumType, realType)                                                                       \
    m_im2colFuncMap[enumType] = std::mem_fn(&RefIm2Col::internalDoIm2Col<realType>);

#define REGISTER_TYPE_COL2IM(enumType, realType)                                                                       \
    m_col2imFuncMap[enumType] = std::mem_fn(&RefIm2Col::internalDoCol2Im<realType>);

RefIm2Col::RefIm2Col(int max_tensor_dims, const RoundingMode rm) : m_rm(rm), m_mme_max_tensor_dims(max_tensor_dims)
{
    // Col2Im - works on output so only supported for fp32 type
    REGISTER_TYPE_COL2IM(EMmeDataType::e_type_fp32, fp32_t);
    // Im2Col - float
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_fp32_ieee, float);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_fp32, fp32_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_tf32, tf32_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_bf16, bf16_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_ufp16, ufp16_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_fp16, fp16_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_fp8_143, fp8_143_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_fp8_152, fp8_152_t);
    // Im2Col - int
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_int8, int8_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_uint8, int8_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_int4, int4x2_t);
    REGISTER_TYPE_IM2COL(EMmeDataType::e_type_uint4, uint4x2_t);
}

void RefIm2Col::doIm2Col(const MmeSimTensor& tensor,
                         const ConvolutionParams& params,
                         CommonRefMatrix& refMatrix,
                         const SizeArray& kernelShape,
                         const SizeArray& outputShape)
{
    auto doIm2ColFunc = m_im2colFuncMap.at(tensor.getElementType());
    MME_ASSERT(m_im2colFuncMap.count({tensor.getElementType()}), "type is not registered in function map !");
    doIm2ColFunc(this, tensor, params, refMatrix, kernelShape, outputShape);
}

void RefIm2Col::doCol2Im(MmeSimTensor& tensor,
                         const ConvolutionParams& params,
                         const CommonRefMatrix& refMatrix,
                         const SizeArray& kernelShape,
                         const SizeArray& outputShape)
{
    auto doCol2ImFunc = m_col2imFuncMap.at(tensor.getElementType());
    MME_ASSERT(m_col2imFuncMap.count({tensor.getElementType()}), "type is not registered in function map !");
    doCol2ImFunc(this, tensor, params, refMatrix, kernelShape, outputShape);
}

template<typename T>
void RefIm2Col::internalDoIm2Col(const MmeSimTensor& tensor,
                                 const ConvolutionParams& params,
                                 CommonRefMatrix& refMatrix,
                                 const SizeArray& kernelShape,
                                 const SizeArray& outputShape)
{
    // create padding vector size of tensor DIM_C
    unsigned FCDSize = tensor.getSize(0);

    std::vector<T> paddingVector(FCDSize);
    for (int i = 0; i < FCDSize; i++)
    {
        paddingVector[i] = *((T*) &params.paddingValue.int32);
    }
    m_tensorShape = tensor.getSizes();
    unsigned matrixHeight = refMatrix.getHeight();
    unsigned matrixWidth = refMatrix.getWidth();
    // run on all columns -
    for (unsigned col = 0; col < matrixWidth; col += FCDSize)
    {
        // generate relevant kernel
        CoordArray currentKernelCoord = getCurrentKernelPosition<T>(kernelShape, col, FCDSize);
        // go over all rows
        for (unsigned row = 0; row < matrixHeight; row++)
        {
            bool pad = false;
            // get tensor position
            CoordArray tPos = getCurrentTensorPosition<T>(&pad, row, outputShape, currentKernelCoord, params);
            // find the matrix position
            auto currentMatrixPos = reinterpret_cast<T*>(refMatrix.getElementAt(row, col));
            // get the tensor value - real or padding.
            auto tensorData = pad ? paddingVector.data() : reinterpret_cast<T*>(tensor.getElementAt(tPos));
            // copy tensor value to matrix position.
            memcpy((void*) currentMatrixPos, (void*) tensorData, sizeof(T) * FCDSize);
        }
    }
}

template<typename T>
void RefIm2Col::internalDoCol2Im(MmeSimTensor& tensor,
                                 const ConvolutionParams& params,
                                 const CommonRefMatrix& refMatrix,
                                 const SizeArray& kernelShape,
                                 const SizeArray& outputShape)
{
    // create padding vector size of tensor DIM_C
    unsigned FCDSize = tensor.getSize(0);
    std::vector<T> paddingVector(FCDSize, (T) params.paddingValue.int32);
    m_tensorShape = tensor.getSizes();

    T zero = T(0.0f);
    tensor.fill(reinterpret_cast<byte*>(&zero));
    unsigned matrixHeight = refMatrix.getHeight();
    unsigned matrixWidth = refMatrix.getWidth();
    // run on all columns -
    for (unsigned col = 0; col < matrixWidth; col += FCDSize)
    {
        // generate relevant kernel
        CoordArray currentKernelCoord = getCurrentKernelPosition<T>(kernelShape, col, FCDSize);
        // go over all rows
        for (unsigned row = 0; row < matrixHeight; row++)
        {
            bool pad = false;
            // get tensor position
            CoordArray tPos = getCurrentTensorPosition<T>(&pad, row, outputShape, currentKernelCoord, params);

            // get the matrix value - real or padding.
            auto currentMatrixPos = reinterpret_cast<const T*>(refMatrix.getElementAt(row, col));
            // find the tensor position
            auto tensorData = pad ? paddingVector.data() : reinterpret_cast<T*>(tensor.getElementAt(tPos));
            // copy matrix value to tensor.
            for (unsigned i = 0; i < FCDSize; i++)
            {
                add(tensorData[i], currentMatrixPos[i], &tensorData[i]);
            }
        }
    }
}

template<typename T>
CoordArray RefIm2Col::getCurrentKernelPosition(const SizeArray& kernelShape, unsigned int column, unsigned int FCDSize)
{
    CoordArray kernel = {0};
    unsigned divCol = column / FCDSize;
    auto dim = std::next(std::next(kernelShape.begin()));  // start from DIM_S
    for (auto& element : kernel)
    {
        unsigned kernelSize = (dim == kernelShape.end()) ? 1 : *dim;
        element = divCol % kernelSize;
        divCol /= kernelSize;
        if (dim != kernelShape.end()) dim++;
    }
    return kernel;
}

template<typename T>
SizeArray RefIm2Col::getCurrentTensorPosition(bool* shouldPad,
                                              unsigned int row,
                                              const SizeArray& outputShape,
                                              const CoordArray& currentKernelCoord,
                                              const ConvolutionParams& params)
{
    CoordArray tensorPosition = {0};
    unsigned divRow = row;
    for (unsigned dim = 1; dim < m_mme_max_tensor_dims; dim++)
    {
        unsigned prevDim = dim - 1;
        unsigned outputPos = divRow % outputShape[dim];
        divRow /= outputShape[dim];
        // find tensor position - N = MS+RD-P
        tensorPosition[dim] = (outputPos * params.convStride[prevDim]) +
                              (currentKernelCoord[prevDim] * params.dilation[prevDim]) - params.padding[prevDim];
        // set pad if necessary
        *shouldPad = needPadding(tensorPosition[dim], m_tensorShape[dim]);
        if (*shouldPad) break;
    }
    return tensorPosition;
}

template<>
void RefIm2Col::add<float>(const float& a, const float& b, float* result) const
{
    float32 aF(a), bF(b);
    float32 cF = float32::add(aF, bF, m_rm);
    *result = cF.toFloat();
}