#pragma once
#include "convolution_params.h"
#include "data_types/non_standard_dtypes.h"
#include "sim_tensor.h"
#include <functional>
#include <map>
#include <vector>

using CoordArray = MmeCommon::SizeArray;

class RefIm2Col final
{
public:
    using doIm2ColFunc = std::function<void(RefIm2Col*,
                                            const MmeSimTensor&,
                                            const ConvolutionParams&,
                                            CommonRefMatrix&,
                                            const MmeCommon::SizeArray&,
                                            const MmeCommon::SizeArray&)>;
    using doCol2ImFunc = std::function<void(RefIm2Col*,
                                            MmeSimTensor&,
                                            const ConvolutionParams&,
                                            const CommonRefMatrix&,
                                            const MmeCommon::SizeArray&,
                                            const MmeCommon::SizeArray&)>;
    RefIm2Col(int max_tensor_dims, const MmeCommon::RoundingMode rm);
    ~RefIm2Col() = default;

    void doIm2Col(const MmeSimTensor& tensor,
                  const ConvolutionParams& params,
                  CommonRefMatrix& refMatrix,
                  const MmeCommon::SizeArray& kernelShape,
                  const MmeCommon::SizeArray& outputShape);
    void doCol2Im(MmeSimTensor& tensor,
                  const ConvolutionParams& params,
                  const CommonRefMatrix& refMatrix,
                  const MmeCommon::SizeArray& kernelShape,
                  const MmeCommon::SizeArray& outputShape);

private:
    template<typename T>
    void internalDoIm2Col(const MmeSimTensor& tensor,
                          const ConvolutionParams& params,
                          CommonRefMatrix& refMatrix,
                          const MmeCommon::SizeArray& kernelShape,
                          const MmeCommon::SizeArray& outputShape);
    template<typename T>
    void internalDoCol2Im(MmeSimTensor& tensor,
                          const ConvolutionParams& params,
                          const CommonRefMatrix& refMatrix,
                          const MmeCommon::SizeArray& kernelShape,
                          const MmeCommon::SizeArray& outputShape);
    // get current coordinates in kernel
    template<typename T>
    CoordArray getCurrentKernelPosition(const MmeCommon::SizeArray& kernelShape, unsigned column, unsigned FCDSize);
    // get the current tensor position offset N = M*S + R*D - P
    template<typename T>
    CoordArray getCurrentTensorPosition(bool* shouldPad,
                                        unsigned row,
                                        const MmeCommon::SizeArray& outputShape,
                                        const CoordArray& currentKernelCoord,
                                        const ConvolutionParams& params);
    // check if current position is outside of tensor boundries, which means we
    // need to get a padding value;
    bool needPadding(int currentTensorPosition, unsigned totalDimSize)
    {
        return (currentTensorPosition < 0) || (currentTensorPosition >= totalDimSize);
    }

    template<typename T>
    void add(const T& a, const T& b, T* result) const
    {
        T one(1.0f, MmeCommon::RoundingMode::RoundToNearest);
        *result = T::fma(a, one, b, m_rm);
    }

    // todo AlonG: This is a mystery function... Need to resolve that
    void add(const int4x2_t& a, const int4x2_t& b, int4x2_t* result)
    {
        *result = a;  //  + b;
    }
    MmeCommon::SizeArray m_tensorShape = {0};
    std::map<MmeCommon::EMmeDataType, doIm2ColFunc> m_im2colFuncMap;
    std::map<MmeCommon::EMmeDataType, doCol2ImFunc> m_col2imFuncMap;
    MmeCommon::RoundingMode m_rm = MmeCommon::RoundingMode::RoundToNearest;
    int m_mme_max_tensor_dims;
};