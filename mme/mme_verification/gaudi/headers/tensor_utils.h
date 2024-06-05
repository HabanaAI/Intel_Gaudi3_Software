#ifndef _TENSOR_UTILS_H
#define _TENSOR_UTILS_H

#include "sim_tensor.h"
#include "convolution_params.h"
#include "gaudi/mme.h"

void genRandomValues(MmeSimTensor* t,
                     MmeSimTensor::f32_t minValue,
                     MmeSimTensor::f32_t maxValue,
                     bool fp,
                     unsigned scale);
bool transposeDims(int dim1, int dim2, MmeCommon::SizeArray strides, MmeCommon::SizeArray sizes);
#endif
