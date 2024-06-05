
#include <synapse_types_operators.h>

bool operator==(const synConvolution3DParamsV2& lhs, const synConvolution3DParamsV2& rhs)
{
    return (lhs.kernel[CONV_KERNEL_WIDTH] == rhs.kernel[CONV_KERNEL_WIDTH]) &&
           (lhs.kernel[CONV_KERNEL_HEIGHT] == rhs.kernel[CONV_KERNEL_HEIGHT]) &&
           (lhs.kernel[CONV_KERNEL_DEPTH] == rhs.kernel[CONV_KERNEL_DEPTH]) &&
           (lhs.stride[CONV_STRIDE_WIDTH] == rhs.stride[CONV_STRIDE_WIDTH]) &&
           (lhs.stride[CONV_STRIDE_HEIGHT] == rhs.stride[CONV_STRIDE_HEIGHT]) &&
           (lhs.stride[CONV_STRIDE_DEPTH] == rhs.stride[CONV_STRIDE_DEPTH]) &&
           (lhs.padding[CONV_PAD_LEFT] == rhs.padding[CONV_PAD_LEFT]) &&
           (lhs.padding[CONV_PAD_RIGHT] == rhs.padding[CONV_PAD_RIGHT]) &&
           (lhs.padding[CONV_PAD_TOP] == rhs.padding[CONV_PAD_TOP]) &&
           (lhs.padding[CONV_PAD_BOTTOM] == rhs.padding[CONV_PAD_BOTTOM]) &&
           (lhs.padding[CONV_PAD_FRONT] == rhs.padding[CONV_PAD_FRONT]) &&
           (lhs.padding[CONV_PAD_BACK] == rhs.padding[CONV_PAD_BACK]) &&
           (lhs.dilation[CONV_DIL_WIDTH] == rhs.dilation[CONV_DIL_WIDTH]) &&
           (lhs.dilation[CONV_DIL_HEIGHT] == rhs.dilation[CONV_DIL_HEIGHT]) && (lhs.activation == rhs.activation) &&
           (lhs.paddingType == rhs.paddingType) && (lhs.nGroups == rhs.nGroups);
}

bool operator==(const synActivationParams& lhs, const synActivationParams& rhs)
{
    return lhs.reluEnable == rhs.reluEnable;
}
