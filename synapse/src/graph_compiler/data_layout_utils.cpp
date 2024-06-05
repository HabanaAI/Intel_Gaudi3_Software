#include "data_layout_utils.h"

bool isDenseAfterPermute(const TensorPtr& input, const gc::Permutation& perm)
{
    // get the sizes
    TSize sizes[Tensor::c_tensorMaxNDim];
    input->getAllNSizesInElements(sizes);
    perm.permuteShape(sizes, input->getDim());

    // get the strides
    TStride strides[Tensor::c_numOfNStrides];
    input->getNStridesInBytes(strides);
    perm.permuteShape(strides, input->getDim());

    // create the potential tensor for the assertion
    TensorPtr potentialTensor =
        std::make_shared<Tensor>(input->getDim(), sizes, input->getElementType(), nullptr, strides);
    return potentialTensor->isDenseLayout();
}