#include "test_tensor_dimensions.hpp"

void TestTensorsDimensions::setDimensions(bool isInput, unsigned tensorIndex, const TensorDimensions& rTensorDimensions)
{
    m_tensorsDimensions.insert(std::make_pair(TensorKey(isInput, tensorIndex), rTensorDimensions));
}

const TensorDimensions* TestTensorsDimensions::getDimensions(bool isInput, unsigned tensorIndex) const
{
    TestTensorDimensionsMap::const_iterator iter = m_tensorsDimensions.find(TensorKey(isInput, tensorIndex));
    if (iter == m_tensorsDimensions.end())
    {
        return nullptr;
    }
    return &(iter->second);
}