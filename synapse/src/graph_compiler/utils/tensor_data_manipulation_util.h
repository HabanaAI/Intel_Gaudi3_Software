#pragma once

#include "tensor.h"

namespace TensorElementAccess
{
template<typename T>
T& getElement(const TensorPtr& tensor, const CoordArray& coordinates)
{
    // It is assumed that coordinates is valid
    HB_ASSERT(tensor->getData() != nullptr, "expecting that tensor {} has allocated data", tensor->getName());
    uint64_t offset = 0;
    for (unsigned int i = 0; i < tensor->getDim(); ++i)
    {
        offset += coordinates[i] * tensor->getStrideInElements(i);
    }
    return *(reinterpret_cast<T*>(tensor->getData()) + offset);
}

template<typename T>
void setElement(const TensorPtr& tensor, const CoordArray& coordinates, T newElem)
{
    T& oldElemPtr = TensorElementAccess::getElement<T>(tensor, coordinates);
    oldElemPtr     = newElem;
}
}  // namespace TensorElementAccess