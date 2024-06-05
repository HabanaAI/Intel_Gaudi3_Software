#pragma once

#include <stdint.h>

template<class T, class V>
void setBuffer(void* buffer, uint64_t numOfElements, V getVal)
{
    T* data = static_cast<T*>(buffer);
    for (uint64_t j = 0; j < numOfElements; j++)
    {
        data[j] = getVal();
    }
}