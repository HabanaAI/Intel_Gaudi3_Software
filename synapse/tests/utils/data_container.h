#pragma once

#include "synapse_common_types.h"
#include <string>
#include <vector>
#include <iostream>

class DataContainer
{
public:
    virtual ~DataContainer() = default;

    virtual std::vector<uint8_t>     getBuffer(const std::string& tensorName) const      = 0;
    virtual std::vector<TSize>       getShape(const std::string& tensorName) const       = 0;
    virtual std::vector<uint8_t>     getPermutation(const std::string& tensorName) const = 0;
    virtual synDataType              getDataType(const std::string& tensorName) const    = 0;
    virtual std::vector<std::string> getTensorsNames() const                             = 0;

    template<class T>
    std::vector<T> getElements(const std::string& tensorName) const
    {
        auto           buffer = getBuffer(tensorName);
        std::vector<T> ret(buffer.size() / sizeof(T));
        memcpy(ret.data(), buffer.data(), buffer.size());
        return ret;
    }
};