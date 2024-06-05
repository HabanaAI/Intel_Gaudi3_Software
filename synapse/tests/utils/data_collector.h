#pragma once

#include "hpp/syn_host_buffer.hpp"
#include "hpp/syn_recipe.hpp"
#include "data_container.h"

#include <map>
#include <string>
#include <vector>
#include <cstring>

class DataCollector : public DataContainer
{
public:
    DataCollector(const syn::Recipe& recipe)
    {
        auto tensorsInfo = recipe.getLaunchTensorsInfoExt();
        for (const auto& i : tensorsInfo)
        {
            m_tensorsInfo[i.tensorName] = i;
        }
    }

    const synRetrievedLaunchTensorInfoExt& getInfo(const std::string& tensorName) const
    {
        return m_tensorsInfo.at(tensorName);
    }

    void setBuffer(const std::string& tensorName, const syn::HostBuffer& buffer) { m_tensors[tensorName] = buffer; }

    void removeBuffer(const std::string& tensorName) { m_tensors.erase(tensorName); }

    std::vector<uint8_t> getBuffer(const std::string& tensorName) const override
    {
        const syn::HostBuffer& hostBuffer = m_tensors.at(tensorName);
        return std::vector<uint8_t>(hostBuffer.getAs<uint8_t>(), hostBuffer.getAs<uint8_t>() + hostBuffer.getSize());
    }

    syn::HostBuffer getHostBuffer(const std::string& tensorName) const { return m_tensors.at(tensorName); }

    std::vector<std::string> getTensorsNames() const override
    {
        std::vector<std::string> tensorsNames;
        tensorsNames.reserve(m_tensors.size());

        for (const auto& t : m_tensors)
        {
            tensorsNames.emplace_back(t.first);
        }
        return tensorsNames;
    }

    std::vector<TSize> getShape(const std::string& tensorName) const override
    {
        const synRetrievedLaunchTensorInfoExt info = getInfo(tensorName);
        return std::vector<TSize>(info.tensorMaxSize, info.tensorMaxSize + info.tensorDims);
    }

    std::vector<uint8_t> getPermutation(const std::string& tensorName) const override
    {
        const synRetrievedLaunchTensorInfoExt info = getInfo(tensorName);
        return std::vector<uint8_t>(info.tensorPermutation, info.tensorPermutation + info.tensorDims);
    }

    synDataType getDataType(const std::string& tensorName) const override
    {
        const synRetrievedLaunchTensorInfoExt info = getInfo(tensorName);
        return info.tensorDataType;
    }

private:
    std::map<std::string, syn::HostBuffer>                 m_tensors;
    std::map<std::string, synRetrievedLaunchTensorInfoExt> m_tensorsInfo;
};