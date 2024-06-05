#pragma once
#include "hlgcfg_log.hpp"
namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{

template<class T>
GcfgDefaultItem<T>::GcfgDefaultItem(const T& defaultVal)
: m_default(defaultVal)
{
}

template<class T>
GcfgDefaultItem<T>& GcfgDefaultItem<T>::operator<<(std::pair<uint32_t, T> const & deviceValuePair)
{
    if (deviceValuePair.first < m_defaultsPerDevice.size())
    {
        m_defaultsPerDevice[deviceValuePair.first] = deviceValuePair.second;
    }
    else
    {
        HLGCFG_LOG_CRITICAL("device type {} (value {}) is not supported for default item. ignore.",
                            (int32_t)deviceValuePair.first,
                            toString(deviceValuePair.second));
        m_default = deviceValuePair.second;
    }
    return *this;
}

template<class T>
const T& GcfgDefaultItem<T>::value(uint32_t device) const
{
    return (device < m_defaultsPerDevice.size() && m_defaultsPerDevice[device].has_value()) ? m_defaultsPerDevice[device].value()
                                                                                            : m_default.value();
}

template<class T>
std::string GcfgDefaultItem<T>::getValueStr() const
{
    std::string out;
    bool first = true;
    auto processDefault = [&](std::optional<T> const & defaultValue, uint32_t deviceType){
        if (defaultValue.has_value())
        {
            if (!first)
            {
                out += " | ";
            }
            out += std::to_string((int32_t)deviceType) + ": " + toString(defaultValue.value());
            first = false;
        }
    };
    for (uint32_t deviceT = 0; deviceT < m_defaultsPerDevice.size(); ++deviceT)
    {
        processDefault(m_defaultsPerDevice[deviceT], deviceT);
    }
    processDefault(m_default, InvalidDeviceType);
    return out;
}
}}