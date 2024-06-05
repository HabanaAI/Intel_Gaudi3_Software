#pragma once
#include "../string_utils.hpp"
#include "../hlgcfg_item_observer.hpp"
#include "../hlgcfg.hpp"

#include "hlgcfg_log.hpp"
#include <cmath>
#include <exception>
#include <list>
#include <string>
#include <typeinfo>
#include "../hlgcfg_item_observer.hpp"
#include <cxxabi.h>

namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{

namespace details {
inline std::vector<std::string> buildNameVec(const std::string &name, const std::vector<std::string> &aliases) {
    std::vector<std::string> res{name};
    res.insert(res.end(), aliases.cbegin(), aliases.cend());
    return res;
}
}

inline
GcfgItem::GcfgItem(const std::string&              name,
                   const std::vector<std::string>& aliases,
                   const std::string&              description,
                   bool                            isPublic,
                   ObserversList                   observers)
: m_setFrom(GcfgSource::default_src)
, m_names(details::buildNameVec(name, aliases))
, m_description(description)
, m_isPublic(isPublic)
, m_shouldUnregister(true)
, m_observers(std::move(observers))
{
    auto ret = registerGcfgItem(*this);
    if (ret.has_error())
    {
        if (ret.errorCode() == ErrorCode::configNameAlreadyRegistered)
        {
            m_shouldUnregister = false;
        }
    }
}

inline
GcfgItem::GcfgItem(const std::string& name,
                   const std::string& description,
                   bool               isPublic,
                   ObserversList      observers)
: GcfgItem(name, {}, description, isPublic, std::move(observers))
{
}

inline
GcfgItem::~GcfgItem()
{
    if (m_shouldUnregister)
    {
        unregisterGcfgItem(*this);
    }
}

inline
VoidOutcome GcfgItem::updateFromEnv(bool enableExperimental)
{
    for (const auto& name : m_names)
    {
        const char* envValue = getenv(name.c_str());
        if (envValue == nullptr) continue;

        if (m_isPublic || enableExperimental)
        {

            if (m_setFrom == GcfgSource::env)
            {
                const bool areEqual = std::equal(m_usedEnvValue.begin(), m_usedEnvValue.end(),
                                                 envValue, envValue + strlen(envValue),
                                                 [](char a, char b) {
                                                     return tolower(a) == tolower(b);
                                                 });
                if (areEqual)
                {
                    continue;
                }
                auto valueStr = getValueStr();
                const bool areEqualWithValueStr = std::equal(valueStr.begin(), valueStr.end(),
                                                             envValue, envValue + strlen(envValue),
                                                             [](char a, char b) {
                                                                 return tolower(a) == tolower(b);
                                                             });
                if (areEqualWithValueStr)
                {
                    continue;
                }
                // might be the same value with a different conversion (true<->1 false<->0)
                auto ret = setFromString(std::string(envValue));
                if (ret && valueStr == getValueStr())
                {
                    continue;
                }
                setFromString(m_usedEnvValue);
                HLGCFG_RETURN_ERR(envAliasesConflict,
                                  "multiple conflicting Global Configuration env variables ({}=\"{}\" vs {}=\"{}\")!",
                                  name, envValue, m_usedEnvAliasName, m_usedEnvValue);
            }

            HLGCFG_LOG_INFO("Identified Global Configuration env variable ({})", name);
            auto ret = setFromString(std::string(envValue));
            if (!ret)
            {
                return  ret;
            }
            m_setFrom = GcfgSource::env;
            m_usedEnvAliasName = name;
            m_usedEnvValue = envValue;
        }
        else
        {
            HLGCFG_LOG_WARN("Using Global Configuration private env variable without ENABLE_EXPERIMENTAL_FLAGS=true "
                           "({}=\"{}\" vs {}=\"{}\")!",
                            name,
                            envValue,
                            m_usedEnvAliasName,
                            m_usedEnvValue);
            HLGCFG_RETURN_ERR(privateConfigAccess, "{} is internal config. cannot be set without ENABLE_EXPERIMENTAL_FLAGS", name);
        }
    }
    return {};

}

inline
VoidOutcome GcfgItem::updateObservers(const std::string& value)
{
    for (auto* const observer : m_observers)
    {
        const std::string& observerName = observer->primaryName();
        HLGCFG_LOG_INFO("name: {}, observer: {} valueToUpdate: {}", primaryName(), observerName, value);
        auto ret = observer->updateValue(value);
        if (!ret)
        {
            return ret;
        }
    }
    return {};
}

template<class T, class R>
GcfgItemImpl<T,R>::GcfgItemImpl(const std::string&              name,
                                const std::vector<std::string>& aliases,
                                const std::string&              description,
                                const GcfgDefaultItem<T>&       defaultValue,
                                bool                            isPublic,
                                ObserversList                   observersList)
: GcfgItem(name, aliases, description, isPublic, std::move(observersList))
, m_defaultValue(defaultValue)
{
    m_value = m_defaultValue.value(InvalidDeviceType);
    auto ret = updateObservers(toString(m_value));
    if (!ret)
    {
        throw std::invalid_argument(ret.errorDesc());
    }
}

template<class T, class R>
GcfgItemImpl<T,R>::GcfgItemImpl(const std::string&        name,
                                const std::string&        description,
                                const GcfgDefaultItem<T>& defaultValue,
                                bool                      isPublic,
                                ObserversList             observersList)
: GcfgItemImpl(name, {}, description, defaultValue, isPublic, std::move(observersList))
{
}

template<class T, class R>
const R& GcfgItemImpl<T,R>::value() const
{
    return m_setFrom == GcfgSource::default_src ? m_defaultValue.value(hl_gcfg::getDeviceType()) : m_value;
}

template<class T, class R>
const R& GcfgItemImpl<T,R>::getDefaultValue(uint32_t device) const
{
    return m_defaultValue.value(device);
}

template<class T, class R>
VoidOutcome GcfgItemImpl<T,R>::setValue(const T& value)
{
    return setValue(value, false);
}

template<class T, class R>
void GcfgItemImpl<T,R>::reset()
{
    // Clear setFrom and re-init value with the default one (received in the constructor)
    m_setFrom = GcfgSource::default_src;
    m_value = m_defaultValue.value(InvalidDeviceType);
}

template<class T, class R>
VoidOutcome GcfgItemImpl<T,R>::setFromString(const std::string& str, bool fromObserver)
{
    return setValue(str, fromObserver);
}

template<class T, class U>
T getTypedValue(const U& val) { return val; }

template<class T>
T getTypedValue(const std::string& val) { return hl_gcfg::fromString<T>(val); }


template<class T, class R>
template<class U>
VoidOutcome GcfgItemImpl<T, R>::setValue(const U& value, bool fromObserver)
{
    // Environment variable take precedence
    if (m_setFrom == GcfgSource::env)
    {
        try
        {
            T newValue = getTypedValue<T>(value);
            if (newValue == m_value)
            {
                // it's ok to set the same value as it was set before
                return {};
            }
        }
        catch(...)
        {
        }
        HLGCFG_RETURN_ERR(valueWasAlreadySetFromEnv, "{} was already set from env variable ({}).", primaryName(), getUsedEnvAlias());
    }

    if (m_setFrom == GcfgSource::observer && !fromObserver)
    {
        HLGCFG_LOG_WARN("override {} value that already set from observation", primaryName());
    }

    T newValue{};
    try
    {
        newValue = getTypedValue<T>(value);
    }
    catch (const std::exception& e)
    {
        std::string typeName = typeid(T).name();
        int status = -4;
        std::unique_ptr<char, decltype(std::free)&> demangled_name {nullptr, std::free};
        demangled_name.reset(abi::__cxa_demangle(typeName.c_str(), nullptr, nullptr, &status));
        if (demangled_name != nullptr && status == 0)
        {
           typeName = demangled_name.get();
        }

        HLGCFG_RETURN_ERR(invalidString, "invalid string value '{}' for global conf {}. '{}' type expected", toString(value), primaryName(), typeName);
    }

    std::swap(m_value, newValue);
    HLGCFG_LOG_DEBUG("configuration parameter name={}, value={}", primaryName(), toString(m_value));
    if (fromObserver)
    {
        m_setFrom = GcfgSource::observer;
    }
    else
    {
        m_setFrom = GcfgSource::runtime;
    }

    if (!m_observers.empty())
    {

        auto ret = updateObservers(toString(m_value));
        if (!ret)
        {
            // update failed - tyry to revert to previous value
            auto retRevert = updateObservers(toString(newValue));
            if (!retRevert)
            {
                // inconsistent state (neither new nor previous)
                throw std::invalid_argument(ret.errorDesc());
            }
            std::swap(m_value, newValue);
            return ret;
        }
    }
    return {};
}

template<class T, class R>
std::string GcfgItemImpl<T, R>::getValueStr() const
{
    return toString(value());
}

template<>
std::string GcfgItemImpl<hl_gcfg::SizeParam, uint64_t>::getValueStr() const;

template<class T, class R>
std::string GcfgItemImpl<T,R>::getDefaultValuesStr() const
{
    return m_defaultValue.getValueStr();
}

template<>
inline std::string GcfgItemImpl<hl_gcfg::SizeParam, uint64_t>::getValueStr() const
{
    return m_setFrom == GcfgSource::default_src ? m_defaultValue.value(hl_gcfg::getDeviceType()).getString()
                                                : m_value.getString();
}

template<>
inline const uint64_t& GcfgItemImpl<hl_gcfg::SizeParam, uint64_t>::value() const
{
    return m_setFrom == GcfgSource::default_src ? getDefaultValue(hl_gcfg::getDeviceType()) : m_value.getByteVal();
}

template<>
inline const uint64_t& GcfgItemImpl<hl_gcfg::SizeParam, uint64_t>::getDefaultValue(uint32_t device) const
{
    return m_defaultValue.value(device).getByteVal();
}
}}

#include "hlgcfg_item_observer.inl"