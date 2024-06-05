#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <optional>

#include <hl_gcfg/hlgcfg.hpp>
#include "log_manager.h"

// Changes the configuration only for this scope
class ScopedConfigurationChange
{
public:
    ScopedConfigurationChange(std::string        keyName,
                              const std::string& newValue,
                              bool enableExperimental = false)
    : m_keyName(std::move(keyName)), m_enableExperimental(enableExperimental)
    {
        auto v = hl_gcfg::getGcfgItemValue(m_keyName.data());
        if (v.has_value())
        {
            m_prevValue = v.value();
        }
        else
        {
            LOG_ERR(GC_CONF, "{}: {}", HLLOG_FUNC, v.errorDesc());
        }
        auto ret = hl_gcfg::setGcfgItemValue(m_keyName, newValue, m_enableExperimental);
        if (ret.has_error())
        {
            LOG_ERR(GC_CONF, "{}: {}", HLLOG_FUNC, ret.errorDesc());
        }
    }

    ~ScopedConfigurationChange()
    {
        if (m_prevValue)
        {
            auto ret = hl_gcfg::setGcfgItemValue(m_keyName, m_prevValue.value(), m_enableExperimental);
            if (ret.has_error())
            {
                LOG_ERR(GC_CONF, "{}: {}", HLLOG_FUNC, ret.errorDesc());
            }
        }
    }

    // no copyable
    ScopedConfigurationChange(const ScopedConfigurationChange&) = delete;
    ScopedConfigurationChange& operator=(const ScopedConfigurationChange&) = delete;

    // no movable
    ScopedConfigurationChange(ScopedConfigurationChange&&) = delete;
    ScopedConfigurationChange& operator=(ScopedConfigurationChange&&) = delete;

private:
    std::optional<std::string> m_prevValue;
    std::string m_keyName;
    bool m_enableExperimental;
};

class ScopedEnvChange
{
public:
    ScopedEnvChange(std::string varName, const std::string& newVal) : m_varName(std::move(varName))
    {
        m_oldVal = (getenv(m_varName.c_str()) == nullptr) ? "" : getenv(m_varName.c_str());
        LOG_INFO_T(SYN_API, "{} setenv {} to {}. Old value {}", HLLOG_FUNC, m_varName, newVal, m_oldVal);
        int rc = setenv(m_varName.c_str(), newVal.c_str(), 1);
        (void)rc;
        assert(rc == 0);
    }

    ~ScopedEnvChange()
    {
        if (m_oldVal.empty())
        {
            LOG_INFO_T(SYN_API, "{} unset {}", HLLOG_FUNC, m_varName);
            int rc = unsetenv(m_varName.c_str());
            (void)rc;
            assert(rc == 0);
        }
        else
        {
            LOG_INFO_T(SYN_API, " {} setenv back {} to {}.", HLLOG_FUNC, m_varName, m_oldVal);
            int rc = setenv(m_varName.c_str(), m_oldVal.c_str(), 1);
            (void)rc;
            assert(rc == 0);
        }
    }

private:
    std::string m_varName;
    std::string m_oldVal;
};
