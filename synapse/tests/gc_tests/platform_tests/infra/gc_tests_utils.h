#pragma once

#include "hpp/syn_context.hpp"
#include "synapse_common_types.h"

static inline std::string synDeviceTypeToString(synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGoya2:
            return "goya2";
        case synDeviceGaudi:
            return "gaudi1";
        case synDeviceGaudi2:
            return "gaudi2";
        case synDeviceGaudi3:
            return "gaudi3";
        case synDeviceEmulator:
            return "emulator";
        case synDeviceTypeInvalid:
            return "invalid";
        case synDeviceTypeSize:
            return "size";
        default:
            return "unknown";
    }
}

class ScopedConfig
{
public:
    ScopedConfig(syn::Context ctx, const std::string& config, const std::string& value) : m_ctx(ctx), m_config(config)
    {
        m_originalValue = m_ctx.getConfiguration(m_config);
        m_ctx.setConfiguration(m_config, value);
    }

    ~ScopedConfig() { m_ctx.setConfiguration(m_config, m_originalValue); }

private:
    syn::Context m_ctx;
    std::string  m_config;
    std::string  m_originalValue;
};
