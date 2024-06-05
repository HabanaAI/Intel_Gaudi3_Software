#pragma once

#include "hpp/synapse.hpp"
#include <gtest/gtest.h>

class SynBaseTest : public ::testing::Test
{
public:
    SynBaseTest() {}
    virtual ~SynBaseTest() = default;

protected:
    syn::Context m_ctx;
};

class SynTypedDeviceTest : public SynBaseTest
{
public:
    SynTypedDeviceTest()
    {
        const char* deviceType = getenv("SYN_DEVICE_TYPE");

        m_deviceType = deviceType == nullptr ? synDeviceGaudi : synDeviceType(std::stoi(deviceType));
    }
    virtual ~SynTypedDeviceTest() = default;

    syn::Device tryAcquireDevice(const std::set<synDeviceType>& deviceTypes)
    {
        for (const auto& t : deviceTypes)
        {
            if (m_ctx.getDeviceCount(t) == 0) continue;
            try
            {
                return m_ctx.acquire(t);
            }
            catch (...)
            {
                // failed to acquire
            }
        }
        return syn::Device();
    }

    virtual syn::Device acquireDevice()
    {
        syn::Device dev = tryAcquireDevice({m_deviceType});
        if (dev) return dev;
        dev = tryAcquireDevice(m_optionalDeviceTypes);
        if (dev) return dev;
        throw std::runtime_error("failed to acquire device");
    }

protected:
    synDeviceType           m_deviceType;
    std::set<synDeviceType> m_optionalDeviceTypes;
};