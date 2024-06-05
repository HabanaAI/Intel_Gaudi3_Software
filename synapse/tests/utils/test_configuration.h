#pragma once

#include <vector>

#include "synapse_common_types.h"
#include "supported_devices_macros.h"
#include "test_config_types.h"

class TestConfiguration
{
public:
    TestConfiguration();
    bool shouldRunTest(const synDeviceType& testDevice, uint64_t numDevices) const;
    const std::string& skipReason() const { return m_skipReason; }

    std::vector<synDeviceType> m_supportedDeviceTypes;
    TestCompilationMode        m_compilationMode;
    TestPackage                m_testPackage;
    uint16_t                   m_numOfTestDevices;

private:
    bool isSupportedDeviceForTest(const synDeviceType& testDevice) const;
    bool isNumDevicesSufficient(uint16_t numDevices) const;
    bool isCompilationModeAlligned(const synDeviceType& testDevice) const;
    bool isTestPackageAlligned() const;
    mutable std::string m_skipReason;
};
