#pragma once

#include "../utils/gtest_synapse.hpp"
#include "../infra/test_types.hpp"
#include "package_support_macros.h"
#include "syn_test_filter_factory.hpp"

struct TestConfig;

class SynBaseTest : public ::testing::Test
{
public:
    SynBaseTest();

    ~SynBaseTest() override;

    void SetUp() override;
    void TearDown() override;

protected:
    virtual void afterSynInitialize();

    static std::string getTestName();

    static std::string getTestSuiteName();

    typedef std::function<void()> TestFunc;

    void invokeMultiThread(unsigned nbThreads, unsigned nbIterations, TestFunc testFunc);

    void invokeMultiThread(unsigned nbThreads, std::chrono::milliseconds maxDuration, TestFunc testFunc);

    synDeviceType getDeviceType() const;

    void setSupportedDevices(std::initializer_list<synDeviceType> deviceTypes);

    bool isSupportedDeviceTypeForTest(synDeviceType deviceType) const;

    void setSupportedPackages(std::initializer_list<synTestPackage> testPackages);

    bool shouldRunTest();

    static const synDeviceType  INVALID_DEVICE_TYPE;
    const TestConfig&           mTestConfig;
    synDeviceType               m_deviceType;
    std::vector<synDeviceType>  m_supportedDeviceTypes;
    std::vector<synTestPackage> m_supportedTestPackages;
};
