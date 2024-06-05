#pragma once

#include "test_device_manager.h"
#include "hpp/synapse.hpp"
#include "infra/gc_tests_types.h"
#include <gtest/gtest.h>

namespace gc_tests
{
class EnvSet
{
public:
    EnvSet();
    ~EnvSet();

private:
    std::optional<std::string> m_experimentalOrigVal;
};

class SynBaseTest : public ::testing::Test
{
public:
    SynBaseTest();
    virtual ~SynBaseTest() = default;

    void setTestPackage(TestPackage testPackage);

    void SetUp() override;
    void TearDown() override;

    bool isCompilationModeAlligned() const;

protected:
    std::string         m_testName;
    EnvSet              m_envSet;
    syn::Context        m_ctx;
    TestConfig          m_testConfig;
    TestPackage         m_testPackage;
    TestCompilationMode m_compilationMode;
};

class SynTrainingBaseTest : public SynBaseTest
{
public:
    SynTrainingBaseTest(synDeviceType deviceType);
    static const std::set<synDeviceType>& allDevices();

protected:
    synDeviceType m_deviceType;
};

class SynTrainingRunBaseTest : public SynTrainingBaseTest
{
public:
    SynTrainingRunBaseTest(synDeviceType deviceType);

    void SetUp() override;

    bool isSupportedDeviceForTest() const;

    synDeviceType m_requestedDeviceType;
    syn::Device   m_device;

private:
    std::shared_ptr<TestDeviceManager::DeviceReleaser> m_deviceReleaser;
};

class SynTrainingCompileTest
: public SynTrainingBaseTest
, public testing::WithParamInterface<synDeviceType>
{
public:
    SynTrainingCompileTest();
};

class SynTrainingRunTest
: public SynTrainingRunBaseTest
, public testing::WithParamInterface<synDeviceType>
{
public:
    SynTrainingRunTest();
};

template<typename T>
class SynWithParamInterface
: public SynTrainingBaseTest
, public testing::WithParamInterface<std::tuple<synDeviceType, T>>
{
public:
    SynWithParamInterface() : SynTrainingBaseTest(std::get<0>(SynWithParamInterface::GetParam())) {}
};

template<typename T>
class SynTrainingRunParamTest
: public SynTrainingRunBaseTest
, public testing::WithParamInterface<std::tuple<synDeviceType, T>>
{
public:
    SynTrainingRunParamTest() : SynTrainingRunBaseTest(std::get<0>(SynTrainingRunParamTest::GetParam())) {}
};
}  // namespace gc_tests