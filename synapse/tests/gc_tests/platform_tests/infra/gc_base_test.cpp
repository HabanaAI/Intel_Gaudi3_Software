#include "gc_base_test.h"
#include "gtest/gtest.h"
#include "shared_resources.h"
#include "synapse_common_types.h"
#include "test_device_manager.h"
#include "utils.h"

static constexpr const char* EXPERIMENTAL_STR = "ENABLE_EXPERIMENTAL_FLAGS";
namespace gc_tests
{
EnvSet::EnvSet()
{
    const char* experimentalVal = getenv(EXPERIMENTAL_STR);

    if (!experimentalVal)
    {
        setenv(EXPERIMENTAL_STR, "1", true);
        m_experimentalOrigVal = "0";  // indicate that env var was created by the test
    }
}

EnvSet::~EnvSet()
{  // Remove the env var when it was created by the test
    if (m_experimentalOrigVal.has_value())
    {
        unsetenv(EXPERIMENTAL_STR);
    }
}

SynBaseTest::SynBaseTest()
: m_testName(sanitizeFileName(::testing::UnitTest::GetInstance()->current_test_info()->name())),
  m_testConfig(SharedResources::config()),
  m_testPackage(TEST_PACKAGE_DEFAULT),
  m_compilationMode(COMP_GRAPH_MODE_TEST)
{
}

void SynBaseTest::setTestPackage(TestPackage testPackage)
{
    m_testPackage = testPackage;
}

void SynBaseTest::SetUp()
{
    if (!m_testConfig.groupIds.empty() && m_testConfig.groupIds.find(m_testPackage) == m_testConfig.groupIds.end())
    {
        GTEST_SKIP() << fmt::format("Test package {} isn't enabled", static_cast<uint32_t>(m_testPackage));
    }
    if (!isCompilationModeAlligned())
    {
        GTEST_SKIP() << fmt::format("Compilation mode testing set to {}, and test compilation mode is {}",
                                    m_testConfig.m_compilationMode,
                                    m_compilationMode);
    }
}

void SynBaseTest::TearDown() {}

bool SynBaseTest::isCompilationModeAlligned() const
{
    bool ret = (m_testConfig.m_compilationMode == COMP_BOTH_MODE_TESTS) ||
               (m_testConfig.m_compilationMode == m_compilationMode);

    ret &= (m_testConfig.m_compilationMode != COMP_EAGER_MODE_TEST);

    return ret;
}

SynTrainingBaseTest::SynTrainingBaseTest(synDeviceType deviceType) : m_deviceType(deviceType) {}

const std::set<synDeviceType>& SynTrainingBaseTest::allDevices()
{
    static const std::set<synDeviceType> DEVICES {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3};
    return DEVICES;
}

SynTrainingRunBaseTest::SynTrainingRunBaseTest(synDeviceType deviceType)
: SynTrainingBaseTest(deviceType),
  m_requestedDeviceType(this->m_testConfig.deviceType.value()),
  m_deviceReleaser(TestDeviceManager::instance().acquireDevice(m_requestedDeviceType))
{
    m_device = m_deviceReleaser->dev();
}

void SynTrainingRunBaseTest::SetUp()
{
    if (!isSupportedDeviceForTest())
    {
        GTEST_SKIP() << fmt::format("Run device {} is different from test device type {}",
                                    toString(m_requestedDeviceType),
                                    toString(m_deviceType));
    }
}

bool SynTrainingRunBaseTest::isSupportedDeviceForTest() const
{
    return m_deviceType == m_requestedDeviceType;
}

SynTrainingCompileTest::SynTrainingCompileTest() : SynTrainingBaseTest(GetParam()) {}

SynTrainingRunTest::SynTrainingRunTest() : SynTrainingRunBaseTest(GetParam()) {}
}  // namespace gc_tests