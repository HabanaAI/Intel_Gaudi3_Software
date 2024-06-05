#include "infra/gc_test_configuration.h"

#include "graph_compiler/utils.h"
#include "eager/eager_interface.h"
#include "infra/shared_resources.h"

using namespace gc_tests;

TestConfiguration::TestConfiguration():
  m_compilationMode(COMP_GRAPH_MODE_TEST),
  m_testPackage(TEST_PACKAGE_DEFAULT),
  m_numOfTestDevices(1)
{
}

bool TestConfiguration::shouldRunTest(const synDeviceType& testDevice, uint64_t numDevices) const
{
    return isSupportedDeviceForTest(testDevice) && isNumDevicesSufficient(numDevices) &&
           isCompilationModeAlligned(testDevice) &&
           isTestPackageAlligned();
}

bool TestConfiguration::isSupportedDeviceForTest(const synDeviceType& testDevice) const
{
    bool ret = std::find(m_supportedDeviceTypes.begin(), m_supportedDeviceTypes.end(), testDevice) !=
               m_supportedDeviceTypes.end();

    if (!ret)
    {
        m_skipReason =
            fmt::format("Device {} not supported for test, supported devices: {}",
                        toString(testDevice),
                        toString(m_supportedDeviceTypes, ',', [](synDeviceType type) { return toString(type); }));
    }

    return ret;
}

bool TestConfiguration::isNumDevicesSufficient(uint16_t numDevices) const
{
    /*m_numOfTestDevices  - set by env var - default is 1
     _getNumOfDevices()  - can vary from test to test - default is 1
     we want to run test only when - _getNumOfDevices() == m_numOfTestDevices == 1 or
     _getNumOfDevices() != 1 && _getNumOfDevices() <= m_numOfTestDevices*/

    bool ret = numDevices <= m_numOfTestDevices;

    if (!ret)
    {
        m_skipReason =
            fmt::format("Num devices {} is not compliance with test num devices: {}", m_numOfTestDevices, numDevices);
    }

    return ret;
}

bool TestConfiguration::isCompilationModeAlligned(const synDeviceType& testDevice) const
{
    bool ret = (SharedResources::config().m_compilationMode == COMP_BOTH_MODE_TESTS) ||
               (SharedResources::config().m_compilationMode == m_compilationMode);

    ret &= (m_compilationMode != COMP_EAGER_MODE_TEST || eager_mode::isValidForEager(testDevice));

    if (!ret)
    {
        m_skipReason = fmt::format("Compilation mode testing set to {}, and test compilation mode is {}",
                                   SharedResources::config().m_compilationMode,
                                   m_compilationMode);
    }

    return ret;
}

bool TestConfiguration::isTestPackageAlligned() const
{
    if (SharedResources::config().groupIds.empty())
    {
        return true;
    }

    for (auto const& p : SharedResources::config().groupIds)
    {
        if (p == m_testPackage)
        {
            return true;
        }
    }

    m_skipReason = fmt::format("Test package {} isn't enabled", m_testPackage);

    return false;
}
