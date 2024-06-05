#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "test_resources.hpp"
#include "test_config.hpp"

const synDeviceType SynBaseTest::INVALID_DEVICE_TYPE = synDeviceTypeInvalid;

SynBaseTest::SynBaseTest() : mTestConfig(TestResources::getTestConfig()), m_deviceType(INVALID_DEVICE_TYPE) {}

SynBaseTest::~SynBaseTest() {}

void SynBaseTest::afterSynInitialize() {}

void SynBaseTest::SetUp()
{
    LOG_INFO(SYN_RT_TEST, "Running test: {}", getTestName());
    synStatus status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "Failed to synInitialize";

    afterSynInitialize();

    synDeviceType deviceType = getDeviceType();
    ASSERT_NE(deviceType, INVALID_DEVICE_TYPE) << "getDeviceType failed";

    if (!isSupportedDeviceTypeForTest(deviceType))
    {
        GTEST_SKIP() << "Not supported deviceType " << deviceType;
    }

    m_deviceType = deviceType;
}

void SynBaseTest::TearDown()
{
    synStatus status = synDestroy();
    ASSERT_EQ(status, synSuccess) << "Failed to synDestroy";
}

std::string SynBaseTest::getTestName()
{
    std::string testName(::testing::UnitTest::GetInstance()->current_test_info()->name());

    // remove / char in case it appears in string
    std::string toReplace   = "/";
    std::string replaceWith = "_";
    std::size_t pos         = testName.find(toReplace);
    if (pos != std::string::npos)
    {
        testName = testName.replace(pos, toReplace.length(), replaceWith);
    }
    return testName;
}

std::string SynBaseTest::getTestSuiteName()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
}

void SynBaseTest::invokeMultiThread(unsigned nbThreads, unsigned nbIterations, SynBaseTest::TestFunc testFunc)
{
    auto threadFunc = [&](unsigned threadId) {
        for (unsigned i = 0; i < nbIterations; ++i)
        {
            testFunc();
        }
    };

    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        threads.emplace_back(threadFunc, i);
    }

    for (auto& t : threads)
    {
        t.join();
    }
}

void SynBaseTest::invokeMultiThread(unsigned                  nbThreads,
                                    std::chrono::milliseconds maxDuration,
                                    SynBaseTest::TestFunc     testFunc)
{
    auto threadFunc = [&](unsigned threadId) {
        auto start = std::chrono::steady_clock::now();

        while (1)
        {
            testFunc();
            auto finish = std::chrono::steady_clock::now();

            if (finish - start >= maxDuration)
            {
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        threads.emplace_back(threadFunc, i);
    }

    for (auto& t : threads)
    {
        t.join();
    }
}

synDeviceType SynBaseTest::getDeviceType() const
{
    // Check if there's an explicit device type from user test command
    // If there is - do getDeviceByType, and either work with it or fail
    // If there isn't - check for deviceCount on every device until you stumble upon a non-zero
    // Then use that device
    if (!mTestConfig.deviceType.has_value())
    {
        unsigned deviceIndex = synDeviceGaudi;
        for (; deviceIndex < synDeviceTypeSize; ++deviceIndex)
        {
            uint32_t  count;
            synStatus status = synDeviceGetCountByDeviceType(&count, (synDeviceType)deviceIndex);
            if (status != synSuccess)
            {
                return INVALID_DEVICE_TYPE;
            }

            if (count > 0)
            {
                break;
            }
        }

        if (deviceIndex == synDeviceTypeSize)
        {
            return INVALID_DEVICE_TYPE;
        }

        return (synDeviceType)deviceIndex;
    }
    else
    {
        return mTestConfig.deviceType.value();
    }
}

void SynBaseTest::setSupportedDevices(std::initializer_list<synDeviceType> deviceTypes)
{
    m_supportedDeviceTypes.clear();
    for (auto deviceType : deviceTypes)
    {
        m_supportedDeviceTypes.push_back(deviceType);
    }
}

bool SynBaseTest::isSupportedDeviceTypeForTest(synDeviceType deviceType) const
{
    bool isSupported = std::find(m_supportedDeviceTypes.begin(), m_supportedDeviceTypes.end(), deviceType) !=
                       m_supportedDeviceTypes.end();

    return isSupported;
}

void SynBaseTest::setSupportedPackages(std::initializer_list<synTestPackage> testPackages)
{
    m_supportedTestPackages.clear();
    for (auto testPackage : testPackages)
    {
        m_supportedTestPackages.push_back(testPackage);
    }
}

bool SynBaseTest::shouldRunTest()
{
    if (mTestConfig.includedTestPackages.size() == 0 && mTestConfig.excludedTestPackages.size() == 0)
    {
        LOG_TRACE(SYN_RT_TEST, "No test packages provided");
        return true;
    }

    std::vector<synTestPackage>* suitePackageList {};
    if (m_supportedTestPackages.empty())
    {
        std::string testSuiteName = this->getTestSuiteName();
        suitePackageList          = &SynTestFilterFactory::getSuiteDefaultPackages(testSuiteName);
    }
    else
    {
        suitePackageList = &m_supportedTestPackages;
    }

    bool shouldRun = true;

    if (mTestConfig.includedTestPackages.size() != 0)
    {
        shouldRun = false;
        for (synTestPackage pkg : mTestConfig.includedTestPackages)
        {
            if (std::find(suitePackageList->begin(), suitePackageList->end(), pkg) != suitePackageList->end())
            {
                shouldRun = true;
            }
        }
    }

    if (mTestConfig.excludedTestPackages.size() != 0)
    {
        for (synTestPackage pkg : mTestConfig.excludedTestPackages)
        {
            if (std::find(suitePackageList->begin(), suitePackageList->end(), pkg) != suitePackageList->end())
            {
                shouldRun = false;
                LOG_DEBUG(SYN_RT_TEST, "Skipped test due to excluded package");
            }
        }
    }

    return shouldRun;
}
