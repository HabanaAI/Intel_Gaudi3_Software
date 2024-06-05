#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <system_error>
#include "filesystem.h"

#include "argparser.hpp"

#include "../infra/test_types.hpp"
#include "syn_test_filter_factory.hpp"
#include "test_recipe_interface.hpp"
#include "test_config.hpp"
#include "test_resources.hpp"

static synDeviceType deviceTypeFromString(const std::string& deviceName)
{
    if (deviceName == "gaudi") return synDeviceGaudi;
    if (deviceName == "gaudiM") return synDeviceGaudi;
    if (deviceName == "gaudi2") return synDeviceGaudi2;
    if (deviceName == "gaudi3") return synDeviceGaudi3;
    throw std::runtime_error("Invalid device type: " + deviceName);
}

class ThrowListener : public testing::EmptyTestEventListener
{
    void OnTestPartResult(const testing::TestPartResult& result) override
    {
        if (result.type() == testing::TestPartResult::kFatalFailure)
        {
            throw testing::AssertionException(result);
        }
    }
};
int main(int argc, char** argv)
{
    try
    {
        InputParser inputParser(argc, argv);

        if (inputParser.cmdOptionExists("-h"))
        {
            std::cout << "HELP!! ¯\\_(ツ)_/¯" << std::endl;
        }

        // First validate all test suites are registered to any package
        std::vector<std::string> nonRegisteredTestSuites;
        for (auto& suite : SynTestFilterFactory::getSuiteRegistrationMap())
        {
            if (suite.second == false)
            {
                nonRegisteredTestSuites.push_back(suite.first);
            }
        }
        if (nonRegisteredTestSuites.size() > 0)
        {
            std::cout << "Test suites: " << std::endl;
            for (auto& suite : nonRegisteredTestSuites)
            {
                std::cout << suite << std::endl;
            }
            std::cout << "are not registered to any package, please use REGISTER_SUITE macro" << std::endl;
            return EXIT_FAILURE;
        }

        // Global test configuration created
        TestConfig testConfig;

        const std::vector<std::string> deviceTypes = inputParser.getCmdOption("--device-type");
        if (!deviceTypes.empty())
        {
            synDeviceType deviceType = deviceTypeFromString(deviceTypes[0]);
            testConfig.deviceType    = deviceType;
        }
        else
        {
            char* pChar = getenv("SYN_DEVICE_TYPE");

            if (pChar != nullptr)
            {
                synDeviceType deviceType = (synDeviceType)std::stoi(pChar);
                testConfig.deviceType    = deviceType;
            }
        }

        std::vector<std::string> testPackages = inputParser.getCmdOption("--test-packages");
        std::vector<std::string> exPackages   = inputParser.getCmdOption("--ex-packages");
        std::string              gtestFilter  = SynTestFilterFactory::buildFilter(testPackages, exPackages);
        if (!testPackages.empty())
        {
            for (std::string pkgName : testPackages)
            {
                testConfig.includedTestPackages.push_back(SynTestFilterFactory::getPackageFromName(pkgName));
            }
        }
        if (!exPackages.empty())
        {
            for (std::string pkgName : exPackages)
            {
                testConfig.excludedTestPackages.push_back(SynTestFilterFactory::getPackageFromName(pkgName));
            }
        }

        std::vector<std::string> specificTests = inputParser.getCmdOptionEQ("--gtest_filter");
        if (!specificTests.empty() && (gtestFilter != ""))
        {
            // --gtest_filter overrides any packages specified
            //   otherwise you may have contradictions, so don't use them both
            std::cout << "specific --gtest_filter was found. ignoring packages translation of " << gtestFilter
                      << std::endl;
            gtestFilter = "";
        }

        // Global test configuration set.
        // All additions to config should be added before this line.
        std::ignore = ::testing::AddGlobalTestEnvironment(new TestResources(testConfig));  // released by gtest

        // Add the built filter to arguments list
        int                argNum = argc + 1;
        std::vector<char*> gtestArgs(argv, argv + argc);
        gtestArgs.push_back(gtestFilter.data());

        if (!inputParser.cmdOptionExists("--keep-resources"))
        {
            std::cout << "Clearing resource files" << std::endl;
            TestRecipeInterface::clearResourceFiles();
        }

        // Pass updated arguments to gtest
        fs::create_directory(TEST_RESOURCE_PATH);
        // make ASSERT_... throw exceptions.
        // this way we can ASSET_ in any nested function
        testing::UnitTest::GetInstance()->listeners().Append(new ThrowListener);
        ::testing::InitGoogleTest(&argNum, gtestArgs.data());
        return RUN_ALL_TESTS();
    }
    catch (const std::exception& e)
    {
        std::cerr << "synapse tests run failed, error: " << e.what();
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "synapse tests run failed with unknown error";
        return EXIT_FAILURE;
    }
}