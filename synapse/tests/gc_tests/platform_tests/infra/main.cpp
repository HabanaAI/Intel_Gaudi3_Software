#include "arg_parse.h"
#include "compiler_types.h"
#include "test_device_manager.h"
#include "hpp/syn_context.hpp"
#include "shared_resources.h"
#include "synapse_common_types.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <system_error>

static synDeviceType deviceTypeFromString(const std::string& deviceName)
{
    if (deviceName == "gaudi") return synDeviceGaudi;
    if (deviceName == "gaudiM") return synDeviceGaudi;
    if (deviceName == "gaudi2") return synDeviceGaudi2;
    if (deviceName == "gaudi3") return synDeviceGaudi3;

    throw std::runtime_error("Invalid device type: " + deviceName);
}

static TestCompilationMode testCompilationModeFromString(const std::string& typeName)
{
    if (typeName == "graph") return TestCompilationMode::COMP_GRAPH_MODE_TEST;
    if (typeName == "eager") return TestCompilationMode::COMP_EAGER_MODE_TEST;
    if (typeName == "any") return TestCompilationMode::COMP_BOTH_MODE_TESTS;

    throw std::runtime_error("Invalid test compilation type: " + typeName);
}

int main(int argc, char** argv)
{
    try
    {
        {
            // creating a context will print synapse info before the tests output
            syn::Context ctx;
        }
        ::testing::InitGoogleTest(&argc, argv);

        ArgParser p;

        auto deviceTypeArg = p.add<std::string>("-c",
                                                "--device-type",
                                                "",
                                                "device type",
                                                {"greco", "gaudi", "gaudiM", "gaudi2", "gaudi3"},
                                                false);
        auto compilationModeArg =
            p.add<std::string>("", "--compilation-mode", "graph", "compilation mode", {"graph", "eager", "any"}, false);
        auto testPackageArg = p.addMulti<uint32_t>("", "--test-packages", "", "test packages ids", {}, false);

        p.parse(argc, argv);

        if (p.helpRequested())
        {
            std::cout << p.getHelp() << std::endl;
            return EXIT_SUCCESS;
        }

        synDeviceType deviceType =
            deviceTypeArg ? deviceTypeFromString(deviceTypeArg.getValue()) : synDeviceType::synDeviceTypeInvalid;

        TestCompilationMode   testCompilationMode = testCompilationModeFromString(compilationModeArg.getValue());
        std::set<TestPackage> groupIds;
        for (const auto& e : testPackageArg.getValues())
        {
            groupIds.insert(static_cast<TestPackage>(e));
        }
        TestConfig config = {std::set<TestPackage> {groupIds.begin(), groupIds.end()}, deviceType, testCompilationMode};

        std::ignore = ::testing::AddGlobalTestEnvironment(new SharedResources(config));  // released by gtest

        return RUN_ALL_TESTS();
    }
    catch (const std::exception& e)
    {
        std::cerr << "gc tests run failed, error: " << e.what();
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "gc tests run failed with unknown error";
        return EXIT_FAILURE;
    }
}
