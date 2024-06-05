#include "shared_resources.h"
#include "test_device_manager.h"

static TestConfig sConfig;

SharedResources::SharedResources(const TestConfig& config)
{
    sConfig = config;
}

void SharedResources::SetUp() {}

void SharedResources::TearDown()
{
    TestDeviceManager::instance().reset();
}

const TestConfig& SharedResources::config()
{
    return sConfig;
}
