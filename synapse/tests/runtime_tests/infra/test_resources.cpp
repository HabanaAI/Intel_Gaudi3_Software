#include "test_resources.hpp"
#include "test_config.hpp"

static TestConfig sConfig;

TestResources::TestResources(const TestConfig& rTestConfig)
{
    sConfig = rTestConfig;
}

void TestResources::SetUp() {}

void TestResources::TearDown() {}

const TestConfig& TestResources::getTestConfig()
{
    return sConfig;
}
