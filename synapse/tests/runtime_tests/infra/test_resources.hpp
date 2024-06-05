#pragma once

#include "../utils/gtest_synapse.hpp"

struct TestConfig;

class TestResources : public ::testing::Environment
{
public:
    TestResources(const TestConfig& rTestConfig);

    void SetUp() override;

    void TearDown() override;

    static const TestConfig& getTestConfig();
};
