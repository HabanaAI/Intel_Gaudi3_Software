#pragma once

#include "infra/gc_tests_types.h"
#include "synapse_common_types.h"
#include <gtest/gtest.h>

class SharedResources : public ::testing::Environment
{
public:
    SharedResources(const TestConfig& config);
    void SetUp() override;
    void TearDown() override;

    static const TestConfig& config();
};