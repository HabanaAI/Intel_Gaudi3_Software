#pragma once

#include "synapse_common_types.h"
#include <cstdint>
#include <optional>
#include <set>
#include "utils/test_config_types.h"

struct TestConfig
{
    std::set<TestPackage>        groupIds;
    std::optional<synDeviceType> deviceType;
    TestCompilationMode          m_compilationMode;
};