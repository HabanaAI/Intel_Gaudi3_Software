#pragma once

#include "synapse_common_types.h"
#include "../infra/test_types.hpp"

struct TestConfig
{
    std::optional<synDeviceType> deviceType;
    std::vector<synTestPackage>  includedTestPackages;
    std::vector<synTestPackage>  excludedTestPackages;
};
