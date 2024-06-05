#pragma once

#include "test_recipe_base.hpp"
#include <cstdint>
#include <vector>

class TestRecipeAssertAsync : public TestRecipeBase
{
public:
    TestRecipeAssertAsync(synDeviceType deviceType);
    ~TestRecipeAssertAsync() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;
};