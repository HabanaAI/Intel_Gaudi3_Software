#pragma once

#include "test_recipe_base.hpp"

class TestRecipeHcl : public TestRecipeBase
{
public:
    TestRecipeHcl(synDeviceType deviceType, bool isSfgGraph);
    ~TestRecipeHcl() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;
};