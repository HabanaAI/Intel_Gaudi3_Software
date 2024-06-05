#pragma once

#include "test_recipe_base.hpp"

class TestRecipeTpc : public TestRecipeBase
{
public:
    TestRecipeTpc(synDeviceType deviceType);
    ~TestRecipeTpc() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;
};