#pragma once

#include "test_recipe_base.hpp"

class TestRecipeGemmAddf32 : public TestRecipeBase
{
public:
    TestRecipeGemmAddf32(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode);
    ~TestRecipeGemmAddf32() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;

    const std::vector<TSize> m_sizes;
};