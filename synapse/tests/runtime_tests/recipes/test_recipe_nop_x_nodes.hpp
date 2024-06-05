#pragma once

#include "test_recipe_base.hpp"

class TestRecipeNopXNodes : public TestRecipeBase
{
public:
    TestRecipeNopXNodes(synDeviceType deviceType, unsigned nodes = 1);
    ~TestRecipeNopXNodes() {}

    void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void     _graphCreation() override;
    unsigned m_nodes;
};