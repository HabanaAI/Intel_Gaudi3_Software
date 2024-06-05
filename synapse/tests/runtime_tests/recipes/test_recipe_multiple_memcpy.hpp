#pragma once

#include "test_recipe_base.hpp"

class TestRecipeMultipleMemcpy : public TestRecipeBase
{
public:
    TestRecipeMultipleMemcpy(synDeviceType deviceType, unsigned numbOfNodes);

    virtual ~TestRecipeMultipleMemcpy() {};

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;

    unsigned m_numNodes;
};