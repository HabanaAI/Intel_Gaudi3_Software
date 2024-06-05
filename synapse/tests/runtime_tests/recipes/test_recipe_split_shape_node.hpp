#pragma once

#include "test_recipe_base.hpp"

class TestRecipeSplitShapeNode : public TestRecipeBase
{
public:
    TestRecipeSplitShapeNode(synDeviceType deviceType);

    virtual ~TestRecipeSplitShapeNode() = default;

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;
};