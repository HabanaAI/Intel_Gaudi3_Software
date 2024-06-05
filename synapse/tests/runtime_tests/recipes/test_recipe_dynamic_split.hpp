#pragma once

#include "test_recipe_base.hpp"

#include "synapse_api_types.h"

class TestRecipeDynamicSplit : public TestRecipeBase
{
public:
    TestRecipeDynamicSplit(synDeviceType deviceType);

    ~TestRecipeDynamicSplit() = default;

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    synNodeId getUniqueNodeId() const { return m_uniqueNodeId; }

private:
    void _graphCreation() override;

    synNodeId m_uniqueNodeId;
};