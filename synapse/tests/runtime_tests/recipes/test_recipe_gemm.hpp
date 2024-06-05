#pragma once

#include "test_recipe_base.hpp"

class TestRecipeGemm : public TestRecipeBase
{
public:
    TestRecipeGemm(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode = false);
    ~TestRecipeGemm() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    synNodeId getUniqueNodeId() const { return m_uniqueNodeId; }

private:
    void _graphCreation() override;

    const std::vector<TSize> m_sizes;
    synNodeId                m_uniqueNodeId;
};