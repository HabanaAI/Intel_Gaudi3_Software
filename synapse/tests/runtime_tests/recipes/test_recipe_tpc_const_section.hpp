#pragma once

#include "test_host_buffer.hpp"
#include "test_recipe_base.hpp"

class TestRecipeTpcConstSection : public TestRecipeBase
{
public:
    TestRecipeTpcConstSection(synDeviceType deviceType);
    ~TestRecipeTpcConstSection() {}

    virtual void validateResults(const LaunchTensorMemory&    rLaunchTensorMemory,
                                 const std::vector<uint64_t>& rConstSectionsHostBuffers) const override;

    void _graphCreation() override;

    void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;
};