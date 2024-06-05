#pragma once

#include "test_recipe_base.hpp"
#include <cstdint>
#include <vector>

class TestRecipeAddf32 : public TestRecipeBase
{
public:
    TestRecipeAddf32(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode = false);

    TestRecipeAddf32(TestRecipeAddf32&& other) noexcept : TestRecipeBase(std::move(other)), m_sizes(other.m_sizes) {}

    ~TestRecipeAddf32() = default;

    bool read_file(const std::string& file_name, float* output, uint32_t num_of_elements);

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    void _graphCreation() override;

private:
    std::vector<TSize> m_sizes;
};