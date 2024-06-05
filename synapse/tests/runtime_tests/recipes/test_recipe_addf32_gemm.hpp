#pragma once

#include "test_recipe_base.hpp"

class TestRecipeAddf32Gemm : public TestRecipeBase
{
public:
    TestRecipeAddf32Gemm(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode, std::string const& uniqueRecipeName = "");
    ~TestRecipeAddf32Gemm() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

protected:
    void _graphCreation() override;

    const std::vector<TSize> m_sizes;
};

class TestRecipeAddf32GemmSections : public TestRecipeAddf32Gemm
{
public:
    TestRecipeAddf32GemmSections(synDeviceType deviceType, std::vector<TSize> const& sizes, bool eagerMode)
        : TestRecipeAddf32Gemm(deviceType, sizes, eagerMode, makeUniqueRecipeName<TestRecipeAddf32GemmSections>(eagerMode ? "eager" : "graph", sizes))
        {}
    ~TestRecipeAddf32GemmSections() {}

protected:
    void _graphCreation() override;
    void _destroyGraphHandle() override;

    synSectionHandle m_testSection;
};