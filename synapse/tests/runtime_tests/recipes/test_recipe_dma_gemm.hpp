#pragma once

#include "test_recipe_base.hpp"

class TestRecipeDmaGemm : public TestRecipeBase
{
public:
    TestRecipeDmaGemm(synDeviceType deviceType, TSize hDim, TSize bDim);
    ~TestRecipeDmaGemm() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;

    unsigned m_matrixSize;
};