#pragma once

#include "test_recipe_base.hpp"
#include <cstdint>
#include <vector>
#include <memory>

class TestRecipeDsdDma : public TestRecipeBase
{
public:
    TestRecipeDsdDma(synDeviceType        deviceType,
                     std::array<TSize, 4> inputMax,
                     std::array<TSize, 4> inputMin,
                     std::array<TSize, 4> outputMax,
                     std::array<TSize, 4> outputMin);
    ~TestRecipeDsdDma() = default;

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    void setExecutionDynamicSize(unsigned op1ActualSize) { m_op1ActualSize = op1ActualSize; };

private:
    void _graphCreation() override;

    unsigned m_op1ActualSize = 0;
};