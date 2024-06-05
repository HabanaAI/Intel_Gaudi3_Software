#pragma once

#include "test_recipe_base.hpp"

class TestRecipeDsdGemm : public TestRecipeBase
{
public:
    // Op1 - H may be dynamic according to isOp1Dynamic
    // Op2 - W is dynamic
    TestRecipeDsdGemm(synDeviceType        deviceType,
                      bool                 isDynamic,
                      bool                 isSharedInputSection,
                      std::array<TSize, 2> op1Max,
                      std::array<TSize, 2> op1Min,
                      std::array<TSize, 2> op2Max,
                      std::array<TSize, 2> op2Min,
                      std::array<TSize, 2> outMax,
                      std::array<TSize, 2> outMin);

    TestRecipeDsdGemm(TestRecipeDsdGemm&& other) noexcept
    : TestRecipeBase(std::move(other)),
      m_op1ActualSize(other.m_op1ActualSize),
      m_op2ActualSize(other.m_op2ActualSize),
      m_gemmParams(other.m_gemmParams)
    {
    }

    ~TestRecipeDsdGemm() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    void setExecutionDynamicSize(unsigned op1ActualSize, unsigned op2ActualSize)
    {
        m_op1ActualSize = op1ActualSize;
        m_op2ActualSize = op2ActualSize;
    };

    unsigned getActualSize(bool isOp1) { return isOp1 ? m_op1ActualSize : m_op2ActualSize; };

private:
    void _graphCreation() override;

    unsigned m_op1ActualSize = 0;
    unsigned m_op2ActualSize = 0;

    synGEMMParams m_gemmParams;
};