#pragma once

#include "test_recipe_base.hpp"
#include <cstdint>
#include <vector>
#include <memory>

class TestRecipeDma : public TestRecipeBase
{
public:
    TestRecipeDma(synDeviceType deviceType,
                  TSize         zDim,
                  TSize         wDim,
                  float         initValue,
                  bool          isConstTensor,
                  synDataType   dataType,
                  unsigned      threadIndex    = 0,
                  unsigned      iterationIndex = 0);
    ~TestRecipeDma() = default;

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;
    void         _graphCreation() override;

private:
    std::string m_outputName;
    std::string m_inputName;

    float m_constInputTensorInitVal;
};