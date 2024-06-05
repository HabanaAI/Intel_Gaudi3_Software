#pragma once

#include "test_recipe_base.hpp"

class TestRecipeRndRelu : public TestRecipeBase
{
public:
    TestRecipeRndRelu(synDeviceType deviceType);
    ~TestRecipeRndRelu() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    bool read_file(const std::string& file_name, float* output, uint32_t readLength);

private:
    void _graphCreation() override;
};