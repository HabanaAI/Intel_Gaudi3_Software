#pragma once

#include "test_recipe_base.hpp"

class TestRecipeRelu : public TestRecipeBase
{
public:
    TestRecipeRelu(synDeviceType deviceType);
    ~TestRecipeRelu() {}

    void         validateResultsWithFile(const LaunchTensorMemory& rLaunchTensorMemory) const;
    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    bool read_file(const std::string& file_name, float* output, uint32_t readLength) const;

    std::string getPathPrefix() { return m_pathPrefix; }

private:
    void _graphCreation() override;

    std::string m_pathPrefix = "";
};