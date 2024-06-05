#pragma once

#include "test_recipe_base.hpp"
#include <cstdint>
#include <vector>
#include <map>

class TestRecipeResnet : public TestRecipeBase
{
public:
    TestRecipeResnet(synDeviceType deviceType, bool isSfg = false);
    ~TestRecipeResnet() {}

    std::vector<size_t> getExternalTensorIds() { return m_tensorExtIdx; }

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

    synStatus generateSfg();

private:
    void compileGraph() override;

    // Not supported
    void _graphCreation() override {};

    bool           m_isSfg;
    const unsigned m_numOfExternalTensors = 3;

    std::vector<size_t> m_tensorExtIdx;
    std::string         m_pathPrefix;

    std::vector<std::string> m_tensorNames;
};