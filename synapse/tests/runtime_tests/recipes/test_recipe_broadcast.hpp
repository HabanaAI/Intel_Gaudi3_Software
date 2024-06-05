#pragma once

#include "test_recipe_base.hpp"
#include <array>

using TestSizes = std::array<TSize, SYN_MAX_TENSOR_DIM>;

class TestRecipeBroadcast : public TestRecipeBase
{
public:
    TestRecipeBroadcast(synDeviceType deviceType,
                        TestSizes     inSize,
                        TestSizes     outSize,
                        uint32_t      dims,
                        bool          isFcdMultipleDimsBroadcast);
    ~TestRecipeBroadcast() {}

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const override;

private:
    void _graphCreation() override;

    bool m_isFcdMultipleDimsBroadcast = false;
};