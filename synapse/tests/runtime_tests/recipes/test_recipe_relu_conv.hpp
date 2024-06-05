#pragma once

#include "test_recipe_base.hpp"

class TestRecipeReluConv : public TestRecipeBase
{
public:
    TestRecipeReluConv(synDeviceType deviceType);
    ~TestRecipeReluConv() {}

private:
    void _graphCreation() override;
};