#pragma once

#include "base_test.h"

namespace json_tests
{
class RecipeRunner : public RunTypedDeviceTest
{
public:
    RecipeRunner(const ArgParser& args);

    void run() override;

private:
    void runTest(bool run);

protected:
    std::string m_recipeFilePath;
};
}  // namespace json_tests