#pragma once

#include "base_test.h"

namespace json_tests
{
class ModelTest : public BaseTest
{
public:
    ModelTest(const ArgParser& args);

    void run() override;

protected:
    std::string m_comObjFilePath;
    bool        m_compile;
    bool        m_run;
};
}  // namespace json_tests