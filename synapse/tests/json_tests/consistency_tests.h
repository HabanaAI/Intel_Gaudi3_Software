#pragma once

#include "base_test.h"
#include <optional>

namespace json_tests
{

class ConsistencyTest
: public TypedDeviceTest
, public JsonTest
{
public:
    ConsistencyTest(const ArgParser& args);
    void run() override;

private:
    void        setup();
    bool        checkConsistency(const size_t graphIndex);
    syn::Recipe compileGraph(const size_t graphIndex);

    int m_testIterations;
    bool m_keepGoing;
};

}  // namespace json_tests