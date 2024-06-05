#include "infra/gc_base_test.h"
#include "infra/gtest_macros.h"
#include "synapse_common_types.h"
#include "gtest/gtest-param-test.h"
#include <string>

using namespace gc_tests;

GC_TEST_F(SynTrainingCompileTest, DISABLED_all_devices)
{
    syn::Graph invalidGraph;
    EXPECT_FALSE(invalidGraph);
    syn::Graph graph = m_ctx.createGraph(m_deviceType);
    EXPECT_TRUE(graph);
}

GC_TEST_F_INC(SynTrainingCompileTest, DISABLED_include_specific_devices, synDeviceGaudi2)
{
    syn::EagerGraph invalidGraph;
    EXPECT_FALSE(invalidGraph);
    syn::EagerGraph graph = m_ctx.createEagerGraph(m_deviceType);
    EXPECT_TRUE(graph);
}

GC_TEST_F_EXC(SynTrainingCompileTest, DISABLED_exclude_specific_devices, synDeviceGaudi, synDeviceGaudi3)
{
    syn::EagerGraph invalidGraph;
    EXPECT_FALSE(invalidGraph);
    syn::EagerGraph graph = m_ctx.createEagerGraph(m_deviceType);
    EXPECT_TRUE(graph);
}

class SynTrainingCompileParamExampleTest : public SynWithParamInterface<std::tuple<int, std::string>>
{
};

auto params = ::testing::Values(std::make_tuple<int, std::string>(1, "1"),
                                std::make_tuple<int, std::string>(2, "2"),
                                std::make_tuple<int, std::string>(100, "100"));

GC_TEST_P(SynTrainingCompileParamExampleTest, DISABLED_parmeterized_test, params)
{
    auto              params = std::get<1>(GetParam());  // index 0 is saved for device type
    int               intVal = std::get<0>(params);
    const std::string strVal = std::get<1>(params);
    EXPECT_EQ(std::to_string(intVal), strVal);
}

class SynTrainingRunParamExampleTest : public SynTrainingRunParamTest<std::tuple<int, std::string>>
{
};

GC_TEST_P_INC(SynTrainingRunParamExampleTest, DISABLED_parmeterized_test1, params, synDeviceGaudi)
{
    auto              params = std::get<1>(GetParam());  // index 0 is saved for device type
    int               intVal = std::get<0>(params);
    const std::string strVal = std::get<1>(params);
    EXPECT_EQ(std::to_string(intVal), strVal);
}
