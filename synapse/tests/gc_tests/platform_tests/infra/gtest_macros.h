#pragma once

#include "infra/gc_tests_utils.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"

namespace gc_tests
{
static inline std::set<synDeviceType> excludeDevices(const std::set<synDeviceType>&    all,
                                                     const std::vector<synDeviceType>& toExclude)
{
    std::set<synDeviceType> ret = all;
    for (const auto& e : toExclude)
    {
        ret.erase(e);
    }
    return ret;
}

struct PrintToStringParamName
{
    template<class ParamType>
    std::string operator()(const ParamType& info) const
    {
        return synDeviceTypeToString(info.param);
    }
};

struct PrintToStringParamNameWithIndex
{
    template<class ParamType>
    std::string operator()(const ParamType& info) const
    {
        synDeviceType deviceType = std::get<0>(info.param);
        return synDeviceTypeToString(deviceType) + "_" + std::to_string(info.index);
    }
};
}  // namespace gc_tests

#define GC_TEST_P_(test_suite_name, test_name, values, nameGen)                                                        \
    struct test_suite_name##_##test_name : public test_suite_name                                                      \
    {                                                                                                                  \
    };                                                                                                                 \
    INSTANTIATE_TEST_SUITE_P(, test_suite_name##_##test_name, values, nameGen {});                                     \
    TEST_P(test_suite_name##_##test_name, test_name)

#define GC_TEST_F_INC(test_suite_name, test_name, ...)                                                                 \
    GC_TEST_P_(test_suite_name, test_name, ::testing::Values(__VA_ARGS__), PrintToStringParamName)

#define GC_TEST_F_EXC(test_suite_name, test_name, ...)                                                                 \
    GC_TEST_P_(test_suite_name,                                                                                        \
               test_name,                                                                                              \
               ::testing::ValuesIn(excludeDevices(test_suite_name::allDevices(), {__VA_ARGS__})),                      \
               PrintToStringParamName)

#define GC_TEST_F(test_suite_name, test_name)                                                                          \
    GC_TEST_P_(test_suite_name, test_name, ::testing::ValuesIn(test_suite_name::allDevices()), PrintToStringParamName)

#define GC_TEST_P_INC(test_suite_name, test_name, values, ...)                                                         \
    GC_TEST_P_(test_suite_name,                                                                                        \
               test_name,                                                                                              \
               ::testing::Combine(::testing::Values(__VA_ARGS__), values),                                             \
               PrintToStringParamNameWithIndex)

#define GC_TEST_P_EXC(test_suite_name, test_name, values, ...)                                                         \
    GC_TEST_P_(test_suite_name,                                                                                        \
               test_name,                                                                                              \
               ::testing::Combine(::testing::ValuesIn(excludeDevices({__VA_ARGS__})), values),                         \
               PrintToStringParamNameWithIndex)

#define GC_TEST_P(test_suite_name, test_name, values)                                                                  \
    GC_TEST_P_(test_suite_name,                                                                                        \
               test_name,                                                                                              \
               ::testing::Combine(::testing::ValuesIn(test_suite_name::allDevices()), values),                         \
               PrintToStringParamNameWithIndex)
