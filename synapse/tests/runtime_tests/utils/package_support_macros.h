#pragma once

#include "../utils/gtest_synapse.hpp"
#include <gtest/gtest-typed-test.h>
#include <variant>

#ifndef GTEST_DISALLOW_ASSIGN_
// A macro to disallow copy operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_ASSIGN_(type) type& operator=(type const&) = delete
#endif

#ifndef GTEST_DISALLOW_COPY_AND_ASSIGN_
// A macro to disallow copy constructor and operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type)                                                                          \
    type(type const&) = delete;                                                                                        \
    GTEST_DISALLOW_ASSIGN_(type)
#endif

// unfortunately a templated class as a test base breaks CLion gtests support
// that's why instaed of a class we inject members directly
#define TEST_CONFIGURATION_SUPPORT_MEMBERS(...)                                                                        \
                                                                                                                       \
    virtual void TestBody() override                                                                                   \
    {                                                                                                                  \
        if (!this->shouldRunTest())                                                                                    \
        {                                                                                                              \
            GTEST_SKIP();                                                                                              \
        }                                                                                                              \
        TestBodyInternal(0, 0);                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    void TestBodyInternal(unsigned _threadIdx, unsigned _deviceIdx);                                                   \
                                                                                                                       \
protected:                                                                                                             \
    void _setTestConfiguration() {}                                                                                    \
    void _setTestConfiguration(std::initializer_list<synTestPackage> supportedTestPackages)                            \
    {                                                                                                                  \
        this->setSupportedPackages(supportedTestPackages);                                                             \
    }                                                                                                                  \
    void _setTestConfiguration(std::initializer_list<synDeviceType> supportedDevices)                                  \
    {                                                                                                                  \
        this->setSupportedDevices(supportedDevices);                                                                   \
    }                                                                                                                  \
    void _setTestConfiguration(std::initializer_list<synTestPackage> supportedTestPackages,                            \
                               std::initializer_list<synDeviceType>  supportedDevices)                                 \
    {                                                                                                                  \
        this->setSupportedPackages(supportedTestPackages);                                                             \
        this->setSupportedDevices(supportedDevices);                                                                   \
    }                                                                                                                  \
    void SetUp() override                                                                                              \
    {                                                                                                                  \
        _setTestConfiguration(__VA_ARGS__);                                                                            \
        base_class::SetUp();                                                                                           \
    }

#define GTEST_TEST_SYN_(test_case_name, test_name, parent_class, parent_id, ...)                                       \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class                                      \
    {                                                                                                                  \
        using base_class = parent_class;                                                                               \
        TEST_CONFIGURATION_SUPPORT_MEMBERS(__VA_ARGS__)                                                                \
    public:                                                                                                            \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                                                         \
                                                                                                                       \
    private:                                                                                                           \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;                                          \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name, test_name));                            \
    };                                                                                                                 \
                                                                                                                       \
    ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_info_ =                         \
        ::testing::internal::MakeAndRegisterTestInfo(                                                                  \
            #test_case_name,                                                                                           \
            #test_name,                                                                                                \
            NULL,                                                                                                      \
            NULL,                                                                                                      \
            ::testing::internal::CodeLocation(__FILE__, __LINE__),                                                     \
            (parent_id),                                                                                               \
            parent_class::SetUpTestCase,                                                                               \
            parent_class::TearDownTestCase,                                                                            \
            new ::testing::internal::TestFactoryImpl<GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);              \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBodyInternal(unsigned _threadIdx, unsigned _deviceIdx)

// Defines a test that uses a test fixture with supported devices and supported packages.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case. The third is an
// std::vector of supported devices. If no devices are specified then
// the default supported devices are used.
// If all devices are supported in the test use TEST_F instead.
//
// A test fixture class must be declared earlier.  The user should put
// the test code between braces after using this macro.  Example:
//
//   class FooTest : public SynTest {};
//
//   TEST_F_SYN(FooTest, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_F_SYN(FooTest, example2, {synDeviceGaud2}, {synTestPackage::CI}) {}
//   TEST_F_SYN(FooTest, example3) {}
//   TEST_F_SYN(FooTest, example4, std::vector<synDeviceType>({synDeviceGaudi})) {}
//   TEST_F_SYN(FooTest, example5, std::vector<synDeviceType>{synDeviceGaudi, synDeviceGaudi2}) {}
//
// Example2:
//
//   #include <vector>
//   #include <algorithm>
//   #include "synapse_common_types.h"
//
//   class FooTest2 : public ::testing::Test
//   {
//   public:
//       FooTest2()
//       {
//           m_deviceType = synDeviceGaudi;
//           setSupportedDevices({synDeviceGaudi});
//       }
//   protected:
//       virtual void SetUpTest() {}
//       virtual void TearDownTest() {}
//       void setSupportedDevices(std::vector<synDeviceType> supportedDeviceTypes)
//       {
//           m_supportedDeviceTypes = supportedDeviceTypes;
//       }
//       bool shouldRunTest()
//       {
//           return std::find(m_supportedDeviceTypes.begin(), m_supportedDeviceTypes.end(), m_deviceType) !=
//           m_supportedDeviceTypes.end();
//       }
//
//       synDeviceType m_deviceType;
//       std::vector<synDeviceType> m_supportedDeviceTypes;
//   };
//
//   TEST_F_SYN(FooTest2, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_F_SYN(FooTest2, example2, {synDeviceGaudi}) {}
//   TEST_F_SYN(FooTest2, example3) {}
#define TEST_F_SYN(test_fixture, test_name, ...)                                                                       \
    CREATE_SUITE_REGISTRATION(test_fixture)                                                                            \
    GTEST_TEST_SYN_(test_fixture, test_name, test_fixture, ::testing::internal::GetTypeId<test_fixture>(), __VA_ARGS__)

// Defines a value-parameterized test that uses a test fixture with supported devices.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case. The third is an
// std::vector of supported devices. If no devices are specified
// then the default supported devices are used.
// If all devices are supported in the test use TEST_F instead.
//
// A parametrized test fixture class must be declared earlier.  The user should put
// the test code between braces after using this macro.  Example:
//
//   class FooTest : public ::testing::TestWithParam<int>, public SynTest {};
//
//   TEST_P_SYN(FooTest, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_P_SYN(FooTest, example2, {synDeviceGaudi},{synTestPackage::ASIC_CI}) {}
//   TEST_P_SYN(FooTest, example3) {}
//   TEST_P_SYN(FooTest, example4, std::vector<synDeviceType>({synDeviceGaudi})) {}
//   TEST_P_SYN(FooTest, example5, std::vector<synDeviceType>{synDeviceGaudi, synDeviceGaudi2}) {}
//
//   INSTANTIATE_TEST_SUITE_P(SomePrefix, FooTest, testing::Values(1, 2, 3));
//
// Example2:
//
//   #include <vector>
//   #include <algorithm>
//   #include "synapse_common_types.h"
//
//   class FooTest2 : public ::testing::TestWithParam<int>, public ::testing::Test
//   {
//   public:
//       FooTest2()
//       {
//           m_deviceType = synDeviceGaudi;
//           setSupportedDevices({synDeviceGaudi});
//       }
//   protected:
//       virtual void SetUpTest() {}
//       virtual void TearDownTest() {}
//       void setSupportedDevices(std::vector<synDeviceType> supportedDeviceTypes)
//       {
//           m_supportedDeviceTypes = supportedDeviceTypes;
//       }
//       bool shouldRunTest()
//       {
//           return std::find(m_supportedDeviceTypes.begin(), m_supportedDeviceTypes.end(), m_deviceType) !=
//           m_supportedDeviceTypes.end();
//       }
//
//       synDeviceType m_deviceType;
//       std::vector<synDeviceType> m_supportedDeviceTypes;
//   };
//
//   TEST_P_SYN(FooTest2, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_P_SYN(FooTest2, example2, {synDeviceGaudi}) {}
//   TEST_P_SYN(FooTest2, example3) {}
//
//   INSTANTIATE_TEST_SUITE_P(SomePrefix, FooTest2, testing::Values(1, 2, 3));
#define TEST_P_SYN(test_case_name, test_name, ...)                                                                     \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name                                    \
    {                                                                                                                  \
        using base_class = test_case_name;                                                                             \
        TEST_CONFIGURATION_SUPPORT_MEMBERS(__VA_ARGS__)                                                                \
    public:                                                                                                            \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                                                         \
                                                                                                                       \
    private:                                                                                                           \
        static int AddToRegistry()                                                                                     \
        {                                                                                                              \
            ::testing::UnitTest::GetInstance()                                                                         \
                ->parameterized_test_registry()                                                                        \
                .GetTestCasePatternHolder<test_case_name>(#test_case_name,                                             \
                                                          ::testing::internal::CodeLocation(__FILE__, __LINE__))       \
                ->AddTestPattern(                                                                                      \
                    #test_case_name,                                                                                   \
                    #test_name,                                                                                        \
                    new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>(),     \
                    ::testing::internal::CodeLocation(__FILE__, __LINE__));                                            \
            return 0;                                                                                                  \
        }                                                                                                              \
        static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;                                                   \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name, test_name));                            \
    };                                                                                                                 \
    int GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::gtest_registering_dummy_ =                                  \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry();                                            \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBodyInternal(unsigned _threadIdx, unsigned _deviceIdx)

#define TYPED_TEST_SYN(test_case_name, test_name, ...)                                                                 \
    template<typename gtest_TypeParam_>                                                                                \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name<gtest_TypeParam_>                  \
    {                                                                                                                  \
        TEST_CONFIGURATION_SUPPORT_MEMBERS(__VA_ARGS__)                                                                \
                                                                                                                       \
    private:                                                                                                           \
        typedef test_case_name<gtest_TypeParam_> TestFixture;                                                          \
        typedef gtest_TypeParam_                 TypeParam;                                                            \
    };                                                                                                                 \
    static bool gtest_##test_case_name##_##test_name##_registered_ GTEST_ATTRIBUTE_UNUSED_ =                           \
        ::testing::internal::TypeParameterizedTest<                                                                    \
            test_case_name,                                                                                            \
            ::testing::internal::TemplateSel<GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>,                       \
            GTEST_TYPE_PARAMS_(                                                                                        \
                test_case_name)>::Register("",                                                                         \
                                           ::testing::internal::CodeLocation(__FILE__, __LINE__),                      \
                                           GTEST_STRINGIFY_(test_case_name),                                           \
                                           GTEST_STRINGIFY_(test_name),                                                \
                                           0,                                                                          \
                                           ::testing::internal::GenerateNames<GTEST_NAME_GENERATOR_(test_case_name),   \
                                                                              GTEST_TYPE_PARAMS_(test_case_name)>());  \
    template<typename gtest_TypeParam_>                                                                                \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)<gtest_TypeParam_>::TestBodyInternal(unsigned _threadIdx,    \
                                                                                               unsigned _deviceIdx)
