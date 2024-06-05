#pragma once

#include <gtest/gtest.h>
#include <gtest/gtest-typed-test.h>
#include <variant>

#ifndef GTEST_DISALLOW_ASSIGN_
// A macro to disallow copy operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_ASSIGN_(type) \
  type& operator=(type const &) = delete
#endif

#ifndef GTEST_DISALLOW_COPY_AND_ASSIGN_
// A macro to disallow copy constructor and operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type) \
  type(type const &) = delete; \
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
                                                                                                                       \
        auto testFunc = [this](unsigned threadIdx, unsigned deviceIdx) {                                               \
            this->m_threadDeviceId = deviceIdx < this->m_deviceIds.size() ? this->m_deviceIds.at(deviceIdx) : 0;       \
            if (this->m_MultiThreadConf.iterations != 0)                                                               \
            {                                                                                                          \
                for (unsigned i = 0; i < this->m_MultiThreadConf.iterations && !::testing::Test::HasFailure(); ++i)    \
                {                                                                                                      \
                    TestBodyInternal(threadIdx, deviceIdx);                                                            \
                }                                                                                                      \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                auto start = std::chrono::steady_clock::now();                                                         \
                do                                                                                                     \
                {                                                                                                      \
                    TestBodyInternal(threadIdx, deviceIdx);                                                            \
                    auto finish = std::chrono::steady_clock::now();                                                    \
                    if (finish - start >= this->m_MultiThreadConf.maxDuration || ::testing::Test::HasFailure())        \
                    {                                                                                                  \
                        break;                                                                                         \
                    }                                                                                                  \
                } while (true);                                                                                        \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        if (this->m_MultiThreadConf.nbThreads <= 1 &&                                                                  \
            (this->m_simultaneousDeviceExecution == false || this->m_deviceIds.size() < 2))                            \
        {                                                                                                              \
            this->m_nbTestThreads = 1;                                                                                 \
            testFunc(0, 0);                                                                                            \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            std::vector<std::thread> threads;                                                                          \
            this->m_nbTestThreads = this->m_deviceIds.size() * this->m_MultiThreadConf.nbThreads;                      \
            for (unsigned deviceIdx = 0; deviceIdx < this->m_deviceIds.size(); ++deviceIdx)                            \
            {                                                                                                          \
                threads.reserve(this->m_MultiThreadConf.nbThreads);                                                    \
                for (unsigned i = 0; i < this->m_MultiThreadConf.nbThreads; ++i)                                       \
                {                                                                                                      \
                    threads.emplace_back(testFunc, i, deviceIdx);                                                      \
                }                                                                                                      \
                if (this->m_simultaneousDeviceExecution == false)                                                      \
                {                                                                                                      \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
            for (auto& t : threads)                                                                                    \
            {                                                                                                          \
                t.join();                                                                                              \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    void TestBodyInternal(unsigned _threadIdx, unsigned _deviceIdx);                                                   \
                                                                                                                       \
protected:                                                                                                             \
    void _setTestConfiguration() {}                                                                                    \
    void _setTestConfiguration(MultiThread mtConf) { this->m_MultiThreadConf = mtConf; }                               \
                                                                                                                       \
    void _setTestConfiguration(std::vector<std::variant<char, synDeviceType>> supportedDeviceTypes)                    \
    {                                                                                                                  \
        this->setSupportedDevices(supportedDeviceTypes);                                                               \
    }                                                                                                                  \
                                                                                                                       \
    void _setTestConfiguration(std::vector<std::variant<char, synDeviceType>> supportedDeviceTypes,                    \
                               MultiThread                                    mtConf)                                  \
    {                                                                                                                  \
        _setTestConfiguration(mtConf);                                                                                 \
        this->setSupportedDevices(supportedDeviceTypes);                                                               \
    }                                                                                                                  \
    void SetUp() override                                                                                              \
    {                                                                                                                  \
        _setTestConfiguration(__VA_ARGS__);                                                                            \
        this->SetUpTest();                                                                                             \
    }                                                                                                                  \
    void TearDown() override { this->TearDownTest(); }

#define GTEST_TEST_GC_(test_case_name, test_name, parent_class, parent_id, ...)                                        \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class                                      \
    {                                                                                                                  \
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

// Defines a test that uses a test fixture with supported devices.
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
//   TEST_F_GC(FooTest, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_F_GC(FooTest, example2, {synDeviceGaud2}) {}
//   TEST_F_GC(FooTest, example3) {}
//   TEST_F_GC(FooTest, example3, MiltiThread(100s, 2_threads)) {} // run the test in a loop for 100s in 2 threads
//   TEST_F_GC(FooTest, example4, std::vector<synDeviceType>({synDeviceGaudi})) {}
//   TEST_F_GC(FooTest, example5, std::vector<synDeviceType>{synDeviceGaudi, synDeviceGaudi2}) {}
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
//   TEST_F_GC(FooTest2, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_F_GC(FooTest2, example2, {synDeviceGaudi}) {}
//   TEST_F_GC(FooTest2, example3) {}
#define TEST_F_GC(test_fixture, test_name, ...)                                                                        \
    GTEST_TEST_GC_(test_fixture, test_name, test_fixture, ::testing::internal::GetTypeId<test_fixture>(), __VA_ARGS__)

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
//   TEST_P_GC(FooTest, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_P_GC(FooTest, example2, {synDeviceGaudi}) {}
//   TEST_P_GC(FooTest, example3) {}
//   TEST_P_GC(FooTest, example4, std::vector<synDeviceType>({synDeviceGaudi})) {}
//   TEST_P_GC(FooTest, example5, std::vector<synDeviceType>{synDeviceGaudi, synDeviceGaudi2}) {}
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
//   TEST_P_GC(FooTest2, example1, {synDeviceGaudi, synDeviceGaudi2}) {}
//   TEST_P_GC(FooTest2, example2, {synDeviceGaudi}) {}
//   TEST_P_GC(FooTest2, example3) {}
//
//   INSTANTIATE_TEST_SUITE_P(SomePrefix, FooTest2, testing::Values(1, 2, 3));
#define TEST_P_GC(test_case_name, test_name, ...)                                                                      \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name                                    \
    {                                                                                                                  \
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

#define TYPED_TEST_GC(test_case_name, test_name, ...)                                                                  \
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
