#pragma once

#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// shameless patching of Google own macro  at npu-stack/3rd-parties/googletest_1_10/googletest/include/gtest/gtest-param-test.h
// use as
//   TEST_P_CHKDEV(YourClass, your_test,{GAUDI3})
//      where YourClass should derive from SCALTest
//            your_test - your test name
//            the last parameter is [optional] a vector of scalSupportedDevices ({ALL},{GAUDI2}, {GAUDI2,GAUDi3} etc.)
//
//   TEST_P is for tests that use parameters (e.g. call GetParam();) and expected to be called as  test_name/0  test_name/1  etc.
//   see TEST_F below for others
//  the wrapper behind SetUp(), TestBody() and TearDown() is void Test::Run()  in
//      3rd-parties/googletest_1_10/googletest/src/gtest.cc
#define TEST_P_CHKDEV(test_suite_name, test_name,...)                          \
  class GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)                     \
      : public test_suite_name {                                               \
   public:                                                                     \
    GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() {}                    \
    void SetUp() override {                                                    \
        if (!CheckDevice(__VA_ARGS__)) { GTEST_SKIP();}                        \
        test_suite_name::SetUp();                                              \
    }                                                                          \
    void TestBody() override {                                                 \
        TestBodyImpl();                                                        \
    }                                                                          \
    void TestBodyImpl() ;                                                      \
                                                                               \
   private:                                                                    \
    static int AddToRegistry() {                                               \
      ::testing::UnitTest::GetInstance()                                       \
          ->parameterized_test_registry()                                      \
          .GetTestSuitePatternHolder<test_suite_name>(                         \
              GTEST_STRINGIFY_(test_suite_name),                               \
              ::testing::internal::CodeLocation(__FILE__, __LINE__))           \
          ->AddTestPattern(                                                    \
              GTEST_STRINGIFY_(test_suite_name), GTEST_STRINGIFY_(test_name),  \
              new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_( \
                  test_suite_name, test_name)>(),                              \
              ::testing::internal::CodeLocation(__FILE__, __LINE__));          \
      return 0;                                                                \
    }                                                                          \
    static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;               \
  };                                                                           \
  int GTEST_TEST_CLASS_NAME_(test_suite_name,                                  \
                             test_name)::gtest_registering_dummy_ =            \
      GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)::AddToRegistry();     \
  void GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)::TestBodyImpl()



////////////////////////////////////////////////////////////////////////////////////////////////
#define TEST_F_CHKDEV(test_fixture, test_name,...)                                     \
  GTEST_TEST_CHKDEV(test_fixture, test_name, test_fixture,                             \
              ::testing::internal::GetTypeId<test_fixture>(), __VA_ARGS__)

#define GTEST_TEST_CHKDEV(test_suite_name, test_name, parent_class, parent_id,...)      \
  static_assert(sizeof(GTEST_STRINGIFY_(test_suite_name)) > 1,                          \
                "test_suite_name must not be empty");                                   \
  static_assert(sizeof(GTEST_STRINGIFY_(test_name)) > 1,                                \
                "test_name must not be empty");                                         \
  class GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)                              \
      : public parent_class {                                                           \
   public:                                                                              \
    GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() {}                             \
    ~GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() override = default;           \
                                                                                        \
   private:                                                                             \
    void SetUp() override {                                                             \
        if (!CheckDevice(__VA_ARGS__)) { GTEST_SKIP();}                                 \
        test_suite_name::SetUp();                                                       \
    }                                                                                   \
    void TestBody() override {                                                          \
        TestBodyImpl();                                                                 \
    }                                                                                   \
    void TestBodyImpl() ;                                                               \
    static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;               \
  };                                                                                    \
                                                                                        \
  ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_suite_name,                    \
                                                    test_name)::test_info_ =            \
      ::testing::internal::MakeAndRegisterTestInfo(                                     \
          #test_suite_name, #test_name, nullptr, nullptr,                               \
          ::testing::internal::CodeLocation(__FILE__, __LINE__), (parent_id),           \
          ::testing::internal::SuiteApiResolver<                                        \
              parent_class>::GetSetUpCaseOrSuite(__FILE__, __LINE__),                   \
          ::testing::internal::SuiteApiResolver<                                        \
              parent_class>::GetTearDownCaseOrSuite(__FILE__, __LINE__),                \
          new ::testing::internal::TestFactoryImpl<GTEST_TEST_CLASS_NAME_(              \
              test_suite_name, test_name)>);                                            \
  void GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)::TestBodyImpl()

////////////////////////////////////////////////////////////////////////////////////////////////