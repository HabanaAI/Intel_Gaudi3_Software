#include "mme_test_base.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

template<class T>
class MMECommonTestExecutor
: public MMEVerification
, public testing::WithParamInterface<std::string>
{
public:
    static std::vector<std::string> getTestList()
    {
        std::vector<std::string> testList;
        std::string testsPath = T::getTestsPath();
        fs::path p(testsPath);
        for (auto& testFile : fs::directory_iterator(p))
        {
            testList.push_back(testFile.path().stem().string());
        }
        return testList;
    }
    static std::string getDirName()
    {
        assert(0);
        return "";
    };
    void executeParameterizedTest()
    {
        std::string testPath = T::getTestsPath() + "/" + GetParam();
#ifdef GAUDI3_EN
        MmeCommon::ChipType chipType = MmeCommon::e_mme_Gaudi3;
#elif GAUDI2_EN
        MmeCommon::ChipType chipType = MmeCommon::e_mme_Gaudi2;
#else
        ASSERT_TRUE(false) << "either Gaudi2 or Gaudi3 have to be enabled";
#endif
        runTest(testPath, chipType);
    }
    static std::string getTestsPath()
    {
        std::string configPath;
        char* mmeRoot = getenv("MME_ROOT");
        configPath = mmeRoot;
        configPath += std::string("/mme_verification/") + "common" + "/configs/";
        return configPath + T::getDirName();
    }
};

class MMEGaudi2RegressionTests : public MMECommonTestExecutor<MMEGaudi2RegressionTests>
{
public:
    static std::string getDirName() { return "regression"; }
};

class MMEGaudi2SBReuseTests : public MMECommonTestExecutor<MMEGaudi2SBReuseTests>
{
public:
    static std::string getDirName() { return "sbReuse"; }
};

class MMEGaudi2SBReuse2Tests : public MMECommonTestExecutor<MMEGaudi2SBReuse2Tests>
{
public:
    static std::string getDirName() { return "sbReuse2"; }
};

class MMEGaudi2ReductionTests : public MMECommonTestExecutor<MMEGaudi2ReductionTests>
{
public:
    static std::string getDirName() { return "reduction"; }
};
class MMEGaudi2WorkloadTests : public MMECommonTestExecutor<MMEGaudi2WorkloadTests>
{
public:
    static std::string getDirName() { return "workloads"; }
};
class MMEGaudi2ResnetTests : public MMECommonTestExecutor<MMEGaudi2ResnetTests>
{
public:
    static std::string getDirName() { return "resnet_tests"; }
};
/***************     Common tests     ****************/
TEST_P(MMEGaudi2SBReuseTests, sbReuse_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2SBReuseTests, ::testing::ValuesIn(MMEGaudi2SBReuseTests::getTestList()));

TEST_P(MMEGaudi2SBReuse2Tests, sbReuse2_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2SBReuse2Tests, ::testing::ValuesIn(MMEGaudi2SBReuse2Tests::getTestList()));

TEST_P(MMEGaudi2ReductionTests, reduction_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2ReductionTests, ::testing::ValuesIn(MMEGaudi2ReductionTests::getTestList()));

TEST_P(MMEGaudi2WorkloadTests, workload_test)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2WorkloadTests, ::testing::ValuesIn(MMEGaudi2WorkloadTests::getTestList()));

TEST_P(MMEGaudi2RegressionTests, regression_test)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2RegressionTests, ::testing::ValuesIn(MMEGaudi2RegressionTests::getTestList()));

TEST_P(MMEGaudi2ResnetTests, resnet_test)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2ResnetTests, ::testing::ValuesIn(MMEGaudi2ResnetTests::getTestList()));