#include "mme_test_base.h"
#include "gaudi2/mme_descriptor_generator.h"

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
class MMEGaudi2TestExecutor
: public MMEGaudi2Verification
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
    void executeParameterizedTest()
    {
        std::string testPath = T::getTestsPath() + "/" + GetParam();
        runTest(testPath, MmeCommon::e_mme_Gaudi2);
    }
    static std::string getTestsPath()
    {
        std::string configPath;
        char* mmeRoot = getenv("MME_ROOT");
        configPath = mmeRoot;
        configPath += std::string("/mme_verification/") + "gaudi2" + "/configs/";
        return configPath + T::getDirName();
    }
};

class MMEGaudi2VlsiTests : public MMEGaudi2TestExecutor<MMEGaudi2VlsiTests>
{
public:
    static std::string getDirName() { return "vlsi_tests"; }
};

class MMEGaudi2SanityTests : public MMEGaudi2TestExecutor<MMEGaudi2SanityTests>
{
public:
    static std::string getDirName() { return "sanity_tests"; }
};

class MMEGaudi2BGemmTests : public MMEGaudi2TestExecutor<MMEGaudi2BGemmTests>
{
public:
    static std::string getDirName() { return "bgemm_tests"; }
};

class MMEGaudi2ConvTests : public MMEGaudi2TestExecutor<MMEGaudi2ConvTests>
{
public:
    static std::string getDirName() { return "conv_tests"; }
};

class MMEGaudi2Dedw2xTests : public MMEGaudi2TestExecutor<MMEGaudi2Dedw2xTests>
{
public:
    static std::string getDirName() { return "dedw_2x"; }
};

class MMEGaudi2DedwFp8Tests : public MMEGaudi2TestExecutor<MMEGaudi2DedwFp8Tests>
{
public:
    static std::string getDirName() { return "dedw_fp8"; }
};

class MMEGaudi2LongTests : public MMEGaudi2TestExecutor<MMEGaudi2LongTests>
{
public:
    static std::string getDirName() { return "long_tests"; }
};
/***************     Gaudi2 tests     ****************/

TEST_P(MMEGaudi2SanityTests, sanity_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2SanityTests, ::testing::ValuesIn(MMEGaudi2SanityTests::getTestList()));

TEST_P(MMEGaudi2BGemmTests, bgemm_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2BGemmTests, ::testing::ValuesIn(MMEGaudi2BGemmTests::getTestList()));

TEST_P(MMEGaudi2ConvTests, conv_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2ConvTests, ::testing::ValuesIn(MMEGaudi2ConvTests::getTestList()));

TEST_P(MMEGaudi2LongTests, long_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2LongTests, ::testing::ValuesIn(MMEGaudi2LongTests::getTestList()));

TEST_P(MMEGaudi2VlsiTests, vlsi_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2VlsiTests, ::testing::ValuesIn(MMEGaudi2VlsiTests::getTestList()));

TEST_P(MMEGaudi2Dedw2xTests, dedw2x_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2Dedw2xTests, ::testing::ValuesIn(MMEGaudi2Dedw2xTests::getTestList()));

TEST_P(MMEGaudi2DedwFp8Tests, dedw_fp8_test)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi2DedwFp8Tests, ::testing::ValuesIn(MMEGaudi2DedwFp8Tests::getTestList()));