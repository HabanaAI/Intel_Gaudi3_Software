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
template <class T>
class MMEGaudi3TestExecutor : public MMEGaudi3Verification, public testing::WithParamInterface<std::string>
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
    static std::string getDirName() {assert(0); return "";};
    static std::string getTestName(::testing::TestParamInfo<std::string> p) { return p.param; }
    void executeParameterizedTest()
    {
        std::string testPath = T::getTestsPath() + "/" + GetParam();
        runTest(testPath, MmeCommon::e_mme_Gaudi3);
    }
    static std::string getTestsPath()
    {
        std::string configPath;
        char* mmeRoot = getenv("MME_ROOT");
        configPath = mmeRoot;
        configPath += std::string("/mme_verification/") + "gaudi3" + "/configs/";
        return configPath + T::getDirName();
    }
};

class MMEGaudi3VlsiTests : public MMEGaudi3TestExecutor<MMEGaudi3VlsiTests>
{
public:
    static std::string getTestsPath()
    {
        assert(getenv("SOFTWARE_LFS_DATA") != nullptr && "SOFTWARE_LFS_DATA is not defined !");
        std::string lfsPath = getenv("SOFTWARE_LFS_DATA");
        return lfsPath + "/synapse/tests/mme_tests/vlsi_tests";
    }
};

class MMEGaudi3BasicTests : public MMEGaudi3TestExecutor<MMEGaudi3BasicTests>
{
public:
    static std::string getDirName() { return "basic_tests"; }
};

class MMEGaudi3NumericTests : public MMEGaudi3TestExecutor<MMEGaudi3NumericTests>
{
public:
    static std::string getDirName() { return "numerics"; }
};

class MMEGaudi3BigTests : public MMEGaudi3TestExecutor<MMEGaudi3BigTests>
{
public:
    static std::string getDirName() { return "big_tests"; }
};

class MMEGaudi3DmaTests : public MMEGaudi3TestExecutor<MMEGaudi3DmaTests>
{
public:
    static std::string getDirName() { return "dma_tests"; }
};

class MMEGaudi3ReductionTests : public MMEGaudi3TestExecutor<MMEGaudi3ReductionTests>
{
public:
    static std::string getDirName() { return "reduction"; }
};

/***************     Gaudi3 tests     ****************/
TEST_P(MMEGaudi3BasicTests, basic_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests,
                        MMEGaudi3BasicTests,
                        ::testing::ValuesIn(MMEGaudi3BasicTests::getTestList()),
                        MMEGaudi3BasicTests::getTestName);

TEST_P(MMEGaudi3VlsiTests, vlsi_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests,
                        MMEGaudi3VlsiTests,
                        ::testing::ValuesIn(MMEGaudi3VlsiTests::getTestList()),
                        MMEGaudi3VlsiTests::getTestName);

TEST_P(MMEGaudi3NumericTests, numeric_tests)  // will enable once it passes tests
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests,
                        MMEGaudi3NumericTests,
                        ::testing::ValuesIn(MMEGaudi3NumericTests::getTestList()),
                        MMEGaudi3NumericTests::getTestName);

TEST_P(MMEGaudi3DmaTests, dma_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests,
                        MMEGaudi3DmaTests,
                        ::testing::ValuesIn(MMEGaudi3DmaTests::getTestList()),
                        MMEGaudi3DmaTests::getTestName);

TEST_P(MMEGaudi3BigTests, big_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests,
                        MMEGaudi3BigTests,
                        ::testing::ValuesIn(MMEGaudi3BigTests::getTestList()),
                        MMEGaudi3BigTests::getTestName);

TEST_P(MMEGaudi3ReductionTests, reduction_tests)
{
    executeParameterizedTest();
}
INSTANTIATE_TEST_CASE_P(Tests, MMEGaudi3ReductionTests, ::testing::ValuesIn(MMEGaudi3ReductionTests::getTestList()));
