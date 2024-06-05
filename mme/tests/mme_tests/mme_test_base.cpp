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

#ifdef GAUDI3_EN
#include "mme_verification/gaudi3/mme_test/gaudi3_mme_test_manager.h"
#include "mme_verification/common/mme_test.h"
#endif
#ifdef GAUDI2_EN
#include "mme_verification/gaudi2/mme_test/gaudi2_mme_test_manager.h"
#include "mme_verification/common/mme_test.h"
#endif
#ifdef GAUDI_EN
int executeMMEGaudiTest(int argc, char **argv);
#endif

unsigned MMEVerification::m_seed = 0;
bool MMEVerification::m_chipTest = false;
bool MMEVerification::m_simTest = true;
bool MMEVerification::m_checkRoi = false;
bool MMEVerification::m_shouldIncrementSeed = true;
unsigned MMEVerification::m_mmeLimit = 0;
unsigned MMEVerification::m_threadLimit = 0;

std::string MMEVerification::getConfigPath()
{
    // TODO: point to specific directory for CI tests.
    std::string configPath = getenv("MME_ROOT");
    configPath += "/mme_verification/" + getPlatform() + "/configs/" + getSubFolder();
    return configPath;
}

void MMEVerification::runTest(const std::string& testName, MmeCommon::ChipType chipType, unsigned repeats)
{
    printTimeStamp();
    std::string configPath = getTestConfigPath(testName, ".json");
    switch (chipType)
    {
        case MmeCommon::e_mme_Gaudi3:
        {
#ifdef GAUDI3_EN
            MmeCommon::MMETest test(std::make_unique<gaudi3::Gaudi3MmeTestManager>(), MmeCommon::e_mme_Gaudi3);
            int status = test.runFromGtest(configPath, repeats, m_seed, m_chipTest, m_simTest, m_checkRoi, m_mmeLimit, m_threadLimit);
            ASSERT_EQ(status, EXIT_SUCCESS);
            break;
#else
    ASSERT_TRUE(false) << "Gaudi3 stack doesnt exists";
#endif
        }
        case MmeCommon::e_mme_Gaudi2:
        {
#ifdef GAUDI2_EN
            MmeCommon::MMETest test(std::make_unique<gaudi2::Gaudi2MmeTestManager>(), MmeCommon::e_mme_Gaudi2);
            int status = test.runFromGtest(configPath, repeats, m_seed, m_chipTest, m_simTest, m_checkRoi, m_mmeLimit, m_threadLimit);
            ASSERT_EQ(status, EXIT_SUCCESS);
            break;
#else
            ASSERT_TRUE(false) << "Gaudi2 stack doesnt exists";
#endif
        }
        default:
            MME_ASSERT(0, "Chip not supported yet");
    }
}

void MMEGaudiVerification::runTest(const std::string& testName, MmeCommon::ChipType chipType, unsigned repeats)
{
    std::string configPath = getTestConfigPath(testName, ".cfg");
    std::vector<char*> argv;
    std::vector<std::string> args = parseArgs(configPath, m_seed);
    std::cerr << "Random Seed Used : " << m_seed << std::endl;
    for (auto& arg : args)
    {
        argv.push_back((char*)arg.data());
    }
    argv.push_back(nullptr);

#ifdef GAUDI_EN
    int status = executeMMEGaudiTest(argv.size() - 1, argv.data());
    ASSERT_EQ(status, EXIT_SUCCESS);
#else
    ASSERT_TRUE(false) << "Gaudi stack doesnt exists";
#endif
}

std::vector<std::string> MMEGaudiVerification::parseArgs(std::string testName, unsigned seed)
{
    std::vector<std::string> flags = {"test_type", "cfg", "seed"};
    std::vector<std::string> args;

    args.emplace_back(""); // first arg is the executable name - not relevant in this case.
    for (const auto& flag : flags)
    {
        std::string argument = flag + "=";
        if (flag == "test_type")
        {
            argument += m_chipTest ? "device_sim_chip" : "device_sim_null";
        }
        else if (flag == "cfg")
        {
            argument += testName;
        }
        else if (flag == "seed")
        {
            if (!seed) seed = time(nullptr);
            argument += std::to_string(seed);
        }
        else
        {
            assert(0);
        }
        args.emplace_back(argument);
    }
    return args;
}
std::string MMEVerification::getTestConfigPath(const std::string& testName, const std::string& suffix)
{
    std::string test_config_path;
    fs::path configPath(testName);
    if (configPath.is_relative())
    {
        configPath = fs::path(getConfigPath());
        configPath /= fs::path(testName);
    }
    if (!configPath.has_extension())
    {
        configPath.replace_extension(suffix);
    }

    std::cerr << "Test path : " << configPath.string() << std::endl;
    return configPath.string();
}

void MMEVerification::printTimeStamp() const
{
    // Get the current time in human-readable format
    time_t now = time(nullptr);
    char* time_str = ctime(&now);
    std::cerr << "Timestamp: " << time_str << std::endl;
}
