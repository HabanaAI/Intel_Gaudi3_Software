#include "mme_test.h"

namespace MmeCommon
{
int MMETest::run(int argc, char** argv)
{
    int status = EXIT_SUCCESS;
    do
    {
        try
        {
            // parse general args
            if (!m_parser.parse(argc, argv))
            {
                status = EXIT_FAILURE;
                break;
            }
            // parse config file
            if (!parseConfigFile(m_parser.getTestConfig()))
            {
                status = EXIT_FAILURE;
                break;
            }
            // run test and compare results
            if (!runTest(m_configParser.getParsedTests()))
            {
                status = EXIT_FAILURE;
                break;
            }
        }
        catch (const std::exception& exc)
        {
            std::cerr << "ERROR: " << exc.what() << "\n";
            status = EXIT_FAILURE;
        }
    } while (false);
    return status;
}

int MMETest::runFromGtest(const std::string& testConfigPath,
                          unsigned repeats,
                          unsigned seed,
                          bool chipTest,
                          bool simTest,
                          bool checkRoi,
                          unsigned mmeLimit,
                          unsigned threadLimit)
{
    bool exitStatus = EXIT_SUCCESS;
    try
    {
        m_configParser.setSeed(seed);
        m_configParser.parseJsonFile(testConfigPath, repeats);

        testVector& tests = m_configParser.getParsedTests();
        std::string dumpDir = "";
        std::string dumpUnit = "";
        EMmeDump dumpMmes = e_mme_dump_none;
        unsigned mmeDumpIdx = 0;
        std::string lfsrDir = "";
        std::vector<unsigned> deviceIdxs = {0};
        MME_ASSERT(simTest != false || chipTest != false, "cannot have test without device and without simulator");
        DeviceType devAType = simTest ? e_sim : e_null;
        DeviceType devBType = chipTest ? e_chip : e_null;

        std::cerr << "Random Seed Used : " << seed << std::endl;
        // run test and compare results
        if (!m_testManager->runTests(tests,
                                     dumpDir,
                                     dumpUnit,
                                     dumpMmes,
                                     mmeDumpIdx,
                                     lfsrDir,
                                     devAType,
                                     devBType,
                                     deviceIdxs,
                                     seed,
                                     threadLimit,
                                     mmeLimit,
                                     checkRoi))
        {
            exitStatus = EXIT_FAILURE;
        }
    }
    catch (...)
    {
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}

bool MMETest::parseConfigFile(const std::string& configFilePath)
{
    unsigned seed = m_parser.getSeed();
    unsigned repeats = m_parser.getRepeats();
    unsigned b2bTestLimit = m_parser.getB2BTestLimit();
    m_configParser.setSeed(seed);
    return m_configParser.parseJsonFile(configFilePath, repeats, b2bTestLimit);
}

bool MMETest::runTest(testVector& tests)
{
    auto testType = m_parser.getTestType();
    auto dumpDir = m_parser.getOutDir();
    auto dumpUnit = m_parser.getDumpUnit();
    auto dumpMmes = m_parser.getDumpMem();
    auto mmeDumpIdx = m_parser.getMmeIdx();
    auto lfsrDir = m_parser.getLFSRPath();
    auto devIdxs = m_parser.getDeviceIdxs();
    auto seed = m_parser.getSeed();
    auto mmeLimit = m_parser.getMmeLimit();
    auto checkRoi = m_parser.getCheckRoi();
    auto scalFw = m_parser.getScalFw();
    auto numOfThreads = m_parser.getThreadLimit();
    auto chipAlternative = m_parser.getChipAlternative(); // gaudiM/Gaudi2B etc..
    bool exitStatus;
    m_testManager->setscalFw(scalFw);
    switch (testType)
    {
        case EMMETestType::sim_null:
            exitStatus = m_testManager->runTests(tests,
                                                 dumpDir,
                                                 dumpUnit,
                                                 dumpMmes,
                                                 mmeDumpIdx,
                                                 lfsrDir,
                                                 e_sim,
                                                 e_null,
                                                 devIdxs,
                                                 seed,
                                                 numOfThreads,
                                                 mmeLimit,
                                                 checkRoi,
                                                 chipAlternative);
            break;
        case EMMETestType::sim_chip:
            exitStatus = m_testManager->runTests(tests,
                                                 dumpDir,
                                                 dumpUnit,
                                                 dumpMmes,
                                                 mmeDumpIdx,
                                                 lfsrDir,
                                                 e_sim,
                                                 e_chip,
                                                 devIdxs,
                                                 seed,
                                                 numOfThreads,
                                                 mmeLimit,
                                                 checkRoi,
                                                 chipAlternative);
            break;
        case EMMETestType::chip_null:
            exitStatus = m_testManager->runTests(tests,
                                                 dumpDir,
                                                 dumpUnit,
                                                 dumpMmes,
                                                 mmeDumpIdx,
                                                 lfsrDir,
                                                 e_chip,
                                                 e_null,
                                                 devIdxs,
                                                 seed,
                                                 numOfThreads,
                                                 mmeLimit,
                                                 checkRoi,
                                                 chipAlternative);
            break;
        case EMMETestType::null_null:
            exitStatus = m_testManager->runTests(tests,
                                                 dumpDir,
                                                 dumpUnit,
                                                 dumpMmes,
                                                 mmeDumpIdx,
                                                 lfsrDir,
                                                 e_null,
                                                 e_null,
                                                 devIdxs,
                                                 seed,
                                                 numOfThreads,
                                                 mmeLimit,
                                                 checkRoi,
                                                 chipAlternative);
            break;
        default:
            std::cerr << "ERROR: Unknown test type. \n";
            exitStatus = false;
    }

    return exitStatus;
}

}  // namespace MmeCommon
