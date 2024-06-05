#pragma once

#include "argument_parser.h"
#include "config_parser.h"
#include <cstdlib>
#include <iostream>
#include "mme_test_manager.h"

namespace MmeCommon
{
class MMETest
{
public:
    MMETest(std::unique_ptr<MmeCommon::MmeTestManager> testManager, ChipType chipType)
    : m_testManager(std::move(testManager)), m_configParser(chipType) {};
    virtual ~MMETest() = default;
    virtual int run(int argc, char** argv);
    virtual int runFromGtest(const std::string& testConfigPath,
                             unsigned repeats,
                             unsigned seed,
                             bool chipTest,
                             bool simTest,
                             bool checkRoi,
                             unsigned mmeLimit = 0,
                             unsigned threadLimit = 0);

protected:
    bool parseConfigFile(const std::string& configFilePath);
    bool runTest(testVector& tests);

    std::unique_ptr<MmeCommon::MmeTestManager> m_testManager;
    MMETestArgumentParser m_parser;
    MmeCommon::MMETestConfigParser m_configParser;
};

}  // namespace Gaudi2
