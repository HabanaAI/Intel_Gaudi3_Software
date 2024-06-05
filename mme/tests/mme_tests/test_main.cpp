#include <gtest/gtest.h>
#include "mme_test_base.h"
#include <string>

int main(int argc, char** argv)
{
    std::vector<char*> argList = {};
    for (unsigned idx = 0 ; idx < argc ; idx++)
    {
        std::string argStr(argv[idx]);
        if (argStr == "--chip_test")
        {
            // device_sim_chip
            MMEVerification::setChipTest(true);
            continue;
        }
        else if (argStr == "--sim_test")
        {
            // device_sim_null
            MMEVerification::setChipTest(false);
            MMEVerification::setSimTest(true);
            continue;
        }
        else if (argStr == "--drop_sim_test")
        {
            // should be used for device_chip_null, there is an assert later on to prevent device_null_null
            MMEVerification::setSimTest(false);
        }
        else if (argStr == "--seed")
        {
            // seed
            idx++;
            unsigned s = std::stoi(std::string(argv[idx]));
            MMEVerification::setSeed(s);
            continue;
        }
        else if (argStr == "--no-color")
        {
            // no color - TBD
            continue;
        }
        else if (argStr == "--check_roi")
        {
            MMEVerification::setCheckRoi(true);
            continue;
        }
        else if (argStr == "--mme_limit")
        {
            idx++;
            unsigned limit = std::stoi(std::string(argv[idx]));
            MMEVerification::setMmeLimit(limit);
            continue;
        }
        else if (argStr == "-j")
        {
            idx++;
            unsigned threadLimit = std::stoi(std::string(argv[idx]));
            MMEVerification::setThreadLimit(threadLimit);
            continue;
        }
        else
        {
            argList.push_back(argv[idx]);
        }

    }
    int argsNr = (int)argList.size();
    ::testing::InitGoogleTest(&argsNr, argList.data());
    return RUN_ALL_TESTS();
}