#pragma once

#include "hl_logger/hllog_core.hpp"
#include "hcl_api.hpp"

#include "gtest/gtest.h"
#include <fstream>
#include <regex>

inline namespace DfaFileChkVer_3
{
class dfaFilesCheck
{
public:
    enum DfaLoggerEnum
    {
        loggerSynDevFail,
        loggerDmesgCpy,
        loggerFailedRecipe,
        loggerDfaNicInfo,
        loggerDfaApiInfo
    };

    struct DfaLogInfo
    {
        DfaLoggerEnum         loggerEnum;
        hl_logger::LoggerSPtr logger;
        std::string           logFile;
        hl_logger::SinksSPtr  orgSinks;
        bool                  sinksChanged = false;
    };

    struct ExpectedWords
    {
        std::string                             word;
        uint32_t                                cnt;
        std::function<bool(uint32_t, uint32_t)> compare;
    };

    DfaLoggersV3 m_dfaLoggers;

    dfaFilesCheck() { }

    void init()
    {
        m_dfaLoggers = getDfaLoggersV3();
        createDfaFileNames();
    }

    std::vector<DfaLogInfo> m_dfaLogInfo {{DfaLoggerEnum::loggerSynDevFail},
                                          {DfaLoggerEnum::loggerDmesgCpy},
                                          {DfaLoggerEnum::loggerFailedRecipe},
                                          {DfaLoggerEnum::loggerDfaNicInfo},
                                          {DfaLoggerEnum::loggerDfaApiInfo}};

    /*
     ***************************************************************************************************
     *   @brief getLogFileName() returns the dfa file name. It is the original file name + the test name
     *
     *   set the expected file names in the array
     *
     ***************************************************************************************************
     */
    std::string getLogFileName(DfaLogInfo const& dfaLogInfo)
    {
        const char* testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return dfaLogInfo.logFile + ".forTest." + testName + ".txt";
    }

    /*
     ***************************************************************************************************
     *   @brief checkDfaBegin()
     *
     *   The function goes over all the DFA files, checks they all have the #DFA begin mark with the same
     *   id (timestamp) next to them
     *
     ***************************************************************************************************
     */
    void checkDfaBegin()
    {
        std::vector<std::string> foundStrings(m_dfaLogInfo.size() - 1);

        for (unsigned i = 0; i < m_dfaLogInfo.size(); i++)
        {
            auto&         entry = m_dfaLogInfo[i];
            const auto&   file  = getLogFileName(entry);
            bool          found = false;
            std::ifstream ifs(file);

            ASSERT_TRUE(ifs.is_open()) << " file " << m_dfaLogInfo[i].logFile << "not found \n";

            if (entry.loggerEnum != DfaLoggerEnum::loggerDfaApiInfo)
            {
                std::string line;
                while (getline(ifs, line))
                {
                    auto pos = line.find("#DFA begin");
                    if (pos != std::string::npos)
                    {
                        ASSERT_FALSE(found) << "should be only once";
                        found           = true;
                        foundStrings[i] = line.substr(pos);
                    }
                }
                ASSERT_TRUE(found) << " 'Begin' was not found in " << m_dfaLogInfo[i].logFile << "\n";
            }
        }

        // Verify all are the same
        for (const auto& found : foundStrings)
        {
            ASSERT_EQ(foundStrings[0], found) << "fail comparing " << foundStrings[0] << " vs " << found << "\n";
        }
    }

    /*
     ***************************************************************************************************
     *   @brief expectedInFile()
     *
     *   Check if the given expected-words are in the given file
     *
     ***************************************************************************************************
     */
    void expectedInFile(DfaLoggerEnum loggerEnum, const std::vector<ExpectedWords>& expected)
    {
        std::string   filename;

        bool found = false;
        for (auto const& entry : m_dfaLogInfo)
        {
            if (entry.loggerEnum == loggerEnum)
            {
                found    = true;
                filename = getLogFileName(entry);
                break;
            }
        }

        ASSERT_EQ(found, true);

        std::ifstream ifs(filename);
        ASSERT_EQ(ifs.is_open(), true) << "Could not open file " << filename << "\n";

        std::vector<uint32_t> actualCount(expected.size());
        std::string           line;

        // Go over all lines
        while (getline(ifs, line))
        {
            // Go over all expected words
            for (unsigned i = 0; i < expected.size(); i++)
            {
                const auto& word = expected[i].word;
                auto        pos  = line.find(word);
                if (pos != std::string::npos)
                {
                    actualCount[i]++;  // found, increase the word's counter
                }
            }
        }

        // Check the counters against expected values
        for (unsigned i = 0; i < expected.size(); i++)
        {
            bool ok = expected[i].compare(actualCount[i], expected[i].cnt);
            if (!ok)
            {
                printf("ERROR: %s - actual %d expected %d in file %s\n",
                       expected[i].word.c_str(),
                       actualCount[i],
                       expected[i].cnt,
                       filename.c_str());
            }
            EXPECT_EQ(ok, true);
        }
    }

    /*
     ***************************************************************************************************
     *   @brief createDfaFileNames()
     *
     *   set the expected file names in the array
     *
     ***************************************************************************************************
     */
    void createDfaFileNames()
    {
        for (auto& entry : m_dfaLogInfo)
        {
            switch (entry.loggerEnum)
            {
                case loggerSynDevFail   : entry.logger = m_dfaLoggers.dfaSynDevFailLogger;   break;
                case loggerDmesgCpy     : entry.logger = m_dfaLoggers.dfaDmesgLogger;        break;
                case loggerFailedRecipe : entry.logger = m_dfaLoggers.dfaFailedRecipeLogger; break;
                case loggerDfaNicInfo   : entry.logger = m_dfaLoggers.dfaNicInfoLogger;      break;
                case loggerDfaApiInfo   : entry.logger = m_dfaLoggers.dfaApiInfo;            break;
            }

            std::vector<std::string> names = hl_logger::getSinksFilenames(entry.logger);

            // assume we use only one file for this logger, if not, return "" - it will break the test
            assert(names.size() == 1); // at least in debug mode, crash the test
            entry.logFile = (names.size() ==  1) ? names[0] : "";
        }
    }

    /*
     ***************************************************************************************************
     *   @brief setTestLoggers() set the dfa logs to go to special files.
     *          This is to avoid losing the actual files that might have important content of other tests
     *
     *   set the expected file names in the array
     *
     ***************************************************************************************************
     */
    void setTestLoggers()
    {
        for (auto& entry : m_dfaLogInfo)
        {
            entry.orgSinks     = hl_logger::getSinks(entry.logger);
            entry.sinksChanged = true;
            std::remove(getLogFileName(entry).c_str());  // Remove previous files
            hl_logger::setSinks(entry.logger);
            hl_logger::addFileSink(entry.logger, getLogFileName(entry), 2*1024*1024*1024ull, 0);
        }
    }

    /*
     ***************************************************************************************************
     *   @brief removeTestDfaFiles() Removes the dfa files created for the test. This function
                should be called if the test passed
     *
     *   set the expected file names in the array
     *
     ***************************************************************************************************
     */
    void removeTestDfaFiles()
    {
        for (auto& entry : m_dfaLogInfo)
        {
            std::remove(getLogFileName(entry).c_str());
        }
    }

    /*
     ***************************************************************************************************
     *   @brief setOriginalLoggers() set the dfa logs back to the original files
     *
     *   set the expected file names in the array
     *
     ***************************************************************************************************
     */
    void setOriginalLoggers()
    {
        for (auto& entry : m_dfaLogInfo)
        {
            if (entry.sinksChanged)  // recover org-sinks if we changed them
            {
                hl_logger::flush(entry.logger);
                hl_logger::setSinks(entry.logger, entry.orgSinks);
            }
        }
    }

    /*
     ***************************************************************************************************
     *   @brief wordCountInDmesg()
     *
     *   Count how many times a word appears in a given number of dmesg log lines.
     *   In case timestamp is 'null' the function will replace it with the first valid timestamp
     *   within numOfLines range, if no timestamp found 'not found' timestamp is returned.
     *   In case timestamp is not 'null' the function will search for the given pattern from this
     *   timestamp as a reference of a starting point in dmesg log and by numOfLines argument.
     *   If timestamp is 'not found' it means no timestamp found of the previous call and the function
     *   will search for the given pattern with no timestamp reference and no lines range.
     *
     ***************************************************************************************************
     */
    void wordCountInDmesg(uint32_t& dmesgMsgCount, const std::string& word, std::string& timestamp, const uint32_t numOfLines)
    {
        std::string                  cmd;
        std::pair<bool, std::string> cmdOutput;

        // Search for a timestamp in numOfLines range
        if (timestamp == "null")
        {
            timestamp = "not found";
            cmd       = "dmesg | tail -n " + std::to_string(numOfLines);
            cmdOutput = exec(cmd);
            EXPECT_TRUE(cmdOutput.first) << cmdOutput.second;

            std::istringstream       iss(cmdOutput.second);
            std::vector<std::string> lines;
            std::string              line;
            std::regex               timestampRegex("\\d+\\.\\d+");
            std::smatch              matchTimestamp;

            while (std::getline(iss, line))
            {
                if (std::regex_search(line, matchTimestamp, timestampRegex))
                {
                    timestamp = matchTimestamp[0].str();
                    break;
                }
            }
        }

        // No timestamp found - probably dmesg was cleaned
        if (timestamp == "not found")
        {
            cmd = "dmesg | egrep '" + word + "' | wc | awk '{print $1}'";
        }
        else
        {
            cmd = "dmesg | grep " + timestamp + " -A " + std::to_string(numOfLines) + " | egrep '" + word + "' | wc | awk '{print $1}'";
        }

        cmdOutput = exec(cmd);
        EXPECT_TRUE(cmdOutput.first) << cmdOutput.second;

        dmesgMsgCount = stoul(cmdOutput.second);
    }

    std::pair<bool, std::string> exec(std::string cmd)
    {
        std::array<char, 4096> buffer;
        std::string            result;

        cmd += " 2>&1";
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        if (!pipe)
        {
            return {false, "couldn't open pipe"};
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        {
            result += buffer.data();
        }
        return {true, result};
    }
}; // class

} // namespace
