#pragma once

#include <exception>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <thread>
#include "mme_assert.h"
#include "mme_common/mme_common_enum.h"

enum class EMMETestType
{
    sim_null,
    sim_chip,
    chip_null,
    null_null
};

class MMETestArgumentParser
{
    using ArgMap = std::unordered_map<std::string, std::string>;

public:
    MMETestArgumentParser() = default;
    ~MMETestArgumentParser() = default;
    bool parse(int argc, char** argv);

    const EMMETestType& getTestType() const { return m_testType; }
    const std::string& getTestConfig() const { return m_testConfigPath; }
    const std::string& getOutDir() const { return m_outDir; }
    const std::string& getDumpUnit() const { return m_dumpUnit; }
    const unsigned& getSeed() const { return m_seed; }
    const unsigned& getRepeats() const { return m_repeats; }
    const std::string& getPole() const { return m_pole; }
    const MmeCommon::EMmeDump& getDumpMem() const { return m_dumpMem; }
    const unsigned& getMmeIdx() const { return m_mmeIdx; }
    const std::string& getLFSRPath() const { return m_lfsrPath; }
    const std::vector<unsigned>& getDeviceIdxs() const { return m_deviceIdxs; }
    const unsigned& getMmeLimit() const { return m_mmeLimit; }
    const unsigned& getDieNr() const { return m_dieNr; }
    const bool& getCheckRoi() const { return m_checkRoi; }
    const unsigned& getB2BTestLimit() const {return m_b2bTestLimit;}
    const unsigned& getThreadLimit() const {return m_threadLimit;}
    const bool& getScalFw() const { return m_scalFw; }
    const bool& getChipAlternative() const { return m_chipAlternative; }
private:
    ArgMap parseCommandLineToArgMap(int argc, char** argv);
    void parseArgs(ArgMap& argMap);
    void checkRequired(const ArgMap& argMap);
    void printHelp();

    // parsers
    void parseNum(const std::string& arg, unsigned& num) { num = std::stoi(arg); }
    void parseNumList(const std::string& arg, const std::string& delimiter, std::vector<unsigned>& val);
    void parseBool(const std::string& arg, bool& val)
    {
        val = (arg == "True" || arg == "true" || arg == "T" || arg == "t" || arg == "1");
    }

    void parseTestType(const std::string& type);
    void parsePath(const std::string& argName, const std::string& path, std::string& memberVal, bool create = false);
    void parseDumpMemory(const std::string& dumpMode);

    // required arguments
    EMMETestType m_testType;
    std::string m_testConfigPath;
    // optional arguments
    std::string m_outDir;
    std::string m_dumpUnit;
    unsigned m_seed = time(nullptr);
    unsigned m_repeats = 1;
    std::string m_pole = "none";
    MmeCommon::EMmeDump m_dumpMem = MmeCommon::e_mme_dump_none;
    unsigned m_mmeIdx = 0;
    std::string m_lfsrPath;
    std::vector<unsigned> m_deviceIdxs = {0};
    unsigned m_mmeLimit = 0;
    unsigned m_dieNr = 0;
    bool m_checkRoi = false;
    unsigned m_b2bTestLimit = 1;
    unsigned m_threadLimit = 0;
    bool m_scalFw = false;
    bool m_chipAlternative = false;
};

class RequiredArgMissing : public std::exception
{
public:
    RequiredArgMissing(const char* arg)
    {
        m_msg = std::string("argument ") + std::string(arg) + std::string(" is required but missing");
    }
    ~RequiredArgMissing() noexcept override = default;

    const char* what() const noexcept override { return m_msg.c_str(); }

private:
    std::string m_msg;
};

class InvalidArgument : public std::exception
{
public:
    InvalidArgument(const char* arg, const char* val)
    {
        m_msg = std::string("argument ") + std::string(arg) + std::string(" has invalid value - ") + std::string(val);
    }
    ~InvalidArgument() noexcept override = default;
    const char* what() const noexcept override { return m_msg.c_str(); }

private:
    std::string m_msg;
};
