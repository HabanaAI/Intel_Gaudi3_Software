#pragma once
#include <gtest/gtest.h>
#include <fstream>
#include "include/mme_common/mme_common_enum.h"

typedef void (*unitTestPtr)(void);

class MMEVerification : public ::testing::Test
{
public:
    MMEVerification() = default;
    virtual ~MMEVerification() = default;
    virtual void
    runTest(const std::string& testName, MmeCommon::ChipType chipType = MmeCommon::e_mme_Gaudi2, unsigned repeats = 1);
    static void setSeed(unsigned seed) {m_seed = seed;}
    static void setChipTest(bool val) {m_chipTest = val;}
    static void setSimTest(bool val) {m_simTest = val;}
    static void setCheckRoi(bool val) {m_checkRoi = val;}
    static void setMmeLimit(unsigned val) { m_mmeLimit = val; }
    static void setThreadLimit(unsigned val) { m_threadLimit = val; }
    static void SetUpTestCase()
    {
        if (!m_seed)
        {
            m_seed = time(nullptr);
        }
        m_shouldIncrementSeed = false;
    };
private:
    void(*m_unitTest)() = nullptr;
    bool m_onlyUnitTest = false;
protected:
    static unsigned m_seed;
    static bool m_shouldIncrementSeed;
    static bool m_chipTest;
    static bool m_simTest;
    static bool m_checkRoi;
    static unsigned m_mmeLimit;
    static unsigned m_threadLimit;
    virtual void SetUp() override
    {
        char* enableConsole = getenv("ENABLE_CONSOLE");
        if (!enableConsole || strcmp(enableConsole, "true") != 0)
        {
            testing::internal::CaptureStdout();
        }
        if (!m_seed)
        {
            m_seed = time(nullptr);
        }
        if (m_shouldIncrementSeed)
        {
            m_seed++;
        }
        else
        {
            m_shouldIncrementSeed = true;
        }
    }
    virtual void TearDown() override
    {
        char* enableConsole = getenv("ENABLE_CONSOLE");
        if (!enableConsole || strcmp(enableConsole, "true") != 0)
        {
            std::string logContent = testing::internal::GetCapturedStdout();
            if (HasFailure())
            {
                std::cout << logContent << std::endl;
            }
            std::ofstream logFile(getLogFileName(), std::ofstream::out | std::ofstream::app);
            logFile << logContent << std::endl;
        }
    }
    virtual std::string getLogFileName()
    {
        const std::string logFileName = "mme_test_" + getPlatform() + ".log";
        std::string directory = std::string("HOME");
        char* env_habana_log_dir = getenv("HABANA_LOGS");
        if (env_habana_log_dir != nullptr)
        {
            directory = env_habana_log_dir;
        }
        else
        {
            directory += "/.habana_logs";
        }
        if (mkdir(directory.c_str(), 0777) != 0 && errno != EEXIST) assert(0);
        return directory + "/" + logFileName;
    }
    virtual std::string getConfigPath();
    std::string getTestConfigPath(const std::string& testName, const std::string& suffix);
    virtual std::string getPlatform() { return "common"; }
    virtual std::string getSubFolder() { return ""; }
    void printTimeStamp() const;
};

class MMEGaudi3Verification : public MMEVerification
{
protected:
    virtual std::string getPlatform() override {return "gaudi3";}
};

class MMEGaudi2Verification : public MMEVerification
{
protected:
    virtual std::string getPlatform() override {return "gaudi2";}
    virtual std::string getSubFolder() override { return ""; }
};

class MMEGaudiVerification : public MMEVerification
{
public:
    virtual void runTest(const std::string& testName,
                         MmeCommon::ChipType chipType = MmeCommon::e_mme_Gaudi,
                         unsigned repeats = 1) override;

protected:
    std::vector<std::string> parseArgs(std::string testName, unsigned seed = 0);
    virtual std::string getPlatform() override {return "gaudi";}
};