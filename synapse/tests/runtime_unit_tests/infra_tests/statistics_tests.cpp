//
// Created by esdeor on 11/17/22.
//

#include "statistics.hpp"
#include <gtest/gtest.h>
#include "scoped_configuration_change.h"
#include "global_conf_manager.h"


static std::string getPerfFileName()
{
    const char* logDir = std::getenv("HABANA_LOGS");
    return  std::string(logDir) + "/" + "perf_measure.log";
}

static void removePerfFile()
{
    std::string fileName = getPerfFileName();
    std::remove(fileName.c_str());
}

static uint64_t getPerfFileSize()
{
    synapse::LogManager::instance().flush();

    struct stat stat_buf {};
    int rc = stat(getPerfFileName().c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

enum class StatPoints {p1, p2, p3, LAST};
//    enum class StatPoints {p1, p2, p3, p4, LAST}; // This fails to compile (as it should)

/* This fails compilation as it should because the enum order is not correct
    static constexpr auto enumNamePoints = toStatArray<StatPoints>(
        {{StatPoints::p1, "p1"},
         {StatPoints::p3, "p2"},
         {StatPoints::p2, "p3"}});
*/
static constexpr auto enumNamePoints = toStatArray<StatPoints>(
    {{StatPoints::p1, "p1"},
     {StatPoints::p2, "p2"},
     {StatPoints::p3, "p3"}});

TEST(runtime_stats, basic)
{
    removePerfFile();
    GlobalConfManager::instance().init("");

    ScopedConfigurationChange experimental_flags("EXP_FLAGS", "true");
    ScopedConfigurationChange statsFreq("STATS_FREQ", "1");

    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);

    {
        Statistics<enumNamePoints> stat("Test", 5, true);

        stat.collect(StatPoints::p2, 1);
        stat.collect(StatPoints::p2, 2);

        auto get = stat.get(StatPoints::p2);

        ASSERT_EQ(get.count, 2);
        ASSERT_EQ(get.sum, 3);

        std::string name(get.name);

        ASSERT_EQ(name, "p2");
    }

    ASSERT_NE(getPerfFileSize(), 0) << "We should have something in the file";
}

TEST(runtime_stats, disabled1)
{
    removePerfFile();
    GlobalConfManager::instance().init("");

    ScopedConfigurationChange experimental_flags("EXP_FLAGS", "true");
    ScopedConfigurationChange statsFreq("STATS_FREQ", "1");

    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);

    Statistics<enumNamePoints> stat("Test", 5, false);

    stat.collect(StatPoints::p2, 0);
    stat.collect(StatPoints::p2, 0);

    auto get = stat.get(StatPoints::p2);

    ASSERT_EQ(get.count, 0);
    ASSERT_EQ(get.sum, 0);

    std::string name(get.name);

    ASSERT_EQ(name, "p2");
    ASSERT_EQ(getPerfFileSize(), -1) << "File should be empty";
}

TEST(runtime_stats, disabled2)
{
    removePerfFile();
    GlobalConfManager::instance().init("");

    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);

    Statistics<enumNamePoints> stat("Test", 5, true);

    stat.collect(StatPoints::p2, 0);
    stat.collect(StatPoints::p2, 0);

    auto get = stat.get(StatPoints::p2);

    ASSERT_EQ(get.count, 0);
    ASSERT_EQ(get.sum, 0);

    std::string name(get.name);

    ASSERT_EQ(name, "p2");
    ASSERT_EQ(getPerfFileSize(), -1) << "File should be empty";
}

TEST(runtime_stats, copy)
{
    removePerfFile();
    GlobalConfManager::instance().init("");

    ScopedConfigurationChange experimental_flags("EXP_FLAGS", "true");
    ScopedConfigurationChange statsFreq("STATS_FREQ", "1");

    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);

    Statistics<enumNamePoints> stat("Test", 5, true);

    stat.collect(StatPoints::p2, 1);
    stat.collect(StatPoints::p2, 2);

    auto get = stat.get(StatPoints::p2);

    ASSERT_EQ(get.count, 2);
    ASSERT_EQ(get.sum, 3);

    Statistics<enumNamePoints> statCopy = stat;

    stat.collect(StatPoints::p2, 1);
    statCopy.collect(StatPoints::p2, 100);

    get = stat.get(StatPoints::p2);

    ASSERT_EQ(get.count, 3);
    ASSERT_EQ(get.sum, 4);

    get = statCopy.get(StatPoints::p2);

    ASSERT_EQ(get.count, 1);
    ASSERT_EQ(get.sum, 100);
}

TEST(runtime_stats, EnableDisable)
{
    removePerfFile();
    GlobalConfManager::instance().init("");

    ScopedConfigurationChange experimental_flags("EXP_FLAGS", "true");
    ScopedConfigurationChange statsFreq("STATS_FREQ", "1");

    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);

    Statistics<enumNamePoints> stat("Test", 5, false);

    stat.collect(StatPoints::p2, 10); // stats disabled, should do nothing
    auto get = stat.get(StatPoints::p2);
    ASSERT_EQ(get.sum, 0);

    stat.setEnableState(true);
    stat.collect(StatPoints::p2, 5); // stats enabled, should collect
    get = stat.get(StatPoints::p2);
    ASSERT_EQ(get.sum, 5);

    stat.setEnableState(false); // disable again
    stat.collect(StatPoints::p2, 3); // stats disabled, should not change
    get = stat.get(StatPoints::p2);
    ASSERT_EQ(get.sum, 0);
}

TEST(runtime_stats, basicInt)
{
    removePerfFile();
    GlobalConfManager::instance().init("");

    ScopedConfigurationChange experimental_flags("EXP_FLAGS", "true");
    ScopedConfigurationChange statsFreq("STATS_FREQ", "1");

    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::PERF, 2);
    synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::RECIPE_STATS, 2);

    {
        std::vector<StatEnumMsg<int>> points = {{1,"1"}, {2, "2"}, {3,"3"}};

        StatisticsVec stat("Test", points, 5, true);

        stat.collect(2, 1);
        stat.collect(2, 2);

        auto get = stat.get(2);

        ASSERT_EQ(get.count, 2);
        ASSERT_EQ(get.sum, 3);

        std::string name(get.name);

        ASSERT_EQ(name, "3");
    }

    ASSERT_NE(getPerfFileSize(), 0) << "We should have something in the file";
}
