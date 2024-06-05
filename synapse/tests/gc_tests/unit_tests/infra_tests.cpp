#include <atomic>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

#include "infra/global_conf_manager.h"
#include "infra/threads/thread_pool.h"
#include "infra/threads/thread_work_item.h"
#include "graph_optimizer_test.h"

class InfraTest : public GraphOptimizerTest {};

TEST_F(InfraTest, global_conf_from_string)
{
    // int64 global conf check
    GlobalConfInt64 g_64("g_64", "empty description", 0);
    g_64.setFromString("400000000001");
    ASSERT_EQ(g_64.value(), 400000000001);
    g_64.setFromString("-400000000001");
    ASSERT_EQ(g_64.value(), -400000000001);

    // bool check
    GlobalConfBool g_bool("g_bool", "empty description", false);
    g_bool.setFromString("true");
    ASSERT_EQ(g_bool.value(), true);
    g_bool.setFromString("0");
    ASSERT_EQ(g_bool.value(), false);
    g_bool.setFromString("1");
    ASSERT_EQ(g_bool.value(), true);
    g_bool.setFromString("false");
    ASSERT_EQ(g_bool.value(), false);

    // float check
    GlobalConfFloat g_float("g_float", "empty description", 0);
    g_float.setFromString("40.001");
    ASSERT_EQ(g_float.value(), (float)40.001);
    g_float.setFromString("-40.001");
    ASSERT_EQ(g_float.value(), (float)-40.001);

    // SizeParam check
    GlobalConfSize         g_size("g_size", "empty description", hl_gcfg::SizeParam());
    constexpr unsigned int KB_SIZE = (1024);
    constexpr unsigned int MB_SIZE = (KB_SIZE * KB_SIZE);
    constexpr unsigned int GB_SIZE = (KB_SIZE * KB_SIZE * KB_SIZE);

    g_size.setFromString("8MB");
    uint64_t    byteVal = (uint64_t)8 * MB_SIZE;
    std::string strVal  = std::to_string((uint64_t)8 * MB_SIZE) + " (8MB)";
    ASSERT_EQ(g_size.value(), byteVal);
    ASSERT_EQ(g_size.getValueStr(), strVal);

    g_size.setFromString("64k");
    byteVal = (uint64_t)64 * KB_SIZE;
    strVal  = std::to_string(byteVal) + " (64k)";
    ASSERT_EQ(g_size.value(), byteVal);
    ASSERT_EQ(g_size.getValueStr(), strVal);

    g_size.setFromString("4gb");
    byteVal = (uint64_t)4 * GB_SIZE;
    strVal  = std::to_string(byteVal) + " (4gb)";
    ASSERT_EQ(g_size.value(), byteVal);
    ASSERT_EQ(g_size.getValueStr(), strVal);

    g_size.setFromString("4096");
    byteVal = (uint64_t)4096;
    strVal  = std::to_string(byteVal) + " (4096B)";
    ASSERT_EQ(g_size.value(), byteVal);
    ASSERT_EQ(g_size.getValueStr(), strVal);
}

TEST_F(InfraTest, global_conf_manager_exist_file)
{
    GlobalConfFloat g_float("g_float", "empty description", 2.56);
    GlobalConfInt64 g_64("g_64", "empty description", 3);
    GlobalConfBool  g_bool("g_bool", "empty description", false);

    std::set<std::string> stringRep({"g_float=4.75", "g_64=400", "g_bool=true"});

    static const std::string testFileName = "g_conf_test.ini";
    std::ofstream file(testFileName.c_str());
    ASSERT_TRUE(file.good());
    if (! file.good()) return;

    for (auto line : stringRep)
    {
        file << line << std::endl;
    }
    file.close();

    GlobalConfManager::instance().init("");

    GlobalConfManager::instance().setGlobalConf("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GlobalConfManager::instance().load(testFileName);
    GlobalConfManager::instance().setGlobalConf("ENABLE_EXPERIMENTAL_FLAGS", "false");

    ASSERT_EQ(g_float.value(), 4.75);
    ASSERT_EQ(g_64.value(), 400);
    ASSERT_EQ(g_bool.value(), true);

    remove(testFileName.c_str());
}

TEST_F(InfraTest, global_conf_default_by_type)
{
    GlobalConfInt64 g_64("g_64", "empty description", hl_gcfg::DfltInt64(3) << hl_gcfg::deviceValue(synDeviceGaudi, 2));
    ASSERT_EQ(g_64.value(), 3);

    ASSERT_EQ(g_64.getDefaultValue(synDeviceTypeInvalid), 3);
    ASSERT_EQ(g_64.getDefaultValue(synDeviceGaudi), 2);
    ASSERT_EQ(g_64.getDefaultValue(synDeviceGaudi2), 3);

    GlobalConfManager::instance().init("");
    ASSERT_EQ(g_64.value(), 3);
    GlobalConfManager::instance().setDeviceType(synDeviceGaudi);
    ASSERT_EQ(g_64.value(), 2);
}

class TestWorkItem : public synapse::ThreadWorkItem
{
public:
    TestWorkItem(std::atomic<int>& count) : m_count(count) {}

    virtual ~TestWorkItem() {}

    virtual void doWork() override
    {
        ++m_count;
    }

    std::atomic<int>& m_count;
};

TEST_F(InfraTest, thread_pool_work)
{
    static const uint32_t numOfWorks = 100;
    synapse::ThreadPool tp(8);
    tp.start();

    std::atomic<int> count(0);
    for (uint32_t i = 0; i< numOfWorks; ++i)
        tp.addJob(new TestWorkItem(count));

    tp.finish();
    ASSERT_EQ(count, numOfWorks);
}
