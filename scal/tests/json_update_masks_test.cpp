#include <gtest/gtest.h>
//#include "common/json.hpp"
#include <iostream>
#include <fstream>
#include "infra/json_update_mask.hpp"
#include "scal_basic_test.h"

static void dumpJson(const scaljson::json& json, const char* filename)
{
    std::fstream f;
    f.open(filename, std::fstream::out | std::fstream::trunc);
    if (!f.is_open())
    {
        printf("Failed to open file %s\n", filename);
        return;
    }
    f << std::setw(4) << json;
    f.close();
}

class JsonUpdateTest : public SCALTestDevice
{
protected:
    void openOrgJson();
    void run();

    scaljson::json                  m_json;
    JsonUpdateMask::EngMasks*       m_engMasks = nullptr;
    std::unique_ptr<JsonUpdateMask> m_jsonUpdateMask;
};

void JsonUpdateTest::openOrgJson()
{
    std::ifstream jsonFile(":/default.json");
    ASSERT_EQ(!jsonFile, false) << "failed to open json file";

    // parse the json file
    try
    {
        m_json = scaljson::json::parse(jsonFile, nullptr, true, true);
    }
    catch (const std::exception &e)
    {
        ASSERT_TRUE(false) << "failed to parse json file";
    }

    dumpJson(m_json, "before.json");

    m_jsonUpdateMask = std::make_unique<JsonUpdateMask>(m_json, m_fd);
    bool rtn = m_jsonUpdateMask->initMasks();
    if (rtn != true)
    {
        ASSERT_TRUE(false) << "Faied to init masks";
    }

    m_engMasks = &m_jsonUpdateMask->testingOnlyGetEngMasks();
}

void JsonUpdateTest::run()
{
    m_jsonUpdateMask->run();
    dumpJson(m_jsonUpdateMask->getPatch(), "change.json");

    auto newJson = m_json.patch(m_jsonUpdateMask->getPatch());
    dumpJson(newJson, "after.json");
}

// Test is disabled as it is used for development only
TEST_F_CHKDEV(JsonUpdateTest, DISABLED_no_netwrok_hd_0_2_4_6, {GAUDI3})
{
    openOrgJson();
    m_engMasks->mme &= 0x55;
    m_engMasks->tpc &= 0x00FF00FF00FF00FF;
    m_engMasks->rot &= 0;
    m_engMasks->dma &= 0x0;
    m_engMasks->nic &= 0;

    run();
}

// Test is disabled as it is used for development only
TEST_F_CHKDEV(JsonUpdateTest, DISABLED_network_only_override, {GAUDI3})
{
    openOrgJson();
    m_engMasks->mme &= 0;
    m_engMasks->tpc &= 0;
    m_engMasks->rot &= 0x41;
    m_engMasks->dma &= 0xFF;
    m_engMasks->nic &= 0xFFFFFF;

    run();
}
