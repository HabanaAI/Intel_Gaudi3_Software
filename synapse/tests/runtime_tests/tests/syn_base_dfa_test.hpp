#pragma once

#include <string>

#include "syn_base_test.hpp"
#include "test_config_types.h"

#include "runtime/common/device/device_interface.hpp"

#include <chrono>
#include <vector>
#include <functional>
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "internal/tests/dfa_files_check.hpp"
/*
 ***************************************************************************************************
 *   Class SynBaseDfaTest
 *
 *   This is the testing class. Note, it is inherits from SynTest but overrides SetUpTest() so
 *   the tests using it can start "clean", before synInit
 *
 ***************************************************************************************************
 */
class SynBaseDfaTest
: public SynBaseTest
, public dfaFilesCheck
{
public:
    SynBaseDfaTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }

    ~SynBaseDfaTest() { setOriginalLoggers(); }

    virtual void SetUp() override
    {
        SynBaseTest::SetUp();

        if (::testing::Test::IsSkipped())
        {
            return;
        }

        dfaFilesCheck::init();
        setTestLoggers();
    }

    virtual void TearDown() override {}

    bool isScalDevice();

    static constexpr int CHK_LAUNCH_DUMP_NUM = 3;

    void getExpectedDevFail(std::vector<ExpectedWords>& rtn, bool chkHlSmi);

    const std::vector<ExpectedWords> getCommonExpectedDevFail();
    const std::vector<ExpectedWords> getScalExpectedDevFail();
    const std::vector<ExpectedWords> getHlSmiDevFail();
    const std::vector<ExpectedWords> getCommonExpectedDmesg();

    const std::vector<ExpectedWords> m_commonExpectedDevFail = getCommonExpectedDevFail();
    const std::vector<ExpectedWords> m_scalExpectedDevFail   = getScalExpectedDevFail();
    const std::vector<ExpectedWords> m_hlSimDevFail          = getHlSmiDevFail();
};
