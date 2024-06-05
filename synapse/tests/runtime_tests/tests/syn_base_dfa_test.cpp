#include "syn_base_dfa_test.hpp"

REGISTER_SUITE(SynBaseDfaTest, ALL_TEST_PACKAGES);

bool SynBaseDfaTest::isScalDevice()
{
    switch (m_deviceType)
    {
        case synDeviceGaudi3:
        case synDeviceGaudi2:
            return true;

        case synDeviceGaudi:
        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            return false;
    }
    return false;
}

/*
 ***************************************************************************************************
 *   @brief getExpectedDevFail()
 *
 *   Get the expected words in dev-fail file. Concat common-expected + device-specific-expected
 *
 ***************************************************************************************************
 */
void SynBaseDfaTest::getExpectedDevFail(std::vector<ExpectedWords>& rtn, bool chkHlSmi)
{
    rtn = m_commonExpectedDevFail;

    switch (m_deviceType)
    {
        case synDeviceGaudi2:
            rtn.insert(rtn.end(), m_scalExpectedDevFail.begin(), m_scalExpectedDevFail.end());
            break;

        case synDeviceGaudi3:
            rtn.insert(rtn.end(), m_scalExpectedDevFail.begin(), m_scalExpectedDevFail.end());
            break;

        default:
            break;  // just use common
    }

    if (chkHlSmi && !(m_deviceType == synDeviceGaudi3))  // gaudi3 is running asic tests on simulator that doesn't
                                                         // support hl-smi
    {
        rtn.insert(rtn.end(), m_hlSimDevFail.begin(), m_hlSimDevFail.end());
    }
}

// Below are the expected string to find in each file
const std::vector<SynBaseDfaTest::ExpectedWords> SynBaseDfaTest::getCommonExpectedDevFail()
{
    return {{"Version:", 1, std::equal_to<uint32_t>()},
            {"Synapse:", 1, std::equal_to<uint32_t>()},
            {"HCL:", 1, std::equal_to<uint32_t>()},
            {"MME:", 1, std::equal_to<uint32_t>()},
            {"SCAL:", 1, std::equal_to<uint32_t>()},
            {"actualSize of engine dump", 1, std::equal_to<uint32_t>()},
            {"Number of queues", 1, std::greater_equal<uint32_t>()},  // in gaudi1 we log more than  once
            {"#device type", 1, std::equal_to<uint32_t>()},
            {"#is simulator", 1, std::equal_to<uint32_t>()},
            {std::string("is copied to ") + DMESG_COPY_FILE, 1, std::equal_to<uint32_t>()},
            {"#registers", 20, std::greater<uint32_t>()},  // at least 20 blocks of registers for all devices
            {"Failed reading registers", 0, std::equal_to<uint32_t>()},
            {"#DFA end", 1, std::equal_to<uint32_t>()},
            {"Log GCFG values", 1, std::equal_to<uint32_t>()},  // verify we log GCFG
            {"STATS_FREQ", 1, std::equal_to<uint32_t>()},       // verify an example of one GCFG value
            {"hw_ip fd", 1, std::equal_to<uint32_t>()}};
}

const std::vector<SynBaseDfaTest::ExpectedWords> SynBaseDfaTest::getScalExpectedDevFail()
{
    return {
        {"longSo: curr/target/prev", 13, std::greater<uint32_t>()},  // at least 13 (4*(compute+tx+rx))
        {"No-Progress TDR", 1, std::equal_to<uint32_t>()},
        {"Logging engines heartbeat values", 1, std::equal_to<uint32_t>()},
        {"TDR status: TDR not triggered", 6, std::greater_equal<uint32_t>()},
    };
}

const std::vector<SynBaseDfaTest::ExpectedWords> SynBaseDfaTest::getHlSmiDevFail()
{
    return {
        {"HL-SMI LOG", 1, std::equal_to<uint32_t>()},
        {"Product Name", 1, std::greater_equal<uint32_t>()},
        {"UUID", 1, std::equal_to<uint32_t>()},
        {"Network Information", 1, std::equal_to<uint32_t>()},
    };
}

const std::vector<SynBaseDfaTest::ExpectedWords> SynBaseDfaTest::getCommonExpectedDmesg()
{
    return {
        {"#uptime:", 1, std::equal_to<uint32_t>()},
        {"Failure occurred on device", 1, std::equal_to<uint32_t>()},
        {"#device type", 1, std::equal_to<uint32_t>()},
        {"#is simulator", 1, std::equal_to<uint32_t>()},
        {"dmesg copy start", 1, std::equal_to<uint32_t>()},
        {"dmesg copy done", 1, std::equal_to<uint32_t>()},
    };
}
