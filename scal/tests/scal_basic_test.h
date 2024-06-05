#pragma once

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_macros.h"
#include "common/pci_ids.h"

enum scalTestColor
{
    red,
    green,
    blue
};

enum scalSupportedDevices
{
    GAUDI2 = 0,
    GAUDI3 = 1,
    ALL = 9999
};

typedef std::vector<scalSupportedDevices> scalDeviceList;
class scalConsolePrint
{
public:
    scalConsolePrint(scalTestColor color);
    friend std::ostream& operator<<(std::ostream& os, const scalConsolePrint& print);
    std::ostream& operator<<(const scalConsolePrint& print);
    std::ostream& operator<<(std::ostream& (*pf)(std::ostream&));
    std::ostream& operator<<(const std::string& s);

private:
    std::string m_style;
};

class SCALTest : public ::testing::Test
{
public:
    SCALTest() : m_cout(scalTestColor::blue){ };

    virtual ~SCALTest(){};

    virtual bool CheckDevice(scalDeviceList list);
    virtual bool CheckDevice();
    int getScalDeviceTypeEnv();

protected:
    virtual void SetUp();
    virtual void TearDown();

    scalConsolePrint m_cout;
};

// A void test-function using  ASSERT_ or EXPECT_ calls should be encapsulated
// by this macro. Example: CHECK_FOR_FAILURES(MyCheckForEquality(counter, 42), "for counter=42")
#define CHECK_FOR_FAILURES(statement)     \
{                                         \
    ASSERT_NO_FATAL_FAILURE((statement)); \
}

enum scalDeviceType{
    dtOther  = -1,
    dtGaudi2 = 0,
    dtGaudi3 = 1
};

inline scalDeviceType getScalDeviceType(unsigned device_id)
{
    scalDeviceType ret = dtOther;
    switch (device_id)
    {
    case PCI_IDS_GAUDI2:
    case PCI_IDS_GAUDI2_SIMULATOR:
    case PCI_IDS_GAUDI2_FPGA:
        ret = dtGaudi2;
        break;

    case PCI_IDS_GAUDI3:
    case PCI_IDS_GAUDI3_DIE1:
    case PCI_IDS_GAUDI3_SINGLE_DIE:
    case PCI_IDS_GAUDI3_SIMULATOR:
    case PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE:
        ret = dtGaudi3;
        break;
    default:
        break;
    }
    return ret;
}

struct TestParams
{
    const char* first_stream_name;
    const char* second_stream_name;
    const char* first_completion_group_name;
    const char* second_completion_group_name;
    uint8_t     first_engine_group;
    uint8_t     second_engine_group;
    bool        is_supported_by_gaudi2;
};

class SCALTestDevice : public SCALTest
{
    protected:

        void SetUp() override
        {
            m_fd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
            if (m_fd <= 0)
            {
                LOG_ERR(SCAL,"{}:Can't open a device file errno {} {}",__FUNCTION__,  errno, std::strerror(errno));
                ASSERT_EQ(m_fd, 0);
            }

            int ret;
            ret = hlthunk_get_hw_ip_info(m_fd, &m_hw_ip);
            ASSERT_EQ(ret, 0);
        }

        void TearDown() override
        {
            if (m_fd != -1)
            {
                int rc = hlthunk_close(m_fd);
                ASSERT_EQ(rc, 0);
            }
        }

        scalDeviceType getScalDeviceType() { return ::getScalDeviceType(m_hw_ip.device_id);}

        int m_fd = -1;
        struct hlthunk_hw_ip_info m_hw_ip;
};

class SCALTestDmPdma : public SCALTestDevice, public testing::WithParamInterface<TestParams>
{
    protected:
        void pdma_host_sync_internal();
};
