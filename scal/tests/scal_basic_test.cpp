#include "scal_basic_test.h"

scalConsolePrint::scalConsolePrint(scalTestColor color)
{
    switch (color)
    {
    case scalTestColor::blue:
    default:
        m_style = "\033[0;34m[ SCALOut ]\033[0m ";
        break;
    }
}

void SCALTest::SetUp()
{
}

void SCALTest::TearDown()
{
    // scalDestroy();
}

bool SCALTest::CheckDevice()
{
    // if the user didn't specify any supported devices
    // we assume all are (optimistics, aren't we ..)
    return true;
}

bool SCALTest::CheckDevice(scalDeviceList okDeviceList)
{
    int scal_dev_type = getScalDeviceTypeEnv();
    for (auto dev : okDeviceList)
    {
        if (dev == scalSupportedDevices::ALL)
            return true;
        if (dev == (scalSupportedDevices)scal_dev_type)
            return true;
    }
    return false;
}

int SCALTest::getScalDeviceTypeEnv()
{
    int scal_device = 0; // default is Gaudi2
    const char* scal_device_type = getenv("SCAL_DEVICE_TYPE");
    if (scal_device_type != nullptr)
    {
        try
        {
            // Ensure its a real device
            scal_device = std::stoi(scal_device_type);
        }
        catch (const std::invalid_argument& ia)
        {
            LOG_ERR(SCAL,"illegal value in SCAL_DEVICE_TYPE {}", scal_device_type);
            scal_device = 0;
        }
    }
    return scal_device;
}

std::ostream& operator<<(std::ostream& os, const scalConsolePrint& print)
{
    std::cout << print.m_style;
    return os;
}

std::ostream& scalConsolePrint::operator<<(const scalConsolePrint& print)
{
    std::cout << print.m_style;
    return std::cout;
}

std::ostream& scalConsolePrint::operator<<(std::ostream& (*pf)(std::ostream&))
{
    std::cout << m_style << pf;
    return std::cout;
}

std::ostream& scalConsolePrint::operator<<(const std::string& s)
{
    std::cout << m_style << s;
    return std::cout;
}


