#include "device_handler.h"
#include "print_utils.h"
#include "src/mme_common/mme_hal_reader.h"

namespace MmeCommon
{
void DeviceHandler::createDevices(const uint64_t pqBaseAddr, unsigned dieNr)
{
    m_dieNr = dieNr;
    if (m_runOnChip)
    {
        for (const auto& idx : m_deviceIdxs)
        {
            coral::DriverDeviceBase* device = getChipDriverDevice();
            MME_ASSERT_PTR(device);
            m_driverDevices.push_back(device);
        }
    }
    if (m_runOnSim)
    {
        m_simDevice = getChipSimDevice(pqBaseAddr);
    }
}

bool DeviceHandler::openDevices()
{
    if (m_runOnSim)
    {
        if (!openSimDevice()) return false;
    }
    if (m_runOnChip)
    {
        if (!openDriverDevices()) return false;
    }
    return true;
}

bool DeviceHandler::openSimDevice()
{
    if (!m_simDevice->openDevice(0))
    {
        atomicColoredPrint(COLOR_RED, "FATAL ERROR: failed initializing Simulator.\n");
        return false;
    }
    return true;
}

bool DeviceHandler::openDriverDevices()
{
    auto currDriverDeviceItr = m_driverDevices.begin();
    std::optional<unsigned> driverDieNr;
    std::optional<bool> isCacheMode;
    for (const auto& idx : m_deviceIdxs)
    {
        if (!(*currDriverDeviceItr)->openDevice(idx))
        {
            atomicColoredPrint(COLOR_RED, "FATAL ERROR: failed initializing driver device ID %d.\n", idx);
            return false;
        }
        unsigned curDieNr = (*currDriverDeviceItr)->getNumDies();
        bool curCacheMode =  (*currDriverDeviceItr)->isCacheMode();
        assertOrAssign(driverDieNr, curDieNr, "all driver devices must have the same amount of dies");
        assertOrAssign(isCacheMode, curCacheMode, "all driver devices must have the same cache configuration");
        currDriverDeviceItr++;
    }
    return true;
}

void DeviceHandler::closeDevices()
{
    for (auto& device : m_driverDevices)
    {
        device->closeDevice();
        delete device;
    }

    if (m_simDevice != nullptr)
    {
        m_simDevice->closeDevice();
        delete m_simDevice;
    }
}

unsigned DeviceHandler::getDieNr()
{
    if (m_runOnChip)
    {
        //  all devices must have the same number of dies, simply query the first one.
        return m_driverDevices.front()->getNumDies();
    }
    else
    {
        return m_simDevice->getNumDies();
    }
}

DeviceHandler::DeviceHandler(DeviceType devA,
                             DeviceType devB,
                             std::vector<unsigned int>& deviceIdxs,
                             const MmeCommon::MmeHalReader& mmeHal)
: m_deviceIdxs(deviceIdxs), m_mmeHal(mmeHal)
{
    if (devA == e_sim || devB == e_sim)
    {
        m_runOnSim = true;
    }
    if (devA == e_chip || devB == e_chip)
    {
        m_runOnChip = true;
    }
    if (devA == e_null || devB == e_null)
    {
        m_runOnRef = true;
    }
}

DeviceHandler::~DeviceHandler()
{
    atomicColoredPrint(COLOR_YELLOW, "INFO: Closing devices.\n");
    closeDevices();
}

void DeviceHandler::configureMeshSniffersAndDumpDir(const EMmeDump poleDump,
                                                    const unsigned mmeIdx,
                                                    CoralMmeHBWSniffer& hbwSniffer,
                                                    CoralMmeLBWSniffer& lbwSniffer,
                                                    const std::string& dumpDir,
                                                    const std::string& dumpUnit)
{
    MME_ASSERT(m_runOnSim, "trying to set sniffers without simulated device");
    if (poleDump == EMmeDump::e_mme_dump_single)
    {
        MME_ASSERT(mmeIdx < m_mmeHal.getMmeNr(), "mem dump mme idx is out of bounds");
        configureMmeSniffer(mmeIdx, hbwSniffer, lbwSniffer);
        m_simDevice->getCluster()->setMmeDumpDir(mmeIdx, dumpDir, dumpUnit);
    }

    if (poleDump == EMmeDump::e_mme_dump_all)
    {
        unsigned mmeNr = m_mmeHal.getMmeNr();
        for (unsigned idx = 0; idx < mmeNr; idx++)
        {
            configureMmeSniffer(idx, hbwSniffer, lbwSniffer);
            m_simDevice->getCluster()->setMmeDumpDir(idx, dumpDir, dumpUnit);
        }
    }

    if (poleDump != EMmeDump::e_mme_dump_none)
    {
        m_simDevice->getCluster()->setHbwSniffer(&hbwSniffer);
        m_simDevice->getCluster()->setLbwSniffer(&lbwSniffer);
        hbwSniffer.enable();
        lbwSniffer.enable();
    }
}
}  // namespace MmeCommon
