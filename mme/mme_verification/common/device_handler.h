#pragma once
// Sniffer
#include "include/mme_common/mme_common_enum.h"
#include "mme_coralhbw_sniffer.h"
#include "mme_corallbw_sniffer.h"
// Coral includes
#include "coral_user_utils.h"
#include "coral_user_program_base.h"
#include "coral_user_device.h"
#include "coral_user_driver_device_base.h"
#include "coral_user_simdev_base.h"

namespace MmeCommon
{
class MmeHalReader;

enum DeviceType
{
    e_null = 0,
    e_sim,
    e_chip
};

class DeviceHandler
{
public:
    DeviceHandler(DeviceType devA,
                  DeviceType devB,
                  std::vector<unsigned>& deviceIdxs,
                  const MmeCommon::MmeHalReader& mmeHal);
    virtual ~DeviceHandler();
    void createDevices(const uint64_t pqBaseAddr, unsigned simDieNr = 1);
    bool openDevices();
    bool openSimDevice();
    bool openDriverDevices();
    void closeDevices();
    SimDeviceBase* getSimDevice() { return m_simDevice; }
    bool isRunOnSim() const { return m_runOnSim; }
    bool isRunOnChip() const { return m_runOnChip; }
    bool isRunOnRef() const { return m_runOnRef; }
    std::vector<coral::DriverDeviceBase*>& getDriverDevices() { return m_driverDevices; }
    unsigned getNumOfDriverDevices() const { return m_driverDevices.size(); }
    unsigned getDieNr();

    void configureMeshSniffersAndDumpDir(const EMmeDump poleDump,
                                         const unsigned mmeIdx,
                                         CoralMmeHBWSniffer& hbwSniffer,
                                         CoralMmeLBWSniffer& lbwSniffer,
                                         const std::string& dumpDir,
                                         const std::string& dumpUnit);
    void setChipAlternative(bool value) {m_chipAlternative = value;}
protected:
    virtual SimDeviceBase* getChipSimDevice(const uint64_t pqBaseAddr) = 0;
    virtual coral::DriverDeviceBase* getChipDriverDevice() = 0;
    virtual void
    configureMmeSniffer(const unsigned mmeIdx, CoralMmeHBWSniffer& hbwSniffer, CoralMmeLBWSniffer& lbwSniffer) = 0;
    unsigned m_dieNr = 0;
    bool m_chipAlternative = false;

private:
    bool openDriverDevice(std::vector<coral::DriverDeviceBase*>& devices, unsigned idx);
    bool m_runOnSim = false;
    bool m_runOnChip = false;
    bool m_runOnRef = false;
    const MmeCommon::MmeHalReader& m_mmeHal;
    const std::vector<unsigned> m_deviceIdxs = {};
    SimDeviceBase* m_simDevice = nullptr;
    std::vector<coral::DriverDeviceBase*> m_driverDevices = {};
};
}  // namespace MmeCommon
