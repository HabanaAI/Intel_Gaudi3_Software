#pragma once

#include "device_handler.h"

namespace gaudi3
{
class Gaudi3DeviceHandler : public MmeCommon::DeviceHandler
{
public:
    Gaudi3DeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs);

protected:
    virtual SimDeviceBase* getChipSimDevice(const uint64_t pqBaseAddr) override;
    virtual coral::DriverDeviceBase* getChipDriverDevice() override;
    virtual void
    configureMmeSniffer(const unsigned mmeIdx, CoralMmeHBWSniffer& hbwSniffer, CoralMmeLBWSniffer& lbwSniffer) override;
};
}  // namespace gaudi3