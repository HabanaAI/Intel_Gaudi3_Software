#include "device_info_interface.hpp"

#include "defs.h"

using namespace common;

uint64_t DeviceInfoInterface::getMonitorPayloadAddrHighRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const
{
    return getSmBase(dcoreId) + getMonitorPayloadAddrHighRegisterSmOffset(monitorId);
}

uint64_t DeviceInfoInterface::getMonitorPayloadAddrLowRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const
{
    return getSmBase(dcoreId) + getMonitorPayloadAddrLowRegisterSmOffset(monitorId);
}

uint64_t DeviceInfoInterface::getMonitorPayloadDataRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const
{
    return getSmBase(dcoreId) + getMonitorPayloadDataRegisterSmOffset(monitorId);
}

uint64_t DeviceInfoInterface::getMonitorConfigRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const
{
    return getSmBase(dcoreId) + getMonitorConfigRegisterSmOffset(monitorId);
}

uint64_t DeviceInfoInterface::getMonitorArmRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const
{
    return getSmBase(dcoreId) + getMonitorArmRegisterSmOffset(monitorId);
}
