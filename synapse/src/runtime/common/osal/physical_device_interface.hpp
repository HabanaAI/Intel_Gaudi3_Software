#pragma once

#include "synapse_api_types.h"
#include "hlthunk.h"
#include <chrono>

class PhysicalDeviceInterface
{
public:
    virtual ~PhysicalDeviceInterface() = default;

    virtual int open(hlthunk_device_name deviceName, const char* busid) = 0;

    virtual int openByModuleId(synModuleId moduleid) = 0;

    virtual int getPciBusIdFromFd(int fd, char* pci_bus_id, int len) = 0;

    virtual int getDeviceIndexFromPciBusId(const char* busid) = 0;

    virtual uint32_t getDevModuleIdx(int fd) = 0;

    virtual uint32_t getDevIdType(int fd) = 0;

    virtual int openControl(int dev_id, const char* busid) = 0;

    virtual int openControlByModuleId(synModuleId module_id) = 0;

    virtual int openControlByBusId(const char* busid) = 0;

    virtual int getInfo(int fd, struct hl_info_args* info) = 0;

    virtual int getHwIpInfo(int fd, struct hlthunk_hw_ip_info* hw_ip) = 0;

    virtual int getDeviceStatus(int dev_id, bool& deviceActive, bool& deviceInRelease) = 0;

    virtual int waitForDeviceOperational(int fd, std::chrono::seconds timeout_sec) = 0;

    virtual int close(int fd) = 0;
};
