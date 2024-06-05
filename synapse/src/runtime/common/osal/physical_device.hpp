#pragma once

#include "physical_device_interface.hpp"

class PhysicalDevice : public PhysicalDeviceInterface
{
public:
    virtual ~PhysicalDevice() = default;

    virtual int open(hlthunk_device_name deviceName, const char* busid) override;

    virtual int openByModuleId(synModuleId moduleid) override;

    virtual int getPciBusIdFromFd(int fd, char* pci_bus_id, int len) override;

    virtual int getDeviceIndexFromPciBusId(const char* busid) override;

    virtual uint32_t getDevModuleIdx(int fd) override;

    virtual uint32_t getDevIdType(int fd) override;

    virtual int openControl(int dev_id, const char* busid) override;

    virtual int openControlByModuleId(synModuleId module_id) override;

    virtual int openControlByBusId(const char* busid) override;

    virtual int getInfo(int fd, struct hl_info_args* info) override;

    int getHwIpInfo(int fd, struct hlthunk_hw_ip_info* hw_ip) override;

    virtual int getDeviceStatus(int fd, bool& deviceActive, bool& deviceInRelease) override;

    virtual int waitForDeviceOperational(int fd, std::chrono::seconds timeout_sec) override;

    virtual int close(int fd) override;
};
