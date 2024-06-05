#include "physical_device.hpp"
#include "hlthunk.h"
#include "log_manager.h"
#include <limits>
#include <thread>
#include <unistd.h>
#include <chrono>
using namespace std::chrono_literals;

int PhysicalDevice::open(hlthunk_device_name deviceName, const char* busid)
{
    return hlthunk_open(deviceName, busid);
}

int PhysicalDevice::openByModuleId(synModuleId moduleid)
{
    return hlthunk_open_by_module_id(moduleid);
}

int PhysicalDevice::getPciBusIdFromFd(int fd, char* pci_bus_id, int len)
{
    return hlthunk_get_pci_bus_id_from_fd(fd, pci_bus_id, len);
}

int PhysicalDevice::getDeviceIndexFromPciBusId(const char* busid)
{
    return hlthunk_get_device_index_from_pci_bus_id(busid);
}

uint32_t PhysicalDevice::getDevModuleIdx(int fd)
{
    hlthunk_hw_ip_info hwIp;

    int ret = hlthunk_get_hw_ip_info(fd, &hwIp);

    if (ret < 0)
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return hwIp.module_id;
}

uint32_t PhysicalDevice::getDevIdType(int fd)
{
    hlthunk_hw_ip_info hwIp;

    int ret = hlthunk_get_hw_ip_info(fd, &hwIp);

    if (ret < 0)
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return hwIp.device_id;
}

int PhysicalDevice::openControl(int dev_id, const char* busid)
{
    return hlthunk_open_control(dev_id, busid);
}

int PhysicalDevice::openControlByModuleId(synModuleId module_id)
{
    return hlthunk_open_control_by_module_id(module_id);
}

int PhysicalDevice::openControlByBusId(const char* busid)
{
    return hlthunk_open_control_by_bus_id(busid);
}

int PhysicalDevice::getDeviceStatus(int dev_id, bool& deviceActive, bool& deviceInRelease)
{
    hlthunk_open_stats_info deviceCtrlInfo;
    int                     ret = hlthunk_get_open_stats(dev_id, &deviceCtrlInfo);

    deviceActive    = deviceCtrlInfo.is_compute_ctx_active;
    deviceInRelease = deviceCtrlInfo.compute_ctx_in_release;

    return ret;
}

int PhysicalDevice::waitForDeviceOperational(int fd, std::chrono::seconds timeout_sec)
{
    std::chrono::microseconds elapsed_usec = 0us;
    std::chrono::microseconds sleep_usec   = 500ms;
    std::chrono::microseconds timeout_usec = std::chrono::duration_cast<std::chrono::microseconds>(timeout_sec);
    int device_status;

    while (elapsed_usec < timeout_usec)
    {
        device_status = hlthunk_get_device_status_info(fd);
        if (device_status == HL_DEVICE_STATUS_OPERATIONAL) return 0;

        if (device_status < 0)
        {
            LOG_ERR(SYN_OSAL,
                    "hlthunk_get_device_status_info failed with rc {} error {} {}",
                    device_status,
                    errno,
                    std::strerror(errno));
            return device_status;
        }

        if (device_status != HL_DEVICE_STATUS_IN_RESET &&
            device_status != HL_DEVICE_STATUS_IN_RESET_AFTER_DEVICE_RELEASE)
            return device_status;

        elapsed_usec += sleep_usec;
        if (elapsed_usec < timeout_usec) std::this_thread::sleep_for(sleep_usec);
    }

    return -ETIMEDOUT;
}

int PhysicalDevice::getInfo(int fd, struct hl_info_args* info)
{
    return hlthunk_get_info(fd, info);
}

int PhysicalDevice::getHwIpInfo(int fd, struct hlthunk_hw_ip_info* hw_ip)
{
    return hlthunk_get_hw_ip_info(fd, hw_ip);
}

int PhysicalDevice::close(int fd)
{
    return hlthunk_close(fd);
}
