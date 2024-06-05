#include <gtest/gtest.h>
#include "runtime/common/osal/osal.hpp"
#include "runtime/common/osal/physical_device_interface.hpp"

class PhysicalDeviceMock : public PhysicalDeviceInterface
{
public:
    virtual ~PhysicalDeviceMock() = default;

    virtual int open(hlthunk_device_name deviceName, const char *busid) override { return fdCompute; }

    virtual int openByModuleId(synModuleId moduleid) override { return fdCompute; }

    virtual int getPciBusIdFromFd(int fd, char* pci_bus_id, int len) override { return 0; }

    virtual int getDeviceIndexFromPciBusId(const char* busid) override { return 0; }

    virtual uint32_t getDevModuleIdx(int fd) override { return 5; };

    virtual uint32_t getDevIdType(int fd) override { return PCI_IDS_GAUDI2; };

    virtual int openControl(int dev_id, const char* busid) override { return contolFd; }

    virtual int openControlByModuleId(synModuleId module_id) override { return contolFd; }

    virtual int openControlByBusId(const char* busid) override { return contolFd; }

    virtual int getInfo(int fd, struct hl_info_args* info) override
    {
        ((hl_info_hw_ip_info*)(info->return_pointer))->device_id = PCI_IDS_GAUDI_SIMULATOR;
        info->return_size                                        = sizeof(hl_info_hw_ip_info);
        return 0;
    }

    virtual int getHwIpInfo(int fd, struct hlthunk_hw_ip_info *hw_ip) override
    {
        hw_ip->device_id = PCI_IDS_GAUDI_SIMULATOR;
        return 0;
    }

    int getDeviceStatus(int dev_id, bool& deviceActive, bool& deviceInRelease) override
    {
        deviceActive    = false;
        deviceInRelease = false;
        return 0;
    }

    int waitForDeviceOperational(int fd, std::chrono::seconds timeout_sec) override
    {
        return 0;
    }

    virtual int close(int fd) override { return 0; }

    int getComputeFd() { return fdCompute; }

    int getControlFd() { return contolFd; }

    const int fdCompute = 0;
    int       contolFd  = 1;
};

class UTOsalTest : public ::testing::Test
{
};

TEST_F(UTOsalTest, acquire_release_device)
{
    PhysicalDeviceInterface* pInterfaceOrig = OSAL::testGetInterface();
    PhysicalDeviceMock       mock;
    OSAL::testSetInterface(&mock);
    const char*         pciBus     = nullptr;
    const synDeviceType deviceType = synDeviceGaudi;
    const synModuleId   moduleId   = 0;
    synStatus           status     = OSAL::getInstance().acquireDevice(pciBus, deviceType, moduleId);
    EXPECT_EQ(status, synSuccess);
    status = OSAL::getInstance().releaseAcquiredDevice();
    EXPECT_EQ(status, synSuccess);

    mock.contolFd++;

    status = OSAL::getInstance().acquireDevice(pciBus, deviceType, moduleId);
    EXPECT_EQ(status, synSuccess);
    status = OSAL::getInstance().releaseAcquiredDevice();
    EXPECT_EQ(status, synSuccess);

    OSAL::testSetInterface(pInterfaceOrig);
}

TEST_F(UTOsalTest, acquire_multiple_devices_on_single_process)
{
    PhysicalDeviceInterface* pInterfaceOrig = OSAL::testGetInterface();
    PhysicalDeviceMock       mock;
    OSAL::testSetInterface(&mock);
    const char*         pciBus     = nullptr;
    const synDeviceType deviceType = synDeviceGaudi;
    synModuleId         moduleId   = 0;
    synStatus           status     = OSAL::getInstance().acquireDevice(pciBus, deviceType, moduleId);
    EXPECT_EQ(status, synSuccess);

    mock.contolFd++;
    moduleId    = 1;
    status      = OSAL::getInstance().acquireDevice(pciBus, deviceType, moduleId);
    EXPECT_EQ(status, synDeviceAlreadyAcquired);

    status = OSAL::getInstance().releaseAcquiredDevice();
    EXPECT_EQ(status, synSuccess);

    OSAL::testSetInterface(pInterfaceOrig);
}