/*****************************************************************************
 * Copyright (C) 2016 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Tzachi Cohen <tcohen@gaudilabs.com>
 * Oded Gabbay <ogabbay@gaudilabs.com>
 ******************************************************************************
 */

#ifndef OSAL_HPP
#define OSAL_HPP

#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>

#include "synapse_types.h"
#include "common/pci_ids.h"
#include "hlthunk.h"
#include "define_synapse_common.hpp"

using namespace std::chrono_literals;

class BufferAllocator;
class PhysicalDeviceInterface;

typedef struct DevicePllInfo
{
    DevicePllInfo() : isValid(false) {}

    bool     isValid;
    uint32_t psocPciPllNr;
    uint32_t psocPciPllNf;
    uint32_t psocPciPllOd;
    uint32_t psocPciPllDivFactor1;
} DevicePllInfo;

typedef struct DeviceClockRateInfo
{
    uint32_t currentClockRate;
    uint32_t maxClockRate;
} DeviceClockRateInfo;

typedef enum
{
    DEVICE_SUB_TYPE_A,
    DEVICE_SUB_TYPE_B,
    DEVICE_SUB_TYPE_C,
    DEVICE_SUB_TYPE_D
} synDeviceSubType;

class OSAL
{
public:
    static OSAL& getInstance()
    {
        static OSAL s_hal;
        return s_hal;
    }
    OSAL(OSAL&&)  = delete;
    OSAL& operator=(OSAL&&) = delete;

    typedef uint64_t                                                   DeviceVirtualAddress;
    typedef std::unordered_map<DeviceVirtualAddress, BufferAllocator*> DeviceVirtualAddressMap;
    typedef std::unordered_map<std::string, synDeviceType>             DeviceNameToDeviceTypeMap;

    void* mapMem(size_t len, off_t offset);
    int   unmapMem(void* addr, size_t len);

    synStatus getDeviceInfoByModuleId(const synDeviceInfo*& pDeviceInfo,
                                      uint32_t&             currentClockRate,
                                      const synModuleId     moduleId);

    synStatus acquireDevice(const char* pciBus, synDeviceType deviceType, synModuleId moduleId);
    synStatus releaseAcquiredDevice();

    synStatus setBufferAllocator(DeviceVirtualAddress deviceVA, BufferAllocator* pBufferAllocator);
    synStatus getBufferAllocator(DeviceVirtualAddress deviceVA, BufferAllocator** ppBufferAllocator);
    synStatus clearBufferAllocator(DeviceVirtualAddress deviceVA);
    synStatus releaseAcquiredDeviceBuffers();

    int           getFd() const;
    synDeviceType getDeviceType() const;

    synStatus          GetDeviceInfo(synDeviceInfo& deviceInfo);
    synStatus          getDeviceLimitationInfo(synDeviceLimitationInfo& deviceLimitationInfo);
    synStatus          GetDeviceControlFd(int& rFdControl);
    synStatus          getDeviceClockRateInfo(DeviceClockRateInfo& deviceClockRateInfo);
    synStatus          getDeviceHlIdx(uint32_t& hlIdx);
    synStatus          getDeviceModuleIdx(uint32_t& moduleIdx);
    synStatus          getDeviceIdType(uint32_t& devIdType);
    synStatus          getPciAddr(std::string& pciAddr);
    synStatus          getDeviceName(char* pName, const int len);
    synStatus          getPCIBusId(char* pPciBusId, const int len);
    static std::string GetDeviceNameByDevType(synDeviceType devType);

    int getPageSize() const;

    unsigned getDevicesCount(synDeviceType deviceType = synDeviceTypeSize);

    inline bool isAcquired() const { return m_isAcquired; }

    synStatus writeI2cReg(uint32_t deviceId, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t value);
    synStatus
    readI2cReg(uint32_t deviceId, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t* pValue);
    synStatus setLedState(uint32_t deviceId, uint32_t ledId, bool state);
    synStatus setFrequency(uint32_t deviceId, uint32_t pllId, uint32_t frequency);
    synStatus getFrequency(uint32_t deviceId, uint32_t pllId, uint32_t* pFrequency);

    synStatus getAvailableNicElementsMask(uint32_t& availableNicMask);

    static void                     testSetInterface(PhysicalDeviceInterface* interface) { s_interface = interface; }
    static PhysicalDeviceInterface* testGetInterface() { return s_interface; }

    uint32_t getTotalHugePages();
    uint32_t getFreeHugePagesAtAppStart();
    void     printHugePageInfo();
    void     GetModuleIDsList(uint32_t *pDeviceModuleIds, uint32_t*  size);

private:
    friend class SynApiTests;

    OSAL();
    ~OSAL();

    // Open device based on module ID
    synStatus openByModuleID(synModuleId moduleId, synDeviceType deviceType);

    // Open device based on PCI bus ID
    synStatus openByPciID(const char* pciBus, synDeviceType deviceType);

    // Open any device
    synStatus openAnyDevice(synDeviceType deviceType);

    // Open control FD based on module ID
    synStatus openControlFD(synModuleId moduleId);

    // Open control FD based on PCI bus ID
    synStatus openControlFD(const char* pciBus);

    // Open control FD based on any module ID
    synStatus openControlFDAny(synModuleId& rModuleId);

    // Open compute FD based on module ID
    synStatus openComputeFD(synModuleId moduleId);

    // Open compute FD based on PCI bus ID
    synStatus openComputeFD(const char* pciBus);

    // Close control and compute FDs
    synStatus closeFD();

    synStatus _getDeviceInfo(synDeviceInfo& deviceInfo, int fd, synDeviceLimitationInfo& pDeviceLimitationInfo);
    synStatus _getPllId(uint32_t pllId, std::string& kmdPllName);

    synStatus _destroyDeviceBuffers();
    synStatus _closeDeviceFD();

    synStatus _waitForCloseComputeDevice(int controlFD, bool isOpen);

    std::vector<hlthunk_device_name> _getHlThunkDevices(synDeviceType    deviceType,
                                                        synDeviceSubType subType = DEVICE_SUB_TYPE_A);

    static void _updateDeviceCount(uint32_t                         outCount[synDeviceTypeSize],
                                   unsigned&                        numberOfDevices,
                                   const DeviceNameToDeviceTypeMap& nameToDeviceType,
                                   const std::string&               buffer,
                                   bool                             shouldStripSimulator);

    synStatus deviceInfoSet(synDeviceType deviceType);
    void      deviceInfoReset();

    static std::pair<bool, std::string> runCmd(const char* cmd);

    synStatus getDeviceParentName(std::string& parentName, uint32_t deviceId);

    static int returnNumberFromLine(const std::string& input);

    void logAcquiredDevicePid(synModuleId moduleId, std::string* deviceFailLog = nullptr);

    std::string checkExposedLinuxDevicesDir();

    std::string logLinuxDevicesDirOutputByType(std::string devicePath);

    typedef std::unordered_map<synModuleId, synDeviceInfo> DeviceInfoByModuleIdDB;

    int                                      m_controlFD    = -1;
    int                                      m_computeFD    = -1;
    synDeviceInfo                            m_deviceInfo {};
    DeviceInfoByModuleIdDB                   m_deviceInfoByModuleId;
    synDeviceLimitationInfo                  m_deviceLimitationInfo {};
    DevicePllInfo                            m_devicePllClockInfo {};
    std::string                              m_pciAddr      = "";
    unsigned                                 m_devHlIdx     = -1;
    uint32_t                                 m_devModuleIdx = -1;
    uint32_t                                 m_devIdType    = -1;
    std::unique_ptr<DeviceVirtualAddressMap> m_deviceVirtualAddressMap {};
    std::mutex                               m_vaMapLock;
    std::atomic<bool>                        m_isAcquired = false;
    uint32_t                                 m_totalHugePagesNum;
    uint32_t                                 m_freeHugePagesInitialized;
    uint32_t                                 m_HugePageSize;
    static constexpr std::chrono::seconds    c_waitTimForReleaseTriesSec = 60s;
    static PhysicalDeviceInterface*          s_interface;
};

#endif /* OSAL_HPP */
