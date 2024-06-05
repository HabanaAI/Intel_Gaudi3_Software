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

#ifndef _WIN32

#include <dirent.h>
#include <string>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <regex>
#ifdef USE_VALGRIND
#include <valgrind/memcheck.h>
#endif

#include "syn_logging.h"
#include "buffer_allocator.hpp"
#include "osal.hpp"
#include "synapse_runtime_logging.h"
#include "defs.h"
#include "defenders.h"
#include "physical_device.hpp"
#include "habana_global_conf.h"

using namespace std;

PhysicalDeviceInterface* OSAL::s_interface = new PhysicalDevice();

OSAL::OSAL()
{
    // init huge pages data
    auto hugePagesInfo  = runCmd("grep -E -i 'HugePages_|Hugepagesize' /proc/meminfo");
    uint32_t    totalHp = 0, freeHp = 0, hpSize = 0;
    if (hugePagesInfo.first)
    {
        std::stringstream ss(hugePagesInfo.second);
        std::string       line;
        while (std::getline(ss, line, '\n'))
        {
            if (line.find("Total") != string::npos)
            {
                totalHp = returnNumberFromLine(line);
            }
            else if (line.find("Free") != string::npos)
            {
                freeHp = returnNumberFromLine(line);
            }
            else if (line.find("size") != string::npos)
            {
                hpSize = returnNumberFromLine(line);
            }
        }
    }

    m_deviceInfo.fd            = -1;
    m_totalHugePagesNum        = totalHp;
    m_freeHugePagesInitialized = freeHp;
    m_HugePageSize             = hpSize;
}

OSAL::~OSAL()
{
    delete s_interface;
}

synStatus OSAL::getDeviceInfoByModuleId(const synDeviceInfo*& pDeviceInfo,
                                        uint32_t&             currentClockRate,
                                        const synModuleId     moduleId)
{
    synStatus retStatus = synSuccess;
    std::string exposedLinuxDevicesDirOutput = checkExposedLinuxDevicesDir();

    auto deviceInfoIter = m_deviceInfoByModuleId.find(moduleId);
    if (deviceInfoIter != m_deviceInfoByModuleId.end())
    {
        pDeviceInfo = &(deviceInfoIter->second);
        return synSuccess;
    }

    // Open control FD
    const int controlFD = s_interface->openControlByModuleId(moduleId);
    if (controlFD < 0)
    {
        LOG_ERR(SYN_OSAL, "{} failed to get control FD for module ID {} controlFD {}", HLLOG_FUNC, moduleId, controlFD);
        LOG_ERR(SYN_OSAL, "{}", exposedLinuxDevicesDirOutput);
        return synNoDeviceFound;
    }
    LOG_INFO(SYN_OSAL, "FD control {} is open", controlFD);

    // Get Device-Info
    synDeviceLimitationInfo deviceLimitationInfo {}; // At the moment, this is not required for non-allocated devices
    synDeviceInfo&          deviceInfo = m_deviceInfoByModuleId[moduleId];
    synStatus               status     = _getDeviceInfo(deviceInfo, controlFD, deviceLimitationInfo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "{} Failed to get device-info (status {})", HLLOG_FUNC, status);

        m_deviceInfoByModuleId.erase(moduleId);
        retStatus = synNoDeviceFound;
    }

    // Get CLK-Rate info
    DeviceClockRateInfo deviceClockRateInfo;
    int rc =
        hlthunk_get_clk_rate(m_controlFD, &deviceClockRateInfo.currentClockRate, &deviceClockRateInfo.maxClockRate);
    if (rc != 0)
    {
        LOG_ERR(SYN_OSAL, "Get-Clock-Rate ioctl failed");
        retStatus = synDeviceReset;
    }
    currentClockRate = deviceClockRateInfo.currentClockRate;

    // Close control FD
    //
    // If LKD is currently under reset as of a previous release, we would like to wait until it is finished
    status = _waitForCloseComputeDevice(controlFD, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "_waitForCloseComputeDevice failed with synStatus {}", status);
        retStatus = status;
    }
    s_interface->close(controlFD);

    // Handle failures
    if (retStatus != synSuccess)
    {
        m_deviceInfoByModuleId.erase(moduleId);
    }

    pDeviceInfo = &deviceInfo;
    return retStatus;
}

synStatus OSAL::acquireDevice(const char* pciBus, synDeviceType deviceType, synModuleId moduleId)
{
    if (isAcquired())
    {
        return synDeviceAlreadyAcquired;
    }

    synStatus status;

    if (moduleId != INVALID_MODULE_ID)
    {
        status = openByModuleID(moduleId, deviceType);
    }
    else if (pciBus != nullptr && std::string(pciBus) != "")
    {
        status = openByPciID(pciBus, deviceType);
    }
    else
    {
        status = openAnyDevice(deviceType);
    }

    return status;
}

synStatus OSAL::openByModuleID(synModuleId moduleId, synDeviceType deviceType)
{
    synStatus status = openControlFD(moduleId);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Open control FD failed with status {}, API called: {}", status, __FUNCTION__);
        return status;
    }

    LOG_INFO(SYN_OSAL, "FD control {} is open", m_controlFD);

    status = deviceInfoSet(deviceType);
    if (status != synSuccess)
    {
        closeFD();
        return status;
    }

    status = openComputeFD(moduleId);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Open compute FD failed with status {}, API called: {}", status, __FUNCTION__);
        logAcquiredDevicePid(moduleId);
        closeFD();
        deviceInfoReset();
        return status;
    }

    LOG_INFO(SYN_OSAL, "FD compute {} is open", m_computeFD);

    m_isAcquired = true;

    return synSuccess;
}

synStatus OSAL::openByPciID(const char* pciBus, synDeviceType deviceType)
{
    synStatus status = openControlFD(pciBus);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Open control FD failed with status {}, API called: {}", status, __FUNCTION__);
        return status;
    }

    LOG_INFO(SYN_OSAL, "FD control {} is open", m_controlFD);

    status = deviceInfoSet(deviceType);
    if (status != synSuccess)
    {
        closeFD();
        return status;
    }

    status = openComputeFD(pciBus);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Open compute FD failed with status {}, API called: {}", status, __FUNCTION__);
        logAcquiredDevicePid(m_devModuleIdx);
        closeFD();
        deviceInfoReset();
        return status;
    }

    LOG_INFO(SYN_OSAL, "FD compute {} is open", m_computeFD);

    m_isAcquired = true;

    return synSuccess;
}

synStatus OSAL::openAnyDevice(synDeviceType deviceType)
{
    std::string failedAcquiredDevices[MAX_NUM_OF_DEVICES_PER_HOST] = { "" };
    int controlFD = -1;
    int computeFD = -1;

    std::string exposedLinuxDevicesDirOutput = checkExposedLinuxDevicesDir();

    // Iterate 8 devices since this is the max amount on a HLS box
    for (synModuleId moduleId = 0; moduleId < MAX_NUM_OF_DEVICES_PER_HOST; moduleId++)
    {
        controlFD = s_interface->openControlByModuleId(moduleId);
        if (controlFD == -EBADF)
        {
            LOG_DEBUG(SYN_OSAL,
                      "{} device with moduleId {} was not found (Bad file descriptor). Continue to the next module.",
                      HLLOG_FUNC,
                      moduleId);
            continue;
        }

        if (controlFD < 0)
        {
            LOG_ERR(SYN_OSAL,
                    "{} failed to get control FD for moduleId {} controlFD {}, API called: {}",
                    HLLOG_FUNC,
                    moduleId,
                    controlFD,
                    __FUNCTION__);
            return synFail;
        }

        // If LKD is currently under reset as of a previous release, we would like to wait until it is finished and
        // then try to allocate
        synStatus status = _waitForCloseComputeDevice(controlFD, true);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_OSAL, "_waitForCloseComputeDevice failed with synStatus {}.", status);
            s_interface->close(controlFD);
            return status;
        }

        m_controlFD = controlFD;
        status = deviceInfoSet(deviceType);
        if (status != synSuccess)
        {
            s_interface->close(m_controlFD);
            m_controlFD = -1;
            return status;
        }

        computeFD = s_interface->openByModuleId(moduleId);
        if (computeFD < 0)
        {
            logAcquiredDevicePid(moduleId, &failedAcquiredDevices[moduleId]);
            deviceInfoReset();
            closeFD();
            continue;
        }
        else
        {
            m_computeFD = computeFD;
            m_deviceInfo.fd = computeFD;
            LOG_INFO(SYN_OSAL, "FD control {} is open", m_controlFD);
            LOG_INFO(SYN_OSAL, "FD compute {} is open", m_computeFD);
            m_isAcquired = true;
            return synSuccess;
        }
    }

    LOG_ERR(SYN_OSAL, "No available device to allocate. FD from thunk call: controlFD: {} computeFD: {} error reason: {}",
            controlFD,
            computeFD,
            controlFD < 0 ? strerror(controlFD * -1) : strerror(controlFD));

    for (synModuleId moduleId = 0; moduleId < MAX_NUM_OF_DEVICES_PER_HOST; moduleId++)
    {
        if (failedAcquiredDevices[moduleId] != "")
        {
            LOG_ERR(SYN_OSAL, "{}", failedAcquiredDevices[moduleId]);
        }
    }

    LOG_ERR(SYN_OSAL, "{}", exposedLinuxDevicesDirOutput);
    return synNoDeviceFound;
}

synStatus OSAL::openControlFD(synModuleId moduleId)
{
    std::string exposedLinuxDevicesDirOutput = checkExposedLinuxDevicesDir();
    const int controlFD = s_interface->openControlByModuleId(moduleId);
    if (controlFD < 0)
    {
        LOG_ERR(SYN_OSAL, "{} failed to get control FD for module ID {} controlFD {}", HLLOG_FUNC, moduleId, controlFD);
        LOG_ERR(SYN_OSAL, "{}", exposedLinuxDevicesDirOutput);
        return synNoDeviceFound;
    }
    // If LKD is currently under reset as of a previous release, we would like to wait until it is finished and then
    // try to allocate
    synStatus status = _waitForCloseComputeDevice(controlFD, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "_waitForCloseComputeDevice failed with synStatus {}.", status);
        s_interface->close(controlFD);
        return status;
    }
    m_controlFD = controlFD;

    return synSuccess;
}

synStatus OSAL::openControlFD(const char* pciBus)
{
    std::string exposedLinuxDevicesDirOutput = checkExposedLinuxDevicesDir();
    const int controlFD = s_interface->openControlByBusId(pciBus);
    if (controlFD < 0)
    {
        LOG_ERR(SYN_OSAL, "{} failed to get control FD for bus ID {} controlFD {}", HLLOG_FUNC, pciBus, controlFD);
        LOG_ERR(SYN_OSAL, "{}", exposedLinuxDevicesDirOutput);
        return synNoDeviceFound;
    }

    // If LKD is currently under reset as of a previous release, we would like to wait until it is finished and then
    // try to allocate
    synStatus status = _waitForCloseComputeDevice(controlFD, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "_waitForCloseComputeDevice failed with synStatus {}.", status);
        s_interface->close(controlFD);
        return status;
    }
    m_controlFD = controlFD;

    return synSuccess;
}

synStatus OSAL::openComputeFD(synModuleId moduleId)
{
    const int computeFD = s_interface->openByModuleId(moduleId);
    if (computeFD < 0)
    {
        LOG_ERR(SYN_OSAL, "{} failed to get compute FD by module ID {} computeFD {}", HLLOG_FUNC, moduleId, computeFD);
        return synNoDeviceFound;
    }

    m_computeFD     = computeFD;
    m_deviceInfo.fd = m_computeFD;

    return synSuccess;
}

synStatus OSAL::openComputeFD(const char* pciBus)
{
    const int computeFD = s_interface->open(HLTHUNK_DEVICE_INVALID, pciBus);
    if (computeFD < 0)
    {
        LOG_ERR(SYN_OSAL, "{} failed to get compute FD by bus ID {} computeFD {}", HLLOG_FUNC, pciBus, computeFD);
        return synFail;
    }

    m_computeFD     = computeFD;
    m_deviceInfo.fd = m_computeFD;

    return synSuccess;
}

synStatus OSAL::openControlFDAny(synModuleId& rModuleId)
{
    int         controlFD = -1;
    synModuleId moduleId  = 0;

    std::string exposedLinuxDevicesDirOutput = checkExposedLinuxDevicesDir();

    // Iterate 8 devices since this is the max amount on a HLS box
    for (; moduleId < MAX_NUM_OF_DEVICES_PER_HOST; moduleId++)
    {
        controlFD = s_interface->openControlByModuleId(moduleId);
        if (controlFD == -EBADF)
        {
            LOG_DEBUG(SYN_OSAL,
                      "{} device with moduleId {} was not found (Bad file descriptor). Continue to the next module.",
                      HLLOG_FUNC,
                      moduleId);
            continue;
        }

        if (controlFD < 0)
        {
            LOG_ERR(SYN_OSAL,
                    "{} failed to get control FD for moduleId {} controlFD {}",
                    HLLOG_FUNC,
                    moduleId,
                    controlFD);
            return synFail;
        }

        break;
    }

    if (moduleId == MAX_NUM_OF_DEVICES_PER_HOST)
    {
        LOG_ERR(SYN_OSAL, "no habanalabs devices were found in this VM (Bad file descriptor)");
        LOG_ERR(SYN_OSAL, "{}", exposedLinuxDevicesDirOutput);
        return synNoDeviceFound;
    }

    LOG_INFO(SYN_OSAL, "moduleId {}", moduleId);
    rModuleId = moduleId;

    // If LKD is currently under reset as of a previous release, we would like to wait until it is finished and
    // then try to allocate
    synStatus status = _waitForCloseComputeDevice(controlFD, true);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "_waitForCloseComputeDevice failed with synStatus {}.", status);
        s_interface->close(controlFD);
        return status;
    }

    m_controlFD = controlFD;

    return synSuccess;
}

synStatus OSAL::deviceInfoSet(synDeviceType deviceType)
{
    synDeviceInfo           deviceInfo {};
    synDeviceLimitationInfo deviceLimitationInfo {};
    synStatus               status = _getDeviceInfo(deviceInfo, m_controlFD, deviceLimitationInfo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "{} No HabanaLabs devices found (status {})", HLLOG_FUNC, status);
        return synNoDeviceFound;
    }

    switch (deviceInfo.deviceType)
    {
        case synDeviceGaudi:
        case synDeviceGaudi2:
        case synDeviceGaudi3:
            break;
        default:
        {
            LOG_ERR(SYN_OSAL, "{} Invalid device type {}", HLLOG_FUNC, deviceInfo.deviceType);
            return synInvalidArgument;
        }
    }

    if ((deviceType != synDeviceTypeInvalid) && (deviceInfo.deviceType != deviceType))
    {
        LOG_ERR(SYN_OSAL,
                "User requested device type {} is not available. (Available device type: {})",
                deviceType,
                deviceInfo.deviceType);
        return synDeviceTypeMismatch;
    }

    m_deviceInfo = deviceInfo;
    // The compute FD is unknown at this stage and it will be set later once the control FD is opened
    m_deviceInfo.fd        = -1;
    m_deviceLimitationInfo = deviceLimitationInfo;

    // Retrieve all info for the device
    char busId[13];
    int  res = s_interface->getPciBusIdFromFd(m_controlFD, busId, sizeof(busId));
    if (res != 0)
    {
        LOG_ERR(SYN_OSAL, "Failed to get busId from fd {}", m_controlFD);
        deviceInfoReset();
        return synFail;
    }
    std::string pciAddr(busId);
    LOG_TRACE(SYN_OSAL, "PciBusId {}", busId);
    m_pciAddr = pciAddr;

    int devHlIdx = s_interface->getDeviceIndexFromPciBusId(busId);
    if (devHlIdx < 0)
    {
        LOG_ERR(SYN_OSAL, "Failed to get device index from busId {}", busId);
        deviceInfoReset();
        return synFail;
    }
    LOG_TRACE(SYN_OSAL, "devHlIdx {}", devHlIdx);
    m_devHlIdx = devHlIdx;

    uint32_t devModuleIdx = s_interface->getDevModuleIdx(m_controlFD);
    if (devModuleIdx == std::numeric_limits<uint32_t>::max())
    {
        LOG_ERR(SYN_OSAL, "Failed to get devModuleIdx, fd {}", m_controlFD);
        deviceInfoReset();
        return synFail;
    }
    LOG_TRACE(SYN_OSAL, "devModuleIdx {}", devModuleIdx);
    m_devModuleIdx = devModuleIdx;

    uint32_t devIdType;
    devIdType = s_interface->getDevIdType(m_controlFD);
    // Todo delete m_devIdType or enable this code
    // if (devIdType == std::numeric_limits<uint32_t>::max())
    //{
    //    LOG_ERR(SYN_OSAL, "Failed to get getDevIdType, fd {}", m_controlFD);
    //    deviceInfoReset();
    //    return synFail;
    //}
    LOG_TRACE(SYN_OSAL, "devIdType {}", devIdType);
    m_devIdType = devIdType;

    // Create an entity
    {
        std::lock_guard<std::mutex> lock(m_vaMapLock);
        m_deviceVirtualAddressMap = std::make_unique<DeviceVirtualAddressMap>();
    }

    return synSuccess;
}

int OSAL::getFd() const
{
    if (!isAcquired())
    {
        LOG_ERR(SYN_OSAL, "Device not opened so has no file descriptor");
        return -1;
    }

    return m_computeFD;
}

synDeviceType OSAL::getDeviceType() const
{
    if (!isAcquired())
    {
        LOG_ERR(SYN_OSAL, "Device not opened so has no type");
        return synDeviceTypeSize;
    }

    return m_deviceInfo.deviceType;
}

void* OSAL::mapMem(size_t len, off_t offset)
{
    void* pResource = nullptr;

    if (!isAcquired())
    {
        LOG_ERR(SYN_OSAL, "Device not opened so can not call mmap !!!");
        return nullptr;
    }

    pResource = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, m_computeFD, offset);

    if (pResource == MAP_FAILED)
    {
        LOG_ERR(SYN_OSAL, "Can not map on device, size {}. Got errno {}", len, errno);
        pResource = nullptr;
    }

    return pResource;
}

uint32_t OSAL::getTotalHugePages()
{
    return m_totalHugePagesNum;
}

uint32_t OSAL::getFreeHugePagesAtAppStart()
{
    return m_freeHugePagesInitialized;
}

void OSAL::printHugePageInfo()
{
    LOG_INFO(SYN_OSAL,
             "Hugepage info: total: {}, free: {}, singleHpSize {}KB",
             m_totalHugePagesNum,
             m_freeHugePagesInitialized,
             m_HugePageSize);
}

int OSAL::returnNumberFromLine(const std::string& input)
{
    std::string output = std::regex_replace(input, std::regex("[^0-9]*([0-9]+).*"), std::string("$1"));
    return stoi(output);
}

std::pair<bool, std::string> OSAL::runCmd(const char* cmd)
{
    std::array<char, 128>                    buffer;
    std::string                              result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        LOG_ERR(SYN_OSAL, "popen failed");
        return {false, "Failed to run command"};
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return {true, result};
}

synStatus OSAL::getDeviceParentName(std::string& parentName, uint32_t deviceId)
{
    std::string cmd = "cat /sys/class/accel/accel" + std::to_string(deviceId) + "/device/parent_device";
    auto deviceName = runCmd(cmd.c_str());
    if (!deviceName.first)
    {
        LOG_ERR(SYN_OSAL, "Failed to get parent device name");
        return synFail;
    }

    parentName = deviceName.second;

    size_t endLineCharPos = parentName.find('\n');
    if (endLineCharPos != std::string::npos)
    {
        parentName.erase(endLineCharPos, 1);
    }

    return synSuccess;
}

int OSAL::unmapMem(void* addr, size_t len)
{
    int status = munmap(addr, len);

    return status;
}

synStatus OSAL::setBufferAllocator(DeviceVirtualAddress deviceVA, BufferAllocator* pBufferAllocator)
{
    synStatus status;
    LOG_TRACE(SYN_OSAL, "{} Taking m_vaMapLock on thread ID 0x{:x}", HLLOG_FUNC, (uint64_t)pthread_self());
    std::lock_guard<std::mutex> lock(m_vaMapLock);
    if (m_deviceVirtualAddressMap.get() == nullptr)
    {
        LOG_WARN(SYN_OSAL, "Invalid argument for {}", HLLOG_FUNC);
        status = synInvalidArgument;
    }
    else
    {
        DeviceVirtualAddressMap& deviceVAMap = *(m_deviceVirtualAddressMap.get());
        if (deviceVAMap.find(deviceVA) != deviceVAMap.end())
        {
            LOG_WARN(SYN_OSAL, "BufferAllocator already set");
            status = synInvalidArgument;
        }
        else
        {
            deviceVAMap[deviceVA] = pBufferAllocator;
            status                = synSuccess;
        }
    }

    return status;
}

synStatus OSAL::getBufferAllocator(DeviceVirtualAddress deviceVA, BufferAllocator** ppBufferAllocator)
{
    synStatus status;
    LOG_TRACE(SYN_OSAL, "{} Taking m_vaMapLock on thread ID 0x{:x}", HLLOG_FUNC, (uint64_t)pthread_self());
    std::lock_guard<std::mutex> lock(m_vaMapLock);
    if ((m_deviceVirtualAddressMap.get() == nullptr) || (ppBufferAllocator == nullptr) ||
        m_deviceVirtualAddressMap.get()->count(deviceVA) == 0)
    {
        LOG_WARN(SYN_OSAL, "Invalid argument for {}", HLLOG_FUNC);
        status = synInvalidArgument;
    }
    else
    {
        *ppBufferAllocator = m_deviceVirtualAddressMap.get()->at(deviceVA);
        status             = synSuccess;
    }

    return status;
}

synStatus OSAL::clearBufferAllocator(DeviceVirtualAddress deviceVA)
{
    LOG_TRACE(SYN_OSAL, "{} Taking m_vaMapLock on thread ID 0x{:x}", HLLOG_FUNC, (uint64_t)pthread_self());
    std::lock_guard<std::mutex> lock(m_vaMapLock);
    if (m_deviceVirtualAddressMap.get() == nullptr || m_deviceVirtualAddressMap.get()->count(deviceVA) == 0)
    {
        LOG_WARN(SYN_API, "{}: deviceVA {} was not found in DB, can not clear it", HLLOG_FUNC, deviceVA);
        return synInvalidArgument;
    }

    m_deviceVirtualAddressMap.get()->erase(deviceVA);

    LOG_TRACE(SYN_API, "erased deviceVAMapIter for {} successfully", deviceVA);

    return synSuccess;
}

synStatus OSAL::releaseAcquiredDeviceBuffers()
{
    LOG_TRACE(SYN_OSAL, "{} Taking m_vaMapLock on thread ID 0x{:x}", HLLOG_FUNC, (uint64_t)pthread_self());
    std::lock_guard<std::mutex> lock(m_vaMapLock);
    if (_destroyDeviceBuffers() != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Failed to destroy all device-buffers of released device");
        return synFail;
    }

    m_deviceVirtualAddressMap.reset();

    return synSuccess;
}

synStatus OSAL::releaseAcquiredDevice()
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    m_isAcquired = false;

    deviceInfoReset();

    if (closeFD() != synSuccess)
    {
        LOG_ERR(SYN_OSAL, "Failed to close FD of released device");
        return synFail;
    }

    return synSuccess;
}

synStatus OSAL::GetDeviceInfo(synDeviceInfo& deviceInfo)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    deviceInfo = m_deviceInfo;
    return synSuccess;
}

synStatus OSAL::getDeviceLimitationInfo(synDeviceLimitationInfo& deviceLimitationInfo)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    deviceLimitationInfo = m_deviceLimitationInfo;
    return synSuccess;
}

synStatus OSAL::GetDeviceControlFd(int& rFdControl)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    rFdControl = m_controlFD;
    if (rFdControl > 0)
    {
        return synSuccess;
    }
    return synFail;
}

synStatus OSAL::getDeviceHlIdx(uint32_t& hlIdx)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    hlIdx = m_devHlIdx;

    if (hlIdx >= 0)
    {
        return synSuccess;
    }
    return synNoDeviceFound;
}

synStatus OSAL::getDeviceModuleIdx(uint32_t& moduleIdx)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    moduleIdx = m_devModuleIdx;

    if (moduleIdx == -1)
    {
        return synNoDeviceFound;
    }
    return synSuccess;
}

synStatus OSAL::getDeviceIdType(uint32_t& devIdType)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    devIdType = m_devIdType;

    if (devIdType == std::numeric_limits<uint32_t>::max())
    {
        return synNoDeviceFound;
    }
    return synSuccess;
}

synStatus OSAL::getPciAddr(std::string& pciAddr)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    pciAddr = m_pciAddr;

    return synSuccess;
}

synStatus OSAL::getDeviceClockRateInfo(DeviceClockRateInfo& deviceClockRateInfo)
{
    if (!isAcquired())
    {
        return synNoDeviceFound;
    }

    int rc =
        hlthunk_get_clk_rate(m_controlFD, &deviceClockRateInfo.currentClockRate, &deviceClockRateInfo.maxClockRate);
    if (rc != 0)
    {
        LOG_ERR(SYN_OSAL, "Get-Clock-Rate ioctl failed");
        return synDeviceReset;
    }

    return synSuccess;
}

int OSAL::getPageSize() const
{
    return getpagesize();
}

synStatus OSAL::getPCIBusId(char* pPciBusId, const int len)
{
    int fd = getFd();
    if (fd == -1)
    {
        LOG_ERR(SYN_OSAL, "Device not opened so has no file descriptor");
        return synNoDeviceFound;
    }

    int rc = s_interface->getPciBusIdFromFd(fd, pPciBusId, len);
    if (rc < 0)
    {
        LOG_ERR(SYN_OSAL, "Failed to get pci bud ID from FD {} with rc {}", fd, rc);
        return synDeviceReset;
    }
    return synSuccess;
}

synStatus OSAL::getDeviceName(char* pName, const int len)
{
    if (!isAcquired())
    {
        LOG_ERR(SYN_OSAL, "Device not opened so has no file descriptor");
        return synNoDeviceFound;
    }

    string deviceName = GetDeviceNameByDevType(m_deviceInfo.deviceType);

    std::strncpy(pName, deviceName.c_str(), len);
    if (deviceName.size() >= len)
    {
        pName[len - 1] = '\0';
        LOG_WARN(SYN_OSAL, "{}: Given pName lentgth is shorter than real device name", HLLOG_FUNC);
    }

    return synSuccess;
}

unsigned OSAL::getDevicesCount(synDeviceType deviceType)
{
    int devicesCount = 0;
    int tmpCount     = 0;

    switch (deviceType)
    {
        case synDeviceTypeSize:
            devicesCount = hlthunk_get_device_count(HLTHUNK_DEVICE_DONT_CARE);
            if (devicesCount < 0)
            {
                LOG_DEBUG(SYN_OSAL, "hlthunk_get_device_count failed for total count (HLTHUNK_DEVICE_DONT_CARE)");
            }
            break;
        case synDeviceGaudi2:
            tmpCount = hlthunk_get_device_count(HLTHUNK_DEVICE_GAUDI2);
            if (tmpCount < 0)
            {
                LOG_DEBUG(SYN_OSAL, "hlthunk_get_device_count failed for synDeviceGaudi2 (HLTHUNK_DEVICE_GAUDI2)");
            }
            else
            {
                devicesCount += tmpCount;
            }
            tmpCount = hlthunk_get_device_count(HLTHUNK_DEVICE_GAUDI2B);
            if (tmpCount < 0)
            {
                LOG_DEBUG(SYN_OSAL, "hlthunk_get_device_count failed for synDeviceGaudi2 (HLTHUNK_DEVICE_GAUDI2B)");
            }
            else
            {
                devicesCount += tmpCount;
            }
            tmpCount = hlthunk_get_device_count(HLTHUNK_DEVICE_GAUDI2C);
            if (tmpCount < 0)
            {
                LOG_DEBUG(SYN_OSAL, "hlthunk_get_device_count failed for synDeviceGaudi2 (HLTHUNK_DEVICE_GAUDI2C)");
            }
            else
            {
                devicesCount += tmpCount;
            }
            tmpCount = hlthunk_get_device_count(HLTHUNK_DEVICE_GAUDI2D);
            if (tmpCount < 0)
            {
                LOG_DEBUG(SYN_OSAL, "hlthunk_get_device_count failed for synDeviceGaudi2 (HLTHUNK_DEVICE_GAUDI2D)");
            }
            else
            {
                devicesCount += tmpCount;
            }
            break;
        default:
            // get all possible hlthunk device names, count and sum
            for (hlthunk_device_name hlThunkDevName : _getHlThunkDevices(deviceType))
            {
                tmpCount = hlthunk_get_device_count(hlThunkDevName);
                if (tmpCount < 0)
                {
                    LOG_DEBUG(SYN_OSAL, "hlthunk_get_device_count failed for synDeviceType {}", deviceType);
                }
                else
                {
                    devicesCount += tmpCount;
                }
            }
            break;
    }

    return (devicesCount < 0) ? 0 : devicesCount;
};

synStatus
OSAL::writeI2cReg(uint32_t deviceId, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t value)
{
    std::string deviceParentName {};
    synStatus   ret = getDeviceParentName(deviceParentName, deviceId);
    if (ret != synSuccess)
    {
        return ret;
    }

    std::string basePath = "/sys/kernel/debug/accel/" + deviceParentName + "/i2c_";
    ofstream    busFile(basePath + "bus");
    ofstream    addrFile(basePath + "addr");
    ofstream    regFile(basePath + "reg");
    ofstream    dataFile(basePath + "data");

    if (busFile.is_open() && addrFile.is_open() && regFile.is_open() && dataFile.is_open())
    {
        busFile << to_string(i2cBus);
        addrFile << to_string(i2cAddress);
        regFile << to_string(regAddress);
        std::stringstream ss;
        ss << std::hex << value;
        dataFile << ss.str();
        busFile.flush();
        addrFile.flush();
        regFile.flush();
        dataFile.flush();

        LOG_INFO(SYN_OSAL, "Writting to i2c bus {} address {} reg {} data {}", i2cBus, i2cAddress, regAddress, value);
    }
    else
    {
        LOG_ERR(SYN_OSAL, "Failed to open one of the files");
        ret = synFail;
    }

    busFile.close();
    addrFile.close();
    regFile.close();
    dataFile.close();

    return ret;
}

synStatus
OSAL::readI2cReg(uint32_t deviceId, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t* pValue)
{
    std::string deviceParentName {};
    synStatus   ret = getDeviceParentName(deviceParentName, deviceId);
    if (ret != synSuccess)
    {
        return ret;
    }

    std::string basePath = "/sys/kernel/debug/accel/" + deviceParentName + "/i2c_";
    ofstream    busFile(basePath + "bus");
    ofstream    addrFile(basePath + "addr");
    ofstream    regFile(basePath + "reg");
    ifstream    dataFile(basePath + "data");

    if (busFile.is_open() && addrFile.is_open() && regFile.is_open() && dataFile.is_open())
    {
        busFile << to_string(i2cBus);
        addrFile << to_string(i2cAddress);
        regFile << to_string(regAddress);
        busFile.flush();
        addrFile.flush();
        regFile.flush();

        char buffer[256];
        dataFile.getline(buffer, 256);
        // check if a number
        string str(buffer);
        if (!str.empty() && str.find_first_not_of("0123456789") != string::npos)
        {
            *pValue = stoi(str, nullptr, 0);
            LOG_INFO(SYN_OSAL,
                     "Reading from i2c bus {} address {} reg {} data {}",
                     i2cBus,
                     i2cAddress,
                     regAddress,
                     *pValue);
        }
        else
        {
            LOG_ERR(SYN_OSAL, "Failed to read from the files");
            ret = synFail;
        }
    }
    else
    {
        LOG_ERR(SYN_OSAL, "Failed to open one of the files");
        ret = synFail;
    }

    busFile.close();
    addrFile.close();
    regFile.close();
    dataFile.close();

    return ret;
}

synStatus OSAL::setLedState(uint32_t deviceId, uint32_t ledId, bool state)
{
    std::string deviceParentName {};
    synStatus   ret = getDeviceParentName(deviceParentName, deviceId);
    if (ret != synSuccess)
    {
        return ret;
    }

    std::string ledFileName = "/sys/kernel/debug/accel/" + deviceParentName + "/led" + std::to_string(0 + ledId);
    ofstream    ledFile(ledFileName);

    if (ledFile.is_open())
    {
        LOG_INFO(SYN_OSAL, "Setting led id {} to state {}", ledId, state);
        ledFile << to_string(state ? 1 : 0);
        ledFile.close();

        return synSuccess;
    }
    else
    {
        LOG_ERR(SYN_OSAL, "Failed to open the led file {} errno {}", ledFileName, errno);
        return synFail;
    }
}

synStatus OSAL::setFrequency(uint32_t deviceId, uint32_t pllId, uint32_t frequency)
{
    string kmdPllName;
    if (_getPllId(pllId, kmdPllName) != synSuccess)
    {
        return synFail;
    }

    stringstream freqFileName;
    freqFileName.str("");
    freqFileName << "/sys/devices/virtual/habanalabs/hl" << (0 + deviceId) << "/" << kmdPllName;

    synStatus ret = synSuccess;
    ofstream  freqFile(freqFileName.str());
    if (freqFile.is_open())
    {
        freqFile << to_string(frequency);
        freqFile.close();

        LOG_INFO(SYN_OSAL, "Setting frequency {} to file {}", frequency, kmdPllName);
    }
    else
    {
        LOG_ERR(SYN_OSAL, "Failed to open the frequency file {} errno {}", freqFileName.str(), errno);
        ret = synFail;
    }

    return ret;
}

synStatus OSAL::getFrequency(uint32_t deviceId, uint32_t pllId, uint32_t* pFrequency)
{
    string kmdPllName;
    if (_getPllId(pllId, kmdPllName) != synSuccess)
    {
        return synFail;
    }

    stringstream freqFileName;
    freqFileName.str("");
    freqFileName << "/sys/devices/virtual/habanalabs/hl" << (0 + deviceId) << "/" << kmdPllName;

    synStatus ret = synSuccess;
    ifstream  freqFile(freqFileName.str());
    if (freqFile.is_open())
    {
        char   buffer[256];
        size_t size;
        freqFile.getline(buffer, 256);
        // check if a number
        string str(buffer);
        if (!str.empty() && str.find_first_not_of("0123456789") == string::npos)
        {
            *pFrequency = stoi(buffer, &size);
            LOG_INFO(SYN_OSAL, "Getting frequency from {}, the frequency is {}", kmdPllName, *pFrequency);
        }
        else
        {
            LOG_ERR(SYN_OSAL, "Failed to read the frequency from file");
            ret = synFail;
        }

        freqFile.close();
    }
    else
    {
        LOG_ERR(SYN_OSAL, "Failed to open the frequency file {} errno {}", freqFileName.str(), errno);
        ret = synFail;
    }

    return ret;
}

synStatus OSAL::getAvailableNicElementsMask(uint32_t& availableNicMask)
{
    if (!isAcquired())
    {
        LOG_ERR(SYN_OSAL, "Device not opened so has no NIC elements mask");
        return synNoDeviceFound;
    }

    synDeviceType deviceType = m_deviceInfo.deviceType;

    struct hlthunk_mac_addr_info macAddressInfo;
    int                          rc = hlthunk_get_mac_addr_info(m_computeFD, &macAddressInfo);
    if (rc != 0)
    {
        LOG_WARN(SYN_API, "Failed to get NICs MAC Address information rc = {}", rc);
        return synFail;
    }

    availableNicMask = macAddressInfo.mask[0];
    return synSuccess;
}

inline bool isEnvExist(const char* s)
{
    return std::getenv(s) != nullptr;
}

inline int64_t getEvnInt(const char* s, int64_t val = 0)
{
    return (isEnvExist(s) ? std::stoll(std::getenv(s)) : val);
}

void OSAL::deviceInfoReset()
{
    {
        LOG_TRACE(SYN_OSAL, "{} Taking m_vaMapLock on thread ID 0x{:x}", HLLOG_FUNC, (uint64_t)pthread_self());
        std::lock_guard<std::mutex> lock(m_vaMapLock);
        m_deviceVirtualAddressMap.reset();
    }

    m_devIdType            = -1;
    m_devModuleIdx         = -1;
    m_devHlIdx             = -1;
    m_pciAddr              = "";
    m_deviceInfo           = {};
    m_deviceInfo.fd        = -1;
    m_deviceLimitationInfo = {};
}

synStatus OSAL::_getDeviceInfo(synDeviceInfo& deviceInfo, int fd, synDeviceLimitationInfo& deviceLimitationInfo)
{
    synStatus                 status   = synSuccess;
    struct hlthunk_hw_ip_info hwIpInfo = {0};

    int ret = s_interface->getHwIpInfo(fd, &hwIpInfo);
    if (ret != 0)
    {
        LOG_ERR(SYN_OSAL, "HW IP info ioctl failed with ret {}", ret);
        status = synDeviceReset;
        return status;
    }

    deviceInfo.fd              = fd;
    deviceInfo.dramBaseAddress = hwIpInfo.dram_base_address;
    deviceInfo.dramEnabled     = hwIpInfo.dram_enabled;
    deviceInfo.dramSize        = hwIpInfo.dram_size;
    deviceInfo.sramBaseAddress = hwIpInfo.sram_base_address;
    deviceInfo.sramSize        = hwIpInfo.sram_size;
    // For Gaudi: Masking to 9 bit value
    deviceInfo.tpcEnabledMask = hwIpInfo.tpc_enabled_mask & ((1 << 8) - 1);
    deviceInfo.deviceId       = hwIpInfo.module_id;

    DevicePllInfo& devicePllClkInfo = m_devicePllClockInfo;

    if (!devicePllClkInfo.isValid)
    {
        devicePllClkInfo.isValid              = true;
        devicePllClkInfo.psocPciPllNr         = hwIpInfo.psoc_pci_pll_nr;
        devicePllClkInfo.psocPciPllNf         = hwIpInfo.psoc_pci_pll_nf;
        devicePllClkInfo.psocPciPllOd         = hwIpInfo.psoc_pci_pll_od;
        devicePllClkInfo.psocPciPllDivFactor1 = hwIpInfo.psoc_pci_pll_div_factor;
    }

    LOG_INFO(SYN_OSAL, "Got info for device (fd {})", fd);
    LOG_INFO(SYN_OSAL, "\t Device revision number {}", hwIpInfo.revision_id);
    LOG_INFO(SYN_OSAL, "\t Dram base address 0x{:x}", deviceInfo.dramBaseAddress);
    LOG_INFO(SYN_OSAL, "\t Dram size {}", deviceInfo.dramSize);
    LOG_INFO(SYN_OSAL, "\t Dram is enabled {}", deviceInfo.dramEnabled ? "true" : "false");
    LOG_INFO(SYN_OSAL, "\t Sram base address 0x{:x}", deviceInfo.sramBaseAddress);
    LOG_INFO(SYN_OSAL, "\t Sram size {}", deviceInfo.sramSize);

    switch (hwIpInfo.device_id)
    {
        case PCI_IDS_GAUDI_SIMULATOR:
            deviceInfo.deviceType = synDeviceGaudi;
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI simulator");
            break;
        case PCI_IDS_GAUDI:
        case PCI_IDS_GAUDI_SEC:
            deviceInfo.deviceType = synDeviceGaudi;
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI");
            break;
        case PCI_IDS_GAUDI2:
        case PCI_IDS_GAUDI2_SIMULATOR:
            deviceInfo.deviceType = synDeviceGaudi2;
            // on gaudi2 masking to 24-bits value
            deviceInfo.tpcEnabledMask = hwIpInfo.tpc_enabled_mask_ext & ((1 << 24) - 1);
            // Check device revision to distinguish between gaudi2/gaudi2b/gaudi2c/gaudi2d
            switch (hwIpInfo.revision_id)
            {
                case 1:
                    deviceLimitationInfo.fp32Limited = false;
                    LOG_INFO(SYN_OSAL,
                             "\t Device type GAUDI2{}",
                             hwIpInfo.device_id == PCI_IDS_GAUDI2 ? "" : " simulator");
                    break;
                case 2:
                    deviceLimitationInfo.fp32Limited = true;
                    LOG_INFO(SYN_OSAL,
                             "\t Device type GAUDI2B{}",
                             hwIpInfo.device_id == PCI_IDS_GAUDI2 ? "" : " simulator");
                    break;
                case 3:
                    deviceLimitationInfo.fp32Limited = true;
                    LOG_INFO(SYN_OSAL,
                             "\t Device type GAUDI2C{}",
                             hwIpInfo.device_id == PCI_IDS_GAUDI2 ? "" : " simulator");
                    break;
                case 4:
                    deviceLimitationInfo.fp32Limited = true;
                    LOG_INFO(SYN_OSAL,
                             "\t Device type GAUDI2D{}",
                             hwIpInfo.device_id == PCI_IDS_GAUDI2 ? "" : " simulator");
                    break;
                default:
                    LOG_ERR(SYN_OSAL, "Device type GAUDI2 with unknown revision");
                    return synFail;
            }
            break;
        case PCI_IDS_GAUDI2B_SIMULATOR:
        case PCI_IDS_GAUDI2C_SIMULATOR:
        case PCI_IDS_GAUDI2D_SIMULATOR:
            deviceInfo.deviceType = synDeviceGaudi2;
            // on gaudi2 masking to 24-bits value
            deviceInfo.tpcEnabledMask        = hwIpInfo.tpc_enabled_mask_ext & ((1 << 24) - 1);
            deviceLimitationInfo.fp32Limited = true;
            switch (hwIpInfo.device_id)
            {
                case PCI_IDS_GAUDI2B_SIMULATOR:
                    LOG_INFO(SYN_OSAL, "\t Device type GAUDI2B simulator");
                    break;
                case PCI_IDS_GAUDI2C_SIMULATOR:
                    LOG_INFO(SYN_OSAL, "\t Device type GAUDI2C simulator");
                    break;
                case PCI_IDS_GAUDI2D_SIMULATOR:
                    LOG_INFO(SYN_OSAL, "\t Device type GAUDI2D simulator");
                    break;
            }
            break;
        case PCI_IDS_GAUDI_HL2000M:
        case PCI_IDS_GAUDI_HL2000M_SEC:
            deviceInfo.deviceType = synDeviceGaudi;
            LOG_INFO(SYN_OSAL, "\t Device type GAUDIB");
            break;
        case PCI_IDS_GAUDI_HL2000M_SIMULATOR:
            deviceInfo.deviceType = synDeviceGaudi;
            LOG_INFO(SYN_OSAL, "\t Device type GAUDIB simulator");
            break;
        case PCI_IDS_GAUDI3:
            deviceInfo.deviceType = synDeviceGaudi3;
            // on gaudi3 (double die) masking to 64-bits value
            deviceInfo.tpcEnabledMask = hwIpInfo.tpc_enabled_mask_ext;
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI3");
            // We cannot check that the SW (via GCFG) is defined to be double die as well,
            // as the defaults are different between the two
            // But... the HW supports the SW, in such a case, so it is fine
            break;
        case PCI_IDS_GAUDI3_SINGLE_DIE:
            deviceInfo.deviceType = synDeviceGaudi3;
            // on gaudi3 (single die) masking to 32-bits value
            deviceInfo.tpcEnabledMask = hwIpInfo.tpc_enabled_mask_ext & (((uint64_t)1 << 32) - 1);
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI3 - Half Chip, Die0");
            HB_ASSERT(GCFG_GAUDI3_SINGLE_DIE_CHIP.value() == true,
                      "GAUDI3 Single die flavor requires GAUDI3_SINGLE_DIE_CHIP=true");
            break;
        case PCI_IDS_GAUDI3_DIE1:
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI3 - Half Chip, Die1");
            HB_ASSERT(false, "GAUDI3 Single die flavor of Die1 is not supported");
            break;
        case PCI_IDS_GAUDI3_SIMULATOR:
            deviceInfo.deviceType = synDeviceGaudi3;
            // on gaudi3 (double die) masking to 64-bits value
            deviceInfo.tpcEnabledMask = hwIpInfo.tpc_enabled_mask_ext;
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI3 simulator");
            // We cannot check that the SW (via GCFG) is defined to be double die as well,
            // as the defaults are different between the two
            // But... the HW supports the SW, in such a case, so it is fine
            break;
        case PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE:
            deviceInfo.deviceType = synDeviceGaudi3;
            // on gaudi3 (single die) masking to 32-bits value
            deviceInfo.tpcEnabledMask = hwIpInfo.tpc_enabled_mask_ext & (((uint64_t)1 << 32) - 1);
            HB_ASSERT(GCFG_GAUDI3_SINGLE_DIE_CHIP.value() == true,
                      "GAUDI3 Single die flavor requires GAUDI3_SINGLE_DIE_CHIP=true");
            LOG_INFO(SYN_OSAL, "\t Device type GAUDI3 simulator");
            break;
        default:
            LOG_ERR(SYN_OSAL, "habana device 0x{:x} not supported", hwIpInfo.device_id);
            status = synFail;
    }
    LOG_INFO(SYN_OSAL, "\t Tpc enabled mask 0x{:x}", deviceInfo.tpcEnabledMask);

    return status;
}

synStatus OSAL::_destroyDeviceBuffers()
{
    if (m_deviceVirtualAddressMap.get() == nullptr)
    {
        return synInvalidArgument;
    }

    DeviceVirtualAddressMap& virtualAddressMap = *(m_deviceVirtualAddressMap.get());
    synStatus                status            = synSuccess;

    LOG_INFO(SYN_OSAL, "Trying to close all device buffers");

    for (auto it = virtualAddressMap.begin(); it != virtualAddressMap.end(); it++)
    {
        BufferAllocator* pBufAlloc = it->second;

        if (pBufAlloc == nullptr)
        {
            LOG_ERR(SYN_OSAL, "No device BufferAllocator to destroy VA {}", it->first);
        }
        else if (pBufAlloc->FreeMemory() == synSuccess)
        {
            LOG_TRACE(SYN_OSAL, "Destroyed device buffer successfully");
            delete pBufAlloc;
        }
        else
        {
            LOG_ERR(SYN_OSAL, "Failed to destroy device buffer");
            status = synFail;
            delete pBufAlloc;
        }
    }

    virtualAddressMap.clear();

    return status;
}

synStatus OSAL::_waitForCloseComputeDevice(int controlFD, bool isOpen)
{
    if (controlFD < 0)
    {
        LOG_ERR(SYN_OSAL, "{} called with bad FD {}", HLLOG_FUNC, controlFD);
        return synFail;
    }

    bool isActive, inRelease;
    int  waitNumbSec = 0;

    int ret = s_interface->getDeviceStatus(controlFD, isActive, inRelease);
    if (ret != 0)
    {
        LOG_ERR(SYN_OSAL, "getDeviceStatus {} failed with error", ret);
        return synFail;
    }
    else
    {
        LOG_INFO(SYN_OSAL,
                 "{}: Before {} device active state {} and release {} state",
                 HLLOG_FUNC,
                 isOpen ? "opening" : "closing",
                 isActive,
                 inRelease);
    }

    int rc = 0;
    rc     = s_interface->waitForDeviceOperational(controlFD, c_waitTimForReleaseTriesSec);
    if (rc != 0)
    {
        LOG_ERR(SYN_OSAL, "_wait_until_not_in_reset failed with rc {}", rc);
        return synFail;
    }
    ret = s_interface->getDeviceStatus(controlFD, isActive, inRelease);
    if (ret != 0)
    {
        LOG_ERR(SYN_OSAL, "getDeviceStatus {} failed with error", ret);
        return synFail;
    }
    else
    {
        LOG_INFO(SYN_OSAL,
                 "{} device active state {} and release {} state after {} sec",
                 HLLOG_FUNC,
                 isActive,
                 inRelease,
                 waitNumbSec);
    }

    return synSuccess;
}

synStatus OSAL::closeFD()
{
    synStatus status = synSuccess;

    LOG_INFO(SYN_OSAL, "Trying to close FD of device");

    int retCompute = 0;
    if (m_computeFD == -1)
    {
        LOG_DEBUG(SYN_OSAL, "No compute FD found for device");
    }
    else
    {
        retCompute = s_interface->close(m_computeFD);
        if (retCompute != 0)
        {
            LOG_ERR(SYN_OSAL, "Closing device computeFD {} failed. Returned {}.", m_computeFD, retCompute);
        }
        else
        {
            LOG_INFO(SYN_OSAL, "FD compute {} is closed", m_computeFD);
            m_deviceInfo.fd = -1;
            m_computeFD     = -1;
        }
    }

    int retContol = 0;
    if (m_controlFD == -1)
    {
        LOG_DEBUG(SYN_OSAL, "No control FD found for device, controlFD {}.", m_controlFD);
    }
    else
    {
        status = _waitForCloseComputeDevice(m_controlFD, false);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_OSAL, "_waitForCloseComputeDevice failed with synStatus {}.", status);
        }

        retContol = s_interface->close(m_controlFD);
        if (retContol != 0)
        {
            LOG_ERR(SYN_OSAL, "Closing device, controlFD {} failed with synStatus {}.", m_controlFD, retContol);
        }
        else
        {
            LOG_INFO(SYN_OSAL, "FD Control {} is closed", m_controlFD);
            m_controlFD = -1;
        }
    }

    if ((retCompute != 0) && (retContol != 0))
    {
        return synFail;
    }

    return status;
}

synStatus OSAL::_getPllId(uint32_t pllId, string& kmdPllName)
{
    switch (pllId)
    {
        case PLL_CPU:
            kmdPllName = "cpu_clk";
            break;
        case PLL_IC:
            kmdPllName = "ic_clk";
            break;
        case PLL_MC:
            kmdPllName = "mc_clk";
            break;
        case PLL_MME:
            kmdPllName = "mme_clk";
            break;
        case PLL_PCI:
            kmdPllName = "pci_clk";
            break;
        case PLL_EMMC:
            kmdPllName = "emmc_clk";
            break;
        case PLL_TPC:
            kmdPllName = "tpc_clk";
            break;
        case PLL_SIZE:
        default:
            LOG_ERR(SYN_OSAL, "Used incorrect PLL id {}", pllId);
            kmdPllName = "";
            return synFail;
    }

    LOG_INFO(SYN_OSAL, "Got pll file name {}", kmdPllName);
    return synSuccess;
}

std::vector<hlthunk_device_name> OSAL::_getHlThunkDevices(synDeviceType deviceType, synDeviceSubType subType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return {HLTHUNK_DEVICE_GAUDI, HLTHUNK_DEVICE_GAUDI_HL2000M};
        case synDeviceGaudi2:
            if (subType == DEVICE_SUB_TYPE_B)
            {
                return {HLTHUNK_DEVICE_GAUDI2B};
            }
            else if (subType == DEVICE_SUB_TYPE_C)
            {
                return {HLTHUNK_DEVICE_GAUDI2C};
            }
            else if (subType == DEVICE_SUB_TYPE_D)
            {
                return {HLTHUNK_DEVICE_GAUDI2D};
            }
            return {HLTHUNK_DEVICE_GAUDI2};
        case synDeviceGaudi3:
            return {HLTHUNK_DEVICE_GAUDI3};
        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            return {HLTHUNK_DEVICE_INVALID};
    }
}

void OSAL::_updateDeviceCount(uint32_t                         outCount[synDeviceTypeSize],
                              unsigned&                        numberOfDevices,
                              const DeviceNameToDeviceTypeMap& rNameToDeviceType,
                              const std::string&               deviceName,
                              bool                             shouldStripSimulator)

{
    std::string tmpDeviceName;
    size_t      deviceNameLastPosFound = 0;

    // Finding the longest substr from deviceName, which is a known device-name (defined by rNameToDeviceType DB)
    // We do that by adding token-after-token, and searching whether there is a known device-name on that DB
    do
    {
        if (deviceNameLastPosFound == std::string::npos)
        {
            break;
        }

        tmpDeviceName = deviceName;

        size_t strPos = tmpDeviceName.find_first_of(" ", deviceNameLastPosFound + 1);
        if (strPos == std::string::npos)
        {
            strPos = tmpDeviceName.find_first_of("\n", deviceNameLastPosFound + 1);
        }

        if (strPos != std::string::npos)
        {
            tmpDeviceName.erase(strPos);
        }
        auto partialNameFound = rNameToDeviceType.find(tmpDeviceName);
        if (partialNameFound == rNameToDeviceType.end())
        {
            break;
        }
        deviceNameLastPosFound = strPos;
    } while (1);

    tmpDeviceName = deviceName;
    tmpDeviceName = deviceName.substr(0, deviceNameLastPosFound);

    if (shouldStripSimulator)
    {
        tmpDeviceName.erase(tmpDeviceName.find_last_not_of(" Simulator") + 1);
    }
    auto nameToDeviceTypeEntry = rNameToDeviceType.find(tmpDeviceName);
    if (nameToDeviceTypeEntry != rNameToDeviceType.end())
    {
        LOG_DEBUG(SYN_OSAL,
                  "Found device desciption ({}, {}) for device-type {}",
                  nameToDeviceTypeEntry->first,
                  nameToDeviceTypeEntry->second,
                  deviceName);

        ++outCount[nameToDeviceTypeEntry->second];
        ++numberOfDevices;
    }
    else
    {
        LOG_WARN(SYN_OSAL, "Not found device type mapping for {}", deviceName);
    }
}

void OSAL::logAcquiredDevicePid(synModuleId moduleId, std::string* deviceFailLog)
{
    std::string thisPidStr    = std::to_string(getpid());
    std::string hlDevicesPath = HLTHUNK_DEV_NAME_CONTROL;
    hlDevicesPath = hlDevicesPath.substr(0, hlDevicesPath.length() - 2);

    char        readBuffer[128];
    std::string CmdOutput = "";
    std::string cmd =
        "ls -l /proc/*/fd/* 2> /dev/null | grep '" + hlDevicesPath + "' | grep /proc/" + thisPidStr;

    FILE* pipe = popen(cmd.c_str(), "r");
    while (!feof(pipe))
    {
        if (fgets(readBuffer, 128, pipe) != nullptr) CmdOutput += readBuffer;
    }
    pclose(pipe);

    if (CmdOutput == "")
    {
        LOG_TRACE(SYN_OSAL, "PID {} trying to acquire device module ID {} couldn't be found", thisPidStr, moduleId);
        return;
    }

    unsigned    startOffset = CmdOutput.find(hlDevicesPath);
    unsigned    endOffset   = CmdOutput.length() - 1;
    std::string thisDevice  = CmdOutput.substr(startOffset, endOffset - startOffset);

    CmdOutput = "";
    cmd = "ls -l /proc/*/fd/* 2> /dev/null | grep '" + hlDevicesPath + "'";

    pipe = popen(cmd.c_str(), "r");
    while (!feof(pipe))
    {
        if (fgets(readBuffer, 128, pipe) != nullptr) CmdOutput += readBuffer;
    }
    pclose(pipe);

    if (CmdOutput == "")
    {
        LOG_TRACE(SYN_OSAL, "No habanalabs devices acquired by processes can be found in: /proc");
        return;
    }

    char* pidEntryPtr = strtok(const_cast<char*>(CmdOutput.c_str()), "\n");
    while (pidEntryPtr != nullptr)
    {
        std::string pidEntry = pidEntryPtr;
        startOffset       = pidEntry.find("/proc/") + 6;
        endOffset         = pidEntry.find("/fd/");
        std::string pidId = pidEntry.substr(startOffset, endOffset - startOffset);

        startOffset      = pidEntry.find("/fd/") + 4;
        endOffset        = pidEntry.find(" ->");
        std::string FdId = pidEntry.substr(startOffset, endOffset - startOffset);

        startOffset        = pidEntry.find(hlDevicesPath);
        endOffset          = pidEntry.length();
        std::string device = pidEntry.substr(startOffset, endOffset - startOffset);

        if (device == thisDevice && pidId != thisPidStr)
        {
            if (deviceFailLog != nullptr)
            {
                *deviceFailLog = "Device module ID " +
                                 std::to_string(moduleId) +
                                 " failed to acquire, already acquired by PID " +
                                 pidId +
                                 ", with control FD " +
                                 FdId;
                return;
            }

            LOG_ERR(SYN_OSAL,
                    "Device module ID {} failed to acquire, already acquired by PID {}, with control FD {}",
                    moduleId,
                    pidId,
                    FdId);
            return;
        }

        pidEntryPtr = strtok(nullptr, "\n");
    }
}

string OSAL::GetDeviceNameByDevType(synDeviceType devType)
{
    switch (devType)
    {
        case synDeviceGaudi:
            return string("GAUDI");

        case synDeviceGaudi2:
            return string("GAUDI2");

        case synDeviceGaudi3:
            return string("GAUDI3");

        default:
            HB_ASSERT(0, "No such device type");
            break;
    };

    return string("");
}

std::string OSAL::logLinuxDevicesDirOutputByType(std::string devicePath)
{
    std::string cmdOutput = "";
    std::string resultOutput = "";
    char readBuffer[128];
    std::string hlDevicesPath = devicePath.substr(0, devicePath.length() - 2);
    std::string devicesDirCmd = "ls -l " + hlDevicesPath + "[0-7] 2> /dev/null";

    FILE* pipe = popen(devicesDirCmd.c_str(), "r");
    while (!feof(pipe))
    {
        if (fgets(readBuffer, 128, pipe) != nullptr)
            cmdOutput += readBuffer;
    }
    pclose(pipe);

    if (cmdOutput == "")
    {
        resultOutput = "/dev/ directory does not contains any habanalabs ";
        resultOutput.append(std::string(HLTHUNK_DEV_NAME_PRIMARY) == devicePath ?
                            "compute devices, " : "control devices, ");
    }
    else
    {
        resultOutput = "habanalabs ";
        resultOutput.append(std::string(HLTHUNK_DEV_NAME_PRIMARY) == devicePath ?
                            "compute devices found: " : "control devices found: ");
        resultOutput.append(cmdOutput);
    }

    return resultOutput;
}

std::string OSAL::checkExposedLinuxDevicesDir()
{
    std::string exposedLinuxDevicesDirOutput;
    struct stat devicesDir;
    if (stat("/dev/", &devicesDir) != 0)
    {
        exposedLinuxDevicesDirOutput = "/dev/ directory is not exist on this VM";
        return exposedLinuxDevicesDirOutput;
    }

    exposedLinuxDevicesDirOutput = logLinuxDevicesDirOutputByType(HLTHUNK_DEV_NAME_PRIMARY);
    exposedLinuxDevicesDirOutput += logLinuxDevicesDirOutputByType(HLTHUNK_DEV_NAME_CONTROL);

    return exposedLinuxDevicesDirOutput;
}

void OSAL::GetModuleIDsList(uint32_t *pDeviceModuleIds, uint32_t*  size)
{
    int       controlFD  = -1;
    synModuleId moduleId = 0;
    uint32_t i           = 0;
    *size = 0;
    // Iterate 8 devices since this is the max amount on a HLS box
    for (; moduleId < MAX_NUM_OF_DEVICES_PER_HOST; moduleId++)
    {
        controlFD = s_interface->openControlByModuleId(moduleId);
        // Device not found, this module is not present on the VM, try the next module
        if (controlFD == -EBADF)
        {
            continue;
        }
        else
        {
            pDeviceModuleIds[i] = moduleId;
            i++;
        }
    }
    *size = i;
}

#endif  //_WIN32
