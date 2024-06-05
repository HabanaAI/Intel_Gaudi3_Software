#include "device_manager.hpp"

#include "defenders.h"
#include "device_utils.hpp"
#include "graph_compiler/graph_traits.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "infra/global_conf_manager.h"
#include "runtime/common/osal/osal.hpp"
#include "runtime/qman/gaudi/device_gaudi.hpp"
#include "runtime/scal/gaudi2/device_gaudi2scal.hpp"
#include "runtime/scal/gaudi3/device_gaudi3scal.hpp"

extern HalReaderPtr instantiateGaudiHalReader();
extern HalReaderPtr instantiateGaudiMHalReader();

const static uint16_t NUM_OF_SYNC_OBJECTS = 2048;

synStatus DeviceManager::getDeviceAttributesByModuleId(const synModuleId         moduleId,
                                                       const synDeviceAttribute* deviceAttr,
                                                       const unsigned            querySize,
                                                       uint64_t*                 retVal) const
{
    uint32_t             currentClockRate = 0;
    const synDeviceInfo* pDeviceInfo      = nullptr;
    synStatus status = OSAL::getInstance().getDeviceInfoByModuleId(pDeviceInfo, currentClockRate, moduleId);
    if ((status != synSuccess) || (pDeviceInfo == nullptr))
    {
        return (status != synSuccess) ? status : synFail;
    }

    return extractDeviceAttributes(deviceAttr, querySize, retVal, *pDeviceInfo, &currentClockRate, nullptr);
}

synStatus DeviceManager::acquireDevice(uint32_t* pDeviceId, const char* pciBus, synDeviceType deviceType, synModuleId moduleId)
{
    std::unique_lock<std::mutex> guard(m_mutex);

    VERIFY_IS_NULL_POINTER(SYN_API, pDeviceId, "Device-Id");

    synStatus status = OSAL::getInstance().acquireDevice(pciBus, deviceType, moduleId);
    if (status == synDeviceAlreadyAcquired)
    {
        if (m_device == nullptr)
        {
            LOG_CRITICAL(SYN_API, "Can not acquire device, it is marked as already acquired but not found???");
        }
        else
        {
            LOG_ERR(SYN_API, "Already acquired device successfully");
        }
        return status;
    }
    else if (status != synSuccess)
    {
        *pDeviceId = UINT_MAX;
        LOG_WARN(SYN_API, "Acquiring a Synapse device aborted");
        return status;
    }

    status = _addDevice();
    if (status != synSuccess)
    {
        guard.unlock();
        releaseDevice();
        return status;
    }

    uint32_t moduleIdx;
    OSAL::getInstance().getDeviceModuleIdx(moduleIdx);
    LOG_DEBUG(SYN_DEVICE, "Acquired Synapse device successfully, device module ID {}", moduleIdx);

    DeviceInterface* pDeviceInterface = m_device.get();
    synDeviceType    actualDeviceType = pDeviceInterface->getDevType();
    GlobalConfManager::instance().setDeviceType(actualDeviceType);

    status = pDeviceInterface->acquire(NUM_OF_SYNC_OBJECTS);
    if (status != synSuccess)
    {
        guard.unlock();
        releaseDevice();
        return status;
    }

    *pDeviceId = 0;

    return status;
}

synStatus DeviceManager::releaseDevice()
{
    std::unique_lock<std::mutex> guard(m_mutex);

    if (m_device == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Device is not acquired", HLLOG_FUNC);
        return synNoDeviceFound;
    }

    if (m_device.use_count() > 1)
    {
        LOG_ERR(SYN_API, "Release device called while it is being used by {} threads",
                          m_device.use_count() - 1);
        return synFail;
    }

    // Set the device to release state, wait for idle state and then release
    synStatus status = m_device->release(m_deviceBeingReleased);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "Can not release device");
        return status;
    }

    m_device.reset();
    m_deviceBeingReleased = false;

    return synSuccess;
}

uint16_t DeviceManager::getNumDevices() const
{
    return m_device != nullptr ? 1 : 0;
}

synStatus DeviceManager::_addDevice()
{
    synDeviceInfo deviceInfo {};
    synStatus     status = OSAL::getInstance().GetDeviceInfo(deviceInfo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get device info");
        return status;
    }

    int fdControl;
    status = OSAL::getInstance().GetDeviceControlFd(fdControl);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get device control FD");
        return status;
    }

    uint32_t hlIdx;
    status = OSAL::getInstance().getDeviceHlIdx(hlIdx);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get device index");
        return status;
    }

    uint32_t devModuleIdx;
    status = OSAL::getInstance().getDeviceModuleIdx(devModuleIdx);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get device module ID");
        return status;
    }

    uint32_t devIdType;
    status = OSAL::getInstance().getDeviceIdType(devIdType);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get device type");
        return status;
    }

    std::string pciAddr;
    status = OSAL::getInstance().getPciAddr(pciAddr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to get device pciAddr");
        return status;
    }

    DeviceConstructInfo deviceConstructInfo {.deviceInfo   = deviceInfo,
                                             .fdControl    = fdControl,
                                             .hlIdx        = hlIdx,
                                             .devModuleIdx = devModuleIdx,
                                             .devIdType    = devIdType,
                                             .pciAddr      = pciAddr};

    LOG_INFO_T(SYN_DEVICE,
               "Creating a new device, type {} hlIdx {} fd {}",
               deviceConstructInfo.deviceInfo.deviceType,
               deviceConstructInfo.hlIdx,
               deviceConstructInfo.deviceInfo.fd);

    switch (deviceConstructInfo.deviceInfo.deviceType)
    {
        case synDeviceGaudi:
            m_device = std::make_shared<DeviceGaudi>(deviceConstructInfo);
            break;

        case synDeviceGaudi2:
            m_device = std::make_shared<DeviceGaudi2scal>(deviceConstructInfo);
            break;

        case synDeviceGaudi3:
            m_device = std::make_shared<DeviceGaudi3scal>(deviceConstructInfo);
            break;

        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            HB_ASSERT(false, "Bad device type");
    }

    if (m_device == nullptr)
    {
        LOG_ERR(SYN_API, "{}: Internal-device pointer is nullptr", HLLOG_FUNC);
        return synFail;
    }

    LOG_INFO_T(SYN_DEVICE, "created new device, ptr {}", TO64P(m_device.get()));

    return synSuccess;
}

std::shared_ptr<DeviceInterface> DeviceManager::getDeviceInterface(const char* funcName) const
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = m_device;

    if (pDeviceInterface == nullptr)
    {
        if (funcName != nullptr)
        {
            LOG_ERR(SYN_DEVICE, "{}: No device allocated", funcName);
        }
        return nullptr;
    }

    if (m_deviceBeingReleased)
    {
        if (funcName != nullptr)
        {
            LOG_ERR(SYN_DEVICE, "{}: Device is being released by another thread", funcName);
        }
        return nullptr;
    }

    return pDeviceInterface;
}

std::shared_ptr<DeviceGaudi> DeviceManager::getDeviceGaudi(const char* funcName) const
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        return nullptr;
    }

    std::shared_ptr<DeviceGaudi> pDeviceGaudi = std::dynamic_pointer_cast<DeviceGaudi>(pDeviceInterface);

    if (pDeviceGaudi == nullptr)
    {
        synDeviceType devType = pDeviceInterface->getDevType();

        if (funcName) LOG_ERR(SYN_DEVICE, "{}: Invalid devType {}", funcName, devType);
        return nullptr;
    }

    return pDeviceGaudi;
}

void DeviceManager::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(nullptr);

    if (pDeviceInterface != nullptr)
    {
        pDeviceInterface->notifyRecipeRemoval(rRecipeHandle);
    }
}

synStatus DeviceManager::allocateDeviceMemory(uint64_t           size,
                                              uint32_t           flags,
                                              void**             buffer,
                                              bool               isUserRequest,
                                              uint64_t           reqVAAddress,
                                              const std::string& mappingDesc,
                                              uint64_t*          deviceVA)
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        return synFail;
    }

    return pDeviceInterface->allocateMemory(size, flags, buffer, isUserRequest, reqVAAddress, mappingDesc, deviceVA);
}

synStatus DeviceManager::deallocateDeviceMemory(void* pBuffer, uint32_t flags, bool isUserRequest)
{
    // This API is not checking m_deviceBeingReleased because on release HCL use it to free device memory
    std::shared_ptr<DeviceInterface> pDeviceInterface = m_device;
    if (pDeviceInterface == nullptr)
    {
         return synSuccess;
    }

    return pDeviceInterface->deallocateMemory(pBuffer, flags, isUserRequest);
}

synStatus DeviceManager::mapBufferToDevice(uint64_t           size,
                                           void*              buffer,
                                           bool               isUserRequest,
                                           uint64_t           reqVAAddress,
                                           const std::string& mappingDesc)
{
    LOG_TRACE(SYN_API,
              "{}: buffer 0x{:x} size 0x{:x} isUserRequest {}",
              HLLOG_FUNC,
              (uint64_t)buffer,
              size,
              isUserRequest);

    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        LOG_ERR(SYN_API,
                "{}: Device does not exist for mapping Buffer 0x{:x} (isUserRequest {})",
                HLLOG_FUNC,
                (uint64_t)buffer,
                isUserRequest);
        return synInvalidArgument;
    }

    return pDeviceInterface->mapBufferToDevice(size, buffer, isUserRequest, reqVAAddress, mappingDesc);
}

synStatus DeviceManager::unmapBufferFromDevice(void* buffer, bool isUserRequest, uint64_t* bufferSize)
{
    LOG_TRACE(SYN_API, "{}: buffer 0x{:x} isUserRequest {}", HLLOG_FUNC, (uint64_t)buffer, isUserRequest);

    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        LOG_ERR(SYN_API,
                "{}: device not exist for unmapping Buffer 0x{:x} (isUserRequest {})",
                HLLOG_FUNC,
                (uint64_t)buffer,
                isUserRequest);
        return synInvalidArgument;
    }

    return pDeviceInterface->unmapBufferFromDevice(buffer, isUserRequest, bufferSize);
}

synStatus DeviceManager::deviceGetCount(uint32_t* pCount)
{
    *pCount = OSAL::getInstance().getDevicesCount();
    LOG_TRACE(SYN_API, "{}: Number of devices found: {}", HLLOG_FUNC, *pCount);

    return synSuccess;
}

synStatus DeviceManager::deviceGetModuleIds(uint32_t *pDeviceModuleIds, uint32_t*  size)
{
    uint32_t  numOfDevice = OSAL::getInstance().getDevicesCount();
    if(*size < numOfDevice)
    {
        LOG_ERR(SYN_API, "{}: Number of devices found: {} while user provided {}", HLLOG_FUNC, numOfDevice, *size);
        return synInvalidArgument;

    }
    OSAL::getInstance().GetModuleIDsList(pDeviceModuleIds, size);
    return synSuccess;
}

synStatus DeviceManager::deviceGetCountByDeviceType(uint32_t* pCount, const synDeviceType deviceType)
{
    *pCount = OSAL::getInstance().getDevicesCount(deviceType);
    LOG_TRACE(SYN_API, "{}: Number of devices found: {}", HLLOG_FUNC, *pCount);
    return synSuccess;
}

synStatus DeviceManager::deviceCount(uint32_t count[synDeviceTypeSize])
{
    LOG_TRACE(SYN_API, "{}", HLLOG_FUNC);

    for (uint32_t device = 0; device < synDeviceTypeSize - 1; ++device)
    {
        count[device] = OSAL::getInstance().getDevicesCount(synDeviceType(device));
    }
    return synSuccess;
}

synStatus DeviceManager::deviceGetPCIBusId(char* pPciBusId, const int len)
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        return synFail;
    }

    return pDeviceInterface->getPCIBusId(pPciBusId, len);
}

synStatus DeviceManager::deviceGetFd(int* pFd)
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        return synFail;
    }

    *pFd = OSAL::getInstance().getFd();

    if (*pFd < 0)
    {
        LOG_ERR(SYN_API, "{}: File descriptor not found", HLLOG_FUNC);
        return synFail;
    }

    return synSuccess;
}

synStatus DeviceManager::getDeviceInfo(synDeviceInfo* pDeviceInfo) const
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        return synFail;
    }

    pDeviceInterface->getDeviceInfo(*pDeviceInfo);

    return synSuccess;
}

synStatus DeviceManager::getDeviceInfo(synDeviceInfoV2* pDeviceInfo) const
{
    std::shared_ptr<DeviceInterface> pDeviceInterface = getDeviceInterface(__FUNCTION__);
    if (pDeviceInterface == nullptr)
    {
        return synFail;
    }

    pDeviceInterface->getDeviceInfo(*pDeviceInfo);

    return synSuccess;
}

synStatus DeviceManager::getDeviceName(char* pName, const int len)
{
    return OSAL::getInstance().getDeviceName(pName, len);
}

synStatus DeviceManager::getDeviceTypeAttribute(synDeviceType             deviceType,
                                                const synDeviceAttribute* deviceAttr,
                                                const unsigned            querySize,
                                                uint64_t*                 retVal)
{
    VERIFY_IS_NULL_POINTER(SYN_API, retVal, "Device-attribute array");
    VERIFY_IS_NULL_POINTER(SYN_API, deviceAttr, "Device-attribute identifier");

    synStatus status       = synSuccess;
    uint64_t* currentValue = retVal;
    for (unsigned queryIndex = 0; queryIndex < querySize; queryIndex++, currentValue++)
    {
        switch (deviceAttr[queryIndex])
        {
            case DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.sramBaseAddress;
                }
                else
                {
                    LOG_ERR(SYN_API,
                            "{}: Unsupported attribute {} in query index {} for device type {}",
                            HLLOG_FUNC,
                            deviceAttr[queryIndex],
                            queryIndex,
                            _deviceTypeToStrings(deviceType)[0]);
                    status = synUnsupported;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.dramBaseAddress;
                }
                else
                {
                    GraphTraits                graphTraits(deviceType);
                    std::shared_ptr<HalReader> pHalReader = graphTraits.getHalReader();
                    *currentValue                         = pHalReader->getDRAMBaseAddr();
                }
                break;
            }
            case DEVICE_ATTRIBUTE_SRAM_SIZE:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.sramSize;
                }
                else
                {
                    LOG_ERR(SYN_API,
                            "{}: Unsupported attribute {} in query index {} for device type {}",
                            HLLOG_FUNC,
                            deviceAttr[queryIndex],
                            queryIndex,
                            _deviceTypeToStrings(deviceType)[0]);
                    status = synUnsupported;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_SIZE:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.dramSize;
                }
                else
                {
                    LOG_ERR(SYN_API,
                            "{}: Unsupported attribute {} in query index {} for device type {}",
                            HLLOG_FUNC,
                            deviceAttr[queryIndex],
                            queryIndex,
                            _deviceTypeToStrings(deviceType)[0]);
                    status = synUnsupported;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_TPC_ENABLED_MASK:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.tpcEnabledMask;
                }
                else
                {
                    GraphTraits                graphTraits(deviceType);
                    std::shared_ptr<HalReader> pHalReader = graphTraits.getHalReader();
                    *currentValue                         = pHalReader->getTpcEnginesMask();
                }
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_ENABLED:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.dramEnabled;
                }
                else
                {
                    GraphTraits                graphTraits(deviceType);
                    std::shared_ptr<HalReader> pHalReader = graphTraits.getHalReader();
                    *currentValue                         = pHalReader->getDRAMSizeInBytes() > 0;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_DEVICE_TYPE:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    synDeviceInfo synapseDeviceInfo {};
                    status = OSAL::getInstance().GetDeviceInfo(synapseDeviceInfo);
                    if (status != synSuccess)
                    {
                        LOG_ERR(SYN_DEVICE, "Failed to get device info");
                        return status;
                    }
                    *currentValue = synapseDeviceInfo.deviceType;
                }
                else
                {
                    *currentValue = deviceType;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_CLK_RATE:
            {
                if (OSAL::getInstance().isAcquired())
                {
                    DeviceClockRateInfo deviceClockRateInfo;
                    status = OSAL::getInstance().getDeviceClockRateInfo(deviceClockRateInfo);
                    if (status != synSuccess)
                    {
                        return status;
                    }
                    else
                    {
                        *currentValue = deviceClockRateInfo.currentClockRate;
                    }
                }
                else
                {
                    LOG_ERR(SYN_API,
                            "{}: Unsupported attribute {} in query index {} for device type {}",
                            HLLOG_FUNC,
                            deviceAttr[queryIndex],
                            queryIndex,
                            _deviceTypeToStrings(deviceType)[0]);
                    status = synUnsupported;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_MAX_RMW_SIZE:
            {
                *currentValue = GCFG_RMW_SECTION_MAX_SIZE_BYTES.value();
                if (*currentValue == 0)
                {
                    LOG_ERR(SYN_API,
                            "{}: Unsupported attribute DEVICE_ATTRIBUTE_MAX_RMW_SIZE for device type {}",
                            HLLOG_FUNC,
                            _deviceTypeToStrings(deviceType)[0]);
                    status = synUnsupported;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE:
            {
                GraphTraits                graphTraits(deviceType);
                std::shared_ptr<HalReader> pHalReader = graphTraits.getHalReader();
                *currentValue                         = pHalReader->getAddressAlignmentSizeInBytes();
                break;
            }
            case DEVICE_ATTRIBUTE_MAX_DIMS:
            {
                *currentValue = SYN_GAUDI_MAX_TENSOR_DIM;
                break;
            }
            // DEVICE_ATTRIBUTE_DRAM_FREE_SIZE attribute require device
            case DEVICE_ATTRIBUTE_DRAM_FREE_SIZE:
            // DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE attribute require device
            case DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE:
            default:
            {
                LOG_ERR(SYN_API,
                        "{}: Unsupported attribute {} in query index {} for device type {}",
                        HLLOG_FUNC,
                        deviceAttr[queryIndex],
                        queryIndex,
                        _deviceTypeToStrings(deviceType)[0]);
                status = synUnsupported;
            }
        }
    }

    return status;
}

const std::unordered_map<std::string, synDeviceType>& DeviceManager::_stringToDeviceType()
{
    static const std::unordered_map<std::string, synDeviceType> DEVICE_TYPE_BY_NAME = []() {
        std::unordered_map<std::string, synDeviceType> ret;
        for (uint32_t device = 0; device < synDeviceTypeSize - 1; ++device)
        {
            auto nameVec = _deviceTypeToStrings(synDeviceType(device));
            for (const auto& name : nameVec)
            {
                ret.emplace(name, synDeviceType(device));
            }
        }
        return ret;
    }();

    return DEVICE_TYPE_BY_NAME;
}

std::vector<std::string> DeviceManager::_deviceTypeToStrings(synDeviceType deviceType)
{
    std::vector<std::string> vectorOfDeviceTypesStr;
    switch (deviceType)
    {
        case synDeviceGaudi:
            vectorOfDeviceTypesStr.push_back("GAUDI");
            return vectorOfDeviceTypesStr;
        case synDeviceGaudi2:
            vectorOfDeviceTypesStr.push_back("GAUDI2");
            vectorOfDeviceTypesStr.push_back("GAUDI2B");
            vectorOfDeviceTypesStr.push_back("GAUDI2C");
            vectorOfDeviceTypesStr.push_back("GAUDI2D");
            return vectorOfDeviceTypesStr;
        case synDeviceGaudi3:
            vectorOfDeviceTypesStr.push_back("GAUDI3");
            return vectorOfDeviceTypesStr;
        case synDeviceEmulator:
            vectorOfDeviceTypesStr.push_back("GAUDI Simulator");
            return vectorOfDeviceTypesStr;
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            LOG_ERR(SYN_API, "{} called with unfamiliar device type :{}", HLLOG_FUNC, deviceType);
            return vectorOfDeviceTypesStr;
    }
    return vectorOfDeviceTypesStr;
}

synStatus DeviceManager::getHalReader(synDeviceType deviceType, std::shared_ptr<HalReader>& rpHalReader)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            rpHalReader = instantiateGaudiHalReader();
            break;

        case synDeviceGaudi2:
            rpHalReader = Gaudi2HalReader::instance();
            break;

        case synDeviceGaudi3:
            rpHalReader = Gaudi3HalReader::instance();
            break;

        default:
            LOG_ERR(SYN_API, "{}: Device type {} not supported", HLLOG_FUNC, deviceType);
            return synUnsupported;
    }

    return synSuccess;
}