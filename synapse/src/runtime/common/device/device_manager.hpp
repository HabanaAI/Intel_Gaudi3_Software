#pragma once

#include <memory>
#include <mutex>

#include "synapse_common_types.h"
#include "synapse_api_types.h"
#include "define_synapse_common.hpp"
#include "runtime/common/device/device_interface.hpp"

struct InternalRecipeHandle;
struct DeviceConstructInfo;
class DeviceGaudi;
class HalReader;

class DeviceManager
{
public:
    DeviceManager() = default;

    virtual ~DeviceManager() = default;

    synStatus getDeviceAttributesByModuleId(const synModuleId         moduleId,
                                            const synDeviceAttribute* deviceAttr,
                                            const unsigned            querySize,
                                            uint64_t*                 retVal) const;

    synStatus acquireDevice(uint32_t* pDeviceId, const char* pciBus, synDeviceType deviceType, synModuleId moduleId);

    synStatus releaseDevice();

    uint16_t getNumDevices() const;

    std::shared_ptr<DeviceInterface> getDeviceInterface(const char* funcName) const;

    std::shared_ptr<DeviceGaudi> getDeviceGaudi(const char* funcName) const;

    void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle);

    synStatus allocateDeviceMemory(uint64_t           size,
                                   uint32_t           flags,
                                   void**             buffer,
                                   bool               isUserRequest,
                                   uint64_t           reqVAAddress,
                                   const std::string& mappingDesc,
                                   uint64_t*          deviceVA = nullptr);

    synStatus deallocateDeviceMemory(void*    pBuffer,
                                     uint32_t flags,
                                     bool     isUserRequest);

    synStatus mapBufferToDevice(uint64_t           size,
                                void*              buffer,
                                bool               isUserRequest,
                                uint64_t           reqVAAddress,
                                const std::string& mappingDesc);

    synStatus unmapBufferFromDevice(void*        buffer,
                                    bool         isUserRequest,
                                    uint64_t*    bufferSize = nullptr);

    synStatus deviceGetCount(uint32_t* pCount);

    synStatus deviceGetCountByDeviceType(uint32_t* pCount, const synDeviceType deviceType);

    synStatus deviceCount(uint32_t count[synDeviceTypeSize]);

    synStatus deviceGetPCIBusId(char* pPciBusId, const int len);

    synStatus deviceGetFd(int *pFd);

    synStatus getDeviceInfo(synDeviceInfo* pDeviceInfo) const;
    synStatus getDeviceInfo(synDeviceInfoV2* pDeviceInfo) const;

    synStatus getDeviceName(char *pName, const int len);

    static synStatus getDeviceTypeAttribute(synDeviceType             deviceType,
                                            const synDeviceAttribute* deviceAttr,
                                            const unsigned            querySize,
                                            uint64_t*                 retVal);

    static synStatus getHalReader(synDeviceType deviceType, std::shared_ptr<HalReader>& rpHalReader);

    synStatus deviceGetModuleIds(uint32_t *pDeviceModuleIds, uint32_t*  size);

private:
    synStatus _addDevice();

    static const std::unordered_map<std::string, synDeviceType>& _stringToDeviceType();
    static std::vector<std::string>                              _deviceTypeToStrings(synDeviceType deviceType);

    mutable std::mutex               m_mutex;
    std::shared_ptr<DeviceInterface> m_device;
    std::atomic<bool>                m_deviceBeingReleased = false;
};
