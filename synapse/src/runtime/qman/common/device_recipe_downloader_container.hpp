#pragma once

#include <unordered_map>
#include <mutex>
#include "synapse_common_types.h"

struct InternalRecipeHandle;
class DeviceRecipeDownloaderInterface;
class DeviceRecipeDownloader;
namespace generic
{
class CommandBufferPktGenerator;
}
class QmanDefinitionInterface;
class DeviceDownloaderInterface;
class DeviceMapperInterface;
struct DeviceAgnosticRecipeInfo;
struct basicRecipeInfo;
class DeviceRecipeAddressesGenerator;

class DeviceRecipeDownloaderContainerInterface
{
public:
    virtual ~DeviceRecipeDownloaderContainerInterface() = default;

    virtual void addDeviceRecipeDownloader(uint32_t                          amountOfEnginesInArbGroup,
                                           InternalRecipeHandle&             rRecipeHandle,
                                           DeviceRecipeDownloaderInterface*& rpDeviceRecipeDownloader) = 0;
};

class DeviceRecipeDownloaderContainer : public DeviceRecipeDownloaderContainerInterface
{
public:
    DeviceRecipeDownloaderContainer(synDeviceType                             deviceType,
                                    DeviceRecipeAddressesGenerator&           rDeviceRecipeAddressesGenerator,
                                    const QmanDefinitionInterface&            rQmansDefinition,
                                    const generic::CommandBufferPktGenerator& rCmdBuffPktGenerator,
                                    DeviceDownloaderInterface&                rDeviceDownloader,
                                    DeviceMapperInterface&                    rDeviceMapper);

    virtual ~DeviceRecipeDownloaderContainer() = default;

    virtual void addDeviceRecipeDownloader(uint32_t                          amountOfEnginesInArbGroup,
                                           InternalRecipeHandle&             rRecipeHandle,
                                           DeviceRecipeDownloaderInterface*& rpDeviceRecipeDownloader) override;

    void removeDeviceRecipeInfo(InternalRecipeHandle& rRecipeHandle);

    void removeAllDeviceRecipeInfo();

private:
    typedef std::unordered_map<InternalRecipeHandle*, DeviceRecipeDownloader*>
        InternalRecipeHandleToDeviceRecipeInfoMap;

    const synDeviceType                       m_deviceType;
    DeviceRecipeAddressesGenerator&           m_rDeviceRecipeAddressesGenerator;
    const QmanDefinitionInterface&            m_rQmansDefinition;
    const generic::CommandBufferPktGenerator& m_rCmdBuffPktGenerator;
    DeviceDownloaderInterface&                m_rDeviceDownloader;
    DeviceMapperInterface&                    m_rDeviceMapper;

    InternalRecipeHandleToDeviceRecipeInfoMap m_deviceRecipeInfoMap;
    mutable std::mutex                        m_deviceRecipeInfoMapMutex;
};
