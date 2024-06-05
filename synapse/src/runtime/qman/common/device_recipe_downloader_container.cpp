#include "device_recipe_downloader_container.hpp"
#include "device_recipe_downloader.hpp"
#include "device_recipe_addresses_generator.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

DeviceRecipeDownloaderContainer::DeviceRecipeDownloaderContainer(
    synDeviceType                             deviceType,
    DeviceRecipeAddressesGenerator&           rDeviceRecipeAddressesGenerator,
    const QmanDefinitionInterface&            rQmansDefinition,
    const generic::CommandBufferPktGenerator& rCmdBuffPktGenerator,
    DeviceDownloaderInterface&                rDeviceDownloader,
    DeviceMapperInterface&                    rDeviceMapper)
: m_deviceType(deviceType),
  m_rDeviceRecipeAddressesGenerator(rDeviceRecipeAddressesGenerator),
  m_rQmansDefinition(rQmansDefinition),
  m_rCmdBuffPktGenerator(rCmdBuffPktGenerator),
  m_rDeviceDownloader(rDeviceDownloader),
  m_rDeviceMapper(rDeviceMapper)
{
}

void DeviceRecipeDownloaderContainer::addDeviceRecipeDownloader(
    uint32_t                          amountOfEnginesInArbGroup,
    InternalRecipeHandle&             rRecipeHandle,
    DeviceRecipeDownloaderInterface*& rpDeviceRecipeDownloader)
{
    std::lock_guard<std::mutex> lock(m_deviceRecipeInfoMapMutex);

    auto iter = m_deviceRecipeInfoMap.find(&rRecipeHandle);
    if (iter != m_deviceRecipeInfoMap.end())
    {
        rpDeviceRecipeDownloader = iter->second;
    }
    else
    {
        DeviceRecipeDownloader* pDeviceRecipeDownloader =
            new DeviceRecipeDownloader(m_deviceType,
                                       m_rQmansDefinition,
                                       m_rCmdBuffPktGenerator,
                                       m_rDeviceDownloader,
                                       m_rDeviceMapper,
                                       rRecipeHandle.deviceAgnosticRecipeHandle,
                                       rRecipeHandle.basicRecipeHandle,
                                       rRecipeHandle.recipeSeqNum);

        m_deviceRecipeInfoMap.insert({&rRecipeHandle, pDeviceRecipeDownloader});

        rpDeviceRecipeDownloader = pDeviceRecipeDownloader;
    }
}

void DeviceRecipeDownloaderContainer::removeDeviceRecipeInfo(InternalRecipeHandle& rRecipeHandle)
{
    std::lock_guard<std::mutex> lock(m_deviceRecipeInfoMapMutex);

    const auto& iter = m_deviceRecipeInfoMap.find(&rRecipeHandle);
    if (iter != m_deviceRecipeInfoMap.end())
    {
        DeviceRecipeDownloader* pDeviceRecipeDownloader = iter->second;
        pDeviceRecipeDownloader->notifyRecipeDestroy();
        m_rDeviceRecipeAddressesGenerator.notifyRecipeDestroy(rRecipeHandle.basicRecipeHandle.recipe);
        delete pDeviceRecipeDownloader;
        m_deviceRecipeInfoMap.erase(iter);
    }
}

void DeviceRecipeDownloaderContainer::removeAllDeviceRecipeInfo()
{
    std::lock_guard<std::mutex> lock(m_deviceRecipeInfoMapMutex);

    for (auto const& entry : m_deviceRecipeInfoMap)
    {
        InternalRecipeHandle&   rRecipeHandle           = *entry.first;
        DeviceRecipeDownloader* pDeviceRecipeDownloader = entry.second;
        pDeviceRecipeDownloader->notifyRecipeDestroy();
        m_rDeviceRecipeAddressesGenerator.notifyRecipeDestroy(rRecipeHandle.basicRecipeHandle.recipe);
        delete pDeviceRecipeDownloader;
    }

    m_deviceRecipeInfoMap.clear();
}
