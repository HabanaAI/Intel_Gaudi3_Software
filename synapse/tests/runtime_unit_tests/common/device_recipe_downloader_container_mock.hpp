#pragma once

#include "runtime/qman/common/device_recipe_downloader_container.hpp"

class DeviceRecipeDownloaderContainerMock : public DeviceRecipeDownloaderContainerInterface
{
public:
    DeviceRecipeDownloaderContainerMock(DeviceRecipeDownloaderInterface* pDeviceRecipeDownloader)
    : m_pDeviceRecipeDownloader(pDeviceRecipeDownloader)
    {
    }

    virtual void addDeviceRecipeDownloader(uint32_t                          amountOfEnginesInArbGroup,
                                           InternalRecipeHandle&             rRecipeHandle,
                                           DeviceRecipeDownloaderInterface*& rpDeviceRecipeDownloader) override
    {
        rpDeviceRecipeDownloader = m_pDeviceRecipeDownloader;
    }

    DeviceRecipeDownloaderInterface* m_pDeviceRecipeDownloader;
};
