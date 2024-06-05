#pragma once

#include "runtime/qman/common/device_downloader_interface.hpp"

class DeviceDownloaderMock : public DeviceDownloaderInterface
{
public:
    DeviceDownloaderMock();

    virtual ~DeviceDownloaderMock() = default;

    virtual synStatus downloadProgramCodeBuffer(uint64_t               recipeId,
                                                QueueInterface*        pPreviousStream,
                                                internalMemcopyParams& rMemcpyParams,
                                                uint64_t               hostBufferSize) const;

    virtual synStatus downloadProgramDataBuffer(QueueInterface*        pPreviousStream,
                                                internalMemcopyParams& rMemcpyParams,
                                                SpRecipeProgramBuffer* pRecipeProgramDataBuffer) const;

    static uint64_t s_downloadCounter;
};
