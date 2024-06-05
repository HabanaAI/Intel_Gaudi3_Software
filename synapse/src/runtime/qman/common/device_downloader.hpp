#pragma once

#include "device_downloader_interface.hpp"

class QueueInterface;

class DeviceDownloader : public DeviceDownloaderInterface
{
public:
    DeviceDownloader(QueueInterface& rStreamCopy);

    virtual ~DeviceDownloader() override = default;

    virtual synStatus downloadProgramCodeBuffer(uint64_t               recipeId,
                                                QueueInterface*        pPreviousStream,
                                                internalMemcopyParams& rMemcpyParams,
                                                uint64_t               hostBufferSize) const override;

    virtual synStatus downloadProgramDataBuffer(QueueInterface*        pPreviousStream,
                                                internalMemcopyParams& rMemcpyParams,
                                                SpRecipeProgramBuffer* pRecipeProgramDataBuffer) const override;

private:
    QueueInterface& m_rStreamCopy;
};
