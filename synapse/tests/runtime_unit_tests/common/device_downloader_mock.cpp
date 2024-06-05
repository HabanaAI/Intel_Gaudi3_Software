#include "device_downloader_mock.hpp"

uint64_t DeviceDownloaderMock::s_downloadCounter = 0;

DeviceDownloaderMock::DeviceDownloaderMock()
{
    s_downloadCounter = 0;
}

synStatus DeviceDownloaderMock::downloadProgramCodeBuffer(uint64_t               recipeId,
                                                          QueueInterface*        pPreviousStream,
                                                          internalMemcopyParams& rMemcpyParams,
                                                          uint64_t               hostBufferSize) const
{
    s_downloadCounter++;
    return synSuccess;
}

synStatus DeviceDownloaderMock::downloadProgramDataBuffer(QueueInterface*        pPreviousStream,
                                                          internalMemcopyParams& rMemcpyParams,
                                                          SpRecipeProgramBuffer* pRecipeProgramDataBuffer) const
{
    s_downloadCounter++;
    return synSuccess;
}
