#include "device_recipe_downloader_mock.hpp"

DeviceRecipeDownloaderMock::DeviceRecipeDownloaderMock(const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                       const CachedAndNot&             cpDmaChunksAmount,
                                                       synStatus                       downloadStatus)
: m_rDeviceAgnosticRecipeInfo(rDeviceAgnosticRecipeInfo), m_downloadStatus(downloadStatus)
{
    m_recipeStaticInfo.setCpDmaChunksAmount(EXECUTION_STAGE_ACTIVATE, cpDmaChunksAmount);
    m_recipeStaticInfo.setCpDmaChunksAmount(EXECUTION_STAGE_ENQUEUE, cpDmaChunksAmount);
}

synStatus DeviceRecipeDownloaderMock::processRecipe(uint64_t               workspaceAddress,
                                                    uint64_t               dcSizeCpDma,
                                                    uint64_t               dcSizeCommand,
                                                    bool                   programCodeInCache,
                                                    uint64_t               programCodeHandle,
                                                    uint64_t               programCodeDeviceAddress,
                                                    std::vector<uint64_t>& rProgramCodeDeviceAddresses)
{
    return synSuccess;
}

synStatus DeviceRecipeDownloaderMock::downloadExecutionBufferCache(bool                   programCodeInCache,
                                                                   uint64_t               programCodeHandle,
                                                                   std::vector<uint64_t>& rProgramCodeDeviceAddresses)
{
    return m_downloadStatus;
}

synStatus DeviceRecipeDownloaderMock::downloadProgramDataBufferCache(uint64_t              workspaceAddress,
                                                                     bool                  programDataInCache,
                                                                     uint64_t              programDataHandle,
                                                                     uint64_t              programDataDeviceAddress,
                                                                     SpRecipeProgramBuffer programDataRecipeBuffer)
{
    return m_downloadStatus;
}

synStatus DeviceRecipeDownloaderMock::downloadExecutionBufferWorkspace(QueueInterface* pComputeStream,
                                                                       bool            programCodeInCache,
                                                                       uint64_t        programCodeHandle,
                                                                       uint64_t        programCodeDeviceAddress,
                                                                       bool&           rIsDownloadWorkspaceProgramCode)
{
    return m_downloadStatus;
}

synStatus DeviceRecipeDownloaderMock::downloadProgramDataBufferWorkspace(QueueInterface* pComputeStream,
                                                                         bool            programDataInCache,
                                                                         uint64_t        programDataHandle,
                                                                         uint64_t        programDataDeviceAddress,
                                                                         bool& rIsDownloadWorkspaceProgramData,
                                                                         SpRecipeProgramBuffer programDataRecipeBuffer)
{
    return m_downloadStatus;
}

const DeviceAgnosticRecipeInfo& DeviceRecipeDownloaderMock::getDeviceAgnosticRecipeInfo() const
{
    return m_rDeviceAgnosticRecipeInfo;
}

const RecipeStaticInfo& DeviceRecipeDownloaderMock::getRecipeStaticInfo() const
{
    return m_recipeStaticInfo;
}
