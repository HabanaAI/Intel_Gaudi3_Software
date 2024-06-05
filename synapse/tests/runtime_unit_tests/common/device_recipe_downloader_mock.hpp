#pragma once

#include "runtime/qman/common/device_recipe_downloader_interface.hpp"
#include "runtime/qman/common/recipe_static_information.hpp"

class DeviceRecipeDownloaderMock : public DeviceRecipeDownloaderInterface
{
public:
    DeviceRecipeDownloaderMock(const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                               const CachedAndNot&             cpDmaChunksAmount,
                               synStatus                       downloadStatus);

    virtual synStatus processRecipe(uint64_t               workspaceAddress,
                                    uint64_t               dcSizeCpDma,
                                    uint64_t               dcSizeCommand,
                                    bool                   programCodeInCache,
                                    uint64_t               programCodeHandle,
                                    uint64_t               programCodeDeviceAddress,
                                    std::vector<uint64_t>& rProgramCodeDeviceAddresses) override;

    virtual synStatus downloadExecutionBufferCache(bool                   programCodeInCache,
                                                   uint64_t               programCodeHandle,
                                                   std::vector<uint64_t>& rProgramCodeDeviceAddresses) override;

    virtual synStatus downloadProgramDataBufferCache(uint64_t              workspaceAddress,
                                                     bool                  programDataInCache,
                                                     uint64_t              programDataHandle,
                                                     uint64_t              programDataDeviceAddress,
                                                     SpRecipeProgramBuffer programDataRecipeBuffer) override;

    virtual synStatus downloadExecutionBufferWorkspace(QueueInterface* pComputeStream,
                                                       bool            programCodeInCache,
                                                       uint64_t        programCodeHandle,
                                                       uint64_t        programCodeDeviceAddress,
                                                       bool&           rIsDownloadWorkspaceProgramCode) override;

    virtual synStatus downloadProgramDataBufferWorkspace(QueueInterface*       pComputeStream,
                                                         bool                  programDataInCache,
                                                         uint64_t              programDataHandle,
                                                         uint64_t              programDataDeviceAddress,
                                                         bool&                 rIsDownloadWorkspaceProgramData,
                                                         SpRecipeProgramBuffer programDataRecipeBuffer) override;

    const DeviceAgnosticRecipeInfo& getDeviceAgnosticRecipeInfo() const override;

    const RecipeStaticInfo& getRecipeStaticInfo() const override;

private:
    const DeviceAgnosticRecipeInfo& m_rDeviceAgnosticRecipeInfo;
    RecipeStaticInfo                m_recipeStaticInfo;
    const synStatus                 m_downloadStatus;
};