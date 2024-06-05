#pragma once

#include "device_recipe_downloader_interface.hpp"
#include "synapse_common_types.h"
#include "define_synapse_common.hpp"
#include "runtime/qman/common/recipe_static_information.hpp"
#include "infra/threads/single_execution_owner.hpp"

class DeviceDownloaderInterface;
class DeviceMapperInterface;
struct DeviceAgnosticRecipeInfo;
namespace generic
{
class CommandBufferPktGenerator;
}
class QmanDefinitionInterface;

class DeviceRecipeDownloader : public DeviceRecipeDownloaderInterface
{
public:
    DeviceRecipeDownloader(synDeviceType                             deviceType,
                           const QmanDefinitionInterface&            rQmansDefinition,
                           const generic::CommandBufferPktGenerator& rCmdBuffPktGenerator,
                           DeviceDownloaderInterface&                rDeviceDownloader,
                           DeviceMapperInterface&                    rDeviceMapper,
                           const DeviceAgnosticRecipeInfo&           rDeviceAgnosticRecipeInfo,
                           const basicRecipeInfo&                    rBasicRecipeInfo,
                           uint64_t                                  recipeId);

    virtual ~DeviceRecipeDownloader() = default;

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

    const DeviceAgnosticRecipeInfo& getDeviceAgnosticRecipeInfo() const override { return m_rDeviceAgnosticRecipeInfo; }

    const RecipeStaticInfo& getRecipeStaticInfo() const override { return m_recipeStaticInfo; }

    void notifyRecipeDestroy();

private:
    static synStatus _processRecipe(synDeviceType                   deviceType,
                                    DeviceMapperInterface&          rDeviceMapper,
                                    uint64_t                        dcSizeCpDma,
                                    uint64_t                        dcSizeCommand,
                                    const basicRecipeInfo&          rBasicRecipeInfo,
                                    const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                    uint64_t                        workspaceAddress,
                                    RecipeStaticInfo*               pRecipeInfo,
                                    SingleExecutionOwner*           pSingleExecutionOwner,
                                    std::vector<uint64_t>&          rProgramCodeDeviceAddresses,
                                    uint64_t                        programCodeDeviceAddress,
                                    bool                            programCodeInCache,
                                    uint64_t                        programCodeHandle);

    static synStatus _downloadExecutionBufferCache(uint64_t                   recipeId,
                                                   DeviceDownloaderInterface* pDeviceDownloader,
                                                   const basicRecipeInfo&     rBasicRecipeInfo,
                                                   SingleExecutionOwner*      pSingleExecutionOwner,
                                                   std::vector<uint64_t>&     rProgramCodeDeviceAddresses,
                                                   bool                       programCodeInCache,
                                                   uint64_t                   programCodeHandle);

    static void _generateDownloadProgramCodeCacheMemCopyParams(const recipe_t&              rRecipe,
                                                               const std::vector<uint64_t>& rProgramCodeDeviceAddresses,
                                                               internalMemcopyParams&       rMemcpyParams,
                                                               uint64_t&                    hostBufferSize);

    static synStatus _downloadProgramDataBufferToCache(const DeviceDownloaderInterface& rDownloader,
                                                       const basicRecipeInfo&           rBasicRecipeInfo,
                                                       uint64_t                         workspaceAddress,
                                                       RecipeStaticInfo*                pRecipeInfo,
                                                       SingleExecutionOwner*            pSingleExecutionOwner,
                                                       bool                             programDataInCache,
                                                       uint64_t                         programDataHandle,
                                                       uint64_t                         programDataDeviceAddress,
                                                       SpRecipeProgramBuffer            programDataRecipeBuffer);

    const synDeviceType m_deviceType;

    const QmanDefinitionInterface& m_rQmansDefinition;

    const generic::CommandBufferPktGenerator& m_rCmdBuffPktGenerator;

    DeviceDownloaderInterface& m_rDeviceDownloader;

    DeviceMapperInterface& m_rDeviceMapper;

    const basicRecipeInfo& m_rBasicRecipeInfo;

    const DeviceAgnosticRecipeInfo& m_rDeviceAgnosticRecipeInfo;

    RecipeStaticInfo m_recipeStaticInfo;

    // A user requires to perform two operations, with the following order:
    //      1) Recipe Static Processing (with new device-addresses)
    //      2) DWL both parts to the recipe-cache (device)
    // Each operation requires SEO, while there is no restriction that the same user will be the handler of both
    //
    // We are using the RC-Handle to assure unique-ownership is taken, for the DWL to those acquired RC-blocks.
    //
    // We will use the same PRG-Code's handle for the SEO of the processing.
    // "Working from Workspace" - Seems to be broken.
    //                          - We perform the processing, even for those cases,
    //                            but has the handle is unique (WS-address), a second user may update that data,
    //                            while still working on it. There is no critical-section protection (mutex) over
    //                            that code-part
    //                          - Possible fix - process *after* locking the Launch operation
    //                          - We will use the programCodeHandle for taking ownership
    //                          - When PRG-Code is in WS, we will force ownership
    SingleExecutionOwner m_processingSingleExecutionOwner;
    SingleExecutionOwner m_prgCodeDwlToCacheSingleExecutionOwner;
    SingleExecutionOwner m_prgDataDwlToCacheSingleExecutionOwner;

    uint64_t m_recipeId;
};
