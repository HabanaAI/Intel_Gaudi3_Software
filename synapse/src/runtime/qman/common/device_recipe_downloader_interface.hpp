#pragma once

#include "synapse_common_types.h"

#include <memory>
#include <vector>

class QueueInterface;
struct basicRecipeInfo;
struct DeviceAgnosticRecipeInfo;
class RecipeStaticInfo;
class DataChunksAllocatorCommandBuffer;

class RecipeProgramBuffer;
typedef std::shared_ptr<RecipeProgramBuffer> SpRecipeProgramBuffer;

class DeviceRecipeDownloaderInterface
{
public:
    virtual ~DeviceRecipeDownloaderInterface() = default;

    virtual synStatus processRecipe(uint64_t               workspaceAddress,
                                    uint64_t               dcSizeCpDma,
                                    uint64_t               dcSizeCommand,
                                    bool                   programCodeInCache,
                                    uint64_t               programCodeHandle,
                                    uint64_t               programCodeDeviceAddress,
                                    std::vector<uint64_t>& rProgramCodeDeviceAddresses) = 0;

    virtual synStatus downloadExecutionBufferCache(bool                   programCodeInCache,
                                                   uint64_t               programCodeHandle,
                                                   std::vector<uint64_t>& rProgramCodeDeviceAddresses) = 0;

    virtual synStatus downloadProgramDataBufferCache(uint64_t              workspaceAddress,
                                                     bool                  programDataInCache,
                                                     uint64_t              programDataHandle,
                                                     uint64_t              programDataDeviceAddress,
                                                     SpRecipeProgramBuffer programDataRecipeBuffer) = 0;

    virtual synStatus downloadExecutionBufferWorkspace(QueueInterface* pComputeStream,
                                                       bool            programCodeInCache,
                                                       uint64_t        programCodeHandle,
                                                       uint64_t        programCodeDeviceAddress,
                                                       bool&           rIsDownloadWorkspaceProgramCode) = 0;

    virtual synStatus downloadProgramDataBufferWorkspace(QueueInterface*       pComputeStream,
                                                         bool                  programDataInCache,
                                                         uint64_t              programDataHandle,
                                                         uint64_t              programDataDeviceAddress,
                                                         bool&                 rIsDownloadWorkspaceProgramData,
                                                         SpRecipeProgramBuffer programDataRecipeBuffer) = 0;

    virtual const DeviceAgnosticRecipeInfo& getDeviceAgnosticRecipeInfo() const = 0;

    virtual const RecipeStaticInfo& getRecipeStaticInfo() const = 0;
};
