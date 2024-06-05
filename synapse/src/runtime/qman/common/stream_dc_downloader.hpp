#pragma once

#include "synapse_common_types.h"
#include "command_submission_data_chunks.hpp"
#include "dynamic_info_processor_interface.hpp"
#include "defs.h"

class QmanDefinitionInterface;
struct DeviceAgnosticRecipeInfo;
class RecipeStaticInfo;

namespace generic
{
class CommandBufferPktGenerator;
}

class StreamDcDownloader
{
public:
    StreamDcDownloader(synDeviceType deviceType, uint32_t physicalQueueOffset);

    virtual ~StreamDcDownloader() = default;
    /**
     * @brief Copies the program (patchable) blobs into DC and create and copies CP-DMAs to their own DC
     *        Creates mapping between a blob and its DC
     */
    void downloadProgramCodeBuffer(uint64_t                        arbMasterBaseQmanId,
                                   CommandSubmissionDataChunks&    csDataChunks,
                                   std::vector<uint64_t>&          patchableBlobsDcAddresses,
                                   DataChunksDB&                   cpDmaDataChunks,
                                   uint64_t                        dcSizeCommand,
                                   const recipe_t*                 recipeHandle,
                                   const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                   const RecipeStaticInfo&         rRecipeStaticInfo,
                                   const blob_t*                   blobsForCpDma,
                                   eExecutionStage                 stage,
                                   const uint8_t*                  patchableBlobsBuffer,
                                   staticBlobsDeviceAddresses*     staticBlobsDevAddresses = nullptr) const;

    /**
     * @brief copy patchable blobs buffer into (stream copy) to DC
     *        and store the DC base addresses
     */
    static void downloadPatchableBlobsBuffer(const uint8_t*         patchableBlobsBuffer,
                                             uint64_t               patchableBlobsBufferSize,
                                             DataChunksDB&          commandsDataChunks,
                                             std::vector<uint64_t>& commandsDcsAddresses);

private:
    void _copyStaticBlobsCpDmaToDc(DataChunksDB&                cpDmaDataChunks,
                                   uint32_t&                    curDataChunkIdx,
                                   const blob_t*                pCurrentBlob,
                                   uint64_t                     blobIndex,
                                   CommandSubmissionDataChunks& csDataChunks,
                                   uint64_t                     engineId,
                                   const RecipeStaticInfo&      rRecipeStaticInfo,
                                   staticBlobsDeviceAddresses*  staticBlobsDevAddresses) const;

    void _addPatchableBlobsSingleCpDmaToDc(const blob_t*                pCurrentBlob,
                                           DataChunksDB&                cpDmaDataChunks,
                                           uint32_t&                    curDataChunkIdx,
                                           uint64_t                     engineId,
                                           uint64_t                     blobIndex,
                                           CommandSubmissionDataChunks& csDataChunks,
                                           std::vector<uint64_t>&       patchableBlobsDcAddresses,
                                           const RecipeStaticInfo&      rRecipeStaticInfo) const;

    void _addPatchableBlobsMultipleCpDmaToDc(const blob_t*                pCurrentBlob,
                                             DataChunksDB&                cpDmaDataChunks,
                                             uint32_t&                    curDataChunkIdx,
                                             uint64_t                     engineId,
                                             uint64_t                     blobIndex,
                                             CommandSubmissionDataChunks& csDataChunks,
                                             std::vector<uint64_t>&       patchableBlobsDcAddresses,
                                             const uint8_t*               patchableBlobsBuffer,
                                             uint64_t                     dcSizeCommand) const;

    static void
    _addSingleCpDmaMapping(DataChunk* pCpDmaDataChunk, uint64_t engineId, CommandSubmissionDataChunks& csDataChunks);

    void _fillDataChunksAndStore(DataChunksDB&                cpDmaDataChunks,
                                 uint32_t&                    curDataChunkIdx,
                                 void*                        cpDmaPktAddr,
                                 uint64_t                     pktSize,
                                 uint64_t                     engineId,
                                 CommandSubmissionDataChunks& csDataChunks) const;

    void _fillDataChunksAndStoreForPatchable(DataChunksDB&                cpDmaDataChunks,
                                             uint32_t&                    curDataChunkIdx,
                                             uint64_t                     blobsAddressInDc,
                                             uint64_t                     currentCpDmaSize,
                                             uint64_t                     pktSize,
                                             uint64_t                     engineId,
                                             CommandSubmissionDataChunks& csDataChunks) const;

    static void _resetDataChunks(DataChunksDB& dataChunks);

    static void _updateStageDataChunksInfo(DataChunksDB&            cpDmaDataChunks,
                                           uint32_t&                prevDataChunkIdx,
                                           uint32_t                 dataChunkIdx,
                                           uint32_t&                curStageIdx,
                                           std::vector<StagedInfo>& stagesInfo);

    void _copySingleProgramToDataChunks(const recipe_t*                 recipeHandle,
                                        const blob_t*                   blobs,
                                        const blob_t*                   blobsForCpDma,
                                        staticBlobsDeviceAddresses*     staticBlobsDevAddresses,
                                        CommandSubmissionDataChunks&    csDataChunks,
                                        DataChunksDB&                   cpDmaDataChunks,
                                        std::vector<uint64_t>&          patchableBlobsDcAddresses,
                                        const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                        const RecipeStaticInfo&         rRecipeStaticInfo,
                                        std::vector<uint32_t>&          stagesNodes,
                                        std::vector<StagedInfo>&        stagesInfo,
                                        uint64_t                        engineId,
                                        uint64_t                        programIndex,
                                        uint64_t                        arbCommandSize,
                                        uint64_t                        arbSetCmdHostAddress,
                                        uint64_t                        arbReleaseCmdHostAddress,
                                        uint32_t&                       curDataChunkIdx,
                                        uint32_t&                       prevDataChunkIdx,
                                        uint32_t                        physicalQueueOffset,
                                        eExecutionStage                 stage,
                                        bool                            useArbitration,
                                        bool                            syncWithStreamMaster,
                                        bool                            isArbMasterQman,
                                        bool                            stageSubmissionEnabled) const;

    inline DataChunk* _getChunk(DataChunksDB& cpDmaDataChunks, uint32_t dataChunkIdx) const
    {
        const uint32_t dataChunkIdxMax = cpDmaDataChunks.size();
        HB_ASSERT((dataChunkIdx < dataChunkIdxMax),
                  "invalid data chunk index access dataChunkIdx {} dataChunkIdxMax {}",
                  dataChunkIdx,
                  dataChunkIdxMax);
        return cpDmaDataChunks[dataChunkIdx];
    }

    const synDeviceType                 m_deviceType;
    const uint32_t                      m_physicalQueueOffset;
    generic::CommandBufferPktGenerator* m_packetGenerator;
    QmanDefinitionInterface*            m_qmansDef;
    bool                                m_useArbitration;
};
