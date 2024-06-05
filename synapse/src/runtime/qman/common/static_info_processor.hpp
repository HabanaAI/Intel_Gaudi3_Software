#pragma once

#include <map>
#include <vector>

#include "synapse_api_types.h"
#include "recipe_package_types.hpp"

struct recipe_t;
struct DeviceAgnosticRecipeInfo;
struct shape_plane_graph_t;
struct blob_t;
struct data_chunk_patch_point_t;
struct job_t;
struct patch_point_t;
struct program_t;
struct data_chunk_sm_patch_point_t;
struct sm_patch_point_t;
struct tensor_info_t;
struct section_group_t;

struct basicRecipeInfo;
struct DataChunkPatchPointsInfo;
struct DataChunkSmPatchPointsInfo;

class QmanDefinitionInterface;
class RecipeStaticInfo;
class DeviceMapperInterface;
class DataChunksAllocatorCommandBuffer;
class DeviceDownloaderInterface;

namespace generic
{
class CommandBufferPktGenerator;
}

// Todo Rename class to DeviceRecipeResourcesProcessor
class StaticInfoProcessor
{
public:
    static bool allocateResourcesAndProcessRecipe(DeviceMapperInterface*          pDeviceMapper,
                                                  const basicRecipeInfo&          rBasicRecipeHandle,
                                                  const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                  synDeviceType                   deviceType,
                                                  RecipeStaticInfo&               rRecipeInfo,
                                                  std::vector<uint64_t>&          rProgramCodeDeviceAddresses,
                                                  uint64_t                        sectionAddressForProgram,
                                                  bool                            programCodeInCache,
                                                  uint64_t                        workspaceAddress,
                                                  uint64_t                        dcSizeCpDma,
                                                  uint64_t                        dcSizeCommand);

    // Todo Rename to deallocateResources
    static void destroyProcessor(const DeviceMapperInterface& rDeviceMapper, RecipeStaticInfo& rRecipeInfo);

    // pRecipe - should be supplied only by the DSD test (SynGaudiRtDynamicShapesTest)
    static inline bool
    debugBuildDcPatchPointsOnBlobsChunksDatabases(shape_plane_graph_t*            pShapePlanRecipe,
                                                  synDeviceType                   deviceType,
                                                  const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                  RecipeStaticInfo&               rRecipeInfo,
                                                  const recipe_t&                 rRecipe,
                                                  uint64_t                        dcSizeCommand,
                                                  uint64_t                        patchingBlobsChunksSize,
                                                  uint64_t                        patchingBlobsChunkDataChunksAmount,
                                                  bool                            isDsd)
    {
        return _buildDcPatchPointsOnBlobsChunksDatabases(pShapePlanRecipe,
                                                         deviceType,
                                                         rDeviceAgnosticRecipeInfo,
                                                         rRecipeInfo,
                                                         rRecipe,
                                                         dcSizeCommand,
                                                         patchingBlobsChunksSize,
                                                         patchingBlobsChunkDataChunksAmount,
                                                         isDsd,
                                                         nullptr,
                                                         nullptr);
    }

private:
    // We will set information according to given workspaceAddress, and wll update dynamically, upon change
    // The blobs are patched as part of the dynamic processing, while the download is a static operation
    //
    // PS, currently:
    //  - noraml-mode workspace-address does not affect the program-code
    //  - gaudi-demo mode workspace-address does not change -> update is a TODO item
    //  [The TODO item - store the original WA, and add API that updates the content upon change,
    //  upon diff between the WA values
    //  At the moment it is in the statis-part, but it should probably be moved to the dynamic, as part of this
    //  change]
    static bool processRecipeInfo(const basicRecipeInfo&          rBasicRecipeHandle,
                                  const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                  RecipeStaticInfo&               rRecipeInfo,
                                  const DeviceMapperInterface&    rDeviceMapper,
                                  synDeviceType                   deviceType,
                                  uint64_t                        dcSizeCpDma,
                                  uint64_t                        dcSizeCommand,
                                  std::vector<uint64_t>&          rProgramCodeDeviceAddresses,
                                  uint64_t                        sectionAddressForProgram,
                                  bool                            programCodeInCache,
                                  uint64_t                        workspaceAddress);

    static bool createCpDmaForStaticBlobsAndStore(synDeviceType                deviceType,
                                                  const recipe_t&              rRecipe,
                                                  RecipeStaticInfo&            rRecipeInfo,
                                                  std::vector<uint64_t>&       rProgramCodeDeviceAddresses,
                                                  const DeviceMapperInterface& rDeviceMapper,
                                                  uint64_t                     sectionAddressForProgram);

    static bool calculateProgramChunksAmount(synDeviceType                   deviceType,
                                             const recipe_t&                 rRecipe,
                                             const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                             RecipeStaticInfo&               rRecipeInfo,
                                             uint64_t                        chunkSize);

    // Todo consider change the map to has table and reserve memory
    // A map between program index to CP_DMA packets amount
    typedef std::map<uint64_t, CachedAndNot> programsCpDmaPacketsDB;
    // A map between blob index to CP_DMA packets amount
    typedef std::map<uint64_t, CachedAndNot> blobsCpDmaPacketsAmountDB;
    // A vector of blob's host address to the device addresses of this split blob
    // NOTE: Order is important
    typedef std::vector<HostAndDevAddr> HostAndDevAddrVec;

    static void _createCpDmaForStaticBlobsAndStore(const recipe_t&   pRecipe,
                                                   RecipeStaticInfo* pRecipeInfo,
                                                   synDeviceType     deviceType);

    static bool _calculateProgramChunksAmount(synDeviceType                   deviceType,
                                              const recipe_t&                 rRecipe,
                                              const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                              RecipeStaticInfo&               rRecipeInfo,
                                              uint64_t                        dcSizeCpDma,
                                              eExecutionStage                 stage);

    static bool _calculateProgramCpDmaPktAmount(const recipe_t&            rRecipe,
                                                uint64_t                   programIndex,
                                                programsCpDmaPacketsDB&    programCpDmaAmountDB,
                                                blobsCpDmaPacketsAmountDB& blobsCpDmaAmountDB,
                                                uint64_t                   dcSizeCpDma,
                                                uint64_t                   cpDmaPacketSize,
                                                RecipeStaticInfo&          rRecipeInfo);

    static bool unmapAndClearProgramCode(const DeviceMapperInterface& rDeviceMapper, RecipeStaticInfo& rRecipeInfo);

    static bool mapAndSetProgramData(const recipe_t&              rRecipe,
                                     const DeviceMapperInterface& rDeviceMapper,
                                     RecipeStaticInfo&            rRecipeInfo);

    static bool mapAndSetProgramCode(const recipe_t&              rRecipe,
                                     const DeviceMapperInterface& rDeviceMapper,
                                     RecipeStaticInfo&            rRecipeInfo);

    static bool unmapAndClearMappedBuffers(const DeviceMapperInterface& rDeviceMapper, RecipeStaticInfo& rRecipeInfo);

    static void _clearPatchPointsDcInfoDbs(RecipeStaticInfo* pRecipeInfo);

    static void
    allocateCpDmaStaticBlobsBuffer(const recipe_t& rRecipe, RecipeStaticInfo& rRecipeInfo, synDeviceType deviceType);

    static void freeCpDmaStaticBlobsBuffer(RecipeStaticInfo& rRecipeInfo);

    static bool _buildDcPatchPointsOnBlobsChunksDatabases(shape_plane_graph_t*            pShapePlanRecipe,
                                                          synDeviceType                   deviceType,
                                                          const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                          RecipeStaticInfo&               rRecipeInfo,
                                                          const recipe_t&                 rRecipe,
                                                          uint64_t                        dcSizeCommand,
                                                          uint64_t                        patchingBlobsChunksSize,
                                                          uint64_t        patchingBlobsChunkDataChunksAmount,
                                                          bool            isDsd,
                                                          const blob_t**  blobs,
                                                          uint64_t* const blobsBuffer);

    static bool _emplaceDcPatchPointsOnBlobsChunksDatabases(
                                                const recipe_t&        rRecipe,
                                                RecipeStaticInfo&      rRecipeInfo,
                                                eExecutionStage        stage,
                                                const section_group_t* sectionGroupPpInfo,
                                                uint64_t               sgPpFirstEnqueueIndex,
                                                const blob_t*          patchOnDataChunksBlobs,
                                                const uint8_t*         patchingBlobsBuffer,
                                                uint64_t               dcSizeCommand,
                                                uint64_t               programCommandsChunksAmount,
                                                bool                   isSobj);

    static bool _updateSmPatchPointsInfoDb(const recipe_t&                    rRecipe,
                                           shape_plane_graph_t*               pShapePlanRecipe,
                                           DataChunkSmPatchPointsInfo* const& pPatchPointsDataChunksInfoDb,
                                           const blob_t* const&               blobs,
                                           uint64_t                           programCommandsChunksAmount,
                                           uint64_t                           dcSizeCommand,
                                           uint64_t                           blobsBufferAddress);

    static bool _updatePatchPointsInfoDb(DataChunkPatchPointsInfo* const& pPatchPointsDataChunksInfoDb,
                                         uint64_t                         programCommandsChunksAmount,
                                         uint64_t                         dcSizeCommand,
                                         data_chunk_patch_point_t*        dataChunkPatchPoints,
                                         const uint32_t*                  sectionGroupPpIndices,
                                         const uint64_t                   typePatchPointAmount,
                                         const patch_point_t* const&      patchPoints,
                                         uint64_t                         patchPointsAmount,
                                         const blob_t* const&             blobs,
                                         uint64_t                         blobsAmount,
                                         uint64_t                         blobsBufferAddress);

    static bool _buildSmPatchPointsOnDataChunkDb(const recipe_t&      rRecipe,
                                                 shape_plane_graph_t* pShapePlanRecipe,
                                                 RecipeStaticInfo&    rRecipeInfo,
                                                 uint64_t             programCommandsChunksAmount,
                                                 uint64_t             dcSizeCommand,
                                                 const blob_t* const& blobs,
                                                 uint64_t             blobsAmount,
                                                 uint64_t             patchingBlobsBufferAddress);

    static bool calcPatchableBlobsOffsetsAndStore(shape_plane_graph_t*            pShapePlanRecipe,
                                                  synDeviceType                   deviceType,
                                                  const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                  RecipeStaticInfo&               rRecipeInfo,
                                                  const recipe_t&                 rRecipe,
                                                  uint64_t                        dcSizeCommand,
                                                  uint64_t                        patchingBlobsChunksSize,
                                                  uint64_t                        patchingBlobsChunksDataChunksAmount);

    static void _calcPatchableBlobsOffsetsAndStore(const blob_t*&    pRecipeBlobsElements,
                                                   uint64_t          blobsBuffer,
                                                   uint64_t          blobsAmount,
                                                   uint64_t          dcSizeCommand,
                                                   RecipeStaticInfo& rRecipeInfo);

    static uint32_t _calculateTotalAmountOfSmPatchPoints(shape_plane_graph_t* pShapePlanRecipe);

    static generic::CommandBufferPktGenerator* _getPacketGenerator(synDeviceType deviceType);

    static bool allocateArbitrationPackets(RecipeStaticInfo* pRecipeInfo, synDeviceType deviceType);

    static void freeArbitrationPackets(RecipeStaticInfo& rRecipeInfo);

    static DataChunkPatchPointsInfo* const allocatePatchPointsDcInfoDbs(RecipeStaticInfo& rRecipeInfo,
                                                                        eExecutionStage   executionStage,
                                                                        uint64_t          patchPointsTypeId,
                                                                        uint32_t          chunksAmount,
                                                                        uint64_t          patchPointsAmount,
                                                                        uint32_t          dcSizeCommand,
                                                                        bool              isSobj);

    static void clearStagePatchPointsDcInfoDbs(RecipeStaticInfo* pRecipeInfo, eExecutionStage executionStage);

    static bool storeNonPatchableBlobsDeviceAddresses(const DeviceMapperInterface& rDeviceMapper,
                                                      const recipe_t&              rRecipe,
                                                      RecipeStaticInfo&            rRecipeInfo,
                                                      std::vector<uint64_t>&       rProgramCodeDeviceAddresses);

    static bool _addProgramCodeBlobDeviceAddress(const recipe_t&                        rRecipe,
                                                 uint8_t*                               currentAddrInBuffer,
                                                 uint64_t                               blobIdx,
                                                 std::vector<uint64_t>::const_iterator& blockAddrEnd,
                                                 std::vector<uint64_t>::const_iterator& blocksAddrIterator,
                                                 uint64_t&                              blockSizeLeft,
                                                 RecipeStaticInfo*                      pRecipeInfo,
                                                 uint64_t                               blockSize);

    friend class UTGaudiRtDynamicShapesTest;
};