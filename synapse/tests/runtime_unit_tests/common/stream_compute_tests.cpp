#include "runtime/qman/common/queue_info.hpp"
#include <gtest/gtest.h>
#include "habana_global_conf_runtime.h"
#include "hpp/syn_context.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"
#include "runtime/qman/common/device_mapper.hpp"
#include "runtime/qman/common/recipe_static_information.hpp"
#include "runtime/qman/common/static_info_processor.hpp"
#include "runtime/qman/common/stream_master_helper_mock.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "device_agnostic_recipe_processor.hpp"
#include "runtime/qman/common/queue_compute_qman.hpp"
#include "stream_copy_mock.hpp"
#include "physical_queues_manager_mock.hpp"
#include "dev_memory_alloc_mock.hpp"
#include "device_recipe_addresses_generator_mock.hpp"
#include "device_mapper_mock.hpp"
#include "dynamic_info_processor_mock.hpp"
#include "device_downloader_mock.hpp"
#include "device_recipe_downloader_mock.hpp"
#include "device_recipe_downloader_container_mock.hpp"
#include "common/work_completion_manager_mock.hpp"
#include "submit_command_buffers_mock.hpp"
#include "recipe_processing_utils.hpp"

const uint8_t API_ID = 0;

class UTStreamComputeTest : public ::testing::Test
{
public:
    void recipeLaunch(synDeviceType             deviceType,
                      DynamicInfoProcessorMock& rDynamicInfoProcessor,
                      InternalRecipeHandle&     rInternalRecipeHandle,
                      uint32_t                  launchTensorsAmount,
                      unsigned                  launchNum,
                      const CachedAndNot&       rCpDmaChunksAmount,
                      synStatus                 recipeDownloadStatus);

    void
    singleProgramCodeBlobRecipeLaunch(synDeviceType deviceType, unsigned launchNum, synStatus recipeDownloadStatus);

    void singlePatchableProgramCodeBlobRecipeLaunch(synDeviceType deviceType, unsigned launchNum);

    void checkWcmOrder(synDeviceType deviceType);

    void checkNotifyCsCompleted(synDeviceType deviceType, bool csFailed);

private:
    static uint64_t getStreamMasterQueueIdForCompute(synDeviceType deviceType);
};

void UTStreamComputeTest::recipeLaunch(synDeviceType             deviceType,
                                       DynamicInfoProcessorMock& rDynamicInfoProcessor,
                                       InternalRecipeHandle&     rInternalRecipeHandle,
                                       uint32_t                  launchTensorsAmount,
                                       unsigned                  launchNum,
                                       const CachedAndNot&       rCpDmaChunksAmount,
                                       synStatus                 recipeDownloadStatus)
{
    // Initialization
    syn::Context context;

    // Construction
    const BasicQueueInfo                basicQueueInfo {{0, 0},
                                         INTERNAL_STREAM_TYPE_COMPUTE,
                                         TRAINING_QUEUE_COMPUTE_0,
                                         0,
                                         std::make_shared<QueueInfo>(0, 0, 0, 0)};
    PhysicalQueuesManagerMock           physicalStreamsManager;
    WorkCompletionManagerMock           workCompletionManager;
    SubmitCommandBuffersMock            submitter;
    DevMemoryAllocMock                  devMemoryAlloc;
    DeviceRecipeAddressesGeneratorMock  devRecipeAddress;
    DeviceRecipeDownloaderMock          deviceRecipeDownloader(rInternalRecipeHandle.deviceAgnosticRecipeHandle,
                                                      rCpDmaChunksAmount,
                                                      recipeDownloadStatus);
    DeviceRecipeDownloaderContainerMock deviceRecipeDownloaderContainer(&deviceRecipeDownloader);
    QueueMock                           streamCopy;
    std::unique_ptr<StreamMasterHelperInterface> streamMasterHelper = std::make_unique<StreamMasterHelperMock>();

    QueueComputeQman stream {basicQueueInfo,
                             0,
                             0,
                             false,
                             deviceType,
                             &physicalStreamsManager,
                             workCompletionManager,
                             submitter,
                             devMemoryAlloc,
                             devRecipeAddress,
                             deviceRecipeDownloaderContainer,
                             streamCopy,
                             std::move(streamMasterHelper)};

    stream.initAllocators();

    // Launch
    std::array<synLaunchTensorInfoExt, 0x10> launchTensorsInfo;

    const uint64_t tensorDeviceAddr = 0x10;
    for (unsigned tensorIndex = 0; tensorIndex < launchTensorsAmount; tensorIndex++)
    {
        launchTensorsInfo[tensorIndex].tensorName     = nullptr;
        launchTensorsInfo[tensorIndex].pTensorAddress = (uint64_t)&tensorDeviceAddr;
        launchTensorsInfo[tensorIndex].tensorType     = HOST_TO_DEVICE_TENSOR;
        memset(launchTensorsInfo[tensorIndex].tensorSize, 0, sizeof(launchTensorsInfo[0].tensorSize));
        launchTensorsInfo[tensorIndex].tensorId = tensorIndex;
    }

    uint64_t                workspaceAddress = 0;
    uint32_t                flags            = 0;
    DeviceMapperMock        deviceMapper;
    EventWithMappedTensorDB eventList;

    for (unsigned launchIter = 0; launchIter < launchNum; launchIter++)
    {
        synStatus status = stream.launch(launchTensorsInfo.data(),
                                         launchTensorsAmount,
                                         workspaceAddress,
                                         &rInternalRecipeHandle,
                                         0,
                                         flags,
                                         eventList,
                                         API_ID);
        EXPECT_EQ(status, recipeDownloadStatus);
    }
    workCompletionManager.notifyCsCompleted(false);

    EXPECT_EQ(devRecipeAddress.isClean(), true);
}

void UTStreamComputeTest::singleProgramCodeBlobRecipeLaunch(synDeviceType deviceType,
                                                            unsigned      launchNum,
                                                            synStatus     recipeDownloadStatus)
{
    recipe_t       recipe {};

    const uint64_t executionBlobsBufferSize = 0x10;
    char      executionBlobsBuffer[executionBlobsBufferSize] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint64_t* pExecutionBlobsBuffer    = (uint64_t*)&executionBlobsBuffer[0];
    recipe.execution_blobs_buffer_size = executionBlobsBufferSize;
    recipe.execution_blobs_buffer      = pExecutionBlobsBuffer;
    const uint32_t blobNum             = 1;
    blob_t         blobs[blobNum];
    blob_t*        pBlobs = &blobs[0];
    blobs[0].blob_type    = {0, 1, 0, 0x0};
    blobs[0].size         = executionBlobsBufferSize;
    blobs[0].data         = pExecutionBlobsBuffer;
    recipe.blobs_nr       = blobNum;
    recipe.blobs          = pBlobs;

    uint64_t blobIndices[1];
    blobIndices[0] = 0;

    uint32_t  programsNr = 1;        // The number of programs on the recipe.
    program_t programs[programsNr];  // The recipe's programs.
    programs[0].program_length = 1;
    programs[0].blob_indices   = &blobIndices[0];

    recipe.programs_nr = programsNr;
    recipe.programs    = &programs[0];

    RecipeProcessingUtils::getExecutionJobs(deviceType, recipe.execute_jobs_nr, recipe.execute_jobs);

    uint32_t       program_blobs_nr[] {programs[0].program_length};
    node_program_t node[] {{&program_blobs_nr[0], 0}};
    recipe.node_exe_list = &node[0];
    recipe.node_nr       = 1;

    uint64_t workspace_sizes[3] {0, 0, recipe.execution_blobs_buffer_size};
    recipe.workspace_nr    = sizeof(workspace_sizes) / sizeof(uint64_t);
    recipe.workspace_sizes = &workspace_sizes[0];

    gc_conf_t recipeConf[] {{deviceType, gc_conf_t::DEVICE_TYPE},
                            {GCFG_TPC_ENGINES_ENABLED_MASK.value(), gc_conf_t::TPC_ENGINE_MASK}};
    recipe.recipe_conf_nr     = sizeof(recipeConf) / sizeof(gc_conf_t);
    recipe.recipe_conf_params = &recipeConf[0];

    InternalRecipeHandle internalRecipeHandle;
    internalRecipeHandle.basicRecipeHandle.recipe                           = &recipe;
    internalRecipeHandle.basicRecipeHandle.shape_plan_recipe                = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfo                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfoSize              = 0;
    internalRecipeHandle.basicRecipeHandle.recipeAllocator                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeStats.numbSuccessfulLaunch = 0;
    synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle.basicRecipeHandle,
                                                              internalRecipeHandle.deviceAgnosticRecipeHandle);
    EXPECT_EQ(status, synSuccess);

    DynamicInfoProcessorMock dynamicInfoProcessor(internalRecipeHandle);

    QueueComputeQman::s_pTestDynamicInfoProcessor = &dynamicInfoProcessor;

    recipeLaunch(deviceType, dynamicInfoProcessor, internalRecipeHandle, 0, launchNum, {1, 1}, recipeDownloadStatus);

    QueueComputeQman::s_pTestDynamicInfoProcessor = nullptr;

    // Check
    if (recipeDownloadStatus == synSuccess)
    {
        EXPECT_EQ(dynamicInfoProcessor.getExecutionHandle(), launchNum);
        EXPECT_EQ(dynamicInfoProcessor.m_csHandles.size(), launchNum);
    }
    else
    {
        EXPECT_EQ(dynamicInfoProcessor.getExecutionHandle(), 0);
        EXPECT_EQ(dynamicInfoProcessor.m_csHandles.size(), 0);
    }
}

void UTStreamComputeTest::singlePatchableProgramCodeBlobRecipeLaunch(synDeviceType deviceType, unsigned launchNum)
{
    recipe_t       recipe {};
    const uint64_t patchingBlobsBufferSize = 0x10;
    char           patchingBlobsBuffer[patchingBlobsBufferSize] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint64_t*      pPatchingBlobsBuffer = (uint64_t*)&patchingBlobsBuffer[0];
    recipe.patching_blobs_buffer_size   = patchingBlobsBufferSize;
    recipe.patching_blobs_buffer        = pPatchingBlobsBuffer;
    const uint32_t blobNum              = 1;
    blob_t         blobs[blobNum];
    blob_t*        pBlobs = &blobs[0];
    blobs[0].blob_type    = {1, 0, 0, 0x0};
    blobs[0].size         = patchingBlobsBufferSize;
    blobs[0].data         = pPatchingBlobsBuffer;
    recipe.blobs_nr       = blobNum;
    recipe.blobs          = pBlobs;

    uint64_t blobIndices[1];
    blobIndices[0] = 0;

    uint32_t  programsNr = 1;        // The number of programs on the recipe.
    program_t programs[programsNr];  // The recipe's programs.
    programs[0].program_length = 1;
    programs[0].blob_indices   = &blobIndices[0];

    recipe.programs_nr = programsNr;
    recipe.programs    = &programs[0];

    RecipeProcessingUtils::getExecutionJobs(deviceType, recipe.execute_jobs_nr, recipe.execute_jobs);

    uint32_t       program_blobs_nr[] {programs[0].program_length};
    node_program_t node[] {{&program_blobs_nr[0], 0}};
    recipe.node_exe_list = &node[0];
    recipe.node_nr       = 1;

    uint64_t workspace_sizes[3] {0, 0, 0};
    recipe.workspace_nr    = sizeof(workspace_sizes) / sizeof(uint64_t);
    recipe.workspace_sizes = &workspace_sizes[0];

    const uint32_t patchPointsNum = 1;
    patch_point_t  patchPoints[patchPointsNum];

    patchPoints[0].type                                 = patch_point_t::SIMPLE_DW_LOW_MEM_PATCH_POINT;
    patchPoints[0].blob_idx                             = 0;
    patchPoints[0].dw_offset_in_blob                    = 0;
    patchPoints[0].memory_patch_point.effective_address = 0;
    patchPoints[0].memory_patch_point.section_idx       = 0;

    recipe.patch_points_nr = patchPointsNum;
    recipe.patch_points    = &patchPoints[0];

    gc_conf_t recipeConf[] {{deviceType, gc_conf_t::DEVICE_TYPE},
                            {GCFG_TPC_ENGINES_ENABLED_MASK.value(), gc_conf_t::TPC_ENGINE_MASK}};
    recipe.recipe_conf_nr     = sizeof(recipeConf) / sizeof(gc_conf_t);
    recipe.recipe_conf_params = &recipeConf[0];

    recipe.permute_tensors_views_nr = 0;

    recipe.persist_tensors_nr = 1;
    persist_tensor_info_t tensors[1];
    recipe.tensors                      = tensors;
    recipe.tensors[0].name              = "a_very_unique_test_tensor";
    recipe.tensors[0].section_idx       = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    recipe.tensors[0].offset_in_section = 0;
    recipe.tensors[0].size              = 11;
    recipe.tensors[0].elementType       = 8;
    recipe.tensors[0].zp                = 1.5;
    recipe.tensors[0].scale             = 3.2;
    recipe.tensors[0].dimensions        = SYN_MAX_TENSOR_DIM;
    for (size_t idx = 0; idx < SYN_MAX_TENSOR_DIM; idx++)
    {
        recipe.tensors[0].dimensionsSize[idx] = 0;
    }
    recipe.tensors[0].batchSize              = 1;
    recipe.tensors[0].isInput                = true;
    recipe.tensors[0].section_type           = 0;
    recipe.tensors[0].layout                 = "NCHW";
    recipe.tensors[0].multi_views_indices_nr = 0;

    InternalRecipeHandle internalRecipeHandle;
    internalRecipeHandle.basicRecipeHandle.recipe                           = &recipe;
    internalRecipeHandle.basicRecipeHandle.shape_plan_recipe                = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfo                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfoSize              = 0;
    internalRecipeHandle.basicRecipeHandle.recipeAllocator                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeStats.numbSuccessfulLaunch = 0;
    synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle.basicRecipeHandle,
                                                              internalRecipeHandle.deviceAgnosticRecipeHandle);
    EXPECT_EQ(status, synSuccess);

    DynamicInfoProcessorMock dynamicInfoProcessor(internalRecipeHandle);

    QueueComputeQman::s_pTestDynamicInfoProcessor = &dynamicInfoProcessor;

    recipeLaunch(deviceType, dynamicInfoProcessor, internalRecipeHandle, 1, launchNum, {1, 1}, synSuccess);

    QueueComputeQman::s_pTestDynamicInfoProcessor = nullptr;

    // Check
    EXPECT_EQ(dynamicInfoProcessor.getExecutionHandle(), launchNum);
}

uint64_t UTStreamComputeTest::getStreamMasterQueueIdForCompute(synDeviceType deviceType)
{
    uint64_t streamMasterQueueIdForCompute;
    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            streamMasterQueueIdForCompute = gaudi::QmansDefinition::getInstance()->getStreamMasterQueueIdForCompute();
            break;
        }
        default:
        {
            streamMasterQueueIdForCompute = 0;
        }
    }
    return streamMasterQueueIdForCompute;
}

TEST_F(UTStreamComputeTest, gaudiSingleProgramCodeBlobRecipeLaunchSingle)
{
    singleProgramCodeBlobRecipeLaunch(synDeviceGaudi, 1, synSuccess);
}

TEST_F(UTStreamComputeTest, gaudiSingleProgramCodeBlobRecipeLaunchMulti)
{
    singleProgramCodeBlobRecipeLaunch(synDeviceGaudi, 10, synSuccess);
}

TEST_F(UTStreamComputeTest, gaudiSinglePatchableProgramCodePBlobRecipeLaunchSingle)
{
    singlePatchableProgramCodeBlobRecipeLaunch(synDeviceGaudi, 1);
}

void UTStreamComputeTest::checkWcmOrder(synDeviceType deviceType)
{
    recipe_t recipe {};

    const uint64_t executionBlobsBufferSize = 0x10;
    char      executionBlobsBuffer[executionBlobsBufferSize] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint64_t* pExecutionBlobsBuffer    = (uint64_t*)&executionBlobsBuffer[0];
    recipe.execution_blobs_buffer_size = executionBlobsBufferSize;
    recipe.execution_blobs_buffer      = pExecutionBlobsBuffer;
    const uint32_t blobNum             = 1;
    blob_t         blobs[blobNum];
    blob_t*        pBlobs = &blobs[0];
    blobs[0].blob_type    = {0, 1, 0, 0x0};
    blobs[0].size         = executionBlobsBufferSize;
    blobs[0].data         = pExecutionBlobsBuffer;
    recipe.blobs_nr       = blobNum;
    recipe.blobs          = pBlobs;

    uint64_t blobIndices[1];
    blobIndices[0] = 0;

    uint32_t  programsNr = 1;        // The number of programs on the recipe.
    program_t programs[programsNr];  // The recipe's programs.
    programs[0].program_length = 1;
    programs[0].blob_indices   = &blobIndices[0];

    recipe.programs_nr = programsNr;
    recipe.programs    = &programs[0];

    RecipeProcessingUtils::getExecutionJobs(deviceType, recipe.execute_jobs_nr, recipe.execute_jobs);

    uint32_t       program_blobs_nr[] {programs[0].program_length};
    node_program_t node[] {{&program_blobs_nr[0], 0}};
    recipe.node_exe_list = &node[0];
    recipe.node_nr       = 1;

    uint64_t workspace_sizes[3] {0, 0, recipe.execution_blobs_buffer_size};
    recipe.workspace_nr    = sizeof(workspace_sizes) / sizeof(uint64_t);
    recipe.workspace_sizes = &workspace_sizes[0];

    gc_conf_t recipeConf[] {{deviceType, gc_conf_t::DEVICE_TYPE},
                            {GCFG_TPC_ENGINES_ENABLED_MASK.value(), gc_conf_t::TPC_ENGINE_MASK}};
    recipe.recipe_conf_nr     = sizeof(recipeConf) / sizeof(gc_conf_t);
    recipe.recipe_conf_params = &recipeConf[0];

    InternalRecipeHandle internalRecipeHandle;
    internalRecipeHandle.basicRecipeHandle.recipe                           = &recipe;
    internalRecipeHandle.basicRecipeHandle.shape_plan_recipe                = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfo                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfoSize              = 0;
    internalRecipeHandle.basicRecipeHandle.recipeAllocator                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeStats.numbSuccessfulLaunch = 0;
    synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle.basicRecipeHandle,
                                                              internalRecipeHandle.deviceAgnosticRecipeHandle);
    EXPECT_EQ(status, synSuccess);

    DynamicInfoProcessorMock dynamicInfoProcessor(internalRecipeHandle);

    QueueComputeQman::s_pTestDynamicInfoProcessor = &dynamicInfoProcessor;

    // Initialization
    syn::Context context;

    // Construction
    const BasicQueueInfo      basicQueueInfo {{0, 0},
                                         INTERNAL_STREAM_TYPE_COMPUTE,
                                         TRAINING_QUEUE_COMPUTE_0,
                                         0,
                                         std::make_shared<QueueInfo>(0, 0, 0, 0)};
    PhysicalQueuesManagerMock physicalStreamsManager;

    WorkCompletionManagerMock           workCompletionManager;
    SubmitCommandBuffersMock            submitter;
    DevMemoryAllocMock                  devMemoryAlloc;
    DeviceRecipeAddressesGeneratorMock  devRecipeAddress;
    DeviceRecipeDownloaderMock          deviceRecipeDownloader(internalRecipeHandle.deviceAgnosticRecipeHandle,
                                                      {0, 0},
                                                      synSuccess);
    DeviceRecipeDownloaderContainerMock deviceRecipeDownloaderContainer(&deviceRecipeDownloader);
    QueueMock                                    streamCopy;
    std::unique_ptr<StreamMasterHelperInterface> streamMasterHelper = std::make_unique<StreamMasterHelperMock>();

    WcmCsHandleQueue csHandles;
    {
        QueueComputeQman stream {basicQueueInfo,
                                 0,
                                 0,
                                 false,
                                 deviceType,
                                 &physicalStreamsManager,
                                 workCompletionManager,
                                 submitter,
                                 devMemoryAlloc,
                                 devRecipeAddress,
                                 deviceRecipeDownloaderContainer,
                                 streamCopy,
                                 std::move(streamMasterHelper)};

        stream.initAllocators();

        // Launch
        std::array<synLaunchTensorInfoExt, 0x10> launchTensorsInfo;

        const uint64_t tensorDeviceAddr     = 0x10;
        launchTensorsInfo[0].tensorName     = nullptr;
        launchTensorsInfo[0].pTensorAddress = (uint64_t)&tensorDeviceAddr;
        launchTensorsInfo[0].tensorType     = HOST_TO_DEVICE_TENSOR;
        memset(launchTensorsInfo[0].tensorSize, 0, sizeof(launchTensorsInfo[0].tensorSize));
        launchTensorsInfo[0].tensorId = 0;

        uint64_t                workspaceAddress = 0;
        uint32_t                flags            = 0;
        DeviceMapperMock        deviceMapper;
        EventWithMappedTensorDB eventList;

        status = stream.launch(launchTensorsInfo.data(),
                               1,
                               workspaceAddress,
                               &internalRecipeHandle,
                               0,
                               flags,
                               eventList,
                               API_ID);
        EXPECT_EQ(status, synSuccess);

        status = stream.launch(launchTensorsInfo.data(),
                               1,
                               workspaceAddress,
                               &internalRecipeHandle,
                               0,
                               flags,
                               eventList,
                               API_ID);
        EXPECT_EQ(status, synSuccess);

        // Store CS handles since they are about to be removed on call to WorkCompletionManagerMock::notifyCsCompleted
        csHandles = workCompletionManager.m_csHandles;

        // Notify completion in reverse order to the insertion order
        workCompletionManager.notifyCsCompleted(workCompletionManager.m_csHandles[1], false);
        workCompletionManager.notifyCsCompleted(workCompletionManager.m_csHandles[0], false);
    }
    QueueComputeQman::s_pTestDynamicInfoProcessor = nullptr;

    EXPECT_EQ(dynamicInfoProcessor.m_csHandles.size(), csHandles.size());
    EXPECT_EQ(dynamicInfoProcessor.m_csHandles[0].first, csHandles[1]);
    EXPECT_EQ(dynamicInfoProcessor.m_csHandles[1].first, csHandles[0]);
}

void UTStreamComputeTest::checkNotifyCsCompleted(synDeviceType deviceType, bool csFailed)
{
    recipe_t recipe {};

    const uint64_t executionBlobsBufferSize = 0x10;
    char      executionBlobsBuffer[executionBlobsBufferSize] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint64_t* pExecutionBlobsBuffer    = (uint64_t*)&executionBlobsBuffer[0];
    recipe.execution_blobs_buffer_size = executionBlobsBufferSize;
    recipe.execution_blobs_buffer      = pExecutionBlobsBuffer;
    const uint32_t blobNum             = 1;
    blob_t         blobs[blobNum];
    blob_t*        pBlobs = &blobs[0];
    blobs[0].blob_type    = {0, 1, 0, 0x0};
    blobs[0].size         = executionBlobsBufferSize;
    blobs[0].data         = pExecutionBlobsBuffer;
    recipe.blobs_nr       = blobNum;
    recipe.blobs          = pBlobs;

    uint64_t blobIndices[1];
    blobIndices[0] = 0;

    uint32_t  programsNr = 1;        // The number of programs on the recipe.
    program_t programs[programsNr];  // The recipe's programs.
    programs[0].program_length = 1;
    programs[0].blob_indices   = &blobIndices[0];

    recipe.programs_nr = programsNr;
    recipe.programs    = &programs[0];

    RecipeProcessingUtils::getExecutionJobs(deviceType, recipe.execute_jobs_nr, recipe.execute_jobs);

    uint32_t       program_blobs_nr[] {programs[0].program_length};
    node_program_t node[] {{&program_blobs_nr[0], 0}};
    recipe.node_exe_list = &node[0];
    recipe.node_nr       = 1;

    uint64_t workspace_sizes[3] {0, 0, recipe.execution_blobs_buffer_size};
    recipe.workspace_nr    = sizeof(workspace_sizes) / sizeof(uint64_t);
    recipe.workspace_sizes = &workspace_sizes[0];

    gc_conf_t recipeConf[] {{deviceType, gc_conf_t::DEVICE_TYPE},
                            {GCFG_TPC_ENGINES_ENABLED_MASK.value(), gc_conf_t::TPC_ENGINE_MASK}};
    recipe.recipe_conf_nr     = sizeof(recipeConf) / sizeof(gc_conf_t);
    recipe.recipe_conf_params = &recipeConf[0];

    InternalRecipeHandle internalRecipeHandle;
    internalRecipeHandle.basicRecipeHandle.recipe                           = &recipe;
    internalRecipeHandle.basicRecipeHandle.shape_plan_recipe                = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfo                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeDebugInfoSize              = 0;
    internalRecipeHandle.basicRecipeHandle.recipeAllocator                  = nullptr;
    internalRecipeHandle.basicRecipeHandle.recipeStats.numbSuccessfulLaunch = 0;
    synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle.basicRecipeHandle,
                                                              internalRecipeHandle.deviceAgnosticRecipeHandle);
    EXPECT_EQ(status, synSuccess);

    DynamicInfoProcessorMock dynamicInfoProcessor(internalRecipeHandle);

    QueueComputeQman::s_pTestDynamicInfoProcessor = &dynamicInfoProcessor;

    // Initialization
    syn::Context context;

    // Construction
    const BasicQueueInfo                         basicQueueInfo {{0, 0},
                                         INTERNAL_STREAM_TYPE_COMPUTE,
                                         TRAINING_QUEUE_COMPUTE_0,
                                         0,
                                         std::make_shared<QueueInfo>(0, 0, 0, 0)};
    PhysicalQueuesManagerMock                    physicalStreamsManager;
    WorkCompletionManagerMock           workCompletionManager;
    SubmitCommandBuffersMock            submitter;
    DevMemoryAllocMock                  devMemoryAlloc;
    DeviceRecipeAddressesGeneratorMock  devRecipeAddress;
    DeviceRecipeDownloaderMock          deviceRecipeDownloader(internalRecipeHandle.deviceAgnosticRecipeHandle,
                                                      {0, 0},
                                                      synSuccess);
    DeviceRecipeDownloaderContainerMock deviceRecipeDownloaderContainer(&deviceRecipeDownloader);
    QueueMock                                    streamCopy;
    std::unique_ptr<StreamMasterHelperInterface> streamMasterHelper = std::make_unique<StreamMasterHelperMock>();

    WcmCsHandleQueue csHandles;
    {
        QueueComputeQman stream {basicQueueInfo,
                                 0,
                                 0,
                                 false,
                                 deviceType,
                                 &physicalStreamsManager,
                                 workCompletionManager,
                                 submitter,
                                 devMemoryAlloc,
                                 devRecipeAddress,
                                 deviceRecipeDownloaderContainer,
                                 streamCopy,
                                 std::move(streamMasterHelper)};

        stream.initAllocators();

        // Launch
        std::array<synLaunchTensorInfoExt, 0x10> launchTensorsInfo;

        const uint64_t tensorDeviceAddr     = 0x10;
        launchTensorsInfo[0].tensorName     = nullptr;
        launchTensorsInfo[0].pTensorAddress = (uint64_t)&tensorDeviceAddr;
        launchTensorsInfo[0].tensorType     = HOST_TO_DEVICE_TENSOR;
        memset(launchTensorsInfo[0].tensorSize, 0, sizeof(launchTensorsInfo[0].tensorSize));
        launchTensorsInfo[0].tensorId = 0;

        uint64_t                workspaceAddress = 0;
        uint32_t                flags            = 0;
        DeviceMapperMock        deviceMapper;
        EventWithMappedTensorDB eventList;

        status = stream.launch(launchTensorsInfo.data(),
                               1,
                               workspaceAddress,
                               &internalRecipeHandle,
                               0,
                               flags,
                               eventList,
                               API_ID);
        EXPECT_EQ(status, synSuccess);

        // Store CS handles since they are about to be removed on call to WorkCompletionManagerMock::notifyCsCompleted
        csHandles = workCompletionManager.m_csHandles;

        workCompletionManager.notifyCsCompleted(workCompletionManager.m_csHandles[0], csFailed);
    }
    QueueComputeQman::s_pTestDynamicInfoProcessor = nullptr;

    EXPECT_EQ(dynamicInfoProcessor.m_csHandles.size(), csHandles.size());
    EXPECT_EQ(dynamicInfoProcessor.m_csHandles[0].first, csHandles[0]);
    EXPECT_EQ(dynamicInfoProcessor.m_csHandles[0].second, csFailed);
    EXPECT_EQ(devRecipeAddress.m_recipeIdNotified, devRecipeAddress.m_recipeIdGenerated);
}

TEST_F(UTStreamComputeTest, gaudiCheckDifferentWcmOrder)
{
    checkWcmOrder(synDeviceGaudi);
}

TEST_F(UTStreamComputeTest, gaudiCheckNotifyCsCompletedFailed)
{
    checkNotifyCsCompleted(synDeviceGaudi, true);
}

TEST_F(UTStreamComputeTest, gaudiSingleProgramCodeBlobRecipeLaunchSingleRecipeDownloadFailure)
{
    singleProgramCodeBlobRecipeLaunch(synDeviceGaudi, 1, synFail);
}