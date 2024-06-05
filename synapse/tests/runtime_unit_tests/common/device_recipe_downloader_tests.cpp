#include <gtest/gtest.h>
#include "device_mock.hpp"
#include "runtime/qman/common/device_recipe_downloader.hpp"
#include "device_downloader_mock.hpp"
#include "device_agnostic_recipe_processor.hpp"
#include "device_agnostic_recipe_info.hpp"
#include "device_mapper_mock.hpp"
#include "recipe.h"
#include "basic_recipe_info.hpp"
#include "habana_global_conf.h"
#include "recipe_processing_utils.hpp"

class UTDeviceRecipeDownloaderTest : public ::testing::Test
{
public:
    // Todo add DeviceRecipeDownloader tests
};

TEST_F(UTDeviceRecipeDownloaderTest, basic_flow)
{
    const synDeviceType   deviceType = synDeviceGaudi;
    const QmanDefinitionInterface&            rQmansDefinition     = *getDeviceQmansDefinition(deviceType);
    const generic::CommandBufferPktGenerator& rCmdBuffPktGenerator = *getDeviceCommandBufferPktGenerator(deviceType);
    DeviceDownloaderMock  deviceDownloader;
    DeviceMapperMock                          deviceMapper;

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

    basicRecipeInfo          basicRecipeInfo {&recipe, nullptr, nullptr, 0, nullptr};
    DeviceAgnosticRecipeInfo deviceAgnosticRecipeHandle {};
    synStatus status = DeviceAgnosticRecipeProcessor::process(basicRecipeInfo, deviceAgnosticRecipeHandle);
    EXPECT_EQ(status, synSuccess);
    DeviceRecipeDownloader deviceRecipeDownloader(deviceType,
                                                  rQmansDefinition,
                                                  rCmdBuffPktGenerator,
                                                  deviceDownloader,
                                                  deviceMapper,
                                                  deviceAgnosticRecipeHandle,
                                                  basicRecipeInfo,
                                                  0);

    // Todo call downloadExecutionBufferCache method

    // Todo call downloadProgramDataBufferCache method

    // Todo call downloadProgramDataBufferWorkspace method

    // Todo call downloadExecutionBufferWorkspace method

    deviceRecipeDownloader.notifyRecipeDestroy();
}