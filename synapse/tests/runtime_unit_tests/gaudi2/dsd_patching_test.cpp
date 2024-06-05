#include "common/dsd_recipe.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>

class UTGaudi2RtDynamicShapesTest : public DsdRecipe
{
};

TEST_F(UTGaudi2RtDynamicShapesTest, dynamicPatching_gaudi2)
{
    initDynamicPatchingTest();

    blobs[NUM_BLOBS - 1].blob_type_all         = blob_t::DYNAMIC;
    blobs[NUM_BLOBS - 1].blob_type.dynamic_exe = 1;

    recipe.dynamic_blobs_buffer      = blobData[NUM_BLOBS - 1];
    recipe.dynamic_blobs_buffer_size = BLOB_DATA_SIZE;

    recipe.patching_blobs_buffer_size = (NUM_BLOBS - 1) * BLOB_DATA_SIZE;

    addrPP[NUM_ADDR_PP - 1].blob_idx = NUM_BLOBS - 1;

    ASSERT_EQ(DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi2,
                                                               recipeInfo,
                                                               deviceAgnosticRecipeInfo.m_recipeStaticInfoScal),
              synSuccess);

    const RecipeAddrPatcher* pRecipeAddrPatcher = nullptr;
    DynamicRecipe dynamicRecipe(recipeInfo,
                                deviceAgnosticRecipeInfo,
                                &deviceAgnosticRecipeInfo.m_recipeStaticInfoScal.recipeDsdPpInfo.getDsdDCPatchingInfo(),
                                pRecipeAddrPatcher);

    uint32_t NUM_USER_TENSORS = ARRAY_SIZE(launchTensors);  // Persistent + shape tensors

    synLaunchTensorInfoExt launchTensorInfo[NUM_USER_TENSORS];
    for (uint64_t i = 0; i < NUM_USER_TENSORS; i++)
    {
        launchTensorInfo[i].tensorName     = launchTensors[i].tensorName;
        launchTensorInfo[i].pTensorAddress = launchTensors[i].pTensorAddress;
        launchTensorInfo[i].tensorType     = launchTensors[i].tensorType;
        launchTensorInfo[i].tensorId       = launchTensors[i].tensorId;
        memcpy(launchTensorInfo[i].tensorSize, launchTensors[i].tensorSize, sizeof(launchTensorInfo[i].tensorSize));
    }

    std::unique_ptr<SingleDataChunkHostBuffer[]> dataChunksHostBuffers(new SingleDataChunkHostBuffer[2]);
    for (uint16_t i = 0; i < 2; i++)
    {
        dataChunksHostBuffers[i].reset(new uint8_t[TOTAL_DATA_SIZE]);
    }
    std::vector<uint64_t> dataChunksHostAddresses;
    dataChunksHostAddresses.resize(2);
    for (uint16_t i = 0; i < 2; i++)
    {
        dataChunksHostAddresses[i] = (uint64_t)dataChunksHostBuffers[i].get();
    }

    bool res = dynamicRecipe.runSifOnAllNodes(launchTensorInfo,
                                              NUM_USER_TENSORS,
                                              tensorIdx2userIdx,
                                              0 /* programDataHostAddress - NA*/);
    ASSERT_EQ(true, res) << "Failed DSD SIF";

    res = dynamicRecipe.runSmfOnAllNodes(dataChunksHostAddresses);
    ASSERT_EQ(true, res) << "Failed DSD SMF";
}

class UTGaudi3RtDynamicShapesTest : public DsdRecipe
{
};

TEST_F(UTGaudi3RtDynamicShapesTest, dynamicPatching_gaudi3)
{
    initDynamicPatchingTest();

    blobs[NUM_BLOBS - 1].blob_type_all         = blob_t::DYNAMIC;
    blobs[NUM_BLOBS - 1].blob_type.dynamic_exe = 1;

    recipe.dynamic_blobs_buffer      = blobData[NUM_BLOBS - 1];
    recipe.dynamic_blobs_buffer_size = BLOB_DATA_SIZE;

    recipe.patching_blobs_buffer_size = (NUM_BLOBS - 1) * BLOB_DATA_SIZE;

    addrPP[NUM_ADDR_PP - 1].blob_idx = NUM_BLOBS - 1;

    ASSERT_EQ(DeviceAgnosticRecipeStaticProcessorScal::process(synDeviceGaudi3,
                                                               recipeInfo,
                                                               deviceAgnosticRecipeInfo.m_recipeStaticInfoScal),
              synSuccess);

    DynamicRecipe dynamicRecipe(recipeInfo,
                                deviceAgnosticRecipeInfo,
                                &deviceAgnosticRecipeInfo.m_recipeStaticInfoScal.recipeDsdPpInfo.getDsdDCPatchingInfo(),
                                &deviceAgnosticRecipeInfo.m_recipeStaticInfoScal.recipeAddrPatcher);

    uint32_t NUM_USER_TENSORS = ARRAY_SIZE(launchTensors);  // Persistent + shape tensors

    synLaunchTensorInfoExt launchTensorInfo[NUM_USER_TENSORS];
    for (uint64_t i = 0; i < NUM_USER_TENSORS; i++)
    {
        launchTensorInfo[i].tensorName     = launchTensors[i].tensorName;
        launchTensorInfo[i].pTensorAddress = launchTensors[i].pTensorAddress;
        launchTensorInfo[i].tensorType     = launchTensors[i].tensorType;
        launchTensorInfo[i].tensorId       = launchTensors[i].tensorId;
        memcpy(launchTensorInfo[i].tensorSize, launchTensors[i].tensorSize, sizeof(launchTensorInfo[i].tensorSize));
    }

    std::unique_ptr<SingleDataChunkHostBuffer[]> dataChunksHostBuffers(new SingleDataChunkHostBuffer[2]);
    for (uint16_t i = 0; i < 2; i++)
    {
        dataChunksHostBuffers[i].reset(new uint8_t[TOTAL_DATA_SIZE]);
    }
    std::vector<uint64_t> dataChunksHostAddresses;
    dataChunksHostAddresses.resize(2);
    for (uint16_t i = 0; i < 2; i++)
    {
        dataChunksHostAddresses[i] = (uint64_t)dataChunksHostBuffers[i].get();
    }

    bool res = dynamicRecipe.runSifOnAllNodes(launchTensorInfo,
                                              NUM_USER_TENSORS,
                                              tensorIdx2userIdx,
                                              0 /* programDataHostAddress - NA*/);
    ASSERT_EQ(true, res) << "Failed DSD SIF";

    res = dynamicRecipe.runSmfOnAllNodes(dataChunksHostAddresses);
    ASSERT_EQ(true, res) << "Failed DSD SMF";
}