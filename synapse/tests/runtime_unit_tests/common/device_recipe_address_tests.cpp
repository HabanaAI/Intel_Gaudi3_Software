#include "dev_memory_alloc_mock.hpp"
#include <gtest/gtest.h>
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/qman/common/device_recipe_addresses_generator.hpp"
#include "runtime/qman/common/recipe_cache_manager.hpp"

class UTDeviceRecipeAddressTest : public ::testing::Test
{
public:
    // Todo add DeviceRecipeAddressesGenerator tests
};

TEST(UTDeviceRecipeAddressTest, basic_flow_cache)
{
    const synDeviceType deviceType = synDeviceGaudi;

    recipe_t recipe {};
    uint64_t workspace_sizes[3] {0, 0x1000, 0};
    recipe.workspace_nr                = sizeof(workspace_sizes) / sizeof(uint64_t);
    recipe.workspace_sizes             = &workspace_sizes[0];
    recipe.execution_blobs_buffer_size = 0x2000;
    DevMemoryAllocMock             devMemoryAlloc;
    DeviceRecipeAddressesGenerator devAddr(deviceType, devMemoryAlloc);
    synStatus status = devAddr.allocate();
    ASSERT_EQ(status, synSuccess);

    const uint64_t        workspaceAddress = 0x3000;
    const uint64_t        refCount         = 1;

    uint64_t              recipeId;
    uint64_t              programCodeDeviceAddress;
    uint64_t              programDataDeviceAddress;
    bool                  programCodeInCache;
    bool                  programDataInCache;
    uint64_t              programCodeHandle;
    uint64_t              programDataHandle;
    std::vector<uint64_t> executionBlocksAddresses;

    InternalRecipeHandle internalRecipeHandle {};
    internalRecipeHandle.basicRecipeHandle.recipe = &recipe;
    internalRecipeHandle.recipeSeqNum             = 0x12;

    status = devAddr.generateDeviceRecipeAddresses(internalRecipeHandle,
                                                   workspaceAddress,
                                                   0 /* launchSeqId */,
                                                   refCount,
                                                   recipeId,
                                                   programCodeDeviceAddress,
                                                   programDataDeviceAddress,
                                                   programCodeInCache,
                                                   programDataInCache,
                                                   programCodeHandle,
                                                   programDataHandle,
                                                   executionBlocksAddresses);
    ASSERT_EQ(status, synSuccess);

    ASSERT_EQ(programCodeInCache, true);
    ASSERT_EQ(programDataInCache, true);

    devAddr.notifyDeviceRecipeAddressesAreNotUsed(recipeId,
                                                  refCount,
                                                  RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID);

    status = devAddr.release();
    ASSERT_EQ(status, synSuccess);
}