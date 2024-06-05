#include "synapse_test.hpp"
#include "runtime/qman/common/recipe_cache_manager.hpp"
#include "habana_global_conf_runtime.h"
#include "math.h"

class UTSynGaudiRecipeCaching : public ::testing::Test
{
public:
    void SetUp() override
    {
        // internal cache stats requires syn_singleton
        synInitialize();
    }
    void TearDown() override { synDestroy(); }

    void checkReferenceCount(uint64_t refCount, bool faultyRcNotification);
};

void UTSynGaudiRecipeCaching::checkReferenceCount(uint64_t refCount, bool faultyRcNotification)
{
    uint64_t recipeCacheSize = 250 * KB;  // in KB

    const uint64_t recipeId = 0x555;

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());  // deviceId, base address;

    uint64_t recipeSectionsSizes[RecipeCacheManager::ST_AMOUNT] = {0x1000, 0x2000};
    uint64_t sectionsHandles[RecipeCacheManager::ST_AMOUNT];

    std::vector<uint64_t> prgCodeBlocksVector;
    std::vector<uint64_t> prgDataBlocksVector;

    RecipeCacheManager::BlockPtrVector sectionsBlockLists[RecipeCacheManager::ST_AMOUNT] = {&prgCodeBlocksVector,
                                                                                            &prgDataBlocksVector};

    const uint64_t execBlobsSubTypeAllocId       = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    const uint64_t prgDataSingularSubTypeAllocId = RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;

    RecipeCacheManager::AllSubTypesAllocationRequests allocationRequests;
    RecipeCacheManager::addSingleSubTypeAllocationRequestL(
        allocationRequests, RecipeCacheManager::ST_EXECUTION_BLOBS,
        execBlobsSubTypeAllocId, recipeSectionsSizes[RecipeCacheManager::ST_EXECUTION_BLOBS]);
    RecipeCacheManager::addSingleSubTypeAllocationRequestL(
        allocationRequests, RecipeCacheManager::ST_PROGRAM_DATA,
        prgDataSingularSubTypeAllocId, recipeSectionsSizes[RecipeCacheManager::ST_PROGRAM_DATA]);

    RecipeCacheManager::RecipeCacheAllocationId execBlobsAllocId =
        std::make_pair(recipeId, execBlobsSubTypeAllocId);
    RecipeCacheManager::RecipeCacheAllocationId prgDataSingularAllocId  =
        std::make_pair(recipeId, prgDataSingularSubTypeAllocId);


    bool status =
        cache.acquireBlocks(recipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
    ASSERT_TRUE(status) << "acquireBlocksForAllSections failed";

    if (faultyRcNotification)
    {
        refCount++;
    }

    status = cache.setRecipeNotInUse(prgDataSingularAllocId, refCount);
    ASSERT_TRUE(status) << "setRecipeNotInUse PROGRAM_DATA failed";

    status = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
    ASSERT_TRUE(status) << "setRecipeNotInUse EXECUTION_BLOBS failed";
}

static void updateBlocksAmount(uint64_t&                                 totalBlocksAmount,
                               RecipeCacheManager::BlockAllocatorResult& requiredStatus,
                               uint64_t                                  allocatedMemoryInMB)
{
    uint64_t blockSize                = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t blockAmoutInCurrentAlloc = std::ceil((KB * KB * allocatedMemoryInMB) / (float)blockSize);
    if (totalBlocksAmount >= blockAmoutInCurrentAlloc)
    {
        totalBlocksAmount -= blockAmoutInCurrentAlloc;
        requiredStatus = RecipeCacheManager::BAR_ALLOCATED;
    }
    else
    {
        requiredStatus = RecipeCacheManager::BAR_FAILURE;
    }
}

TEST_F(UTSynGaudiRecipeCaching, check_reference_count1)
{
    checkReferenceCount(1, false);
}

TEST_F(UTSynGaudiRecipeCaching, check_reference_count2)
{
    checkReferenceCount(2, false);
}

TEST_F(UTSynGaudiRecipeCaching, check_reference_count2_faulty_notification)
{
    checkReferenceCount(2, true);
}

TEST_F(UTSynGaudiRecipeCaching, basic_LRU)
{
    const uint64_t refCount    = 1;
    uint64_t counter           = 0;
    uint64_t blockSize         = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t recipeCacheSize   = 250 * KB;  // in KB
    uint64_t totalBlocksAmount = std::ceil((recipeCacheSize * KB) / (float)blockSize);

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());  // deviceId, base address;
    std::map<uint64_t, uint64_t> recipesIdToSize;             // std::map<deviceId, size>

    for (size_t i = 0; i <= 200; i++)
    {
        recipesIdToSize[i] = i * KB * KB;  // in MB
    }

    std::vector<uint64_t> blockAddresses;

    RecipeCacheManager::BlockAllocatorResult failurestatus  = RecipeCacheManager::BAR_FAILURE;
    RecipeCacheManager::BlockAllocatorResult allocStatus    = failurestatus;
    RecipeCacheManager::BlockAllocatorResult requiredStatus = failurestatus;

    bool     status         = false;
    uint64_t recipeId       = 200;
    uint64_t subTypeAllocId = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;

    RecipeCacheManager::SubType                 recipeSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::RecipeCacheAllocationId allocId       = std::make_pair(recipeId, subTypeAllocId);

    updateBlocksAmount(totalBlocksAmount, requiredStatus, 200);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[200], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 200 MB";

    updateBlocksAmount(totalBlocksAmount, requiredStatus, 50);
    allocId.first = 50;
    allocStatus   = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[50], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 50 MB";

    // this should fail as cache is full
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 30);
    allocId.first = 30;
    allocStatus   = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[30], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "should have failed to cache 30 MB";

    uint64_t blockAmoutInCurrentAlloc = std::ceil(recipesIdToSize[50] / (float)blockSize);
    totalBlocksAmount += blockAmoutInCurrentAlloc;
    // set recipe id "50" to notInUse
    allocId.first = 50;
    status        = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 50 MB";

    // now we should succeed
    allocId.first = 30;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 30);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[30], blockAddresses, refCount, counter);
    ASSERT_NE(failurestatus, allocStatus) << "Failed to cache 30 MB";

    // free space should be 20

    // allocate an existing id ("30"), only refcount should change
    allocId.first = 30;
    allocStatus   = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[30], blockAddresses, refCount, counter);
    ASSERT_TRUE(allocStatus) << "Failed to cache 30 MB again";

    // this should fail as cache is full
    allocId.first = 40;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 40);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[40], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Should have failed to cache 40 MB";

    // should succeed
    allocId.first = 20;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 20);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[20], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 20 MB";

    // set (non existing anymore) recipe id "50" to notInUse
    allocId.first = 50;
    status        = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_FALSE(status) << "setRecipeNotInUse for 50 MB (already deleted) should have failed";

    // set  (non existent) recipe id "40" to notInUse
    allocId.first = 40;
    status        = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_FALSE(status) << "Should have failed setRecipeNotInUse for 40 MB";

    // free space should be 0

    // set  recipe id "20" to notInUse
    blockAmoutInCurrentAlloc = std::ceil(recipesIdToSize[20] / (float)blockSize);
    totalBlocksAmount += blockAmoutInCurrentAlloc;
    allocId.first = 20;
    status        = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 20 MB";

    // now this should succeed as we cleared some space
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 18);
    allocId.first = 18;
    allocStatus   = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[18], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 18 MB ";

    // 2 should be free

    // set recipe id "30" to notInUse, but since refcount("30") should be 2, no space should be release
    allocId.first = 30;
    status        = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 30 MB";

    // now this should fail since we only dec the refcount
    allocId.first = 3;
    allocStatus   = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[3], blockAddresses, refCount, counter);
    ASSERT_EQ(RecipeCacheManager::BAR_FAILURE, allocStatus) << "should have failed to cache 3 MB ";

    // set recipe id "30" to notInUse again, now it should move it to notInUse
    allocId.first            = 30;
    blockAmoutInCurrentAlloc = std::ceil(recipesIdToSize[30] / (float)blockSize);
    totalBlocksAmount += blockAmoutInCurrentAlloc;
    status = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 30 MB";

    // now free space should be  29

    // now this should succeed
    allocId.first = 29;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 29);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[29], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "failed to cache 29 MB ";
}

TEST_F(UTSynGaudiRecipeCaching, check_LRU_policy)
{
    // contiguous allocation is for program data only
    const uint64_t refCount    = 1;
    uint64_t counter           = 0;
    uint64_t blockSize         = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t recipeCacheSize   = 250 * KB;  // in KB
    uint64_t totalBlocksAmount = std::ceil((recipeCacheSize * KB) / (float)blockSize);

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());
    std::map<uint64_t, uint64_t> recipesIdToSize;

    for (size_t i = 0; i <= 200; i++)
    {
        recipesIdToSize[i] = i * KB * KB;  // in MB
    }

    std::vector<uint64_t> blockAddresses;

    RecipeCacheManager::BlockAllocatorResult failurestatus  = RecipeCacheManager::BAR_FAILURE;
    RecipeCacheManager::BlockAllocatorResult allocStatus    = failurestatus;
    RecipeCacheManager::BlockAllocatorResult requiredStatus = failurestatus;

    bool     status         = false;
    uint64_t recipeId       = 50;
    uint64_t subTypeAllocId = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;

    RecipeCacheManager::SubType                 recipeSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::RecipeCacheAllocationId allocId       = std::make_pair(recipeId, subTypeAllocId);

    updateBlocksAmount(totalBlocksAmount, requiredStatus, 50);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[50], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 50 MB";

    allocId.first = 60;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 60);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[60], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 60 MB";

    allocId.first = 90;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 90);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[90], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 90 MB";

    allocId.first = 30;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 30);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[30], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 30 MB";

    // now only 20M free in cache

    allocId.first = 50;
    // promote id 50 to be with the highest priority
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[50], blockAddresses, refCount, counter);
    ASSERT_EQ(RecipeCacheManager::BAR_ALREADY_IN_CACHE, allocStatus) << "Failed to cache 50 MB";

    // should fail... no much space left, all recipes in use
    allocId.first = 80;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 80);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[80], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 80 MB";

    allocId.first                     = 60;
    uint64_t blockAmoutInCurrentAlloc = std::ceil(recipesIdToSize[60] / (float)blockSize);
    totalBlocksAmount += blockAmoutInCurrentAlloc;
    status = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 60 MB";

    allocId.first            = 90;
    blockAmoutInCurrentAlloc = std::ceil(recipesIdToSize[90] / (float)blockSize);
    totalBlocksAmount += blockAmoutInCurrentAlloc;
    status = cache.setRecipeNotInUse(allocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 90 MB";

    // success, now 60 should be out and the cache is full
    allocId.first = 80;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 80);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[80], blockAddresses, refCount, counter);
    ASSERT_NE(failurestatus, allocStatus) << "Failed to cache 80 MB";

    // enter 60 again - should allocate again - BAR_ALLOCATED
    allocId.first = 60;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 60);
    allocStatus = cache._allocateMemoryForSingleSubType(
        allocId.first, recipeSubType, allocId.second, recipesIdToSize[60], blockAddresses, refCount, counter);
    ASSERT_NE(failurestatus, allocStatus) << "Failed to cache 60 MB";
}

TEST_F(UTSynGaudiRecipeCaching, test_contiguous_allocation)
{
    const uint64_t refCount    = 1;
    uint64_t counter           = 0;
    uint64_t blockSize         = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t recipeCacheSize   = 250 * KB;  // in KB
    uint64_t totalBlocksAmount = std::ceil((recipeCacheSize * KB) / (float)blockSize);

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());  // deviceId, base address;
    std::map<uint64_t, uint64_t> recipesIdToSize;                              // std::map<deviceId, size>

    for (size_t i = 0; i <= 200; i++)
    {
        recipesIdToSize[i] = i * KB * KB;  // in MB
    }

    std::vector<uint64_t> blockAddresses;

    RecipeCacheManager::BlockAllocatorResult failurestatus  = RecipeCacheManager::BAR_FAILURE;
    RecipeCacheManager::BlockAllocatorResult allocStatus    = failurestatus;
    RecipeCacheManager::BlockAllocatorResult requiredStatus = failurestatus;

    bool     status                  = false;
    uint64_t execBlobsRecipeId       = 60;
    uint64_t prgDataRecipeId         = 30;
    uint64_t execBlobsSubTypeAllocId = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    uint64_t prgDataSubTypeAllocId   = RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;

    RecipeCacheManager::SubType execBlobsSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::SubType prgDataSubType   = RecipeCacheManager::ST_PROGRAM_DATA;

    RecipeCacheManager::RecipeCacheAllocationId execBlobsAllocId =
        std::make_pair(execBlobsRecipeId, execBlobsSubTypeAllocId);
    RecipeCacheManager::RecipeCacheAllocationId prgDataAllocId   =
        std::make_pair(prgDataRecipeId, prgDataSubTypeAllocId);

    updateBlocksAmount(totalBlocksAmount, requiredStatus, 60);
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            execBlobsAllocId.first, execBlobsSubType, execBlobsAllocId.second,
            recipesIdToSize[60], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 60 MB";

    updateBlocksAmount(totalBlocksAmount, requiredStatus, 30);
    allocStatus = cache._allocateMemoryForSingleSubType(
        prgDataAllocId.first, prgDataSubType, prgDataAllocId.second,
        recipesIdToSize[30], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 30 MB";

    execBlobsAllocId.first = 20;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 20);
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            execBlobsAllocId.first, execBlobsSubType, execBlobsAllocId.second,
            recipesIdToSize[20], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 20 MB";

    prgDataAllocId.first = 5;
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 5);
    allocStatus = cache._allocateMemoryForSingleSubType(
        prgDataAllocId.first, prgDataSubType, prgDataAllocId.second,
        recipesIdToSize[5], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 5 MB";

    execBlobsAllocId.first = 135;
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            execBlobsAllocId.first, execBlobsSubType, execBlobsAllocId.second,
            recipesIdToSize[135], blockAddresses, refCount, counter);
    ASSERT_EQ(requiredStatus, allocStatus) << "Failed to cache 135 MB";

    // cache is full

    execBlobsAllocId.first = 60;
    status                 = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for sub-type ID 60";

    execBlobsAllocId.first = 20;
    status                 = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for sub-type ID 20";

    // should fail... no much contiguous space left
    prgDataAllocId.first = 80;
    allocStatus          = cache._allocateMemoryForSingleSubType(
        prgDataAllocId.first, prgDataSubType, prgDataAllocId.second,
        recipesIdToSize[80], blockAddresses, refCount, counter);
    ASSERT_EQ(RecipeCacheManager::BAR_FAILURE, allocStatus) << "Failed to cache 80 MB";

    prgDataAllocId.first = 30;
    status               = cache.setRecipeNotInUse(prgDataAllocId, refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for 30 MB";

    // should success
    prgDataAllocId.first = 80;
    allocStatus          = cache._allocateMemoryForSingleSubType(
        prgDataAllocId.first, prgDataSubType, prgDataAllocId.second,
        recipesIdToSize[80], blockAddresses, refCount, counter);
    ASSERT_EQ(RecipeCacheManager::BAR_ALLOCATED, allocStatus) << "Failed to cache 80 MB";
}

TEST_F(UTSynGaudiRecipeCaching, unique_ids_allocations)
{
    const uint64_t refCount    = 1;
    uint64_t handle            = 0;
    uint64_t blockSize         = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t recipeCacheSize   = 250 * KB;  // in KB
    uint64_t totalBlocksAmount = std::ceil((recipeCacheSize * KB) / (float)blockSize);

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());

    // std::map<recipeId, size>
    std::map<uint64_t, uint64_t> recipesIdToSize;
    for (size_t i = 0; i <= 200; i++)
    {
        recipesIdToSize[i] = i * KB * KB;  // in MB
    }

    std::vector<uint64_t> blockAddresses;

    RecipeCacheManager::BlockAllocatorResult failurestatus  = RecipeCacheManager::BAR_FAILURE;
    RecipeCacheManager::BlockAllocatorResult allocStatus    = failurestatus;
    RecipeCacheManager::BlockAllocatorResult requiredStatus = failurestatus;

    bool     status                        = false;
    uint64_t execBlobsRecipeId             = 30;
    uint64_t prgDataRecipeId               = 30;
    uint64_t execBlobsSubTypeAllocId       = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    uint64_t prgDataSingularSubTypeAllocId = RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;

    const uint64_t prgDataUniqueSubTypeAllocId = 0;

    RecipeCacheManager::SubType execBlobsSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::SubType prgDataSubType   = RecipeCacheManager::ST_PROGRAM_DATA;

    RecipeCacheManager::RecipeCacheAllocationId execBlobsAllocId =
        std::make_pair(execBlobsRecipeId, execBlobsSubTypeAllocId);
    RecipeCacheManager::RecipeCacheAllocationId prgDataSingularAllocId  =
        std::make_pair(prgDataRecipeId, prgDataSingularSubTypeAllocId);
    RecipeCacheManager::RecipeCacheAllocationId prgDataUniqueAllocId  =
        std::make_pair(prgDataRecipeId, prgDataUniqueSubTypeAllocId);

    // First unique-ID allocation
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 10);
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            prgDataUniqueAllocId.first, prgDataSubType, prgDataUniqueAllocId.second,
            recipesIdToSize[10], blockAddresses, refCount, handle);
    ASSERT_EQ(RecipeCacheManager::BAR_ALLOCATED, allocStatus) << "Failed to cache Unique-ID";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(1, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // Re-try to allocate same unique-ID
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 20);
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            prgDataUniqueAllocId.first, prgDataSubType, prgDataUniqueAllocId.second,
            recipesIdToSize[20], blockAddresses, refCount, handle);
    ASSERT_EQ(failurestatus, allocStatus) << "Expected to fail to cache same Unique-ID twice";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(1, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // Second unique-ID allocation
    prgDataUniqueAllocId.second++;
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            prgDataUniqueAllocId.first, prgDataSubType, prgDataUniqueAllocId.second,
            recipesIdToSize[20], blockAddresses, refCount, handle);
    ASSERT_EQ(RecipeCacheManager::BAR_ALLOCATED, allocStatus) << "Expected to cache a second Unique-ID";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(2, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // Try to allocate singular-ID over the same recipe's sub-type
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 30);
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            prgDataSingularAllocId.first, prgDataSubType, prgDataSingularAllocId.second,
            recipesIdToSize[30], blockAddresses, refCount, handle);
    ASSERT_EQ(failurestatus, allocStatus) << "Expected to fail to cache Singular-ID";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(2, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // First singular-ID allocation, for exe-blobs ST
    updateBlocksAmount(totalBlocksAmount, requiredStatus, 30);
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            execBlobsAllocId.first, execBlobsSubType, execBlobsAllocId.second,
            recipesIdToSize[90], blockAddresses, refCount, handle);
    ASSERT_EQ(RecipeCacheManager::BAR_ALLOCATED, allocStatus) << "Failed to cache Singular-ID for exec-blobs sub-type";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(3, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // Re-use old unique-ID, when not used
    status = cache.setRecipeNotInUse(std::make_pair(prgDataRecipeId, prgDataUniqueSubTypeAllocId), refCount);
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for " << cache._getAllocationIdDescL(prgDataUniqueAllocId);
    ASSERT_EQ(1, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(2, cache.m_inUseList.size()) << "Unexpected In-use list size";
    //
    prgDataUniqueAllocId.second++;
    allocStatus =
        cache._allocateMemoryForSingleSubType(
            prgDataUniqueAllocId.first, prgDataSubType, prgDataUniqueAllocId.second,
            recipesIdToSize[20], blockAddresses, refCount, handle);
    ASSERT_EQ(RecipeCacheManager::BAR_ALLOCATED, allocStatus) << "Failed to cache Unique-ID, after unuse of another";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(3, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // Check PRG-Data is in cache
    Settable<uint64_t> programDataAddress = cache.getProgramDataAddressIfAlreadyInCache(prgDataUniqueAllocId.first);
    ASSERT_EQ(true, programDataAddress.is_set()) << "PRG-Data had not been in cache";
    ASSERT_EQ(programDataAddress.value(), blockAddresses.front()) << "PRG-Data address is not the last unique-ID allocated";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(3, cache.m_inUseList.size()) << "Unexpected In-use list size";

    // Check PRG-Data is in cache, even after completing usage
    // Releasing twice - one for the "Launch" and one for the "Kernel printf"
    status = cache.setRecipeNotInUse(prgDataUniqueAllocId, refCount);
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(3, cache.m_inUseList.size()) << "Unexpected In-use list size";
    status = cache.setRecipeNotInUse(prgDataUniqueAllocId, refCount);
    ASSERT_EQ(1, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(2, cache.m_inUseList.size()) << "Unexpected In-use list size";
    ASSERT_TRUE(status) << "Failed setRecipeNotInUse for " << cache._getAllocationIdDescL(prgDataUniqueAllocId);
    //
    programDataAddress = cache.getProgramDataAddressIfAlreadyInCache(prgDataUniqueAllocId.first);
    ASSERT_EQ(true, programDataAddress.is_set()) << "PRG-Data had not been in cache";
    ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size";
    ASSERT_EQ(3, cache.m_inUseList.size()) << "Unexpected In-use list size";
    //
    ASSERT_EQ(prgDataUniqueAllocId.second, cache.m_inUseList.front().second)
        << "Unexpected allocation-ID found at in-use list";
}

TEST_F(UTSynGaudiRecipeCaching, acquire_blocks)
{
    const uint64_t refCount    = 1;
    uint64_t blockSize         = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t recipeCacheSize   = 250 * KB;  // in KB

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());

    std::vector<uint64_t> prgDataBlocksVector;
    std::vector<uint64_t> prgCodeBlocksVector;
    //
    RecipeCacheManager::BlockPtrVector sectionsBlockLists[RecipeCacheManager::ST_AMOUNT];
    sectionsBlockLists[RecipeCacheManager::ST_PROGRAM_DATA]    = &prgDataBlocksVector;
    sectionsBlockLists[RecipeCacheManager::ST_EXECUTION_BLOBS] = &prgCodeBlocksVector;

    uint64_t sectionsHandles[RecipeCacheManager::ST_AMOUNT];

    bool     status        = false;
    uint64_t recipeId      = 30;
    uint64_t requestedSize = blockSize;

    uint64_t execBlobsSubTypeAllocId       = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    uint64_t prgDataSingularSubTypeAllocId = RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;

    RecipeCacheManager::SubType execBlobsSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::SubType prgDataSubType   = RecipeCacheManager::ST_PROGRAM_DATA;

    // Singular allocation and release operations (both sub-types are Singulars)
    {
        RecipeCacheManager::RecipeCacheAllocationId execBlobsAllocId =
            std::make_pair(recipeId, execBlobsSubTypeAllocId);
        RecipeCacheManager::RecipeCacheAllocationId prgDataSingularAllocId  =
            std::make_pair(recipeId, prgDataSingularSubTypeAllocId);

        RecipeCacheManager::AllSubTypesAllocationRequests allocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               prgDataSingularSubTypeAllocId, requestedSize);

        // First Acquire - Both Singular
        status = cache.acquireBlocks(recipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (first singular allocation)";
        ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size (first singular allocation)";
        ASSERT_EQ(2, cache.m_inUseList.size())    << "Unexpected In-use list size (first singular allocation)";
        ASSERT_EQ(1, cache.m_allocationIdHandlesDB[prgDataSingularAllocId])
            << "Unexpected PRG-Data handle (first singular allocation)";
        ASSERT_EQ(2, cache.m_allocationIdHandlesDB[execBlobsAllocId])
            << "Unexpected PRG-Code handle (first singular allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::SINGULAR,
                  cache.m_recipesInfoTable[recipeId].allocState[prgDataSubType])
            << "Unexpected PRG-Data allocation-state (first singular allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::SINGULAR,
                  cache.m_recipesInfoTable[recipeId].allocState[execBlobsSubType])
            << "Unexpected PRG-Code allocation-state (first singular allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[prgDataSubType].size())
            << "Unexpected PRG-Data sub-type IDs' amount (first singular allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[execBlobsSubType].size())
            << "Unexpected PRG-Code sub-type IDs' amount (first singular allocation)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[prgDataSingularAllocId].refCount)
            << "Unexpected PRG-Data reference-count (first singular allocation)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (first singular allocation)";

        // Second Acquire (same recipe) - Both Singular (same IDs)
        status = cache.acquireBlocks(recipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (second singular allocation)";
        ASSERT_EQ(0, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size (second singular allocation)";
        ASSERT_EQ(2, cache.m_inUseList.size())    << "Unexpected In-use list size (second singular allocation)";
        ASSERT_EQ(1, cache.m_allocationIdHandlesDB[prgDataSingularAllocId])
            << "Unexpected PRG-Data handle (second singular allocation)";
        ASSERT_EQ(2, cache.m_allocationIdHandlesDB[execBlobsAllocId])
            << "Unexpected PRG-Code handle (second singular allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::SINGULAR,
                  cache.m_recipesInfoTable[recipeId].allocState[prgDataSubType])
            << "Unexpected PRG-Data allocation-state (second singular allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::SINGULAR,
                  cache.m_recipesInfoTable[recipeId].allocState[execBlobsSubType])
            << "Unexpected PRG-Code allocation-state (second singular allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[prgDataSubType].size())
            << "Unexpected PRG-Data sub-type IDs' amount (second singular allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[execBlobsSubType].size())
            << "Unexpected PRG-Code sub-type IDs' amount (second singular allocation)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[prgDataSingularAllocId].refCount)
            << "Unexpected PRG-Data reference-count (second singular allocation)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (second singular allocation)";

        // First (PrgData) release
        status = cache.setRecipeNotInUse(prgDataSingularAllocId, refCount);
        ASSERT_EQ(0, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (first PRG-Data singular release)";
        ASSERT_EQ(2, cache.m_inUseList.size())
            << "Unexpected In-use list size (first PRG-Data singular release)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[prgDataSingularAllocId].refCount)
            << "Unexpected PRG-Data reference-count (first PRG-Data singular release)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (first PRG-Data singular release)";

        // Second (PrgData) release
        status = cache.setRecipeNotInUse(prgDataSingularAllocId, refCount);
        ASSERT_EQ(1, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (second PRG-Data singular release)";
        ASSERT_EQ(1, cache.m_inUseList.size())
            << "Unexpected In-use list size (second PRG-Data singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[prgDataSingularAllocId].refCount)
            << "Unexpected PRG-Data reference-count (second PRG-Data singular release)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (second PRG-Data singular release)";

        // First (PrgCode) release
        status = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
        ASSERT_EQ(1, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (first PRG-Code singular release)";
        ASSERT_EQ(1, cache.m_inUseList.size())
            << "Unexpected In-use list size (first PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[prgDataSingularAllocId].refCount)
            << "Unexpected PRG-Data reference-count (first PRG-Code singular release)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (first PRG-Code singular release)";

        // Second (PrgCode) release
        status = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
        ASSERT_EQ(2, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_inUseList.size())
            << "Unexpected In-use list size (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[prgDataSingularAllocId].refCount)
            << "Unexpected PRG-Data reference-count (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (second PRG-Code singular release)";
    }

    // Unique allocation and release operations (only PrgData sub-types is Unique)
    {
        recipeId++; // A different recipe

        uint64_t firstPrgDataUniqueSubTypeAllocId  = 0;
        uint64_t secondPrgDataUniqueSubTypeAllocId = 1;

        RecipeCacheManager::RecipeCacheAllocationId execBlobsAllocId =
            std::make_pair(recipeId, execBlobsSubTypeAllocId);
        RecipeCacheManager::RecipeCacheAllocationId firstPrgDataUniqueAllocId  =
            std::make_pair(recipeId, firstPrgDataUniqueSubTypeAllocId);
        RecipeCacheManager::RecipeCacheAllocationId secondPrgDataUniqueAllocId  =
            std::make_pair(recipeId, secondPrgDataUniqueSubTypeAllocId);

        // First Acquire
        RecipeCacheManager::AllSubTypesAllocationRequests firstAllocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(firstAllocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(firstAllocationRequests, prgDataSubType,
                                                               firstPrgDataUniqueSubTypeAllocId, requestedSize);

        status = cache.acquireBlocks(recipeId, sectionsBlockLists, sectionsHandles, refCount, firstAllocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (first unique allocation)";
        ASSERT_EQ(2, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size (first unique allocation)";
        ASSERT_EQ(2, cache.m_inUseList.size())    << "Unexpected In-use list size (first unique allocation)";
        ASSERT_EQ(3, cache.m_allocationIdHandlesDB[firstPrgDataUniqueAllocId])
            << "Unexpected PRG-Data handle (first unique allocation)";
        ASSERT_EQ(4, cache.m_allocationIdHandlesDB[execBlobsAllocId])
            << "Unexpected PRG-Code handle (first unique allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::UNIQUE,
                  cache.m_recipesInfoTable[recipeId].allocState[prgDataSubType])
            << "Unexpected PRG-Data allocation-state (first unique allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::SINGULAR,
                  cache.m_recipesInfoTable[recipeId].allocState[execBlobsSubType])
            << "Unexpected PRG-Code allocation-state (first unique allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[prgDataSubType].size())
            << "Unexpected PRG-Data sub-type IDs' amount (first unique allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[execBlobsSubType].size())
            << "Unexpected PRG-Code sub-type IDs' amount (first unique allocation)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[firstPrgDataUniqueAllocId].refCount)
            << "Unexpected PRG-Data reference-count (first unique allocation)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (first unique allocation)";

        // Second Acquire (same recipe) - New Unique allocation-ID
        RecipeCacheManager::AllSubTypesAllocationRequests secondAllocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(secondAllocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(secondAllocationRequests, prgDataSubType,
                                                               secondPrgDataUniqueSubTypeAllocId, requestedSize);

        status = cache.acquireBlocks(recipeId, sectionsBlockLists, sectionsHandles, refCount, secondAllocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (second unique allocation)";
        ASSERT_EQ(2, cache.m_notInUseList.size()) << "Unexpected Not-in-use list size (second unique allocation)";
        ASSERT_EQ(3, cache.m_inUseList.size())    << "Unexpected In-use list size (second unique allocation)";
        ASSERT_EQ(5, cache.m_allocationIdHandlesDB[secondPrgDataUniqueAllocId])
            << "Unexpected PRG-Data handle (second unique allocation)";
        ASSERT_EQ(4, cache.m_allocationIdHandlesDB[execBlobsAllocId])
            << "Unexpected PRG-Code handle (second unique allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::UNIQUE,
                  cache.m_recipesInfoTable[recipeId].allocState[prgDataSubType])
            << "Unexpected PRG-Data allocation-state (second unique allocation)";
        ASSERT_EQ(RecipeCacheManager::AllocationState::SINGULAR,
                  cache.m_recipesInfoTable[recipeId].allocState[execBlobsSubType])
            << "Unexpected PRG-Code allocation-state (second unique allocation)";
        ASSERT_EQ(2, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[prgDataSubType].size())
            << "Unexpected PRG-Data sub-type IDs' amount (second unique allocation)";
        ASSERT_EQ(1, cache.m_recipesInfoTable[recipeId].subTypeAllocIds[execBlobsSubType].size())
            << "Unexpected PRG-Code sub-type IDs' amount (second unique allocation)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[secondPrgDataUniqueAllocId].refCount)
            << "Unexpected second PRG-Data reference-count (second unique allocation)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (second unique allocation)";

        // First (PrgData) release
        status = cache.setRecipeNotInUse(firstPrgDataUniqueAllocId, refCount);
        ASSERT_EQ(3, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (first PRG-Data unique release)";
        ASSERT_EQ(2, cache.m_inUseList.size())
            << "Unexpected In-use list size (first PRG-Data unique release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[firstPrgDataUniqueAllocId].refCount)
            << "Unexpected first PRG-Data reference-count (first PRG-Data unique release)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[secondPrgDataUniqueAllocId].refCount)
            << "Unexpected second PRG-Data reference-count (first PRG-Data unique release)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (first PRG-Data unique release)";

        // Second (PrgData) release
        status = cache.setRecipeNotInUse(secondPrgDataUniqueAllocId, refCount);
        ASSERT_EQ(4, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (second PRG-Data unique release)";
        ASSERT_EQ(1, cache.m_inUseList.size())
            << "Unexpected In-use list size (second PRG-Data unique release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[firstPrgDataUniqueAllocId].refCount)
            << "Unexpected first PRG-Data reference-count (second PRG-Data unique release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[secondPrgDataUniqueAllocId].refCount)
            << "Unexpected second PRG-Data reference-count (second PRG-Data unique release)";
        ASSERT_EQ(2, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (second PRG-Data unique release)";

        // First (PrgCode) release
        status = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
        ASSERT_EQ(4, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (first PRG-Code singular release)";
        ASSERT_EQ(1, cache.m_inUseList.size())
            << "Unexpected In-use list size (first PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[firstPrgDataUniqueAllocId].refCount)
            << "Unexpected first PRG-Data reference-count (first PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[secondPrgDataUniqueAllocId].refCount)
            << "Unexpected second PRG-Data reference-count (first PRG-Code singular release)";
        ASSERT_EQ(1, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (first PRG-Code singular release)";

        // Second (PrgCode) release
        status = cache.setRecipeNotInUse(execBlobsAllocId, refCount);
        ASSERT_EQ(5, cache.m_notInUseList.size())
            << "Unexpected Not-in-use list size (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_inUseList.size())
            << "Unexpected In-use list size (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[firstPrgDataUniqueAllocId].refCount)
            << "Unexpected first PRG-Data reference-count (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[secondPrgDataUniqueAllocId].refCount)
            << "Unexpected second PRG-Data reference-count (second PRG-Code singular release)";
        ASSERT_EQ(0, cache.m_recipesSubTypeTable[execBlobsAllocId].refCount)
            << "Unexpected PRG-Code reference-count (second PRG-Code singular release)";
    }
}

TEST_F(UTSynGaudiRecipeCaching, recipe_cache_usage)
{
    const uint64_t refCount    = 1;
    uint64_t blockSize         = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * KB;
    uint64_t recipeCacheSize   = 250 * KB;  // in KB

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());

    std::vector<uint64_t> prgDataBlocksVector;
    std::vector<uint64_t> prgCodeBlocksVector;
    //
    RecipeCacheManager::BlockPtrVector sectionsBlockLists[RecipeCacheManager::ST_AMOUNT];
    sectionsBlockLists[RecipeCacheManager::ST_PROGRAM_DATA]    = &prgDataBlocksVector;
    sectionsBlockLists[RecipeCacheManager::ST_EXECUTION_BLOBS] = &prgCodeBlocksVector;

    uint64_t sectionsHandles[RecipeCacheManager::ST_AMOUNT];

    bool     status        = false;
    uint64_t requestedSize = blockSize;

    uint64_t uniqueRecipeId   = 10;
    uint64_t singularRecipeId = 20;

    uint64_t execBlobsSubTypeAllocId       = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    uint64_t prgDataSingularSubTypeAllocId = RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;
    uint64_t prgDataUniqueSubTypeAllocId   = 0;

    RecipeCacheManager::SubType execBlobsSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::SubType prgDataSubType   = RecipeCacheManager::ST_PROGRAM_DATA;

    // Unique Allocations & Release
    {
        RecipeCacheManager::AllSubTypesAllocationRequests allocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               ++prgDataUniqueSubTypeAllocId, requestedSize);

        status = cache.acquireBlocks(uniqueRecipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (first unique allocation)";

        allocationRequests.clear();
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               ++prgDataUniqueSubTypeAllocId, requestedSize);
        status = cache.acquireBlocks(uniqueRecipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (second unique allocation)";

        cache.eraseRecipeFromDb(uniqueRecipeId);
    }

    // Singular Allocations & Release
    {
        RecipeCacheManager::AllSubTypesAllocationRequests allocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               prgDataSingularSubTypeAllocId, requestedSize);

        status = cache.acquireBlocks(singularRecipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (first singular allocation)";

        allocationRequests.clear();
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               prgDataSingularSubTypeAllocId, requestedSize);
        status = cache.acquireBlocks(singularRecipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (second singular allocation)";

        cache.eraseRecipeFromDb(singularRecipeId);
    }
}

TEST_F(UTSynGaudiRecipeCaching, acquire_and_override)
{
    const uint64_t refCount    = 1;

    uint64_t blockSize       = GCFG_RECIPE_CACHE_BLOCK_SIZE.value(); // In KB
    // Having only two blocks - one for each SubType for a single recipe
    uint64_t recipeCacheSize = blockSize * 2; // In KB

    RecipeCacheManager cache(0x20000000,
                             synDeviceGaudi,
                             recipeCacheSize,
                             GCFG_RECIPE_CACHE_BLOCK_SIZE.value());

    std::vector<uint64_t> prgDataBlocksVector;
    std::vector<uint64_t> prgCodeBlocksVector;
    //
    RecipeCacheManager::BlockPtrVector sectionsBlockLists[RecipeCacheManager::ST_AMOUNT];
    sectionsBlockLists[RecipeCacheManager::ST_PROGRAM_DATA]    = &prgDataBlocksVector;
    sectionsBlockLists[RecipeCacheManager::ST_EXECUTION_BLOBS] = &prgCodeBlocksVector;

    uint64_t sectionsHandles[RecipeCacheManager::ST_AMOUNT];

    bool     status        = false;
    uint64_t requestedSize = blockSize * KB;

    uint64_t firstRecipeId  = 10;
    uint64_t secondRecipeId = 20;

    uint64_t execBlobsSubTypeAllocId       = RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    uint64_t prgDataSingularSubTypeAllocId = RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;

    RecipeCacheManager::SubType execBlobsSubType = RecipeCacheManager::ST_EXECUTION_BLOBS;
    RecipeCacheManager::SubType prgDataSubType   = RecipeCacheManager::ST_PROGRAM_DATA;

    // First recipe acquire
    {
        RecipeCacheManager::AllSubTypesAllocationRequests allocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               prgDataSingularSubTypeAllocId, requestedSize);

        status = cache.acquireBlocks(firstRecipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (first singular allocation)";
    }

    // Set Not-In-Use for the allocations of the first recipe
    {
        status = cache.setRecipeNotInUse(std::make_pair(firstRecipeId, execBlobsSubTypeAllocId), refCount);
        ASSERT_TRUE(status) << "Failed setRecipeNotInUse for execution-blobs SubType";

        status = cache.setRecipeNotInUse(std::make_pair(firstRecipeId, prgDataSingularSubTypeAllocId), refCount);
        ASSERT_TRUE(status) << "Failed setRecipeNotInUse for prg-data-blobs SubType";
    }

    // Second recipe acquire - should delete the first recipe allocations (eviction)
    {
        RecipeCacheManager::AllSubTypesAllocationRequests allocationRequests;
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, execBlobsSubType,
                                                               execBlobsSubTypeAllocId, requestedSize);
        RecipeCacheManager::addSingleSubTypeAllocationRequestL(allocationRequests, prgDataSubType,
                                                               prgDataSingularSubTypeAllocId, requestedSize);

        status = cache.acquireBlocks(secondRecipeId, sectionsBlockLists, sectionsHandles, refCount, allocationRequests);
        ASSERT_EQ(true, status) << "Failed to acquire-blocks (first singular allocation)";
    }

    // Erasing the first recipe from cache
    cache.eraseRecipeFromDb(firstRecipeId);
}