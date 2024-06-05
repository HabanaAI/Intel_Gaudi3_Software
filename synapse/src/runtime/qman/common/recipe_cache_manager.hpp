#pragma once

#include "hal_reader/hal_reader.h"
#include "infra/defs.h"
#include "runtime/common/osal/buffer_allocator.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

// NOTE: L at the end of the function name (xxxxL), means the function is called with a mutex lock

/*********************************************************************************/
/*                                                                               */
/*                                 BlockAllocator                                */
/*                                                                               */
/*********************************************************************************/
#define RECIPE_CACHE_ALIGNMENT 32
#define KB                     1024
// No thread-safeness is guaranteed on the following classes
// todo support non-contiguous allocations currently supporting only contiguous allocations
class BlockAllocator
{
public:
    // Initialize freeList, cache size and block size configured in global variable XXX
    BlockAllocator(uint64_t         poolSize,
                   uint64_t         blockSize,
                   deviceAddrOffset baseAddr,
                   uint32_t         cacheAllignment = RECIPE_CACHE_ALIGNMENT);
    ~BlockAllocator();
    // allocate blockNr number of blocks, returned in blockList
    // If no contiguous blocks available return fail
    bool blockAllocateL(uint64_t blockNr, std::vector<uint64_t>& blockList);

    bool contiguousBlockAllocateL(uint64_t blockNr, std::vector<uint64_t>& blockList);

    bool freeL(std::vector<uint64_t>& blockList);

    uint64_t getFreeNumBlocksL() { return m_nbFreeBlocks; }

private:
    // Pointer to free blocks of memory
    uint32_t            m_nbFreeBlocks;
    const uint64_t      m_poolSize;
    const uint64_t      m_blockSize;
    uint64_t            m_nBlocks;
    uint64_t            m_baseAddr;
    std::vector<bool>   m_takenBlocks;
};

/*********************************************************************************/
/*                                                                               */
/*                                RecipeCacheManager                             */
/*                                                                               */
/*********************************************************************************/
class RecipeCacheManager
{
public:
    using BlockPtrVector = std::vector<uint64_t>*;

    enum SubType : uint8_t
    {
        ST_FIRST,

        ST_PROGRAM_DATA = ST_FIRST,
        ST_EXECUTION_BLOBS,  // non patchable, program code

        ST_AMOUNT            // always the last
    };

    static const uint64_t INVALID_ALLOCATION_ID;
    //
    // Reserved sub-type ID for the execution PRG-Code, as there will be is a single entry per recipe,
    // for that part
    static const uint64_t EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    //
    // PRG-Data invalid ID
    static const uint64_t PRG_DATA_INVALID_ALLOCATION_ID;
    // PRG-Data Singular ID
    static const uint64_t PRG_DATA_SINGULAR_ALLOCATION_ID;
    // PRG-Data last unique ID
    static const uint64_t PRG_DATA_LAST_UNIQUE_ALLOCATION_ID;

    // When allocating all TYPE_IDs, we will first perform the allocation of the contiguous TYPE_ID(s),
    // as those have a more strict rules

    enum BlockAllocatorResult : uint8_t
    {
        BAR_ALLOCATED,
        BAR_ALREADY_IN_CACHE,
        BAR_FAILURE
    };

    // Pair of <recipeHandle, Sub-Type Allocation-ID>
    typedef std::pair<uint64_t, uint64_t> RecipeCacheAllocationId;

    typedef struct _AllocationRequestInfo
    {
        uint64_t subTypeId;
        uint64_t size;
    } AllocationRequestInfo;
    // Pair of <Sub-Type ID, Sub-Type Allocation-ID>
    typedef std::pair<SubType, AllocationRequestInfo>   SingleSubTypeAllocationRequest;
    // Recipe-cache allocation-requests, for numerous sub-types
    typedef std::deque<SingleSubTypeAllocationRequest>  AllSubTypesAllocationRequests;

    enum class AllocationState
    {
        NOT_SET,
        UNIQUE,
        SINGULAR
    };

    // Holds information about a given recipe
    typedef struct _RecipeInfoEntry
    {
        // A uniue-allocation is one that is used exactly once
        // At the moment it will "hang out", after completion, at the not-in-use DB
        AllocationState      allocState[ST_AMOUNT];
        // Holding unique allocations (Meaning, only a single Singular allocation ID. No reason of duplications)
        std::deque<uint64_t> subTypeAllocIds[ST_AMOUNT];
    } RecipeInfoEntry;

    // Holds information about a given recipe sub-type's entry
    // To be used in the notInUse (both of them) and InUse lists
    typedef struct _RecipeSubTypeEntry
    {
        uint64_t                                     refCount = 0;
        std::list<RecipeCacheAllocationId>::iterator iterator;  // must always be valid
        std::vector<uint64_t>                        blockList;
        SubType                                      entrySubType;
    } RecipeSubTypeEntry;

    // A hash function used to hash a pair of any kind
    struct hash_pair
    {
        size_t operator()(const std::pair<uint64_t, uint64_t>& p) const
        {
            auto hash1 = std::hash<uint64_t> {}(p.first);
            auto hash2 = std::hash<uint64_t> {}(p.second);
            return hash1 ^ hash2;
        }
    };

    // Called in synDeviceAcquire. Allocate memory for cache as defined in global parameter
    RecipeCacheManager(uint64_t      baseAddress,
                       synDeviceType devType,
                       uint64_t      cacheSize,
                       uint64_t      recipeCacheBlockSize);

    // Remove copy c'tor and assigment operator
    RecipeCacheManager(const RecipeCacheManager&) = delete;
    RecipeCacheManager& operator=(RecipeCacheManager const&) = delete;

    // Frees memory for cache, frees datastructures used
    ~RecipeCacheManager();

    // Returns recipe Information on device ( to generate the linDMA commands)
    // maybe return value here should be of list of contiguous memory ( if there are some blocks that are contiguous
    // to simplify the "patching" of the linDMA commands (2nd phase)
    bool getRecipeCacheInfo(RecipeCacheAllocationId allocationId, std::list<uint64_t>& blockAddresses);

    bool acquireBlocks(uint64_t                             recipeId,
                       BlockPtrVector*                      sectionsBlockLists,
                       // the handles are used for exploring CS-reuse
                       uint64_t*                            sectionsHandles,
                       uint64_t                             refCount,
                       const AllSubTypesAllocationRequests& allocationRequests);

    Settable<uint64_t> getProgramDataAddressIfAlreadyInCache(uint64_t recipeId);

    // When detected that recipe is not in use ( on every synLaunch we can check state on driver), should
    // update recipeCache to notInUse to make room for new recipes in the cache.
    // No need to memset, just move this recipe's blocks to "not-in-use" list
    bool setRecipeNotInUse(RecipeCacheAllocationId recipeId, uint64_t refCount);

    bool setRecipeInUse(RecipeCacheAllocationId recipeId, uint64_t refCount);

    void eraseRecipeFromDb(uint64_t recipeId);

    void getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress) const;

    // A method which assists creating a single allocation request
    static void addSingleSubTypeAllocationRequestL(AllSubTypesAllocationRequests& allocationRequests,
                                                   SubType                        subType,
                                                   uint64_t                       subTypeId,
                                                   uint64_t                       requestedSize);
private:
#define FRIEND_TEST(test_case_name, test_name) friend class test_case_name##_##test_name##_Test

    FRIEND_TEST(UTSynGaudiRecipeCaching, basic_LRU);
    FRIEND_TEST(UTSynGaudiRecipeCaching, check_LRU_policy);
    FRIEND_TEST(UTSynGaudiRecipeCaching, test_contiguous_allocation);
    FRIEND_TEST(UTSynGaudiRecipeCaching, unique_ids_allocations);
    FRIEND_TEST(UTSynGaudiRecipeCaching, acquire_blocks);
    FRIEND_TEST(UTSynGaudiRecipeCaching, recipe_cache_usage);

#undef FRIEND_TEST

    // Try to allocate memory for a recipe sub-type
    // When no room available, try to invalidate according to LRU policy (look at the not-in-use recipes)
    // when success, Add recipe to recipeCacheDB  <name, recipeEntry>
    // use deviceMemoryAllocator API
    BlockAllocatorResult _allocateMemoryForSingleSubType(uint64_t                recipeId,
                                                         SubType                 subType,
                                                         uint64_t                subTypeAllocId,
                                                         uint64_t                allocSize,
                                                         std::vector<uint64_t>&  blockAddresses,
                                                         uint64_t                refCount,
                                                         uint64_t&               handle);

    void _deleteAllocationIdFromHandelsDB(RecipeCacheAllocationId allocationId);

    void _deleteRecipeFromHandelsDB(uint64_t recipeId);

    // Removes recipe from DB. If not found returns fail
    bool _invalidateAllocationL(RecipeCacheAllocationId allocationId);

    // Evict allocations unil a size of sizeToFree had been free
    bool _freeMemoryBySizeL(uint64_t sizeToFree, bool& hasRecipeToFree);

    // Remove recipe whether its in "not in use" or "in-use" list,
    bool _deleteAllocationL(RecipeCacheAllocationId allocationId, bool shouldDeleteBlocks);

    bool _isAllocationInUseL(RecipeCacheAllocationId allocationId);

    std::list<RecipeCacheAllocationId>& _getRelavantRecipeListL(RecipeCacheAllocationId allocationId);

    bool setRecipeInLRUListL(RecipeCacheAllocationId allocationId, uint64_t refCount, bool isInUse);

    std::string _getAllocationIdDescL(RecipeCacheAllocationId allocationId);

    bool _incrementAllocationIdHandleL(RecipeCacheAllocationId allocationId, uint64_t& newHandle);

    bool _getAllocationIdHandle(RecipeCacheAllocationId allocationId, uint64_t& handle);

    void _releaseWaitingLaunches();

    uint64_t _getRecipeSubTypeLastAllocationId(uint64_t recipeId, SubType subType) const;

    // return true, in case of a valid order
    static bool _validateAllocationsRequest(const AllSubTypesAllocationRequests& allocationRequests,
                                            uint64_t                             recipeId);

    BlockAllocatorResult _addNewAllocationL(RecipeCacheAllocationId allocationId,
                                            RecipeSubTypeEntry&     recipeEntry,
                                            std::vector<uint64_t>&  blockAddresses,
                                            uint64_t&               handle,
                                            uint64_t                refCount,
                                            uint64_t                allocSize,
                                            SubType                 subType);


    uint64_t m_baseAddress;
    uint64_t m_lastAddress;
    uint64_t m_blockSize;

    synDeviceType m_deviceType;
    HalReaderPtr  m_pHalReader;

    // Each Allocation-ID (and not a pair of [recipe-ID, Sub-Type ID]) has its own handle
    // DB of <RecipeCacheAllocationId, handle>
    std::unordered_map<RecipeCacheAllocationId, uint64_t, hash_pair> m_allocationIdHandlesDB;

    // DB of <RecipeCacheAllocationId, RecipeSubTypeEntry>
    // Holds acquired entries information
    std::unordered_map<RecipeCacheAllocationId, RecipeSubTypeEntry, hash_pair> m_recipesSubTypeTable;

    // DB of <recipeId, RecipeInfoEntry>
    // Holds allocations-info, per recipe-ID
    std::unordered_map<uint64_t, RecipeInfoEntry> m_recipesInfoTable;

    // LRU managed
    //
    // Order of allocation:
    //   1) If in cache - use it
    //   2) In case there are enough free blocks - acquire them
    //   3) Perform eviction from the notInUseList
    // Optional: Add another DB for the unique-allocations, so they will be chosen before the non-unique,
    //           while not being free otherwise, and hence not breaking the kernel-printf support
    std::list<RecipeCacheAllocationId> m_inUseList;
    std::list<RecipeCacheAllocationId> m_notInUseList;

    BlockAllocator m_pBlockAllocator;
    std::mutex     m_mutex;

    mutable std::mutex              m_condVarMutex;
    mutable std::condition_variable m_condVar;
    // all waiters will get a notifyIndedx and when a notify will occour
    // the notifyId will be incremented, new waiters will get the incremented value so they won't be released
    // from older notifies
    uint64_t m_notifyIndex;

    // For Singular Sub-Type:
    //      For a given [recipe, sub-type] pair, a new allocation might be required, due to eviction,
    //      and for the same allocationId
    //      Hence, we need to provide a new handle for the same allocationId
    //
    // For the Unique Sub-Type:
    //      For a given [recipe, sub-type] pair, there might have multiple Unique allocationIds
    //      As we need to distinguish between them, increment a global (static) handle,
    //      for having a unique handle, for each allocation ID, although may share the same [recipe, sub-type]
    uint64_t m_globalHandle;
};
