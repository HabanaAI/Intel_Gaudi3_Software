#include "recipe_cache_manager.hpp"

#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "infra/defs.h"
#include "recipe.h"
#include "runtime/common/common_types.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"

#include <cstdint>
#include <mutex>

extern HalReaderPtr instantiateGaudiHalReader();

//
// INVALID              = u64::max
// EXEC_BLOBS_PRG_CODE
// PRG_DATA_SINGULAR
// LAST_UNIQUE_PRG_CODE
// ...
// First UNIQUE_PRG_CODE = 0
//
const uint64_t RecipeCacheManager::INVALID_ALLOCATION_ID = std::numeric_limits<uint64_t>::max();
//
// Reserved sub-type ID for the execution PRG-Code, as there will be is a single entry per recipe,
// for that part
const uint64_t RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID = INVALID_ALLOCATION_ID - 1;
//
// PRG-Data invalid ID
const uint64_t RecipeCacheManager::PRG_DATA_INVALID_ALLOCATION_ID      = EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
// PRG-Data Singular ID
const uint64_t RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID     = PRG_DATA_INVALID_ALLOCATION_ID - 1;
// PRG-Data last unique ID
const uint64_t RecipeCacheManager::PRG_DATA_LAST_UNIQUE_ALLOCATION_ID  = PRG_DATA_SINGULAR_ALLOCATION_ID - 1;


// Recipe cache manager is on-device cache that stores recently used static part of recipes
// on pre allocated memory.
// Cache management is on recipe granularity
// Cache invalidation policy is LRU based.
BlockAllocator::BlockAllocator(uint64_t         poolSize,
                               uint64_t         blockSize,
                               deviceAddrOffset baseAddr,
                               uint32_t         cacheAllignment)
: m_poolSize(poolSize), m_blockSize(blockSize), m_nBlocks(0), m_baseAddr(baseAddr)
{
    // Initialize freeList
    m_baseAddr = ((m_baseAddr + cacheAllignment - 1) / cacheAllignment) * cacheAllignment;
    // Todo, implement Heap allocator with block size
    HB_ASSERT(m_poolSize % m_blockSize == 0, "Pool size must be a multiple of block size");

    // initialize free blocks list
    m_nbFreeBlocks = m_poolSize / m_blockSize;
    if (m_baseAddr != baseAddr)
    {
        m_nbFreeBlocks--;
    }
    m_nBlocks = m_nbFreeBlocks;

    // Initialize takenBlock status - this list made for finding contiguous free blocks
    // for program data allocations
    m_takenBlocks.resize(m_nBlocks, false);

    LOG_TRACE(SYN_RCPE_CACHE,
              "BlockAllocator: poolSize {} blockSize {} cacheAllignment {} nBlocks {}",
              poolSize,
              blockSize,
              cacheAllignment,
              m_nBlocks);
}

BlockAllocator::~BlockAllocator() = default;

bool BlockAllocator::blockAllocateL(uint64_t blockNr, std::vector<uint64_t>& blockList)
{
    if (blockNr > m_nbFreeBlocks)
    {
        LOG_ERR(SYN_RCPE_CACHE, "{}: Requested {} blocks > available {} blocks", HLLOG_FUNC, blockNr, m_nbFreeBlocks);
        return false;
    }
    return contiguousBlockAllocateL(blockNr, blockList);
}

bool BlockAllocator::contiguousBlockAllocateL(uint64_t blockNr, std::vector<uint64_t>& blockList)
{
    uint64_t firstBlockIndex = 0, i = 0;
    uint64_t blockAmount = blockNr;

    while (i < m_nBlocks && blockAmount > 0)
    {
        if (m_takenBlocks[i] == false)
        {
            firstBlockIndex = i;
            while (blockAmount > 0 && i < m_nBlocks)
            {
                if (m_takenBlocks[i] == true)
                {
                    blockAmount = blockNr;
                    break;
                }
                i++;
                blockAmount--;
            }
        }
        i++;
    }

    // Didn't find
    if (blockAmount != 0)
    {
        return false;
    }

    // Calculate the addresses - in case we found contiguous blocks
    uint64_t insertAddress = m_baseAddr + (firstBlockIndex * m_blockSize);

    blockList.reserve(blockList.size() + blockNr);
    for (i = 0; i < blockNr; i++)
    {
        blockList.push_back(insertAddress);
        m_takenBlocks.at(firstBlockIndex + i) = true;
        insertAddress += m_blockSize;
    }
    m_nbFreeBlocks -= blockNr;
    return true;
}

bool BlockAllocator::freeL(std::vector<uint64_t>& blockList)
{
    for (auto blockAddr : blockList)
    {
        uint64_t blockIndex          = (blockAddr - m_baseAddr) / m_blockSize;
        m_takenBlocks.at(blockIndex) = false;
    }
    m_nbFreeBlocks += blockList.size();
    return true;
}

// cache size in KB
RecipeCacheManager::RecipeCacheManager(uint64_t      baseAddress,
                                       synDeviceType devType,
                                       uint64_t      cacheSize,
                                       uint64_t      recipeCacheBlockSize)
: m_baseAddress(baseAddress),
  m_lastAddress(baseAddress + GCFG_RECIPE_CACHE_SIZE.value() * 1024 - 1),
  m_blockSize(recipeCacheBlockSize * KB),
  m_deviceType(devType),
  m_pBlockAllocator(cacheSize * KB, m_blockSize, baseAddress),
  m_notifyIndex(0),
  m_globalHandle(0)
{
    LOG_TRACE(SYN_RCPE_CACHE,
              "{}: baseAddress {:#x} cacheSize {:#x} blockSize {:#x}",
              HLLOG_FUNC,
              baseAddress,
              cacheSize,
              recipeCacheBlockSize);

    m_pHalReader = nullptr;

    if (m_deviceType == synDeviceGaudi)
    {
        m_pHalReader = instantiateGaudiHalReader();
    }
}

// Frees memory for cache, frees datastructures used
RecipeCacheManager::~RecipeCacheManager() {}

// maybe return value here should be of list of contiguous memory ( if there are some blocks that are contiguous
// to simplify the "patching" of the linDMA commands (2nd phase
bool RecipeCacheManager::getRecipeCacheInfo(RecipeCacheAllocationId allocationId, std::list<uint64_t>& blockAddresses)
{
    HB_ASSERT(0, "not implemented");
    return false;
}

Settable<uint64_t> RecipeCacheManager::getProgramDataAddressIfAlreadyInCache(uint64_t recipeId)
{
    Settable<uint64_t>    programDataAddress;
    std::vector<uint64_t> blockAddressesList;

    std::unique_lock<std::mutex> mlock(m_mutex);
    uint64_t allocationId = _getRecipeSubTypeLastAllocationId(recipeId, ST_PROGRAM_DATA);
    if (allocationId == INVALID_ALLOCATION_ID)
    {
        LOG_ERR(SYN_RCPE_CACHE, "No PRG-Data allocation for recipeId {}", recipeId);
        return programDataAddress;
    }

    RecipeCacheAllocationId programDataId = std::make_pair(recipeId, allocationId);

    auto subTypeIter = m_recipesSubTypeTable.find(programDataId);
    if (subTypeIter == m_recipesSubTypeTable.end())
    {
        return programDataAddress;
    }

    const uint64_t refCount = 1;
    setRecipeInLRUListL(programDataId, refCount, true);

    programDataAddress = subTypeIter->second.blockList.front();
    return programDataAddress;
}

bool RecipeCacheManager::acquireBlocks(uint64_t                             recipeId,
                                       BlockPtrVector*                      sectionsBlockLists,
                                       uint64_t*                            sectionsHandles,
                                       uint64_t                             refCount,
                                       const AllSubTypesAllocationRequests& allocationRequests)
{
    if (!_validateAllocationsRequest(allocationRequests, recipeId))
    {
        return false;
    }

    do
    {
        std::unique_lock<std::mutex> mlock(m_mutex);
        BlockAllocatorResult allocateStatus = BAR_FAILURE;
        std::deque<uint64_t> subTypesAllocationIds;

        for (auto singleAllocReq : allocationRequests)
        {
            SubType  allocSubType   = singleAllocReq.first;
            uint64_t subTypeAllocId = singleAllocReq.second.subTypeId;
            uint64_t allocationSize = singleAllocReq.second.size;

            allocateStatus = _allocateMemoryForSingleSubType(recipeId,
                                                             allocSubType,
                                                             subTypeAllocId,
                                                             allocationSize,
                                                             *sectionsBlockLists[allocSubType],
                                                             refCount,
                                                             sectionsHandles[allocSubType]);
            if (allocateStatus == BAR_FAILURE)
            {
                // delete all formers
                RecipeCacheAllocationId deleteAllocationId;
                deleteAllocationId.first = recipeId;
                for (auto currSubTypeAllocationId : subTypesAllocationIds)
                {
                    deleteAllocationId.second = currSubTypeAllocationId;

                    // Theoretically, we may have allocated a Singular part but failed with a Unique one
                    if (!setRecipeInLRUListL(deleteAllocationId, refCount, false))
                    {
                        LOG_ERR(SYN_RCPE_CACHE,
                                "{} Failed to set allocation as not in-use (rollback) ({})",
                                HLLOG_FUNC,
                                _getAllocationIdDescL(deleteAllocationId));
                    }

                    if (m_recipesSubTypeTable[deleteAllocationId].refCount == 0)
                    {
                        _invalidateAllocationL(deleteAllocationId);
                    }
                }
                break;
            }
            else if (allocateStatus == BAR_ALREADY_IN_CACHE)
            {
                LOG_TRACE(SYN_RCPE_CACHE,
                          "{} cache already includes allocation-request (recipe-ID: 0x{:x} sub-type ID: {})",
                          HLLOG_FUNC,
                          recipeId,
                          subTypeAllocId);
            }
            subTypesAllocationIds.push_back(subTypeAllocId);
        }

        if (allocateStatus == BAR_FAILURE)
        {
            if (m_recipesSubTypeTable.size() == 0)
            {
                LOG_ERR(SYN_RCPE_CACHE,
                        "{} cache is empty and still can not allocate this recipe 0x{:x}",
                        HLLOG_FUNC,
                        recipeId);
                return false;
            }

            std::unique_lock<std::mutex> condMutex(m_condVarMutex);
            uint64_t                     launchWakeUpId = m_notifyIndex;
            LOG_TRACE(SYN_RCPE_CACHE, "sleep on launchWakeUPID: {} recipe 0x{:x}", launchWakeUpId, recipeId);
            mlock.unlock();
            m_condVar.wait(condMutex, [&] { return launchWakeUpId < m_notifyIndex; });
        }
        else
        {
            return true;
        }
    } while (true);

    return true;
}

// Try to allocate memory for recipe
// When no room available, try to invalidate according to LRU policy (look at the not-in-use recipes)
// when success, Add recipe to recipeCacheDB  <name, subTypeEntry>
// use deviceMemoryAllocator API
RecipeCacheManager::BlockAllocatorResult
RecipeCacheManager::_allocateMemoryForSingleSubType(uint64_t                recipeId,
                                                    SubType                 allocSubType,
                                                    uint64_t                subTypeAllocId,
                                                    uint64_t                allocSize,
                                                    std::vector<uint64_t>&  blockAddresses,
                                                    uint64_t                refCount,
                                                    uint64_t&               handle)
{
    uint64_t recipeCacheSize = GCFG_RECIPE_CACHE_SIZE.value() * 1024;

    RecipeCacheAllocationId allocationId(recipeId, subTypeAllocId);

    // validation done on earlier stage, but just for making sure...
    HB_ASSERT_DEBUG_ONLY(allocSize <= recipeCacheSize,
                         "Requested allocation size ({}) is bigger than cache size {}",
                         _getAllocationIdDescL(allocationId),
                         allocSize);

    auto recipeTableEntry = m_recipesSubTypeTable.find(allocationId);
    if (recipeTableEntry != m_recipesSubTypeTable.end())
    {   // recipe already cached.
        auto recipeInfoIter = m_recipesInfoTable.find(recipeId);
        if (recipeInfoIter == m_recipesInfoTable.end())
        {
            LOG_WARN(SYN_RCPE_CACHE,
                     "{}: allocationId {} already cached, while recipe-info (of {}) is missing",
                     HLLOG_FUNC,
                     _getAllocationIdDescL(allocationId),
                     recipeId);
            return BAR_FAILURE;
        }

        if (recipeInfoIter->second.allocState[allocSubType] == AllocationState::UNIQUE)
        {
            LOG_WARN(SYN_RCPE_CACHE,
                     "{}: allocationId {} is unique, and cannot be re-allocated",
                     HLLOG_FUNC,
                     _getAllocationIdDescL(allocationId));
            return BAR_FAILURE;
        }

        RecipeSubTypeEntry& subTypeEntry = recipeTableEntry->second;
        setRecipeInLRUListL(allocationId, refCount, true);
        blockAddresses.assign(subTypeEntry.blockList.begin(), subTypeEntry.blockList.end());
        LOG_TRACE(SYN_RCPE_CACHE,
                  "{}: allocationId {} already cached, returned the relevant blockAddresses",
                  HLLOG_FUNC,
                  _getAllocationIdDescL(allocationId));
        _getAllocationIdHandle(allocationId, handle);
        STAT_COLLECT_COND(1, allocSubType == ST_EXECUTION_BLOBS, nonPatchableAlreadyIn, prgDataAlreadyIn);
        return BAR_ALREADY_IN_CACHE;
    }

    // else - recipe is not in cache
    bool     hasRecipeToFree      = false;
    uint64_t numberOfBlocksNeeded = (allocSize + m_blockSize - 1) / m_blockSize;  // round up
    uint64_t freeMemory           = m_pBlockAllocator.getFreeNumBlocksL() * m_blockSize;

    // Unique handle replacement (eviction)
    auto recipeInfoIter = m_recipesInfoTable.find(recipeId);
    if (recipeInfoIter != m_recipesInfoTable.end())
    {
        AllocationState subTypeAllocState    = recipeInfoIter->second.allocState[allocSubType];
        bool            isSubTypeUniqueAlloc = (subTypeAllocState == AllocationState::UNIQUE);

        if ((isSubTypeUniqueAlloc) &&
            (recipeInfoIter->second.subTypeAllocIds[allocSubType].size() != 0))
        {
            RecipeCacheAllocationId oldAllocationId(recipeId, recipeInfoIter->second.subTypeAllocIds[allocSubType].front());
            if (!_isAllocationInUseL(oldAllocationId))
            {
                LOG_DEBUG(SYN_RCPE_CACHE,
                          "{}: Replacing allocationId {} with {}",
                          HLLOG_FUNC,
                          _getAllocationIdDescL(oldAllocationId),
                          _getAllocationIdDescL(allocationId));
                STAT_GLBL_COLLECT(1, recipeCacheUniqueReUse);

                RecipeSubTypeEntry& subTypeEntry = m_recipesSubTypeTable[allocationId];

                subTypeEntry.blockList = m_recipesSubTypeTable[oldAllocationId].blockList;
                _deleteAllocationL(oldAllocationId, false);
                return _addNewAllocationL(allocationId,
                                         subTypeEntry,
                                         blockAddresses,
                                         handle,
                                         refCount,
                                         allocSize,
                                         allocSubType);
            }
        }
    }

    // LRU eviction
    if (allocSize > freeMemory)
    {
        if (!_freeMemoryBySizeL(allocSize - freeMemory, hasRecipeToFree))
        {
            LOG_DEBUG(SYN_RCPE_CACHE,
                      "{}: Not enough space to add {}, size: {}, free memory in cache: {}",
                      HLLOG_FUNC,
                      _getAllocationIdDescL(allocationId),
                      allocSize,
                      freeMemory);
            STAT_GLBL_COLLECT(1, recipeCacheNoMemory);
            return BAR_FAILURE;
        }
    }

    RecipeSubTypeEntry& recipeAllocEntry = m_recipesSubTypeTable[allocationId];
    do
    {
        if (m_pBlockAllocator.blockAllocateL(numberOfBlocksNeeded, recipeAllocEntry.blockList))
        {
            return _addNewAllocationL(allocationId,
                                      recipeAllocEntry,
                                      blockAddresses,
                                      handle,
                                      refCount,
                                      allocSize,
                                      allocSubType);
        }

        LOG_TRACE(SYN_RCPE_CACHE,
                  "{}: Can not find contiguous block for recipe (free blocks - {}): {}",
                  HLLOG_FUNC,
                  _getAllocationIdDescL(allocationId),
                  allocSize);
        _freeMemoryBySizeL(allocSize, hasRecipeToFree);
        STAT_GLBL_COLLECT(1, RecipeCacheFreeMemory);
        // keep searching for contiguous memory in cache while there is at least one recipe to evict
    } while (hasRecipeToFree);

    m_recipesSubTypeTable.erase(allocationId);
    STAT_GLBL_COLLECT(1, recipeCacheNoMemoryAfterFree);
    return BAR_FAILURE;
}

// When detected that recipe is not in use ( on every synLaunch we can check state on driver), should
// update recipeCache to notInUse to make room for new recipes in the cache.
// No need to memset, just move this recipe's blocks to "not-in-use" list
bool RecipeCacheManager::setRecipeNotInUse(RecipeCacheAllocationId allocationId, uint64_t refCount)
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    return setRecipeInLRUListL(allocationId, refCount, false);
}

// cached recipe - changing its state from not in use to in use
bool RecipeCacheManager::setRecipeInUse(RecipeCacheAllocationId allocationId, uint64_t refCount)
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    return setRecipeInLRUListL(allocationId, refCount, true);
}

bool RecipeCacheManager::setRecipeInLRUListL(RecipeCacheAllocationId allocationId, uint64_t refCount, bool isInUse)
{
    auto recipeSubTypeIter = m_recipesSubTypeTable.find(allocationId);
    if (recipeSubTypeIter == m_recipesSubTypeTable.end())
    {
        LOG_DEBUG(SYN_RCPE_CACHE, "{}: {} was not found in cache", HLLOG_FUNC, _getAllocationIdDescL(allocationId));
        return false;
    }

    /*
      Removes and adds to list, even in case of same list, for LRU manners
    */
    RecipeSubTypeEntry& subTypeEntry = m_recipesSubTypeTable[allocationId];

    if (isInUse == false && subTypeEntry.refCount < refCount)
    {
        // Todo change it to assert in the future - invalid release request from cache might indicate a caller bug
        LOG_WARN(SYN_RCPE_CACHE,
                 "{}: {} invalid unused request. reducing RC from {} to {}",
                 HLLOG_FUNC,
                 _getAllocationIdDescL(allocationId),
                 refCount,
                 subTypeEntry.refCount);
        refCount = subTypeEntry.refCount;
    }

    if (isInUse == false && subTypeEntry.refCount > refCount)
    {
        subTypeEntry.refCount -= refCount;
    }
    else
    {
        std::list<RecipeCacheAllocationId>& LRUlist = _getRelavantRecipeListL(allocationId);
        LRUlist.erase(subTypeEntry.iterator);
        if (isInUse)
        {
            m_inUseList.push_front(allocationId);
            subTypeEntry.iterator = m_inUseList.begin();
            subTypeEntry.refCount += refCount;
        }
        else
        {
            m_notInUseList.push_front(allocationId);
            subTypeEntry.iterator = m_notInUseList.begin();
            // if we got here recount must be 1
            subTypeEntry.refCount = 0;
            _releaseWaitingLaunches();
        }
    }
    LOG_TRACE(SYN_RCPE_CACHE,
              "{}: {} set as {}in use (new refCount = {})",
              HLLOG_FUNC,
              _getAllocationIdDescL(allocationId),
              isInUse ? "" : "not ",
              subTypeEntry.refCount);

    return true;
}

void RecipeCacheManager::_releaseWaitingLaunches()
{
    std::unique_lock<std::mutex> condMutex(m_condVarMutex);
    m_notifyIndex++;
    LOG_TRACE(SYN_RCPE_CACHE, "notify Index: {}", m_notifyIndex);
    m_condVar.notify_all();
}

uint64_t RecipeCacheManager::_getRecipeSubTypeLastAllocationId(uint64_t recipeId, SubType allocSubType) const
{
    if (allocSubType >= ST_AMOUNT)
    {
        return INVALID_ALLOCATION_ID;
    }

    auto iter = m_recipesInfoTable.find(recipeId);
    if (iter == m_recipesInfoTable.end())
    {
        return INVALID_ALLOCATION_ID;
    }

    const RecipeInfoEntry& recipeAllocInfo = iter->second;

    uint64_t allocationId = (allocSubType == ST_PROGRAM_DATA) ?
                            PRG_DATA_SINGULAR_ALLOCATION_ID : EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID;
    if (recipeAllocInfo.allocState[allocSubType] == AllocationState::UNIQUE)
    {
        allocationId = (recipeAllocInfo.subTypeAllocIds[allocSubType].size() != 0) ?
                        recipeAllocInfo.subTypeAllocIds[allocSubType].back() :
                        INVALID_ALLOCATION_ID;
    }

    return allocationId;
}

bool RecipeCacheManager::_validateAllocationsRequest(const AllSubTypesAllocationRequests& allocationRequests,
                                                     uint64_t                             recipeId)
{
    uint64_t recipeCacheSize = GCFG_RECIPE_CACHE_SIZE.value() * KB;

    if (allocationRequests.size() > RecipeCacheManager::ST_AMOUNT)
    {
        LOG_ERR(SYN_RCPE_CACHE, "Too many allocation-requests {} (limit {})",
                allocationRequests.size(), RecipeCacheManager::ST_AMOUNT);
        return false;
    }

    SubType lastTypeFound            = ST_AMOUNT;
    uint64_t totalAllocSizeRequested = 0;

    for (auto singleRequest : allocationRequests)
    {
        uint64_t allocationSize = singleRequest.second.size;
        // Not mandatory, as total size validation is reuired, but for debug information purposes...
        if (allocationSize > recipeCacheSize)
        {
            LOG_ERR(SYN_RCPE_CACHE,
                    "Requested allocation size (for recipe-ID: 0x{:x}, sub-type ID: {}) is bigger than cache size {}",
                    recipeId,
                    singleRequest.second.subTypeId,
                    allocationSize);
            return false;
        }

        totalAllocSizeRequested += allocationSize;
        if (singleRequest.first == ST_EXECUTION_BLOBS)
        {
            lastTypeFound = ST_EXECUTION_BLOBS;
        }
        else if (singleRequest.first == ST_PROGRAM_DATA)
        {
            if (lastTypeFound == ST_EXECUTION_BLOBS)
            {
                LOG_ERR(SYN_RCPE_CACHE, "Invalid order of allocations-request");
                return false;
            }
            lastTypeFound = ST_PROGRAM_DATA;
        }
        else
        {
            LOG_ERR(SYN_RCPE_CACHE, "Invalid sub-type in allocations-request");
            return false;
        }
    }

    if (totalAllocSizeRequested > recipeCacheSize)
    {
        LOG_ERR(SYN_RCPE_CACHE,
                "Allocation requested for 0x{:x} is bigger than cache size {}",
                recipeId,
                totalAllocSizeRequested);

        return false;
    }

    return true;
}

void RecipeCacheManager::eraseRecipeFromDb(uint64_t recipeId)
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    LOG_TRACE(SYN_RCPE_CACHE, "{}: Erasing recipe {}", HLLOG_FUNC, recipeId);

    // Recipe's Handles
    _deleteRecipeFromHandelsDB(recipeId);

    // Recipe's Allocations
    auto recipeInfoIter = m_recipesInfoTable.find(recipeId);
    if (recipeInfoIter == m_recipesInfoTable.end())
    {
        LOG_DEBUG(SYN_RCPE_CACHE, "{}: Recipe not found in recipe-info table", HLLOG_FUNC);
        return;
    }
    //
    RecipeCacheAllocationId allocationId;
    allocationId.first = recipeId;
    for (uint8_t currSubType = ST_FIRST; currSubType < ST_AMOUNT; currSubType++)
    {
        for (auto subTypeAllocationId : recipeInfoIter->second.subTypeAllocIds[currSubType])
        {
            allocationId.second = subTypeAllocationId;
            _invalidateAllocationL(allocationId);
        }
    }

    m_recipesInfoTable.erase(recipeId);
}

void RecipeCacheManager::getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress) const
{
    baseAddress = m_baseAddress;
    lastAddress = m_lastAddress;
}

void RecipeCacheManager::addSingleSubTypeAllocationRequestL(AllSubTypesAllocationRequests& allocationRequests,
                                                            SubType                        allocSubType,
                                                            uint64_t                       subTypeId,
                                                            uint64_t                       requestedSize)
{
    SingleSubTypeAllocationRequest request =
        std::make_pair(allocSubType, AllocationRequestInfo({.subTypeId = subTypeId, .size = requestedSize}));

    // Contiguous sub-types will be ordered in front of others
    if (allocSubType == ST_EXECUTION_BLOBS)
    {
        allocationRequests.push_back(request);
    }
    else
    {
        allocationRequests.push_front(request);
    }
}

// Remove allocation whether its in "not in use" or "in-use" list,
bool RecipeCacheManager::_deleteAllocationL(RecipeCacheAllocationId allocationId, bool shouldDeleteBlocks)
{
    auto recipeTableEntry = m_recipesSubTypeTable.find(allocationId);
    if (recipeTableEntry == m_recipesSubTypeTable.end())
    { // recipe is not in cache
        LOG_DEBUG(SYN_RCPE_CACHE, "{}: {} was not found in cache", HLLOG_FUNC, _getAllocationIdDescL(allocationId));
        return false;
    }

    // else - allocation is in cache => delete it
    RecipeSubTypeEntry& subTypeEntry = recipeTableEntry->second;
    SubType             entrySubType = subTypeEntry.entrySubType;

    std::list<RecipeCacheAllocationId>& LRUlist = _getRelavantRecipeListL(allocationId);

    if (shouldDeleteBlocks)
    {
        m_pBlockAllocator.freeL(subTypeEntry.blockList);
    }
    LRUlist.erase(subTypeEntry.iterator);
    m_recipesSubTypeTable.erase(allocationId);

    std::deque<uint64_t>& subTypeAllocIds = m_recipesInfoTable[allocationId.first].subTypeAllocIds[entrySubType];
    auto subTypeAlloIter = std::find(subTypeAllocIds.begin(), subTypeAllocIds.end(), allocationId.second);
    subTypeAllocIds.erase(subTypeAlloIter);

    _deleteAllocationIdFromHandelsDB(allocationId);

    LOG_TRACE(SYN_RCPE_CACHE, "{}: {} removed from cache", HLLOG_FUNC, _getAllocationIdDescL(allocationId));

    return true;
}

bool RecipeCacheManager::_isAllocationInUseL(RecipeCacheAllocationId allocationId)
{
    auto recipeSubTypeIter = m_recipesSubTypeTable.find(allocationId);
    if (recipeSubTypeIter == m_recipesSubTypeTable.end())
    {
        LOG_ERR(SYN_RCPE_CACHE, "{}: {} not found in cache", HLLOG_FUNC, _getAllocationIdDescL(allocationId));

        // TODO - fix this. It is on neither of the lists
        return true;
    }

    return (recipeSubTypeIter->second.refCount > 0) ? true : false;
}

std::list<RecipeCacheManager::RecipeCacheAllocationId>&
RecipeCacheManager::_getRelavantRecipeListL(RecipeCacheAllocationId allocationId)
{
    return _isAllocationInUseL(allocationId) ? m_inUseList : m_notInUseList;
}

bool RecipeCacheManager::_invalidateAllocationL(RecipeCacheAllocationId allocationId)
{
    auto recipeTableEntry = m_recipesSubTypeTable.find(allocationId);
    if (recipeTableEntry != m_recipesSubTypeTable.end())
    {
        RecipeSubTypeEntry& subTypeEntry = recipeTableEntry->second;
        if (subTypeEntry.refCount == 0)
        {
            return _deleteAllocationL(allocationId, true);
        }
        else
        {
            LOG_ERR(SYN_RCPE_CACHE,
                    "{}: allocation: {} has ref count ({}) different than 0!!",
                    HLLOG_FUNC,
                    _getAllocationIdDescL(allocationId),
                    subTypeEntry.refCount);
        }
    }
    else
    {
        LOG_DEBUG(SYN_RCPE_CACHE, "{}: {} was not found in cache", HLLOG_FUNC, _getAllocationIdDescL(allocationId));
    }

    return false;
}

// LRU invalidation
bool RecipeCacheManager::_freeMemoryBySizeL(uint64_t sizeToFree, bool& hasRecipeToFree)
{
    hasRecipeToFree = false;

    RecipeCacheAllocationId allocationId;

    uint64_t sizeOfNotInUseList = m_notInUseList.size();
    for (size_t i = 0; i < sizeOfNotInUseList; i++)
    {
        allocationId = m_notInUseList.back();

        auto recipeTableEntry = m_recipesSubTypeTable.find(allocationId);
        if (recipeTableEntry == m_recipesSubTypeTable.end())
        {
            // Although we found an invalid allocationId, we will keep searching the DB (no harm)
            LOG_DEBUG(SYN_RCPE_CACHE, "{}: {} was not found in cache", HLLOG_FUNC, _getAllocationIdDescL(allocationId));
            continue;
        }

        RecipeSubTypeEntry& subTypeEntry = recipeTableEntry->second;
        uint64_t            recipeSize   = subTypeEntry.blockList.size() * m_blockSize;

        _invalidateAllocationL(allocationId);
        hasRecipeToFree = true;
        if (recipeSize >= sizeToFree)
        {
            // we have invalidated enough recipes
            return true;
        }

        sizeToFree -= recipeSize;
    }

    LOG_DEBUG(SYN_RCPE_CACHE, "{}: Cant freeMemoryBySize", HLLOG_FUNC);
    return false;
}

std::string RecipeCacheManager::_getAllocationIdDescL(RecipeCacheAllocationId allocationId)
{
    return fmt::format("recipe-ID: 0x{:x}, sub-type ID: {}", allocationId.first, allocationId.second);
}

void RecipeCacheManager::_deleteAllocationIdFromHandelsDB(RecipeCacheAllocationId allocationId)
{
    auto handlesIter = m_allocationIdHandlesDB.find(allocationId);
    if (handlesIter == m_allocationIdHandlesDB.end())
    {
        LOG_DEBUG(SYN_RCPE_CACHE,
                  "{}: allocationId {} not found it handles DB",
                  HLLOG_FUNC,
                  _getAllocationIdDescL(allocationId));
        return;
    }

    m_allocationIdHandlesDB.erase(handlesIter);
}

void RecipeCacheManager::_deleteRecipeFromHandelsDB(uint64_t recipeId)
{
    auto recipeIter = m_recipesInfoTable.find(recipeId);
    if (recipeIter == m_recipesInfoTable.end())
    {
        return;
    }

    RecipeCacheAllocationId allocationId;
    allocationId.first = recipeId;
    for (uint8_t currSubType = ST_FIRST; currSubType < ST_AMOUNT; currSubType++)
    {
        for (auto subTypeAllocationId : recipeIter->second.subTypeAllocIds[currSubType])
        {
            allocationId.second = subTypeAllocationId;
            _deleteAllocationIdFromHandelsDB(allocationId);
        }
    }
}

bool RecipeCacheManager::_incrementAllocationIdHandleL(RecipeCacheAllocationId allocationId, uint64_t& newHandle)
{
    uint64_t handle = m_globalHandle;

    auto dbIterator = m_allocationIdHandlesDB.find(allocationId);
    if (dbIterator != m_allocationIdHandlesDB.end())
    {
        handle = dbIterator->second;
    }
    else
    {
        ++m_globalHandle;
    }

    if (m_pHalReader == nullptr)
    {
        LOG_ERR(SYN_RCPE_CACHE, "{}: recipe cache hal reader is null", HLLOG_FUNC);
        return false;
    }

    const uint64_t hbmArea    = m_pHalReader->getDRAMBaseAddr();
    const uint64_t hbmEndArea = hbmArea + m_pHalReader->getDRAMSizeInBytes();
    HB_ASSERT(hbmArea != INVALID_HANDLE_VALUE + 1, "Invalid handle value is set to start of HBM area");
    if (handle + 1 == hbmArea)
    {
        handle = hbmEndArea;
    }
    else if (handle + 1 != INVALID_HANDLE_VALUE)
    {
        handle++;
    }
    else
    {
        handle = INVALID_HANDLE_VALUE + 1;
    }

    m_allocationIdHandlesDB[allocationId] = handle;
    newHandle = handle;
    return true;
}

bool RecipeCacheManager::_getAllocationIdHandle(RecipeCacheAllocationId allocationId, uint64_t& handle)
{
    auto dbIterator = m_allocationIdHandlesDB.find(allocationId);
    if (dbIterator == m_allocationIdHandlesDB.end())
    {
        return false;
    }

    handle = dbIterator->second;
    return true;
}

RecipeCacheManager::BlockAllocatorResult RecipeCacheManager::_addNewAllocationL(RecipeCacheAllocationId allocationId,
                                                                                RecipeSubTypeEntry&     subTypeEntry,
                                                                                std::vector<uint64_t>&  blockAddresses,
                                                                                uint64_t&               handle,
                                                                                uint64_t                refCount,
                                                                                uint64_t                allocSize,
                                                                                SubType                 allocSubType)
{
    bool     isRecipeEntryExists = false;
    uint64_t recipeId            = allocationId.first;
    auto     recipeInfoIter      = m_recipesInfoTable.find(recipeId);
    if (recipeInfoIter != m_recipesInfoTable.end())
    {
        isRecipeEntryExists = true;
    }

    RecipeInfoEntry& recipeInfoEntry   = m_recipesInfoTable[recipeId];
    AllocationState& subTypeAllocState = recipeInfoEntry.allocState[allocSubType];

    bool isSubTypeUniqueAlloc = (subTypeAllocState   == AllocationState::UNIQUE);
    bool isCurrUniqueAlloc    = (allocationId.second <= PRG_DATA_LAST_UNIQUE_ALLOCATION_ID);
    if ((subTypeAllocState != AllocationState::NOT_SET) &&
        (isCurrUniqueAlloc != isSubTypeUniqueAlloc))
    {
        LOG_WARN(SYN_RCPE_CACHE,
                 "{}: Different allocation-state for allocationId ({})"
                 " and DB's sub-type {} (is-unique: alloc {} DB {})",
                 HLLOG_FUNC,
                 _getAllocationIdDescL(allocationId),
                 allocSubType,
                 isCurrUniqueAlloc,
                 isSubTypeUniqueAlloc);
        return BAR_FAILURE;
    }
    subTypeAllocState = isCurrUniqueAlloc ? AllocationState::UNIQUE : AllocationState::SINGULAR;

    // Increment allocation-id handle
    bool status = _incrementAllocationIdHandleL(allocationId, handle);
    if (!status)
    {
        if (!isRecipeEntryExists)
        {
            recipeInfoIter = m_recipesInfoTable.find(recipeId);
            m_recipesInfoTable.erase(recipeInfoIter);
        }

        m_pBlockAllocator.freeL(subTypeEntry.blockList);
        return BAR_FAILURE;
    }

    if ((recipeInfoEntry.subTypeAllocIds[allocSubType].size() == 0) ||
        (recipeInfoEntry.allocState[allocSubType] == AllocationState::UNIQUE))
    {
        recipeInfoEntry.subTypeAllocIds[allocSubType].push_back(allocationId.second);
    }

    m_inUseList.push_front(allocationId);

    subTypeEntry.entrySubType = allocSubType;
    subTypeEntry.iterator     = m_inUseList.begin();
    subTypeEntry.refCount     = refCount;

    blockAddresses.assign(subTypeEntry.blockList.begin(), subTypeEntry.blockList.end());

    LOG_TRACE(SYN_API, "Recipe cache - Blocks-list handle {}:", handle);
    uint64_t sizeLeft = allocSize;
    for (auto const& entry : subTypeEntry.blockList)
    {
        if (sizeLeft < m_blockSize)
        {
            LOG_TRACE(SYN_API, "Recipe-Cache: Block-address(partial) 0x{:x} Size {}", entry, sizeLeft);
            sizeLeft -= sizeLeft;
        }
        else
        {
            LOG_TRACE(SYN_API, "Recipe-Cache: Block-address(full) 0x{:x} Size {}", entry, m_blockSize);
            sizeLeft -= m_blockSize;
        }
    }

    LOG_TRACE(SYN_RCPE_CACHE,
              "{}: {} added to cache, alloc_size: {} handle: {} sub-type: {} refCount: {}",
              HLLOG_FUNC,
              _getAllocationIdDescL(allocationId),
              allocSize,
              handle,
              allocSubType,
              refCount);

    STAT_COLLECT_COND(1, allocSubType == ST_EXECUTION_BLOBS, allocatedNonPatchable, allocatedPrgData);

    return BAR_ALLOCATED;
}
