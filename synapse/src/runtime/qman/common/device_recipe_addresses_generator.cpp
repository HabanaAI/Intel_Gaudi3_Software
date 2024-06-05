#include "device_recipe_addresses_generator.hpp"
#include "defs.h"
#include "define_synapse_common.hpp"
#include "runtime/common/device/device_mem_alloc.hpp"
#include "habana_global_conf_runtime.h"
#include "math_utils.h"
#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "runtime/common/queues/queue_compute_utils.hpp"
#include "runtime/qman/common/recipe_cache_manager.hpp"
#include "runtime/qman/common/qman_types.hpp"

DeviceRecipeAddressesGenerator::DeviceRecipeAddressesGenerator(synDeviceType            devType,
                                                               DevMemoryAllocInterface& rDevMemAlloc)
: m_devType(devType), m_rDevMemAlloc(rDevMemAlloc), m_dramRecipeCacheBaseAddress(0)
{
}

DeviceRecipeAddressesGenerator::~DeviceRecipeAddressesGenerator() = default;

synStatus
DeviceRecipeAddressesGenerator::generateDeviceRecipeAddresses(const InternalRecipeHandle& rInternalRecipeHandle,
                                                              uint64_t                    workspaceAddress,
                                                              uint64_t                    refCount,
                                                              uint64_t                    prgDataSubTypeAllocId,
                                                              uint64_t&                   rRecipeId,
                                                              uint64_t&                   rProgramCodeDeviceAddress,
                                                              uint64_t&                   rProgramDataDeviceAddress,
                                                              bool&                       rProgramCodeInCache,
                                                              bool&                       rProgramDataInCache,
                                                              uint64_t&                   rProgramCodeHandle,
                                                              uint64_t&                   rProgramDataHandle,
                                                              std::vector<uint64_t>&      rExecutionBlocksAddresses)
{
    rProgramCodeInCache = false;
    rProgramDataInCache = false;

    rProgramDataHandle = INVALID_HANDLE_VALUE;
    rProgramCodeHandle = INVALID_HANDLE_VALUE;

    rProgramDataDeviceAddress = INVALID_DEVICE_ADDR;
    rProgramCodeDeviceAddress = INVALID_DEVICE_ADDR;

    // At the moment we use the recipe address and a unique handle (note, all DSD recipes use the same handle)
    const uint64_t recipeId = getRecipeId(rInternalRecipeHandle);

    do
    {
        if (m_pRecipeCacheManager)
        {
            bool success = _allocateRecipeMemoryInCache(m_pRecipeCacheManager.get(),
                                                        prgDataSubTypeAllocId,
                                                        rInternalRecipeHandle.basicRecipeHandle.recipe,
                                                        recipeId,
                                                        refCount,
                                                        rExecutionBlocksAddresses,
                                                        rProgramCodeDeviceAddress,
                                                        rProgramDataDeviceAddress,
                                                        rProgramCodeInCache,
                                                        rProgramDataInCache,
                                                        rProgramCodeHandle,
                                                        rProgramDataHandle);
            if (!success)
            {
                return synAllResourcesTaken;
            }
            rRecipeId = recipeId;
            return synSuccess;
        }
    } while (0);

    // allocate on WS
    rProgramCodeInCache = false;
    rProgramDataInCache = false;

    rProgramCodeDeviceAddress =
        getProgramCodeSectionAddress(workspaceAddress, *rInternalRecipeHandle.basicRecipeHandle.recipe);
    rProgramCodeHandle = rProgramCodeDeviceAddress;

    rProgramDataDeviceAddress =
        getProgramDataSectionAddress(workspaceAddress, *rInternalRecipeHandle.basicRecipeHandle.recipe);
    rProgramDataHandle = rProgramDataDeviceAddress;

    rRecipeId = recipeId;
    return synSuccess;
}

void DeviceRecipeAddressesGenerator::notifyDeviceRecipeAddressesAreNotUsed(uint64_t recipeId,
                                                                           uint64_t refCount,
                                                                           uint64_t prgDataSubTypeAllocId)
{
    bool status = false;
    if (m_pRecipeCacheManager)
    {
        if ((prgDataSubTypeAllocId == RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID) ||
            (prgDataSubTypeAllocId <= RecipeCacheManager::PRG_DATA_LAST_UNIQUE_ALLOCATION_ID))
        {
            status =
                m_pRecipeCacheManager->setRecipeNotInUse(std::make_pair(recipeId, prgDataSubTypeAllocId), refCount);
            if (!status)
            {
                LOG_WARN(SYN_RCPE_CACHE, "{} failed to set recipe {} program data as not in use", HLLOG_FUNC, recipeId);
            }
        }

        status = m_pRecipeCacheManager->setRecipeNotInUse(
            std::make_pair(recipeId, RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID),
            refCount);
        if (!status)
        {
            LOG_WARN(SYN_RCPE_CACHE, "{} failed to set recipe {} program code as not in use", HLLOG_FUNC, recipeId);
        }
    }
}

synStatus DeviceRecipeAddressesGenerator::getProgramDataAddress(const InternalRecipeHandle& rInternalRecipeHandle,
                                                                uint64_t&                   rProgramDataAddress) const
{
    rProgramDataAddress = INITIAL_WORKSPACE_ADDRESS;
    if (isRecipeCacheValid())
    {
        const uint64_t     recipeId              = getRecipeId(rInternalRecipeHandle);
        Settable<uint64_t> prgDataAddressInCache = getProgramDataAddressIfAlreadyInCache(recipeId);

        if (!prgDataAddressInCache.is_set())
        {
            LOG_ERR(SYN_API, "program data was not in cache");
            return synFail;
        }
        rProgramDataAddress = prgDataAddressInCache.value();
    }
    else
    {
        LOG_ERR(SYN_API, "cache is disabled although fallback to WS is disabled");
        return synFail;
    }
    return synSuccess;
}

synStatus DeviceRecipeAddressesGenerator::allocate()
{
    HB_ASSERT(m_pRecipeCacheManager == nullptr, "m_pRecipeCacheManager is not nullptr");

    uint64_t cache_size = GCFG_RECIPE_CACHE_SIZE.value() * 1024;
    // Allocate DRAM buffer
    void*       buffer = 0;
    std::string mappingDesc("Recipe Cache");
    if (m_rDevMemAlloc.allocateMemory(cache_size + MANDATORY_KERNEL_ALIGNMENT,
                                      synMemFlags::synMemDevice,
                                      &buffer,
                                      false,
                                      0,
                                      mappingDesc) != synSuccess)
    {
        LOG_ERR(SYN_DEVICE, "Failed to allocate Recipe Cache");
        return synFail;
    }
    m_dramRecipeCacheBaseAddress = (uint64_t)buffer;
    uint64_t alignedDramAddress  = round_to_multiple(m_dramRecipeCacheBaseAddress, MANDATORY_KERNEL_ALIGNMENT);
    m_pRecipeCacheManager.reset(new RecipeCacheManager(alignedDramAddress,
                                                       m_devType,
                                                       GCFG_RECIPE_CACHE_SIZE.value(),
                                                       GCFG_RECIPE_CACHE_BLOCK_SIZE.value()));

    return synSuccess;
}

synStatus DeviceRecipeAddressesGenerator::release()
{
    synStatus status = synSuccess;

    if (m_pRecipeCacheManager)
    {
        m_pRecipeCacheManager.reset();

        if (m_dramRecipeCacheBaseAddress != 0)
        {
            if (m_rDevMemAlloc.deallocateMemory((void*)m_dramRecipeCacheBaseAddress, false, 0) != synSuccess)
            {
                status = synFail;
            }
        }
    }

    return status;
};

synStatus DeviceRecipeAddressesGenerator::getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress) const
{
    if (m_pRecipeCacheManager == nullptr)
    {
        return synFail;
    }

    m_pRecipeCacheManager->getCacheDeviceAddressRange(baseAddress, lastAddress);
    return synSuccess;
}

Settable<uint64_t> DeviceRecipeAddressesGenerator::getProgramDataAddressIfAlreadyInCache(uint64_t recipeId) const
{
    return m_pRecipeCacheManager->getProgramDataAddressIfAlreadyInCache(recipeId);
}

void DeviceRecipeAddressesGenerator::notifyRecipeDestroy(const recipe_t* pRecipe)
{
    HB_ASSERT_PTR(pRecipe);

    if (m_pRecipeCacheManager != nullptr)
    {
        m_pRecipeCacheManager->eraseRecipeFromDb((uint64_t)pRecipe);
    }
}

uint64_t DeviceRecipeAddressesGenerator::getRecipeId(const InternalRecipeHandle& rInternalRecipeHandle)
{
    return (uint64_t)rInternalRecipeHandle.recipeSeqNum;
}

bool DeviceRecipeAddressesGenerator::_allocateRecipeMemoryInCache(
    RecipeCacheManager*    pRecipeCacheManager,
    uint64_t               prgDataSubTypeAllocId,
    const recipe_t*        pRecipe,
    uint64_t               recipeId,
    uint64_t               refCount,
    std::vector<uint64_t>& rExecutionBlocksDeviceAddresses,
    uint64_t&              rProgramCodeDeviceAddress,
    uint64_t&              rProgramDataDeviceAddress,
    bool&                  rProgramCodeInCache,
    bool&                  rProgramDataInCache,
    uint64_t&              rProgramCodeHandle,
    uint64_t&              rProgramDataHandle)
{
    uint64_t recipeSectionsSizes[RecipeCacheManager::ST_AMOUNT];
    recipeSectionsSizes[RecipeCacheManager::ST_PROGRAM_DATA] =
        pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA];
    recipeSectionsSizes[RecipeCacheManager::ST_EXECUTION_BLOBS] = pRecipe->execution_blobs_buffer_size;

    std::vector<uint64_t>              tempPrgDataBlocksVector;
    RecipeCacheManager::BlockPtrVector sectionsBlockLists[RecipeCacheManager::ST_AMOUNT];
    sectionsBlockLists[RecipeCacheManager::ST_PROGRAM_DATA]    = &tempPrgDataBlocksVector;
    sectionsBlockLists[RecipeCacheManager::ST_EXECUTION_BLOBS] = &rExecutionBlocksDeviceAddresses;

    uint64_t sectionsHandles[RecipeCacheManager::ST_AMOUNT];

    RecipeCacheManager::AllSubTypesAllocationRequests rcmAllocationRequests;
    RecipeCacheManager::addSingleSubTypeAllocationRequestL(
        rcmAllocationRequests,
        RecipeCacheManager::ST_EXECUTION_BLOBS,
        RecipeCacheManager::EXECUTION_PRG_CODE_RESERVED_ALLOCATION_ID,
        recipeSectionsSizes[RecipeCacheManager::ST_EXECUTION_BLOBS]);

    RecipeCacheManager::addSingleSubTypeAllocationRequestL(rcmAllocationRequests,
                                                           RecipeCacheManager::ST_PROGRAM_DATA,
                                                           prgDataSubTypeAllocId,
                                                           recipeSectionsSizes[RecipeCacheManager::ST_PROGRAM_DATA]);

    bool status = pRecipeCacheManager->acquireBlocks(recipeId,
                                                     sectionsBlockLists,
                                                     sectionsHandles,
                                                     refCount,
                                                     rcmAllocationRequests);
    if (status)
    {
        if (recipeSectionsSizes[RecipeCacheManager::ST_EXECUTION_BLOBS] > 0)
        {
            rProgramCodeDeviceAddress = rExecutionBlocksDeviceAddresses.front();
        }
        if (recipeSectionsSizes[RecipeCacheManager::ST_PROGRAM_DATA] > 0)
        {
            rProgramDataDeviceAddress = tempPrgDataBlocksVector.front();
        }

        rProgramCodeInCache = true;
        rProgramDataInCache = true;

        rProgramCodeHandle = sectionsHandles[RecipeCacheManager::ST_EXECUTION_BLOBS];
        rProgramDataHandle = sectionsHandles[RecipeCacheManager::ST_PROGRAM_DATA];
    }
    return status;
}
