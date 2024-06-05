#include "device_recipe_addresses_generator_mock.hpp"
#include <gtest/gtest.h>
#include "runtime/common/recipe/recipe_handle_impl.hpp"

bool programCodeInCache = true;
bool programDataInCache = true;

DeviceRecipeAddressesGeneratorMock::DeviceRecipeAddressesGeneratorMock() : m_recipeIdGenerated(0), m_recipeIdNotified(0)
{
}

synStatus
DeviceRecipeAddressesGeneratorMock::generateDeviceRecipeAddresses(const InternalRecipeHandle& rInternalRecipeHandle,
                                                                  uint64_t                    workspaceAddress,
                                                                  uint64_t                    refCount,
                                                                  uint64_t                    launchSeqId,
                                                                  uint64_t&                   rRecipeId,
                                                                  uint64_t&                   rProgramCodeDeviceAddress,
                                                                  uint64_t&                   rProgramDataDeviceAddress,
                                                                  bool&                       rProgramCodeInCache,
                                                                  bool&                       rProgramDataInCache,
                                                                  uint64_t&                   rProgramCodeHandle,
                                                                  uint64_t&                   rProgramDataHandle,
                                                                  std::vector<uint64_t>&      rExecutionBlocksAddresses)
{
    std::map<uint64_t, std::pair<const recipe_t*, uint64_t>>::iterator iter;
    for (iter = mRecipeIdsEntries.begin(); iter != mRecipeIdsEntries.end(); ++iter)
    {
        if (iter->second.first == rInternalRecipeHandle.basicRecipeHandle.recipe)
        {
            break;
        }
    }

    if (iter != mRecipeIdsEntries.end())
    {
        uint64_t& rReferenceCount = iter->second.second;
        rReferenceCount += refCount;
        rRecipeId = iter->first;
    }
    else
    {
        m_recipeIdGenerated++;
        rRecipeId = m_recipeIdGenerated;
        mRecipeIdsEntries.insert({rRecipeId, {rInternalRecipeHandle.basicRecipeHandle.recipe, refCount}});
    }

    if (programCodeInCache)
    {
        rProgramCodeInCache = true;
        rProgramCodeHandle = rProgramCodeDeviceAddress = 0x1234;
    }
    else
    {
        rProgramCodeInCache = false;
        rProgramCodeDeviceAddress =
            getProgramCodeSectionAddress(workspaceAddress, *rInternalRecipeHandle.basicRecipeHandle.recipe);
        rProgramCodeHandle = rProgramCodeDeviceAddress;
    }

    if (programDataInCache)
    {
        rProgramDataInCache = true;
        rProgramDataHandle = rProgramDataDeviceAddress = 0x5678;
    }
    else
    {
        rProgramDataInCache = false;
        rProgramDataDeviceAddress =
            getProgramDataSectionAddress(workspaceAddress, *rInternalRecipeHandle.basicRecipeHandle.recipe);
        rProgramDataHandle = rProgramDataDeviceAddress;
    }

    return synSuccess;
}

void DeviceRecipeAddressesGeneratorMock::notifyDeviceRecipeAddressesAreNotUsed(uint64_t recipeId,
                                                                               uint64_t refCount,
                                                                               uint64_t prgDataSubTypeAllocId)
{
    std::map<uint64_t, std::pair<const recipe_t*, uint64_t>>::iterator iter = mRecipeIdsEntries.find(recipeId);
    if (iter == mRecipeIdsEntries.end())
    {
        ASSERT_TRUE(false) << "invalid recipe ID " << recipeId;
        return;
    }

    uint64_t& rReferenceCount = iter->second.second;

    ASSERT_TRUE(rReferenceCount >= refCount) << "rReferenceCount " << rReferenceCount << "refCount " << refCount;
    rReferenceCount -= refCount;

    if (rReferenceCount == 0)
    {
        mRecipeIdsEntries.erase(iter);
    }

    m_recipeIdNotified = recipeId;
}

synStatus DeviceRecipeAddressesGeneratorMock::getProgramDataAddress(const InternalRecipeHandle& rInternalRecipeHandle,
                                                                    uint64_t& rProgramDataAddress) const
{
    return synSuccess;
}