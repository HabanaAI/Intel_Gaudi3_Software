#pragma once

#include "runtime/qman/common/device_recipe_addresses_generator_interface.hpp"
#include <map>

class DeviceRecipeAddressesGeneratorMock : public DeviceRecipeAddressesGeneratorInterface
{
public:
    DeviceRecipeAddressesGeneratorMock();

    // Decides if the sections will be in the recipe cache or in the WS
    virtual synStatus generateDeviceRecipeAddresses(const InternalRecipeHandle& rInternalRecipeHandle,
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
                                                    std::vector<uint64_t>&      rExecutionBlocksAddresses) override;

    virtual void notifyDeviceRecipeAddressesAreNotUsed(uint64_t recipeId,
                                                       uint64_t refCount,
                                                       uint64_t prgDataSubTypeAllocId) override;

    virtual synStatus getProgramDataAddress(const InternalRecipeHandle& rInternalRecipeHandle,
                                            uint64_t&                   rProgramDataAddress) const override;

    inline bool isClean() const { return mRecipeIdsEntries.empty(); }

    uint64_t m_recipeIdGenerated;
    uint64_t m_recipeIdNotified;

    using RecipeIdEntry = std::pair<const recipe_t*, uint64_t>;
    std::map<uint64_t, RecipeIdEntry> mRecipeIdsEntries;
};