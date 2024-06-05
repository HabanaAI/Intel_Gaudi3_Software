#pragma once

#include "synapse_common_types.h"
#include <vector>

struct InternalRecipeHandle;
struct recipe_t;

class DeviceRecipeAddressesGeneratorInterface
{
public:
    virtual ~DeviceRecipeAddressesGeneratorInterface() = default;

    // Note that generateDeviceRecipeAddresses and notifyDeviceRecipeAddressesAreNotUsed are paired methods
    // which means that once the user is done working with the generated addresses, he has to call
    // notifyDeviceRecipeAddressesAreNotUsed
    virtual synStatus generateDeviceRecipeAddresses(const InternalRecipeHandle& rInternalRecipeHandle,
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
                                                    std::vector<uint64_t>&      rExecutionBlocksAddresses) = 0;

    // Todo remove isProgramData since they are allocated and released as together
    virtual void
    notifyDeviceRecipeAddressesAreNotUsed(uint64_t recipeId, uint64_t refCount, uint64_t prgDataSubTypeAllocId) = 0;

    virtual synStatus getProgramDataAddress(const InternalRecipeHandle& rInternalRecipeHandle,
                                            uint64_t&                   rProgramDataAddress) const = 0;

protected:
    static uint64_t getProgramCodeSectionAddress(uint64_t workspaceAddress, const recipe_t& rRecipe);
    static uint64_t getProgramDataSectionAddress(uint64_t workspaceAddress, const recipe_t& rRecipe);
};
