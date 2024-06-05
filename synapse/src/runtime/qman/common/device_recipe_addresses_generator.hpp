#pragma once

#include "device_recipe_addresses_generator_interface.hpp"
#include <memory>
#include "settable.h"

class DevMemoryAllocInterface;
class RecipeCacheManager;

class DeviceRecipeAddressesGenerator : public DeviceRecipeAddressesGeneratorInterface
{
public:
    DeviceRecipeAddressesGenerator(synDeviceType devType, DevMemoryAllocInterface& rDevMemAlloc);

    virtual ~DeviceRecipeAddressesGenerator();

    // Decides if the sections will be in the recipe cache or in the WS
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
                                                    std::vector<uint64_t>&      rExecutionBlocksAddresses) override;

    virtual void
    notifyDeviceRecipeAddressesAreNotUsed(uint64_t recipeId, uint64_t refCount, uint64_t programDataSubTypeId) override;

    virtual synStatus getProgramDataAddress(const InternalRecipeHandle& rInternalRecipeHandle,
                                            uint64_t&                   rProgramDataAddress) const override;

    synStatus allocate();

    synStatus release();

    synStatus getCacheDeviceAddressRange(uint64_t& baseAddress, uint64_t& lastAddress) const;

    bool isRecipeCacheValid() const { return (m_pRecipeCacheManager != nullptr); }

    void notifyRecipeDestroy(const recipe_t* pRecipe);

private:
    static uint64_t getRecipeId(const InternalRecipeHandle& rInternalRecipeHandle);

    static bool _allocateRecipeMemoryInCache(RecipeCacheManager*    pRecipeCacheManager,
                                             uint64_t               prgDataSubTypeAllocId,
                                             const recipe_t*        pRecipe,
                                             uint64_t               recipeId,
                                             uint64_t               refCount,
                                             std::vector<uint64_t>& rRecipeBlocksAddr,
                                             uint64_t&              rProgramCodeDeviceAddress,
                                             uint64_t&              rProgramDataDeviceAddress,
                                             bool&                  rProgramCodeInCache,
                                             bool&                  rProgramDataInCache,
                                             uint64_t&              rProgramCodeHandle,
                                             uint64_t&              rProgramDataHandle);

    Settable<uint64_t> getProgramDataAddressIfAlreadyInCache(uint64_t recipeId) const;

    const synDeviceType m_devType;

    DevMemoryAllocInterface& m_rDevMemAlloc;

    std::unique_ptr<RecipeCacheManager> m_pRecipeCacheManager;

    uint64_t m_dramRecipeCacheBaseAddress;
};
