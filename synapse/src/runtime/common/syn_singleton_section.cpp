#include "syn_singleton.hpp"

#include "defenders.h"
#include "recipe_allocator.h"
#include "section_handle.hpp"
#include "synapse_common_types.h"

#include "runtime/common/recipe/recipe_handle_impl.hpp"

bool synSingleton::_validateSectionForGraph(InternalSectionHandle* sectionHandle)
{
    if (!sectionHandle)
    {
        LOG_ERR(SYN_API, "{}: Empty section handle is invalid", HLLOG_FUNC);
        return false;
    }
    return m_graphEntries.validateGraphSection(sectionHandle);
}

synStatus synSingleton::_sectionLockAndSetID(InternalSectionHandle* sectionHandle)
{
    HB_ASSERT_PTR(sectionHandle);
    return m_graphEntries.sectionLockAndSetID(sectionHandle);
}

synStatus
synSingleton::sectionCreate(synSectionHandle* phSection, uint64_t sectionDescriptor, const synGraphHandle graphHandle)
{
    LOG_SINGLETON_API();

    if (phSection == nullptr)
    {
        LOG_ERR(SYN_API, "Can not create section handle with NULL phSection");
        return synFail;
    }

    uint32_t graphId;
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    if (synSuccess != m_graphEntries.getGraphId(graphHandle, graphId))
    {
        return synFail;
    }

    auto [handle, ptr] = m_sectionHndlSlopMap.insert(0, graphId);

    if (ptr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not create section handle");
        return synFail;
    }

    ptr->setSectionHandle((synSectionHandle)handle);
    *phSection = (synSectionHandle)handle;
    m_graphEntries.setSection(graphId, *phSection);

    return synSuccess;
}

synStatus synSingleton::sectionDestroy(synSectionHandle hSection)
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];
    if (sectionPtr == nullptr)
    {
        LOG_TRACE(SYN_API,
                  "{}: Section 0x{:x} not found, it was probably already destroyed",
                  HLLOG_FUNC,
                  (SMHandle)hSection);
        return synSuccess;
    }
    auto graphId = sectionPtr->getGraphID();
    // section deletion requires to release the ref count
    // held by the smart ptr wrapper, otherwise ConcurrentSlotMap erase
    // would fail.
    sectionPtr.reset();
    m_sectionHndlSlopMap.erase((SMHandle)hSection);
    // remove the section from the graph entry
    std::unique_lock<std::mutex> guard(m_graphsMutex);
    m_graphEntries.removeSection(graphId, hSection);
    return synSuccess;
}

synStatus synSingleton::sectionGroupSet(synSectionHandle hSection, uint64_t group) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not set group of an empty section handle {:x}", TO64(hSection));
        return synFail;
    }
    if (group > 255ull)
    {
        LOG_ERR(SYN_API, "Unsupported section group: {}, Allowed range: 0-255", group);
        return synUnsupported;
    }

    bool success = sectionPtr->setGroup(group);
    if (!success)
    {
        LOG_ERR(SYN_API, "Failed setting a group to section handle (section: 0x{:x})", TO64(hSection));
        return synFail;
    }

    return synSuccess;
}

synStatus synSingleton::sectionGroupGet(synSectionHandle hSection, uint64_t* group) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not get group of an empty section handle {:x}", TO64(hSection));
        return synFail;
    }
    VERIFY_IS_NULL_POINTER(SYN_API, group, "sectionGroup");

    *group = sectionPtr->getGroup();
    return synSuccess;
}

synStatus synSingleton::sectionPersistentSet(synSectionHandle hSection, bool isPersistent) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not set an empty section handle as persistent {:x}", TO64(hSection));
        return synFail;
    }

    bool success = sectionPtr->setPersistent(isPersistent);
    if (!success)
    {
        LOG_ERR(SYN_API, "Failed setting persistency to section handle (section: 0x{:x})", TO64(hSection));
        return synFail;
    }

    return synSuccess;
}

synStatus synSingleton::sectionPersistentGet(synSectionHandle hSection, bool* isPersistent) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not get an empty section handle's persistent state {:x}", TO64(hSection));
        return synFail;
    }
    VERIFY_IS_NULL_POINTER(SYN_API, isPersistent, "sectionIsPersistent");

    *isPersistent = sectionPtr->getPersistent();
    return synSuccess;
}

synStatus synSingleton::sectionConstSet(synSectionHandle hSection, bool isConst) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not set an empty section handle as const {:x}", TO64(hSection));
        return synFail;
    }

    bool success = sectionPtr->setConst(isConst);
    if (!success)
    {
        LOG_ERR(SYN_API, "Failed setting consistency to section handle (section: 0x{:x})", TO64(hSection));
        return synFail;
    }

    return synSuccess;
}

synStatus synSingleton::sectionConstGet(synSectionHandle hSection, bool* isConst) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not get an empty section handle's const state {:x}", TO64(hSection));
        return synFail;
    }
    VERIFY_IS_NULL_POINTER(SYN_API, isConst, "sectionIsConst");

    *isConst = sectionPtr->getConst();
    return synSuccess;
}

synStatus synSingleton::sectionGetProp(const synRecipeHandle  pRecipeHandle,
                                       const synSectionId     sectionId,
                                       const synSectionProp   prop,
                                       uint64_t*              propertyPtr) const
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, pRecipeHandle, "pRecipeHandle");

    recipe_t* recipe = pRecipeHandle->basicRecipeHandle.recipe;
    VERIFY_IS_NULL_POINTER(SYN_API, recipe, "internal recipe");

    const_section_t* constSections       = recipe->const_sections;
    uint32_t         constSectionsAmount = recipe->const_sections_nr;

    switch (prop)
    {
        case SECTION_SIZE:
            for (int i = 0; i < constSectionsAmount; i++, constSections++)
            {
                if (constSections->section_idx == sectionId)
                {
                    *propertyPtr = constSections->size;
                    return synSuccess;
                }
            }
            *propertyPtr = 0;
            return synFail;

        case SECTION_DATA:
            for (int i = 0; i < constSectionsAmount; i++, constSections++)
            {
                if (constSections->section_idx == sectionId)
                {
                    if (constSections->data == (char*)INVALID_CONST_SECTION_DATA)
                    {
                        LOG_WARN(SYN_API, "{}: Const-section data is not set (probably had been cleared)", HLLOG_FUNC);
                        return synInvalidArgument;
                    }

                    *propertyPtr = reinterpret_cast<uint64_t>(constSections->data);
                    return synSuccess;
                }
            }
            propertyPtr = nullptr;
            return synFail;

        case IS_CONST:
            *propertyPtr = false;
            for (int i = 0; i < constSectionsAmount; i++, constSections++)
            {
                if (constSections->section_idx == sectionId)
                {
                    *propertyPtr = true;
                    break;
                }
            }
            return synSuccess;

        default:
            LOG_ERR(SYN_API, "{} property is not a valid one", HLLOG_FUNC);
            return synFail;
    }
}

synStatus synSingleton::sectionRMWSet(synSectionHandle hSection, bool isRMW) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not set an empty section handle as RMW {:x}", TO64(hSection));
        return synFail;
    }

    bool success = sectionPtr->setRMW(isRMW);
    if (!success)
    {
        LOG_ERR(SYN_API, "Failed setting RMW to section handle (section: 0x{:x})", TO64(hSection));
        return synFail;
    }

    return synSuccess;
}

synStatus synSingleton::sectionRMWGet(synSectionHandle hSection, bool* isRMW) const
{
    LOG_SINGLETON_API();

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)hSection];

    if (sectionPtr == nullptr)
    {
        LOG_ERR(SYN_API, "Can not get an empty section handle's RMW state {:x}", TO64(hSection));
        return synFail;
    }
    VERIFY_IS_NULL_POINTER(SYN_API, isRMW, "sectionIsRMW");

    *isRMW = sectionPtr->getRMW();
    return synSuccess;
}

synStatus synSingleton::sectionSetDeviceAddress(synSectionHandle hSection, uint64_t deviceAddress) const
{
    HB_ASSERT(false, "sectionSetDeviceAddress not implemented. to be removed after complex guid refactoring");
    return synFail;
}

synStatus synSingleton::sectionGetDeviceAddress(synSectionHandle hSection, uint64_t* deviceAddress) const
{
    HB_ASSERT(false, "sectionGetDeviceAddress not implemented. to be removed after complex guid refactoring");
    return synFail;
}

synStatus synSingleton::sectionsClearHostBuffer( synRecipeHandle     recipeHandle,
                                                 const synSectionId* sectionIds,
                                                 size_t              numOfSections )
{
    LOG_SINGLETON_API();
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle,  "recipeHandle");
    VERIFY_IS_NULL_POINTER(SYN_API, sectionIds,    "sectionIds");

    synStatus status(synSuccess);

    RecipeAllocator* rAllocator          = recipeHandle->basicRecipeHandle.recipeAllocator;
    recipe_t*        recipe              = recipeHandle->basicRecipeHandle.recipe;
    const_section_t* constSections       = recipe->const_sections;
    uint32_t         constSectionsAmount = recipe->const_sections_nr;

    const synSectionId* pCurrSectionId = sectionIds;
    for (size_t i = 0; i < numOfSections; i++, pCurrSectionId++)
    {
        synSectionId     sectionId         = *pCurrSectionId;
        const_section_t* pCurrConstSection = constSections;
        bool             isSectionIdFound  = false;
        for (int i = 0; i < constSectionsAmount; i++, pCurrConstSection++)
        {
            if (pCurrConstSection->section_idx == sectionId)
            {
                isSectionIdFound = true;
                if ((uint64_t)pCurrConstSection->data == INVALID_CONST_SECTION_DATA)
                {
                    LOG_WARN(SYN_API, "{}: Invalid const-section data (probably had been cleared)", HLLOG_FUNC);
                    status = (status == synFail) ? synFail : synInvalidArgument;
                    continue;
                }

                if (!rAllocator->freeSingleEntry(pCurrConstSection->data))
                {
                    status = synFail;
                    continue;
                }

                pCurrConstSection->data = (char*)INVALID_CONST_SECTION_DATA;
            }
        }

        if (!isSectionIdFound)
        {
            LOG_WARN(SYN_API, "{}: Section-ID {} is not part of recipe's const-sections", HLLOG_FUNC, sectionId);
            status = synInvalidArgument;
        }
    }

    return status;
}

SlotMapItemSptr<InternalSectionHandle> getSectionPtrFromHandle(synSectionHandle handle)
{
    // We get to this function during synInit from eager code. At this point we don't have _SYN_SINGLETON_INTERNAL.
    // We need to intercept the case of handle==0
    if (handle == nullptr)
    {
        return {};
    }
    return _SYN_SINGLETON_INTERNAL->sectionHandleToPtr(handle);
}