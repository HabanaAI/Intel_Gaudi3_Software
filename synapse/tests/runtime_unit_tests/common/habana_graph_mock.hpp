#pragma once

#include "habana_graph.h"
#include <synapse_common_types.h>
#include <vector>

class MemoryAllocatorMock : public MemoryAllocator
{
public:
    MemoryAllocatorMock()  = default;
    ~MemoryAllocatorMock() = default;
};

class HabanaGraphMock : public HabanaGraph
{
public:
    HabanaGraphMock(synDeviceType deviceType, const std::vector<uint64_t>& workspaceSizes);
    virtual ~HabanaGraphMock() = default;
    virtual recipe_t*                    serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    virtual HabanaGraphPtr               clone(bool cloneAllocators = false, bool keepMappings = false) const override
    {
        return nullptr;
    }
    virtual bool                         isPersistentTensor(const pTensor& tensor) const override { return false; }
    virtual bool                         isUserManagedDram(const pTensor& tensor) const override { return false; }
    virtual synDeviceType                getDeviceType() const override { return synDeviceTypeInvalid; }
    virtual HabanaGraphPtr               createEmptyGraph() const override { return nullptr; }
    virtual bool                         compile() override;

private:
    void initializeRecipe(RecipeAllocator& rRecipeAlloc, recipe_t& rRecipe) const;

    synDeviceType         m_deviceType;
    MemoryAllocatorMock* mpMemoryAllocator;
    std::vector<uint64_t> m_workspaceSizes;
};
