#pragma once

#include "recipe.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

struct basicRecipeInfo;

typedef enum
{
    RECIPE_TYPE_NORMAL,
    RECIPE_TYPE_DSD,
    RECIPE_TYPE_DSD_AND_IH2D
} DummyRecipeType;

class TestDummyRecipe
{
public:
    TestDummyRecipe(DummyRecipeType recipeType      = RECIPE_TYPE_NORMAL,
                    uint64_t        patchingSize    = 0x2000,
                    uint64_t        execBlobsSize   = 0x1000,
                    uint64_t        dynamicSize     = 0x3000,
                    uint64_t        programDataSize = 0x2800,
                    uint64_t        ecbListSize     = 0x800,
                    size_t          numberOfSobPps  = 0,
                    synDeviceType   deviceType      = synDeviceGaudi2);

    ~TestDummyRecipe();

    recipe_t*             getRecipe() { return &m_recipe; }
    basicRecipeInfo*      getBasicRecipeInfo() { return &m_internalRecipeHandle.basicRecipeHandle; }
    uint64_t              getRecipeSeqId() { return m_seqId; }
    InternalRecipeHandle* getInternalRecipeHandle() { return &m_internalRecipeHandle; }
    bool                  isDsd() { return m_isDsd; }
    bool                  isIH2DRecipe() { return m_isIH2DRecipe; }

    static const uint32_t BLOB_SIZE        = 0x1000;
    const uint64_t        wsScratchpadSize = 0x2200;
    const uint64_t        wsPrgDataSize    = 0x3400;

    std::vector<patch_point_t>         m_patchPoints;
    std::vector<persist_tensor_info_t> m_tensors;
    std::vector<tensor_info_t>         m_dsdTensors;
    std::vector<uint64_t>              m_wsSizes;
    std::vector<gc_conf_t>             m_gc_conf;

    using SingleBuff64 = std::vector<uint64_t>;
    std::vector<SingleBuff64> m_ecbBuffers;

    void        createValidEcbLists();
    static void createValidEcbLists(recipe_t* recipe);

private:
    recipe_t             m_recipe;
    uint64_t             m_seqId;
    InternalRecipeHandle m_internalRecipeHandle;
    bool                 m_isDsd;
    bool                 m_isIH2DRecipe;

    static uint64_t m_s_runningId;

    void fillBuff(SingleBuff64& buff, uint64_t val)
    {
        for (int i = 0; i < buff.size(); i++)
        {
            (buff)[i] = (m_seqId << 48) + (val << 32) + i + 1;
        }
    }
    static void createSingleValidEcbList(ecb_t list);
    template<typename Buff_t>
    void createBuff(Buff_t** buff, uint64_t sizeGivenInBytes, uint64_t val);
};

template<typename Buff_t>
void TestDummyRecipe::createBuff(Buff_t** buff, uint64_t sizeGivenInBytes, uint64_t val)
{
    uint32_t sizeElements = sizeGivenInBytes / sizeof(Buff_t);

    *buff = new Buff_t[sizeElements];

    for (int i = 0; i < sizeElements; i++)
    {
        (*buff)[i] = (m_seqId << 48) + (val << 32) + i + 1;
    }
}
