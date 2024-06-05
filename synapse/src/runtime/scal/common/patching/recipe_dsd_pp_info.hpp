#pragma once
#include "runtime/common/recipe/patching/define.hpp"

struct RecipeStaticInfoScal;
struct basicRecipeInfo;

class RecipeDsdPpInfo
{
public:
    bool
    init(const basicRecipeInfo& rBasicRecipeInfo, const RecipeStaticInfoScal& rRecipeStaticInfoScal, uint32_t dcSize);

    const DataChunkSmPatchPointsInfo& getDsdDCPatchingInfo() const { return m_pSmPatchPointsDataChunksInfo; }

private:
    void setDsdDcPp(data_chunk_sm_patch_point_t& dcPp,
                    sm_patch_point_t&            recipePp,
                    const recipe_t&              recipe,
                    uint32_t                     dcSize,
                    const RecipeStaticInfoScal&  rRecipeStaticInfoScal);

    DataChunkSmPatchPointsInfo m_pSmPatchPointsDataChunksInfo;
};
