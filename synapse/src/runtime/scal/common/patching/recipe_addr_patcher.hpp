#pragma once

#include "recipe.h"

#include "runtime/scal/common/recipe_launcher/mem_mgrs_types.hpp"
#include "types.h"

#include <cstdint>
#include <limits>
#include <unordered_map>

/**************************************************************************************/
/* The structure DataChunkPatchPoint holds information based on the patch points from */
/* the recipe. The patch point location is hold in terms of data-chunk/offset         */
/* instead of blobIdx/offset that is given by the recipe                              */
/**************************************************************************************/
static const uint32_t PP_TYPE_SIZE_BITS = 2;
static const uint32_t OFFSET_SIZE_BITS  = (32 - PP_TYPE_SIZE_BITS);

struct DataChunkPatchPoint
{
    // offset (in Bytes) to the patch-point location inside the data-chunk
    uint32_t offset_in_data_chunk : OFFSET_SIZE_BITS;
    uint32_t type : PP_TYPE_SIZE_BITS;

    uint8_t data_chunk_index;

    struct
    {
        uint16_t section_idx;
        uint64_t effective_address;
    } memory_patch_point;
};

struct UnpatchSingleRes
{
    uint64_t val       = 0;
    bool     lowValid  = false;
    bool     highValid = false;
};

/**************************************************************************************/
/* This class (RecipeAddrPatcher) is responsible for SCAL-architecture patching       */
/* An instance is created per recipe. It first translates the recipe patch points     */
/* database to something more efficient (int) and then can be called for the patching.*/
/* More details in the cpp file next to each function                                 */
/**************************************************************************************/
class RecipeAddrPatcher
{
public:
    RecipeAddrPatcher();
    bool init(const recipe_t& recipe, uint32_t dcSize);

    void patchAll(const uint64_t* sectionAddrDb, MemoryMappedAddrVec& dcAddr) const;

    bool verifySectionsAddrFromDc(const MemoryMappedAddrVec dcVec,
                                  uint32_t                  dcSize,
                                  const uint64_t*           expectedAddr,
                                  uint32_t                  expectedSize) const;

    uint64_t* getPatchPointEffectiveAddr(uint32_t patchPointIndex)
    {
        return &m_dcPp[patchPointIndex].memory_patch_point.effective_address;
    }

    void copyPatchPointDb(const RecipeAddrPatcher& recipeAddrPatcherToCopy);

private:
    static void setDcPp(DataChunkPatchPoint& dcPp, const patch_point_t& recipePp, const recipe_t& recipe, uint32_t dcSize);
    uint8_t*    getPpAddr(std::vector<patch_point_t>& sortedPp, uint32_t ppIdx) const;
    bool        getSectionsAddrFromDc(const MemoryMappedAddrVec&     dcVec,
                                   uint32_t                       dcSize,
                                   std::vector<UnpatchSingleRes>& sectionsAddr) const;  // return the sections addr
    void        unprotectDcs(MemoryMappedAddrVec& dcAddr) const;
    void        protectDcs(MemoryMappedAddrVec& dcAddr) const;

    using DcPpDb = SmallVector<DataChunkPatchPoint, 8>;

    DcPpDb          m_dcPp;
    const recipe_t* m_recipe;
    uint64_t        m_dcSize;
    bool            m_protectMem;
};
