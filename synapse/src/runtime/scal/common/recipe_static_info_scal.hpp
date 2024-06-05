#pragma once

#include "synapse_common_types.h"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "runtime/scal/common/patching/recipe_addr_patcher.hpp"
#include "runtime/scal/common/patching/recipe_dsd_pp_info.hpp"
#include "types.h"

enum SectionType
{
    PATCHABLE,      // GLB HBM
    PROGRAM_DATA,   // GLB HBM
    NON_PATCHABLE,  // GLB HBM
    DYNAMIC,        // ARC HBM
    FIRST_IN_ARC = DYNAMIC,
    ECB_LIST_FIRST  // ARC HBM
};

inline const char* ScalSectionTypeToString(SectionType sec)
{
    switch (sec)
    {
        case PATCHABLE:
            return "PATCHABLE";
        case PROGRAM_DATA:
            return "PROGRAM_DATA";
        case NON_PATCHABLE:
            return "NON_PATCHABLE";
        case DYNAMIC:
            return "DYNAMIC";
        default:
            return "ECB_LIST";
    }
}

struct RecipeSingleSection
{
    uint8_t* recipeAddr   = nullptr;
    uint64_t size         = 0;
    uint32_t align        = 0;
    uint32_t offsetMapped = 0;
    uint32_t offsetHbm    = 0;
};

typedef SmallVector<RecipeSingleSection, 5 * 2 + ECB_LIST_FIRST> RecipeSingleSectionVec;

struct RecipeStaticInfoScal
{
    RecipeSingleSectionVec recipeSections;
    RecipeAddrPatcher      recipeAddrPatcher;
    RecipeDsdPpInfo        recipeDsdPpInfo;

    uint64_t m_mappedSizeNoPatch = 0;  // memory needed on mapped memory without patch section
    uint64_t m_mappedSizePatch   = 0;  // memory needed on mapped memory for patchable section(s)
    uint64_t m_arcHbmSize        = 0;  // memory needed on arcHbm
    uint64_t m_glbHbmSizeTotal   = 0;  // memory needed on glbHbm, including patch section
    uint64_t m_glbHbmSizeNoPatch = 0;  // memory needed on glbHbm without patch section
    uint64_t m_ecbListsTotalSize = 0;  // memory needed for all ebc list sections.

    EngineGrpArr m_computeEngineGrpArr {}; // used only for "update recipe base" command - doesn't include CME
};
