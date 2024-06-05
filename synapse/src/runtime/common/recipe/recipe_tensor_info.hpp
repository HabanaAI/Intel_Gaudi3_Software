#pragma once
#include <cstdint>
#include <unordered_map>
#include "synapse_common_types.h"
#include "recipe_patch_processor.hpp"
#include <unordered_set>
#include "types.h"

typedef std::pair<uint64_t, synTensorType> IdxAndType;

#pragma pack(push, 8)
struct RecipeTensorsInfo
{
    synStatus tensorRetrieveIds(uint64_t* tensorIds, const uint32_t numOfTensors) const;

    synStatus tensorRetrieveIds(const char** tensorsNames, uint64_t* tensorIds, const uint32_t numOfTensors) const;

    synStatus tensorRetrieveId(const char* tensorName, uint64_t* tensorId) const;

    uint32_t getTensorAmount() const;

    patching::SectionToSectionType            m_sectionToSectionType;
    SmallVector<SectionSizeTensor, 8>         m_sectionsInfo;

    std::unordered_set<uint64_t> m_constZeroSizeTensors;
    std::unordered_set<uint64_t> m_constZeroSizeSections;

    shape_plane_graph_t*                      m_shapePlanRecipe = nullptr;
    recipe_t*                                 m_recipe          = nullptr;

    bool     m_isTensorName2idxInit = false;
    uint64_t m_maxSectionId         = 0;
    uint64_t m_numSectionsToPatch   = 0;
};
#pragma pack(pop)