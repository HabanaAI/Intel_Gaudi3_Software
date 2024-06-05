#pragma once

#include "mem_mgrs_types.hpp"

#include "runtime/scal/common/recipe_static_info_scal.hpp"

class MappedMemorySectionsUtils
{
public:
    static void
    memcpyToMapped(const MemorySectionsScal& rSections, const RecipeSingleSectionVec& rRecipeSections, bool isDsd, bool isIH2DRecipe);

private:
    static void memcpySectionsToMapped(const RecipeSingleSectionVec& rRecipeSections,
                                       int                           from,
                                       int                           to,
                                       const MemoryMappedAddrVec&    mappedAddr,
                                       uint64_t                      dcSize);

    static void memcpyBufferToPatchableMapped(uint64_t                   size,
                                              const MemoryMappedAddrVec& patchableMappedAddr,
                                              uint8_t*                   progDataHostBuffer,
                                              uint32_t                   offsetMapped,
                                              uint64_t                   dcSize);

    static void
    copyToDc(const MemoryMappedAddrVec& dst, uint8_t* src, uint64_t size, uint64_t offsetInDst, uint64_t dcSize);
};
