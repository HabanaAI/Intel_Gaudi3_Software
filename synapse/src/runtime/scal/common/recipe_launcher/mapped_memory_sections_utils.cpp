#include "mapped_memory_sections_utils.hpp"

#include "habana_global_conf_runtime.h"
#include "log_manager.h"

#include "memory_management/memory_protection.hpp"

#include "log_manager.h"
#include "runtime/scal/common/infra/scal_types.hpp"

void MappedMemorySectionsUtils::copyToDc(const MemoryMappedAddrVec& dst,
                                         uint8_t*                   src,
                                         uint64_t                   size,
                                         uint64_t                   offsetInDst,
                                         uint64_t                   dcSize)
{
    uint16_t currDc     = offsetInDst / dcSize;
    uint64_t offsetDc   = offsetInDst % dcSize;
    uint64_t leftToCpy  = size;
    uint64_t offsetSrc  = 0;
    bool     protectMem = (GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & PROTECT_MAPPED_MEM) == PROTECT_MAPPED_MEM;

    while (leftToCpy > 0)
    {
        uint64_t cpyNow = std::min(leftToCpy, dcSize - offsetDc);

        LOG_TRACE(SYN_PROG_DWNLD,
                       "leftToCpy {:x} currDc {} addr dc {:x}/{:x} offsetDc {:x} src {:x} offsetSrc {:x} cpyNow {:x}",
                       leftToCpy,
                       currDc,
                       TO64(dst[currDc].hostAddr),
                       dst[currDc].devAddr,
                       offsetDc,
                       TO64(src),
                       offsetSrc,
                       cpyNow);

        if (protectMem)
        {
            MemProtectUtils::memWrUnprotectPages(dst[currDc].hostAddr + offsetDc, cpyNow);
        }

        memcpy(dst[currDc].hostAddr + offsetDc, src + offsetSrc, cpyNow);

        if (protectMem)
        {
            MemProtectUtils::memWrProtectPages(dst[currDc].hostAddr + offsetDc, cpyNow);
        }

        offsetDc += cpyNow;
        if (offsetDc >= dcSize)
        {
            offsetDc -= dcSize;
            currDc++;
        }
        leftToCpy -= cpyNow;
        offsetSrc += cpyNow;
    }
}

void MappedMemorySectionsUtils::memcpySectionsToMapped(const RecipeSingleSectionVec& rRecipeSections,
                                                       int                           from,
                                                       int                           to,
                                                       const MemoryMappedAddrVec&    mappedAddr,
                                                       uint64_t                      dcSize)
{
    for (int i = from; i < to; i++)
    {
        uint64_t size = rRecipeSections[i].size;

        if (size == 0) continue;

        LOG_TRACE(SYN_PROG_DWNLD, "memcopy to mapped section {}", i);
        copyToDc(mappedAddr, rRecipeSections[i].recipeAddr, size, rRecipeSections[i].offsetMapped, dcSize);
    }
}

void MappedMemorySectionsUtils::memcpyBufferToPatchableMapped(uint64_t                   size,
                                                              const MemoryMappedAddrVec& patchableMappedAddr,
                                                              uint8_t*                   progDataHostBuffer,
                                                              uint32_t                   offsetMapped,
                                                              uint64_t                   dcSize)
{
    if (size == 0) return;

    LOG_TRACE(SYN_PROG_DWNLD,
                   "memcopy from host buffer address {:x} to patchable mapped offset {:x}, buffer size: {:x}",
                   TO64(progDataHostBuffer),
                   offsetMapped,
                   size);

    copyToDc(patchableMappedAddr, progDataHostBuffer, size, offsetMapped, dcSize);
}

void MappedMemorySectionsUtils::memcpyToMapped(const MemorySectionsScal&     rSections,
                                               const RecipeSingleSectionVec& rRecipeSections,
                                               bool                          isDsd,
                                               bool                          isIH2DRecipe)
{
    if (rSections.m_inMappedNoPatch == OUT)
    {
        memcpySectionsToMapped(rRecipeSections,
                               NON_PATCHABLE,
                               FIRST_IN_ARC,
                               rSections.m_nonPatchableMappedAddr,
                               rSections.m_nonPatchableDcSize);
        if (!isIH2DRecipe)
        {
            memcpySectionsToMapped(rRecipeSections,
                                   PROGRAM_DATA,
                                   NON_PATCHABLE,
                                   rSections.m_nonPatchableMappedAddr,
                                   rSections.m_nonPatchableDcSize);
        }
    }

    // In case of IH2D recipe, we want to copy the program data to the patchable mapped addr every time
    if (isIH2DRecipe)
    {
        memcpyBufferToPatchableMapped(rRecipeSections[PROGRAM_DATA].size,
                                      rSections.m_patchableMappedAddr,
                                      rSections.m_ih2dBuffer.get(),
                                      rRecipeSections[PROGRAM_DATA].offsetMapped,
                                      rSections.m_patchableDcSize);
    }

    if (rSections.m_inMappedNoPatch == OUT)
    {
        if (!isDsd)
        {
            memcpySectionsToMapped(rRecipeSections,
                                   DYNAMIC,
                                   DYNAMIC + 1,
                                   rSections.m_nonPatchableMappedAddr,
                                   rSections.m_nonPatchableDcSize);
        }

        memcpySectionsToMapped(rRecipeSections,
                               ECB_LIST_FIRST,
                               rRecipeSections.size(),
                               rSections.m_nonPatchableMappedAddr,
                               rSections.m_nonPatchableDcSize);
    }

    // in Dsd we need to copy the patchable buffers everytime
    if (rSections.m_inMappedPatch == OUT || isDsd)
    {
        memcpySectionsToMapped(rRecipeSections,
                               PATCHABLE,
                               PATCHABLE + 1,
                               rSections.m_patchableMappedAddr,
                               rSections.m_patchableDcSize);
        if (isDsd)
        {
            memcpySectionsToMapped(rRecipeSections,
                                   DYNAMIC,
                                   DYNAMIC + 1,
                                   rSections.m_patchableMappedAddr,
                                   rSections.m_patchableDcSize);
        }
    }
}
