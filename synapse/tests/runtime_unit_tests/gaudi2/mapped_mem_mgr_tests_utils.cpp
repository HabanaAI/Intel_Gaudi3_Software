#include "mapped_mem_mgr_tests_utils.hpp"
#include "log_manager.h"
#include "runtime/scal/common/recipe_launcher/mem_mgrs_types.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "log_manager.h"
#include <cstdint>

bool MappedMemMgrTestUtils::testingCompareWithRecipePatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                              uint64_t                      id,
                                                              const MemorySectionsScal&     rSections)
{
    bool rtn = true;

    uint64_t size    = rRecipeSections[PATCHABLE].size;
    uint32_t startDc = 0;
    uint32_t lastDc  = (size - 1) / rSections.m_patchableDcSize;

    for (uint32_t dc = startDc; dc <= lastDc; dc++)
    {
        uint64_t* dcAddr = (uint64_t*)rSections.m_patchableMappedAddr[dc].hostAddr;

        LOG_TRACE(SYN_API, "Patchable compare: dc {:x} addr {:x} id {:x}", dc, TO64(dcAddr), id);

        uint32_t lastByte = (dc < lastDc) ? rSections.m_patchableDcSize : size % rSections.m_patchableDcSize;

        for (int i = 0; i < lastByte / sizeof(uint64_t); i++)
        {
            if (dcAddr[i] != id)
            {
                if (rtn == true)  // first error
                {
                    LOG_ERR(SYN_API,
                                 "Patchable compare failed. Expected {:x} actual {:x} for DC {:x} i {:x}",
                                 id,
                                 TO64(dcAddr[i]),
                                 dc,
                                 i);
                    rtn = false;
                }
            }
        }
    }
    return rtn;
}

bool MappedMemMgrTestUtils::testingCompareWithRecipeDynamicPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                                     uint64_t                      id,
                                                                     const MemorySectionsScal&     rSections)
{
    bool rtn = true;

    uint64_t size    = rRecipeSections[DYNAMIC].size;
    uint32_t startDc = (rRecipeSections[DYNAMIC].offsetMapped / rSections.m_patchableDcSize);
    uint32_t lastDc  = (rRecipeSections[DYNAMIC].offsetMapped + size - 1) / rSections.m_patchableDcSize;

    for (uint32_t dc = startDc; dc <= lastDc; dc++)
    {
        uint64_t* dcAddr    = (uint64_t*)rSections.m_patchableMappedAddr[dc].hostAddr;
        uint32_t  firstByte = (dc == startDc) ? rRecipeSections[DYNAMIC].offsetMapped % rSections.m_patchableDcSize : 0;

        LOG_TRACE(SYN_API, "Dynamic Patchable compare: dc {:x} addr {:x} id {:x}", dc, TO64(dcAddr), id);

        uint32_t lastByte = (dc < lastDc) ? rSections.m_patchableDcSize : size % rSections.m_patchableDcSize;

        LOG_TRACE(SYN_API, "Dynamic Patchable compare: firstByte {:x} lastByte {:x}", firstByte, lastByte);

        for (int i = firstByte; i < lastByte / sizeof(uint64_t); i++)
        {
            if (dcAddr[i] != id)
            {
                if (rtn == true)  // first error
                {
                    LOG_ERR(SYN_API,
                                 "Dynamic Patchable compare failed. Expected {:x} actual {:x} for DC {:x} i {:x}",
                                 id,
                                 TO64(dcAddr[i]),
                                 dc,
                                 i);
                    rtn = false;
                }
            }
        }
    }
    return rtn;
}

bool MappedMemMgrTestUtils::testingCompareWithRecipeNonPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                                 int                           sectionId,
                                                                 const MemorySectionsScal&     rSections)
{
    bool rtn = true;

    uint64_t size    = rRecipeSections[sectionId].size;
    uint32_t startDc = rRecipeSections[sectionId].offsetMapped / rSections.m_nonPatchableDcSize;
    uint32_t lastDc  = (rRecipeSections[sectionId].offsetMapped + size - 1) / rSections.m_nonPatchableDcSize;

    uint32_t checkedUntilNow = 0;
    for (uint32_t dc = startDc; dc <= lastDc; dc++)
    {
        uint32_t sizeToChk = rSections.m_nonPatchableDcSize;
        uint32_t firstByte = 0;

        if (dc == startDc)
        {
            sizeToChk -= rRecipeSections[sectionId].offsetMapped % rSections.m_nonPatchableDcSize;
            firstByte = rRecipeSections[sectionId].offsetMapped % rSections.m_nonPatchableDcSize;
        }
        if (dc == lastDc)
        {
            uint32_t lastByte = (rRecipeSections[sectionId].offsetMapped + size) % rSections.m_nonPatchableDcSize;
            if (lastByte == 0) lastByte = rSections.m_nonPatchableDcSize;
            sizeToChk -= (rSections.m_nonPatchableDcSize - lastByte);
        }

        uint8_t* dcAddr = rSections.m_nonPatchableMappedAddr[dc].hostAddr;

        LOG_TRACE(SYN_PROG_DWNLD,
                       "section {} compare: offsetMapped {:x} dc {:x} dcAddr {:x} first Byte {:x} size {:x} startDc "
                       "{:x} lastDc {:x}"
                       " checkedUntilNow {:x}",
                       sectionId,
                       rRecipeSections[sectionId].offsetMapped,
                       dc,
                       TO64(dcAddr),
                       firstByte,
                       sizeToChk,
                       startDc,
                       lastDc,
                       checkedUntilNow);

        uint8_t* addrInRecipe = rRecipeSections[sectionId].recipeAddr + checkedUntilNow;
        int      res          = memcmp(dcAddr + firstByte, addrInRecipe, sizeToChk);
        LOG_TRACE(SYN_PROG_DWNLD,
                       "memcmp res {} sectionId {} dc {:x} of {:x}->{:x} size {:x}",
                       res,
                       sectionId,
                       dc,
                       TO64(dcAddr + firstByte),
                       TO64(addrInRecipe),
                       sizeToChk);
        if (res != 0)
        {
            rtn = false;
            LOG_ERR(SYN_PROG_DWNLD, "the above compare gave bad result");

            for (int i = 0; i < std::min(sizeToChk / sizeof(uint64_t), 8UL); i++)
            {
                uint64_t* addr       = (uint64_t*)(dcAddr + firstByte);
                uint64_t* addrRecipe = (uint64_t*)(addrInRecipe);

                LOG_TRACE(SYN_API,
                               "mismatch section {} expected / actual {:x}/{:x}",
                               sectionId,
                               addrRecipe[i],
                               addr[i]);
            }
        }
        checkedUntilNow += sizeToChk;
    }
    return rtn;
}

bool MappedMemMgrTestUtils::testingCompareWithRecipeSimulatedPatch(const RecipeSingleSectionVec& rRecipeSections,
                                                                   uint64_t                      id,
                                                                   const MemorySectionsScal&     rSections,
                                                                   bool                          isDsd,
                                                                   bool                          isIH2DRecipe)
{
    bool rtn = true;

    // all sections except patchable
    for (int i = 0; i < DYNAMIC; i++) // Dynamic is handled after the loop
    {
        if (i == PATCHABLE || (i == PROGRAM_DATA && isIH2DRecipe))
        {
            bool res = testingCompareWithRecipePatchable(rRecipeSections, id, rSections);
            if (res != true)
            {
                rtn = false;
            }
        }
        else
        {
            bool res = testingCompareWithRecipeNonPatchable(rRecipeSections, i, rSections);
            if (res != true)
            {
                rtn = false;
            }
        }
    }

    if (isDsd)
    {
        bool res = testingCompareWithRecipeDynamicPatchable(rRecipeSections, id, rSections);
        if (res != true)
        {
            rtn = false;
        }
    }
    else
    {
        bool res = testingCompareWithRecipeNonPatchable(rRecipeSections, DYNAMIC, rSections);
        if (res != true)
        {
            rtn = false;
        }
    }

    for (int i = FIRST_IN_ARC + 1; i < rRecipeSections.size(); i++)
    {
        bool res = testingCompareWithRecipeNonPatchable(rRecipeSections, i, rSections);
        if (res != true)
        {
            rtn = false;
        }
    }

    return rtn;
}

void MappedMemMgrTestUtils::testingFillMappedPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                       uint64_t                      id,
                                                       const MemorySectionsScal&     rSections)
{
    uint64_t size    = rRecipeSections[PATCHABLE].size;
    uint32_t startDc = 0;
    uint32_t lastDc  = (size - 1) / rSections.m_patchableDcSize;

    for (uint32_t dc = startDc; dc <= lastDc; dc++)
    {
        uint64_t* dcAddr = (uint64_t*)rSections.m_patchableMappedAddr[dc].hostAddr;

        uint32_t lastByte = (dc < lastDc) ? rSections.m_patchableDcSize : size % rSections.m_patchableDcSize;

        if (lastByte == 0) lastByte = rSections.m_patchableDcSize;

        LOG_TRACE(SYN_API,
                       "filling patchable addr: dc {:x} addr {:x} lastByte {:x} id {:x}",
                       dc,
                       TO64(dcAddr),
                       lastByte,
                       id);

        for (int i = 0; i < lastByte / sizeof(uint64_t); i++)
        {
            dcAddr[i] = id;
        }
    }
}

void MappedMemMgrTestUtils::testingFillMappedDsdPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                          uint64_t                      id,
                                                          const MemorySectionsScal&     rSections)
{
    uint64_t size    = rRecipeSections[DYNAMIC].size;
    uint32_t startDc = (rRecipeSections[DYNAMIC].offsetMapped / rSections.m_patchableDcSize);
    uint32_t lastDc  = (rRecipeSections[DYNAMIC].offsetMapped + size - 1) / rSections.m_patchableDcSize;

    for (uint32_t dc = startDc; dc <= lastDc; dc++)
    {
        uint32_t  firstByte = (dc == startDc) ? rRecipeSections[DYNAMIC].offsetMapped % rSections.m_patchableDcSize : 0;
        uint64_t* dcAddr    = (uint64_t*)(rSections.m_patchableMappedAddr[dc].hostAddr);

        uint32_t lastByte = (dc < lastDc) ? rSections.m_patchableDcSize : size % rSections.m_patchableDcSize;

        if (lastByte == 0) lastByte = rSections.m_patchableDcSize;

        LOG_TRACE(SYN_API,
                       "filling patchable addr: dc {:x} addr {:x} lastByte {:x} id {:x} firstByte {:x}",
                       dc,
                       TO64(dcAddr),
                       lastByte,
                       id,
                       firstByte);

        for (int i = firstByte; i < lastByte / sizeof(uint64_t); i++)
        {
            dcAddr[i] = id;
        }
    }
}

bool MappedMemMgrTestUtils::testingVerifyMappedToDevOffset(uint64_t offset, const MemorySectionsScal& rSections)
{
    if (rSections.m_inMappedPatch != BUSY)
    {
        for (int i = 0; i < rSections.m_patchableMappedAddr.size(); i++)
        {
            if ((uint64_t)rSections.m_patchableMappedAddr[i].hostAddr - rSections.m_patchableMappedAddr[i].devAddr !=
                offset)
            {
                LOG_ERR(SYN_API,
                        "patch offset error for segment {:x} {:x}-{:x} != {:x}",
                        i,
                        TO64(rSections.m_patchableMappedAddr[i].hostAddr),
                        rSections.m_patchableMappedAddr[i].devAddr,
                        offset);
                return false;
            }
        }
    }
    if (rSections.m_inMappedNoPatch != BUSY)
    {
        for (int i = 0; i < rSections.m_nonPatchableMappedAddr.size(); i++)
        {
            if ((uint64_t)rSections.m_nonPatchableMappedAddr[i].hostAddr -
                    rSections.m_nonPatchableMappedAddr[i].devAddr !=
                offset)
            {
                LOG_ERR(SYN_API,
                        "patch offset error for segment {:x} {:x}-{:x} != {:x}",
                        i,
                        TO64(rSections.m_nonPatchableMappedAddr[i].hostAddr),
                        rSections.m_nonPatchableMappedAddr[i].devAddr,
                        offset);
                return false;
            }
        }
    }
    return true;
}

void MappedMemMgrTestUtils::testingCopyToArc(const RecipeSingleSectionVec& rRecipeSections,
                                             const MemorySectionsScal&     rSections)
{
    for (int i = FIRST_IN_ARC; i < rRecipeSections.size(); i++)
    {
        LOG_TRACE(SYN_API,
                       "testing arc memcpy {:x} {:x}<-{:x} size {:x}",
                       i,
                       rSections.m_arcHbmAddr + rRecipeSections[i].offsetHbm,
                       TO64(rRecipeSections[i].recipeAddr),
                       rRecipeSections[i].size);

        memcpy((uint8_t*)rSections.m_arcHbmAddr + rRecipeSections[i].offsetHbm,
               rRecipeSections[i].recipeAddr,
               rRecipeSections[i].size);
    }
}

bool MappedMemMgrTestUtils::testingCheckArcSections(const RecipeSingleSectionVec& rRecipeSections,
                                                    uint64_t                      offset,
                                                    const MemorySectionsScal&     rSections)
{
    bool same = true;
    for (int i = FIRST_IN_ARC; i < rRecipeSections.size(); i++)
    {
        LOG_TRACE(SYN_API,
                       "testing arc memcmp {:x},{:x} size {:x} offset {:x}",
                       rSections.m_arcHbmAddr + rRecipeSections[i].offsetHbm,
                       TO64(rRecipeSections[i].recipeAddr),
                       rRecipeSections[i].size,
                       offset);
        bool res = memcmp((uint8_t*)rSections.m_arcHbmAddr + rRecipeSections[i].offsetHbm,
                          rRecipeSections[i].recipeAddr,
                          rRecipeSections[i].size);
        if (res != 0)
        {
            same = false;
            LOG_ERR(SYN_API, "memcmp failed for {}", i);
        }
    }

    if (rSections.m_arcHbmAddr - rSections.m_arcHbmCoreAddr != offset)
    {
        same = false;
        LOG_ERR(SYN_API,
                     "addrDev-addrCore offset is wrong {:x} {:x} offset {:x}",
                     rSections.m_arcHbmAddr,
                     rSections.m_arcHbmCoreAddr,
                     offset);
    }
    return same;
}

void MappedMemMgrTestUtils::testingCopyToGlb(const RecipeSingleSectionVec& rRecipeSections,
                                             const MemorySectionsScal&     rSections)
{
    for (int i = 0; i < FIRST_IN_ARC; i++)
    {
        LOG_TRACE(SYN_API,
                       "testing glb memcpy {:x}<-{:x} size {:x}",
                       rSections.m_glbHbmAddr + rRecipeSections[i].offsetHbm,
                       TO64(rRecipeSections[i].recipeAddr),
                       rRecipeSections[i].size);
        memcpy((uint8_t*)rSections.m_glbHbmAddr + rRecipeSections[i].offsetHbm,
               rRecipeSections[i].recipeAddr,
               rRecipeSections[i].size);
    }
}

bool MappedMemMgrTestUtils::testingCheckGlbSections(const RecipeSingleSectionVec& rRecipeSections,
                                                    const MemorySectionsScal&     rSections)
{
    bool same = true;
    for (int i = 0; i < FIRST_IN_ARC; i++)
    {
        LOG_TRACE(SYN_API,
                       "testing hbm glb memcmp {:x},{:x} size {:x}",

                       rSections.m_glbHbmAddr + rRecipeSections[i].offsetHbm,
                       TO64(rRecipeSections[i].recipeAddr),
                       rRecipeSections[i].size);
        bool res = memcmp((uint8_t*)rSections.m_glbHbmAddr + rRecipeSections[i].offsetHbm,
                          rRecipeSections[i].recipeAddr,
                          rRecipeSections[i].size);
        if (res != 0)
        {
            same = false;
            LOG_ERR(SYN_API, "memcmp failed for {}", i);
        }
    }
    return same;
}
