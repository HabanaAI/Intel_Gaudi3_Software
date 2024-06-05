#include "recipe_addr_patcher.hpp"

#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "utils.h"

#include "memory_management/memory_protection.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

RecipeAddrPatcher::RecipeAddrPatcher()
: m_protectMem((GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & PROTECT_MAPPED_MEM) == PROTECT_MAPPED_MEM)
{
}

/*
 ***************************************************************************************************
 *   @brief init() - translates the recipe patch-points to a new database that is more efficient
 *                   when doing the patching. Needs to be called once
 *
 *   Note: we have one set of patch-points that includes all.
 *         And N sets of patch points, one per section type
 *
 *   @param  recipe, data-chunk size (that we later do the patching on)
 *   @return bool (false-fail, true-OK)
 *
 ***************************************************************************************************
 */
bool RecipeAddrPatcher::init(const recipe_t& recipe, uint32_t dcSize)
{
    static_assert((1 << PP_TYPE_SIZE_BITS) >= patch_point_t::PP_TYPE_LAST, "not enough bits");

    m_recipe = &recipe;

    if (dcSize == 0)
    {
        LOG_ERR(SYN_RECIPE, "dcSize is 0, illegal");
        return false;
    }

    if ((1 << OFFSET_SIZE_BITS) < dcSize)
    {
        LOG_ERR(SYN_RECIPE,
                "not enough bits 2^{}={:x} to cover dc size {:x}",
                OFFSET_SIZE_BITS,
                1 << OFFSET_SIZE_BITS,
                dcSize);
        return false;
    }

    // dc is only 8 bits, make sure patchable is not more than 256 dc
    if (recipe.patching_blobs_buffer_size > ((1 << 8) * dcSize))
    {
        LOG_ERR(SYN_RECIPE, "patching blob buffer size too big, won't fit in 256 (8bit) DCs total size {:x} dc size {:x}",
                recipe.patching_blobs_buffer_size, dcSize);
        return false;
    }

    m_dcSize = dcSize;

    // build the data base for all the patch-points
    uint64_t     numPP = recipe.patch_points_nr;
    m_dcPp.resize(numPP);

    for (uint32_t pp = 0; pp < numPP; pp++)
    {
        const uint32_t ppIdx    = pp;
        patch_point_t& recipePp = recipe.patch_points[ppIdx];

        setDcPp(m_dcPp[pp], recipePp, recipe, dcSize);
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief setDcPp() - utility function used by init
 *
 *   @param
 *   @return None
 *
 ***************************************************************************************************
 */
void RecipeAddrPatcher::setDcPp(DataChunkPatchPoint& dcPp,
                                const patch_point_t& recipePp,
                                const recipe_t&      recipe,
                                uint32_t             dcSize)
{
    uint32_t recipeBlobIdx = recipePp.blob_idx;
    uint8_t* ppBlobAddr    = (uint8_t*)(((uint32_t*)recipe.blobs[recipeBlobIdx].data) + recipePp.dw_offset_in_blob);
    uint64_t offset        = ppBlobAddr - (uint8_t*)recipe.patching_blobs_buffer;

    dcPp.type                                 = recipePp.type;
    dcPp.memory_patch_point.section_idx       = recipePp.memory_patch_point.section_idx;
    dcPp.memory_patch_point.effective_address = recipePp.memory_patch_point.effective_address;
    dcPp.data_chunk_index                     = offset / dcSize;
    dcPp.offset_in_data_chunk                 = offset - (dcPp.data_chunk_index * dcSize);
}

/*
 ***************************************************************************************************
 *   @brief patchAll() - Does the patching (using all the patch-points) on the user given data.
 *                       Data is given in data-chunks
 *
 *   @param  sectionAddrDb - array of the section addresses
 *   @param  dcAddr        - vector of data chunks addresses
 *   @return None
 *
 ***************************************************************************************************
 */
void RecipeAddrPatcher::patchAll(const uint64_t* sectionAddrDb, MemoryMappedAddrVec& dcAddr) const
{
    bool shouldLog = LOG_LEVEL_AT_LEAST_TRACE(SYN_PATCHING);

    if (m_protectMem)
    {
        unprotectDcs(dcAddr);
    }

    // Go over the patch points and do the patching
    for (uint32_t ppIdx = 0; ppIdx < m_dcPp.size(); ppIdx++)
    {
        auto&          pp                          = m_dcPp[ppIdx];
        const auto     dataChunkIdx                = pp.data_chunk_index;
        const uint64_t currentDataChunkHostAddress = (uint64_t)dcAddr[dataChunkIdx].hostAddr;
        uint64_t       offsetInDataChunk           = pp.offset_in_data_chunk;

        // calculate patch point value
        uint64_t sectionAddress    = sectionAddrDb[pp.memory_patch_point.section_idx];
        ptrToInt patchedPointValue = {.u64 = sectionAddress + pp.memory_patch_point.effective_address};

        // actual patching
        uint32_t* pPatchedLocation = (uint32_t*)(currentDataChunkHostAddress + offsetInDataChunk);
        *pPatchedLocation          = patchedPointValue.u32[(pp.type & 1)];
        if (pp.type == patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT)
        {
            *(pPatchedLocation + 1) = patchedPointValue.u32[1];
        }

        if (unlikely(shouldLog))
        {
            LOG_TRACE(SYN_PATCHING,
                      "{}: Patching (DC {} PP-Index in Stage {:x}) DC-Address 0x{:x} Offset-in-DC 0x{:x},"
                      " section {}, effective-address 0x{:x} patched-location 0x{:x} patch-type {} value 0x{:x}",
                      HLLOG_FUNC,
                      dataChunkIdx,
                      ppIdx,
                      currentDataChunkHostAddress,
                      offsetInDataChunk,
                      pp.memory_patch_point.section_idx,
                      //                pp.node_exe_index,
                      pp.memory_patch_point.effective_address,
                      (uint64_t)currentDataChunkHostAddress + offsetInDataChunk,
                      pp.type,
                      patchedPointValue.u64);
        }
    }
    LOG_TRACE(SYN_PATCHING, "{}: Patching execution successfully completed {} patch-points", HLLOG_FUNC, m_dcPp.size());

    if (m_protectMem)
    {
        protectDcs(dcAddr);
    }
}

/*
 ***************************************************************************************************
 *   @brief getPpAddr() - utility function used by verifySectionsAddrFromDc. Calculates the patching
 *                        address from the patch-point index and the vector of patch-points
 *
 *   @param  sortedPp - vector of patch-point
 *   @param  ppIdx    - index of patch-points that we want the address for
 *   @return uint8_t* - address of the patching location
 *
 ***************************************************************************************************
 */
uint8_t* RecipeAddrPatcher::getPpAddr(std::vector<patch_point_t>& sortedPp, uint32_t ppIdx) const
{
    if (ppIdx >= m_recipe->patch_points_nr)
    {
        return nullptr;
    }

    const patch_point_t& pp      = sortedPp[ppIdx];
    uint32_t             blobIdx = pp.blob_idx;
    uint64_t             dw      = pp.dw_offset_in_blob;
    uint8_t*             addr    = (uint8_t*)m_recipe->blobs[blobIdx].data + sizeof(uint32_t) * dw;

    return addr;
}

/*
 ***************************************************************************************************
 *   @brief verifySectionsAddrFromDc() - The function gets a vector of data-chunks.
 *                                       It calls a function that calculates the sections addresses
 *                                       based on the patched data (in the data chunks). It then compares
 *                                       it to the user given expected sections addresses (usually from the
 *                                       user tensors).
 *                                       It is used to verify the patching was done correctly.
 *
 *   @param  dcVec  - vector of data-chunks
 *   @param  dcSize - size of the data-chunks
 *   @param  expectedAddr - array of the user expected sections addresses
 *   @param  expectedSize - size of the expectedAddr array
 *   @return bool (true=OK, false=fail)
 *
 ***************************************************************************************************
 */
bool RecipeAddrPatcher::verifySectionsAddrFromDc(const MemoryMappedAddrVec dcVec,
                                                 uint32_t                  dcSize,
                                                 const uint64_t*           expectedAddr,
                                                 uint32_t                  expectedSize) const
{
    LOG_DEBUG(SYN_PATCHING, "{} expectedSize {:x}", HLLOG_FUNC, expectedSize);
    std::vector<UnpatchSingleRes> calcAddr;

    bool res = getSectionsAddrFromDc(dcVec, dcSize, calcAddr);
    if (!res) return res;

    // Sanity check - we should calculate no more sections than the user gave us
    if (calcAddr.size() > expectedSize)
    {
        LOG_ERR(SYN_PATCHING, "Got more sections from calc {:x} than expected {:x}", calcAddr.size(), expectedSize);
        return false;
    }

    const uint64_t LOW = 0xFFFFFFFF;

    // Go over each section, compare only if valid bit is on
    for (int i = 0; i < calcAddr.size(); i++)
    {
        LOG_TRACE(SYN_PATCHING,
                  "{} section {:x} calc {:x} expected {:x} valid hi/low {:x}/{:x}",
                  HLLOG_FUNC,
                  i,
                  calcAddr[i].val,
                  expectedAddr[i],
                  calcAddr[i].highValid,
                  calcAddr[i].lowValid);

        if (calcAddr[i].lowValid)
        {
            if ((calcAddr[i].val & LOW) != (expectedAddr[i] & LOW))
            {
                LOG_ERR(SYN_PATCHING,
                        "For section {:x}, low is different {:x} {:x}",
                        i,
                        calcAddr[i].val,
                        expectedAddr[i]);
                return false;
            }
        }

        if (calcAddr[i].highValid)
        {
            if ((calcAddr[i].val >> 32) != (expectedAddr[i] >> 32))
            {
                LOG_ERR(SYN_PATCHING,
                        "For section {:x}, high is different {:x} {:x}",
                        i,
                        calcAddr[i].val,
                        expectedAddr[i]);
                return false;
            }
        }
    }
    return true;
}

void RecipeAddrPatcher::copyPatchPointDb(const RecipeAddrPatcher& recipeAddrPatcherToCopy)
{
    HB_ASSERT(m_dcPp.size() == recipeAddrPatcherToCopy.m_dcPp.size(),
        "copyPatchPointDb called with different vector sizes");

    memcpy(m_dcPp.data(), recipeAddrPatcherToCopy.m_dcPp.data(), m_dcPp.size() * sizeof(m_dcPp[0]));
}

/*
 ***************************************************************************************************
 *   @brief getSectionsAddrFromDc() - calculates the sections addresses from the patched data-chunks.
 *                                    Note1: in some cases not all the sections can be calculated so
 *                                    it returns also a valid indication.
 *                                    Note2: in some cases only part of the section address can be calculated
 *                                    (32 bits high or low), so there are two valid indications per 64 bit address
 *
 *   @param  dcVec  - vector of data-chunks
 *   @param  dcSize - size of the data-chunks
 *   @param  sectionsAddr - vector of sections addresses and valid indications
 *   @return bool (true=OK, false=fail)
 *
 ***************************************************************************************************
 */
bool RecipeAddrPatcher::getSectionsAddrFromDc(const MemoryMappedAddrVec&     dcVec,
                                              uint32_t                       dcSize,
                                              std::vector<UnpatchSingleRes>& sectionsAddr) const
{
    // find max section idx
    uint16_t maxSection = 0;
    for (int pp = 0; pp < m_recipe->patch_points_nr; pp++)
    {
        maxSection = std::max(maxSection, m_recipe->patch_points[pp].memory_patch_point.section_idx);
    }

    sectionsAddr.resize(maxSection + 1);

    // Sort the patch points based on the address. This is done so when we go over the buffer we know what
    // is the next patch points.
    std::vector<patch_point_t> sortedPp(m_recipe->patch_points, m_recipe->patch_points + m_recipe->patch_points_nr);
    // sorting by blob idx should be enough (as blobs are sorted)
    std::sort(std::begin(sortedPp), std::end(sortedPp), [&](const patch_point_t& a, const patch_point_t& b) {
        uint32_t  aBlobIdx = a.blob_idx;
        uint32_t* aAddr    = (uint32_t*)m_recipe->blobs[aBlobIdx].data + a.dw_offset_in_blob;

        uint32_t  bBlobIdx = b.blob_idx;
        uint32_t* bAddr    = (uint32_t*)m_recipe->blobs[bBlobIdx].data + b.dw_offset_in_blob;

        return aAddr < bAddr;
    });

    uint32_t ppIdx  = 0;
    uint8_t* ppAddr = getPpAddr(sortedPp, ppIdx);  // next addresses that was patched
    for (uint64_t dwBuff = 0; dwBuff < m_recipe->patching_blobs_buffer_size; dwBuff += 4)  // go in 4 bytes increment
    {
        uint32_t* buffAddr = (uint32_t*)((uint8_t*)m_recipe->patching_blobs_buffer + dwBuff);
        uint32_t  buffVal  = *buffAddr;

        uint32_t  dc       = dwBuff / dcSize;
        uint32_t  offsetDc = dwBuff % dcSize;
        uint32_t* dcAddr   = (uint32_t*)(dcVec[dc].hostAddr + offsetDc);
        uint32_t  dcVal    = *dcAddr;

        // if not a patch point, values in recipe and data-chunk should be the same
        if (ppAddr != (uint8_t*)buffAddr)
        {
            if (dcVal == buffVal) continue;
            LOG_ERR(SYN_PATCHING,
                    "Non patched values are not the same. dcVal {:x} buffVal {:x} dc {:x} offsetDc {:x} dcAddr {:x}"
                    " ppIdx {:x} ppAddr {:x} blobIdx {:x} dwBlob {:x} buffBaseAddr {:x} buffAddr {:x} dwBuff {:x}",
                    dcVal,
                    buffVal,
                    dc,
                    offsetDc,
                    TO64(dcAddr),
                    ppIdx,
                    TO64(ppAddr),
                    sortedPp[ppIdx].blob_idx,
                    sortedPp[ppIdx].dw_offset_in_blob,
                    TO64(m_recipe->patching_blobs_buffer),
                    TO64(buffAddr),
                    dwBuff);
            return false;
        }

        const uint64_t HI  = 0xFFFFFFFF00000000;
        const uint64_t LOW = 0x00000000FFFFFFFF;

        // We are looking at a patch point, calculate the section address
        const patch_point_t& pp      = sortedPp[ppIdx];
        uint32_t             section = pp.memory_patch_point.section_idx;

        switch (pp.type)
        {
            case patch_point_t::SIMPLE_DW_LOW_MEM_PATCH_POINT:
            {
                uint32_t sectionAddrLow = dcVal - (uint32_t)pp.memory_patch_point.effective_address;

                LOG_TRACE(SYN_PATCHING,
                          "{} dcAddr {:x} PP LOW sectionAddrLow {:x} dcVal {:x} effective adddr {:x} ppIdx {:x}",
                          HLLOG_FUNC,
                          TO64(dcAddr),
                          sectionAddrLow,
                          dcVal,
                          pp.memory_patch_point.effective_address,
                          ppIdx);

                if (sectionsAddr[section].lowValid)  // already set, verify the same
                {
                    if ((uint32_t)sectionsAddr[section].val != sectionAddrLow)
                    {
                        LOG_ERR(SYN_PATCHING,
                                "LOW: Current val {:x} different from previous {:x}",
                                sectionAddrLow,
                                (uint32_t)sectionsAddr[section].val);
                        return false;
                    }
                }

                sectionsAddr[section].lowValid = true;
                sectionsAddr[section].val      = (sectionsAddr[section].val & HI) | sectionAddrLow;
                break;
            }

            case patch_point_t::SIMPLE_DW_HIGH_MEM_PATCH_POINT:
            {
                uint32_t sectionAddrHigh = dcVal - (pp.memory_patch_point.effective_address >> 32);

                LOG_TRACE(SYN_PATCHING,
                          "{} dcAddr {:x} PP HIGH sectionAddrHigh {:x} dcVal {:x} effective adddr {:x} ppIdx {:x}",
                          HLLOG_FUNC,
                          TO64(dcAddr),
                          sectionAddrHigh,
                          dcVal,
                          pp.memory_patch_point.effective_address,
                          ppIdx);

                if (sectionsAddr[section].highValid)  // already set, verify the same
                {
                    if ((sectionsAddr[section].val >> 32) != sectionAddrHigh)
                    {
                        LOG_ERR(SYN_PATCHING,
                                "HIGH: Current val {:x} different from previous {:x}",
                                sectionAddrHigh,
                                sectionsAddr[section].val >> 32);
                        return false;
                    }
                }

                sectionsAddr[section].highValid = true;
                sectionsAddr[section].val       = (sectionsAddr[section].val & LOW) | ((uint64_t)sectionAddrHigh << 32);
                break;
            }

            case patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT:
            {
                uint64_t dcVal64       = (uint64_t)(*dcAddr) + ((uint64_t)(*(dcAddr + 1)) << 32);  //*dcAddr64;
                uint64_t sectionAddr64 = dcVal64 - pp.memory_patch_point.effective_address;

                LOG_TRACE(SYN_PATCHING,
                          "{} dcAddr {:x} PP DDW sectionAddr64 {:x} dcVal64 {:x} effective adddr {:x} ppIdx {:x}",
                          HLLOG_FUNC,
                          TO64(dcAddr),
                          sectionAddr64,
                          dcVal64,
                          pp.memory_patch_point.effective_address,
                          ppIdx);

                if (sectionsAddr[section].lowValid)  // already set, verify the same (low)
                {
                    if ((uint32_t)sectionsAddr[section].val != (uint32_t)sectionAddr64)
                    {
                        LOG_ERR(SYN_PATCHING,
                                "DW LOW: Current val {:x} different from previous {:x}",
                                sectionAddr64,
                                sectionsAddr[section].val);
                        return false;
                    }
                }

                if (sectionsAddr[section].highValid)  // already set, verify the same (high)
                {
                    if ((sectionsAddr[section].val >> 32) != (sectionAddr64 >> 32))
                    {
                        LOG_ERR(SYN_PATCHING,
                                "DW HIGH: Current val {:x} different from previous {:x}",
                                sectionAddr64,
                                sectionsAddr[section].val);
                        return false;
                    }
                }

                sectionsAddr[section].val       = sectionAddr64;
                sectionsAddr[section].lowValid  = true;
                sectionsAddr[section].highValid = true;

                dwBuff += 4;
                break;
            }

            case patch_point_t::SOB_PATCH_POINT:
            {
                HB_ASSERT_DEBUG_ONLY(false, "If we get here, we should handle it");
            }
        }  // switch

        // prepare pp for next run
        ppIdx++;
        ppAddr = getPpAddr(sortedPp, ppIdx);
    }
    return true;
}

void RecipeAddrPatcher::unprotectDcs(MemoryMappedAddrVec& dcAddr) const
{
    for (auto& dc : dcAddr)
    {
        MemProtectUtils::memWrUnprotectPages(dc.hostAddr, m_dcSize);
    }
}

void RecipeAddrPatcher::protectDcs(MemoryMappedAddrVec& dcAddr) const
{
    for (auto& dc : dcAddr)
    {
        MemProtectUtils::memWrProtectPages(dc.hostAddr, m_dcSize);
    }
}
