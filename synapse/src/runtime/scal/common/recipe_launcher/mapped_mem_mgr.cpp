#include "mapped_mem_mgr.hpp"

#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"

#include "log_manager.h"
#include "runtime/scal/common/recipe_static_info_scal.hpp"

#include <utils.inl>

/*
 ***************************************************************************************************
 *   @brief MappedMemMgr() - constructor
 *
 *   @param  name for statistics
 *   @return None
 *
 ***************************************************************************************************
 */
MappedMemMgr::MappedMemMgr(const std::string& name, DevMemoryAllocInterface& devMemoryAllocInterface)
: m_stat("MappedMemoryMgr " + name, 100, true /*enable*/),
  m_devMemoryAllocInterface(devMemoryAllocInterface),
  m_protectMem((GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & PROTECT_MAPPED_MEM) == PROTECT_MAPPED_MEM)
{
}

MappedMemMgr::~MappedMemMgr()
{
    if (m_hostAddr != nullptr)
    {
        LOG_INFO(SYN_PROG_DWNLD, "Releasing mmm memory, addr {:x}", TO64(m_hostAddr));
        synStatus status = m_devMemoryAllocInterface.deallocateMemory((void*)m_hostAddr, synMemFlags::synMemHost, false);

        if (status != synSuccess)
        {
            LOG_ERR(SYN_PROG_DWNLD,
                         "Could not release mmm memory, addr {:x}, return code {}",
                         TO64(m_hostAddr),
                         status);
        }
    }
}

/*
 ***************************************************************************************************
 *   @brief init() - set the pre-allocated memory info - host/dev start addr, size
 *
 *   @param  devAddr, hostAddr, size - the pre-allocated memory for this class to use
 *   @return None
 *
 ***************************************************************************************************
 */
synStatus MappedMemMgr::init()
{
    m_segmentSize = getDcSize();  // cache the value

    if ((m_segmentSize % PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES) != 0)
    {
        LOG_CRITICAL_T(SYN_PROG_DWNLD,
                       "chunk size {:x} must be multiple of PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES {:x}",
                       m_segmentSize,
                       PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES);
        return synFail;
    }

    m_numSegments = GCFG_STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP.value();

    uint64_t neededSize = m_numSegments * m_segmentSize;
    void*    hostVoidAddr;

    synStatus status = m_devMemoryAllocInterface.allocateMemory(neededSize,
                                                                synMemFlags::synMemHost,
                                                                &hostVoidAddr,
                                                                false,
                                                                0,
                                                                "mapped-memory-mgr",
                                                                &m_mappedAddr);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_PROG_DWNLD,
                     "can not allocate memory for mapped-memory-mgr. num/size segments {:x}/{:x}. needed size {:x}. "
                     "return status {}",
                     m_numSegments,
                     m_segmentSize,
                     neededSize,
                     status);
        return status;
    }

    m_hostAddr = (uint8_t*)hostVoidAddr;

    m_segmentAlloc.init(m_numSegments);

    LOG_INFO(SYN_PROG_DWNLD,
                  "Allocated memory for mapped-memory-mgr. num/size segments {:x}/{:x}. NeededSize {:x} hostAddr {:x} "
                  "mappedAddr {:x}",
                  m_numSegments,
                  m_segmentSize,
                  neededSize,
                  TO64(m_hostAddr),
                  TO64(m_mappedAddr));

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief testingOnlyInitialMappedSize() - For testing, returns the size fo the mapped memory
 *
 *   @param  info - the info on this entry
 *   @return None
 *
 ***************************************************************************************************
 */
uint64_t MappedMemMgr::testingOnlyInitialMappedSize()
{
    return getNumDc() * getDcSize();
}

/*
 ***************************************************************************************************
 *   @brief getDcSize() - For testing, returns the segment size of the mapped memory
 *
 *   @param  info - the info on this entry
 *   @return None
 *
 ***************************************************************************************************
 */
uint64_t MappedMemMgr::getDcSize()
{
    return GCFG_STREAM_COMPUTE_ARC_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP.value() * 1024;
}

uint64_t MappedMemMgr::getNumDc()
{
    return GCFG_STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP.value();
}

/*
 ***************************************************************************************************
 *   @brief freeEntryMemory() - frees the memory for a given entry. It will free the chunk for the
 *                              non-patchable sections and all the chunks for the patchable sections
 *
 *   @param  info - the info on this entry
 *   @return None
 *
 ***************************************************************************************************
 */
void MappedMemMgr::freeEntryMemory(MappedRecipeInfo& info)
{
    while (!info.patchableFree.empty())
    {
        freeSingleFreePatchableEntry(info);
    }
    for (int i = 0; i < info.nonPatchableAddr.size(); i++)
    {
        LOG_TRACE(SYN_PROG_DWNLD, "Going to release non-patchable {:x}", TO64(info.nonPatchableAddr[i]));
    }
    m_segmentAlloc.releaseSegments(info.nonPatchableAddr);
}

/*
 ***************************************************************************************************
 *   @brief freeSingleFreePatchableEntry() - frees a single patchable free-memory chunk for a givne entry.
 *
 *   @param  pSelectedMmi - A pointer to the info of that entry
 *   @return None
 *
 ***************************************************************************************************
 */
void MappedMemMgr::freeSingleFreePatchableEntry(MappedRecipeInfo& selectedMmi)
{
    if (unlikely(selectedMmi.patchableFree.empty()))
    {
        return;
    }

    PatchableInfo& patchableInfo = selectedMmi.patchableFree.front();
    auto&          segments      = patchableInfo.segments;

    for (int i = 0; i < segments.size(); i++)
    {
        LOG_TRACE(SYN_PROG_DWNLD, "Going to release patchable segment {:x}", TO64(segments[i]));
    }
    m_segmentAlloc.releaseSegments(segments);
    selectedMmi.patchableFree.pop_front();
}

/*
 ***************************************************************************************************
 *   @brief removeId() - This function is called when a recipe is destroyed. It finds the entry in the DB,
 *                       calls a function to release the memory and deletes it from the DB. we assume that this method
 *                       is called from recipe destroy flow only, thus, we use info and not errrors.
 *
 *   @param  id - the recipe to be removed
 *   @return None
 *
 ***************************************************************************************************
 */
void MappedMemMgr::removeId(RecipeSeqId id)
{
    auto it = m_recipeDb.find(id.val);
    if (it == m_recipeDb.end())
    {
        // This can happen if recipe was evicted from the mmm (because we needed its space for a different recipe_
        LOG_DEBUG_T(SYN_PROG_DWNLD, "removeId not found {:x}", id.val);
        return;
    }

    MappedRecipeInfo& info = it->second;

    if (!info.patchableUsed.empty())
    {
        LOG_ERR_T(SYN_PROG_DWNLD, "removeId is still used {:x}, has {:x} used", id.val, info.patchableUsed.size());
        return;
    }

    freeEntryMemory(info);

    m_recipeDb.erase(it);
}

void MappedMemMgr::removeAllId()
{
    for (auto& recipeDb : m_recipeDb)
    {
        RecipeSeqId       id(recipeDb.first);
        MappedRecipeInfo& info = recipeDb.second;

        if (!info.patchableUsed.empty())
        {
            LOG_ERR_T(SYN_PROG_DWNLD, "removeId is still used {:x}, has {:x} used", id.val, info.patchableUsed.size());
            return;
        }

        freeEntryMemory(info);
    }

    m_recipeDb.clear();
}

/*
 ***************************************************************************************************
 *   @brief fillSectionsInfo() - This function is called to update the sections' information with the
 *                               addresses/status
 *
 *   @param  runningId - free running counter, for sanity checks, debug
 *   @param  Info      - the information in the DB for this recipe
 *   @param  recipeSections - the sections information
 *   @param  addrPatch      - address of the patchable section
 *   @param  nonPatchableStatus - the status to put in the non-patchable sections (for the patchable sections, this
 *                                is filled by the caller
 *   @return None
 *
 ***************************************************************************************************
 */
void MappedMemMgr::fillSectionsInfo(uint64_t            runningId,
                                    MappedRecipeInfo&   info,
                                    MemorySectionsScal& rSections,
                                    PatchableInfo&      patchableInfo)
{
    size_t numSegments          = patchableInfo.segments.size();
    rSections.m_patchableDcSize = m_segmentSize;

    rSections.m_patchableMappedAddr.resize(numSegments);
    for (int i = 0; i < numSegments; i++)
    {
        uint64_t offset = patchableInfo.segments[i] * m_segmentSize;

        rSections.m_patchableMappedAddr[i] = {m_hostAddr + offset, m_mappedAddr + offset};

        LOG_TRACE(SYN_MEM_MAP,
                       "runId {:x} patchable {}/{} segment {:x} host/dev addr {:x}/{:x} used {:x}",
                       runningId,
                       i,
                       numSegments,
                       TO64(patchableInfo.segments[i]),
                       TO64(rSections.m_patchableMappedAddr[i].hostAddr),
                       rSections.m_patchableMappedAddr[i].devAddr,
                       info.patchableUsed.size());
    }

    if (rSections.m_inMappedPatch == OUT)
    {
        patchableInfo.hostAddrPatchingInfo =
            std::unique_ptr<patching::HostAddressPatchingInformation>(new patching::HostAddressPatchingInformation);
    }
    rSections.m_hostAddrPatchInfo = patchableInfo.hostAddrPatchingInfo.get();
    // add the patchable info to the back of the used list
    info.patchableUsed.push_back({std::move(patchableInfo), runningId});

    // Fill the other sections
    rSections.m_nonPatchableDcSize = m_segmentSize;

    rSections.m_nonPatchableMappedAddr.resize(info.nonPatchableAddr.size());
    for (int i = 0; i < info.nonPatchableAddr.size(); i++)
    {
        uint64_t offset = info.nonPatchableAddr[i] * m_segmentSize;

        rSections.m_nonPatchableMappedAddr[i] = {m_hostAddr + offset, m_mappedAddr + offset};

        LOG_TRACE(SYN_MEM_MAP,
                       "non-patch {}/{}  segment {:x} host/dev addr {:x}/{:x}",
                       i,
                       info.nonPatchableAddr.size(),
                       TO64(info.nonPatchableAddr[i]),
                       TO64(rSections.m_nonPatchableMappedAddr[i].hostAddr),
                       rSections.m_nonPatchableMappedAddr[i].devAddr);
    }
}

/*
 ***************************************************************************************************
 *   @brief getAddr() - This function is called to get the mapped memory address for all the sections.
 *                        It tries to get memory, if it fails, it tries to release an entry.
 *                        If an entry is released, it tries to get mapped memory again, if not,
 *                        it sets BUSY to all sections status and returns
 *
 *   @param  entryIds       - recipe + running Id
 *   @param  recipeSections - the sections information to be updated
 *   @return None
 *
 ***************************************************************************************************
 */
void MappedMemMgr::getAddrForId(const RecipeStaticInfoScal& rRecipeStaticInfoScal,
                                EntryIds                    entryIds,
                                MemorySectionsScal&         rSections)
{
    LOG_TRACE(SYN_PROG_DWNLD, "seqId {:x} runningId {:x}", entryIds.recipeId.val, entryIds.runningId);
    while (true)
    {
        tryGetAddrForId(rRecipeStaticInfoScal, entryIds, rSections);

        if (!rSections.anyBusyInMapped())
        {
            m_stat.collect(StatPoints::requests, 1);
            return;
        }

        bool releaseRes = releaseEntry();
        if (releaseRes == false)  // couldn't release
        {
            m_stat.collect(StatPoints::busyReturned, 1);
            LOG_DEBUG(SYN_PROG_DWNLD,
                           "Failed to release an Entry, returning BUSY. Can not allocate for {:x}/{:x}",
                           entryIds.recipeId.val,
                           entryIds.runningId);
            m_stat.collect(StatPoints::requests, 1);
            return;
        }
        else
        {
            m_stat.collect(StatPoints::released, 1);
        }
    }
    m_stat.collect(StatPoints::requests, 1);
}

/*
 ***************************************************************************************************
 *   @brief releaseEntry() - This function releases an unused entry. For now, it just releases the first one it finds.
 *                           TODO: release the oldest one?
 *
 *   @param  None
 *   @return true (released) / false (nothing to release)
 *
 ***************************************************************************************************
 */
bool MappedMemMgr::releaseEntry()
{
    MappedRecipeInfo* pSelectedMmi                    = nullptr;
    unsigned          selectedMmiFreePatchableEntries = 0;

    // For now, just release the first one. TODO: release the oldest one? Release if more than X copies of patchable?
    for (auto it = m_recipeDb.begin(); it != m_recipeDb.end(); it++)
    {
        MappedRecipeInfo& info = it->second;
        if (info.patchableUsed.empty())  // not in use
        {
            LOG_TRACE(SYN_PROG_DWNLD, "For recipe {:x}, no patchable used, releasing", it->first);
            freeEntryMemory(info);
            m_recipeDb.erase(it);

            STAT_GLBL_COLLECT(1, mmmReleasedEntry);
            return true;  // released
        }

        unsigned freePatchableEntries = info.patchableFree.size();
        if (freePatchableEntries > selectedMmiFreePatchableEntries)
        {
            selectedMmiFreePatchableEntries = freePatchableEntries;
            pSelectedMmi                    = &info;
        }
    }

    if (pSelectedMmi != nullptr)
    {
        freeSingleFreePatchableEntry(*pSelectedMmi);
        STAT_GLBL_COLLECT(1, mmmReleasedSegment);
        return true;
    }

    STAT_GLBL_COLLECT(1, mmmCouldNotRelease);
    return false;
}

/*
 ***************************************************************************************************
 *   @brief logPatchableDb() - utility to help with logging
 ***************************************************************************************************
 */
static void logPatchableDb(EntryIds entryIds, const std::vector<uint16_t>& patchableDb, const char* msg)
{
    for (int i = 0; i < patchableDb.size(); i++)
    {
        LOG_TRACE(SYN_PROG_DWNLD,
                       "Recipe {:x} runId {:x} {} {}/{}: {:x}",
                       entryIds.recipeId.val,
                       entryIds.runningId,
                       msg,
                       i,
                       patchableDb.size(),
                       patchableDb[i]);
    }
}

/*
 ***************************************************************************************************
*   @brief getDcIfNeeded() - returns success (true) if size is 0, if not allocates Data chunks

*   @param  dataChunks   - to be filled
*   @param  dataSize     - needed size
*   @return true-OK, false-Fail
*
***************************************************************************************************
*/
bool MappedMemMgr::getDcIfNeeded(std::vector<uint16_t>& segments, uint64_t dataSize)
{
    if (dataSize == 0)
    {
        return true;
    }

    uint16_t neededSegments = (dataSize + (m_segmentSize - 1)) / m_segmentSize;

    segments = m_segmentAlloc.getSegments(neededSegments);

    if (segments.empty())
    {
        return false;
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief tryGetAddrForId() - This functions tries to allocate mapped memory for a given recipe and
 *                           fills the information in the sections information.
 *                           If recipe in DB - if have free patchable section - use it, done
 *                           if recipe in DB - no free patchable section - allocate:
 *                                allocate fail: return false, patchable=BUSY
 *                                allocate OK:   return true, patchable=OUT, non-patchable=IN
 *
 *                           if recipe not in DB - try to allocate for all. Return OUT or BUSY
 *
 *   @param  entryIds - recipe + runningId
 *   @param  recipeSections - information per section - to be updated
 *   @return None (status in recipeSections)
 *
 ***************************************************************************************************
 */
void MappedMemMgr::tryGetAddrForId(const RecipeStaticInfoScal& rRecipeStaticInfoScal,
                                   EntryIds                    entryIds,
                                   MemorySectionsScal&         rSections)
{
    // Check if recipe already in mapped memory
    uint64_t& recipeId  = entryIds.recipeId.val;
    uint64_t& runningId = entryIds.runningId;

    auto it = m_recipeDb.find(entryIds.recipeId.val);
    if (it != m_recipeDb.end())  // Recipe already in mapped
    {
        MappedRecipeInfo& info = it->second;

        PatchableInfo patchableInfo;
        if (info.patchableFree.empty())  // We don't have a patchable we can use, allocate one
        {
            bool ok = getDcIfNeeded(patchableInfo.segments, rRecipeStaticInfoScal.m_mappedSizePatch);
            if (ok == false)  // NP in, P Allocation failed
            {
                rSections.m_inMappedPatch   = BUSY;
                rSections.m_inMappedNoPatch = IN;

                LOG_TRACE(SYN_PROG_DWNLD,
                               "Recipe {:x} runId {:x} is in (except patchable) but can not allocate for patchable",
                               recipeId,
                               runningId);
                m_stat.collect(StatPoints::recipePatchBusy, 1);
                return;
            }
            else  // NP in, P allocation OK
            {
                m_stat.collect(StatPoints::recipeNewPatch, 1);
                rSections.m_inMappedPatch   = OUT;
                rSections.m_inMappedNoPatch = IN;

                logPatchableDb(entryIds, patchableInfo.segments, "got new patchable");
            }
        }
        else  // NP in, reuse P
        {
            m_stat.collect(StatPoints::recipePatchFound, 1);
            patchableInfo = std::move(info.patchableFree.front());
            info.patchableFree.pop_front();

            rSections.m_inMappedPatch   = IN;
            rSections.m_inMappedNoPatch = IN;

            logPatchableDb(entryIds, patchableInfo.segments, "is all in patcable");
        }
        fillSectionsInfo(entryIds.runningId, info, rSections, patchableInfo);
        return;
    }
    else  // recipe not in mapped memory, need to allocate for it
    {
        PatchableInfo patchableInfo;
        bool          ok = getDcIfNeeded(patchableInfo.segments, rRecipeStaticInfoScal.m_mappedSizePatch);

        if (ok == false)  // NP out, failed P
        {
            LOG_TRACE(SYN_PROG_DWNLD,
                           "Recipe {:x} runId {:x} out, can not allocate for patchable",
                           recipeId,
                           runningId);

            rSections.m_inMappedPatch   = BUSY;
            rSections.m_inMappedNoPatch = BUSY;

            m_stat.collect(StatPoints::newRecipeBusyPatch, 1);
            return;
        }

        std::vector<uint16_t> nonPatchableSegments;
        ok = getDcIfNeeded(nonPatchableSegments, rRecipeStaticInfoScal.m_mappedSizeNoPatch);

        if (ok == false)  // NP out, failed NP
        {
            LOG_TRACE(SYN_PROG_DWNLD,
                           "Recipe {:x} runId {:x} out, can not allocate for non-patchable",
                           recipeId,
                           runningId);

            m_segmentAlloc.releaseSegments(patchableInfo.segments);

            rSections.m_inMappedNoPatch = BUSY;
            rSections.m_inMappedPatch   = BUSY;

            m_stat.collect(StatPoints::newRecipeBusyNonPatch, 1);
            return;
        }
        m_stat.collect(StatPoints::newRecipeOK, 1);
        rSections.m_inMappedPatch   = OUT;
        rSections.m_inMappedNoPatch = OUT;

        MappedRecipeInfo& info = m_recipeDb[entryIds.recipeId.val];
        info.nonPatchableAddr  = std::move(nonPatchableSegments);

        fillSectionsInfo(entryIds.runningId, info, rSections, patchableInfo);
        return;
    }
}

/*
 ***************************************************************************************************
 *   @brief unuseId() - This functions marks a recipe as unused. It moves the patchable section from
 *                      the used list to the free list. In case of an error, will remove the entry
 *                      from the list and free the allocated section of the entry. Note, a recipe with
 *                      an empty patchable-used list is considered unused and can be removed from the DB
 *                      to free space for antoher recipe
 *
 *   @param  entryIds - recipe + runningId
 *   @return true (OK) / false (error - should never happen)
 *
 ***************************************************************************************************
 */
bool MappedMemMgr::unuseId(EntryIds ids, bool error)
{
    LOG_TRACE(SYN_PROG_DWNLD, "seqId {:x} runningID {:x}", ids.recipeId.val, ids.runningId);

    bool rtn = true;  // assume all is good

    auto it = m_recipeDb.find(ids.recipeId.val);
    if (it == m_recipeDb.end())  // something is wrong
    {
        LOG_ERR_T(SYN_PROG_DWNLD, "recipe {:x}, running id {:x} not found for unset", ids.recipeId.val, ids.runningId);
        return false;
    }

    MappedRecipeInfo& info = it->second;

    // If error, remove last one added. If not, remove older one
    PatchableUsedInfo usedInfo = error ? std::move(info.patchableUsed.back()) : std::move(info.patchableUsed.front());
    uint64_t          expectedRunningId = usedInfo.runningId;
    if (expectedRunningId != ids.runningId)  // we expect the recipes to finish in order (FIFO)
    {
        LOG_ERR_T(SYN_PROG_DWNLD,
                  "for patchable, expectedRunningId != runningId {:x} != {:x}",
                  expectedRunningId,
                  ids.runningId);
        HB_ASSERT(0, "for patchable, expectedRunningId != runningId");
        return false;
    }

    // If error, remove last one added. If not, remove older one
    // On normal case, move from used to free for re-use of the entry
    // On error case, do not move to free since the data is not valid and also free all segments
    auto& segments = usedInfo.patchableInfo.segments;
    if (error)
    {
        LOG_TRACE(SYN_PROG_DWNLD, "Error in patching for recipe {} runningId {}, releasing patchable segments", ids.recipeId.val, ids.runningId);
        for (int i = 0; i < segments.size(); i++)
        {
            LOG_TRACE(SYN_PROG_DWNLD, "Due to error, going to release patchable segment {:x}", TO64(segments[i]));
        }
        m_segmentAlloc.releaseSegments(segments);
        info.patchableUsed.pop_back();

        // If the failed launch was the first launch of this recipe, we need to remove the recipe from the DB
        // since the nonPatchable entry is not filled and we want to create it from the beginning on the next launch
        if (info.patchableUsed.size() == 0)
        {
            removeId(ids.recipeId);
        }
    }
    else
    {
        for (int i = 0; i < segments.size(); i++)
        {
            LOG_TRACE(SYN_MEM_MAP,
                           "Unuse (moving from used to free) {:x}/{:x} num used patchable before {:x} segment {:x} "
                           "from err flow {}",
                           ids.recipeId.val,
                           ids.runningId,
                           info.patchableUsed.size(),
                           TO64(usedInfo.patchableInfo.segments[i]),
                           error);
        }

        info.patchableFree.push_back(std::move(usedInfo.patchableInfo));
        info.patchableUsed.pop_front();
    }

    return rtn;
}

void MappedMemMgr::unuseIdOnError(EntryIds ids)
{
    unuseId(ids, true);
}
