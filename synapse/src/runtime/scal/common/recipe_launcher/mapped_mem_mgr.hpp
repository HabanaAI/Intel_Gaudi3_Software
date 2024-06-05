#pragma once

#include "infra/memory_management/segment_alloc.hpp"
#include "mem_mgrs_types.hpp"
#include "statistics.hpp"

#include "runtime/common/recipe/patching/host_address_patcher.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

#include <deque>
#include <memory>
#include <unordered_map>
#include "device/device_mem_alloc.hpp"

class HostBuffersMapper;
struct RecipeStaticInfoScal;

struct PatchableInfo
{
    std::vector<uint16_t>                                     segments;
    std::unique_ptr<patching::HostAddressPatchingInformation> hostAddrPatchingInfo;  // information about the sections
                                                                                     // addresses used when it was last
                                                                                     // patched
};

struct PatchableUsedInfo
{
    PatchableInfo patchableInfo;
    uint64_t      runningId;
};

struct MappedRecipeInfo
{
    std::vector<uint16_t>         nonPatchableAddr;
    std::deque<PatchableInfo>     patchableFree;
    std::deque<PatchableUsedInfo> patchableUsed;
};

/*********************************************************************************
 * This class is used to allocate mapped memory for the recipe. It uses an allocator
 * to get memory from pre-allocated memory.
 * Per recipe it allocates one chunk of memory for all the non-patchable sections and
 * multiple chunks of memory (as needed) for the patchable section
 *
 * NOTE: This class assumes only one thread is using it. If we need to make it thread safe:
 *       There is a gap between getting the memory and copying to it, need to handle  the case where
 *       another thread asks for the same recipe during this gap.
 *********************************************************************************/
class MappedMemMgr
{
private:
    enum class StatPoints
    {
        requests,
        busyReturned,
        released,
        newRecipeBusyPatch,
        newRecipeBusyNonPatch,
        newRecipeOK,
        recipePatchBusy,
        recipeNewPatch,
        recipePatchFound,
        LAST
    };

    static constexpr auto enumNamePoints = toStatArray<StatPoints>(
    {
        {StatPoints::requests,              "requests"},
        {StatPoints::busyReturned,          "busyReturned"},
        {StatPoints::released,              "released"},
        {StatPoints::newRecipeBusyPatch,    "newRecipeBusyPatch"},
        {StatPoints::newRecipeBusyNonPatch, "newRecipeBusyNonPatch"},
        {StatPoints::newRecipeOK,           "newRecipeOK"},
        {StatPoints::recipePatchBusy,       "recipePatchBusy"},
        {StatPoints::recipeNewPatch,        "recipeNewPatch"},
        {StatPoints::recipePatchFound,      "recipePatchFound"}
    });

public:
    MappedMemMgr(const std::string& name, DevMemoryAllocInterface& devMemoryAllocInterface);

    virtual ~MappedMemMgr();

    synStatus init();

    void removeId(RecipeSeqId id);

    void removeAllId();

    bool unuseId(EntryIds ids, bool error = false);

    void unuseIdOnError(EntryIds ids);

    void
    getAddrForId(const RecipeStaticInfoScal& rRecipeStaticInfoScal, EntryIds entryIds, MemorySectionsScal& rSections);

    size_t getNumRecipes() const { return m_recipeDb.size(); }

    uint64_t getMappedMemorySize() const { return m_segmentSize * m_numSegments; }

    static uint64_t testingOnlyInitialMappedSize();

    static uint64_t getDcSize();

    static uint64_t getNumDc();

private:
    bool getDcIfNeeded(std::vector<uint16_t>& segments, uint64_t dataSize);
    void tryGetAddrForId(const RecipeStaticInfoScal& rRecipeStaticInfoScal,
                         EntryIds                    entryIds,
                         MemorySectionsScal&         rSections);
    bool releaseEntry();
    void freeEntryMemory(MappedRecipeInfo& info);
    void freeSingleFreePatchableEntry(MappedRecipeInfo& selectedMmi);

    void fillSectionsInfo(uint64_t            runningId,
                          MappedRecipeInfo&   info,
                          MemorySectionsScal& sections,
                          PatchableInfo&      patchableInfo);

    const uint64_t        ALIGN      = 128;
    static const uint32_t NUM_CHUNKS = 8;

    Statistics<enumNamePoints> m_stat;

    uint64_t                 m_segmentSize;
    uint16_t                 m_numSegments;
    uint8_t*                 m_hostAddr = nullptr;
    uint64_t                 m_mappedAddr;
    DevMemoryAllocInterface& m_devMemoryAllocInterface;
    SegmentAlloc             m_segmentAlloc;

    std::unordered_map<uint64_t, MappedRecipeInfo> m_recipeDb;
    const bool                                     m_protectMem;
};
