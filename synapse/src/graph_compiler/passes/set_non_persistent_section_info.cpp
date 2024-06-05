#include "gaudi_graph.h"
#include "habana_global_conf.h"
#include "habana_pass.h"
#include "liveness_analysis.h"
#include "settable.h"

#include <unordered_set>

// Creates 2 maps from non-persistent section id to its unique tensors sorted by exec order:
// multiBufferSections - includes tensors with non-persistent section id + buffering-level ->
//                       need to caluclate both offset and allocation-size for tensors in this list
// userManagedSections - includes tensors with non-persistent section id + offset ->
//                       offset already set, need to calculate allocation-size for tensors in this list
static void collectMemorySecionTensors(HabanaGraph&                      g,
                                       std::map<uint64_t, TensorVector>& multiBufferSections,  /*OUT*/
                                       std::map<uint64_t, TensorVector>& userManagedSections)  /*OUT*/
{
    std::unordered_set<const Tensor*> handledTensors;

    for (const NodePtr& n : g.getExeSortedNodes())
    {
        for (TensorPtr t : n->getOperands())
        {
            GET_REAL_TENSOR_IF_NULL_CONTINUE(t);

            const auto& si = t->getTensorAnnotation().nonPersistentSectionInfo;

            if (si.sectionId.is_set())
            {
                const uint64_t id = si.sectionId.value();

                if (!handledTensors.insert(t.get()).second) continue;

                if (si.bufferingLevel.is_set())
                {
                    // This also makes this pass non-reenterant for multi-buffers.
                    HB_ASSERT(!si.offsetFromBase.is_set(),
                              "Can't set both bufferingLevel and offsetFromBase for tensor \"{}\"!",
                              t->getName());

                    multiBufferSections[id].push_back(t);
                }
                else
                {
                    HB_ASSERT(si.offsetFromBase.is_set(),
                              "Must set bufferingLevel or offsetFromBase when setting sectionId for tensor \"{}\"!",
                              t->getName());
                    userManagedSections[id].push_back(t);
                }
            }
            else
            {
                HB_ASSERT(!si.offsetFromBase.is_set() && !si.bufferingLevel.is_set(),
                          "Can't set offsetFromBase or bufferingLevel without sectionId for tensor \"{}\"!",
                          t->getName());
            }
        }
    }
}

// Decide upon and set offsetFromBase in non-persistent section tensors annotations and
// update the sizeToAllocate for the tensor to include the whole non-persistent section.
static void handleMultiBuffers(HabanaGraph& g, const std::map<uint64_t, TensorVector>& multiBufferSections)
{
    const bool supportDRAM = GCFG_ENABLE_DRAM_MULTI_BUFFERING.value();
    const bool supportSRAM = GCFG_ENABLE_SRAM_MULTI_BUFFERING.value();

    if (multiBufferSections.empty()) return;

    const auto lifetimes = AllLivenessAnalysis(&g).getTensorLifetimeMap();

    for (const auto& mb : multiBufferSections)
    {
        const auto  id      = mb.first;
        const auto& tensors = mb.second;

        TempLogContextSetter logCtx(fmt::format("MultiBuffer_{}", id));

        HB_ASSERT(!tensors.empty(), "MultiBuffer {} contains no tensors", id);

        auto       level = tensors[0]->getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.value();
        const auto loc   = tensors[0]->location();
        HB_ASSERT(level > 0, "Multibuffer {} bufferingLevel is 0", id);
        HB_ASSERT(loc == TensorLocation::TENSOR_IN_DRAM || loc == TensorLocation::TENSOR_IN_SRAM,
                  "Multibuffer tensors location is expected to be either DRAM or SRAM");
        if (loc != TensorLocation::TENSOR_IN_SRAM && !supportDRAM)
        {
            LOG_DEBUG(MEMORY_SECTION, "Skipping multibuffer for buffer id {} as ENABLE_DRAM_MULTI_BUFFERING is unset", id);
            continue;
        }
        if (loc == TensorLocation::TENSOR_IN_SRAM && !supportSRAM)
        {
            LOG_DEBUG(MEMORY_SECTION, "Skipping multibuffer for buffer id {} as ENABLE_SRAM_MULTI_BUFFERING is unset", id);
            continue;
        }

        // Get maxChunkSize and verify consistency within the multibuffer

        uint64_t maxChunkSize = getWriteSpaceForTensor(tensors[0]);
        for (size_t i = 1; i < tensors.size(); ++i)
        {
            maxChunkSize = std::max(maxChunkSize, getWriteSpaceForTensor(tensors[i]));

            const auto newLevel = tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.value();
            HB_ASSERT(level == newLevel,
                      "Multibuffer {} levels - tensor 0: {} and tensor {}: {}",
                      id,
                      level,
                      i,
                      newLevel);
            const auto newLoc = tensors[i]->location();
            HB_ASSERT(loc == newLoc, "Multibuffer {} locations - tensor 0: {} and tensor {}: {}", id, loc, i, newLoc);
        }
        HB_ASSERT(maxChunkSize > 0, "Multibuffer {} with maxChunkSize 0", id);

        if (tensors.size() == level)
        {
            LOG_INFO(MEMORY_SECTION, "{} tensors and {} level is pointless", tensors.size(), level);
        }
        if (tensors.size() < level)
        {
            LOG_WARN(MEMORY_SECTION,
                     "{} tensors and {} level is wasteful; Changing level to {} to save memory",
                     tensors.size(),
                     level,
                     tensors.size());
            level = tensors.size();
        }

        if (LOG_LEVEL_AT_LEAST_TRACE(MEMORY_SECTION))
        {
            std::vector<std::string> lines;
            for (size_t i = 0; i < tensors.size(); ++i)
            {
                lines.push_back(fmt::format("\ttensors[{}] is {} with needed write size: {}",
                                            i,
                                            tensors[i]->getName(),
                                            getWriteSpaceForTensor(tensors[i])));
            }

            LOG_TRACE(MEMORY_SECTION,
                      "loc: {}, level: {} maxChunkSize: {}, decided based on:\n{}",
                      loc == TensorLocation::TENSOR_IN_DRAM ? "DRAM" : "SRAM",
                      level,
                      maxChunkSize,
                      fmt::join(begin(lines), end(lines), "\n"));
        }

        std::vector<int64_t> freeAt(level, -1);
        int                  roundRobinIdx = -1;

        for (const TensorPtr& t : tensors)
        {
            auto& si = t->getTensorAnnotation().nonPersistentSectionInfo;

            // A tensor might be an input to multiple things that we've already
            // allocated so we deliberately only take unique real tensors.
            HB_ASSERT(!si.offsetFromBase.is_set(), "Unique Real Tensors expected");

            auto              idx            = roundRobinIdx;
            const TensorPtr&  realTensor     = Tensor::getRealTensor(t);
            const auto        tensorLifetime = lifetimes.at(realTensor);
            Settable<int64_t> minFreeTime;
            auto              newIdx = idx;

            for (size_t i = 0; i < level; ++i)
            {
                newIdx = (newIdx + 1) % level;
                // In case the tensor can be placed in more than one level -
                // choose the level idx that gets free first to optimize the performance.
                if ((freeAt[newIdx] < tensorLifetime.m_start) &&
                    (!minFreeTime.is_set() || (minFreeTime.value() > freeAt[newIdx])))
                {
                    idx         = newIdx;
                    minFreeTime = freeAt[newIdx];
                }
            }

            // We could assign some slot here and let the sync manager handle this by adding a sync but it means that
            // the multibuffer planning (id assignment) wasn't done correctly.
            HB_ASSERT(minFreeTime.is_set(), // -> a valid index was found
                      "Multibuffer {} has no free slot for tensor {}. lifetime end of slots: [{}], tensor "
                      "lifetime: [{}, {}]",
                      id,
                      t->getName(),
                      fmt::join(begin(freeAt), end(freeAt), ", "),
                      tensorLifetime.m_start,
                      tensorLifetime.m_end);

            freeAt[idx]   = tensorLifetime.m_end;
            roundRobinIdx = (idx + 1) % level;

            const uint64_t offset         = idx * maxChunkSize;
            const uint64_t sizeToAllocate = level * maxChunkSize;
            LOG_DEBUG(MEMORY_SECTION,
                      "Tensor \"{}\" buffer index set to {} (offset {}) and sizeToAllocate to {}",
                      t->getName(),
                      idx,
                      offset,
                      sizeToAllocate);

            // The functions side effect
            si.offsetFromBase.set(offset);
            t->getTensorAnnotation().sizeToAllocate.set(sizeToAllocate);
        }
    }
}

// Calculate allocation size for each tensor in the non-persistent memory section according to tensor's offsets.
static void handleUserManagedSections(HabanaGraph& g, const std::map<uint64_t, TensorVector>& userManagedSections)
{
    for (const auto& section : userManagedSections)
    {
        const auto  id      = section.first;
        const auto& tensors = section.second;

        TempLogContextSetter logCtx(fmt::format("MemorySection_{}", id));

        HB_ASSERT(!tensors.empty(), "MemorySection {} contains no tensors", id);

        // Calcluate allocation size for the memory section:
        // max(T1_size + T1_offset, T2_Size + T2_offset,..., Tn_size + Tn_offset)
        // where T1...Tn tensors in the section
        uint64_t allocationSize           = 0;
        uint64_t minOffsetInMemorySection = std::numeric_limits<uint64_t>::max();
        for (const auto& t : tensors)
        {
            uint64_t currentOffset         = t->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value();
            uint64_t currentAllocationSize = getWriteSpaceForTensor(t) + currentOffset;

            minOffsetInMemorySection = std::min(minOffsetInMemorySection, currentOffset);
            allocationSize           = std::max(allocationSize, currentAllocationSize);
        }
        HB_ASSERT(allocationSize > 0, "MemorySection {} with allocation-size 0", id);

        // Log warning in case the minimal offset is not zero
        if (minOffsetInMemorySection > 0)
        {
            LOG_WARN(MEMORY_SECTION,
                     "MemorySection {} is not optimized, minimal offset is {}",
                     id,
                     minOffsetInMemorySection);
        }

        if (LOG_LEVEL_AT_LEAST_TRACE(MEMORY_SECTION))
        {
            std::vector<std::string> lines;
            for (size_t i = 0; i < tensors.size(); ++i)
            {
                lines.push_back(fmt::format("\ttensors[{}] is {} with offset {} and needed write size: {}",
                                            i,
                                            tensors[i]->getName(),
                                            tensors[i]->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value(),
                                            getWriteSpaceForTensor(tensors[i])));
            }

            LOG_TRACE(MEMORY_SECTION,
                      "loc: {}, allocation-size: {}, decided based on:\n{}",
                      tensors[0]->location() == TensorLocation::TENSOR_IN_DRAM ? "DRAM" : "SRAM",
                      allocationSize,
                      fmt::join(begin(lines), end(lines), "\n"));
        }

        // Update allocation size for the tensors in the section
        for (const TensorPtr& t : tensors)
        {
            auto& si = t->getTensorAnnotation().nonPersistentSectionInfo;

            LOG_DEBUG(MEMORY_SECTION,
                      "Tensor \"{}\" section id: {} offset: {} -> sizeToAllocate set to {}",
                      t->getName(),
                      si.sectionId.value(),
                      si.offsetFromBase.value(),
                      allocationSize);

            t->getTensorAnnotation().sizeToAllocate.set(allocationSize);
        }
    }
}

// This pass goes over all the non-persistent memory sections and for each tensor in a memory section
// assinges allocation size + offset (if not already set by user).
bool setNonPersistentSectionInfo(HabanaGraph& g)
{
    std::map<uint64_t, TensorVector> multiBufferSections;
    std::map<uint64_t, TensorVector> userManagedSections;
    collectMemorySecionTensors(g, multiBufferSections, userManagedSections);

    handleMultiBuffers(g, multiBufferSections);
    handleUserManagedSections(g, userManagedSections);

    return true;
}