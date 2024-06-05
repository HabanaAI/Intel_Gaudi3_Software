#include "non_persistent_section_util.h"

#include "allocators_utils.h"
#include "infra/defs.h"
#include "graph_annotation.h"
#include "habana_graph.h"
#include "liveness_analysis.h"
#include "tensor.h"

#include <algorithm>

NonPersistentSectionAllocTracker::NonPersistentSectionAllocTracker(HabanaGraph& g, bool sram)
{
    std::map<uint64_t, TensorSet> id2Tensors;
    for (auto t : g.getTensors())
    {
        t = Tensor::getRealTensor(t);

        const auto& si = t->getTensorAnnotation().nonPersistentSectionInfo;

        HB_ASSERT(!si.offsetFromBase.is_set() || si.sectionId.is_set(),
                  "{}: tensor \"{}\" from non-persistent section has an offset without an id",
                  __func__,
                  t->getName());

        if (sram != t->inSram()) continue;

        // Note that offsetFromBase has to be used as a filter since a tensor might have non-persistent section id/level set for it
        // but still not be placed if the setNonPersistentSectionInfo pass deems it as unapropriate as a result of a global
        // config like GCFG_ENABLE_DRAM_MULTI_BUFFERING being set to false.
        if (si.offsetFromBase.is_set())
        {
            id2Tensors[si.sectionId.value()].insert(t);
        }
    }

    for (const auto& v : id2Tensors)
    {
        const uint64_t id      = v.first;
        const auto&    tensors = v.second;

        m_nonPersistentSectionId2desc.emplace(id, NonPersistentSectionDesc({begin(tensors), end(tensors)}));
        for (const TensorPtr& t : tensors)
        {
            m_tensor2status.emplace(t, TensorStatus::unallocated);
        }
    }

    tracePrint(fmt::format("{} c'tor", __func__), nullptr, true);
}

NonPersistentSectionAllocTracker::~NonPersistentSectionAllocTracker()
{
    tracePrint(__func__, nullptr, false);
}

void NonPersistentSectionAllocTracker::setSectionBaseAddr(uint64_t sectionId, uint64_t addr)
{
    LOG_TRACE(MEMORY_SECTION, "{}: Section ID {} base addr set to {}", HLLOG_FUNC, sectionId, addr);

    auto it = m_nonPersistentSectionId2desc.find(sectionId);
    HB_ASSERT(it != m_nonPersistentSectionId2desc.end(), "{}: Section ID {} not found!", __func__, sectionId);
    auto& desc = it->second;

    // validate expected state
    if (addr != -1)
    {
        HB_ASSERT(desc.addr == -1, "{}: Addr of section {} is already set to {}!", __func__, sectionId, desc.addr);
        HB_ASSERT(desc.totalAllocated == 1 && desc.totalFreed == 0,
                  "{}: Addr was unset with totalAllocated={} and totalFreed={}!",
                  __func__,
                  desc.totalAllocated,
                  desc.totalFreed);
    }
    else  // rollback - all tensors should be moved from "allocated" and "freed" to "unallocated" state
    {
        HB_ASSERT(desc.addr != -1, "{}: Addr is already unset!", __func__);
        HB_ASSERT(desc.totalAllocated == 0 && desc.totalFreed == 0,
                  "{}: Unsetting addr with totalAllocated={} and totalFreed={}!",
                  __func__,
                  desc.totalAllocated,
                  desc.totalFreed);
    }
    desc.addr = addr;
}

bool NonPersistentSectionAllocTracker::markAsAlloc(const TensorPtr& tensor, /*OUT*/ size_t& offset)
{
    auto& status = getTensorStatus(tensor);
    HB_ASSERT(status != TensorStatus::freed,
              "{}: tensor \"{}\" allocated after free without rollback!",
              __func__,
              tensor->getName());

    bool firstTensorAllocated = false;

    if (status == TensorStatus::allocated)
    {
        offset = tensor->getTensorOffset();
        LOG_TRACE(MEMORY_SECTION,
                  "{}: tensor \"{}\" is already present, offset (0x{:x}) should remain unmodified.",
                  HLLOG_FUNC,
                  tensor->getName(),
                  offset);
    }
    else
    {
        HB_ASSERT(status == TensorStatus::unallocated, "");

        const auto&      si  = tensor->getTensorAnnotation().nonPersistentSectionInfo;
        const auto       id   = si.sectionId.value();
        NonPersistentSectionDesc& desc = m_nonPersistentSectionId2desc.at(id);

        if (desc.totalAllocated == 0 && desc.totalFreed == 0)  // not yet allocated
        {
            firstTensorAllocated = true;
            offset               = -1;
            LOG_TRACE(MEMORY_SECTION,
                      "{}: tensor \"{}\" is the first to be alocated within non-persistent section {}",
                      HLLOG_FUNC,
                      tensor->getName(),
                      id);
        }
        else
        {
            const auto sectionAddr    = desc.addr;
            const auto offsetFromBase = si.offsetFromBase.value();
            offset                    = sectionAddr + offsetFromBase;
            LOG_TRACE(MEMORY_SECTION,
                      "{}: tensor \"{}\" offset should be set to 0x{:x} (0x{:x} + 0x{:x})",
                      HLLOG_FUNC,
                      tensor->getName(),
                      offset,
                      sectionAddr,
                      offsetFromBase);
        }

        HB_ASSERT(desc.totalUnallocated > 0, "");
        --desc.totalUnallocated;
        ++desc.totalAllocated;
    }

    status = TensorStatus::allocated;
    return firstTensorAllocated;
}

bool NonPersistentSectionAllocTracker::markAsFree(const TensorPtr& tensor, bool rollback)
{
    auto& status = getTensorStatus(tensor);

    const auto       id   = tensor->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
    NonPersistentSectionDesc& desc = m_nonPersistentSectionId2desc.at(id);

    switch (status)
    {
        case TensorStatus::allocated:
            HB_ASSERT(desc.totalAllocated > 0, "");
            --desc.totalAllocated;
            break;
        case TensorStatus::freed:
            // Note that the status should normally be only allocated but this may be
            // used in TensorsEpochAllocator to free a previous bundles tensors which
            // was already freed
            //TODO: [SW-28478]
            HB_ASSERT(desc.totalFreed > 0, "");
            --desc.totalFreed;
            break;
        default:
            HB_ASSERT(false,
                      "{}: tensor \"{}\" freed with status ({}) isn't allocated ({}) nor freed ({})!",
                      __func__,
                      tensor->getName(),
                      static_cast<int>(status),
                      static_cast<int>(TensorStatus::allocated),
                      static_cast<int>(TensorStatus::freed));
    }

    if (rollback)
    {
        ++desc.totalUnallocated;
        status = TensorStatus::unallocated;
    }
    else
    {
        ++desc.totalFreed;
        status = TensorStatus::freed;
    }

    const bool shouldFree = desc.totalAllocated == 0 && (desc.totalFreed == 0 || desc.totalUnallocated == 0);
    return shouldFree;
}

void NonPersistentSectionAllocTracker::handleUnallocatedTensorFallback(const TensorPtr& tensor)
{
    const auto& si = tensor->getTensorAnnotation().nonPersistentSectionInfo;
    if (!si.offsetFromBase.is_set()) return;

    auto& status = getTensorStatus(tensor);
    switch (status)
    {
        case TensorStatus::unallocated:
            break;
        case TensorStatus::freed:
        {
            NonPersistentSectionDesc& desc = m_nonPersistentSectionId2desc.at(si.sectionId.value());
            HB_ASSERT(desc.totalFreed > 0, "");
            --desc.totalFreed;
            ++desc.totalUnallocated;
            if (desc.totalFreed == 0 && desc.totalAllocated == 0)
            {
                // need to reset section address as part of resetting the state
                setSectionBaseAddr(si.sectionId.value(), (uint64_t)-1);
            }
            status = TensorStatus::unallocated;
            break;
        }
        default:
            HB_ASSERT(false,
                      "{}: tensor \"{}\" with status ({}) which isn't unallocated ({}) nor freed ({})!",
                      __func__,
                      tensor->getName(),
                      static_cast<int>(status),
                      static_cast<int>(TensorStatus::unallocated),
                      static_cast<int>(TensorStatus::freed));
    }
}

template<class C, class V>
static bool contains(const C& c, const V& v)
{
    return std::find(begin(c), end(c), v) != end(c);
}

bool NonPersistentSectionAllocTracker::trackPlannedNonPersistentSectionTensor(const TensorPtr& tensor)
{
    const auto& si = tensor->getTensorAnnotation().nonPersistentSectionInfo;
    if (!si.offsetFromBase.is_set())
    {
        LOG_TRACE(MEMORY_SECTION, "{}: tensor \"{}\" isn't part of a non-persistent section.", HLLOG_FUNC, tensor->getName());
        return false;
    }

    // Note that this is done before checking m_currentPlannedNonPersistentSectionIds to assert it's unallocated
    const auto status = getTensorStatus(tensor);
    HB_ASSERT(status == TensorStatus::unallocated,
              "{}: called for tensor \"{}\" with status ({}) != unallocated ({})",
              __func__,
              tensor->getName(),
              static_cast<int>(status),
              static_cast<int>(TensorStatus::unallocated));

    const auto id = si.sectionId.value();
    if (contains(m_currentPlannedNonPersistentSectionIds, id))
    {
        LOG_TRACE(MEMORY_SECTION,
                  "- {}: {} is reusing allocated or planned for the current epoch non-persistent section, requiring no "
                  "additional write space.",
                  HLLOG_FUNC,
                  tensor->getName());
        return true;
    }

    // Note that we don't mind adding even if it was allocated etc.
    m_currentPlannedNonPersistentSectionIds.push_back(id);

    const NonPersistentSectionDesc& desc = m_nonPersistentSectionId2desc.at(id);
    HB_ASSERT(desc.totalUnallocated > 0, "");

    if (desc.totalAllocated || desc.totalFreed)
    {
        LOG_TRACE(MEMORY_SECTION,
                  "{}: tensor \"{}\" is part of non-persistent section #{} which IS already allocated.",
                  HLLOG_FUNC,
                  tensor->getName(),
                  id);
        return true;
    }

    LOG_TRACE(MEMORY_SECTION,
              "{}: tensor \"{}\" is part of non-persistent section #{} which IS NOT currently allocated, yet.",
              HLLOG_FUNC,
              tensor->getName(),
              id);
    return false;
}

void NonPersistentSectionAllocTracker::discardPlannedTensors()
{
    m_currentPlannedNonPersistentSectionIds.clear();
}

void NonPersistentSectionAllocTracker::tracePrint(const std::string& label, const bool* res, bool showAll) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(MEMORY_SECTION)) return;

    std::stringstream ss;

    {
        const auto resLabel = res != nullptr ? fmt::format(" => res={}", *res) : "";
        const auto summary  = [&] {
            std::array<size_t, 3> overallStateCount {};
            for (const auto& v : m_nonPersistentSectionId2desc)
            {
                overallStateCount[0] += v.second.totalUnallocated;
                overallStateCount[1] += v.second.totalAllocated;
                overallStateCount[2] += v.second.totalFreed;
            }
            return fmt::format("[{},{},{}] ::", overallStateCount[0], overallStateCount[1], overallStateCount[2]);
        }();

        if (m_nonPersistentSectionId2desc.size() > 1)
        {
            ss << fmt::format("{}{}: {}\n", label, resLabel, summary);
        }
        else if (m_nonPersistentSectionId2desc.size() == 1)
        {
            ss << fmt::format("{}{}: {} ", label, resLabel, summary);
        }
        else
        {
            ss << fmt::format("{}{}: Empty!\n", label, resLabel);
        }
    }

    std::map<uint64_t, NonPersistentSectionDesc> sortedNonPersistentSectionId2desc(
        m_nonPersistentSectionId2desc.begin(),
        m_nonPersistentSectionId2desc.end());  // Sort by section ID
    for (const auto& v : sortedNonPersistentSectionId2desc)
    {
        const NonPersistentSectionDesc& desc = v.second;
        if (!showAll && desc.totalAllocated == 0 && desc.totalFreed == 0) continue;

        ss << fmt::format("\tnon-persistent section id #{}: [0:{}, 1:{} 2:{}] => [ ",
                          v.first,
                          desc.totalUnallocated,
                          desc.totalAllocated,
                          desc.totalFreed);

        const bool allSame = !!desc.totalUnallocated + !!desc.totalAllocated + !!desc.totalFreed == 1;
        if (!showAll && allSame)
        {
            ss << "All " << (desc.totalUnallocated != 0 ? '0' : desc.totalAllocated != 0 ? '1' : '2') << ' ';
        }
        else
        {
            for (const TensorPtr& t : desc.tensors)
            {
                if (showAll || desc.totalAllocated != 0 || desc.totalFreed != 0)
                {
                    ss << fmt::format("{{\"{}\": {}}}, ", t->getName(), static_cast<int>(m_tensor2status.at(t)));
                }
            }
        }
        ss << fmt::format("]\n");
    }
    const auto& str = ss.str();
    LOG_TRACE(MEMORY_SECTION, "{}", str);
}

NonPersistentSectionAllocTracker::NonPersistentSectionDesc::NonPersistentSectionDesc(
    TensorVector nonPersistentSectionTensors)
: totalUnallocated(nonPersistentSectionTensors.size()), tensors(std::move(nonPersistentSectionTensors))
{
}

bool NonPersistentSectionAllocTracker::NonPersistentSectionDesc::operator==(const NonPersistentSectionDesc& o) const
{
    return std::make_tuple(totalUnallocated, totalAllocated, totalFreed, tensors) ==
           std::make_tuple(o.totalUnallocated, o.totalAllocated, o.totalFreed, o.tensors);
}

bool NonPersistentSectionAllocTracker::verifyDataInSync() const
{
    std::unordered_map<uint64_t, NonPersistentSectionDesc> sanity;
    for (const auto& v : m_tensor2status)
    {
        const TensorPtr&   t      = v.first;
        const TensorStatus status = v.second;

        const uint64_t   id   = t->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
        NonPersistentSectionDesc& desc = sanity[id];
        switch (status)
        {
            case TensorStatus::unallocated:
                ++desc.totalUnallocated;
                break;
            case TensorStatus::allocated:
                ++desc.totalAllocated;
                break;
            case TensorStatus::freed:
                ++desc.totalFreed;
                break;
        }
        desc.tensors.push_back(t);
    }
    for (auto& desc : sanity)
    {
        std::sort(begin(desc.second.tensors), end(desc.second.tensors), TensorComparator());
    }
    return sanity == m_nonPersistentSectionId2desc;
}

void NonPersistentSectionAllocTracker::verifyAllDone() const
{
    HB_ASSERT(verifyDataInSync(), "Non persistent section consistency check failed");

    for (const auto& v : m_nonPersistentSectionId2desc)
    {
        const NonPersistentSectionDesc& desc = v.second;
        HB_ASSERT(desc.totalAllocated == 0 && ((desc.totalUnallocated == 0) != (desc.totalFreed == 0)),
                  "{}: non-persistent section: {} totals are: {{ unallocated: {}, allocated: {}, freed: {} }}, whereas all freed "
                  "(For handled non-persistent section) or all unallocated was expected.",
                  __func__,
                  v.first,
                  desc.totalUnallocated,
                  desc.totalAllocated,
                  desc.totalFreed);
        HB_ASSERT((desc.addr == -1) == (desc.totalFreed == 0),
                  "{}: addr=0x{:x} should be -1 iff no tensors were allocated but totalFreed is {}.",
                  __func__,
                  desc.addr,
                  desc.totalFreed);
    }
}

const NonPersistentSectionAllocTracker::TensorStatus& NonPersistentSectionAllocTracker::getTensorStatus(const TensorPtr& t) const
{
    const auto& si = t->getTensorAnnotation().nonPersistentSectionInfo;
    HB_ASSERT(si.offsetFromBase.is_set(), "{}: tensor \"{}\" isn't part of a non-persistent section", __func__, t->getName());

    const auto it = m_tensor2status.find(t);
    HB_ASSERT(it != end(m_tensor2status), "{}: unknown tensor \"{}\"", __func__, t->getName());

    return it->second;
}

NonPersistentSectionAllocTracker::TensorStatus& NonPersistentSectionAllocTracker::getTensorStatus(const TensorPtr& t)
{
    return const_cast<TensorStatus&>(const_cast<const NonPersistentSectionAllocTracker&>(*this).getTensorStatus(t));
}