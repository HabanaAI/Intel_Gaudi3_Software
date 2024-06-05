#pragma once

#include "compilation_hal_reader.h"
#include "cache_types.h"

namespace gc::layered_brain
{
struct CacheRequirementProfilerIfc
{
    struct InputCacheUsageProfile
    {
        size_t totalReads = 1;
        size_t dcoreReads = 1;

        uint64_t size = 0;

        uint32_t nofConsumers = 1;  // There will always be at least one - the one triggered the input profiling.

        bool produced                  = false;
        // Indicates that the producer is perforated on the same dim and one of this tensor dims is mapped to it.
        bool localized                 = false;
        bool lastConsumer              = false;
        bool allRequired               = false;
        bool bpt                       = false;

        void log() const;
    };

    struct OutputCacheUsageProfile
    {
        uint64_t size = 0;

        bool rmw                            = false;
        bool lastRmwWriter                  = false;
        bool hasConsumers                   = false;
        // Indicates that the first consumer is perforated on the same dim and one of this tensor dims is mapped to it.
        bool localized = false;

        void log() const;
    };

    // Profile how the node with bundle operation index (schedule) 'opIdx' is accessing its 'inputIdx' input
    virtual InputCacheUsageProfile inputProfile(size_t opIdx, size_t inputIdx) = 0;

    // Profile how the node with bundle operation index (schedule) 'opIdx' is accessing its 'outputIdx' output
    virtual OutputCacheUsageProfile outputProfile(size_t opIdx, size_t outputIdx) = 0;

    virtual ~CacheRequirementProfilerIfc() = default;
};
using CacheRequirementProfilerPtr = std::shared_ptr<CacheRequirementProfilerIfc>;
using InputCacheUsageProfile      = CacheRequirementProfilerIfc::InputCacheUsageProfile;
using OutputCacheUsageProfile     = CacheRequirementProfilerIfc::OutputCacheUsageProfile;

struct CacheRequirementsAnalyzerIfc
{
    struct RequirementDetails
    {
        // What to do with the data after the cache was accessed
        enum class PostAccessAction
        {
            RELEASE,  // Data is no longer needed.
            NONE,     // Data will be needed.
        };

        // How to release or yield the data
        enum class ReleaseType
        {
            DEGRADE_CME,
            DISCARD_CME,
            DEGRADE_CLASS,
            NONE,
        };

        uint64_t         capacity   = 0;
        CacheDirective   directive  = CacheDirective::NoAllocate;
        CacheClass       cacheClass = CacheClass::Low;
        PostAccessAction postAccess = PostAccessAction::NONE;
        ReleaseType      release    = ReleaseType::NONE;

        bool cachingRequired() const { return directive != CacheDirective::NoAllocate; }
        bool releaseRequired() const { return postAccess == PostAccessAction::RELEASE; }
        bool yieldAllowed() const { return cachingRequired() && !releaseRequired(); }

        CacheMaintenanceAction cmAction() const
        {
            switch (release)
            {
                case ReleaseType::DEGRADE_CME:
                    return CacheMaintenanceAction::DEGRADE;
                case ReleaseType::DISCARD_CME:
                    return CacheMaintenanceAction::DISCARD;
                default:
                    return CacheMaintenanceAction::NOP;
            }
        }

        CacheDirective releaseCacheDirective() const
        {
            // When releasing using the read class, there is no point in a directive that cache the data. If the data is
            // already in the cache, it will be read from there regardless of the directive. If it is not, caching it
            // may evict other entries and there will not be any cache reuse gain to offset this cost (DEGRADE_CLASS is
            // only selected when there is no cache reuse)
            return release == ReleaseType::DEGRADE_CLASS ? CacheDirective::NoAllocate : directive;
        }

        CacheClass releaseCacheClass() const
        {
            return release == ReleaseType::DEGRADE_CLASS ? CacheClass::Low : cacheClass;
        }

        static RequirementDetails noAlloc()
        {
            return RequirementDetails {.capacity   = 0,
                                       .directive  = CacheDirective::NoAllocate,
                                       .cacheClass = CacheClass::Low,
                                       .postAccess = PostAccessAction::NONE,
                                       .release    = ReleaseType::NONE};
        }

        static RequirementDetails allocH(uint64_t capacity)
        {
            return RequirementDetails {.capacity   = capacity,
                                       .directive  = CacheDirective::HomeAllocate,
                                       .cacheClass = CacheClass::High,
                                       .postAccess = PostAccessAction::NONE,
                                       .release    = ReleaseType::NONE};
        }

        static RequirementDetails allocD(uint64_t capacity)
        {
            auto details      = RequirementDetails::allocH(capacity);
            details.directive = CacheDirective::DcoreAllocate;
            return details;
        }

        static RequirementDetails allocDH(uint64_t baseCapacity)
        {
            const uint64_t numDCores = CompilationHalReader::isHalReaderSet()
                                           ? CompilationHalReader::getHalReader()->getNumDcores()
                                           : 4;

            auto details      = RequirementDetails::allocH(baseCapacity * numDCores);
            details.directive = CacheDirective::SharedAllocate;
            return details;
        }
    };

    // List the cache requirements for the node with bundle schedule 'opIdx' when accessing input 'inputIdx'
    virtual RequirementDetails inputRequirement(size_t opIdx, size_t inputIdx) const = 0;

    // List the cache requirements for the node with bundle schedule 'opIdx' when accessing output 'outputIdx'
    virtual RequirementDetails outputRequirement(size_t opIdx, size_t outputIdx) const = 0;

    virtual ~CacheRequirementsAnalyzerIfc() = default;
};

struct NodeCacheSetterIfc
{
    // Set a node's cache directive and return whether all accesses succeeded as required.
    virtual bool setDirectives(size_t nodeIdx, CacheRequirementsAnalyzerIfc* requirementAnalyzer) = 0;

    virtual ~NodeCacheSetterIfc() = default;
};

}  // namespace gc::layered_brain