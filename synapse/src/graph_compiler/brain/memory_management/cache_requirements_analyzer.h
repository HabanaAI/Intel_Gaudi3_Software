#pragma once

#include "cache_management_apis.h"
#include "memory_usage_db.h"

namespace gc::layered_brain
{
class CacheRequirementsAnalyzer : public CacheRequirementsAnalyzerIfc
{
public:
    CacheRequirementsAnalyzer(const CacheRequirementProfilerPtr& profiler) : m_profiler(profiler) {}

    RequirementDetails inputRequirement(size_t opIdx, size_t inputIdx) const override;
    RequirementDetails outputRequirement(size_t opIdx, size_t outputIdx) const override;

    virtual ~CacheRequirementsAnalyzer() = default;

private:
    CacheRequirementProfilerPtr m_profiler;

    InputCacheUsageProfile  inputProfile(size_t opIdx, size_t inputIdx) const;
    OutputCacheUsageProfile outputProfile(size_t opIdx, size_t outputIdx) const;

    RequirementDetails inputRequirement(const InputCacheUsageProfile& profile) const;
    RequirementDetails bundleInputRequirement(const InputCacheUsageProfile& profile) const;
    RequirementDetails intermediateConsumerRequirement(const InputCacheUsageProfile& profile) const;

    bool numReadsOverride(const InputCacheUsageProfile& profile) const;

    RequirementDetails outputRequirement(const OutputCacheUsageProfile& profile) const;

    void setInputRelease(RequirementDetails& req, const InputCacheUsageProfile& profile) const;
    void setOutputRelease(RequirementDetails& req, const OutputCacheUsageProfile& profile) const;
};

}  // namespace gc::layered_brain