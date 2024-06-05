#include "cache_requirements_analyzer.h"
#include "layered_brain.h"
#include "brain_conf.h"

using namespace gc::layered_brain;

CacheRequirementsAnalyzerIfc::RequirementDetails CacheRequirementsAnalyzer::inputRequirement(size_t opIdx,
                                                                                             size_t inputIdx) const
{
    const auto profile = inputProfile(opIdx, inputIdx);
    profile.log();
    auto req = inputRequirement(profile);
    setInputRelease(req, profile);
    return req;
}

InputCacheUsageProfile CacheRequirementsAnalyzer::inputProfile(size_t opIdx, size_t inputIdx) const
{
    return m_profiler->inputProfile(opIdx, inputIdx);
}

CacheRequirementsAnalyzerIfc::RequirementDetails
CacheRequirementsAnalyzer::inputRequirement(const InputCacheUsageProfile& profile) const
{
    if (numReadsOverride(profile))
    {
        return RequirementDetails::allocDH(profile.size);
    }
    if (profile.produced)
    {
        return intermediateConsumerRequirement(profile);
    }
    return bundleInputRequirement(profile);
}

// List of scenarios where the number of reads may be inaccurate or irrelevant and so should be ignored.
bool CacheRequirementsAnalyzer::numReadsOverride(const InputCacheUsageProfile& profile) const
{
    if (profile.size <= GCFG_SMALL_INPUT_FORCE_CACHING_MAX_SIZE_BYTES.value())
    {
        return true;
    }
    if (profile.allRequired && profile.size < GCFG_ALL_REQUIRED_INPUT_CACHING_MAX_SIZE_BYTES.value())
    {
        return true;
    }
    return false;
}

CacheRequirementsAnalyzerIfc::RequirementDetails
CacheRequirementsAnalyzer::bundleInputRequirement(const InputCacheUsageProfile& profile) const
{
    HB_ASSERT(!numReadsOverride(profile), "Unexpected input profile requires not to be based on no. of reads.");

    /*
    From the spec:
    - Input from HBM
        * Single read: no alloc, class=00
        * Single read per dcore, shared between dcores: allocH, class=10
        * Multiple reads per dcore, shared between dcores: allocDH, class=10
        * Multiple reads per dcore, no shared: allocD, class=10
    */

    const bool cacheDueToMultipleConsumers = GCFG_ENABLE_LB_CACHE_REUSED_SLICES.value() && profile.nofConsumers > 1;
    if (profile.totalReads == 1 && !cacheDueToMultipleConsumers)
    {
        return RequirementDetails::noAlloc();
    }
    if (profile.dcoreReads == 1)
    {
        return RequirementDetails::allocH(profile.size);
    }
    if (profile.totalReads > profile.dcoreReads)
    {
        return RequirementDetails::allocDH(profile.size);
    }
    return RequirementDetails::allocD(profile.size);
}

CacheRequirementsAnalyzerIfc::RequirementDetails
CacheRequirementsAnalyzer::intermediateConsumerRequirement(const InputCacheUsageProfile& profile) const
{
    HB_ASSERT(!numReadsOverride(profile), "Unexpected input profile requires not to be based on no. of reads.");

    /*
    From the spec:
    - Ephemeral (producer/consumer pipelined)
        * If producer and consumer use same perforation: allocD, class=10
        * If producer and consumer use different perforation:
            > Write: ...
            > Read:
                - If total reads=dcore reads then allocD, class=10
                - If total reads>dcore reads then allocDH, class=10
    */
    if (profile.localized && !profile.allRequired)
    {
        return RequirementDetails::allocD(profile.size);
    }
    else
    {
        if (profile.dcoreReads == 1)
        {
            return RequirementDetails::allocH(profile.size);
        }
        else if (profile.totalReads == profile.dcoreReads)
        {
            return RequirementDetails::allocD(profile.size);
        }
        else
        {
            return RequirementDetails::allocDH(profile.size);
        }
    }
}

void CacheRequirementsAnalyzer::setInputRelease(RequirementDetails& req, const InputCacheUsageProfile& profile) const
{
    req.release    = RequirementDetails::ReleaseType::NONE;  // default
    req.postAccess = profile.lastConsumer ? RequirementDetails::PostAccessAction::RELEASE
                                          : RequirementDetails::PostAccessAction::NONE;

    if (!profile.bpt && profile.lastConsumer && GCFG_ENABLE_LB_NON_BPT_SLICES_DISCARDING.value())
    {
        // Non-BPT can be discarded (may be overridden) after the last read. Discarding the data saves the BW
        // required for a write-back to HBM of useless data. Only a CME can perform this action.
        req.release = RequirementDetails::ReleaseType::DISCARD_CME;
    }
    else
    {
        // BPT data should outlive the bundle or ephemeral that should not be discarded yet. In case of ephemeral, the
        // release details are set in case yielding it is required.
        if (profile.totalReads == 1 && !numReadsOverride(profile))
        {
            // When the data is read once by a single engine, that engine can reduce the cache class while reading
            // the data. This saves the resources needed for the cache maintenance engine (CME) to perform this
            // task.
            req.release = RequirementDetails::ReleaseType::DEGRADE_CLASS;
        }
        else
        {
            // When the data is read multiple times, if the first reader would reduce the class, an eviction of the
            // data from the cache becomes more likely and may cause threshing or increase BW to a farther memory.
            // In this case, we need to use the CME to perform the degradation after all the reads are over.
            req.release = RequirementDetails::ReleaseType::DEGRADE_CME;
        }
    }
}

CacheRequirementsAnalyzerIfc::RequirementDetails CacheRequirementsAnalyzer::outputRequirement(size_t opIdx,
                                                                                              size_t outputIdx) const
{
    const auto profile = outputProfile(opIdx, outputIdx);
    profile.log();
    auto req = outputRequirement(profile);
    setOutputRelease(req, profile);
    return req;
}

OutputCacheUsageProfile CacheRequirementsAnalyzer::outputProfile(size_t opIdx, size_t outputIdx) const
{
    return m_profiler->outputProfile(opIdx, outputIdx);
}

CacheRequirementsAnalyzerIfc::RequirementDetails
CacheRequirementsAnalyzer::outputRequirement(const OutputCacheUsageProfile& profile) const
{
    /*
    From the spec:
    - Ephemeral (producer/consumer pipelined)
      * If producer and consumer use same perforation: allocD, class=10
      * If producer and consumer use different perforation:
        > Write: allocH, class=10
        > Read:...
    - RMW output
      * AllocD, class=11 (must not evict)
    - Bundle output
      * No allocation , class=00
    */
    if (profile.rmw)
    {
        auto req = RequirementDetails::allocD(profile.size);
        // Can't use class degrade for RMW, since multiple engines may be writing the same data in parallel.
        // Release of RMW cache will always be through CME because of this, and the class would always be Top.
        req.cacheClass = CacheClass::Top;
        return req;
    }
    if (profile.hasConsumers)
    {
        if (profile.localized)
        {
            return RequirementDetails::allocD(profile.size);
        }
        else
        {
            return RequirementDetails::allocH(profile.size);
        }
    }
    return RequirementDetails::noAlloc();
}

void CacheRequirementsAnalyzer::setOutputRelease(RequirementDetails& req, const OutputCacheUsageProfile& profile) const
{
    req.release = RequirementDetails::ReleaseType::NONE;            // default
    req.postAccess = RequirementDetails::PostAccessAction::NONE;  // default

    if (profile.rmw && profile.lastRmwWriter && !profile.hasConsumers)
    {
        // The only case of releasing an output is when writing a bundle output (without consumers) with RMW. The last
        // writer needs to trigger a CME and can't degrade the class itself, since it may have multiple engines writing
        // at the same time.
        req.release = RequirementDetails::ReleaseType::DEGRADE_CME;
        req.postAccess = RequirementDetails::PostAccessAction::RELEASE;
    }
}