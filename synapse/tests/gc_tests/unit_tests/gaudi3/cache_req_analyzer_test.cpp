#include "brain_conf.h"
#include "cache_requirements_analyzer.h"
#include "../gaudi2/layered_brain_test.h"
#include "hal_reader/gaudi3/hal_reader.h"

/*
Validate from the spec:
- Input from HBM
  * Single read: no alloc, class=00
  * Single read per dcore, shared between dcores: allocH, class=10
  * Multiple reads per dcore, shared between dcores: allocDH, class=10
  * Multiple reads per dcore, no shared: allocD, class=10
- Input embeddings (very large all-required tensor, sparse access)
  * NO SUPPORT YET - will be added if real use case is encountered
  * allocH, Class=01
    note: we must ensure this content does not remove any other planned capacity content. Should we use noalloc?
Mostly, we will not have enough capacity to store large amount of this content.
- TPC all-required input tensor
  * allocDH, class=10 note: need to evaluate demotion of allocDH to allocH when BW is “low enough”, per kernel cost
    model.
- Ephemeral (producer/consumer pipelined)
  * If producer and consumer use same perforation: allocD, class=10
  * If producer and consumer use different perforation:
    > Write: allocH, class=10
    > Read:
      - If total reads=dcore reads then allocD, class=10
      - If total reads>dcore reads then allocDH, class=10
- RMW output
  * AllocD, class=11 (must not evict)
- Bundle output
  * No allocation , class=00

*/

class CacheRequirementsAnalyzerTest : public LayeredBrainTest
{
public:
    using RD = CacheRequirementsAnalyzerIfc::RequirementDetails;

    static constexpr uint64_t CAP = 1000;

    class PreDefinedInputProfiler : public CacheRequirementProfilerIfc
    {
    public:
        PreDefinedInputProfiler(const InputCacheUsageProfile& inputProfile) : m_profile(inputProfile) {}

        InputCacheUsageProfile  inputProfile(size_t opIdx, size_t inputIdx) override { return m_profile; }
        OutputCacheUsageProfile outputProfile(size_t opIdx, size_t outputIdx) override { return {}; }

    private:
        InputCacheUsageProfile m_profile;
    };

    CacheRequirementProfilerPtr getInputCacheProfiler(const InputCacheUsageProfile& profile)
    {
        return CacheRequirementProfilerPtr(new PreDefinedInputProfiler(profile));
    }

    class PreDefinedOutputProfiler : public CacheRequirementProfilerIfc
    {
    public:
        PreDefinedOutputProfiler(const OutputCacheUsageProfile& outputProfile) : m_profile(outputProfile) {}

        InputCacheUsageProfile  inputProfile(size_t opIdx, size_t inputIdx) override { return {}; }
        OutputCacheUsageProfile outputProfile(size_t opIdx, size_t outputIdx) override { return m_profile; }

    private:
        OutputCacheUsageProfile m_profile;
    };

    CacheRequirementsAnalyzerTest()
    {
        // This is required so that allocDH capacity requirement will be calculated by Gaudi3's no. of DCOREs.
        CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
    }

    void SetUp() override
    {
        LayeredBrainTest::SetUp();
        setGlobalConfForTest(GCFG_SMALL_INPUT_FORCE_CACHING_MAX_SIZE_BYTES, "0");
        setGlobalConfForTest(GCFG_ENABLE_LB_CACHE_REUSED_SLICES, "True");
    }

    CacheRequirementProfilerPtr getOutputCacheProfiler(const OutputCacheUsageProfile& profile) const
    {
        return CacheRequirementProfilerPtr(new PreDefinedOutputProfiler(profile));
    }

    void validateNoAlloc(const RD& req) const
    {
        EXPECT_EQ(CacheDirective::NoAllocate, req.directive);
        EXPECT_EQ(CacheClass::Low, req.cacheClass);
        EXPECT_EQ(0, req.capacity);
    }

    void validateAlloc(const RD& req, CacheDirective directive, CacheClass cacheClass, uint64_t capacity) const
    {
        EXPECT_EQ(directive, req.directive);
        EXPECT_EQ(cacheClass, req.cacheClass);
        EXPECT_EQ(capacity, req.capacity);
    }

    void validateAllocD(const RD& req, CacheClass cls = CacheClass::High) const
    {
        validateAlloc(req, CacheDirective::DcoreAllocate, cls, CAP);
    }

    void validateAllocH(const RD& req, CacheClass cls = CacheClass::High) const
    {
        validateAlloc(req, CacheDirective::HomeAllocate, cls, CAP);
    }

    void validateAllocDH(const RD& req, CacheClass cls = CacheClass::High) const
    {
        validateAlloc(req, CacheDirective::SharedAllocate, cls, 4 * CAP);
    }

    void validateNoRelease(const RD& req) const { EXPECT_EQ(RD::PostAccessAction::NONE, req.postAccess); }

    void validateRelease(const RD& req, RD::ReleaseType releaseType) const
    {
        EXPECT_EQ(RD::PostAccessAction::RELEASE, req.postAccess);
        EXPECT_EQ(releaseType, req.release);
    }

    void validateYield(const RD& req, RD::ReleaseType releaseType) const
    {
        EXPECT_EQ(RD::PostAccessAction::NONE, req.postAccess);
        EXPECT_EQ(releaseType, req.release);
    }
};

TEST_F(CacheRequirementsAnalyzerTest, cra_should_require_no_alloc_for_singly_read_inputs)
{
    createGraph(1);

    // Given profile of a bundle input that is read only once
    InputCacheUsageProfile profile {};
    profile.produced   = false;
    profile.totalReads = 1;
    profile.dcoreReads = 1;
    profile.size       = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect no allocation
    validateNoAlloc(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_singly_read_input_with_multiple_consumers)
{
    createGraph(1);

    // Given profile of a bundle input that is read only once
    InputCacheUsageProfile profile {};
    profile.produced     = false;
    profile.totalReads   = 1;
    profile.dcoreReads   = 1;
    profile.nofConsumers = 5;
    profile.size         = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocH
    validateAllocH(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_single_read_per_dcore_shared_inputs)
{
    createGraph(1);

    // Given a profile of a bundle input where each datum is read once from each dcore
    InputCacheUsageProfile profile {};
    profile.produced   = false;
    profile.totalReads = 4;
    profile.dcoreReads = 1;
    profile.size       = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocH
    validateAllocH(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_multiple_reads_per_dcore_inputs)
{
    createGraph(1);

    // Given a profile of a bundle input where each datum is read multiple times from a single dcore
    InputCacheUsageProfile profile {};
    profile.produced   = false;
    profile.totalReads = 2;
    profile.dcoreReads = 2;
    profile.size       = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD
    validateAllocD(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_multiple_reads_per_dcore_shared_inputs)
{
    createGraph(1);

    // Given a profile of a bundle input where each datum is read multiple times from multiple dcores
    InputCacheUsageProfile profile {};
    profile.produced   = false;
    profile.totalReads = 8;
    profile.dcoreReads = 2;
    profile.size       = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocDH
    validateAllocDH(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_all_required_inputs)
{
    createGraph(1);

    // Given a profile of an all required bundle input (no real data on number of reads), with small size
    InputCacheUsageProfile profile {};
    profile.produced    = false;
    profile.allRequired = true;
    profile.size        = CAP;
    profile.dcoreReads  = 5;
    profile.totalReads  = 5;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocDH
    validateAllocDH(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_all_required_big_inputs)
{
    createGraph(1);

    // Given a profile of an all required bundle input (no real data on number of reads), with big size
    InputCacheUsageProfile profile {};
    profile.produced    = false;
    profile.allRequired = true;
    profile.size        = CAP * 10000;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect no alloc
    validateNoAlloc(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_perforated_intermediate_input)
{
    createGraph(1);

    // Given a profile of reading an intermediate input where the producer has the same perforation as the current
    // consumer
    InputCacheUsageProfile profile {};
    profile.produced                  = true;
    profile.localized                 = true;
    profile.size                      = CAP;
    profile.dcoreReads                = 1;  // To make sure the perforation is the reason for allocD
    profile.totalReads                = 4;  // To make sure the release is not degrade_class

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD
    validateAllocD(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_differently_perforated_intermediate)
{
    createGraph(1);

    // Given a profile of reading an intermediate input where the producer has a different perforation than the current
    // consumer, but each dcore reads different data (#total-reads == #dcore-reads)
    InputCacheUsageProfile profile {};
    profile.produced                  = true;
    profile.localized                 = false;
    profile.totalReads                = 2;
    profile.dcoreReads                = 2;
    profile.size                      = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD
    validateAllocD(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_differently_perforated_shared_intermediate)
{
    createGraph(1);

    // Given a profile of reading an intermediate input where the producer has a different perforation than the current
    // consumer, and the same data is read by multiple dcores (#total-reads > #dcore-reads)
    InputCacheUsageProfile profile {};
    profile.produced                  = true;
    profile.localized                 = false;
    profile.totalReads                = 8;
    profile.dcoreReads                = 2;
    profile.size                      = CAP;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocDH
    validateAllocDH(cra.inputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_require_no_release_for_non_last_read)
{
    createGraph(1);

    // Given a profile of a bundle input where each datum is read once from each dcore on a non-last consumer
    InputCacheUsageProfile profile {};
    profile.produced     = false;
    profile.size         = CAP;
    profile.bpt          = false;
    profile.lastConsumer = false;
    profile.totalReads   = 4;
    profile.dcoreReads   = 1;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect no release
    auto req = cra.inputRequirement(0, 0);
    validateAllocH(req);
    validateNoRelease(req);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_require_discard_release_for_last_read_non_bpts)
{
    setGlobalConfForTest(GCFG_ENABLE_LB_NON_BPT_SLICES_DISCARDING, "true");
    createGraph(1);

    // Given a profile of a non-bpt input where each datum is shared among dcores on the last consumer
    InputCacheUsageProfile profile {};
    profile.produced     = true;
    profile.size         = CAP;
    profile.bpt          = false;
    profile.lastConsumer = true;
    profile.totalReads   = 4;
    profile.dcoreReads   = 1;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect discard CME
    auto req = cra.inputRequirement(0, 0);
    validateAllocH(req);  // single dcore read
    validateRelease(req, RD::ReleaseType::DISCARD_CME);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_require_degrade_cme_release_for_last_read_bpts)
{
    createGraph(1);

    // Given a profile of an intermediate bpt input where each datum is read multiple times from a single dcore on the
    // last consumer
    InputCacheUsageProfile profile {};
    profile.produced     = true;
    profile.size         = CAP;
    profile.bpt          = true;
    profile.lastConsumer = true;
    profile.totalReads   = 2;
    profile.dcoreReads   = 2;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect degrade CME
    auto req = cra.inputRequirement(0, 0);
    validateAllocD(req);
    validateRelease(req, RD::ReleaseType::DEGRADE_CME);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_require_degrade_class_release_for_last_read_bpts_with_single_access)
{
    createGraph(1);

    // Given a profile of an intermediate bpt input where each datum is read once on the last consumer
    InputCacheUsageProfile profile {};
    profile.produced     = true;
    profile.size         = CAP;
    profile.bpt          = true;
    profile.lastConsumer = true;
    profile.totalReads   = 1;
    profile.dcoreReads   = 1;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect degrade class
    auto req = cra.inputRequirement(0, 0);
    validateAllocH(req);
    validateRelease(req, RD::ReleaseType::DEGRADE_CLASS);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_requires_cme_yielding_for_unreleased_input_with_multiple_reads)
{
    createGraph(1);

    // Given a profile of a shared input that will be read again much later in the bundle
    InputCacheUsageProfile profile {};
    profile.produced        = true;
    profile.size            = CAP;
    profile.bpt             = false;
    profile.lastConsumer    = false;
    profile.totalReads      = 4;
    profile.dcoreReads      = 1;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect CME based yielding
    auto req = cra.inputRequirement(0, 0);
    validateAllocH(req);
    validateYield(req, RD::ReleaseType::DEGRADE_CME);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_requires_degrade_yielding_for_unreleased_input_with_single_read)
{
    createGraph(1);

    // Given a profile of a shared input that will be read again much later in the bundle
    InputCacheUsageProfile profile {};
    profile.produced        = true;
    profile.size            = CAP;
    profile.bpt             = false;
    profile.lastConsumer    = false;
    profile.totalReads      = 1;
    profile.dcoreReads      = 1;

    auto                      profiler = getInputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect degraded class based yielding
    auto req = cra.inputRequirement(0, 0);
    validateAllocH(req);
    validateYield(req, RD::ReleaseType::DEGRADE_CLASS);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_should_require_no_allocation_for_bundle_output)
{
    createGraph(1);

    // Given a profile of a non-RMW bundle output
    OutputCacheUsageProfile profile {};

    auto                      profiler = getOutputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect no alloc
    validateNoAlloc(cra.outputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_requirements_for_rmw_bundle_output)
{
    createGraph(1);

    // Given a profile of a RMW bundle output
    OutputCacheUsageProfile profile {};
    profile.size = CAP;
    profile.rmw  = true;

    auto                      profiler = getOutputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD with highest cache class
    validateAllocD(cra.outputRequirement(0, 0), CacheClass::Top);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_same_perforation_intermediate_output)
{
    createGraph(1);

    // Given a profile of an intermediate producer with same perforation as the consumer
    OutputCacheUsageProfile profile {};
    profile.size                           = CAP;
    profile.hasConsumers                   = true;
    profile.localized                      = true;

    auto                      profiler = getOutputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD
    validateAllocD(cra.outputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_required_allocation_for_different_perforation_intermediate_output)
{
    createGraph(1);

    // Given a profile of an intermediate producer with different perforation than the consumer
    OutputCacheUsageProfile profile {};
    profile.size                           = CAP;
    profile.hasConsumers                   = true;
    profile.localized                      = false;

    auto                      profiler = getOutputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocH
    validateAllocH(cra.outputRequirement(0, 0));
}

TEST_F(CacheRequirementsAnalyzerTest, cra_should_set_output_release_for_last_rmw_writer)
{
    createGraph(1);
    // Given a profile of a RMW bundle output where the producer is the last to read-modify-write
    OutputCacheUsageProfile profile {};
    profile.size          = CAP;
    profile.rmw           = true;
    profile.lastRmwWriter = true;

    auto                      profiler = getOutputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD with highest cache class and degrade CME
    auto req = cra.outputRequirement(0, 0);
    validateAllocD(req, CacheClass::Top);
    validateRelease(req, RD::ReleaseType::DEGRADE_CME);
}

TEST_F(CacheRequirementsAnalyzerTest, cra_should_not_set_output_release_for_last_rmw_writer_with_consumer)
{
    createGraph(1);
    // Given a profile of a RMW bundle output where the producer is the last to read-modify-write
    OutputCacheUsageProfile profile {};
    profile.size          = CAP;
    profile.rmw           = true;
    profile.lastRmwWriter = true;
    profile.hasConsumers  = true;

    auto                      profiler = getOutputCacheProfiler(profile);
    CacheRequirementsAnalyzer cra(profiler);

    // Expect allocD with high cache class and no release
    auto req = cra.outputRequirement(0, 0);
    validateAllocD(req, CacheClass::Top);
    validateNoRelease(req);
}