#include <cstddef>

#include "brain_conf.h"
#include "node_cache_setter.h"
#include "bundle_memory_preprocessor.h"

#include "../gaudi2/layered_brain_test.h"
#include "scoped_configuration_change.h"

using namespace gc::layered_brain;

class NodeCacheSetterTest : public LayeredBrainTest
{
public:
    BundleNodes   m_bundle;
    MemoryUsageDB m_db;

    using ReqDet = CacheRequirementsAnalyzerIfc::RequirementDetails;

    void SetUp() override
    {
        LayeredBrainTest::SetUp();
        // For easier tracking of actual budget:
        setGlobalConfForTest(GCFG_FRAGMENTATION_COMPENSATION_FACTOR, "1.0");
        // To test yielding flows:
        setGlobalConfForTest(GCFG_ENABLE_LB_CACHE_YIELDING, "true");
        // To make sure ctrl deps are added when testing for dependency assurance
        setGlobalConfForTest(GCFG_LAYERED_BRAIN_CACHE_THRESHING_PREVENTION_MODE, "0");
    }

    struct RequirementAnalyzerNotRequired : public CacheRequirementsAnalyzerIfc
    {
        RequirementDetails inputRequirement(size_t nodeIdx, size_t inputIdx) const override
        {
            return RequirementDetails::noAlloc();
        }
        RequirementDetails outputRequirement(size_t nodeIdx, size_t outputIdx) const override
        {
            return RequirementDetails::noAlloc();
        }
    };

    struct RequirementAnalyzerConstCap : public CacheRequirementsAnalyzerIfc
    {
        RequirementAnalyzerConstCap(
            uint64_t                        inCap,
            std::optional<uint64_t>         outCap        = {},
            bool                            releaseInputs = false,
            RequirementDetails::ReleaseType releaseType   = RequirementDetails::ReleaseType::DEGRADE_CME)
        : IN_CAP(inCap), OUT_CAP(outCap.value_or(inCap)), RELEASE_IN(releaseInputs), RELEASE_TYPE(releaseType)
        {
        }

        RequirementDetails inputRequirement(size_t nodeIdx, size_t inputIdx) const override
        {
            auto req    = RequirementDetails::allocH(IN_CAP);
            req.release = RELEASE_TYPE;
            if (RELEASE_IN) req.postAccess = RequirementDetails::PostAccessAction::RELEASE;
            return req;
        }
        RequirementDetails outputRequirement(size_t nodeIdx, size_t outputIdx) const override
        {
            return RequirementDetails::allocH(OUT_CAP);
        }

        const uint64_t                        IN_CAP;
        const uint64_t                        OUT_CAP;
        const bool                            RELEASE_IN;
        const RequirementDetails::ReleaseType RELEASE_TYPE;
    };

    class RequirementAnalyzerMapped : public CacheRequirementsAnalyzerIfc
    {
    public:
        // operand idx --> requirement
        using Map = std::unordered_map<size_t, RequirementDetails>;

        RequirementAnalyzerMapped(const Map& inputMap, const Map& outputMap)
        : m_inputMap(inputMap), m_outputMap(outputMap)
        {
        }

        RequirementDetails inputRequirement(size_t nodeIdx, size_t inputIdx) const override
        {
            return m_inputMap.at(inputIdx);
        }

        RequirementDetails outputRequirement(size_t nodeIdx, size_t outputIdx) const override
        {
            return m_outputMap.at(outputIdx);
        }

    private:
        Map m_inputMap;
        Map m_outputMap;
    };

    void init(int numNodes)
    {
        createGraph(numNodes);
        m_bundle = bundleNodes(0, 0, numNodes);
        m_db     = runPreprocessor();
    }

    MemoryUsageDB runPreprocessor()
    {
        POCBundleMemoryPreProcessor bmpp {m_graph, m_bundle};
        return bmpp.buildMemUsageDB();
    }

    // Legacy creation of NodeCacheSetter by budget. Kept to avoid re-writing all tests.
    std::vector<BundleCacheState> m_cacheStates;
    NodeCacheSetter getNCS(BundleCacheState::Capacity budget)
    {
        m_cacheStates.emplace_back(budget);
        return getNCS(m_cacheStates.back());
    }

    NodeCacheSetter getNCS(BundleCacheState& cacheStateTracker)
    {
        const unsigned pipelineDepth = 2;  // To make sure the yield barrier is 2 threads
        return NodeCacheSetter(m_graph, m_bundle, m_db, cacheStateTracker, pipelineDepth);
    }

    // CMD = Cache Meta Data
    void validateCMDVecDirectives(const std::vector<CacheMetaData>&            cmds,
                                  const std::initializer_list<CacheDirective>& expDirectives) const
    {
        size_t cmdIdx = 0;
        ASSERT_EQ(cmds.size(), expDirectives.size());
        for (const auto& directive : expDirectives)
        {
            EXPECT_EQ(directive, cmds[cmdIdx++].cacheDirective) << "Failure at cmd " << cmdIdx - 1;
        }
    }
};

TEST_F(NodeCacheSetterTest, ncs_should_set_no_alloc_directive_when_no_cache_required)
{
    // Given [in]->n->[out]
    init(1);

    auto ncs = getNCS(0);

    // When setting directives with no cache requirements
    RequirementAnalyzerNotRequired analyzer;
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer));

    // Expect noAlloc directive set for each tensor
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {NoAllocate});
    validateCMDVecDirectives(n->getNodeAnnotation().outputsCacheMetaData, {NoAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_set_cache_usage_directive_when_within_budget)
{
    // Given [in]->n->[out]
    init(1);
    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 20b cache required for each tensor
    RequirementAnalyzerConstCap analyzer(20);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer));

    // Expect allocH directive set for each tensor
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    validateCMDVecDirectives(n->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_set_no_alloc_when_exceeding_budget)
{
    // Given [in]->n->[out]
    init(1);
    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 120b cache required for each tensor
    RequirementAnalyzerConstCap analyzer(120);
    ASSERT_FALSE(ncs.setDirectives(0, &analyzer));

    // Expect noAlloc directive set for each tensor
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {NoAllocate});
    validateCMDVecDirectives(n->getNodeAnnotation().outputsCacheMetaData, {NoAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_set_in_cache_the_operands_that_are_within_budget__input_exceeds)
{
    // Given [in]->n->[out]
    init(1);
    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 120b cache required for input and 80b cache required for output
    RequirementAnalyzerConstCap analyzer(120, 80);
    ASSERT_FALSE(ncs.setDirectives(0, &analyzer));

    // Expect only the output to have allocH
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {NoAllocate});
    validateCMDVecDirectives(n->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_set_in_cache_the_operands_that_are_within_budget__output_exceeds)
{
    // Given [in]->n->[out]
    init(1);
    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 60b cache required for input and 120b cache required for output
    RequirementAnalyzerConstCap analyzer(60, 120);
    ASSERT_FALSE(ncs.setDirectives(0, &analyzer));

    // Expect only the input to have allocH
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    validateCMDVecDirectives(n->getNodeAnnotation().outputsCacheMetaData, {NoAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_limit_assignments_to_cache_budget)
{
    // Given [in1, in2]->n->[out]
    init(1);
    m_bundle.front()->addInput(1, newTensor());
    m_db = runPreprocessor();  // re-process to add the extra input

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 40b required for each operand
    RequirementAnalyzerConstCap analyzer(40, 40, true);

    // Expect failure to allocate the 2nd input, since the first input should be released only after all the operands
    // were allocated
    ASSERT_FALSE(ncs.setDirectives(0, &analyzer));

    // Expect only the second input allocation to fail, as all 3 operands together would exceed the budget
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate, NoAllocate});
    validateCMDVecDirectives(n->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_not_release_uncached_tensors)
{
    // Given [in0, in1, in2, in3]->n->[out0, out1, out3]
    init(1);
    NodePtr node = m_bundle[0];
    node->addInput(1, newTensor());
    node->addInput(2, newTensor());
    node->addInput(3, newTensor());
    node->addOutput(newTensor());
    node->addOutput(newTensor());
    m_db = runPreprocessor();  // re-process to add the extra inputs and otuputs

    auto ncs = getNCS(100);

    auto allocDReq       = ReqDet::allocD(10);
    allocDReq.release    = ReqDet::ReleaseType::DEGRADE_CME;
    allocDReq.postAccess = ReqDet::PostAccessAction::RELEASE;

    auto allocHReq       = ReqDet::allocH(10);
    allocHReq.release    = ReqDet::ReleaseType::NONE;
    allocHReq.postAccess = ReqDet::PostAccessAction::NONE;

    auto allocDHReq       = ReqDet::allocDH(10);
    allocDHReq.release    = ReqDet::ReleaseType::DEGRADE_CLASS;
    allocDHReq.postAccess = ReqDet::PostAccessAction::NONE;

    RequirementAnalyzerMapped cacheReqAnalyzer(
        // input map
        RequirementAnalyzerMapped::Map {{0, allocDReq}, {1, allocHReq}, {2, allocDHReq}, {3, ReqDet::noAlloc()}},
        // output map
        RequirementAnalyzerMapped::Map {{0, allocDReq}, {1, allocHReq}, {2, ReqDet::noAlloc()}});

    ASSERT_TRUE(ncs.setDirectives(0, &cacheReqAnalyzer));

    // Expect the first operand to be cached and released. The second cached and not released. Third cached with
    // DEGRADE_CLASS release details (not actually released) and the forth is not cached
    validateCMDVecDirectives(node->getNodeAnnotation().inputsCacheMetaData,
                             {DcoreAllocate, HomeAllocate, SharedAllocate, NoAllocate});
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(0).cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(0).mcid, 2);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(0).cacheClass, CacheClass::High);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(1).cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(1).mcid, 0);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(1).cacheClass, CacheClass::High);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(2).cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(2).mcid, 0);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(2).cacheClass, CacheClass::High);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(3).cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(node->getNodeAnnotation().inputsCacheMetaData.at(3).cacheClass, CacheClass::Normal);  // default
    validateCMDVecDirectives(node->getNodeAnnotation().outputsCacheMetaData, {DcoreAllocate, HomeAllocate, NoAllocate});
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(0).cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(0).mcid, 1);
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(0).cacheClass, CacheClass::High);
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(1).cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(1).mcid, 0);
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(1).cacheClass, CacheClass::High);
    EXPECT_EQ(node->getNodeAnnotation().outputsCacheMetaData.at(2).cmAction, CacheMaintenanceAction::NOP);
}

TEST_F(NodeCacheSetterTest, ncs_should_reuse_dead_tensors_budget)
{
    // Given [in]->n0->[t0]->n1->[t1]
    init(2);

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 40b required for each operand
    RequirementAnalyzerConstCap analyzer(40, 40, true);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer));

    // Expect success to allocate cache for the accesses of n1 since t0 is already allocated and 'in' can be reused
    NodePtr n0 = m_bundle[0];
    NodePtr n1 = m_bundle[1];
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer));
    validateCMDVecDirectives(n1->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    validateCMDVecDirectives(n1->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});

    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 1);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].mcid, 2);
}

TEST_F(NodeCacheSetterTest, ncs_should_allocate_and_release_multiple_views_of_the_same_tensor_only_once)
{
    // Given [t0]->n0->[t1]->n1->[t2]->n2->[t3]
    //                  |^        |    ^
    //                  |+-alias--+    |
    //                  +--------------+
    // Where t2 is an alias of t1 and t1 is an input to n2 (n0 is in the role of a fork node - framing the bundle, since
    // aliasing analysis is only done for intermediate tensors)
    init(3);
    NodePtr   n1 = m_bundle.at(1);
    NodePtr   n2 = m_bundle.at(2);
    TensorPtr t1 = n1->getInput(0);
    TensorPtr t2 = n2->getInput(0);
    n2->addInput(1, t1);
    t2->setAsAliasSubTensor(t1);
    m_db = runPreprocessor();  // re-process to add the extra connections

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 20b required for each input and 80b for the output and releasing the inputs
    RequirementAnalyzerConstCap analyzer(20, 80, true);
    ASSERT_TRUE(ncs.setDirectives(2, &analyzer));

    // Expect success to allocate cache for the accesses of n2
    validateCMDVecDirectives(n2->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate, HomeAllocate});
    validateCMDVecDirectives(n2->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});
    EXPECT_EQ(n2->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n2->getNodeAnnotation().inputsCacheMetaData[0].mcid, 1);
    // the 2nd CME is redundant, but it will be optimized out by code generation, so to keep the implementation simple,
    // we don't care.
    EXPECT_EQ(n2->getNodeAnnotation().inputsCacheMetaData[1].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n2->getNodeAnnotation().inputsCacheMetaData[1].mcid, 2);
}

TEST_F(NodeCacheSetterTest, ncs_should_reuse_yielded_tensors_budget)
{
    // Given [in]->n0->[t0]->n1->[t1],
    // where n0 threadIdx = 0
    // and n1 threadIdx = 2
    init(2);
    NodePtr n0                                      = m_bundle[0];
    NodePtr n1                                      = m_bundle[1];
    n0->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n1->getNodeAnnotation().bundleInfo->threadIndex = 2;

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 40b required for each operand, yielding the input
    RequirementAnalyzerConstCap yieldingAnalyzer(40);
    ASSERT_TRUE(ncs.setDirectives(0, &yieldingAnalyzer));
    // At this point the input of n0 is expected not to be released.
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);

    // Expect success to allocate cache for the accesses of n1 since t0 is already allocated and 'in' can be yielded
    RequirementAnalyzerConstCap analyzer(40, 40, true);
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer));
    validateCMDVecDirectives(n1->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    validateCMDVecDirectives(n1->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});

    // Expect n0 release to have been updated to yield the budget for t1
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 1);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].mcid, 2);
}

TEST_F(NodeCacheSetterTest, ncs_should_reuse_multiple_yielded_tensors_budget)
{
    // Given [t0.0]->n0->[t1]->n1->[t2]->n2->[t3],
    //               ^
    //               |
    //       [t0.1]--+
    //
    // where n0, n1 threadIdx = 0
    // and n2 threadIdx = 2
    createGraph(3);
    m_bundle = bundleNodes(0, 0, 3);

    NodePtr n0 = m_bundle.at(0);
    NodePtr n1 = m_bundle.at(1);
    NodePtr n2 = m_bundle.at(2);
    GraphEditor::editNode(m_graph, n0, [&]() { n0->addInput(1, newTensor()); });  // add t0.1

    n0->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n1->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n2->getNodeAnnotation().bundleInfo->threadIndex = 2;
    m_db                                            = runPreprocessor();

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 25b required for each operand, the inputs are yielding candidates
    RequirementAnalyzerConstCap yieldingAnalyzer(25);
    ASSERT_TRUE(ncs.setDirectives(0, &yieldingAnalyzer));  // 25b for t0.0, t0.1, t1 (total 75b)
    ASSERT_TRUE(ncs.setDirectives(1, &yieldingAnalyzer));  // 25b for t2 (total 100b)
    // At this point the inputs of n0 and n1 are expected not to be released.
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[1].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[1].mcid, 0);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);

    // Expect success to allocate cache for the accesses of n2 since t2 is already allocated and t0.*, t1 can be yielded
    RequirementAnalyzerConstCap analyzer(25, 75, true);
    ASSERT_TRUE(ncs.setDirectives(2, &analyzer));

    // Expect n0 release to have been updated to yield the budget for t1
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 1);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[1].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[1].mcid, 2);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].mcid, 3);
}

TEST_F(NodeCacheSetterTest, ncs_should_not_downgrade_class_or_directive_when_not_releasing)
{
    // Given [in]->n->[out]
    init(1);
    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with degrade class but no release
    RequirementAnalyzerConstCap analyzer(20, 30, false, ReqDet::ReleaseType::DEGRADE_CLASS);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer));

    // Expect high class for inputs and outputs
    const NodePtr& n = m_bundle[0];
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    EXPECT_EQ(CacheClass::High, n->getNodeAnnotation().inputsCacheMetaData.at(0).cacheClass);
    EXPECT_EQ(CacheClass::High, n->getNodeAnnotation().outputsCacheMetaData.at(0).cacheClass);
}

TEST_F(NodeCacheSetterTest, ncs_should_downgrade_class_and_directive_when_releasing)
{
    // Given [in]->n->[out]
    init(1);
    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with degrade class and releasing the input
    RequirementAnalyzerConstCap analyzer(20, 30, true, ReqDet::ReleaseType::DEGRADE_CLASS);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer));

    // Expect high class for outputs and low for inputs
    const NodePtr& n = m_bundle[0];
    EXPECT_EQ(CacheClass::Low, n->getNodeAnnotation().inputsCacheMetaData.at(0).cacheClass);
    EXPECT_EQ(CacheClass::High, n->getNodeAnnotation().outputsCacheMetaData.at(0).cacheClass);
    // see explanation in RequirementDetails::releaseCacheDirective
    validateCMDVecDirectives(n->getNodeAnnotation().inputsCacheMetaData, {NoAllocate});
}

TEST_F(NodeCacheSetterTest, ncs_should_downgrade_class_and_directive_when_yielding)
{
    // Given [in]->n0->[t0]->n1->[t1],
    // where n0 threadIdx = 0
    // and n1 threadIdx = 2
    init(2);
    NodePtr n0                                      = m_bundle[0];
    NodePtr n1                                      = m_bundle[1];
    n0->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n1->getNodeAnnotation().bundleInfo->threadIndex = 2;

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 40b required for each operand, not releasing the input (=> yield candidate)
    RequirementAnalyzerConstCap yieldingAnalyzer(40, 40, false, ReqDet::ReleaseType::DEGRADE_CLASS);
    ASSERT_TRUE(ncs.setDirectives(0, &yieldingAnalyzer));
    // At this point the input of n0 is expected not to be released.
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cacheClass, CacheClass::High);

    // Expect success to allocate cache for the accesses of n1 since t0 is already allocated and 'in' can be yielded
    RequirementAnalyzerConstCap analyzer(40, 40, true);
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer));
    validateCMDVecDirectives(n1->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    validateCMDVecDirectives(n1->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});

    // Expect n0 release to have been updated using cache class to yield the budget required for t1
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cacheClass, CacheClass::Low);
    // see explanation in RequirementDetails::releaseCacheDirective
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cacheDirective, NoAllocate);

    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].mcid, 1);
    EXPECT_EQ(n1->getNodeAnnotation().inputsCacheMetaData[0].cacheClass, CacheClass::High);
}

TEST_F(NodeCacheSetterTest, ncs_should_not_yield_tensors_when_it_wouldnt_help)
{
    // Given [in]->n0->[t0]->n1->[t1],
    // where n0 threadIdx = 0
    // and n1 threadIdx = 2
    init(2);
    NodePtr n0                                      = m_bundle[0];
    NodePtr n1                                      = m_bundle[1];
    n0->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n1->getNodeAnnotation().bundleInfo->threadIndex = 2;

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 40b required for each operand, yielding the input
    RequirementAnalyzerConstCap yieldingAnalyzer(40);
    ASSERT_TRUE(ncs.setDirectives(0, &yieldingAnalyzer));
    // At this point the input of n0 is expected not to be released.
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);

    // Expect failure to allocate 70b cache for the output access of n1 since 'in' is not big enough to help if it's
    // yielded.
    RequirementAnalyzerConstCap analyzer(30, 70, true);
    ASSERT_FALSE(ncs.setDirectives(1, &analyzer));

    // Expect n0 input to stay unreleased
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);
}

TEST_F(NodeCacheSetterTest, ncs_should_not_reuse_yielded_tensors_budget_of_recent_threads)
{
    // Given [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[t3], where
    // n0 threadIdx = 0
    // n1 threadIdx = 1
    // n2, n3 threadIdx = 2
    init(4);
    NodePtr n0                                      = m_bundle[0];
    NodePtr n1                                      = m_bundle[1];
    NodePtr n2                                      = m_bundle[2];
    NodePtr n3                                      = m_bundle[3];
    n0->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n1->getNodeAnnotation().bundleInfo->threadIndex = 1;
    n2->getNodeAnnotation().bundleInfo->threadIndex = 2;
    n3->getNodeAnnotation().bundleInfo->threadIndex = 2;

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // When setting directives with 35b required for each operand, yielding the input
    RequirementAnalyzerConstCap analyzer(25, 25);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer));  // in, t0 allocated 25b, 'in' is yielded (total 50b)
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer));  // t1 allocated 25b, t0 is yielded (total 75b)
    ASSERT_TRUE(ncs.setDirectives(2, &analyzer));  // t2 allocated 25b, t1 is yielded (total 100b)

    // Expect failure to allocate 30b for the output of n3, since it can only use the yielded capacity of 'in'.
    RequirementAnalyzerConstCap analyzer30bCap(25, 30);
    ASSERT_FALSE(ncs.setDirectives(3, &analyzer30bCap));
    // At this point the input of n0 ('in') is expected not to be released.
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::NOP);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 0);

    // Expect success to allocate 25b for the output of n3, since 'in' can be released for that.
    RequirementAnalyzerConstCap analyzer25bCap(25, 25);
    ASSERT_TRUE(ncs.setDirectives(3, &analyzer25bCap));
    validateCMDVecDirectives(n3->getNodeAnnotation().inputsCacheMetaData, {HomeAllocate});
    validateCMDVecDirectives(n3->getNodeAnnotation().outputsCacheMetaData, {HomeAllocate});
    // At this point the input of n0 ('in') should change to be released.
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].cmAction, CacheMaintenanceAction::DEGRADE);
    EXPECT_EQ(n0->getNodeAnnotation().inputsCacheMetaData[0].mcid, 1);
}

TEST_F(NodeCacheSetterTest, ncs_should_be_able_to_revive_yielded_tensors)
{
    // Given [t0]->n0->[t1]->n1->[t2]->n2->[t3]   n3->[t4]
    //         |        |                         ^
    //         |        |                         |
    //         +--------+-------------------------+
    // where:
    // n0 threadIdx = 1
    // n1 threadIdx = 0
    // n2 threadIdx = 3
    // n3 threadIdx = 3, input[0]=t1, input[1]=t0
    init(4);
    NodePtr n0                                      = m_bundle.at(0);
    NodePtr n1                                      = m_bundle.at(1);
    NodePtr n2                                      = m_bundle.at(2);
    NodePtr n3                                      = m_bundle.at(3);
    n0->getNodeAnnotation().bundleInfo->threadIndex = 1;
    n1->getNodeAnnotation().bundleInfo->threadIndex = 0;
    n2->getNodeAnnotation().bundleInfo->threadIndex = 3;
    n3->getNodeAnnotation().bundleInfo->threadIndex = 3;

    TensorPtr t0 = n0->getInput(0);
    TensorPtr t1 = n1->getInput(0);
    GraphEditor::editNode(m_graph, n3, [&]() {
        n3->replaceInput(0, t1);
        n3->addInput(1, t0);
    });

    // And a budget of 100 bytes
    auto ncs = getNCS(100);

    // This flow checks the bug fix of a situation where both t0 and t1 are candidates for yielding. it is enough to
    // yield t0, capacity-wise, but n0 has higher thread-idx then n1, so t1 may have higher priority for yielding. This
    // will cause both to be freed, but only t0 to be reclaimed (since it has lower access index). This caused t1 to
    // have to be revived from a 'dead' state, which is not supported in the current design (needs to be properly
    // reclaimed and the dependencies ensured).

    RequirementAnalyzerConstCap analyzer0(50, 10);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer0));  // t0, t1 allocated 50b, 10b, t0 is yielded from thread-1.

    RequirementAnalyzerConstCap analyzer1(10, 10);
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer1));  // t2 allocated 10b, t1 is yielded from thread-0 (total 70b).

    RequirementAnalyzerConstCap analyzer2(10, 50);
    ASSERT_TRUE(ncs.setDirectives(2, &analyzer2));  // t3 allocated 50b by reclaiming t0 (still 70b).

    RequirementAnalyzerConstCap analyzer3(10, 20);
    ASSERT_TRUE(ncs.setDirectives(3, &analyzer3));  // t4 allocated 20b. t1 is 'revived', t0 allocated 10b (total 100b).
}

TEST_F(NodeCacheSetterTest, ncs_should_set_dependency_when_reusing_reclaimed_budget)
{
    // Given
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    // Transformed to:
    // [in]->n0->[t0]->n1->[t1]
    //   |
    //   +------------>n2->[t2]
    // (n2 reads 'in' instead of t1)
    init(3);
    NodePtr n0 = m_bundle.at(0);
    NodePtr n1 = m_bundle.at(1);
    NodePtr n2 = m_bundle.at(2);
    GraphEditor::replaceInput(m_graph, n2, 0, n0->getInput(0));

    ASSERT_FALSE(m_graph.isAncestor(n1, n2)) << "Sanity failed after setup";

    m_db = runPreprocessor();

    // And a budget of 100 bytes (effective)
    auto ncs = getNCS(100);

    // First allocate 20b for each of n0 operands (without releasing any)
    RequirementAnalyzerConstCap analyzer0(20, 20, false);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer0));

    // Next allocate 20b for n1 operands and release t0
    RequirementAnalyzerConstCap analyzer1(20, 20, true);
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer1));

    // At this point, there should be:
    // 20b allocated for 'in'
    // 20b allocated for t1
    // 20b allocated and freed for t0
    // So when allocating 50b for t2:
    RequirementAnalyzerConstCap analyzer2(20, 50, true);
    ASSERT_TRUE(ncs.setDirectives(2, &analyzer2));

    // need to reclaim the budget of t0, so dependency needs to be added between n1 and n2.
    EXPECT_TRUE(m_graph.isAncestor(n1, n2));
}

TEST_F(NodeCacheSetterTest, ncs_should_provide_max_cache_usage)
{
    constexpr int      numNodes  = 3;
    constexpr uint64_t budget    = 96ull * 1024 * 1024;
    constexpr uint64_t sliceSize = 1ull * 1024 * 1024;

    // Given a chain
    init(numNodes);

    // And a budget of 96MB
    BundleCacheState cacheStateTracker(budget);
    auto             ncs = getNCS(cacheStateTracker);

    // When setting directives with 1MB for each operand and not releasing anything
    RequirementAnalyzerConstCap analyzer(sliceSize, sliceSize, false);
    for (size_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++)
    {
        EXPECT_TRUE(ncs.setDirectives(nodeIdx, &analyzer));
    }

    // In a chain, the number of tensors is one more than the number of nodes
    EXPECT_EQ((numNodes + 1) * sliceSize, cacheStateTracker.maxLiveCapacity());
}

TEST_F(NodeCacheSetterTest, ncs_steady_state_test)
{
    constexpr int      numNodes  = 13;
    constexpr uint64_t budget    = 96ull * 1024 * 1024;
    constexpr uint64_t writeSize = 5ull * 1024 * 1024;
    constexpr uint64_t readSize  = 20ull * 1024 * 1024;

    // Given a long chain
    init(numNodes);

    // And a budget of 96MB
    BundleCacheState cacheStateTracker(budget);
    auto             ncs = getNCS(cacheStateTracker);

    // When setting directives with different read and write sizes and releasing input once it's used
    RequirementAnalyzerConstCap analyzer(readSize, writeSize, true);
    for (size_t nodeIdx = 0; nodeIdx < numNodes; nodeIdx++)
    {
        EXPECT_TRUE(ncs.setDirectives(nodeIdx, &analyzer));
        // At any given moment only a single input and output are alive in the cache
        EXPECT_EQ(readSize + writeSize, cacheStateTracker.maxLiveCapacity()) << "Failure in index " << nodeIdx;
    }
}

TEST_F(NodeCacheSetterTest, ncs_should_support_release_separately_from_caching)
{
    // [t0]->n0->[t1]->n1->[t2]
    //   |             ^
    //   +-------------+
    init(2);
    NodePtr   n0 = m_bundle.at(0);
    NodePtr   n1 = m_bundle.at(1);
    TensorPtr t0 = n0->getInput(0);
    TensorPtr t1 = n1->getInput(0);
    GraphEditor::editNode(m_graph, n1, [&]() { n1->addInput(1, t0); });
    m_db = runPreprocessor();

    // Budget = 100
    BundleCacheState cacheStateTracker(100);
    auto             ncs = getNCS(cacheStateTracker);

    // When n0 doesn't require release, but do require caching
    RequirementAnalyzerConstCap analyzer0(10, 10, false);
    ASSERT_TRUE(ncs.setDirectives(0, &analyzer0));
    EXPECT_TRUE(cacheStateTracker.isCached(t0));
    EXPECT_TRUE(cacheStateTracker.isCached(t1));
    EXPECT_EQ(20, cacheStateTracker.totalLive());  // 10 input, 10 output

    // Then n1 does require release, but not caching
    auto releaseNoCacheReq       = ReqDet::noAlloc();
    releaseNoCacheReq.release    = ReqDet::ReleaseType::DEGRADE_CME;
    releaseNoCacheReq.postAccess = ReqDet::PostAccessAction::RELEASE;
    RequirementAnalyzerMapped analyzer1(
        // input map
        RequirementAnalyzerMapped::Map {{0, releaseNoCacheReq}, {1, releaseNoCacheReq}},
        // output map
        RequirementAnalyzerMapped::Map {{0, releaseNoCacheReq}});
    ASSERT_TRUE(ncs.setDirectives(1, &analyzer1));
    EXPECT_FALSE(cacheStateTracker.isCached(t0));
    EXPECT_FALSE(cacheStateTracker.isCached(t1));
    EXPECT_EQ(0, cacheStateTracker.totalLive());
}