#include "../gaudi2/layered_brain_test.h"
#include "bundle_memory_preprocessor.h"
#include "cache_requirements_profiler.h"
#include "memory_usage_db.h"
#include "tpc_kernel_loader.h"
#include "compilation_hal_reader.h"

class CacheRequirementsProfilerTest : public LayeredBrainTest
{
public:
    static constexpr size_t NUM_NODES = 3;
    static constexpr unsigned BUNDLE_ID = 8;

    CacheRequirementsProfilerTest() : m_halSetter(&m_graph) {}

    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    void initBundle(size_t numNodes = NUM_NODES, bool createTheGraph = true)
    {
        if (createTheGraph)  // enable pre generation of the graph and node chain
        {
            createGraph(numNodes);
        }

        // Load TPC kernel, so it will have access pattern
        TpcKernelLoader tpcKernelLoader(&m_graph);
        ASSERT_TRUE(tpcKernelLoader.load());

        bundleNodes(BUNDLE_ID, 0, numNodes);

        n.resize(numNodes);
        t.resize(numNodes);

        in = m_nodeChain[0]->getInput(0);
        for (size_t i = 0; i < numNodes; i++)
        {
            n[i] = m_nodeChain[i];
            t[i] = n[i]->getOutput(0);
        }
    }

    void initDB() { m_db = POCBundleMemoryPreProcessor(m_graph, m_nodeChain).buildMemUsageDB(); }

    CacheRequirementProfilerPtr getProfiler(const StrategyPtr& strategy = nullptr)
    {
        return CacheRequirementProfilerPtr(new CacheRequirementProfiler(m_graph, m_db, strategy));
    }

    // Replaces the 'idx'th node in the chain with an identity static reshape node
    bool replaceWithLogical(size_t idx)
    {
        NodePtr n = m_nodeChain[idx];
        NodePtr rs = NodeFactory::createNode(n->getInputs(),
                                             n->getOutputs(),
                                             nullptr,
                                             NodeFactory::staticReshapeNodeTypeName,
                                             fmt::format("reshape{}", idx));
        m_nodeChain[idx] = rs;
        return REPLACE_NODE_SUCCESS == GraphEditor::replaceNodes(m_graph, {n}, {rs});
    }

    void setNodesPerforationGroup(const std::initializer_list<unsigned>& nodesIndices, unsigned perforationGroup)
    {
        for (unsigned i : nodesIndices)
        {
            n[i]->getNodeAnnotation().bundleInfo->perforationGroup = perforationGroup;
        }
    }

    MemoryUsageDB m_db;

    std::vector<NodePtr>   n;
    TensorPtr              in;
    std::vector<TensorPtr> t;

    // For reduced BVDs detection and partials detection
    CompilationHalReaderSetter m_halSetter;
};

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_producer_existence_to_each_input)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    initBundle();
    initDB();

    auto crp = getProfiler();

    // Expect all inputs except 'in' to be marked as produced
    auto inProfile = crp->inputProfile(0, 0);
    EXPECT_FALSE(inProfile.produced);

    for (size_t i = 1; i < NUM_NODES; i++)
    {
        auto tiProfile = crp->inputProfile(i, 0);
        EXPECT_TRUE(tiProfile.produced);
    }
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_last_consumer)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    //                 ^          ^
    //                 |          |
    //         [t3]----+--------- +
    initBundle();
    TensorPtr t3 = newTensor();
    n[1]->addInput(1, t3);
    n[2]->addInput(1, t3);
    initDB();

    auto crp = getProfiler();

    // Expect n1 access to t3 to not be marked as last consumer
    auto n1Profile = crp->inputProfile(1, 1);
    EXPECT_FALSE(n1Profile.lastConsumer);
    EXPECT_EQ(2, n1Profile.nofConsumers);
    // Expect n2 access to t3 to be marked as last consumer
    auto n2Profile = crp->inputProfile(2, 1);
    EXPECT_TRUE(n2Profile.lastConsumer);
    EXPECT_EQ(2, n2Profile.nofConsumers);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_last_physical_consumer)
{
    //                         logical
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    //                 ^          ^
    //                 |          |
    //         [t3]----+--------- +
    createGraph(3);
    replaceWithLogical(2);
    initBundle(3, false);
    TensorPtr t3 = newTensor();
    n[1]->addInput(1, t3);
    n[2]->addInput(1, t3);
    initDB();

    auto crp = getProfiler();

    // Expect n1 access to t3 to be marked as last consumer
    auto n1Profile = crp->inputProfile(1, 1);
    EXPECT_TRUE(n1Profile.lastConsumer);
    EXPECT_EQ(1, n1Profile.nofConsumers);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_last_consumer_with_alias)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    //            |          ^
    //            |          |
    //            +--alias-- +
    initBundle();
    t[0]->setAsAliasSubTensor(t[1]);
    initDB();

    auto crp = getProfiler();

    // Expect n1 access to t0 to not be marked as last consumer
    auto n1Profile = crp->inputProfile(1, 0);
    EXPECT_FALSE(n1Profile.lastConsumer);
    EXPECT_EQ(2, n1Profile.nofConsumers);
    // Expect n2 access to t1 to be marked as last consumer
    auto n2Profile = crp->inputProfile(2, 0);
    EXPECT_TRUE(n2Profile.lastConsumer);
    EXPECT_EQ(2, n2Profile.nofConsumers);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_if_an_input_is_bpt)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    initBundle();
    initDB();
    // Make n2 a 'join' in the DB
    m_db.slices.at(t[1]).properties.joinedBy = n[2];

    auto crp = getProfiler();

    // Expect n0 access to 'in' to show it's a bpt (bundle input)
    auto n0Profile = crp->inputProfile(0, 0);
    EXPECT_TRUE(n0Profile.bpt);
    // Expect n1 access to t0 to show it's not a bpt
    auto n1Profile = crp->inputProfile(1, 0);
    EXPECT_FALSE(n1Profile.bpt);
    // Expect n2 access to t1 to show it's a bpt
    auto n2Profile = crp->inputProfile(2, 0);
    EXPECT_TRUE(n2Profile.bpt);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_the_tensor_size)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    initBundle();
    initDB();
    TSize sizes[] = {128, 128, 1, 1};
    t[0]->reshape(4, sizes);

    auto crp = getProfiler();

    // Expect n1 access to t0 to show its size
    auto inputProfile = crp->inputProfile(1, 0);
    EXPECT_EQ(128 * 128 * in->getElementSizeInBytes(), inputProfile.size);
    // Expect n0 access to t0 to show its size
    auto outputProfile = crp->outputProfile(0, 0);
    EXPECT_EQ(128 * 128 * in->getElementSizeInBytes(), outputProfile.size);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_consumer_existence_to_each_output)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    initBundle();
    initDB();

    auto crp = getProfiler();

    // Expect all outputs except 't2' to be marked as hasConsumers
    for (size_t i = 0; i < NUM_NODES - 1; i++)
    {
        auto tiProfile = crp->outputProfile(i, 0);
        EXPECT_TRUE(tiProfile.hasConsumers);
    }

    auto t2Profile = crp->outputProfile(2, 0);
    EXPECT_FALSE(t2Profile.hasConsumers);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_non_logical_consumer_existence_to_each_output)
{
    //                         logical
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]
    createGraph(3);
    replaceWithLogical(2);
    initBundle(3, false);
    initDB();

    auto crp = getProfiler();

    auto t1Profile = crp->outputProfile(1, 0);
    EXPECT_FALSE(t1Profile.hasConsumers);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_rmw_properties_for_outputs)
{
    //                     [t3]->n3->[t4]->┌─────────┐
    //                                     │Reduction│
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->└─────────┘
    initBundle();
    addNodeToChain();  // create n3. It's created as consumer of t2
    n.push_back(m_nodeChain.back());
    t.push_back(newTensor());        // create t3
    n[3]->replaceInput(0, t[3]);     // disconnect n3 from the chain
    bundleNode(BUNDLE_ID, n[3], 3);  // bundle n3 as the last scheduled.

    NodePtr reduction = NodeFactory::createNode({n[3]->getOutput(0), n[2]->getOutput(0)},
                                                {newTensor()},
                                                nullptr,
                                                NodeFactory::reductionNodeTypeName,
                                                "reduction");
    bundleNode(BUNDLE_ID, reduction, 4);
    m_nodeChain.push_back(reduction);  // to make sure it's in the DB
    initDB();

    auto crp = getProfiler();

    // Expect n3 to show its output is rmw and the last writer
    auto n3Profile = crp->outputProfile(3, 0);
    EXPECT_TRUE(n3Profile.rmw);
    EXPECT_TRUE(n3Profile.lastRmwWriter);
    // Expect n2 to show its output is rmw and not the last writer
    auto n2Profile = crp->outputProfile(2, 0);
    EXPECT_TRUE(n2Profile.rmw);
    EXPECT_FALSE(n2Profile.lastRmwWriter);
    // Expect n0 and n1 to show their outputs are not rmw
    EXPECT_FALSE(crp->outputProfile(1, 0).rmw);
    EXPECT_FALSE(crp->outputProfile(0, 0).rmw);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_perforation_compatibility)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[t3]->n4->[t4]
    // n0 and n1 are perforated the same
    // n2 is perforated on different bvd then n0, n1
    // n3 has the same perforation decision as n2, but no dcore ROIs (not actually perforated)
    // n4 is perforated with the same decision as n3
    initBundle(5);
    initDB();

    setNodesPerforationGroup({0, 1}, 1);
    setNodesPerforationGroup({2, 3, 4}, 2);

    unsigned perforationDim = 0;  // First node dim, tensor dim 0 is mapped to it
    // Add dcore rois to simulate actual perforation. The ROI details are irrelevant
    for (auto& ni : n)
    {
        if (ni == n[3]) continue;
        ni->getNodeAnnotation().m_dcoreROIs.resize(4);
        ni->getNodeAnnotation().perforationDim = perforationDim;
        ASSERT_TRUE(ni->getNodeAccessPattern());
        ASSERT_EQ(ni->getNodeAccessPattern()->getIndexSpaceDim(ni->getInput(0), 0), perforationDim);
        ASSERT_EQ(ni->getNodeAccessPattern()->getIndexSpaceDim(ni->getOutput(0), 0), perforationDim);
    }

    auto crp = getProfiler();

    std::vector<bool> expectedSamePerforationResult = {
        true,   // n0<>n1 - perforated the same
        false,  // n1<>n2 - perforated differently
        false,  // n2<>n3 - n3 not actually perforated
        false,  // n3<>n4 - n3 not actually perforated
    };

    for (size_t i = 0; i < 4; i++)
    {
        auto outProfile = crp->outputProfile(i, 0);
        EXPECT_EQ(expectedSamePerforationResult[i], outProfile.localized) << "Failure in n" << i << " output profile";
        auto inProfile = crp->inputProfile(i + 1, 0);  // n_i+1 input would match exp[i]
        EXPECT_EQ(expectedSamePerforationResult[i], inProfile.localized) << "Failure in n" << i + 1 << " input profile";
    }
}

TEST_F(CacheRequirementsProfilerTest, crp_should_ignore_logicals_to_determine_perforation_compatibility)
{
    //               logical               logical
    // [in]->n0->[t0]-->n1-->[t1]->n2->[t2]-->n3-->[t3]->n4->[t4]
    //            ^     ^     |         |           ^
    //            |     |     |         |           |
    //            |  [shape]  |         |           |
    //            |           |         |           |
    //            +---alias---+         +---alias---+
    // n0, n1, n2 and n3 are perforated the same
    // n1 and n3 are logical and n1 has a shape tensor and no dcore-rois (n3 has them on purpose. See below)
    // n4 is perforated differently
    createGraph(5);
    ASSERT_TRUE(replaceWithLogical(1));
    ASSERT_TRUE(replaceWithLogical(3));
    initBundle(5, false);
    auto shape = std::make_shared<Tensor>();
    shape->setShapeTensor(SHAPE_TENSOR);
    n[1]->addInput(1, shape);
    t[1]->setAsAliasSubTensor(t[0]);
    t[2]->setAsAliasSubTensor(t[3]);
    initDB();

    setNodesPerforationGroup({0, 1, 2, 3}, 1);
    setNodesPerforationGroup({4}, 2);

    unsigned perforationDim = 0;  // First node dim, tensor dim 0 is mapped to it
    // Add dcore rois to simulate actual perforation. The ROI details are irrelevant
    for (auto& ni : n)
    {
        // When a node doesn't have dcore rois, it is considered not-perforated.
        // This test checks that n0<>n2 comparisons ignore n1 because it's logical even though it has no dcore rois,
        // and n2<>n4 comparison ignores n3 because it's logical even though it _has_ dcore rois.
        if (ni == n[1]) continue;
        ni->getNodeAnnotation().m_dcoreROIs.resize(4);
        ni->getNodeAnnotation().perforationDim = perforationDim;
        if (!ni->isLogicalOperation())
        {
            ASSERT_TRUE(ni->getNodeAccessPattern());
            ASSERT_EQ(ni->getNodeAccessPattern()->getIndexSpaceDim(ni->getInput(0), 0), perforationDim);
            ASSERT_EQ(ni->getNodeAccessPattern()->getIndexSpaceDim(ni->getOutput(0), 0), perforationDim);
        }
    }

    auto crp = getProfiler();

    std::vector<bool> expectedSamePerforationResult = {
        true,  // n0<>n2 - perforated the same
        false,  // n2<>n4 - perforated differently
    };

    for (size_t i = 0; i < 2; i++)
    {
        auto outProfile = crp->outputProfile(2 * i, 0);
        EXPECT_EQ(expectedSamePerforationResult.at(i), outProfile.localized)
            << "Failure in n" << 2 * i << " output profile";
        auto inProfile = crp->inputProfile(2 * i + 2, 0);  // 2i+2 input would match exp[i]
        EXPECT_EQ(expectedSamePerforationResult.at(i), inProfile.localized)
            << "Failure in n" << 2 * i + 2 << " input profile";
    }
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_perforation_compatibility_for_unperforated_tensors)
{
    // [in]->n0->[t0]->n1->[t1]
    // n0 and n1 are perforated the same, but t0 dims are not mapped to the perforation dim.
    initBundle(2);
    initDB();

    setNodesPerforationGroup({0, 1}, 1);

    unsigned perforationDim = 199;
    ASSERT_TRUE(n[0]->getNodeAccessPattern());
    ASSERT_TRUE(n[1]->getNodeAccessPattern());
    // Make sure t0 dims are not mapped to the perforation dim -> should not be localized
    ASSERT_EQ(n[0]->getOutput(0), n[1]->getInput(0));
    for (auto tensorDim = 0; tensorDim < n[0]->getOutput(0)->getDim(); tensorDim++)
    {
        ASSERT_NE(n[0]->getNodeAccessPattern()->getIndexSpaceDim(n[0]->getOutput(0), tensorDim), perforationDim);
        ASSERT_NE(n[1]->getNodeAccessPattern()->getIndexSpaceDim(n[1]->getInput(0), tensorDim), perforationDim);
    }
    // Add dcore rois to simulate actual perforation. The ROI details are irrelevant
    for (auto& ni : n)
    {
        ni->getNodeAnnotation().m_dcoreROIs.resize(4);
        ni->getNodeAnnotation().perforationDim = perforationDim;
    }

    auto crp = getProfiler();

    auto outProfile = crp->outputProfile(0, 0);
    EXPECT_EQ(false, outProfile.localized);
    auto inProfile = crp->inputProfile(1, 0);
    EXPECT_EQ(false, inProfile.localized);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_read_mme_nof_accesses_from_the_strategy)
{
    auto  inA     = newTensor();
    auto  inB     = newTensor();
    auto  out     = newTensor();
    TSize sizes[] = {128, 128};
    inA->reshape(2, sizes);
    inB->reshape(2, sizes);
    out->reshape(2, sizes);

    synGEMMParams params {};
    auto          gemm = NodeFactory::createNode({inA, inB}, {out}, &params, NodeFactory::gemmNodeTypeName, "gemm");

    bundleNode(BUNDLE_ID, gemm, 0);
    m_nodeChain.push_back(gemm);
    initDB();

    auto mmeSolutionParams                                   = std::make_shared<SolutionParams>();
    mmeSolutionParams->perfAttr.memoryAttrA.accessesPerChip  = 31;
    mmeSolutionParams->perfAttr.memoryAttrA.accessesPerDcore = 23;
    mmeSolutionParams->perfAttr.memoryAttrB.accessesPerChip  = 13;
    mmeSolutionParams->perfAttr.memoryAttrB.accessesPerDcore = 82;

    auto mmeStrategy = std::make_shared<MmeSolution>();
    mmeStrategy->QORs.emplace(gemm, mmeSolutionParams);

    StrategyPtr strategy = std::make_shared<Strategy>(mmeStrategy);

    auto crp = getProfiler(strategy);

    auto inAProfile = crp->inputProfile(0, 0);
    EXPECT_EQ(31, inAProfile.totalReads);
    EXPECT_EQ(23, inAProfile.dcoreReads);
    auto inBProfile = crp->inputProfile(0, 1);
    EXPECT_EQ(13, inBProfile.totalReads);
    EXPECT_EQ(82, inBProfile.dcoreReads);
}

TEST_F(CacheRequirementsProfilerTest, crp_should_determine_all_required_based_on_access_pattern)
{
    auto  fm          = newTensor();
    auto  bias        = newTensor();
    auto  out         = newTensor();
    TSize fmSizes[]   = {1, 32768};
    TSize biasSizes[] = {1, 1};  // will force the bias to be all-required

    fm->reshape(2, fmSizes);
    bias->reshape(2, biasSizes);
    out->reshape(2, fmSizes);

    auto add = NodeFactory::createNode({fm, bias}, {out}, nullptr, "add_f32", "add");

    // Load TPC kernel, so it will have access pattern
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add));
    TpcKernelLoader tpcKernelLoader(&m_graph);
    ASSERT_TRUE(tpcKernelLoader.load());

    bundleNode(BUNDLE_ID, add, 0);
    m_nodeChain.push_back(add);
    initDB();

    auto crp = getProfiler();

    auto inAProfile = crp->inputProfile(0, 0);
    EXPECT_FALSE(inAProfile.allRequired);
    auto inBProfile = crp->inputProfile(0, 1);
    EXPECT_TRUE(inBProfile.allRequired);
}