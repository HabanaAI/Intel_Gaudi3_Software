#include "layered_brain_test.h"
#include "memory_usage_db.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "bundle_memory_directive_executor.h"
#include "gtest/gtest.h"

class POCBundleMemoryDirectiveExecutorTest : public LayeredBrainTest
{
public:
    using Placement = MemoryUsageDB::SliceEntry::Directives::Placement;

protected:
    void SetUp() override
    {
        LayeredBrainTest::SetUp();
        CompilationHalReader::setHalReader(Gaudi2HalReader::instance());
    }
    void initBundle(int numBundleNodes)
    {
        static constexpr unsigned BUNDLE_ID = 12;
        // Create bundle with numBundleNodes
        // Add all bundle nodes to the steps of the DB.
        // Set consuming steps and producing steps of every tensor.
        createGraph(numBundleNodes);
        bundleNodes(BUNDLE_ID, 0, numBundleNodes);

        for (int n = 0; n < numBundleNodes; n++)
        {
            const auto& node = m_nodeChain[n];
            m_db.steps.push_back({n, node});
            const auto& input  = node->getInput(0);
            const auto& output = node->getOutput(0);
            m_db.slices[input].properties.consumingSteps.insert(n);
            m_db.slices[input].properties.immediateConsumingSteps.insert(n);
            m_db.slices[output].properties.producingStep = n;
        }
    }

    bool invokeExecutor(const TensorPtr& slice)
    {
        POCBundleMemoryDirectiveExecutor executor {m_graph, m_db};
        return executor.executeDirectivesFor(slice);
    }

    MemoryUsageDB m_db;
};

TEST_F(POCBundleMemoryDirectiveExecutorTest, directive_executor_should_spill_joined_slice_with_sram_placement)
{
    // [in]->n0->[t0]->n1->[out]
    //             |         ^
    //             |         |
    //             +--alias--+
    // n0 is a fork node and t0 has SRAM placement directive
    initBundle(2);
    TensorPtr t0                           = m_nodeChain[0]->getOutput(0);
    m_db.slices.at(t0).properties.joinedBy = m_nodeChain[1];
    t0->setAsAliasSubTensor(m_nodeChain[1]->getOutput(0));
    m_db.slices.at(t0).directives.placement = Placement::SRAM;

    ASSERT_TRUE(invokeExecutor(t0));

    // Expected t0 to be spilled out of SRAM:
    // [in]->n0->[t0 <sram>]->spill->[t0_spilled]->n1->[out]
    //                                     |             ^
    //                                     |             |
    //                                     +--  alias  --+

    EXPECT_TRUE(t0->inSram());

    const auto& consumers = m_graph.getTensorConsumers(t0);
    ASSERT_EQ(1, consumers.size());
    const auto& spill                = consumers.front();
    const auto& t0Spilled            = spill->getOutput(0);
    const auto& secondLevelConsumers = m_graph.getNodeConsumers(spill);

    EXPECT_FALSE(spill->isLogicalOperation()) << "Spill node can't be logical";
    ASSERT_EQ(1, secondLevelConsumers.size());
    EXPECT_EQ(m_nodeChain[1], *secondLevelConsumers.begin());

    EXPECT_FALSE(t0->isAliasedTensor());
    EXPECT_TRUE(t0Spilled->isAliasedTensor());
    EXPECT_EQ(m_nodeChain[1]->getOutput(0), t0Spilled->getAliasTensor());

    // Expect also the operation index of the fill node to be the same as the consumer, so that it will be scheduled
    // right before it.
    ASSERT_TRUE(spill->getNodeAnnotation().bundleInfo.is_set());
    const auto& producerAnn = *m_nodeChain[0]->getNodeAnnotation().bundleInfo;
    const auto& spillAnn    = *spill->getNodeAnnotation().bundleInfo;
    EXPECT_EQ(producerAnn.bundleIndex, spillAnn.bundleIndex);
    EXPECT_EQ(producerAnn.operationIndex, spillAnn.operationIndex);
}

TEST_F(POCBundleMemoryDirectiveExecutorTest, HBM_join_so_directive_executor_should_do_nothing)
{
    initBundle(2);
    const TensorPtr& t0 = m_nodeChain[0]->getOutput(0);
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    m_db.slices.at(t0).directives.placement = Placement::HBM;
    m_db.slices.at(t0).properties.joinedBy  = m_nodeChain[1];

    ASSERT_TRUE(invokeExecutor(t0));

    ASSERT_TRUE(t0->inDram());
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    ASSERT_EQ(m_graph.getNumNodes(), 2);  // memcpy node should not be inserted
}

TEST_F(POCBundleMemoryDirectiveExecutorTest, directive_executor_should_fill_forked_sram_slice)
{
    // [in]->n0->[t0]->n1->[out]
    //   ^         |
    //   |         |
    //   +--alias--+
    // n0 is a fork node and t0 has SRAM placement directive
    initBundle(2);
    TensorPtr t0                        = m_nodeChain[0]->getOutput(0);
    m_db.slices[t0].properties.forkedBy = m_nodeChain[0];
    t0->setAsAliasSubTensor(m_nodeChain[0]->getInput(0));
    m_db.slices[t0].directives.placement = Placement::SRAM;

    ASSERT_TRUE(invokeExecutor(t0));

    // Expect t0 to be filled into SRAM:
    // [in]->n0->[t0_preFill]->fill->[t0 <sram>]->n1->[out]
    //   ^             |
    //   |             |
    //   +--  alias  --+
    EXPECT_TRUE(t0->inSram());

    const auto& fill = m_graph.getTensorProducer(t0);
    ASSERT_NE(nullptr, fill);
    const auto& t0PreFill            = fill->getInput(0);
    const auto& secondLevelProducers = m_graph.getNodeProducers(fill);

    EXPECT_FALSE(fill->isLogicalOperation()) << "Fill node can't be logical";
    ASSERT_EQ(1, secondLevelProducers.size());
    EXPECT_EQ(m_nodeChain[0], *secondLevelProducers.begin());

    EXPECT_FALSE(t0->isAliasedTensor());
    EXPECT_TRUE(t0PreFill->isAliasedTensor());
    EXPECT_EQ(m_nodeChain[0]->getInput(0), t0PreFill->getAliasTensor());

    // Expect also the operation index of the fill node to be the same as the consumer, so that it will be scheduled
    // right before it.
    ASSERT_TRUE(fill->getNodeAnnotation().bundleInfo.is_set());
    const auto& consumerAnn = *m_nodeChain[1]->getNodeAnnotation().bundleInfo;
    const auto& fillAnn     = *fill->getNodeAnnotation().bundleInfo;
    EXPECT_EQ(consumerAnn.bundleIndex, fillAnn.bundleIndex);
    EXPECT_EQ(consumerAnn.operationIndex, fillAnn.operationIndex);
}

TEST_F(POCBundleMemoryDirectiveExecutorTest, UNSET_join_dramInit_so_directive_executor_should_leave_it_on_dram)
{
    initBundle(2);
    const TensorPtr& t0 = m_nodeChain[0]->getOutput(0);
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    m_db.slices.at(t0).directives.placement = Placement::UNSET;
    m_db.slices.at(t0).properties.joinedBy  = m_nodeChain[1];
    t0->setTensorInDram();

    ASSERT_TRUE(invokeExecutor(t0));

    EXPECT_TRUE(t0->inDram());  // should be the same placement as before the execute
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    ASSERT_EQ(m_graph.getNumNodes(), 2);  // memcpy node should not be inserted
}

TEST_F(POCBundleMemoryDirectiveExecutorTest, UNSET_join_sramInit_so_directive_executor_should_leave_it_on_sram)
{
    initBundle(2);
    const TensorPtr& t0 = m_nodeChain[0]->getOutput(0);
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    m_db.slices.at(t0).directives.placement = Placement::UNSET;
    m_db.slices.at(t0).properties.joinedBy  = m_nodeChain[1];
    t0->setTensorInSram();

    ASSERT_TRUE(invokeExecutor(t0));

    EXPECT_TRUE(t0->inSram());  // should be the same placement as before the execute
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    ASSERT_EQ(m_graph.getNumNodes(), 2);  // memcpy node should not be inserted
}

TEST_F(POCBundleMemoryDirectiveExecutorTest, directive_executor_should_place_the_slice_on_sram_without_memcpy_insertion)
{
    initBundle(2);
    const TensorPtr& t0 = m_nodeChain[0]->getOutput(0);
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    m_db.slices.at(t0).directives.placement = Placement::SRAM;

    ASSERT_TRUE(invokeExecutor(t0));

    ASSERT_TRUE(t0->inSram());
    ASSERT_EQ(m_graph.getNumNodes(), 2);            // memcpy node should not be inserted
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);     // input should not change
}

TEST_F(POCBundleMemoryDirectiveExecutorTest, HBM_directive_slice_so_directive_executor_should_set_slice_on_dram)
{
    initBundle(2);
    const TensorPtr& t0 = m_nodeChain[0]->getOutput(0);
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
    m_db.slices.at(t0).directives.placement = Placement::HBM;

    ASSERT_TRUE(invokeExecutor(t0));

    ASSERT_TRUE(t0->inDram());
    ASSERT_EQ(m_graph.getNumNodes(), 2);
    ASSERT_EQ(m_nodeChain[1]->getInput(0), t0);
}