#include "memory_management/bundle_memory_preprocessor.h"

#include "layered_brain_test.h"
#include "tensor_view_node.h"

using namespace gc::layered_brain;

class BundleMemPreProcTest : public LayeredBrainTest
{
protected:
    static constexpr unsigned bundleId = 33;

    BundleNodes m_bundle;

    void setup(int numNodes)
    {
        createGraph(numNodes);
        m_bundle = bundleNodes(bundleId, 0, numNodes);
    }

    MemoryUsageDB runPreprocessor()
    {
        POCBundleMemoryPreProcessor bmpp {m_graph, m_bundle};
        return bmpp.buildMemUsageDB();
    }

    void
    validateSliceConsumers(const MemoryUsageDB& db, const TensorPtr& slice, const std::unordered_set<int>& expConsumers)
    {
        EXPECT_EQ(expConsumers, db.slices.at(slice).properties.consumingSteps);
    }

    void validateRealSlice(const MemoryUsageDB& db, const TensorPtr& slice, const TensorPtr& expRealSlice) const
    {
        EXPECT_EQ(expRealSlice, db.slices.at(slice).properties.realSlice);
    }
};

TEST_F(BundleMemPreProcTest, bmpp_should_have_step_per_bundle_node_in_order)
{
    setup(10);
    addNodeToChain();  // Add an extra unbundled node

    MemoryUsageDB db = runPreprocessor();

    ASSERT_EQ(10, db.steps.size());
    for (int n = 0; n < 10; n++)
    {
        EXPECT_EQ(m_bundle[n], db.steps[n].sliceNode);
    }
}

TEST_F(BundleMemPreProcTest, bmpp_should_have_slice_entry_for_all_bundle_node_operands)
{
    setup(10);

    MemoryUsageDB db = runPreprocessor();

    for (const NodePtr& node : m_bundle)
    {
        for (const TensorPtr& tensorSlice : node->getOperands())
        {
            ASSERT_NE(db.slices.end(), db.slices.find(tensorSlice));
        }
    }
}

TEST_F(BundleMemPreProcTest, bmpp_should_set_producer_step_for_bundle_produced_slices)
{
    setup(5);

    MemoryUsageDB db = runPreprocessor();

    for (int n = 0; n < 5; n++)
    {
        const TensorPtr& output = m_nodeChain[n]->getOutput(0);
        ASSERT_NO_THROW(db.slices.at(output));
        const auto& outputEntry = db.slices[output];
        ASSERT_TRUE(outputEntry.properties.producingStep.has_value());
        EXPECT_EQ(outputEntry.properties.producingStep, n);
    }
}

TEST_F(BundleMemPreProcTest, bmpp_should_set_consumer_steps_for_bundle_produced_slices)
{
    setup(5);

    MemoryUsageDB db = runPreprocessor();

    for (int n = 0; n < 5; n++)
    {
        const TensorPtr& input = m_nodeChain[n]->getInput(0);
        ASSERT_NO_THROW(db.slices.at(input));
        const auto& inputEntry = db.slices[input];
        ASSERT_EQ(1, inputEntry.properties.consumingSteps.size()) << "no consumers in step " << n;
        EXPECT_EQ(*inputEntry.properties.consumingSteps.begin(), n);
    }
}

TEST_F(BundleMemPreProcTest, bmpp_should_indicate_external_consumers)
{
    // [in]->n0->[t0]->n1...n4->[out]->n5
    //   |         |                   ^
    //   |         |                   |
    //   +---------+-------------------+
    //
    // n5 is out of the bundle. Expect in, t0 and out to have ext consumer indication.

    setup(5);
    addNodeToChain();
    const TensorPtr& in  = m_bundle.front()->getInput(0);
    const TensorPtr& t0  = m_bundle.front()->getOutput(0);
    const TensorPtr& t2  = m_bundle[2]->getOutput(0);
    const TensorPtr& out = m_bundle.back()->getOutput(0);
    GraphEditor::editNode(m_graph, m_nodeChain.back(), [&]() {
        m_nodeChain.back()->addInput(1, in);
        m_nodeChain.back()->addInput(2, t0);
    });

    MemoryUsageDB db = runPreprocessor();

    const auto& inEntry = db.slices.at(in);
    EXPECT_TRUE(inEntry.properties.consumedExternally);

    const auto& t0Entry = db.slices.at(t0);
    EXPECT_TRUE(t0Entry.properties.consumedExternally);

    const auto& outEntry = db.slices.at(out);
    EXPECT_TRUE(outEntry.properties.consumedExternally);

    // Sample one of the other tensors to make sure they are not marked as externally consumed
    const auto& t2Entry = db.slices.at(t2);
    EXPECT_FALSE(t2Entry.properties.consumedExternally);
}

TEST_F(BundleMemPreProcTest, bmpp_should_add_alias_consumers_to_real_slices__fwd_alias)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    //            ^         |
    //            |         |
    //            +--alias--+
    setup(4);
    const TensorPtr& t0 = m_bundle[0]->getOutput(0);
    const TensorPtr& t1 = m_bundle[1]->getOutput(0);
    const TensorPtr& t2 = m_bundle[2]->getOutput(0);
    t1->setAsAliasSubTensor(t0);

    MemoryUsageDB db = runPreprocessor();

    validateSliceConsumers(db, t0, {1, 2});

    validateRealSlice(db, t0, nullptr);
    validateRealSlice(db, t1, t0);
    validateRealSlice(db, t2, nullptr);
}

TEST_F(BundleMemPreProcTest, bmpp_should_add_alias_consumers_to_real_slices__bwd_alias)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    //            |         ^
    //            |         |
    //            +--alias--+
    setup(4);
    const TensorPtr& t0 = m_bundle[0]->getOutput(0);
    const TensorPtr& t1 = m_bundle[1]->getOutput(0);
    const TensorPtr& t2 = m_bundle[2]->getOutput(0);
    t0->setAsAliasSubTensor(t1);

    MemoryUsageDB db = runPreprocessor();

    validateSliceConsumers(db, t1, {1, 2});

    validateRealSlice(db, t0, t1);
    validateRealSlice(db, t1, nullptr);
    validateRealSlice(db, t2, nullptr);
}

TEST_F(BundleMemPreProcTest, bmpp_should_add_alias_consumers_to_real_slices__alias_chain)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    //            |^         |         ^
    //            ||         |         |
    //            |+--alias--+         |
    //            +--------alias-------+
    setup(4);
    const TensorPtr& t0 = m_bundle[0]->getOutput(0);
    const TensorPtr& t1 = m_bundle[1]->getOutput(0);
    const TensorPtr& t2 = m_bundle[2]->getOutput(0);
    t0->setAsAliasSubTensor(t2);
    t1->setAsAliasSubTensor(t0);

    MemoryUsageDB db = runPreprocessor();

    validateSliceConsumers(db, t2, {1, 2, 3});

    validateRealSlice(db, t0, t2);
    validateRealSlice(db, t1, t2);
    validateRealSlice(db, t2, nullptr);
}

enum class JoinTestType
{
    INTERNAL_CONCAT,
    TENSOR_VIEW
};

class BundleMemPreProcJoinTest
: public BundleMemPreProcTest
, public ::testing::WithParamInterface<JoinTestType>
{
protected:
    NodePtr getJoinNode(const JoinTestType& aggType, const TensorVector& tensors)
    {
        switch (aggType)
        {
            case JoinTestType::INTERNAL_CONCAT:
            {
                synConcatenateParams params {};
                params.axis     = 0;
                const auto& out = newTensor();
                TSize       concatOutputSize =
                    std::accumulate(tensors.begin(), tensors.end(), 0, [&](uint64_t acc, const TensorPtr& t) {
                        return acc + (t ? t->getSizeInElements(params.axis) : 0);
                    });
                SizeArray outSizes       = out->getAllSizesInElements();
                outSizes.at(params.axis) = concatOutputSize;
                out->reshape(out->getDim(), outSizes.data());
                return NodeFactory::createNode(tensors,
                                               {out},
                                               &params,
                                               NodeFactory::concatenateNodeInternalTypeName,
                                               "concat");
            }
            case JoinTestType::TENSOR_VIEW:
            {
                auto tensorview = std::make_shared<TensorViewNode>(newTensor(), false, "tensorview");
                for (const TensorPtr& t : tensors)
                {
                    tensorview->addView(t, SizeVector(t->getDim(), 0));
                }
                return tensorview;
            }
        }
        return nullptr;  // dummy - can't get here
    }

    NodePtr addJoin(const JoinTestType& aggType, const TensorVector& tensors, bool bundleJoin = true)
    {
        NodePtr join = getJoinNode(aggType, tensors);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, join));
        for (const TensorPtr& t : tensors)
        {
            t->setAsAliasSubTensor(join->getOutput(0));
        }
        if (bundleJoin)
        {
            bundleNode(bundleId, join, 41);
            m_bundle.push_back(join);
        }
        return join;
    }
};

INSTANTIATE_TEST_SUITE_P(join_types,
                         BundleMemPreProcJoinTest,
                         ::testing::Values(JoinTestType::INTERNAL_CONCAT, JoinTestType::TENSOR_VIEW));

TEST_P(BundleMemPreProcJoinTest, bmpp_should_indicate_joined_slices)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]->join
    //             |         |                         ^
    //             |         |                         |
    //             +---------+-------------------------+

    setup(4);
    const TensorPtr& in  = m_bundle[0]->getInput(0);
    const TensorPtr& t0  = m_bundle[0]->getOutput(0);
    const TensorPtr& t1  = m_bundle[1]->getOutput(0);
    const TensorPtr& t2  = m_bundle[2]->getOutput(0);
    const TensorPtr& out = m_bundle.back()->getOutput(0);

    NodePtr join = addJoin(GetParam(), {t0, t1, out});

    MemoryUsageDB db = runPreprocessor();

    const auto& inEntry = db.slices.at(in);
    EXPECT_EQ(nullptr, inEntry.properties.joinedBy);

    const auto& t0Entry = db.slices.at(t0);
    EXPECT_EQ(join, t0Entry.properties.joinedBy);
    EXPECT_EQ(std::unordered_set<int>({1}), t0Entry.properties.consumingSteps);

    const auto& t1Entry = db.slices.at(t1);
    EXPECT_EQ(join, t1Entry.properties.joinedBy);
    EXPECT_EQ(std::unordered_set<int>({2}), t1Entry.properties.consumingSteps);

    const auto& t2Entry = db.slices.at(t2);
    EXPECT_EQ(nullptr, t2Entry.properties.joinedBy);

    const auto& outEntry = db.slices.at(out);
    EXPECT_EQ(join, outEntry.properties.joinedBy);
    EXPECT_EQ(std::unordered_set<int>({}), outEntry.properties.consumingSteps);
}

TEST_P(BundleMemPreProcJoinTest, bmpp_should_indicate_joined_slices_for_non_bundled_join)
{
    // [in]->n0->[t0]->n1->[out]->extJoin
    //             |                ^
    //             |                |
    //             +----------------+

    setup(2);
    const TensorPtr& t0  = m_bundle[0]->getOutput(0);
    const TensorPtr& out = m_bundle.back()->getOutput(0);

    NodePtr extJoin = addJoin(GetParam(), {t0, out}, false);

    MemoryUsageDB db = runPreprocessor();

    const auto& t0Entry = db.slices.at(t0);
    EXPECT_EQ(extJoin, t0Entry.properties.joinedBy);
    EXPECT_EQ(std::unordered_set<int>({1}), t0Entry.properties.consumingSteps);

    const auto& outEntry = db.slices.at(out);
    EXPECT_EQ(extJoin, outEntry.properties.joinedBy);
    EXPECT_EQ(std::unordered_set<int>({}), outEntry.properties.consumingSteps);
}

TEST_P(BundleMemPreProcJoinTest, bmpp_should_not_add_alias_consumers_to_aggr_output)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[out]
    //             |         |
    //             +---------+-->join->[aggOut]
    // t0 and t1 are aliases to aggOut
    setup(3);
    const TensorPtr& t0          = m_bundle[0]->getOutput(0);
    const TensorPtr& t1          = m_bundle[1]->getOutput(0);
    NodePtr          join        = addJoin(GetParam(), {t0, t1});
    const TensorPtr& aggOut      = join->getOutput(0);

    MemoryUsageDB db = runPreprocessor();

    // The semantics of the join output are different from those of intermediate slices: When an intermediate
    // slice, s0, is an alias of another intermediate slice, s1, we want s0's consumers to be considered also s1's
    // consumers (since they really consume s1). But a join output is always aliased by intermediate slices and
    // we never want their consumers to be considered the output's, because a- we don't free it and b- we don't want it
    // to seem like an intermediate slice.
    validateSliceConsumers(db, aggOut, {});
}

TEST_P(BundleMemPreProcJoinTest, bmpp_should_aggregate_aliases_of_joined_reduced_slices)
{
    // [t0]->n0->[t1]->n1->[t2]-->n3->[t4]->Join->[t5]
    //                 n2->[t3]----^
    // t2 and t3 are aliases of t4 (n3 acts as Reduction node), which is an alias of t5
    setup(4);
    const TensorPtr& t2 = m_bundle[1]->getOutput(0);
    const TensorPtr& t3 = m_bundle[2]->getOutput(0);
    const TensorPtr& t4 = m_bundle[3]->getOutput(0);
    t2->setAsAliasSubTensor(t4);
    t3->setAsAliasSubTensor(t4);

    GraphEditor::editNode(m_graph, m_bundle[2], [&]() { m_bundle[2]->removeInput(t2); });
    GraphEditor::editNode(m_graph, m_bundle[3], [&]() { m_bundle[3]->addInput(1, t2); });
    NodePtr join = addJoin(GetParam(), {t4});

    MemoryUsageDB db = runPreprocessor();

    validateRealSlice(db, t2, t4);
    validateRealSlice(db, t3, t4);
    validateRealSlice(db, t4, nullptr);
}

enum class ForkTestType
{
    INTERNAL_SPLIT,
    TENSOR_VIEW
};

class BundleMemPreProcForkTest
: public BundleMemPreProcTest
, public ::testing::WithParamInterface<ForkTestType>
{
protected:
    TensorVector m_t {5};
    TensorVector m_in {3};
    NodePtr      m_disaggr;

    // setup [in0]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[t3]->n4->[t4]
    // And transform it into:
    //
    //               +-->[in0]->n0->[t0]->n1->[t1]
    // [in]->disaggr-+-->[in1]->n2->[t2]
    //               +-->[in2]->n3->[t3]->n4->[t4]
    // Where in0-2 are aliases to 'in'
    void setup(bool bundleFork = true)
    {
        BundleMemPreProcTest::setup(5);
        for (int i = 0; i < 5; i++)
        {
            m_t[i] = m_nodeChain[i]->getOutput(0);
        }
        m_in[0] = m_nodeChain.front()->getInput(0);
        m_in[1] = newTensor();
        m_in[2] = newTensor();
        GraphEditor::replaceInput(m_graph, m_nodeChain[2], 0, m_in[1]);
        GraphEditor::replaceInput(m_graph, m_nodeChain[3], 0, m_in[2]);

        m_disaggr = addFork(GetParam(), m_in, bundleFork);
    }

    NodePtr addFork(const ForkTestType& aggType, const TensorVector& tensors, bool bundleFork)
    {
        NodePtr disaggr = getForkNode(aggType, tensors);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, disaggr));
        for (const TensorPtr& t : tensors)
        {
            t->setAsAliasSubTensor(disaggr->getInput(0));
        }
        if (bundleFork)
        {
            bundleNode(bundleId, disaggr, 41);
            m_bundle.insert(m_bundle.begin(), disaggr);  // Add to the start
        }
        return disaggr;
    }

    NodePtr getForkNode(const ForkTestType& disaggType, const TensorVector& tensors)
    {
        switch (disaggType)
        {
            case ForkTestType::INTERNAL_SPLIT:
            {
                synSplitParams params {};
                return NodeFactory::createNode({newTensor()},
                                               tensors,
                                               &params,
                                               NodeFactory::splitNodeInternalTypeName,
                                               "split");
            }
            case ForkTestType::TENSOR_VIEW:
            {
                auto tensorview = std::make_shared<TensorViewNode>(newTensor(), true, "tensorview");
                for (const TensorPtr& t : tensors)
                {
                    tensorview->addView(t, SizeVector(t->getDim(), 0));
                }
                return tensorview;
            }
        }
        return nullptr;  // dummy - can't get here
    }
};

INSTANTIATE_TEST_SUITE_P(fork_types,
                         BundleMemPreProcForkTest,
                         ::testing::Values(ForkTestType::INTERNAL_SPLIT, ForkTestType::TENSOR_VIEW));

TEST_P(BundleMemPreProcForkTest, bmpp_should_indicate_disggregated_slices)
{
    // setup [in0]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[t3]->n4->[t4]
    // And transform it into:
    //
    //               +-->[in0]->n0->[t0]->n1->[t1]
    // [in]->disaggr-+-->[in1]->n2->[t2]
    //               +-->[in2]->n3->[t3]->n4->[t4]
    // Where in0-2 are aliases to 'in'
    setup();

    MemoryUsageDB db = runPreprocessor();

    // Expect join indication for in* slices and no join for t*
    for (const TensorPtr& in : m_in)
    {
        const MemoryUsageDB::SliceEntry& entry = db.slices.at(in);
        EXPECT_EQ(m_disaggr, entry.properties.forkedBy);
        EXPECT_FALSE(entry.properties.producingStep);
    }
    for (const TensorPtr& t : m_t)
    {
        const MemoryUsageDB::SliceEntry& entry = db.slices.at(t);
        EXPECT_EQ(nullptr, entry.properties.forkedBy);
    }
}

TEST_P(BundleMemPreProcForkTest, bmpp_should_not_add_alias_consumers_to_disaggr_input)
{
    // setup [in0]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[t3]->n4->[t4]
    // And transform it into:
    //
    //               +-->[in0]->n0->[t0]->n1->[t1]
    // [in]->disaggr-+-->[in1]->n2->[t2]
    //               +-->[in2]->n3->[t3]->n4->[t4]
    // Where in0-2 are aliases to 'in'
    setup();

    MemoryUsageDB db = runPreprocessor();

    // Only the fork (step 0) is expected to be considered a consumer of its input
    validateSliceConsumers(db, m_disaggr->getInput(0), {0});
    // And the fork is not expected to be considered a consumer of its outputs
    // Notice: ni is now step i+1 since the fork was pushed to the front of the bundle steps
    validateSliceConsumers(db, m_in[0], {1});  // in0 is consumed by n0 which is now step1
    validateSliceConsumers(db, m_in[1], {3});  // in1 is consumed by n2 which is now step3
    validateSliceConsumers(db, m_in[2], {4});  // in2 is consumed by n3 which is now step4
}

TEST_P(BundleMemPreProcForkTest, bmpp_should_indicate_disaggration_for_bundle_inputs_with_disaggr_producer)
{
    // setup [in0]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[t3]->n4->[t4]
    // And transform it into:
    //
    //                  +-->[in0]->n0->[t0]->n1->[t1]
    // [in]->extDisaggr-+-->[in1]->n2->[t2]
    //                  +-->[in2]->n3->[t3]->n4->[t4]
    // Where in0-2 are aliases to 'in' and extDisaggr is not bundled
    setup(false);

    MemoryUsageDB db = runPreprocessor();

    // Expect join indication for in* slices and no join for t*
    for (const TensorPtr& in : m_in)
    {
        const MemoryUsageDB::SliceEntry& entry = db.slices.at(in);
        EXPECT_EQ(m_disaggr, entry.properties.forkedBy);
        EXPECT_FALSE(entry.properties.producingStep);
    }
    for (const TensorPtr& t : m_t)
    {
        const MemoryUsageDB::SliceEntry& entry = db.slices.at(t);
        EXPECT_EQ(nullptr, entry.properties.forkedBy);
    }
}