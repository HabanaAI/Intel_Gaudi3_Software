#include "layered_brain_test.h"
#include "bundle_memory_insulator.h"

class POCBundleMemoryInsulatorTest : public LayeredBrainTest
{
protected:
    // Add external consumer to the output of step 'stepIdx' of the bundle.
    NodePtr addExternalConsumer(int stepIdx)
    {
        NodePtr externalConsumer = newNode(77);  // created as [out]->extCons->[t_new]

        const TensorPtr& t = m_nodeChain[stepIdx]->getOutput(0);
        externalConsumer->replaceInput(0, t);  // changed to [t]->extCons->[t_new]

        GraphEditor::addNode(m_graph, externalConsumer);

        return externalConsumer;
    }

    struct DoNothingSliceInsulator : public POCBundleMemorySliceInsulator
    {
        void insulate(NodePtr node, TensorPtr slice, BundleInsulation* insulation) override {}
    } m_doNothingSliceInsulator;

    BundleNodes insulateOutputs(const BundleNodes& bundle)
    {
        POCBundleMemoryInsulator insulator {m_graph, bundle, &m_doNothingSliceInsulator, nullptr};
        return insulator.getInsulatedBundle();
    }

    BundleNodes insulateInputs(const BundleNodes& bundle)
    {
        POCBundleMemoryInsulator insulator {m_graph, bundle, nullptr, &m_doNothingSliceInsulator};
        return insulator.getInsulatedBundle();
    }

    // Find the node that insulates the output of step 'stepIdx' (assumed to be intermediate)
    NodePtr findInsulatingNode(int stepIdx)
    {
        const TensorPtr& output = m_nodeChain[stepIdx]->getOutput(0);
        for (const NodePtr& consumer : m_graph.getTensorConsumers(output))
        {
            if (std::find(m_nodeChain.begin(), m_nodeChain.end(), consumer) == m_nodeChain.end())
            {
                return consumer;
            }
        }
        return nullptr;
    }
};

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_not_change_pure_intermediate_slices)
{
    // [in]->n0->[t0]->n1->[out]
    createGraph(2);
    auto      bundle = bundleNodes(0, 0, 2);
    TensorPtr t0     = bundle[0]->getOutput(0);

    BundleNodes insulatedBundle = insulateOutputs(bundle);

    ASSERT_EQ(2, insulatedBundle.size());
    EXPECT_EQ(bundle[0], insulatedBundle[0]);
    EXPECT_EQ(t0, insulatedBundle[0]->getOutput(0));
    EXPECT_EQ(bundle[1], insulatedBundle[1]);
    EXPECT_EQ(t0, insulatedBundle[1]->getInput(0));
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_add_join_for_ext_consumed_intermediate_slices)
{
    // [in]->n0->[t0]->n1->[out]
    //             |
    //             +-->extCons
    // n0 and n1 are bundled.
    createGraph(2);
    TensorPtr   t0      = m_nodeChain[0]->getOutput(0);
    NodePtr     extCons = addExternalConsumer(0);
    BundleNodes bundle  = bundleNodes(54, 0, 2);

    BundleNodes insulatedBundle = insulateOutputs(bundle);

    // Expected:
    // [in]->n0->[t0_tile]->n1->[out]
    //             |
    //             +-->join->[t0]->extCons
    // n0, n1 and join are bundled.
    // Bundle order is n0, join, n1

    const TensorPtr& t0_tile = m_nodeChain[0]->getOutput(0);
    EXPECT_NE(t0, t0_tile);
    EXPECT_EQ(t0_tile, m_nodeChain[1]->getInput(0));
    NodePtr join = findInsulatingNode(0);
    ASSERT_NE(nullptr, join);
    ASSERT_EQ(1, join->getNumOutputs());
    EXPECT_EQ(t0, join->getOutput(0));
    BundleNodes expBundle = {m_nodeChain[0], join, m_nodeChain[1]};
    EXPECT_EQ(expBundle, insulatedBundle);
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_retarget_internal_aliases)
{
    //            +--alias--+--alias---+
    //            |         |          |
    //            |         v          |
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    //                      |
    //                      +-->extCons
    // n0-n3 are bundled.
    createGraph(4);
    TensorPtr t0 = m_nodeChain[0]->getOutput(0);
    TensorPtr t1 = m_nodeChain[1]->getOutput(0);
    TensorPtr t2 = m_nodeChain[2]->getOutput(0);
    t0->setAsAliasSubTensor(t1);
    t2->setAsAliasSubTensor(t1);
    NodePtr     extCons = addExternalConsumer(1);
    BundleNodes bundle  = bundleNodes(54, 0, 4);

    insulateOutputs(bundle);

    // Expected:
    //            +---alias----+---alias----+
    //            |            |            |
    //            |            v            |
    // [in]->n0->[t0]->n1->[t1_tile]->n2->[t2]->n3->[out]
    //                         |
    //                         +-->join->[t1]->extCons

    const TensorPtr& t1_tile = m_nodeChain[1]->getOutput(0);
    EXPECT_EQ(t1_tile, t0->getAliasTensor());
    EXPECT_EQ(t1_tile, t2->getAliasTensor());
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_set_alias_of_join_node)
{
    // [in]->n0->[t0]->n1->[out]
    //             |
    //             +-->extCons
    // n0 and n1 are bundled.
    createGraph(2);
    TensorPtr   t0      = m_nodeChain[0]->getOutput(0);
    NodePtr     extCons = addExternalConsumer(0);
    BundleNodes bundle  = bundleNodes(54, 0, 2);

    insulateOutputs(bundle);

    // Expected:
    // [in]->n0->[t0_tile]->n1->[out]
    //             :   |
    //             :   +-->join->[t0]->extCons
    //             :                   ^
    //             :                   :
    //             + - - alias - - - - +
    // n0, n1 and join are bundled.
    // Bundle order is n0, join, n1

    const TensorPtr& t0_tile = m_nodeChain[0]->getOutput(0);
    EXPECT_EQ(t0, t0_tile->getAliasTensor());
    NodePtr aggr = findInsulatingNode(0);
    ASSERT_NE(nullptr, aggr);
    ASSERT_TRUE(aggr->isLogicalOperation());
    EXPECT_TRUE(std::static_pointer_cast<LogicalOpNode>(aggr)->getRunLogicalOperationDone());
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_handle_ext_consumed_alias)
{
    //            +--alias--+
    //            |         |
    //            |         v
    // [in]->n0->[t0]->n1->[t1]->n2->[out]
    //            |
    //            +-->extCons
    // n0-n2 are bundled.
    createGraph(3);
    TensorPtr t0 = m_nodeChain[0]->getOutput(0);
    TensorPtr t1 = m_nodeChain[1]->getOutput(0);
    t0->setAsAliasSubTensor(t1);
    NodePtr     extCons = addExternalConsumer(0);
    BundleNodes bundle  = bundleNodes(322, 0, 3);

    BundleNodes insulatedBundle = insulateOutputs(bundle);

    // Currently expected to plant memcpy
    //            +----alias-----+
    //            |              |
    //            |              v
    // [in]->n0->[t0_tile]->n1->[t1]->n2->[out]
    //            |
    //            +-->memcpy->[t0]-->extCons
    // n0-n2 are bundled.
    const TensorPtr& t0_tile = m_nodeChain[0]->getOutput(0);
    EXPECT_NE(t0, t0_tile);
    EXPECT_EQ(t1, t0_tile->getAliasTensor());
    EXPECT_FALSE(t0->isAliasedTensor());
    NodePtr memcpy = findInsulatingNode(0);
    ASSERT_NE(nullptr, memcpy);
    EXPECT_FALSE(memcpy->isLogicalOperation());  // All we care about is that it does not require aliasing.
    EXPECT_EQ(t0, memcpy->getOutput(0));
    EXPECT_EQ(t0, extCons->getInput(0));
    BundleNodes expBundle = bundle;
    expBundle.insert(std::next(expBundle.begin()), memcpy);
    EXPECT_EQ(expBundle, insulatedBundle);
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_handle_ext_consumed_ext_alias)
{
    // [in]->n0->[t0]->n1->[out]
    //            :|
    //            :+-->extCons->[tExt]
    //            :              ^
    //            :              :
    //            +- - alias - - +
    // n0, n1 are bundled.
    createGraph(2);
    NodePtr   extCons = addExternalConsumer(0);
    TensorPtr t0      = extCons->getInput(0);
    TensorPtr tExt    = extCons->getOutput(0);
    t0->setAsAliasSubTensor(tExt);

    BundleNodes bundle = bundleNodes(29, 0, 2);
    insulateOutputs(bundle);

    // Expect
    // [in]->n0->[t0_tile]->n1->[out]
    //            :  |
    //            :  +-->join->[ t0 ]->extCons->[tExt]
    //            :                     ^ :               ^
    //            :                     : :               :
    //            +- - - - alias - - - -+ + - - alias - - +
    const TensorPtr& t0_tile = m_nodeChain[0]->getOutput(0);
    NodePtr          aggr    = findInsulatingNode(0);
    EXPECT_NE(aggr, extCons);
    EXPECT_TRUE(aggr->isLogicalOperation());
    EXPECT_EQ(t0, t0_tile->getAliasTensor());
    EXPECT_EQ(tExt, t0->getAliasTensor());
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_not_change_forked_bundle_inputs)
{
    // [in]->n0->[t0]->n1->[out]
    createGraph(2);
    TensorPtr in  = m_nodeChain[0]->getInput(0);
    TensorPtr t0  = m_nodeChain[1]->getInput(0);
    TensorPtr out = m_nodeChain[1]->getOutput(0);

    // Change into [in]->split->[t0]->n1->[out]
    synSplitParams splitParams {0};
    NodePtr split = NodeFactory::createNode({in}, {t0}, &splitParams, NodeFactory::splitNodeInternalTypeName, "SPLIT");
    ASSERT_EQ(REPLACE_NODE_SUCCESS, GraphEditor::replaceNodes(m_graph, {m_nodeChain[0]}, {split}));
    m_nodeChain[0] = split;

    BundleNodes bundle          = bundleNodes(0x23, 0, 2);
    BundleNodes insulatedBundle = insulateInputs(bundle);

    // Expect no change: [in]->split->[t0]->n1->[out]
    auto exeSorted = m_graph.getExeSortedNodes();
    EXPECT_EQ(BundleNodes(exeSorted.begin(), exeSorted.end()), bundle);
    EXPECT_EQ(insulatedBundle, bundle);
    EXPECT_EQ(t0, insulatedBundle.front()->getOutput(0));
    EXPECT_EQ(t0, insulatedBundle.back()->getInput(0));
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_fork_directly_accessed_bundle_inputs)
{
    // [in]->n0->[t0]->n1->[out]
    createGraph(2);
    TensorPtr in = m_nodeChain[0]->getInput(0);
    TensorPtr t0 = m_nodeChain[1]->getInput(0);

    BundleNodes bundle          = bundleNodes(0x55, 0, 2);
    BundleNodes insulatedBundle = insulateInputs(bundle);

    // Expect
    // [in]->fork->[inTile]->n0->[t0]->n1->[out]
    //   ^                    |
    //   +--------alias-------+

    TensorPtr inTile = m_nodeChain[0]->getInput(0);
    EXPECT_NE(in, inTile);
    EXPECT_EQ(inTile, m_nodeChain[0]->getInput(0));
    EXPECT_EQ(t0, m_nodeChain[1]->getInput(0)) << "Expected only bundle input to get insulated";

    NodePtr fork = m_graph.getTensorProducer(inTile);
    ASSERT_NE(nullptr, fork);
    ASSERT_EQ(1, fork->getNumOutputs());
    EXPECT_EQ(in, fork->getInput(0));

    EXPECT_TRUE(fork->isLogicalOperation());
    EXPECT_EQ(inTile->getAliasTensor(), in);
    EXPECT_TRUE(std::static_pointer_cast<LogicalOpNode>(fork)->getRunLogicalOperationDone());

    BundleNodes expBundle = {fork, m_nodeChain[0], m_nodeChain[1]};
    EXPECT_EQ(expBundle, insulatedBundle);
}

TEST_F(POCBundleMemoryInsulatorTest, insulator_should_fill_directly_accessed_input_aliases)
{
    // [in]->n0->[t0]->n1->[out]
    //  |         ^
    //  +--alias--+
    createGraph(2);
    TensorPtr in = m_nodeChain[0]->getInput(0);
    TensorPtr t0 = m_nodeChain[1]->getInput(0);
    in->setAsAliasSubTensor(t0);

    BundleNodes bundle          = bundleNodes(0xfe, 0, 2);
    BundleNodes insulatedBundle = insulateInputs(bundle);

    // Expect:
    // [in]->fill->[inTile]->n0->[t0]->n1->[out]
    //                 |           ^
    //                 +---alias---+

    TensorPtr inTile = m_nodeChain[0]->getInput(0);
    NodePtr   fill   = m_graph.getTensorProducer(inTile);
    EXPECT_FALSE(in->isAliasedTensor());
    EXPECT_TRUE(inTile->isAliasedTensor());
    EXPECT_EQ(t0, inTile->getAliasTensor());
    EXPECT_FALSE(fill->isLogicalOperation());
}