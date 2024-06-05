#include "gtest/gtest.h"
#include <algorithm>
#include <platform/gaudi/graph_compiler/gaudi_graph.h>
#include <graph_compiler/habana_nodes/node_factory.h>
#include <graph_compiler/bundle_plane_graph.h>
#include "graph_optimizer_test.h"
#include "types.h"

class BundlePlaneValidationTest : public GraphOptimizerTest
{
protected:
    TensorPtr createTensor()
    {
        TSize    sizes[] = {128, 128};
        unsigned dim     = ARRAY_SIZE(sizes);
        return std::make_shared<Tensor>(dim, sizes, syn_type_float);
    }

    NodePtr addNop(const TensorVector& inputs, const TensorVector& outputs)
    {
        static uint32_t nofNops = 0;
        NodePtr         nop     = NodeFactory::createNode(inputs,
                                              outputs,
                                              nullptr,
                                              NOP_KERNEL_NAME,
                                              std::string("nop_") + std::to_string(nofNops++));
        GraphEditor::addNode(m_graph, nop);
        return nop;
    }

    NodePtr addNop(const TensorVector& inputs, const TensorVector& outputs, unsigned bundleIndex)
    {
        NodePtr nop = addNop(inputs, outputs);
        setBundleInfo(nop, bundleIndex, BundleType::TPC);
        return nop;
    }

    NodePtr addGemm(const TensorVector& inputs, const TensorVector& outputs)
    {
        static uint32_t nofConvs = 0;
        synGEMMParams   params {};
        NodePtr         nop = NodeFactory::createNode(inputs,
                                              outputs,
                                              &params,
                                              NodeFactory::gemmNodeTypeName,
                                              std::string("gemm_") + std::to_string(nofConvs++));
        GraphEditor::addNode(m_graph, nop);
        return nop;
    }

    NodePtr addGemm(const TensorVector& inputs, const TensorVector& outputs, unsigned bundleIndex)
    {
        NodePtr gemm = addGemm(inputs, outputs);
        setBundleInfo(gemm, bundleIndex, BundleType::MME);
        return gemm;
    }

    void setBundleInfo(NodePtr& node, unsigned bundleIndex, BundleType type)
    {
        node->getNodeAnnotation().bundleInfo.set(BundleInfo(bundleIndex, type));
    }

    GaudiGraph m_graph;
};

TEST_F(BundlePlaneValidationTest, bp_should_be_able_to_create_a_bundle_for_a_node)
{
    auto t   = createTensor();
    auto nop = addNop({}, {t});

    BundlePlane bp {m_graph};
    EXPECT_TRUE(bp.addNodeToBundle(nop));
}

TEST_F(BundlePlaneValidationTest, bp_should_fail_to_create_a_bundle_for_a_bundled_node)
{
    auto t   = createTensor();
    auto nop = addNop({}, {t});

    BundlePlane bp {m_graph};
    EXPECT_TRUE(bp.addNodeToBundle(nop));
    EXPECT_FALSE(bp.addNodeToBundle(nop));
}

TEST_F(BundlePlaneValidationTest, bp_should_fuse_multiple_nodes_into_bundles)
{
    // Given 3 nop nodes graph
    NodeVector nops;
    for (int i = 0; i < 3; i++)
    {
        nops.push_back(addNop({}, {}));
    }
    BundlePlane bp {m_graph};

    // When bundling them
    EXPECT_TRUE(bp.addNodeToBundle(nops[0]));
    EXPECT_TRUE(bp.addNodeToBundle(nops[1]));

    // Then
    const Graph* bpGraph = bp.getBundlePlaneGraph();
    EXPECT_EQ(bpGraph->getNumNodes(), 2) << "2 original nodes are in a single bundle and one free";

    EXPECT_EQ(bp.getBundlePlaneRepresentation(nops[0]), bp.getBundlePlaneRepresentation(nops[1]));
    EXPECT_NE(bp.getBundlePlaneRepresentation(nops[0]), bp.getBundlePlaneRepresentation(nops[2]));

    const BundlePlaneNode* freeNode = BundlePlane::downcastToBundlePlaneNode(bp.getBundlePlaneRepresentation(nops[2]));
    EXPECT_FALSE(freeNode->isBundle());

    const BundlePlaneNode* bundleNode =
        BundlePlane::downcastToBundlePlaneNode(bp.getBundlePlaneRepresentation(nops[0]));
    EXPECT_TRUE(bundleNode->isBundle());
    const NodeVector& nodesInBundle = bundleNode->getBundledNodes();
    EXPECT_EQ(nodesInBundle.size(), 2);
    EXPECT_NE(std::find(nodesInBundle.begin(), nodesInBundle.end(), nops[0]), nodesInBundle.end());
    EXPECT_NE(std::find(nodesInBundle.begin(), nodesInBundle.end(), nops[1]), nodesInBundle.end());
}

TEST_F(BundlePlaneValidationTest, bp_builder_should_build_using_existing_bundle_annotation)
{
    // Given 3 nops in the graph where 0 and 1 are already bundled.
    NodeVector nops;
    for (int i = 0; i < 3; i++)
    {
        nops.push_back(addNop({}, {}));
    }

    setBundleInfo(nops[0], 0, BundleType::TPC);
    setBundleInfo(nops[1], 0, BundleType::TPC);

    // When using the builder to build the bundle plane
    BundlePlane bp(m_graph, true);

    // Then
    const Graph* bpGraph = bp.getBundlePlaneGraph();
    EXPECT_EQ(bpGraph->getNumNodes(), 2) << "2 original nodes are in a single bundle and one free";

    EXPECT_EQ(bp.getBundlePlaneRepresentation(nops[0]), bp.getBundlePlaneRepresentation(nops[1]));
    EXPECT_NE(bp.getBundlePlaneRepresentation(nops[0]), bp.getBundlePlaneRepresentation(nops[2]));

    const BundlePlaneNode* freeNode = BundlePlane::downcastToBundlePlaneNode(bp.getBundlePlaneRepresentation(nops[2]));
    EXPECT_FALSE(freeNode->isBundle());

    const BundlePlaneNode* bundleNode =
        BundlePlane::downcastToBundlePlaneNode(bp.getBundlePlaneRepresentation(nops[0]));
    EXPECT_TRUE(bundleNode->isBundle());
    const NodeVector& nodesInBundle = bundleNode->getBundledNodes();
    EXPECT_EQ(nodesInBundle.size(), 2);
    EXPECT_NE(std::find(nodesInBundle.begin(), nodesInBundle.end(), nops[0]), nodesInBundle.end());
    EXPECT_NE(std::find(nodesInBundle.begin(), nodesInBundle.end(), nops[1]), nodesInBundle.end());
}

TEST_F(BundlePlaneValidationTest, bp_producer_adding_should_eliminate_the_produced_input)
{
    // Given Graph:
    // nop1 --> [t] --> nop2

    TensorPtr t = createTensor();

    NodePtr nop1 = addNop({}, {t});
    NodePtr nop2 = addNop({t}, {});

    BundlePlane bp {m_graph};

    // When bundling nop2 and then adding nop1 to the same bundle:
    bp.addNodeToBundle(nop2);
    bp.addNodeToBundle(nop1);

    // t is expected to be a eliminated
    const auto& bundle = bp.getBundlePlaneRepresentation(nop2);
    EXPECT_EQ(bundle->getNumInputs(), 0);
    EXPECT_EQ(bundle->getNumOutputs(), 0);
}

TEST_F(BundlePlaneValidationTest, bp_producer_adding_should_turn_input_to_output)
{
    // Given Graph:
    // nop1 --> [t] --> nop2
    //             \--> nop3

    TensorPtr t = createTensor();

    NodePtr nop1 = addNop({}, {t});
    NodePtr nop2 = addNop({t}, {});
    NodePtr nop3 = addNop({t}, {});

    BundlePlane bp {m_graph};

    // When bundling nop2 and then adding nop1 to the same bundle:
    bp.addNodeToBundle(nop2);
    bp.addNodeToBundle(nop1);

    // t is expected to be a bundle output and not a bundle input
    const auto& bundle = bp.getBundlePlaneRepresentation(nop2);
    EXPECT_EQ(bundle->getNumInputs(), 0);
    EXPECT_EQ(bundle->getNumOutputs(), 1);
    EXPECT_EQ(bundle->getOutput(0), bp.getBundlePlaneRepresentation(t));
}

TEST_F(BundlePlaneValidationTest, bp_producer_adding_should_add_new_inputs_and_outputs)
{
    // Given graph
    // [t1] --> nop0 --> [t2] --> nop1
    //            \----> [t3]
    TensorPtr t1 = createTensor();
    TensorPtr t2 = createTensor();
    TensorPtr t3 = createTensor();

    NodePtr nop0 = addNop({t1}, {t2, t3});
    NodePtr nop1 = addNop({t2}, {});

    BundlePlane bp {m_graph};

    // When bundling nop1 and then adding nop0
    bp.addNodeToBundle(nop1);
    bp.addNodeToBundle(nop0);

    // t1 is expected to become the only bundle input and t3 the only bundle output
    // [t1] --> (nop0, nop1 bundle) --> [t3]
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop0);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop1));
    EXPECT_EQ(bundleNode->getNumInputs(), 1);
    EXPECT_EQ(bundleNode->getInput(0), bp.getBundlePlaneRepresentation(t1));
    EXPECT_EQ(bundleNode->getNumOutputs(), 1);
    EXPECT_EQ(bundleNode->getOutput(0), bp.getBundlePlaneRepresentation(t3));
}

TEST_F(BundlePlaneValidationTest, bp_producer_adding_should_eliminate_multiple_bundle_input_usages)
{
    // Given graph:
    // nop0 -> [t] --+--> nop1
    //               +--> nop2
    TensorPtr t    = createTensor();
    NodePtr   nop0 = addNop({}, {t});
    NodePtr   nop1 = addNop({t}, {});
    NodePtr   nop2 = addNop({t}, {});

    BundlePlane bp {m_graph};

    // When first adding nop1 and nop2 to the bundle, t is seen as a bundle input twice:
    // nop0 -> [t] ==> bundle
    bp.addNodeToBundle(nop1);
    bp.addNodeToBundle(nop2);

    // And when now adding nop0 to the bundle
    bp.addNodeToBundle(nop0);

    // The bundle should become a node with no inputs or outputs
    EXPECT_TRUE(bp.getBundlePlaneGraph()->isAcyclicGraph());
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop0);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop1));
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop2));
    EXPECT_EQ(bundleNode->getNumInputs(), 0);
    EXPECT_EQ(bundleNode->getNumOutputs(), 0);
}

TEST_F(BundlePlaneValidationTest, bp_producer_adding_should_turn_multiple_bundle_input_usages_to_output)
{
    // Given graph:
    //               +--> nop1
    // nop0 -> [t] --+--> nop2
    //               +--> nop3
    TensorPtr t    = createTensor();
    NodePtr   nop0 = addNop({}, {t});
    NodePtr   nop1 = addNop({t}, {});
    NodePtr   nop2 = addNop({t}, {});
    NodePtr   nop3 = addNop({t}, {});

    BundlePlane bp {m_graph};

    // When first adding nop1 and nop2 to the bundle, t is seen as a bundle input twice:
    // nop0 -> [t] --+==> bundle
    //               +--> nop3
    bp.addNodeToBundle(nop1);
    bp.addNodeToBundle(nop2);

    // And when now adding nop0 to the bundle
    bp.addNodeToBundle(nop0);

    // t should become a bundle output (and stop being an input, or we'll get a cycle)
    EXPECT_TRUE(bp.getBundlePlaneGraph()->isAcyclicGraph());
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop0);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop1));
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop2));
    EXPECT_EQ(bundleNode->getNumInputs(), 0);
    EXPECT_EQ(bundleNode->getNumOutputs(), 1);
    EXPECT_EQ(bundleNode->getOutput(0), bp.getBundlePlaneRepresentation(t));
}

TEST_F(BundlePlaneValidationTest, bp_consumer_adding_should_eliminate_the_consumed_output)
{
    // Given graph:
    // nop0 --> [t] --> nop1
    TensorPtr t    = createTensor();
    NodePtr   nop0 = addNop({}, {t});
    NodePtr   nop1 = addNop({t}, {});

    BundlePlane bp {m_graph};

    // When adding nop0 to the bundle and then nop1
    bp.addNodeToBundle(nop0);
    bp.addNodeToBundle(nop1);

    // t becomes and internal tensor and should be eliminated from the bundle node outputs
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop0);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop1));
    EXPECT_EQ(bundleNode->getNumInputs(), 0);
    EXPECT_EQ(bundleNode->getNumOutputs(), 0);
}

TEST_F(BundlePlaneValidationTest, bp_consumer_adding_should_not_eliminate_externally_consumed_output)
{
    // Given graph:
    // nop0 --> [t] --> nop1
    //             \--> nop2
    TensorPtr t    = createTensor();
    NodePtr   nop0 = addNop({}, {t});
    NodePtr   nop1 = addNop({t}, {});
    NodePtr   nop2 = addNop({t}, {});

    BundlePlane bp {m_graph};

    // When adding nop0 to the bundle and then nop1
    bp.addNodeToBundle(nop0);
    bp.addNodeToBundle(nop1);

    // t is still consumed outside the bundle so it should remain a bundle output
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop0);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop1));
    EXPECT_EQ(bundleNode->getNumInputs(), 0);
    EXPECT_EQ(bundleNode->getNumOutputs(), 1);
    EXPECT_EQ(bundleNode->getOutput(0), bp.getBundlePlaneRepresentation(t));
}

TEST_F(BundlePlaneValidationTest, bp_consumer_adding_should_add_new_inputs_and_outputs)
{
    // Given graph:
    // nop0 --> [t0] --> nop1 --> [t2]
    //          [t1] -----^
    TensorPtr t0 = createTensor();
    TensorPtr t1 = createTensor();
    TensorPtr t2 = createTensor();

    NodePtr nop0 = addNop({}, {t0});
    NodePtr nop1 = addNop({t0, t1}, {t2});

    BundlePlane bp {m_graph};

    // When bundling nop0 and then nop1
    bp.addNodeToBundle(nop0);
    bp.addNodeToBundle(nop1);

    // t0 should be eliminated, t1 becomes the bundle input and t2 the bundle output
    // t1 --> bundle --> t2
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop0);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop1));
    EXPECT_EQ(bundleNode->getNumInputs(), 1);
    EXPECT_EQ(bundleNode->getInput(0), bp.getBundlePlaneRepresentation(t1));
    EXPECT_EQ(bundleNode->getNumOutputs(), 1);
    EXPECT_EQ(bundleNode->getOutput(0), bp.getBundlePlaneRepresentation(t2));
}

TEST_F(BundlePlaneValidationTest, bp_should_ignore_null_inputs_and_outputs)
{
    TensorPtr t    = createTensor();
    NodePtr   nop1 = addNop({nullptr, t, nullptr}, {nullptr, nullptr});
    NodePtr   nop2 = addNop({nullptr, nullptr}, {nullptr, t, nullptr});

    BundlePlane bp {m_graph};

    bp.addNodeToBundle(nop1);
    const NodePtr& bundleNode = bp.getBundlePlaneRepresentation(nop1);
    EXPECT_EQ(bundleNode->getNumOutputs(), 0);
    EXPECT_EQ(bundleNode->getNumInputs(), 1);

    bp.addNodeToBundle(nop2);
    EXPECT_EQ(bundleNode, bp.getBundlePlaneRepresentation(nop2));
    EXPECT_EQ(bundleNode->getNumOutputs(), 0);
    EXPECT_EQ(bundleNode->getNumInputs(), 0);
}

TEST_F(BundlePlaneValidationTest, bp_shdould_validate_single_path_producer)
{
    // Given graph:
    // nop --> mme
    TensorPtr a = createTensor();
    TensorPtr b = createTensor();
    TensorPtr c = createTensor();

    NodePtr nop = addNop({}, {a});
    NodePtr mme = addGemm({a, b}, {c});

    BundlePlane bp {m_graph};
    bp.addNodeToBundle(mme);

    // With nop as a candidate to a strategy containing only the MME
    NodeVector acceptedNodes = {
        mme,
    };
    const auto& stitchedOperand = a;
    const auto& candidateNode   = nop;

    EXPECT_TRUE(bp.validateCandidate(candidateNode, stitchedOperand, acceptedNodes));
}

TEST_F(BundlePlaneValidationTest, bp_should_invalidate_multiple_path_producer)
{
    // Given Graph
    // nop1 -------->[a]-------------> mme -->[c]
    //      \-->[b0]--> nop2 -->[b1]---^
    TensorPtr a  = createTensor();
    TensorPtr b0 = createTensor();
    TensorPtr b1 = createTensor();
    TensorPtr c  = createTensor();

    NodePtr nop1 = addNop({}, {a, b0});
    NodePtr nop2 = addNop({b0}, {b1});
    NodePtr mme  = addGemm({a, b1}, {c});

    BundlePlane bp {m_graph};
    bp.addNodeToBundle(mme);

    // With nop1 as candidate to a strategy containing only the MME
    NodeVector acceptedNodes = {
        mme,
    };
    const auto& nop1StitchedTensor = a;
    EXPECT_FALSE(bp.validateCandidate(nop1, nop1StitchedTensor, acceptedNodes));

    // With nop2 as a candidate to a strategy containing only the MME
    const auto& nop2StitchedTensor = b1;
    EXPECT_TRUE(bp.validateCandidate(nop2, nop2StitchedTensor, acceptedNodes));

    // With nop1 as a candidate to a strategy containing both nop2 and mme
    acceptedNodes = {mme, nop2};
    EXPECT_FALSE(bp.validateCandidate(nop1, nop1StitchedTensor, acceptedNodes));

    // With nop2 as a candidate to a strategy containing both nop1 and mme
    acceptedNodes = {mme, nop1};
    EXPECT_FALSE(bp.validateCandidate(nop2, nop2StitchedTensor, acceptedNodes));
}

TEST_F(BundlePlaneValidationTest, bp_should_consider_a_bundle_as_node_in_the_path_count)
{
    // Given a graph:
    // nop1 ---------------------------------> mme1 (bundle1)
    //    |                                     ^
    //    +----> mme2 (bundle2)                 |
    //            ^                             |
    //            |                             |
    // nop2 ------+--> mme3 (bundle2) ----------+
    //
    // where mme2 and mme3 are bundled together

    TensorPtr nop1Out = createTensor();
    TensorPtr nop2Out = createTensor();

    NodePtr nop1 = addNop({}, {nop1Out});
    NodePtr nop2 = addNop({}, {nop2Out});

    TensorPtr mme1Out = createTensor();
    TensorPtr mme2Out = createTensor();
    TensorPtr mme3InB = createTensor();
    TensorPtr mme3Out = createTensor();

    NodePtr mme1 = addGemm({nop1Out, mme3Out}, {mme1Out}, 1);
    NodePtr mme2 = addGemm({nop1Out, nop2Out}, {mme2Out}, 2);
    NodePtr mme3 = addGemm({nop2Out, mme3InB}, {mme3Out}, 2);

    // When
    BundlePlane bp(m_graph, true);

    // Expect to identify 2 paths from nop1 to mme1 in the bundle plane graph
    EXPECT_FALSE(bp.validateCandidate(nop1, nop1Out, {mme1}));
}

TEST_F(BundlePlaneValidationTest, bp_remove_bundle)
{
    // Given graph:
    // nop0 --> [t0] --> nop1 --> [t2]
    //          [t1] -----^
    TensorPtr t0 = createTensor();
    TensorPtr t1 = createTensor();
    TensorPtr t2 = createTensor();

    NodePtr nop0 = addNop({}, {t0});
    NodePtr nop1 = addNop({t0, t1}, {t2});

    BundlePlane bp {m_graph};

    // When bundling nop0 and then nop1
    bp.addNodeToBundle(nop0);
    bp.addNodeToBundle(nop1);

    // verify after bundle removal that everything goes back to how it was before.
    bp.removeBundle(nop0);
    const NodePtr& bpNode0 = bp.getBundlePlaneRepresentation(nop0);
    const NodePtr& bpNode1 = bp.getBundlePlaneRepresentation(nop1);
    EXPECT_NE(bpNode0, bpNode1);
    EXPECT_EQ(bpNode0->getNumInputs(), 0);
    EXPECT_EQ(bpNode1->getNumInputs(), 2);
    EXPECT_EQ(bpNode0->getOutput(0), bp.getBundlePlaneRepresentation(t0));
    EXPECT_EQ(bpNode1->getOutput(0), bp.getBundlePlaneRepresentation(t2));
}

TEST_F(BundlePlaneValidationTest, bp_multiple_node_create_bundle)
{
    // Given graph:
    // nop0 --> [t0] --> nop1 --> [t2] -> nop2 -> [t3] -> nop4 -> [t5] -> nop5 -> [t6]
    //          [t1] -----^         |---> nop3 -> [t4] ----|
    TensorPtr t0 = createTensor();
    TensorPtr t1 = createTensor();
    TensorPtr t2 = createTensor();
    TensorPtr t3 = createTensor();
    TensorPtr t4 = createTensor();
    TensorPtr t5 = createTensor();
    TensorPtr t6 = createTensor();

    NodePtr nop0 = addNop({}, {t0});
    NodePtr nop1 = addNop({t0, t1}, {t2});
    NodePtr nop2 = addNop({t2}, {t3});
    NodePtr nop3 = addNop({t2}, {t4});
    NodePtr nop4 = addNop({t3, t4}, {t5});
    NodePtr nop5 = addNop({t5}, {t6});

    BundlePlane bp {m_graph};

    // verify create bundle from multiple nodes works ok
    bp.createBundleFromNodes({nop1, nop2, nop3, nop4}, BundleInfo(0, BundleType::COMPLEX_GUID));

    const NodePtr& bpNode0 = bp.getBundlePlaneRepresentation(nop0);
    const NodePtr& bpNode1 = bp.getBundlePlaneRepresentation(nop1);
    const NodePtr& bpNode5 = bp.getBundlePlaneRepresentation(nop5);

    EXPECT_EQ(bpNode1, bp.getBundlePlaneRepresentation(nop2));
    EXPECT_EQ(bpNode1, bp.getBundlePlaneRepresentation(nop3));
    EXPECT_EQ(bpNode1, bp.getBundlePlaneRepresentation(nop4));
    EXPECT_EQ(bpNode1->getNumInputs(), 2);
    EXPECT_EQ(bpNode1->getNumOutputs(), 1);
    EXPECT_EQ(bpNode1->getOutput(0), bp.getBundlePlaneRepresentation(t5));
    EXPECT_NE(bpNode0, bpNode1);
    EXPECT_NE(bpNode5, bpNode1);

    // after removal of bundle bp graph returns to it's previous state.
    bp.removeBundle(nop4);
    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 6);
    EXPECT_EQ(bp.getBundlePlaneGraph()->getTensors().size(), 7);
}

TEST_F(BundlePlaneValidationTest, bp_unbundle_node_sanity)
{
    // Given graph:
    // nop0 --> [t0] --> nop1 --> [t1] --> nop2 --> [t2] --> nop3 --> [t3] --> nop4 --> [t4]
    const TensorVector testTensors {createTensor(), createTensor(), createTensor(), createTensor(), createTensor()};

    NodeVector  testNodes {addNop({}, {testTensors.at(0)}),
                          addNop({testTensors.at(0)}, {testTensors.at(1)}),
                          addNop({testTensors.at(1)}, {testTensors.at(2)}),
                          addNop({testTensors.at(2)}, {testTensors.at(3)}),
                          addNop({testTensors.at(3)}, {testTensors.at(4)})};
    const auto  bi(BundleInfo(0, BundleType::UNDEFINED));
    BundlePlane bp {m_graph};
    std::for_each(testNodes.begin(), testNodes.end(), [this, &bi](NodePtr& n) {
        setBundleInfo(n, bi.bundleIndex, bi.bundleType);
    });
    bp.createBundleFromNodes(testNodes, BundleInfo(0, BundleType::UNDEFINED));
    const NodePtr& bpNode0 = bp.getBundlePlaneRepresentation(testNodes.at(0));
    for (const auto& n : {testNodes.at(1), testNodes.at(2), testNodes.at(3), testNodes.at(4)})
    {
        EXPECT_EQ(bpNode0, bp.getBundlePlaneRepresentation(n));
    }

    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 1);

    // valid unbundling - unbundle the last bundled node
    bp.unbundleNode(testNodes.at(4));

    // at this point bundle 0 contains nop0,nop1,nop2,nop3 and the bp graph is as follows:
    // (bundle0)->[t3]->(nop4)->[t4]
    EXPECT_TRUE(m_graph.isAcyclicGraph());
    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 2);

    // valid unbundling - unbundle the last bundled node
    bp.unbundleNode(testNodes.at(3));

    // at this point bundle 0 contains nop0,nop1,nop2 and the bp graph is as follows:
    // bundle0 --> [t2] --> nop3 --> [t3] --> nop4 --> [t4]
    EXPECT_TRUE(m_graph.isAcyclicGraph());
    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 3);
}

TEST_F(BundlePlaneValidationTest, bp_unbundle_node_agnostic_to_bundling_order)
{
    // Given graph:
    // [t0] --> nop0 --> [t1] --> nop1 --> [t3] --> nop2 --> [t4] --> nop3 --> [t5] --> nop4 -> [t6] --> nop5 --> [t7]
    //                   [t2] -----^

    const TensorVector testTensors {createTensor(),
                                    createTensor(),
                                    createTensor(),
                                    createTensor(),
                                    createTensor(),
                                    createTensor(),
                                    createTensor(),
                                    createTensor()};

    NodeVector  testNodes {addNop({testTensors.at(0)}, {testTensors.at(1)}),
                          addNop({testTensors.at(1), testTensors.at(2)}, {testTensors.at(3)}),
                          addNop({testTensors.at(3)}, {testTensors.at(4)}),
                          addNop({testTensors.at(4)}, {testTensors.at(5)}),
                          addNop({testTensors.at(5)}, {testTensors.at(6)}),
                          addNop({testTensors.at(6)}, {testTensors.at(7)})};
    const auto  bi(BundleInfo(0, BundleType::UNDEFINED));
    BundlePlane bp {m_graph};
    std::for_each(testNodes.begin(), testNodes.end(), [this, &bi](NodePtr& n) {
        setBundleInfo(n, bi.bundleIndex, bi.bundleType);
    });

    // bundling order:
    // nop1, nop2, nop3, nop0, nop4, nop5
    const NodeVector bundlingOrder {testNodes.at(1),
                                    testNodes.at(2),
                                    testNodes.at(3),
                                    testNodes.at(0),
                                    testNodes.at(4),
                                    testNodes.at(5)};
    for (const auto& n : bundlingOrder)
    {
        bp.addNodeToBundle(n, bi);
    }

    // all nodes expected to be bundled
    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 1);

    // valid unbundling - unbundle the last bundled node
    bp.unbundleNode(testNodes.back());
    EXPECT_TRUE(m_graph.isAcyclicGraph());
    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 2);

    // bp graph: [t0] --> bundle0 --> [t6] -> nop5 --> [t7]
    //           [t2] ----^

    // valid unbundling - unbundle peripheral node nop0
    bp.unbundleNode(testNodes.front());
    EXPECT_TRUE(m_graph.isAcyclicGraph());
    EXPECT_EQ(bp.getBundlePlaneGraph()->getNumNodes(), 3);

    // bp graph: [t0] --> nop0 --> bundle0 --> [t6] -> nop5 --> [t7]
    //                       [t2] ----^
}
