#include "graph_optimizer_test.h"
#include "gaudi3_graph.h"
#include "layered_brain_test_common.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "bundler/layered_brain_bundle.h"
#include "synapse_common_types.h"
#include <vector>
#include "node_factory.h"

using namespace gc::layered_brain;

class LayeredBrainBundleTest : public LayeredBrainCommonTest<Gaudi3Graph>
{
public:
    LayeredBrainBundleTest() : m_tensorSizes({31, 13, 23, 37}), m_guid("relu_fwd_bf16")
    {
        CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
    }

protected:
    unsigned getNextNodeId() { return m_nextNodeId++; }

    // adds a node to the end of the chain [0]->[1]->[2]->...->[n-1]
    // returns pointer to the new node
    NodePtr addNode()
    {
        auto in = m_nodes.empty() ? createTensor(m_tensorSizes, m_dtype) : m_nodes.back()->getOutput(0);
        HB_ASSERT_PTR(in);
        auto       out    = createTensor(m_tensorSizes, m_dtype);
        HB_ASSERT_PTR(out);
        const auto nextId = getNextNodeId();
        NodePtr    relu =
            NodeFactory::createGenericTPCNode({in}, {out}, nullptr, m_guid.c_str(), fmt::format("relu_{}", nextId));
        HB_ASSERT_PTR(relu);
        const auto success = GraphEditor::addNode(m_graph, relu);
        HB_ASSERT(success, "Expecting add node to graph to succeed");
        m_nodes.push_back(relu);
        return relu;
    }

    std::vector<NodePtr> addNodes(unsigned n)
    {
        std::vector<NodePtr> newNodes {};
        newNodes.reserve(n);
        while (n--)
        {
            newNodes.push_back(addNode());
        }
        return newNodes;
    }

    const std::vector<TSize> m_tensorSizes;
    const synDataType        m_dtype = syn_type_bf16;
    const std::string        m_guid;
    unsigned                 m_nextNodeId = 0;
    std::vector<NodePtr>     m_nodes;
    Gaudi3Graph              m_graph;
};

TEST_F(LayeredBrainBundleTest, sanity)
{
    BPGraphContext bpgCtx(m_graph);
    auto           bundle = Bundle::create(m_graph);

    // obviously no candidates on init
    EXPECT_FALSE(bundle->hasCandidates());

    // add some candidates, still a candidate bundle
    {
        auto nodes = addNodes(4);
        bundle->addCandidate(nodes);
        EXPECT_TRUE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 4);  // 4 candidates
    }

    // after accepting candidates should have no candidates
    bundle->acceptCandidates();
    EXPECT_FALSE(bundle->hasCandidates());

    // add more candidates to valid bundle w/o candidates
    {
        auto nodes = addNodes(3);
        bundle->addCandidate(nodes);
        EXPECT_TRUE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 4 + 3);  // 4 accepted, 3 candidates

        // reject candidates
        bundle->rejectCandidates();
        EXPECT_FALSE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 4);  // 4 accepted
    }
}

TEST_F(LayeredBrainBundleTest, accept_and_reject_without_candidates)
{
    BPGraphContext bpgCtx(m_graph);
    auto           bundle = Bundle::create(m_graph);
    // add 3 candidates and commit
    {
        auto nodes = addNodes(3);
        bundle->addCandidate(nodes);
        bundle->acceptCandidates();
        EXPECT_FALSE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 3);  // 3 accepted nodes
    }

    // expecting no change to bundle
    bundle->acceptCandidates();
    EXPECT_FALSE(bundle->hasCandidates());
    EXPECT_EQ(bundle->getNodes().size(), 3);  // 3 accepted nodes

    // expecting no change to bundle
    bundle->rejectCandidates();
    EXPECT_FALSE(bundle->hasCandidates());
    EXPECT_EQ(bundle->getNodes().size(), 3);  // 3 accepted nodes
}

TEST_F(LayeredBrainBundleTest, candidate_bundle_cleanup_on_destruction)
{
    BPGraphContext bpgCtx(m_graph);
    auto           nodes = addNodes(5);

    // add 5 candidates without committing
    {
        auto bundle = Bundle::create(m_graph);
        bundle->addCandidate(nodes);
        EXPECT_TRUE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 5);  // 5 candidates
        const auto bundleCandidateIdx = bundle->index();
        for (const auto& n : nodes)
        {
            const auto& bundleInfo = n->getNodeAnnotation().bundleInfo;
            EXPECT_TRUE(bundleInfo.is_set());
            EXPECT_EQ(bundleInfo->bundleIndex, bundleCandidateIdx);
        }
    }
    // on bundle destruction all candidates should be rejected
    for (const auto& n : nodes)
    {
        const auto& bundleInfo = n->getNodeAnnotation().bundleInfo;
        EXPECT_FALSE(bundleInfo.is_set());
    }
}

TEST_F(LayeredBrainBundleTest, accepted_nodes_and_candidates_cleanup_on_destruction)
{
    BPGraphContext bpgCtx(m_graph);
    auto           nodes      = addNodes(5);
    auto           candidates = addNodes(3);
    {
        // add 5 candidates and commit
        auto bundle = Bundle::create(m_graph);
        bundle->addCandidate(nodes);
        EXPECT_TRUE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 5);  // 5 candidates
        bundle->acceptCandidates();
        EXPECT_FALSE(bundle->hasCandidates());
        EXPECT_EQ(bundle->getNodes().size(), 5);  // 5 bundle nodes

        // add 3 candidates without committing
        bundle->addCandidate(candidates);
        EXPECT_EQ(bundle->getNodes().size(), 5 + 3);  // 5 bundle nodes + 3 candidates
        EXPECT_TRUE(bundle->hasCandidates());
    }

    // on bundle destruction all candidates should be rejected
    // and accepted nodes should be untouched
    for (const auto& n : nodes)
    {
        const auto& bundleInfo = n->getNodeAnnotation().bundleInfo;
        EXPECT_TRUE(bundleInfo.is_set());
    }

    for (const auto& n : candidates)
    {
        const auto& bundleInfo = n->getNodeAnnotation().bundleInfo;
        EXPECT_FALSE(bundleInfo.is_set());
    }

    const auto& graphNodes    = m_graph.getNodes();
    const auto  nBundledNodes = std::count_if(graphNodes.begin(), graphNodes.end(), [](const auto& n) {
        return n && n->getNodeAnnotation().bundleInfo.is_set();
    });

    EXPECT_EQ(nodes.size(), nBundledNodes);
}

TEST_F(LayeredBrainBundleTest, direct_add_to_bundle)
{
    BPGraphContext bpgCtx(m_graph);
    auto           bundle     = Bundle::create(m_graph);
    auto           nodes      = addNodes(5);
    auto           candidates = addNodes(3);
    // add 5 nodes directly
    bundle->add(nodes);
    EXPECT_EQ(bundle->getNodes().size(), 5);
    EXPECT_FALSE(bundle->hasCandidates());
}

TEST_F(LayeredBrainBundleTest, dismantle_bundle)
{
    BPGraphContext bpgCtx(m_graph);
    auto           bundle = Bundle::create(m_graph);

    // directly add 5 nodes
    auto nodes = addNodes(5);
    bundle->add(nodes);
    EXPECT_EQ(bundle->getNodes().size(), nodes.size());
    EXPECT_FALSE(bundle->hasCandidates());

    // dismantle bundle
    bundle->dismantle();
    EXPECT_EQ(bundle->getNodes().size(), 0);
    EXPECT_FALSE(bundle->hasCandidates());

    // directly add 3 more nodes
    auto moreNodes = addNodes(3);
    bundle->add(moreNodes);
    EXPECT_EQ(bundle->getNodes().size(), moreNodes.size());
    EXPECT_FALSE(bundle->hasCandidates());

    // add 3 more candidates
    auto candidates = addNodes(3);
    bundle->addCandidate(candidates);
    EXPECT_EQ(bundle->getNodes().size(), candidates.size() + moreNodes.size());
    EXPECT_TRUE(bundle->hasCandidates());

    // dismantle bundle
    bundle->dismantle();
    EXPECT_EQ(bundle->getNodes().size(), 0);
    EXPECT_FALSE(bundle->hasCandidates());
}
