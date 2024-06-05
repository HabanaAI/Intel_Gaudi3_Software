#include "gaudi2_graph.h"
#include "habana_graph.h"
#include "sram_management/pipeline_management/common_tile_size_calculator.h"
#include "graph_optimizer_test.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"

using TestParamTuple = std::tuple<unsigned,   // dim0 granularity
                                  unsigned,   // dim1 granularity
                                  bool,       // transpose
                                  unsigned>;  // chain length

// Test params tuple extractor
class CommonTileSizeParams
{
public:
    static const unsigned          NUM_DIMS = 2;
    std::array<unsigned, NUM_DIMS> granularityPerDim;
    bool                           transpose;
    unsigned                       chainLength;

    CommonTileSizeParams() = delete;

    CommonTileSizeParams(const TestParamTuple& paramsTuple)
    : granularityPerDim {std::get<0>(paramsTuple), std::get<1>(paramsTuple)},
      transpose(std::get<2>(paramsTuple)),
      chainLength(std::get<3>(paramsTuple))
    {
    }
};
// c multiple trees graphs
// create nodes with parametrized AP
// simple test - kernel requests same ISME from each tensor and validate LCM
// complex test - kernel with diff ISMEs per tensor, but only validate the common tile is valid - is a mult of the
// granularity in all directions
class CommonTileSizeTest
: public GraphOptimizerTest
, public testing::WithParamInterface<TestParamTuple>
{
protected:
    CommonTileSizeParams m_params;
    Gaudi2Graph          m_graph;
    NodeSet              m_nodes;
    TensorSet            m_tensors;
    TileSizePerTensor    m_tensorsTiles;
    TileSizePerNode      m_nodeTiles;

    CommonTileSizeTest() : m_params(GetParam()) {}

    void testMmeWithTpcChain()
    {
        createMmeWithTpcProducersChainGraph();
        calculateCommonTileSizes();
        validateLegalTileSizes();
        validateSingleNodeGraphGranularity();
    }

    NodePtr createTPCNode(unsigned nodeIndex, const TensorPtr& input)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        unsigned                        nodeFactor = nodeIndex % 4;
        for (auto dim = 0; dim < CommonTileSizeParams::NUM_DIMS; dim++)
        {
            unsigned dimFactor   = (dim % 2) + 2;
            unsigned granularity = m_params.granularityPerDim[dim] + dimFactor * nodeFactor;
            TSize    size        = input ? input->getSizeInElements(dim) : 1000 * dimFactor;
            nodeParams.dims.emplace_back(size, granularity);
        }
        // The common tile size algorithm doesn't really care about the actual tensors sizes, so picking random numbers
        nodeParams.transpose = (nodeIndex % 2 == 0) ? m_params.transpose : !m_params.transpose;

        LOG_TRACE(GO_TEST,
                  "Node {}: granularity {}x{}, input size {}x{}",
                  nodeIndex,
                  nodeParams.dims[0].granularity,
                  nodeParams.dims[1].granularity,
                  nodeParams.dims[0].size,
                  nodeParams.dims[1].size);

        return TPCCustomIndexSpaceNode::create(nodeParams, input);
    }

    NodeVector createTpcChain()
    {
        // Create nodes and connect them to a single chain
        // The replaced tensors don't have the same size, but we don't care about the sizes for this test
        NodeVector chain;
        NodePtr prevNode;
        for (unsigned nodeIndex = 0; nodeIndex < m_params.chainLength; nodeIndex++)
        {
            TensorPtr input   = prevNode ? prevNode->getOutput(0) : nullptr;
            auto      tpcNode = createTPCNode(nodeIndex, input);
            chain.push_back(tpcNode);
            prevNode = tpcNode;
        }
        return chain;
    }

    NodePtr createGemm(const TensorPtr& tpcChainOut)
    {
        synGEMMParams gemmParams {m_params.transpose, false};
        TSize         aSizes[]      = {128, 256};
        TensorPtr     a             = tpcChainOut ? tpcChainOut : TensorPtr(new Tensor(2, aSizes, syn_type_float));
        TSize         commonDimSize = m_params.transpose ? a->getSizeInElements(1) : a->getSizeInElements(0);
        TSize         outHeight     = m_params.transpose ? a->getSizeInElements(0) : a->getSizeInElements(1);
        TSize         bSizes[]      = {256, commonDimSize};
        TSize         outSizes[]    = {256, outHeight};
        TensorPtr     b             = TensorPtr(new Tensor(2, bSizes, syn_type_float));
        TensorPtr     gemmOut       = TensorPtr(new Tensor(2, outSizes, syn_type_float));
        return NodeFactory::createNode({a, b}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    }

    void createMmeWithTpcProducersChainGraph()
    {
        NodeVector tpcChain = createTpcChain();
        for (const auto& n : tpcChain)
        {
            ASSERT_TRUE(GraphEditor::addNode(m_graph, n));
        }
        // add gemm at the end of the chain
        TensorPtr lastOutput = tpcChain.empty() ? nullptr : tpcChain.back()->getOutput(0);
        NodePtr   gemm       = createGemm(lastOutput);
        ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm));
    }

    void calculateCommonTileSizes()
    {
        NodeSet   nodes;
        TensorSet tensors;

        for (const auto& n : m_graph.getExeSortedNodes())
        {
            nodes.insert(n);
            // Add tensors to the set only if there is TPC producers chain. Otherwise test the single node scenario
            if (m_params.chainLength > 0)
            {
                tensors.insert(n->getInput(0));
                if (HabanaGraph::runsOnMME(n))
                {
                    // For the gemm insert both its input with producers chain and its output
                    tensors.insert(n->getInput(0));
                    tensors.insert(n->getOutput(0));
                }
            }
        }

        // Calculate the minimal tile sizes
        std::tie(m_tensorsTiles, m_nodeTiles) =
            CommonTileSizeCalculator::getMinCommonTilesSizes(nodes, tensors, m_graph);
        m_nodes   = nodes;
        m_tensors = tensors;
    }

    // Validate the tiles are a multiplication of the tensors granularity for all tensor dims
    void validateLegalTileSizes()
    {
        for (auto& t : m_tensors)
        {
            // Validate the calc set the tensor granularity
            ASSERT_NE(m_tensorsTiles.find(t), m_tensorsTiles.end());
            // Validate the chosen granularity is aligned with all connected nodes access pattern
            for (auto& n : CommonTileSizeCalculator::getConnectedNodesInGroup(t, m_nodes, m_graph))
            {
                auto tensorTile        = m_tensorsTiles[t];
                auto tensorGranularity = n->getNodeAccessPattern()->getTensorGranularity(t).geometry;
                for (auto dim = 0; dim < t->getDim(); dim++)
                {
                    LOG_TRACE(GO_TEST,
                              "Tensor {}[{}] - for {}: granularity {} tileSize {}",
                              t->getName(),
                              dim,
                              n->getNodeName(),
                              tensorGranularity[dim],
                              tensorTile[dim]);
                    ASSERT_EQ(tensorTile[dim] % tensorGranularity[dim], 0);
                }
            }
        }
    }

    // validate single node graph granularity is all 1s
    void validateSingleNodeGraphGranularity()
    {
        if (m_params.chainLength == 0)
        {
            ASSERT_EQ(m_nodes.size(), 1);
            const NodePtr& n = *m_nodes.begin();
            ASSERT_EQ(m_nodeTiles[n], NodeTile::Geometry(m_nodeTiles[n].size(), 1));
        }
    }
};

TEST_P(CommonTileSizeTest, common_tile_size_test)
{
    testMmeWithTpcChain();
    // TODO SW-75959 - add missing tests: (details in jira)
    // Validate tile size is minimal
    // Validate nodes ISR is valid - map from tensor to ISR and back
    // test trees in addition to chain
}

INSTANTIATE_TEST_SUITE_P(min_common_tile_size,
                         CommonTileSizeTest,
                         testing::Combine(testing::Values(2, 3, 5, 8, 9, 11, 15),   // dim0 granularity
                                          testing::Values(2, 5, 7, 8, 11, 13, 16),  // dim1 granularity
                                          testing::Values(false, true),             // transpose
                                          testing::Range(0U, 9U)                    // chain length
                                          ));
