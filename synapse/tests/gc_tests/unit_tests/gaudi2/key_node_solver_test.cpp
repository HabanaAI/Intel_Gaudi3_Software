#include "compilation_hal_reader.h"
#include "gaudi2_graph.h"
#include "slicer/bundle_views_collector.h"
#include "slicer/mme_key_node_solver.h"
#include "slicer/legacy_mme_key_node_solver.h"
#include "strategy.h"
#include "graph_optimizer_test.h"
#include "scoped_configuration_change.h"
#include "slicing_brain.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"

using namespace gc::layered_brain;

class MMEKeyNodeSolverTest
: public GraphOptimizerTest
, public ::testing::WithParamInterface<std::tuple<bool,      // transpose A
                                                  bool,      // transpose B
                                                  TSize,     // height A
                                                  TSize,     // common-dim size
                                                  TSize,     // width B
                                                  unsigned,  // num batch dims
                                                  TSize,     // batch dim size
                                                  TSize,     // batch dim granularity
                                                  bool,      // true - use MME brain, false - use stub (PM node solver)
                                                  unsigned,  // num nodes
                                                  bool       // shared input
                                                  >>
{
protected:
    MMEKeyNodeSolverTest() : m_halSetter(&m_graph)
    {
        std::tie(m_transposeA,
                 m_transposeB,
                 m_heightA,
                 m_commonDim,
                 m_widthB,
                 m_numBatchDims,
                 m_batchDimSize,
                 m_batchDimGranularity,
                 m_useMMEBrain,
                 m_numNodes,
                 m_sharedInput) = GetParam();
    }

    std::unique_ptr<KeyNodeSolver> getKeyNodeSolver(const NodePtr& keyNode) const
    {
        if (m_useMMEBrain)
        {
            return std::make_unique<MMEKeyNodeSolver>(m_graph, keyNode, MAX_TILE_SIZE, MIN_COMMON_DIM_FOR_PARTIALS);
        }
        return std::make_unique<LegacyMMEKeyNodeSolver>(m_graph, keyNode, MAX_TILE_SIZE, MIN_COMMON_DIM_FOR_PARTIALS);
    }

    void test()
    {
        ScopedConfigurationChange scc("HARD_TRIM_MME_BRAIN_STRATEGIES", "0");

        SlicingBrain dummyBrain(m_graph);  // Required to intialize SlicingBrain::knobs for legacy solver
        createNodes();
        createBundleViews();

        StrategyContainer strategies;
        for (const auto& node : m_nodes)
        {
            strategies = getKeyNodeSolver(node)->getSlicingStrategies(m_bundleViews, strategies);
        }
        ASSERT_EQ(strategies.strategies.size(), strategies.mmeSolutions.size());
        ASSERT_EQ(strategies.nodes.size(), m_nodes.size());
        for (const auto& node : m_nodes)
        {
            ASSERT_TRUE(std::find(strategies.nodes.begin(), strategies.nodes.end(), node) != strategies.nodes.end());
        }
        if (m_useMMEBrain)
        {
            ASSERT_GE(strategies.strategies.size(), 1);
            for (const auto& strategy : strategies.strategies)
            {
                validateStrategy(strategy);
            }
        }
        else
        {
            ASSERT_EQ(m_nodes.size(), 1);  // The legacy solver supports a single node only
            ASSERT_EQ(strategies.strategies.size(), 1);
            const TStride minSliceSizeBytes = getMinSliceSizeBytes(m_nodes.front());
            if (minSliceSizeBytes <= SlicingBrain::knobs.maxSRAMCapInBytes)
            {
                validateStrategy(strategies.strategies.front());
            }
            else
            {
                // Default strategy should be returned in this case - all BVDs should be unsliced
                for (BundleViewId bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); bvdId++)
                {
                    ASSERT_FALSE(strategies.strategies[0]->getBVDMultiplier(bvdId).isSliced());
                }
            }
        }
    }

private:
    std::pair<TileSizePerTensor, TileSizePerNode> getGranularityConstraints() const
    {
        TileSizePerTensor granularityPerTensor;
        TileSizePerNode   granularityPerNode;

        for (const auto& node : m_nodes)
        {
            const auto& tensors = node->getOperands();

            // Init node granularity to 1
            granularityPerNode[node] = NodeTile::Geometry(node->getNodeAccessPattern()->getNodeResolution().size(), 1);
            // Init tensors granularity to 1
            std::for_each(tensors.begin(), tensors.end(), [&](const TensorPtr& t) {
                granularityPerTensor[t] = TensorTile::Geometry(t->getDim(), 1);
            });

            // Update batch dims granularity
            for (Dim tensorDim = DIM_GEMM_BATCH; tensorDim < node->getOutput(0)->getDim(); tensorDim++)
            {
                for (const auto& tensor : tensors)
                {
                    granularityPerTensor[tensor][tensorDim] = m_batchDimGranularity;
                    const Dim nodeDim = node->getNodeAccessPattern()->getIndexSpaceDim(tensor, tensorDim);
                    granularityPerNode[node][nodeDim] = m_batchDimGranularity;
                }
            }
        }

        return {granularityPerTensor, granularityPerNode};
    }

    void createBundleViews()
    {
        const auto& [granularityPerTensor, granularityPerNode] = getGranularityConstraints();
        BundleViewsCollector bundleViewsCollector(m_nodes);
        m_bundleViews = bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
    }

    TensorPtr createTensor(std::vector<TSize> shape, bool transposed, const std::vector<TSize>& batchDims) const
    {
        if (transposed)
        {
            std::swap(shape[0], shape[1]);
        }
        if (!batchDims.empty())
        {
            shape.insert(shape.end(), batchDims.begin(), batchDims.end());
        }
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_bf16);
    }

    NodePtr addGEMMNode(unsigned nodeIdx)
    {
        std::vector<TSize> batchDims(m_numBatchDims, m_batchDimSize);

        std::string   guid = batchDims.empty() ? NodeFactory::gemmNodeTypeName : NodeFactory::batchGemmNodeTypeName;
        synGEMMParams params(m_transposeA, m_transposeB);

        TensorVector inputs;
        inputs.push_back(createTensor({m_commonDim, m_heightA}, m_transposeA, batchDims));
        inputs.push_back(createTensor({m_widthB, m_commonDim}, m_transposeB, batchDims));
        TensorPtr output = createTensor({m_widthB, m_heightA}, false, batchDims);

        NodePtr node = NodeFactory::createNode(inputs,
                                               {output},
                                               &params,
                                               guid.c_str(),
                                               (batchDims.empty() ? "GEMM" : "BGEMM") + std::to_string(nodeIdx));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        return node;
    }

    void createNodes()
    {
        static constexpr auto MAX_STRIDE =
            std::numeric_limits<std::remove_reference<decltype(MmeCommon::MmeTensorView::strides[0])>::type>::max();

        for (auto i = 0; i < m_numNodes; i++)
        {
            NodePtr mmeNode = addGEMMNode(i);
            for (const TensorPtr& t : mmeNode->getOperands())
            {
                ASSERT_LT(t->getStrideInElements(t->getDim()), MAX_STRIDE);
            }
            m_nodes.push_back(mmeNode);
            if (m_sharedInput && i > 0)
            {
                mmeNode->replaceInput(0, m_nodes.front()->getInput(0));  // All nodes are sharing operand A
            }
        }
    }

    // Temp until real MME brain is ready.
    // Currently MAX_TILE_SIZE is ignored and the NodeSolver inflates the tile given the max SRAM capacity.
    // In order to check if the NodeSolver will return a valid strategy need to check if min slice size fits SRAM.
    // When the NodeSolver will be replaced with real MME brain implementation - we will need to verify that
    // each tile size is below MAX_TILE_SIZE threshold.
    TStride getMinSliceSizeBytes(const NodePtr& mmeNode) const
    {
        unsigned    slicedInputIdx    = 0;
        const auto& slicedInputInSram = mmeNode->getInput(slicedInputIdx);
        const auto& slicedDims        = NodeSolver::getInputSlicingDims(mmeNode, slicedInputIdx);
        HB_ASSERT(!slicedDims.empty(), "Missing slicing dim for node {}", mmeNode->getNodeName());
        auto        slicedDim                = slicedDims.front();
        const TSize slicedDimGranularity     = m_bundleViews->getGranularityForTensorDim(slicedInputInSram, slicedDim);
        const TSize minSlicedDimSizeElements = std::min(
            MmeNodeSolver::getMinSlicedDimSizeInElements(mmeNode, slicedInputInSram, slicedDim, slicedDimGranularity),
            slicedInputInSram->getSizeInElements(slicedDim));
        const TSize minSliceSizeElements = slicedInputInSram->getDenseSizeInElements() /
                                           slicedInputInSram->getSizeInElements(slicedDim) * minSlicedDimSizeElements;
        const TStride minSliceSizeBytes = minSliceSizeElements * slicedInputInSram->getElementSizeInBytes();
        return minSliceSizeBytes;
    }

    void validateStrategy(const StrategyPtr& strategy) const
    {
        ASSERT_TRUE(strategy->getMmeSolution());
        for (const auto& node : m_nodes)
        {
            ASSERT_TRUE(strategy->getMmeSolution()->QORs.find(node) != strategy->getMmeSolution()->QORs.end());
            ASSERT_TRUE(strategy->getMmeSolution()->QORs.at(node));
            if (m_useMMEBrain)
            {
                const auto& ap = node->getNodeAccessPattern();
                ASSERT_TRUE(ap);
                ASSERT_TRUE(strategy->getMmeSolution()->brainSolution.find(node) !=
                            strategy->getMmeSolution()->brainSolution.end());
                const auto& brainSolution = strategy->getMmeSolution()->brainSolution.at(node);
                ASSERT_TRUE(brainSolution);
                for (auto nodeDim = 0; nodeDim < ap->getNodeResolution().size(); nodeDim++)
                {
                    if (m_bundleViews->isNodeDimMappedToBVD(node, nodeDim))
                    {
                        auto bvd = m_bundleViews->getBVDForNodeDim(node, nodeDim);
                        if (strategy->getBVDMultiplier(bvd).isSliced())
                        {
                            ASSERT_EQ(brainSolution->solutionDimMultipliers.at(nodeDim),
                                      strategy->getBVDMultiplier(bvd).getMultiplier());
                        }
                        else
                        {
                            ASSERT_EQ(brainSolution->solutionDimMultipliers.at(nodeDim),
                                      m_bundleViews->getBundleView(bvd).resolution);
                        }
                    }
                }
            }
        }
        // MAX_TILE_SIZE and MIN_COMMON_DIM_FOR_PARTIALS are ignored in current implementation and not validated.
        // Should be validated with real MME brain.
        for (BundleViewId bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); bvdId++)
        {
            const auto& bundleView = m_bundleViews->getBundleView(bvdId);
            // Validate that each BVD has a multiplier in the strategy.
            auto granularityMultiplier = strategy->getBVDMultiplier(bvdId);
            // Validate that the multiplier is not bigger than the max multiplier allowed by the node AP.
            if (granularityMultiplier.isSliced())  // BVD is sliced
            {
                ASSERT_LE(granularityMultiplier.getMultiplier(), bundleView.resolution);
                ASSERT_NE(granularityMultiplier.getMultiplier(), 0);
            }
        }
    }

    static constexpr uint64_t MAX_TILE_SIZE               = 4 * 1024 * 1024;
    static constexpr uint64_t MIN_COMMON_DIM_FOR_PARTIALS = 2 * 1024;

    Gaudi2Graph                m_graph;
    CompilationHalReaderSetter m_halSetter;

    bool     m_transposeA;
    bool     m_transposeB;
    TSize    m_heightA;
    TSize    m_commonDim;
    TSize    m_widthB;
    unsigned m_numBatchDims;
    TSize    m_batchDimSize;
    TSize    m_batchDimGranularity;
    bool     m_useMMEBrain;
    unsigned m_numNodes;
    bool     m_sharedInput;  // Share operand A between the nodes (false - no sharing)

    NodeVector             m_nodes;
    BundleViewContainerPtr m_bundleViews;
};

TEST_P(MMEKeyNodeSolverTest, get_strategies_for_mme_node)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(
    bgemm_legacy_key_node_solver_test,
    MMEKeyNodeSolverTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false, true),  // transpose B
                       ::testing::Values(1023, 128),    // height A
                       ::testing::Values(1023, 128),    // common-dim size
                       ::testing::Values(1023, 128),    // width B
                       ::testing::Values(1, 2, 3),      // num batch dims
                       ::testing::Values(1, 7, 13),     // batch dim size
                       ::testing::Values(1, 3),         // batch dim granularity
                       ::testing::Values(false),        // true - use MME brain, false - use stub (PM node solver)
                       ::testing::Values(1),            // num nodes
                       ::testing::Values(false)));      // shared input

INSTANTIATE_TEST_SUITE_P(
    gemm_legacy_key_node_solver_test,
    MMEKeyNodeSolverTest,
    ::testing::Combine(::testing::Values(false, true),            // transpose A
                       ::testing::Values(false, true),            // transpose B
                       ::testing::Values(517, 1023, 128, 64002),  // height A
                       ::testing::Values(517, 1023, 128, 64002),  // common-dim size
                       ::testing::Values(517, 1023, 128, 64002),  // width B
                       ::testing::Values(0),                      // num batch dims
                       ::testing::Values(1),                      // batch dim size (not relevant for GEMM)
                       ::testing::Values(1),                      // batch dim granularity (not relevant for GEMM)
                       ::testing::Values(false),    // true - use MME brain, false - use stub (PM node solver)
                       ::testing::Values(1),        // num nodes
                       ::testing::Values(false)));  // shared input

INSTANTIATE_TEST_SUITE_P(
    bgemm_key_node_solver_test,
    MMEKeyNodeSolverTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false, true),  // transpose B
                       ::testing::Values(517, 1355),    // height A
                       ::testing::Values(517, 1355),    // common-dim size
                       ::testing::Values(517, 1355),    // width B
                       ::testing::Values(1, 2, 3),      // num batch dims
                       ::testing::Values(1, 7, 13),     // batch dim size
                       ::testing::Values(1, 3),         // batch dim granularity
                       ::testing::Values(true),         // true - use MME brain, false - use stub (PM node solver)
                       ::testing::Values(1),            // num nodes
                       ::testing::Values(false)));      // shared input

INSTANTIATE_TEST_SUITE_P(
    gemm_key_node_solver_test,
    MMEKeyNodeSolverTest,
    ::testing::Combine(::testing::Values(false, true),            // transpose A
                       ::testing::Values(false, true),            // transpose B
                       ::testing::Values(517, 1023, 128, 64002),  // height A
                       ::testing::Values(517, 1023, 128, 64002),  // common-dim size
                       ::testing::Values(517, 1023, 128, 64002),  // width B
                       ::testing::Values(0),                      // num batch dims
                       ::testing::Values(1),                      // batch dim size (not relevant for GEMM)
                       ::testing::Values(1),                      // batch dim granularity (not relevant for GEMM)
                       ::testing::Values(true),     // true - use MME brain, false - use stub (PM node solver)
                       ::testing::Values(1),        // num nodes
                       ::testing::Values(false)));  // shared input

INSTANTIATE_TEST_SUITE_P(
    multi_bgemm_key_node_solver_test,
    MMEKeyNodeSolverTest,
    ::testing::Combine(::testing::Values(false),          // transpose A
                       ::testing::Values(false),          // transpose B
                       ::testing::Values(1355),           // height A
                       ::testing::Values(519),            // common-dim size
                       ::testing::Values(1232),           // width B
                       ::testing::Values(1, 2),           // num batch dims
                       ::testing::Values(7),              // batch dim size
                       ::testing::Values(3),              // batch dim granularity
                       ::testing::Values(true),           // true - use MME brain, false - use stub (PM node solver)
                       ::testing::Values(2, 3),           // num nodes
                       ::testing::Values(false, true)));  // shared input

INSTANTIATE_TEST_SUITE_P(
    multi_gemm_key_node_solver_test,
    MMEKeyNodeSolverTest,
    ::testing::Combine(::testing::Values(false),          // transpose A
                       ::testing::Values(false),          // transpose B
                       ::testing::Values(1093),           // height A
                       ::testing::Values(1223),           // common-dim size
                       ::testing::Values(128),            // width B
                       ::testing::Values(0),              // num batch dims
                       ::testing::Values(1),              // batch dim size (not relevant for GEMM)
                       ::testing::Values(1),              // batch dim granularity (not relevant for GEMM)
                       ::testing::Values(true),           // true - use MME brain, false - use stub (PM node solver)
                       ::testing::Values(2, 3),           // num nodes
                       ::testing::Values(false, true)));  // shared input