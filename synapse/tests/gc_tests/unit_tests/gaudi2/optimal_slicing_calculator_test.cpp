#include "compilation_hal_reader.h"
#include "gaudi2_graph.h"
#include "slicer/optimal_slicing_calculator.h"
#include "slicer/bundle_views_collector.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"
#include "graph_optimizer_test.h"

using namespace gc::layered_brain;

class OptimalSlicingCalculatorTest : public GraphOptimizerTest
{
protected:
    NodePtr addTPCNode(const TensorPtr& input = nullptr, const TensorPtr& output = nullptr)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        for (auto i = 0; i < m_numDims; i++)
        {
            nodeParams.dims.emplace_back(m_dimSize, 1, 0);
        }
        nodeParams.transpose = false;
        NodePtr node         = TPCCustomIndexSpaceNode::create(nodeParams, input, output);
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        return node;
    }

    NodePtr addGEMMNode()
    {
        synGEMMParams      params(false, false);
        std::vector<TSize> sizes(m_numDims, m_dimSize);

        TensorVector inputs;
        inputs.push_back(std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float));
        inputs.push_back(std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float));
        TensorVector outputs;
        outputs.push_back(std::make_shared<Tensor>(sizes.size(), sizes.data(), syn_type_float));
        NodePtr node = NodeFactory::createNode(inputs,
                                               outputs,
                                               &params,
                                               NodeFactory::gemmNodeTypeName,
                                               "GEMM" + std::to_string(m_nodeId++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        return node;
    }

    std::pair<TileSizePerTensor, TileSizePerNode> getMinCommonTilesSizes(const NodeVector& bundleNodes) const
    {
        TensorSet bundleTensorsSet;
        NodeSet   bundleNodesSet(bundleNodes.begin(), bundleNodes.end());
        for (const auto& n : bundleNodes)
        {
            for (const auto& nodeOperand : n->getOperands())
            {
                if (nodeOperand)
                {
                    bundleTensorsSet.emplace(nodeOperand);
                }
            }
        }
        return CommonTileSizeCalculator::getMinCommonTilesSizes(bundleNodesSet, bundleTensorsSet, m_graph);
    }

    void test()
    {
        NodeVector bundleNodes;
        NodePtr    gemm = addGEMMNode();
        bundleNodes.emplace_back(gemm);
        bundleNodes.emplace_back(addTPCNode(nullptr, gemm->getInput(0)));  // operand A producer
        bundleNodes.emplace_back(addTPCNode(nullptr, gemm->getInput(1)));  // operand B producer
        bundleNodes.emplace_back(addTPCNode(gemm->getOutput(0)));          // output consumer

        const auto& [granularityPerTensor, granularityPerNode] = getMinCommonTilesSizes(bundleNodes);
        BundleViewsCollector          bundleViewsCollector(bundleNodes);
        const BundleViewContainerPtr& bundleViews =
            bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);

        ASSERT_EQ(bundleViews->getNumOfBundleViews(), 3);

        auto mmeSolution        = std::make_shared<MmeSolution>();
        mmeSolution->QORs[gemm] = std::make_shared<SolutionParams>();
        StrategyPtr strategy    = std::make_shared<Strategy>(mmeSolution);

        // Granularity multiplier for BVD 1 is omitted and expected to be filled by the optimal slicing calculator.
        strategy->setBVDMultiplier(0, BVDMultiplier(1UL));
        strategy->setBVDMultiplier(2, BVDMultiplier(2UL));

        OptimalSlicingCalculator optimalSlicingCalculator(m_graph, SlicingPolicy::ENOUGH_REUSE, MAX_TILE_SIZE);
        const auto& optimalStrategy = optimalSlicingCalculator.getOptimalStrategy(bundleViews, {strategy}, bundleNodes);
        validateStrategy(optimalStrategy, bundleViews);
    }

    void validateStrategy(const StrategyPtr& optimalStrategy, const BundleViewContainerPtr& bundleViews) const
    {
        unsigned numOfSlicedBVDs = 0;
        for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
        {
            if (optimalStrategy->getBVDMultiplier(bvdId).isSliced())
            {
                numOfSlicedBVDs++;
            }
        }
        ASSERT_LE(numOfSlicedBVDs, 1);
    }

    Gaudi2Graph                m_graph;
    CompilationHalReaderSetter m_halSetter {&m_graph};
    const unsigned             m_numDims     = 2;
    const unsigned             m_dimSize     = 128;
    unsigned                   m_nodeId      = 0;
    static constexpr uint64_t  MAX_TILE_SIZE = 4 * 1024 * 1024;
};

TEST_F(OptimalSlicingCalculatorTest, single_strategy)
{
    test();
}