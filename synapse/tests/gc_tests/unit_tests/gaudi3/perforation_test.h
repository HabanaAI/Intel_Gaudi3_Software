#pragma once

#include "bundle_view.h"
#include "compilation_hal_reader.h"
#include "graph_optimizer_test.h"
#include "gaudi3_graph.h"
#include "node_factory.h"
#include "strategy.h"
#include "tpc_slicing_test_infra.h"
#include "common_tile_size_calculator.h"
#include "slicer/bundle_views_collector.h"
#include "types.h"

using namespace gc::layered_brain;

class PerforationTest : public GraphOptimizerTest
{
protected:
    TensorPtr createTensor(const std::vector<TSize>& shape)
    {
        TensorPtr tensor = std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_bf16);
        m_bundleTensors.insert(tensor);
        return tensor;
    }

    void addNodeToGraph(const NodePtr& node)
    {
        ASSERT_TRUE(node);
        ASSERT_TRUE(GraphEditor::addNode(m_graph, node));
        m_bundleNodes.push_back(node);
    }

    void createSingleMMEGraph()
    {
        // [TPC] -> [GEMM] -> [TPC]

        synGEMMParams gemmParams(false, false);
        m_gemm = NodeFactory::createNode({createTensor({DIM_SIZE, DIM_SIZE}), createTensor({DIM_SIZE, DIM_SIZE})},
                                         {createTensor({DIM_SIZE, DIM_SIZE})},
                                         &gemmParams,
                                         NodeFactory::gemmNodeTypeName,
                                         "GEMM");
        addNodeToGraph(m_gemm);
        NodePtr tpcProducer =
            TPCCustomIndexSpaceNode::createSliceableNode(createTensor({DIM_SIZE, DIM_SIZE}), m_gemm->getInput(0));
        addNodeToGraph(tpcProducer);
        NodePtr tpcConsumer =
            TPCCustomIndexSpaceNode::createSliceableNode(m_gemm->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(tpcConsumer);
    }

    void createUnsliceableGraph()
    {
        // [TPC] -> [TPC] -> [TPC]

        TPCCustomIndexSpaceNode::Params nodeParams {};
        // All dims are all-required (size == granularity)
        nodeParams.dims.emplace_back(DIM_SIZE, DIM_SIZE);
        nodeParams.dims.emplace_back(DIM_SIZE, DIM_SIZE);
        nodeParams.transpose = false;

        NodePtr node1 = TPCCustomIndexSpaceNode::create(nodeParams,
                                                        createTensor({DIM_SIZE, DIM_SIZE}),
                                                        createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(node1);
        NodePtr node2 =
            TPCCustomIndexSpaceNode::create(nodeParams, node1->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(node2);
        NodePtr node3 =
            TPCCustomIndexSpaceNode::create(nodeParams, node2->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(node3);
    }

    void createMultiMMEGraph()
    {
        // [TPC] -> [GEMM] -> [TPC] -> [GEMM]

        synGEMMParams gemmParams(false, false);
        m_gemm = NodeFactory::createNode({createTensor({DIM_SIZE, DIM_SIZE}), createTensor({DIM_SIZE, DIM_SIZE})},
                                         {createTensor({DIM_SIZE, DIM_SIZE})},
                                         &gemmParams,
                                         NodeFactory::gemmNodeTypeName,
                                         "GEMM");
        addNodeToGraph(m_gemm);
        NodePtr tpcProducer =
            TPCCustomIndexSpaceNode::createSliceableNode(createTensor({DIM_SIZE, DIM_SIZE}), m_gemm->getInput(0));
        addNodeToGraph(tpcProducer);
        NodePtr tpcConsumer =
            TPCCustomIndexSpaceNode::createSliceableNode(m_gemm->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(tpcConsumer);
        auto gemm2 = NodeFactory::createNode({tpcConsumer->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE})},
                                             {createTensor({DIM_SIZE, DIM_SIZE})},
                                             &gemmParams,
                                             NodeFactory::gemmNodeTypeName,
                                             "GEMM2");
        addNodeToGraph(gemm2);
    }

    void createSharedInputMultiMMEGraph()
    {
        // Create a graph with 3 MMEs sharing input A.
        synGEMMParams gemmParams(false, false);
        m_gemm = NodeFactory::createNode({createTensor({DIM_SIZE, DIM_SIZE}), createTensor({DIM_SIZE, DIM_SIZE})},
                                         {createTensor({DIM_SIZE, DIM_SIZE})},
                                         &gemmParams,
                                         NodeFactory::gemmNodeTypeName,
                                         "GEMM1");
        addNodeToGraph(m_gemm);

        auto gemm2 = NodeFactory::createNode({m_gemm->getInput(0), createTensor({DIM_SIZE, DIM_SIZE})},
                                             {createTensor({DIM_SIZE, DIM_SIZE})},
                                             &gemmParams,
                                             NodeFactory::gemmNodeTypeName,
                                             "GEMM2");
        addNodeToGraph(gemm2);

        auto gemm3 = NodeFactory::createNode({m_gemm->getInput(0), createTensor({DIM_SIZE, DIM_SIZE})},
                                             {createTensor({DIM_SIZE, DIM_SIZE})},
                                             &gemmParams,
                                             NodeFactory::gemmNodeTypeName,
                                             "GEMM3");
        addNodeToGraph(gemm3);

        // Add consumer to the first GEMM
        NodePtr gemm1Consumer =
            TPCCustomIndexSpaceNode::createSliceableNode(m_gemm->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(gemm1Consumer);

        // Add producer to input B and consumer to the second GEMM
        NodePtr gemm2Producer =
            TPCCustomIndexSpaceNode::createSliceableNode(createTensor({DIM_SIZE, DIM_SIZE}), gemm2->getInput(1));
        addNodeToGraph(gemm2Producer);
        NodePtr gemm2Consumer =
            TPCCustomIndexSpaceNode::createSliceableNode(gemm2->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(gemm2Consumer);

        // The third GEMM doesn't have producers/consumer
    }

    void createTPCOnlyGraph()
    {
        // [TPC] -> [TPC] -> [TPC]

        NodePtr tpc1 = TPCCustomIndexSpaceNode::createSliceableNode(createTensor({DIM_SIZE, DIM_SIZE}),
                                                                    createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(tpc1);
        NodePtr tpc2 =
            TPCCustomIndexSpaceNode::createSliceableNode(tpc1->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(tpc2);
        NodePtr tpc3 =
            TPCCustomIndexSpaceNode::createSliceableNode(tpc2->getOutput(0), createTensor({DIM_SIZE, DIM_SIZE}));
        addNodeToGraph(tpc3);
    }

    void createBundleViews()
    {
        NodeSet bundleNodesSet(m_bundleNodes.begin(), m_bundleNodes.end());
        const auto& [granularityPerTensor, granularityPerNode] =
            CommonTileSizeCalculator::getMinCommonTilesSizes(bundleNodesSet, m_bundleTensors, m_graph);

        BundleViewsCollector bundleViewsCollector(m_bundleNodes);
        m_bundleViews = bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
    }

    StrategyPtr createDefaultStrategy()
    {
        auto mmeSolution = std::make_shared<MmeSolution>();
        for (const auto& node : m_bundleNodes)
        {
            if (HabanaGraph::runsOnMME(node))
            {
                mmeSolution->QORs[node] = std::make_shared<SolutionParams>();
            }
        }
        StrategyPtr strategy = std::make_shared<Strategy>(mmeSolution);
        for (BundleViewId bvd = 0; bvd < m_bundleViews->getNumOfBundleViews(); bvd++)
        {
            strategy->setBVDMultiplier(bvd, BVDMultiplier());
        }
        return strategy;
    }

    static constexpr unsigned  NUM_DCORES = 4;
    static constexpr TSize     DIM_SIZE   = 4096;
    Gaudi3Graph                m_graph;
    CompilationHalReaderSetter m_halSetter {&m_graph};
    NodeVector                 m_bundleNodes;
    NodePtr                    m_gemm;
    TensorSet                  m_bundleTensors;
    BundleViewContainerPtr     m_bundleViews;
};