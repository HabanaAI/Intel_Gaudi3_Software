#include "habana_norm_slicing_test.h"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "habana_pass.h"
#include "node.h"
#include "node_annotation.h"
#include "node_factory.h"
#include "node_utils.h"
#include "perf_lib_layer_params.h"
#include "types.h"
#include "utils.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <memory>

HabanaNormsSlicingTest::HabanaNormsSlicingTest()
: m_graph(),
  m_sliceCollector(std::make_shared<PatternNodesCollector>()),
  m_normsHandler(m_graph, m_sliceCollector),
  m_opIdx(0),
  m_bundleIdx(0),
  m_tpcsPerSlice(std::get<0>(GetParam())),
  m_numSliceTensors(std::get<1>(GetParam())),
  m_totalNumTpcs(m_tpcsPerSlice * m_numSliceTensors),
  m_ofmInSram(std::get<2>(GetParam())),
  m_sliceTensorInSram(std::get<3>(GetParam())),
  m_tilePattern(std::get<4>(GetParam()))
{
}

// Example graph that can be created:
//                +--------+                                 +--------+
//                | tensor +-+                               | tensor +-+
//                | Slice  | |                               | Slice  | |
//                +--------+ |                               +--------+ |
//                           |                                          |
// +------+   +------+       |  +------+      +------+   +------+       |  +------+
// |      |   |tensor|       +->|      |      |      |   |tensor|       +->|      |
// | Prod +-->| OFM  |-------+->| Norm |      | Prod +-->| OFM  |-------+->| Norm |
// +------+   +------+       |  +------+      +------+   +------+       |  +------+
//                           |                                          |
// +------+   +------+       |  +------+      +------+   +------+       |  +------+
// |      |   |tensor|       +->|      |      |      |   |tensor|       +->|      |
// | Prod +-->| OFM  |-------+->| Norm |      | Prod +-->| OFM  |-------+->| Norm |
// +------+   +------+       |  +------+      +------+   +------+       |  +------+
//                           |                                          |
// +------+   +------+       |  +------+      +------+   +------+       |  +------+
// |      |   |tensor|       +->|      |      |      |   |tensor|       +->|      |
// | Prod +-->| OFM  |-------+->| Norm |      | Prod +-->| OFM  |-------+->| Norm |
// +------+   +------+          +------+      +------+   +------+          +------+
//
// The params would be: m_tpcsPeSlice = 3, m_numSliceTensors = 2.
void HabanaNormsSlicingTest::createGraph()
{
    initOrigSliceNodeMap();
    createSliceTensors();
    addProducers();
    addReshapeNodes();
    addNormNodes();
    addSplitNode();
}

void HabanaNormsSlicingTest::initOrigSliceNodeMap()
{
    TensorPtr origSliceIn  = std::make_shared<Tensor>(syn_type_float, "origOfm");
    TensorPtr origSliceOut = std::make_shared<Tensor>(syn_type_float, "origSlice");
    // Those slice params are not interesting and probably incorrect:
    SliceNode::SliceNodeStaticParams sliceParams  = {.starts = {0, 0, 0, 0, 0},
                                                    .ends   = {4, 4, 4, 4, 4},
                                                    .steps  = {1, 1, 1, 1, 1}};
    m_sliceNode                                   = NodeFactory::createNode({origSliceIn},
                                                                            {origSliceOut},
                                          &sliceParams,
                                          NodeFactory::logicalSliceFwdNodeTypeName,
                                          "origSliceNode");
    m_sliceNode->getNodeAnnotation().originatedFromCguid = true;
    m_sliceNode->getNodeAnnotation().bundleInfo.set(
        BundleInfo(m_bundleIdx, BundleType::UNDEFINED, m_opIdx++));  // why is it needed?
    if (m_tilePattern)
    {
        ns_TileKernel::ParamsV2 params      = {4, 1, 1, 1, 1};
        TensorPtr               origTileOut = std::make_shared<Tensor>(syn_type_float, "origTileOut");
        m_tileNode = NodeFactory::createNode({origSliceOut}, {origTileOut}, &params, "tile_fwd_f32", "origTileNode");
    }
    m_sliceCollector->registerPattern(SliceTilePattern(m_sliceNode, m_tileNode));
}

void HabanaNormsSlicingTest::addSplitNode()
{
    if (m_tpcsPerSlice <= 1 || m_tilePattern) return;
    TensorPtr      splitIn  = std::make_shared<Tensor>(syn_type_float, "origSlice");
    synSplitParams splitDim = {0};  // It doesn't matter for the sake of this test
    NodePtr        splitNode =
        NodeFactory::createNode({splitIn}, m_sliceTensors, &splitDim, NodeFactory::splitNodeTypeName, "split");
    splitNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, BundleType::UNDEFINED, m_opIdx++));
    GraphEditor::addNode(m_graph, splitNode);
}

void HabanaNormsSlicingTest::createSliceTensors()
{
    for (unsigned i = 0; i < m_numSliceTensors; i++)
    {
        TensorPtr slice = std::make_shared<Tensor>(syn_type_float, ("slice_" + std::to_string(i)).c_str());
        slice->getTensorAnnotation().origBigTensor = m_sliceNode->getOutput(0);
        if (m_sliceTensorInSram) slice->setTensorInSram();
        m_sliceTensors.push_back(slice);
    }
}

void HabanaNormsSlicingTest::addProducers()
{
    for (unsigned i = 0; i < m_totalNumTpcs; i++)
    {
        TensorPtr ofm = std::make_shared<Tensor>(syn_type_float, ("ofm_" + std::to_string(i)).c_str());
        ofm->getTensorAnnotation().origBigTensor = m_sliceNode->getInput(0);
        NodePtr producerNode =
            NodeFactory::createNode({}, {ofm}, nullptr, NOP_KERNEL_NAME, "producer_" + std::to_string(i));
        producerNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, BundleType::UNDEFINED, m_opIdx++));
        GraphEditor::addNode(m_graph, producerNode);
        m_ofmTensors.push_back(ofm);
    }
}

void HabanaNormsSlicingTest::addNormNodes()
{
    for (unsigned i = 0; i < m_totalNumTpcs; i++)
    {
        unsigned sliceIdx = i / m_tpcsPerSlice;
        NodePtr  normNode =
            NodeFactory::createNode({m_tilePattern ? m_reshapeOutputTensors[i] : m_ofmTensors[i],
                                     m_tilePattern ? m_tileNode->getOutput(0) : m_sliceTensors[sliceIdx]},
                                    {},
                                    nullptr,
                                    NOP_KERNEL_NAME,
                                    "norm_" + std::to_string(i));
        normNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, BundleType::UNDEFINED, m_opIdx++));
        GraphEditor::addNode(m_graph, normNode);
    }
}

void HabanaNormsSlicingTest::addReshapeNodes()
{
    if (!m_tilePattern) return;
    for (unsigned i = 0; i < m_totalNumTpcs; i++)
    {
        TensorPtr reshapeOut  = std::make_shared<Tensor>(syn_type_float, "origReshapeOut");
        NodePtr   reshapeNode = NodeFactory::createNode({m_ofmTensors[i]},
                                                        {reshapeOut},
                                                      nullptr,
                                                      NodeFactory::reshapeNodeTypeName,
                                                      "reshape_" + std::to_string(i));
        reshapeNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, BundleType::UNDEFINED, m_opIdx++));
        GraphEditor::addNode(m_graph, reshapeNode);
        m_reshapeOutputTensors.push_back(reshapeOut);
    }
}

void HabanaNormsSlicingTest::runTest()
{
    graphVisualizationPre(m_graph);
    m_normsHandler.handleRemovedSliceNormNodes();
    graphVisualizationPost(m_graph);
}

void HabanaNormsSlicingTest::validateSlice()
{
    unsigned expectedSliceNodesNum = m_numSliceTensors;
    unsigned actualSliceNodesNum   = 0;
    for (const auto& node : m_graph.getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_SLICE)
        {
            actualSliceNodesNum++;
            auto     sliceConsumers = m_graph.getNodeConsumers(node);
            unsigned expectedSliceConsumers =
                node->getInput(0)->inSram() ? 1 /* memcpy consumer if the input is in sram*/ : m_tpcsPerSlice;
            ASSERT_EQ(sliceConsumers.size(), expectedSliceConsumers);
            if (sliceConsumers.size() == 1 &&
                isMemcpy(**sliceConsumers.begin()))  // If the only consumer is memcpy, iterate the memcpy's consumers
            {
                sliceConsumers = m_graph.getNodeConsumers(*sliceConsumers.begin());
                ASSERT_EQ(sliceConsumers.size(), m_tpcsPerSlice);
            }
            // get all the non-slice producers for the norms, and validate that the slice is connected to the producer
            // with the lowest execution index
            NodeSet allProducersBesideSliceAndMemcpy;
            for (const auto& sliceConsumer : sliceConsumers)
            {
                for (const NodePtr& prod : m_graph.getNodeProducers(sliceConsumer))
                {
                    if (prod->getNodeType() != Node::TYPE_SLICE && !isMemcpy(*prod))
                    {
                        allProducersBesideSliceAndMemcpy.insert(prod);
                    }
                    else if (isMemcpy(*prod))
                    {
                        ASSERT_TRUE(node->getInput(0)->inSram());
                    }
                    else  // if prod->getNodeType() == Node::TYPE_SLICE
                    {
                        ASSERT_FALSE(node->getInput(0)->inSram());
                    }
                }
            }
            auto sliceProducer = m_graph.getTensorProducer(node->getInput(0));
            ASSERT_NE(allProducersBesideSliceAndMemcpy.find(sliceProducer), allProducersBesideSliceAndMemcpy.end());
            unsigned producerExecIdx = sliceProducer->getExecutionOrderedIndex();
            // Validate slice producer has lowest execution order index
            for (const auto& producer : allProducersBesideSliceAndMemcpy)
            {
                if (producer == sliceProducer) continue;
                EXPECT_GT(producer->getExecutionOrderedIndex(), producerExecIdx);
            }
        }
    }
    ASSERT_EQ(actualSliceNodesNum, expectedSliceNodesNum);
}

// The expected graph after handling should look as follows:
//                 +-------+     +----------+    +------+      +----------+
//                 | Slice +---> |slice_out +--->| Tile +----> | tile_out |
//                 | Node  |     +----------+    | Node |      +-----+----+
//                 +-------+                     +------+            |
//                     ^                                             |
//                     |                                             v
// +----------+   +----+-----+    +--------+   +-------------+   +--------+
// | Producer +-->| prod_out +--->|Reshape +-->| reshape_out +-->|Consumer|
// | Node     |   +----------+    |Node    |   +-------------+   |Node    |
// +----------+                   +--------+                     +--------+
// Number of producer/reshape/consumer nodes depends on the test params
void HabanaNormsSlicingTest::validateSliceAndTile()
{
    unsigned expectedSliceNodesNum = 1;
    unsigned actualSliceNodesNum   = 0;
    unsigned expectedTileNodesNum  = 1;
    unsigned actualTileNodesNum    = 0;
    for (const auto& node : m_graph.getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_SLICE)
        {
            actualSliceNodesNum++;
            const auto& sliceConsumers = m_graph.getNodeConsumers(node);
            ASSERT_EQ(sliceConsumers.size(), 1);
            ASSERT_NE((*sliceConsumers.begin())->getGUID().find("tile"), std::string::npos); /* tile consumer */
            const NodePtr& tile          = (*sliceConsumers.begin());
            const auto&    tileConsumers = m_graph.getNodeConsumers(tile);
            actualTileNodesNum++;
            // get all the non-tile producers for the norms, and validate that the slice and tile are connected to the
            // producer with the lowest execution index
            NodeSet ofmRealProducers;
            for (const auto& tileConsumer : tileConsumers)
            {
                for (const NodePtr& prod : m_graph.getNodeProducers(tileConsumer))
                {
                    if (prod->getGUID().find("tile") == std::string::npos)
                    {
                        ASSERT_TRUE(isReshapeNode(prod));
                        ofmRealProducers.insert(m_graph.getTensorProducer(prod->getInput(0)));
                    }
                }
            }
            auto sliceProducer = m_graph.getTensorProducer(node->getInput(0));
            ASSERT_NE(ofmRealProducers.find(sliceProducer), ofmRealProducers.end());
            unsigned producerExecIdx = sliceProducer->getExecutionOrderedIndex();
            // Validate slice producer has lowest execution order index
            for (const auto& producer : ofmRealProducers)
            {
                if (producer == sliceProducer) continue;
                EXPECT_GT(producer->getExecutionOrderedIndex(), producerExecIdx);
            }
        }
    }
    ASSERT_EQ(actualSliceNodesNum, expectedSliceNodesNum);
    ASSERT_EQ(actualTileNodesNum, expectedTileNodesNum);
}

void HabanaNormsSlicingTest::validateResult()
{
    if (m_tilePattern)
    {
        validateSliceAndTile();
        return;
    }
    validateSlice();
}

TEST_P(HabanaNormsSlicingTest, handle_removed_slice_norms)
{
    createGraph();
    runTest();
    validateResult();
}

INSTANTIATE_TEST_SUITE_P(
    SlicePattern,
    HabanaNormsSlicingTest,
    ::testing::Combine(::testing::Values<unsigned>(1, 3, 6),  // m_tpcsPerSlice - number of tpc nodes that consume a
                                                              // single "k"-slice tensor in the post-slicing graph
                       ::testing::Values<unsigned>(1, 5),  // m_numSliceTensors - number of tensors that represent the
                                                           // "k"-slice in the post-slicing graph
                       ::testing::Values<bool>(true, false),
                       ::testing::Values<bool>(true, false),
                       ::testing::Values<bool>(false)));

INSTANTIATE_TEST_SUITE_P(
    SliceAndTilePattern,
    HabanaNormsSlicingTest,
    ::testing::Combine(::testing::Values<unsigned>(1, 3, 6),  // m_tpcsPerSlice - number of tpc nodes that consume a
                                                              // single "k"-slice tensor in the post-slicing graph
                       ::testing::Values<unsigned>(1),  // m_numSliceTensors - number of tensors that represent the
                                                        // "k"-slice in the post-slicing graph
                       ::testing::Values<bool>(true, false),
                       ::testing::Values<bool>(false),
                       ::testing::Values<bool>(true)));