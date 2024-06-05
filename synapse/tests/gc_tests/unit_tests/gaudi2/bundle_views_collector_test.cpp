#include "gaudi2_graph.h"
#include "slicer/bundle_views_collector.h"
#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"

using namespace gc::layered_brain;

class BundleViewsCollectorTest : public GraphOptimizerTest
{
protected:
    NodePtr addTPCNode(unsigned         numDims,
                       TSize            dimSize,
                       unsigned         granularity,
                       int              inputOverlap,
                       bool             transpose,
                       const TensorPtr& input  = nullptr,
                       const TensorPtr& output = nullptr)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        for (auto i = 0; i < numDims; i++)
        {
            nodeParams.dims.emplace_back(dimSize, granularity, inputOverlap);
        }
        nodeParams.transpose = transpose;
        NodePtr node         = TPCCustomIndexSpaceNode::create(nodeParams, input, output);
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        m_bundleNodes.push_back(node);
        return node;
    }

    NodePtr addAllRequiredTPCNode(unsigned numDims)
    {
        TPCCustomIndexSpaceMappingNode::Params params;
        params.tensorRank           = numDims;
        params.nodeResolutionRank   = 1;
        params.dimMappingForInputs  = {{}};
        params.dimMappingForOutputs = {{}};
        NodePtr node                = TPCCustomIndexSpaceMappingNode::create(params);
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        m_bundleNodes.push_back(node);
        return node;
    }

    NodeVector addTpcChain(unsigned chainLength,
                           unsigned numDims,
                           TSize    dimSize,
                           unsigned granularity,
                           int      inputOverlap,
                           bool     transpose,
                           bool     isFirstAllRequired)
    {
        NodeVector tpcChain;
        NodePtr    prevNode;
        for (auto i = 0; i < chainLength; i++)
        {
            TensorPtr input = prevNode ? prevNode->getOutput(0) : nullptr;
            NodePtr   node =
                (i == 0 && isFirstAllRequired)
                      ? addAllRequiredTPCNode(numDims)
                      : addTPCNode(numDims, dimSize, granularity, (i == 0) ? inputOverlap : 0, transpose, input);
            prevNode        = node;
            tpcChain.push_back(node);
        }
        return tpcChain;
    }

    NodePtr addGEMMNode(bool             transposeA,
                        bool             transposeB,
                        TSize            dimSize,
                        unsigned         numBatchDims,
                        const TensorPtr& inputA = nullptr,
                        const TensorPtr& inputB = nullptr,
                        const TensorPtr& output = nullptr)
    {
        std::vector<TSize> batchDims(numBatchDims, dimSize);

        std::string   guid = (batchDims.empty()) ? NodeFactory::gemmNodeTypeName : NodeFactory::batchGemmNodeTypeName;
        synGEMMParams params(transposeA, transposeB);

        TensorVector inputs;
        inputs.push_back(inputA ? inputA : createTensorForGemm({dimSize, dimSize}, params.transpose_a, batchDims));
        inputs.push_back(inputB ? inputB : createTensorForGemm({dimSize, dimSize}, params.transpose_b, batchDims));

        TensorVector outputs;
        outputs.push_back(output ? output : createTensorForGemm({dimSize, dimSize}, false, batchDims));

        NodePtr node =
            NodeFactory::createNode(inputs, outputs, &params, guid.c_str(), "GEMM" + std::to_string(m_nodeId++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        m_bundleNodes.push_back(node);
        return node;
    }

    void addGemmWithTpcChains(unsigned producerChainLength,
                              unsigned consumerChainLength,
                              TSize    dimSize,
                              unsigned granularity,
                              int      inputOverlap,
                              bool     transpose,
                              bool     isFirstAllRequired,
                              bool     transposeA,
                              bool     transposeB,
                              unsigned numBatchDims)
    {
        // Create a single GEMM with producer chain for both inputs and consumer chain for the output.
        const unsigned    numDims        = 2 + numBatchDims;
        const NodeVector& producerAChain = addTpcChain(producerChainLength,
                                                       numDims,
                                                       dimSize,
                                                       granularity,
                                                       inputOverlap,
                                                       transpose,
                                                       isFirstAllRequired);
        const NodeVector& producerBChain = addTpcChain(producerChainLength,
                                                       numDims,
                                                       dimSize,
                                                       granularity,
                                                       inputOverlap,
                                                       transpose,
                                                       isFirstAllRequired);
        const NodeVector& consumerChain =
            addTpcChain(consumerChainLength, numDims, dimSize, granularity, 0, transpose, isFirstAllRequired);

        const TensorPtr& inputA = producerAChain.back()->getOutput(0);
        EXPECT_TRUE(inputA);
        const TensorPtr& inputB = producerBChain.back()->getOutput(0);
        EXPECT_TRUE(inputB);
        const TensorPtr& output = consumerChain.front()->getInput(0);
        EXPECT_TRUE(output);
        addGEMMNode(transposeA, transposeB, dimSize, numBatchDims, inputA, inputB, output);
    }

    void addMultiGemmsWithTpcChains(unsigned producerChainLength,
                                    unsigned consumerChainLength,
                                    TSize    dimSize,
                                    unsigned granularity,
                                    int      inputOverlap,
                                    bool     transpose,
                                    bool     isFirstAllRequired,
                                    bool     transposeA,
                                    bool     transposeB,
                                    unsigned numBatchDims)
    {
        // Create 3 GEMMs with shared inputs.
        // Each input has a producer chain and each output has a consumer chain.

        const unsigned    numDims                 = 2 + numBatchDims;
        const NodeVector& sharedProducerChain1    = addTpcChain(producerChainLength,
                                                             numDims,
                                                             dimSize,
                                                             granularity,
                                                             inputOverlap,
                                                             transpose,
                                                             isFirstAllRequired);
        const NodeVector& sharedProducerChain2    = addTpcChain(producerChainLength,
                                                             numDims,
                                                             dimSize,
                                                             granularity,
                                                             inputOverlap,
                                                             transpose,
                                                             isFirstAllRequired);
        const NodeVector& nonSharedProducerChain1 = addTpcChain(producerChainLength,
                                                                numDims,
                                                                dimSize,
                                                                granularity,
                                                                inputOverlap,
                                                                transpose,
                                                                isFirstAllRequired);
        const NodeVector& nonSharedProducerChain2 = addTpcChain(producerChainLength,
                                                                numDims,
                                                                dimSize,
                                                                granularity,
                                                                inputOverlap,
                                                                transpose,
                                                                isFirstAllRequired);

        const NodeVector& consumerChain1 =
            addTpcChain(consumerChainLength, numDims, dimSize, granularity, 0, transpose, isFirstAllRequired);
        const NodeVector& consumerChain2 =
            addTpcChain(consumerChainLength, numDims, dimSize, granularity, 0, transpose, isFirstAllRequired);
        const NodeVector& consumerChain3 =
            addTpcChain(consumerChainLength, numDims, dimSize, granularity, 0, transpose, isFirstAllRequired);

        const TensorPtr& sharedIn1 = sharedProducerChain1.back()->getOutput(0);
        EXPECT_TRUE(sharedIn1);
        const TensorPtr& sharedIn2 = sharedProducerChain2.back()->getOutput(0);
        EXPECT_TRUE(sharedIn2);
        const TensorPtr& nonSharedIn1 = nonSharedProducerChain1.back()->getOutput(0);
        EXPECT_TRUE(nonSharedIn1);
        const TensorPtr& nonSharedIn2 = nonSharedProducerChain2.back()->getOutput(0);
        EXPECT_TRUE(nonSharedIn2);

        const TensorPtr& output1 = consumerChain1.front()->getInput(0);
        EXPECT_TRUE(output1);
        const TensorPtr& output2 = consumerChain2.front()->getInput(0);
        EXPECT_TRUE(output2);
        const TensorPtr& output3 = consumerChain3.front()->getInput(0);
        EXPECT_TRUE(output3);

        addGEMMNode(transposeA, transposeB, dimSize, numBatchDims, nonSharedIn1, sharedIn1, output1);
        addGEMMNode(transposeA, transposeB, dimSize, numBatchDims, sharedIn1, sharedIn2, output2);
        addGEMMNode(transposeA, transposeB, dimSize, numBatchDims, sharedIn2, nonSharedIn2, output3);
    }

    void test()
    {
        std::tie(m_granularityPerTensor, m_granularityPerNode) = getMinCommonTilesSizes(m_bundleNodes);
        BundleViewsCollector       bundleViewsCollector(m_bundleNodes);
        const BundleViewContainerPtr& bundleViews =
            bundleViewsCollector.getAllBundleViews(m_granularityPerTensor, m_granularityPerNode);
        validateBundleViews(bundleViews);
    }

private:
    TensorPtr createTensorForGemm(std::vector<TSize> shape, bool transposed, const std::vector<TSize>& batchDims) const
    {
        if (transposed)
        {
            std::swap(shape[0], shape[1]);
        }
        if (!batchDims.empty())
        {
            shape.insert(shape.end(), batchDims.begin(), batchDims.end());
        }
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
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

    void validateTensorDimsMapping(const BundleViewContainerPtr& bundleViews) const
    {
        for (const auto& node : m_bundleNodes)
        {
            for (const auto& tensor : node->getOperands())
            {
                if (!tensor) continue;
                for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
                {
                    unsigned numOfBVDs = 0;
                    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
                    {
                        const auto& bundleView = bundleViews->getBundleView(bvdId);
                        if (bundleView.tensorDimsGranularity.find({tensor, tensorDim}) !=
                            bundleView.tensorDimsGranularity.end())
                        {
                            numOfBVDs++;
                            ASSERT_TRUE(m_granularityPerTensor.find(tensor) != m_granularityPerTensor.end());
                            ASSERT_EQ(bundleView.tensorDimsGranularity.at({tensor, tensorDim}),
                                      m_granularityPerTensor.at(tensor)[tensorDim]);
                            ASSERT_EQ(bundleViews->getBVDForTensorDim(tensor, tensorDim), bundleView.id);
                        }
                    }
                    ASSERT_EQ(numOfBVDs, 1);
                }
            }
        }
    }

    void validateNodeDimsMapping(const BundleViewContainerPtr& bundleViews) const
    {
        for (const auto& node : m_bundleNodes)
        {
            const auto& nodeAp = node->getNodeAccessPattern();
            ASSERT_TRUE(nodeAp);
            for (Dim nodeDim = 0; nodeDim < nodeAp->getNodeResolution().size(); nodeDim++)
            {
                unsigned numOfBVDs = 0;
                for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
                {
                    const auto& bundleView = bundleViews->getBundleView(bvdId);
                    if (bundleView.nodeDimsGranularity.find({node, nodeDim}) != bundleView.nodeDimsGranularity.end())
                    {
                        numOfBVDs++;
                        ASSERT_TRUE(m_granularityPerNode.find(node) != m_granularityPerNode.end());
                        ASSERT_EQ(bundleView.nodeDimsGranularity.at({node, nodeDim}),
                                  m_granularityPerNode.at(node)[nodeDim]);
                        ASSERT_EQ(bundleViews->getBVDForNodeDim(node, nodeDim), bundleView.id);
                        ASSERT_EQ(bundleView.resolution,
                                  div_round_up(nodeAp->getNodeResolution()[nodeDim],
                                               bundleView.nodeDimsGranularity.at({node, nodeDim})));
                    }
                }
                ASSERT_LE(numOfBVDs, 1);
            }
        }

        for (const auto& node : m_bundleNodes)
        {
            const auto& nodeAp = node->getNodeAccessPattern();
            ASSERT_TRUE(nodeAp);
            for (Dim indexSpaceDim = 0; indexSpaceDim < nodeAp->getNodeResolution().size(); indexSpaceDim++)
            {
                std::set<std::pair<TensorPtr, Dim>> mappedTensorsDims;
                for (const auto& tensor : node->getOperands())
                {
                    ASSERT_TRUE(tensor);
                    for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
                    {
                        if (nodeAp->getIndexSpaceDim(tensor, tensorDim) == indexSpaceDim)
                        {
                            mappedTensorsDims.emplace(tensor, tensorDim);
                        }
                    }
                }
                if (!mappedTensorsDims.empty())
                {
                    BundleViewId bvdId = bundleViews->getBVDForNodeDim(node, indexSpaceDim);
                    for (const auto& mappedTensorDim : mappedTensorsDims)
                    {
                        ASSERT_EQ(bundleViews->getBVDForTensorDim(mappedTensorDim.first, mappedTensorDim.second),
                                  bvdId);
                    }
                }
            }
        }
    }

    void validateBundleViews(const BundleViewContainerPtr& bundleViews) const
    {
        // 1) Each tensor dim in the bundle should be mapped to exactly one BVD.
        // 2) Each node dim in the bundle should be mapped to at most one BVD.
        // (index-space dims that none of the tensor dims mapped to them will not appear in any BVD).
        // 3) All tensor dims that are mapped to the same index-space dim should be in the same BVD.
        // 4) Granularity for each tensor dim and node dim should be taken from LCM output.

        for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
        {
            ASSERT_EQ(bundleViews->getBundleView(bvdId).id, bvdId);
        }
        validateTensorDimsMapping(bundleViews);
        validateNodeDimsMapping(bundleViews);
    }

    Gaudi2Graph       m_graph;
    NodeVector        m_bundleNodes;
    TileSizePerTensor m_granularityPerTensor;
    TileSizePerNode   m_granularityPerNode;
    unsigned          m_nodeId = 0;
};

// #################################################################################

class BundleViewsCollectorTpcChainTest
: public BundleViewsCollectorTest
, public ::testing::WithParamInterface<std::tuple<unsigned,  // chain length
                                                  unsigned,  // num dims
                                                  TSize,     // dim size
                                                  unsigned,  // dim granularity
                                                  int,       // input overlap
                                                  bool,      // TPC output is transposed relative to the input
                                                  bool       // is first node in chain all-required
                                                  >>
{
protected:
    BundleViewsCollectorTpcChainTest()
    {
        std::tie(m_chainLength,
                 m_numDims,
                 m_dimSize,
                 m_granularity,
                 m_inputOverlap,
                 m_transposeTpcOut,
                 m_isFirstAllRequired) = GetParam();
    }

    void createGraph()
    {
        addTpcChain(m_chainLength,
                    m_numDims,
                    m_dimSize,
                    m_granularity,
                    m_inputOverlap,
                    m_transposeTpcOut,
                    m_isFirstAllRequired);
    }

    unsigned m_chainLength;
    unsigned m_numDims;
    TSize    m_dimSize;
    unsigned m_granularity;
    int      m_inputOverlap;
    bool     m_transposeTpcOut;
    bool     m_isFirstAllRequired;
};

TEST_P(BundleViewsCollectorTpcChainTest, collect_bundle_views_for_tpc_chain)
{
    if ((m_numDims < 2) && m_transposeTpcOut)
    {
        return;  // Skip invalid configs
    }
    createGraph();
    test();
}

INSTANTIATE_TEST_SUITE_P(tpc_chain_test,
                         BundleViewsCollectorTpcChainTest,
                         ::testing::Combine(::testing::Values(1, 3, 5, 13),    // chain length
                                            ::testing::Values(1, 2, 3, 4),     // num dims
                                            ::testing::Values(128),            // dim size
                                            ::testing::Values(1, 2, 128),      // dim granularity
                                            ::testing::Values(0, -2, 2),       // input overlap
                                            ::testing::Values(false, true),    // transpose TPC output
                                            ::testing::Values(false, true)));  // is first node in chain all-required

// #################################################################################

class BundleViewsCollectorSingleGemmTest
: public BundleViewsCollectorTest
, public ::testing::WithParamInterface<std::tuple<bool,     // transpose A
                                                  bool,     // transpose B
                                                  TSize,    // dim size
                                                  unsigned  // num of batch dims
                                                  >>
{
protected:
    BundleViewsCollectorSingleGemmTest()
    {
        std::tie(m_transposeA, m_transposeB, m_dimSize, m_numBatchDims) = GetParam();
    }
    void     createGraph() { addGEMMNode(m_transposeA, m_transposeB, m_dimSize, m_numBatchDims); }
    bool     m_transposeA;
    bool     m_transposeB;
    TSize    m_dimSize;
    unsigned m_numBatchDims;
};

TEST_P(BundleViewsCollectorSingleGemmTest, collect_bundle_views_for_single_gemm)
{
    createGraph();
    test();
}

INSTANTIATE_TEST_SUITE_P(single_gemm_test,
                         BundleViewsCollectorSingleGemmTest,
                         ::testing::Combine(::testing::Values(false, true),   // transpose A
                                            ::testing::Values(false, true),   // transpose B
                                            ::testing::Values(128),           // dim size
                                            ::testing::Values(0, 1, 2, 3)));  // num of batch dims

// #################################################################################

class BundleViewsCollectorSingleGemmWithTpcChainsTest
: public BundleViewsCollectorTest
, public ::testing::WithParamInterface<std::tuple<unsigned,  // producer chain length
                                                  unsigned,  // consumer chain length
                                                  TSize,     // dim size
                                                  unsigned,  // dim granularity
                                                  int,       // input overlap
                                                  bool,      // TPC output is transposed relative to the input
                                                  bool,      // is first node in chain all-required
                                                  bool,      // transpose A
                                                  bool,      // transpose B
                                                  unsigned   // num of batch dims
                                                  >>
{
protected:
    BundleViewsCollectorSingleGemmWithTpcChainsTest()
    {
        std::tie(m_producerChainLength,
                 m_consumerChainLength,
                 m_dimSize,
                 m_granularity,
                 m_inputOverlap,
                 m_transposeTpcOut,
                 m_isFirstAllRequired,
                 m_transposeA,
                 m_transposeB,
                 m_numBatchDims) = GetParam();
    }
    void createGraph()
    {
        addGemmWithTpcChains(m_producerChainLength,
                             m_consumerChainLength,
                             m_dimSize,
                             m_granularity,
                             m_inputOverlap,
                             m_transposeTpcOut,
                             m_isFirstAllRequired,
                             m_transposeA,
                             m_transposeB,
                             m_numBatchDims);
    }
    unsigned m_producerChainLength;
    unsigned m_consumerChainLength;
    TSize    m_dimSize;
    unsigned m_granularity;
    int      m_inputOverlap;
    bool     m_transposeTpcOut;
    bool     m_isFirstAllRequired;
    bool     m_transposeA;
    bool     m_transposeB;
    unsigned m_numBatchDims;
};

TEST_P(BundleViewsCollectorSingleGemmWithTpcChainsTest, collect_bundle_views_for_single_gemm_with_tpc_chains)
{
    createGraph();
    test();
}

INSTANTIATE_TEST_SUITE_P(single_gemm_with_tpc_chains_test,
                         BundleViewsCollectorSingleGemmWithTpcChainsTest,
                         ::testing::Combine(::testing::Values(1, 3, 13),     // producer chain length
                                            ::testing::Values(1, 4, 11),     // consumer chain length
                                            ::testing::Values(128),          // dim size
                                            ::testing::Values(1, 2, 128),    // dim granularity
                                            ::testing::Values(0, -2, 2),     // input overlap
                                            ::testing::Values(false, true),  // transpose TPC output
                                            ::testing::Values(false, true),  // is first node in chain all-required
                                            ::testing::Values(false, true),  // transpose A
                                            ::testing::Values(false, true),  // transpose B
                                            ::testing::Values(0, 1, 2)));    // batch dims

// #################################################################################

class BundleViewsCollectorMultiGemmsWithTpcChainsTest
: public BundleViewsCollectorTest
, public ::testing::WithParamInterface<std::tuple<unsigned,  // producer chain length
                                                  unsigned,  // consumer chain length
                                                  TSize,     // dim size
                                                  unsigned,  // dim granularity
                                                  int,       // input overlap
                                                  bool,      // TPC output is transposed relative to the input
                                                  bool,      // is first node in chain all-required
                                                  bool,      // transpose A
                                                  bool,      // transpose B
                                                  unsigned   // num of batch dims
                                                  >>
{
protected:
    BundleViewsCollectorMultiGemmsWithTpcChainsTest()
    {
        std::tie(m_producerChainLength,
                 m_consumerChainLength,
                 m_dimSize,
                 m_granularity,
                 m_inputOverlap,
                 m_transposeTpcOut,
                 m_isFirstAllRequired,
                 m_transposeA,
                 m_transposeB,
                 m_numBatchDims) = GetParam();
    }
    void createGraph()
    {
        addMultiGemmsWithTpcChains(m_producerChainLength,
                                   m_consumerChainLength,
                                   m_dimSize,
                                   m_granularity,
                                   m_inputOverlap,
                                   m_transposeTpcOut,
                                   m_isFirstAllRequired,
                                   m_transposeA,
                                   m_transposeB,
                                   m_numBatchDims);
    }
    unsigned m_producerChainLength;
    unsigned m_consumerChainLength;
    TSize    m_dimSize;
    unsigned m_granularity;
    int      m_inputOverlap;
    bool     m_transposeTpcOut;
    bool     m_isFirstAllRequired;
    bool     m_transposeA;
    bool     m_transposeB;
    unsigned m_numBatchDims;
};

TEST_P(BundleViewsCollectorMultiGemmsWithTpcChainsTest, collect_bundle_views_for_multi_gemms_with_tpc_chains)
{
    createGraph();
    test();
}

INSTANTIATE_TEST_SUITE_P(multi_gemms_with_tpc_chains_test,
                         BundleViewsCollectorMultiGemmsWithTpcChainsTest,
                         ::testing::Combine(::testing::Values(1, 3, 13),     // producer chain length
                                            ::testing::Values(1, 4, 11),     // consumer chain length
                                            ::testing::Values(128),          // dim size
                                            ::testing::Values(1, 2, 128),    // dim granularity
                                            ::testing::Values(0, -2, 2),     // input overlap
                                            ::testing::Values(false, true),  // transpose TPC output
                                            ::testing::Values(false, true),  // is first node in chain all-required
                                            ::testing::Values(false, true),  // transpose A
                                            ::testing::Values(false, true),  // transpose B
                                            ::testing::Values(0, 1, 2)));    // batch dims