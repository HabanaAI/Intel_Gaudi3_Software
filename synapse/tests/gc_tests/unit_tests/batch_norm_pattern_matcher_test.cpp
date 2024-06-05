#include "batch_norm_pattern_matcher.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "perf_lib_layer_params.h"
#include "node_factory.h"
#include <string>

class InstanceNormToBatchNormPatternMatcherTest : public GraphOptimizerTest
{
protected:
    Gaudi2Graph m_graph;

    TensorPtr createTensor(const std::vector<TSize>& shape) const
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_single);
    }

    NodeVector createBNPattern(const TSize batchSize)
    {
        const TSize C = 32;
        const TSize H = 160;
        const TSize W = 192;

        TensorPtr ifm    = createTensor({C, W, H, batchSize});
        TensorPtr gradIn = createTensor({C, W, H, batchSize});
        TensorPtr mean   = createTensor({C, batchSize});
        TensorPtr istd   = createTensor({C, batchSize});
        TensorPtr gamma  = createTensor({C});

        TensorPtr gradOut   = createTensor({C, W, H, batchSize});
        TensorPtr gradBeta  = createTensor({C, batchSize});
        TensorPtr gradGamma = createTensor({C, batchSize});

        TensorVector splitIfm;
        TensorVector splitGradIn;
        TensorVector splitMean;
        TensorVector splitIstd;
        TensorVector concatGradOut;
        TensorVector concatGradBeta;
        TensorVector concatGradGamma;

        for (auto i = 0; i < batchSize; i++)
        {
            splitIfm.push_back(createTensor({C, W, H, 1}));
            splitGradIn.push_back(createTensor({C, W, H, 1}));
            splitMean.push_back(createTensor({C}));
            splitIstd.push_back(createTensor({C}));
            concatGradOut.push_back(createTensor({C, W, H, 1}));
            concatGradBeta.push_back(createTensor({C}));
            concatGradGamma.push_back(createTensor({C}));
        }

        synSplitParams splitIfmParams;
        splitIfmParams.axis  = 3;
        NodePtr splitIfmNode = NodeFactory::createNode({ifm},
                                                       splitIfm,
                                                       &splitIfmParams,
                                                       NodeFactory::splitNodeInternalTypeName,
                                                       "splitIfm_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, splitIfmNode));

        synSplitParams splitGradInParams;
        splitGradInParams.axis  = 3;
        NodePtr splitGradInNode = NodeFactory::createNode({gradIn},
                                                          splitGradIn,
                                                          &splitGradInParams,
                                                          NodeFactory::splitNodeInternalTypeName,
                                                          "splitGradIn_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, splitGradInNode));

        synSplitParams splitMeanParams;
        splitMeanParams.axis  = 1;
        NodePtr splitMeanNode = NodeFactory::createNode({mean},
                                                        splitMean,
                                                        &splitMeanParams,
                                                        NodeFactory::splitNodeInternalTypeName,
                                                        "splitMean_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, splitMeanNode));

        synSplitParams splitIstdParams;
        splitIstdParams.axis  = 1;
        NodePtr splitIstdNode = NodeFactory::createNode({istd},
                                                        splitIstd,
                                                        &splitIstdParams,
                                                        NodeFactory::splitNodeInternalTypeName,
                                                        "splitIstd_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, splitIstdNode));

        synConcatenateParams concatGradOutParams;
        concatGradOutParams.axis  = 3;
        NodePtr concatGradOutNode = NodeFactory::createNode(concatGradOut,
                                                            {gradOut},
                                                            &concatGradOutParams,
                                                            NodeFactory::concatenateNodeInternalTypeName,
                                                            "concatGradOut_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, concatGradOutNode));

        synConcatenateParams concatGradBetaParams;
        concatGradBetaParams.axis  = 1;
        NodePtr concatGradBetaNode = NodeFactory::createNode(concatGradBeta,
                                                             {gradBeta},
                                                             &concatGradBetaParams,
                                                             NodeFactory::concatenateNodeInternalTypeName,
                                                             "concatGradBeta_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, concatGradBetaNode));

        synConcatenateParams concatGradGammaParams;
        concatGradGammaParams.axis  = 1;
        NodePtr concatGradGammaNode = NodeFactory::createNode(concatGradGamma,
                                                              {gradGamma},
                                                              &concatGradGammaParams,
                                                              NodeFactory::concatenateNodeInternalTypeName,
                                                              "concatGradGamma_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, concatGradGammaNode));

        NodeVector bnNodes;
        for (auto i = 0; i < batchSize; i++)
        {
            ns_BatchNormKernel::Params bnParams {};
            NodePtr                    bnNode = NodeFactory::createNode(
                {splitIfmNode->getOutput(i),
                 splitGradInNode->getOutput(i),
                 splitMeanNode->getOutput(i),
                 splitIstdNode->getOutput(i),
                 gamma},
                {concatGradOutNode->getInput(i), concatGradBetaNode->getInput(i), concatGradGammaNode->getInput(i)},
                &bnParams,
                "batch_norm_bwd_f32",
                "BN_" + std::to_string(i) + "_" + std::to_string(batchSize));
            EXPECT_TRUE(GraphEditor::addNode(m_graph, bnNode));
            bnNodes.push_back(bnNode);
        }

        return bnNodes;
    }

    NodePtr createSingleBn(const TSize batchSize)
    {
        const TSize C = 32;
        const TSize H = 160;
        const TSize W = 192;

        TensorPtr ifm    = createTensor({C, W, H, batchSize});
        TensorPtr gradIn = createTensor({C, W, H, batchSize});
        TensorPtr mean   = createTensor({C});
        TensorPtr istd   = createTensor({C});
        TensorPtr gamma  = createTensor({C});

        TensorPtr gradOut   = createTensor({C, W, H, batchSize});
        TensorPtr gradBeta  = createTensor({C});
        TensorPtr gradGamma = createTensor({C});

        ns_BatchNormKernel::Params bnParams {};
        NodePtr                    bnNode = NodeFactory::createNode({ifm, gradIn, mean, istd, gamma},
                                                 {gradOut, gradBeta, gradGamma},
                                                 &bnParams,
                                                 "batch_norm_bwd_f32",
                                                 "BN_" + std::to_string(batchSize));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, bnNode));

        return bnNode;
    }

    void validateBNPatttern(const TSize                                  batchSize,
                            const NodeVector&                            bnNodes,
                            const InstanceNormToBatchNormPatternMatcher& bnPatternMatcher) const
    {
        for (const auto& bnNode : bnNodes)
        {
            const auto& [valid, concurrencyLevel] = bnPatternMatcher.matchPattern(bnNode);
            ASSERT_TRUE(valid);
            ASSERT_EQ(concurrencyLevel, batchSize);
        }
    }

    void validateNoPattern(const NodeVector&                            bnNodes,
                           const InstanceNormToBatchNormPatternMatcher& bnPatternMatcher) const
    {
        for (const auto& bnNode : bnNodes)
        {
            const auto& [valid, concurrencyLevel] = bnPatternMatcher.matchPattern(bnNode);
            ASSERT_FALSE(valid);
            ASSERT_EQ(concurrencyLevel, 0);
        }
    }
};

TEST_F(InstanceNormToBatchNormPatternMatcherTest, batch_norm_pattern_matcher_test)
{
    setGlobalConfForTest(GCFG_SKIP_BN_SPLIT_FOR_IN_REDUCED_TO_BN, "true");

    NodeVector bnNodesBatch4  = createBNPattern(4);
    NodeVector bnNodesBatch7  = createBNPattern(7);
    NodeVector bnNodesBatch63 = createBNPattern(63);

    NodePtr bnSingleBatch15 = createSingleBn(15);
    NodePtr bnSingleBatch1  = createSingleBn(1);

    InstanceNormToBatchNormPatternMatcher bnPatternMatcher(m_graph);

    validateBNPatttern(4, bnNodesBatch4, bnPatternMatcher);
    validateBNPatttern(7, bnNodesBatch7, bnPatternMatcher);
    validateBNPatttern(63, bnNodesBatch63, bnPatternMatcher);

    validateNoPattern({bnSingleBatch15, bnSingleBatch1}, bnPatternMatcher);
}