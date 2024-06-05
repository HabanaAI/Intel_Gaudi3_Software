#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "slicer/bundle_views_collector.h"
#include "slicer/sliced_bundle_graph_generator.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"
#include "types.h"

using namespace gc::layered_brain;

class SlicedBundleGraphGeneratorTest : public GraphOptimizerTest
{
protected:
    NodePtr addGEMMNode(bool             transposeA,
                        bool             transposeB,
                        TSize            heightA,
                        TSize            commonDim,
                        TSize            widthB,
                        unsigned         numBatchDims,
                        TSize            batchDimSize,
                        const TensorPtr& inputA     = nullptr,
                        const TensorPtr& inputB     = nullptr,
                        const TensorPtr& output     = nullptr,
                        bool             isInBundle = true)
    {
        std::vector<TSize> batchDims(numBatchDims, batchDimSize);

        std::string   guid = (batchDims.empty()) ? NodeFactory::gemmNodeTypeName : NodeFactory::batchGemmNodeTypeName;
        synGEMMParams params(transposeA, transposeB);

        TensorVector inputs;
        inputs.push_back(inputA ? inputA : createTensorForGemm({commonDim, heightA}, params.transpose_a, batchDims));
        inputs.push_back(inputB ? inputB : createTensorForGemm({widthB, commonDim}, params.transpose_b, batchDims));

        TensorVector outputs;
        outputs.push_back(output ? output : createTensorForGemm({widthB, heightA}, false, batchDims));

        NodePtr node = NodeFactory::createNode(inputs,
                                               outputs,
                                               &params,
                                               guid.c_str(),
                                               (batchDims.empty() ? "GEMM" : "BGEMM") + std::to_string(m_nodeId++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addBroadcastBGEMMNode(bool     transposeA,
                                  bool     transposeB,
                                  TSize    heightA,
                                  TSize    commonDim,
                                  TSize    widthB,
                                  unsigned numBatchDims,
                                  TSize    batchDimSize,
                                  bool     isOperandABroadcasted,
                                  bool     isBroadcastDimPresent,
                                  bool     isInBundle = true)
    {
        HB_ASSERT(numBatchDims > 0, "Expected at least a single batch dim");

        std::vector<TSize> batchDimsOpA;
        std::vector<TSize> batchDimsOpB;
        if (isOperandABroadcasted)
        {
            if (isBroadcastDimPresent)
            {
                batchDimsOpA.resize(numBatchDims, 1);
            }
            batchDimsOpB.resize(numBatchDims, batchDimSize);
        }
        else  // Operand B is broadcasted
        {
            if (isBroadcastDimPresent)
            {
                batchDimsOpB.resize(numBatchDims, 1);
            }
            batchDimsOpA.resize(numBatchDims, batchDimSize);
        }
        std::vector<TSize> batchDimsOut(numBatchDims, batchDimSize);

        synGEMMParams params(transposeA, transposeB);

        TensorVector inputs;
        inputs.push_back(createTensorForGemm({commonDim, heightA}, params.transpose_a, batchDimsOpA));
        inputs.push_back(createTensorForGemm({widthB, commonDim}, params.transpose_b, batchDimsOpB));

        TensorVector outputs;
        outputs.push_back(createTensorForGemm({widthB, heightA}, false, batchDimsOut));

        NodePtr node = NodeFactory::createNode(inputs,
                                               outputs,
                                               &params,
                                               NodeFactory::batchGemmNodeTypeName,
                                               "BGEMM" + std::to_string(m_nodeId++));
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addTPCNode(unsigned         numDims,
                       const SizeArray& sizes,
                       unsigned         granularity,
                       int              inputOverlap,
                       bool             transpose,
                       const TensorPtr& input      = nullptr,
                       const TensorPtr& output     = nullptr,
                       bool             isInBundle = true)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        for (auto i = 0; i < numDims; i++)
        {
            nodeParams.dims.emplace_back(sizes[i], granularity, inputOverlap);
        }
        nodeParams.transpose = transpose;
        NodePtr node         = TPCCustomIndexSpaceNode::create(nodeParams, input, output);
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addTPCNodeWithBroadcast(const SizeVector& inAsizes, const SizeVector& inBSizes, bool isInBundle = true)
    {
        TensorPtr inA = std::make_shared<Tensor>(inAsizes.size(), inAsizes.data(), syn_type_float);
        TensorPtr inB = std::make_shared<Tensor>(inBSizes.size(), inBSizes.data(), syn_type_float);

        // Init output shape according to inputs.
        const auto numDimsInOutTensor = std::max(inAsizes.size(), inBSizes.size());
        SizeVector outSizes(numDimsInOutTensor);
        for (auto i = 0; i < numDimsInOutTensor; i++)
        {
            if ((i < inAsizes.size()) && (i < inBSizes.size()))
            {
                // Both dims in A and B should be equal, or one is broadcasted.
                EXPECT_TRUE((inAsizes[i] == inBSizes[i]) || (inAsizes[i] == 1) || (inBSizes[i] == 1));
                outSizes[i] = std::max(inAsizes[i], inBSizes[i]);
            }
            else
            {
                outSizes[i] = (inAsizes.size() > inBSizes.size()) ? inAsizes[i] : inBSizes[i];
            }
        }
        TensorPtr out = std::make_shared<Tensor>(outSizes.size(), outSizes.data(), syn_type_float);

        pNode node = NodeFactory::createGenericTPCNode({inA, inB}, {out}, nullptr, "add_fwd_f32", "ADD");
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addReshapeNode(const SizeVector& inSizes,
                           const SizeVector& outSizes,
                           const TensorPtr&  input      = nullptr,
                           const TensorPtr&  output     = nullptr,
                           bool              isInBundle = true)
    {
        TensorVector inputs;
        inputs.push_back(input ? input : std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float));
        inputs.push_back(std::make_shared<Tensor>(outSizes.size(),
                                                  outSizes.data(),
                                                  syn_type_uint32,
                                                  nullptr,
                                                  nullptr,
                                                  false,
                                                  false,
                                                  INVALID_BATCH_POS,
                                                  nullptr,
                                                  SHAPE_TENSOR));

        TensorVector outputs;
        outputs.push_back(output ? output : std::make_shared<Tensor>(outSizes.size(), outSizes.data(), syn_type_float));

        pNode node = NodeFactory::createNode(inputs, outputs, nullptr, NodeFactory::reshapeNodeTypeName, "RESHAPE");
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addTransposeNode(const SizeVector&      inSizes,
                             const SizeVector&      outSizes,
                             const std::vector<Dim> permutation,
                             const TensorPtr&       input      = nullptr,
                             const TensorPtr&       output     = nullptr,
                             bool                   isInBundle = true)
    {
        TensorVector inputs;
        inputs.push_back(input ? input : std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float));
        TensorVector outputs;
        outputs.push_back(output ? output : std::make_shared<Tensor>(outSizes.size(), outSizes.data(), syn_type_float));

        EXPECT_EQ(inputs[0]->getDim(), outputs[0]->getDim());
        EXPECT_EQ(permutation.size(), outputs[0]->getDim());
        synTransposeParams params;
        params.tensorDim = outputs[0]->getDim();
        for (auto i = 0; i < permutation.size(); i++)
        {
            params.permutation[i] = TransposePermutationDim(permutation[i]);
        }

        pNode node = NodeFactory::createNode(inputs,
                                             outputs,
                                             &params,
                                             sizeof(params),
                                             NodeFactory::transposeNodeTypeName,
                                             "TRANSPOSE");
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addExpandDimsNode(const SizeVector& inSizes,
                              unsigned          expandDim,
                              const TensorPtr&  input      = nullptr,
                              bool              isInBundle = true)
    {
        TensorVector inputs;
        inputs.push_back(input ? input : std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float));

        SizeVector outSizes = inSizes;
        outSizes.insert(outSizes.begin() + expandDim, 1);
        auto output = std::make_shared<Tensor>(outSizes.size(), outSizes.data(), syn_type_float);

        pNode node =
            NodeFactory::createNode(inputs, {output}, &expandDim, NodeFactory::expandDimsNodeTypeName, "EXPAND_DIMS");
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    NodePtr addSqueezeNode(const SizeVector& inSizes, const TensorPtr& input = nullptr, bool isInBundle = true)
    {
        TensorVector inputs;
        inputs.push_back(input ? input : std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float));

        SizeVector outSizes = inSizes;
        // Remove all ones from the output
        outSizes.erase(std::remove(outSizes.begin(), outSizes.end(), 1), outSizes.end());
        auto output = std::make_shared<Tensor>(outSizes.size(), outSizes.data(), syn_type_float);

        pNode node = NodeFactory::createNode(inputs, {output}, nullptr, NodeFactory::squeezeNodeTypeName, "SQUEEZE");
        EXPECT_TRUE(node);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, node));
        if (isInBundle)
        {
            m_bundleNodes.push_back(node);
        }
        return node;
    }

    void markPersistentTensors()
    {
        for (auto& tensor : m_graph.getTensors())
        {
            const auto& producer  = m_graph.getTensorProducer(tensor);
            const auto  numConsumers = m_graph.getNumberOfTensorConsumers(tensor);
            // Mark graph inputs and outputs as persistent
            if (!producer || numConsumers == 0)
            {
                synMemoryDescriptor memDesc(true);
                tensor->setMemoryDescriptor(memDesc);
            }
        }
    }

    void test(const TensorPtr&                     slicedTensor,
              Dim                                  slicedTensorDim,
              uint64_t                             multiplier,
              BundleEngine                         bundleEngine,
              const std::unordered_set<TensorPtr>& bundleInputs,
              const std::unordered_set<TensorPtr>& bundleOutputs,
              const TensorVector&                  reductionTensors)
    {
        m_bundleInputs     = bundleInputs;
        m_bundleOutputs    = bundleOutputs;
        m_reductionTensors = reductionTensors;

        ASSERT_TRUE(validateNodesLayout(m_graph));
        ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));

        createBundleViews();
        StrategyPtr slicingStrategy = createStrategy(slicedTensor, slicedTensorDim, multiplier);

        SlicedBundleGraphGenerator slicedGraphGenerator(m_graph,
                                                        m_bundleIdx,
                                                        m_bundleNodes,
                                                        m_bundleViews,
                                                        slicingStrategy);
        HabanaGraphPtr       slicedGraph = slicedGraphGenerator.createSlicedGraph();
        validateSlicedGraph(slicedGraph, multiplier, bundleEngine);
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

    void createBundleViews()
    {
        TensorSet bundleTensorsSet;
        NodeSet   bundleNodesSet(m_bundleNodes.begin(), m_bundleNodes.end());
        for (const auto& n : m_bundleNodes)
        {
            for (const auto& nodeOperand : n->getOperands())
            {
                if (nodeOperand)
                {
                    bundleTensorsSet.emplace(nodeOperand);
                }
            }
        }
        const auto& [granularityPerTensor, granularityPerNode] =
            CommonTileSizeCalculator::getMinCommonTilesSizes(bundleNodesSet, bundleTensorsSet, m_graph);

        BundleViewsCollector bundleViewsCollector(m_bundleNodes);
        m_bundleViews = bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
    }

    StrategyPtr createStrategy(const TensorPtr& slicedTensor, Dim slicedTensorDim, uint64_t multiplier)
    {
        StrategyPtr slicingStrategy = std::make_shared<Strategy>();

        m_slicedBVD = m_bundleViews->getBVDForTensorDim(slicedTensor, slicedTensorDim);
        for (BundleViewId bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); bvdId++)
        {
            if (bvdId == m_slicedBVD)
            {
                slicingStrategy->setBVDMultiplier(bvdId, BVDMultiplier(multiplier));
            }
            else
            {
                slicingStrategy->setBVDMultiplier(bvdId, BVDMultiplier());
            }
        }

        return slicingStrategy;
    }

    void validateSlicedGraph(const HabanaGraphPtr& slicedGraph, uint64_t multiplier, BundleEngine bundleEngine)
    {
        ASSERT_TRUE(slicedGraph);

        auto numOfSlices = div_round_up(m_bundleViews->getBundleView(m_slicedBVD).resolution, multiplier);

        validateNumSlices(slicedGraph, numOfSlices);
        validateConnectivity(slicedGraph);
        validateForkAndJoinsForBPTs(slicedGraph, numOfSlices);
        validateReductionNodes(slicedGraph, numOfSlices);
        validateBundleInfo(slicedGraph, bundleEngine);
    }

    bool isNodeExpectedToBeSliced(const NodePtr& node)
    {
        const auto& slicedBVD = m_bundleViews->getBundleView(m_slicedBVD);
        for (const auto& nodeDimGranularity : slicedBVD.nodeDimsGranularity)
        {
            if (nodeDimGranularity.first.first == node)
            {
                return true;  // Node is mapped to sliced BVD
            }
        }
        return false;
    }

    bool isTensorExpectedToBeSliced(const TensorPtr& tensor)
    {
        const auto& slicedBVD = m_bundleViews->getBundleView(m_slicedBVD);
        for (const auto& tensorDimGranularity : slicedBVD.tensorDimsGranularity)
        {
            if (tensorDimGranularity.first.first == tensor)
            {
                return true;  // Tensor is mapped to sliced BVD
            }
        }
        return false;
    }

    void validateNumSlices(const HabanaGraphPtr& slicedGraph, uint64_t numOfSlices)
    {
        for (const auto& origNode : m_bundleNodes)
        {
            const NodeVector& slicedNodes = getSlicedNodes(slicedGraph, origNode);
            ASSERT_EQ(slicedNodes.size(), isNodeExpectedToBeSliced(origNode) ? numOfSlices : 1);
            for (const auto& origTensor : origNode->getOperands())
            {
                const TensorVector& slicedTensors = getSlicedTensors(slicedGraph, origTensor);
                if (m_bundleViews->getBundleView(m_slicedBVD).resolution == 1)  // BVD is not sliced
                {
                    ASSERT_EQ(slicedTensors.size(), 1);
                    ASSERT_EQ(slicedTensors.front()->getAllSizesInElements(), origTensor->getAllSizesInElements());
                }
                else if (isTensorExpectedToBeSliced(origTensor))
                {
                    ASSERT_EQ(slicedTensors.size(), numOfSlices);
                    TSize              totalSlicedDimSize = 0;
                    std::optional<Dim> slicedTensorDim;
                    for (const auto& slicedTensor : slicedTensors)
                    {
                        ASSERT_EQ(slicedTensor->getDim(), origTensor->getDim());
                        for (Dim tensorDim = 0; tensorDim < slicedTensor->getDim(); tensorDim++)
                        {
                            if (m_bundleViews->getBVDForTensorDim(origTensor, tensorDim) == m_slicedBVD)
                            {
                                totalSlicedDimSize += slicedTensor->getSizeInElements(tensorDim);
                                if (slicedTensorDim.has_value())
                                {
                                    ASSERT_EQ(slicedTensorDim.value(), tensorDim);
                                }
                                slicedTensorDim = tensorDim;
                            }
                            else
                            {
                                ASSERT_EQ(slicedTensor->getSizeInElements(tensorDim),
                                          origTensor->getSizeInElements(tensorDim));
                            }
                        }
                    }
                    ASSERT_TRUE(slicedTensorDim.has_value());
                    ASSERT_EQ(totalSlicedDimSize, origTensor->getSizeInElements(slicedTensorDim.value()));
                }
                else
                {
                    ASSERT_EQ(slicedTensors.front()->getAllSizesInElements(), origTensor->getAllSizesInElements());
                    // In case of reduction - both inputs and output tensors of the reduction node are marked as slices
                    // of the original reduction tensor.
                    bool isReductionTensor =
                        (std::find(m_reductionTensors.begin(), m_reductionTensors.end(), origTensor) !=
                         m_reductionTensors.end());
                    ASSERT_EQ(slicedTensors.size(), isReductionTensor ? (1 + numOfSlices) : 1);
                }
            }
        }
    }

    void validateConnectivity(const HabanaGraphPtr& slicedGraph)
    {
        ASSERT_TRUE(slicedGraph->isConnectedGraph());

        for (const auto& node : m_bundleNodes)
        {
            for (const auto& slicedNode : getSlicedNodes(slicedGraph, node))
            {
                ASSERT_EQ(slicedNode->getNumInputs(), node->getNumInputs());
                for (auto i = 0; i < slicedNode->getNumInputs(); i++)
                {
                    ASSERT_EQ(slicedNode->getInput(i)->getTensorAnnotation().origBigTensor, node->getInput(i));
                }

                ASSERT_EQ(slicedNode->getNumOutputs(), node->getNumOutputs());
                for (auto i = 0; i < slicedNode->getNumOutputs(); i++)
                {
                    ASSERT_EQ(slicedNode->getOutput(i)->getTensorAnnotation().origBigTensor, node->getOutput(i));
                }
            }
        }
    }

    void validateForkAndJoinsForBPTs(const HabanaGraphPtr& slicedGraph, uint64_t numOfSlices)
    {
        unsigned numOfTensorViewNodes = 0;
        for (const auto& slicedNode : slicedGraph->getNodes())
        {
            if (slicedNode->getNodeType() == Node::TYPE_TENSOR_VIEW)
            {
                numOfTensorViewNodes++;
            }
        }
        ASSERT_EQ(m_bundleInputs.size() + m_bundleOutputs.size(), numOfTensorViewNodes);

        for (const auto& bundleInput : m_bundleInputs)
        {
            ASSERT_EQ(slicedGraph->getNumberOfTensorConsumers(bundleInput), 1);
            NodePtr inputConsumer = slicedGraph->getTensorConsumers(bundleInput).front();
            ASSERT_EQ(inputConsumer->getNodeType(), Node::TYPE_TENSOR_VIEW);
            ASSERT_EQ(inputConsumer->getNumInputs(), 1);
            ASSERT_EQ(inputConsumer->getNumOutputs(), isTensorExpectedToBeSliced(bundleInput) ? numOfSlices : 1);
        }

        for (const auto& bundleOutput : m_bundleOutputs)
        {
            ASSERT_EQ(slicedGraph->getNumberOfTensorProducers(bundleOutput), 1);
            NodePtr outputProducer = slicedGraph->getTensorProducer(bundleOutput);
            ASSERT_EQ(outputProducer->getNodeType(), Node::TYPE_TENSOR_VIEW);
            ASSERT_EQ(outputProducer->getNumOutputs(), 1);
            ASSERT_EQ(outputProducer->getNumInputs(), isTensorExpectedToBeSliced(bundleOutput) ? numOfSlices : 1);
        }

        ASSERT_TRUE(slicedGraph->getLayeredBrainData());
    }

    NodeVector getSlicedNodes(const HabanaGraphPtr& slicedGraph, const NodePtr& origNode) const
    {
        NodeVector slicedNodes;
        for (const auto& node : slicedGraph->getNodes())
        {
            if (node->getNodeAnnotation().origBigNode == origNode)
            {
                slicedNodes.push_back(node);
            }
        }
        return slicedNodes;
    }

    TensorVector getSlicedTensors(const HabanaGraphPtr& slicedGraph, const TensorPtr& origTensor) const
    {
        TensorVector slicedTensors;
        for (const auto& tensor : slicedGraph->getTensors())
        {
            if (tensor->getTensorAnnotation().origBigTensor == origTensor)
            {
                slicedTensors.push_back(tensor);
            }
        }
        return slicedTensors;
    }

    void validateReductionNodes(const HabanaGraphPtr& slicedGraph, uint64_t numOfSlices)
    {
        NodeVector reductionNodes;
        for (const auto& slicedNode : slicedGraph->getNodes())
        {
            if (slicedNode->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
            {
                ASSERT_EQ(slicedNode->getNumInputs(), numOfSlices);
                ASSERT_EQ(slicedNode->getNumOutputs(), 1);
                reductionNodes.push_back(slicedNode);
            }
        }
        if (m_bundleViews->getBundleView(m_slicedBVD).resolution == 1)
        {
            ASSERT_EQ(reductionNodes.size(), 0);  // BVD is not sliced - no reduction nodes should be added
        }
        else
        {
            ASSERT_EQ(reductionNodes.size(), m_reductionTensors.size());
            for (const auto& reductionTensor : m_reductionTensors)
            {
                auto iter = std::find_if(reductionNodes.begin(), reductionNodes.end(), [&](const NodePtr& node) {
                    return (node->getOutput(0)->getTensorAnnotation().origBigTensor == reductionTensor);
                });
                ASSERT_TRUE(iter != reductionNodes.end());
                for (const auto& tensor : (*iter)->getOperands())
                {
                    ASSERT_EQ(tensor->getTensorAnnotation().origBigTensor, reductionTensor);
                    ASSERT_EQ(tensor->getAllSizesInElements(), reductionTensor->getAllSizesInElements());
                }
            }
        }
    }

    void validateBundleInfo(const HabanaGraphPtr& slicedGraph, BundleEngine bundleEngine)
    {
        for (const auto& slicedNode : slicedGraph->getNodes())
        {
            const auto& bundleInfo = slicedNode->getNodeAnnotation().bundleInfo;
            ASSERT_TRUE(bundleInfo.is_set());
            ASSERT_EQ(bundleInfo->bundleIndex, m_bundleIdx);
            ASSERT_EQ(bundleInfo->bundleEngine, bundleEngine);
        }
    }

    Gaudi2Graph m_graph;
    unsigned    m_nodeId = 0;

    NodeVector m_bundleNodes;
    BundleIdx  m_bundleIdx = 17;

    BundleViewContainerPtr m_bundleViews;
    BundleViewId           m_slicedBVD;

    std::unordered_set<TensorPtr> m_bundleInputs;
    std::unordered_set<TensorPtr> m_bundleOutputs;
    TensorVector                  m_reductionTensors;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

class SlicedBundleGraphGeneratorSingleGemmTest
: public SlicedBundleGraphGeneratorTest
, public ::testing::WithParamInterface<std::tuple<bool,     // transpose A
                                                  bool,     // transpose B
                                                  TSize,    // height A
                                                  TSize,    // common-dim
                                                  TSize,    // width B
                                                  uint64_t  // multiplier
                                                  >>
{
protected:
    SlicedBundleGraphGeneratorSingleGemmTest()
    {
        std::tie(m_transposeA, m_transposeB, m_heightA, m_commonDim, m_widthB, m_multiplier) = GetParam();
    }
    void createGraph()
    {
        m_gemm       = addGEMMNode(m_transposeA, m_transposeB, m_heightA, m_commonDim, m_widthB, 0, 0);
        m_inputBPTs  = {m_gemm->getInput(0), m_gemm->getInput(1)};
        m_outputBPTs = {m_gemm->getOutput(0)};
        markPersistentTensors();
    }
    bool                          m_transposeA;
    bool                          m_transposeB;
    TSize                         m_heightA;
    TSize                         m_commonDim;
    TSize                         m_widthB;
    uint64_t                      m_multiplier;
    NodePtr                       m_gemm;
    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;
};

TEST_P(SlicedBundleGraphGeneratorSingleGemmTest, operandA_sliced_on_non_common_dim)
{
    createGraph();
    test(m_gemm->getInput(0),
         m_transposeA ? 0 : 1,
         m_multiplier,
         BundleEngine::ENGINE_MME,
         m_inputBPTs,
         m_outputBPTs,
         {});
}

TEST_P(SlicedBundleGraphGeneratorSingleGemmTest, operandB_sliced_on_non_common_dim)
{
    createGraph();
    test(m_gemm->getInput(1),
         m_transposeB ? 1 : 0,
         m_multiplier,
         BundleEngine::ENGINE_MME,
         m_inputBPTs,
         m_outputBPTs,
         {});
}

TEST_P(SlicedBundleGraphGeneratorSingleGemmTest, slice_on_common_dim)
{
    createGraph();
    test(m_gemm->getInput(0),
         m_transposeA ? 1 : 0,
         m_multiplier,
         BundleEngine::ENGINE_MME,
         m_inputBPTs,
         m_outputBPTs,
         {m_gemm->getOutput(0)});
}

INSTANTIATE_TEST_SUITE_P(single_gemm_test,
                         SlicedBundleGraphGeneratorSingleGemmTest,
                         ::testing::Combine(::testing::Values(false, true),  // transpose A
                                            ::testing::Values(false, true),  // transpose B
                                            ::testing::Values(1024, 317),    // height A
                                            ::testing::Values(1024, 317),    // common-dim
                                            ::testing::Values(1024, 317),    // width B
                                            ::testing::Values(4, 7)));       // multiplier

///////////////////////////////////////////////////////////////////////////////////////////////////////

class SlicedBundleGraphGeneratorSingleBGemmTest
: public SlicedBundleGraphGeneratorTest
, public ::testing::WithParamInterface<std::tuple<bool,     // transpose A
                                                  bool,     // transpose B
                                                  TSize,    // height A
                                                  TSize,    // common-dim
                                                  TSize,    // width B
                                                  TSize,    // batch size
                                                  uint64_t  // multiplier
                                                  >>
{
protected:
    SlicedBundleGraphGeneratorSingleBGemmTest()
    {
        std::tie(m_transposeA, m_transposeB, m_heightA, m_commonDim, m_widthB, m_batch, m_multiplier) = GetParam();
    }
    void createGraph()
    {
        m_bgemm      = addGEMMNode(m_transposeA, m_transposeB, m_heightA, m_commonDim, m_widthB, 2, m_batch);
        m_inputBPTs  = {m_bgemm->getInput(0), m_bgemm->getInput(1)};
        m_outputBPTs = {m_bgemm->getOutput(0)};
        markPersistentTensors();
    }
    bool                          m_transposeA;
    bool                          m_transposeB;
    TSize                         m_heightA;
    TSize                         m_commonDim;
    TSize                         m_widthB;
    TSize                         m_batch;
    uint64_t                      m_multiplier;
    NodePtr                       m_bgemm;
    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;
};

TEST_P(SlicedBundleGraphGeneratorSingleBGemmTest, slice_on_batch_dim)
{
    createGraph();
    test(m_bgemm->getInput(0),
         m_bgemm->getInput(0)->getDim() - 1,
         m_multiplier,
         BundleEngine::ENGINE_MME,
         m_inputBPTs,
         m_outputBPTs,
         {});
}

INSTANTIATE_TEST_SUITE_P(single_bgemm_test,
                         SlicedBundleGraphGeneratorSingleBGemmTest,
                         ::testing::Combine(::testing::Values(false),   // transpose A
                                            ::testing::Values(true),    // transpose B
                                            ::testing::Values(1021),    // height A
                                            ::testing::Values(1021),    // common-dim
                                            ::testing::Values(1021),    // width B
                                            ::testing::Values(24, 19),  // batch size
                                            ::testing::Values(1, 8)));  // multiplier

///////////////////////////////////////////////////////////////////////////////////////////////////////

class SlicedBundleGraphGeneratorGemmWithTPCTest
: public SlicedBundleGraphGeneratorTest
, public ::testing::WithParamInterface<std::tuple<bool,     // transpose A
                                                  bool,     // transpose B
                                                  TSize,    // height A
                                                  TSize,    // common-dim
                                                  TSize,    // width B
                                                  uint64_t  // multiplier
                                                  >>
{
protected:
    SlicedBundleGraphGeneratorGemmWithTPCTest()
    {
        std::tie(m_transposeA, m_transposeB, m_heightA, m_commonDim, m_widthB, m_multiplier) = GetParam();
    }
    void createGraph()
    {
        // GEMM node with producers chain for each input and consumer chain for the output.

        m_gemm = addGEMMNode(m_transposeA, m_transposeB, m_heightA, m_commonDim, m_widthB, 0, 0);

        auto tpc1ProducerA =
            addTPCNode(2, m_gemm->getInput(0)->getAllSizesInElements(), 2, 0, false, nullptr, m_gemm->getInput(0));
        auto tpc2ProducerA = addTPCNode(2,
                                        m_gemm->getInput(0)->getAllSizesInElements(),
                                        2,
                                        0,
                                        false,
                                        nullptr,
                                        tpc1ProducerA->getInput(0));

        auto tpc1ProducerB =
            addTPCNode(2, m_gemm->getInput(1)->getAllSizesInElements(), 2, 0, false, nullptr, m_gemm->getInput(1));
        auto tpc2ProducerB = addTPCNode(2,
                                        m_gemm->getInput(1)->getAllSizesInElements(),
                                        2,
                                        0,
                                        false,
                                        nullptr,
                                        tpc1ProducerB->getInput(0));

        auto tpc1Consumer =
            addTPCNode(2, m_gemm->getOutput(0)->getAllSizesInElements(), 2, 0, false, m_gemm->getOutput(0), nullptr);
        auto tpc2Consumer = addTPCNode(2,
                                       m_gemm->getOutput(0)->getAllSizesInElements(),
                                       2,
                                       0,
                                       false,
                                       tpc1Consumer->getOutput(0),
                                       nullptr);

        m_inputBPTs  = {tpc2ProducerA->getInput(0), tpc2ProducerB->getInput(0)};
        m_outputBPTs = {tpc2Consumer->getOutput(0)};
        markPersistentTensors();
    }
    bool                          m_transposeA;
    bool                          m_transposeB;
    TSize                         m_heightA;
    TSize                         m_commonDim;
    TSize                         m_widthB;
    uint64_t                      m_multiplier;
    NodePtr                       m_gemm;
    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;
};

TEST_P(SlicedBundleGraphGeneratorGemmWithTPCTest, operandA_sliced_on_non_common_dim)
{
    createGraph();
    test(m_gemm->getInput(0),
         m_transposeA ? 0 : 1,
         m_multiplier,
         BundleEngine::ENGINE_MME_TPC,
         m_inputBPTs,
         m_outputBPTs,
         {});
}

TEST_P(SlicedBundleGraphGeneratorGemmWithTPCTest, slice_on_common_dim)
{
    createGraph();
    test(m_gemm->getInput(0),
         m_transposeA ? 1 : 0,
         m_multiplier,
         BundleEngine::ENGINE_MME_TPC,
         m_inputBPTs,
         m_outputBPTs,
         {m_gemm->getOutput(0)});
}

INSTANTIATE_TEST_SUITE_P(gemm_with_tpc_test,
                         SlicedBundleGraphGeneratorGemmWithTPCTest,
                         ::testing::Combine(::testing::Values(false, true),  // transpose A
                                            ::testing::Values(false, true),  // transpose B
                                            ::testing::Values(1024, 614),    // height A
                                            ::testing::Values(1024, 614),    // common-dim size
                                            ::testing::Values(1024),         // width B
                                            ::testing::Values(4, 7)));       // multiplier

///////////////////////////////////////////////////////////////////////////////////////////////////////

class SlicedBundleGraphGeneratorSharedInputGemmsWithTPCTest
: public SlicedBundleGraphGeneratorTest
, public ::testing::WithParamInterface<std::tuple<bool,     // producer to shared input
                                                  bool,     // consumer
                                                  TSize,    // dim size
                                                  uint64_t  // multiplier
                                                  >>
{
protected:
    SlicedBundleGraphGeneratorSharedInputGemmsWithTPCTest()
    {
        std::tie(m_producerToSharedInput, m_consumer, m_dimSize, m_multiplier) = GetParam();
    }
    void createGraph()
    {
        // 2 GEMM nodes sharing an input, with producer/consumer for each of the operands.

        std::vector<TSize> shape = {m_dimSize, m_dimSize};
        m_sharedIn               = std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);

        m_gemm1 = addGEMMNode(false, false, m_dimSize, m_dimSize, m_dimSize, 0, 0, m_sharedIn);
        m_gemm2 = addGEMMNode(false, false, m_dimSize, m_dimSize, m_dimSize, 0, 0, m_sharedIn);

        NodePtr sharedProducerA;
        if (m_producerToSharedInput)
        {
            sharedProducerA = addTPCNode(2, m_sharedIn->getAllSizesInElements(), 2, 0, false, nullptr, m_sharedIn);
        }

        auto tpc1ProducerB =
            addTPCNode(2, m_gemm1->getInput(1)->getAllSizesInElements(), 2, 0, false, nullptr, m_gemm1->getInput(1));
        auto tpc2ProducerB =
            addTPCNode(2, m_gemm2->getInput(1)->getAllSizesInElements(), 2, 0, false, nullptr, m_gemm2->getInput(1));

        NodePtr tpc1Consumer;
        NodePtr tpc2Consumer;
        if (m_consumer)
        {
            tpc1Consumer = addTPCNode(2,
                                      m_gemm1->getOutput(0)->getAllSizesInElements(),
                                      2,
                                      0,
                                      false,
                                      m_gemm1->getOutput(0),
                                      nullptr);
            tpc2Consumer = addTPCNode(2,
                                      m_gemm2->getOutput(0)->getAllSizesInElements(),
                                      2,
                                      0,
                                      false,
                                      m_gemm2->getOutput(0),
                                      nullptr);
        }

        m_inputBPTs  = {tpc1ProducerB->getInput(0),
                       tpc2ProducerB->getInput(0),
                       m_producerToSharedInput ? sharedProducerA->getInput(0) : m_sharedIn};
        m_outputBPTs = {tpc1Consumer ? tpc1Consumer->getOutput(0) : m_gemm1->getOutput(0),
                        tpc2Consumer ? tpc2Consumer->getOutput(0) : m_gemm2->getOutput(0)};
        markPersistentTensors();
    }
    bool                          m_producerToSharedInput;
    bool                          m_consumer;
    TSize                         m_dimSize;
    uint64_t                      m_multiplier;
    TensorPtr                     m_sharedIn;
    NodePtr                       m_gemm1;
    NodePtr                       m_gemm2;
    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;
};

TEST_P(SlicedBundleGraphGeneratorSharedInputGemmsWithTPCTest, shared_input_sliced_on_non_common_dim)
{
    createGraph();
    test(m_sharedIn, 1, m_multiplier, BundleEngine::ENGINE_MME_TPC, m_inputBPTs, m_outputBPTs, {});
}

TEST_P(SlicedBundleGraphGeneratorSharedInputGemmsWithTPCTest, shared_input_sliced_on_common_dim)
{
    createGraph();
    test(m_sharedIn,
         0,
         m_multiplier,
         BundleEngine::ENGINE_MME_TPC,
         m_inputBPTs,
         m_outputBPTs,
         {m_gemm1->getOutput(0), m_gemm2->getOutput(0)});
}

INSTANTIATE_TEST_SUITE_P(gemms_with_shared_input_and_tpc_test,
                         SlicedBundleGraphGeneratorSharedInputGemmsWithTPCTest,
                         ::testing::Combine(::testing::Values(false, true),  // shared input producer
                                            ::testing::Values(false, true),  // consumer
                                            ::testing::Values(1024),         // dim size
                                            ::testing::Values(21)));         // multiplier

///////////////////////////////////////////////////////////////////////////////////////////////////////

class SlicedBundleGraphGeneratorGemmWithTPCBroadcastTest
: public SlicedBundleGraphGeneratorTest
, public ::testing::WithParamInterface<std::tuple<SizeVector,  // TPC input A sizes
                                                  SizeVector,  // TPC input B sizes
                                                  unsigned,    // sliced dim
                                                  uint64_t,    // multiplier
                                                  bool         // true - for TPC producer, false - for TPC consumer
                                                  >>
{
protected:
    SlicedBundleGraphGeneratorGemmWithTPCBroadcastTest()
    {
        std::tie(m_tpcInputASizes, m_tpcInputBSizes, m_slicedDim, m_multiplier, m_isProducer) = GetParam();
    }
    void createGraph()
    {
        m_tpc                          = addTPCNodeWithBroadcast(m_tpcInputASizes, m_tpcInputBSizes);
        const TSize gemmMissingDimSize = 998;
        if (m_isProducer)
        {
            // GEMM node with TPC producer that one of its inputs is broadcasted.
            m_gemm       = addGEMMNode(false,
                                 false,
                                 m_tpc->getOutput(0)->getSizeInElements(1),
                                 m_tpc->getOutput(0)->getSizeInElements(0),
                                 gemmMissingDimSize,
                                 0,
                                 0,
                                 m_tpc->getOutput(0),
                                 nullptr,
                                 nullptr);
            m_inputBPTs  = {m_gemm->getInput(1), m_tpc->getInput(0), m_tpc->getInput(1)};
            m_outputBPTs = {m_gemm->getOutput(0)};
        }
        else
        {
            // GEMM node with TPC consumer that one of its inputs is broadcasted.
            m_gemm       = addGEMMNode(false,
                                 false,
                                 m_tpc->getInput(0)->getSizeInElements(1),
                                 gemmMissingDimSize,
                                 m_tpc->getInput(0)->getSizeInElements(0),
                                 0,
                                 0,
                                 nullptr,
                                 nullptr,
                                 m_tpc->getInput(0));
            m_inputBPTs  = {m_gemm->getInput(0), m_gemm->getInput(1), m_tpc->getInput(1)};
            m_outputBPTs = {m_tpc->getOutput(0)};
        }

        markPersistentTensors();
    }

    SizeVector m_tpcInputASizes;
    SizeVector m_tpcInputBSizes;
    unsigned   m_slicedDim;
    uint64_t   m_multiplier;
    bool       m_isProducer;

    NodePtr                       m_tpc;
    NodePtr                       m_gemm;
    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;
};

TEST_P(SlicedBundleGraphGeneratorGemmWithTPCBroadcastTest, gemm_with_tpc_broadcast)
{
    createGraph();

    TensorVector reductionTensors;
    if (m_isProducer && (m_slicedDim == 0))  // GEMM will be sliced on common-dim
    {
        reductionTensors.push_back(m_gemm->getOutput(0));
    }
    test(m_tpc->getInput(0),
         m_slicedDim,
         m_multiplier,
         BundleEngine::ENGINE_MME_TPC,
         m_inputBPTs,
         m_outputBPTs,
         reductionTensors);
}

INSTANTIATE_TEST_SUITE_P(gemm_with_tpc_broadcast_test,
                         SlicedBundleGraphGeneratorGemmWithTPCBroadcastTest,
                         ::testing::Combine(::testing::Values(SizeVector {1024, 876},  // TPC input A sizes
                                                              SizeVector {1, 876},
                                                              SizeVector {1024, 1}),
                                            ::testing::Values(SizeVector {1024, 876},  // TPC input B sizes
                                                              SizeVector {1024},
                                                              SizeVector {1024, 1},
                                                              SizeVector {1},
                                                              SizeVector {1, 876},
                                                              SizeVector {1, 1}),
                                            ::testing::Values(0, 1),           // sliced dim
                                            ::testing::Values(2),              // multiplier
                                            ::testing::Values(false, true)));  // producer / consumer

///////////////////////////////////////////////////////////////////////////////////////////////////////

class SlicedBundleGraphGeneratorBroadcastBGemmTest
: public SlicedBundleGraphGeneratorTest
, public ::testing::WithParamInterface<std::tuple<bool,      // transpose A
                                                  bool,      // transpose B
                                                  TSize,     // height A
                                                  TSize,     // common-dim
                                                  TSize,     // width B
                                                  TSize,     // batch size
                                                  unsigned,  // num batch dims
                                                  bool,      // true - operand A is broadcasted, false - operand B is
                                                             // broadcasted
                                                  bool,      // is broadcast dim present
                                                  Dim,       // sliced dim
                                                  uint64_t,  // multiplier
                                                  bool       // add TPC producers/consumer
                                                  >>
{
protected:
    SlicedBundleGraphGeneratorBroadcastBGemmTest()
    {
        std::tie(m_transposeA,
                 m_transposeB,
                 m_heightA,
                 m_commonDim,
                 m_widthB,
                 m_batchSize,
                 m_numBatchDims,
                 m_isOpABroadcasted,
                 m_isBroadcastDimPresent,
                 m_slicedDim,
                 m_multiplier,
                 m_addTpcExpansions) = GetParam();
    }
    void createGraph()
    {
        m_bgemm = addBroadcastBGEMMNode(m_transposeA,
                                        m_transposeB,
                                        m_heightA,
                                        m_commonDim,
                                        m_widthB,
                                        m_numBatchDims,
                                        m_batchSize,
                                        m_isOpABroadcasted,
                                        m_isBroadcastDimPresent);

        if (m_addTpcExpansions)
        {
            auto tpcProducerA = addTPCNode(m_bgemm->getInput(0)->getDim(),
                                           m_bgemm->getInput(0)->getAllSizesInElements(),
                                           2,
                                           0,
                                           false,
                                           nullptr,
                                           m_bgemm->getInput(0));
            auto tpcProducerB = addTPCNode(m_bgemm->getInput(1)->getDim(),
                                           m_bgemm->getInput(1)->getAllSizesInElements(),
                                           2,
                                           0,
                                           false,
                                           nullptr,
                                           m_bgemm->getInput(1));
            auto tpcConsumer  = addTPCNode(m_bgemm->getOutput(0)->getDim(),
                                          m_bgemm->getOutput(0)->getAllSizesInElements(),
                                          2,
                                          0,
                                          false,
                                          m_bgemm->getOutput(0));

            m_inputBPTs  = {tpcProducerA->getInput(0), tpcProducerB->getInput(0)};
            m_outputBPTs = {tpcConsumer->getOutput(0)};
        }
        else
        {
            m_inputBPTs  = {m_bgemm->getInput(0), m_bgemm->getInput(1)};
            m_outputBPTs = {m_bgemm->getOutput(0)};
        }

        markPersistentTensors();
    }

    bool     m_transposeA;
    bool     m_transposeB;
    TSize    m_heightA;
    TSize    m_commonDim;
    TSize    m_widthB;
    TSize    m_batchSize;
    unsigned m_numBatchDims;
    bool     m_isOpABroadcasted;
    bool     m_isBroadcastDimPresent;
    Dim      m_slicedDim;
    uint64_t m_multiplier;
    bool     m_addTpcExpansions;

    NodePtr                       m_bgemm;
    std::unordered_set<TensorPtr> m_inputBPTs;
    std::unordered_set<TensorPtr> m_outputBPTs;
};

TEST_P(SlicedBundleGraphGeneratorBroadcastBGemmTest, broadcast_bgemm)
{
    auto numDimsInSlicedOpA = 2 + ((m_isOpABroadcasted && !m_isBroadcastDimPresent) ? 0 : m_numBatchDims);
    if (m_slicedDim >= numDimsInSlicedOpA)
    {
        return;  // Skip invalid test params
    }

    createGraph();

    TensorVector reductionTensors;
    if (((m_slicedDim == 0) && !m_transposeA) ||
        ((m_slicedDim == 1) && m_transposeA))  // BGEMM will be sliced on common-dim
    {
        reductionTensors.push_back(m_bgemm->getOutput(0));
    }
    test(m_bgemm->getInput(0),
         m_slicedDim,
         m_multiplier,
         m_addTpcExpansions ? BundleEngine::ENGINE_MME_TPC : BundleEngine::ENGINE_MME,
         m_inputBPTs,
         m_outputBPTs,
         reductionTensors);
}

INSTANTIATE_TEST_SUITE_P(
    broadcast_bgemm_test,
    SlicedBundleGraphGeneratorBroadcastBGemmTest,
    ::testing::Combine(::testing::Values(false, true),  // transpose A
                       ::testing::Values(false),        // transpose B
                       ::testing::Values(1021),         // height A
                       ::testing::Values(1021),         // common-dim
                       ::testing::Values(1021),         // width B
                       ::testing::Values(1, 23),        // batch size
                       ::testing::Values(1, 2),         // num batch dims
                       ::testing::Values(true,
                                         false),  // true - operand A is broadcasted, false - operand B is broadcasted
                       ::testing::Values(true, false),    // is broadcast dim present
                       ::testing::Values(0, 1, 2, 3),     // sliced dim
                       ::testing::Values(2),              // multiplier
                       ::testing::Values(true, false)));  // add TPC producers/consumer

///////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(SlicedBundleGraphGeneratorTest, unsliced_graph_multiplier_equal_to_bvd_resolution)
{
    auto gemm = addGEMMNode(false, false, 1024, 1024, 1024, 0, 0);
    auto tpcProducer =
        addTPCNode(2, gemm->getInput(0)->getAllSizesInElements(), 1, 0, false, nullptr, gemm->getInput(0));

    markPersistentTensors();
    test(gemm->getInput(0),
         0,
         1024,  // multiplier is equal to dim size, granularity is 1 => No slicing is expected
         BundleEngine::ENGINE_MME_TPC,
         {tpcProducer->getInput(0), gemm->getInput(1)},
         {gemm->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, unsliced_graph_multiplier_bigger_than_bvd_resolution)
{
    auto gemm = addGEMMNode(false, false, 1024, 1024, 1024, 0, 0);
    auto tpcProducer =
        addTPCNode(2, gemm->getInput(0)->getAllSizesInElements(), 1, 0, false, nullptr, gemm->getInput(0));

    markPersistentTensors();
    test(gemm->getInput(0),
         0,
         2000,  // multiplier is bigger than dim size, granularity is 1 => No slicing is expected
         BundleEngine::ENGINE_MME_TPC,
         {tpcProducer->getInput(0), gemm->getInput(1)},
         {gemm->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, unsliced_graph_all_required_node)
{
    auto gemm = addGEMMNode(false, false, 1024, 1024, 1024, 0, 0);

    // All required TPC producer (granularity equal to dim size) => No slicing is expected
    auto tpcProducer =
        addTPCNode(2, gemm->getInput(0)->getAllSizesInElements(), 1024, 0, false, nullptr, gemm->getInput(0));

    markPersistentTensors();
    test(gemm->getInput(0),
         0,
         1,
         BundleEngine::ENGINE_MME_TPC,
         {tpcProducer->getInput(0), gemm->getInput(1)},
         {gemm->getOutput(0)},
         {});
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_gemm_with_2_consumers_in_bundle)
{
    // GEMM node with 2 consumers in bundle
    auto gemm = addGEMMNode(false, false, 1024, 1024, 1024, 0, 0);
    auto consumer1 =
        addTPCNode(2, gemm->getOutput(0)->getAllSizesInElements(), 2, 0, false, gemm->getOutput(0), nullptr);
    auto consumer2 =
        addTPCNode(2, gemm->getOutput(0)->getAllSizesInElements(), 2, 0, false, gemm->getOutput(0), nullptr);

    markPersistentTensors();
    test(gemm->getInput(0),
         1,
         100,
         BundleEngine::ENGINE_MME_TPC,
         {gemm->getInput(0), gemm->getInput(1)},
         {consumer1->getOutput(0), consumer2->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_with_intermediate_output_bpt)
{
    // GEMM node with one consumer in bundle and one consumer outside the bundle.
    auto gemm = addGEMMNode(false, false, 1024, 1024, 1024, 0, 0);
    auto internalConsumer =
        addTPCNode(2, gemm->getOutput(0)->getAllSizesInElements(), 2, 0, false, gemm->getOutput(0), nullptr);
    auto externalConsumer =
        addTPCNode(2, gemm->getOutput(0)->getAllSizesInElements(), 2, 0, false, gemm->getOutput(0), nullptr, false);

    markPersistentTensors();
    test(gemm->getInput(0),
         1,
         100,
         BundleEngine::ENGINE_MME_TPC,
         {gemm->getInput(0), gemm->getInput(1)},
         {gemm->getOutput(0), internalConsumer->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_with_intermediate_persistent_tensor)
{
    // GEMM node with persistent output and consumer in bundle.
    auto                gemm = addGEMMNode(false, false, 1024, 1024, 1024, 0, 0);
    synMemoryDescriptor memDesc(true);
    gemm->getOutput(0)->setMemoryDescriptor(memDesc);
    auto consumer =
        addTPCNode(2, gemm->getOutput(0)->getAllSizesInElements(), 2, 0, false, gemm->getOutput(0), nullptr);

    markPersistentTensors();
    test(gemm->getInput(0),
         1,
         100,
         BundleEngine::ENGINE_MME_TPC,
         {gemm->getInput(0), gemm->getInput(1)},
         {gemm->getOutput(0), consumer->getOutput(0)},
         {});
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_with_reshape_between_mme_and_tpc)
{
    // TPC -> Reshape -> BGEMM

    constexpr TSize m = 2570, k = 1100, n = 9865, b = 16;
    auto            bgemm   = addGEMMNode(false, false, m, k, n, 1, b);
    auto            reshape = addReshapeNode({k, m * b}, {k, m, b}, nullptr, bgemm->getInput(0));
    auto tpc = addTPCNode(2, reshape->getInput(0)->getAllSizesInElements(), 2, 0, false, nullptr, reshape->getInput(0));

    markPersistentTensors();

    test(bgemm->getInput(0),
         2,  // Batch dim is sliced
         4,
         BundleEngine::ENGINE_MME_TPC,
         {tpc->getInput(0), bgemm->getInput(1)},
         {bgemm->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_with_transpose_between_mme_and_tpc)
{
    // TPC -> Transpose -> BGEMM

    constexpr unsigned m = 2570, k = 1100, n = 9865, b = 16;
    auto               bgemm     = addGEMMNode(false, false, m, k, n, 1, b);
    auto               transpose = addTransposeNode({k, b, m}, {k, m, b}, {0, 2, 1}, nullptr, bgemm->getInput(0));
    auto               tpc =
        addTPCNode(3, transpose->getInput(0)->getAllSizesInElements(), 2, 0, false, nullptr, transpose->getInput(0));

    markPersistentTensors();

    test(bgemm->getInput(0),
         2,  // Batch dim is sliced
         2,
         BundleEngine::ENGINE_MME_TPC,
         {tpc->getInput(0), bgemm->getInput(1)},
         {bgemm->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_with_expand_dims_between_mme_and_tpc)
{
    // BGEMM -> Expand-dims -> TPC

    constexpr unsigned m = 2570, k = 1100, n = 9865, b = 16;
    auto               bgemm      = addGEMMNode(false, false, m, k, n, 1, b);
    auto               expandDims = addExpandDimsNode({n, m, b}, 2, bgemm->getOutput(0));
    auto tpc = addTPCNode(4, expandDims->getOutput(0)->getAllSizesInElements(), 2, 0, false, expandDims->getOutput(0));

    markPersistentTensors();

    test(bgemm->getInput(0),
         2,  // Batch dim is sliced
         2,
         BundleEngine::ENGINE_MME_TPC,
         {bgemm->getInput(0), bgemm->getInput(1)},
         {tpc->getOutput(0)},
         {});
}

TEST_F(SlicedBundleGraphGeneratorTest, sliced_graph_with_squeeze_between_mme_and_tpc)
{
    // TPC -> Squeeze -> BGEMM

    constexpr unsigned m = 2570, k = 1100, n = 9865, b = 16;

    auto tpc     = addTPCNode(4, {k, m, 1, b}, 2, 0, false);
    auto squeeze = addSqueezeNode({k, m, 1, b}, tpc->getOutput(0));
    auto bgemm   = addGEMMNode(false, false, m, k, n, 1, b, squeeze->getOutput(0));

    markPersistentTensors();

    test(bgemm->getInput(0),
         2,  // Batch dim is sliced
         2,
         BundleEngine::ENGINE_MME_TPC,
         {tpc->getInput(0), bgemm->getInput(1)},
         {bgemm->getOutput(0)},
         {});
}