#include "slicer/bundle_views_collector.h"
#include "common_tile_size_calculator.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "slicer/sliced_bundle_graph_generator.h"
#include "synapse_common_types.h"
#include "tpc_slicing_test_infra.h"
#include "node_factory.h"

using namespace gc::layered_brain;

class SlicedBundleGraphGeneratorMultipleSlicedBVDsTest : public GraphOptimizerTest,
       public ::testing::WithParamInterface<std::tuple<bool,     // requires F32 reduction
                                                       bool,     // requires memset
                                                       bool>>    // slice on CD
{
protected:
    void addNodeToBundle(const NodePtr& node)
    {
        ASSERT_TRUE(node);
        ASSERT_TRUE(GraphEditor::addNode(m_graph, node));
        m_bundleNodes.push_back(node);
    }

    BundleViewContainerPtr createBundleViews() const
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
        return bundleViewsCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
    }

    uint64_t
    getNumSlicesPerBVD(const BundleViewContainerPtr& bundleViews, BundleViewId bvdId, const StrategyPtr& strategy) const
    {
        const auto& multiplier = strategy->getBVDMultiplier(bvdId);
        if (multiplier.isSliced())
        {
            return div_round_up(bundleViews->getBundleView(bvdId).resolution, multiplier.getMultiplier());
        }
        return 1;
    }

    HabanaGraphPtr sliceGraph(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy) const
    {
        SlicedBundleGraphGenerator slicedGraphGenerator(m_graph, 0, m_bundleNodes, bundleViews, strategy);
        return slicedGraphGenerator.createSlicedGraph();
    }

private:
    Gaudi2Graph m_graph;
    NodeVector  m_bundleNodes;
};

TEST_P(SlicedBundleGraphGeneratorMultipleSlicedBVDsTest, test_gemm_with_multiple_sliced_dims)
{
    bool                     requireF32Reduction = std::get<0>(GetParam());
    bool                     requireMemset       = std::get<1>(GetParam());
    bool                     sliceOnCD           = std::get<2>(GetParam());
    const std::vector<TSize> inputAShape = {512, 1024};
    const std::vector<TSize> inputBShape = {2048, 512};
    const std::vector<TSize> outShape    = {2048, 1024};
    const uint64_t           multiplier  = 256;
    const synDataType        origDataType        = syn_type_bf16;
    TensorVector             inputs;
    inputs.push_back(std::make_shared<Tensor>(inputAShape.size(), inputAShape.data(), origDataType));
    inputs.push_back(std::make_shared<Tensor>(inputBShape.size(), inputBShape.data(), origDataType));
    TensorVector outputs;
    outputs.push_back(std::make_shared<Tensor>(outShape.size(), outShape.data(), origDataType));
    synGEMMParams params(false, false);
    NodePtr       gemm = NodeFactory::createNode(inputs, outputs, &params, NodeFactory::gemmNodeTypeName, "GEMM");
    // Mark inputs/output as persistent
    synMemoryDescriptor memDesc(true);
    gemm->getInput(0)->setMemoryDescriptor(memDesc);
    gemm->getInput(1)->setMemoryDescriptor(memDesc);
    gemm->getOutput(0)->setMemoryDescriptor(memDesc);
    addNodeToBundle(gemm);

    const auto& bundleViews = createBundleViews();
    ASSERT_EQ(bundleViews->getNumOfBundleViews(), 3);
    auto mmeSolution            = std::make_shared<MmeSolution>();
    mmeSolution->QORs[gemm]     = std::make_shared<SolutionParams>();
    StrategyPtr slicingStrategy = std::make_shared<Strategy>(mmeSolution);
    BundleViewId cdBVD           = bundleViews->getBVDForTensorDim(gemm->getInput(0), 0);
    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
    {
        slicingStrategy->setBVDMultiplier(bvdId,
                                          ((bvdId == cdBVD) && !sliceOnCD) ? BVDMultiplier()
                                                                           : BVDMultiplier(multiplier));
    }
    // Granularity is 1 in all dims, multiplier set to 256:
    uint64_t numCommonDimSlices = !sliceOnCD ? 1 : (inputAShape[0] / multiplier);  // 512/256
    uint64_t numHeightSlices    = inputAShape[1] / multiplier;  // 1024/256
    uint64_t numWidthSlices     = inputBShape[0] / multiplier;  // 2048/256
    ASSERT_EQ(getNumSlicesPerBVD(bundleViews, bundleViews->getBVDForTensorDim(gemm->getInput(0), 0), slicingStrategy),
              numCommonDimSlices);
    ASSERT_EQ(getNumSlicesPerBVD(bundleViews, bundleViews->getBVDForTensorDim(gemm->getInput(0), 1), slicingStrategy),
              numHeightSlices);
    ASSERT_EQ(getNumSlicesPerBVD(bundleViews, bundleViews->getBVDForTensorDim(gemm->getInput(1), 0), slicingStrategy),
              numWidthSlices);

    if (requireF32Reduction)
    {
        slicingStrategy->getMmeSolution()->QORs.at(gemm)->solutionRequirements.requiresCast = true;
    }
    if (requireMemset)
    {
        slicingStrategy->getMmeSolution()->QORs.at(gemm)->solutionRequirements.requiresMemset = true;
    }

    auto slicedGraph = sliceGraph(bundleViews, slicingStrategy);

    unsigned numOfGEMMs      = 0;
    unsigned numOfForks      = 0;
    unsigned numOfJoins      = 0;
    unsigned numOfReductions = 0;
    unsigned numOfCasts      = 0;
    unsigned numOfMemsets    = 0;
    for (const auto& node : slicedGraph->getNodes())
    {
        switch (node->getNodeType())
        {
            case Node::TYPE_GEMM:
                numOfGEMMs++;
                ASSERT_EQ(node->getNodeAnnotation().origBigNode, gemm);
                ASSERT_EQ(node->getNumInputs(), 2);
                ASSERT_EQ(node->getInput(0)->getTensorAnnotation().origBigTensor, gemm->getInput(0));
                ASSERT_EQ(node->getInput(1)->getTensorAnnotation().origBigTensor, gemm->getInput(1));
                ASSERT_EQ(node->getNumOutputs(), 1);
                ASSERT_EQ(node->getOutput(0)->getTensorAnnotation().origBigTensor, gemm->getOutput(0));
                ASSERT_EQ(node->getOutput(0)->getElementType(), requireF32Reduction ? syn_type_single : origDataType);
                break;
            case Node::TYPE_INTERNAL_REDUCTION:
                numOfReductions++;
                ASSERT_EQ(node->getNumInputs(), numCommonDimSlices + requireMemset);
                ASSERT_EQ(node->getNumOutputs(), 1);
                for (const auto& input : node->getInputs())
                {
                    ASSERT_TRUE(slicedGraph->getTensorProducer(input));
                    ASSERT_TRUE((slicedGraph->getTensorProducer(input)->getNodeType() == Node::TYPE_GEMM) ||
                                (slicedGraph->getTensorProducer(input)->getNodeType() == Node::TYPE_MEMSET));
                }
                break;
            case Node::TYPE_TENSOR_VIEW:
                if (node->getNumOutputs() == 1)
                {
                    numOfJoins++;
                    ASSERT_EQ(node->getOutput(0), gemm->getOutput(0));
                    ASSERT_EQ(node->getNumInputs(), numHeightSlices * numWidthSlices);
                    for (const auto& input : node->getInputs())
                    {
                        ASSERT_TRUE(slicedGraph->getTensorProducer(input));
                        if (requireF32Reduction)
                        {
                            ASSERT_TRUE(slicedGraph->getTensorProducer(input)->isCast());
                        }
                        else
                        {
                            ASSERT_EQ(slicedGraph->getTensorProducer(input)->getNodeType(),
                                      Node::TYPE_INTERNAL_REDUCTION);
                        }
                    }
                }
                else
                {
                    numOfForks++;
                    ASSERT_EQ(node->getNumInputs(), 1);
                    if (node->getInput(0) == gemm->getInput(0))  // Fork for input A
                    {
                        ASSERT_EQ(node->getNumOutputs(), numCommonDimSlices * numHeightSlices);
                        for (const auto& output : node->getOutputs())
                        {
                            ASSERT_EQ(slicedGraph->getNumberOfTensorConsumers(output), numWidthSlices);
                            for (const auto& consumer : slicedGraph->getTensorConsumers(output))
                            {
                                ASSERT_EQ(consumer->getNodeType(), Node::TYPE_GEMM);
                            }
                        }
                    }
                    else  // Fork for input B
                    {
                        ASSERT_EQ(node->getInput(0), gemm->getInput(1));
                        ASSERT_EQ(node->getNumOutputs(), numCommonDimSlices * numWidthSlices);
                        for (const auto& output : node->getOutputs())
                        {
                            ASSERT_EQ(slicedGraph->getNumberOfTensorConsumers(output), numHeightSlices);
                            for (const auto& consumer : slicedGraph->getTensorConsumers(output))
                            {
                                ASSERT_EQ(consumer->getNodeType(), Node::TYPE_GEMM);
                            }
                        }
                    }
                }
                break;
            case Node::TYPE_USER:
                ASSERT_TRUE(node->isCast());
                numOfCasts++;
                ASSERT_EQ(node->getNumInputs(), 1);
                ASSERT_EQ(node->getNumOutputs(), 1);
                ASSERT_EQ(node->getInput(0)->getElementType(), syn_type_single);
                ASSERT_EQ(node->getOutput(0)->getElementType(), origDataType);
                ASSERT_TRUE(slicedGraph->getTensorProducer(node->getInput(0)));
                ASSERT_EQ(slicedGraph->getTensorProducer(node->getInput(0))->getNodeType(),
                          Node::TYPE_INTERNAL_REDUCTION);
                break;
            case Node::TYPE_MEMSET:
                numOfMemsets++;
                ASSERT_EQ(node->getNumInputs(), 0);
                ASSERT_EQ(node->getNumOutputs(), 1);
                ASSERT_EQ(slicedGraph->getNumberOfTensorConsumers(node->getOutput(0)), 1);
                ASSERT_EQ(slicedGraph->getTensorConsumers(node->getOutput(0)).front()->getNodeType(),
                          Node::TYPE_INTERNAL_REDUCTION);
                ASSERT_EQ(node->getOutput(0)->getElementType(), requireF32Reduction ? syn_type_single : origDataType);
                break;
            default:
                FAIL() << "Unexpected node in graph";
        }
    }

    ASSERT_TRUE(slicedGraph->isConnectedGraph());
    ASSERT_EQ(numOfGEMMs, numCommonDimSlices * numHeightSlices * numWidthSlices);
    ASSERT_EQ(numOfForks, 2);
    ASSERT_EQ(numOfJoins, 1);
    if (sliceOnCD)
    {
        ASSERT_EQ(numOfReductions, numHeightSlices * numWidthSlices);
    }
    else
    {
        ASSERT_EQ(numOfReductions, requireMemset ? numOfGEMMs : 0);
    }
    ASSERT_EQ(numOfCasts, requireF32Reduction ? numOfReductions : 0);
    ASSERT_EQ(numOfMemsets, requireMemset ? numOfReductions : 0);
}

INSTANTIATE_TEST_SUITE_P(gemm_with_multiple_sliced_dims_cd_slicing_without_concurrency,
                         SlicedBundleGraphGeneratorMultipleSlicedBVDsTest,
                         ::testing::Combine(::testing::Values(false, true),  // requires F32 reduction
                                            ::testing::Values(false),        // requires memset
                                            ::testing::Values(true)));       // slice on cd

INSTANTIATE_TEST_SUITE_P(gemm_with_multiple_sliced_dims_cd_concurrency,
                         SlicedBundleGraphGeneratorMultipleSlicedBVDsTest,
                         ::testing::Combine(::testing::Values(false, true),    // requires F32 reduction
                                            ::testing::Values(true),           // requires memset
                                            ::testing::Values(false, true)));  // slice on cd

TEST_F(SlicedBundleGraphGeneratorMultipleSlicedBVDsTest, test_tpc_with_multiple_reduced_sliced_dims)
{
    // Create a TPC node with all-required output and 1:1 mapping between input dims and node dims.
    // Validate that a single reduction node is added.
    TPCCustomIndexSpaceMappingNode::Params params;
    params.tensorRank           = 4;
    params.nodeResolutionRank   = 4;
    params.dimMappingForInputs  = {{{0, false}, {1, false}, {2, false}, {3, false}}};
    params.dimMappingForOutputs = {{}};
    NodePtr tpc                 = TPCCustomIndexSpaceMappingNode::create(params);
    // Mark input/output as persistent
    synMemoryDescriptor memDesc(true);
    tpc->getInput(0)->setMemoryDescriptor(memDesc);
    tpc->getOutput(0)->setMemoryDescriptor(memDesc);
    addNodeToBundle(tpc);

    const auto& bundleViews     = createBundleViews();
    StrategyPtr slicingStrategy = std::make_shared<Strategy>();
    // Init all BVDs as UNSLICED.
    for (BundleViewId bvdId = 0; bvdId < bundleViews->getNumOfBundleViews(); bvdId++)
    {
        slicingStrategy->setBVDMultiplier(bvdId, BVDMultiplier());
    }
    // Slice input dims 0 and 3.
    const uint64_t multiplier = 1;
    slicingStrategy->setBVDMultiplier(bundleViews->getBVDForTensorDim(tpc->getInput(0), 0), BVDMultiplier(multiplier));
    slicingStrategy->setBVDMultiplier(bundleViews->getBVDForTensorDim(tpc->getInput(0), 3), BVDMultiplier(multiplier));

    auto slicedGraph = sliceGraph(bundleViews, slicingStrategy);

    auto expectedNumSlices =
        getNumSlicesPerBVD(bundleViews, bundleViews->getBVDForTensorDim(tpc->getInput(0), 0), slicingStrategy) *
        getNumSlicesPerBVD(bundleViews, bundleViews->getBVDForTensorDim(tpc->getInput(0), 3), slicingStrategy);

    unsigned numOfTPCs       = 0;
    unsigned numOfForks      = 0;
    unsigned numOfJoins      = 0;
    unsigned numOfReductions = 0;
    for (const auto& node : slicedGraph->getNodes())
    {
        switch (node->getNodeType())
        {
            case Node::TYPE_USER:
                numOfTPCs++;
                ASSERT_EQ(node->getNodeAnnotation().origBigNode, tpc);
                ASSERT_EQ(node->getNumInputs(), 1);
                ASSERT_EQ(node->getInput(0)->getTensorAnnotation().origBigTensor, tpc->getInput(0));
                ASSERT_EQ(node->getNumOutputs(), 1);
                ASSERT_EQ(node->getOutput(0)->getTensorAnnotation().origBigTensor, tpc->getOutput(0));
                break;
            case Node::TYPE_INTERNAL_REDUCTION:
                numOfReductions++;
                ASSERT_EQ(node->getNumInputs(), expectedNumSlices);
                ASSERT_EQ(node->getNumOutputs(), 1);
                for (const auto& input : node->getInputs())
                {
                    ASSERT_TRUE(slicedGraph->getTensorProducer(input));
                    ASSERT_EQ(slicedGraph->getTensorProducer(input)->getNodeType(), Node::TYPE_USER);
                }
                break;
            case Node::TYPE_TENSOR_VIEW:
                if (node->getNumOutputs() == 1)
                {
                    numOfJoins++;
                    ASSERT_EQ(node->getOutput(0), tpc->getOutput(0));
                    ASSERT_EQ(node->getNumInputs(), 1);
                    ASSERT_TRUE(slicedGraph->getTensorProducer(node->getInput(0)));
                    ASSERT_EQ(slicedGraph->getTensorProducer(node->getInput(0))->getNodeType(),
                              Node::TYPE_INTERNAL_REDUCTION);
                }
                else
                {
                    numOfForks++;
                    ASSERT_EQ(node->getNumInputs(), 1);
                    ASSERT_EQ(node->getInput(0), tpc->getInput(0));
                    ASSERT_EQ(node->getNumOutputs(), expectedNumSlices);
                    for (const auto& output : node->getOutputs())
                    {
                        ASSERT_EQ(slicedGraph->getNumberOfTensorConsumers(output), 1);
                        ASSERT_EQ(slicedGraph->getTensorConsumers(output).front()->getNodeType(), Node::TYPE_USER);
                    }
                }
                break;
            default:
                FAIL() << "Unexpected node in graph";
        }
    }
    ASSERT_EQ(numOfTPCs, expectedNumSlices);
    ASSERT_EQ(numOfForks, 1);
    ASSERT_EQ(numOfJoins, 1);
    ASSERT_EQ(numOfReductions, 1);
}