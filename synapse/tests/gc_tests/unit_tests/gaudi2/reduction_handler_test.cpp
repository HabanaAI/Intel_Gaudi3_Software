#include "slicer/reduction_handler.h"
#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "types.h"
#include "reduction_node.h"

using namespace gc::layered_brain;

class ReductionHandlerTest : public GraphOptimizerTest
{
protected:
    TensorPtr createTensor()
    {
        const std::vector<TSize> tensorSizes = {128, 128};
        return std::make_shared<Tensor>(tensorSizes.size(), tensorSizes.data(), syn_type_bf16);
    }

    NodePtr createMMENode()
    {
        synGEMMParams params(false, false);
        NodePtr       node = NodeFactory::createNode({createTensor(), createTensor()},
                                               {createTensor()},
                                               &params,
                                               NodeFactory::gemmNodeTypeName,
                                               "MME");
        EXPECT_TRUE(node);
        return node;
    }

    NodePtr createTPCNode()
    {
        NodePtr node =
            NodeFactory::createNode({createTensor()}, {createTensor()}, nullptr, TPCNode::NOP_KERNEL_NAME, "TPC");
        EXPECT_TRUE(node);
        return node;
    }

    TensorPtr getTensorSlice(const TensorPtr& tensor)
    {
        TensorPtr slice                            = tensor->clone(false, false, false);
        slice->getTensorAnnotation().origBigTensor = tensor;
        return slice;
    }

    NodePtr   getNodeSlice(const NodePtr& node) { return node->getSlice(); }

    const BundleIdx     m_bundleIdx  = 4;
};

TEST_F(ReductionHandlerTest, single_producer_for_each_slice)
{
    ReductionHandler reductionHandler(m_bundleIdx, {}, {});

    NodePtr mmeNode       = createMMENode();
    NodePtr slicedMmeNode = mmeNode->getSlice();
    slicedMmeNode->replaceInput(0, getTensorSlice(mmeNode->getInput(0)));
    slicedMmeNode->replaceInput(1, getTensorSlice(mmeNode->getInput(1)));
    slicedMmeNode->replaceOutput(0, getTensorSlice(mmeNode->getOutput(0)));
    reductionHandler.addProducerForTensorSlice(slicedMmeNode->getOutput(0), slicedMmeNode, mmeNode, 0);

    NodePtr tpcNode       = createTPCNode();
    NodePtr slicedTpcNode = tpcNode->getSlice();
    slicedTpcNode->replaceInput(0, getTensorSlice(tpcNode->getInput(0)));
    slicedTpcNode->replaceOutput(0, getTensorSlice(tpcNode->getOutput(0)));
    reductionHandler.addProducerForTensorSlice(slicedTpcNode->getOutput(0), slicedTpcNode, tpcNode, 0);

    const auto& reductionNodes = reductionHandler.createReductionNodes();
    // No reduction nodes expected - a single producer for each slice.
    ASSERT_TRUE(reductionNodes.empty());
}

TEST_F(ReductionHandlerTest, slices_with_multiple_producers)
{
    ReductionHandler reductionHandler(m_bundleIdx, {}, {});

    NodePtr    mmeNode               = createMMENode();
    TensorPtr  reductionTensorForMme = getTensorSlice(mmeNode->getOutput(0));
    unsigned   numOfSlicesForMmeNode = 3;
    NodeVector slicedMmeNodes;
    for (auto i = 0; i < numOfSlicesForMmeNode; i++)
    {
        NodePtr slicedMmeNode = mmeNode->getSlice();
        slicedMmeNodes.push_back(slicedMmeNode);
        slicedMmeNode->replaceInput(0, getTensorSlice(mmeNode->getInput(0)));
        slicedMmeNode->replaceInput(1, getTensorSlice(mmeNode->getInput(1)));
        slicedMmeNode->replaceOutput(0, reductionTensorForMme);  // Multiple producers for output tensor
        reductionHandler.addProducerForTensorSlice(slicedMmeNode->getOutput(0), slicedMmeNode, mmeNode, 0);
    }

    NodePtr    tpcNode               = createTPCNode();
    TensorPtr  reductionTensorForTpc = getTensorSlice(tpcNode->getOutput(0));
    unsigned   numOfSlicesForTpcNode = 5;
    NodeVector slicedTpcNodes;
    for (auto i = 0; i < numOfSlicesForTpcNode; i++)
    {
        NodePtr slicedTpcNode = tpcNode->getSlice();
        slicedTpcNodes.push_back(slicedTpcNode);
        slicedTpcNode->replaceInput(0, getTensorSlice(tpcNode->getInput(0)));
        slicedTpcNode->replaceOutput(0, reductionTensorForTpc);  // Multiple producers for output tensor
        reductionHandler.addProducerForTensorSlice(slicedTpcNode->getOutput(0), slicedTpcNode, tpcNode, 0);
    }

    const auto& reductionNodes = reductionHandler.createReductionNodes();
    ASSERT_EQ(reductionNodes.size(), 2);
    for (const auto& node : reductionNodes)
    {
        const auto& reductionNode = std::dynamic_pointer_cast<ReductionNode>(node);
        ASSERT_TRUE(reductionNode);
        ASSERT_TRUE(reductionNode->getNodeAnnotation().bundleInfo.is_set());
        ASSERT_EQ(reductionNode->getNodeAnnotation().bundleInfo->bundleIndex, m_bundleIdx);
        ASSERT_EQ(reductionNode->getNodeType(), Node::TYPE_INTERNAL_REDUCTION);
        ASSERT_EQ(reductionNode->getNumOutputs(), 1);
        if (reductionNode->getOutput(0) == reductionTensorForMme)
        {
            ASSERT_EQ(reductionNode->getNumInputs(), numOfSlicesForMmeNode);
            for (auto i = 0; i < numOfSlicesForMmeNode; i++)
            {
                ASSERT_EQ(slicedMmeNodes[i]->getOutput(0), reductionNode->getInput(i));
            }
            // For MME nodes - reduction operation set to REDUCTION_ADD.
            ASSERT_EQ(reductionNode->getReductionOperation(), ReductionOperation::REDUCTION_ADD);
        }
        else
        {
            ASSERT_EQ(reductionNode->getOutput(0), reductionTensorForTpc);
            ASSERT_EQ(reductionNode->getNumInputs(), numOfSlicesForTpcNode);
            for (auto i = 0; i < numOfSlicesForTpcNode; i++)
            {
                ASSERT_EQ(slicedTpcNodes[i]->getOutput(0), reductionNode->getInput(i));
            }
            // For TPC nodes - reduction operation set to REDUCTION_UNORDERED_SET.
            ASSERT_EQ(reductionNode->getReductionOperation(), ReductionOperation::REDUCTION_UNORDERED_SET);
        }
    }
}

TEST_F(ReductionHandlerTest, convert_reduction_tensors_to_f32)
{
    NodePtr mmeNode = createMMENode();
    ASSERT_EQ(mmeNode->getOutput(0)->getElementType(), syn_type_bf16);

    ReductionHandler reductionHandler(m_bundleIdx, {mmeNode}, {});

    TensorPtr  reductionTensorForMme = getTensorSlice(mmeNode->getOutput(0));
    unsigned   numOfSlicesForMmeNode = 4;
    NodeVector slicedMmeNodes;
    for (auto i = 0; i < numOfSlicesForMmeNode; i++)
    {
        NodePtr slicedMmeNode = mmeNode->getSlice();
        slicedMmeNodes.push_back(slicedMmeNode);
        slicedMmeNode->replaceInput(0, getTensorSlice(mmeNode->getInput(0)));
        slicedMmeNode->replaceInput(1, getTensorSlice(mmeNode->getInput(1)));
        slicedMmeNode->replaceOutput(0, reductionTensorForMme);  // Multiple producers for output tensor
        reductionHandler.addProducerForTensorSlice(slicedMmeNode->getOutput(0), slicedMmeNode, mmeNode, 0);
    }

    const auto& newNodes = reductionHandler.createReductionNodes();
    for (const auto& node : newNodes)
    {
        ASSERT_TRUE(node->getNodeAnnotation().bundleInfo.is_set());
        ASSERT_EQ(node->getNodeAnnotation().bundleInfo->bundleIndex, m_bundleIdx);
    }

    // Validate that the new nodes are reduction -> cast
    ASSERT_EQ(newNodes.size(), 2);
    auto castIt = std::find_if(newNodes.begin(), newNodes.end(), [](const NodePtr& n) { return n->isCast(); });
    ASSERT_TRUE(castIt != newNodes.end());
    auto castNode = *castIt;
    ASSERT_EQ(castNode->getNumInputs(), 1);
    ASSERT_EQ(castNode->getNumOutputs(), 1);
    auto reductionIt = std::find_if(newNodes.begin(), newNodes.end(), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_INTERNAL_REDUCTION;
    });
    ASSERT_TRUE(reductionIt != newNodes.end());
    auto reductionNode = std::dynamic_pointer_cast<ReductionNode>(*reductionIt);
    ASSERT_TRUE(reductionNode);
    ASSERT_EQ(reductionNode->getReductionOperation(), ReductionOperation::REDUCTION_ADD);
    ASSERT_EQ(reductionNode->getNumInputs(), numOfSlicesForMmeNode);
    ASSERT_EQ(reductionNode->getNumOutputs(), 1);
    for (auto i = 0; i < numOfSlicesForMmeNode; i++)
    {
        ASSERT_EQ(slicedMmeNodes[i]->getOutput(0), reductionNode->getInput(i));
        ASSERT_EQ(slicedMmeNodes[i]->getOutput(0)->getElementType(), syn_type_single);  // MME should output F32
    }
    ASSERT_EQ(reductionNode->getOutput(0)->getElementType(), syn_type_single);
    ASSERT_EQ(reductionNode->getOutput(0), castNode->getInput(0));
    ASSERT_EQ(castNode->getOutput(0)->getElementType(),
              mmeNode->getOutput(0)->getElementType());  // Cast back to original data-type
}

TEST_F(ReductionHandlerTest, single_producer_with_memset)
{
    NodePtr          mmeNode = createMMENode();
    ReductionHandler reductionHandler(m_bundleIdx, {}, {mmeNode});

    NodePtr slicedMmeNode = mmeNode->getSlice();
    slicedMmeNode->replaceInput(0, getTensorSlice(mmeNode->getInput(0)));
    slicedMmeNode->replaceInput(1, getTensorSlice(mmeNode->getInput(1)));
    slicedMmeNode->replaceOutput(0, getTensorSlice(mmeNode->getOutput(0)));
    reductionHandler.addProducerForTensorSlice(slicedMmeNode->getOutput(0), slicedMmeNode, mmeNode, 0);

    const auto& newNodes = reductionHandler.createReductionNodes();
    ASSERT_EQ(newNodes.size(), 2);
    auto reduction = std::find_if(newNodes.begin(), newNodes.end(), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_INTERNAL_REDUCTION;
    });
    ASSERT_TRUE(reduction != newNodes.end());
    auto memset = std::find_if(newNodes.begin(), newNodes.end(), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_MEMSET;
    });
    ASSERT_TRUE(memset != newNodes.end());
    const auto& reductionInputs = (*reduction)->getInputs();
    ASSERT_TRUE(std::find(reductionInputs.begin(), reductionInputs.end(), (*memset)->getOutput(0)) !=
                reductionInputs.end());
    ASSERT_TRUE(std::find(reductionInputs.begin(), reductionInputs.end(), slicedMmeNode->getOutput(0)) !=
                reductionInputs.end());
}

TEST_F(ReductionHandlerTest, multiple_producers_with_memset)
{
    NodePtr          mmeNode = createMMENode();
    ReductionHandler reductionHandler(m_bundleIdx, {}, {mmeNode});

    TensorPtr  reductionTensorForMme = getTensorSlice(mmeNode->getOutput(0));
    unsigned   numOfSlices           = 3;
    NodeVector slicedMmeNodes;
    for (auto i = 0; i < numOfSlices; i++)
    {
        NodePtr slicedMmeNode = mmeNode->getSlice();
        slicedMmeNodes.push_back(slicedMmeNode);
        slicedMmeNode->replaceInput(0, getTensorSlice(mmeNode->getInput(0)));
        slicedMmeNode->replaceInput(1, getTensorSlice(mmeNode->getInput(1)));
        slicedMmeNode->replaceOutput(0, reductionTensorForMme);  // Multiple producers for output tensor
        reductionHandler.addProducerForTensorSlice(slicedMmeNode->getOutput(0), slicedMmeNode, mmeNode, 0);
    }

    const auto& newNodes = reductionHandler.createReductionNodes();
    ASSERT_EQ(newNodes.size(), 2);
    auto reduction = std::find_if(newNodes.begin(), newNodes.end(), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_INTERNAL_REDUCTION;
    });
    ASSERT_TRUE(reduction != newNodes.end());
    auto memset = std::find_if(newNodes.begin(), newNodes.end(), [](const NodePtr& n) {
        return n->getNodeType() == Node::TYPE_MEMSET;
    });
    ASSERT_TRUE(memset != newNodes.end());
    const auto& reductionInputs = (*reduction)->getInputs();
    ASSERT_TRUE(std::find(reductionInputs.begin(), reductionInputs.end(), (*memset)->getOutput(0)) !=
                reductionInputs.end());
    for (const auto& slicedNode : slicedMmeNodes)
    {
        ASSERT_TRUE(std::find(reductionInputs.begin(), reductionInputs.end(), slicedNode->getOutput(0)) !=
                    reductionInputs.end());
    }
}