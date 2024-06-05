#include "global_conf_test_setter.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "synapse_common_types.hpp"
#include "transpose_node.h"
#include "node.h"
#include "transpose_utils.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace gaudi
{
class TransposeTest : public GraphOptimizerTest
{
};

static void createTransposeGraph(GaudiGraph&                      g,
                                 bool                             preferLogicalBeforePhisical,
                                 const TransposePermutationArray& permutation,
                                 const std::vector<TSize>&        sizes)
{
    const unsigned numOfElements = multiplyElements(sizes);

    const unsigned int dimsNum = permutation.size();

    // Set output dimensions according to the transpose permutation
    std::vector<TSize> outputDims(dimsNum);
    applyPermutation(sizes.data(), permutation, outputDims.data());

    // Initialize input data
    std::vector<float> inputData;
    for(auto i = 0; i < numOfElements; i++)
    {
        inputData.push_back(i);
    }

    // Create input and output tensors
    pTensor in(new Tensor(dimsNum, sizes.data(), syn_type_float, reinterpret_cast<char*>(inputData.data()), nullptr, false, true));
    pTensor out(new Tensor(dimsNum, outputDims.data(), syn_type_float, nullptr, nullptr, true, false));

    // Set graph's input tensor as persistent
    synMemoryDescriptor inMemDesc(true);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    in->setMemoryDescriptor(inMemDesc);

    // Set graph's output tensor as persistent
    synMemoryDescriptor outMemDesc(true);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR+1);
    out->setMemoryDescriptor(outMemDesc);

    // Create transpose node
    synTransposeParamsNDims transposeParams;
    transposeParams.tensorDim = dimsNum;
    for (int i = 0; i < dimsNum; ++i)
    {
        transposeParams.permutation[i] = permutation[i];
    }
    pNode node = NodeFactory::createNode({in}, {out}, &transposeParams, NodeFactory::transposeNodeTypeName, "transpose1");
    std::shared_ptr<TransposeNode> transpose = std::dynamic_pointer_cast<TransposeNode>(node);
    ASSERT_TRUE(transpose != nullptr);
    if(preferLogicalBeforePhisical)
    {
        transpose->setPreferLogicalBeforePhysical(true);
    }

    // Compile graph
    GraphEditor::addNode(g, node);
    ASSERT_EQ(g.compile(), true) << "Failed to compile graph";

    // Run nodes on CPU
    for (const auto& node : g.getExeSortedNodes())
    {
        node->RunOnCpu();
    }

    // Check maximum 2 actual DMAs are created
    uint32_t countNonLogical = 0;
    for (const auto& node : g.getExeSortedNodes())
    {
        if (!node->isLogicalOperation())
        {
            countNonLogical += 1;
        }
    }
    auto maxNonLogical = preferLogicalBeforePhisical ? 3 : 2;
    ASSERT_TRUE(countNonLogical <= maxNonLogical);
    NodePtr logicalTranspose = preferLogicalBeforePhisical ? g.getExeSortedNodes().front() : g.getExeSortedNodes().back();
    auto physicalTranspose = *std::find_if(g.getExeSortedNodes().begin(), g.getExeSortedNodes().end(), [](const NodePtr& n) {
        return n->isDma() && dynamic_cast<DMANode*>(n.get())->isTranspose();
    });
    ASSERT_EQ(logicalTranspose->getNodeType(), Node::TYPE_LOGICAL_TRANSPOSE);
    ASSERT_TRUE(logicalTranspose->isLogicalOperation());
    ASSERT_TRUE(physicalTranspose->isDma());

    ASSERT_EQ(g.GetNodeROIs(physicalTranspose)->size(), g.getPipelineDepth(*physicalTranspose));
    ASSERT_EQ(g.getPipelineDepth(*physicalTranspose), g.getDefaultPipelineDepth());
}

static void runTransposeTest(const TransposePermutationArray& permutation,
                             const std::vector<TSize>&        sizes = {14, 7, 23, 51})
{
    GaudiGraph g1;
    GaudiGraph g2;

    createTransposeGraph(g1, false, permutation, sizes);
    createTransposeGraph(g2, true, permutation, sizes);

    ASSERT_EQ(g1.getGraphOutputs().size(), 1) << "Wrong number of outputs for g1";
    ASSERT_EQ(g2.getGraphOutputs().size(), 1) << "Wrong number of outputs for g2";

    const pTensor output1 = g1.getGraphOutputs().front();
    const pTensor output2 = g2.getGraphOutputs().front();

    ASSERT_EQ(output1->getTotalElements(), output2->getTotalElements());
    ASSERT_EQ(output1->getTotalSizeInBytes(), output2->getTotalSizeInBytes());

    float* output1Data = (float*)output1->map();
    float* output2Data = (float*)output2->map();

    for (unsigned int i = 0; i < output1->getTotalElements(); ++i)
    {
        ASSERT_FLOAT_EQ(output1Data[i], output2Data[i]) << "Mismatch at index " << i << ":"
                                                        << " output1 (physical before logical): " << output1Data[i]
                                                        << " output2 (logical before physical): " << output2Data[i];
    }
}

TEST_F(TransposeTest, transpose_order_bwch)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    TransposePermutationArray permutation({TPD_4Dim_Batch, TPD_Width, TPD_Channel, TPD_Height});
    runTransposeTest(permutation);
}

TEST_F(TransposeTest, transpose_order_wchb)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    TransposePermutationArray permutation({TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch});
    runTransposeTest(permutation);
}

TEST_F(TransposeTest, transpose_order_wchb_1023x255x7x7)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    TransposePermutationArray permutation({TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch});
    runTransposeTest(permutation);
}

TEST_F(TransposeTest, up_to_only_2_dma_operations_per_transpose)
{
    std::vector<uint32_t>     sizes = {16, 16, 17, 17};
    TransposePermutationArray permutation({TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch});
}

TEST_F(TransposeTest, transpose_5dim)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    TransposePermutationArray permutation(
        {TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch, TransposePermutationDim(4)});

    std::vector<TSize> sizes = {2, 3, 4, 5, 6};
    runTransposeTest(permutation, sizes);
}

class DoubleTransposeTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
{
};

TEST_P(DoubleTransposeTest, double_transpose)
{
    setGlobalConfForTest(GCFG_TRANSPOSE_SPLITTING_THRESHOLD, "1.0");
    TransposePermutationArray permutation({TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch});
    std::vector<TSize>     sizes         = {std::get<0>(GetParam()),
                                   std::get<1>(GetParam()),
                                   std::get<2>(GetParam()),
                                   std::get<3>(GetParam())};
    const unsigned            numOfElements = multiplyElements(sizes);

    const unsigned int dimsNum = permutation.size();

    // Set output dimensions according to the transpose permutation
    TSize outputDims[dimsNum];
    applyPermutation(sizes.data(), permutation, outputDims);

    // Initialize input data
    std::vector<float> inputData;
    for (auto i = 0; i < numOfElements; i++)
    {
        inputData.push_back(i);
    }

    // Create input and output tensors
    pTensor in(new Tensor(dimsNum,
                          sizes.data(),
                          syn_type_float,
                          reinterpret_cast<char*>(inputData.data()),
                          nullptr,
                          false,
                          true));
    pTensor out(new Tensor(dimsNum, outputDims, syn_type_float, nullptr, nullptr, true, false));

    // Set graph's input tensor as persistent
    synMemoryDescriptor inMemDesc(true);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    in->setMemoryDescriptor(inMemDesc);

    // Set graph's output tensor as persistent
    synMemoryDescriptor outMemDesc(true);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    out->setMemoryDescriptor(outMemDesc);

    // Create transpose node
    synTransposeParams transposeParams;
    transposeParams.tensorDim = dimsNum;
    for (int i = 0; i < dimsNum; ++i)
    {
        transposeParams.permutation[i] = permutation[i];
    }
    pNode node =
        NodeFactory::createNode({in}, {out}, &transposeParams, NodeFactory::transposeNodeTypeName, "transpose1");
    std::shared_ptr<TransposeNode> transpose = std::dynamic_pointer_cast<TransposeNode>(node);
    ASSERT_TRUE(transpose != nullptr);
    GaudiGraph g;
    // Compile graph
    GraphEditor::addNode(g, node);
    ASSERT_EQ(g.compile(), true) << "Failed to compile graph";
    // Run nodes on CPU
    int countDma = 0;
    for (const auto& node : g.getExeSortedNodes())
    {
        if (node->isLogicalOperation()) continue;
        ASSERT_TRUE(node->isDma());
        auto dmaNode = std::dynamic_pointer_cast<DMANode>(node);
        HB_ASSERT_PTR(dmaNode);
        ASSERT_TRUE(dmaNode->isTranspose());
        countDma++;
    }
    ASSERT_EQ(countDma, 2);
}

INSTANTIATE_TEST_SUITE_P(,
                         DoubleTransposeTest,
                         ::testing::Values(std::make_tuple(64, 12, 8, 8),
                                           std::make_tuple(64, 2, 16, 4),
                                           std::make_tuple(64, 11, 16, 4),
                                           std::make_tuple(64, 15, 16, 4),
                                           std::make_tuple(64, 19, 16, 6)));

}  // namespace gaudi
