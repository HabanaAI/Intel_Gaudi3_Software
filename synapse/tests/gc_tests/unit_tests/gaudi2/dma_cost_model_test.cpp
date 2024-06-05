#include "defs.h"
#include "graph_optimizer_test.h"
#include "habana_nodes.h"
#include "hal_reader/hal_reader.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "graph_compiler/dma_cost_model.h"
#include "hal_reader/gaudi2/hal_reader.h"
#include "transpose_utils.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <memory>

constexpr auto         GB_TO_BYTES = 1 << 30;
constexpr auto         SEC_TO_USEC = 1000000;
constexpr unsigned int HBM_BW      = 1945;
constexpr unsigned int DMA_BW      = 970;

class DmaCostModelTest : public GraphOptimizerTest
{
    void SetUp() override { GraphOptimizerTest::SetUp(); }

public:
    void createTest(Gaudi2Graph*             g,
                    const unsigned           tensor_dim,
                    const TSize*             inSizes,
                    const TSize*             outSizes,
                    std::string_view         guid,
                    synTransposeParamsNDims* params = nullptr);
    void compareCost(double costRef, NodePtr node);
};

void DmaCostModelTest::createTest(Gaudi2Graph*             g,
                                  const unsigned           tensor_dim,
                                  const TSize*             inSizes,
                                  const TSize*             outSizes,
                                  std::string_view         guid,
                                  synTransposeParamsNDims* params)
{
    uint64_t     totalSize  = 0;
    unsigned int paramsSize = params ? sizeof(*params) : 0;

    TensorPtr           o      = TensorPtr(new Tensor(tensor_dim, outSizes, syn_type_single));
    TensorVector        inputs = {};
    synMemoryDescriptor memDesc(true);  // persistent

    if (guid.compare("memset") != 0)
    {
        ASSERT_TRUE(inSizes) << "input sizes is nullptr";
        totalSize = dataTypeSizeInBytes(syn_type_single) * multiplyElements(inSizes, inSizes + tensor_dim);

        TensorPtr i = TensorPtr(new Tensor(tensor_dim, inSizes, syn_type_single));
        inputs.push_back(i);
        i->setTensorInDram();
        i->setDramOffset(0x1000);
        i->setMemoryDescriptor(memDesc);
    }

    o->setTensorInDram();
    o->setDramOffset(0x1000 + totalSize);
    o->setMemoryDescriptor(memDesc);

    NodePtr n = NodeFactory::createNode(inputs, {o}, params, paramsSize, guid, "node1");

    GraphEditor::addNode(*g, n);
    ASSERT_TRUE(g->compile()) << "failed to compile graph";
}

void DmaCostModelTest::compareCost(double costRef, NodePtr node)
{
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(node);
    HB_ASSERT(dmaNode, "Failed to convert to DMA node");

    DmaCostModel cost(*Gaudi2HalReader::instance().get());
    ASSERT_EQ(costRef, cost.getCostModelResult(*dmaNode.get()).durationInUsec);
}

TEST_F(DmaCostModelTest, memcpy)
{
    Gaudi2Graph    g;
    const unsigned tensor_dim        = 4;
    const TSize    sizes[tensor_dim] = {128, 32, 16, 4};
    double         totalSize = dataTypeSizeInBytes(syn_type_single) * multiplyElements(sizes, sizes + tensor_dim);
    createTest(&g, tensor_dim, sizes, sizes, "memcpy");

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 1);

    double costRef = std::max((totalSize * 2) / HBM_BW, totalSize / DMA_BW) / GB_TO_BYTES * SEC_TO_USEC;
    compareCost(costRef, nodes.front());
}

TEST_F(DmaCostModelTest, memset)
{
    Gaudi2Graph    g;
    const unsigned tensor_dim        = 4;
    const TSize    sizes[tensor_dim] = {128, 32, 16, 4};
    double         totalSize = dataTypeSizeInBytes(syn_type_single) * multiplyElements(sizes, sizes + tensor_dim);
    createTest(&g, tensor_dim, nullptr, sizes, "memset");

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 1);

    double costRef = std::max(totalSize / HBM_BW, totalSize / DMA_BW) / GB_TO_BYTES * SEC_TO_USEC;
    compareCost(costRef, nodes.front());
}

TEST_F(DmaCostModelTest, transpose)
{
    Gaudi2Graph    g;
    const unsigned tensor_dim           = 4;
    const TSize    inSizes[tensor_dim]  = {128, 32, 16, 4};
    const TSize    outSizes[tensor_dim] = {32, 16, 4, 128};
    double         totalSize = dataTypeSizeInBytes(syn_type_single) * multiplyElements(inSizes, inSizes + tensor_dim);
    TransposePermutationArray permutation({TPD_Width, TPD_Height, TPD_4Dim_Batch, TPD_Channel});
    synTransposeParamsNDims   params = permutationToParams(permutation);
    createTest(&g, tensor_dim, inSizes, outSizes, "transpose", &params);

    const NodeVector& nodes        = g.getExeSortedNodes();
    NodePtr           dmaTranspose = nullptr;
    for (auto node : nodes)
    {
        if (node->getGUID().compare(NodeFactory::transposeDmaNodeTypeName) == 0)
        {
            dmaTranspose = node;
        }
    }
    HB_ASSERT(dmaTranspose, "No dma transpose node in graph");

    double costRef = std::max((totalSize * 2) / HBM_BW, totalSize / DMA_BW) / GB_TO_BYTES * SEC_TO_USEC;
    compareCost(costRef, dmaTranspose);
}

TEST_F(DmaCostModelTest, broadcast)
{
    Gaudi2Graph    g;
    const unsigned tensor_dim           = 4;
    const TSize    inSizes[tensor_dim]  = {2, 8, 16, 1};
    const TSize    outSizes[tensor_dim] = {2, 8, 16, 4};
    double         totalSize = dataTypeSizeInBytes(syn_type_single) * multiplyElements(outSizes, outSizes + tensor_dim);

    createTest(&g, tensor_dim, inSizes, outSizes, "broadcast");

    const NodeVector& nodes   = g.getExeSortedNodes();
    NodePtr           dmaNode = nullptr;
    for (auto node : nodes)
    {
        if (node->getNodeType() == Node::TYPE_DMA)
        {
            dmaNode = node;
        }
    }

    HB_ASSERT(dmaNode, "No dma node in graph");

    double costRef = std::max((totalSize * 2) / HBM_BW, totalSize / DMA_BW) / GB_TO_BYTES * SEC_TO_USEC;
    compareCost(costRef, dmaNode);
}