#include "graph_optimizer_test.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "dma_transpose_node.h"
#include "node_factory.h"

namespace gaudi
{
class GraphOptimizerTestTranspose : public GraphOptimizerTest
{
};

TEST_F(GraphOptimizerTestTranspose, transpose_test_2d_in_sram)
{
    setGlobalConfForTest(GCFG_ENABLE_TWO_DIM_TRANSPOSE_RESHAPER, "false");

    setGlobalConfForTest(GCFG_DMA_TRANSPOSE_SOLVER_MAX_SCD_SIZE, "127");
    setGlobalConfForTest(GCFG_DMA_TRANSPOSE_SOLVER_MIN_FCD_SIZE, "15000");
    TSize inSizes[]  = {1638400, 4};
    TSize outSizes[] = {4, 1638400};

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};
    GaudiGraph         g;
    TensorPtr          in  = std::make_shared<Tensor>(2, inSizes, syn_type_float);
    TensorPtr          out = std::make_shared<Tensor>(2, outSizes, syn_type_float);

    synMemoryDescriptor memDescPersist(true);
    in->setMemoryDescriptor(memDescPersist);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    out->setMemoryDescriptor(memDescPersist);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    in->setTensorInDram();
    out->setTensorInDram();

    NodePtr transposeNode = NodeFactory::createNode({in}, {out}, &transposeParams, "transpose", "transpose_test");

    ASSERT_TRUE(GraphEditor::addNode(g, transposeNode));
    ASSERT_TRUE(g.compile());

    unsigned dmaTransposeNodeCount = 0;
    for (const auto& node : g.getNodes())
    {
        if (std::dynamic_pointer_cast<DMATransposeNode>(node) != nullptr)
        {
            dmaTransposeNodeCount++;
            ASSERT_TRUE(node->getOutput(0)->inSram());
        }
    }
    ASSERT_EQ(dmaTransposeNodeCount, 2);
}

TEST_F(GraphOptimizerTestTranspose, transpose_test_2d_in_dram)
{
    setGlobalConfForTest(GCFG_ENABLE_TWO_DIM_TRANSPOSE_RESHAPER, "false");

    setGlobalConfForTest(GCFG_DMA_TRANSPOSE_SOLVER_MAX_SCD_SIZE, "127");
    setGlobalConfForTest(GCFG_DMA_TRANSPOSE_SOLVER_MIN_FCD_SIZE, "15000");

    TSize inSizes[]  = {2000, 4000};
    TSize outSizes[] = {4000, 2000};

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};
    GaudiGraph         g;
    TensorPtr          in  = std::make_shared<Tensor>(2, inSizes, syn_type_float);
    TensorPtr          out = std::make_shared<Tensor>(2, outSizes, syn_type_float);

    synMemoryDescriptor memDescPersist(true);
    in->setMemoryDescriptor(memDescPersist);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    out->setMemoryDescriptor(memDescPersist);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    in->setTensorInDram();
    out->setTensorInDram();

    NodePtr transposeNode = NodeFactory::createNode({in}, {out}, &transposeParams, "transpose", "transpose_test");

    ASSERT_TRUE(GraphEditor::addNode(g, transposeNode));
    ASSERT_TRUE(g.compile());

    unsigned dmaTransposeNodeCount = 0;
    for (const auto& node : g.getNodes())
    {
        if (std::dynamic_pointer_cast<DMATransposeNode>(node) != nullptr)
        {
            dmaTransposeNodeCount++;
            ASSERT_TRUE(node->getOutput(0)->isUserManagedDram());
        }
    }
    ASSERT_EQ(dmaTransposeNodeCount, 1);
}
}  // namespace gaudi
