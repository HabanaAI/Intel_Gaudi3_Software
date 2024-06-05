#include "graph_optimizer_test.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"

class Gaudi2DescriptorGeneratorTest : public GraphOptimizerTest {};

TEST_F(Gaudi2DescriptorGeneratorTest, add_dma_descriptors_to_graph_nothing_to_do)
{
    Gaudi2Graph                 graph;
    gaudi2::DescriptorGenerator tested(downcaster<Gaudi2CodeGenerator>(graph.getCodeGenerator().get()));
    TensorPtr                   inputTensor  = std::make_shared<Tensor>(syn_type_bf16);
    TensorPtr                   outputTensor = std::make_shared<Tensor>(syn_type_bf16);
    std::string                 nodeName;
    auto dmaNodePtr  = std::make_shared<DMANode>(inputTensor, outputTensor, nodeName, DMA_TYPE_SPILL);
    auto expectedPtr = std::make_shared<DMANode>(inputTensor, outputTensor, nodeName, DMA_TYPE_SPILL);

    ASSERT_NO_THROW(tested.visit(dmaNodePtr.get()));
    ASSERT_EQ(*(expectedPtr.get()), *(dmaNodePtr.get()));
}

TEST_F(Gaudi2DescriptorGeneratorTest, generate_dma_descriptors_empty_roi)
{
    Gaudi2Graph                 graph;
    gaudi2::DescriptorGenerator dg(downcaster<Gaudi2CodeGenerator>(graph.getCodeGenerator().get()));
    TensorPtr                   outputTensor = std::make_shared<Tensor>(syn_type_bf16);
    TensorPtr                   inputTensor  = std::make_shared<Tensor>(syn_type_bf16);
    const std::list<NodeROI>    physicalRois;

    inputTensor->setSramOffset(1000);
    outputTensor->setDramOffset(500);

    DMANode dmaNode(inputTensor, outputTensor, "nodeName", DMA_TYPE_INTERNAL);

    gaudi2::DescriptorGenerator::DmaDescriptorsList descriptors;

    ASSERT_NO_THROW(dg.generateDmaDescriptors(dmaNode, physicalRois, descriptors));
    ASSERT_TRUE(descriptors.empty());
}
