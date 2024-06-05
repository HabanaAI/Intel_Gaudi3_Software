#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "identity_node.h"
#include "node_factory.h"
#include "tensor.h"
#include "types_exception.h"

class IdentityTest : public GraphOptimizerTest
{
};

TEST_F(IdentityTest, test_only_output_persistent)
{
    static const TSize channels = 16;
    static const TSize width    = 32;
    static const TSize height   = 256;
    static const TSize batch    = 512;

    TSize originalSize[] = {channels, width, height, batch};
    pTensor sourceTensor(new Tensor(4, originalSize, syn_type_bf16));
    pTensor middleTensor(new Tensor(4, originalSize, syn_type_bf16));
    pTensor targetTensor(new Tensor(4, originalSize, syn_type_bf16));
    sourceTensor->setAsNonUserManaged();
    middleTensor->setAsNonUserManaged();
    synMemoryDescriptor descriptor;
    descriptor.m_isPersistent = true;
    targetTensor->setMemoryDescriptor(descriptor);
    pNode identity1 = NodeFactory::createNode({sourceTensor}, {middleTensor}, nullptr, NodeFactory::identityNodeTypeName, "");
    pNode identity2 = NodeFactory::createNode({middleTensor}, {targetTensor}, nullptr, NodeFactory::identityNodeTypeName, "");
    GaudiGraph g;
    GraphEditor::addNode(g, identity1);
    GraphEditor::addNode(g, identity2);
    handleLogicalOps(g);
    // No memcopy should be inserted, each tensor is an alias to the tensor after it
    ASSERT_EQ(g.getNodes().size(), 2);
}

TEST_F(IdentityTest, test_both_persistent)
{
    static const TSize channels = 16;
    static const TSize width    = 32;
    static const TSize height   = 256;
    static const TSize batch    = 512;

    TSize originalSize[] = {channels, width, height, batch};
    pTensor sourceTensor(new Tensor(4, originalSize, syn_type_bf16));
    pTensor middleTensor(new Tensor(4, originalSize, syn_type_bf16));
    pTensor targetTensor(new Tensor(4, originalSize, syn_type_bf16));
    synMemoryDescriptor descriptor;
    descriptor.m_isPersistent = true;
    sourceTensor->setAsNonUserManaged();
    middleTensor->setMemoryDescriptor(descriptor);
    middleTensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
    targetTensor->setMemoryDescriptor(descriptor);
    targetTensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM);
    pNode identity1 = NodeFactory::createNode({sourceTensor}, {middleTensor}, nullptr, NodeFactory::identityNodeTypeName, "");
    pNode identity2 = NodeFactory::createNode({middleTensor}, {targetTensor}, nullptr, NodeFactory::identityNodeTypeName, "");
    GaudiGraph g;
    GraphEditor::addNode(g, identity1);
    GraphEditor::addNode(g, identity2);
    handleLogicalOps(g);
    // One memcopy should be inserted, from the first persistent tensor to the middle tensor
    ASSERT_EQ(g.getNodes().size(), 3);
}


TEST_F(IdentityTest, test_invalid_node)
{
    TSize originalSize[] = {1, 2, 3, 4};
    TSize differentSize[] = {5, 6, 7, 8};
    pTensor sourceTensor(new Tensor(4, originalSize, syn_type_bf16));
    pTensor targetTensor(new Tensor(4, differentSize, syn_type_bf16));
    bool isValidNode = true;
    try
    {
        pNode identity = NodeFactory::createNode({sourceTensor}, {targetTensor}, nullptr, NodeFactory::identityNodeTypeName, "");
    } catch(const InvalidNodeParamsException& e)
    {
        isValidNode = false;
    }
    ASSERT_EQ(isValidNode, false);
}
