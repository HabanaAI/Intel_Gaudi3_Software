#include "graph_editor.h"
#include "graph_optimizer_test.h"
#include "graph_visualization.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "gaudi2_graph.h"
#include "types.h"
#include "utils.h"

class RemoveRedundantMemcpyTest : public GraphOptimizerTest
{
public:
    TensorPtr
    createPersistentTensor(unsigned dim, const SizeArray& sizes, synDataType type, const std::string name = "")
    {
        synMemoryDescriptor persistentMemoryDesc(true);
        const auto          t = createTensor(dim, sizes, type, name);
        t->setMemorySectionID(m_sectionId++);
        t->setMemoryDescriptor(persistentMemoryDesc);
        return t;
    }

    TensorPtr createTensor(unsigned dim, const SizeArray& sizes, synDataType type, const std::string& name = "")
    {
        TensorPtr t = std::make_shared<Tensor>(dim, sizes.data(), type);
        t->setName(name);
        return t;
    }

    NodePtr addNode(const TensorVector& inputs, const TensorVector& outputs, const char * guid, const char * name)
    {
        NodePtr n = NodeFactory::createNode(inputs, outputs, nullptr, guid, name);
        GraphEditor::addNode(m_graph, n);
        return n;
    }

    unsigned getNumMemcpyInGraph()
    {
        unsigned numMemcpy = 0;
        for (const auto& n: m_graph.getNodes())
        {
            if (isMemcpy(*n)) numMemcpy++;
        }
        return numMemcpy;
    }

    void runTest()
    {
        graphVisualizationPre(m_graph);
        removeRedundantMemcpyNodes(m_graph);
        graphVisualizationPost(m_graph);
    }

private:
    Gaudi2Graph m_graph;
    unsigned m_sectionId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
};

TEST_F(RemoveRedundantMemcpyTest, basic_memcpy_remove)
{
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut = createTensor(dim, sizes, syn_type_float, "reluOut");
    NodePtr relu = addNode({reluIn},{reluOut}, "relu_fwd_f32", "relu");

    auto memcpyOut = createTensor(dim, sizes, syn_type_float, "memcpyOut");
    NodePtr memcpy = addNode({reluOut}, {memcpyOut}, NodeFactory::memcpyNodeTypeName, "memcpy");

    auto addIn2 = createTensor(dim, sizes, syn_type_float, "addIn2");
    auto addOut = createTensor(dim, sizes, syn_type_float, "addOut");
    NodePtr add = addNode({memcpyOut, addIn2}, {addOut}, "add_fwd_f32", "add");

    runTest();

    ASSERT_EQ(getNumMemcpyInGraph(), 0) << "Memcpy node is redundant - should be removed";
    ASSERT_TRUE(relu->getOutput(0) == reluOut)
        << "Since memcpy out is not persistent, it can be removed. The producer should not be modified.";
    ASSERT_TRUE(add->getInput(0) == reluOut)
        << "After memcpy removal, the memcpy's consumer should consume the original memcpy's input";
}

TEST_F(RemoveRedundantMemcpyTest, handle_persistent_output)
{
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut = createTensor(dim, sizes, syn_type_float, "reluOut");
    NodePtr relu = addNode({reluIn}, {reluOut}, "relu_fwd_f32", "relu");

    auto memcpyOut = createPersistentTensor(dim, sizes, syn_type_float, "memcpyOut");
    NodePtr memcpy = addNode({reluOut}, {memcpyOut}, NodeFactory::memcpyNodeTypeName, "memcpy");

    auto addIn2 = createTensor(dim, sizes, syn_type_float, "addIn2");
    auto addOut = createTensor(dim, sizes, syn_type_float, "addOut");
    NodePtr add = addNode({memcpyOut, addIn2}, {addOut}, "add_fwd_f32", "add");

    runTest();

    ASSERT_EQ(getNumMemcpyInGraph(), 0) << "Memcpy node is redundant - should be removed";
    ASSERT_TRUE(relu->getOutput(0) == memcpyOut)
        << "After memcpy removal, the memcpy's producer should produce the persistent tensor (original memcpy out)";
    ASSERT_TRUE(add->getInput(0) == memcpyOut)
        << "After memcpy removal, the memcpy's consumer should consume the persistent tensor (original memcpy out)";
}

TEST_F(RemoveRedundantMemcpyTest, handle_persistent_input)
{
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut = createPersistentTensor(dim, sizes, syn_type_float, "reluOut");
    NodePtr relu = addNode({reluIn}, {reluOut}, "relu_fwd_f32", "relu");

    auto memcpyOut = createTensor(dim, sizes, syn_type_float, "memcpyOut");
    NodePtr memcpy = addNode({reluOut}, {memcpyOut}, NodeFactory::memcpyNodeTypeName, "memcpy");

    auto addIn2 = createTensor(dim, sizes, syn_type_float, "addIn2");
    auto addOut = createTensor(dim, sizes, syn_type_float, "addOut");
    NodePtr add = addNode({memcpyOut, addIn2}, {addOut}, "add_fwd_f32", "add");

    runTest();

    ASSERT_EQ(getNumMemcpyInGraph(), 0) << "Memcpy node is redundant - should be removed";
    ASSERT_TRUE(relu->getOutput(0) == reluOut)
        << "Since the memcpy's input is persistent, it should be unchanged";
    ASSERT_TRUE(add->getInput(0) == reluOut)
        << "Since the memcpy's input is persistent, it should be unchanged";
}

TEST_F(RemoveRedundantMemcpyTest, handle_persistent_input_and_output)
{
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut = createPersistentTensor(dim, sizes, syn_type_float, "reluOut");
    NodePtr relu = addNode({reluIn}, {reluOut}, "relu_fwd_f32", "relu");

    auto memcpyOut = createPersistentTensor(dim, sizes, syn_type_float, "memcpyOut");
    NodePtr memcpy = addNode({reluOut}, {memcpyOut}, NodeFactory::memcpyNodeTypeName, "memcpy");

    auto addIn2 = createTensor(dim, sizes, syn_type_float, "addIn2");
    auto addOut = createTensor(dim, sizes, syn_type_float, "addOut");
    NodePtr add = addNode({memcpyOut, addIn2}, {addOut}, "add_fwd_f32", "add");

    runTest();

    ASSERT_EQ(getNumMemcpyInGraph(), 1) << "Memcpy node is NOT redundant - both input and output are persistent";
}

TEST_F(RemoveRedundantMemcpyTest, dont_remove_memcpy_that_changes_dtype)
{
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto memcpyIn = createTensor(dim, sizes, syn_type_bf16, "memcpyIn");
    auto memcpyOut = createTensor(dim, sizes, syn_type_float, "memcpyOut");
    NodePtr memcpy = addNode({memcpyIn}, {memcpyOut}, NodeFactory::memcpyNodeTypeName, "memcpy");

    auto addIn2 = createTensor(dim, sizes, syn_type_float, "addIn2");
    auto addOut = createTensor(dim, sizes, syn_type_float, "addOut");
    NodePtr add = addNode({memcpyOut, addIn2}, {addOut}, "add_fwd_f32", "add");

    runTest();

    ASSERT_EQ(getNumMemcpyInGraph(), 1) << "Memcpy node is NOT redundant - it functions as a cast, so it cant be removed";
}
