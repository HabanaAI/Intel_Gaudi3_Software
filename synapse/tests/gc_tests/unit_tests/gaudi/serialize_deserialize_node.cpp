#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "gaudi_graph.h"
#include "compilation_hal_reader.h"

class GaudiSerializeDeserializeNodes : public GraphOptimizerTest
{
    public:
        GaudiSerializeDeserializeNodes()
        {
            CompilationHalReader::setHalReader(m_graph.getHALReader());
        }
        ~GaudiSerializeDeserializeNodes()
        {
            CompilationHalReader::setHalReader(nullptr);
        }
        GaudiGraph& getGraph()
        {
            return m_graph;
        }
    private:
        GaudiGraph m_graph;

};

TEST_F(GaudiSerializeDeserializeNodes, serialize_node)
{
    TSize srcSizes[] = {2, 4, 4};
    TSize destSizes[] = {2, 4, 4};
    unsigned dim = 3;

    uint64_t srcStrides[]  = {4, 8, 40, 160, 160, 160};
    uint64_t destStrides[] = {4, 8, 32, 128, 128, 128};

    float stridedData[] =
    {
            1, 1,       1, 1,       1, 1,       1, 1,       0, 0,
            2, 2,       2, 2,       2, 2,       2, 2,       0, 0,
            3, 3,       3, 3,       3, 3,       3, 3,       0, 0,
            4, 4,       4, 4,       4, 4,       4, 4,       0, 0
    };

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr inTensor = TensorPtr (new Tensor(dim, srcSizes, syn_type_float, (char*)stridedData, srcStrides));
    inTensor->setName("input");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    synMemoryDescriptor persistentMemoryDesc2(true);

    TensorPtr outTensor = TensorPtr (new Tensor(dim, destSizes, syn_type_float, nullptr, destStrides));
    inTensor->setName("output");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inTensor->setMemoryDescriptor(persistentMemoryDesc2);

    NodePtr node = NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::getSerializeNodeGUID(), "Serialize");

    GraphEditor::addNode(getGraph(), node);
    ASSERT_TRUE(getGraph().compile());
}

TEST_F(GaudiSerializeDeserializeNodes, serialize_node_invalid_size)
{
    TSize srcSizes[] = {2, 4, 4};
    TSize destSizes[] = {2, 4, 5};
    unsigned dim = 3;

    uint64_t srcStrides[]  = {4, 8, 40, 160, 160, 160};
    uint64_t destStrides[] = {4, 8, 32, 160, 160, 160};

    float stridedData[] =
            {
                    1, 1,       1, 1,       1, 1,       1, 1,       0, 0,
                    2, 2,       2, 2,       2, 2,       2, 2,       0, 0,
                    3, 3,       3, 3,       3, 3,       3, 3,       0, 0,
                    4, 4,       4, 4,       4, 4,       4, 4,       0, 0
            };

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr inTensor = TensorPtr (new Tensor(dim, srcSizes, syn_type_float, (char*)stridedData, srcStrides));
    inTensor->setName("input");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    synMemoryDescriptor persistentMemoryDesc2(true);

    TensorPtr outTensor = TensorPtr (new Tensor(dim, destSizes, syn_type_float, nullptr, destStrides));
    inTensor->setName("output");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inTensor->setMemoryDescriptor(persistentMemoryDesc2);

    EXPECT_ANY_THROW(NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::getSerializeNodeGUID(), "Serialize"));
}

TEST_F(GaudiSerializeDeserializeNodes, serialize_node_invalid_stride)
{
    TSize srcSizes[] = {2, 4, 4};
    TSize destSizes[] = {2, 4, 4};
    unsigned dim = 3;

    uint64_t srcStrides[]  = {4, 8, 40, 160, 160, 160};
    uint64_t destStrides[] = {4, 8, 32, 192, 192, 192};

    float stridedData[] =
            {
                    1, 1,       1, 1,       1, 1,       1, 1,       0, 0,
                    2, 2,       2, 2,       2, 2,       2, 2,       0, 0,
                    3, 3,       3, 3,       3, 3,       3, 3,       0, 0,
                    4, 4,       4, 4,       4, 4,       4, 4,       0, 0
            };

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr inTensor = TensorPtr (new Tensor(dim, srcSizes, syn_type_float, (char*)stridedData, srcStrides));
    inTensor->setName("input");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    synMemoryDescriptor persistentMemoryDesc2(true);

    TensorPtr outTensor = TensorPtr (new Tensor(dim, destSizes, syn_type_float, nullptr, destStrides));
    inTensor->setName("output");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inTensor->setMemoryDescriptor(persistentMemoryDesc2);

    EXPECT_ANY_THROW(NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::getSerializeNodeGUID(), "Serialize"));
}

TEST_F(GaudiSerializeDeserializeNodes, deserialize_node)
{
    TSize srcSizes[] = {2, 4, 4};
    TSize destSizes[] = {2, 4, 4};
    unsigned dim = 3;

    uint64_t srcStrides[]  = {4, 8, 32, 128, 128, 128};
    uint64_t destStrides[] = {4, 8, 40, 160, 160, 160};

    float denseData[] =
    {
            1, 1,       1, 1,       1, 1,       1, 1,
            2, 2,       2, 2,       2, 2,       2, 2,
            3, 3,       3, 3,       3, 3,       3, 3,
            4, 4,       4, 4,       4, 4,       4, 4
    };

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr inTensor = TensorPtr (new Tensor(dim, srcSizes, syn_type_float, (char*)denseData, srcStrides));
    inTensor->setName("input");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    synMemoryDescriptor persistentMemoryDesc2(true);

    TensorPtr outTensor = TensorPtr (new Tensor(dim, destSizes, syn_type_float, nullptr, destStrides));
    inTensor->setName("output");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inTensor->setMemoryDescriptor(persistentMemoryDesc2);

    NodePtr node = NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::getDeserializeNodeGUID(), "Deserialize");

    GraphEditor::addNode(getGraph(), node);
    ASSERT_TRUE(getGraph().compile());
}

TEST_F(GaudiSerializeDeserializeNodes, deserialize_node_invalid_size)
{
    TSize srcSizes[] = {2, 4, 5};
    TSize destSizes[] = {2, 4, 4};
    unsigned dim = 3;

    uint64_t srcStrides[]  = {4, 8, 32, 160, 160, 160};
    uint64_t destStrides[] = {4, 8, 40, 160, 160, 160};

    float denseData[] =
    {
            1, 1,       1, 1,       1, 1,       1, 1,
            2, 2,       2, 2,       2, 2,       2, 2,
            3, 3,       3, 3,       3, 3,       3, 3,
            4, 4,       4, 4,       4, 4,       4, 4,
    };

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr inTensor = TensorPtr (new Tensor(dim, srcSizes, syn_type_float, (char*)denseData, srcStrides));
    inTensor->setName("input");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    synMemoryDescriptor persistentMemoryDesc2(true);

    TensorPtr outTensor = TensorPtr (new Tensor(dim, destSizes, syn_type_float, nullptr, destStrides));
    inTensor->setName("output");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inTensor->setMemoryDescriptor(persistentMemoryDesc2);

    EXPECT_ANY_THROW(NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::getDeserializeNodeGUID(), "Deserialize"));
}

TEST_F(GaudiSerializeDeserializeNodes, deserialize_node_invalid_stride)
{
    TSize srcSizes[] = {2, 4, 4};
    TSize destSizes[] = {2, 4, 4};
    unsigned dim = 3;

    uint64_t srcStrides[]  = {4, 8, 32, 192, 192, 192};
    uint64_t destStrides[] = {4, 8, 40, 160, 160, 160};

    float denseData[] =
    {
            1, 1,       1, 1,       1, 1,       1, 1,
            2, 2,       2, 2,       2, 2,       2, 2,
            3, 3,       3, 3,       3, 3,       3, 3,
            4, 4,       4, 4,       4, 4,       4, 4
    };

    synMemoryDescriptor persistentMemoryDesc(true);

    TensorPtr inTensor = TensorPtr (new Tensor(dim, srcSizes, syn_type_float, (char*)denseData, srcStrides));
    inTensor->setName("input");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    synMemoryDescriptor persistentMemoryDesc2(true);

    TensorPtr outTensor = TensorPtr (new Tensor(dim, destSizes, syn_type_float, nullptr, destStrides));
    inTensor->setName("output");
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    inTensor->setMemoryDescriptor(persistentMemoryDesc2);

    EXPECT_ANY_THROW(NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::getDeserializeNodeGUID(), "Deserialize"));
}
