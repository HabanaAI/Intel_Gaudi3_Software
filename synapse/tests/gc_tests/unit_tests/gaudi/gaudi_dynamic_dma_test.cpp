#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "gaudi_recipe_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "gaudi_code_generator.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "queue_dispatcher.h"
#include "params_file_manager.h"
#include "graph_optimizer_test.h"

using namespace std;

TEST_F(GaudiRecipeTest, verify_patching_points_dynamic_dma)
{
    static const unsigned c_parallelLevel = 2;

    GaudiGraph      g;

    // Tensor(unsigned dim, const unsigned* sizes, synDataType type, const unsigned* minSize  s);
    //
    //       dimensions 0 and 2 are dynamic
    TSize sizes[3] = { 1024, 1024, 16 };
    TSize minSizes[3] = { 128, 1024, 8 };
    const unsigned  tensor_dim = sizeof(sizes)/sizeof(sizes[0]);

    TensorPtr i = std::make_shared<Tensor>(tensor_dim, sizes, syn_type_single, minSizes);
    TensorPtr o = std::make_shared<Tensor>(tensor_dim, sizes, syn_type_single, minSizes);
    NodePtr   n = NodeFactory::createNode({i}, {o}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "N1");

    std::dynamic_pointer_cast<DMANode>(n)->setParallelLevel(c_parallelLevel);

    synMemoryDescriptor memDescPersist(true);
    i->setMemoryDescriptor(memDescPersist);
    i->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o->setMemoryDescriptor(memDescPersist);
    o->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    GraphEditor::addNode(g, n);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";

    gaudi::DMADescriptorsWrappers descWrap = downcaster<GaudiCodeGenerator>(g.getCodeGenerator().get())->getDMANodeDescriptorsWrappers(n);

    for (auto dw : descWrap)
    {
        const BasicFieldsContainerInfo& fci = dw.getBasicFieldsContainerInfo();
        const BasicFieldInfoSet&        bfis = fci.retrieveBasicFieldInfoSet();

        ASSERT_EQ(bfis.size(), 4); // 2 dynamic dimensions in src and dst tensors each
        for (const BasicFieldInfoPair& bfip: bfis)
        {
            auto type = bfip.second->getType();
            ASSERT_TRUE(type == FieldType::FIELD_DYNAMIC_DMA_SRC ||
                        type == FieldType::FIELD_DYNAMIC_DMA_DST);
        }

    }
}
