#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "params_file_manager.h"
#include "platform/gaudi3/graph_compiler/passes.h"

using namespace gaudi3;
class Gaudi3GraphTest : public GraphOptimizerTest
{
};

TEST_F(Gaudi3GraphTest, create_and_compile_gaudi3_graph)
{
    const unsigned      tensor_dim = 1;
    const TSize         size       = 1;
    Gaudi3Graph         g;
    TensorPtr           i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr           o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    NodePtr             n = NodeFactory::createDebugNode(i, o, "");
    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i->setDramOffset(0x1000);
    o->setDramOffset(0x2000);
    i->setMemoryDescriptor(memDesc);
    o->setMemoryDescriptor(memDesc);
    GraphEditor::addNode(g, n);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
}

TEST_F(Gaudi3GraphTest, single_add_tpc_node_test)
{
    const unsigned      tensor_dim = 1;
    const TSize         size       = 1;
    Gaudi3Graph         g;
    TensorPtr           i1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr           i2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr           o  = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    NodePtr             n  = NodeFactory::createNode({i1, i2}, {o}, nullptr, "add_fwd_f32", "add");
    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i1->setDramOffset(0x1000);
    i2->setDramOffset(0x2000);
    o->setDramOffset(0x3000);
    i1->setMemoryDescriptor(memDesc);
    i2->setMemoryDescriptor(memDesc);
    o->setMemoryDescriptor(memDesc);

    ASSERT_TRUE(GraphEditor::addNode(g, n));
    ASSERT_TRUE(g.compile()) << "failed to compile graph";

    ASSERT_EQ(g.getNodes().size(), 1) << "Expecting a single node in graph";
    for (const auto& node : g.getNodes())
    {
        ASSERT_TRUE(g.runsOnTPC(node)) << "Expecting memcpy to run on tpc";
    }
}

TEST_F(Gaudi3GraphTest, update_tpc_descriptor_wrapper)
{
    TensorVector    inTensors;
    TensorVector    outTensors;
    std::string     nodeName;
    auto            node = std::make_shared<TPCNode>(inTensors, outTensors, nodeName);
    Gaudi3Graph     tested;
    gaudi3::TpcDesc tpcDescriptor;
    ValidityMask<gaudi3::TpcDesc> tpcMask;
    NodeROI         roi;
    tpc_wd_ctxt_t   emptyTpcFwCtx = {0};
    TpcDescriptorGenerator::McidTpcUsage mcidTpcUsage;
    tested.updateTPCDescriptorWrapper(*node.get(), tpcDescriptor, tpcMask, {emptyTpcFwCtx}, roi, mcidTpcUsage);

    auto wrappers = tested.getTpcNodeDescriptorsWrappers(tested.getNodeSharedPtr(*node.get()));
    ASSERT_EQ(wrappers.size(), 1);
}

class Gaudi3TpcMemcpyTest
: public GraphOptimizerTest
, public testing::WithParamInterface<synDataType>
{
};

TEST_P(Gaudi3TpcMemcpyTest, single_memcpy_tpc_node_test)
{
    const unsigned tensor_dim = 4;
    const TSize    sizes[]    = {10, 16, 2, 1};
    auto           dataType   = GetParam();

    Gaudi3Graph         g;
    TensorPtr           i = TensorPtr(new Tensor(tensor_dim, sizes, dataType));
    TensorPtr           o = TensorPtr(new Tensor(tensor_dim, sizes, dataType));
    NodePtr             n = NodeFactory::createNode({i}, {o}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i->setDramOffset(0x1000);
    o->setDramOffset(0x2000);
    i->setMemoryDescriptor(memDesc);
    o->setMemoryDescriptor(memDesc);
    i->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    GraphEditor::addNode(g, n);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";

    uint32_t tpcNodeCounter = 0;

    for (const auto& node : g.getNodes())
    {
        if (g.runsOnTPC(node))
        {
            tpcNodeCounter++;
        }
    }

    ASSERT_EQ(tpcNodeCounter, 1) << "Expecting memcpy to run on tpc";
}

INSTANTIATE_TEST_SUITE_P(,
                         Gaudi3TpcMemcpyTest,
                         ::testing::Values(syn_type_int8,
                                           syn_type_uint8,
                                           syn_type_int16,
                                           syn_type_uint16,
                                           syn_type_int32,
                                           syn_type_bf16,
                                           syn_type_single,
                                           syn_type_float,
                                           syn_type_fp16,
                                           syn_type_fp8_143,
                                           syn_type_fp8_152,
                                           syn_type_tf32,
                                           syn_type_hb_float));