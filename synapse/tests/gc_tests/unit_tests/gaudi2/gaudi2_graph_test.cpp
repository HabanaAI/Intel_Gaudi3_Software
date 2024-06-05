#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "gaudi2_code_generator.h"
#include "params_file_manager.h"

class Gaudi2GraphTest : public GraphOptimizerTest {};

TEST_F(Gaudi2GraphTest, create_and_compile_gaudi2_graph)
{
    const unsigned      tensor_dim = 1;
    const TSize         size = 1;
    Gaudi2Graph         g;
    TensorPtr           i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr           o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    NodePtr             n = NodeFactory::createDebugNode(i, o, "");
    synMemoryDescriptor memDesc(true); // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i->setDramOffset(0x1000);
    o->setDramOffset(0x2000);
    i->setMemoryDescriptor(memDesc);
    o->setMemoryDescriptor(memDesc);
    GraphEditor::addNode(g, n);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
}

TEST_F(Gaudi2GraphTest, update_dma_descriptor_wrapper)
{
    TensorPtr       tensor;
    std::string     nodeName;
    auto            nodePtr = std::make_shared<DMANode>(tensor, nodeName);
    Gaudi2Graph     tested;
    auto            testedCodeGen = downcaster<Gaudi2CodeGenerator>(tested.getCodeGenerator().get());
    gaudi2::DmaDesc dmaDescriptor;
    edma_wd_ctxt_t  fwCtx = {0};
    ValidityMask<gaudi2::DmaDesc> dmaMask;
    NodeROI         roi;

    testedCodeGen->updateDMADescriptorWrapper(*(nodePtr.get()), dmaDescriptor, dmaMask, fwCtx, roi);

    auto wrappers = testedCodeGen->getDmaNodeDescriptorsWrappers(tested.getNodeSharedPtr(*(nodePtr.get())));
    ASSERT_EQ(wrappers.size(), 1);
}

TEST_F(Gaudi2GraphTest, update_tpc_descriptor_wrapper)
{
    TensorVector    inTensors;
    TensorVector    outTensors;
    std::string     nodeName;
    auto            node = std::make_shared<TPCNode>(inTensors, outTensors, nodeName);
    Gaudi2Graph     tested;
    auto            testedCodeGen = downcaster<Gaudi2CodeGenerator>(tested.getCodeGenerator().get());
    gaudi2::TpcDesc tpcDescriptor;
    ValidityMask<gaudi2::TpcDesc> tpcMask;
    NodeROI         roi;
    tpc_wd_ctxt_t   emptyTpcFwCtx;
    testedCodeGen->updateTPCDescriptorWrapper(*node.get(), tpcDescriptor, tpcMask, emptyTpcFwCtx, roi);

    auto wrappers = testedCodeGen->getTpcNodeDescriptorsWrappers(tested.getNodeSharedPtr(*node.get()));
    ASSERT_EQ(wrappers.size(), 1);
}
