#include <gtest/gtest.h>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/gaudi3_code_generator.h"

class CodeGenTest : public GraphOptimizerTest
{
};

TEST_F(CodeGenTest, validate_control_edge_dependency)
{
    Gaudi3Graph g;

    // Add mme node
    const TSize sizes_x[] = {256, 256, 1, 1};
    const TSize sizes_w[] = {256, 256, 1, 1};
    const TSize sizes_y[] = {256, 256, 1, 1};
    TensorPtr x = TensorPtr(new Tensor(4U, sizes_x, syn_type_fp16));
    TensorPtr w = TensorPtr(new Tensor(4U, sizes_w, syn_type_fp16));
    TensorPtr y = TensorPtr(new Tensor(4U, sizes_y, syn_type_fp16));
    synMemoryDescriptor memDesc1(true);  // persistent

    // set some bogus addresses to the tensors and allocate host memory so we won't assert
    x->setDramOffset(0x1000);
    w->setDramOffset(0x2000);
    y->setDramOffset(0x3000);
    x->setMemoryDescriptor(memDesc1);
    w->setMemoryDescriptor(memDesc1);
    y->setMemoryDescriptor(memDesc1);
    x->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    w->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    y->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    synConvolutionParams params {};
    NodePtr mmeNode = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "mme_node");
    ASSERT_TRUE(GraphEditor::addNode(g, mmeNode));

    // Add TPC node
    const unsigned tensor_dim = 1;
    const TSize size          = 1;
    TensorPtr xx = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr ww = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr yy = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    synMemoryDescriptor memDesc2(true);  // persistent

    // set some bogus addresses to the tensors and allocate host memory so we won't assert
    xx->setDramOffset(0x40000);
    ww->setDramOffset(0x50000);
    yy->setDramOffset(0x60000);
    xx->setMemoryDescriptor(memDesc2);
    ww->setMemoryDescriptor(memDesc2);
    yy->setMemoryDescriptor(memDesc2);
    x->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    w->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);
    y->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 7);
    NodePtr tpcNode = NodeFactory::createNode({xx, ww}, {yy}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(GraphEditor::addNode(g, tpcNode));

    // Add SYNC control dependency
    g.addControlDependency({mmeNode}, {tpcNode}, Tensor::ControlEdgeType::SYNC);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getExeSortedNodes().size(), 2);

    const NodePtr& tpcNodeAfterCompile = g.getExeSortedNodes()[1];

    ASSERT_EQ(tpcNodeAfterCompile->getNodeAnnotation().arcSyncScheme.size(), 1);
    ASSERT_EQ(tpcNodeAfterCompile->getNodeAnnotation().arcSyncScheme[0].dependencies.size(), 1); // dependency on mme
    ASSERT_EQ(tpcNodeAfterCompile->getNodeAnnotation().arcSyncScheme[0].dependencies[0], 1);     // dependency on mme
}

TEST_F(CodeGenTest, reduce_pipeline_dependencies)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "true");
    uint8_t     dim      = 4;
    auto        guid     = NodeFactory::transposeNodeTypeName;
    auto        dataType = syn_type_single;
    Gaudi3Graph g;

    TSize inSize[]  = {64, 16, 2, 1};
    TSize outSize[] = {2, 16, 64, 1};

    TensorPtr in  = TensorPtr(new Tensor(dim, inSize, dataType));
    TensorPtr out = TensorPtr(new Tensor(dim, outSize, dataType));

    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    in->setDramOffset(0x1000);
    out->setDramOffset(0x5000);
    in->setMemoryDescriptor(memDesc);
    out->setMemoryDescriptor(memDesc);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    // Create transpose node
    synTransposeParamsNDims transposeParams;
    transposeParams.tensorDim = dim;
    for (int i = 0; i < dim; ++i)
    {
        transposeParams.permutation[i] = i;
    }

    transposeParams.permutation[0] = 2;
    transposeParams.permutation[1] = 1;
    transposeParams.permutation[2] = 0;
    transposeParams.tensorDim      = dim;

    NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());

    uint32_t mmeNodeCounter = 0;
    uint32_t tpcNodeCounter = 0;

    for (const auto& node : g.getNodes())
    {
        if (g.runsOnMME(node)) mmeNodeCounter++;
        if (g.runsOnTPC(node)) tpcNodeCounter++;
    }
    ASSERT_EQ(mmeNodeCounter, 1) << "Expecting a single MME node in post graph";
    ASSERT_EQ(tpcNodeCounter, 1) << "Expecting one tpc memcpy node in post graph";

    Gaudi3CodeGenerator& codeGen = dynamic_cast<Gaudi3CodeGenerator&>(*(g.getCodeGenerator()));
    DependencyMap depMap;
    depMap[DEVICE_XPS_LOGICAL_QUEUE] = 1;
    depMap[DEVICE_TPC_LOGICAL_QUEUE] = 1;
    codeGen.removeRedundantDependencies(depMap, 0);
    ASSERT_EQ(depMap.size(), 1);
    ASSERT_TRUE(depMap.find(DEVICE_XPS_LOGICAL_QUEUE) != depMap.end()) << "XPS dependency should have been kept";
    ASSERT_TRUE(depMap.find(DEVICE_TPC_LOGICAL_QUEUE) == depMap.end()) << "TPC dependency should have been dropped";
}
