#include <gtest/gtest.h>
#include <iostream>
#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "transpose_utils.h"

class Gaudi3GraphTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<const char*, synDataType>>
{
};

using Dim = TransposePermutationDim;

TEST_P(Gaudi3GraphTest, gaudi3_mme_node_gc_test)
{
    auto guid     = std::get<0>(GetParam());
    auto dataType = std::get<1>(GetParam());

    Gaudi3Graph g;

    const TSize sizes_x[] = {256, 256, 1, 1};
    const TSize sizes_w[] = {256, 256, 1, 1};
    const TSize sizes_y[] = {256, 256, 1, 1};

    TensorPtr x = TensorPtr(new Tensor(4U, sizes_x, dataType));
    TensorPtr w = TensorPtr(new Tensor(4U, sizes_w, dataType));
    TensorPtr y = TensorPtr(new Tensor(4U, sizes_y, dataType));

    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    x->setDramOffset(0x10000);
    w->setDramOffset(0x20000);
    y->setDramOffset(0x30000);
    x->setMemoryDescriptor(memDesc);
    w->setMemoryDescriptor(memDesc);
    y->setMemoryDescriptor(memDesc);
    x->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    w->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    y->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    synConvolutionParams params {};

    NodePtr conv = NodeFactory::createNode({x, w}, {y}, &params, guid, "mme_node");
    GraphEditor::addNode(g, conv);

    ASSERT_TRUE(g.compile());
}

INSTANTIATE_TEST_SUITE_P(_,
                         Gaudi3GraphTest,
                         ::testing::Values(std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_fp8_143),
                                           std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_fp8_152),
                                           std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_bf16),
                                           std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_fp16),
                                           std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_tf32),
                                           std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_single),
                                           std::make_tuple(NodeFactory::convolutionNodeTypeName, syn_type_hb_float),

                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_fp8_143),
                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_fp8_152),
                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_bf16),
                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_fp16),
                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_tf32),
                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_single),
                                           std::make_tuple(NodeFactory::deDxNodeTypeName, syn_type_hb_float),

                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_fp8_143),
                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_fp8_152),
                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_bf16),
                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_fp16),
                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_tf32),
                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_single),
                                           std::make_tuple(NodeFactory::deDwNodeTypeName, syn_type_hb_float)));

TEST_F(Gaudi3GraphTest, gaudi3_mme_transpose_node_gc_test)
{
    uint8_t dim      = 2;
    auto    guid     = NodeFactory::transposeNodeTypeName;
    auto    dataType = syn_type_single;

    Gaudi3Graph g;

    TSize inSize[]  = {64, 16};
    TSize outSize[] = {16, 64};

    TensorPtr in  = TensorPtr(new Tensor(dim, inSize, dataType));
    TensorPtr out = TensorPtr(new Tensor(dim, outSize, dataType));

    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    in->setDramOffset(0x1000);
    out->setDramOffset(0x2000);
    in->setMemoryDescriptor(memDesc);
    out->setMemoryDescriptor(memDesc);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    // Create transpose node
    synTransposeParamsNDims transposeParams = permutationToParams({Dim(1), Dim(0)});
    NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 1) << "Expecting a single node in graph";

    for (const auto& node : g.getNodes())
    {
        ASSERT_TRUE(g.runsOnMME(node)) << "Expecting transpose to run on mme";
    }
}

TEST_F(Gaudi3GraphTest, gaudi3_mme_transpose_node_gc_test1)
{
    uint8_t dim      = 3;
    auto    guid     = NodeFactory::transposeNodeTypeName;
    auto    dataType = syn_type_single;

    Gaudi3Graph g;

    TSize inSize[]  = {64, 16, 2};
    TSize outSize[] = {16, 64, 2};

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
    synTransposeParamsNDims transposeParams = permutationToParams({Dim(1), Dim(0), Dim(2)});

    NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 1) << "Expecting a single node in graph";

    for (const auto& node : g.getNodes())
    {
        ASSERT_TRUE(g.runsOnMME(node)) << "Expecting transpose to run on mme";
    }
}

TEST_F(Gaudi3GraphTest, gaudi3_mme_transpose_node_gc_test2)
{
    uint8_t dim      = 4;
    auto    guid     = NodeFactory::transposeNodeTypeName;
    auto    dataType = syn_type_single;

    Gaudi3Graph g;

    TSize inSize[]  = {64, 16, 2, 1};
    TSize outSize[] = {2, 64, 16, 1};

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
    synTransposeParamsNDims transposeParams = permutationToParams({Dim(2), Dim(0), Dim(1), Dim(3)});

    NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 3) << "Expecting 3 nodes in graph (physical transpose + reshapes)";

    uint32_t mmeNodeCounter = 0;
    uint32_t rspNodeCounter = 0;

    for (const auto& node : g.getNodes())
    {
        if (g.runsOnMME(node))
        {
            mmeNodeCounter++;
        }

        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            rspNodeCounter++;
        }
    }

    ASSERT_EQ(mmeNodeCounter, 1) << "Expecting a single MME node in post graph";
    ASSERT_EQ(rspNodeCounter, 2) << "Expecting two reshape nodes in post graph";
}

TEST_F(Gaudi3GraphTest, gaudi3_mme_transpose_node_gc_test3)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "true");
    uint8_t dim      = 4;
    auto    guid     = NodeFactory::transposeNodeTypeName;
    auto    dataType = syn_type_single;

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
    synTransposeParamsNDims transposeParams = permutationToParams({Dim(2), Dim(1), Dim(0), Dim(3)});

    NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 5) << "Expecting 5 nodes in graph: physical and logical transposes, flatten and "
                                         "unflatten and memcpy";

    uint32_t mmeNodeCounter = 0;
    uint32_t rspNodeCounter = 0;
    uint32_t tpcNodeCounter = 0;

    for (const auto& node : g.getNodes())
    {
        if (g.runsOnMME(node))
        {
            mmeNodeCounter++;
        }

        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            rspNodeCounter++;
        }

        if (g.runsOnTPC(node))
        {
            tpcNodeCounter++;
        }
    }

    ASSERT_EQ(mmeNodeCounter, 1) << "Expecting a single MME node in post graph";
    ASSERT_EQ(rspNodeCounter, 2) << "Expecting two reshape nodes in post graph";
    ASSERT_EQ(tpcNodeCounter, 1) << "Expecting one tpc memcpy node in post graph";
}

TEST_F(Gaudi3GraphTest, gaudi3_mme_transpose_node_gc_test4)
{
    uint8_t dim      = 3;
    auto    guid     = NodeFactory::transposeNodeTypeName;
    auto    dataType = syn_type_single;

    Gaudi3Graph g;

    TSize inSize[]  = {2, 1024, 512};
    TSize outSize[] = {1024, 512, 2};

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
    synTransposeParamsNDims transposeParams = permutationToParams({Dim(1), Dim(2), Dim(0)});

    NodePtr transpose = NodeFactory::createNode({in}, {out}, &transposeParams, guid, "mme_transpose_node");
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());
}
