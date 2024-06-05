#include <gtest/gtest.h>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"

class Gaudi3GraphTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<const char*, synDataType>>
{
};

/*
 * Test to verify that fuseTransposeMme pass is fusing MmeTranspose nodes
 * In the pre graph:   bgemm => transpose (permutation: [1,0,3,2])
 * After extractMultiNodes: bgemm => MmeTranspose (permutation: [1,0,2,3]) => LogicalTranspose (permutation: [0,1,3,2])
 * After fuseTransposeMme (and post graph): bgemm (transpose_a & transpose_b) => LogicalTranspose (permutation:
 * [0,1,3,2])
 */
// TODO [SW-178847]: Enable test once fuser can identify
// a reshaped physical transpose as a fusion candidate
TEST_F(Gaudi3GraphTest, DISABLED_gaudi3_fuseTransposeMme_test)
{
    Gaudi3Graph g;
    uint8_t     dim      = 4;
    auto        dataType = syn_type_single;

    // batch gemm node creation
    const TSize sizesIn0[]     = {64, 16, 2, 3};
    const TSize sizesIn1[]     = {64, 64, 2, 3};
    TSize       bgemmOutSize[] = {64, 16, 2, 3};

    auto bgemmGuid = NodeFactory::batchGemmNodeTypeName;

    TensorPtr bgemmInput0 = TensorPtr(new Tensor(dim, sizesIn0, dataType));
    TensorPtr bgemmInput1 = TensorPtr(new Tensor(dim, sizesIn1, dataType));
    TensorPtr bgemmOut    = TensorPtr(new Tensor(dim, bgemmOutSize, dataType));

    synMemoryDescriptor memDesc(true);  // persistent

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    bgemmInput0->setDramOffset(0x10000);
    bgemmInput1->setDramOffset(0x20000);
    bgemmInput0->setMemoryDescriptor(memDesc);
    bgemmInput1->setMemoryDescriptor(memDesc);
    bgemmInput0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    bgemmInput1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    synGEMMParams params {};

    NodePtr bgemm = NodeFactory::createNode({bgemmInput0, bgemmInput1}, {bgemmOut}, &params, bgemmGuid, "bgemm_node");

    // transpose node creation
    auto transpose_guid = NodeFactory::transposeNodeTypeName;

    TSize     transposeOutSize[] = {16, 64, 3, 2};
    TensorPtr transposeOut       = TensorPtr(new Tensor(dim, transposeOutSize, dataType));

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    transposeOut->setDramOffset(0x30000);
    transposeOut->setMemoryDescriptor(memDesc);
    transposeOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);

    synTransposeParamsNDims transposeParams;
    transposeParams.tensorDim      = dim;
    transposeParams.permutation[0] = 1;
    transposeParams.permutation[1] = 0;
    transposeParams.permutation[2] = 3;
    transposeParams.permutation[3] = 2;
    transposeParams.tensorDim      = dim;

    NodePtr transpose =
        NodeFactory::createNode({bgemmOut}, {transposeOut}, &transposeParams, transpose_guid, "mme_transpose_node");

    GraphEditor::addNode(g, bgemm);
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile());
    ASSERT_EQ(g.getNodes().size(), 2) << "Expecting two nodes in graph";

    uint32_t mmeCounter              = 0;
    uint32_t logicalTransposeCounter = 0;

    for (const auto& node : g.getNodes())
    {
        if (g.runsOnMME(node))
        {
            mmeCounter++;
        }

        if (node->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE)
        {
            logicalTransposeCounter++;
        }
    }

    ASSERT_EQ(mmeCounter, 1) << "Expecting a single MME node (batch gemm) in post graph";
    ASSERT_EQ(logicalTransposeCounter, 1) << "Expecting a single logical transpose node in post graph";
}