#include "gaudi_graph.h"
#include "graph_compiler/types.h"
#include "graph_optimizer_test.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "node_factory.h"

TEST_F(GraphOptimizerTest, gaudi_logical_transpose_with_relu_gc_test)
{
    // Graph: relu_in(persist)->relu_fwd_32->relu_out/transpose_in->transpose->transpose_out(persist)

    // Default alias direction for logical transpose is OUTPUT_TO_INPUT.
    // In this case, we are not able to set transpose_out as alias since it is persist tensor.
    // We can swap the alias direction and avoid adding memcpy node. (transpose_in will be aliased to transpose_out).
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    const TSize FCD = 1;
    const TSize WIDTH = 2;
    const TSize HEIGHT = 3;
    const TSize BATCH = 1;

    TSize input_dimensions[] = {FCD, WIDTH, HEIGHT, BATCH};

    // Transpose W and H dimensions (CWHB to CHWB). Since no change in FCD we will create only logical transpose.
    TransposePermutationArray permutation({TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch});

    const unsigned int dim_num = permutation.size();

    // Set output dimensions according to the transpose permutation
    TSize output_dimensions[dim_num];
    for (unsigned int index = 0; index < dim_num; ++index)
    {
        output_dimensions[index] = input_dimensions[permutation[index]];
    }

    const unsigned inputSize = FCD * WIDTH * HEIGHT * BATCH;
    float *inputArray = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputArray[i] = (float) i;
    }

    pTensor relu_in       = pTensor(new Tensor(dim_num, input_dimensions, syn_type_float,
                                    reinterpret_cast<char*>(inputArray), nullptr, false, true));
    pTensor relu_out      = pTensor(new Tensor(dim_num, input_dimensions,  syn_type_float));
    pTensor transpose_out = pTensor(new Tensor(dim_num, output_dimensions, syn_type_float,
                               nullptr,nullptr, true, false));

    // Set graph's input tensor as persistent
    synMemoryDescriptor relu_memDesc(true);
    relu_in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    relu_in->setMemoryDescriptor(relu_memDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor transpose_memDesc(true);
    transpose_out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR+1);
    transpose_out->setMemoryDescriptor(transpose_memDesc);

    pNode relu = NodeFactory::createGenericTPCNode({relu_in}, {relu_out}, nullptr, "relu_fwd_f32", "relu");
    pNode transpose = NodeFactory::createNode({relu_out}, {transpose_out}, &permutation, "transpose_logic", "transpose_logic");

    GaudiGraph g;

    GraphEditor::addNode(g, relu);
    GraphEditor::addNode(g, transpose);

    ASSERT_TRUE(g.compile()) << "Failed to compile graph";

    // Make sure no memcpy/DMA nodes are in compiled graph
    const NodeVector& nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        LOG_DEBUG(GO_TEST, "Node {} in graph", node->getNodeTypeStr());
        ASSERT_TRUE(node->getNodeType() == Node::TYPE_LOGICAL_TRANSPOSE ||
                    node->getNodeType() == Node::TYPE_USER)
            << "Found an unexpected node in graph";
    }

    delete[] inputArray;
}