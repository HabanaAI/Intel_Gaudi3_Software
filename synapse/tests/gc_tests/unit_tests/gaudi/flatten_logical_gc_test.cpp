#include "gaudi_graph.h"
#include "graph_compiler/types.h"
#include "graph_optimizer_test.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "node_factory.h"

TEST_F(GraphOptimizerTest, DISABLED_gaudi_logical_flatten_with_relu_gc_test)
{
    // Graph: relu_in(persist)->relu_fwd_32->relu_out/flatten_in->flatten->flatten_out(persist)

    // Default alias direction for logical flatten is OUTPUT_TO_INPUT.
    // In this case, we are not able to set flatten_out as alias since it is persist tensor.
    // We can swap the alias direction and avoid adding memcpy node. (flatten_in will be aliased to flatten_out).
    const TSize FCD = 4;
    const TSize WIDTH = 4;
    const TSize HEIGHT = 4;
    const TSize BATCH = 1;

    TSize relu_input_dimensions[] = {FCD, WIDTH, HEIGHT, BATCH};
    TSize flat_input_dimensions[] = {FCD * WIDTH, HEIGHT, BATCH};

    const unsigned int relu_dim_num = 4;
    const unsigned int flat_dim_num = 2;

    const TSize inputSize = FCD * WIDTH * HEIGHT * BATCH;
    float *inputArray = new float[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputArray[i] = (float) i;
    }

    pTensor relu_in       = pTensor(new Tensor(relu_dim_num, relu_input_dimensions, syn_type_float,
                                               reinterpret_cast<char*>(inputArray), nullptr, false, true));
    pTensor relu_out      = pTensor(new Tensor(relu_dim_num, relu_input_dimensions,  syn_type_float));
    pTensor flatten_out   = pTensor(new Tensor(flat_dim_num, flat_input_dimensions, syn_type_float,
                                               nullptr,nullptr, true, false));

    // Set graph's input tensor as persistent
    synMemoryDescriptor relu_memDesc(true);
    relu_in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    relu_in->setMemoryDescriptor(relu_memDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor flatten_memDesc(true);
    flatten_out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR+1);
    flatten_out->setMemoryDescriptor(flatten_memDesc);

    pNode relu = NodeFactory::createGenericTPCNode({relu_in}, {relu_out}, nullptr, "relu_fwd_f32", "relu");

    // Flatten setting
    synFlattenParams    flattenAttr;
    flattenAttr.axis    = 1;

    pNode flatten = NodeFactory::createNode({relu_out}, {flatten_out}, &flattenAttr, NodeFactory::flattenNodeTypeName, "flatten_logic");

    GaudiGraph g;

    GraphEditor::addNode(g, relu);
    GraphEditor::addNode(g, flatten);

    ASSERT_TRUE(g.compile()) << "Failed to compile graph";

    // Make sure no memcpy/DMA nodes are in compiled graph
    const NodeVector& nodes = g.getExeSortedNodes();
    for (const pNode& node : nodes)
    {
        ASSERT_TRUE(node->getNodeType() == Node::TYPE_INTERNAL_FLATTEN ||
                    node->getNodeType() == Node::TYPE_INTERNAL_PACKING ||
                    node->getNodeType() == Node::TYPE_USER
        )
                    << "Found an unexpected node in graph: " << node->getNodeType();

        // validate flatten logical node input/output dims
        if (node->getNodeType() == Node::TYPE_INTERNAL_FLATTEN)
        {
            node->getOutputs().front()->validateFlattenSubTensor(node->getInputs().front(), flattenAttr.axis);
        }
    }

    // validate graph input/output dims
    flatten_out->validateFlattenSubTensor(relu_in, flattenAttr.axis);

    delete[] inputArray;
}
