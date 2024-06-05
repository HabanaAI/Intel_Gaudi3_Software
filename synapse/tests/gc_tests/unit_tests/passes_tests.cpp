#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "fuse_pad_pass_tests_common.h"
#include "gaudi2_graph.h"

using namespace std;

class PASSES : public GraphOptimizerTest {};

TEST_F(PASSES, DISABLED_eliminate_pass)
{
    Gaudi2Graph  g;
    g.setInferenceMode(true);

    const TSize kW    = 1;
    const TSize kH    = 1;
    const TSize dW    = 1;
    const TSize dH    = 1;
    const TSize nOFM  = 256;
    const TSize wOFM  = 2;
    const TSize hOFM  = 1;
    const TSize nIFM  = 256;
    const TSize padH  = 0;
    const TSize padW  = 0;
    const TSize batch = 1;

    synConvolutionParams params;
    params.dH   = dH;
    params.dW   = dW;
    params.kH   = kH;
    params.kW   = kW;
    params.padT = padH;
    params.padB = padH;
    params.padL = padW;
    params.padR = padW;
    params.dilH = 1;
    params.dilW = 1;

    //o = ((i - k + 2 * pad) / stride) + 1
    const TSize wIFM = ((wOFM - 1) * params.dW) + (params.kW - 1) * params.dilW + 1 - (params.padL + params.padR);
    const TSize hIFM = ((hOFM - 1) * params.dH) + (params.kH - 1) * params.dilH + 1 - (params.padT + params.padB);

    float weights[nIFM * nOFM * params.kW * params.kH];
    char ifm[nIFM * wIFM * hIFM * batch];
    weights[0]=1;
    ifm[0]=1;

    const TSize i_sizes[] = { nIFM, wIFM, hIFM, batch };
    const TSize o_sizes[] = { nOFM, wOFM, hOFM, batch };
    const TSize w_sizes[] = { nOFM, nIFM, params.kH, params.kW};

    const unsigned SIZE = 3;
    TensorPtr      tensors[SIZE+1];
    TensorPtr      wtensors[SIZE];
    NodePtr        nodes[SIZE];

    // IFM
    tensors[0] = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed, reinterpret_cast<char*>(ifm)));

    for (unsigned i = 0; i < SIZE; ++i)
    {
        // OFM
        tensors[i+1] = TensorPtr(new Tensor(4U, o_sizes, syn_type_fixed));

        // Weights
        wtensors[i]  = TensorPtr(new Tensor(4U, w_sizes, syn_type_fixed,
                reinterpret_cast<char*>(weights)));
        wtensors[i]->setAsWeights();
    }


    for (unsigned i = 0; i < SIZE; ++i)
    {
        NodePtr n = getConvNodeWithGoyaLayouts(tensors[i], wtensors[i],
                nullptr, tensors[i+1], params, "");
        GraphEditor::addNode(g, n);
        nodes[i] = n;
    }

    Gaudi2Graph                        copy1(g);
    Gaudi2Graph                        copy2(g);

    // Adding to 'g' graph reshape tensor with *static input* tensor.
    TensorPtr static_input = TensorPtr(new Tensor(4U, w_sizes, syn_type_fixed, reinterpret_cast<char*>(weights)));
    static_input->setAsStaticParam(true);
    NodePtr node_to_remove = NodeFactory::createNode({static_input}, {wtensors[0]}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape_to_remove");
    GraphEditor::addNode(g, node_to_remove);
    g.compile();

    // Adding to 'copy1' graph reshape tensor with dynamic input tensor.
    const TSize non_static_tensor_sizes[] = { wOFM, nIFM, params.kH, params.kW};
    const NodeVector& inputs1                   = copy1.getExeSortedNodes();
    TensorPtr      not_static_input =
        TensorPtr(new Tensor(4U, non_static_tensor_sizes, syn_type_fixed, reinterpret_cast<char*>(weights)));
    NodePtr node_not_to_remove = NodeFactory::createNode({not_static_input}, {inputs1.front()->getInput(0)}, nullptr, NodeFactory::reshapeNodeTypeName,"reshape_not_to_remove");
    GraphEditor::addNode(copy1, node_not_to_remove);
    copy1.compile();

    // Adding to 'copy2' graph reshape tensor with dynamic input tensor.
    const NodeVector& inputs2            = copy2.getExeSortedNodes();
    TensorPtr new_inter = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed));
    TensorPtr new_input_tensor_a = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed, reinterpret_cast<char*>(ifm)));
    TensorPtr new_input_tensor_b = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed, reinterpret_cast<char*>(ifm)));
    TensorPtr add_1_output = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed));
    TensorPtr reshape_output = TensorPtr(new Tensor(4U, i_sizes, syn_type_fixed));

    TensorPtr src_input_tensor = inputs2.front()->getInput(0);

    new_input_tensor_b->setAsStaticParam(true);

    NodePtr add_1   = NodeFactory::createNode({new_input_tensor_a, new_input_tensor_b}, {add_1_output}, nullptr, NodeFactory::addNodeTypeName, "");
    NodePtr node_to_remove_2_a = NodeFactory::createNode({new_input_tensor_b}, {reshape_output}, nullptr, NodeFactory::reshapeNodeTypeName ,"reshape_to_remove");
    NodePtr add_2   = NodeFactory::createNode({add_1_output, reshape_output}, {src_input_tensor}, nullptr, NodeFactory::addNodeTypeName, "");

    GraphEditor::addNode(copy2, add_1);
    GraphEditor::addNode(copy2, node_to_remove_2_a);
    GraphEditor::addNode(copy2, add_2);
    copy2.compile();

    const NodeVector& g_all_nodes      = g.getExeSortedNodes();
    const NodeVector& copy1_all_nodes  = copy1.getExeSortedNodes();
    auto copy1_all_inputs = copy1.getGraphInputs();
    auto copy2_all_inputs = copy2.getGraphInputs();

    ASSERT_EQ(g_all_nodes.size(), SIZE + 2); // +1 for DMA out, as the input is static now (reshape should be deleted)

    ASSERT_EQ(copy1_all_nodes.size(), SIZE + 2 + 1); // '+ 2' for DMA nodes +1 for added reshape
    ASSERT_EQ(copy1_all_inputs.size(), 4); // 3 weight tensor + 1 static input
    ASSERT_EQ(copy2_all_inputs.size(), 6); // 3 weight tensor + 1 static input (which was not deleted) + 2 new inputs

}

void permuteShape(TSize* inSizes, TSize* outSizes, unsigned dimNum, TransposePermutationDim* perm)
{
    for (unsigned int index = 0; index < dimNum; ++index)
    {
        outSizes[index] = inSizes[perm[index]];
    }
}

TEST_F(PASSES, DISABLED_contiguous_transposes_pass)
{

    TSize input_dimensions[] = {6,3,4,2};
    TransposePermutationArray permutation({TPD_Channel, TPD_Width, TPD_Height, TPD_4Dim_Batch });

    const unsigned int dim_num = permutation.size();

    const int INPUT_SIZE =  std::accumulate (input_dimensions,
                                             input_dimensions + dim_num,
                                             1, // initial acc
                                             [](unsigned int acc, unsigned int dim){return acc * dim;});

    int8_t* inputArray = new int8_t [INPUT_SIZE];
    std::generate (inputArray, inputArray + INPUT_SIZE, Test_Random_Number_Creator (std::array<int, 2>({0,10})));
    TensorPtr IFM     =
        TensorPtr(new Tensor(dim_num, input_dimensions,  syn_type_fixed, reinterpret_cast<char*>(inputArray)));

    Gaudi2Graph g;

    TransposePermutationDim perm[MAX_DIMENSIONS_NUM] = {TPD_Channel, TPD_Width, TPD_Height, TPD_4Dim_Batch, TBD_5DimSize};
    TransposePermutationDim before[MAX_DIMENSIONS_NUM] = {TPD_Width, TPD_4Dim_Batch, TPD_Channel, TPD_Height, TBD_5DimSize};
    TransposePermutationDim after[MAX_DIMENSIONS_NUM] = {TPD_Height, TPD_Channel, TPD_4Dim_Batch, TPD_Width, TBD_5DimSize };
    TransposePermutationDim identity[MAX_DIMENSIONS_NUM] = {TPD_Channel, TPD_Width, TPD_Height, TPD_4Dim_Batch, TBD_5DimSize };
    TransposePermutationDim messA[MAX_DIMENSIONS_NUM] = {TPD_Width,TPD_Channel, TPD_4Dim_Batch, TPD_Height, TBD_5DimSize };
    TransposePermutationDim messB[MAX_DIMENSIONS_NUM] = {TPD_Width,TPD_Channel, TPD_Height, TPD_4Dim_Batch, TBD_5DimSize };

    synTransposeParams tranposeParams_before, tranposeParams_after, tranposeParams_identity,
                       tranposeParams_messA, tranposeParams_messB, tranposeParams;

    std::copy(std::begin(before), std::end(before), std::begin(tranposeParams_before.permutation));
    std::copy(std::begin(after), std::end(after ), std::begin(tranposeParams_after.permutation));
    std::copy(std::begin(identity), std::end(identity), std::begin(tranposeParams_identity.permutation));
    std::copy(std::begin(messA), std::end(messA), std::begin(tranposeParams_messA.permutation));
    std::copy(std::begin(messB), std::end(messB), std::begin(tranposeParams_messB.permutation));
    std::copy(std::begin(perm), std::end(perm), std::begin(tranposeParams.permutation));

    tranposeParams_before.tensorDim = 4;
    tranposeParams_after.tensorDim = 4;
    tranposeParams_identity.tensorDim = 4;
    tranposeParams_messA.tensorDim = 4;
    tranposeParams_messB.tensorDim = 4;
    tranposeParams.tensorDim = 4;

    TSize INT1_sizes[dim_num];
    permuteShape(input_dimensions, INT1_sizes, dim_num, tranposeParams_before.permutation);
    TSize INT2_sizes[dim_num];
    permuteShape(INT1_sizes, INT2_sizes, dim_num, tranposeParams_after.permutation);
    TSize INT3_sizes[dim_num];
    permuteShape(INT2_sizes, INT3_sizes, dim_num, tranposeParams_before.permutation);
    TSize INT4_0_sizes[dim_num];
    permuteShape(INT3_sizes, INT4_0_sizes, dim_num, tranposeParams_identity.permutation);
    TSize INT4_sizes[dim_num];
    permuteShape(INT4_0_sizes, INT4_sizes, dim_num, tranposeParams_after.permutation);
    TSize INT4_2_sizes[dim_num];
    permuteShape(INT4_sizes, INT4_2_sizes, dim_num, tranposeParams_messA.permutation);
    TSize INT5_sizes[dim_num];
    permuteShape(INT4_2_sizes, INT5_sizes, dim_num, tranposeParams_messB.permutation);
    TSize OFM_sizes[dim_num];
    permuteShape(INT5_sizes, OFM_sizes, dim_num, tranposeParams.permutation);

    TensorVector INT1, INT2, INT3, INT4, INT4_0, INT4_2, INT5, IFM_V, OFM_V;
    INT1.push_back(TensorPtr(new Tensor(dim_num, INT1_sizes,  syn_type_fixed)));
    INT2.push_back(TensorPtr(new Tensor(dim_num, INT2_sizes,  syn_type_fixed)));
    INT3.push_back(TensorPtr(new Tensor(dim_num, INT3_sizes,  syn_type_fixed)));
    INT4.push_back(TensorPtr(new Tensor(dim_num, INT4_sizes,  syn_type_fixed)));
    INT5.push_back(TensorPtr(new Tensor(dim_num, INT5_sizes,  syn_type_fixed)));
    INT4_0.push_back(TensorPtr(new Tensor(dim_num, INT4_0_sizes,  syn_type_fixed)));
    INT4_2.push_back(TensorPtr(new Tensor(dim_num, INT4_2_sizes,  syn_type_fixed)));
    OFM_V.push_back(TensorPtr(new Tensor(dim_num, OFM_sizes,  syn_type_fixed)));
    IFM_V.push_back(IFM);

    NodeVector transposeNodes;
    transposeNodes.push_back(NodeFactory::createNode(IFM_V, INT1, &tranposeParams_before, "transpose", "T1"));
    transposeNodes.push_back(NodeFactory::createNode(INT1, INT2, &tranposeParams_after, "transpose","T2"));
    transposeNodes.push_back(NodeFactory::createNode(INT2, INT3, &tranposeParams_before, "transpose","T3"));
    transposeNodes.push_back(NodeFactory::createNode(INT3, INT4_0, &tranposeParams_identity, "transpose","T4_0"));
    transposeNodes.push_back(NodeFactory::createNode(INT4_0, INT4, &tranposeParams_after, "transpose","T4"));
    transposeNodes.push_back(NodeFactory::createNode(INT4, INT4_2, &tranposeParams_messA, "transpose","T4_2"));
    transposeNodes.push_back(NodeFactory::createNode(INT4_2, INT5, &tranposeParams_messB, "transpose","T5"));
    transposeNodes.push_back(NodeFactory::createNode(INT5, OFM_V, &tranposeParams, "transpose","T6"));

    for (auto node : transposeNodes)
    {
        GraphEditor::addNode(g, node);
    }

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";


    std::vector<std::string> listOfNotRemoved{"T4_2", "T5"};
    std::vector<std::string> listOfRemovedNodes{"T1", "T2", "T3", "T4_0", "T4", "T6"};

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 7);  // 2 DMA, 1 memcopy , 2 transposes extracted to 2*2 (T4_2, T5),
                                 // 1 reshape node replaces the transpose (T6)
                                 // and one memcpy at the end of the graph

    int numOfFoundedNames = 0;
    int numOfRemovedNodesThatWasFound = 0;

    for (NodePtr node : nodes)
    {
        string name = node->getNodeName();
        if (std::find(std::begin(listOfNotRemoved), std::end(listOfNotRemoved), name) != std::end(listOfNotRemoved))
        {
            ++numOfFoundedNames;
        }

        if (std::find(std::begin(listOfRemovedNodes), std::end(listOfRemovedNodes), name) != std::end(listOfRemovedNodes))
        {
            ++numOfRemovedNodesThatWasFound;
        }
    }
    ASSERT_EQ(numOfFoundedNames, 2) << "Wrong node was removed";
    ASSERT_EQ(numOfRemovedNodesThatWasFound, 0) << "Not all nodes were removed";
    delete[] inputArray;
}

TEST_F(PASSES, DISABLED_mask_invalid_softmax_pass)
{
    Gaudi2Graph              g;
    g.setInferenceMode(true);
    TSize dims1[2] = { 1, 1 };
    TSize dims2[1] = { 1 };
    int16_t buff1[1][1];
    int16_t buff2[1];
    TensorPtr               smInput1   = TensorPtr(new Tensor(2, dims1, syn_type_int16, reinterpret_cast<char*>(buff1)));
    TensorPtr               smInput2   = TensorPtr(new Tensor(1, dims2, syn_type_int16, reinterpret_cast<char*>(buff2)));
    TensorPtr               smOutput   = TensorPtr(new Tensor(2, dims1, syn_type_int16));
    ns_SequenceMask::Params smParams;
    smParams.mask_value = -9999;
    smParams.use_sequence_length = 1;
    NodePtr smNode    = NodeFactory::createGenericTPCNode({smInput1, smInput2}, {smOutput}, nullptr, "sequence_mask_i16", "");
    TPCNodePtr sequenceMaxTpcNode = std::dynamic_pointer_cast<TPCNode>(smNode);
    sequenceMaxTpcNode->storeParamsInBuffer(&smParams, sizeof(ns_SequenceMask::Params));

    TensorPtr          softmaxOutput   = TensorPtr(new Tensor(2, dims1, syn_type_int16));
    ns_Softmax::Params softmaxParams;
    softmaxParams.dim = 0;
    NodePtr  softmaxNode    = NodeFactory::createGenericTPCNode({smOutput}, {softmaxOutput}, nullptr, "softmax_i16", "");
    TPCNodePtr softMaxtpcNode = std::dynamic_pointer_cast<TPCNode>(softmaxNode);
    softMaxtpcNode->storeParamsInBuffer(&softmaxParams, sizeof(ns_Softmax::Params));

    GraphEditor::addNode(g, smNode);
    GraphEditor::addNode(g, softmaxNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";
    const NodeVector& nodes = g.getExeSortedNodes();
    // 1 tpc node, 3 dma nodes for 2 inputs and 1 output.
    ASSERT_EQ(nodes.size(), 4) << "Unexpected graph size";
    NodePtr  node = *std::next(nodes.begin(), 2);
    TPCNodePtr maskInvalidSoftmaxNode = std::dynamic_pointer_cast<TPCNode>(node);
    ASSERT_TRUE(std::string(maskInvalidSoftmaxNode->getGUID()) == std::string("softmax_i16"));
    // ASSERT_EQ(node->getNumInputs(), 3); // add the aux tensor
    ASSERT_EQ(node->getNumOutputs(), 1);
    ASSERT_EQ(node->getInput(0), smInput1);
    ASSERT_EQ(node->getInput(1), smInput2);
    ASSERT_EQ(node->getOutput(0), softmaxOutput);
}

// Temp disable these test - SW-54573 to track isolation

class GOYA_FUSE_PAD_CONV_PASSES : public FUSE_PAD_CONV_PASSES<Gaudi2Graph>
{
};

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_conv_symmetric_goya)
{
    fuse_pad_into_conv_symmetric();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_conv_asymmetric_goya)
{
    fuse_pad_into_conv_asymmetric();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_non_zero_pad_into_padded_conv_goya)
{
    fuse_non_zero_pad_into_padded_conv();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_non_zero_pad_into_int8_conv_goya)
{
    fuse_non_zero_pad_into_int8_conv();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_non_zero_pad_into_int16_conv_goya)
{
    fuse_non_zero_pad_into_int16_conv();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_conv_non_spatial_goya)
{
    fuse_pad_into_conv_non_spatial();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_max_pool_symmetric_goya)
{
    fuse_pad_into_max_pool_symmetric();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_max_pool_asymmetric_goya)
{
    fuse_pad_into_max_pool_asymmetric();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_avg_pool_symmetric_goya)
{
    fuse_pad_into_avg_pool_symmetric();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_conv_pad_into_max_pool_remove_cast_goya)
{
    fuse_conv_pad_into_max_pool_remove_cast();
}

/*
in the following test,  the output of the mme goes into both pad and another mme,
 and so the cast is inserted only on the direction of the pad, but the original tensor before the
 cast has to remain since it goes into the other mme
*/
TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_conv_pad_into_max_pool_keep_cast_goya)
{
    fuse_conv_pad_into_max_pool_keep_cast();
}

TEST_F(GOYA_FUSE_PAD_CONV_PASSES, DISABLED_fuse_pad_into_avg_pool_asymmetric_goya)
{
    fuse_pad_into_avg_pool_asymmetric();
}

TEST_F(PASSES, DISABLED_merge_casts_between_logicalNode)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const unsigned n     = 1;
    const unsigned w     = 3;
    const unsigned h     = 3;
    const unsigned batch = 1;

    char tesnor1[n * w * h * batch];

    const TSize sizes1[] = { n, w, h, batch };
    const TSize sizes2[] = { w, n, h, batch };

    TensorPtr T1 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T1->setName("T1");
    TensorPtr T2 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T2->setName("T2");
    TensorPtr T3 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T3->setName("T3");
    TensorPtr T4 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T4->setName("T4");

    NodePtr relu3 = NodeFactory::createGenericTPCNode({T1}, {T2}, nullptr, "relu_i16", "relu_1");
    GraphEditor::addNode(g, relu3);

    NodePtr reshape1 = NodeFactory::createNode({T2}, {T3}, nullptr, "reshape", "reshape_1");
    GraphEditor::addNode(g, reshape1);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({T3}, {T4}, nullptr, "relu_f32", "relu_2");
    GraphEditor::addNode(g, relu4);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned int castNodeCount = 0;
    for (const auto &node : g.getExeSortedNodes())
    {
        if ((node != nullptr)
            && (node->isCast()))
        {
            ++castNodeCount;
        }
    }
    ASSERT_EQ(castNodeCount, 3) << "{} Unexpected cast nodes found in graph" << castNodeCount;
}

TEST_F(PASSES, DISABLED_merge_casts_between_logicalNode_multi_producers)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char tesnor1[n * w * h * batch];
    char tesnor2[2* n * w * h * batch];

    const TSize sizes1[] = { n, w, h, batch };
    const TSize sizes2[] = { 2*n, w, h, batch };

    TensorPtr T1 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T1->setName("T1");
    TensorPtr T2 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T2->setName("T2");
    TensorPtr T3 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T3->setName("T3");
    TensorPtr T4 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T4->setName("T4");
    TensorPtr T5 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T3->setName("T5");
    TensorPtr T6 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T4->setName("T6");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({T1}, {T2}, nullptr, "relu_f32", "relu_1");
    GraphEditor::addNode(g, relu1);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({T3}, {T4}, nullptr, "relu_f32", "relu_2");
    GraphEditor::addNode(g, relu2);

    NodePtr concat = NodeFactory::createNode({T2, T4}, {T5}, nullptr, "concat", "concat_1");
    GraphEditor::addNode(g, concat);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({T5}, {T6}, nullptr, "relu_i16", "relu_3");
    GraphEditor::addNode(g, relu3);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned int castNodeCount = 0;
    for (const auto &node : g.getExeSortedNodes())
    {
        if ((node != nullptr)
            && (node->isCast()))
        {
            ++castNodeCount;
        }
    }
    ASSERT_EQ(castNodeCount, 5) << "{} Unexpected cast nodes found in graph" << castNodeCount;
}

TEST_F(PASSES, DISABLED_merge_casts_between_logicalNode_multi_consumers)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char tesnor1[2*n * w * h * batch];
    char tesnor2[n * w * h * batch];
    char tesnor3[n * w * h * batch];

    const TSize sizes1[] = { n, 2*w, h, batch };
    const TSize sizes2[] = { n, w, h, batch };

    TensorPtr T1 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T1->setName("T1");
    TensorPtr T2 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T2->setName("T2");
    TensorPtr T3 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T3->setName("T3");
    TensorPtr T4 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor3)));
    T4->setName("T4");
    TensorPtr T5 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T3->setName("T5");
    TensorPtr T6 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T4->setName("T6");

    unsigned int splitParams= 1;
    NodePtr      relu1 = NodeFactory::createGenericTPCNode({T1}, {T2}, nullptr, "relu_i16", "relu_1");
    GraphEditor::addNode(g, relu1);

    NodePtr split = NodeFactory::createNode({T2}, {T3, T4}, &splitParams, "split", "split");
    GraphEditor::addNode(g, split);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({T3}, {T5}, nullptr, "relu_f32", "relu_2");
    GraphEditor::addNode(g, relu2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({T4}, {T6}, nullptr, "relu_f32", "relu_3");
    GraphEditor::addNode(g, relu3);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned int castNodeCount = 0;
    for (const auto &node : g.getExeSortedNodes())
    {
        if ((node != nullptr)
            && (node->isCast()))
        {
            ++castNodeCount;
        }
    }
    ASSERT_EQ(castNodeCount, 4) << "{} Unexpected cast nodes found in graph" << castNodeCount;
}

TEST_F(PASSES, DISABLED_redundant_casts_between_logicalNode)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char tesnor1[n * w * h * batch];

    const TSize sizes1[] = { n, w, h, batch };
    const TSize sizes2[] = { w, n, h, batch };

    TensorPtr T1 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T1->setName("T1");
    TensorPtr T2 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T2->setName("T2");
    TensorPtr T3 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T3->setName("T3");
    TensorPtr T4 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T4->setName("T4");

    NodePtr relu3 = NodeFactory::createGenericTPCNode({T1}, {T2}, nullptr, "relu_f32", "relu_1");
    GraphEditor::addNode(g, relu3);

    NodePtr reshape1 = NodeFactory::createNode({T2}, {T3}, nullptr, "reshape", "reshape_1");
    GraphEditor::addNode(g, reshape1);

    NodePtr relu4 = NodeFactory::createGenericTPCNode({T3}, {T4}, nullptr, "relu_f32", "relu_2");
    GraphEditor::addNode(g, relu4);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned int castNodeCount = 0;
    for (const auto &node : g.getExeSortedNodes())
    {
        if ((node != nullptr)
            && (node->isCast()))
        {
            ++castNodeCount;
        }
    }
    ASSERT_EQ(castNodeCount, 2) << "{} Unexpected cast nodes found in graph" << castNodeCount;
}

TEST_F(PASSES, DISABLED_redundant_casts_between_logicalNode_multi_consumers)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char tesnor1[2*n * w * h * batch];
    char tesnor2[n * w * h * batch];
    char tesnor3[n * w * h * batch];

    const TSize sizes1[] = { n, 2*w, h, batch };
    const TSize sizes2[] = { n, w, h, batch };

    TensorPtr T1 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T1->setName("T1");
    TensorPtr T2 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T2->setName("T2");
    TensorPtr T3 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T3->setName("T3");
    TensorPtr T4 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor3)));
    T4->setName("T4");
    TensorPtr T5 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T3->setName("T5");
    TensorPtr T6 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T4->setName("T6");

    unsigned int splitParams= 1;
    NodePtr      relu1 = NodeFactory::createGenericTPCNode({T1}, {T2}, nullptr, "relu_f32", "relu_1");
    GraphEditor::addNode(g, relu1);

    NodePtr split = NodeFactory::createNode({T2}, {T3, T4}, &splitParams, "split", "split");
    GraphEditor::addNode(g, split);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({T3}, {T5}, nullptr, "relu_f32", "relu_2");
    GraphEditor::addNode(g, relu2);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({T4}, {T6}, nullptr, "relu_f32", "relu_3");
    GraphEditor::addNode(g, relu3);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned int castNodeCount = 0;
    for (const auto &node : g.getExeSortedNodes())
    {
        if ((node != nullptr)
            && (node->isCast()))
        {
            ++castNodeCount;
        }
    }
    ASSERT_EQ(castNodeCount, 3) << "{} Unexpected cast nodes found in graph" << castNodeCount;
}

TEST_F(PASSES, DISABLED_redundant_casts_between_logicalNode_multi_producers)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize n     = 1;
    const TSize w     = 3;
    const TSize h     = 3;
    const TSize batch = 1;

    char tesnor1[n * w * h * batch];
    char tesnor2[2* n * w * h * batch];

    const TSize sizes1[] = { n, w, h, batch };
    const TSize sizes2[] = { 2*n, w, h, batch };

    TensorPtr T1 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T1->setName("T1");
    TensorPtr T2 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T2->setName("T2");
    TensorPtr T3 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T3->setName("T3");
    TensorPtr T4 = TensorPtr(new Tensor(4U, sizes1, syn_type_fixed, reinterpret_cast<char*>(tesnor1)));
    T4->setName("T4");
    TensorPtr T5 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T3->setName("T5");
    TensorPtr T6 = TensorPtr(new Tensor(4U, sizes2, syn_type_fixed, reinterpret_cast<char*>(tesnor2)));
    T4->setName("T6");

    NodePtr relu1 = NodeFactory::createGenericTPCNode({T1}, {T2}, nullptr, "relu_f32", "relu_1");
    GraphEditor::addNode(g, relu1);

    NodePtr relu2 = NodeFactory::createGenericTPCNode({T3}, {T4}, nullptr, "relu_f32", "relu_2");
    GraphEditor::addNode(g, relu2);

    NodePtr reshape1 = NodeFactory::createNode({T2, T4}, {T5}, nullptr, "concat", "concat_1");
    GraphEditor::addNode(g, reshape1);

    NodePtr relu3 = NodeFactory::createGenericTPCNode({T5}, {T6}, nullptr, "relu_f32", "relu_3");
    GraphEditor::addNode(g, relu3);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    unsigned int castNodeCount = 0;
    for (const auto &node : g.getExeSortedNodes())
    {
        if ((node != nullptr)
            && (node->isCast()))
        {
            ++castNodeCount;
        }
    }
    ASSERT_EQ(castNodeCount, 3) << "{} Unexpected cast nodes found in graph" << castNodeCount;
}

TEST_F(PASSES, DISABLED_merge_two_contiguous_cast_node)
{
    static const char* i16tof32KernelName = "cast_i16_to_f32";
    static const char* f32toi8KernelName  = "cast_f32_to_i8";
    static const char* mergedKernelName   = "cast_i16_to_i8";
    static const char* notKernelName  = "not_i8";

    TSize n = 4;
    TSize c = 4;

    const unsigned TOTAL_TENSOR_SIZE  = n * c;

    int16_t*  firstCastInputData   = new int16_t[TOTAL_TENSOR_SIZE];
    float_t*  firstCastOutputData  = new float_t[TOTAL_TENSOR_SIZE];
    int8_t*   secondCastOutputData = new int8_t[TOTAL_TENSOR_SIZE];
    int8_t*   notOutputData        = new int8_t[TOTAL_TENSOR_SIZE];

    const TSize tensorSize[]  = {n,c};

    bool ret;

    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setRecipeName("mergeCasts");
    Node::NodeProperties p;

    // Create cast node i16_to_f32
    TensorPtr       firstCastInputTensor  = std::make_shared<Tensor>(2U, tensorSize, synDataType::syn_type_int16, reinterpret_cast<char*>(firstCastInputData));
    TensorPtr       firstCastOutptuTensor = std::make_shared<Tensor>(2U, tensorSize, synDataType::syn_type_single, reinterpret_cast<char*>(firstCastOutputData));
    NodePtr         firstCastNode = NodeFactory::createNode({firstCastInputTensor}, {firstCastOutptuTensor}, nullptr, i16tof32KernelName, "firstCast", p);
    NodeAnnotation& firstNodeAnnotation = firstCastNode->getNodeAnnotation();
    firstNodeAnnotation.insertedNode = true; // to make sure node could be removed by 'removeRedundantCastNodes'

    // Create cast node f32_to_i8
    TensorPtr       secondCastOutptuTensor =  std::make_shared<Tensor>(2U, tensorSize, synDataType::syn_type_fixed, reinterpret_cast<char*>(secondCastOutputData));
    NodePtr         secondCastNode = NodeFactory::createNode({firstCastOutptuTensor}, {secondCastOutptuTensor}, nullptr, f32toi8KernelName, "secondCast", p);
    NodeAnnotation& secondNodeAnnotation = secondCastNode->getNodeAnnotation();
    secondNodeAnnotation.insertedNode = true; // to make sure node could be removed by 'removeRedundantCastNodes'

    // Create not node
    TensorPtr notOutputTensor =  std::make_shared<Tensor>(2U, tensorSize, synDataType::syn_type_fixed, reinterpret_cast<char*>(notOutputData));
    NodePtr   notNode = NodeFactory::createNode({secondCastOutptuTensor}, {notOutputTensor}, nullptr, notKernelName, "not", p);

    GraphEditor::addNode(g, firstCastNode);
    GraphEditor::addNode(g, secondCastNode);
    GraphEditor::addNode(g, notNode);

    ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph with merged casts";

    const NodeVector& nodes = g.getExeSortedNodes();

    ASSERT_EQ(nodes.size(), 4); // 1 merged cast node one not node + 2 DMA nodes
    bool firstTpcNode = true;

    for (const NodePtr& n : nodes)
    {
        TPCNodePtr tpc = std::dynamic_pointer_cast<TPCNode>(n);
        if (tpc != nullptr)
        {
            if (firstTpcNode)
            {
                firstTpcNode = false;
                ASSERT_TRUE(tpc->getGUID() == mergedKernelName);
            }
            else
            {
                ASSERT_TRUE(tpc->getGUID() == notKernelName);
            }
        }
    }

    std::string gcfgUsedFileName = g.getRecipeName() + ".used";
    std::remove(gcfgUsedFileName.c_str());

    delete[] firstCastInputData;
    delete[] firstCastOutputData;
    delete[] secondCastOutputData;
    delete[] notOutputData;
}

TEST_F(PASSES, DISABLED_remove_unrequired_requant_nodes)
{
    static const char* not16KernelName     = "not_i16";
    static const char* i16toi8KernelName   = "cast_i16_to_i8";
    static const char* not8KernelName      = "not_i8";
    static const char* requant8KernelName  = "requant_i8";
    static const char* requant16KernelName = "requant_i16";
    static const char* i8toi16KernelName   = "cast_i8_to_i16";

    TSize n = 4;
    TSize c = 4;

    const TSize TOTAL_TENSOR_SIZE = n * c;

    // Creating:  {NOT-1} -> {CAST-1} -> {REQUANT-1} -> {NOT-2} -> {REQUANT-2} -> {CAST-2} -> {NOT-3} -> {REQUANT-3} -> {RESHAPE-1, NOT-3}
    // Expecting: {NOT-1} -> {CAST-1} -> {NOT-2} -> {CAST-2} -> {NOT-3} -> {RESHAPE-1, NOT-4}
    int16_t* not1InputData      = new int16_t[TOTAL_TENSOR_SIZE];
    int16_t* not1OutputData     = new int16_t[TOTAL_TENSOR_SIZE];
    int8_t * cast1OutputData    = new int8_t[TOTAL_TENSOR_SIZE];
    int8_t * requant1OutputData = new int8_t[TOTAL_TENSOR_SIZE];
    int8_t * not2OutputData     = new int8_t[TOTAL_TENSOR_SIZE];
    int8_t * requant2OutputData = new int8_t[TOTAL_TENSOR_SIZE];
    int16_t* cast2OutputData    = new int16_t[TOTAL_TENSOR_SIZE];
    int16_t* not3OutputData     = new int16_t[TOTAL_TENSOR_SIZE];
    int16_t* requant3OutputData = new int16_t[TOTAL_TENSOR_SIZE];
    int16_t* reshape1OutputData = new int16_t[TOTAL_TENSOR_SIZE];
    int16_t* not4OutputData     = new int16_t[TOTAL_TENSOR_SIZE];

    const TSize tensorSize[] = {n, c};
    const TSize reshapeOutTensorSize[] = {n, sqrt(c), sqrt(c)};

    bool                              ret;

    Gaudi2Graph g;
    g.setInferenceMode(true);
    g.setRecipeName("mergeCasts");
    Node::NodeProperties p;

    // Create NOT-1 node
    TensorPtr not1InputTensor  = std::make_shared<Tensor>(2U,
                                                        tensorSize,
                                                        synDataType::syn_type_int16,
                                                        reinterpret_cast<char*>(not1InputData));
    TensorPtr not1OutputTensor = std::make_shared<Tensor>(2U,
                                                        tensorSize,
                                                        synDataType::syn_type_int16,
                                                        reinterpret_cast<char*>(not1OutputData));
    NodePtr   not1Node         = NodeFactory::createNode({not1InputTensor},
                                                       {not1OutputTensor},
                                                       nullptr,
                                                       not16KernelName,
                                                       "not1",
                                                       p);

    // Create cast-1 node i16_to_i8
    TensorPtr cast1OutputTensor = std::make_shared<Tensor>(2U,
                                                         tensorSize,
                                                         synDataType::syn_type_int8,
                                                         reinterpret_cast<char*>(cast1OutputData));
    NodePtr   cast1Node         = NodeFactory::createNode({not1OutputTensor},
                                                        {cast1OutputTensor},
                                                        nullptr,
                                                        i16toi8KernelName,
                                                        "cast1",
                                                        p);
    NodeAnnotation& cast1NodeAnnotation = cast1Node->getNodeAnnotation();
    cast1NodeAnnotation.insertedNode = true; // to make sure node could be removed by 'removeRedundantCastNodes'

    // Create requant-1 node
    TensorPtr requant1OutputTensor = std::make_shared<Tensor>(2U,
                                                            tensorSize,
                                                            synDataType::syn_type_int8,
                                                            reinterpret_cast<char*>(requant1OutputData));
    NodePtr   requant1Node         = NodeFactory::createNode({cast1OutputTensor},
                                                           {requant1OutputTensor},
                                                           nullptr,
                                                           requant8KernelName,
                                                           "requant1",
                                                           p);

    // Create NOT-2 node
    TensorPtr not2OutputTensor = std::make_shared<Tensor>(2U,
                                                        tensorSize,
                                                        synDataType::syn_type_int8,
                                                        reinterpret_cast<char*>(not2OutputData));
    NodePtr   not2Node         = NodeFactory::createNode({requant1OutputTensor},
                                                       {not2OutputTensor},
                                                       nullptr,
                                                       not8KernelName,
                                                       "not2",
                                                       p);

    // Create requant-2 node
    TensorPtr requant2OutputTensor = std::make_shared<Tensor>(2U,
                                                            tensorSize,
                                                            synDataType::syn_type_int8,
                                                            reinterpret_cast<char*>(requant2OutputData));
    NodePtr   requant2Node         = NodeFactory::createNode({not2OutputTensor},
                                                           {requant2OutputTensor},
                                                           nullptr,
                                                           requant8KernelName,
                                                           "requant2",
                                                           p);

    // Create cast-2 node i8_to_i16
    TensorPtr cast2OutputTensor = std::make_shared<Tensor>(2U,
                                                         tensorSize,
                                                         synDataType::syn_type_int16,
                                                         reinterpret_cast<char*>(cast2OutputData));
    NodePtr   cast2Node         = NodeFactory::createNode({requant2OutputTensor},
                                                        {cast2OutputTensor},
                                                        nullptr,
                                                        i8toi16KernelName,
                                                        "cast2",
                                                        p);
    NodeAnnotation& cast2NodeAnnotation = cast2Node->getNodeAnnotation();
    cast2NodeAnnotation.insertedNode = true; // to make sure node could be removed by 'removeRedundantCastNodes'

    // Create NOT-3 node
    TensorPtr not3OutputTensor = std::make_shared<Tensor>(2U,
                                                        tensorSize,
                                                        synDataType::syn_type_int16,
                                                        reinterpret_cast<char*>(not3OutputData));
    NodePtr   not3Node         = NodeFactory::createNode({cast2OutputTensor},
                                                       {not3OutputTensor},
                                                       nullptr,
                                                       not16KernelName,
                                                       "not3",
                                                       p);

    // Create REQ-3 node
    TensorPtr req3OutputTensor = std::make_shared<Tensor>(2U,
                                                        tensorSize,
                                                        synDataType::syn_type_int16,
                                                        reinterpret_cast<char*>(requant3OutputData));
    NodePtr   requant3Node     = NodeFactory::createNode({not3OutputTensor},
                                                       {req3OutputTensor},
                                                       nullptr,
                                                       requant16KernelName,
                                                       "requant3",
                                                       p);

    // Create RESHAPE-1
    TensorPtr reshape1OutputTensor = std::make_shared<Tensor>(3U,
                                                            reshapeOutTensorSize,
                                                            synDataType::syn_type_int16,
                                                            reinterpret_cast<char*>(reshape1OutputData));
    NodePtr   reshape1Node         = NodeFactory::createNode({req3OutputTensor},
                                                           {reshape1OutputTensor},
                                                           nullptr,
                                                           NodeFactory::reshapeNodeTypeName,
                                                           "reshape1");

    // Create NOT-4
    TensorPtr not4OutputTensor = std::make_shared<Tensor>(2U,
                                                        tensorSize,
                                                        synDataType::syn_type_int16,
                                                        reinterpret_cast<char*>(not4OutputData));
    NodePtr   not4Node         = NodeFactory::createNode({req3OutputTensor},
                                                       {not4OutputTensor},
                                                       nullptr,
                                                       not16KernelName,
                                                       "not3",
                                                       p);

    GraphEditor::addNode(g, not1Node);
    GraphEditor::addNode(g, cast1Node);
    GraphEditor::addNode(g, requant1Node);
    GraphEditor::addNode(g, not2Node);
    GraphEditor::addNode(g, requant2Node);
    GraphEditor::addNode(g, cast2Node);
    GraphEditor::addNode(g, not3Node);
    GraphEditor::addNode(g, requant3Node);
    GraphEditor::addNode(g, reshape1Node);
    GraphEditor::addNode(g, not4Node);

    ret = g.compile();

    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    string unexpectedNodeNames[] = {"requant1", "requant2", "requant3"};

    for (const NodePtr& node : nodes)
    {
        for (string unexpectedName : unexpectedNodeNames)
        {
            ASSERT_NE(node->getNodeName(), unexpectedName);
        }
    }

    delete[] not1InputData;
    delete[] not1OutputData;
    delete[] cast1OutputData;
    delete[] requant1OutputData;
    delete[] not2OutputData;
    delete[] requant2OutputData;
    delete[] cast2OutputData;
    delete[] not3OutputData;
    delete[] requant3OutputData;
    delete[] reshape1OutputData;
    delete[] not4OutputData;
}

TEST_F(PASSES, DISABLED_scatter_reuse_tensor)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    TSize dims1[2] = {1, 5};
    TSize dims2[2] = {1, 2};
    int16_t data_buff[1][5];
    int16_t indices_buff[1][2];
    TensorPtr inputDataTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single, reinterpret_cast<char*>(data_buff)));
    TensorPtr inputIndices = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_int32, reinterpret_cast<char*>(indices_buff)));
    TensorPtr inputUpdates = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_single, reinterpret_cast<char*>(indices_buff)));
    TensorPtr outputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));

    NodePtr scatterNode = NodeFactory::createNode({inputDataTensor, inputIndices, inputUpdates}, {outputTensor},
                                                nullptr, "scatter_f32", "scatter_node");

    ns_ScatterKernel::Params params;
    params.axis = 1;
    TPCNodePtr scatterTpcNode = std::dynamic_pointer_cast<TPCNode>(scatterNode);
    scatterTpcNode->storeParamsInBuffer(&params, sizeof(ns_ScatterKernel::Params));

    GraphEditor::addNode(g, scatterNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    // 2 memcpy nodes, 1 tpc node, 4 dma nodes for 3 inputs and 1 output.
    ASSERT_EQ(nodes.size(), 7) << "Unexpected graph size";

    NodePtr memcpy_node_in = *std::next(nodes.begin(), 1);
    ASSERT_EQ(memcpy_node_in->getNodeTypeStr(), "memcpy_f32");

    NodePtr  node = *std::next(nodes.begin(), 4);
    TPCNodePtr tpcNode = std::dynamic_pointer_cast<TPCNode>(node);

    NodePtr memcpy_node_out = *std::next(nodes.begin(), 5);
    ASSERT_EQ(memcpy_node_out->getNodeTypeStr(), "memcpy_f32");

    //Memcpy node in
    ASSERT_EQ(memcpy_node_in->getInput(0), inputDataTensor);
    ASSERT_EQ(memcpy_node_in->getOutput(0), tpcNode->getInput(0));

    //Memcpy node out
    ASSERT_EQ(memcpy_node_out->getInput(0), tpcNode->getOutput(0));
    ASSERT_EQ(memcpy_node_out->getOutput(0), outputTensor);

    ASSERT_EQ(tpcNode->getNodeName(), "scatter_node");
    ASSERT_EQ(tpcNode->getNumInputs(), 3);
    ASSERT_EQ(tpcNode->getNumOutputs(), 1);
    ASSERT_EQ(tpcNode->getInput(1), inputIndices);
    ASSERT_EQ(tpcNode->getInput(2), inputUpdates);
    ASSERT_EQ(tpcNode->getOutput(0), memcpy_node_out->getInput(0));

    ASSERT_EQ((tpcNode->getOutput(0))->getAliasTensor(), tpcNode->getInput(0));
}

TEST_F(PASSES, DISABLED_state_update_reuse_tensor)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    TSize dimsBig[] = {1, 2, 4, 8};
    TSize dimsSmall[] = {1, 1, 4, 8};
    TSize dimsState[] {1};

    int16_t big_buff[1][2][4][8];
    int16_t small_buff[1][1][4][8];
    int16_t state_buff[1];

    TensorPtr inputBigTensor = std::make_shared<Tensor>(Tensor(4U, dimsBig, syn_type_int16, reinterpret_cast<char*>(big_buff)));
    TensorPtr inputSmallTensor = std::make_shared<Tensor>(Tensor(4U, dimsSmall, syn_type_int16, reinterpret_cast<char*>(small_buff)));
    TensorPtr inputStateTensor = std::make_shared<Tensor>(Tensor(1U, dimsState, syn_type_int16, reinterpret_cast<char*>(state_buff)));
    TensorPtr outputTensor = std::make_shared<Tensor>(Tensor(4U, dimsBig, syn_type_int16));

    NodePtr state_update_node = NodeFactory::createNode({inputBigTensor, inputSmallTensor, inputStateTensor}, {outputTensor},
                                                nullptr, "state_update_i16", "state_update");
    GraphEditor::addNode(g, state_update_node);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    // 2 memcpy nodes, 1 tpc node, 4 dma nodes for 3 inputs and 1 output.
    ASSERT_EQ(nodes.size(), 7) << "Unexpected graph size";

    NodePtr memcpy_node_in = *std::next(nodes.begin(), 1);
    ASSERT_EQ(memcpy_node_in->getNodeTypeStr(), "memcpy_i16");

    NodePtr  node = *std::next(nodes.begin(), 4);
    TPCNodePtr stateUpdateNode = std::dynamic_pointer_cast<TPCNode>(node);

    NodePtr memcpy_node_out = *std::next(nodes.begin(), 5);
    ASSERT_EQ(memcpy_node_out->getNodeTypeStr(), "memcpy_i16");

    //Memcpy node in
    ASSERT_EQ(memcpy_node_in->getInput(0), inputBigTensor);
    ASSERT_EQ(memcpy_node_in->getOutput(0), stateUpdateNode->getInput(0));

    //Memcpy node out
    ASSERT_EQ(memcpy_node_out->getInput(0), stateUpdateNode->getOutput(0));
    ASSERT_EQ(memcpy_node_out->getOutput(0), outputTensor);

    ASSERT_EQ(stateUpdateNode->getNodeName(), "state_update");
    ASSERT_EQ(stateUpdateNode->getNumInputs(), 3);
    ASSERT_EQ(stateUpdateNode->getNumOutputs(), 1);
    ASSERT_EQ(stateUpdateNode->getInput(1), inputSmallTensor);
    ASSERT_EQ(stateUpdateNode->getInput(2), inputStateTensor);
    ASSERT_EQ(stateUpdateNode->getOutput(0), memcpy_node_out->getInput(0));

    ASSERT_EQ((stateUpdateNode->getOutput(0))->getAliasTensor(), stateUpdateNode->getInput(0));
}

TEST_F(PASSES, DISABLED_scatter_multi_consumers_reuse)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    TSize dims1[2] = {1, 5};
    TSize dims2[2] = {1, 2};
    int16_t data_buff[1][5];
    int16_t indices_buff[1][2];
    TensorPtr    inputDataTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single, reinterpret_cast<char*>(data_buff)));
    TensorPtr inputIndices = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_int32, reinterpret_cast<char*>(indices_buff)));
    TensorPtr inputUpdates = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_single, reinterpret_cast<char*>(indices_buff)));
    TensorPtr outputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));
    TensorPtr outputSinTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));

    NodePtr scatterNode = NodeFactory::createNode({inputDataTensor, inputIndices, inputUpdates}, {outputTensor},
                                                nullptr, "scatter_f32", "scatter_node");

    NodePtr sinNode = NodeFactory::createNode({inputDataTensor}, {outputSinTensor},
                                            nullptr, "sin_f32", "sin_node");

    ns_ScatterKernel::Params params;
    params.axis = 1;
    TPCNodePtr scatterTpcNode = std::dynamic_pointer_cast<TPCNode>(scatterNode);
    scatterTpcNode->storeParamsInBuffer(&params, sizeof(ns_ScatterKernel::Params));

    GraphEditor::addNode(g, scatterNode);
    GraphEditor::addNode(g, sinNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    // 2 memcpy node, 1 scatter node, 5 dma scatter nodes (for 4 inputs and 1 output), 1 sin node, 1 dma sin output.
    ASSERT_EQ(nodes.size(), 9) << "Unexpected graph size";

    NodePtr  sin_node = *std::next(nodes.begin(), 1);
    TPCNodePtr sinGNode = std::dynamic_pointer_cast<TPCNode>(sin_node);
    ASSERT_EQ(sin_node->getNodeName(), "sin_node");

    NodePtr memcpy_node_in = *std::next(nodes.begin(), 3);
    ASSERT_EQ(memcpy_node_in->getNodeTypeStr(), "memcpy_f32");

    NodePtr  scatter_node = *std::next(nodes.begin(), 6);
    TPCNodePtr scatterGNode = std::dynamic_pointer_cast<TPCNode>(scatter_node);
    ASSERT_EQ(scatterGNode->getNodeName(), "scatter_node");

    NodePtr memcpy_node_out = *std::next(nodes.begin(), 7);
    ASSERT_EQ(memcpy_node_out->getNodeTypeStr(), "memcpy_f32");

    //Memcpy node in
    ASSERT_EQ(memcpy_node_in->getInput(0), inputDataTensor);
    ASSERT_EQ(memcpy_node_in->getOutput(0), scatter_node->getInput(0));

    //Memcpy node out
    ASSERT_EQ(memcpy_node_out->getInput(0), scatter_node->getOutput(0));
    ASSERT_EQ(memcpy_node_out->getOutput(0), outputTensor);

    //Sin node
    ASSERT_EQ(sin_node->getNumInputs(), 1);
    ASSERT_EQ(sin_node->getNumOutputs(), 1);
    ASSERT_EQ(sin_node->getInput(0), inputDataTensor);
    ASSERT_EQ(sin_node->getOutput(0), outputSinTensor);

    //Scatter node
    ASSERT_EQ(scatter_node->getNumInputs(), 3);
    ASSERT_EQ(scatter_node->getNumOutputs(), 1);
    ASSERT_EQ(scatter_node->getInput(0), memcpy_node_in->getOutput(0));
    ASSERT_EQ(scatter_node->getInput(1), inputIndices);
    ASSERT_EQ(scatter_node->getInput(2), inputUpdates);
    ASSERT_EQ(scatter_node->getOutput(0), memcpy_node_out->getInput(0));

    ASSERT_EQ((scatter_node->getOutput(0))->getAliasTensor(), scatter_node->getInput(0));
}

TEST_F(PASSES, DISABLED_scatter_input_memcpy)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    TSize dims1[2] = {1, 5};
    TSize dims2[2] = {1, 2};
    int16_t data_buff[1][5];
    int16_t indices_buff[1][2];
    TensorPtr    inputDataTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single, reinterpret_cast<char*>(data_buff)));
    TensorPtr inputIndices = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_int32, reinterpret_cast<char*>(indices_buff)));
    TensorPtr inputUpdates = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_single, reinterpret_cast<char*>(indices_buff)));
    TensorPtr scatterOutputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));
    TensorPtr outputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));

    NodePtr scatterNode = NodeFactory::createNode({inputDataTensor, inputIndices, inputUpdates}, {scatterOutputTensor},
                                                nullptr, "scatter_f32", "scatter_node");

    NodePtr sinNode = NodeFactory::createNode({scatterOutputTensor}, {outputTensor},
                                            nullptr, "sin_f32", "sin_node");

    ns_ScatterKernel::Params params;
    params.axis = 1;
    TPCNodePtr scatterTpcNode = std::dynamic_pointer_cast<TPCNode>(scatterNode);
    scatterTpcNode->storeParamsInBuffer(&params, sizeof(ns_ScatterKernel::Params));

    GraphEditor::addNode(g, scatterNode);
    GraphEditor::addNode(g, sinNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    // 1 memcpy node, 1 scatter node, 4 dma nodes, 1 sin node.
    ASSERT_EQ(nodes.size(), 7) << "Unexpected graph size";

    NodePtr memcpy_node_in = *std::next(nodes.begin(), 1);
    ASSERT_EQ(memcpy_node_in->getNodeTypeStr(), "memcpy_f32");

    NodePtr  scatter_node = *std::next(nodes.begin(), 4);
    TPCNodePtr scatterGNode = std::dynamic_pointer_cast<TPCNode>(scatter_node);
    ASSERT_EQ(scatterGNode->getNodeName(), "scatter_node");

    NodePtr  sin_node = *std::next(nodes.begin(), 5);
    TPCNodePtr sinGNode = std::dynamic_pointer_cast<TPCNode>(sin_node);
    ASSERT_EQ(sin_node->getNodeName(), "sin_node");

    //Memcpy node in
    ASSERT_EQ(memcpy_node_in->getInput(0), inputDataTensor);
    ASSERT_EQ(memcpy_node_in->getOutput(0), scatter_node->getInput(0));

    //Scatter node
    ASSERT_EQ(scatter_node->getNumInputs(), 3);
    ASSERT_EQ(scatter_node->getNumOutputs(), 1);
    ASSERT_EQ(scatter_node->getInput(0), memcpy_node_in->getOutput(0));
    ASSERT_EQ(scatter_node->getInput(1), inputIndices);
    ASSERT_EQ(scatter_node->getInput(2), inputUpdates);
    ASSERT_EQ(scatter_node->getOutput(0), scatterOutputTensor);

    //Sin node
    ASSERT_EQ(sin_node->getNumInputs(), 1);
    ASSERT_EQ(sin_node->getNumOutputs(), 1);
    ASSERT_EQ(sin_node->getInput(0), scatterOutputTensor);
    ASSERT_EQ(sin_node->getOutput(0), outputTensor);

    ASSERT_EQ((scatter_node->getOutput(0))->getAliasTensor(), scatter_node->getInput(0));
}

TEST_F(PASSES, DISABLED_scatter_output_memcpy)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    TSize dims1[2] = {1, 5};
    TSize dims2[2] = {1, 2};
    int16_t data_buff[1][5];
    int16_t indices_buff[1][2];
    TensorPtr    inputDataTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single, reinterpret_cast<char*>(data_buff)));
    TensorPtr inputIndices = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_int32, reinterpret_cast<char*>(indices_buff)));
    TensorPtr inputUpdates = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_single, reinterpret_cast<char*>(indices_buff)));
    TensorPtr sinOutputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));
    TensorPtr outputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));

    NodePtr sinNode = NodeFactory::createNode({inputDataTensor}, {sinOutputTensor},
                                            nullptr, "sin_f32", "sin_node");

    NodePtr scatterNode = NodeFactory::createNode({sinOutputTensor, inputIndices, inputUpdates}, {outputTensor},
                                                nullptr, "scatter_f32", "scatter_node");

    ns_ScatterKernel::Params params;
    params.axis = 1;
    TPCNodePtr scatterTpcNode = std::dynamic_pointer_cast<TPCNode>(scatterNode);
    scatterTpcNode->storeParamsInBuffer(&params, sizeof(ns_ScatterKernel::Params));

    GraphEditor::addNode(g, sinNode);
    GraphEditor::addNode(g, scatterNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    // 1 memcpy node, 1 scatter node, 4 dma nodes, 1 sin node.
    ASSERT_EQ(nodes.size(), 7) << "Unexpected graph size";

    NodePtr  sin_node = *std::next(nodes.begin(), 1);
    TPCNodePtr sinGNode = std::dynamic_pointer_cast<TPCNode>(sin_node);
    ASSERT_EQ(sin_node->getNodeName(), "sin_node");

    NodePtr  scatter_node = *std::next(nodes.begin(), 4);
    TPCNodePtr scatterGNode = std::dynamic_pointer_cast<TPCNode>(scatter_node);
    ASSERT_EQ(scatterGNode->getNodeName(), "scatter_node");

    NodePtr memcpy_node_out = *std::next(nodes.begin(), 5);
    ASSERT_EQ(memcpy_node_out->getNodeTypeStr(), "memcpy_f32");

    //Sin node
    ASSERT_EQ(sin_node->getNumInputs(), 1);
    ASSERT_EQ(sin_node->getNumOutputs(), 1);
    ASSERT_EQ(sin_node->getInput(0), inputDataTensor);
    ASSERT_EQ(sin_node->getOutput(0), sinOutputTensor);

    //Scatter node
    ASSERT_EQ(scatter_node->getNumInputs(), 3);
    ASSERT_EQ(scatter_node->getNumOutputs(), 1);
    ASSERT_EQ(scatter_node->getInput(0), sinOutputTensor);
    ASSERT_EQ(scatter_node->getInput(1), inputIndices);
    ASSERT_EQ(scatter_node->getInput(2), inputUpdates);
    ASSERT_EQ(scatter_node->getOutput(0), memcpy_node_out->getInput(0));

    //Memcpy node out
    ASSERT_EQ(memcpy_node_out->getInput(0), scatter_node->getOutput(0));
    ASSERT_EQ(memcpy_node_out->getOutput(0), outputTensor);

    ASSERT_EQ((scatter_node->getOutput(0))->getAliasTensor(), scatter_node->getInput(0));
}

TEST_F(PASSES, DISABLED_scatter_no_memcpy_nodes)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    TSize dims1[2] = {1, 5};
    TSize dims2[2] = {1, 2};
    int16_t data_buff[1][5];
    int16_t indices_buff[1][2];
    TensorPtr    inputDataTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single, reinterpret_cast<char*>(data_buff)));
    TensorPtr inputIndices = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_int32, reinterpret_cast<char*>(indices_buff)));
    TensorPtr inputUpdates = std::make_shared<Tensor>(Tensor(2, dims2, syn_type_single, reinterpret_cast<char*>(indices_buff)));
    TensorPtr outputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));
    TensorPtr scatterOutputTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));
    TensorPtr outputSinTensor = std::make_shared<Tensor>(Tensor(2, dims1, syn_type_single));

    NodePtr sinNode1 = NodeFactory::createNode({inputDataTensor}, {outputSinTensor}, nullptr, "sin_f32", "sin_node1");

    NodePtr scatterNode = NodeFactory::createNode({outputSinTensor, inputIndices, inputUpdates}, {scatterOutputTensor},
                                                nullptr, "scatter_f32", "scatter_node");

    NodePtr sinNode2 = NodeFactory::createNode({scatterOutputTensor}, {outputTensor}, nullptr, "sin_f32", "sin_node2");

    ns_ScatterKernel::Params params;
    params.axis = 1;
    TPCNodePtr scatterTpcNode = std::dynamic_pointer_cast<TPCNode>(scatterNode);
    scatterTpcNode->storeParamsInBuffer(&params, sizeof(ns_ScatterKernel::Params));

    GraphEditor::addNode(g, sinNode1);
    GraphEditor::addNode(g, scatterNode);
    GraphEditor::addNode(g, sinNode2);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    // 1 scatter node, 4 dma nodes, 2 sin nodes.
    ASSERT_EQ(nodes.size(), 7) << "Unexpected graph size";

    NodePtr  sin_node1 = *std::next(nodes.begin(), 1);
    TPCNodePtr sin1Node = std::dynamic_pointer_cast<TPCNode>(sin_node1);
    ASSERT_EQ(sin1Node->getNodeName(), "sin_node1");

    NodePtr  scatter_node = *std::next(nodes.begin(), 4);
    TPCNodePtr scatterGNode = std::dynamic_pointer_cast<TPCNode>(scatter_node);
    ASSERT_EQ(scatterGNode->getNodeName(), "scatter_node");

    NodePtr  sin_node2 = *std::next(nodes.begin(), 5);
    TPCNodePtr sin2Node = std::dynamic_pointer_cast<TPCNode>(sin_node2);
    ASSERT_EQ(sin2Node->getNodeName(), "sin_node2");

    //Sin1 node
    ASSERT_EQ(sin1Node->getNumInputs(), 1);
    ASSERT_EQ(sin1Node->getNumOutputs(), 1);
    ASSERT_EQ(sin1Node->getInput(0), inputDataTensor);
    ASSERT_EQ(sin1Node->getOutput(0), outputSinTensor);

    //Scatter node
    ASSERT_EQ(scatter_node->getNumInputs(), 3);
    ASSERT_EQ(scatter_node->getNumOutputs(), 1);
    ASSERT_EQ(scatter_node->getInput(0), outputSinTensor);
    ASSERT_EQ(scatter_node->getInput(1), inputIndices);
    ASSERT_EQ(scatter_node->getInput(2), inputUpdates);
    ASSERT_EQ(scatter_node->getOutput(0), scatterOutputTensor);

    //Sin2 node
    ASSERT_EQ(sin2Node->getNumInputs(), 1);
    ASSERT_EQ(sin2Node->getNumOutputs(), 1);
    ASSERT_EQ(sin2Node->getInput(0), scatterOutputTensor);
    ASSERT_EQ(sin2Node->getOutput(0), outputTensor);

    ASSERT_EQ((scatter_node->getOutput(0))->getAliasTensor(), scatter_node->getInput(0));
}

typedef enum
{
    GELU_TEST_MODE_REGULAR = 0,              /* No changes */
    GELU_TEST_MODE_SWAP_MULT_SCALAR1_INPUTS, /* Switch between multscalar1 inputs */
    GELU_TEST_MODE_SWAP_MULT_SCALAR2_INPUTS, /* Switch between multscalar1 inputs */
    GELU_TEST_MODE_SWAP_MULXMULHALF,         /* Switch between the mult in 0.5 and mult in x operations */
    GELU_TEST_MODE_SWAP_MULHALF_INPUTS,      /* Switch between the multHalf operands*/
    GELU_TEST_MODE_SWAP_MULX_INPUTS,         /* Switch between the mult-x arguments */
    GELU_TEST_MODE_SWAP_ADD_X_INPUTS,        /* Switch between the mult in 0.5 and mult in x operations */
    GELU_TEST_MODE_X_POW_3,                  /* X^3 will be represented by pow(x,3) */
    GELU_TEST_MODE_X_POW2_MUL_X,             /* X^3 will be represented by x*x^2 */
    GELU_TEST_MODE_X_MUL_X_POW2,             /* X^3 will be represented by (x^2)*x */
    GELU_TEST_MODE_X_MUL_X_MUL_X_VAR1,       /* X^3 will be represented by x*(x*x) */
    GELU_TEST_MODE_X_MUL_X_MUL_X_VAR2,       /* X^3 will be represented by (x*x)*x */
    NUM_OF_GELU_TEST_MODES
}GeluTestMode;

TEST_F(PASSES, DISABLED_fuse_gelu_pattern)
{
    for (unsigned testMode = GELU_TEST_MODE_REGULAR; testMode < NUM_OF_GELU_TEST_MODES; testMode++)
    {
        Gaudi2Graph   g;
        g.setInferenceMode(true);

        TSize   xDims[] = {5, 5};
        float_t x_data_buff[5][5];
        float_t powOutput_data_buff[5][5];
        float_t pow2Output_data_buff[5][5];
        float_t scalar1MultOutput_data_buff[5][5];
        float_t addXOutput_data_buff[5][5];
        float_t scalar2MultOutput_data_buff[5][5];
        float_t tanhOutput_data_buff[5][5];
        float_t plus1Output_data_buff[5][5];
        float_t multHalfOutput_data_buff[5][5];
        float_t multXOutput_data_buff[5][5];

        TensorPtr x                 = TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(x_data_buff)));
        x->setName("x");
        TensorPtr powOutput         =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(powOutput_data_buff)));
        powOutput->setName("powOutput");
        TensorPtr pow2Output         =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(pow2Output_data_buff)));
        pow2Output->setName("pow2Output");
        TensorPtr scalar1MultOutput =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(scalar1MultOutput_data_buff)));
        scalar1MultOutput->setName("scalar1MultOutput");
        TensorPtr addXOutput        =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(addXOutput_data_buff)));
        addXOutput->setName("addXOutput");
        TensorPtr scalar2MultOutput =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(scalar2MultOutput_data_buff)));
        scalar2MultOutput->setName("scalar2MultOutput");
        TensorPtr tanhOutput        =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(tanhOutput_data_buff)));
        tanhOutput->setName("tanhOutput");
        TensorPtr plus1Output       =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(plus1Output_data_buff)));
        plus1Output->setName("plus1Output");
        TensorPtr multHalfOutput    =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(multHalfOutput_data_buff)));
        multHalfOutput->setName("multHalfOutput");
        TensorPtr multXOutput       =
            TensorPtr(new Tensor(2,
                                                       xDims,
                                                       syn_type_float,
                                                       reinterpret_cast<char*>(multXOutput_data_buff)));
        multXOutput->setName("multXOutput");

        // Static tensor:
        TSize scalarDims[] = {1};
        float threeVal     = 3.0;
        float scalar1Val   = 0.044715;
        float scalar2Val   = 0.79788;
        float oneVal       = 1.0;
        float halfVal      = 0.5;

        TensorPtr three = TensorPtr(new Tensor(1, scalarDims, syn_type_float, reinterpret_cast<char*>(&threeVal)));
        three->setAsStaticParam(true);
        three->setName("three");
        TensorPtr scalar1 = TensorPtr(new Tensor(1, scalarDims, syn_type_float, reinterpret_cast<char*>(&scalar1Val)));
        scalar1->setAsStaticParam(true);
        scalar1->setName("scalar1");
        TensorPtr scalar2 = TensorPtr(new Tensor(1, scalarDims, syn_type_float, reinterpret_cast<char*>(&scalar2Val)));
        scalar2->setAsStaticParam(true);
        scalar2->setName("scalar2");
        TensorPtr one = TensorPtr(new Tensor(1, scalarDims, syn_type_float, reinterpret_cast<char*>(&oneVal)));
        one->setAsStaticParam(true);
        one->setName("one2");
        TensorPtr half = TensorPtr(new Tensor(1, scalarDims, syn_type_float, reinterpret_cast<char*>(&halfVal)));
        half->setAsStaticParam(true);
        half->setName("half");

        // Nodes:
        NodePtr pow3Node;
        NodePtr xmultx;
        NodePtr xMulXPow2Node;
        NodePtr pow2Node;
        if (testMode == GELU_TEST_MODE_X_POW_3)
        {
            pow3Node = NodeFactory::createGenericTPCNode({x, three}, {powOutput}, nullptr, "pow_f32", "pow3");
            GraphEditor::addNode(g, pow3Node);
        }
        else if (testMode == GELU_TEST_MODE_X_POW2_MUL_X || testMode == GELU_TEST_MODE_X_MUL_X_POW2)
        {
            pow2Node = NodeFactory::createGenericTPCNode({x}, {pow2Output}, nullptr, "pow2_f32", "pow2");
            if(testMode == GELU_TEST_MODE_X_POW2_MUL_X)
            {
                xMulXPow2Node = NodeFactory::createGenericTPCNode({pow2Output, x}, {powOutput}, nullptr, "mult_f32", "xmultxpow2");
            }
            else
            {
                xMulXPow2Node = NodeFactory::createGenericTPCNode({x, pow2Output}, {powOutput}, nullptr, "mult_f32", "xmultxpow2");
            }
            GraphEditor::addNode(g, pow2Node);
            GraphEditor::addNode(g, xMulXPow2Node);
        }
        else /* X_MUL_X_MUL_X_VAR1 || X_MUL_X_MUL_X_VAR2*/
        {
            xmultx = NodeFactory::createGenericTPCNode({x, x}, {pow2Output}, nullptr, "pow2_f32", "pow2");
            if(testMode == GELU_TEST_MODE_X_MUL_X_MUL_X_VAR1)
            {
                xMulXPow2Node = NodeFactory::createGenericTPCNode({pow2Output, x}, {powOutput}, nullptr, "mult_f32", "xmultxpow2");
            }
            else
            {
                xMulXPow2Node = NodeFactory::createGenericTPCNode({x, pow2Output}, {powOutput}, nullptr, "mult_f32", "xmultxpow2");
            }
            GraphEditor::addNode(g, xmultx);
            GraphEditor::addNode(g, xMulXPow2Node);
        }

        NodePtr multScalar1Node;
        if (testMode == GELU_TEST_MODE_SWAP_MULT_SCALAR1_INPUTS)
        {
            multScalar1Node = NodeFactory::createGenericTPCNode({powOutput, scalar1},
                                                                {scalar1MultOutput},
                                                                nullptr,
                                                                "mult_f32",
                                                                "multScalar1");
        }
        else
        {
            multScalar1Node = NodeFactory::createGenericTPCNode({scalar1, powOutput},
                                                                {scalar1MultOutput},
                                                                nullptr,
                                                                "mult_f32",
                                                                "multScalar1");
        }
        GraphEditor::addNode(g, multScalar1Node);

        NodePtr addXNode;
        if (testMode == GELU_TEST_MODE_SWAP_ADD_X_INPUTS)
        {
            addXNode = NodeFactory::createGenericTPCNode({scalar1MultOutput, x},
                                                         {addXOutput},
                                                         nullptr,
                                                         "add_f32",
                                                         "addX");
        }
        else
        {
            addXNode = NodeFactory::createGenericTPCNode({x, scalar1MultOutput},
                                                         {addXOutput},
                                                         nullptr,
                                                         "add_f32",
                                                         "addX");
        }
        GraphEditor::addNode(g, addXNode);

        NodePtr multScalar2Node;
        if (testMode == GELU_TEST_MODE_SWAP_MULT_SCALAR2_INPUTS)
        {
            multScalar2Node = NodeFactory::createGenericTPCNode({addXOutput, scalar2},
                                                                {scalar2MultOutput},
                                                                nullptr,
                                                                "mult_f32",
                                                                "multScalar2");
        }
        else
        {
            multScalar2Node = NodeFactory::createGenericTPCNode({scalar2, addXOutput},
                                                                {scalar2MultOutput},
                                                                nullptr,
                                                                "mult_f32",
                                                                "multScalar2");
        }
        GraphEditor::addNode(g, multScalar2Node);

        NodePtr tanhNode        = NodeFactory::createGenericTPCNode({scalar2MultOutput},
                                                                  {tanhOutput},
                                                                  nullptr,
                                                                  "tanh_f32",
                                                                  "tanh");
        GraphEditor::addNode(g, tanhNode);

        NodePtr add1Node        = NodeFactory::createGenericTPCNode({tanhOutput, one},
                                                                  {plus1Output},
                                                                  nullptr,
                                                                  "add_f32",
                                                                  "add1");
        GraphEditor::addNode(g, add1Node);

        NodePtr multHalfNode;
        NodePtr multXNode;
        if (testMode == GELU_TEST_MODE_SWAP_MULXMULHALF)
        {

            multXNode = NodeFactory::createGenericTPCNode({x, plus1Output},
                                                          {multXOutput},
                                                          nullptr,
                                                          "mult_f32",
                                                          "multX");

            multHalfNode = NodeFactory::createGenericTPCNode({multXOutput, half},
                                                             {multHalfOutput},
                                                             nullptr,
                                                             "mult_f32",
                                                             "multHalf");

        }
        else
        {
            if (testMode == GELU_TEST_MODE_SWAP_MULHALF_INPUTS)
            {
                multHalfNode = NodeFactory::createGenericTPCNode({plus1Output, half},
                                                                 {multHalfOutput},
                                                                 nullptr,
                                                                 "mult_f32",
                                                                 "multHalf");
            }
            else
            {
                multHalfNode = NodeFactory::createGenericTPCNode({half, plus1Output},
                                                                 {multHalfOutput},
                                                                 nullptr,
                                                                 "mult_f32",
                                                                 "multHalf");
            }

            if (testMode == GELU_TEST_MODE_SWAP_MULX_INPUTS)
            {
                multXNode = NodeFactory::createGenericTPCNode({multHalfOutput, x},
                                                              {multXOutput},
                                                              nullptr,
                                                              "mult_f32",
                                                              "multX");
            }
            else
            {
                multXNode = NodeFactory::createGenericTPCNode({x, multHalfOutput},
                                                              {multXOutput},
                                                              nullptr,
                                                              "mult_f32",
                                                              "multX");
            }
        }
        GraphEditor::addNode(g, multHalfNode);
        GraphEditor::addNode(g, multXNode);

        bool        ret         = g.compile();
        std::string testModeStr = std::to_string(testMode);
        ASSERT_EQ(ret, true) << "Failed to compile graph on test mode " + testModeStr;

        const NodeVector& nodes = g.getExeSortedNodes();
        int  size  = nodes.size();
        ASSERT_EQ(size, 3) << "Unexpected graph size on test mode " + testModeStr;

    }

}

// TODO: clean up Greco tests SW-137061
TEST_F(PASSES, DISABLED_remove_redundant_logical_nodes)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    std::list<std::string> removeNodeNames;
    std::list<std::string> keepNodeNames;

    TSize Dims6_4[]   = {6,  4, 1, 1, 1};
    TSize Dims2_3_4[] = {2,  3, 4, 1, 1};
    TSize Dims4_3_2[] = {4,  3, 2, 1, 1};
    TSize Dims24[]    = {24, 1, 1, 1, 1};
    TSize Dims4_2_2[] = {4,  2, 2, 1, 1};
    TSize Dims3_2_2[] = {3,  2, 2, 1, 1};
    TSize Dims2_2_2[] = {2,  2, 2, 1, 1};
    TSize Dims2_2_1[] = {2,  2, 1, 1, 1};
    TSize Dims2_2[]   = {2,  2, 1, 1, 1};
    TSize Dims2_4[]   = {2,  4, 1, 1, 1};

    /* Reshape */
    /* Reshape1: same dims different sizes
     * Reshape2: same dims, same sizes
     * Reshape3: different dims */
    float_t inReshape1Buff[6][4];
    float_t outReshape1Buff[24];
    float_t outReshape2Buff[24];
    float_t outReshape3Buff[2][3][4];

    TensorPtr inReshape1  = TensorPtr(new Tensor(2U, Dims6_4, syn_type_float, reinterpret_cast<char*>(inReshape1Buff)));
    inReshape1->setName("InReshape1");
    TensorPtr outReshape1 = TensorPtr(new Tensor(1U, Dims24, syn_type_float, reinterpret_cast<char*>(outReshape1Buff)));
    outReshape1->setName("outReshape1");
    TensorPtr outReshape2 = TensorPtr(new Tensor(1U, Dims24, syn_type_float, reinterpret_cast<char*>(outReshape2Buff)));
    outReshape2->setName("outReshape2");
    TensorPtr outReshape3 =
        TensorPtr(new Tensor(3U, Dims2_3_4, syn_type_float, reinterpret_cast<char*>(outReshape3Buff)));
    outReshape3->setName("outReshape3");

    NodePtr reshape1Node = NodeFactory::createNode({inReshape1}, {outReshape1}, nullptr, "reshape", "reshape1");
    NodePtr reshape2Node = NodeFactory::createNode({outReshape1}, {outReshape2}, nullptr, "reshape", "reshape2");
    NodePtr reshape3Node = NodeFactory::createNode({outReshape2}, {outReshape3}, nullptr, "reshape", "reshape3");
    GraphEditor::addNode(g, reshape1Node);
    GraphEditor::addNode(g, reshape2Node);
    GraphEditor::addNode(g, reshape3Node);
    keepNodeNames.push_back("reshape1");
    removeNodeNames.push_back("reshape2");
    keepNodeNames.push_back("reshape3");

    /* Transpose
     * Transpose1: permutation kept the same
     * Transpose2: permutation change*/
    float_t outTrans1Buff[2][3][4];
    float_t outTrans2Buff[4][3][2];

    synTransposeParams trans1Params = {{TPD_Channel, TPD_Width, TPD_Height}, 3};
    synTransposeParams trans2Params = {{TPD_Height, TPD_Width, TPD_Channel}, 3};

    TensorPtr outTrans1 = TensorPtr(new Tensor(3U, Dims2_3_4, syn_type_float, reinterpret_cast<char*>(outTrans1Buff)));
    outTrans1->setName("outTrans1");
    TensorPtr outTrans2 = TensorPtr(new Tensor(3U, Dims4_3_2, syn_type_float, reinterpret_cast<char*>(outTrans2Buff)));
    outTrans2->setName("outTrans2");

    NodePtr trans1Node = NodeFactory::createNode({outReshape3}, {outTrans1}, &trans1Params, "transpose", "trans1");
    NodePtr trans2Node = NodeFactory::createNode({outTrans1}, {outTrans2}, &trans2Params, "transpose", "trans2");

    GraphEditor::addNode(g, trans1Node);
    GraphEditor::addNode(g, trans2Node);
    removeNodeNames.push_back("trans1");
    keepNodeNames.push_back("trans2");

    /* Slice */
    /* Slice1: new_start = 0, new_end < end, step = 1
     * Slice2: new_start = 0, new_end = end, step = 1
     * Slice3: new_start > 0, new_end = end, step = 1 */
    float_t outSlice1Buff[4][2][2]; /* sliced from 4,3,2 */
    float_t outSlice2Buff[4][2][2];
    float_t outSlice3Buff[3][2][2]; /* sliced 1 element from the begining */

    TensorPtr outSlice1 = TensorPtr(new Tensor(3U, Dims4_2_2, syn_type_float, reinterpret_cast<char*>(outSlice1Buff)));
    outSlice1->setName("outSlice1");
    TensorPtr outSlice2 = TensorPtr(new Tensor(3U, Dims4_2_2, syn_type_float, reinterpret_cast<char*>(outSlice2Buff)));
    outSlice2->setName("outSlice2");
    TensorPtr outSlice3 = TensorPtr(new Tensor(3U, Dims3_2_2, syn_type_float, reinterpret_cast<char*>(outSlice3Buff)));
    outSlice3->setName("outSlice3");

    synSliceParams slice1p = {{0, 1, 2}, {0, 0, 0}, {4, 2, 2}, {1, 1, 1}};
    synSliceParams slice2p = {{0, 1, 2}, {0, 0, 0}, {4, 2, 2}, {1, 1, 1}};
    synSliceParams slice3p = {{0, 1, 2}, {1, 0, 0}, {4, 2, 2}, {1, 1, 1}};

    NodePtr slice1Node = NodeFactory::createNode({outTrans2}, {outSlice1}, &slice1p, "slice", "slice1");
    NodePtr slice2Node = NodeFactory::createNode({outSlice1}, {outSlice2}, &slice2p, "slice", "slice2");
    NodePtr slice3Node = NodeFactory::createNode({outSlice2}, {outSlice3}, &slice3p, "slice", "slice3");

    GraphEditor::addNode(g, slice1Node);
    GraphEditor::addNode(g, slice2Node);
    GraphEditor::addNode(g, slice3Node);
    keepNodeNames.push_back("slice1");
    removeNodeNames.push_back("slice2");
    keepNodeNames.push_back("slice3");

    /* SliceAxis
     * SliceAxis1: new_start = 0, new_end < end
     * SliceAxis2: new_start = 0, new_end = end
     * SliceAxis3: new_start > 0, new_end = end */
    float_t outSliceAxis1Buff[2][2][2]; /* sliced from 3,2,2 */
    float_t outSliceAxis2Buff[2][2][2];
    float_t outSliceAxis3Buff[2][2][1]; /* sliced 1 element from the begining */

    TensorPtr outSliceAxis1 =
        TensorPtr(new Tensor(3U, Dims2_2_2, syn_type_float, reinterpret_cast<char*>(outSliceAxis1Buff)));
    outSlice1->setName("outSliceAxis1");
    TensorPtr outSliceAxis2 =
        TensorPtr(new Tensor(3U, Dims2_2_2, syn_type_float, reinterpret_cast<char*>(outSliceAxis2Buff)));
    outSlice2->setName("outSliceAxis2");
    TensorPtr outSliceAxis3 =
        TensorPtr(new Tensor(3U, Dims2_2_1, syn_type_float, reinterpret_cast<char*>(outSliceAxis3Buff)));
    outSlice3->setName("outSliceAixs3");

    synSliceAxisParams sliceAxis1p = {0, 0, 2};
    synSliceAxisParams sliceAxis2p = {0, 0, 2};
    synSliceAxisParams sliceAxis3p = {2, 1, 2};

    NodePtr sliceAxis1Node = NodeFactory::createNode({outSlice3},     {outSliceAxis1}, &sliceAxis1p, "slice_axis", "slice_axis1");
    NodePtr sliceAxis2Node = NodeFactory::createNode({outSliceAxis1}, {outSliceAxis2}, &sliceAxis2p, "slice_axis", "slice_axis2");
    NodePtr sliceAxis3Node = NodeFactory::createNode({outSliceAxis2}, {outSliceAxis3}, &sliceAxis3p, "slice_axis", "slice_axis3");

    GraphEditor::addNode(g, sliceAxis1Node);
    GraphEditor::addNode(g, sliceAxis2Node);
    GraphEditor::addNode(g, sliceAxis3Node);
    keepNodeNames.push_back("slice_axis1");
    removeNodeNames.push_back("slice_axis2");
    keepNodeNames.push_back("slice_axis3");

    /* Flatten
     * Flatten1: axis=2, reduce from (2,2,1) to (2,2)
     * Flatten2: axis=0, keep (2,2) */
    float_t outFlatten1Buff[2][2];
    float_t outFlatten2Buff[2][2];

    TensorPtr outFlatten1 =
        TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(outFlatten1Buff)));
    outFlatten1->setName("outFlatten1");
    TensorPtr outFlatten2 =
        TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(outFlatten2Buff)));
    outFlatten2->setName("outFlatten2");

    synFlattenParams flattenParams1 = {0};
    synFlattenParams flattenParams2 = {0};

    NodePtr flatten1Node = NodeFactory::createNode({outSliceAxis3}, {outFlatten1}, &flattenParams1, "flatten", "flatten1");
    NodePtr flatten2Node = NodeFactory::createNode({outFlatten1},   {outFlatten2}, &flattenParams2, "flatten", "flatten2");

    GraphEditor::addNode(g, flatten1Node);
    GraphEditor::addNode(g, flatten2Node);
    keepNodeNames.push_back("flatten1");
    removeNodeNames.push_back("flatten2");

    /* Concat
     * Concat1: concat 2 tensors on dim1
     * Concat2: concat 1 tensor  on dim0 */
    float_t inConcat1Buff[2][2];
    float_t outConcat1Buff[2][4];
    float_t outConcat2Buff[2][4];

    TensorPtr inConcat1 = TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(inConcat1Buff)));
    inConcat1->setName("inConcat1");
    TensorPtr outConcat1 = TensorPtr(new Tensor(2U, Dims2_4, syn_type_float, reinterpret_cast<char*>(outConcat1Buff)));
    outConcat1->setName("outConcat1");
    TensorPtr outConcat2 = TensorPtr(new Tensor(2U, Dims2_4, syn_type_float, reinterpret_cast<char*>(outConcat2Buff)));
    outConcat2->setName("outConcat2");

    unsigned concat1Param = 1; /* dim = 1 */
    unsigned concat2Param = 0; /* dim = 0 */

    NodePtr concat1Node = NodeFactory::createNode({outFlatten2, inConcat1}, {outConcat1}, &concat1Param, "concat", "concat1");
    NodePtr concat2Node = NodeFactory::createNode({outConcat1}, {outConcat2}, &concat2Param, "concat", "concat2");

    GraphEditor::addNode(g, concat1Node);
    GraphEditor::addNode(g, concat2Node);
    keepNodeNames.push_back("concat1");
    removeNodeNames.push_back("concat2");

    /* Split
     * Split1: split to 2 tensor on dim 1
     * Split2: split to 1 tensor on dim 0
     * Split3: split to 1 tensor on dim 0 (also output node, should be removed) */
    float_t split1Out1Buff[2][2];
    float_t split1Out2Buff[2][2];
    float_t split2OutBuff[2][2];
    float_t split3OutBuff[2][2];

    TensorPtr split1Out1 = TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(split1Out1Buff)));
    split1Out1->setName("split1Out1");
    TensorPtr split1Out2 = TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(split1Out2Buff)));
    split1Out2->setName("split1Out2");
    TensorPtr split2Out = TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(split2OutBuff)));
    split2Out->setName("split2Out");
    TensorPtr split3Out = TensorPtr(new Tensor(2U, Dims2_2, syn_type_float, reinterpret_cast<char*>(split3OutBuff)));
    split3Out->setName("split3Out");

    unsigned splitParams1 = 1;
    unsigned splitParams2 = 0;
    unsigned splitParams3 = 0;

    NodePtr split1Node = NodeFactory::createNode({outConcat2}, {split1Out1, split1Out2}, &splitParams1, "split", "split1");
    NodePtr split2Node = NodeFactory::createNode({split1Out1}, {split2Out}, &splitParams2, "split", "split2");
    NodePtr split3Node = NodeFactory::createNode({split2Out}, {split3Out}, &splitParams3, "split", "split3");

    GraphEditor::addNode(g, split1Node);
    GraphEditor::addNode(g, split2Node);
    GraphEditor::addNode(g, split3Node);
    keepNodeNames.push_back("split1");
    removeNodeNames.push_back("split2");
    removeNodeNames.push_back("split3");

    /*
     * Compiling the graph
     */
    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        /* Checking that this node doesn't exist in the need to remove nodes */
        ASSERT_EQ(std::find(removeNodeNames.begin(), removeNodeNames.end(), node->getNodeName()), removeNodeNames.end())
            << "Did not remove node " << node->getNodeName();

        keepNodeNames.remove(node->getNodeName());
    }

    ASSERT_EQ(keepNodeNames.size(), 0) << "Nodes which should have been kept, were removed";
}

TEST_F(PASSES, handle_identity_cast_nodes)
{
    // handleIdentityCastNodes requires GCFG_SYNAPSE_DATA_TYPE_SELECTION to be on
    GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(true);
    Gaudi2Graph   g;
    g.setInferenceMode(true);
    bool      ret;

    std::list<std::string> removeNodeNames;
    std::list<std::string> keepNodeNames;

    const TSize sizes[] = {1, 2, 2, 1};

    pTensor t1 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t2 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t3 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t4 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t5 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t6 = pTensor(new Tensor(4U, sizes, syn_type_int32));
    pTensor t7 = pTensor(new Tensor(4U, sizes, syn_type_int32));

    pNode tanh1 = NodeFactory::createNode({t1}, {t2}, nullptr, "tanh", "tanh1");
    pNode cast1 = NodeFactory::createNode({t2}, {t3}, nullptr, "cast_i8_to_i16", "cast1");
    pNode cast2 = NodeFactory::createNode({t3}, {t4}, nullptr, "cast_i16_to_i16", "cast2");
    pNode cast3 = NodeFactory::createNode({t4}, {t5}, nullptr, "cast_f32_to_f32", "cast3");
    pNode tanh2 = NodeFactory::createNode({t5}, {t6}, nullptr, "tanh", "tanh2");
    pNode cast4 = NodeFactory::createNode({t6}, {t7}, nullptr, "cast_f32_to_f32", "cast4");
    GraphEditor::addNode(g, tanh1);
    GraphEditor::addNode(g, cast1);
    GraphEditor::addNode(g, cast2);
    GraphEditor::addNode(g, cast3);
    GraphEditor::addNode(g, tanh2);
    GraphEditor::addNode(g, cast4);

    keepNodeNames.push_back("tanh1");
    keepNodeNames.push_back("cast1");
    removeNodeNames.push_back("cast2");
    removeNodeNames.push_back("cast3");
    keepNodeNames.push_back("tanh2");
    removeNodeNames.push_back("cast4");

    // run the pass
    ret = handleIdentityCastNodes(g);
    ASSERT_TRUE(ret) << "failed to run handle identity cast nodes pass";

    const NodeVector& sortedNodes = g.getExeSortedNodes();
    for (const NodePtr& node : sortedNodes)
    {
        // Checking all nodes in removeNodeNames were removed
        ASSERT_EQ(std::find(removeNodeNames.begin(), removeNodeNames.end(), node->getNodeName()),
                  removeNodeNames.end()) << "Did not remove node " << node->getNodeName();

        keepNodeNames.remove(node->getNodeName());
    }

    ASSERT_EQ(keepNodeNames.size(), 0) << "Nodes which should have been kept, were removed";
}

TEST_F(PASSES, remove_contiguous_reshapes)
{
    Gaudi2Graph   graph;
    graph.setInferenceMode(true);

    SizeArray firstSize = {5,5,5,5,1};
    TensorPtr tensor1 = TensorPtr(new Tensor(4U, firstSize.data(), syn_type_float));
    TensorPtr tensor2 = TensorPtr(new Tensor(4U, firstSize.data(), syn_type_float));
    NodePtr   tpc1      = NodeFactory::createNode({tensor1}, {tensor2}, nullptr, NOP_KERNEL_NAME, "tpc_nop1");
    GraphEditor::addNode(graph, tpc1);

    SizeArray reshapeSize1 = {5,25,1,5,1};
    TensorPtr tensor3 = TensorPtr(new Tensor(4U, reshapeSize1.data(), syn_type_float));
    NodePtr   reshape1 = NodeFactory::createNode({tensor2}, {tensor3}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape1");
    GraphEditor::addNode(graph, reshape1);

    SizeArray reshapeSize2 = {5,125,1,1,1};
    TensorPtr tensor4 = TensorPtr(new Tensor(4U, reshapeSize2.data(), syn_type_float));
    NodePtr   reshape2 = NodeFactory::createNode({tensor3}, {tensor4}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape2");
    GraphEditor::addNode(graph, reshape2);

    TensorPtr tensor5 = TensorPtr(new Tensor(4U, reshapeSize2.data(), syn_type_float));
    NodePtr   tpc2    = NodeFactory::createNode({tensor4}, {tensor5}, nullptr, NOP_KERNEL_NAME, "tpc_nop2");
    GraphEditor::addNode(graph, tpc2);

    SizeArray reshapeSize3 = {625,1,1,1,1};
    TensorPtr tensor6 = TensorPtr(new Tensor(4U, reshapeSize3.data(), syn_type_float));
    NodePtr   reshape3 = NodeFactory::createNode({tensor5}, {tensor6}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape3");
    GraphEditor::addNode(graph, reshape3);

    TensorPtr tensor7 = TensorPtr(new Tensor(4U, reshapeSize3.data(), syn_type_float));
    NodePtr   tpc3    = NodeFactory::createNode({tensor6}, {tensor7}, nullptr, NOP_KERNEL_NAME, "tpc_nop3");
    GraphEditor::addNode(graph, tpc3);

    SizeArray reshapeSize4 = {25,25,1,1,1};
    TensorPtr tensor8 = TensorPtr(new Tensor(4U, reshapeSize4.data(), syn_type_float));
    NodePtr   reshape4 = NodeFactory::createNode({tensor6}, {tensor8}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape4");
    GraphEditor::addNode(graph, reshape4);

    TensorPtr tensor9 = TensorPtr(new Tensor(4U, reshapeSize4.data(), syn_type_float));
    NodePtr   tpc4    = NodeFactory::createNode({tensor8}, {tensor9}, nullptr, NOP_KERNEL_NAME, "tpc_nop4");
    GraphEditor::addNode(graph, tpc4);

    removeContiguousReshapeNodes(graph);

    unsigned leftReshapes = 0;
    for (auto node : graph.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            leftReshapes++;
        }
    }
    ASSERT_EQ(leftReshapes, 3);
}

TEST_F(PASSES, remove_contiguous_reshapes_6d)
{
    Gaudi2Graph   graph;
    graph.setInferenceMode(true);

    SizeArray firstSize = {5, 5, 5, 5, 1};
    TensorPtr tensor1   = TensorPtr(new Tensor(4U, firstSize.data(), syn_type_float));
    TensorPtr tensor2   = TensorPtr(new Tensor(4U, firstSize.data(), syn_type_float));
    NodePtr   tpc1      = NodeFactory::createNode({tensor1}, {tensor2}, nullptr, NOP_KERNEL_NAME, "tpc_nop1");
    GraphEditor::addNode(graph, tpc1);

    NSizeArray reshapeSize1 = {5, 25, 1, 5, 1, 1};
    TensorPtr  tensor3      = TensorPtr(new Tensor(6U, reshapeSize1.data(), syn_type_float));
    NodePtr    reshape1 =
        NodeFactory::createNode({tensor2}, {tensor3}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape1");
    GraphEditor::addNode(graph, reshape1);

    SizeArray reshapeSize2 = {5, 125, 1, 1, 1};
    TensorPtr tensor4      = TensorPtr(new Tensor(4U, reshapeSize2.data(), syn_type_float));
    NodePtr   reshape2 =
        NodeFactory::createNode({tensor3}, {tensor4}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape2");
    GraphEditor::addNode(graph, reshape2);

    TensorPtr tensor5 = TensorPtr(new Tensor(4U, reshapeSize2.data(), syn_type_float));
    NodePtr   tpc2    = NodeFactory::createNode({tensor4}, {tensor5}, nullptr, NOP_KERNEL_NAME, "tpc_nop2");
    GraphEditor::addNode(graph, tpc2);

    removeContiguousReshapeNodes(graph);

    unsigned leftReshapes = 0;
    for (auto node : graph.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            leftReshapes++;
        }
    }
    ASSERT_EQ(leftReshapes, 1);
}

TEST_F(PASSES, DISABLED_fuse_layer_norm)
{
    Gaudi2Graph   g;
    g.setInferenceMode(true);

    const float EPS   = 0.0001;
    const float BETA  = 0;
    const float GAMMA = 1;

    /* shape: */
    TSize inputShape[]     = {768, 128, 1, 1};
    TSize meanVarShape[]   = {1, 128, 1, 1};
    TSize betaGammaShape[] = {768, 1, 1, 1};
    TSize epsilonShape[]   = {1, 1, 1, 1};

    /* Tensors: */
    TensorPtr input            = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr reduceMean1Out   = TensorPtr(new Tensor(3U, meanVarShape, syn_type_float));
    TensorPtr squareDiffSubOut = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr squareDiffMulOut = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr reduceMean2Out   = TensorPtr(new Tensor(3U, meanVarShape, syn_type_float));
    TensorPtr batchNormAdd1Out = TensorPtr(new Tensor(3U, meanVarShape, syn_type_float));
    TensorPtr sqrtOut          = TensorPtr(new Tensor(3U, meanVarShape, syn_type_float));
    TensorPtr reciprocalOut    = TensorPtr(new Tensor(3U, meanVarShape, syn_type_float));
    TensorPtr batchNormMul1Out = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr batchNormMul2Out = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr batchNormMul3Out = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr batchNormSubOut  = TensorPtr(new Tensor(3U, inputShape, syn_type_float));
    TensorPtr batchNormAdd2Out = TensorPtr(new Tensor(3U, inputShape, syn_type_float));

    /* Static tensors: */
    float_t betaBuff[768];
    float_t gammaBuff[768];
    float_t epsBuff[1];

    for (unsigned i = 0; i < 768; i++)
    {
        betaBuff[i] = BETA;
        gammaBuff[i] = GAMMA;
    }
    epsBuff[0] = EPS;

    TensorPtr batchNormMul1Scalar =
        TensorPtr(new Tensor(1U, betaGammaShape, syn_type_float, reinterpret_cast<char*>(gammaBuff)));// gamma
    TensorPtr batchNormSubScalar  =
        TensorPtr(new Tensor(1U, betaGammaShape, syn_type_float, reinterpret_cast<char*>(betaBuff)));// beta
    TensorPtr batchNormAdd1Scalar =
        TensorPtr(new Tensor(1U, epsilonShape, syn_type_float, reinterpret_cast<char*>(epsBuff))); //epsilon
    batchNormAdd1Scalar->setAsStaticParam(true);

    /* Nodes: */
    ns_Reduction::Params reduceParams = {0};
    NodePtr              reduceMean1 = NodeFactory::createGenericTPCNode({input}, {reduceMean1Out}, &reduceParams,
                                                          "reduce_mean", "reduce_mean1");
    GraphEditor::addNode(g, reduceMean1);

    NodePtr squareDiffSub = NodeFactory::createGenericTPCNode({input, reduceMean1Out}, {squareDiffSubOut},
                                                            nullptr, "sub", "squareDiffSub");
    GraphEditor::addNode(g, squareDiffSub);

    NodePtr squareDiffMul = NodeFactory::createGenericTPCNode({squareDiffSubOut, squareDiffSubOut}, {squareDiffMulOut},
                                                            nullptr, "mult", "squareDiffMul");
    GraphEditor::addNode(g, squareDiffMul);

    NodePtr reduceMean2 = NodeFactory::createGenericTPCNode({squareDiffMulOut}, {reduceMean2Out},
                                                          &reduceParams, "reduce_mean", "reduce_mean2");
    GraphEditor::addNode(g, reduceMean2);

    NodePtr batchNormAdd1 = NodeFactory::createGenericTPCNode({reduceMean2Out, batchNormAdd1Scalar}, {batchNormAdd1Out},
                                                            nullptr, "add", "batchNormAdd1");
    GraphEditor::addNode(g, batchNormAdd1);

    NodePtr sqrt = NodeFactory::createGenericTPCNode({batchNormAdd1Out}, {sqrtOut},
                                                   nullptr, "sqrt", "sqrt");
    GraphEditor::addNode(g, sqrt);

    NodePtr reciprocal = NodeFactory::createGenericTPCNode({sqrtOut}, {reciprocalOut},
                                                         nullptr, "reciprocal", "reciprocal");
    GraphEditor::addNode(g, reciprocal);

    NodePtr batchNormMul1 = NodeFactory::createGenericTPCNode({reciprocalOut, batchNormMul1Scalar}, {batchNormMul1Out},
                                                            nullptr, "mult", "batchNormMul1");
    GraphEditor::addNode(g, batchNormMul1);

    NodePtr batchNormMul2 = NodeFactory::createGenericTPCNode({input, batchNormMul1Out}, {batchNormMul2Out},
                                                            nullptr, "mult", "batchNormMul2");
    GraphEditor::addNode(g, batchNormMul2);

    NodePtr batchNormMul3 = NodeFactory::createGenericTPCNode({reduceMean1Out, batchNormMul1Out}, {batchNormMul3Out},
                                                            nullptr, "mult", "batchNormMul3");
    GraphEditor::addNode(g, batchNormMul3);

    NodePtr batchNormSub = NodeFactory::createGenericTPCNode({batchNormSubScalar, batchNormMul3Out}, {batchNormSubOut},
                                                           nullptr, "sub", "batchNormSub");
    GraphEditor::addNode(g, batchNormSub);

    NodePtr batchNormAdd2 = NodeFactory::createGenericTPCNode({batchNormMul2Out, batchNormSubOut}, {batchNormAdd2Out},
                                                            nullptr, "add", "batchNormAdd2");
    GraphEditor::addNode(g, batchNormAdd2);

    /*
    * Compiling the graph
    */
    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    /* 4 DMA nodes, and 1 fused kernel */
    ASSERT_EQ(nodes.size(), 5) << "Expecting fusion of layern norm pattern";

    TPCNodePtr layerNormNode = std::dynamic_pointer_cast<TPCNode>(nodes[3]);
    ASSERT_NE(layerNormNode, nullptr) << "fused node is not a TPC node";
    ASSERT_EQ(layerNormNode->getGUID(), "layer_norm_f32") <<"Expecting layer norm node";

    ns_LayerNormKernel::Params *params = (ns_LayerNormKernel::Params*)layerNormNode->getParams();
    ASSERT_EQ(params->epsValid, true) << "Epsilon param is not valid";
    ASSERT_EQ(params->eps, EPS) << "Epsilon param is matching the expected value";

    TensorPtr betaTensor = layerNormNode->getInput(1);
    ASSERT_NE(betaTensor, nullptr) << "null node on input 1 of fused kernel. expecting beta tensor";
    ASSERT_EQ(betaTensor->getId(), batchNormSubScalar->getId()) << "unexpected beta tensor";

    TensorPtr gammaTensor = layerNormNode->getInput(2);
    ASSERT_NE(gammaTensor, nullptr) << "null node on input 1 of fused kernel. expecting gamma tensor";
    ASSERT_EQ(gammaTensor->getId(), batchNormMul1Scalar->getId()) << "unexpected gamma tensor";
}

TEST_F(PASSES, DISABLED_check_beam_multinode_pass)
{
    /* This test verifies that for small value of K, topk node is being replace with topk_st1 and topk_st2 nodes */
    /* Test for large value of K, will be performed through habana_py_test */
    Gaudi2Graph   g;
    g.setInferenceMode(true);

    TSize H = 37;
    TSize W = 42;
    TSize K = 21;

    TSize inputDims[] = {H, W};
    TSize outputDims[] = {H, K};

    int16_t      input_data_buff[H][W];
    int16_t      output_data_buff[H][K];
    int32_t      indices_data_buff[H][K];

    TensorPtr inputTensor   =
        TensorPtr(new Tensor(2, inputDims, syn_type_int16, reinterpret_cast<char*>(input_data_buff)));
    TensorPtr outputTensor  =
        TensorPtr(new Tensor(2, outputDims, syn_type_int16, reinterpret_cast<char*>(output_data_buff)));
    TensorPtr indicesTensor =
        TensorPtr(new Tensor(2, outputDims, syn_type_int32, reinterpret_cast<char*>(indices_data_buff)));

    synBeamParams params;
    params.bsw = K;
    params.axis = 1;

    NodePtr topkNode = NodeFactory::createNode({inputTensor},{outputTensor, indicesTensor}, &params, "topk", "topk");

    GraphEditor::addNode(g, topkNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();

    bool found_st1 = false;
    bool found_st2 = false;

    for (const NodePtr& node : nodes)
    {
        TPCNodePtr n = std::dynamic_pointer_cast<TPCNode>(node);
        if ( n == nullptr)
        {
            continue;
        }

        if (std::string(n->getGUID()).find("top_k_st1") != std::string::npos)
        {
            found_st1 = true;
        }
        else if (std::string(n->getGUID()).find("top_k_st2") != std::string::npos)
        {
            found_st2 = true;
        }
    }

    ASSERT_EQ(found_st1, true) << "could not find topk_st1 node in the graph";
    ASSERT_EQ(found_st2, true) << "could not find topk_st2 node in the graph";

}

TEST_F(PASSES, DISABLED_check_beam_multinode_pass_with_5D_padding)
{
    /* This test verifies that for  K > 37 and a 5D input tensor the beam search node is replaced
     * with pad node followed by sort_step nodes.
     * The number of sort nodes should be (n * (n + 1)) / 2 ,
     * where n is log2(x) and x is the closest power of 2 to K , while x >= K*/

    Gaudi2Graph   g;
    g.setInferenceMode(true);

    TSize H = 40;
    TSize W = 42;
    TSize C = 3;
    TSize N = 10;
    TSize D = 1;
    TSize K = 38; //beam search size

    TSize inputDims[]  = {N, H, W, C, D};
    TSize outputDims[] = {N, H, K, C, D};

    int16_t      input_data_buff[N][H][W][C][D];
    int16_t      output_data_buff[N][H][K][C][D];
    int32_t      indices_data_buff[N][H][K][C][D];

    TensorPtr inputTensor   =
        TensorPtr(new Tensor(5, inputDims,  syn_type_int16, reinterpret_cast<char*>(input_data_buff)));
    TensorPtr outputTensor  =
        TensorPtr(new Tensor(5, outputDims, syn_type_int16, reinterpret_cast<char*>(output_data_buff)));
    TensorPtr indicesTensor =
        TensorPtr(new Tensor(5, outputDims, syn_type_int32, reinterpret_cast<char*>(indices_data_buff)));

    synBeamParams params;
    params.bsw  = K;
    params.axis = 1;

    NodePtr topkNode = NodeFactory::createNode({inputTensor},{outputTensor, indicesTensor}, &params, "topk", "topk");

    GraphEditor::addNode(g, topkNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    //convert K to closest large power of 2
    float log2Len                = ceil(log2(K));
    float newLen                 = pow(2, log2Len);
    unsigned log2NewLen          = log2(newLen);
    unsigned expectedNumOfSortNodes = (log2NewLen * (log2NewLen + 1)) / 2;

    const NodeVector& nodes = g.getExeSortedNodes();

    unsigned numOfFoundSortNodes = 0;
    unsigned numOfFoundPadNodes  = 0;

    //verify that pad node and correct number of sort nodes were created
    for (const NodePtr& node : nodes)
    {
        TPCNodePtr n = std::dynamic_pointer_cast<TPCNode>(node);
        if ( n == nullptr)
        {
            continue;
        }

        if (std::string(n->getGUIDWithoutDtype()).find("sort_step") != std::string::npos)
        {
            numOfFoundSortNodes++;
        }
        else if (std::string(n->getGUIDWithoutDtype()).find("pad") != std::string::npos)
        {
            numOfFoundPadNodes++;
        }
    }

    ASSERT_EQ(numOfFoundSortNodes, expectedNumOfSortNodes) << "Unexpected number of sort step nodes";
    ASSERT_EQ(numOfFoundPadNodes, 1) << "Unexpected number of pad nodes";

}

TEST_F(PASSES, DISABLED_nms_test_extraction_one_chunk)
{
    /* This test verifies that NMS multinode is being extracted properly - to the expected nodes */
    Gaudi2Graph   g;
    g.setInferenceMode(true);

    TSize B = 30;
    TSize C = 8;
    TSize N = 16;

    TSize max_out_size = 5;
    TSize outputDim1 = max_out_size * C * N;
    TSize outputDim2 = 3;

    TSize scoresInputDims[] = {B, C, N};
    TSize boxesInputDims[] = {B, 4, N};
    TSize outputIndicesDims[] = {outputDim1, outputDim2};

    int32_t      indices_data_buff[B][C][N];
    int32_t      boxes_data_buff[B][4][N];
    int32_t      output_data_buff[outputDim1][outputDim2];

    TensorPtr boxesTensor   =
        TensorPtr(new Tensor(3, boxesInputDims, syn_type_int32, reinterpret_cast<char*>(boxes_data_buff)));
    TensorPtr scoresTensor =
        TensorPtr(new Tensor(3, scoresInputDims, syn_type_int32, reinterpret_cast<char*>(indices_data_buff)));
    TensorPtr outputTensor  =
        TensorPtr(new Tensor(2, outputIndicesDims, syn_type_int32, reinterpret_cast<char*>(output_data_buff)));

    synNMSParams nmsParams;
    nmsParams.iouTh      = 1;
    nmsParams.maxOutSize = max_out_size;
    nmsParams.scoreTh    = 0;

    NodePtr nmsNode = NodeFactory::createNode({boxesTensor, scoresTensor},
                                            {outputTensor},
                                            &nmsParams,
                                            "non_max_suppression",
                                            "nms");

    GraphEditor::addNode(g, nmsNode);

    bool ret = g.compile();
    ASSERT_EQ(ret, true) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    NodePtr  currNode;
    TPCNodePtr tpcNode;

    /* expecting:
     * 1. filter_and_squeeze - 1
     * 2. bitonic sort - 1
     * 3. scalar_merge - 0 (only one chunk)
     * 4. slice - 2 (one for scores and one for indices)
     * 5. split - N * 2 (N for boxes and N for indices)
     * 6. gather - N
     * 7. concat - 1
     * 8. nms_f32 - 1
     * 9. post_nms_f32 - 1
     * 10. reshape - 0 (no reshape is required anymore as bitonic support correct shapes)
     * 11. pad - 2, need to pad scores and indices
     * */
    unsigned countFilter = 0, countSlice = 0, countSplit = 0, countGather = 0, countConcat = 0,
             countNms = 0, countPostNms = 0, countReshape = 0, countPad = 0, countPreprocess = 0,
             countGenerateBitonicChunks = 0;

    for (const NodePtr& node : nodes)
    {
        tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
        if (tpcNode != nullptr)
        {
            if (tpcNode->getGUIDWithoutDtype() == "filter_and_squeeze")
            {
                countFilter++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "sort_pre_process")
            {
                countPreprocess++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "generate_bitonic_chunks")
            {
                countGenerateBitonicChunks++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "gather")
            {
                countGather++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "post_nms")
            {
                countPostNms++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "nms")
            {
                countNms++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "pad")
            {
                countPad++;
            }
            else if (tpcNode->getGUIDWithoutDtype() == "cast" || tpcNode->getGUIDWithoutDtype() == "memcpy")
            {
                continue;
            }
            else
            {
                ASSERT_TRUE(false) << "Unexpected TPC node with guid " + tpcNode->getGUID();
            }
        }
        else
        {
            switch (node->getNodeType())
            {
            case Node::TYPE_SLICE:
            case Node::TYPE_SLICE_AXIS:
                countSlice++;
                break;
            case Node::TYPE_INTERNAL_SPLIT:
                countSplit++;
                break;
            case Node::TYPE_INTERNAL_CONCAT:
                countConcat++;
                break;
            case Node::TYPE_INTERNAL_RESHAPE:
                countReshape++;
                break;
            case Node::TYPE_DMA:
            case Node::TYPE_MEMCOPY:
            case Node::TYPE_INTERNAL_PACKING:
                /* ignore nodes */
                break;
            default:
                ASSERT_TRUE(false) << "Unexpected Node type " + std::to_string(node->getNodeType()) + ", name is:" + node->getNodeName();
            }
        }
    }

    ASSERT_EQ(countFilter, 1);
    ASSERT_EQ(countSlice, 2);
    ASSERT_EQ(countSplit, 2);
    ASSERT_EQ(countGather, N);
    ASSERT_EQ(countConcat, 1);
    ASSERT_EQ(countNms, 1);
    ASSERT_EQ(countPostNms, 1);
    ASSERT_EQ(countPad, 2);
    ASSERT_EQ(countPreprocess, 1);
    ASSERT_EQ(countGenerateBitonicChunks, 1);
}
