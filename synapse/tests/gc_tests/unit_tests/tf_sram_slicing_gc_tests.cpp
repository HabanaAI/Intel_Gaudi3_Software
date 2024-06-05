#include "gc_tests/unit_tests/graph_optimizer_test.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "graph_factory.h"
#include "perf_lib_layer_params.h"

class GaudiTFSramSlicingTest
: public GraphOptimizerTest
, public testing::WithParamInterface<synDeviceType>
{
public:
    GaudiTFSramSlicingTest() { m_graphPtr = GraphFactory::createGraph(GetParam(), CompilationMode::Graph); }
    TensorPtr createPersistentTensor(unsigned          dim,
                                     const SizeArray&  sizes,
                                     synDataType       type,
                                     const std::string name     = "",
                                     const TSize*      minSizes = nullptr)
    {
        synMemoryDescriptor persistentMemoryDesc(true);
        const auto          t = createTensor(dim, sizes, type, name, minSizes);
        t->setDramOffset(m_nextTensorOffset);
        t->setMemorySectionID(m_sectionId++);
        t->setMemoryDescriptor(persistentMemoryDesc);
        m_nextTensorOffset += t->getTotalSizeInBytes();
        return t;
    }
    void addNodeToGraph(const char*        guid,
                        TensorVector       inputTensorIndices,
                        TensorVector       outputTensorIndices,
                        UserParams         userParams = nullptr,
                        const unsigned     paramsSize = 0,
                        const std::string& nodeName   = "")
    {
        GraphEditor::addNode(
            *m_graphPtr,
            NodeFactory::createNode(inputTensorIndices, outputTensorIndices, userParams, paramsSize, guid, nodeName));
    }
    TensorPtr createTensor(unsigned           dim,
                           const SizeArray&   sizes,
                           synDataType        type,
                           const std::string& name     = "",
                           const TSize*       minSizes = nullptr)
    {
        TensorPtr t;
        if (minSizes == nullptr)
        {
            t = std::make_shared<Tensor>(dim, sizes.data(), type);
        }
        else
        {
            t = std::make_shared<Tensor>(dim, sizes.data(), type, minSizes);
        }
        t->setName(name);
        return t;
    }
    TensorPtr createShapeTensor(unsigned           dim,
                                const SizeArray&   sizes,
                                synDataType        type,
                                const std::string& name     = "",
                                const TSize*       minSizes = nullptr)
    {
        auto t = std::make_shared<
            Tensor>(dim, sizes.data(), type, nullptr, nullptr, false, false, INVALID_BATCH_POS, minSizes, SHAPE_TENSOR);
        t->setName(name);
        return t;
    }
    template<typename SizesContainer>
    TensorPtr createPersistentTensor(const SizesContainer& shape,
                                     synDataType           type,
                                     const std::string&    name     = "",
                                     const TSize*          minSizes = nullptr)
    {
        SizeArray sizes;
        std::copy(shape.begin(), shape.end(), sizes.begin());
        return createPersistentTensor(shape.size(), sizes, type, name, minSizes);
    }
    template<typename SizesContainer>
    TensorPtr createTensor(const SizesContainer& shape,
                           synDataType           type,
                           const std::string&    name     = "",
                           const TSize*          minSizes = nullptr)
    {
        SizeArray sizes;
        std::copy(shape.begin(), shape.end(), sizes.begin());
        return createTensor(shape.size(), sizes, type, name, minSizes);
    }
    void compile() { ASSERT_TRUE(m_graphPtr->compile()) << "Failed to compile graph"; }

private:
    unsigned         m_sectionId        = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
    deviceAddrOffset m_nextTensorOffset = 0x1000;
    HabanaGraphPtr   m_graphPtr;
};

TEST_P(GaudiTFSramSlicingTest, bn_stage1_bwd_dedw_1024x2048_batch_128)
{
    /*
     * Problematic Scenario: bn_stage1_bwd -> DEDW(stride=2) with tensors:
     * BN Output MME Input [0] tensor_1547 : 2048x7x7x128, Sliced : 2048x7x7x11, Num of slices: 12, Buffers: 2, inSram:
     * true MME Input [1] tensor_1557 : 1024x14x14x128, Sliced : 512x14x14x11, Num of slices: 24, Buffers: 2, inSram:
     * true BN  Input [2] tensor_1543 : 2048x7x7x128, Sliced : 2048x7x7x11, Num of slices: 12, Buffers: 1, inSram: false
     * BN  Input [3] tensor_1542 : 2048x7x7x128, Sliced : 2048x7x7x11, Num of slices: 12, Buffers: 1, inSram: false
     * BN  Input [4] tensor_1545_resized_concat_with_tensor_1555_resized_out : 2048x2, Sliced : 2048x2, Num of slices:
     * 1, Buffers: 1, inSram: false BN  Input [5] TPC5355_out : 2048x2, Sliced : 2048x2, Num of slices: 1, Buffers: 1,
     * inSram: false BN  Input [6] TPC5355_beta_fake_resized_concat_with_tensor_1544_resized_out : 2048x2, Sliced :
     * 2048x2, Num of slices: 1, Buffers: 1, inSram: false BN  Output [7] TPC5355_grad_beta_gamma : 2048x2, Sliced :
     * 2048x2, Num of slices: 1, Buffers: 1, inSram: false MME Output tensor_1558 : 2048x1024x1x1, Sliced :
     * 2048x512x1x1, Num of slices: 2, Buffers: 2, inSram: true
     *
     * Compilation failed with multiple producers to a tensor
     */
    constexpr unsigned b = 128;
    constexpr unsigned h = 14;
    constexpr unsigned w = 14;
    constexpr unsigned c = 1024;
    constexpr unsigned k = 2048;

    synConvolutionParams params {};
    params.dH = 2;
    params.dW = 2;
    params.kH = 1;
    params.kW = 1;

    const SizeArray dwSizes = {k, c, params.kW, params.kH, 1};
    const SizeArray xSizes  = {c, w, h, b, 1};
    const SizeArray dySizes = {
        k,
        convOutputDimSize(xSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
        convOutputDimSize(xSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
        xSizes[3],
        1};

    const SizeArray bnParamSizes = {k, 3, 1, 1, 1};

    constexpr unsigned twoDims = 2, fourDims = 4;

    // BN inputs
    const auto inputFeatureMap = createPersistentTensor(fourDims, dySizes, syn_type_bf16, "inputFeatureMap");
    const auto gradIn          = createPersistentTensor(fourDims, dySizes, syn_type_bf16, "gradIn");
    const auto meanIstd        = createPersistentTensor(twoDims, bnParamSizes, syn_type_float, "meanIstd");
    const auto betaGamma       = createPersistentTensor(twoDims, bnParamSizes, syn_type_float, "bettaGamma");
    const auto sumDotP         = createPersistentTensor(twoDims, bnParamSizes, syn_type_float, "SumDotP");
    // BN Outputs
    const auto gradBetaGamma = createTensor(twoDims, bnParamSizes, syn_type_float);
    const auto gradOut       = createTensor(fourDims, dySizes, syn_type_bf16);

    ns_BatchNormStage1Kernel::Params bnParams {};
    bnParams.N = multiplyElements(dySizes.data() + 1, dySizes.data() + 4);

    addNodeToGraph("batch_norm_stage1_bwd_bf16",
                   {inputFeatureMap, gradIn, meanIstd, sumDotP, betaGamma},
                   {gradOut, gradBetaGamma},
                   &bnParams,
                   sizeof(bnParams),
                   "bn_stage1_bwd");

    // DEDW inputs
    const auto dy = gradOut;
    const auto x  = createPersistentTensor(fourDims, xSizes, syn_type_bf16, "x");
    // DEDW output
    auto dw = createPersistentTensor(fourDims, dwSizes, syn_type_bf16, "dw");

    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dy, x}, {dw}, &params, sizeof(params), "DEDW");

    compile();
}
TEST_P(GaudiTFSramSlicingTest, relu_bwd_conv_dedw_relu_L2)
{
    /*
     * Original Input [0] tensor_1178 : 256x14x14x512, Sliced : 256x14x14x6, Num of slices: 86, Buffers: 2, inSram: true
     * Original Input [1] tensor_1134 : 256x28x28x512, Sliced : 256x28x28x6, Num of slices: 86, Buffers: 2, inSram: true
     * Original Input [2] tensor_1133 : 256x28x28x512, Sliced : 256x28x28x6, Num of slices: 86, Buffers: 1, inSram:
     * false Original Input [3] tensor_1126 : 256x28x28x512, Sliced : 256x28x28x6, Num of slices: 86, Buffers: 1,
     * inSram: false Original Input [4] tensor_1140 : 256x256x2x2, Sliced : 256x256x2x2, Num of slices: 1, Buffers: 1,
     * inSram: true Original Input [5] tensor_1141 : 256x14x14x512, Sliced : 256x14x14x6, Num of slices: 86, Buffers: 2,
     * inSram: true Original Input [6] tensor_1146_reshaped : 256x14x14x512, Sliced : 256x14x14x6, Num of slices: 86,
     * Buffers: 1, inSram: false Original Input [7] tensor_1147_reshaped : 256x14x14x512, Sliced : 256x14x14x6, Num of
     * slices: 86, Buffers: 1, inSram: false Original Output tensor_1179 : 256x256x2x2, Sliced : 256x256x2x2, Num of
     * slices: 1, Buffers: 1, inSram: true
     *
     * Compilation failed with tensor_1134_slice_0_0_0_85_0__0__bundle_93 is reused while still alive in node
     * tensor_1134_bundle_93_memcpy_1
     */

    constexpr unsigned b = 512;
    constexpr unsigned h = 28;
    constexpr unsigned w = 28;
    constexpr unsigned c = 256;
    constexpr unsigned k = 256;

    synConvolutionParams params {};
    params.dH = 2;
    params.dW = 2;
    params.kH = 2;
    params.kW = 2;

    const SizeArray wSizes  = {k, c, params.kW, params.kH, 1};
    const SizeArray dwSizes = {k, c, params.kW, params.kH, 1};
    const SizeArray dySizes = {c, w / 2, h / 2, b, 1};
    const SizeArray xSizes  = {c, w, h, b, 1};
    const SizeArray ySizes  = {
        k,
        convOutputDimSize(xSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
        convOutputDimSize(xSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
        xSizes[3],
        1};

    constexpr unsigned fourDims = 4;

    // relu bwd in
    const auto gradIn = createPersistentTensor(fourDims, xSizes, syn_type_float, "gradIn");  // tensor_1133
    const auto IFM    = createPersistentTensor(fourDims, xSizes, syn_type_float, "ifm");     // tensor_1126
    // relu bwd out
    const auto gradOut = createPersistentTensor(fourDims, xSizes, syn_type_float, "gradOut");  // tensor_1134

    addNodeToGraph("relu_bwd_f32", {gradIn, IFM}, {gradOut}, nullptr, 0, "relu");

    // // Conv inputs
    const auto x_conv = gradOut;  // tensor_1134

    const auto wgh = createPersistentTensor(fourDims, wSizes, syn_type_float, "w");  // tensor_1140

    // Conv output
    const auto y = createPersistentTensor(fourDims, ySizes, syn_type_float, "y");  // tensor_1141

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {x_conv, wgh}, {y}, &params, sizeof(params), "Conv");

    // DEDW inputs
    const auto x = gradOut;  // tensor_1134

    const auto dy = createPersistentTensor(fourDims, dySizes, syn_type_float, "dy");  // tensor_1178

    // DEDW output
    const auto dw = createPersistentTensor(fourDims, dwSizes, syn_type_float, "dw");  // tensor_1179

    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dy, x}, {dw}, &params, sizeof(params), "dedw");

    // ADD inputs
    const auto in1 = createPersistentTensor(fourDims, dwSizes, syn_type_float, "in1");  // tensor_1180

    const auto in2 = dw;  // tensor_1179

    // ADD output
    const auto add = createPersistentTensor(fourDims, dwSizes, syn_type_float, "add");  // tensor_1181

    addNodeToGraph("add_fwd_f32", {in1, in2}, {add});

    // // relu in
    const auto reluIn3 = y;                                                                     // tensor_1141
    const auto reluIn4 = createPersistentTensor(fourDims, dySizes, syn_type_float, "reluIn4");  // tensor_1146_reshaped

    // relu out
    auto reluOut2 = createPersistentTensor(fourDims, dySizes, syn_type_float, "reluOut2");  // tensor_1179

    addNodeToGraph("relu_bwd_f32", {reluIn3, reluIn4}, {reluOut2}, nullptr, 0, "relu2");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, cast_conv_dedw)
{
    /*
     * Original Input [0] tensor_3520 : 256x14x14x512, Sliced : 256x14x14x12, Num of slices: 43, Buffers: 2, inSram:
     * true, alignedToCL:false Original Input [1] tensor_3516 : 256x28x28x512, Sliced : 256x28x28x12, Num of slices: 43,
     * Buffers: 2, inSram: true, alignedToCL:false Original Input [2] tensor_3509 : 256x28x28x512, Sliced :
     * 256x28x28x12, Num of slices: 43, Buffers: 1, inSram: false, alignedToCL:false Original Input [3] tensor_3517 :
     * 256x256x2x2, Sliced : 256x256x2x2, Num of slices: 1, Buffers: 1, inSram: true, alignedToCL:false Original Input
     * [4] tensor_3518 : 256x14x14x512, Sliced : 256x14x14x12, Num of slices: 43, Buffers: 1, inSram: false,
     * alignedToCL:false Original Output tensor_3521 : 256x256x2Handling all operations for bundle 125x2, Sliced :
     * 256x256x2x2, Num of slices: 1, Buffers: 1, inSram: true, alignedToCL:false
     *
     * Compilation failed with tensor_3516_slice_0_0_0_42_0__0__bundle_125 is reused while still alive in node
     * tensor_3516_bundle_125_memcpy_1
     */

    constexpr unsigned b = 512;
    constexpr unsigned h = 28;
    constexpr unsigned w = 28;
    constexpr unsigned c = 256;
    constexpr unsigned k = 256;

    synConvolutionParams params {};
    params.dH = 2;
    params.dW = 2;
    params.kH = 2;
    params.kW = 2;

    const SizeArray wSizes  = {k, c, params.kW, params.kH, 1};
    const SizeArray dwSizes = {k, c, params.kW, params.kH, 1};
    const SizeArray dySizes = {c, w / 2, h / 2, b, 1};
    const SizeArray xSizes  = {c, w, h, b, 1};
    const SizeArray ySizes  = {
        k,
        convOutputDimSize(xSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
        convOutputDimSize(xSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
        xSizes[3],
        1};

    constexpr unsigned fourDims = 4;

    // // cast in
    const auto castIn = createPersistentTensor(fourDims, xSizes, syn_type_float, "IFM");  // tensor_3509

    // cast out
    const auto castOut = createPersistentTensor(fourDims, xSizes, syn_type_bf16, "castOut");  // tensor_3516

    addNodeToGraph("cast_f32_to_bf16", {castIn}, {castOut}, nullptr, 0, "cast");

    // Conv inputs
    const auto x_conv = castOut;  // tensor_3516

    const auto wgh = createPersistentTensor(fourDims, wSizes, syn_type_bf16, "w");

    // Conv output
    const auto y = createPersistentTensor(fourDims, ySizes, syn_type_bf16, "y");  // tensor_3518

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {x_conv, wgh}, {y}, &params, sizeof(params), "Conv");

    // DEDW inputs
    const auto x = castOut;  // tensor_3516

    const auto dy = createPersistentTensor(fourDims, dySizes, syn_type_bf16, "dy");  // tensor_3520

    // DEDW output
    const auto dw = createPersistentTensor(fourDims, dwSizes, syn_type_bf16, "dw");  // tensor_3521

    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dy, x}, {dw}, &params, sizeof(params), "dedw");

    // // cast in
    const auto castIn2 = dw;  // tensor_3521

    // cast out
    const auto castOut2 = createPersistentTensor(fourDims, dwSizes, syn_type_float, "castOut2");  // tensor_3522

    addNodeToGraph("cast_bf16_to_f32", {castIn2}, {castOut2}, nullptr, 0, "cast2");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, relu_conv_3x3_strided_bn_inf_f32)
{
    /*
     * Problematic Scenario: relu -> conv 3x3 strided -> batch_norm_inf:
     * Original Input [0] tensor_11099 : 512x14x14x128, Sliced : 512x14x14x16, Num of slices: 8, Buffers: 2, inSram:
     * true Original Input [1] tensor_10603 : 512x512x3x3, Sliced : 128x512x3x3, Num of slices: 4, Buffers: 2, inSram:
     * true Original Input [2] tensor_11097 : 512x14x14x128, Sliced : 512x14x14x16, Num of slices: 8, Buffers: 1,
     * inSram: false Original Input [3] tensor_11108 : 512, Sliced : 128, Num of slices: 4, Buffers: 1, inSram: false
     * Original Input [4] tensor_11107 : 512, Sliced : 128, Num of slices: 4, Buffers: 1, inSram: false
     * Original Input [5] tensor_11109 : 512, Sliced : 128, Num of slices: 4, Buffers: 1, inSram: false
     * Original Input [6] tensor_11110 : 512, Sliced : 128, Num of slices: 4, Buffers: 1, inSram: false
     * Original Input [7] tensor_11111 : 512x7x7x128, Sliced : 128x7x7x16, Num of slices: 32, Buffers: 1, inSram: false
     * Original Output tensor_11100 : 512x7x7x128, Sliced : 128x7x7x16, Num of slices: 32, Buffers: 2, inSram: true
     * Compilation failed with multiple producers to a tensor
     */

    constexpr unsigned b = 128;
    constexpr unsigned h = 14;
    constexpr unsigned w = 14;
    constexpr unsigned c = 512;
    constexpr unsigned k = 512;

    synConvolutionParams params = {};
    params.dH                   = 2;
    params.dW                   = 2;
    params.kH                   = 3;
    params.kW                   = 3;

    const SizeArray wSizes = {k, c, params.kW, params.kH, 1};
    const SizeArray xSizes = {c, w, h, b, 1};
    const SizeArray ySizes = {
        k,
        convOutputDimSize(xSizes[1], params.kW, params.dW, params.padL + params.padR, params.dilW),
        convOutputDimSize(xSizes[2], params.kH, params.dH, params.padT + params.padB, params.dilH),
        xSizes[3],
        1};

    const SizeArray bnParamSizes = {k, 1, 1, 1, 1};

    constexpr unsigned fourDims = 4, oneDim = 1;

    // relu in
    const auto reluIn = createPersistentTensor(fourDims, xSizes, syn_type_float, "relu_in");

    // relu out
    const auto reluOut = createTensor(fourDims, xSizes, syn_type_float);

    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut}, nullptr, 0, "relu");

    // Conv inputs
    const auto x = reluOut;

    const auto wgh = createPersistentTensor(fourDims, wSizes, syn_type_float, "w");

    // Conv output
    const auto y = createTensor(fourDims, ySizes, syn_type_float);

    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {x, wgh}, {y}, &params, sizeof(params), "Conv");

    // BN inputs
    const auto inputFeatureMap = y;
    const auto beta            = createPersistentTensor(oneDim, bnParamSizes, syn_type_float, "beta");
    const auto gamma           = createPersistentTensor(oneDim, bnParamSizes, syn_type_float, "gamma");
    const auto mean            = createPersistentTensor(oneDim, bnParamSizes, syn_type_float, "mean");
    const auto var             = createPersistentTensor(oneDim, bnParamSizes, syn_type_float, "var");

    // BN outputs
    const auto gradOut = createPersistentTensor(fourDims, ySizes, syn_type_float, "bn_out");

    ns_BatchNormKernel::Params bnParams {};

    addNodeToGraph("batch_norm_inf_f32",
                   {inputFeatureMap, beta, gamma, mean, var},
                   {gradOut},
                   &bnParams,
                   sizeof(bnParams),
                   "batch_norm_inf_f32");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, cast_gemm)
{
    // just simulate the same problem as appeared in
    // TestCiResnet.test_ci_resnet_bfloat16_acc_target[overfitTrue-lr0.01-batch32-timeout900]
    //
    //  original graph :  (cast_f32_to_bf16)->[t2]->(gemm)
    //
    // (original problem - after the sram slicing passs,
    // the pass optimizeTPCKernels added a "packing"(reshape) node in the middle of the bundle (correctly)
    // but then the optimize logical operations pass added "memcpy" node between the "split" and the "packing" nodes.
    // This new "memcpy" node in the middle of the bundle didn't contain bundle id and therefore messed the
    // execution schedule

    constexpr unsigned w = 32;
    constexpr unsigned c = 2048;
    constexpr unsigned k = 1001;

    const SizeArray gemmIn1Sizes = {c, w};
    const SizeArray gemmIn2Sizes = {k, c};
    const SizeArray gemmOutSizes = {k, w};

    constexpr unsigned twoDims = 2;

    // tpc cast in
    const auto tpcCastIn = createPersistentTensor(twoDims, gemmIn2Sizes, syn_type_float, "tpc_in");

    // tpc cast out
    const auto tpcCastOut = createTensor(twoDims, gemmIn2Sizes, syn_type_bf16);

    addNodeToGraph("cast_f32_to_bf16", {tpcCastIn}, {tpcCastOut}, nullptr, 0, "Cast_f32_to_bf16");

    // gemm inputs
    const auto gemmIn1 = createPersistentTensor(twoDims, gemmIn1Sizes, syn_type_bf16, "gemm_in1");

    const auto gemmIn2 = tpcCastOut;

    const auto gemmOut = createTensor(twoDims, gemmOutSizes, syn_type_bf16);

    synGEMMParams params {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {gemmIn1, gemmIn2}, {gemmOut}, &params, sizeof(params), "GEMM");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, cast_gemm_narrow_producer)
{
    // Problem appeared in BERT TF script:
    /*
     * Original Input [0] tensor_2407 : 4096x4096, Sliced : 4096x1024, Num of slices: 4, Buffers: 2, inSram: true
     * Original Input [1] tensor_1707 : 1024x4096, Sliced : 128x4096, Num of slices: 8, Buffers: 2, inSram: true
     * Original Input [2] tensor_1706 : 1024x4096, Sliced : 128x4096, Num of slices: 8, Buffers: 1, inSram: false
     * Original Output tensor_2408 : 1024x4096, Sliced : 128x1024, Num of slices: 32, Buffers: 1, inSram: false
     */

    constexpr unsigned w = 4096;
    constexpr unsigned c = 4096;
    constexpr unsigned k = 1024;

    const SizeArray gemmIn1Sizes = {c, w};
    const SizeArray gemmIn2Sizes = {k, c};
    const SizeArray gemmOutSizes = {k, w};

    constexpr unsigned twoDims = 2;

    // tpc cast in
    const auto tpcCastIn = createPersistentTensor(twoDims, gemmIn2Sizes, syn_type_float, "tpc_in");

    // tpc cast out
    const auto tpcCastOut = createTensor(twoDims, gemmIn2Sizes, syn_type_bf16);

    addNodeToGraph("cast_f32_to_bf16", {tpcCastIn}, {tpcCastOut}, nullptr, 0, "Cast_f32_to_bf16");

    // gemm inputs
    const auto gemmIn1 = createPersistentTensor(twoDims, gemmIn1Sizes, syn_type_bf16, "gemm_in1");

    const auto gemmIn2 = tpcCastOut;

    const auto gemmOut = createTensor(twoDims, gemmOutSizes, syn_type_bf16);

    synGEMMParams params {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {gemmIn1, gemmIn2}, {gemmOut}, &params, sizeof(params), "GEMM");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, cast_batch_gemm_cast)
{
    constexpr unsigned b = 16;
    constexpr unsigned w = 32;
    constexpr unsigned c = 32;
    constexpr unsigned k = 32;

    const SizeArray xsize = {c, w, b};
    const SizeArray wsize = {k, c, b};
    const SizeArray ysize = {k, w, b};

    constexpr unsigned threeDims = 3;

    const auto tpcCastIn0 = createPersistentTensor(threeDims, xsize, syn_type_float, "IFM");
    // tpc cast out
    const auto tpcCastOut0 = createTensor(threeDims, xsize, syn_type_bf16);

    addNodeToGraph("cast_f32_to_bf16", {tpcCastIn0}, {tpcCastOut0}, nullptr, 0, "Cast_f32_to_bf16");

    const auto gemmX = tpcCastOut0;
    const auto gemmW = createPersistentTensor(threeDims, wsize, syn_type_bf16, "gemmW");

    const auto gemmY = createTensor(threeDims, ysize, syn_type_bf16);

    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {gemmX, gemmW}, {gemmY}, &gemmParams, sizeof(gemmParams));

    const auto tpcCastIn1 = gemmY;
    // tpc cast out
    const auto tpcCastOut1 = createPersistentTensor(threeDims, ysize, syn_type_float);

    addNodeToGraph("cast_bf16_to_f32", {tpcCastIn1}, {tpcCastOut1}, nullptr, 0, "cast_bf16_to_f32");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, cast_conv_bn)
{
    // From: test_ci_resnet.py::TestCiResnet::test_accuracy_resnet50_quick_overfit
    //
    // Bundle-ID:  52 (Solved)   Node: TPC3290 [cast_f32_to_bf16]
    // Bundle-ID:  52 (Solved)   Node: Convolution4657 [Convolution]
    // Bundle-ID: N/A (N/A)      Node: TPC4661_memset_zero [Memset]
    // Bundle-ID:  52 (Solved)   Node: TPC4661_stage1 [batch_norm_stage1_fwd_bf16]
    //
    // Slicing Strategy - Left-to-Right , 4Wx1H, graph size optimized: false
    // Original Input [0] tensor_2406_flattened : 512x196x1x1, Sliced : 512x196x1x1, Num of slices: 1, Buffers: 1,
    // inSram: true Original Input [1] tensor_515 : 2048x512x1x1, Sliced : 512x512x1x1, Num of slices: 4, Buffers: 2,
    // inSram: true Original Input [2] tensor_514 : 2048x512x1x1, Sliced : 512x512x1x1, Num of slices: 4, Buffers: 1,
    // inSram: false Original Input [3] tensor_2420 : 2048, Sliced : 512, Num of slices: 4, Buffers: 1, inSram: false
    // Original Input [4] TPC4661_in : 2048x2, Sliced : 512x2, Num of slices: 4, Buffers: 1, inSram: false
    // Original Output tensor_2407_flattened : 2048x196x1x1, Sliced : 512x196x1x1, Num of slices: 4, Buffers: 2, inSram:
    // true

    const SizeArray ifmSizes = {512, 196, 1, 1};
    const SizeArray wghSizes = {2048, 512, 1, 1};
    const SizeArray ofmSizes = {2048, 196, 1, 1};
    const SizeArray channels = {2048};

    constexpr unsigned fourDims = 4, oneDim = 1;

    const auto castIn  = createPersistentTensor(fourDims, wghSizes, syn_type_float);
    const auto castOut = createTensor(fourDims, wghSizes, syn_type_bf16);
    addNodeToGraph("cast_f32_to_bf16", {castIn}, {castOut}, nullptr, 0, "cast");

    const auto convIn  = createPersistentTensor(fourDims, ifmSizes, syn_type_bf16);
    const auto convWGH = castOut;
    const auto convOut = createTensor(fourDims, ofmSizes, syn_type_bf16);

    synConvolutionParams convParams {};
    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {convIn, convWGH},
                   {convOut},
                   &convParams,
                   sizeof convParams,
                   "conv");

    const auto bnIn            = convOut;
    const auto bnBetta         = createPersistentTensor(oneDim, channels, syn_type_float);
    const auto bnGamma         = createPersistentTensor(oneDim, channels, syn_type_float);
    const auto bnInRunningMean = createPersistentTensor(oneDim, channels, syn_type_float);
    const auto bnInRunningVar  = createPersistentTensor(oneDim, channels, syn_type_float);

    const auto bnOut            = createTensor(fourDims, ofmSizes, syn_type_bf16);
    const auto bnMean           = createPersistentTensor(oneDim, channels, syn_type_float);
    const auto bnIstd           = createPersistentTensor(oneDim, channels, syn_type_float);
    const auto bnOutRunningMean = createPersistentTensor(oneDim, channels, syn_type_float);
    const auto bnOutRunningVar  = createPersistentTensor(oneDim, channels, syn_type_float);

    ns_BatchNormKernel::Params bnParams {};
    bnParams.epsilon     = 1e-5;
    bnParams.momentum    = 0.1;
    bnParams.threshold.f = 1.;
    addNodeToGraph("batch_norm_fwd_bf16",
                   {bnIn, bnBetta, bnGamma, bnInRunningMean, bnInRunningVar},
                   {bnOut, bnMean, bnIstd, bnOutRunningMean, bnOutRunningVar},
                   &bnParams,
                   sizeof bnParams,
                   "BN");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, gemm_reshape_add)
{
    /*
     * BERT compilation failed on the following graph due to mismatch dimension count between the GEMM and ADD.
     *  GEMM Input [0] tensor_1109 : 2x384, Sliced : 2x384, Num of slices: 1, Buffers: 1, inSram: true
     *  GEMM Input [1] tensor_1110 : 768x2, Sliced : 416x2, Num of slices: 2, Buffers: 2, inSram: true
     *  GEMM Output tensor_1111 : 768x384, Sliced : 416x384, Num of slices: 2, Buffers: 2, inSram: true
     *  Reshape GEMM output to Add input (add an outer dimension with size=1)
     *  Add  Input tensor_1113 : 768x384x1, Sliced : 416x384x1, Num of slices: 2, Buffers: 1, inSram: false
     *  Add  Output tensor_1114 : 768x384x1, Sliced : 416x384x1, Num of slices: 2, Buffers: 1, inSram: false
     */

    constexpr unsigned m = 384, k = 2, n = 768;

    const SizeArray aSize   = {k, m};
    const SizeArray bSize   = {n, k};
    const SizeArray oSize   = {n, m};
    const SizeArray addSize = {n, m, 1};

    constexpr unsigned twoDims = 2, threeDims = 3;

    const auto a = createPersistentTensor(twoDims, aSize, syn_type_float);
    const auto b = createPersistentTensor(twoDims, bSize, syn_type_float);

    const auto o = createTensor(twoDims, oSize, syn_type_float);

    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {a, b}, {o}, &gemmParams, sizeof gemmParams, "GEMM");

    const auto reshapeIn = o;

    const auto reshapeOut = createTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {reshapeIn}, {reshapeOut});

    const auto addIn0 = reshapeOut;
    const auto addIn1 = createPersistentTensor(threeDims, addSize, syn_type_float);
    const auto addOut = createPersistentTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph("add_fwd_f32", {addIn0, addIn1}, {addOut});

    compile();
}

TEST_P(GaudiTFSramSlicingTest, add_reshape_gemm)
{
    /* Symmetric case to gemm_reshape_add, just to verify */

    constexpr unsigned m = 384, k = 2, n = 768;

    const SizeArray aSize   = {k, m};
    const SizeArray bSize   = {n, k};
    const SizeArray oSize   = {n, m};
    const SizeArray addSize = {n, k, 1};

    constexpr unsigned twoDims = 2, threeDims = 3;

    const auto addIn0 = createPersistentTensor(threeDims, addSize, syn_type_float);
    const auto addIn1 = createPersistentTensor(threeDims, addSize, syn_type_float);

    const auto addOut = createTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph("add_fwd_f32", {addIn0, addIn1}, {addOut});

    const auto reshapeIn = addOut;

    const auto reshapeOut = createTensor(twoDims, bSize, syn_type_float);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {reshapeIn}, {reshapeOut});

    const auto    a = createPersistentTensor(twoDims, aSize, syn_type_float);
    const auto    b = reshapeOut;
    const auto    o = createTensor(twoDims, oSize, syn_type_float);
    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {a, b}, {o}, &gemmParams, sizeof gemmParams, "GEMM");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, cast_narrow_conv_bn_stage1)
{
    /*
     * Cast Input                   : 2048x512x1x1, Sliced : 512x512x1x1, Num of slices: 4, Buffers: 1, inSram: false
     * Conv Input [0]               : 512x3136x1x1, Sliced : 512x448x1x1, Num of slices: 7, Buffers: 2, inSram: true
     * Conv Input [1] (cast output) : 2048x512x1x1, Sliced : 512x512x1x1, Num of slices: 4, Buffers: 2, inSram: true
     * Conv Output bn input [0]     : 2048x3136x1x1, Sliced : 512x448x1x1, Num of slices: 28, Buffers: 2, inSram: true
     * BN Input [1]                 : 2048, Sliced : 512, Num of slices: 4, Buffers: 1, inSram: false
     * BN output                    : 2048x2, Sliced : 512x2, Num of slices: 4, Buffers: 1, inSram: false
     */

    constexpr unsigned b = 1, h = 1, w = 3136, c = 512, k = 2048, r = 1, s = 1;

    const SizeArray convInSize  = {c, w, h, b};
    const SizeArray convWghSize = {k, c, s, r};
    const SizeArray convOutSize = {k, w, h, b};

    const SizeArray bn1DSize = {k};

    constexpr unsigned fourDims = 4, oneDim = 1;

    const auto castIn = createPersistentTensor(fourDims, convWghSize, syn_type_float);

    const auto castOut = createTensor(fourDims, convWghSize, syn_type_bf16);
    addNodeToGraph("cast_f32_to_bf16", {castIn}, {castOut}, nullptr, 0, "Cast");

    const auto convIn  = createPersistentTensor(fourDims, convInSize, syn_type_bf16);
    const auto convWgh = castOut;
    const auto convOut = createTensor(fourDims, convOutSize, syn_type_bf16);

    synConvolutionParams convParams {};
    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {convIn, convWgh},
                   {convOut},
                   &convParams,
                   sizeof convParams,
                   "Conv");

    const auto bnIFM = convOut;
    const auto bnIn0 = createPersistentTensor(oneDim, bn1DSize, syn_type_float);
    const auto bnIn1 = createPersistentTensor(oneDim, bn1DSize, syn_type_float);
    const auto bnIn2 = createPersistentTensor(oneDim, bn1DSize, syn_type_float);
    const auto bnIn3 = createPersistentTensor(oneDim, bn1DSize, syn_type_float);

    const auto bnOut0 = createTensor(fourDims, convOutSize, syn_type_bf16);
    const auto bnOut1 = createTensor(oneDim, bn1DSize, syn_type_float);
    const auto bnOut2 = createTensor(oneDim, bn1DSize, syn_type_float);
    const auto bnOut3 = createTensor(oneDim, bn1DSize, syn_type_float);
    const auto bnOut4 = createTensor(oneDim, bn1DSize, syn_type_float);

    ns_BatchNormKernel::Params bnParams {};
    bnParams.threshold.f = 0.1;
    bnParams.momentum    = 0.1;
    bnParams.epsilon     = 1e-5;
    addNodeToGraph("batch_norm_fwd_bf16",
                   {bnIFM, bnIn0, bnIn1, bnIn2, bnIn3},
                   {bnOut0, bnOut1, bnOut2, bnOut3, bnOut4},
                   &bnParams,
                   sizeof bnParams,
                   "BN");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, gemm_reshape_add_broadcast)
{
    //   GEMM Input [0] tensor_0 : 768x4, Sliced : 768x4, Num of slices: 1, Buffers: 1, inSram: true
    //   GEMM Input [1] tensor_1 : 768x768, Sliced : 721x768, Num of slices: 2, Buffers: 2, inSram: true
    //   GEMM Output tensor_2 : 768x4, Sliced : 721x4, Num of slices: 2, Buffers: 2, inSram: true
    //   Reshape GEMM output to Add input (add an outer dimension with size=1)
    //   TPC add (broadcast) Input [2] tensor_3 : 768x1x1, Sliced : 721x1x1, Num of slices: 2, Buffers: 1, inSram: false
    //   TPC add Input [3] tensor_4 : 768x4x1, Sliced : 721x4x1, Num of slices: 2, Buffers: 1, inSram: false

    constexpr unsigned h = 4, cd = 768, w = 768;

    const SizeArray aSize            = {cd, h};
    const SizeArray bSize            = {w, cd};
    const SizeArray oSize            = {w, h};
    const SizeArray addSize          = {w, h, 1};
    const SizeArray addBroadcastSize = {w, 1, 1};

    constexpr unsigned twoDims = 2, threeDims = 3;

    const auto    a = createPersistentTensor(twoDims, aSize, syn_type_float);
    const auto    b = createPersistentTensor(twoDims, bSize, syn_type_float);
    const auto    o = createTensor(twoDims, oSize, syn_type_float);
    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {a, b}, {o}, &gemmParams, sizeof gemmParams, "GEMM");

    const auto reshapeIn  = o;
    const auto reshapeOut = createTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {reshapeIn}, {reshapeOut});

    const auto addInBroadcast  = createPersistentTensor(threeDims, addBroadcastSize, syn_type_float);
    const auto addInReshapeOut = reshapeOut;
    const auto addOut          = createPersistentTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph("add_fwd_f32", {addInBroadcast, addInReshapeOut}, {addOut});

    compile();
}

TEST_P(GaudiTFSramSlicingTest, trnsposed_shared_input_gemms_with_producer)
{
    /*
     * https://jira.habana-labs.com/browse/SW-15477
     * Compilation issue in DLRM dot.
     *
     * ADD Input [0]: 1024x128, Sliced : 192x128, Num of slices: 6, Buffers: 1, inSram: false
     * ADD Input [1]: 1024x128, Sliced : 192x128, Num of slices: 6, Buffers: 1, inSram: false
     * ADD output +
     * GEMM0 Input [1] +
     * GEMM1 Input [0]: 1024x128, Sliced : 192x128, Num of slices: 6, Buffers: 2, inSram: true
     *
     * GEMM0 Input [0]: 383x128, Sliced : 192x128, Num of slices: 2, Buffers: 2, inSram: true
     * GEMM0 Output: 1024x383, Sliced : 192x192, Num of slices: 12, Buffers: 1, inSram: false
     *
     * GEMM1 Input [1]: 1024x383, Sliced : 192x383, Num of slices: 6, Buffers: 2, inSram: true
     * GEMM1 Output: 383x128, Sliced : 383x128, Num of slices: 1, Buffers: 1, inSram: true
     */

    const SizeArray sharedSizes = {1024, 128};

    const SizeArray gemm0nonSharedSizes = {383, 128};
    const SizeArray gemm0OutputSizes    = {1024, 383};

    const SizeArray gemm1nonSharedSizes = {1024, 383};
    const SizeArray gemm1OutputSizes    = {383, 128};

    constexpr unsigned twoDims = 2;

    const auto addIn0 = createPersistentTensor(twoDims, sharedSizes, syn_type_float);
    const auto addIn1 = createPersistentTensor(twoDims, sharedSizes, syn_type_float);
    const auto addOut = createTensor(twoDims, sharedSizes, syn_type_float);
    addNodeToGraph("add_fwd_f32", {addIn0, addIn1}, {addOut}, nullptr, 0, "ADD");

    const auto gemm0In1gemm1In0 = addOut;

    const auto    gemm0In0 = createPersistentTensor(twoDims, gemm0nonSharedSizes, syn_type_float);
    const auto    gemm0Out = createTensor(twoDims, gemm0OutputSizes, syn_type_float);
    synGEMMParams gemm0Params {};
    gemm0Params.transpose_a = true;

    const auto    gemm1In1 = createPersistentTensor(twoDims, gemm1nonSharedSizes, syn_type_float);
    const auto    gemm1Out = createTensor(twoDims, gemm1OutputSizes, syn_type_float);
    synGEMMParams gemm1Params {};
    gemm1Params.transpose_b = true;

    // The topology does not dictate order between gemm 1 and 0. In order to reproduce the scenario above, GEMM 0 needs
    // to be the primary gemm in the bundle (master). For that, its original bundle needs to be the first expanded.
    // This will happen if it is scheduled after gemm1, which will happen if it has a bigger node ID than GEMM1.
    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {gemm0In1gemm1In0, gemm1In1},
                   {gemm1Out},
                   &gemm1Params,
                   sizeof gemm1Params,
                   "GEMM1");
    addNodeToGraph(NodeFactory::gemmNodeTypeName,
                   {gemm0In0, gemm0In1gemm1In0},
                   {gemm0Out},
                   &gemm0Params,
                   sizeof gemm0Params,
                   "GEMM0");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, gemm_reshape_add_broadcast_2)
{
    // JIRA: SW-23846

    //   Slicing Strategy - Left-to-Right , 4Wx1H, graph size optimized: false
    //   Original Input [0] autoGenPersistInputTensorName_0 : 768x4096, Sliced : 768x256, Num of slices: 16, Buffers: 2,
    //   inSram: true Original Input [1] autoGenPersistInputTensorName_1 : 768x768, Sliced : 256x768, Num of slices: 3,
    //   Buffers: 2, inSram: true Original Input [2] autoGenPersistInputTensorName_2 : 768x1x1, Sliced : 256x1x1, Num of
    //   slices: 3, Buffers: 1, inSram: false Original Input [3] autoGenPersistOutputTensorName_3_reshaped : 768x4096,
    //   Sliced : 256x256, Num of slices: 48, Buffers: 1, inSram: false Original Output Tensor-2 : 768x4096, Sliced :
    //   256x256, Num of slices: 48, Buffers: 2, inSram: true

    constexpr unsigned h = 4096, cd = 768, w = 768, k = 128, p = 32;

    const SizeArray aSize            = {cd, h};
    const SizeArray bSize            = {w, cd};
    const SizeArray oSize            = {w, h};
    const SizeArray addSize          = {w, k, p};
    const SizeArray addBroadcastSize = {w, 1, 1};

    constexpr unsigned twoDims = 2, threeDims = 3;

    const auto    a = createPersistentTensor(twoDims, aSize, syn_type_float);
    const auto    b = createPersistentTensor(twoDims, bSize, syn_type_float);
    const auto    o = createTensor(twoDims, oSize, syn_type_float);
    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {a, b}, {o}, &gemmParams, sizeof(gemmParams), "GEMM");
    const auto reshapeIn = o;

    const auto reshapeOut = createTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {reshapeIn}, {reshapeOut});
    const auto addInBroadcast  = createPersistentTensor(threeDims, addBroadcastSize, syn_type_float);
    const auto addInReshapeOut = reshapeOut;
    const auto addOut          = createPersistentTensor(threeDims, addSize, syn_type_float);
    addNodeToGraph("add_fwd_f32", {addInReshapeOut, addInBroadcast}, {addOut});
    compile();
}

TEST_P(GaudiTFSramSlicingTest, tf_dsd_basic)
{
    // changeTensorsOffsetDiffsToBeWide();

    /*************
     * n8_Conv2D node
     * inputs: [t13[1, 28, 28, 64](dtype=float32), t11[32, 1, 5, 5](dtype=float32)]
     * output: [t15_Conv2D_0[32, 28, 28, 64](dtype=float32)]
     *************/

    // create t13 tensor
    const SizeArray t13_sizes = {1, 28, 28, 64};
    const auto      t13       = createPersistentTensor(4, t13_sizes, syn_type_single, "t13");

    // create t11 tensor
    const SizeArray t11_sizes = {32, 1, 5, 5};
    const auto      t11       = createPersistentTensor(4, t11_sizes, syn_type_single, "t11");

    // create t15_Conv2D_0 tensor
    const SizeArray t15_Conv2D_0_sizes = {32, 28, 28, 64};
    const auto      t15_Conv2D_0       = createTensor(4, t15_Conv2D_0_sizes, syn_type_single, "t15_Conv2D_0");

    unsigned char n8_Conv2D_params[] = {5, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0,  0,  2, 0, 0, 0,
                                        2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 39, 80, 1, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {t13, t11},
                   {t15_Conv2D_0},
                   (void*)n8_Conv2D_params,
                   sizeof(n8_Conv2D_params),
                   "n8_Conv2D");

    /*************
     * n9_BiasAdd node
     * inputs: [t14[32](dtype=float32), t18[32, 1, 1, 1](dtype=uint32)]
     * output: [t17[32, 1, 1, 1](dtype=float32)]
     *************/

    // create t14 tensor
    const SizeArray t14_sizes = {32};
    const auto      t14       = createPersistentTensor(1, t14_sizes, syn_type_single, "t14");

    // create t18 tensor
    const SizeArray t18_sizes = {32, 1, 1, 1};
    const auto      t18       = createShapeTensor(4, t18_sizes, syn_type_uint32, "t18");

    // create t17 tensor
    const SizeArray t17_sizes = {32, 1, 1, 1};
    const auto      t17       = createTensor(4, t17_sizes, syn_type_single, "t17");

    addNodeToGraph("reshape", {t14, t18}, {t17}, nullptr, 0, "n9_BiasAdd");

    /*************
     * n10_BiasAdd node
     * inputs: [t15_Conv2D_0[32, 28, 28, 64](dtype=float32), t17[32, 1, 1, 1](dtype=float32)]
     * output: [t16_BiasAdd_0[32, 28, 28, 64](dtype=float32)]
     *************/

    // create t16_BiasAdd_0 tensor
    const SizeArray t16_BiasAdd_0_sizes = {32, 28, 28, 64};
    const auto      t16_BiasAdd_0       = createTensor(4, t16_BiasAdd_0_sizes, syn_type_single, "t16_BiasAdd_0");

    addNodeToGraph("add_fwd_f32", {t15_Conv2D_0, t17}, {t16_BiasAdd_0}, nullptr, 0, "n10_BiasAdd");

    /*************
     * n11_Relu node
     * inputs: [t16_BiasAdd_0[32, 28, 28, 64](dtype=float32)]
     * output: [t19_Relu_0[32, 28, 28, 64](dtype=float32)]
     *************/

    // create t19_Relu_0 tensor
    const SizeArray t19_Relu_0_sizes = {32, 28, 28, 64};
    const auto      t19_Relu_0       = createPersistentTensor(4, t19_Relu_0_sizes, syn_type_single, "t19_Relu_0");

    addNodeToGraph("relu_fwd_f32", {t16_BiasAdd_0}, {t19_Relu_0}, nullptr, 0, "n11_Relu");

    /*************
     * n12_pool1 node
     * inputs: [t19_Relu_0[32, 28, 28, 64](dtype=float32)]
     * output: [t20[32, 14, 14, 64](dtype=uint8), t21_pool1_0[32, 14, 14, 64](dtype=float32)]
     *************/

    // create t20 tensor
    const SizeArray t20_sizes = {32, 14, 14, 64};
    const auto      t20       = createPersistentTensor(4, t20_sizes, syn_type_uint8, "t20");

    // create t21_pool1_0 tensor
    const SizeArray t21_pool1_0_sizes = {32, 14, 14, 64};
    const auto      t21_pool1_0       = createPersistentTensor(4, t21_pool1_0_sizes, syn_type_single, "t21_pool1_0");

    unsigned char n12_pool1_params[] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                                        0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_fwd_f32", {t19_Relu_0}, {t20, t21_pool1_0}, (void*)n12_pool1_params, 44, "n12_pool1");

    /*************
     * n13_Conv2D_1 node
     * inputs: [t21_pool1_0[32, 14, 14, 64](dtype=float32), t12[64, 32, 5, 5](dtype=float32)]
     * output: [t22_Conv2D_1_0[64, 14, 14, 64](dtype=float32)]
     *************/

    // create t12 tensor
    const SizeArray t12_sizes = {64, 32, 5, 5};
    const auto      t12       = createPersistentTensor(4, t12_sizes, syn_type_single, "t12");

    // create t22_Conv2D_1_0 tensor
    const SizeArray t22_Conv2D_1_0_sizes = {64, 14, 14, 64};
    const auto      t22_Conv2D_1_0       = createTensor(4, t22_Conv2D_1_0_sizes, syn_type_single, "t22_Conv2D_1_0");

    unsigned char n13_Conv2D_1_params[] = {5, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0,  0,  2, 0, 0, 0,
                                           2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 17, 80, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0};
    addNodeToGraph("spatial_convolution",
                   {t21_pool1_0, t12},
                   {t22_Conv2D_1_0},
                   (void*)n13_Conv2D_1_params,
                   sizeof(n13_Conv2D_1_params),
                   "n13_Conv2D_1");

    /*************
     * n14_BiasAdd_1 node
     * inputs: [t8[64](dtype=float32), t25[64, 1, 1, 1](dtype=uint32)]
     * output: [t24[64, 1, 1, 1](dtype=float32)]
     *************/

    // create t8 tensor
    const SizeArray t8_sizes = {64};
    const auto      t8       = createPersistentTensor(1, t8_sizes, syn_type_single, "t8");

    // create t25 tensor
    const SizeArray t25_sizes = {64, 1, 1, 1};
    const auto      t25       = createShapeTensor(4, t25_sizes, syn_type_uint32, "t25");

    // create t24 tensor
    const SizeArray t24_sizes = {64, 1, 1, 1};
    const auto      t24       = createTensor(4, t24_sizes, syn_type_single, "t24");

    addNodeToGraph("reshape", {t8, t25}, {t24}, nullptr, 0, "n14_BiasAdd_1");

    /*************
     * n15_BiasAdd_1 node
     * inputs: [t22_Conv2D_1_0[64, 14, 14, 64](dtype=float32), t24[64, 1, 1, 1](dtype=float32)]
     * output: [t23_BiasAdd_1_0[64, 14, 14, 64](dtype=float32)]
     *************/

    // create t23_BiasAdd_1_0 tensor
    const SizeArray t23_BiasAdd_1_0_sizes = {64, 14, 14, 64};
    const auto      t23_BiasAdd_1_0       = createTensor(4, t23_BiasAdd_1_0_sizes, syn_type_single, "t23_BiasAdd_1_0");

    addNodeToGraph("add_fwd_f32", {t22_Conv2D_1_0, t24}, {t23_BiasAdd_1_0}, nullptr, 0, "n15_BiasAdd_1");

    /*************
     * n16_Relu_1 node
     * inputs: [t23_BiasAdd_1_0[64, 14, 14, 64](dtype=float32)]
     * output: [t26_Relu_1_0[64, 14, 14, 64](dtype=float32)]
     *************/

    // create t26_Relu_1_0 tensor
    const SizeArray t26_Relu_1_0_sizes = {64, 14, 14, 64};
    const auto      t26_Relu_1_0       = createPersistentTensor(4, t26_Relu_1_0_sizes, syn_type_single, "t26_Relu_1_0");

    addNodeToGraph("relu_fwd_f32", {t23_BiasAdd_1_0}, {t26_Relu_1_0}, nullptr, 0, "n16_Relu_1");

    /*************
     * n17_MaxPool2d node
     * inputs: [t26_Relu_1_0[64, 14, 14, 64](dtype=float32)]
     * output: [t27[64, 7, 7, 64](dtype=uint8), t28_MaxPool2d_0[64, 7, 7, 64](dtype=float32)]
     *************/

    // create t27 tensor
    const SizeArray t27_sizes = {64, 7, 7, 64};
    const auto      t27       = createPersistentTensor(4, t27_sizes, syn_type_uint8, "t27");

    // create t28_MaxPool2d_0 tensor
    const SizeArray t28_MaxPool2d_0_sizes = {64, 7, 7, 64};
    const auto t28_MaxPool2d_0 = createPersistentTensor(4, t28_MaxPool2d_0_sizes, syn_type_single, "t28_MaxPool2d_0");

    unsigned char n17_MaxPool2d_params[] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                                            0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    addNodeToGraph("maxpool_2d_fwd_f32",
                   {t26_Relu_1_0},
                   {t27, t28_MaxPool2d_0},
                   (void*)n17_MaxPool2d_params,
                   44,
                   "n17_MaxPool2d");

    /*************
     * n18_Reshape node
     * inputs: [t28_MaxPool2d_0[64, 7, 7, 64](dtype=float32), t30[3136, 64](dtype=uint32)]
     * output: [t29_Reshape_0[3136, 64](dtype=float32)]
     *************/

    // create t30 tensor
    const SizeArray t30_sizes = {3136, 64};
    const auto      t30       = createShapeTensor(2, t30_sizes, syn_type_uint32, "t30");

    // create t29_Reshape_0 tensor
    const SizeArray t29_Reshape_0_sizes = {3136, 64};
    const auto      t29_Reshape_0 = createPersistentTensor(2, t29_Reshape_0_sizes, syn_type_single, "t29_Reshape_0");

    addNodeToGraph("reshape", {t28_MaxPool2d_0, t30}, {t29_Reshape_0}, nullptr, 0, "n18_Reshape");

    /*************
     * n19_MatMul node
     * inputs: [t29_Reshape_0[3136, 64](dtype=float32), t9[1024, 3136](dtype=float32)]
     * output: [t31_MatMul_0[1024, 64](dtype=float32)]
     *************/

    // create t9 tensor
    const SizeArray t9_sizes = {1024, 3136};
    const auto      t9       = createPersistentTensor(2, t9_sizes, syn_type_single, "t9");

    // create t31_MatMul_0 tensor
    const SizeArray t31_MatMul_0_sizes = {1024, 64};
    const auto      t31_MatMul_0       = createPersistentTensor(2, t31_MatMul_0_sizes, syn_type_single, "t31_MatMul_0");

    unsigned char n19_MatMul_params[] = {0, 0};
    addNodeToGraph("gemm", {t29_Reshape_0, t9}, {t31_MatMul_0}, (void*)n19_MatMul_params, 2, "n19_MatMul");

    /*************
     * n20_Add node
     * inputs: [t10[1024](dtype=float32), t34[1024, 1](dtype=uint32)]
     * output: [t33[1024, 1](dtype=float32)]
     *************/

    // create t10 tensor
    const SizeArray t10_sizes = {1024};
    const auto      t10       = createPersistentTensor(1, t10_sizes, syn_type_single, "t10");

    // create t34 tensor
    const SizeArray t34_sizes = {1024, 1};
    const auto      t34       = createShapeTensor(2, t34_sizes, syn_type_uint32, "t34");

    // create t33 tensor
    const SizeArray t33_sizes = {1024, 1};
    const auto      t33       = createTensor(2, t33_sizes, syn_type_single, "t33");

    addNodeToGraph("reshape", {t10, t34}, {t33}, nullptr, 0, "n20_Add");

    /*************
     * n21_Add node
     * inputs: [t33[1024, 1](dtype=float32), t31_MatMul_0[1024, 64](dtype=float32)]
     * output: [t32_Add_0[1024, 64](dtype=float32)]
     *************/

    // create t32_Add_0 tensor
    const SizeArray t32_Add_0_sizes = {1024, 64};
    const auto      t32_Add_0       = createTensor(2, t32_Add_0_sizes, syn_type_single, "t32_Add_0");

    addNodeToGraph("add_fwd_f32", {t33, t31_MatMul_0}, {t32_Add_0}, nullptr, 0, "n21_Add");

    /*************
     * n22_Relu_2 node
     * inputs: [t32_Add_0[1024, 64](dtype=float32)]
     * output: [t35_Relu_2_0[1024, 64](dtype=float32)]
     *************/

    // create t35_Relu_2_0 tensor
    const SizeArray t35_Relu_2_0_sizes = {1024, 64};
    const auto      t35_Relu_2_0       = createPersistentTensor(2, t35_Relu_2_0_sizes, syn_type_single, "t35_Relu_2_0");

    addNodeToGraph("relu_fwd_f32", {t32_Add_0}, {t35_Relu_2_0}, nullptr, 0, "n22_Relu_2");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, onehot_gemm)
{
    /* MLPerf MaskRCNN case one_hot followed by gemm. OneHot is not sliceable */
    constexpr unsigned m = 1100, k = 1100, n = 1;
    const SizeArray    aSize      = {k, m};
    const SizeArray    bSize      = {n, k};
    const SizeArray    oSize      = {n, m};
    const SizeArray    onehotSize = {m};

    constexpr unsigned oneDim = 1, twoDims = 2;

    const auto onehotIn  = createPersistentTensor(oneDim, onehotSize, syn_type_int32);
    const auto onehotOut = createPersistentTensor(twoDims, aSize, syn_type_float);

    ns_OneHotKernel::Params params = {-1, k, 1, 0};  // axis, depth, on val, off val
    addNodeToGraph("one_hot_fwd_f32", {onehotIn}, {onehotOut}, (void*)&params, sizeof(params));

    const auto    a = onehotOut;
    const auto    b = createPersistentTensor(twoDims, bSize, syn_type_float);
    const auto    o = createTensor(twoDims, oSize, syn_type_float);
    synGEMMParams gemmParams {};
    addNodeToGraph(NodeFactory::gemmNodeTypeName, {a, b}, {o}, &gemmParams, sizeof gemmParams, "GEMM");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, slice_random_uniform)
{
    // slice->random_uniform (scalar-pipe)

    const SizeArray    sliceSize     = {1};
    const SizeArray    randomSize    = {1001, 2048};
    const SizeArray    randomMinSize = {1001, 2048};
    constexpr unsigned oneDim = 1, twoDims = 2;

    const auto sliceShape = createShapeTensor(oneDim, sliceSize, syn_type_uint32, "sliceShape", sliceSize.data());

    const auto sliceIn = createPersistentTensor(oneDim, sliceSize, syn_type_int32, "sliceIn");

    const auto sliceOut = createTensor(oneDim, sliceSize, syn_type_int32);

    synSliceParams sliceParams {};
    sliceParams.ends[0]  = 1;
    sliceParams.steps[0] = 1;
    addNodeToGraph("slice", {sliceIn, sliceShape}, {sliceOut}, &sliceParams, sizeof sliceParams, "slice");

    const auto randomShape =
        createShapeTensor(twoDims, randomSize, syn_type_uint32, "randomShape", randomMinSize.data());

    const auto randomOut =
        createPersistentTensor(twoDims, randomSize, syn_type_float, "randomOut", randomMinSize.data());

    ns_RandomUniform::Params randomParams {};
    randomParams.high = 100;
    randomParams.low  = 0;
    addNodeToGraph("random_uniform_fwd_f32",
                   {sliceOut, randomShape},
                   {randomOut},
                   &randomParams,
                   sizeof randomParams,
                   "random");

    compile();
}

TEST_P(GaudiTFSramSlicingTest, slice_flat_dedw)
{
    std::vector<TSize> dySize = {512, 24999, 1, 1};
    std::vector<TSize> xSize  = {512, 49999, 1, 1};
    std::vector<TSize> dwSize = {512, 512, 3, 1};

    auto dy = createPersistentTensor(dySize, syn_type_float, "dy");
    auto x  = createPersistentTensor(xSize, syn_type_float, "x");
    auto dw = createPersistentTensor(dwSize, syn_type_float, "dw");

    synConvolutionParams params {};
    params.kW = 3;
    params.dW = 2;

    addNodeToGraph(NodeFactory::deDwNodeTypeName, {dy, x}, {dw}, &params, sizeof(params), "dedw");
    compile();
}

// Test param:
// Device type.
INSTANTIATE_TEST_SUITE_P(, GaudiTFSramSlicingTest, ::testing::Values(synDeviceGaudi, synDeviceGaudi2));