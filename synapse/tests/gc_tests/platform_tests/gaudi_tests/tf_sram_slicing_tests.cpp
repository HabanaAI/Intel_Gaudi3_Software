/*
 * Tests from TF compilation failures
 */

#include "graph_compiler/habana_nodes/node_factory.h"
#include "gc_gaudi_test_infra.h"

class SynGaudiTFSramSlicingTest : public SynGaudiTestInfra
{
public:
    SynGaudiTFSramSlicingTest() { setTestPackage(TEST_PACKAGE_SRAM_SLICING); }
};

TEST_F_GC(SynGaudiTFSramSlicingTest, conv_case_add_fused)
{
    // Conv[1x1] -> tpc_memcpy -> Add_broadcasting
    // The TPC kernels get fused.
    // In this case if the conv is flattened and the fused kernel is stitched. Wrong results were observed.

    unsigned fmSize[]  = {1024, 1, 1, 1024};
    unsigned wghSize[] = {1024, 1024, 1, 1};
    unsigned bxSize[]  = {1024, 1, 1, 1};

    auto x = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 fmSize,
                                 ARRAY_SIZE(fmSize),
                                 syn_type_float);
    auto w = createPersistTensor(INPUT_TENSOR,
                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                 nullptr,
                                 wghSize,
                                 ARRAY_SIZE(wghSize),
                                 syn_type_float);

    auto y = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, fmSize, ARRAY_SIZE(fmSize), syn_type_float);

    auto yCopy = createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, fmSize, ARRAY_SIZE(fmSize), syn_type_float);

    auto bx = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bxSize, ARRAY_SIZE(bxSize), syn_type_float);

    auto out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, fmSize, ARRAY_SIZE(fmSize), syn_type_float);

    synConvolutionParams convParams {};
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {x, w}, {y}, &convParams, sizeof convParams);
    addNodeToGraph("memcpy_f32", {y}, {yCopy});
    addNodeToGraph("add_fwd_f32", {yCopy, bx}, {out});

    compileTopology();
    runTopology(0, true);

    CoordArray outIdx;
    float      expResult;
    TSize sizes[sizeof(fmSize)/sizeof(unsigned)];
    castNcopy(sizes, fmSize, sizeof(fmSize)/sizeof(unsigned));
    ASSERT_TRUE(checkMmeOp(m_tensorDescs[x],
                           reinterpret_cast<char*>(m_hostBuffers[x]),
                           m_tensorDescs[w],
                           reinterpret_cast<char*>(m_hostBuffers[w]),
                           m_tensorDescs[out],
                           reinterpret_cast<char*>(m_hostBuffers[out]),
                           convParams,
                           REFERENCE_OP_FWD,
                           outIdx,
                           m_deviceType,
                           &expResult))
        << "Wrong index: " << toString(outIdx, ',') << ": " << expResult
        << " actual: " << getIndexValue(sizes, outIdx, syn_type_float, m_hostBuffers[out]);
}

TEST_F_GC(SynGaudiTFSramSlicingTest, tpc_padded_conv)
{
    // pad_fwd_f32 -> conv

    TestSizes fmSize   = {1, 64, 64, 1, 1};
    TestSizes wghSize  = {1, 1, 7, 7, 1};
    unsigned padData[] = {0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned padSize[] = {10};

    TestSizes paddedFmSize;
    for (int idx = 0; idx < fmSize.size(); idx++)
    {
        paddedFmSize[idx] = fmSize[idx] + padData[idx] + padData[idx + MAX_DIMENSIONS_NUM];
    }

    auto padH2D = createHost2DeviceTensor(INPUT_TENSOR, padSize, padData, 1, "pad_data");

    auto unpaddedIFM = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           fmSize.data(),
                                           fmSize.size(),
                                           syn_type_float,
                                           nullptr,
                                           "input");

    auto paddedIFM =
        createTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, paddedFmSize.data(), paddedFmSize.size(), syn_type_float);

    auto wgh = createPersistTensor(INPUT_TENSOR,
                                   MEM_INIT_RANDOM_WITH_NEGATIVE,
                                   nullptr,
                                   wghSize.data(),
                                   wghSize.size(),
                                   syn_type_float,
                                   nullptr,
                                   "wgh");

    auto ofm = createPersistTensor(OUTPUT_TENSOR,
                                   MEM_INIT_ALL_ZERO,
                                   nullptr,
                                   fmSize.data(),
                                   fmSize.size(),
                                   syn_type_float,
                                   nullptr,
                                   "output");

    ns_PadKernelEx::Params padParams {};
    padParams.mode    = PAD_MODE_CONSTANT;
    padParams.value.f = 0.f;
    addNodeToGraph("pad_fwd_f32", {unpaddedIFM, padH2D}, {paddedIFM}, &padParams, sizeof padParams, "PAD");

    synConvolutionParams convParams {};
    convParams.kH = wghSize[WEIGHT_DIM_R];
    convParams.kW = wghSize[WEIGHT_DIM_S];
    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {paddedIFM, wgh},
                   {ofm},
                   &convParams,
                   sizeof convParams,
                   "CONV");

    compileAndRun();

    synConvolution3DParams conv3dParams {};
    conv3dParams.kernel[CONV_KERNEL_HEIGHT] = wghSize[WEIGHT_DIM_R];
    conv3dParams.kernel[CONV_KERNEL_WIDTH]  = wghSize[WEIGHT_DIM_S];
    conv3dParams.padding[CONV_PAD_LEFT]     = padData[1];
    conv3dParams.padding[CONV_PAD_RIGHT]    = padData[1 + MAX_DIMENSIONS_NUM];
    conv3dParams.padding[CONV_PAD_TOP]      = padData[2];
    conv3dParams.padding[CONV_PAD_BOTTOM]   = padData[2 + MAX_DIMENSIONS_NUM];
    CoordArray outIdx;
    ASSERT_TRUE(checkFwdConvolution(m_tensorDescs[unpaddedIFM],
                                    reinterpret_cast<char*>(m_hostBuffers[unpaddedIFM]),
                                    m_tensorDescs[wgh],
                                    reinterpret_cast<char*>(m_hostBuffers[wgh]),
                                    m_tensorDescs[ofm],
                                    reinterpret_cast<char*>(m_hostBuffers[ofm]),
                                    conv3dParams,
                                    outIdx,
                                    m_deviceType))
        << "Wrong value in index [" << toString(outIdx, ',') << ']';
}