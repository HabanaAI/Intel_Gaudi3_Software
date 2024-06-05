#include "gc_gaudi_test_infra.h"
#include "sram_slicing_test.h"
#include "syn_gaudi_two_run_compare_test.h"

SynGaudiSpatialSRAMSlicingTest::SynGaudiSpatialSRAMSlicingTest() :
    m_batch(std::get<0>(GetParam())),
    m_xHeight(std::get<1>(GetParam())),
    m_xWidth(std::get<2>(GetParam())),
    m_nIFM(std::get<3>(GetParam())),
    m_nOFM(std::get<4>(GetParam())),
    m_filter(std::get<5>(GetParam())),
    m_stride(std::get<6>(GetParam())),
    m_dilation(std::get<7>(GetParam())),
    m_padBefore(std::get<8>(GetParam())),
    m_padAfter(std::get<9>(GetParam())),
    m_params(m_filter, m_filter, m_stride, m_stride, m_padBefore, m_padAfter, m_padBefore, m_padAfter, m_dilation, m_dilation)
{
    m_yHeight.min = convOutputDimSize(m_xHeight.min, m_params.kH, m_params.dH, m_params.padT + m_params.padB, m_params.dilH);
    m_yHeight.max= convOutputDimSize(m_xHeight.max, m_params.kH, m_params.dH, m_params.padT + m_params.padB, m_params.dilH);
    m_yHeight.actual = convOutputDimSize(m_xHeight.actual, m_params.kH, m_params.dH, m_params.padT + m_params.padB, m_params.dilH);

    m_yWidth.min = convOutputDimSize(m_xWidth.min, m_params.kW, m_params.dW, m_params.padL + m_params.padR, m_params.dilW);
    m_yWidth.max = convOutputDimSize(m_xWidth.max, m_params.kW, m_params.dW, m_params.padL + m_params.padR, m_params.dilW);
    m_yWidth.actual = convOutputDimSize(m_xWidth.actual, m_params.kW, m_params.dW, m_params.padL + m_params.padR, m_params.dilW);

    m_xSizes.min = {m_nIFM, m_xWidth.min, m_xHeight.min, m_batch.min};
    m_xSizes.max = {m_nIFM, m_xWidth.max, m_xHeight.max, m_batch.max};
    m_xSizes.actual = {m_nIFM, m_xWidth.actual, m_xHeight.actual, m_batch.actual};
    m_wSizes.actual = {m_nOFM, m_nIFM, m_filter, m_filter};
    m_ySizes.min = {m_nOFM, m_yWidth.min, m_yHeight.min, m_batch.min};
    m_ySizes.max = {m_nOFM, m_yWidth.max, m_yHeight.max, m_batch.max};
    m_ySizes.actual = {m_nOFM, m_yWidth.actual, m_yHeight.actual, m_batch.actual};
}

void SynGaudiSpatialSRAMSlicingTest::runSingleTest()
{
    if (blockTestForConvParams())
    {
        //std::cout << "SKIP TEST with conv params:" << std::endl;
        //std::cout << "filter = " << m_filter << "; stride = " << m_stride << "; dilation = " << m_dilation <<
        //             "; padBefore = " << m_padBefore << "; padAfter = " << m_padAfter << ";" << std::endl;
        return;
    }

    auto outputTensorIdx = addNode(m_params, m_xSizes, m_ySizes, m_wSizes);

    setActualSizes();

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED", "true");
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED", "true");
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED", "true");
    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_DYNAMIC_4D_CONV_SPATIAL_SLICE_ENABLED", "true");

    // Disable conv flattening to enable spatial slicing for cases that are convertible to GEMM.
    addConfigurationToRun(FIRST_RUN, "ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "false");

    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    compareRunsResults({outputTensorIdx});
}

bool SynGaudiSpatialSRAMSlicingTest::blockTestForConvParams()
{
    // block filter size 1 with dilation > 2.
    // Dilation = 2 is enough to test non-single dilation with filter size 1. the rest are redundant.
    return ( ((m_filter == 1) && (m_dilation > 2)) ||
             (m_padAfter >= (m_filter + (m_filter - 1) * (m_dilation - 1))) );
}

unsigned SynGaudiFwdConvSpatialSRAMSlicingTest::addNode(synConvolutionParams& params,
                                                        ShapeSizes&           xSizes,
                                                        ShapeSizes&           ySizes,
                                                        ShapeSizes&           wSizes)
{
    m_tensorXIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes.max.data(),
                                       xSizes.max.size(), syn_type_float, nullptr, "X", 0, 0, nullptr, xSizes.min.data());
    m_tensorWIdx  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes.actual.data(),
                                        wSizes.actual.size(), syn_type_float, nullptr, "W");
    m_tensorYIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes.max.data(),
                                        ySizes.max.size(), syn_type_float, nullptr, "Y", 0, 0, nullptr, ySizes.min.data());

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   {m_tensorXIdx, m_tensorWIdx},
                   {m_tensorYIdx},
                   &params,
                   sizeof(params),
                   "FwdConvolution");

    return m_tensorYIdx;
}

TEST_P_GC(SynGaudiFwdConvSpatialSRAMSlicingTest, big_image_fwd_conv)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_image_fwd_conv_single_ASIC_CI,
    SynGaudiFwdConvSpatialSRAMSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(128), 128, 8, 3, 2, 1, 0, 0),
                      std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(128), 128, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(64), 256, 4, 4, 3, 2, 2, 0),
                      std::make_tuple(DimSizes(2), DimSizes(64), DimSizes(64), 256, 8, 3, 2, 2, 2, 3),
                      // Flattenable conv
                      std::make_tuple(DimSizes(2), DimSizes(512), DimSizes(512), 32, 32, 1, 1, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(big_image_fwd_conv_full_DAILY,
                        SynGaudiFwdConvSpatialSRAMSlicingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({DimSizes(2)}), //batch
                            ::testing::ValuesIn({DimSizes(256)}), //height
                            ::testing::ValuesIn({DimSizes(64)}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({8}), //channels out
                            ::testing::Range(1, 5), //filter
                            ::testing::Range(1, 4), //stride
                            ::testing::Range(1, 3), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4)  //padAfter
                        ));

INSTANTIATE_TEST_SUITE_P(big_image_fwd_conv_full_large_dilation_DAILY,
                        SynGaudiFwdConvSpatialSRAMSlicingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({DimSizes(2)}), //batch
                            ::testing::ValuesIn({DimSizes(256)}), //height
                            ::testing::ValuesIn({DimSizes(32)}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({8}), //channels out
                            ::testing::Range(2, 5), //filter
                            ::testing::Range(1, 4), //stride
                            ::testing::Range(3, 5), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4)  //padAfter
                        ));

INSTANTIATE_TEST_SUITE_P(big_image_fwd_conv_2_spatial_dims_full_DAILY,
                         SynGaudiFwdConvSpatialSRAMSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({DimSizes(2)}),    // batch
                                            ::testing::ValuesIn({DimSizes(16)}),   // height
                                            ::testing::ValuesIn({DimSizes(512)}),  // width
                                            ::testing::ValuesIn({512}),            // channels in
                                            ::testing::ValuesIn({8}),              // channels out
                                            ::testing::Range(2, 5),                // filter
                                            ::testing::Range(1, 4),                // stride
                                            ::testing::Range(1, 3),                // dilation
                                            ::testing::Range(0, 3),                // padBefore
                                            ::testing::Range(0, 3)                 // padAfter
                                            ));

unsigned SynGaudiDedwSpatialSRAMSlicingTest::addNode(synConvolutionParams& params,
                                                     ShapeSizes&           xSizes,
                                                     ShapeSizes&           ySizes,
                                                     ShapeSizes&           wSizes)
{
    m_tensorYIdx  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, ySizes.max.data(),
                                        ySizes.max.size(), syn_type_float, nullptr, "dY", 0, 0, nullptr, ySizes.min.data());
    m_tensorXIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes.max.data(),
                                       xSizes.max.size(), syn_type_float, nullptr, "X", 0, 0, nullptr, xSizes.min.data());
    m_tensorWIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wSizes.actual.data(),
                                        wSizes.actual.size(), syn_type_float, nullptr, "dW");

    addNodeToGraph(NodeFactory::deDwNodeTypeName, {m_tensorYIdx, m_tensorXIdx}, {m_tensorWIdx}, &params, sizeof(params), "dEdW");

    return m_tensorWIdx;
}

TEST_P_GC(SynGaudiDedwSpatialSRAMSlicingTest, big_image_dedw)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_image_dedw_single_ASIC,
    SynGaudiDedwSpatialSRAMSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(2), DimSizes(5), DimSizes(256), 2000, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(6), DimSizes(12), DimSizes(12), 512, 1024, 3, 1, 1, 1, 1),
                      std::make_tuple(DimSizes(2), DimSizes(45), DimSizes(60), 64, 128, 3, 1, 12, 12, 12),
                      // b,h,w,non-common are sliced
                      std::make_tuple(DimSizes(2), DimSizes(5), DimSizes(4800), 384, 80, 3, 1, 1, 1, 1),
                      // b,w,non-common are sliced
                      std::make_tuple(DimSizes(2), DimSizes(3), DimSizes(3400), 1024, 80, 3, 1, 1, 1, 1)));

INSTANTIATE_TEST_SUITE_P(
    big_image_dedw_single_ASIC_CI,
    SynGaudiDedwSpatialSRAMSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(256), 256, 8, 3, 2, 1, 0, 0),
                      std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(256), 256, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(256), 256, 8, 4, 3, 2, 2, 0),
                      std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(64), 256, 8, 3, 2, 2, 2, 3),
                      std::make_tuple(DimSizes(2), DimSizes(45), DimSizes(60), 64, 128, 3, 1, 12, 12, 12),
                      // Flattenable dedw
                      std::make_tuple(DimSizes(16), DimSizes(512), DimSizes(512), 32, 1, 1, 1, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(big_image_dedw_full_DAILY,
                        SynGaudiDedwSpatialSRAMSlicingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({DimSizes(2)}), //batch
                            ::testing::ValuesIn({DimSizes(256)}), //height
                            ::testing::ValuesIn({DimSizes(256)}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({8}), //channels out
                            ::testing::Range(1, 5), //filter
                            ::testing::Range(1, 4), //stride
                            ::testing::Range(1, 5), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4)  //padAfter
                        )
                );

INSTANTIATE_TEST_SUITE_P(big_image_narrow_dedw_full_DAILY,
                         SynGaudiDedwSpatialSRAMSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({DimSizes(2)}),    // batch
                                            ::testing::ValuesIn({DimSizes(39)}),   // height
                                            ::testing::ValuesIn({DimSizes(256)}),  // width
                                            ::testing::ValuesIn({2048}),           // channels in
                                            ::testing::ValuesIn({8}),              // channels out
                                            ::testing::Range(1, 5, 2),             // filter
                                            ::testing::ValuesIn({1, 10}),          // stride
                                            ::testing::ValuesIn({1, 13}),          // dilation
                                            ::testing::ValuesIn({1}),              // padBefore
                                            ::testing::ValuesIn({1, 7})            // padAfter
                                            ));

INSTANTIATE_TEST_SUITE_P(big_image_dedw_2_spatial_dims_full_DAILY,
                         SynGaudiDedwSpatialSRAMSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({DimSizes(2)}),      // batch
                                            ::testing::ValuesIn({DimSizes(8)}),      // height
                                            ::testing::ValuesIn({DimSizes(16384)}),  // width
                                            ::testing::ValuesIn({512}),              // channels in
                                            ::testing::ValuesIn({8}),                // channels out
                                            ::testing::Range(2, 5),                  // filter
                                            ::testing::Range(1, 4),                  // stride
                                            ::testing::Range(1, 3),                  // dilation
                                            ::testing::Range(0, 3),                  // padBefore
                                            ::testing::Range(0, 3)                   // padAfter
                                            ));

bool SynGaudiDedxSpatialSRAMSlicingTest::blockTestForConvParams()
{
    // block also dedx with stride and dilation which have common divisor, due to MME stack bug - SW-23339
    return ((gcd(m_stride, m_dilation) != 1) || SynGaudiSpatialSRAMSlicingTest::blockTestForConvParams());
}

unsigned SynGaudiDedxSpatialSRAMSlicingTest::addNode(synConvolutionParams& params,
                                                     ShapeSizes&           xSizes,
                                                     ShapeSizes&           ySizes,
                                                     ShapeSizes&           wSizes)
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, ySizes.actual.data(),
                                       ySizes.actual.size(), syn_type_float, nullptr, "dY");
    m_tensorWIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes.actual.data(),
                                       wSizes.actual.size(), syn_type_float, nullptr, "W");
    m_tensorXIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes.actual.data(),
                                       xSizes.actual.size(), syn_type_float, nullptr, "dX");
    addNodeToGraph(NodeFactory::deDxNodeTypeName, {m_tensorYIdx, m_tensorWIdx}, {m_tensorXIdx}, &params, sizeof(params), "dEdX");

    return m_tensorXIdx;
}

TEST_P_GC(SynGaudiDedxSpatialSRAMSlicingTest, big_image_dedx)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_image_dedx_single_ASIC_CI,
    SynGaudiDedxSpatialSRAMSlicingTest,
    ::testing::Values(std::make_tuple(DimSizes(2), DimSizes(256), DimSizes(256), 128, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(512), DimSizes(256), 64, 32, 3, 2, 1, 0, 0),
                      std::make_tuple(DimSizes(2), DimSizes(512), DimSizes(256), 16, 40, 4, 3, 2, 2, 0),
                      std::make_tuple(DimSizes(2), DimSizes(512), DimSizes(256), 32, 16, 3, 2, 3, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(512), DimSizes(256), 8, 128, 3, 3, 4, 2, 0),
                      // Flattenable dedx
                      std::make_tuple(DimSizes(2), DimSizes(512), DimSizes(512), 32, 32, 1, 1, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_full_DAILY,
                        SynGaudiDedxSpatialSRAMSlicingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({DimSizes(2)}), //batch
                            ::testing::ValuesIn({DimSizes(256)}), //height
                            ::testing::ValuesIn({DimSizes(256)}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({16}), //channels out
                            ::testing::Range(2, 5), //filter
                            ::testing::Range(1, 3), //stride
                            ::testing::Range(1, 5), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4)  //padAfter
                        ));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_full_filter_1_DAILY,
                        SynGaudiDedxSpatialSRAMSlicingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({DimSizes(2)}), //batch
                            ::testing::ValuesIn({DimSizes(256)}), //height
                            ::testing::ValuesIn({DimSizes(256)}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({32}), //channels out
                            ::testing::ValuesIn({1}), //filter
                            ::testing::Range(1, 4), //stride
                            ::testing::Range(1, 3), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 3)  //padAfter
                        ));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_full_large_stride_DAILY,
                        SynGaudiDedxSpatialSRAMSlicingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({DimSizes(2)}), //batch
                            ::testing::ValuesIn({DimSizes(512)}), //height
                            ::testing::ValuesIn({DimSizes(256)}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({40}), //channels out
                            ::testing::Range(2, 5), //filter
                            ::testing::Range(3, 4), //stride
                            ::testing::Range(1, 5), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4)  //padAfter
                        ));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_2_spatial_dims_full_DAILY,
                         SynGaudiDedxSpatialSRAMSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({DimSizes(2)}),    // batch
                                            ::testing::ValuesIn({DimSizes(16)}),   // height
                                            ::testing::ValuesIn({DimSizes(512)}),  // width
                                            ::testing::ValuesIn({16}),             // channels in
                                            ::testing::ValuesIn({512}),            // channels out
                                            ::testing::Range(2, 5),                // filter
                                            ::testing::Range(1, 3),                // stride
                                            ::testing::Range(1, 4),                // dilation
                                            ::testing::Range(0, 3),                // padBefore
                                            ::testing::Range(0, 3)                 // padAfter
                                            ));

TEST_F_GC(SynGaudiSRAMSlicingTest, sram_slicing_big_conv_with_producer_stitching_L2)
{
    // Tests big convolution with spatial slicing: each operand has a TPC producer.
    // The TPC producer should be stitched to w operand since it doens't have overlap.

    GlobalConfTestSetter conf_conv("SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED", "true");
    GlobalConfTestSetter conf_dedw("SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED", "true");
    GlobalConfTestSetter conf_dedx("SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED", "true");
    GlobalConfTestSetter conf_dynamic_conv("SRAM_SLICER_DYNAMIC_4D_CONV_SPATIAL_SLICE_ENABLED", "true");

    unsigned b = 3, inH = 128, inW = 128, c = 50, k = 8;
    unsigned kernel = 3;
    synConvolutionParams params(kernel, kernel, 2, 2, 2, 1, 2, 1, 2, 2);
    unsigned outH = convOutputDimSize(inH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned outW = convOutputDimSize(inW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    std::vector<unsigned> xSizes {c, inW, inH, b};
    std::vector<unsigned> wSizes {k, c, kernel, kernel};
    std::vector<unsigned> ySizes {k, outW, outH, b};

    unsigned tensorXProducerIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes.data(), xSizes.size(), syn_type_float);
    unsigned tensorWProducerIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes.data(), wSizes.size(), syn_type_float);
    unsigned tensorXIdx =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes.data(), xSizes.size(), syn_type_float);
    unsigned tensorWIdx =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wSizes.data(), wSizes.size(), syn_type_float);
    unsigned tensorYIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes.data(), ySizes.size(), syn_type_float);

    addNodeToGraph("relu_fwd_f32", {tensorXProducerIdx}, {tensorXIdx});
    addNodeToGraph("relu_fwd_f32", {tensorWProducerIdx}, {tensorWIdx});
    addNodeToGraph(NodeFactory::convolutionNodeTypeName, {tensorXIdx, tensorWIdx}, {tensorYIdx}, &params, sizeof(params), "FwdConvolution");
    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[tensorXIdx];
    synTensorDescriptor wDesc = m_tensorDescs[tensorWIdx];
    synTensorDescriptor yDesc = m_tensorDescs[tensorYIdx];

    auto xProducerData = m_hostBuffers[tensorXProducerIdx];
    auto wProducerData = m_hostBuffers[tensorWProducerIdx];
    auto outData = m_hostBuffers[tensorYIdx];

    uint64_t           reluXOutSize = getNumberOfElements(xDesc.m_sizes, xDesc.m_dims);
    std::vector<float> calcReluXOut(reluXOutSize);
    calculateRelu(xDesc, xProducerData, xDesc, calcReluXOut.data());

    uint64_t           reluWOutSize = getNumberOfElements(wDesc.m_sizes, wDesc.m_dims);
    std::vector<float> calcReluWOut(reluWOutSize);
    calculateRelu(wDesc, wProducerData, wDesc, calcReluWOut.data());

    CoordArray wrongIdx = {0};
    bool       ret      = checkFwdConvolution(xDesc,
                                   (char*)calcReluXOut.data(),
                                   wDesc,
                                   (char*)calcReluWOut.data(),
                                   yDesc,
                                   (char*)outData,
                                   params,
                                   wrongIdx,
                                   m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, yDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, yDesc.m_dataType, outData);
}

TEST_F_GC(SynGaudiSRAMSlicingTest, sram_slicing_big_dedw_with_producer_stitching_L2)
{
    // Tests big dedw with spatial slicing: each operand has a TPC producer.
    // The TPC producer should be stitched to dy operand since it doens't have overlap.

    GlobalConfTestSetter conf_conv("SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED", "true");
    GlobalConfTestSetter conf_dedw("SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED", "true");
    GlobalConfTestSetter conf_dedx("SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED", "true");

    unsigned b = 3, inH = 128, inW = 128, c = 50, k = 8;
    unsigned kernel = 3;
    synConvolutionParams params(kernel, kernel, 2, 2, 2, 1, 2, 1, 2, 2);
    unsigned outH = convOutputDimSize(inH, params.kH, params.dH, params.padT + params.padB, params.dilH);
    unsigned outW = convOutputDimSize(inW, params.kW, params.dW, params.padL + params.padR, params.dilW);
    std::vector<unsigned> xSizes {c, inW, inH, b};
    std::vector<unsigned> wSizes {k, c, kernel, kernel};
    std::vector<unsigned> ySizes {k, outW, outH, b};

    unsigned tensorDyProducerIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, ySizes.data(), ySizes.size(), syn_type_float);
    unsigned tensorXProducerIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes.data(), xSizes.size(), syn_type_float);
    unsigned tensorDyIdx =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ySizes.data(), ySizes.size(), syn_type_float);
    unsigned tensorXIdx =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes.data(), xSizes.size(), syn_type_float);
    unsigned tensorDwIdx  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wSizes.data(), wSizes.size(), syn_type_float);

    addNodeToGraph("relu_fwd_f32", {tensorDyProducerIdx}, {tensorDyIdx});
    addNodeToGraph("relu_fwd_f32", {tensorXProducerIdx}, {tensorXIdx});
    addNodeToGraph(NodeFactory::deDwNodeTypeName, {tensorDyIdx, tensorXIdx}, {tensorDwIdx}, &params, sizeof(params), "dEdW");

    compileAndRun();

    synTensorDescriptor dyDesc = m_tensorDescs[tensorDyIdx];
    synTensorDescriptor xDesc = m_tensorDescs[tensorXIdx];
    synTensorDescriptor dwDesc = m_tensorDescs[tensorDwIdx];

    auto dyProducerData = m_hostBuffers[tensorDyProducerIdx];
    auto xProducerData = m_hostBuffers[tensorXProducerIdx];
    auto outData = m_hostBuffers[tensorDwIdx];

    uint64_t           reluDyOutSize = getNumberOfElements(dyDesc.m_sizes, dyDesc.m_dims);
    std::vector<float> calcReluDyOut(reluDyOutSize);
    calculateRelu(dyDesc, dyProducerData, dyDesc, calcReluDyOut.data());

    uint64_t           reluXOutSize = getNumberOfElements(xDesc.m_sizes, xDesc.m_dims);
    std::vector<float> calcReluXOut(reluXOutSize);
    calculateRelu(xDesc, xProducerData, xDesc, calcReluXOut.data());

    CoordArray wrongIdx = {0};
    bool       ret      = checkDEDW(dyDesc,
                         (char*)calcReluDyOut.data(),
                         xDesc,
                         (char*)calcReluXOut.data(),
                         dwDesc,
                         (char*)outData,
                         params,
                         wrongIdx,
                         m_deviceType);

    TSize sizes[SYN_MAX_TENSOR_DIM];
    castNcopy(sizes, dwDesc.m_sizes, SYN_MAX_TENSOR_DIM);
    ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                     << " Got value: " << getIndexValue(sizes, wrongIdx, dwDesc.m_dataType, outData);
}

class SynGaudi3dConvSpatialSlicingTest
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<int, DimSizes, DimSizes, DimSizes, int, int, int, int, int, int, int>>
// batch, depth, height, width, nIFM, nOFM, filter, stride, dilation, padBefore, padAfter
{
public:
    SynGaudi3dConvSpatialSlicingTest();
    // returns the output tensor index to validate correctness
    virtual unsigned addNode() = 0;
    void             runSingleTest();
    virtual bool     blockTestForConvParams();

protected:
    unsigned m_batch;
    DimSizes m_xDepth;
    DimSizes m_xHeight;
    DimSizes m_xWidth;
    unsigned m_nIFM;
    unsigned m_nOFM;
    unsigned m_filter;
    unsigned m_stride;
    unsigned m_dilation;
    int      m_padBefore;
    int      m_padAfter;

    synConvolution3DParams m_params;

    DimSizes m_yDepth;
    DimSizes m_yHeight;
    DimSizes m_yWidth;

    ShapeSizes m_xSizes;
    ShapeSizes m_wSizes;
    ShapeSizes m_ySizes;

    unsigned m_tensorXIdx;
    unsigned m_tensorWIdx;
    unsigned m_tensorYIdx;
};

SynGaudi3dConvSpatialSlicingTest::SynGaudi3dConvSpatialSlicingTest()
: m_batch(std::get<0>(GetParam())),
  m_xDepth(std::get<1>(GetParam())),
  m_xHeight(std::get<2>(GetParam())),
  m_xWidth(std::get<3>(GetParam())),
  m_nIFM(std::get<4>(GetParam())),
  m_nOFM(std::get<5>(GetParam())),
  m_filter(std::get<6>(GetParam())),
  m_stride(std::get<7>(GetParam())),
  m_dilation(std::get<8>(GetParam())),
  m_padBefore(std::get<9>(GetParam())),
  m_padAfter(std::get<10>(GetParam())),
  m_params(m_filter,
           m_filter,
           m_filter,
           m_stride,
           m_stride,
           m_stride,
           m_padBefore,
           m_padAfter,
           m_padBefore,
           m_padAfter,
           m_padBefore,
           m_padAfter,
           m_dilation,
           m_dilation,
           m_dilation)
{
    m_yDepth.min    = convOutputDimSize(m_xDepth.min,
                                     m_params.kernel[CONV_KERNEL_DEPTH],
                                     m_params.stride[CONV_STRIDE_DEPTH],
                                     m_params.padding[CONV_PAD_FRONT] + m_params.padding[CONV_PAD_BACK],
                                     m_params.dilation[CONV_DIL_DEPTH]);
    m_yDepth.max    = convOutputDimSize(m_xDepth.max,
                                     m_params.kernel[CONV_KERNEL_DEPTH],
                                     m_params.stride[CONV_STRIDE_DEPTH],
                                     m_params.padding[CONV_PAD_FRONT] + m_params.padding[CONV_PAD_BACK],
                                     m_params.dilation[CONV_DIL_DEPTH]);
    m_yDepth.actual = convOutputDimSize(m_xDepth.actual,
                                        m_params.kernel[CONV_KERNEL_DEPTH],
                                        m_params.stride[CONV_STRIDE_DEPTH],
                                        m_params.padding[CONV_PAD_FRONT] + m_params.padding[CONV_PAD_BACK],
                                        m_params.dilation[CONV_DIL_DEPTH]);

    m_yHeight.min    = convOutputDimSize(m_xHeight.min,
                                      m_params.kernel[CONV_KERNEL_HEIGHT],
                                      m_params.stride[CONV_STRIDE_HEIGHT],
                                      m_params.padding[CONV_PAD_TOP] + m_params.padding[CONV_PAD_BOTTOM],
                                      m_params.dilation[CONV_DIL_HEIGHT]);
    m_yHeight.max    = convOutputDimSize(m_xHeight.max,
                                      m_params.kernel[CONV_KERNEL_HEIGHT],
                                      m_params.stride[CONV_STRIDE_HEIGHT],
                                      m_params.padding[CONV_PAD_TOP] + m_params.padding[CONV_PAD_BOTTOM],
                                      m_params.dilation[CONV_DIL_HEIGHT]);
    m_yHeight.actual = convOutputDimSize(m_xHeight.actual,
                                         m_params.kernel[CONV_KERNEL_HEIGHT],
                                         m_params.stride[CONV_STRIDE_HEIGHT],
                                         m_params.padding[CONV_PAD_TOP] + m_params.padding[CONV_PAD_BOTTOM],
                                         m_params.dilation[CONV_DIL_HEIGHT]);

    m_yWidth.min    = convOutputDimSize(m_xWidth.min,
                                     m_params.kernel[CONV_KERNEL_WIDTH],
                                     m_params.stride[CONV_STRIDE_WIDTH],
                                     m_params.padding[CONV_PAD_LEFT] + m_params.padding[CONV_PAD_RIGHT],
                                     m_params.dilation[CONV_DIL_WIDTH]);
    m_yWidth.max    = convOutputDimSize(m_xWidth.max,
                                     m_params.kernel[CONV_KERNEL_WIDTH],
                                     m_params.stride[CONV_STRIDE_WIDTH],
                                     m_params.padding[CONV_PAD_LEFT] + m_params.padding[CONV_PAD_RIGHT],
                                     m_params.dilation[CONV_DIL_WIDTH]);
    m_yWidth.actual = convOutputDimSize(m_xWidth.actual,
                                        m_params.kernel[CONV_KERNEL_WIDTH],
                                        m_params.stride[CONV_STRIDE_WIDTH],
                                        m_params.padding[CONV_PAD_LEFT] + m_params.padding[CONV_PAD_RIGHT],
                                        m_params.dilation[CONV_DIL_WIDTH]);

    m_xSizes.min    = {m_nIFM, m_xWidth.min, m_xHeight.min, m_xDepth.min, m_batch};
    m_xSizes.max    = {m_nIFM, m_xWidth.max, m_xHeight.max, m_xDepth.max, m_batch};
    m_xSizes.actual = {m_nIFM, m_xWidth.actual, m_xHeight.actual, m_xDepth.actual, m_batch};
    m_wSizes.actual = {m_nOFM, m_nIFM, m_filter, m_filter, m_filter};
    m_ySizes.min    = {m_nOFM, m_yWidth.min, m_yHeight.min, m_yDepth.min, m_batch};
    m_ySizes.max    = {m_nOFM, m_yWidth.max, m_yHeight.max, m_yDepth.max, m_batch};
    m_ySizes.actual = {m_nOFM, m_yWidth.actual, m_yHeight.actual, m_yDepth.actual, m_batch};
}

bool SynGaudi3dConvSpatialSlicingTest::blockTestForConvParams()
{
    // block filter size 1 with dilation > 2.
    // Dilation = 2 is enough to test non-single dilation with filter size 1. the rest are redundant.
    return (((m_filter == 1) && (m_dilation > 2)) || (m_padAfter >= (m_filter + (m_filter - 1) * (m_dilation - 1))));
}

void SynGaudi3dConvSpatialSlicingTest::runSingleTest()
{
    if (blockTestForConvParams())
    {
        return;
    }

    unsigned tensorToValidateIdx = addNode();

    addConfigurationToRun(FIRST_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "18874368");
    addConfigurationToRun(SECOND_RUN, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    addConfigurationToRun(SECOND_RUN, "ENABLE_LAYERED_PIPELINE_BRAIN", "false");

    addConfigurationToRun(FIRST_RUN, "ENABLE_PIPELINE_MANAGEMENT", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_PIPELINE_MANAGEMENT", "false");

    // Disable conv flattening to enable spatial slicing for cases that are convertible to GEMM.
    addConfigurationToRun(FIRST_RUN, "ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING", "false");

    compareRunsResults({tensorToValidateIdx});
}

class SynGaudiDevice3dConvFwdSpatialSlicingTest : public SynGaudi3dConvSpatialSlicingTest
{
public:
    SynGaudiDevice3dConvFwdSpatialSlicingTest() : SynGaudi3dConvSpatialSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynGaudiDevice3dConvFwdSpatialSlicingTest::addNode()
{
    unsigned graphIndex = 0;

    m_tensorXIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_xSizes.max.data(),
                                       m_xSizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "X",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_xSizes.min.data());

    m_tensorWIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "W");

    m_tensorYIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_ySizes.max.data(),
                                       m_ySizes.max.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "Y",
                                       graphIndex,
                                       0,
                                       nullptr,
                                       m_ySizes.min.data());

    addNodeToGraph(NodeFactory::convolution3DNodeTypeName,
                   {m_tensorXIdx, m_tensorWIdx},
                   {m_tensorYIdx},
                   &m_params,
                   sizeof(m_params),
                   "conv3d",
                   graphIndex);
    return m_tensorYIdx;
}

TEST_P_GC(SynGaudiDevice3dConvFwdSpatialSlicingTest, big_image_fwd_3dConv)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_image_fwd_3dConv_single_ASIC_CI,
    SynGaudiDevice3dConvFwdSpatialSlicingTest,
    ::testing::Values(std::make_tuple(2, DimSizes(128), DimSizes(8), DimSizes(64), 128, 2, 3, 1, 1, 0, 0),
                      std::make_tuple(2, DimSizes(128), DimSizes(8), DimSizes(64), 256, 3, 3, 2, 1, 0, 0),
                      std::make_tuple(2, DimSizes(128), DimSizes(8), DimSizes(64), 256, 4, 3, 2, 2, 1, 0),
                      std::make_tuple(2, DimSizes(64), DimSizes(16), DimSizes(64), 128, 2, 5, 1, 2, 2, 2),
                      std::make_tuple(2, DimSizes(64), DimSizes(32), DimSizes(32), 128, 3, 4, 3, 2, 3, 1),
                      std::make_tuple(2, DimSizes(80), DimSizes(6), DimSizes(60), 256, 4, 3, 2, 2, 1, 0),
                      // Flattenable conv
                      std::make_tuple(2, DimSizes(8), DimSizes(256), DimSizes(256), 64, 64, 1, 1, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(big_image_fwd_3dconv_full_DAILY,
                         SynGaudiDevice3dConvFwdSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),              // batch
                                            ::testing::ValuesIn({DimSizes(128)}),  // depth
                                            ::testing::ValuesIn({DimSizes(16)}),   // height
                                            ::testing::ValuesIn({DimSizes(32)}),   // width
                                            ::testing::ValuesIn({256}),            // channels in
                                            ::testing::ValuesIn({8}),              // channels out
                                            ::testing::Range(1, 5),                // filter
                                            ::testing::Range(1, 4),                // stride
                                            ::testing::Range(1, 4),                // dilation
                                            ::testing::Range(0, 3),                // padBefore
                                            ::testing::Range(0, 3)                 // padAfter
                                            ));

INSTANTIATE_TEST_SUITE_P(big_image_fwd_3dconv_2_spatial_dims_full_DAILY,
                         SynGaudiDevice3dConvFwdSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),              // batch
                                            ::testing::ValuesIn({DimSizes(16)}),   // depth
                                            ::testing::ValuesIn({DimSizes(64)}),   // height
                                            ::testing::ValuesIn({DimSizes(128)}),  // width
                                            ::testing::ValuesIn({256}),            // channels in
                                            ::testing::ValuesIn({8}),              // channels out
                                            ::testing::Range(2, 4),                // filter
                                            ::testing::Range(1, 4),                // stride
                                            ::testing::Range(1, 3),                // dilation
                                            ::testing::Range(0, 2),                // padBefore
                                            ::testing::Range(0, 2)                 // padAfter
                                            ));

class SynGaudiDevice3dConvDedxSpatialSlicingTest : public SynGaudi3dConvSpatialSlicingTest
{
public:
    SynGaudiDevice3dConvDedxSpatialSlicingTest() : SynGaudi3dConvSpatialSlicingTest() {}
    unsigned addNode() override;
    bool     blockTestForConvParams() override;
};

bool SynGaudiDevice3dConvDedxSpatialSlicingTest::blockTestForConvParams()
{
    // block also dedx with stride and dilation which have common divisor, due to MME stack bug - SW-23339
    return ((gcd(m_stride, m_dilation) != 1) || SynGaudi3dConvSpatialSlicingTest::blockTestForConvParams());
}

unsigned SynGaudiDevice3dConvDedxSpatialSlicingTest::addNode()
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_ySizes.actual.data(),
                                       m_ySizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "dY");

    m_tensorWIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "W");

    m_tensorXIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_xSizes.actual.data(),
                                       m_xSizes.actual.size(),
                                       syn_type_bf16,
                                       nullptr,
                                       "dX");

    addNodeToGraph(NodeFactory::deDx3DNodeTypeName,
                   {m_tensorYIdx, m_tensorWIdx},
                   {m_tensorXIdx},
                   &m_params,
                   sizeof(m_params),
                   "dedx3d");
    return m_tensorXIdx;
}

TEST_P_GC(SynGaudiDevice3dConvDedxSpatialSlicingTest, big_image_dedx_3dConv)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_image_dedx_3dconv_single_ASIC_CI,
    SynGaudiDevice3dConvDedxSpatialSlicingTest,
    ::testing::Values(std::make_tuple(2, DimSizes(64), DimSizes(64), DimSizes(32), 4, 100, 3, 1, 1, 0, 0),
                      std::make_tuple(2, DimSizes(64), DimSizes(64), DimSizes(200), 4, 128, 3, 2, 1, 0, 0),
                      std::make_tuple(2, DimSizes(128), DimSizes(128), DimSizes(128), 4, 128, 4, 3, 2, 2, 0),
                      std::make_tuple(2, DimSizes(128), DimSizes(100), DimSizes(32), 4, 256, 3, 2, 3, 0, 0),
                      std::make_tuple(2, DimSizes(128), DimSizes(128), DimSizes(64), 4, 256, 3, 3, 4, 2, 0),
                      // Flattenable dedx
                      std::make_tuple(2, DimSizes(8), DimSizes(256), DimSizes(256), 64, 64, 1, 1, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_3dconv_full_DAILY,
                         SynGaudiDevice3dConvDedxSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),             // batch
                                            ::testing::ValuesIn({DimSizes(64)}),  // depth
                                            ::testing::ValuesIn({DimSizes(64)}),  // height
                                            ::testing::ValuesIn({DimSizes(32)}),  // width
                                            ::testing::ValuesIn({16}),            // channels in
                                            ::testing::ValuesIn({256}),           // channels out
                                            ::testing::Range(1, 5),               // filter
                                            ::testing::ValuesIn({1}),             // stride
                                            ::testing::Range(1, 5),               // dilation
                                            ::testing::Range(0, 4),               // padBefore
                                            ::testing::Range(0, 3)                // padAfter
                                            ));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_3dconv_full_stride_2_DAILY,
                         SynGaudiDevice3dConvDedxSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),              // batch
                                            ::testing::ValuesIn({DimSizes(128)}),  // depth
                                            ::testing::ValuesIn({DimSizes(64)}),   // height
                                            ::testing::ValuesIn({DimSizes(32)}),   // width
                                            ::testing::ValuesIn({16}),             // channels in
                                            ::testing::ValuesIn({256}),            // channels out
                                            ::testing::Range(1, 5),                // filter
                                            ::testing::ValuesIn({2}),              // stride
                                            ::testing::ValuesIn({1, 3}),           // dilation
                                            ::testing::Range(0, 4),                // padBefore
                                            ::testing::Range(0, 4)                 // padAfter
                                            ));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_3dconv_full_stride_3_DAILY,
                         SynGaudiDevice3dConvDedxSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),              // batch
                                            ::testing::ValuesIn({DimSizes(128)}),  // depth
                                            ::testing::ValuesIn({DimSizes(64)}),   // height
                                            ::testing::ValuesIn({DimSizes(128)}),  // width
                                            ::testing::ValuesIn({40}),             // channels in
                                            ::testing::ValuesIn({256}),            // channels out
                                            ::testing::Range(1, 5),                // filter
                                            ::testing::ValuesIn({3}),              // stride
                                            ::testing::ValuesIn({1, 2, 4}),        // dilation
                                            ::testing::Range(0, 3),                // padBefore
                                            ::testing::Range(0, 3)                 // padAfter
                                            ));

INSTANTIATE_TEST_SUITE_P(big_image_dedx_3dconv_2_spatial_dims_full_DAILY,
                         SynGaudiDevice3dConvDedxSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),              // batch
                                            ::testing::ValuesIn({DimSizes(16)}),   // depth
                                            ::testing::ValuesIn({DimSizes(64)}),   // height
                                            ::testing::ValuesIn({DimSizes(128)}),  // width
                                            ::testing::ValuesIn({8}),              // channels in
                                            ::testing::ValuesIn({512}),            // channels out
                                            ::testing::Range(2, 5),                // filter
                                            ::testing::Range(1, 4),                // stride
                                            ::testing::Range(1, 4),                // dilation
                                            ::testing::Range(0, 2),                // padBefore
                                            ::testing::Range(0, 2)                 // padAfter
                                            ));

class SynGaudiDevice3dConvDedwSpatialSlicingTest : public SynGaudi3dConvSpatialSlicingTest
{
public:
    SynGaudiDevice3dConvDedwSpatialSlicingTest() : SynGaudi3dConvSpatialSlicingTest() {}
    unsigned addNode() override;
};

unsigned SynGaudiDevice3dConvDedwSpatialSlicingTest::addNode()
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_ySizes.max.data(),
                                       m_ySizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "dY",
                                       0,
                                       0,
                                       nullptr,
                                       m_ySizes.min.data());
    m_tensorXIdx = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_xSizes.max.data(),
                                       m_xSizes.max.size(),
                                       syn_type_float,
                                       nullptr,
                                       "X",
                                       0,
                                       0,
                                       nullptr,
                                       m_xSizes.min.data());
    m_tensorWIdx = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       m_wSizes.actual.data(),
                                       m_wSizes.actual.size(),
                                       syn_type_float,
                                       nullptr,
                                       "dW");

    addNodeToGraph(NodeFactory::deDw3DNodeTypeName,
                   {m_tensorYIdx, m_tensorXIdx},
                   {m_tensorWIdx},
                   &m_params,
                   sizeof(m_params),
                   "dedw3d");

    return m_tensorWIdx;
}

TEST_P_GC(SynGaudiDevice3dConvDedwSpatialSlicingTest, big_image_dedw_3dConv)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(
    big_image_dedw_3dconv_single_ASIC_CI,
    SynGaudiDevice3dConvDedwSpatialSlicingTest,
    ::testing::Values(std::make_tuple(2, DimSizes(64), DimSizes(32), DimSizes(32), 64, 8, 3, 2, 1, 0, 0),
                      std::make_tuple(2, DimSizes(64), DimSizes(32), DimSizes(32), 32, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(2, DimSizes(64), DimSizes(100), DimSizes(32), 40, 8, 4, 3, 2, 2, 0),
                      std::make_tuple(2, DimSizes(8), DimSizes(8), DimSizes(256), 320, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(2, DimSizes(128), DimSizes(16), DimSizes(32), 64, 8, 3, 2, 2, 2, 3),
                      std::make_tuple(2, DimSizes(12), DimSizes(12), DimSizes(12), 256, 400, 3, 1, 1, 1, 1),
                      // b,d,h,non-common are sliced
                      std::make_tuple(2, DimSizes(8), DimSizes(384), DimSizes(128), 256, 8, 2, 3, 1, 1, 1),
                      // Flattenable dedw
                      std::make_tuple(2, DimSizes(8), DimSizes(512), DimSizes(512), 32, 1, 1, 1, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(big_image_dedw_3dconv_full_DAILY,
                         SynGaudiDevice3dConvDedwSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),             // batch
                                            ::testing::ValuesIn({DimSizes(64)}),  // depth
                                            ::testing::ValuesIn({DimSizes(64)}),  // height
                                            ::testing::ValuesIn({DimSizes(32)}),  // width
                                            ::testing::ValuesIn({256}),           // channels in
                                            ::testing::ValuesIn({8}),             // channels out
                                            ::testing::Range(1, 5),               // filter
                                            ::testing::Range(1, 4),               // stride
                                            ::testing::Range(1, 4),               // dilation
                                            ::testing::Range(0, 4),               // padBefore
                                            ::testing::Range(0, 3)                // padAfter
                                            ));

INSTANTIATE_TEST_SUITE_P(big_image_dedw_3dconv_2_spatial_dims_full_DAILY,
                         SynGaudiDevice3dConvDedwSpatialSlicingTest,
                         ::testing::Combine(::testing::ValuesIn({2}),              // batch
                                            ::testing::ValuesIn({DimSizes(8)}),    // depth
                                            ::testing::ValuesIn({DimSizes(384)}),  // height
                                            ::testing::ValuesIn({DimSizes(128)}),  // width
                                            ::testing::ValuesIn({256}),            // channels in
                                            ::testing::ValuesIn({8}),              // channels out
                                            ::testing::Range(2, 4),                // filter
                                            ::testing::Range(1, 4),                // stride
                                            ::testing::Range(1, 3),                // dilation
                                            ::testing::Range(0, 2),                // padBefore
                                            ::testing::Range(0, 2)                 // padAfter
                                            ));

const auto unet3dConvSizes = ::testing::Values(
    // batch, depth, height, width, nIFM, nOFM, filter, stride, dilation, padBefore, padAfter
    std::make_tuple(2, DimSizes(128), DimSizes(128), DimSizes(128), 4, 32, 3, 1, 1, 0, 0),
    std::make_tuple(2, DimSizes(128), DimSizes(128), DimSizes(128), 32, 32, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(64), DimSizes(64), DimSizes(64), 128, 64, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(128), DimSizes(128), DimSizes(128), 64, 32, 3, 1, 1, 1, 1),
    std::make_tuple(3, DimSizes(4), DimSizes(512), DimSizes(512), 32, 16, 3, 1, 1, 1, 1));

const auto unet3dConvSanitySizes = ::testing::Values(
    // batch, depth, height, width, nIFM, nOFM, filter, stride, dilation, padBefore, padAfter
    std::make_tuple(2, DimSizes(128), DimSizes(128), DimSizes(128), 32, 64, 3, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(64), DimSizes(64), DimSizes(64), 64, 64, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(64), DimSizes(64), DimSizes(64), 64, 128, 3, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(32), DimSizes(32), DimSizes(32), 128, 128, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(32), DimSizes(32), DimSizes(32), 128, 256, 3, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(16), DimSizes(16), DimSizes(16), 256, 256, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(16), DimSizes(16), DimSizes(16), 256, 320, 3, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(8), DimSizes(8), DimSizes(8), 320, 320, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(8), DimSizes(8), DimSizes(8), 320, 320, 3, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(4), DimSizes(4), DimSizes(4), 320, 320, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(4), DimSizes(4), DimSizes(4), 320, 320, 2, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(8), DimSizes(8), DimSizes(8), 640, 320, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(8), DimSizes(8), DimSizes(8), 320, 256, 2, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(16), DimSizes(16), DimSizes(16), 512, 256, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(16), DimSizes(16), DimSizes(16), 256, 128, 2, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(32), DimSizes(32), DimSizes(32), 256, 128, 3, 1, 1, 1, 1),
    std::make_tuple(2, DimSizes(32), DimSizes(32), DimSizes(32), 128, 64, 2, 2, 1, 0, 0),
    std::make_tuple(2, DimSizes(64), DimSizes(64), DimSizes(64), 64, 32, 2, 2, 1, 0, 0),
    std::make_tuple(3, DimSizes(4), DimSizes(512), DimSizes(512), 16, 1, 3, 1, 1, 1, 1),
    std::make_tuple(3, DimSizes(4), DimSizes(512), DimSizes(512), 16, 16, 3, 1, 1, 1, 1),
    std::make_tuple(3, DimSizes(4), DimSizes(512), DimSizes(512), 16, 32, 2, 2, 1, 0, 0),
    std::make_tuple(3, DimSizes(2), DimSizes(256), DimSizes(256), 32, 32, 3, 1, 1, 1, 1),
    std::make_tuple(3, DimSizes(2), DimSizes(256), DimSizes(256), 64, 32, 3, 1, 1, 1, 1),
    std::make_tuple(3, DimSizes(4), DimSizes(512), DimSizes(512), 16, 32, 3, 2, 1, 0, 1),
    std::make_tuple(3, DimSizes(4), DimSizes(512), DimSizes(512), 1, 16, 3, 1, 1, 1, 1));

INSTANTIATE_TEST_SUITE_P(UNET3D_fwd_3dConv_ASIC, SynGaudiDevice3dConvFwdSpatialSlicingTest, unet3dConvSizes);
INSTANTIATE_TEST_SUITE_P(UNET3D_fwd_3dConv_ASIC_CI, SynGaudiDevice3dConvFwdSpatialSlicingTest, unet3dConvSanitySizes);

INSTANTIATE_TEST_SUITE_P(UNET3D_dedx_3dConv_ASIC, SynGaudiDevice3dConvDedxSpatialSlicingTest, unet3dConvSizes);
INSTANTIATE_TEST_SUITE_P(UNET3D_dedx_3dConv_ASIC_CI, SynGaudiDevice3dConvDedxSpatialSlicingTest, unet3dConvSanitySizes);

INSTANTIATE_TEST_SUITE_P(UNET3D_dedw_3dConv_ASIC, SynGaudiDevice3dConvDedwSpatialSlicingTest, unet3dConvSizes);
INSTANTIATE_TEST_SUITE_P(UNET3D_dedw_3dConv_ASIC_CI, SynGaudiDevice3dConvDedwSpatialSlicingTest, unet3dConvSanitySizes);