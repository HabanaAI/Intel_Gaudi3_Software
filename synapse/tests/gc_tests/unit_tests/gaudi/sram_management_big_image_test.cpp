#include "sram_management_fe_test.h"
#include "platform/gaudi/graph_compiler/passes.h"
namespace gaudi
{
class SRAMManagementBigImageTest: public SRAMManagementTest,
                                  public testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, int, int, bool>>
                                                          // height, width, nIFM, nOFM, filter, stride, dilation, padBefore, padAfter, smallSram
{
public:

    SRAMManagementBigImageTest() :
        m_height(std::get<0>(GetParam())),
        m_width(std::get<1>(GetParam())),
        m_nIFM(std::get<2>(GetParam())),
        m_nOFM(std::get<3>(GetParam())),
        m_filter(std::get<4>(GetParam())),
        m_stride(std::get<5>(GetParam())),
        m_dilation(std::get<6>(GetParam())),
        m_padBefore(std::get<7>(GetParam())),
        m_padAfter(std::get<8>(GetParam())),
        m_smallSram(std::get<9>(GetParam())),
        m_batch(2),
        m_convParams(m_filter, m_filter, m_stride, m_stride, m_padBefore, m_padAfter, m_padBefore, m_padAfter, m_dilation, m_dilation)
    {
        m_xSizes = {m_nIFM, m_width, m_height, m_batch};
        m_ySizes = {m_nOFM, 0, 0, m_batch};
        m_wSizes = {m_nOFM, m_nIFM, m_convParams.kW, m_convParams.kH};
        m_ySizes[1] = convOutputDimSize(m_xSizes[1], m_convParams.kW, m_convParams.dW, m_convParams.getPadL() + m_convParams.getPadR(), m_convParams.dilW);
        m_ySizes[2] = convOutputDimSize(m_xSizes[2], m_convParams.kH, m_convParams.dH, m_convParams.getPadT() + m_convParams.getPadB(), m_convParams.dilH);
    }

protected:

    pNode createFwdConvNode();
    pNode createDedwNode();
    pNode createDedxNode();

    unsigned m_height;
    unsigned m_width;
    unsigned m_nIFM;
    unsigned m_nOFM;
    unsigned m_filter;
    unsigned m_stride;
    unsigned m_dilation;
    unsigned m_padBefore;
    unsigned m_padAfter;
    bool m_smallSram;
    unsigned m_batch;
    synConvolutionParams m_convParams;
    std::vector<TSize> m_xSizes;
    std::vector<TSize> m_ySizes;
    std::vector<TSize> m_wSizes;

};

pNode SRAMManagementBigImageTest::createFwdConvNode()
{
    pTensor x = createTensor(m_xSizes, syn_type_bf16);
    pTensor w = createTensor(m_wSizes, syn_type_bf16);
    pTensor o = createTensor(m_ySizes, syn_type_bf16);
    pNode fwd = NodeFactory::createNode({x, w}, {o}, &m_convParams, NodeFactory::convolutionNodeTypeName, "fwd");
    return fwd;
}

pNode SRAMManagementBigImageTest::createDedwNode()
{
    pTensor dy = createTensor(m_ySizes, syn_type_bf16);
    pTensor xBwd = createTensor(m_xSizes, syn_type_bf16);
    pTensor dw = createTensor(m_wSizes, syn_type_bf16);
    pNode dedw = NodeFactory::createNode({dy, xBwd}, {dw}, &m_convParams, NodeFactory::deDwNodeTypeName, "dedw");
    return dedw;
}

pNode SRAMManagementBigImageTest::createDedxNode()
{
    pTensor dy2 = createTensor(m_ySizes, syn_type_bf16);
    pTensor w = createTensor(m_wSizes, syn_type_bf16);
    pTensor dx = createTensor(m_xSizes, syn_type_bf16);
    pNode dedx = NodeFactory::createNode({dy2, w}, {dx}, &m_convParams, NodeFactory::deDxNodeTypeName, "dedx");
    return dedx;
}

class SRAMManagementBigImageUnitTest: public SRAMManagementBigImageTest
{
public:
    SRAMManagementBigImageUnitTest() : SRAMManagementBigImageTest() {}
    void validateSlicingParams();

protected:
    void validateSlicingSizeAndOverlap(ConvBaseNode* conv, unsigned sliceSize);
};

void SRAMManagementBigImageUnitTest::validateSlicingSizeAndOverlap(ConvBaseNode* conv, unsigned outputSliceSize)
{
    pTensor output = conv->getOutput(0);
    SizeArray outputChunkSizes = output->getAllSizesInElements();
    unsigned sliceDim = DIM_H;
    outputChunkSizes[sliceDim] = outputSliceSize;
    const synConvolution3DParams& convParams = conv->getConvolutionParams();

    //calc 1st slice shape
    std::array<CoordArray,3> sliceCoord;
    std::array<TensorShape,3> inputShape;
    std::array<TensorShape,3> outputShape;
    sliceCoord[0] = {0};
    if (conv->getNodeType() == Node::TYPE_DEDX)
    {
        // first dx output slice starts from the padding
        sliceCoord[0][DIM_W] -= convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(DIM_W).paddingBeforeIndex];
        sliceCoord[0][DIM_H] -= convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(DIM_H).paddingBeforeIndex];
        sliceCoord[0][DIM_D_FOR_5D_TENSOR] -= convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(DIM_D_FOR_5D_TENSOR).paddingBeforeIndex];
    }
    outputShape[0] = TensorShape(output->getDim(), outputChunkSizes, sliceCoord[0]);
    inputShape[0] = conv->getInputShape(outputShape[0], TENSOR_OFM, TENSOR_IFM);
    //calc 2nd slice shape
    sliceCoord[1] = outputShape[0].getBases();
    // advance H dim with the input chunk size
    sliceCoord[1][sliceDim]+= outputChunkSizes[sliceDim];
    outputShape[1] = TensorShape(output->getDim(), outputChunkSizes, sliceCoord[1]);
    inputShape[1] = conv->getInputShape(outputShape[1], TENSOR_OFM, TENSOR_IFM);
    //calc 3rd slice shape
    sliceCoord[2] = outputShape[1].getBases();
    // advance H dim with the input chunk size
    sliceCoord[2][sliceDim]+= outputChunkSizes[sliceDim];
    outputShape[2] = TensorShape(output->getDim(), outputChunkSizes, sliceCoord[2]);
    inputShape[2] = conv->getInputShape(outputShape[2], TENSOR_OFM, TENSOR_IFM);

    // validate all slices have the same size
    std::array<unsigned,3> sliceSize;
    sliceSize[0] = inputShape[0].getSize(sliceDim);
    sliceSize[1] = inputShape[1].getSize(sliceDim);
    sliceSize[2] = inputShape[2].getSize(sliceDim);
    ASSERT_TRUE(sliceSize[0] == sliceSize[1]);
    ASSERT_TRUE(sliceSize[2] == sliceSize[1]);

    // validate the node overlap calculation is equal to the 2nd and 3rd slices overlaps
    int nodeOverlap = conv->getInputROIOverlapForDim(TENSOR_IFM, sliceDim);
    //calc the overlap - first slice end - seconds slice start on H dimension + 1
    int inputSliceOverlap1 = (inputShape[0].getBase(sliceDim) + inputShape[0].getSize(sliceDim) - 1) - inputShape[1].getBase(sliceDim) + 1;
    int inputSliceOverlap2 = (inputShape[1].getBase(sliceDim) + inputShape[1].getSize(sliceDim) - 1) - inputShape[2].getBase(sliceDim) + 1;
    ASSERT_TRUE(inputSliceOverlap1 == nodeOverlap);
    ASSERT_TRUE(inputSliceOverlap1 == inputSliceOverlap2);

    int padBefore = 0;
    int padAfter = 0;

    if (conv->getNodeType() == Node::TYPE_DEDX)
    {
        // validate the y operand shift is the node overlap
        ASSERT_TRUE(nodeOverlap == -inputShape[0].getBase(sliceDim));

        // validate the node X slice padding calculation is equal to projecting the y operand slice back to the X operand
        TensorShape paddedOutputSlice = conv->getXOperandShape(inputShape[0]);
        int paddedEnd = paddedOutputSlice.getBase(sliceDim) + paddedOutputSlice.getSize(sliceDim) - 1;
        int xEnd = outputShape[0].getBase(sliceDim) + outputShape[0].getSize(sliceDim) - 1;
        int paddedStart = paddedOutputSlice.getBase(sliceDim);
        int xStart = outputShape[0].getBase(sliceDim);
        padBefore = xStart - paddedStart;
        padAfter = paddedEnd - xEnd;
        int nodePadBefore = 0, nodePadAfter = 0;
        conv->getXStrideAlignedROIPaddingForDim(sliceDim, outputSliceSize, nodePadBefore, nodePadAfter);
        ASSERT_TRUE(padBefore == nodePadBefore);
        if (padAfter >= 0)
        {
            ASSERT_TRUE(padAfter == nodePadAfter);
        }
        else
        {
            // resetting negative padding to make sure all tensor lines are part of any slice
            ASSERT_TRUE(0 == nodePadAfter);
        }

        // validate the padding is the same for the next slice
        TensorShape paddedOutputSlice2 = conv->getXOperandShape(inputShape[1]);
        ASSERT_TRUE(paddedOutputSlice.getSize(sliceDim) == paddedOutputSlice2.getSize(sliceDim));
        int paddedEnd2 = paddedOutputSlice2.getBase(sliceDim) + paddedOutputSlice2.getSize(sliceDim) - 1;
        int xEnd2 = outputShape[1].getBase(sliceDim) + outputShape[1].getSize(sliceDim) - 1;
        int paddedStart2 = paddedOutputSlice2.getBase(sliceDim);
        int xStart2 = outputShape[1].getBase(sliceDim);
        int padBefore2 = xStart2 - paddedStart2;
        int padAfter2 = paddedEnd2 - xEnd2;
        ASSERT_TRUE(padBefore == padBefore2);
        ASSERT_TRUE(padAfter == padAfter2);
    }
    // validate slicing is correct - only if the padding after is not bigger than kernel size (as blocked by the solver)
    if (m_padAfter < conv->getDimActualKernelSize(sliceDim))
    {
        // validate the number of slices is correct
        std::shared_ptr<Bundle::Solution::SlicedOperand> wideOperand = std::make_shared<Bundle::Solution::SlicedOperand>(conv->getInput(0));
        std::shared_ptr<Bundle::Solution::SlicedOperand> outputOperand = std::make_shared<Bundle::Solution::SlicedOperand>(conv->getOutput(0));
        wideOperand->chunkDimensions[sliceDim] = sliceSize[0];
        outputOperand->chunkDimensions[sliceDim] = outputSliceSize;
        wideOperand->overlapElementsCount[sliceDim] = nodeOverlap;
        if (conv->getNodeType() == Node::TYPE_CONVOLUTION)
        {
            wideOperand->offsetBefore[sliceDim] = convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(sliceDim).paddingBeforeIndex];
            wideOperand->offsetAfter[sliceDim] = convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(sliceDim).paddingAfterIndex];
            wideOperand->minValidSliceSize[sliceDim] = conv->getDimActualKernelSize(sliceDim);
        }
        else if (conv->getNodeType() == Node::TYPE_DEDX)
        {
            outputOperand->offsetBefore[sliceDim] = convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(sliceDim).paddingBeforeIndex];
            outputOperand->offsetAfter[sliceDim] = convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(sliceDim).paddingAfterIndex];
            wideOperand->offsetBefore[sliceDim] = nodeOverlap;
            outputOperand->minValidSliceSize[sliceDim] = conv->getDimActualKernelSize(sliceDim) - padBefore;
            outputOperand->countPaddingOnlySlice = false;
        }
        unsigned numSlicesWide = SlicedOperandUtils::nofSlices(wideOperand, sliceDim);
        unsigned numSlicesOutput = SlicedOperandUtils::nofSlices(outputOperand, sliceDim);
        ASSERT_TRUE(numSlicesWide >= numSlicesOutput);
        // validate the last slice is not out-of-bounds
        // use the output num of slices, as the solution is built based on the output slices. extra input slice is thrown.
        unsigned dimensionOffsetWide = SlicedOperandUtils::getSliceCoordOffset(numSlicesOutput-1,
                                                                        wideOperand->chunkDimensions[sliceDim],
                                                                        wideOperand->overlapElementsCount[sliceDim],
                                                                        wideOperand->offsetBefore[sliceDim]);
        ASSERT_TRUE(wideOperand->finalShape[sliceDim] > dimensionOffsetWide) << " Wide dim offset " << dimensionOffsetWide << " >= dim size" << wideOperand->finalShape[sliceDim] << std::endl;
        unsigned dimensionOffsetOut = SlicedOperandUtils::getSliceCoordOffset(numSlicesOutput-1,
                                                                        outputOperand->chunkDimensions[sliceDim],
                                                                        outputOperand->overlapElementsCount[sliceDim],
                                                                        outputOperand->offsetBefore[sliceDim]);
        ASSERT_TRUE(outputOperand->finalShape[sliceDim] > dimensionOffsetOut) << " Out dim offset " << dimensionOffsetOut << " >= dim size" << outputOperand->finalShape[sliceDim] << std::endl;
    }
}


void SRAMManagementBigImageUnitTest::validateSlicingParams()
{
    pNode fwd = createFwdConvNode();
    pNode dedx = createDedxNode();

    unsigned minSliceSize = std::max(m_padBefore, (m_filter + (m_filter-1) * (m_dilation-1))) + 1;
    unsigned aligned = minSliceSize - (minSliceSize % m_stride) + m_stride;
    for (unsigned sliceSize = aligned; sliceSize <= 256; sliceSize += m_stride)
    {
        //std::cout << "sliceSize = " << sliceSize << "; " << "filter = " << m_filter << "; " <<
        //                "stride = " << m_stride << "; "  << "dilation = " << m_dilation << "; " <<
        //                "padBefore = " << m_padBefore << "; " << "padAfter = " << m_padAfter << ";" << std::endl;

        validateSlicingSizeAndOverlap((ConvBaseNode*)fwd.get(), sliceSize);
        validateSlicingSizeAndOverlap((ConvBaseNode*)dedx.get(), sliceSize);

        ASSERT_FALSE(HasFailure()) << "Failed params: " << "sliceSize = " << sliceSize << "; " << "filter = " << m_filter << "; " <<
                                        "stride = " << m_stride << "; "  << "dilation = " << m_dilation << "; "  <<
                                        "padBefore = " << m_padBefore << "; " << "padAfter = " << m_padAfter << ";"  << std::endl;
    }
}

TEST_P(SRAMManagementBigImageUnitTest, slice_size_and_overlap_calc_big_image)
{
    validateSlicingParams();
}

INSTANTIATE_TEST_SUITE_P(slice_size_and_overlap_calc_big_image_full,
                        SRAMManagementBigImageUnitTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({256}), //height
                            ::testing::ValuesIn({256}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({256}), //channels out
                            ::testing::Range(1, 5), //filter
                            ::testing::Range(1, 5), //stride
                            ::testing::Range(1, 5), //dilation
                            ::testing::Range(0, 10), //padBefore
                            ::testing::Range(0, 10),  //padAfter
                            ::testing::ValuesIn({false}) //smallSram
                        ));


class SRAMManagementBigImageSlicePassTest: public SRAMManagementBigImageTest
{
public:
    SRAMManagementBigImageSlicePassTest() : SRAMManagementBigImageTest() {}
    void sliceMmeInputs();
};

// Call the sliceGraphToSRAMCapacity pass to validate it's running successfully on any configuration.
// Allow for small sizes SRAM just to validate the flow, or validate the inputs are in SRAM for large SRAM.
void SRAMManagementBigImageSlicePassTest::sliceMmeInputs()
{
    setGlobalConfForTest(GCFG_SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED, "true");
    setGlobalConfForTest(GCFG_SRAM_SLICER_4D_DEDW_SPATIAL_SLICE_ENABLED, "true");
    setGlobalConfForTest(GCFG_SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED, "true");

    //std::cout << "*************************" << std::endl;
    //std::cout << "filter = " << m_filter << "; " << "stride = " << m_stride << "; "  << "dilation = " << m_dilation << "; " <<
    //             "padBefore = " << m_padBefore << "; " << "padAfter = " <<m_padAfter << ";" << std::endl;

    if (m_smallSram)
    {
        setGlobalConfForTest(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, "4000000");
    }

    GaudiGraph g;
    pNode fwd = createFwdConvNode();
    ASSERT_TRUE(GraphEditor::addNode(g, fwd));

    pNode dedw = createDedwNode();
    ASSERT_TRUE(GraphEditor::addNode(g, dedw));

    pNode dedx = createDedxNode();
    ASSERT_TRUE(GraphEditor::addNode(g, dedx));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));
    // Validate the inputs are in SRAM if SRAM is set to be big, and conv params allow for 4D slicing.
    if (!m_smallSram && (m_padAfter < (m_filter + (m_filter - 1) * (m_dilation - 1))))
    {
        for (pNode n : g.getExeSortedNodes())
        {
            if (g.runsOnMME(n))
            {
                for (pTensor input : n->getInputs())
                {
                    if (input)
                    {
                        ASSERT_TRUE(input->inSram());
                    }
                }
            }
        }
    }
}

TEST_P(SRAMManagementBigImageSlicePassTest, mme_inputs_slicing_big_image)
{
    sliceMmeInputs();
}

INSTANTIATE_TEST_SUITE_P(mme_inputs_should_be_located_in_sram_big_image,
                         SRAMManagementBigImageSlicePassTest,
                         ::testing::Combine(::testing::ValuesIn({256}),   // height
                                            ::testing::ValuesIn({256}),   // width
                                            ::testing::ValuesIn({256}),   // channels in
                                            ::testing::ValuesIn({256}),   // channels out
                                            ::testing::Range(1, 5),       // filter
                                            ::testing::Range(1, 4),       // stride
                                            ::testing::Range(1, 5),       // dilation
                                            ::testing::Range(0, 4),       // padBefore
                                            ::testing::Range(0, 4),       // padAfter
                                            ::testing::ValuesIn({false})  // smallSram
                                            ));

INSTANTIATE_TEST_SUITE_P(mme_inputs_should_be_located_in_sram_big_image_slice_2_spatials,
                         SRAMManagementBigImageSlicePassTest,
                         ::testing::Combine(::testing::ValuesIn({16}),     // height
                                            ::testing::ValuesIn({16384}),  // width
                                            ::testing::ValuesIn({1024}),   // channels in
                                            ::testing::ValuesIn({1024}),   // channels out
                                            ::testing::Range(1, 3),        // filter
                                            ::testing::Range(1, 3),        // stride
                                            ::testing::Range(1, 3),        // dilation
                                            ::testing::Range(0, 2),        // padBefore
                                            ::testing::Range(0, 2),        // padAfter
                                            ::testing::ValuesIn({false})   // smallSram
                                            ));

INSTANTIATE_TEST_SUITE_P(basic_mme_inputs_slicing_big_image,
                        SRAMManagementBigImageSlicePassTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({256}), //height
                            ::testing::ValuesIn({256}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({16}), //channels out
                            ::testing::Range(1, 5), //filter
                            ::testing::Range(1, 3), //stride
                            ::testing::Range(1, 5), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4),  //padAfter
                            ::testing::ValuesIn({true}) //smallSram
                        ));

INSTANTIATE_TEST_SUITE_P(basic_mme_inputs_slicing_big_image_large_stride,
                        SRAMManagementBigImageSlicePassTest,
                        ::testing::Combine(
                            ::testing::ValuesIn({512}), //height
                            ::testing::ValuesIn({256}), //width
                            ::testing::ValuesIn({256}), //channels in
                            ::testing::ValuesIn({40}), //channels out
                            ::testing::Range(1, 5), //filter
                            ::testing::Range(3, 5), //stride
                            ::testing::Range(1, 5), //dilation
                            ::testing::Range(0, 4), //padBefore
                            ::testing::Range(0, 4),  //padAfter
                            ::testing::ValuesIn({true}) //smallSram
                        ));


class SRAMManagementBigImageGemmTest: public SRAMManagementTest,
                                      public testing::WithParamInterface<std::tuple<int, int, int, int>>
                                                                                // heightA, commonDim, widthB, batch
{
public:

    SRAMManagementBigImageGemmTest() :
        m_heightA(std::get<0>(GetParam())),
        m_commonDim(std::get<1>(GetParam())),
        m_widthB(std::get<2>(GetParam())),
        m_batch(std::get<3>(GetParam()))
    {}

protected:

    unsigned m_heightA;
    unsigned m_commonDim;
    unsigned m_widthB;
    unsigned m_batch;
};

class SRAMManagementBigImageGemmSlicePassTest: public SRAMManagementBigImageGemmTest
{
public:
    SRAMManagementBigImageGemmSlicePassTest() : SRAMManagementBigImageGemmTest() {}
    void sliceMmeInputs();
};

void SRAMManagementBigImageGemmSlicePassTest::sliceMmeInputs()
{
    GaudiGraph g;

    synGEMMParams gemmParams{};
    pTensor a = createTensor({m_commonDim, m_heightA}, syn_type_bf16);
    pTensor b = createTensor({m_widthB, m_commonDim}, syn_type_bf16);
    pTensor gemmOut = createTensor({m_widthB, m_heightA}, syn_type_bf16);
    pNode gemm = NodeFactory::createNode({a, b}, {gemmOut}, &gemmParams, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(GraphEditor::addNode(g, gemm));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));
    // Validate the inputs are in SRAM
    for (pNode n : g.getExeSortedNodes())
    {
        if (g.runsOnMME(n))
        {
            for (pTensor input : n->getInputs())
            {
                if (input)
                {
                    ASSERT_TRUE(input->inSram());
                }
            }
        }
    }
}

TEST_P(SRAMManagementBigImageGemmSlicePassTest, mme_inputs_slicing_big_image)
{
    sliceMmeInputs();
}

INSTANTIATE_TEST_SUITE_P(gemm_inputs_should_be_located_in_sram_big_image,
                        SRAMManagementBigImageGemmSlicePassTest,
                        ::testing::Values(std::make_tuple(2048, 186368, 4, 1),
                                          std::make_tuple(4, 186368, 2048, 1),
                                          std::make_tuple(100, 186368, 2048, 2),
                                          std::make_tuple(4096, 186368, 2048, 1)
                        ));

TEST_F(SRAMManagementTest, tpc_producer_stitching_to_big_conv)
{
    GaudiGraph g;

    synConvolutionParams convParams(3, 3, 2, 2, 1, 2, 1, 2, 2, 2);
    TSize inH = 256, inW = 256, c = 256, k = 256, batch = 2;
    std::vector<TSize> xSizes = {c, inW, inH, batch};
    std::vector<TSize> ySizes = {k, 0, 0, batch};
    std::vector<TSize> wSizes = {k, c, convParams.kW, convParams.kH};
    ySizes[1] = convOutputDimSize(xSizes[1], convParams.kW, convParams.dW, convParams.getPadL() + convParams.getPadR(), convParams.dilW);
    ySizes[2] = convOutputDimSize(xSizes[2], convParams.kH, convParams.dH, convParams.getPadT() + convParams.getPadB(), convParams.dilH);

    pTensor x = createTensor(xSizes, syn_type_float);
    pTensor w = createTensor(wSizes, syn_type_float);
    pTensor o = createTensor(ySizes, syn_type_float);
    pNode convFwd = NodeFactory::createNode({x, w}, {o}, &convParams, NodeFactory::convolutionNodeTypeName, "conv_fwd");
    ASSERT_TRUE(GraphEditor::addNode(g, convFwd));

    pTensor xProducer = createTensor(xSizes, syn_type_float);
    pNode relu1 = NodeFactory::createNode({xProducer}, {x}, nullptr, "relu_fwd_f32", "relu1");
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));

    pTensor wProducer = createTensor(wSizes, syn_type_float);
    pNode relu2 = NodeFactory::createNode({wProducer}, {w}, nullptr, "relu_fwd_f32", "relu2");
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));

    ASSERT_TRUE(loadTpcKernels(g));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));

    // conv_fwd + relu2 should be bundled together (w operand doesn't have overlap).
    // relu1 should not be stitched (x operand has overlap).
    ASSERT_TRUE(convFwd->getNodeAnnotation().bundleInfo.is_set());
    auto bundleIdx = convFwd->getNodeAnnotation().bundleInfo->bundleIndex;
    ASSERT_FALSE(relu1->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_TRUE(relu2->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_EQ(relu2->getNodeAnnotation().bundleInfo->bundleIndex, bundleIdx);
}

TEST_F(SRAMManagementTest, tpc_producer_stitching_to_big_dedw)
{
    GaudiGraph g;

    synConvolutionParams convParams(3, 3, 2, 2, 1, 2, 1, 2, 2, 2);
    TSize inH = 256, inW = 256, c = 256, k = 256, batch = 2;
    std::vector<TSize> xSizes = {c, inW, inH, batch};
    std::vector<TSize> ySizes = {k, 0, 0, batch};
    std::vector<TSize> wSizes = {k, c, convParams.kW, convParams.kH};
    ySizes[1] = convOutputDimSize(xSizes[1], convParams.kW, convParams.dW, convParams.getPadL() + convParams.getPadR(), convParams.dilW);
    ySizes[2] = convOutputDimSize(xSizes[2], convParams.kH, convParams.dH, convParams.getPadT() + convParams.getPadB(), convParams.dilH);

    pTensor dy = createTensor(ySizes, syn_type_float);
    pTensor xBwd = createTensor(xSizes, syn_type_float);
    pTensor dw = createTensor(wSizes, syn_type_float);
    pNode dedw = NodeFactory::createNode({dy, xBwd}, {dw}, &convParams, NodeFactory::deDwNodeTypeName, "dedw");
    ASSERT_TRUE(GraphEditor::addNode(g, dedw));

    pTensor dyProducer = createTensor(ySizes, syn_type_float);
    pNode relu1 = NodeFactory::createNode({dyProducer}, {dy}, nullptr, "relu_fwd_f32", "relu1");
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));

    pTensor xBwdProducer = createTensor(xSizes, syn_type_float);
    pNode relu2 = NodeFactory::createNode({xBwdProducer}, {xBwd}, nullptr, "relu_fwd_f32", "relu2");
    ASSERT_TRUE(GraphEditor::addNode(g, relu2));

    ASSERT_TRUE(loadTpcKernels(g));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));

    // dedw + relu1 should be bundled together (dy operand doesn't have overlap).
    // relu2 should not be stitched (x operand has overlap).
    ASSERT_TRUE(dedw->getNodeAnnotation().bundleInfo.is_set());
    auto bundleIdx = dedw->getNodeAnnotation().bundleInfo->bundleIndex;
    ASSERT_TRUE(relu1->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_EQ(relu1->getNodeAnnotation().bundleInfo->bundleIndex, bundleIdx);
    ASSERT_FALSE(relu2->getNodeAnnotation().bundleInfo.is_set());
}

TEST_F(SRAMManagementTest, two_big_dedw_with_shared_input)
{
    GaudiGraph g;

    synConvolutionParams convParams(3, 3, 2, 2, 1, 2, 1, 2, 2, 2);
    TSize inH = 256, inW = 256, c = 256, k = 256, batch = 2;
    std::vector<TSize> xSizes = {c, inW, inH, batch};
    std::vector<TSize> ySizes = {k, 0, 0, batch};
    std::vector<TSize> wSizes = {k, c, convParams.kW, convParams.kH};
    ySizes[1] = convOutputDimSize(xSizes[1], convParams.kW, convParams.dW, convParams.getPadL() + convParams.getPadR(), convParams.dilW);
    ySizes[2] = convOutputDimSize(xSizes[2], convParams.kH, convParams.dH, convParams.getPadT() + convParams.getPadB(), convParams.dilH);

    pTensor xBwd_shared = createTensor(xSizes, syn_type_float);

    pTensor dy1 = createTensor(ySizes, syn_type_float);
    pTensor dw1 = createTensor(wSizes, syn_type_float);
    pNode dedw1 = NodeFactory::createNode({dy1, xBwd_shared}, {dw1}, &convParams, NodeFactory::deDwNodeTypeName, "dedw1");
    ASSERT_TRUE(GraphEditor::addNode(g, dedw1));

    pTensor dy1Producer = createTensor(ySizes, syn_type_float);
    pNode relu1 = NodeFactory::createNode({dy1Producer}, {dy1}, nullptr, "relu_fwd_f32", "relu1");
    ASSERT_TRUE(GraphEditor::addNode(g, relu1));

    pTensor dy2 = createTensor(ySizes, syn_type_float);
    pTensor dw2 = createTensor(wSizes, syn_type_float);
    pNode dedw2 = NodeFactory::createNode({dy2, xBwd_shared}, {dw2}, &convParams, NodeFactory::deDwNodeTypeName, "dedw2");
    ASSERT_TRUE(GraphEditor::addNode(g, dedw2));

    ASSERT_TRUE(loadTpcKernels(g));

    ASSERT_TRUE(sliceGraphToSRAMCapacity(g));

    // dedw1 + relu1 should be bundled together (dy operand doesn't have overlap).
    // dedw2 should be in a separate bundle (dedw1 has input with overlap so slave MME can't be added).
    ASSERT_TRUE(dedw1->getNodeAnnotation().bundleInfo.is_set());
    auto bundle1Idx = dedw1->getNodeAnnotation().bundleInfo->bundleIndex;
    ASSERT_TRUE(relu1->getNodeAnnotation().bundleInfo.is_set());
    ASSERT_EQ(relu1->getNodeAnnotation().bundleInfo->bundleIndex, bundle1Idx);
    ASSERT_TRUE(dedw2->getNodeAnnotation().bundleInfo.is_set());
    auto bundle2Idx = dedw2->getNodeAnnotation().bundleInfo->bundleIndex;
    ASSERT_NE(bundle1Idx, bundle2Idx);
}

}