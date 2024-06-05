#include "data_layout_test_infra.h"
#include "utils.h"
#include "node_factory.h"

class SynGaudiDataLayoutBasicTest : public SynGaudiDataLayoutTest
{
public:
    synConvolution3DParams paramsTo3DParams(synConvolutionParams params, unsigned q)
    {
        synConvolution3DParams cov3DParams;

        cov3DParams.kernel[CONV_KERNEL_WIDTH]  = params.kW;
        cov3DParams.kernel[CONV_KERNEL_HEIGHT] = params.kH;
        cov3DParams.kernel[CONV_KERNEL_DEPTH]  = q;

        cov3DParams.stride[CONV_STRIDE_WIDTH]  = params.dW;
        cov3DParams.stride[CONV_STRIDE_HEIGHT] = params.dH;
        cov3DParams.stride[CONV_STRIDE_DEPTH]  = 1;

        cov3DParams.padding[CONV_PAD_LEFT]   = params.padL;
        cov3DParams.padding[CONV_PAD_RIGHT]  = params.padR;
        cov3DParams.padding[CONV_PAD_TOP]    = params.padT;
        cov3DParams.padding[CONV_PAD_BOTTOM] = params.padB;
        cov3DParams.padding[CONV_PAD_FRONT]  = 0;
        cov3DParams.padding[CONV_PAD_BACK]   = 0;

        cov3DParams.dilation[CONV_DIL_WIDTH]  = params.dilW;
        cov3DParams.dilation[CONV_DIL_HEIGHT] = params.dilH;
        cov3DParams.dilation[CONV_DIL_DEPTH]  = 1;

        cov3DParams.activation = params.activation;
        cov3DParams.nGroups    = 1;
        return cov3DParams;
    }

    void validateResults(float* results, float* expected, unsigned* out_sizes)
    {
        for (uint64_t i = 0; i < getNumberOfElements(out_sizes); i++)
        {
            ASSERT_EQ(*results, expected[i])
                << "Mismatch at index " << i << " Result:" << *results << " Ref: " << expected[i];
            results++;
        }
    }
};

TEST_F_GC(SynGaudiDataLayoutBasicTest, NCHW_conv)
{
    synConvolutionParams params;

    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 3;
    params.kW   = 3;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    const unsigned batch = 1;
    const unsigned nIFM  = 1;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    float ifmBuffer[] =
            {
                1, 1, 2, 2, 3, 3,
                1, 1, 2, 2, 3, 3,
                4, 3, 1, 3, 2, 1,
                2, 6, 2, 0, 0, 0,
                1, 5, 2, 0, 0, 0,
                4, 6, 5, 0, 0, 0,
            };

    float wghBuffer[]
            {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            };

    // Calculated mannually as infra util only calculates for NHWC layout
    float ofmRefBuffer[]
            {
                3, 6, 6, 6,
                6, 2, 5, 4,
                12, 5, 1, 3,
                12, 8, 2, 0,
            };

    // create_tensor's layout
    unsigned dims = 4;
    // Pass input as NCHW(same as used in PT)
    unsigned ifmDimSizes[] = {wIFM, hIFM, nIFM, batch};
    unsigned ofmDimSizes[] = {wOFM, hOFM, nOFM, batch};
    // Pass weight as KCRS(same as used in PT)
    unsigned wghDimSizes[] = {params.kW, params.kH, nIFM, nOFM};

    unsigned xTensorIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, ifmBuffer, ifmDimSizes, dims, syn_type_single);
    unsigned wTensorIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBuffer, wghDimSizes, dims, syn_type_single);
    unsigned yTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_single);

    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeWithLayouts(NodeFactory::convolutionNodeTypeName,
                       inputIndices,
                       outputIndices,
                       (void*)&params,
                       sizeof(synConvolutionParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[yTensorIndex];
    validateResults(pOutputBuffer, ofmRefBuffer, ofmDimSizes);
}

TEST_F_GC(SynGaudiDataLayoutBasicTest, NCHW_dedx)
{
    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    const unsigned b = 1, h = 3, w = 3, k = 1, c = 1, r = 2, s = 2, oH = 2, oW = 2;
    const unsigned dims = 4;

    unsigned dySizes[]  = {oW, oH, k, b};
    unsigned xSizes[]   = {w, h, c, b};
    unsigned wghSizes[] = {s, r, c, k};

    float dyBuffer[]
            {
                2, 3,
                4, 2,
            };

    float wghBuffer[]
            {
                1, 0,
                0, 2,
            };

    // Calculated mannually as infra util only calculates for NHWC layout
    float dxBuffer[] =
            {
                2, 3, 0,
                4, 6, 6,
                0, 8, 4,
            };

    unsigned dy =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, dyBuffer, dySizes, dims, syn_type_single);
    unsigned wgh =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBuffer, wghSizes, dims, syn_type_single);
    unsigned dx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes, dims, syn_type_single);

    TensorIndices inputIndices  = {dy, wgh};
    TensorIndices outputIndices = {dx};

    addNodeWithLayouts(NodeFactory::deDxNodeTypeName,
                       inputIndices,
                       outputIndices,
                       (void*)&params,
                       sizeof(synConvolutionParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[dx];
    validateResults(pOutputBuffer, dxBuffer, xSizes);
}

TEST_F_GC(SynGaudiDataLayoutBasicTest, NCHW_dedx3d)
{
    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    const unsigned b = 1, h = 3, w = 3, k = 1, c = 1, r = 2, s = 2, oH = 2, oW = 2;
    const unsigned d = 1, q = 1, oD = 1;
    const unsigned dims = 5;

    synConvolution3DParams cov3DParams = paramsTo3DParams(params, q);

    unsigned dySizes[]  = {oW, oH, oD, k, b};
    unsigned xSizes[]   = {w, h, d, c, b};
    unsigned wghSizes[] = {s, r, q, c, k};

    float dyBuffer[]
            {
                2, 3,
                4, 2,
            };

    float wghBuffer[]
            {
                1, 0,
                0, 2,
            };

    // Calculated mannually as infra util only calculates for NHWC layout
    float dxBuffer[] =
            {
                2, 3, 0,
                4, 6, 6,
                0, 8, 4,
            };

    unsigned dy =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, dyBuffer, dySizes, dims, syn_type_single);
    unsigned wgh =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, wghBuffer, wghSizes, dims, syn_type_single);
    unsigned dx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes, dims, syn_type_single);

    TensorIndices inputIndices  = {dy, wgh};
    TensorIndices outputIndices = {dx};

    addNodeWithLayouts(NodeFactory::deDx3DNodeTypeName,
                       inputIndices,
                       outputIndices,
                       (void*)&cov3DParams,
                       sizeof(synConvolution3DParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[dx];
    validateResults(pOutputBuffer, dxBuffer, xSizes);
}

TEST_F_GC(SynGaudiDataLayoutBasicTest, NCHW_dedw)
{
    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    const unsigned b = 1, h = 3, w = 3, k = 1, c = 1, r = 2, s = 2, oH = 2, oW = 2;
    const unsigned dims = 4;

    unsigned dySizes[]  = {oW, oH, k, b};
    unsigned xSizes[]   = {w, h, c, b};
    unsigned wghSizes[] = {s, r, c, k};

    float dyBuffer[]
            {
                1, 2,
                0, 2,
            };

    float xBuffer[] =
            {
                1, 3, 2,
                2, 0, 3,
                0, 5, 4,
            };

    // Calculated mannually as infra util only calculates for NHWC layout
    float dwBuffer[]
        {
            7, 13,
            12, 14,
        };

    unsigned dy =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, dyBuffer, dySizes, dims, syn_type_single);
    unsigned x  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, xBuffer, xSizes, dims, syn_type_single);
    unsigned dw = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wghSizes, dims, syn_type_single);

    TensorIndices inputIndices  = {dy, x};
    TensorIndices outputIndices = {dw};

    addNodeWithLayouts(NodeFactory::deDwNodeTypeName,
                       inputIndices,
                       outputIndices,
                       (void*)&params,
                       sizeof(synConvolutionParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[dw];
    validateResults(pOutputBuffer, dwBuffer, wghSizes);
}

TEST_F_GC(SynGaudiDataLayoutBasicTest, NCHW_dedw3d)
{
    synConvolutionParams params;
    params.dH   = 1;
    params.dW   = 1;
    params.kH   = 2;
    params.kW   = 2;
    params.dilH = 1;
    params.dilW = 1;

    params.padT = 0;
    params.padB = 0;
    params.padL = 0;
    params.padR = 0;

    const unsigned b = 1, h = 3, w = 3, k = 1, c = 1, r = 2, s = 2, oH = 2, oW = 2;
    const unsigned d = 1, q = 1, oD = 1;
    const unsigned dims = 5;

    synConvolution3DParams cov3DParams = paramsTo3DParams(params, q);

    unsigned dySizes[]  = {oW, oH, oD, k, b};
    unsigned xSizes[]   = {w, h, d, c, b};
    unsigned wghSizes[] = {s, r, q, c, k};

    float dyBuffer[]
            {
                1, 2,
                0, 2,
            };

    float xBuffer[] =
            {
                1, 3, 2,
                2, 0, 3,
                0, 5, 4,
            };

    // Calculated mannually as infra util only calculates for NHWC layout
    float dwBuffer[]
        {
            7, 13,
            12, 14,
        };

    unsigned dy =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, dyBuffer, dySizes, dims, syn_type_single);
    unsigned x  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, xBuffer, xSizes, dims, syn_type_single);
    unsigned dw = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, wghSizes, dims, syn_type_single);

    TensorIndices inputIndices  = {dy, x};
    TensorIndices outputIndices = {dw};

    addNodeWithLayouts(NodeFactory::deDw3DNodeTypeName,
                       inputIndices,
                       outputIndices,
                       (void*)&cov3DParams,
                       sizeof(synConvolution3DParams));

    compileAndRun();

    float* pOutputBuffer = (float*)m_hostBuffers[dw];
    validateResults(pOutputBuffer, dwBuffer, wghSizes);
}

class SynGaudiDataLayoutMMETest
: public SynGaudiDataLayoutTest
, public testing::WithParamInterface<std::tuple<bool /* dynamic shape */,
                                                bool /* allow output permute */,
                                                bool /* permute A */,
                                                bool /* permute B */,
                                                ERepefenceOp>>
{
protected:
    void run3DConvolution();

    void validateResults(unsigned x, unsigned y, unsigned w);

    const char*            m_guid;
    synConvolution3DParams m_params;
    unsigned               m_batch;
    unsigned               m_nIFM;
    unsigned               m_nOFM;
    unsigned               m_wIFMMin;
    unsigned               m_wIFMMax;
    unsigned               m_hIFM;
    unsigned               m_dIFM;

    bool m_permuteA;
    bool m_permuteB;
    bool m_allowPermuteOut;
    bool m_dynamicShape;

    static const unsigned DIM = 5;
    static const char*    m_xName;
    static const char*    m_yName;
    static const char*    m_wName;
};

const char* SynGaudiDataLayoutMMETest::m_xName = "X";
const char* SynGaudiDataLayoutMMETest::m_yName = "Y";
const char* SynGaudiDataLayoutMMETest::m_wName = "W";

void SynGaudiDataLayoutMMETest::validateResults(unsigned x, unsigned y, unsigned w)
{
    synTensorDescriptor xDesc = m_tensorDescs[x];
    synTensorDescriptor yDesc = m_tensorDescs[y];
    synTensorDescriptor wDesc = m_tensorDescs[w];

    memcpy(xDesc.m_sizes, xDesc.m_minSizes, sizeof(xDesc.m_minSizes));
    memcpy(yDesc.m_sizes, yDesc.m_minSizes, sizeof(yDesc.m_minSizes));

    char* xData = (char*)m_hostBuffers[x];
    char* yData = (char*)m_hostBuffers[y];
    char* wData = (char*)m_hostBuffers[w];

    const gc::Permutation& activationPerm = m_ptActivationPermutation5D;
    const gc::Permutation& weightPerm     = m_ptWeightPermutation5D;

    ERepefenceOp op;

    // reference is calculated in synapse layout, so if we don't have the expected permutation we need to transpose the
    // data
    bool xPermuted = isPermuted(m_xName, activationPerm);
    bool yPermuted = isPermuted(m_yName, activationPerm);
    bool wPermuted = isPermuted(m_wName, weightPerm);
    if (!xPermuted)
    {
        transposeBuffer(xDesc.m_sizes, DIM, (float*)xData, activationPerm);
    }
    if (!yPermuted)
    {
        transposeBuffer(yDesc.m_sizes, DIM, (float*)yData, activationPerm);
    }
    if (!wPermuted)
    {
        transposeBuffer(wDesc.m_sizes, DIM, (float*)wData, weightPerm);
    }

    if (std::strcmp(m_guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        op = REFERENCE_OP_FWD;
        EXPECT_EQ(m_permuteA, xPermuted);
        EXPECT_EQ(m_permuteB, wPermuted);
        EXPECT_EQ(m_allowPermuteOut, yPermuted);
    }
    else if (std::strcmp(m_guid, NodeFactory::deDx3DNodeTypeName) == 0)
    {
        op = REFERENCE_OP_DEDX;
        EXPECT_EQ(m_permuteA, yPermuted);
        EXPECT_EQ(m_permuteB, wPermuted);
        EXPECT_EQ(m_allowPermuteOut, xPermuted);
    }
    else  // if (std::strcmp(guid, NodeFactory::deDw3DNodeTypeName) == 0)
    {
        op = REFERENCE_OP_DEDW;
        EXPECT_EQ(m_permuteA, yPermuted);
        EXPECT_EQ(m_permuteB, xPermuted);
        EXPECT_EQ(m_allowPermuteOut, wPermuted);
    }

    activationPerm.permuteShape(xDesc.m_sizes, DIM);
    activationPerm.permuteShape(yDesc.m_sizes, DIM);
    weightPerm.permuteShape(wDesc.m_sizes, DIM);

    CoordArray wrongIdx;
    float      expectedResult = 0;

    bool ret =
        checkMmeOp(xDesc, xData, wDesc, wData, yDesc, yData, m_params, op, wrongIdx, m_deviceType, &expectedResult);

    if (std::strcmp(m_guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[y].m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true)
            << "Wrong value for CONV op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
            << getIndexValue(sizes, wrongIdx, m_tensorDescs[y].m_dataType, m_hostBuffers[y])
            << " Expected: " << expectedResult;
    }
    else if (std::strcmp(m_guid, NodeFactory::deDx3DNodeTypeName) == 0)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[x].m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true)
            << "Wrong value for DEDX op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
            << getIndexValue(sizes, wrongIdx, m_tensorDescs[x].m_dataType, m_hostBuffers[x])
            << " Expected: " << expectedResult;
    }
    else  // if (std::strcmp(guid, NodeFactory::deDw3DNodeTypeName) == 0)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[w].m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true)
            << "Wrong value for DEDW op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
            << getIndexValue(sizes, wrongIdx, m_tensorDescs[w].m_dataType, m_hostBuffers[w])
            << " Expected: " << expectedResult;
    }
}

void SynGaudiDataLayoutMMETest::run3DConvolution()
{
    const unsigned wOFMMax = convOutputDimSize(m_wIFMMax,
                                               m_params.kernel[CONV_KERNEL_WIDTH],
                                               m_params.stride[CONV_STRIDE_WIDTH],
                                               m_params.padding[CONV_PAD_LEFT],
                                               m_params.dilation[CONV_DIL_WIDTH]);
    const unsigned wOFMMin = convOutputDimSize(m_wIFMMin,
                                               m_params.kernel[CONV_KERNEL_WIDTH],
                                               m_params.stride[CONV_STRIDE_WIDTH],
                                               m_params.padding[CONV_PAD_LEFT],
                                               m_params.dilation[CONV_DIL_WIDTH]);
    const unsigned hOFM    = convOutputDimSize(m_hIFM,
                                            m_params.kernel[CONV_KERNEL_HEIGHT],
                                            m_params.stride[CONV_STRIDE_HEIGHT],
                                            m_params.padding[CONV_PAD_TOP],
                                            m_params.dilation[CONV_DIL_HEIGHT]);
    const unsigned dOFM    = convOutputDimSize(m_dIFM,
                                            m_params.kernel[CONV_KERNEL_DEPTH],
                                            m_params.stride[CONV_STRIDE_DEPTH],
                                            m_params.padding[CONV_PAD_FRONT],
                                            m_params.dilation[CONV_DIL_DEPTH]);

    unsigned sizesXMax[] = {m_nIFM, m_wIFMMax, m_hIFM, m_dIFM, m_batch};
    unsigned sizesXMin[] = {m_nIFM, m_wIFMMin, m_hIFM, m_dIFM, m_batch};
    unsigned sizesW[]    = {m_nOFM,
                         m_nIFM,
                         m_params.kernel[CONV_KERNEL_WIDTH],
                         m_params.kernel[CONV_KERNEL_HEIGHT],
                         m_params.kernel[CONV_KERNEL_DEPTH]};
    unsigned sizesYMax[] = {m_nOFM, wOFMMax, hOFM, dOFM, m_batch};
    unsigned sizesYMin[] = {m_nOFM, wOFMMin, hOFM, dOFM, m_batch};

    m_ptActivationPermutation5D.getInversePermutation().permuteShape(sizesXMax, DIM);
    m_ptActivationPermutation5D.getInversePermutation().permuteShape(sizesXMin, DIM);
    m_ptActivationPermutation5D.getInversePermutation().permuteShape(sizesYMax, DIM);
    m_ptActivationPermutation5D.getInversePermutation().permuteShape(sizesYMin, DIM);
    m_ptWeightPermutation5D.getInversePermutation().permuteShape(sizesW, DIM);

    TensorUsage xUsage = INPUT_TENSOR;
    TensorUsage yUsage = INPUT_TENSOR;
    TensorUsage wUsage = INPUT_TENSOR;

    if (std::strcmp(m_guid, NodeFactory::deDx3DNodeTypeName) == 0)
    {
        xUsage = OUTPUT_TENSOR;
    }
    else if (std::strcmp(m_guid, NodeFactory::deDw3DNodeTypeName) == 0)
    {
        wUsage = OUTPUT_TENSOR;
    }
    else if (std::strcmp(m_guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        yUsage = OUTPUT_TENSOR;
    }

    synTensorPermutation activationPermutation;
    synTensorPermutation weightPermutation;
    activationPermutation.dims = m_ptActivationPermutation5D.size();
    weightPermutation.dims     = m_ptActivationPermutation5D.size();
    for (unsigned i = 0; i < DIM; i++)
    {
        activationPermutation.permutation[i] = m_ptActivationPermutation5D.getValues()[i];
        weightPermutation.permutation[i]     = m_ptWeightPermutation5D.getValues()[i];
    }

    unsigned x = createPersistTensor(xUsage,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     sizesXMax,
                                     CONV_3D_TENSOR_DIM,
                                     syn_type_float,
                                     nullptr,
                                     m_xName,
                                     0,
                                     0,
                                     nullptr,
                                     sizesXMin);

    unsigned w = createPersistTensor(wUsage,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     sizesW,
                                     CONV_3D_TENSOR_DIM,
                                     syn_type_float,
                                     nullptr,
                                     m_wName);

    unsigned y = createPersistTensor(yUsage,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,  // initializer
                                     sizesYMax,
                                     CONV_3D_TENSOR_DIM,
                                     syn_type_float,
                                     nullptr,
                                     m_yName,
                                     0,
                                     0,
                                     nullptr,
                                     sizesYMin);

    unsigned shapeX = createShapeTensor(INPUT_TENSOR, sizesXMax, sizesXMin, DIM);

    if (std::strcmp(m_guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        if (m_permuteA)
        {
            synTensorSetPermutation(m_tensors[x], &activationPermutation);
        }
        if (m_permuteB)
        {
            synTensorSetPermutation(m_tensors[w], &weightPermutation);
        }
        if (m_allowPermuteOut)
        {
            synTensorSetAllowPermutation(m_tensors[y], true);
        }
        addNodeWithLayouts(NodeFactory::convolution3DNodeTypeName, {x, w}, {y}, &m_params, sizeof(m_params));
    }
    else if (std::strcmp(m_guid, NodeFactory::deDx3DNodeTypeName) == 0)
    {
        if (m_permuteA)
        {
            synTensorSetPermutation(m_tensors[y], &activationPermutation);
        }
        if (m_permuteB)
        {
            synTensorSetPermutation(m_tensors[w], &weightPermutation);
        }
        if (m_allowPermuteOut)
        {
            synTensorSetAllowPermutation(m_tensors[x], true);
        }

        addNodeWithLayouts(NodeFactory::deDx3DNodeTypeName, {y, w, shapeX}, {x}, &m_params, sizeof(m_params));
    }
    else  // if (std::strcmp(guid, NodeFactory::deDw3DNodeTypeName) == 0)
    {
        if (m_permuteA)
        {
            synTensorSetPermutation(m_tensors[y], &activationPermutation);
        }
        if (m_permuteB)
        {
            synTensorSetPermutation(m_tensors[x], &activationPermutation);
        }
        if (m_allowPermuteOut)
        {
            synTensorSetAllowPermutation(m_tensors[w], true);
        }
        addNodeWithLayouts(NodeFactory::deDw3DNodeTypeName, {y, x}, {w}, &m_params, sizeof(m_params));
    }

    compileTopology();

    if (m_dynamicShape)
    {
        ASSERT_NE(m_graphs[0].recipeHandle, nullptr);
        ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
        shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
        ASSERT_NE(recipe, nullptr);

        setActualSizes(shapeX, sizesXMin);
        setActualSizes(x, sizesXMin);
        setActualSizes(y, sizesYMin);
    }

    runTopology(0, false);

    validateResults(x, y, w);
}

// TODO fix these tests [SW-164319]
TEST_P_GC(SynGaudiDataLayoutMMETest, conv3d_test, {synDeviceGaudi, synDeviceGaudi2})
{
    synConvolution3DParams params;
    m_params          = synConvolution3DParams();
    m_batch           = 5;
    m_nIFM            = 16;
    m_nOFM            = 16;
    m_wIFMMax         = 6;
    m_hIFM            = 4;
    m_dIFM            = 4;
    m_dynamicShape    = std::get<0>(GetParam());
    m_wIFMMin         = m_dynamicShape ? 4 : m_wIFMMax;
    m_allowPermuteOut = std::get<1>(GetParam());
    m_permuteA        = std::get<2>(GetParam());
    m_permuteB        = std::get<3>(GetParam());

    ERepefenceOp op = std::get<4>(GetParam());
    if (op == REFERENCE_OP_FWD)
    {
        m_guid = NodeFactory::convolution3DNodeTypeName;
    }
    else if (op == REFERENCE_OP_DEDX)
    {
        m_guid = NodeFactory::deDx3DNodeTypeName;
    }
    else
    {
        m_guid = NodeFactory::deDw3DNodeTypeName;
    }

    run3DConvolution();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynGaudiDataLayoutMMETest,
    ::testing::Combine(::testing::ValuesIn({false, true}),  // dynamic shape
                       ::testing::ValuesIn({false, true}),  // allow output permutation
                       ::testing::ValuesIn({false, true}),  // permute A
                       ::testing::ValuesIn({false, true}),  // permute B
                       ::testing::ValuesIn({REFERENCE_OP_FWD, REFERENCE_OP_DEDX, REFERENCE_OP_DEDW})  // op
                       ));

class SynGaudiDataLayoutDynamicTest : public SynGaudiDataLayoutTest
{
};

TEST_F_GC(SynGaudiDataLayoutDynamicTest, consumed_dynamic_permuted_tensor_test)
{
    unsigned sizes[]            = {4, 5, 2, 3};
    unsigned maxSizes[]         = {4, 5, 10, 3};
    unsigned reshapedSizes[]    = {20, 6};
    unsigned reshapedMaxSizes[] = {20, 30};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned in         = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      maxSizes,
                                      4,
                                      syn_type_single,
                                      nullptr,
                                      "in",
                                      0,
                                      0,
                                      nullptr,
                                      sizes);
    unsigned reshapedIn = createTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       reshapedMaxSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       reshapedSizes);
    unsigned out        = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       reshapedMaxSizes,
                                       2,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       reshapedSizes);
    unsigned shape      = createShapeTensor(INPUT_TENSOR, reshapedMaxSizes, reshapedSizes, 2);
    setPermutation(in, perm);

    addNodeToGraph("reshape", {in, shape}, {reshapedIn}, nullptr, 0, "reshape");
    addNodeToGraph("relu_fwd_f32", {reshapedIn}, {out}, nullptr, 0, "relu");

    compileTopology();

    setActualSizes(in, sizes);
    setActualSizes(shape, reshapedSizes);
    setActualSizes(out, reshapedSizes);

    runTopology();

    float*       inData  = (float*)m_hostBuffers[in];
    const float* outData = (const float*)m_hostBuffers[out];

    // current sizes are in NCHW format.
    perm.permuteShape(sizes, 4);  // treat it as NHWC,
    // and transpose it "back" to NCHW
    transposeBuffer(sizes, 4, inData, perm.getInversePermutation());

    for (int i = 0; i < multiplyElements(sizes, sizes + 4); i++)
    {
        ASSERT_EQ(outData[i], relu(inData[i])) << "index " << i;
    }
}

TEST_F_GC(SynGaudiDataLayoutTest, produced_dynamic_permuted_tensor_test)
{
    TestSizeVec reluMaxSizes {20, 30};
    TestSizeVec reshapeMaxSizes {4, 5, 10, 3};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned in      = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      reluMaxSizes.data(),
                                      reluMaxSizes.size(),
                                      syn_type_single,
                                      nullptr,
                                      "in");
    unsigned reluOut = createTensor(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    reluMaxSizes.data(),
                                    reluMaxSizes.size(),
                                    syn_type_single);

    unsigned reshapeIn = reluOut;

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       reshapeMaxSizes.data(),
                                       reshapeMaxSizes.size(),
                                       syn_type_single,
                                       nullptr,
                                       "out");
    setPermutation(out, perm);
    addNodeToGraph("relu_fwd_f32", {in}, {reluOut}, nullptr, 0, "relu");
    addNodeToGraph("reshape", {reshapeIn}, {out}, nullptr, 0, "reshape");

    compileAndRun();

    const float* inData  = castHostBuffer<float>(in);
    float*       outData = castHostBuffer<float>(out);

    // transpose shape to NHWC
    perm.permuteShape(reshapeMaxSizes.data(), reshapeMaxSizes.size());
    transposeBuffer(reshapeMaxSizes.data(), reshapeMaxSizes.size(), outData, perm.getInversePermutation());

    const auto nElem = multiplyElements(reshapeMaxSizes.begin(), reshapeMaxSizes.end());
    for (int i = 0; i < nElem; i++)
    {
        ASSERT_EQ(outData[i], relu(inData[i])) << "index " << i;
    }
}

TEST_F_GC(SynGaudiDataLayoutTest, mult_add_repeated_permuted_input)
{
    TestSizeVec operandSizes {4, 5, 10, 3};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned in     = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      operandSizes.data(),
                                      operandSizes.size(),
                                      syn_type_single,
                                      nullptr,
                                      "in");
    unsigned addOut = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          operandSizes.data(),
                                          operandSizes.size(),
                                          syn_type_single,
                                          nullptr,
                                          "addOut");

    unsigned multOut = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           operandSizes.data(),
                                           operandSizes.size(),
                                           syn_type_single,
                                           nullptr,
                                           "multOut");

    setPermutation(in, perm);
    addNodeToGraph("mult_fwd_f32", {in, in}, {multOut}, nullptr, 0, "mult");
    addNodeToGraph("add_fwd_f32", {in, in}, {addOut}, nullptr, 0, "add");

    compileAndRun();

    float*       inData      = castHostBuffer<float>(in);
    const float* addOutData  = castHostBuffer<float>(addOut);
    const float* multOutData = castHostBuffer<float>(multOut);

    // transpose shape to NHWC
    perm.permuteShape(operandSizes.data(), operandSizes.size());
    transposeBuffer(operandSizes.data(), operandSizes.size(), inData, perm.getInversePermutation());

    const auto nElem = multiplyElements(operandSizes.begin(), operandSizes.end());
    for (int i = 0; i < nElem; i++)
    {
        ASSERT_LE(std::abs(multOutData[i] - pow(inData[i], 2)), 0.00001) << " mult index " << i;
        ASSERT_LE(std::abs(addOutData[i] - 2 * inData[i]), 0.00001) << " add index " << i;
    }
}

TEST_F_GC(SynGaudiDataLayoutTest, check_random_consistency)
{
    // disable shape manipulation of kernels that use LFSR random generation
    ScopedConfigurationChange config("ENABLE_LFSR_KERNEL_SHAPE_MANIPULATION", "false");

    TestSizeVec operandSizes {64, 4, 2, 3};
    TestSizeVec seedSizes {1};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned in     = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      operandSizes.data(),
                                      operandSizes.size(),
                                      syn_type_single,
                                      nullptr,
                                      "in");
    unsigned addOut = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          operandSizes.data(),
                                          operandSizes.size(),
                                          syn_type_single,
                                          nullptr,
                                          "addOut");

    unsigned random = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, operandSizes.data(), operandSizes.size());
    unsigned seed   = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_ALL_ONES,
                                        nullptr,
                                        seedSizes.data(),
                                        seedSizes.size(),
                                        syn_type_uint32,
                                        nullptr,
                                        "seed");
    unsigned randomOut = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             operandSizes.data(),
                                             operandSizes.size(),
                                             syn_type_single,
                                             nullptr,
                                             "randOut");

    ns_RandomNormal::Params params {0.f, 1.f, 42};
    setPermutation(in, perm);
    addNodeToGraph("random_normal_fwd_f32", {seed}, {random}, &params, sizeof(params), "random1");
    addNodeToGraph("random_normal_fwd_f32", {seed}, {randomOut}, &params, sizeof(params), "random2");
    addNodeToGraph("add_fwd_f32", {in, random}, {addOut}, nullptr, 0, "add");

    compileAndRun();

    float*       inData     = castHostBuffer<float>(in);
    float*       randomData = castHostBuffer<float>(randomOut);
    const float* addOutData = castHostBuffer<float>(addOut);

    // transpose shape to NHWC
    perm.permuteShape(operandSizes.data(), operandSizes.size());
    transposeBuffer(operandSizes.data(), operandSizes.size(), inData, perm.getInversePermutation());

    const auto nElem = multiplyElements(operandSizes.begin(), operandSizes.end());
    for (int i = 0; i < nElem; i++)
    {
        ASSERT_FLOAT_EQ(inData[i] + randomData[i], addOutData[i]);
    }
}

TEST_F_GC(SynGaudiDataLayoutDynamicTest, check_single_transpose_permutation)
{
    TestSizeVec outSizesMax {64, 20, 20, 3};
    TestSizeVec outSizesMin {64, 10, 10, 3};
    TestSizeVec inSizesMax {20, 20, 64, 3};
    TestSizeVec inSizesMin {10, 10, 64, 3};

    gc::Permutation perm = m_ptActivationPermutation4D;

    unsigned in = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      inSizesMax.data(),
                                      inSizesMax.size(),
                                      syn_type_single,
                                      nullptr,
                                      "in",
                                      0,
                                      0,
                                      nullptr,
                                      inSizesMin.data());

    unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       outSizesMax.data(),
                                       outSizesMax.size(),
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       outSizesMin.data());

    setPermutation(in, perm);

    synTransposeParamsNDims params;
    params.tensorDim = perm.size();
    for (unsigned i = 0; i < perm.size(); i++)
    {
        params.permutation[i] = perm.getValues()[i];
    }
    addNodeToGraph("transpose", {in}, {out}, &params, sizeof(params), "transpose");

    compileTopology();
    setActualSizes(in, inSizesMin);
    setActualSizes(out, outSizesMin);
    runTopology();

    float*       inData  = castHostBuffer<float>(in);
    const float* outData = castHostBuffer<float>(out);

    const auto nElem = multiplyElements(outSizesMin.begin(), outSizesMin.end());
    for (int i = 0; i < nElem; i++)
    {
        ASSERT_FLOAT_EQ(inData[i], outData[i]);
    }
}
