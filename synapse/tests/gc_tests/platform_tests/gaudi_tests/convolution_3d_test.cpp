#include "infra/cpu_calculator.h"
#include "scoped_configuration_change.h"
#include "utils.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"

class SynTrainingConv3DTest : public SynTrainingTestInfra
{
public:
    SynTrainingConv3DTest() { setTestPackage(TEST_PACKAGE_CONVOLUTION); }

protected:
    void run3DConvolution(const char*             guid,
                          synConvolution3DParams& params,
                          const unsigned          batch,
                          const unsigned          nIFM,
                          const unsigned          nOFM,
                          const unsigned          wIFM,
                          const unsigned          hIFM,
                          const unsigned          dIFM);
};

void SynTrainingConv3DTest::run3DConvolution(const char*             guid,
                                             synConvolution3DParams& params,
                                             const unsigned          batch,
                                             const unsigned          nIFM,
                                             const unsigned          nOFM,
                                             const unsigned          wIFM,
                                             const unsigned          hIFM,
                                             const unsigned          dIFM)
{
    const unsigned         wOFM = convOutputDimSize(wIFM,
                                            params.kernel[CONV_KERNEL_WIDTH],
                                            params.stride[CONV_STRIDE_WIDTH],
                                            params.padding[CONV_PAD_LEFT],
                                            params.dilation[CONV_DIL_WIDTH]);
    const unsigned         hOFM = convOutputDimSize(hIFM,
                                            params.kernel[CONV_KERNEL_HEIGHT],
                                            params.stride[CONV_STRIDE_HEIGHT],
                                            params.padding[CONV_PAD_TOP],
                                            params.dilation[CONV_DIL_HEIGHT]);
    const unsigned         dOFM = convOutputDimSize(dIFM,
                                            params.kernel[CONV_KERNEL_DEPTH],
                                            params.stride[CONV_STRIDE_DEPTH],
                                            params.padding[CONV_PAD_FRONT],
                                            params.dilation[CONV_DIL_DEPTH]);
    ConvQuantizationParams qParams;

    unsigned sizesX[] = {nIFM, wIFM, hIFM, dIFM, batch};
    unsigned x;
    unsigned sizesW[] = {nOFM,
                         nIFM,
                         params.kernel[CONV_KERNEL_WIDTH],
                         params.kernel[CONV_KERNEL_HEIGHT],
                         params.kernel[CONV_KERNEL_DEPTH]};
    unsigned w;
    unsigned sizesY[] = {nOFM, wOFM, hOFM, dOFM, batch};
    unsigned y;

    if (std::strcmp(guid, NodeFactory::deDx3DNodeTypeName) != 0)
    {
        x = createPersistTensor(INPUT_TENSOR,
                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                nullptr,  // initializer
                                sizesX,
                                CONV_3D_TENSOR_DIM,
                                syn_type_float);
    }
    else
    {
        x = createPersistTensor(OUTPUT_TENSOR,
                                MEM_INIT_ALL_ZERO,
                                nullptr,  // initializer
                                sizesX,
                                CONV_3D_TENSOR_DIM,
                                syn_type_float);
    }

    if (std::strcmp(guid, NodeFactory::deDw3DNodeTypeName) != 0)
    {
        w = createPersistTensor(INPUT_TENSOR,
                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                nullptr,  // initializer
                                sizesW,
                                CONV_3D_TENSOR_DIM,
                                syn_type_float);
    }
    else
    {
        w = createPersistTensor(OUTPUT_TENSOR,
                                MEM_INIT_ALL_ZERO,
                                nullptr,  // initializer
                                sizesW,
                                CONV_3D_TENSOR_DIM,
                                syn_type_float);
    }

    if (std::strcmp(guid, NodeFactory::convolution3DNodeTypeName) != 0)
    {
        y = createPersistTensor(INPUT_TENSOR,
                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                nullptr,  // initializer
                                sizesY,
                                CONV_3D_TENSOR_DIM,
                                syn_type_float);
    }
    else
    {
        y = createPersistTensor(OUTPUT_TENSOR,
                                MEM_INIT_ALL_ZERO,
                                nullptr,  // initializer
                                sizesY,
                                CONV_3D_TENSOR_DIM,
                                syn_type_float);
    }

    if (std::strcmp(guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        addNodeToGraph(NodeFactory::convolution3DNodeTypeName, {x, w}, {y}, &params, sizeof(params));
    }
    else if (std::strcmp(guid, NodeFactory::deDx3DNodeTypeName) == 0)
    {
        addNodeToGraph(NodeFactory::deDx3DNodeTypeName, {y, w}, {x}, &params, sizeof(params));
    }
    else  // if (std::strcmp(guid, NodeFactory::deDw3DNodeTypeName) == 0)
    {
        addNodeToGraph(NodeFactory::deDw3DNodeTypeName, {y, x}, {w}, &params, sizeof(params));
    }

    compileAndRun();

    synTensorDescriptor xDesc = m_tensorDescs[x];
    synTensorDescriptor wDesc = m_tensorDescs[w];
    synTensorDescriptor yDesc = m_tensorDescs[y];

    char* xData = (char*)m_hostBuffers[x];
    char* wData = (char*)m_hostBuffers[w];
    char* yData = (char*)m_hostBuffers[y];

    CoordArray wrongIdx;
    float      expectedResult = 0;

    ERepefenceOp op;
    if (std::strcmp(guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        op = REFERENCE_OP_FWD;
    }
    else if (std::strcmp(guid, NodeFactory::deDx3DNodeTypeName) == 0)
    {
        op = REFERENCE_OP_DEDX;
    }
    else  // if (std::strcmp(guid, NodeFactory::deDw3DNodeTypeName) == 0)
    {
        op = REFERENCE_OP_DEDW;
    }

    bool ret =
        checkMmeOp(xDesc, xData, wDesc, wData, yDesc, yData, params, op, wrongIdx, m_deviceType, &expectedResult);

    if (std::strcmp(guid, NodeFactory::convolution3DNodeTypeName) == 0)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, m_tensorDescs[y].m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_EQ(ret, true)
            << "Wrong value for CONV op at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',') << " Got value: "
            << getIndexValue(sizes, wrongIdx, m_tensorDescs[y].m_dataType, m_hostBuffers[y])
            << " Expected: " << expectedResult;
    }
    else if (std::strcmp(guid, NodeFactory::deDx3DNodeTypeName) == 0)
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

TEST_F_GC(SynTrainingConv3DTest, conv3d_2x2x2_Input_and_kernel_test)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 2;
    params.kernel[CONV_KERNEL_HEIGHT] = 2;
    params.kernel[CONV_KERNEL_DEPTH]  = 2;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 1, 1, 1, 2, 2, 2);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_2x2x2_Input_and_kernel_with_memcpy_test)
{
    const unsigned kW1    = 2;
    const unsigned kH1    = 2;
    const unsigned kD1    = 2;
    const unsigned dW1    = 1;
    const unsigned dH1    = 1;
    const unsigned dD1    = 1;
    const unsigned batch1 = 1;
    const unsigned nOFM1  = 1;
    const unsigned nIFM1  = 1;
    const unsigned wIFM1  = 2;
    const unsigned hIFM1  = 2;
    const unsigned dIFM1  = 2;

    const unsigned wOFM1 = wIFM1 - kW1 + 1;
    const unsigned hOFM1 = hIFM1 - kH1 + 1;
    const unsigned dOFM1 = dIFM1 - kD1 + 1;

    synConvolution3DParams params1;
    params1.stride[CONV_STRIDE_HEIGHT] = dH1;
    params1.stride[CONV_STRIDE_WIDTH]  = dW1;
    params1.stride[CONV_STRIDE_DEPTH]  = dD1;
    params1.kernel[CONV_KERNEL_HEIGHT] = kH1;
    params1.kernel[CONV_KERNEL_WIDTH]  = kW1;
    params1.kernel[CONV_KERNEL_DEPTH]  = kD1;

    unsigned       sizesIn[]    = {nIFM1, wIFM1, hIFM1, dIFM1, batch1};
    const uint64_t tensorSizeIn = getNumberOfElements(sizesIn, CONV_3D_TENSOR_DIM);
    float*         ifm1         = new float[tensorSizeIn];

    unsigned       sizesW[]    = {nOFM1, nIFM1, kW1, kH1, kD1};
    const uint64_t tensorSizeW = getNumberOfElements(sizesW, CONV_3D_TENSOR_DIM);
    float*         weights1    = new float[tensorSizeW];

    unsigned       outSizes[]    = {nOFM1, wOFM1, hOFM1, dOFM1, batch1};
    const uint64_t tensorSizeOut = getNumberOfElements(outSizes, CONV_3D_TENSOR_DIM);

    for (uint64_t i = 0; i < tensorSizeIn; i++)
    {
        ifm1[i] = i + 1;
    }
    for (uint64_t i = 0; i < tensorSizeW; i++)
    {
        weights1[i] = 1;
    }

    float* outRef1 = new float[tensorSizeOut];
    std::memset(outRef1, 0, tensorSizeOut * sizeof(float));

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                ifm1,
                                                sizesIn,
                                                CONV_3D_TENSOR_DIM,
                                                syn_type_single);
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                weights1,
                                                sizesW,
                                                CONV_3D_TENSOR_DIM,
                                                syn_type_single);
    unsigned yTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes, CONV_3D_TENSOR_DIM, syn_type_single);

    synTensorDescriptor xDesc = m_tensorDescs[xTensorIndex];
    synTensorDescriptor wDesc = m_tensorDescs[wTensorIndex];
    synTensorDescriptor yDesc = m_tensorDescs[yTensorIndex];

    auto xData = m_hostBuffers[xTensorIndex];
    auto wData = m_hostBuffers[wTensorIndex];

    calculateFwdConvolution(xDesc, (char*)xData, wDesc, (char*)wData, yDesc, (char*)outRef1, params1, m_deviceType);

    ASSERT_EQ(tensorSizeOut, 1);
    for (uint64_t i = 0; i < tensorSizeOut; i++)
    {
        ASSERT_EQ(outRef1[i], 36) << "Failed runOnCPU 3D conv with results " << outRef1 << " expected results 36";
    }

    addNodeToGraph(NodeFactory::convolution3DNodeTypeName,
                   {xTensorIndex, wTensorIndex},
                   {yTensorIndex},
                   (void*)&params1,
                   sizeof(synConvolution3DParams));

    // memcpy node
    unsigned mInputTensorIndex = connectOutputTensorToInputTensor(yTensorIndex);
    unsigned mOutputTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes, CONV_3D_TENSOR_DIM, syn_type_single);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName,
                   {mInputTensorIndex},
                   {mOutputTensorIndex},
                   (void*)&params1,
                   sizeof(synConvolution3DParams));

    compileAndRun();

    // test result
    float* out1 = (float*)m_hostBuffers[mOutputTensorIndex];
    for (uint64_t i = 0; i < tensorSizeOut; i++)
    {
        ASSERT_EQ(out1[i], outRef1[i]) << "Mismatch at index " << i << " Result:" << out1[i] << " Ref: " << outRef1[i];
    }

    delete[] ifm1;
    delete[] weights1;
    delete[] outRef1;
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_2x2x2_Input_and_kernel_test_2_channels)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 2;
    params.kernel[CONV_KERNEL_HEIGHT] = 2;
    params.kernel[CONV_KERNEL_DEPTH]  = 2;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 1, 1, 2, 2, 2, 2);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_2x2x2_Input_and_kernel_test_2_batches)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 2;
    params.kernel[CONV_KERNEL_HEIGHT] = 2;
    params.kernel[CONV_KERNEL_DEPTH]  = 2;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 2, 1, 1, 2, 2, 2);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_4x4x4_Input_2x2x2_kernel)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 2;
    params.kernel[CONV_KERNEL_HEIGHT] = 2;
    params.kernel[CONV_KERNEL_DEPTH]  = 2;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 1, 1, 1, 4, 4, 4);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_600_600_batch_2_L2)
{
    synConvolution3DParams params;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 2, 600, 600, 4, 4, 4);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_group_conv_float)
{
    const unsigned kW1    = 1;
    const unsigned kH1    = 1;
    const unsigned kD1    = 1;
    const unsigned dW1    = 1;
    const unsigned dH1    = 1;
    const unsigned dD1    = 1;
    const unsigned batch1 = 1;
    const unsigned nIFM1  = 4;
    const unsigned nOFM1  = 6;
    const unsigned wIFM1  = 2;
    const unsigned hIFM1  = 2;
    const unsigned dIFM1  = 2;

    const unsigned groups = 2;

    const unsigned wOFM1 = convOutputDimSize(wIFM1, kW1, dW1, 0, 1);
    const unsigned hOFM1 = convOutputDimSize(hIFM1, kH1, dH1, 0, 1);
    const unsigned dOFM1 = convOutputDimSize(dIFM1, kD1, dD1, 0, 1);

    synConvolution3DParams params1;
    params1.stride[CONV_STRIDE_HEIGHT] = dH1;
    params1.stride[CONV_STRIDE_WIDTH]  = dW1;
    params1.stride[CONV_STRIDE_DEPTH]  = dD1;
    params1.kernel[CONV_KERNEL_HEIGHT] = kH1;
    params1.kernel[CONV_KERNEL_WIDTH]  = kW1;
    params1.kernel[CONV_KERNEL_DEPTH]  = kD1;
    params1.nGroups                    = groups;

    unsigned       sizesIn[]    = {nIFM1, wIFM1, hIFM1, dIFM1, batch1};
    const uint64_t tensorSizeIn = getNumberOfElements(sizesIn, CONV_3D_TENSOR_DIM);
    float*         ifm1         = new float[tensorSizeIn];

    unsigned       sizesW[]    = {nOFM1, nIFM1 / groups, kW1, kH1, kD1};
    const uint64_t tensorSizeW = getNumberOfElements(sizesW, CONV_3D_TENSOR_DIM);
    float*         weights1    = new float[tensorSizeW];

    unsigned       outSizes[]    = {nOFM1, wOFM1, hOFM1, dOFM1, batch1};
    const uint64_t tensorSizeOut = getNumberOfElements(outSizes, CONV_3D_TENSOR_DIM);

    for (uint64_t i = 0; i < tensorSizeIn; i++)
    {
        ifm1[i] = i;
    }
    for (uint64_t i = 0; i < tensorSizeW; i++)
    {
        int val     = rand();
        weights1[i] = ((val % 5) - 2);
    }

    float* outRef1 = new float[tensorSizeOut / groups];
    std::memset(outRef1, 0, tensorSizeOut * sizeof(float) / groups);
    float* outRef2 = new float[tensorSizeOut / groups];
    std::memset(outRef2, 0, tensorSizeOut * sizeof(float) / groups);

    // new ifm for Ref group conv
    float*   ifmRef1 = new float[tensorSizeIn / groups];
    float*   ifmRef2 = new float[tensorSizeIn / groups];
    unsigned itr1    = 0;
    unsigned itr2    = 0;

    for (uint64_t spatialIdx = 0; spatialIdx < tensorSizeIn; ++spatialIdx)
    {
        if (spatialIdx % (nIFM1) < (nIFM1 / groups))
        {
            ifmRef1[itr1] = ifm1[spatialIdx];
            itr1++;
        }
        else
        {
            ifmRef2[itr2] = ifm1[spatialIdx];
            itr2++;
        }
    }

    // new weights for Ref group conv
    itr1 = itr2        = 0;
    float* weightsRef1 = new float[tensorSizeW / groups];
    float* weightsRef2 = new float[tensorSizeW / groups];

    for (uint64_t spatialIdx = 0; spatialIdx < tensorSizeW; ++spatialIdx)
    {
        if (spatialIdx % (nOFM1) < (nOFM1 / groups))
        {
            weightsRef1[itr1] = weights1[spatialIdx];
            itr1++;
        }
        else
        {
            weightsRef2[itr2] = weights1[spatialIdx];
            itr2++;
        }
    }

    unsigned sizesInRef[]  = {nIFM1 / groups, wIFM1, hIFM1, dIFM1, batch1};
    unsigned sizesWRef[]   = {nOFM1 / groups, nIFM1 / groups, kW1, kH1, kD1};
    unsigned sizesOutRef[] = {nOFM1 / groups, wOFM1, hOFM1, dOFM1, batch1};
    unsigned ifmRefIdx1    = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_FROM_INITIALIZER,
                                              ifmRef1,
                                              sizesInRef,
                                              CONV_3D_TENSOR_DIM,
                                              syn_type_single);

    unsigned ifmRefIdx2     = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_FROM_INITIALIZER,
                                              ifmRef2,
                                              sizesInRef,
                                              CONV_3D_TENSOR_DIM,
                                              syn_type_single);
    unsigned weightsRefIdx1 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_FROM_INITIALIZER,
                                                  weightsRef1,
                                                  sizesWRef,
                                                  CONV_3D_TENSOR_DIM,
                                                  syn_type_single);
    unsigned weightsRefIdx2 = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_FROM_INITIALIZER,
                                                  weightsRef2,
                                                  sizesWRef,
                                                  CONV_3D_TENSOR_DIM,
                                                  syn_type_single);
    unsigned outRefIdx1     = createPersistTensor(OUTPUT_TENSOR,
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              sizesOutRef,
                                              CONV_3D_TENSOR_DIM,
                                              syn_type_single);

    synTensorDescriptor ifmRefDesc = m_tensorDescs[ifmRefIdx1];
    synTensorDescriptor wRefDesc   = m_tensorDescs[weightsRefIdx1];
    synTensorDescriptor outDesc    = m_tensorDescs[outRefIdx1];

    auto ifmRefData1 = m_hostBuffers[ifmRefIdx1];
    auto ifmRefData2 = m_hostBuffers[ifmRefIdx2];
    auto wRefData1   = m_hostBuffers[weightsRefIdx1];
    auto wRefData2   = m_hostBuffers[weightsRefIdx2];

    calculateFwdConvolution(ifmRefDesc,
                            (char*)ifmRefData1,
                            wRefDesc,
                            (char*)wRefData1,
                            outDesc,
                            (char*)outRef1,
                            params1,
                            m_deviceType);
    calculateFwdConvolution(ifmRefDesc,
                            (char*)ifmRefData2,
                            wRefDesc,
                            (char*)wRefData2,
                            outDesc,
                            (char*)outRef2,
                            params1,
                            m_deviceType);

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                ifm1,
                                                sizesIn,
                                                CONV_3D_TENSOR_DIM,
                                                syn_type_single);
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_FROM_INITIALIZER,
                                                weights1,
                                                sizesW,
                                                CONV_3D_TENSOR_DIM,
                                                syn_type_single);
    unsigned yTensorIndex =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes, CONV_3D_TENSOR_DIM, syn_type_single);

    addNodeToGraph(NodeFactory::convolution3DNodeTypeName,
                   {xTensorIndex, wTensorIndex},
                   {yTensorIndex},
                   (void*)&params1,
                   sizeof(synConvolution3DParams));

    compileAndRun();

    // test result
    float results[tensorSizeOut];

    for (uint64_t i = 0; i < tensorSizeOut; i++)
    {
        results[i] = ((float*)m_hostBuffers[yTensorIndex])[i];
    }

    itr1 = itr2 = 0;
    for (uint64_t i = 0; i < tensorSizeOut; i++)
    {
        if (i % (nOFM1) < (nOFM1 / 2))
        {
            ASSERT_EQ(results[i], outRef1[itr1])
                << "Failed 3D conv, chip results " << results[i] << ",expected results " << outRef1[i];
            itr1++;
        }
        else
        {
            ASSERT_EQ(results[i], outRef2[itr2])
                << "Failed 3D conv, chip results " << results[i] << ",expected results " << outRef2[i];
            itr2++;
        }
    }

    delete[] ifm1;
    delete[] ifmRef1;
    delete[] ifmRef2;
    delete[] weights1;
    delete[] weightsRef1;
    delete[] weightsRef2;
    delete[] outRef1;
    delete[] outRef2;
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_256_256_batch_10_L2)
{
    synConvolution3DParams params;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 10, 256, 256, 4, 4, 4);
}

// workloads,framework,operator,op_data_mode,output,inputs,weights,stride,padding,dilation,output_padding,transposed,groups
// ['custom:2'],Cudnn,convolution,31,"[1,19,32,32,32]","[1,19,32,32,32]","[1,19,3,3,3]","[1,1,1]","[0,0,0]","[1,1,1]","[0,0,0]",false,1
TEST_F_GC(SynTrainingConv3DTest, conv3d_19_32_32_32_1)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 1, 19, 1, 32, 32, 32);
}

// 'output': [4, 32, 128, 128, 128], 'inputs': [4, 64, 128, 128, 128]
// 'weights': [32, 64, 3, 3, 3], 'stride': [1, 1, 1], 'padding': [1, 1, 1], 'dilation': [1, 1, 1]
TEST_F_GC(SynTrainingConv3DTest, conv3d_64_128_128_128_4)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 4, 64, 32, 3, 3, 3);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_strided_L2)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    params.stride[CONV_STRIDE_HEIGHT] = 2;
    params.stride[CONV_STRIDE_WIDTH]  = 2;
    params.stride[CONV_STRIDE_DEPTH]  = 2;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 4, 64, 32, 3, 3, 3);
}

TEST_F_GC(SynTrainingConv3DTest, conv3d_padded_small)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 2;
    params.kernel[CONV_KERNEL_HEIGHT] = 2;
    params.kernel[CONV_KERNEL_DEPTH]  = 2;
    params.padding[CONV_PAD_LEFT]     = 0;
    params.padding[CONV_PAD_TOP]      = 0;
    params.padding[CONV_PAD_FRONT]    = 1;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 1, 1, 1, 2, 2, 2);
}

// output,inputs,weights,stride,padding,dilation
// "[4,128,32,32,32]","[4,64,64,64,64]","[128,64,3,3,3]","[2,2,2]","[1,1,1]","[1,1,1]"
TEST_F_GC(SynTrainingConv3DTest, conv3d_padded_L2)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    params.padding[CONV_PAD_LEFT]     = 1;
    params.padding[CONV_PAD_TOP]      = 1;
    params.padding[CONV_PAD_FRONT]    = 1;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 4, 64, 32, 3, 3, 3);
}

// output,inputs,weights,stride,padding,dilation
// "[4,128,32,32,32]","[4,64,64,64,64]","[128,64,3,3,3]","[2,2,2]","[1,1,1]","[1,1,1]"
TEST_F_GC(SynTrainingConv3DTest, conv3d_strided_padded_L2)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    params.stride[CONV_STRIDE_HEIGHT] = 2;
    params.stride[CONV_STRIDE_WIDTH]  = 2;
    params.stride[CONV_STRIDE_DEPTH]  = 2;
    params.padding[CONV_PAD_LEFT]     = 1;
    params.padding[CONV_PAD_TOP]      = 1;
    params.padding[CONV_PAD_FRONT]    = 1;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 4, 64, 32, 3, 3, 3);
}

// output,inputs,weights,stride,padding,dilation
// "[4,128,32,32,32]","[4,64,64,64,64]","[128,64,3,3,3]","[2,2,2]","[1,1,1]","[2,2,2]"
TEST_F_GC(SynTrainingConv3DTest, conv3d_strided_dilated_padded_L2)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    params.stride[CONV_STRIDE_WIDTH]  = 2;
    params.stride[CONV_STRIDE_HEIGHT] = 2;
    params.stride[CONV_STRIDE_DEPTH]  = 2;
    params.padding[CONV_PAD_LEFT]     = 1;
    params.padding[CONV_PAD_TOP]      = 1;
    params.padding[CONV_PAD_FRONT]    = 1;
    params.dilation[CONV_DIL_WIDTH]   = 2;
    params.dilation[CONV_DIL_HEIGHT]  = 2;
    params.dilation[CONV_DIL_DEPTH]   = 2;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 4, 64, 32, 10, 10, 10);
}

// output,inputs,weights,stride,padding,dilation
// [4,32,128,128,128]","[4,64,128,128,128]","[32,64,3,3,3]","[1,1,1]","[1,1,1]","[1,1,1]","[0,0,0]"
TEST_F_GC(SynTrainingConv3DTest, DISABLED_conv3d_big_ASIC_CI)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    params.padding[CONV_PAD_LEFT]     = 1;
    params.padding[CONV_PAD_TOP]      = 1;
    params.padding[CONV_PAD_FRONT]    = 1;
    run3DConvolution(NodeFactory::convolution3DNodeTypeName, params, 4, 64, 32, 128, 128, 128);
}

TEST_F_GC(SynTrainingConv3DTest, dedx_test_small)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    run3DConvolution(NodeFactory::deDx3DNodeTypeName, params, 1, 1, 1, 3, 3, 3);
}

TEST_F_GC(SynTrainingConv3DTest, dedx_test_L2)
{
    GlobalConfTestSetter gSet("SRAM_SLICER_MAX_CAPACITY_BYTES", "5000000");

    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    run3DConvolution(NodeFactory::deDx3DNodeTypeName, params, 64, 32, 64, 7, 7, 7);
}

TEST_F_GC(SynTrainingConv3DTest, dedw_test_small)
{
    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    run3DConvolution(NodeFactory::deDw3DNodeTypeName, params, 1, 1, 1, 3, 3, 3);
}

TEST_F_GC(SynTrainingConv3DTest, dedw_test_L2)
{
    GlobalConfTestSetter gSet("SRAM_SLICER_MAX_CAPACITY_BYTES", "5000000");
    // TODO: remove once cd parallel implementation is done
    GlobalConfTestSetter gSetCdParallel("GCFG_ENABLE_CD_PARALLEL", "false");

    synConvolution3DParams params;
    params.kernel[CONV_KERNEL_WIDTH]  = 3;
    params.kernel[CONV_KERNEL_HEIGHT] = 3;
    params.kernel[CONV_KERNEL_DEPTH]  = 3;
    run3DConvolution(NodeFactory::deDw3DNodeTypeName, params, 64, 32, 64, 7, 7, 7);
}
