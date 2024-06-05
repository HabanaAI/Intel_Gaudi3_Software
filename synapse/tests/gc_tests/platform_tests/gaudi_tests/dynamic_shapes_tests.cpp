#include "gc_dynamic_shapes_infra.h"

class SynGaudiDynamicShapesTests : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicShapesTests, create_static_graph)
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

    const unsigned batch = 128;
    const unsigned nIFM  = 128;
    const unsigned nOFM  = 128;
    const unsigned wOFM  = 4;
    const unsigned hOFM  = 4;

    const unsigned wIFM = convInputDimSize(wOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hIFM = convInputDimSize(hOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmDimSizes[] = { nIFM, wIFM, hIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmDimSizes[] = { nOFM, wOFM, hOFM, batch };

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, ifmDimSizes, dims, syn_type_single);
    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, wghDimSizes, dims, syn_type_single);
    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, ofmDimSizes, dims, syn_type_single);

    TensorIndices inputIndices = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params,
                   sizeof(synConvolutionParams));

    for (auto tensorIndex : inputIndices)
    {
        TestStaticTensor(reinterpret_cast<Tensor*>(getTensorByIndex(tensorIndex)));
    }

    for (auto tensorIndex : outputIndices)
    {
        TestStaticTensor(reinterpret_cast<Tensor*>(getTensorByIndex(tensorIndex)));
    }

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    ASSERT_EQ(m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe, nullptr);

    runTopology();
}

TEST_F_GC(SynGaudiDynamicShapesTests, create_dynamic_graph)
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
    const unsigned nIFM  = 128;
    const unsigned nOFM  = 128;
    const unsigned wMaxOFM  = 128;
    const unsigned hMaxOFM  = 128;
    const unsigned wMinOFM  = 64;
    const unsigned hMinOFM  = 64;

    const unsigned wMaxIFM = convInputDimSize(wMaxOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hMaxIFM = convInputDimSize(hMaxOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    const unsigned wMinIFM = convInputDimSize(wMinOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hMinIFM = convInputDimSize(hMinOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    // create_tensor's layout
    unsigned dims = 4;
    unsigned ifmMaxDimSizes[] = { nIFM, wMaxIFM, hMaxIFM, batch };
    unsigned ifmMinDimSizes[] = { nIFM, wMinIFM, hMinIFM, batch };
    unsigned wghDimSizes[] = { nOFM, nIFM, params.kW, params.kH };
    unsigned ofmMaxDimSizes[] = { nOFM, wMaxOFM, hMaxOFM, batch };
    unsigned ofmMinDimSizes[] = { nOFM, wMinOFM, hMinOFM, batch };


    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr,
                                                ifmMaxDimSizes, dims, syn_type_single, nullptr,
                                                nullptr, 0, 0, nullptr, ifmMinDimSizes);

    unsigned wTensorIndex = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr,
                                                wghDimSizes, dims, syn_type_single);

    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr,
                                                ofmMaxDimSizes, dims, syn_type_single, nullptr,
                                                nullptr, 0, 0, nullptr, ofmMinDimSizes);

    TensorIndices inputIndices = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                   inputIndices,
                   outputIndices,
                   (void*)&params,
                   sizeof(synConvolutionParams));

    TestStaticTensor(reinterpret_cast<Tensor*>(getTensorByIndex(wTensorIndex)));

    Tensor* inTensor = reinterpret_cast<Tensor*>(getTensorByIndex(xTensorIndex));

    ASSERT_TRUE(inTensor->isDynamicShape());
    ASSERT_EQ(inTensor->getMinimalElements(), nIFM * wMinIFM * hMinIFM * batch);
    ASSERT_EQ(inTensor->getMinimalSizeInBytes(), nIFM * wMinIFM * hMinIFM * batch * sizeof(float));

    ASSERT_EQ(inTensor->getMinimalSizeInBytes(), nIFM * wMinIFM * hMinIFM * batch * sizeof(float));
    ASSERT_EQ(inTensor->getMinimalSizeInBytes(), nIFM * wMinIFM * hMinIFM * batch * sizeof(float));

    {
        SizeArray minSizesArray = inTensor->getAllMinimalSizesInElements();
        SizeArray minSizes;
        inTensor->getAllMinimalSizesInElements(minSizes);

        SizeArray sizesArray = inTensor->getAllSizesInElements();
        SizeArray sizes;
        inTensor->getAllSizesInElements(sizes);

        for (int i = 0; i < inTensor->getDim(); i++)
        {
            ASSERT_EQ(minSizes[i], ifmMinDimSizes[i]);
            ASSERT_EQ(minSizesArray[i], ifmMinDimSizes[i]);
            ASSERT_EQ(inTensor->getMinimalSizeInBytes(i), ifmMinDimSizes[i] * sizeof(float));
            ASSERT_EQ(inTensor->getMinimalSizeInElements(i), ifmMinDimSizes[i]);

            ASSERT_EQ(sizes[i], ifmMaxDimSizes[i]);
            ASSERT_EQ(sizesArray[i], ifmMaxDimSizes[i]);
            ASSERT_EQ(inTensor->getSizeInBytes(i), ifmMaxDimSizes[i] * sizeof(float));
            ASSERT_EQ(inTensor->getSizeInElements(i), ifmMaxDimSizes[i]);
        }
    }


    Tensor* outTensor = reinterpret_cast<Tensor*>(getTensorByIndex(yTensorIndex));

    ASSERT_TRUE(outTensor->isDynamicShape());
    ASSERT_EQ(outTensor->getMinimalElements(), nOFM * wMinOFM * hMinOFM * batch);
    ASSERT_EQ(outTensor->getMinimalSizeInBytes(), nOFM * wMinOFM * hMinOFM * batch * sizeof(float));

    ASSERT_EQ(outTensor->getMinimalSizeInBytes(), nOFM * wMinOFM * hMinOFM * batch * sizeof(float));
    ASSERT_EQ(outTensor->getMinimalSizeInBytes(), nOFM * wMinOFM * hMinOFM * batch * sizeof(float));

    {
        SizeArray minSizesArray = outTensor->getAllMinimalSizesInElements();
        SizeArray minSizes;
        outTensor->getAllMinimalSizesInElements(minSizes);

        SizeArray sizesArray = outTensor->getAllSizesInElements();
        SizeArray sizes;
        outTensor->getAllSizesInElements(sizes);

        for (int i = 0; i < outTensor->getDim(); i++)
        {
            ASSERT_EQ(minSizes[i], ofmMinDimSizes[i]);
            ASSERT_EQ(minSizesArray[i], ofmMinDimSizes[i]);
            ASSERT_EQ(outTensor->getMinimalSizeInBytes(i), ofmMinDimSizes[i] * sizeof(float));
            ASSERT_EQ(outTensor->getMinimalSizeInElements(i), ofmMinDimSizes[i]);

            ASSERT_EQ(sizes[i], ofmMaxDimSizes[i]);
            ASSERT_EQ(sizesArray[i], ofmMaxDimSizes[i]);
            ASSERT_EQ(outTensor->getSizeInBytes(i), ofmMaxDimSizes[i] * sizeof(float));
            ASSERT_EQ(outTensor->getSizeInElements(i), ofmMaxDimSizes[i]);
        }
    }

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);
}

TEST_F_GC(SynGaudiDynamicShapesTests, create_dynamic_graph_with_high_rank)
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

    const unsigned batch   = 1;
    const unsigned nIFM    = 128;
    const unsigned nOFM    = 128;
    const unsigned wMaxOFM = 128;
    const unsigned hMaxOFM = 128;
    const unsigned wMinOFM = 64;
    const unsigned hMinOFM = 64;

    const unsigned wMaxIFM = convInputDimSize(wMaxOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hMaxIFM = convInputDimSize(hMaxOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    const unsigned wMinIFM = convInputDimSize(wMinOFM, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned hMinIFM = convInputDimSize(hMinOFM, params.kH, params.dH, params.padT + params.padB, params.dilH);

    // create_tensor's layout
    unsigned dims             = 6;
    unsigned ifmMaxDimSizes[] = {nIFM, wMaxIFM, hMaxIFM, batch, 1, 1};
    unsigned ifmMinDimSizes[] = {nIFM, wMinIFM, hMinIFM, batch, 1, 1};
    unsigned wghDimSizes[]    = {nOFM, nIFM, params.kW, params.kH, 1, 1};
    unsigned ofmMaxDimSizes[] = {nOFM, wMaxOFM, hMaxOFM, batch, 1, 1};
    unsigned ofmMinDimSizes[] = {nOFM, wMinOFM, hMinOFM, batch, 1, 1};

    unsigned xTensorIndex = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_POSITIVE,
                                                nullptr,
                                                ifmMaxDimSizes,
                                                dims,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                ifmMinDimSizes);

    unsigned wTensorIndex =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, wghDimSizes, dims, syn_type_single);

    unsigned yTensorIndex = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_RANDOM_POSITIVE,
                                                nullptr,
                                                ofmMaxDimSizes,
                                                dims,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                ofmMinDimSizes);

    TensorIndices inputIndices  = {xTensorIndex, wTensorIndex};
    TensorIndices outputIndices = {yTensorIndex};

    std::vector<synTensor> inTensors;
    std::vector<synTensor> outTensors;
    // put all tensors in continuous array
    for (int i = 0; i < inputIndices.size(); i++)
    {
        assert(i < m_maxNumTensors);
        if (inputIndices[i] == INVALID_TENSOR_INDEX)
        {
            // support nullptr tensors
            inTensors.push_back(nullptr);
        }
        else
        {
            inTensors.push_back(m_tensors[inputIndices[i]]);
        }
    }
    for (auto i : outputIndices)
    {
        assert(i < m_maxNumTensors);
        outTensors.push_back(m_tensors[i]);
    }

    synGraphHandle graphHandle = getGraph(0).graphHandle;

    ASSERT_EQ(synInvalidTensorDimensions,
              synNodeCreate(graphHandle,
                            inTensors.data(),
                            outTensors.data(),
                            inTensors.size(),
                            outTensors.size(),
                            (void*)&params,
                            sizeof(synConvolutionParams),
                            NodeFactory::convolutionNodeTypeName,
                            nullptr,
                            nullptr,
                            nullptr))
        << "Failed to disallow DSD together with high rank";
}

TEST_F_GC(SynGaudiDynamicShapesTests, create_zero_tensor)
{
    unsigned maxSizes[] = {10, 10};
    unsigned minSizes[] = {0, 0};
    auto inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                        syn_type_float, nullptr, nullptr, 0, 0,
                                        nullptr, minSizes);

    auto outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 2,
                                        syn_type_float, nullptr, nullptr, 0, 0,
                                        nullptr, minSizes);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {outTensor});
    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, minSizes);
    setActualSizes(outTensor, minSizes);

    runTopology(0, true);
    float* data = castHostOutBuffer<float>(outTensor);
    for (int i = 0; i < maxSizes[0] * maxSizes[1]; i++)
    {
        ASSERT_EQ(data[i], 0);
    }
}

TEST_F_GC(SynGaudiDynamicShapesTests, create_shape_tensor)
{
    unsigned sizes[] = { 16, 2, 10, 5};
    unsigned minSizes[] = { 4, 2, 10, 5};
    unsigned dims = sizeof(sizes)/sizeof(sizes[0]);

    auto shapeTensor = createShapeTensor(INPUT_TENSOR, sizes, minSizes, dims);

    auto dynamicOutTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO,
            nullptr, sizes, dims, syn_type_single, nullptr,
            nullptr, 0, 0, nullptr, minSizes);

    addNodeToGraph(NodeFactory::memsetNodeTypeName, {shapeTensor}, {dynamicOutTensor});

    setActualSizes(shapeTensor, sizes);
    setActualSizes(dynamicOutTensor, sizes);

    compileTopology();
    // no need to runTopologyDSD as we are not varying tensor sizes here
    runTopology();
}

class SynGaudiDynamicShapesTestsShapeTensor : public SynGaudiDynamicShapesTests,
                                              public testing::WithParamInterface<unsigned>
{
public:
    static constexpr size_t minDimSize = 1;
    static constexpr size_t maxDimSize = 7;
};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicShapesTestsShapeTensor, ::testing::Values(1, 3, 7));

TEST_P_GC(SynGaudiDynamicShapesTestsShapeTensor, broadcast_with_shape_tensor_with_rt)
{
    const unsigned actualDimSize = GetParam();
    const unsigned tensorDim = 2;

    unsigned inMaxSize[]      = {maxDimSize, 1};
    unsigned inMinSize[]      = {minDimSize, 1};
    unsigned inActualShape[]  = {actualDimSize, 1};
    unsigned outMaxSize[]     = {maxDimSize, 3};
    unsigned outMinSize[]     = {minDimSize, 3};
    unsigned outActualShape[] = {actualDimSize, 3};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSize,
                                            tensorDim,
                                            syn_type_single,
                                            nullptr,
                                            "inTensor",
                                            0,
                                            0,
                                            nullptr,
                                            inMinSize);

    unsigned outTensor =  createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                              outMaxSize, tensorDim, syn_type_single,
                                              nullptr, "outTensor", 0, 0, nullptr, outMinSize);

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outMaxSize, outMinSize, tensorDim,
                                             syn_type_int32, "shapeTensor");

    addNodeToGraph(NodeFactory::broadcastNodeTypeName, {inTensor, shapeTensor}, {outTensor});

    compileTopology();

    setActualSizes(inTensor, inActualShape);
    setActualSizes(outTensor, outActualShape);
    setActualSizes(shapeTensor, outActualShape);

    runTopology(0, true);

    float* inputData = castHostBuffer<float>(inTensor);
    float* outputData = castHostBuffer<float>(outTensor);

    size_t inputElements = actualDimSize;
    size_t outputMaxElements = outMaxSize[0] * outMaxSize[1];
    size_t outputActualElement = actualDimSize * outMaxSize[1];

    // verify that the input data was broadcasted
    for(size_t i = 0; i < outputActualElement; ++i)
    {
        auto inputIndex = i % inputElements;
        EXPECT_EQ(inputData[inputIndex], outputData[i]) << "Wrong value at index: "
                                               << i << " value: " << outputData[i] << " expected: " << inputData[0];
    }

    // verify that the output is calculated only for the actual batch range
    for(size_t i = outputActualElement; i < outputMaxElements; ++i)
    {
        ASSERT_EQ(0, outputData[i]) << "Wrong value at index: "
                                    << i << " value: " << outputData[i] << " expected: " << 0;
    }
}

TEST_P_GC(SynGaudiDynamicShapesTestsShapeTensor, reshape_with_shape_tensor_with_rt)
{
    const unsigned tensorDim = 4;
    const unsigned actualBatch = GetParam();
    size_t H = 3;
    size_t W = 2;
    size_t C = 2;

    unsigned inMaxSize[] = {C, W, H, maxDimSize};
    unsigned inMinSize[] = {C, W, H, minDimSize};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    unsigned inMaxShape[] = {1, C * W, H, maxDimSize};
    unsigned inMinShape[] = {1, C * W, H, minDimSize};
    unsigned inActualShape[] = {1, C * W, H, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxShape, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, inMinShape);

    auto shapeTensor = createShapeTensor(INPUT_TENSOR, inMaxShape, inMinShape, tensorDim);

    TensorIndices inputIndices = {inTensor, shapeTensor};
    TensorIndices outputIndices = {outTensor};

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, inputIndices, outputIndices);

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(shapeTensor, inActualShape);
    setActualSizes(outTensor, inActualShape);

    runTopology(0, true);
}

TEST_P_GC(SynGaudiDynamicShapesTestsShapeTensor, broadcast_fcd_with_shape_tensor_with_rt)
{
    const unsigned actualBatch = GetParam();
    const unsigned tensorDim = 4;

    unsigned inMaxSize[]      = {1, 1, 1, 1};
    unsigned outMaxSize[]     = {64, 3, 7, maxDimSize};
    unsigned outMinSize[]     = {64, 3, 7, minDimSize};
    unsigned outActualShape[] = {64, 3, 7, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSize,
                                            tensorDim,
                                            syn_type_single,
                                            nullptr,
                                            "inTensor");

    unsigned outTensor =  createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                              outMaxSize, tensorDim, syn_type_single,
                                              nullptr, "outTensor", 0, 0, nullptr, outMinSize);

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outMaxSize, outMinSize, tensorDim,
                                             syn_type_int32, "shapeTensor");

    addNodeToGraph(NodeFactory::broadcastNodeTypeName, {inTensor, shapeTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(outTensor, outActualShape);
    setActualSizes(shapeTensor, outActualShape);

    runTopology(0, true);

    float* inputData = castHostBuffer<float>(inTensor);
    float* outputData = castHostBuffer<float>(outTensor);
    size_t inputElements = inMaxSize[0] * inMaxSize[1] * inMaxSize[2] * inMaxSize[3];
    size_t outputMaxElements = outMaxSize[0] * outMaxSize[1] * outMaxSize[2] * outMaxSize[3];
    size_t outputActualElement = outMaxSize[0] * outMaxSize[1] * outMaxSize[2] * actualBatch;

    // verify that the input data was broadcasted
    for(size_t i = 0; i < outputActualElement; ++i)
    {
        auto inputIndex = i % inputElements;
        ASSERT_EQ(inputData[inputIndex], outputData[i]) << "Wrong value at index: "
                                               << i << " value: " << outputData[i] << " expected: " << inputData[0];
    }

    // verify that the output is calculated only for the actual batch range
    for(size_t i = outputActualElement; i < outputMaxElements; ++i)
    {
        ASSERT_EQ(0, outputData[i]) << "Wrong value at index: "
                                    << i << " value: " << outputData[i] << " expected: " << 0;
    }
}

TEST_P_GC(SynGaudiDynamicShapesTestsShapeTensor, broadcast_fcd_with_shape_tensor_with_rt_5d)
{
    const unsigned actualBatch = GetParam();
    const unsigned tensorDim = 5;

    unsigned inMaxSize[]      = {1, 4, 4, 4, maxDimSize};
    unsigned outMaxSize[]     = {4, 4, 4, 4, maxDimSize};
    unsigned outMinSize[]     = {4, 4, 4, 4, minDimSize};
    unsigned outActualShape[] = {4, 4, 4, 4, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSize,
                                            tensorDim,
                                            syn_type_single,
                                            nullptr,
                                            "inTensor");

    unsigned outTensor =  createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                              outMaxSize, tensorDim, syn_type_single,
                                              nullptr, "outTensor", 0, 0, nullptr, outMinSize);

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outMaxSize, outMinSize, tensorDim,
                                             syn_type_int32, "shapeTensor");

    addNodeToGraph(NodeFactory::broadcastNodeTypeName, {inTensor, shapeTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(outTensor, outActualShape);
    setActualSizes(shapeTensor, outActualShape);

    runTopology(0, true);

    float* inputData = castHostBuffer<float>(inTensor);
    float* outputData = castHostBuffer<float>(outTensor);
    size_t outputMaxElements = outMaxSize[0] * outMaxSize[1] * outMaxSize[2] * outMaxSize[3] * outMaxSize[4];
    size_t outputActualElement = outMaxSize[0] * outMaxSize[1] * outMaxSize[2] * outMaxSize[3] * actualBatch;

    // verify that the input data was broadcasted
    for(size_t i = 0; i < outputActualElement; ++i)
    {
        auto inputIndex = i / outMaxSize[0];
        ASSERT_EQ(inputData[inputIndex], outputData[i]) << "Wrong value at index: "
                                               << i << " value: " << outputData[i] << " expected: " << inputData[0];
    }

    // verify that the output is calculated only for the actual batch range
    for(size_t i = outputActualElement; i < outputMaxElements; ++i)
    {
        ASSERT_EQ(0, outputData[i]) << "Wrong value at index: "
                                    << i << " value: " << outputData[i] << " expected: " << 0;
    }
}

TEST_P_GC(SynGaudiDynamicShapesTestsShapeTensor, broadcast_fcd_with_shape_tensor_with_rt_5d_scalar)
{
    const unsigned actualBatch = GetParam();
    const unsigned tensorDim = 5;

    unsigned inMaxSize[]      = {1};
    unsigned outMaxSize[]     = {4, 4, 4, 4, maxDimSize};
    unsigned outMinSize[]     = {4, 4, 4, 4, minDimSize};
    unsigned outActualShape[] = {4, 4, 4, 4, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSize,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            "inTensor");

    unsigned outTensor =  createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                              outMaxSize, tensorDim, syn_type_single,
                                              nullptr, "outTensor", 0, 0, nullptr, outMinSize);

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outMaxSize, outMinSize, tensorDim,
                                             syn_type_int32, "shapeTensor");

    addNodeToGraph(NodeFactory::broadcastNodeTypeName, {inTensor, shapeTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(outTensor, outActualShape);
    setActualSizes(shapeTensor, outActualShape);

    runTopology(0, true);

    float* inputData = castHostBuffer<float>(inTensor);
    float* outputData = castHostBuffer<float>(outTensor);
    size_t outputMaxElements = outMaxSize[0] * outMaxSize[1] * outMaxSize[2] * outMaxSize[3] * outMaxSize[4];
    size_t outputActualElement = outMaxSize[0] * outMaxSize[1] * outMaxSize[2] * outMaxSize[3] * actualBatch;

    // verify that the input data was broadcasted
    for(size_t i = 0; i < outputActualElement; ++i)
    {
        auto inputIndex = 0;
        ASSERT_EQ(inputData[inputIndex], outputData[i]) << "Wrong value at index: "
                                               << i << " value: " << outputData[i] << " expected: " << inputData[0];
    }

    // verify that the output is calculated only for the actual batch range
    for(size_t i = outputActualElement; i < outputMaxElements; ++i)
    {
        ASSERT_EQ(0, outputData[i]) << "Wrong value at index: "
                                    << i << " value: " << outputData[i] << " expected: " << 0;
    }
}

TEST_F_GC(SynGaudiDynamicShapesTests, create_invalid_device_shape_tensor)
{
    // Shape tensor correctness is validated during node creation.
    // Therfore we will create a graph and try to add a node with invalid shape tensor.
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    // first create a valid tensor for the node output
    synTensor outputTensor = nullptr;
    synTensorDescriptor outDesc;
    outDesc.m_dataType = syn_type_float;
    outDesc.m_name = "validOutpout";
    outDesc.m_dims = 1;
    outDesc.m_sizes[0] = SYN_MAX_TENSOR_DIM;
    outDesc.m_tensorType = DATA_TENSOR;
    ASSERT_EQ(synSuccess, synTensorCreate(&outputTensor, &outDesc, nullptr, 0));

    // shape tensor with wrong dtype
    synTensor shapeTensorWrongDtype = nullptr;
    synTensorDescriptor wrongDtypeDesc;
    wrongDtypeDesc.m_dataType = syn_type_float;
    wrongDtypeDesc.m_name = "ShapeTensorWrongDtype";
    wrongDtypeDesc.m_dims = 1;
    wrongDtypeDesc.m_sizes[0] = 1;
    wrongDtypeDesc.m_tensorType = DEVICE_SHAPE_TENSOR;
    ASSERT_EQ(synSuccess, synTensorCreate(&shapeTensorWrongDtype, &wrongDtypeDesc, nullptr, 0));
    ASSERT_NE(synSuccess, synNodeCreate(graphHandle , &shapeTensorWrongDtype, &outputTensor, 1, 1,
                                        nullptr, 0, "memcpy", nullptr, nullptr, nullptr));

    // shape tensor with wrong dim
    synTensor shapeTensorWrongDim = nullptr;
    synTensorDescriptor wrongDimDesc;
    wrongDimDesc.m_dataType = syn_type_uint32;
    wrongDimDesc.m_name = "ShapeTensorWrongDim";
    wrongDimDesc.m_dims = 2;
    wrongDimDesc.m_sizes[0] = 1;
    wrongDimDesc.m_sizes[1] = 1;
    wrongDimDesc.m_tensorType = DEVICE_SHAPE_TENSOR;
    ASSERT_EQ(synSuccess, synTensorCreate(&shapeTensorWrongDim, &wrongDimDesc, nullptr, 0));
    ASSERT_NE(synSuccess, synNodeCreate(graphHandle ,&shapeTensorWrongDim, &outputTensor, 1, 1,
                                        nullptr, 0, "memcpy", nullptr, nullptr, nullptr));

    // shape tensor with wrong size
    synTensor shapeTensorWrongSize = nullptr;
    synTensorDescriptor wrongSizeDesc;
    wrongSizeDesc.m_dataType = syn_type_uint32;
    wrongSizeDesc.m_name = "ShapeTensorWrongSize";
    wrongSizeDesc.m_dims = 1;
    wrongSizeDesc.m_sizes[0]--;
    wrongSizeDesc.m_tensorType = DEVICE_SHAPE_TENSOR;
    ASSERT_EQ(synSuccess, synTensorCreate(&shapeTensorWrongSize, &wrongSizeDesc, nullptr, 0));
    ASSERT_NE(synSuccess, synNodeCreate(graphHandle ,&shapeTensorWrongSize, &outputTensor, 1, 1,
                                        nullptr, 0, "memcpy", nullptr, nullptr, nullptr));

    synGraphDestroy(graphHandle);
}

// This class Handles tests for dynamic execute - only disables full activations of an engine.
class SynGaudiDynamicExecuteTestMemcpy : public SynGaudiDynamicShapesTests,
                                         public testing::WithParamInterface<unsigned>
{
public:
    static constexpr size_t minBatch = 2;
    static constexpr size_t maxBatch = 5;
    static constexpr size_t batchIndex = 3;
};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicExecuteTestMemcpy, ::testing::Values(2, 3, 4, 5));

TEST_P_GC(SynGaudiDynamicExecuteTestMemcpy, basic_batch_disable_memcpy)
{
    const unsigned tensorDim = 4;
    const unsigned splitDim = tensorDim - 1;
    const unsigned actualBatch = GetParam();
    size_t H = 2;
    size_t W = 512;
    size_t C = 16;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned outMaxSize[] = {C, W, H, 1};
    unsigned outMinSize[] = {C, W, H, 0};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    const unsigned nOuts = inMaxSize[splitDim];

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    std::vector<unsigned> splitTensorsOutput;
    std::vector<unsigned> copyTensorsInput;

    // Split the input on batch
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned  tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                        tensorDim, syn_type_single, nullptr, minSize);
        splitTensorsOutput.push_back(tensor);
        copyTensorsInput.push_back(connectOutputTensorToInputTensor(tensor));
    }

    addNodeToGraph(NodeFactory::splitNodeTypeName, {inTensor}, splitTensorsOutput, (void*)&splitDim, sizeof(splitDim));
    std::vector<unsigned> concatInputTensors;
    // memcpy each tensor.
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned  tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                        tensorDim, syn_type_single, nullptr, minSize);

        concatInputTensors.push_back(connectOutputTensorToInputTensor(tensor));
        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {copyTensorsInput[i]}, {tensor});
    }

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, inMinSize);

    // concatenateNodeInternal does not create a physical concat node
    addNodeToGraph(NodeFactory::concatenateNodeLogicalInternalTypeName,
                   concatInputTensors,
                   {outTensor},
                   (void*)&splitDim,
                   sizeof(splitDim));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);

    runTopology(0, true);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);
    const uint64_t tensorBatchSizeElements = getNumberOfElements(inMinSize, splitDim);
    const uint64_t tensorSizeElements      = tensorBatchSizeElements * actualBatch;
    const uint64_t garbageElements         = tensorBatchSizeElements * (maxBatch - actualBatch);

    // Test by the actual batch size.
    for (uint64_t i = 0; i < tensorSizeElements; i++)
    {
        ASSERT_EQ(inBuffer[i], outBuffer[i]) << i;
    }
    for (uint64_t i = tensorSizeElements; i < tensorSizeElements + garbageElements; i++)
    {
        ASSERT_EQ(outBuffer[i], 0) << i;
    }
}

class SynGaudiDynamicExecuteTestMemcpyReluBase : public SynGaudiDynamicExecuteTestMemcpy
{
};

class SynGaudiDynamicExecuteTestMemcpyRelu : public SynGaudiDynamicExecuteTestMemcpyReluBase {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicExecuteTestMemcpyRelu, ::testing::Values(2, 3, 4, 5));

TEST_P_GC(SynGaudiDynamicExecuteTestMemcpyRelu, basic_batch_disable_memcpy_relu)
{
    const unsigned tensorDim = 4;
    const unsigned splitDim = tensorDim - 1;
    const unsigned actualBatch = GetParam();
    size_t H = 128;
    size_t W = 512;
    size_t C = 16;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned outMaxSize[] = {C, W, H, 1};
    unsigned outMinSize[] = {C, W, H, 0};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    const unsigned nOuts = inMaxSize[splitDim];

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    std::vector<unsigned> splitTensorsOutput;
    std::vector<unsigned> copyTensorsInput;

    // Split the input on batch
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned  tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                        tensorDim, syn_type_single, nullptr, minSize);
        splitTensorsOutput.push_back(tensor);
        copyTensorsInput.push_back(connectOutputTensorToInputTensor(tensor));
    }

    addNodeToGraph(NodeFactory::splitNodeTypeName, {inTensor}, splitTensorsOutput, (void*)&splitDim, sizeof(splitDim));

    std::vector<unsigned> reluInputTensors;
    // memcpy each tensor.
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned  tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                        tensorDim, syn_type_single, nullptr, minSize);

        reluInputTensors.push_back(connectOutputTensorToInputTensor(tensor));
        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {copyTensorsInput[i]}, {tensor});
    }

    // relu each tensor split.
    std::vector<unsigned> concatInputTensors;
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                       tensorDim, syn_type_single, nullptr, minSize);

        concatInputTensors.push_back(connectOutputTensorToInputTensor(tensor));
        addNodeToGraph("relu_fwd_f32", {reluInputTensors[i]}, {tensor});
    }

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, inMinSize);

    addNodeToGraph(NodeFactory::concatenateNodeLogicalInternalTypeName,
                   concatInputTensors,
                   {outTensor},
                   (void*)&splitDim,
                   sizeof(splitDim));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);

    runTopology(0, true);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);
    const uint64_t tensorBatchSizeElements = getNumberOfElements(inMaxSize, splitDim);
    const uint64_t tensorSizeElements      = tensorBatchSizeElements * actualBatch;
    const uint64_t garbageElements         = tensorBatchSizeElements * (maxBatch - actualBatch);

    // Test by the actual batch size.
    for (uint64_t i = 0; i < tensorSizeElements; i++)
    {
        if (inBuffer[i] > 0)
        {
            ASSERT_EQ(inBuffer[i], outBuffer[i]) << i;
        }
        else
        {
            ASSERT_EQ(0, outBuffer[i]) << i;
        }
    }
    for (uint64_t i = tensorSizeElements; i < tensorSizeElements + garbageElements; i++)
    {
        ASSERT_EQ(outBuffer[i], 0) << i;
    }
}

class SynGaudiDynamicExecuteTestMemcpyReluConv : public SynGaudiDynamicExecuteTestMemcpyReluBase {};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicExecuteTestMemcpyReluConv, ::testing::Values(2, 3, 4, 5));

TEST_P_GC(SynGaudiDynamicExecuteTestMemcpyReluConv, basic_batch_disable_memcpy_relu_conv, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned tensorDim = 4;
    const unsigned splitDim = tensorDim - 1;
    const unsigned actualBatch = GetParam();
    size_t H = 128;
    size_t W = 128;
    size_t C = 16;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, actualBatch};
    unsigned outMaxSize[] = {C, W, H, 1};
    unsigned outMinSize[] = {C, W, H, 0};

    const unsigned nOuts = inMaxSize[splitDim];

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    std::vector<unsigned> splitTensorsOutput;
    std::vector<unsigned> copyTensorsInput;

    // Split the input on batch
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned  tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                        tensorDim, syn_type_single, nullptr, minSize);
        splitTensorsOutput.push_back(tensor);
        copyTensorsInput.push_back(connectOutputTensorToInputTensor(tensor));
    }

    addNodeToGraph(NodeFactory::splitNodeTypeName, {inTensor}, splitTensorsOutput, (void*)&splitDim, sizeof(splitDim));

    std::vector<unsigned> reluInputTensors;
    // memcpy each tensor.
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned  tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                        tensorDim, syn_type_single, nullptr, minSize);

        reluInputTensors.push_back(connectOutputTensorToInputTensor(tensor));
        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {copyTensorsInput[i]}, {tensor});
    }

    // relu each tensor split.
    std::vector<unsigned> convInputTensors;
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = outMinSize;
        if (i < minBatch)
        {
            minSize = outMaxSize;
        }
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outMaxSize,
                                       tensorDim, syn_type_single, nullptr, minSize);

        convInputTensors.push_back(connectOutputTensorToInputTensor(tensor));
        addNodeToGraph("relu_fwd_f32", {reluInputTensors[i]}, {tensor});
    }

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

    unsigned wDimSizes[] = { C, C, params.kW, params.kH };
    unsigned wTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                           wDimSizes, tensorDim, syn_type_single);

    const unsigned convW = convOutputDimSize(W, params.kW, params.dW, params.padL + params.padR, params.dilW);
    const unsigned convH = convOutputDimSize(H, params.kH, params.dH, params.padT + params.padB, params.dilH);

    unsigned convOutMaxDims[] = {C, convW, convH, 1};
    unsigned convOutMinDims[] = {C, convW, convH, 0};

    std::vector<unsigned> concatInputTensors;
    for (unsigned i = 0; i < nOuts; ++i)
    {
        // If i is smaller than min batch, its size is static.
        unsigned* minSize = convOutMinDims;
        if (i < minBatch)
        {
            minSize = convOutMaxDims;
        }
        unsigned tensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, convOutMaxDims,
                                       tensorDim, syn_type_single, nullptr, minSize);

        concatInputTensors.push_back(connectOutputTensorToInputTensor(tensor));
        addNodeToGraph(NodeFactory::convolutionNodeTypeName, {convInputTensors[i], wTensor},
                       {tensor}, (void*)&params, sizeof(synConvolutionParams));
    }

    unsigned concatMaxSize[] = {C, convW, convH, maxBatch};
    unsigned concatMinSize[] = {C, convW, convH, minBatch};
    unsigned concatActualSize[] = {C, convW, convH, actualBatch};
    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             concatMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, concatMinSize);

    addNodeToGraph(NodeFactory::concatenateNodeLogicalInternalTypeName,
                   concatInputTensors,
                   {outTensor},
                   (void*)&splitDim,
                   sizeof(splitDim));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle, nullptr);
    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, concatActualSize);
    runTopology(0, true);

    const auto inDesc  = static_cast<synTensorDescriptor>(getTensorDescriptor(inTensor));
    const auto wDesc   = static_cast<synTensorDescriptor>(getTensorDescriptor(wTensor));
    const auto outDesc = static_cast<synTensorDescriptor>(getTensorDescriptor(outTensor));
    float* inData = castHostInBuffer<float>(inTensor);
    float* wData = castHostInBuffer<float>(wTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < multiplyElements(inMaxSize, inMaxSize + tensorDim); i++)
    {
        inData[i] = inData[i] < 0 ? 0 : inData[i];
    }
    CoordArray wrongIdx = {0};
    float expectedResult = 0;
    bool       ret            = checkMmeOp(inDesc,
                          (char*)inData,
                          wDesc,
                          (char*)wData,
                          outDesc,
                          (char*)outData,
                          params,
                          ERepefenceOp::REFERENCE_OP_FWD,
                          wrongIdx,
                          m_deviceType,
                          &expectedResult);

    if (actualBatch == maxBatch)
    {
        TSize sizes[SYN_MAX_TENSOR_DIM];
        castNcopy(sizes, outDesc.m_sizes, SYN_MAX_TENSOR_DIM);
        ASSERT_TRUE(ret) << "Wrong value at index: " << toString(wrongIdx.begin(), wrongIdx.end(), ',')
                         << " Got value: " << getIndexValue(sizes, wrongIdx, outDesc.m_dataType, outData)
                         << " Expected: " << expectedResult;
        return;
    }

    // If the actual batch isn't the max batch - ret should be false, and the wrong index should be the actual batch offset.
    ASSERT_FALSE(ret);
    size_t batchElements = C * convW * convH;

    for (int d = 0; d < tensorDim - 1; d++)
    {
        ASSERT_EQ(wrongIdx[d], 0);
    }

    ASSERT_EQ(wrongIdx[batchIndex], actualBatch);
    for (int i = batchElements * actualBatch; i < batchElements * maxBatch; i++)
    {
        ASSERT_EQ(outData[i], 0) << i;
    }
}

class SynGaudiDynamicShapesTestsStaticRoi: public SynGaudiDynamicShapesTests {};

TEST_F_GC(SynGaudiDynamicShapesTestsStaticRoi, memcpy_no_pps_for_static_roi)
{
    const unsigned tensorDim = 4;
    const unsigned lastDim = tensorDim - 1;
    const unsigned actualBatch = 8;
    const unsigned minBatch = 6;
    const unsigned maxBatch = 10;
    unsigned H = 2;
    unsigned W = 64;
    unsigned C = 4;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);
    runTopology(0, true);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);
    const uint64_t tensorBatchSizeElements = getNumberOfElements(inMinSize, lastDim);
    const uint64_t tensorSizeElements      = tensorBatchSizeElements * actualBatch;
    const uint64_t garbageElements         = tensorBatchSizeElements * (maxBatch - actualBatch);

    // Test by the actual batch size.
    for (uint64_t i = 0; i < tensorSizeElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, inBuffer[i])) << i;
    }
    for (uint64_t i = tensorSizeElements; i < tensorSizeElements + garbageElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, 0.0f)) << i;
    }
}

class SynGaudiDynamicShapesSliceTests :
    public SynGaudiDynamicShapesTests,
    public testing::WithParamInterface<unsigned>
{};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicShapesSliceTests, ::testing::Values(0, 1, 2, 3, 4, 5));

TEST_P_GC(SynGaudiDynamicShapesSliceTests, slice_test)
{
    const unsigned tensorDim = 4;
    const unsigned actualSliceBatch = GetParam();
    const unsigned minBatch = 5;
    const unsigned maxBatch = 10;
    unsigned H = 2;
    unsigned W = 64;
    unsigned C = 4;

    unsigned inMaxSize[]    = {C, W, H, maxBatch};
    unsigned inMinSize[]    = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, minBatch + actualSliceBatch};

    unsigned outDynamicMaxSize[]    = {C, W, H, 5};
    unsigned outDynamicMinSize[]    = {C, W, H, 0};
    unsigned outDynamicActualSize[] = {C, W, H, actualSliceBatch};

    unsigned outStaticSize[]    = {C, W, H, actualSliceBatch};

    synSliceParams sliceDynamicParams =
    {
        .axes   = {0, 1, 2, 3, 0},
        .starts = {0, 0, 0, minBatch, 0},
        .ends   = {C, W, H, maxBatch, 0},
        .steps  = {1, 1, 1, 1, 0}
    };

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned outTensor1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             outDynamicMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, outDynamicMinSize);

    addNodeToGraph(NodeFactory::sliceNodeTypeName, {inTensor}, {outTensor1}, &sliceDynamicParams, sizeof(sliceDynamicParams), "slice1");

    if (actualSliceBatch > 0)
    {
        unsigned outTensor2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                outStaticSize, tensorDim, syn_type_single, nullptr, nullptr,
                                                0, 0, nullptr, nullptr);

        synSliceParams sliceStaticParams =
        {
            .axes   = {0, 1, 2, 3, 0},
            .starts = {0, 0, 0, 0, 0},
            .ends   = {C, W, H, actualSliceBatch, 0},
            .steps  = {1, 1, 1, 1, 0}
        };

        addNodeToGraph(NodeFactory::sliceNodeTypeName, {inTensor}, {outTensor2}, &sliceStaticParams, sizeof(sliceStaticParams), "slice2");
    }
    compileTopology();

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor1, outDynamicActualSize);
    runTopology(0, true);
}

TEST_P_GC(SynGaudiDynamicShapesSliceTests, slice_with_shape_tensor_with_rt)
{
    const unsigned tensorDim = 2;

    unsigned W = 9;
    unsigned minH = 5;
    unsigned maxH = 10;
    const unsigned actualH = minH + GetParam();

    unsigned maxSize[]    = {W, maxH};
    unsigned minSize[]    = {W, minH};
    unsigned actualSize[] = {W, actualH};

    synSliceParams sliceDynamicParams =
    {
        .axes   = {0, 1},
        .starts = {0, 0},
        .ends   = {W, maxH},
        .steps  = {1, 1}
    };

    unsigned inTensor = createConstTensor(MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSize, tensorDim,
                                          syn_type_single, nullptr, "inTensor");

    auto shapeTensor = createShapeTensor(INPUT_TENSOR, maxSize, minSize, tensorDim, syn_type_int32, "shapeTensor");

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSize, tensorDim, syn_type_single, nullptr, "outTensor",
                                             0, 0, nullptr, minSize);

    addNodeToGraph(NodeFactory::sliceNodeTypeName, {inTensor, shapeTensor}, {outTensor},
                   &sliceDynamicParams, sizeof(sliceDynamicParams), "slice");

    compileTopology();
    ASSERT_FALSE(HasFailure());

    setActualSizes(shapeTensor, actualSize);
    setActualSizes(outTensor, actualSize);

    runTopology(0, true);

    float* inData = castHostBuffer<float>(inTensor);
    float* outData = castHostBuffer<float>(outTensor);

    size_t maxElementsSize = maxSize[0] * maxSize[1];
    size_t actualElementsSize = actualSize[0] * actualSize[1];

    // verify that the data of the actual size is identical to the input
    for(size_t i = 0; i < actualElementsSize; ++i)
    {
        ASSERT_EQ(inData[i], outData[i]);
    }

    // verify that the output is written only for the actual batch range
    for(size_t i = actualElementsSize; i < maxElementsSize; ++i)
    {
        ASSERT_EQ(outData[i], 0);
    }
}

class SynGaudiDynamicShapesSplitTests :
    public SynGaudiDynamicShapesTests,
    public testing::WithParamInterface<unsigned>
{};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicShapesSplitTests, ::testing::Values(2, 4, 6, 8, 12));

TEST_P_GC(SynGaudiDynamicShapesSplitTests, split_test)
{
    unsigned H = 2;
    unsigned W = 64;
    unsigned C = 4;
    const unsigned tensorDim = 4;
    const unsigned actualBatch = 15;
    const unsigned minBatch = 5;
    const unsigned maxBatch = 24;
    unsigned splitDim = 3;
    unsigned inMaxSize[]    = {C, W, H, maxBatch};
    unsigned inMinSize[]    = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    unsigned nOutputs = GetParam(); // Assuming: maxBatch % nOutputs == 0

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);


    std::vector<unsigned> outTensors;
    unsigned outBatchSize = maxBatch / nOutputs;
    unsigned outMaxSize[] = {C, W, H, outBatchSize};
    for (unsigned i = 0; i < minBatch / outBatchSize; i += 1)
    {
        unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                 outMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                                 0, 0, nullptr, nullptr);
        outTensors.push_back(outTensor);

    }
    for (unsigned i = minBatch / outBatchSize; i < nOutputs; ++i)
    {
        unsigned batch = (minBatch > i * outBatchSize)? minBatch - i * outBatchSize: 0;
        unsigned outMinSize[] = {C, W, H, batch};
        unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                 outMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                                 0, 0, nullptr, outMinSize);
        outTensors.push_back(outTensor);
    }

    addNodeToGraph(NodeFactory::splitNodeTypeName, {inTensor}, outTensors, &splitDim, sizeof(splitDim), "split1");
    compileTopology();

    setActualSizes(inTensor, inActualSize);
    for (unsigned i = minBatch / outBatchSize; i < actualBatch / outBatchSize; ++i)
    {
        unsigned outActualSize[] = {C, W, H, outBatchSize};
        setActualSizes(outTensors[i], outActualSize);
    }
    for (unsigned i = actualBatch / outBatchSize; i < nOutputs; ++i)
    {
        unsigned batch = (actualBatch > i * outBatchSize)? actualBatch - i * outBatchSize: 0;
        unsigned outActualSize[] = {C, W, H, batch};
        setActualSizes(outTensors[i], outActualSize);
    }
    runTopology(0, true);
}

TEST_F_GC(SynGaudiDynamicShapesTests, memset_for_reduction_with_shape_tensor, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    GlobalConfTestSetter conf_parallelLevel("ENABLE_INTERNAL_NODES", "true");
    constexpr unsigned   numOfReductions = 3;

    // this is also scheduler test, we create shape tensor which is not input to the graph
    // and he is also "real" producer of the reduction,
    // so the scheduler must shcedule the shape op before the memset node

    unsigned inputMaxSizes[]    = {10, 2, 16, 5};
    unsigned inputMinSizes[]    = {10, 2, 4, 5};
    unsigned inputActualSizes[] = {10, 2, 7, 5};

    unsigned maxSizes[]    = {10, 2, 1, 16, 5};
    unsigned minSizes[]    = {10, 2, 1, 4, 5};
    unsigned actualSizes[] = {10, 2, 1, 7, 5};

    unsigned shapeTensor         = createShapeTensor(INPUT_TENSOR, inputMaxSizes, inputMinSizes, 4);
    unsigned shapeTensorReshaped = createShapeTensor(OUTPUT_TENSOR, maxSizes, minSizes, 5);
    unsigned inputTensor         = createPersistTensor(INPUT_TENSOR,
                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                               nullptr,
                                               inputMaxSizes,
                                               4,
                                               syn_type_single,
                                               nullptr,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               inputMinSizes);
    unsigned inputTensorReshaped =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, maxSizes, 5, syn_type_single, nullptr, minSizes);
    unsigned outputTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ONES,
                                                nullptr,
                                                maxSizes,
                                                5,
                                                syn_type_single,
                                                nullptr,
                                                nullptr,
                                                0,
                                                0,
                                                nullptr,
                                                minSizes);

    addNodeToGraph(NodeFactory::reshapeNodeTypeName, {inputTensor, shapeTensorReshaped}, {inputTensorReshaped});
    unsigned dimToExpand = 2;
    addNodeToGraph(NodeFactory::expandDimsShapeNodeTypeName,
                   {shapeTensor},
                   {shapeTensorReshaped},
                   (void*)&dimToExpand,
                   sizeof(dimToExpand));

    unsigned lastOutput    = inputTensorReshaped;
    unsigned reductionType = REDUCTION_SET;
    for (unsigned i = 0; i < numOfReductions; ++i)
    {
        unsigned output;
        if (i == numOfReductions - 1)
        {
            output = outputTensor;
        }
        else
        {
            output = createTensor(OUTPUT_TENSOR,
                                  MEM_INIT_ALL_ONES,
                                  nullptr,
                                  maxSizes,
                                  5,
                                  syn_type_single,
                                  nullptr,
                                  minSizes);
        }

        unsigned memsetOutput =
            createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, maxSizes, 5, syn_type_single, nullptr, minSizes);
        addNodeToGraph(NodeFactory::memsetNodeTypeName, {shapeTensorReshaped}, {memsetOutput});
        addNodeToGraph(NodeFactory::reductionNodeTypeName,
                       {memsetOutput, lastOutput},
                       {output},
                       (void*)&reductionType,
                       sizeof(reductionType));
        lastOutput = output;
    }

    compileTopology();

    setActualSizes(shapeTensor, inputActualSizes);
    setAsInternalShapeTensor(shapeTensorReshaped);
    setActualSizes(inputTensor, inputActualSizes);
    setActualSizes(outputTensor, actualSizes);

    runTopology();

    float* out = (float*)m_hostBuffers[outputTensor];
    float* in  = (float*)m_hostBuffers[inputTensor];

    unsigned tensorSize = std::accumulate(inputActualSizes, inputActualSizes + 4, 1, std::multiplies<unsigned>());
    for (unsigned i = 0; i < tensorSize; ++i)
    {
        ASSERT_FLOAT_EQ(*in, *out) << "mismatch in index: " << i;
        ++out;
        ++in;
    }
}

TEST_F_GC(SynGaudiDynamicShapesTests, memset_dynamic_shape)
{
    unsigned input_max_sizes[]    = {5, 7, 8, 9, 10};
    unsigned input_min_sizes[]    = {2, 7, 8, 9, 10};
    unsigned input_actual_sizes[] = {5, 7, 8, 9, 10};

    unsigned shape_max_sizes[]    = {12, 7, 8, 9, 10};
    unsigned shape_min_sizes[]    = {4, 7, 8, 9, 10};
    unsigned shape_actual_sizes[] = {5, 7, 8, 9, 10};

    unsigned output_max_sizes[]    = {12, 7, 8, 9, 10};
    unsigned output_min_sizes[]    = {4, 7, 8, 9, 10};
    unsigned output_actual_sizes[] = {5, 7, 8, 9, 10};

    unsigned input = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_ALL_ZERO,
                                         nullptr,
                                         input_max_sizes,
                                         5,
                                         syn_type_single,
                                         nullptr,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         input_min_sizes);

    unsigned shape = createShapeTensor(INPUT_TENSOR, shape_max_sizes, shape_min_sizes, 5);

    unsigned output = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          output_max_sizes,
                                          5,
                                          syn_type_single,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0,
                                          nullptr,
                                          output_min_sizes);

    synSliceParams params;
    memset(&params, 0, sizeof(params));
    params.axes[1]  = 1;
    params.axes[2]  = 2;
    params.axes[3]  = 3;
    params.axes[4]  = 4;
    params.ends[0]  = 10;
    params.ends[1]  = 7;
    params.ends[2]  = 8;
    params.ends[3]  = 9;
    params.ends[4]  = 10;
    params.steps[0] = 2;
    params.steps[1] = 1;
    params.steps[2] = 1;
    params.steps[3] = 1;
    params.steps[4] = 1;

    addNodeToGraph("strided_slice_grad", {input, shape}, {output}, (void*)&params, sizeof(params));

    compileTopology();

    setActualSizes(input, input_actual_sizes);
    setActualSizes(shape, shape_actual_sizes);
    setActualSizes(output, output_actual_sizes);

    runTopology();
}

class SynGaudiIndexSpacePatchingTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiIndexSpacePatchingTest, index_space_patching)
{
    unsigned input_max_shape[] = {1002, 4, 1};
    unsigned input_min_shape[] = {10, 4, 1};
    unsigned input_act_shape[] = {501, 4, 1};

    unsigned output_max_shape[] = {3006, 4, 40};
    unsigned output_min_shape[] = {10, 4, 5};
    unsigned output_act_shape[] = {1503, 4, 20};

    unsigned h2d_data[]     = {3, 1, 40, 1, 1, 5};
    unsigned h2d_act_data[] = {3, 1, 20};
    unsigned h2d_data_size  = 3;

    unsigned input = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         input_max_shape,
                                         2,
                                         syn_type_float,
                                         nullptr,
                                         "input",
                                         0,
                                         0,
                                         nullptr,
                                         input_min_shape);

    unsigned shape = createShapeTensor(INPUT_TENSOR, input_max_shape, input_min_shape, 3, syn_type_uint32, "shape");

    unsigned reshape = createTensor(OUTPUT_TENSOR,
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    input_max_shape,
                                    3,
                                    syn_type_float,
                                    nullptr,
                                    input_min_shape);

    unsigned params = createHost2DeviceTensor(INPUT_TENSOR, &h2d_data_size, h2d_data, 1, "params");

    unsigned output = createPersistTensor(OUTPUT_TENSOR,
                                          MEM_INIT_ALL_ZERO,
                                          nullptr,
                                          output_max_shape,
                                          3,
                                          syn_type_float,
                                          nullptr,
                                          "output",
                                          0,
                                          0,
                                          nullptr,
                                          output_min_shape);

    addNodeToGraph("reshape", {input, shape}, {reshape}, nullptr, 0, "reshape");
    addNodeToGraph("tile_fwd_f32", {reshape, params}, {output}, nullptr, 0, "tile_fwd");

    compileTopology();

    setActualSizes(input, input_act_shape);
    setActualSizes(shape, input_act_shape);
    setActualSizes(output, output_act_shape);
    setActualScalarParametersData(params, h2d_act_data, h2d_data_size * sizeof(h2d_data[0]));

    runTopology();

    float* dataIn  = (float*)m_hostBuffers[input];
    float* dataOut = (float*)m_hostBuffers[output];
    unsigned tileSize = input_act_shape[0]*input_act_shape[1];
    for (unsigned index = 0; index < tileSize; index++)
    {
        float ref       = dataIn[index];                        // reference value
        unsigned line   = index / input_act_shape[0];           // line in input tensor (dim 1)
        unsigned offset = index - line * input_act_shape[0];    // offset in dim 1 in input tensor

        for (unsigned tile_dim2 = 0; tile_dim2 < h2d_act_data[2]; tile_dim2++)
        {
            unsigned offset_dim2 = input_act_shape[0] * h2d_act_data[0] * input_act_shape[1] * h2d_act_data[1] * tile_dim2;
            for (unsigned tile_dim1 = 0; tile_dim1 < h2d_act_data[1]; tile_dim1++)
            {
                unsigned offset_dim1 = input_act_shape[0] * h2d_act_data[0] * input_act_shape[1] * tile_dim1;
                for (unsigned tile_dim0 = 0; tile_dim0 < h2d_act_data[0]; tile_dim0++)
                {
                    //                       selects line in tiled tensor                   selects tile in dim0          base offset
                    unsigned offset_dim0 = line * input_act_shape[0] * h2d_act_data[0] + input_act_shape[0] * tile_dim0 + offset;
                    ASSERT_FLOAT_EQ(ref, dataOut[offset_dim0 + offset_dim1 + offset_dim2]);
                }
            }
        }
    }
}
