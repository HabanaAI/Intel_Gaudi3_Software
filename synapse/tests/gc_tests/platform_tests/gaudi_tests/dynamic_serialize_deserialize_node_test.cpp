#include "gc_dynamic_shapes_infra.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "recipe.h"
#include <vector>

class SynGaudiSerializeTest : public SynGaudiDynamicShapesTestsInfra
{
    private:
        bool m_enableInternalNodes = false;

    public:
        using Base = SynGaudiDynamicShapesTestsInfra;

        void SetUpTest()
        {
            Base::SetUpTest();
            m_enableInternalNodes = GCFG_ENABLE_INTERNAL_NODES.value();
            GCFG_ENABLE_INTERNAL_NODES.setValue(true);
        }
        void TearDownTest()
        {
            GCFG_ENABLE_INTERNAL_NODES.setValue(m_enableInternalNodes);
            Base::TearDownTest();
        }

        const char* getSerializeNodeTypeName()
        {
            return m_deviceType == synDeviceGaudi3 ?
                NodeFactory::serializeTPCNodeTypeName :
                NodeFactory::serializeDMANodeTypeName;
        }
        const char* getDeserializeNodeTypeName()
        {
            return m_deviceType == synDeviceGaudi3 ?
                NodeFactory::deserializeTPCNodeTypeName :
                NodeFactory::deserializeDMANodeTypeName;
        }

        void addSerializeNode(unsigned input, unsigned output)
        {
            addNodeToGraph(getSerializeNodeTypeName(), {input}, {output},
                           nullptr, 0, fmt::format("Serialize_{}", input).c_str());
        }
        void addDeserializeNode(unsigned input, unsigned output)
        {
            addNodeToGraph(getDeserializeNodeTypeName(), {input}, {output},
                           nullptr, 0, fmt::format("Deserialize_{}", input).c_str());
        }
};

TEST_F_GC(SynGaudiSerializeTest, serialize_node, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[]    = {2, 8, 4};
    unsigned minSizes[]    = {2, 4, 4};
    unsigned actualSizes[] = {2, 6, 4};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addSerializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    StridedToDenseBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, recipe_compilation_consistency, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");

    constexpr unsigned numOfGraphs = 2;

    const size_t C           = 16;
    const size_t MAX_W       = 128;
    const size_t MIN_W       = 64;
    const size_t MAX_H       = 128;
    const size_t MIN_H       = 64;
    const size_t BATCH       = 16;
    const size_t TENSOR_DIMS = 4;

    synConvolutionParams convParams;
    convParams.dH   = 1;
    convParams.dW   = 1;
    convParams.kH   = 3;
    convParams.kW   = 3;
    convParams.dilH = 1;
    convParams.dilW = 1;
    convParams.padT = 0;
    convParams.padB = 0;
    convParams.padL = 0;
    convParams.padR = 0;

    const unsigned convOutMaxW =
        convOutputDimSize(MAX_W, convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW);
    const unsigned convOutMinW =
        convOutputDimSize(MIN_W, convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW);

    SpatialReduction2DDef kernel_params;
    kernel_params.pad_w_begin = 0;
    kernel_params.pad_h_end   = 0;
    kernel_params.pad_w_end   = 0;
    kernel_params.pad_h_begin = 0;
    kernel_params.kernel_w    = 2;
    kernel_params.kernel_h    = 2;
    kernel_params.stride_w    = 2;
    kernel_params.stride_h    = 2;
    kernel_params.dilation_w  = 1;
    kernel_params.dilation_h  = 1;

    unsigned poolOutMaxW =
        (convOutMaxW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) / kernel_params.stride_w + 1;
    unsigned poolOutMinW =
        (convOutMinW + 2 * kernel_params.pad_w_begin - (kernel_params.kernel_w - 1) - 1) / kernel_params.stride_w + 1;

    const unsigned convOutMaxH =
        convOutputDimSize(MAX_H, convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH);
    const unsigned convOutMinH =
        convOutputDimSize(MIN_H, convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH);

    unsigned poolOutMaxH =
        (convOutMaxH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) / kernel_params.stride_h + 1;
    unsigned poolOutMinH =
        (convOutMinH + 2 * kernel_params.pad_h_begin - (kernel_params.kernel_h - 1) - 1) / kernel_params.stride_h + 1;

    unsigned maxpoolOutMaxSizes[] = {C, poolOutMaxW, poolOutMaxH, BATCH};
    unsigned maxpoolOutMinSizes[] = {C, poolOutMinW, poolOutMinH, BATCH};

    unsigned biasSizes[] = {C, 1, 1, 1};

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        // The first graph already exists
        createGraph();
    }

    for (unsigned graphIndex = 0; graphIndex < numOfGraphs; graphIndex++)
    {
        unsigned biasTensor         = createPersistTensor(INPUT_TENSOR,
                                                  MEM_INIT_ALL_ONES,
                                                  nullptr,
                                                  biasSizes,
                                                  TENSOR_DIMS,
                                                  syn_type_float,
                                                  nullptr,
                                                  "Persistent Tensor",
                                                  graphIndex);
        unsigned biasShape          = createShapeTensor(INPUT_TENSOR,
                                               maxpoolOutMaxSizes,
                                               maxpoolOutMinSizes,
                                               TENSOR_DIMS,
                                               syn_type_float,
                                               "Shape Tensor",
                                               graphIndex);
        unsigned broadcastOutTensor = createTensor(OUTPUT_TENSOR,
                                                   MEM_INIT_ALL_ZERO,
                                                   nullptr,
                                                   maxpoolOutMaxSizes,
                                                   TENSOR_DIMS,
                                                   syn_type_float,
                                                   nullptr,
                                                   maxpoolOutMinSizes,
                                                   graphIndex);

        addNodeToGraph(NodeFactory::broadcastNodeTypeName,
                       {biasTensor, biasShape},
                       {broadcastOutTensor},
                       nullptr,
                       0,
                       "Broadcast Node",
                       graphIndex);

        compileTopology("serialize_node_consistency_g" + std::to_string(graphIndex), graphIndex);
    }

    const recipe_t& base_recipe = *getGraph(0).recipeHandle->basicRecipeHandle.recipe;

    for (unsigned graphIndex = 1; graphIndex < numOfGraphs; graphIndex++)
    {
        const recipe_t& curr_recipe = *getGraph(graphIndex).recipeHandle->basicRecipeHandle.recipe;
        ASSERT_TRUE(compareRecipes(base_recipe, curr_recipe, true))
            << "recipe comparison failed, recipes are not equal";
    }
}
TEST_F_GC(SynGaudiSerializeTest, serialize_node_all_dim_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[] = {3, 3, 3, 3, 4};
    unsigned minSizes[] = {1, 1, 1, 1, 2};
    unsigned actualSizes[]  = {1, 2, 3, 2, 3};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addSerializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);

    runTopology(0);

    float* inData = castHostInBuffer<float>(inTensor);
    StridedToDenseBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2] * actualSizes[3] * actualSizes[4]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]) << "Failed index " << i;
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_node_first_dim_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[] = {4, 8, 4};
    unsigned minSizes[] = {2, 8, 4};
    unsigned actualSizes[]    = {3, 8, 4};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addSerializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    StridedToDenseBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_node_last_dim_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[] = {2, 8, 4, 2, 3};
    unsigned minSizes[] = {2, 8, 4, 2, 1};
    unsigned actualSizes[]    = {2, 8, 4, 2, 2};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addSerializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);

    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    StridedToDenseBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, deserialize_node, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[] = {2, 8, 4};
    unsigned minSizes[] = {2, 4, 4};
    unsigned actualSizes[]    = {2, 6, 4};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addDeserializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    DenseToStridedBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, deserialize_node_last_dim_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[] = {2, 8, 4, 2, 3};
    unsigned minSizes[] = {2, 8, 4, 2, 1};
    unsigned actualSizes[]    = {2, 8, 4, 2, 2};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addDeserializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);

    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    DenseToStridedBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, deserialize_node_first_dim_dynamic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    unsigned maxSizes[] = {4, 8, 4};
    unsigned minSizes[] = {2, 8, 4};
    unsigned actualSizes[]    = {3, 8, 4};

    unsigned dim = ARRAY_SIZE(maxSizes);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            maxSizes, dim, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, minSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             maxSizes, dim, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, minSizes);

    addDeserializeNode(inTensor, outTensor);
    compileTopology();

    setActualSizes(inTensor, actualSizes);
    setActualSizes(outTensor, actualSizes);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    DenseToStridedBuffer(inData, maxSizes, actualSizes, dim);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < actualSizes[0] * actualSizes[1] * actualSizes[2]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_pass_all_persistent, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "true");
    const unsigned tensorDim = 4;
    const unsigned actualBatch = 4;
    const unsigned minBatch = 2;
    const unsigned maxBatch = 10;
    unsigned H = 2;
    unsigned W = 64;
    unsigned C = 4;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, "Input",
                                            0, 0, nullptr, inMinSize);

    unsigned intermediateTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, "Intermediate",
                                             0, 0, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, "output",
                                             0, 0, nullptr, inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {intermediateTensor});
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {intermediateTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(intermediateTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    float* intermediateData = castHostOutBuffer<float>(intermediateTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < inActualSize[0] * inActualSize[1] * inActualSize[2] * inActualSize[3]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
        ASSERT_EQ(intermediateData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_pass_intermediate_non_persistent, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "true");
    const unsigned tensorDim = 4;
    const unsigned actualBatch = 4;
    const unsigned minBatch = 2;
    const unsigned maxBatch = 10;
    unsigned H = 2;
    unsigned W = 64;
    unsigned C = 4;

    unsigned inMaxSize[] = {C, W, H, maxBatch};
    unsigned inMinSize[] = {C, W, H, minBatch};
    unsigned inActualSize[] = {C, W, H, actualBatch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, "Input",
                                            0, 0, nullptr, inMinSize);

    unsigned intermediateTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                     inMaxSize, tensorDim, syn_type_single, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, "output",
                                             0, 0, nullptr, inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {intermediateTensor});
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {intermediateTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < inActualSize[0] * inActualSize[1] * inActualSize[2] * inActualSize[3]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_pass_mid_dim_all_persistent, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "true");
    const unsigned tensorDim = 4;
    const unsigned actualWidth = 6;
    const unsigned H = 2;
    const unsigned C = 2;
    const unsigned minWidth = 4;
    const unsigned maxWidth = 8;
    const unsigned batch = 4;

    unsigned inMaxSize[] = {C, maxWidth, H, batch};
    unsigned inMinSize[] = {C, minWidth, H, batch};
    unsigned inActualSize[] = {C, actualWidth, H, batch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, "Input",
                                            0, 0, nullptr, inMinSize);

    unsigned intermediateTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                      inMaxSize, tensorDim, syn_type_single, nullptr, "Intermediate",
                                                      0, 0, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, "Output",
                                             0, 0, nullptr, inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {intermediateTensor});
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {intermediateTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(intermediateTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);
    runTopology(0, true);

    float* inData = castHostOutBuffer<float>(inTensor);
    float* outData = castHostOutBuffer<float>(outTensor);
    float* intermediateData = castHostOutBuffer<float>(intermediateTensor);

    for (int i = 0; i < inActualSize[0] * inActualSize[1] * inActualSize[2] * inActualSize[3]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]) << i;
        ASSERT_EQ(intermediateData[i], inData[i]) << i;
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_pass_mid_dim, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "true");
    const unsigned tensorDim = 4;
    const unsigned actualWidth = 32;
    const unsigned H = 2;
    const unsigned C = 4;
    const unsigned minWidth = 16;
    const unsigned maxWidth = 64;
    const unsigned batch = 4;

    unsigned inMaxSize[] = {C, maxWidth, H, batch};
    unsigned inMinSize[] = {C, minWidth, H, batch};
    unsigned inActualSize[] = {C, actualWidth, H, batch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned intermediateTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                                      inMaxSize, tensorDim, syn_type_single, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {intermediateTensor});
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {intermediateTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);
    runTopology(0, true);

    float* inData = castHostInBuffer<float>(inTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < inActualSize[0] * inActualSize[1] * inActualSize[2] * inActualSize[3]; i++)
    {
        ASSERT_EQ(outData[i], inData[i]);
    }
}

TEST_F_GC(SynGaudiSerializeTest, serialize_pass_add_ctrl, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "true");
    const unsigned            tensorDim   = 4;
    const unsigned            actualWidth = 32;
    const unsigned            H           = 2;
    const unsigned            C           = 4;
    const unsigned            minWidth    = 16;
    const unsigned            maxWidth    = 64;
    const unsigned            batch       = 4;

    unsigned inMaxSize[]    = {C, maxWidth, H, batch};
    unsigned inMinSize[]    = {C, minWidth, H, batch};
    unsigned inActualSize[] = {C, actualWidth, H, batch};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSize,
                                            tensorDim,
                                            syn_type_single,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSize);

    unsigned numElements        = inMaxSize[0] * inMaxSize[1] * inMaxSize[2] * inMaxSize[3];
    unsigned section            = createSection(numElements * dataTypeSizeInBytes(syn_type_single));
    unsigned intermediateTensor = createPersistTensor(OUTPUT_TENSOR,
                                                      MEM_INIT_ALL_ZERO,
                                                      nullptr,
                                                      inMaxSize,
                                                      tensorDim,
                                                      syn_type_single,
                                                      nullptr,
                                                      nullptr,
                                                      0,
                                                      0,
                                                      &section,
                                                      inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             inMaxSize,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             &section,
                                             inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {inTensor}, {intermediateTensor});
    addNodeToGraph("relu_fwd_f32", {intermediateTensor}, {outTensor});

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, inActualSize);
    setActualSizes(intermediateTensor, inActualSize);
    setActualSizes(outTensor, inActualSize);
    runTopology(0, true);

    float* inData  = castHostInBuffer<float>(inTensor);
    float* outData = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < inActualSize[0] * inActualSize[1] * inActualSize[2] * inActualSize[3]; i++)
    {
        ASSERT_EQ(outData[i], std::max(inData[i], 0.f));
    }
}
