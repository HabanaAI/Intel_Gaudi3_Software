#include "gc_dynamic_shapes_infra.h"
#include "habana_global_conf.h"

// This class handles tests for dynamic DMA, with both fully disabled and partial ROIs.
//
//

class SynGaudiDynamicDMATestMemcpySingle
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<std::tuple<unsigned, bool>>
{
};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicDMATestMemcpySingle, ::testing::Values(std::make_tuple(2, false),
                                                                                std::make_tuple(3, false),
                                                                                std::make_tuple(5, false),
                                                                                std::make_tuple(8, false),
                                                                                std::make_tuple(9, false),
                                                                                std::make_tuple(10, false)));

void SynGaudiDynamicDMATestMemcpyBase::createRecipe(unsigned graphIndex)
{
    params.inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                          params.inMaxSize, params.tensorDim, syn_type_single, nullptr, nullptr,
                                          graphIndex, 0, nullptr, params.inMinSize);

    params.outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                           params.inMaxSize, params.tensorDim, syn_type_single, nullptr, nullptr,
                                           graphIndex, 0, nullptr, params.inMinSize);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {params.inTensor}, {params.outTensor},
                   nullptr/*userParams*/, 0/*paramSize*/, nullptr/*nodeName*/, graphIndex);

    compileTopology("", graphIndex);

    ASSERT_NE(m_graphs[graphIndex].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[graphIndex].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);
}

void SynGaudiDynamicDMATestMemcpyBase::runRecipe(unsigned actualBatch, unsigned graphIndex)
{
    unsigned inActualSize[] = {params.C, params.W, params.H, actualBatch};
    setActualSizes(params.inTensor, inActualSize, graphIndex);
    setActualSizes(params.outTensor, inActualSize, graphIndex);
    runTopology(graphIndex, true);
}

void SynGaudiDynamicDMATestMemcpyBase::checkResults(unsigned actualBatch)
{
    float *inBuffer = castHostInBuffer<float>(params.inTensor);
    float *outBuffer = castHostOutBuffer<float>(params.outTensor);
    const uint64_t tensorBatchSizeElements = getNumberOfElements(params.inMinSize, params.lastDim);
    const uint64_t tensorSizeElements      = tensorBatchSizeElements * actualBatch;
    const uint64_t garbageElements         = tensorBatchSizeElements * (params.maxBatch - actualBatch);

    // Test by the actual batch size.
    for (int i = 0; i < tensorSizeElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, inBuffer[i])) << i;
        outBuffer[i] = 0;
    }
    for (int i = tensorSizeElements; i < tensorSizeElements + garbageElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, 0.0f)) << i;
    }
}

class SynGaudiDynamicDMATestMemcpy
: public SynGaudiDynamicDMATestMemcpyBase
, public testing::WithParamInterface<std::vector<unsigned>>
{
};

static const std::vector<std::vector<unsigned>> basic_dynamic_memcpyParams{{2},
                                                                    {3},
                                                                    {5},
                                                                    {8},
                                                                    {9},
                                                                    {10},
                                                                    {2, 5, 10, 9, 8, 3}};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicDMATestMemcpy, ::testing::ValuesIn(basic_dynamic_memcpyParams));

TEST_P_GC(SynGaudiDynamicDMATestMemcpy, basic_dynamic_memcpy, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    createRecipe();

    std::vector<unsigned> batches = GetParam();

    for (auto actualBatch : batches)
    {
        runRecipe(actualBatch);
        checkResults(actualBatch);
    }
}

TEST_P_GC(SynGaudiDynamicDMATestMemcpySingle, basic_dynamic_memcpy, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned tensorDim = 4;
    const unsigned lastDim = tensorDim - 1;
    const unsigned actualBatch = testing::get<0>(GetParam());
    const unsigned minBatch = 2;
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
    for(int i = 0; i < tensorSizeElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, inBuffer[i])) << i;
    }
    for (int i = tensorSizeElements; i < tensorSizeElements + garbageElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, 0.0f)) << i;
    }
}

class SynGaudiDynamicDMATestMemcpyMidDim : public SynGaudiDynamicDMATestMemcpySingle
{
protected:
    void afterSynInitialize() override
    {
        const bool bStagedSubmissionMode = testing::get<1>(GetParam());
        if (bStagedSubmissionMode)
        {
            synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
            synConfigurationSet("ENABLE_STAGED_SUBMISSION", "true");
        }
        SynGaudiDynamicDMATestMemcpySingle::afterSynInitialize();
    }
};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicDMATestMemcpyMidDim, ::testing::Values(std::make_tuple(16, false),
                                                                                std::make_tuple(31, false),
                                                                                std::make_tuple(32, false),
                                                                                std::make_tuple(33, false),
                                                                                std::make_tuple(63, false),
                                                                                std::make_tuple(64, false),
                                                                                std::make_tuple(16, true),
                                                                                std::make_tuple(31, true),
                                                                                std::make_tuple(32, true),
                                                                                std::make_tuple(33, true),
                                                                                std::make_tuple(63, true),
                                                                                std::make_tuple(64, true)));

TEST_P_GC(SynGaudiDynamicDMATestMemcpyMidDim, dynamic_memcpy_mid_dim, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    if (testing::get<1>(GetParam()))
    {
        HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test variant");
    }
    const unsigned tensorDim = 4;
    const unsigned actualWidth = testing::get<0>(GetParam());
    const unsigned H = 2;
    const unsigned C = 4;
    const unsigned minWidth = 16;
    const unsigned maxWidth = 64;
    const unsigned batch = 4;

    unsigned inMaxSize[] = {C, maxWidth, H, batch};
    unsigned inMinSize[] = {C, minWidth, H, batch};
    unsigned actualSize[] = {C, actualWidth, H, batch};

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

    setActualSizes(inTensor, actualSize);
    setActualSizes(outTensor, actualSize);
    runTopology(0, true);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    const uint64_t totalElements = getNumberOfElements(actualSize, tensorDim);
    for (uint64_t i = 0; i < totalElements; i++)
    {
        ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, inBuffer[i]));
    }
}

class SynGaudiDynamicTransposeTest
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<std::tuple<unsigned, unsigned, bool>>
{
protected:
    void afterSynInitialize() override
    {
        const bool bStagedSubmissionMode = testing::get<2>(GetParam());
        if (bStagedSubmissionMode)
        {
            synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
            synConfigurationSet("ENABLE_STAGED_SUBMISSION", "true");
        }
        SynGaudiDynamicShapesTestsInfra::afterSynInitialize();
    }
};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicTransposeTest, ::testing::Values(
            std::make_tuple(0, 64, false),
            std::make_tuple(0, 127, false),
            std::make_tuple(0, 128, false),
            std::make_tuple(0, 129, false),
            std::make_tuple(0, 256, false),
            std::make_tuple(1, 64, false),
            std::make_tuple(1, 127, false),
            std::make_tuple(1, 128, false),
            std::make_tuple(1, 129, false),
            std::make_tuple(1, 256, false),
            std::make_tuple(0, 64, true),
            std::make_tuple(0, 127, true),
            std::make_tuple(0, 128, true),
            std::make_tuple(0, 129, true),
            std::make_tuple(0, 256, true),
            std::make_tuple(1, 64, true),
            std::make_tuple(1, 127, true),
            std::make_tuple(1, 128, true),
            std::make_tuple(1, 129, true),
            std::make_tuple(1, 256, true)
            ));

TEST_P_GC(SynGaudiDynamicTransposeTest, dynamic_transpose, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    if (testing::get<2>(GetParam()))
    {
        HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test variant");
    }
    // Disable sram slicing to make the test simpler
    ScopedConfigurationChange enableInternalNodes("ENABLE_INTERNAL_NODES", "true");

    const unsigned tensorDim = 2;
    const unsigned dynamicDim = testing::get<0>(GetParam());
    const unsigned dynamicSize = testing::get<1>(GetParam());
    const unsigned maxSize = 256;
    const unsigned minSize = 64;

    unsigned inMaxSize[]     = {maxSize, maxSize,     1, 1, 1};
    unsigned inMinSize[]     = {minSize, maxSize,     1, 1, 1};
    unsigned outMinSize[]    = {maxSize, minSize,     1, 1, 1};
    unsigned actualInSize[]  = {dynamicSize, maxSize, 1, 1, 1};
    unsigned actualOutSize[] = {maxSize, dynamicSize, 1, 1, 1};

    if (dynamicDim == 1)
    {
        std::swap(inMinSize[0], inMinSize[1]) ;
        std::swap(outMinSize[0], outMinSize[1]) ;
        std::swap(actualInSize[0], actualInSize[1]);
        std::swap(actualOutSize[0], actualOutSize[1]);
    }

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, outMinSize);



    if(m_deviceType == synDeviceGaudi3)
    {
        synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};
        addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams, sizeof (transposeParams));
    } else
    {
        addNodeToGraph(NodeFactory::transposeDmaNodeTypeName, {inTensor}, {outTensor});
    }

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor, actualInSize);
    setActualSizes(outTensor, actualOutSize);
    runTopology(0, true);

    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    // Test actual elements copied
    for(int i = 0; i < actualInSize[1]; ++i)
    {
        for (int j = 0; j < actualInSize[0]; ++j)
        {
            auto inIdx = i * actualInSize[0] + j;
            auto outIdx = j * actualInSize[1] + i;
            ASSERT_EQ(inBuffer[inIdx],outBuffer[outIdx]);
        }
    }
}

class SynGaudiDynamicDMATestMemset
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<unsigned>
{
public:

    void RunTest(unsigned sizes[5], unsigned tensorDim,
                 unsigned minSize, unsigned maxSize, unsigned sizeFactor)
    {
        const unsigned dynamicDim = 0;
        const unsigned actualSize = GetParam() * sizeFactor;

        unsigned outMaxSizes[5], outMinSizes[5], outActualSizes[5];
        std::copy(sizes, sizes+tensorDim, outMinSizes);
        std::copy(sizes, sizes+tensorDim, outMaxSizes);
        std::copy(sizes, sizes+tensorDim, outActualSizes);

        outMinSizes[dynamicDim] = minSize * sizeFactor;
        outMaxSizes[dynamicDim] = maxSize * sizeFactor;
        outActualSizes[dynamicDim] = actualSize;

        unsigned inputShapeTensor = createShapeTensor(INPUT_TENSOR, outMaxSizes,
                                                      outMinSizes, tensorDim, syn_type_single);
        unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                                 outMaxSizes, tensorDim, syn_type_single, nullptr, nullptr,
                                                 0, 0, nullptr, outMinSizes);

        addNodeToGraph(NodeFactory::memsetNodeTypeName, {inputShapeTensor}, {outTensor});

        compileTopology();

        ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
        shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
        ASSERT_NE(recipe, nullptr);

        setActualSizes(inputShapeTensor, outActualSizes);
        setActualSizes(outTensor, outActualSizes);

        runTopology(0, true);

        float* outBuffer = castHostOutBuffer<float>(outTensor);
        const uint64_t tensorTotalElements = getNumberOfElements(outActualSizes, tensorDim);
        const uint64_t tensorMaxElements   = getNumberOfElements(outMaxSizes, tensorDim);

        // Test by the actual batch size.
        for (uint64_t i = 0; i < tensorTotalElements; i++)
        {
            ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, 0.0f)) << i;
        }
        for (uint64_t i = tensorTotalElements; i < tensorMaxElements; i++)
        {
            ASSERT_EQ(std::make_pair(i, outBuffer[i]), std::make_pair(i, 1.0f)) << i;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(, SynGaudiDynamicDMATestMemset, ::testing::Values(2, 4, 5, 9, 12));

TEST_P_GC(SynGaudiDynamicDMATestMemset, basic_dynamic_memset_4d, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned sizes[] = {0, 16, 16, 256};

    RunTest(sizes, sizeof(sizes)/sizeof(sizes[0]), 2, 12, 1);
}

TEST_P_GC(SynGaudiDynamicDMATestMemset, basic_dynamic_memset_1d, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned sizes[1] = {0};

    // Tensor sizes in this test are such that there is more than one
    // ROI across the dynamic dimension, so there are descriptors
    // that actually get fully disabled.

    RunTest(sizes, sizeof(sizes)/sizeof(sizes[0]), 2, 12, 4096);
}
