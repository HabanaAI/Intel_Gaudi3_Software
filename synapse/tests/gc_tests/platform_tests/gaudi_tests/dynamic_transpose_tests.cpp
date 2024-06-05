#include "gc_dynamic_shapes_infra.h"
#include "scoped_configuration_change.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "synapse_common_types.h"
class SynGaudiDynamicTranspose : public SynGaudiDynamicShapesTestsInfra
{

};

TEST_F_GC(SynGaudiDynamicTranspose,
          fcd_dynamic_transpose_with_flatten,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned inMaxSizes[] = {103, 51, 13};
    unsigned inMinSizes[] = {1, 51, 13};

    unsigned outMaxSizes[] = {13, 103, 51};
    unsigned outMinSizes[] = {13, 1, 51};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, 3, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);

    unsigned intermediateTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               outMaxSizes, 3, syn_type_float, nullptr, outMinSizes);


    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                             outMaxSizes, 3, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, outMinSizes);

    synTransposeParams transposeParams = {{TPD_Height, TPD_Channel, TPD_Width}, 3};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {intermediateTensor},
                   &transposeParams, sizeof (transposeParams));

    addNodeToGraph("relu_fwd_f32",
                   {intermediateTensor},
                   {outTensor});

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    TSize inActualSizes[] = {55, 51, 13};
    TSize outActualSizes[] = {13, 55, 51};
    setActualSizes(inTensor, {55, 51, 13});
    setActualSizes(outTensor, {13, 55, 51});

    auto* inData = castHostInBuffer<float>(inTensor);
    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);
    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 3);

    TensorPtr input = std::make_shared<Tensor>(3, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(3, outActualSizes, syn_type_float);

    NodePtr transposeRef = NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName , "");

    transposeRef->RunOnCpu();
    float *ref_resultArray = (float *) output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        if (ref_resultArray[i] < 0)
        {
            ref_resultArray[i] = 0;
        }
        ASSERT_EQ(out[i], ref_resultArray[i]);
    }
}

TEST_F_GC(SynGaudiDynamicTranspose, fcd_dynamic_transpose_dma_only, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Create the static part of the graph that is inserted as weights to the conv.

    unsigned inMaxSizes[] = {7, 56, 112};
    unsigned inMinSizes[] = {7, 56, 1};

    unsigned outMaxSizes[] = {56, 112, 7};
    unsigned outMinSizes[] = {56, 1, 7};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, 3, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);


    unsigned intermediateTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               outMaxSizes, 3, syn_type_float, nullptr, outMinSizes);


    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                             outMaxSizes, 3, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, outMinSizes);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Height, TPD_Channel }, 3};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {intermediateTensor},
                   &transposeParams, sizeof (transposeParams));


    addNodeToGraph("relu_fwd_f32",
                   {intermediateTensor},
                   {outTensor});

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    TSize inActualSizes[] = {7, 56, 110};
    TSize outActualSizes[] = {56, 110, 7};

    setActualSizes(inTensor, {7, 56, 110});
    setActualSizes(outTensor, {56, 110, 7});

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 3);

    TensorPtr input = std::make_shared<Tensor>(3, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(3, outActualSizes, syn_type_float);

    NodePtr transposeRef = NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName , "");

    transposeRef->RunOnCpu();
    float *ref_resultArray = (float *) output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        if (ref_resultArray[i] < 0)
        {
            ref_resultArray[i] = 0;
        }
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

TEST_F_GC(SynGaudiDynamicTranspose, reshape_dynamic_transpose)
{
    //This transpose should be using only reshape

    unsigned inMaxSizes[] = {50, 1, 40};
    unsigned inMinSizes[] = {50, 1, 1};

    unsigned outMaxSizes[] = {50, 40, 1};
    unsigned outMinSizes[] = {50, 1, 1};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, 3, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);


    unsigned intermediateTensor = createTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                               outMaxSizes, 3, syn_type_float, nullptr, outMinSizes);


    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                             outMaxSizes, 3, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, outMinSizes);

    synTransposeParams transposeParams = {{TPD_Channel, TPD_Height,  TPD_Width}, 3};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {intermediateTensor},
                   &transposeParams, sizeof (transposeParams));


    addNodeToGraph("relu_fwd_f32",
                   {intermediateTensor},
                   {outTensor});

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    TSize inActualSizes[] = {50, 1, 30};
    TSize outActualSizes[] = {50, 30, 1};

    setActualSizes(inTensor, {50, 1, 30});
    setActualSizes(outTensor, {50, 30, 1});

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 3);

    TensorPtr input = std::make_shared<Tensor>(3, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(3, outActualSizes, syn_type_float);

    NodePtr transposeRef = NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName , "");

    transposeRef->RunOnCpu();
    float *ref_resultArray = (float *) output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        if (ref_resultArray[i] < 0)
        {
            ref_resultArray[i] = 0;
        }
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

TEST_F_GC(SynGaudiDynamicTranspose, dynamic_transpose_only_logical, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    //This transpose should be using only reshape

    unsigned inMaxSizes[] = {4, 3, 2, 8};
    unsigned inMinSizes[] = {4, 3, 2, 1};

    unsigned outMaxSizes[] = {4, 3, 8, 2};
    unsigned outMinSizes[] = {4, 3, 1, 2};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                            inMaxSizes, 4, syn_type_float, nullptr, nullptr,
                                            0, 0, nullptr, inMinSizes);


    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr,
                                             outMaxSizes, 4, syn_type_float, nullptr, nullptr,
                                             0, 0, nullptr, outMinSizes);

    synTransposeParams transposeParams = {{TPD_Channel, TPD_Width,  TPD_4Dim_Batch, TPD_Height}, 4};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams, sizeof (transposeParams));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    TSize inActualSizes[] = {4, 3, 2, 6};
    TSize outActualSizes[] = {4, 3, 6, 2};

    setActualSizes(inTensor, {4, 3, 2, 6});
    setActualSizes(outTensor, {4, 3, 6, 2});

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 4);

    TensorPtr input = std::make_shared<Tensor>(4, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(4, outActualSizes, syn_type_float);

    NodePtr transposeRef = NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName , "");

    transposeRef->RunOnCpu();
    float *ref_resultArray = (float *) output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

TEST_F_GC(SynGaudiDynamicTranspose, dynamic_transpose_b_c)
{
    unsigned inMaxSizes[] = {4, 6, 1, 2};
    unsigned inMinSizes[] = {3, 6, 1, 1};

    unsigned outMaxSizes[] = {2, 6, 1, 4};
    unsigned outMinSizes[] = {1, 6, 1, 3};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            4,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             4,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_4Dim_Batch, TPD_Width, TPD_Height, TPD_Channel}, 4};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(inTensor, {3, 6, 1, 1});
    setActualSizes(outTensor, {1, 6, 1, 3});

    TSize inActualSizes[]  = {3, 6, 1, 1};
    TSize outActualSizes[] = {1, 6, 1, 3};

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 4);

    TensorPtr input  = std::make_shared<Tensor>(4, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(4, outActualSizes, syn_type_float);

    NodePtr transposeRef =
        NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName, "");

    transposeRef->RunOnCpu();
    float* ref_resultArray = (float*)output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

TEST_F_GC(SynGaudiDynamicTranspose, dynamic_transpose_c_b)
{
    unsigned outMaxSizes[] = {4, 6, 1, 2};
    unsigned outMinSizes[] = {3, 6, 1, 1};

    unsigned inMaxSizes[] = {2, 6, 1, 4};
    unsigned inMinSizes[] = {1, 6, 1, 3};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            4,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             4,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_4Dim_Batch, TPD_Width, TPD_Height, TPD_Channel}, 4};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(inTensor, {1, 6, 1, 3});
    setActualSizes(outTensor, {3, 6, 1, 1});

    TSize inActualSizes[]  = {1, 6, 1, 3};
    TSize outActualSizes[] = {3, 6, 1, 1};

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 4);

    TensorPtr input  = std::make_shared<Tensor>(4, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(4, outActualSizes, syn_type_float);

    NodePtr transposeRef =
        NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName, "");

    transposeRef->RunOnCpu();
    float* ref_resultArray = (float*)output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

TEST_F_GC(SynGaudiDynamicTranspose, fcd_dynamic_transpose_dma_only_orange)
{
    unsigned inMaxSizes[] = {2, 229376};
    unsigned inMinSizes[] = {1, 229376};

    unsigned outMaxSizes[] = {229376, 2};
    unsigned outMinSizes[] = {229376, 1};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            2,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             2,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, 2};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(inTensor, {1, 229376});
    setActualSizes(outTensor,{229376, 1});

    TSize inActualSizes[]  = {1, 229376};
    TSize outActualSizes[] = {229376, 1};

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActualSizes, inActualSizes + 2);

    TensorPtr input  = std::make_shared<Tensor>(2, inActualSizes, syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(2, outActualSizes, syn_type_float);

    NodePtr transposeRef =
        NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName, "");

    transposeRef->RunOnCpu();
    float* ref_resultArray = (float*)output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

TEST_F_GC(SynGaudiTwoRunCompareTest, dynamic_transpose_fully_utilized_ASIC_CI)
{
    unsigned inMaxSizes[] = {512, 512, 512, 4};
    unsigned inMinSizes[] = {512, 512, 512, 1};

    unsigned outMaxSizes[] = {512, 512, 512, 4};
    unsigned outMinSizes[] = {512, 512, 512, 1};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            4,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             4,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_Height, TPD_Channel, TPD_Width, TPD_4Dim_Batch}, 4};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    unsigned inActualSizes[4]  = {512, 512, 512, 2};
    unsigned outActualSizes[4] = {512, 512, 512, 2};

    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, outActualSizes);
    compareRunsResults({outTensor});
}

TEST_F_GC(SynGaudiTwoRunCompareTest,
          dynamic_transpose_fully_utilized_small,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange enableGaudi3DynamicShape("GAUDI3_DSD", "true");
    unsigned inMaxSizes[] = {128, 128, 128, 4};
    unsigned inMinSizes[] = {128, 128, 128, 1};

    unsigned outMaxSizes[] = {128, 128, 128, 4};
    unsigned outMinSizes[] = {128, 128, 128, 1};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            4,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             4,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_Height, TPD_Channel, TPD_Width, TPD_4Dim_Batch}, 4};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    unsigned inActualSizes[4]  = {128, 128, 128, 2};
    unsigned outActualSizes[4] = {128, 128, 128, 2};

    setActualSizes(inTensor, inActualSizes);
    setActualSizes(outTensor, outActualSizes);
    compareRunsResults({outTensor});
}

TEST_F_GC(SynGaudiTwoRunCompareTest, static_transpose_fully_utilized_ASIC_CI)
{
    unsigned inMaxSizes[] = {512, 512, 512, 4};
    unsigned inMinSizes[] = {512, 512, 512, 4};

    unsigned outMaxSizes[] = {512, 512, 512, 4};
    unsigned outMinSizes[] = {512, 512, 512, 4};

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            4,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             4,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    synTransposeParams transposeParams = {{TPD_Height, TPD_Channel, TPD_Width, TPD_4Dim_Batch}, 4};

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    compareRunsResults({outTensor});
}

class SynGaudiDynamicTransposeMany : public SynGaudiDynamicShapesTestsInfra,
                                     public testing::WithParamInterface<std::tuple<std::tuple<TestSizeVec,TestSizeVec,TestSizeVec>, /* sizes */
                                                                                   TestSizeVec /* permutation */>>
{
    public:

        TestSizeVec permuteSizes(const TestSizeVec& sizes, const TestSizeVec& perm)
        {
            TestSizeVec out = sizes;
            for (unsigned i = 0; i < sizes.size(); ++i)
                out[i] = sizes[perm[i]];
            return out;
        }
};

using ThreeSizes = std::tuple<TestSizeVec, TestSizeVec, TestSizeVec>;

TEST_P_GC(SynGaudiDynamicTransposeMany, simple, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{

    auto sizes       = std::get<0>(GetParam());
    auto perm        = std::get<1>(GetParam());

    auto inMaxSizes = std::get<0>(sizes);
    auto inMinSizes = std::get<1>(sizes);
    auto inActSizes = std::get<2>(sizes);

    auto outMaxSizes = permuteSizes(inMaxSizes, perm);
    auto outMinSizes = permuteSizes(inMinSizes, perm);
    auto outActSizes = permuteSizes(inActSizes, perm);

    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes.data(),
                                            inMaxSizes.size(),
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes.data());

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_NONE,
                                             nullptr,
                                             outMaxSizes.data(),
                                             outMaxSizes.size(),
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes.data());

    synTransposeParams transposeParams;
    std::transform(perm.begin(), perm.end(), transposeParams.permutation, [](int x) {return static_cast<TransposePermutationDim>(x);});
    transposeParams.tensorDim = perm.size();

    addNodeToGraph(NodeFactory::transposeNodeTypeName,
                   {inTensor},
                   {outTensor},
                   &transposeParams,
                   sizeof(transposeParams));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(inTensor, inActSizes.data());
    setActualSizes(outTensor, outActSizes.data());

    auto* inData = castHostInBuffer<float>(inTensor);

    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* out = castHostOutBuffer<float>(outTensor);

    auto totalActualSize = multiplyElements(inActSizes.data(), inActSizes.data() + 2);

    std::vector<TSize> inActHugeSizes(inActSizes.begin(), inActSizes.end());
    std::vector<TSize> outActHugeSizes(outActSizes.begin(), outActSizes.end());

    TensorPtr input  = std::make_shared<Tensor>(4, inActHugeSizes.data(), syn_type_float, (char*)inData);
    TensorPtr output = std::make_shared<Tensor>(4, outActHugeSizes.data(), syn_type_float);

    NodePtr transposeRef =
        NodeFactory::createNode({input}, {output}, &transposeParams, NodeFactory::transposeNodeTypeName, "");

    transposeRef->RunOnCpu();
    float* ref_resultArray = (float*)output->map();

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(ref_resultArray[i], out[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(dynamic, SynGaudiDynamicTransposeMany,
        ::testing::Combine(
            ::testing::Values(ThreeSizes{TestSizeVec {3, 4, 1, 2}, TestSizeVec {2, 4, 1, 1}, TestSizeVec {2, 4, 1, 1}},
                              ThreeSizes{TestSizeVec {8, 3, 1, 1}, TestSizeVec {4, 2, 1, 1}, TestSizeVec {8, 3, 1, 1}},
                              ThreeSizes{TestSizeVec {2, 3, 4, 5}, TestSizeVec {2, 2, 2, 2}, TestSizeVec {2, 2, 3, 3}},
                              ThreeSizes{TestSizeVec {3, 9, 1, 1}, TestSizeVec {2, 2, 1, 1}, TestSizeVec {2, 6, 1, 1}},
                              ThreeSizes{TestSizeVec {1, 1, 4, 4}, TestSizeVec {1, 1, 2, 2}, TestSizeVec {1, 1, 3, 3}},
                              ThreeSizes{TestSizeVec {8, 3, 1, 1}, TestSizeVec {4, 2, 1, 1}, TestSizeVec {4, 2, 1, 1}}),
            ::testing::Values(TestSizeVec {1,0,2,3},
                              TestSizeVec {0,2,1,3},
                              TestSizeVec {0,3,1,2},
                              TestSizeVec {0,3,2,1},
                              TestSizeVec {1,0,3,2},
                              TestSizeVec {1,3,2,0},
                              TestSizeVec {1,2,0,3},
                              TestSizeVec {1,2,3,0},
                              TestSizeVec {2,3,0,1},
                              TestSizeVec {2,1,0,3},
                              TestSizeVec {3,2,1,0},
                              TestSizeVec {3,0,1,2},
                              TestSizeVec {3,1,0,2})));
