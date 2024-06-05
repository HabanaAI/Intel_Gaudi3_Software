#include "gc_dynamic_shapes_infra.h"
#include "gc_dynamic_shapes_infra.h"
#include "recipe_handle_impl.hpp"
#include "scoped_configuration_change.h"
// This class handles tests for physical concatenation
//

class ConcatWithShapeTest
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<unsigned>
{
};

class LogicalConcatWithShapeTest : public SynGaudiDynamicShapesTestsInfra
{
};

// TODO add more sizes when dynamicity is supported
INSTANTIATE_TEST_SUITE_P(, ConcatWithShapeTest, ::testing::Values(2, 3, 7, 8));

TEST_P_GC(ConcatWithShapeTest, basic_concat_with_shape, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");

    const unsigned tensorDim = 4;

    const unsigned concatDim = 1;

    const unsigned inMinDynamicSize = 2;
    const unsigned inMaxDynamicSize = 10;

    // skew it so that in max size no longer sums up to out max size
    // and in min size no longer sums up to in min size
    const unsigned outMinDynamicSize = 3 * inMinDynamicSize - 3;
    const unsigned outMaxDynamicSize = 3 * inMaxDynamicSize + 6;

    unsigned H = 2;
    unsigned C = 2;
    unsigned B = 2;

    unsigned inTensorMaxSize[] = {C, inMaxDynamicSize, H, B};
    unsigned inTensorMinSize[] = {C, inMinDynamicSize, H, B};

    const unsigned actualDynamicSizeSmall = GetParam();
    unsigned       actualDynamicSizeNormy = std::min(actualDynamicSizeSmall + 1, inMaxDynamicSize);
    unsigned       actualDynamicSizeLarge = std::min(actualDynamicSizeSmall + 2, inMaxDynamicSize);

    unsigned inActualSizeSmall[] = {C, actualDynamicSizeSmall, H, B};
    unsigned inActualSizeLarge[] = {C, actualDynamicSizeLarge, H, B};
    unsigned inActualSizeNormy[] = {C, actualDynamicSizeNormy, H, B};

    unsigned outMaxSize[]    = {C, outMaxDynamicSize, H, B};
    unsigned outMinSize[]    = {C, outMinDynamicSize, H, B};
    unsigned outActualSize[] = {C, 0, H, B};  // change it  below
    outActualSize[concatDim] =
        inActualSizeSmall[concatDim] + inActualSizeLarge[concatDim] + inActualSizeNormy[concatDim];

    std::vector<float> init1(inTensorMaxSize[0] * inTensorMaxSize[1] * inTensorMaxSize[2] * inTensorMaxSize[3]);
    std::vector<float> init2(inTensorMaxSize[0] * inTensorMaxSize[1] * inTensorMaxSize[2] * inTensorMaxSize[3]);
    std::vector<float> init3(inTensorMaxSize[0] * inTensorMaxSize[1] * inTensorMaxSize[2] * inTensorMaxSize[3]);

    for (auto i = 0; i < init1.size(); ++i)
    {
        init1[i] = 10000 + i;
        init2[i] = 20000 + i;
        init3[i] = 30000 + i;
    }

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outMaxSize, outMinSize, tensorDim);

    unsigned inTensor1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             init1.data(),
                                             inTensorMaxSize,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inTensorMinSize);

    unsigned inTensor2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             init2.data(),
                                             inTensorMaxSize,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inTensorMinSize);

    unsigned inTensor3 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             init3.data(),
                                             inTensorMaxSize,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inTensorMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             outMaxSize,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSize);

    unsigned parameter = concatDim;

    addNodeToGraph(NodeFactory::physicalConcatNodeTypeName,
                   {inTensor1, inTensor2, inTensor3, shapeTensor},
                   {outTensor},
                   (void*)&parameter,
                   sizeof(parameter));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor1, inActualSizeSmall);
    setActualSizes(inTensor2, inActualSizeLarge);
    setActualSizes(inTensor3, inActualSizeNormy);
    setActualSizes(outTensor, outActualSize);
    setActualSizes(shapeTensor, outActualSize);

    synRecipeSerialize(m_graphs[0].recipeHandle, "lol.recipe");

    runTopology(0, true);

    float* inBuffer1       = castHostInBuffer<float>(inTensor1);
    float* inBuffer2       = castHostInBuffer<float>(inTensor2);
    float* inBuffer3       = castHostInBuffer<float>(inTensor3);
    float* concatBuffers[] = {inBuffer1, inBuffer2, inBuffer3};
    float* outBuffer       = castHostOutBuffer<float>(outTensor);

    unsigned i[4];

    for (i[0] = 0; i[0] < outMaxSize[0]; ++i[0])
    {
        for (i[1] = 0; i[1] < outMaxSize[1]; ++i[1])
        {
            for (i[2] = 0; i[2] < outMaxSize[2]; ++i[2])
            {
                for (i[3] = 0; i[3] < outMaxSize[3]; ++i[3])
                {
                    auto outIndex = i[0] + i[1] * outMaxSize[0] + i[2] * outMaxSize[0] * outMaxSize[1] +
                                    i[3] * outMaxSize[0] * outMaxSize[1] * outMaxSize[2];
                    auto outElem = outBuffer[outIndex];

                    if (i[0] >= outActualSize[0] || i[1] >= outActualSize[1] || i[2] >= outActualSize[2] ||
                        i[3] >= outActualSize[3])
                    {
                        ASSERT_EQ(outElem, 0.0)
                            << "Indices of incorrect non-zero " << i[0] << " " << i[1] << " " << i[2] << " " << i[3];
                    }
                    else
                    {
                        unsigned j[4];
                        memcpy(j, i, sizeof(i));
                        unsigned concatIndex;
                        // calculate concatIndex and j[concatDim]
                        // the three rensor are concatenated in this order: small, large, normy
                        if (j[concatDim] >= inActualSizeSmall[concatDim] + inActualSizeLarge[concatDim])
                        {
                            j[concatDim] = i[concatDim] - (inActualSizeSmall[concatDim] + inActualSizeLarge[concatDim]);
                            concatIndex  = 2;
                        }
                        else if (j[concatDim] >= inActualSizeSmall[concatDim])
                        {
                            j[concatDim] = i[concatDim] - inActualSizeSmall[concatDim];
                            concatIndex  = 1;
                        }
                        else
                        {
                            j[concatDim] = i[concatDim];
                            concatIndex  = 0;
                        }

                        auto inIndex = j[0] + j[1] * inTensorMaxSize[0] +
                                       j[2] * inTensorMaxSize[0] * inTensorMaxSize[1] +
                                       j[3] * inTensorMaxSize[0] * inTensorMaxSize[1] * inTensorMaxSize[2];
                        auto inElem = concatBuffers[concatIndex][inIndex];
                        ASSERT_EQ(outElem, inElem) << "Indices of incorrect element " << j[0] << " " << j[1] << " "
                                                   << j[2] << " " << j[3] << " tensor " << concatIndex;
                    }
                }
            }
        }
    }
}

TEST_F_GC(LogicalConcatWithShapeTest, logical_concat_with_shape)
{
    const unsigned tensorDim = 2;
    const unsigned concatDim = 1;
    const unsigned AMax      = 7;
    const unsigned AMin      = 3;
    const unsigned AAct      = 5;
    const unsigned B         = 4;

    unsigned inSizeMax[2]  = {AMax, B};
    unsigned outSizeMax[2] = {AMax, 2 * B};
    unsigned inSizeMin[2]  = {AMin, B};
    unsigned outSizeMin[2] = {AMin, 2 * B};
    unsigned inSizeAct[2]  = {AAct, B};
    unsigned outSizeAct[2] = {AAct, 2 * B};

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outSizeMax, outSizeMin, tensorDim);

    unsigned inTensor1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             inSizeMax,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inSizeMin);

    unsigned inTensor2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             inSizeMax,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inSizeMin);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_NONE,
                                             nullptr,
                                             outSizeMax,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outSizeMin);

    unsigned parameter = concatDim;

    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {inTensor1, inTensor2, shapeTensor},
                   {outTensor},
                   (void*)&parameter,
                   sizeof(parameter));

    compileTopology();
    setActualSizes(inTensor1, inSizeAct);
    setActualSizes(inTensor2, inSizeAct);
    setActualSizes(shapeTensor, outSizeAct);
    setActualSizes(outTensor, outSizeAct);
    runTopology();

    float* inBuffer1 = castHostInBuffer<float>(inTensor1);
    float* inBuffer2 = castHostInBuffer<float>(inTensor2);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < 2 * B; ++i)
    {
        for (int j = 0; j < AAct; ++j)
        {
            auto in  = i < B ? inBuffer1[AAct * i + j] : inBuffer2[AAct * (i - B) + j];
            auto out = outBuffer[AAct * i + j];
            ASSERT_EQ(out, in) << "Incorrect value at indices [" << i << "," << j << "]";
        }
    }
}

TEST_F_GC(LogicalConcatWithShapeTest, fcd_concat_with_shape)
{
    const unsigned tensorDim = 2;
    const unsigned concatDim = 0;
    const unsigned A         = 5;
    const unsigned B         = 4;

    unsigned inSize[2]  = {B, A};
    unsigned outSize[2] = {2 * B, A};

    unsigned shapeTensor = createShapeTensor(INPUT_TENSOR, outSize, outSize, tensorDim);

    unsigned inTensor1 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSize, tensorDim, syn_type_single);

    unsigned inTensor2 =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSize, tensorDim, syn_type_single);

    unsigned outTensor =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_NONE, nullptr, outSize, tensorDim, syn_type_single);

    unsigned parameter = concatDim;

    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {inTensor1, inTensor2, shapeTensor},
                   {outTensor},
                   (void*)&parameter,
                   sizeof(parameter));

    compileAndRun();

    float* inBuffer1 = castHostInBuffer<float>(inTensor1);
    float* inBuffer2 = castHostInBuffer<float>(inTensor2);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    for (int i = 0; i < 2 * B; ++i)
    {
        for (int j = 0; j < A; ++j)
        {
            auto in  = i < B ? inBuffer1[i + j * B] : inBuffer2[i - B + j * B];
            auto out = outBuffer[i + j * B * 2];
            ASSERT_EQ(out, in) << "Incorrect value at indices [" << i << "," << j << "]";
        }
    }
}
