#include "gc_gaudi_test_infra.h"
#include "gc_dynamic_shapes_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

class SynGaudiDynamicBeam : public SynGaudiTestInfra
{
};

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_same_nim_max,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3, synDeviceGaudi3})
{
    unsigned voc_size        = 8200;
    unsigned hypothesesNum   = 2;
    unsigned N               = 2;
    unsigned valid_count_val = voc_size - 1;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]        = {voc_size, hypothesesNum, N, 1};
    unsigned validCountSizes[] = {hypothesesNum, N, 1, 1};
    unsigned outputSizes[]     = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize        = N * voc_size * hypothesesNum;
    const unsigned validCountSize = N * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];
    int32_t* validCountData   = new int32_t[validCountSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)]          = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
            validCountData[i + (k * hypothesesNum)] = valid_count_val;
        }
    }

    unsigned vocTensor          = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocSizes);
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocSizes);
    unsigned validCountTensor   = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    (float*)validCountData,
                                                    validCountSizes,
                                                    2,
                                                    syn_type_int32,
                                                    nullptr,
                                                    "validCount",
                                                    0,
                                                    0,
                                                    nullptr,
                                                    validCountSizes);
    unsigned scoresTensor       = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputSizes);
    unsigned indicesTensor      = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, validCountTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocSizes);
    setActualSizes(indicesInputTensor, vocSizes);
    setActualSizes(validCountTensor, validCountSizes);
    setActualSizes(scoresTensor, outputSizes);
    setActualSizes(indicesTensor, outputSizes);
    runTopology();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx             = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] validCountData;
    delete[] scoresResults;
    delete[] indicesResults;
}

// TODO enable Gaudi3 when [SW-140702] is solved.
TEST_F_GC(SynGaudiDynamicBeam, dynamic_beam_search_bitonic_sort_with_dynamic_k, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    unsigned voc_size        = 8200;
    unsigned hypothesesNum   = 2;
    unsigned N               = 2;
    unsigned valid_count_val = voc_size - 1;

    unsigned bsw       = 40;
    unsigned bswActual = 38;
    unsigned bswMin    = 38;

    synBeamParams params;
    params.bsw     = 0;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size, hypothesesNum, N, 1};
    unsigned validCountSizes[]   = {hypothesesNum, N, 1, 1};
    unsigned outputSizes[]       = {bsw, hypothesesNum, N, 1};
    unsigned outputMinSizes[]    = {bswMin, hypothesesNum, N, 1};
    unsigned kMaxSizes[]         = {bsw};
    unsigned kMinSizes[]         = {bswMin};
    unsigned kActualSizes[]      = {bswActual};
    unsigned kDim                = 1;
    unsigned outputActualSizes[] = {bswActual, hypothesesNum, N, 1};

    const unsigned vocSize        = N * voc_size * hypothesesNum;
    const unsigned validCountSize = N * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];
    int32_t* validCountData   = new int32_t[validCountSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)]          = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
            validCountData[i + (k * hypothesesNum)] = valid_count_val;
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr);

    unsigned validCountTensor = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    (float*)validCountData,
                                                    validCountSizes,
                                                    2,
                                                    syn_type_int32,
                                                    nullptr,
                                                    "validCount",
                                                    0,
                                                    0,
                                                    nullptr);

    unsigned kShapeTensor = createShapeTensor(INPUT_TENSOR, kMaxSizes, kMinSizes, kDim, syn_type_uint32, "t65");

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, validCountTensor, kShapeTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocSizes);
    setActualSizes(indicesInputTensor, vocSizes);
    setActualSizes(validCountTensor, validCountSizes);
    setActualSizes(kShapeTensor, kActualSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = N * hypothesesNum * bswActual;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bswActual; j++)
            {
                int idx             = j + (i * bswActual) + (k * hypothesesNum * bswActual);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] validCountData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_k_no_validCount,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3, synDeviceGaudi3})
{
    unsigned voc_size      = 8200;
    unsigned hypothesesNum = 2;
    unsigned N             = 2;

    unsigned bsw       = 40;
    unsigned bswActual = 38;
    unsigned bswMin    = 38;

    synBeamParams params;
    params.bsw     = 0;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[]       = {bsw, hypothesesNum, N, 1};
    unsigned outputMinSizes[]    = {bswMin, hypothesesNum, N, 1};
    unsigned kMaxSizes[]         = {bsw};
    unsigned kMinSizes[]         = {bswMin};
    unsigned kActualSizes[]      = {bswActual};
    unsigned kDim                = 1;
    unsigned outputActualSizes[] = {bswActual, hypothesesNum, N, 1};

    const unsigned vocSize = N * voc_size * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)]          = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr);

    unsigned kShapeTensor = createShapeTensor(INPUT_TENSOR, kMaxSizes, kMinSizes, kDim, syn_type_uint32, "t65");

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, INVALID_TENSOR_INDEX, kShapeTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(kShapeTensor, kActualSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = N * hypothesesNum * bswActual;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bswActual; j++)
            {
                int idx             = j + (i * bswActual) + (k * hypothesesNum * bswActual);
                scoresResults[idx]  = voc_size - 1 - j;
                indicesResults[idx] = 1000 + voc_size - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_k_no_validCount_1_dim,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test
    // Gets dynamic input tensors (without valid count) that voc_size_min < k max
    // And set in run time the actual size to min size

    unsigned voc_size        = 1224;
    unsigned voc_actal_sizes = 612;
    unsigned voc_size_min    = 2;

    unsigned bsw       = 800;
    unsigned bswActual = 400;
    unsigned bswMin    = 2;

    synBeamParams params;
    params.bsw     = 0;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size};
    unsigned vocMinSizes[]       = {voc_size_min};
    unsigned vocActalSizes[]     = {voc_actal_sizes};
    unsigned outputSizes[]       = {bsw};
    unsigned outputActualSizes[] = {bswActual};
    unsigned outputMinSizes[]    = {bswMin};
    unsigned kMaxSizes[]         = {bsw};
    unsigned kMinSizes[]         = {bswMin};
    unsigned kActualSizes[]      = {bswActual};
    unsigned kDim                = 1;

    const unsigned vocSize = voc_size;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int j = 0; j < voc_size; j++)
    {
        vocData[j]          = j;
        indicesInputData[j] = j + 1000;
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      1,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned kShapeTensor = createShapeTensor(INPUT_TENSOR, kMaxSizes, kMinSizes, kDim, syn_type_uint32, "t65");

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 1,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, INVALID_TENSOR_INDEX, kShapeTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocActalSizes);
    setActualSizes(indicesInputTensor, vocActalSizes);
    setActualSizes(kShapeTensor, kActualSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = bswActual;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int j = 0; j < totalOutputSize; j++)
    {
        int idx             = j;
        scoresResults[idx]  = voc_actal_sizes - 1 - j;
        indicesResults[idx] = 1000 + voc_actal_sizes - 1 - j;
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_k_no_validCount_1_dim_same_k,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test
    // Gets dynamic input tensors (without valid count) when the min and the max sizes
    // of dynamic input and K shape tensor are equal
    // And set in run time the actual size to min size

    unsigned voc_size        = 256;
    unsigned voc_actal_sizes = 254;
    unsigned voc_size_min    = 2;

    unsigned bsw       = 256;
    unsigned bswActual = 2;
    unsigned bswMin    = 2;

    synBeamParams params;
    params.bsw     = 0;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size};
    unsigned vocMinSizes[]       = {voc_size_min};
    unsigned vocActalSizes[]     = {voc_actal_sizes};
    unsigned outputSizes[]       = {bsw};
    unsigned outputActualSizes[] = {bswActual};
    unsigned outputMinSizes[]    = {bswMin};
    unsigned kMaxSizes[]         = {bsw};
    unsigned kMinSizes[]         = {bswMin};
    unsigned kActualSizes[]      = {bswActual};
    unsigned kDim                = 1;

    const unsigned vocSize = voc_size;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int j = 0; j < voc_size; j++)
    {
        vocData[j]          = j;
        indicesInputData[j] = j + 1000;
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      1,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned kShapeTensor = createShapeTensor(INPUT_TENSOR, kMaxSizes, kMinSizes, kDim, syn_type_uint32, "t65");

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 1,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, INVALID_TENSOR_INDEX, kShapeTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocActalSizes);
    setActualSizes(indicesInputTensor, vocActalSizes);
    setActualSizes(kShapeTensor, kActualSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = bswActual;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int j = 0; j < totalOutputSize; j++)
    {
        int idx             = j;
        scoresResults[idx]  = voc_actal_sizes - 1 - j;
        indicesResults[idx] = 1000 + voc_actal_sizes - 1 - j;
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam, dynamic_beam_search_bitonic_sort_with_dynamic_large_k, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic K
    // Gets dynamic K shape tensor (without valid count)
    // And set in run time the actual size to min size
    // K min is larger than input size
    // Expecting to slice via input size and not K

    unsigned voc_size        = 50;
    unsigned hypothesesNum   = 2;
    unsigned N               = 2;
    unsigned valid_count_val = voc_size;

    unsigned bsw       = 50;
    unsigned bswActual = 50;
    unsigned bswMin    = 40;

    synBeamParams params;
    params.bsw     = 0;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size, hypothesesNum, N, 1};
    unsigned validCountSizes[]   = {hypothesesNum, N, 1, 1};
    unsigned outputSizes[]       = {bsw, hypothesesNum, N, 1};
    unsigned outputMinSizes[]    = {bswMin, hypothesesNum, N, 1};
    unsigned kMaxSizes[]         = {bsw};
    unsigned kMinSizes[]         = {bswMin};
    unsigned kActualSizes[]      = {bswActual};
    unsigned kDim                = 1;
    unsigned outputActualSizes[] = {voc_size, hypothesesNum, N, 1};

    const unsigned vocSize        = N * voc_size * hypothesesNum;
    const unsigned validCountSize = N * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];
    int32_t* validCountData   = new int32_t[validCountSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)]          = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
            validCountData[i + (k * hypothesesNum)] = valid_count_val;
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr);

    unsigned validCountTensor = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    (float*)validCountData,
                                                    validCountSizes,
                                                    2,
                                                    syn_type_int32,
                                                    nullptr,
                                                    "validCount",
                                                    0,
                                                    0,
                                                    nullptr);

    unsigned kShapeTensor = createShapeTensor(INPUT_TENSOR, kMaxSizes, kMinSizes, kDim, syn_type_uint32, "t65");

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, validCountTensor, kShapeTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocSizes);
    setActualSizes(indicesInputTensor, vocSizes);
    setActualSizes(validCountTensor, validCountSizes);
    setActualSizes(kShapeTensor, kActualSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = N * hypothesesNum * voc_size;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                int idx             = j + (i * voc_size) + (k * hypothesesNum * voc_size);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] validCountData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam, dynamic_beam_search_bitonic_sort_with_dynamic_input, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test
    // Gets dynamic input tensors (without valid count)
    // And set in run time the actual size to min size

    unsigned voc_size        = 8200;
    unsigned hypothesesNum   = 4;
    unsigned N               = 2;
    unsigned valid_count_val = voc_size;

    unsigned bsw              = 40;
    unsigned minHypothesesNum = hypothesesNum / 2;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size, hypothesesNum, N, 1};
    unsigned vocMinSizes[]       = {voc_size, minHypothesesNum, N, 1};
    unsigned outputSizes[]       = {bsw, hypothesesNum, N, 1};
    unsigned outputMinSizes[]    = {bsw, minHypothesesNum, N, 1};
    unsigned outputActualSizes[] = {bsw, minHypothesesNum, N, 1};

    const unsigned vocSize = N * voc_size * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)]          = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    // pass nullptrs as the 2 optional tensors, to make sure that this case doesn't interfere with the extraction of
    // beam search (should be the same as not passing them, like done in the test below
    // called dynamic_beam_search_bitonic_sort_with_dynamic_input_100)
    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, INVALID_TENSOR_INDEX, INVALID_TENSOR_INDEX},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocMinSizes);
    setActualSizes(indicesInputTensor, vocMinSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = N * minHypothesesNum * bsw;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < minHypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx             = j + (i * bsw) + (k * minHypothesesNum * bsw);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_input_100,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test
    // Gets dynamic input tensors (without valid count)
    // And set in run time the actual size to min size

    unsigned voc_size        = 8200;
    unsigned hypothesesNum   = 4;
    unsigned N               = 2;
    unsigned valid_count_val = 100;

    unsigned bsw = 100;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned vocMinSizes[] = {valid_count_val, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize = N * voc_size * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < valid_count_val; j++)
            {
                // The actual size of the input will be according valid_count_val
                vocData[j + (i * valid_count_val) + (k * hypothesesNum * valid_count_val)]          = j;
                indicesInputData[j + (i * valid_count_val) + (k * hypothesesNum * valid_count_val)] = j + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocMinSizes);
    setActualSizes(indicesInputTensor, vocMinSizes);
    runTopology();

    const int totalOutputSize = N * hypothesesNum * valid_count_val;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < valid_count_val; j++)
            {
                int idx             = j + (i * valid_count_val) + (k * hypothesesNum * valid_count_val);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }
    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];
    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_input_dim_1,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test
    // Gets dynamic input tensors (without valid count)
    // And set in run time the actual size to min size

    unsigned voc_size        = 8200;
    unsigned voc_size_min    = 8000;
    unsigned valid_count_val = voc_size_min;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]          = {voc_size};
    unsigned vocMinSizes[]       = {voc_size_min};
    unsigned outputSizes[]       = {bsw};
    unsigned outputActualSizes[] = {bsw};

    const unsigned vocSize = voc_size;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int j = 0; j < voc_size; j++)
    {
        vocData[j]          = j;
        indicesInputData[j] = j + 1000;
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      1,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                1,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 1,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocMinSizes);
    setActualSizes(indicesInputTensor, vocMinSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = bsw;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int j = 0; j < bsw; j++)
    {
        int idx             = j;
        scoresResults[idx]  = valid_count_val - 1 - j;
        indicesResults[idx] = 1000 + valid_count_val - 1 - j;
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_valid_count,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test using valid count from the user
    // Gets dynamic input tensors with dynamic valid count
    // And set in run time the actual size to min size

    unsigned voc_size        = 8200;
    unsigned hypothesesNum   = 4;
    unsigned N               = 2;
    unsigned valid_count_val = voc_size;

    unsigned minHypothesesNum = hypothesesNum / 2;
    unsigned bsw              = 40;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]           = {voc_size, hypothesesNum, N, 1};
    unsigned vocMinSizes[]        = {voc_size, minHypothesesNum, N, 1};
    unsigned validCountSizes[]    = {hypothesesNum, N, 1, 1};
    unsigned validCountMinSizes[] = {minHypothesesNum, N, 1, 1};
    unsigned outputSizes[]        = {bsw, hypothesesNum, N, 1};
    unsigned outputMinSizes[]     = {bsw, minHypothesesNum, N, 1};
    unsigned outputActualSizes[]  = {bsw, minHypothesesNum, N, 1};

    const unsigned vocSize        = N * voc_size * hypothesesNum;
    const unsigned validCountSize = N * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];
    int32_t* validCountData   = new int32_t[validCountSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)]          = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
            validCountData[i + (k * hypothesesNum)] = valid_count_val;
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned validCountTensor = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    (float*)validCountData,
                                                    validCountSizes,
                                                    2,
                                                    syn_type_int32,
                                                    nullptr,
                                                    "validCount",
                                                    0,
                                                    0,
                                                    nullptr,
                                                    validCountMinSizes);

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, validCountTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocMinSizes);
    setActualSizes(indicesInputTensor, vocMinSizes);
    setActualSizes(validCountTensor, validCountMinSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = N * minHypothesesNum * bsw;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < minHypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx             = j + (i * bsw) + (k * minHypothesesNum * bsw);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] validCountData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiDynamicBeam,
          dynamic_beam_search_bitonic_sort_with_dynamic_valid_count_100,
          {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    // Bitonic sort dynamic input test using valid count from the user
    // Gets dynamic input tensors with dynamic valid count
    // And set in run time the actual size to min size
    // Valid count valu set to 100

    unsigned voc_size        = 8200;
    unsigned hypothesesNum   = 4;
    unsigned N               = 2;
    unsigned valid_count_val = 100;

    unsigned minHypothesesNum = hypothesesNum / 2;
    unsigned bsw              = 100;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]           = {voc_size, hypothesesNum, N, 1};
    unsigned vocMinSizes[]        = {valid_count_val, minHypothesesNum, N, 1};
    unsigned validCountSizes[]    = {hypothesesNum, N, 1, 1};
    unsigned validCountMinSizes[] = {minHypothesesNum, N, 1, 1};
    unsigned outputSizes[]        = {bsw, hypothesesNum, N, 1};
    unsigned outputMinSizes[]     = {bsw, minHypothesesNum, N, 1};
    unsigned outputActualSizes[]  = {bsw, minHypothesesNum, N, 1};

    const unsigned vocSize        = N * voc_size * hypothesesNum;
    const unsigned validCountSize = N * hypothesesNum;

    float*   vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];
    int32_t* validCountData   = new int32_t[validCountSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < valid_count_val; j++)
            {
                // The actual size of the input will be according valid_count_val
                vocData[j + (i * valid_count_val) + (k * hypothesesNum * valid_count_val)]          = j;
                indicesInputData[j + (i * valid_count_val) + (k * hypothesesNum * valid_count_val)] = j + 1000;
            }
            validCountData[i + (k * hypothesesNum)] = valid_count_val;
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc",
                                             0,
                                             0,
                                             nullptr,
                                             vocMinSizes);

    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices",
                                                      0,
                                                      0,
                                                      nullptr,
                                                      vocMinSizes);

    unsigned validCountTensor = createPersistTensor(INPUT_TENSOR,
                                                    MEM_INIT_FROM_INITIALIZER,
                                                    (float*)validCountData,
                                                    validCountSizes,
                                                    2,
                                                    syn_type_int32,
                                                    nullptr,
                                                    "validCount",
                                                    0,
                                                    0,
                                                    nullptr,
                                                    validCountMinSizes);

    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores",
                                                0,
                                                0,
                                                nullptr,
                                                outputMinSizes);

    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput",
                                                 0,
                                                 0,
                                                 nullptr,
                                                 outputMinSizes);

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor, validCountTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileTopology();
    setActualSizes(vocTensor, vocMinSizes);
    setActualSizes(indicesInputTensor, vocMinSizes);
    setActualSizes(validCountTensor, validCountMinSizes);
    setActualSizes(scoresTensor, outputActualSizes);
    setActualSizes(indicesTensor, outputActualSizes);
    runTopology();

    const int totalOutputSize = N * minHypothesesNum * bsw;
    float*    scoresResults   = new float[totalOutputSize];
    int32_t*  indicesResults  = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < minHypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx             = j + (i * bsw) + (k * minHypothesesNum * bsw);
                scoresResults[idx]  = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float*   pScoresOutput  = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(scoresResults[i])
            << " result received: " << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
            << "Wrong result at cell " << i << " correct result should be: " << static_cast<int>(indicesResults[i])
            << " result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] validCountData;
    delete[] scoresResults;
    delete[] indicesResults;
}
