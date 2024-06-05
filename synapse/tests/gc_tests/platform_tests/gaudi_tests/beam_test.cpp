#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

class SynTrainingBeamSearch : public SynTrainingTestInfra
{
};

class SynGaudiBeamSearch : public SynGaudiTestInfra
{
public:
    SynGaudiBeamSearch() { setSupportedDevices({synDeviceGaudi}); }
};

TEST_F_GC(SynTrainingBeamSearch, beam_search_use_full_bitonic_sort_no_merge)
{
    unsigned voc_size = 1024;
    unsigned hypothesesNum = 2;
    unsigned N = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = false;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;

    float* vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_single);
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indicesInputData, vocSizes, 3, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_single);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);

    addNodeToGraph("topk", {vocTensor, indicesInputTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = voc_size - 1 - j;
                indicesResults[idx] = 1000 + voc_size - 1 - j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    validateResult(scoresResults, pScoresOutput, totalOutputSize);
    validateResult(indicesResults, pIndicesOutput, totalOutputSize);

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynTrainingBeamSearch, beam_search_use_full_bitonic_sort_no_merge_smallest_elements_ascending_input)
{
    unsigned voc_size = 1024;
    unsigned hypothesesNum = 2;
    unsigned N = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = true;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;

    float* vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_single);
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indicesInputData, vocSizes, 3, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_single);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);

    addNodeToGraph("topk", {vocTensor, indicesInputTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = j;
                indicesResults[idx] = 1000 + j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    validateResult(scoresResults, pScoresOutput, totalOutputSize);
    validateResult(indicesResults, pIndicesOutput, totalOutputSize);

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynTrainingBeamSearch, beam_search_use_full_bitonic_sort_no_merge_smallest_elements_descending_input)
{
    unsigned voc_size = 1024;
    unsigned hypothesesNum = 2;
    unsigned N = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = true;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;

    float* vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = voc_size - 1 - j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = (voc_size - 1 - j) + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_single);
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indicesInputData, vocSizes, 3, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_single);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);

    addNodeToGraph("topk", {vocTensor, indicesInputTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = j;
                indicesResults[idx] = 1000 + j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    validateResult(scoresResults, pScoresOutput, totalOutputSize);
    validateResult(indicesResults, pIndicesOutput, totalOutputSize);

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiBeamSearch, beam_search_use_complete_bitonic_sort_with_merge)
{
    unsigned voc_size = 8196 * 4 + 11; /*max chunk size is 8196, make sure two chunks for use in bitonic merge */;
    unsigned hypothesesNum = 2;
    unsigned N = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = false;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;

    float* vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_single);
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indicesInputData, vocSizes, 3, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_single);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);

    addNodeToGraph("topk", {vocTensor, indicesInputTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = voc_size - 1 - j;
                indicesResults[idx] = 1000 + voc_size - 1 - j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
                                << "Wrong result at cell " << i << " correct result should be: " <<
                                static_cast<int>(scoresResults[i]) << " result received: "
                                << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
                                << "Wrong result at cell " << i << " correct result should be: " <<
                                static_cast<int>(indicesResults[i]) << " result received: "
                                << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiBeamSearch, beam_search_use_complete_bitonic_sort_no_indices_input)
{
    unsigned voc_size = 8196 * 4 + 11; /*max chunk size is 8196, make sure two chunks for use in bitonic merge */;
    unsigned hypothesesNum = 2;
    unsigned N = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = false;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;

    float* vocData          = new float[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);

    addNodeToGraph("topk", {vocTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = voc_size - 1 - j;
                indicesResults[idx] = voc_size - 1 - j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
                                << "Wrong result at cell " << i << " correct result should be: " <<
                                static_cast<int>(scoresResults[i]) << " result received: "
                                << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
                                << "Wrong result at cell " << i << " correct result should be: " <<
                                static_cast<int>(indicesResults[i]) << " result received: "
                                << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynTrainingBeamSearch, beam_search_use_complete_bitonic_sort_no_indices_input_smallest_elements)
{
    unsigned voc_size = 8196 * 4 + 11; /*max chunk size is 8196, make sure two chunks for use in bitonic merge */;
    unsigned hypothesesNum = 2;
    unsigned N = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = true;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;

    float* vocData          = new float[vocSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j;
            }
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32);

    addNodeToGraph("topk", {vocTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = j;
                indicesResults[idx] = j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
                            << "Wrong result at cell " << i << " correct result should be: " <<
                            static_cast<int>(scoresResults[i]) << " result received: "
                            << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
                            << "Wrong result at cell " << i << " correct result should be: " <<
                            static_cast<int>(indicesResults[i]) << " result received: "
                            << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiBeamSearch, beam_search_use_small_k_L2)
{
    unsigned voc_size      = 50;
    unsigned hypothesesNum = 16;

    unsigned bsw = 8;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 1;

    unsigned vocSizes[]    = {hypothesesNum, voc_size, 1, 1};
    unsigned outputSizes[] = {hypothesesNum, bsw, 1, 1};

    const unsigned vocSize = voc_size * hypothesesNum;

    float* vocData = new float[vocSize];

    // fill each line with (49-line index)
    for (int i = 0; i < vocSize; i++)
    {
        vocData[i] = 49 - ( i / hypothesesNum);
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 2, syn_type_single);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 2, syn_type_single);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 2, syn_type_int32);

    addNodeToGraph("topk", {vocTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileAndRun();

    std::vector<int8_t> indicesResultVec;
    std::vector<int8_t> scoresResultVec;

    for (unsigned i = 0; i < bsw; i++)
    {
        indicesResultVec.insert(indicesResultVec.end(), hypothesesNum, i);
    }

    for (unsigned i = 0; i < bsw; i++)
    {
        scoresResultVec.insert(scoresResultVec.end(), hypothesesNum, 49-i);
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < bsw * hypothesesNum; i++)
    {
        ASSERT_EQ(indicesResultVec.at(i), pIndicesOutput[i]) << "Wrong result at cell " << i << " correct result should be: " <<
                                                             static_cast<int>(indicesResultVec.at(i)) <<" result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    for (int i = 0; i < bsw * hypothesesNum; i++)
    {
        ASSERT_EQ(scoresResultVec.at(i), pScoresOutput[i]) << "Wrong result at cell " << i << " correct result should be: " <<
                                                           static_cast<int>(scoresResultVec.at(i)) <<" result received: " << static_cast<int>(pScoresOutput[i]);
    }

    delete[] vocData;
}

TEST_F_GC(SynGaudiBeamSearch, beam_search_small_k_int32)
{
    unsigned voc_size      = 50;
    unsigned hypothesesNum = 16;

    unsigned bsw = 8;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, 1, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, 1, 1};

    const unsigned vocSize = voc_size * hypothesesNum;

    float* vocData = new float[vocSize];

    // fill each row (0-axis) with 16 times the decreasing index (49's row, 48's row...)
    for (int i = 0; i < vocSize; i++)
    {
        vocData[i] = 49 - (i % voc_size);
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 2, syn_type_int32);
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 2, syn_type_int32);
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 2, syn_type_int32);

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;

    addNodeToGraph("topk", {vocTensor}, {scoresTensor, indicesTensor}, &params, sizeof(synBeamParams));

    compileTopology();
    runTopology(0, true);

    std::vector<int8_t> indicesResultVec;
    std::vector<int8_t> scoresResultVec;

    std::list<int32_t> indicesResultList({0,1,2,3,4,5,6,7});
    for (unsigned i = 0; i < hypothesesNum; i++)
    {
        indicesResultVec.insert(indicesResultVec.end(), indicesResultList.begin(), indicesResultList.end());
    }

    std::list<float> scoresResultList({49,48,47,46,45,44,43,42});
    for (unsigned i = 0; i < hypothesesNum; i++)
    {
        scoresResultVec.insert(scoresResultVec.end(), scoresResultList.begin(), scoresResultList.end());
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < bsw * hypothesesNum; i++)
    {
        ASSERT_EQ(indicesResultVec.at(i), pIndicesOutput[i]) << "Wrong result at cell " << i << " correct result should be: " <<
                                                             static_cast<int>(indicesResultVec.at(i)) <<" result received: " << static_cast<int>(pIndicesOutput[i]);
    }

    for (int i = 0; i < bsw * hypothesesNum; i++)
    {
        ASSERT_EQ(scoresResultVec.at(i), pScoresOutput[i]) << "Wrong result at cell " << i << " correct result should be: " <<
                                                           static_cast<int>(scoresResultVec.at(i)) <<" result received: " << static_cast<int>(pScoresOutput[i]);
    }

    delete[] vocData;
}

TEST_F_GC(SynGaudiBeamSearch, beam_search_use_complete_bitonic_sort_with_validCount)
{
    unsigned voc_size = 8200;
    unsigned hypothesesNum = 2;
    unsigned N = 2;
    unsigned valid_count_val = voc_size - 1;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw = bsw;
    params.axis = 0;
    params.bottomK = false;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned validCountSizes[]    = {hypothesesNum, N, 1, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

    const unsigned vocSize    = N * voc_size * hypothesesNum;
    const unsigned validCountSize    = N * hypothesesNum;

    float* vocData          = new float[vocSize];
    int32_t* indicesInputData = new int32_t[vocSize];
    int32_t* validCountData   = new int32_t[validCountSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < voc_size; j++)
            {
                vocData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j;
                indicesInputData[j + (i * voc_size) + (k * hypothesesNum * voc_size)] = j + 1000;
            }
            validCountData[i + (k * hypothesesNum)] = valid_count_val;
        }
    }

    unsigned vocTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, vocData, vocSizes, 3, syn_type_single,
                                             nullptr, "voc");
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indicesInputData, vocSizes, 3, syn_type_int32,
                                                      nullptr, "indices");
    unsigned validCountTensor = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)validCountData, validCountSizes, 2, syn_type_int32,
                                                    nullptr, "validCount");
    unsigned scoresTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_single,
                                                nullptr, "scores");
    unsigned indicesTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSizes, 3, syn_type_int32,
                                                 nullptr, "indicesOutput");

    addNodeToGraph("topk", {vocTensor, indicesInputTensor, validCountTensor}, {scoresTensor, indicesTensor}, &params,
                   sizeof(synBeamParams));

    compileAndRun();

    const int totalOutputSize = N * hypothesesNum * bsw;
    float* scoresResults = new float[totalOutputSize];
    int32_t* indicesResults = new int32_t[totalOutputSize];

    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < hypothesesNum; i++)
        {
            for (int j = 0; j < bsw; j++)
            {
                int idx = j + (i * bsw) + (k * hypothesesNum * bsw);
                scoresResults[idx] = valid_count_val - 1 - j;
                indicesResults[idx] = 1000 + valid_count_val - 1 - j;
            }
        }
    }

    float* pScoresOutput = (float*)m_hostBuffers[scoresTensor];
    int32_t* pIndicesOutput = (int32_t*)m_hostBuffers[indicesTensor];

    for (int i = 0; i < totalOutputSize; i++)
    {
        ASSERT_EQ(scoresResults[i], pScoresOutput[i])
                                << "Wrong result at cell " << i << " correct result should be: " <<
                                static_cast<int>(scoresResults[i]) << " result received: "
                                << static_cast<int>(pScoresOutput[i]);

        ASSERT_EQ(indicesResults[i], pIndicesOutput[i])
                                << "Wrong result at cell " << i << " correct result should be: " <<
                                static_cast<int>(indicesResults[i]) << " result received: "
                                << static_cast<int>(pIndicesOutput[i]);
    }

    delete[] vocData;
    delete[] indicesInputData;
    delete[] validCountData;
    delete[] scoresResults;
    delete[] indicesResults;
}

TEST_F_GC(SynGaudiBeamSearch, beam_search_use_complete_bitonic_sort_with_validCountShapeTensor)
{
    unsigned voc_size      = 8200;
    unsigned hypothesesNum = 2;
    unsigned N             = 2;

    unsigned bsw = 40;

    synBeamParams params;
    params.bsw     = bsw;
    params.axis    = 0;
    params.bottomK = false;

    unsigned vocSizes[]    = {voc_size, hypothesesNum, N, 1};
    unsigned outputSizes[] = {bsw, hypothesesNum, N, 1};

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

    unsigned vocTensor          = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             vocData,
                                             vocSizes,
                                             3,
                                             syn_type_single,
                                             nullptr,
                                             "voc");
    unsigned indicesInputTensor = createPersistTensor(INPUT_TENSOR,
                                                      MEM_INIT_FROM_INITIALIZER,
                                                      (float*)indicesInputData,
                                                      vocSizes,
                                                      3,
                                                      syn_type_int32,
                                                      nullptr,
                                                      "indices");
    unsigned scoresTensor       = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                outputSizes,
                                                3,
                                                syn_type_single,
                                                nullptr,
                                                "scores");
    unsigned indicesTensor      = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outputSizes,
                                                 3,
                                                 syn_type_int32,
                                                 nullptr,
                                                 "indicesOutput");

    addNodeToGraph("topk",
                   {vocTensor, indicesInputTensor},
                   {scoresTensor, indicesTensor},
                   &params,
                   sizeof(synBeamParams));

    compileAndRun();

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
