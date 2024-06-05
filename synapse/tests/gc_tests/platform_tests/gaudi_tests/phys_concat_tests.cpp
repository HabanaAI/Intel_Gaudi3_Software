#include "gc_dynamic_shapes_infra.h"

// This class handles tests for physical concatenation
//

// For the time being the test is entirely static,
// until dynamicity in PhysicalConcatNode works
//
class SynGaudiPhysicalConcatTest
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<std::tuple<unsigned, bool>>
{
public:
    void            AllDimsPhysicalConcatTest(unsigned size, bool do_serialize);
    static unsigned i5d(const TestSizes& indices, const TestSizes& sizes);
    template<class it>
    static void fixParameters(const char* env, it from, it to);
    template<class it>
    static void printData(const char* title, it from, it to);

protected:
    void afterSynInitialize() override
    {
        const bool bStagedSubmissionMode = testing::get<1>(GetParam());
        if (bStagedSubmissionMode)
        {
            synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
            synConfigurationSet("ENABLE_STAGED_SUBMISSION", "true");
        }
        SynGaudiDynamicShapesTestsInfra::afterSynInitialize();
    }
};

// TODO add more sizes when dynamicity is supported
INSTANTIATE_TEST_SUITE_P(, SynGaudiPhysicalConcatTest, ::testing::Values(std::make_tuple(2, false),
                                                                        std::make_tuple(3, false),
                                                                        std::make_tuple(7, false),
                                                                        std::make_tuple(8, false),
                                                                        std::make_tuple(2, true),
                                                                        std::make_tuple(3, true),
                                                                        std::make_tuple(7, true),
                                                                        std::make_tuple(8, true)));

TEST_P_GC(SynGaudiPhysicalConcatTest, basic_physical_concat, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    if (testing::get<1>(GetParam()))
    {
        HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test variant");
    }
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS", "false");
    const unsigned tensorDim = 4;

    const unsigned concatDim = 1;

    const unsigned minDynamicSize = 2;
    const unsigned maxDynamicSize = 8;
    unsigned H = 2;
    unsigned C = 2;
    unsigned B = 2;
    const unsigned actualDynamicSize = testing::get<0>(GetParam());

    unsigned inMaxSize[]     = {C, maxDynamicSize,    H, B};

    unsigned inMinSize[]     = {C, minDynamicSize,    H, B};

    unsigned inActualSize[]  = {C, actualDynamicSize, H, B};

    unsigned actualDynamicSizeLarger  = std::min(actualDynamicSize+1, maxDynamicSize);

    unsigned inActualSizeLarger[]   = {C, actualDynamicSizeLarger,  H, B};

    unsigned outMaxSize[]    = {C, maxDynamicSize,    H, B};
    unsigned outMinSize[]    = {C, minDynamicSize,    H, B};
    unsigned outActualSize[] = {C, actualDynamicSize, H, B};

    outMaxSize[concatDim] = 3 * inMaxSize[concatDim];
    outMinSize[concatDim] = 2 * inMinSize[concatDim] + inMaxSize[concatDim];

    outActualSize[concatDim] = inActualSize[concatDim] + inActualSizeLarger[concatDim] + inMaxSize[concatDim];

    std::vector<float> init1(C*maxDynamicSize*H*B);
    std::vector<float> init2(C*maxDynamicSize*H*B);
    std::vector<float> init3(C*maxDynamicSize*H*B);

    for (auto i = 0; i < init1.size(); ++i)
    {
        init1[i] = 10000 + i;
        init2[i] = 20000 + i;
        init3[i] = 30000 + i;
    }

    unsigned inTensor1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, init1.data(),
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    // Check the case when a static-size tensor is concatenated to a DSD tensor
    unsigned inTensor2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, init2.data(),
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0);

    unsigned inTensor3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, init3.data(),
                                            inMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             outMaxSize, tensorDim, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, outMinSize);

    addNodeToGraph(NodeFactory::physicalConcatNodeTypeName,
                   {inTensor1, inTensor2, inTensor3},
                   {outTensor},
                   (void*)&concatDim,
                   sizeof(concatDim));

    compileTopology();

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    setActualSizes(inTensor1, inActualSize);
    setActualSizes(inTensor3, inActualSizeLarger);
    setActualSizes(outTensor, outActualSize);

    synRecipeSerialize(m_graphs[0].recipeHandle, "lol.recipe");

    runTopology(0, true);

    float* inBuffer1 = castHostInBuffer<float>(inTensor1);
    float* inBuffer2 = castHostInBuffer<float>(inTensor2);
    float* inBuffer3 = castHostInBuffer<float>(inTensor3);
    float* concatBuffers[] = { inBuffer1, inBuffer2, inBuffer3 };
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    unsigned i[4];

    for (i[0] = 0; i[0] < outMaxSize[0]; ++i[0])
    {
        for (i[1] = 0; i[1] < outMaxSize[1]; ++i[1])
        {
            for (i[2] = 0; i[2] < inMaxSize[2]; ++i[2])
            {
                for (i[3] = 0; i[3] < inMaxSize[3]; ++i[3])
                {
                    auto outIndex = i[0] + i[1]*outMaxSize[0] + i[2]*outMaxSize[0]*outMaxSize[1] + i[3]*outMaxSize[0]*outMaxSize[1]*outMaxSize[2];
                    auto outElem = outBuffer[outIndex];

                    if (i[0] >= outActualSize[0] ||
                        i[1] >= outActualSize[1] ||
                        i[2] >= outActualSize[2] ||
                        i[3] >= outActualSize[3])
                    {
                        ASSERT_EQ(outElem, 0.0) << "Indices of incorrect non-zero " << i[0] << " " << i[1] << " " << i[2] << " " << i[3];
                    }
                    else
                    {
                        unsigned j[4];
                        memcpy(j, i, sizeof(i));
                        unsigned concatIndex;
                        // calculate concatIndex and j[concatDim]
                        // the three rensor are concatenated in this order: inActualSize, inMaxSize, inActualSizeLarger
                        if (j[concatDim] >= inActualSize[concatDim] + inMaxSize[concatDim])
                        {
                            j[concatDim] = i[concatDim] - (inActualSize[concatDim] + inMaxSize[concatDim]);
                            concatIndex = 2;
                        }
                        else if (j[concatDim] >= inActualSize[concatDim])
                        {
                            j[concatDim] = i[concatDim] - inActualSize[concatDim];
                            concatIndex = 1;
                        }
                        else
                        {
                            j[concatDim] = i[concatDim];
                            concatIndex = 0;
                        }

                        auto inIndex = j[0] + j[1]*inMaxSize[0] + j[2]*inMaxSize[0]*inMaxSize[1] + j[3]*inMaxSize[0]*inMaxSize[1]*inMaxSize[2];
                        auto inElem = concatBuffers[concatIndex][inIndex];
                        ASSERT_EQ(outElem, inElem) << "Indices of incorrect element " << j[0] << " " << j[1] << " " << j[2] << " " << j[3] << " tensor " << concatIndex;
                    }
                }
            }
        }
    }
}

TEST_P_GC(SynGaudiPhysicalConcatTest, physical_concat_3_tensors_2_dyn_dims, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    struct tensor_size
    {
        TestSizes minSize;
        TestSizes maxSize;
        TestSizes actSize;
    };

    unsigned tensorDim = 2;
    unsigned size      = testing::get<0>(GetParam());  // 2,3,7,8

    tensor_size inSize1 = {{1, 12}, {size, 24}, {1, 12}};
    tensor_size inSize2 = {{4, 7}, {2 * size, 14}, {4, 7}};
    tensor_size inSize3 = {{4, 5}, {2 * size, 10}, {4, 5}};
    tensor_size outSize = {{5, 12}, {3 * size, 24}, {5, 12}};

    std::vector<float> init1(inSize1.maxSize[0] * inSize1.maxSize[1]);
    std::vector<float> init2(inSize2.maxSize[0] * inSize2.maxSize[1]);
    std::vector<float> init3(inSize3.maxSize[0] * inSize3.maxSize[1]);

    std::iota(init1.begin(), init1.end(), 1000);
    std::iota(init2.begin(), init2.end(), 2000);
    std::iota(init3.begin(), init3.end(), 3000);

    unsigned inTensor1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             init1.data(),
                                             inSize1.maxSize.data(),
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inSize1.minSize.data());

    unsigned inTensor2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             init2.data(),
                                             inSize2.maxSize.data(),
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inSize2.minSize.data());

    unsigned inTensor3 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_FROM_INITIALIZER,
                                             init3.data(),
                                             inSize3.maxSize.data(),
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             inSize3.minSize.data());

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             outSize.maxSize.data(),
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outSize.minSize.data());

    TestSizes intMaxSize = {2 * size, 24};
    unsigned  intTensor  = createTensor(OUTPUT_TENSOR,
                                      MEM_INIT_ALL_ZERO,
                                      nullptr,
                                      intMaxSize.data(),
                                      tensorDim,
                                      syn_type_single,
                                      nullptr,
                                      nullptr);

    const unsigned concatDim0 = 0;
    const unsigned concatDim1 = 1;

    addNodeToGraph(NodeFactory::physicalConcatNodeTypeName,
                   {inTensor1, intTensor},
                   {outTensor},
                   (void*)&concatDim0,
                   sizeof(concatDim0),
                   "physical_concat0");

    addNodeToGraph(NodeFactory::physicalConcatNodeTypeName,
                   {inTensor2, inTensor3},
                   {intTensor},
                   (void*)&concatDim1,
                   sizeof(concatDim1),
                   "physical_concat1");

    compileTopology();

    setActualSizes(inTensor1, inSize1.actSize.data());
    setActualSizes(inTensor2, inSize2.actSize.data());
    setActualSizes(inTensor3, inSize3.actSize.data());
    setActualSizes(outTensor, outSize.actSize.data());

    runTopology(0, true);

    float* outData = castHostOutBuffer<float>(outTensor);
    int    offset  = 0;
    for (int d1 = 0; d1 < inSize1.actSize[1]; d1++)
    {
        for (int d0 = 0; d0 < inSize1.actSize[0]; d0++)
        {
            ASSERT_EQ(init1[d0 + d1 * inSize1.actSize[0]], outData[offset + d0 + d1 * outSize.actSize[0]]);
        }
    }
    offset += inSize1.actSize[0];
    for (int d1 = 0; d1 < inSize2.actSize[1]; d1++)
    {
        for (int d0 = 0; d0 < inSize2.actSize[0]; d0++)
        {
            ASSERT_EQ(init2[d0 + d1 * inSize2.actSize[0]], outData[offset + d0 + d1 * outSize.actSize[0]]);
        }
    }
    offset += inSize2.actSize[1] * outSize.actSize[0];
    for (int d1 = 0; d1 < inSize3.actSize[1]; d1++)
    {
        for (int d0 = 0; d0 < inSize3.actSize[0]; d0++)
        {
            ASSERT_EQ(init3[d0 + d1 * inSize3.actSize[0]], outData[offset + d0 + d1 * outSize.actSize[0]]);
        }
    }
}

unsigned SynGaudiPhysicalConcatTest::i5d(const TestSizes& indices, const TestSizes& sizes)
{
    unsigned stride = 1;
    unsigned res = 0;
    for (unsigned k = 0; k < 5; ++k)
    {
        res += indices[k] * stride;
        stride *= sizes[k];
    }
    return res;
}

template <class it>
void SynGaudiPhysicalConcatTest::fixParameters(const char* env, it from, it to)
{
    const char* val = getenv(env);
    if (val == nullptr) return;
    std::istringstream in(val);
    auto cur = from;
    while (cur != to)
    {
        in >> *cur;
        ++cur;
    }

    std::cerr << "Fixed " << env << " : ";
    cur = from;

    while (cur != to)
    {
        std::cerr << *cur << " ";
        ++cur;
    }
    std::cerr << "\n";

}

template <class it>
void SynGaudiPhysicalConcatTest::printData(const char* title, it from, it to)
{
    if (getenv("PHYS_CONCAT_PRINT_DATA") == nullptr) return;

    std::cerr << title << "\n";
    while (from != to)
    {
        std::cerr << *from++ << " ";
    }
    std::cerr << "\n";
}

void SynGaudiPhysicalConcatTest::AllDimsPhysicalConcatTest(unsigned size, bool do_serialize)
{
    ScopedConfigurationChange serialize_pass("GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS",
            do_serialize ? "true" : "false");

    unsigned concatDim = 3;

    TestSizes maxSizes { size + 2, size + 2, size + 2, size + 2, size + 2 };
    fixParameters ("PHYS_CONCAT_MAX_SIZES", maxSizes.begin(), maxSizes.end());

    TestSizes actualSizes = { size + 1, size + 1, size + 1, size + 1, size + 1 };
    fixParameters ("PHYS_CONCAT_ACTUAL_SIZES", actualSizes.begin(), actualSizes.end());

    TestSizes minSizes = { size, size, size, size, size };
    fixParameters ("PHYS_CONCAT_MIN_SIZES", minSizes.begin(), minSizes.end());

    fixParameters ("PHYS_CONCAT_AXIS", &concatDim, &concatDim + 1);

    TestSizes inMaxSize     = maxSizes;
    TestSizes inMinSize     = minSizes;
    TestSizes inActualSize  = actualSizes;

    TestSizes outMaxSize    = maxSizes;
    TestSizes outMinSize    = minSizes;
    TestSizes outActualSize = actualSizes;

    outMinSize[concatDim] = inMinSize[concatDim] * 2;
    outMaxSize[concatDim] = inMaxSize[concatDim] * 2;
    outActualSize[concatDim] = inActualSize[concatDim] * 2;

    unsigned totalInBufferSize = inMaxSize[0]*inMaxSize[1]*inMaxSize[2]*inMaxSize[3]*inMaxSize[4];
    unsigned actualInBufferSize = inActualSize[0]*inActualSize[1]*inActualSize[2]*inActualSize[3]*inActualSize[4];
    unsigned totalOutBufferSize = totalInBufferSize * 2;

    std::vector<float> init1(totalInBufferSize);
    std::vector<float> init2(totalInBufferSize);

    for (unsigned i = 0; i < actualInBufferSize; ++i)
    {
        init1[i] = 10000+i;
        init2[i] = 20000+i;
    }

    if (!do_serialize)
    {
        DenseToStridedBuffer(init1.data(), inMaxSize.data(), inActualSize.data(), inActualSize.size());
        DenseToStridedBuffer(init2.data(), inMaxSize.data(), inActualSize.data(), inActualSize.size());
    }

    unsigned inTensor1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, init1.data(),
                                            inMaxSize.data(), 5, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize.data());
    unsigned inTensor2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, init2.data(),
                                            inMaxSize.data(), 5, syn_type_single, nullptr, nullptr,
                                            0, 0, nullptr, inMinSize.data());
    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                             outMaxSize.data(), 5, syn_type_single, nullptr, nullptr,
                                             0, 0, nullptr, outMinSize.data());

    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   {inTensor1, inTensor2},
                   {outTensor},
                   (void*)&concatDim,
                   sizeof(concatDim));

    compileTopology();

    setActualSizes(inTensor1, inActualSize.data());
    setActualSizes(inTensor2, inActualSize.data());
    setActualSizes(outTensor, outActualSize.data());

    runTopology(0, true);

    float* outData = castHostOutBuffer<float>(outTensor);

    auto outOffset = inActualSize[concatDim];
    for (unsigned i = 0; i < concatDim; ++i)
    {
            outOffset *= do_serialize ? inActualSize[i] : inMaxSize[i];
    }

    printData("IN_DATA_1", init1.begin(), init1.begin() + totalInBufferSize);
    printData("IN_DATA_2", init2.begin(), init2.begin() + totalInBufferSize);
    printData("OUT_DATA",  outData, outData + totalOutBufferSize);

    for (unsigned i5 = 0; i5 < inActualSize[4]; ++i5)
    {
        for (unsigned i4 = 0; i4 < inActualSize[3]; ++i4)
        {
            for (unsigned i3 = 0; i3 < inActualSize[2]; ++i3)
            {
                for (unsigned i2 = 0; i2 < inActualSize[1]; ++i2)
                {
                    for (unsigned i1 = 0; i1 < inActualSize[0]; ++i1)
                    {
                        auto inIdx  = i5d({i1, i2, i3, i4, i5}, do_serialize? inActualSize : inMaxSize);
                        auto outIdx = i5d({i1, i2, i3, i4, i5}, do_serialize? outActualSize : outMaxSize);
                        ASSERT_EQ(init1[inIdx], outData[outIdx]) << "FIRST BUFFER " << i1 << " " << i2 << " " << i3 << " " << i4 << " " << i5;
                        ASSERT_EQ(init2[inIdx], outData[outIdx+outOffset]) << "SECOND BUFFER " << i1 << " " << i2 << " " << i3 << " " <<   i4 << " " << i5;
                    }
                }
            }
        }
    }
}

TEST_P_GC(SynGaudiPhysicalConcatTest, physical_concat_all_dims_serialize, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned minSize = testing::get<0>(GetParam());

    AllDimsPhysicalConcatTest(minSize, true);
}

TEST_P_GC(SynGaudiPhysicalConcatTest, physical_concat_all_dims_no_serialize, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    const unsigned minSize = testing::get<0>(GetParam());

    AllDimsPhysicalConcatTest(minSize, false);
}
