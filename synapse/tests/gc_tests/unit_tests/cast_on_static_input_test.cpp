#include "cast_utils.hpp"
#include "data_type_utils.h"
#include "graph_factory.h"
#include "graph_optimizer_test.h"
#include "generic_graph_test.h"
#include "layout.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "syn_singleton.hpp"
#include "tensor.h"
#include "test_utils.h"
#include "gaudi2_graph.h"
#include "transpose_node.h"

extern ConcurrentSlotMapAlloc<InternalSectionHandle> sectionHndlSlopMap;

class CastOnStaticTestCommon
{
protected:
    NodePtr   m_convNode         = nullptr;
    NodePtr   m_castOnStaticNode = nullptr;
    TensorPtr m_staticCastInput  = nullptr;
    TensorPtr m_staticCastOutput = nullptr;
    char*     m_castInputBuffer  = nullptr;
    void      createCastTensors(bool                      isInputBufferTypeFp32,
                                synDataType               castInputType,
                                synDataType               castOutputType,
                                const std::vector<TSize>& castSizes)
    {
        synDataType castInputBufferDataType = isInputBufferTypeFp32 ? syn_type_float : castInputType;
        auto weightsElementsNum = std::accumulate(castSizes.begin(), castSizes.end(), 1U, std::multiplies<TSize>());
        m_castInputBuffer =
            reinterpret_cast<char*>(allocateBufferForSynType(castInputBufferDataType, weightsElementsNum));
        m_staticCastInput = TensorPtr(new Tensor(castSizes.size(), castSizes.data(), castInputType, m_castInputBuffer));
        m_staticCastInput->setAsWeights();
        m_staticCastInput->setAsStaticParam(true);
        if (!isInputBufferTypeFp32)
        {
            m_staticCastInput->setAsDataTypeMatchData();
        }
        // weights tensor which is output to cast
        m_staticCastOutput = TensorPtr(new Tensor(castSizes.size(), castSizes.data(), castOutputType));
    }

    void createCastAndConvNodes(HabanaGraph& graph,
                                bool         isInputBufferTypeFp32,
                                synDataType  castInputType,
                                synDataType  castOutputType)
    {
        const TSize H = 5;
        const TSize W = 5;
        const TSize C = 2;
        const TSize N = 1;
        const TSize K = 2;
        const TSize R = 2;
        const TSize S = 2;

        const std::vector<TSize> inputSizes = {C, W, H, N};

        const TSize weightsStride  = 1;
        const TSize weightsPadding = 1;

        const std::vector<TSize> weightsSizes = {K, C, S, R};

        const TSize outW = ((W - R + 2 * weightsPadding) / weightsStride) + 1;
        const TSize outH = ((H - R + 2 * weightsPadding) / weightsStride) + 1;
        const TSize outC = K;
        //                                     C    W    H    N
        const std::vector<TSize> outSizes = {outC, outW, outH, N};

        synConvolutionParams params;
        params.dH   = weightsStride;
        params.dW   = weightsStride;
        params.kH   = S;
        params.kW   = R;
        params.padT = weightsPadding;
        params.padB = weightsPadding;
        params.padL = weightsPadding;

        params.padR = weightsPadding;
        params.dilH = 1;
        params.dilW = 1;

        createCastTensors(isInputBufferTypeFp32, castInputType, castOutputType, weightsSizes);
        const std::string castGuid = getCastGUID(castInputType, castOutputType);
        m_castOnStaticNode =
            NodeFactory::createGenericTPCNode({m_staticCastInput}, {m_staticCastOutput}, nullptr, castGuid.c_str());

        ASSERT_TRUE(GraphEditor::addNode(graph, m_castOnStaticNode));
        TensorPtr    convIn  = TensorPtr(new Tensor(4U, inputSizes.data(), syn_type_bf16));
        TensorPtr    convOut = TensorPtr(new Tensor(4U, outSizes.data(), syn_type_float));
        TensorVector inputs  = {convIn, m_staticCastOutput};
        TensorVector outputs = {convOut};

        m_convNode = NodeFactory::createNode({convIn, m_staticCastOutput},
                                             {convOut},
                                             &params,
                                             NodeFactory::convolutionNodeTypeName,
                                             "conv_node");

        ASSERT_TRUE(GraphEditor::addNode(graph, m_convNode));
    }
};

class CastOnStaticTest
: public GenericGraphTest
, public CastOnStaticTestCommon
{
protected:
    virtual void SetUp() override
    {
        GenericGraphTest::SetUp();
        m_graph->setInferenceMode(true);
        setGlobalConfForTest(GCFG_ENABLE_CONSTANT_FOLDING, "true");
    }
    virtual void TearDown() override
    {
        delete[] m_castInputBuffer;
        GenericGraphTest::TearDown();
    }
};

template<typename... Ts>
class CastOnStaticTupleTest
: public GenericTupleGraphTest<Ts...>
, public CastOnStaticTestCommon
{
protected:
    using TupleTestParent = GenericTupleGraphTest<Ts...>;

    virtual void SetUp() override
    {
        TupleTestParent::SetUp();
        TupleTestParent::m_graph->setInferenceMode(true);
        TupleTestParent::setGlobalConfForTest(GCFG_ENABLE_CONSTANT_FOLDING, "true");
    }
    virtual void TearDown() override
    {
        delete[] m_castInputBuffer;
        TupleTestParent::TearDown();
    }
};

class CastOnStaticSanityTest : public CastOnStaticTupleTest<bool, synDataType, synDataType>
{
};

TEST_P(CastOnStaticSanityTest, remove_cast_on_static_tensor)
{
    /*
     * Test case : static weights tensor is input to a cast node
     */
    createCastAndConvNodes(*m_graph, std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
    // verify cast node in graph
    synNodeId castId = m_castOnStaticNode->getId();
    ASSERT_TRUE(m_graph->getNodeByID(castId) != nullptr);
    ASSERT_EQ(m_graph->getNumNodes(), 2);
    // verify cast output is not static
    ASSERT_FALSE(m_staticCastOutput->isStaticParam());
    ASSERT_TRUE(eliminateNodesWithStaticInputs(*m_graph));
    // verify cast node was removed from graph
    ASSERT_FALSE(m_graph->getNodeByID(castId) != nullptr);
    ASSERT_EQ(m_graph->getNumNodes(), 1);
    // verify conv node is still valid
    ASSERT_TRUE(m_convNode->validateNode());
    // verify original cast output is still the conv input
    ASSERT_EQ(m_staticCastOutput->getId(), m_convNode->getInput(1)->getId());
    // verify it's static now
    ASSERT_TRUE(m_staticCastOutput->isStaticParam());
}
INSTANTIATE_TEST_SUITE_P(,
                         CastOnStaticSanityTest,
                         testing::Combine(testing::ValuesIn({true, false}),     // whether buffer data type is float
                                          testing::ValuesIn({syn_type_float}),  // cast from data type (cast input)
                                          testing::ValuesIn({syn_type_bf16}),   // cast to data type (cast output)
                                          ::testing::Values(synDeviceGaudi2, synDeviceGaudi3)),
                         CastOnStaticSanityTest::GetName());

class castFoldingFp8Test : public CastOnStaticTupleTest<synDataType>
{
};
TEST_P(castFoldingFp8Test, fp8_basic_folding_test)
{
    const std::vector<TSize> sizes        = {2, 3, 4, 1};
    const synDataType        castFromType = syn_type_float;
    const synDataType        castToType   = std::get<0>(GetParam());
    const std::string        castGuid     = getCastGUID(castFromType, castToType);
    const unsigned           elementsNum  = multiplyElements(sizes.begin(), sizes.end());
    std::unique_ptr<char[]>  inputBuffer(reinterpret_cast<char*>(allocateBufferForSynType(castFromType, elementsNum)));
    fillWithRandom(inputBuffer.get(), elementsNum, castFromType);
    TensorPtr castIn = TensorPtr(new Tensor(4U, sizes.data(), castFromType));
    castIn->setName("castIn");
    castIn->setAsStaticParam(true);
    castIn->setTensorBuffer(inputBuffer.get(), elementsNum * sizeof(float), castIn->getBufferDataType());
    TensorPtr castOut = TensorPtr(new Tensor(4U, sizes.data(), castToType));

    NodePtr cast = NodeFactory::createNode({castIn}, {castOut}, nullptr, castGuid, "cast");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, cast));

    TensorPtr idenOut = TensorPtr(new Tensor(4U, sizes.data(), castToType));
    NodePtr   idenNode =
        NodeFactory::createNode({castOut}, {idenOut}, nullptr, NodeFactory::identityNodeTypeName, "identity_node");
    ASSERT_TRUE(GraphEditor::addNode(*m_graph, idenNode));
    ASSERT_EQ(m_graph->getNumNodes(), 2);
    // verify cast output is not static
    ASSERT_FALSE(castOut->isStaticParam());
    ASSERT_TRUE(eliminateNodesWithStaticInputs(*m_graph));
    ASSERT_EQ(m_graph->getNumNodes(), 1);
    // verify identity node is still valid
    ASSERT_TRUE(idenNode->validateNode());
    // verify original cast output is still the identity input
    ASSERT_EQ(castOut->getId(), idenNode->getInput(0)->getId());
    // verify it's static now
    ASSERT_TRUE(castOut->isStaticParam());
}
INSTANTIATE_TEST_SUITE_P(,
                         castFoldingFp8Test,
                         testing::Combine(testing::ValuesIn({syn_type_fp8_152, syn_type_fp8_143}),
                                          ::testing::Values(synDeviceGaudi2, synDeviceGaudi3)),
                         castFoldingFp8Test::GetName());

class CastOnStaticResultsTest : public CastOnStaticTupleTest<synDataType, synDataType>
{
protected:
    using CastOnStaticTupleTestParent = CastOnStaticTupleTest<synDataType, synDataType>;
    virtual void TearDown() override
    {
        delete[] m_outputAsFloatBuffer;
        m_outputAsFloatBuffer = nullptr;
        CastOnStaticTupleTestParent::TearDown();
    }
    float* m_outputAsFloatBuffer = nullptr;
};

TEST_P(CastOnStaticResultsTest, cast_on_cpu_results)
{
    /*
     * Test case : Perform cast on cpu, cast input buffer type is fp32.
     *             Cast the result back to fp32 and verify that results are same as original cast input buffer.
     *             We cast back to fp32 since it's the easiest compare between different data types (all data
     *             types can cast to/from fp32).
     */
    const TSize        elementsNum    = 10;
    std::vector<TSize> castSizes      = {elementsNum};
    synDataType        castInputType  = std::get<0>(GetParam());
    synDataType        castOutputType = std::get<1>(GetParam());
    createCastTensors(true, castInputType, castOutputType, castSizes);
    fillWithRandom(reinterpret_cast<void*>(m_castInputBuffer), elementsNum, syn_type_float);
    // set buffer for the cast output tensor
    void* castOutputBuffer = allocateBufferForSynType(m_staticCastOutput->getElementType(), elementsNum);
    m_staticCastOutput->bind(castOutputBuffer, true);
    auto cpuCaster = CpuCaster(m_staticCastInput, m_staticCastOutput);
    m_staticCastOutput->setAsDataTypeMatchData();
    ASSERT_TRUE(cpuCaster.doCast());

    float* inputAsFloatBuffer = reinterpret_cast<float*>(m_staticCastInput->getData());
    // cast output back to fp32
    if (castOutputType != syn_type_float)
    {
        m_outputAsFloatBuffer         = reinterpret_cast<float*>(allocateBufferForSynType(syn_type_float, elementsNum));
        TensorPtr outputAsFloatTensor = TensorPtr(new Tensor(castSizes.size(),
                                                             castSizes.data(),
                                                             syn_type_float,
                                                             reinterpret_cast<char*>(m_outputAsFloatBuffer)));
        auto      outputAsFloatCaster = CpuCaster(m_staticCastOutput, outputAsFloatTensor);
        ASSERT_TRUE(outputAsFloatCaster.doCast());
    }
    else
    {
        m_outputAsFloatBuffer = reinterpret_cast<float*>(m_staticCastOutput->getData());
        m_staticCastOutput->setShouldFreeBuffer(false);  // buffer will be deleted by tearDown methods
    }
    // verify results are same
    for (unsigned i = 0; i < elementsNum; i++)
    {
        if (!float_eq(inputAsFloatBuffer[i], m_outputAsFloatBuffer[i]))
        {
            // if we arrive here the assert will fail and print the numbers
            EXPECT_EQ(inputAsFloatBuffer[i], m_outputAsFloatBuffer[i]);
        }
    }
}

// supported cast types for Gaudi2, excluding uints since they require detailed handling in quant params.
const std::vector<synDataType> typesForResults =
    {syn_type_int8, syn_type_int16, syn_type_int32, syn_type_bf16, syn_type_fp16, syn_type_float};

INSTANTIATE_TEST_SUITE_P(,
                         CastOnStaticResultsTest,
                         testing::Combine(testing::ValuesIn(typesForResults),  // cast from data type (cast input)
                                          testing::ValuesIn(typesForResults),  // cast to data type (cast output)
                                          ::testing::Values(synDeviceGaudi2, synDeviceGaudi3)),
                         CastOnStaticResultsTest::GetName());

class CastOnStaticResultsFp8Test
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<synDataType, synDataType, float, float, unsigned>>
{
};

TEST_P(CastOnStaticResultsFp8Test, cast_on_cpu_results_fp8)
{
    /*
     * Test case : Perform cast on cpu, cast input buffer type is fp32/bf16/int32.
     *             Cast the result back to original datatype and verify that results are same as original cast input
     * buffer.
     */
    using TensorBufferDtype                                   = float;
    constexpr TSize       elementsNum                         = 1;
    constexpr std::size_t inputBuffSize                       = sizeof(TensorBufferDtype) * elementsNum;
    std::vector<TSize>    castSizes                           = {elementsNum};
    auto& [castInType, castOutType, inVal, expected, expBias] = GetParam();
    std::vector<TensorBufferDtype> input {inVal};

    auto inputBuffer = std::make_unique<char[]>(inputBuffSize);
    std::memcpy(inputBuffer.get(), input.data(), inputBuffSize);

    TensorPtr staticCastInput = std::make_shared<Tensor>(castSizes.size(), castSizes.data(), castInType);
    staticCastInput->setAsWeights();
    staticCastInput->setAsStaticParam(true);
    staticCastInput->setTensorBuffer(inputBuffer.get(), input.size() * sizeof(float), asSynType<TensorBufferDtype>());
    // weights tensor which is output to cast
    std::unique_ptr<char[]> outBuffer(reinterpret_cast<char*>(allocateBufferForSynType(castOutType, elementsNum)));
    TensorPtr               staticCastOutput =
        std::make_shared<Tensor>(castSizes.size(), castSizes.data(), castOutType, outBuffer.get());

    if (expBias != 0)
    {
        staticCastOutput->setExpBias(expBias);
    }
    auto cpuCaster = CpuCaster(staticCastInput, staticCastOutput);
    staticCastOutput->setAsDataTypeMatchData();
    ASSERT_TRUE(cpuCaster.doCast());
    // cast output back to fp32
    std::unique_ptr<float[]> outputFloatBuf(
        reinterpret_cast<float*>(allocateBufferForSynType(syn_type_float, elementsNum)));
    TensorPtr outputAsFloatTensor = std::make_shared<Tensor>(castSizes.size(),
                                                             castSizes.data(),
                                                             syn_type_float,
                                                             reinterpret_cast<char*>(outputFloatBuf.get()));
    auto      outputAsFloatCaster = CpuCaster(staticCastOutput, outputAsFloatTensor);
    ASSERT_TRUE(outputAsFloatCaster.doCast());
    // verify results are same
    if (!float_eq(expected, outputFloatBuf.get()[0]))
    {
        // if we arrive here the assert will fail and print the numbers
        EXPECT_EQ(expected, outputFloatBuf.get()[0]);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         CastOnStaticResultsFp8Test,
                         testing::Values(std::make_tuple(syn_type_float, syn_type_fp8_143, -0.3474, -0.34375, 0),
                                         std::make_tuple(syn_type_float, syn_type_fp8_152, -0.347, -0.375, 0),
                                         std::make_tuple(syn_type_bf16, syn_type_fp8_143, 0.203125, 0.203125, 0),
                                         std::make_tuple(syn_type_bf16, syn_type_fp8_152, 0.203125, 0.1875, 0),
                                         std::make_tuple(syn_type_float, syn_type_fp8_143, 0.03125, 0.03125, 0),
                                         std::make_tuple(syn_type_float, syn_type_fp8_152, 0.03125, 0.03125, 0),
                                         std::make_tuple(syn_type_int32, syn_type_fp8_143, 9, 9, 0),
                                         std::make_tuple(syn_type_int32, syn_type_fp8_152, 9, 8, 0),
                                         // test different expbias for fp8
                                         std::make_tuple(syn_type_float, syn_type_fp8_143, 3800, 3840, 3),
                                         std::make_tuple(syn_type_float, syn_type_fp8_143, -0.93, -0.9375, 15)));

class EliminateNodesWithStaticInputTest : public GenericGraphTest
{
protected:
    virtual void SetUp() override
    {
        GenericGraphTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_CONSTANT_FOLDING, "true");
    }
    virtual void TearDown() override { GenericGraphTest::TearDown(); }

    unsigned numOfGuidNodes(Graph m_graph, std::string guid)
    {
        unsigned counter = 0;
        for (const auto& n : m_graph.getNodes())
        {
            if (n && n->getGUID() == guid) ++counter;
        }
        return counter;
    }

    using SizesVec = std::vector<TSize>;
    using DataVec  = std::vector<float>;

    void removeMultAndCastNodes(const SizesVec inSizes,
                                const SizesVec inScaleSizes,
                                DataVec        input1,
                                DataVec        input2,
                                DataVec        expected)
    {
        (*m_graph).setInferenceMode(true);

        const TSize       elementsNum      = multiplyElements(inSizes.begin(), inSizes.end());
        const TSize       scaleElementsNum = multiplyElements(inScaleSizes.begin(), inScaleSizes.end());
        const std::size_t input1BuffSize   = sizeof(float) * elementsNum;
        const std::size_t input2BuffSize   = sizeof(float) * scaleElementsNum;

        auto inputBuffer1 = std::make_unique<char[]>(input1BuffSize);
        std::memcpy(inputBuffer1.get(), input1.data(), input1BuffSize);

        auto inputBuffer2 = std::make_unique<char[]>(input2BuffSize);
        std::memcpy(inputBuffer2.get(), input2.data(), input2BuffSize);

        // reference for creating and set a const section: recipe_generator_test.cpp
        synMemoryDescriptor persistentMemoryDesc(true);
        auto sectionHandle1 = sectionHndlSlopMap.insert(0, 0);
        sectionHandle1.second->setConst(true);
        uint32_t mysectionId1 =
            (*m_graph).getCodeGenerator()->getNextMemorySectionID(SectionIDGenerator::USER_ALLOCATED_SECTIONS);
        sectionHandle1.second->setIDAndLock(mysectionId1);
        TensorPtr T1 = std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float);
        T1->setName("T1");
        T1->setAsStaticParam(true);
        T1->setTensorBuffer(inputBuffer1.get(), elementsNum * sizeof(float), T1->getBufferDataType(), false);
        T1->setMemoryDescriptor(persistentMemoryDesc);
        T1->setSectionHandle(sectionHandle1.second);
        T1->setMemorySectionID(mysectionId1);

        TensorPtr T2 = std::make_shared<Tensor>(inScaleSizes.size(), inScaleSizes.data(), syn_type_float);
        T2->setName("T2");
        T2->setAsStaticParam(true);
        T2->setTensorBuffer(inputBuffer2.get(), scaleElementsNum * sizeof(float), T2->getBufferDataType(), false);

        TensorPtr T3 = std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float);
        T3->setName("T3");

        NodePtr mult = NodeFactory::createNode({T1, T2}, {T3}, nullptr, "mult_fwd_f32", "mult");
        mult->setInputLayouts({Layout("CWHN"), Layout("KCSR")});
        GraphEditor::addNode(*m_graph, mult);

        TensorPtr T4 = std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_bf16);
        T4->setName("T4");

        NodePtr cast1 = NodeFactory::createNode({T3}, {T4}, nullptr, "cast_f32_to_bf16", "cast1");
        GraphEditor::addNode(*m_graph, cast1);

        TensorPtr T5 = std::make_shared<Tensor>(inSizes.size(), inSizes.data(), syn_type_float);
        T5->setName("T5");

        NodePtr cast2 = NodeFactory::createNode({T4}, {T5}, nullptr, "cast_bf16_to_f32", "cast2");
        GraphEditor::addNode(*m_graph, cast2);

        ASSERT_EQ((*m_graph).getNumNodes(), 3);
        ASSERT_EQ((*m_graph).getTensors().size(), 5);
        ASSERT_TRUE(eliminateNodesWithStaticInputs(*m_graph)) << "Failed to run eliminateNodesWithStaticInputs pass";
        ASSERT_EQ((*m_graph).getNumNodes(), 1);
        ASSERT_EQ(numOfGuidNodes(*m_graph, "mult_fwd_f32"), 0);
        ASSERT_EQ((*m_graph).getTensors().size(), 2);

        bfloat16* pData = reinterpret_cast<bfloat16*>(cast2->getInput(0)->getData());
        for (unsigned i = 0; i < elementsNum; i++)
        {
            EXPECT_EQ((float)pData[i], expected[i]);
        }
    }
};

TEST_P(EliminateNodesWithStaticInputTest, wrap_single_relu_with_transposes)
{
    /*
     * original graph is:
     * [NCHW static tensor with NHWC memory] -> relu
     *
     * the transpose_dont_care_nodes pass should wrap the relu with appropriate transposes:
     * [NCHW static tensor with NHWC memory] -> transpose ((NCHW)->(NHWC)) -> relu -> transpose ((NHWC)->(NCHW))
     *
     * the eliminateNodesWithStaticInputs pass (which is where we are testing changes) should remove the transpose
     * as it can be performed logically:
     * [NHWC static tensor] -> relu -> transpose ((NHWC)->(NCHW))
     */

    (*m_graph).setInferenceMode(true);
    const std::string reluGUIDString = "relu_f32";
    const char*       reluGUID       = reluGUIDString.c_str();

    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    const synDataType dataType    = syn_type_single;
    unsigned          elementsNum = w * h * c * batch;
    char*             inputBuffer = reinterpret_cast<char*>(allocateBufferForSynType(dataType, elementsNum));
    fillWithRandom(inputBuffer, elementsNum, dataType);

    // enable the tensor's permutation to be set prior to the run of setDefaultStrides method
    TensorPtr T1 = std::make_shared<Tensor>(dataType);
    T1->setName("T1");
    // perm = (2, 0, 1, 3) = (WHCN)->(CWHN)
    DimVector       vect {2, 0, 1, 3};
    gc::Permutation t1Permutation(vect);
    T1->setPermutation(t1Permutation);
    T1->reshape(4U, inSizes, nullptr);
    T1->setAsStaticParam(true);
    T1->setTensorBuffer(inputBuffer, elementsNum * sizeof(float), T1->getBufferDataType());

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T2->setName("T2");

    NodePtr relu1 = NodeFactory::createNode({T1}, {T2}, nullptr, reluGUID, "relu_1");
    GraphEditor::addNode(*m_graph, relu1);

    bool ret = adjustDataLayout(*m_graph);
    ret &= transposeDontCareNodes(*m_graph);
    ret &= eliminateNodesWithStaticInputs(*m_graph);
    ASSERT_EQ(ret, true) << "Failed to run passes";

    TensorPtr newReluInput = relu1->getInput(0);
    ASSERT_TRUE(newReluInput->isStaticParam()) << "New relu input isn't static";
    // verify data is the same
    float* originalInputData = reinterpret_cast<float*>(T1->getData());
    float* newReluInputData  = reinterpret_cast<float*>(newReluInput->getData());
    for (unsigned i = 0; i < elementsNum; i++)
    {
        EXPECT_EQ(newReluInputData[i], originalInputData[i]);
    }

    NodePtr transpose = *((*m_graph).getNodeConsumers(relu1).begin());
    ASSERT_TRUE(transpose->isTranspose()) << "Missing a transpose node after {} node" << relu1->getNodeName();
    TransposeNode* transposeNode = dynamic_cast<TransposeNode*>(transpose.get());
    ASSERT_TRUE(transposeNode && gc::Permutation(transposeNode->permutation()) == t1Permutation.getInversePermutation())
        << "Wrong permutation in transpose node {}" << transposeNode->getNodeName();

    delete[] inputBuffer;
}

TEST_P(EliminateNodesWithStaticInputTest, remove_reshape_cast)
{
    (*m_graph).setInferenceMode(true);
    const TSize c                = 3;
    const TSize w                = 20;
    const TSize h                = 10;
    const TSize batch            = 1;
    const TSize inReshapeSizes[] = {h, w, c, batch};
    const TSize inSizes[]        = {w, h, c, batch};

    const synDataType dataTypeCast = syn_type_bf16;
    const synDataType dataType     = syn_type_single;
    const unsigned    elementsNum  = w * h * c * batch;
    char*             inputBuffer1 = reinterpret_cast<char*>(allocateBufferForSynType(dataType, elementsNum));
    char*             inputBuffer2 = reinterpret_cast<char*>(allocateBufferForSynType(dataType, elementsNum));
    fillWithRandom(inputBuffer1, elementsNum, dataType);
    fillWithRandom(inputBuffer2, elementsNum, dataType);

    float expectedOutput[elementsNum];
    for (unsigned i = 0; i < elementsNum; i++)
    {
        expectedOutput[i] = ((float*)inputBuffer1)[i];
    }

    TensorPtr T1 = std::make_shared<Tensor>(4U, inReshapeSizes, dataTypeCast);
    T1->setName("T1");
    T1->setAsStaticParam(true);
    T1->setTensorBuffer(inputBuffer1, elementsNum * sizeof(float), T1->getBufferDataType());

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, dataTypeCast);
    T2->setName("T2");

    NodePtr reshape = NodeFactory::createNode({T1}, {T2}, nullptr, "reshape", "reshape");
    GraphEditor::addNode(*m_graph, reshape);

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T3->setName("T3");

    NodePtr cast = NodeFactory::createNode({T2}, {T3}, nullptr, "cast_bf16_to_f32", "cast");
    GraphEditor::addNode(*m_graph, cast);

    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T4->setName("T4");
    T4->setAsStaticParam(true);
    T4->setTensorBuffer(inputBuffer2, elementsNum * sizeof(float), T3->getBufferDataType());

    TensorPtr T5 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T5->setName("T5");

    NodePtr add = NodeFactory::createNode({T3, T4}, {T5}, nullptr, "add_fwd_f32", "add");
    GraphEditor::addNode(*m_graph, add);

    ASSERT_EQ((*m_graph).getNumNodes(), 3);
    ASSERT_EQ((*m_graph).getTensors().size(), 5);
    ASSERT_EQ(add->getInput(0)->getData(), nullptr);
    ASSERT_NE(add->getInput(0)->getData(), T1->getData());
    ASSERT_TRUE(eliminateNodesWithStaticInputs(*m_graph)) << "Failed to run eliminateNodesWithStaticInputs pass";
    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_EQ(numOfGuidNodes(*m_graph, "add_fwd_f32"), 1);
    ASSERT_EQ((*m_graph).getTensors().size(), 3);
    ASSERT_NE(add->getInput(0)->getData(), nullptr);
    ASSERT_NE(add->getInput(0)->getData(), T1->getData());

    if (inputBuffer1 != nullptr)
    {
        memset(inputBuffer1, 0, elementsNum * sizeof(float));
    }

    float* pFirstData  = reinterpret_cast<float*>(add->getInput(0)->getData());
    float* pSecondData = reinterpret_cast<float*>(add->getInput(1)->getData());
    for (unsigned i = 0; i < elementsNum; i++)
    {
        EXPECT_EQ(pFirstData[i], expectedOutput[i]);
    }
    for (unsigned i = 0; i < elementsNum; i++)
    {
        EXPECT_EQ(pSecondData[i], ((float*)inputBuffer2)[i]);
    }

    delete[] inputBuffer1;
    delete[] inputBuffer2;
}

TEST_P(EliminateNodesWithStaticInputTest, remove_mult_of_single_scale_cast)
{
    SizesVec inSizes      = {1, 2, 3, 1};
    SizesVec inScaleSizes = {1};

    DataVec input1   = {-2.1, 5, 7, 3.5, 0, 9};
    DataVec input2   = {2};
    DataVec expected = {-4.1875, 10, 14, 7, 0, 18};
    removeMultAndCastNodes(inSizes, inScaleSizes, input1, input2, expected);
}

TEST_P(EliminateNodesWithStaticInputTest, remove_mult_of_vector_scale_cast)
{
    std::vector<TSize> inSizes      = {2, 1, 3, 1};
    std::vector<TSize> inScaleSizes = {2};

    DataVec input1   = {-2.1, 6.3, 7, 3.5, 0, 9};
    DataVec input2   = {2, 3};
    DataVec expected = {-4.1875, 18.875, 14, 10.5, 0, 27};
    removeMultAndCastNodes(inSizes, inScaleSizes, input1, input2, expected);
}

// TODO [SW-141316] enable test when split is supported in the pass
TEST_P(EliminateNodesWithStaticInputTest, DISABLED_split_with_static_input_and_one_output)
{
    (*m_graph).setInferenceMode(true);
    const TSize c         = 3;
    const TSize w         = 20;
    const TSize h         = 10;
    const TSize batch     = 1;
    const TSize inSizes[] = {w, h, c, batch};

    const synDataType dataType     = syn_type_single;
    const unsigned    elementsNum  = w * h * c * batch;
    char*             inputBuffer1 = reinterpret_cast<char*>(allocateBufferForSynType(dataType, elementsNum));
    char*             inputBuffer2 = reinterpret_cast<char*>(allocateBufferForSynType(dataType, elementsNum));
    fillWithRandom(inputBuffer1, elementsNum, dataType);
    fillWithRandom(inputBuffer2, elementsNum, dataType);

    float expectedOutput[elementsNum];
    for (unsigned i = 0; i < elementsNum; i++)
    {
        expectedOutput[i] = ((float*)inputBuffer1)[i];
    }

    TensorPtr T1 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T1->setName("T1");
    T1->setAsStaticParam(true);
    T1->setTensorBuffer(inputBuffer1, elementsNum * sizeof(float), T1->getBufferDataType());

    TensorPtr T2 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T2->setName("T2");

    unsigned dim   = 0;
    NodePtr  split = NodeFactory::createNode({T1}, {T2}, &dim, "split", "split");
    GraphEditor::addNode(*m_graph, split);

    TensorPtr T3 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T3->setName("T3");
    T3->setAsStaticParam(true);
    T3->setTensorBuffer(inputBuffer2, elementsNum * sizeof(float), T3->getBufferDataType());

    TensorPtr T4 = std::make_shared<Tensor>(4U, inSizes, dataType);
    T4->setName("T4");

    NodePtr add = NodeFactory::createNode({T2, T3}, {T4}, nullptr, "add_fwd_f32", "add");
    GraphEditor::addNode(*m_graph, add);

    ASSERT_EQ((*m_graph).getNumNodes(), 2);
    ASSERT_EQ((*m_graph).getTensors().size(), 4);
    ASSERT_EQ(add->getInput(0)->getData(), nullptr);
    ASSERT_NE(add->getInput(0)->getData(), T1->getData());
    ASSERT_EQ(T1->getData(), inputBuffer1);

    ASSERT_TRUE(eliminateNodesWithStaticInputs(*m_graph)) << "Failed to run eliminateNodesWithStaticInputs pass";

    ASSERT_EQ((*m_graph).getNumNodes(), 1);
    ASSERT_EQ((*m_graph).getTensors().size(), 3);
    ASSERT_NE(add->getInput(0)->getData(), nullptr);
    ASSERT_NE(add->getInput(0)->getData(), T1->getData());
    ASSERT_EQ(T1->getData(), inputBuffer1);

    if (inputBuffer1 != nullptr)
    {
        memset(inputBuffer1, 0, elementsNum * sizeof(float));
    }

    float* pFirstData  = reinterpret_cast<float*>(add->getInput(0)->getData());
    float* pSecondData = reinterpret_cast<float*>(add->getInput(1)->getData());
    for (unsigned i = 0; i < elementsNum; i++)
    {
        EXPECT_EQ(pFirstData[i], expectedOutput[i]);
    }
    for (unsigned i = 0; i < elementsNum; i++)
    {
        EXPECT_EQ(pSecondData[i], ((float*)inputBuffer2)[i]);
    }

    delete[] inputBuffer1;
    delete[] inputBuffer2;
}

INSTANTIATE_TEST_SUITE_P(,
                         EliminateNodesWithStaticInputTest,
                         ::testing::Values(synDeviceGaudi2, synDeviceGaudi3),
                         GenericGraphTest::GetName());