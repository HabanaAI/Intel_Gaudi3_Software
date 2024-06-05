#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "syn_singleton.hpp"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, test_descandent_node_mapping)
{
    auto singelton = synSingleton::getInstance();

    // generate unique node id
    synNodeId uniqueNodeId;
    ASSERT_EQ(synSuccess, singelton->getUniqueNodeId(uniqueNodeId));

    // create 2 nodes, and set them as descendants of the first .
    // create simple tensors for nodes.
    unsigned  dims[]    = {1, 2, 3, 4};
    auto inFirst = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    auto outFirst = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    synNodeId firstId;
    addNodeToGraph("memcpy" ,{inFirst}, {outFirst}, nullptr, 0, nullptr, 0, &firstId);

    auto inSecond = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    auto outSecond = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dims, DEFAULT_SIZES, syn_type_float);
    synNodeId secondId;
    addNodeToGraph("memcpy" ,{inSecond}, {outSecond}, nullptr, 0, nullptr, 0, &secondId);

    synGraphHandle graphHandle = m_graphs[0].graphHandle;
    ASSERT_EQ(synSuccess, singelton->setOriginalComplexNode(graphHandle, firstId, uniqueNodeId));
    ASSERT_EQ(synSuccess, singelton->setOriginalComplexNode(graphHandle, secondId, uniqueNodeId));
    // set same node should succeed
    ASSERT_EQ(synSuccess, singelton->setOriginalComplexNode(graphHandle, secondId, uniqueNodeId));

    synNodeId anotherUnique;
    ASSERT_EQ(synSuccess, singelton->getUniqueNodeId(anotherUnique));
    // set descendant to different node should fail
    ASSERT_EQ(synFail, singelton->setOriginalComplexNode(graphHandle, secondId, anotherUnique));
}

class SynTrainingComplexGuidNodeControlDependencyTest : public SynTrainingTestInfra
{
protected:
    virtual void SetUpTest() override
    {
        SynTrainingTestInfra::SetUpTest();
        singleton = synSingleton::getInstanceInternal();
        ASSERT_EQ(synSuccess, singleton->createGraph(&graphHandle, m_deviceType, CompilationMode::Graph));
        graph = singleton->getGraph(graphHandle);
    }
    virtual void TearDownTest() override
    {
        ASSERT_EQ(synSuccess, singleton->destroyGraph(graphHandle));
        for (auto section : sectionHandles)
        {
            ASSERT_EQ(synSuccess, singleton->sectionDestroy(section));
        }
        SynTrainingTestInfra::TearDownTest();
    }

    void createTensors(bool convFirst, bool isNestedTest);

    void createTensorsAndNodes(bool convFirst, bool isNestedTest);

    void createGraphInputTensor(std::string name);

    void setTensorPersistent(synTensor tensor);

    synSingleton*  singleton;
    synGraphHandle graphHandle;
    HabanaGraph*   graph;
    std::vector<synSectionHandle> sectionHandles;  // stores section handles for release at test end

    // below variables are common to all tests
    NodePtr m_convNode, m_absNode, m_negNode, m_reluNode, m_ceilNode;

    synNodeId m_convId, m_negId, m_absId, m_reluId, m_ceilId;

    synTensor m_ifm, m_ofm, m_weights, m_negIn, m_negOut, m_absOut, m_reluOut, m_ceilOut;

    synTensorGeometry     inputGeomtery      = {{3, 4, 4, 1}, 4};
    synTensorDeviceLayout tensorDeviceLayout = {{0, 0, 0, 0}, syn_type_float};  // same for all tensors
};

void SynTrainingComplexGuidNodeControlDependencyTest::setTensorPersistent(synTensor tensor)
{
    synSectionHandle sectionHandle;
    ASSERT_EQ(synSuccess, singleton->sectionCreate(&sectionHandle, 0, graphHandle));
    ASSERT_EQ(synSuccess, singleton->sectionPersistentSet(sectionHandle, true));
    ASSERT_EQ(synSuccess, singleton->tensorAssignToSection(tensor, sectionHandle, 0));
    sectionHandles.push_back(sectionHandle);
}

void SynTrainingComplexGuidNodeControlDependencyTest::createGraphInputTensor(std::string name)
{
    synTensorGeometry geometry = inputGeomtery;
    synTensor         inputTensor;

    ASSERT_EQ(synSuccess, singleton->createTensor(&inputTensor, graphHandle, DATA_TENSOR, name.c_str()));
    ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(inputTensor, &geometry, synGeometryMaxSizes));
    ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(inputTensor, &tensorDeviceLayout));
    setTensorPersistent(inputTensor);

    if (name == "ifm")
    {
        m_ifm = inputTensor;
    }
    else if (name == "negIn")
    {
        m_negIn = inputTensor;
    }
    else
    {
        ASSERT_TRUE(false) << "unexpected input tensor name, test error";
    }
}

void SynTrainingComplexGuidNodeControlDependencyTest::createTensors(bool convFirst, bool isNestedTest)
{
    ASSERT_EQ(synSuccess, singleton->createTensor(&m_weights, graphHandle, DATA_TENSOR, "weights"));
    ASSERT_EQ(synSuccess, singleton->createTensor(&m_ofm, graphHandle, DATA_TENSOR, "ofm"));

    synTensorGeometry weightsGeometry = {{3, 3, 2, 2}, 4};
    ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(m_weights, &weightsGeometry, synGeometryMaxSizes));
    synTensorGeometry ofmGeometry = {{3, 3, 3, 1}, 4};
    ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(m_ofm, &ofmGeometry, synGeometryMaxSizes));

    ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(m_weights, &tensorDeviceLayout));
    ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(m_ofm, &tensorDeviceLayout));
    setTensorPersistent(m_weights);

    synTensorGeometry tpcTensorGeometry = convFirst ? ofmGeometry : inputGeomtery;
    ASSERT_EQ(synSuccess, singleton->createTensor(&m_negOut, graphHandle, DATA_TENSOR, "negOut"));
    ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(m_negOut, &tpcTensorGeometry, synGeometryMaxSizes));
    ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(m_negOut, &tensorDeviceLayout));

    ASSERT_EQ(synSuccess, singleton->createTensor(&m_absOut, graphHandle, DATA_TENSOR, "absOut"));
    ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(m_absOut, &tpcTensorGeometry, synGeometryMaxSizes));
    ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(m_absOut, &tensorDeviceLayout));

    if (isNestedTest)
    {
        ASSERT_EQ(synSuccess, singleton->createTensor(&m_reluOut, graphHandle, DATA_TENSOR, "reluOut"));
        ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(m_reluOut, &tpcTensorGeometry, synGeometryMaxSizes));
        ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(m_reluOut, &tensorDeviceLayout));

        ASSERT_EQ(synSuccess, singleton->createTensor(&m_ceilOut, graphHandle, DATA_TENSOR, "ceilOut"));
        ASSERT_EQ(synSuccess, singleton->tensorSetGeometry(m_ceilOut, &tpcTensorGeometry, synGeometryMaxSizes));
        ASSERT_EQ(synSuccess, singleton->tensorSetDeviceLayout(m_ceilOut, &tensorDeviceLayout));
    }
}

void SynTrainingComplexGuidNodeControlDependencyTest::createTensorsAndNodes(bool convFirst, bool isNestedTest)
{
    createTensors(convFirst, isNestedTest);

    synTensor inConv[2];
    synTensor outConv[] = {m_ofm};

    synTensor inNeg[1];
    synTensor outNeg[] = {m_negOut};

    synTensor inAbs[]  = {m_negOut};
    synTensor outAbs[] = {m_absOut};

    synTensor inRelu[]  = {m_absOut};
    synTensor outRelu[] = {m_reluOut};

    synTensor inCeil[]  = {m_reluOut};
    synTensor outCeil[] = {m_ceilOut};

    if (convFirst)  // conv -> neg -> abs ...
    {
        createGraphInputTensor("ifm");
        inConv[0] = m_ifm;
        inConv[1] = m_weights;
        inNeg[0]  = m_ofm;
        if (isNestedTest)  // conv -> neg -> abs -> relu -> ceil
        {
            // ceilOut is graph output
            setTensorPersistent(m_ceilOut);
        }
        else
        {
            // absOut is graph output
            setTensorPersistent(m_absOut);
        }
    }
    else  // neg -> abs -> ... -> conv
    {
        createGraphInputTensor("negIn");
        inConv[1] = m_weights;
        inNeg[0]  = m_negIn;
        setTensorPersistent(m_ofm);  // ofm is graph output
        if (isNestedTest)            // neg -> abs -> relu -> ceil -> conv
        {
            inConv[0] = m_ceilOut;
        }
        else
        {
            inConv[0] = m_absOut;
        }
    }

    // create conv node
    synConvolutionParams params;
    params.kW = 2;
    params.kH = 2;
    ASSERT_EQ(synSuccess,
              singleton->createGenericNodeWithId(graphHandle,
                                                 inConv,
                                                 outConv,
                                                 2,
                                                 1,
                                                 &params,
                                                 sizeof(synConvolutionParams),
                                                 NodeFactory::convolutionNodeTypeName,
                                                 nullptr,
                                                 nullptr,
                                                 "conv",
                                                 &m_convId));
    m_convNode = graph->getNodeByID(m_convId);

    // create neg node
    ASSERT_EQ(synSuccess,
              singleton->createGenericNodeWithId(graphHandle,
                                                 inNeg,
                                                 outNeg,
                                                 1,
                                                 1,
                                                 nullptr,
                                                 0,
                                                 "neg_fwd_f32",
                                                 nullptr,
                                                 nullptr,
                                                 "neg",
                                                 &m_negId));
    m_negNode = graph->getNodeByID(m_negId);

    // create abs node
    ASSERT_EQ(synSuccess,
              singleton->createGenericNodeWithId(graphHandle,
                                                 inAbs,
                                                 outAbs,
                                                 1,
                                                 1,
                                                 nullptr,
                                                 0,
                                                 "abs_fwd_f32",
                                                 nullptr,
                                                 nullptr,
                                                 "abs",
                                                 &m_absId));
    m_absNode = graph->getNodeByID(m_absId);

    if (isNestedTest)
    {
        // create relu node
        ASSERT_EQ(synSuccess,
                  singleton->createGenericNodeWithId(graphHandle,
                                                     inRelu,
                                                     outRelu,
                                                     1,
                                                     1,
                                                     nullptr,
                                                     0,
                                                     "relu_fwd_f32",
                                                     nullptr,
                                                     nullptr,
                                                     "relu",
                                                     &m_reluId));
        m_reluNode = graph->getNodeByID(m_absId);
        // create ceil node
        ASSERT_EQ(synSuccess,
                  singleton->createGenericNodeWithId(graphHandle,
                                                     inCeil,
                                                     outCeil,
                                                     1,
                                                     1,
                                                     nullptr,
                                                     0,
                                                     "ceil_fwd_f32",
                                                     nullptr,
                                                     nullptr,
                                                     "ceil",
                                                     &m_ceilId));
        m_ceilNode = graph->getNodeByID(m_absId);
    }
}
