#include "gaudi_dual_execution_test_infra.h"
#include "habana_global_conf.h"

const std::string SynDualExecutionGaudiTestInfra::GRAPH_SUFFIXES[] = {"_graph", "_eager"};

SynDualExecutionGaudiTestInfra::SynDualExecutionGaudiTestInfra()
{
    m_testConfig.m_numOfTestDevices = 1;
    m_testConfig.m_compilationMode  = COMP_BOTH_MODE_TESTS;
    m_testConfig.m_supportedDeviceTypes.clear();
    setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});
    setTestPackage(TEST_PACKAGE_EAGER);
}

void SynDualExecutionGaudiTestInfra::SetUpTest()
{
    SynGaudiTestInfra::SetUpTest();
    GCFG_FORCE_EAGER.setValue(1);
}

void SynDualExecutionGaudiTestInfra::TearDownTest()
{
    GCFG_FORCE_EAGER.setValue(2);
    SynGaudiTestInfra::TearDownTest();
}

SynDualExecutionGaudiTestInfra::GraphIndexPair SynDualExecutionGaudiTestInfra::createNewGraphPair()
{
    long origConfigVal = GCFG_FORCE_EAGER.value();
    GCFG_FORCE_EAGER.setValue(2);
    unsigned graphModeGraphIdx = createGraph(COMP_GRAPH_MODE_TEST);
    unsigned eagerGraphIdx     = createGraph(COMP_EAGER_MODE_TEST);
    GCFG_FORCE_EAGER.setValue(origConfigVal);
    return {graphModeGraphIdx, eagerGraphIdx};
}

void SynDualExecutionGaudiTestInfra::setTensorAllowPermutations()
{
    for (const auto& tensor : m_tensors)
    {
        synTensorSetAllowPermutation(tensor, 1);
    }
}

void SynDualExecutionGaudiTestInfra::setTensorAllowPermutations(TensorIndexPair tensorIndexPair)
{
    synTensorSetAllowPermutation(m_tensors[tensorIndexPair.graph], 1);
    synTensorSetAllowPermutation(m_tensors[tensorIndexPair.eager], 1);
}

void SynDualExecutionGaudiTestInfra::setTensorPermutation(TensorIndexPair tensorIndexPair, gc::Permutation permutation)
{
    synTensorPermutation perm;
    perm.dims       = permutation.size();
    auto permValues = permutation.getValues();
    for (int i = 0; i < perm.dims; i++)
    {
        perm.permutation[i] = permValues[i];
    }
    synTensorSetPermutation(m_tensors[tensorIndexPair.eager], &perm);
    synTensorSetPermutation(m_tensors[tensorIndexPair.graph], &perm);
}

void SynDualExecutionGaudiTestInfra::addNewlyCreatedPair(const TensorIndexPair& tensorIndexPair, TensorUsage usage)
{
    if (usage == INPUT_TENSOR)
    {
        m_graphModeInputTensors.push_back(tensorIndexPair.graph);
        m_eagerModeInputTensors.push_back(tensorIndexPair.eager);
    }
    else
    {
        m_graphModeOutputTensors.push_back(tensorIndexPair.graph);
        m_eagerModeOutputTensors.push_back(tensorIndexPair.eager);
    }
}

SynDualExecutionGaudiTestInfra::TensorIndexPair
SynDualExecutionGaudiTestInfra::createPersistTensors(TensorUsage     usage,
                                                     MemInitType     initSelect,
                                                     const float*    initializer,
                                                     unsigned*       sizes,
                                                     unsigned        dims,
                                                     synDataType     dataType,
                                                     bool            allowPermutation,
                                                     unsigned*       strides,
                                                     const char*     name,
                                                     GraphIndexPair  graphIndexPair,
                                                     unsigned        offsetInSection,
                                                     const unsigned* sectionIndex,
                                                     unsigned*       minSizes)
{
    TensorIndexPair tensorIndexPair;
    tensorIndexPair.graph = createPersistTensor(usage,
                                                initSelect,
                                                initializer,
                                                sizes,
                                                dims,
                                                dataType,
                                                strides,
                                                name,
                                                graphIndexPair.graph,
                                                offsetInSection,
                                                sectionIndex,
                                                minSizes);

    tensorIndexPair.eager = createPersistTensor(usage,
                                                MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                                static_cast<float*>(m_hostBuffers[tensorIndexPair.graph]),
                                                sizes,
                                                dims,
                                                dataType,
                                                strides,
                                                name,
                                                graphIndexPair.eager,
                                                offsetInSection,
                                                sectionIndex,
                                                minSizes);

    addNewlyCreatedPair(tensorIndexPair, usage);
    if (allowPermutation)
    {
        synTensorSetAllowPermutation(m_tensors[tensorIndexPair.graph], 1);
        synTensorSetAllowPermutation(m_tensors[tensorIndexPair.eager], 1);
    }
    return tensorIndexPair;
}

SynDualExecutionGaudiTestInfra::TensorIndexPair
SynDualExecutionGaudiTestInfra::createConstTensors(MemInitType    initSelect,
                                                   const float*   initializer,
                                                   unsigned*      sizes,
                                                   unsigned       dims,
                                                   synDataType    dataType,
                                                   unsigned*      strides,
                                                   GraphIndexPair graphIndexPair)
{
    TensorIndexPair tensorIndexPair;
    tensorIndexPair.graph =
        createConstTensor(initSelect, initializer, sizes, dims, dataType, strides, nullptr, graphIndexPair.graph);
    tensorIndexPair.eager =
        createConstTensor(initSelect, initializer, sizes, dims, dataType, strides, nullptr, graphIndexPair.eager);
    addNewlyCreatedPair(tensorIndexPair, INPUT_TENSOR);
    return tensorIndexPair;
}

SynDualExecutionGaudiTestInfra::TensorIndexPair
SynDualExecutionGaudiTestInfra::createTensors(TensorUsage    usage,
                                              MemInitType    initSelect,
                                              const float*   initializer,
                                              unsigned*      sizes,
                                              unsigned       dims,
                                              synDataType    dataType,
                                              unsigned*      strides,
                                              GraphIndexPair graphIndexPair)
{
    TensorIndexPair tensorIndexPair;
    tensorIndexPair.graph =
        createTensor(usage, initSelect, initializer, sizes, dims, dataType, strides, nullptr, graphIndexPair.graph);

    tensorIndexPair.eager = createTensor(usage,
                                         MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                         static_cast<float*>(m_hostBuffers[tensorIndexPair.graph]),
                                         sizes,
                                         dims,
                                         dataType,
                                         strides,
                                         nullptr,
                                         graphIndexPair.eager);

    addNewlyCreatedPair(tensorIndexPair, usage);
    return tensorIndexPair;
}

SynDualExecutionGaudiTestInfra::TensorIndexPair
SynDualExecutionGaudiTestInfra::createShapeTensors(TensorUsage    usage,
                                                   unsigned*      sizes,
                                                   unsigned       dims,
                                                   synDataType    dataType,
                                                   const char*    name,
                                                   GraphIndexPair graphIndexPair)
{
    TensorIndexPair tensorIndexPair;
    tensorIndexPair.graph = createShapeTensor(usage, sizes, nullptr, dims, dataType, name, graphIndexPair.graph);
    tensorIndexPair.eager = createShapeTensor(usage, sizes, nullptr, dims, dataType, name, graphIndexPair.eager);
    addNewlyCreatedPair(tensorIndexPair, usage);
    return tensorIndexPair;
}

void SynDualExecutionGaudiTestInfra::addNodesToGraphs(const char*    guid,
                                                      void*          userParams,
                                                      unsigned       paramSize,
                                                      const char*    nodeName,
                                                      GraphIndexPair graphIndexPair,
                                                      synNodeId*     nodeId,
                                                      const char**   inputLayouts,
                                                      const char**   outputLayouts)
{
    addNodeToGraph(guid,
                   m_graphModeInputTensors,
                   m_graphModeOutputTensors,
                   userParams,
                   paramSize,
                   nodeName,
                   graphIndexPair.graph,
                   nodeId,
                   inputLayouts,
                   outputLayouts);
    addNodeToGraph(guid,
                   m_eagerModeInputTensors,
                   m_eagerModeOutputTensors,
                   userParams,
                   paramSize,
                   nodeName,
                   graphIndexPair.eager,
                   nodeId,
                   inputLayouts,
                   outputLayouts);
}

void SynDualExecutionGaudiTestInfra::addNodesToGraphs(const char*              guid,
                                                      const TensorIndicesPair& inputTensorIndices,
                                                      const TensorIndicesPair& outputTensorIndices,
                                                      void*                    userParams,
                                                      unsigned                 paramSize,
                                                      const char*              nodeName,
                                                      GraphIndexPair           graphIndexPair,
                                                      synNodeId*               nodeId,
                                                      const char**             inputLayouts,
                                                      const char**             outputLayouts)
{
    addNodeToGraph(guid,
                   inputTensorIndices.graph,
                   outputTensorIndices.graph,
                   userParams,
                   paramSize,
                   nodeName,
                   graphIndexPair.graph,
                   nodeId,
                   inputLayouts,
                   outputLayouts);
    addNodeToGraph(guid,
                   inputTensorIndices.eager,
                   outputTensorIndices.eager,
                   userParams,
                   paramSize,
                   nodeName,
                   graphIndexPair.eager,
                   nodeId,
                   inputLayouts,
                   outputLayouts);
}

void SynDualExecutionGaudiTestInfra::compileAndRun()
{
    for (auto graphIndex : {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX})
    {
        SynGaudiTestInfra::compileTopology("topology" + GRAPH_SUFFIXES[graphIndex], graphIndex);
        SynGaudiTestInfra::runTopology(graphIndex, false, synSuccess);
    }
}

void SynDualExecutionGaudiTestInfra::compileTopology(const std::string& topologyName, GraphIndexPair graphIndexPair)
{
    SynGaudiTestInfra::compileTopology(topologyName + GRAPH_SUFFIXES[DEFAULT_GRAPH_MODE_INDEX], graphIndexPair.graph);
    SynGaudiTestInfra::compileTopology(topologyName + GRAPH_SUFFIXES[DEFAULT_EAGER_MODE_INDEX], graphIndexPair.eager);
}

void SynDualExecutionGaudiTestInfra::runTopology(GraphIndexPair graphIndexPair,
                                                 bool           initPersistentOutputs,
                                                 synStatus      expectedLaunch)
{
    SynGaudiTestInfra::runTopology(graphIndexPair.graph, false, synSuccess);
    SynGaudiTestInfra::runTopology(graphIndexPair.eager, false, synSuccess);
}