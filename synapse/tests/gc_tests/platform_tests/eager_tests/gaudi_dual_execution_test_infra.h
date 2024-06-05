#pragma once

#include "../gaudi_tests/gc_gaudi_test_infra.h"

/*
    This class comes to provide the required facilities to execute a given graph topology in both
    graph and eager mode flavors.
    Such a functionality is useful for comparing results between eager and graph mode execution.
*/

class SynDualExecutionGaudiTestInfra : public SynGaudiTestInfra
{
public:
    static constexpr unsigned DEFAULT_GRAPH_MODE_INDEX = 0;
    static constexpr unsigned DEFAULT_EAGER_MODE_INDEX = 1;
    static const std::string  GRAPH_SUFFIXES[];

    SynDualExecutionGaudiTestInfra();

    struct GraphIndexPair
    {
        unsigned graph;
        unsigned eager;
    };

    struct TensorIndexPair
    {
        unsigned graph;
        unsigned eager;
    };

    struct TensorIndicesPair
    {
        TensorIndices graph;
        TensorIndices eager;

        void reserve(size_t size)
        {
            graph.reserve(size);
            eager.reserve(size);
        }

        void push_back(TensorIndexPair tensorIndexPair)
        {
            graph.push_back(tensorIndexPair.graph);
            eager.push_back(tensorIndexPair.eager);
        }
    };

    virtual void SetUpTest() override;
    virtual void TearDownTest() override;

    GraphIndexPair createNewGraphPair();

    void setTensorAllowPermutations();
    void setTensorAllowPermutations(TensorIndexPair tensorIndexPair);
    void setTensorPermutation(TensorIndexPair tensorIndexPair, gc::Permutation permutation);

    TensorIndexPair createPersistTensors(TensorUsage     usage,
                                         MemInitType     initSelect       = MEM_INIT_ALL_ZERO,
                                         const float*    initializer      = nullptr,
                                         unsigned*       sizes            = nullptr,
                                         unsigned        dims             = DEFAULT_SIZES,
                                         synDataType     dataType         = syn_type_single,
                                         bool            allowPermutation = false,
                                         unsigned*       strides          = nullptr,
                                         const char*     name             = nullptr,
                                         GraphIndexPair  graphIndexPair   = {DEFAULT_GRAPH_MODE_INDEX,
                                                                          DEFAULT_EAGER_MODE_INDEX},
                                         unsigned        offsetInSection  = 0,
                                         const unsigned* sectionIndex     = nullptr,
                                         unsigned*       minSizes         = nullptr);

    TensorIndexPair createTensors(TensorUsage    usage,
                                  MemInitType    initSelect     = MEM_INIT_ALL_ZERO,
                                  const float*   initializer    = nullptr,
                                  unsigned*      sizes          = nullptr,
                                  unsigned       dims           = DEFAULT_SIZES,
                                  synDataType    dataType       = syn_type_single,
                                  unsigned*      strides        = nullptr,
                                  GraphIndexPair graphIndexPair = {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX});

    TensorIndexPair createConstTensors(MemInitType    initSelect     = MEM_INIT_ALL_ZERO,
                                       const float*   initializer    = nullptr,
                                       unsigned*      sizes          = nullptr,
                                       unsigned       dims           = DEFAULT_SIZES,
                                       synDataType    dataType       = syn_type_single,
                                       unsigned*      strides        = nullptr,
                                       GraphIndexPair graphIndexPair = {DEFAULT_GRAPH_MODE_INDEX,
                                                                        DEFAULT_EAGER_MODE_INDEX});

    TensorIndexPair createShapeTensors(TensorUsage    usage,
                                       unsigned*      sizes,
                                       unsigned       dims           = DEFAULT_SIZES,
                                       synDataType    dataType       = syn_type_single,
                                       const char*    name           = nullptr,
                                       GraphIndexPair graphIndexPair = {DEFAULT_GRAPH_MODE_INDEX,
                                                                        DEFAULT_EAGER_MODE_INDEX});

    void addNodesToGraphs(const char*    guid,
                          void*          userParams     = nullptr,
                          unsigned       paramSize      = 0,
                          const char*    nodeName       = nullptr,
                          GraphIndexPair graphIndexPair = {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX},
                          synNodeId*     nodeId         = nullptr,
                          const char**   inputLayouts   = nullptr,
                          const char**   outputLayouts  = nullptr);

    void addNodesToGraphs(const char*              guid,
                          const TensorIndicesPair& inputTensorIndices,   // Indices of m_inTensors
                          const TensorIndicesPair& outputTensorIndices,  // Indices of m_outTensors
                          void*                    userParams = nullptr,
                          unsigned                 paramSize  = 0,
                          const char*              nodeName   = nullptr,
                          GraphIndexPair graphIndexPair       = {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX},
                          synNodeId*     nodeId               = nullptr,
                          const char**   inputLayouts         = nullptr,
                          const char**   outputLayouts        = nullptr);

    virtual void compileAndRun() override;

    void compileTopology(const std::string& topologyName   = "",
                         GraphIndexPair     graphIndexPair = {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX});

    void runTopology(GraphIndexPair graphIndexPair        = {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX},
                     bool           initPersistentOutputs = false,
                     synStatus      expectedLaunch        = synSuccess);
private:
    void addNewlyCreatedPair(const TensorIndexPair& tensorIndexPair, TensorUsage usage);

    std::vector<unsigned> m_graphModeInputTensors;
    std::vector<unsigned> m_graphModeOutputTensors;
    std::vector<unsigned> m_eagerModeInputTensors;
    std::vector<unsigned> m_eagerModeOutputTensors;
};