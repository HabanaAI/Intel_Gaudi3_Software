#pragma once

#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "pipeline_management/habana_norms_handler.h"
#include "types.h"

class HabanaNormsSlicingTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<unsigned /* m_tpcsPerSlice */,
                                                unsigned /* m_numSliceTensors */,
                                                bool /* m_ofmInSram */,
                                                bool /* m_sliceTensorInSram */,
                                                bool /* m_tilePattern */>>
{
public:
    HabanaNormsSlicingTest();

protected:
    void createGraph();
    void createSliceTensors();
    void addNormNodes();
    void addProducers();
    void addReshapeNodes();
    void addSplitNode();
    void initOrigSliceNodeMap();
    void runTest();
    void validateSlice();
    void validateSliceAndTile();
    void validateResult();

private:
    Gaudi2Graph              m_graph;
    PatternNodesCollectorPtr m_sliceCollector;
    HabanaNormsHandler       m_normsHandler;
    TensorVector             m_sliceTensors;
    TensorVector             m_ofmTensors;
    TensorVector             m_reshapeOutputTensors;
    NodePtr                  m_sliceNode;
    NodePtr                  m_tileNode;
    unsigned                 m_opIdx;
    unsigned                 m_bundleIdx;
    unsigned                 m_tpcsPerSlice;
    unsigned                 m_numSliceTensors;
    unsigned                 m_totalNumTpcs;
    bool                     m_ofmInSram;
    bool                     m_sliceTensorInSram;
    bool                     m_tilePattern;
};