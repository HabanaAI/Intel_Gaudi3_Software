#include "tensor.h"
#include "node.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "graph_optimizer_test.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "types.h"
#include "graph_compiler/habana_pass.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"

namespace gaudi3
{
#define DCORENr 4
using TestSizes      = std::array<unsigned, DCORENr>;
using TestParamTuple = std::tuple<unsigned,  // perforated dim size
                                  unsigned,  // perforated dim granularity
                                  bool,
                                  TestSizes>;

class DcoreSplitTest
: public GraphOptimizerTest
, public testing::WithParamInterface<TestParamTuple>
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
    }
};

TEST_P(DcoreSplitTest, dcore_test)
{
    ScopedConfigurationChange enableDcoreLocality("ENABLE_DCORE_LOCALITY_SPLIT", "true");

    const unsigned  perforationDim = 0;
    const unsigned  dimSize        = std::get<0>(GetParam());
    const unsigned  granularity    = std::get<1>(GetParam());
    const bool      isPerforated   = std::get<2>(GetParam());
    const TestSizes dcoreSplits    = std::get<3>(GetParam());

    const unsigned tensor_dim = 1;
    const TSize    size       = 1;
    Gaudi3Graph    g;
    TensorPtr      i = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr      o = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorVector   inputTensors;
    TensorVector   outputTensors;

    inputTensors.push_back(i);
    outputTensors.push_back(o);

    NodePtr tpcNode = NodeFactory::createNode({i}, {o}, nullptr, "mult_fwd_f32", "mult");

    tpcNode->getNodeAnnotation().perforation = LitePerforationHints {perforationDim, granularity, false};
    GraphEditor::addNode(g, tpcNode);

    NodeROI* tpcNodeRoi = new NodeROI();
    tpcNodeRoi->size[0]  = dimSize;

    std::list<NodeROI>* nodeRois = g.GetNodeROIs(tpcNode);
    nodeRois->push_back(*tpcNodeRoi);

    splitToDcoreROIs(g);

    ASSERT_EQ(tpcNode->getNodeAnnotation().perforation->isPerforated, isPerforated);

    for (unsigned dcore = 0; DCORENr < DCORENr; dcore++)
    {
        ASSERT_EQ(tpcNodeRoi->dcoreROIs[dcore].size[perforationDim], dcoreSplits[dcore]);
    }

    delete tpcNodeRoi;
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DcoreSplitTest,
    ::testing::Values(std::make_tuple(3, 1, false, TestSizes {0, 0, 0, 0}),   // not enough work
                      std::make_tuple(11, 2, false, TestSizes {0, 0, 0, 0}),  // sizes doesnt match granleiry
                      std::make_tuple(8, 2, true, TestSizes {2, 2, 2, 2}),    // basic split
                      std::make_tuple(24, 2, true, TestSizes {6, 6, 6, 6}),   // larger split
                      std::make_tuple(10, 2, true, TestSizes {4, 2, 2, 2}),   // uneven split
                      std::make_tuple(14, 2, true, TestSizes {4, 4, 4, 2})    // uneven split 2
                      ));
}  // namespace gaudi3