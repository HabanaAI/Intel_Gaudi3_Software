
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "types.h"
#include "handle_logical_operations.h"

class DynamicLogicalReshapeTest
: public GraphOptimizerTest
, public testing::WithParamInterface<::testing::tuple<SizeArray, SizeArray, SizeArray, SizeArray>>
{
public:
    struct GetName
    {
        template<class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
        {
            ::std::stringstream ss;

            ss << "inmax_inmin_inactual_outmax_outmin_outactual_"
               << toString(::testing::get<0>(info.param), 'x') << "_"
               << toString(::testing::get<1>(info.param), 'x') << "_"
               << toString(::testing::get<2>(info.param), 'x') << "_"
               << toString(::testing::get<3>(info.param), 'x');

            return ss.str();
        }
    };
};

TEST_P(DynamicLogicalReshapeTest, test_logical_reshape)
{
    GaudiGraph     g;
    const unsigned tensorDim = 4;

    SizeArray inMaxSize  = ::testing::get<0>(GetParam());
    SizeArray inMinSize  = ::testing::get<1>(GetParam());
    SizeArray outMaxSize = ::testing::get<2>(GetParam());
    SizeArray outMinSize = ::testing::get<3>(GetParam());

    auto tFwdIn  = std::make_shared<Tensor>(tensorDim, inMaxSize.data(), syn_type_float, inMinSize.data());
    auto tFwdOut = std::make_shared<Tensor>(tensorDim, outMaxSize.data(), syn_type_float, outMinSize.data());
    auto tShape  = std::make_shared<Tensor>(tensorDim,
                                           outMaxSize.data(),
                                           syn_type_float,
                                           nullptr,
                                           nullptr,
                                           false,
                                           false,
                                           INVALID_BATCH_POS,
                                           outMinSize.data(),
                                           SHAPE_TENSOR);
    // Set graph's input tensor as persistent
    synMemoryDescriptor inputMemDesc(true);
    tFwdIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    tFwdIn->setMemoryDescriptor(inputMemDesc);
    // Set graph's output tensor as persistent
    synMemoryDescriptor outputMemDesc(true);
    tFwdOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tFwdOut->setMemoryDescriptor(outputMemDesc);

    pNode nodeReluBwd = NodeFactory::createNode({tFwdIn, tShape}, {tFwdOut}, nullptr, "reshape", "reshape_it_myman");
    GraphEditor::addNode(g, nodeReluBwd);

    ASSERT_TRUE(g.compile());
    const NodeVector& nodes = g.getExeSortedNodes();
    for (const pNode& node : nodes)
    {
        // Check no TPC inserted since this is logical reshape
        ASSERT_TRUE(node->getNodeType() != Node::TYPE_USER &&
                    node->getNodeType() != Node::TYPE_INTERNAL_PACKING)
            << "Found an unexpected node in graph";
    }
}

INSTANTIATE_TEST_SUITE_P(,
                        DynamicLogicalReshapeTest,
                        ::testing::Values(::testing::make_tuple(SizeArray {5, 5, 5, 5, 1},
                                                                SizeArray {5, 5, 2, 5, 1},
                                                                SizeArray {25, 5, 5, 1, 1},
                                                                SizeArray {25, 2, 5, 1, 1}),
                                          ::testing::make_tuple(SizeArray {6, 5, 7, 11, 1},
                                                                SizeArray {6, 5, 2, 11, 1},
                                                                SizeArray {2, 15, 7, 11, 1},
                                                                SizeArray {2, 15, 2, 11, 1})),
                        DynamicLogicalReshapeTest::GetName());

TEST_F(GraphOptimizerTest, static_reshape)
{
    GaudiGraph g;

    TSize inMaxSizes[]    = {4, 5, 6, 7};
    TSize inMinSizes[]    = {4, 5, 4, 3};
    TSize sliceMaxSizes[] = {1, 5, 6, 7};
    TSize sliceMinSizes[] = {1, 5, 4, 3};
    TSize outMaxSizes[]   = {1, 30, 7};
    TSize outMinSizes[]   = {1, 20, 3};

    TensorPtr in    = std::make_shared<Tensor>(4, inMaxSizes, syn_type_float, inMinSizes);
    TensorPtr midd1 = std::make_shared<Tensor>(4, sliceMaxSizes, syn_type_float, sliceMinSizes);
    TensorPtr midd2 = std::make_shared<Tensor>(3, outMaxSizes, syn_type_float, outMinSizes);
    TensorPtr out   = std::make_shared<Tensor>(3, outMaxSizes, syn_type_float, outMinSizes);

    synMemoryDescriptor memDesc(true);

    in->setMemoryDescriptor(memDesc);
    out->setMemoryDescriptor(memDesc);

    SliceNode::SliceNodeStaticParams p;
    for (auto i = 0; i < 4; ++i)
    {
        p.steps[i]  = 1;
        p.starts[i] = 0;
        p.ends[i]   = sliceMaxSizes[i];
    }

    auto slice =
        NodeFactory::createNode({in}, {midd1}, (void*)&p, sizeof(p), NodeFactory::logicalSliceFwdNodeTypeName, "slice");
    auto staticReshape =
        NodeFactory::createNode({midd1}, {midd2}, nullptr, NodeFactory::staticReshapeNodeTypeName, "static_reshape");
    auto neg = NodeFactory::createNode({midd2}, {out}, nullptr, "neg_f32", "neg");

    GraphEditor::addNode(g, slice);
    GraphEditor::addNode(g, staticReshape);
    GraphEditor::addNode(g, neg);

    LogicalOpsHandler handler(g);
    bool              ret = handler.handleLogicalOps();

    // validations:
    ASSERT_TRUE(ret) << "handleLogicalOperations failed";
    ASSERT_EQ(g.getExeSortedNodes().size(), 3) << "handleLogicalOperations inserted memcpy node";
}
