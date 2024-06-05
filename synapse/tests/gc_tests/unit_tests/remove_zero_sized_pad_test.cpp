#include "gaudi2_graph.h"
#include "perf_lib_layer_params.h"
#include "node_factory.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"

using namespace std;

class RemoveZeroSizedPadTest : public GraphOptimizerTest
{
};

TEST_F(RemoveZeroSizedPadTest, simple_one_zero_pad)
{
    //                 ________            ___
    // Creating:      |        |          |   |
    //          t1->->|Zero_Pad|->->t2->->|Add|->->t4
    //                |________|    t3->->|___|
    //
    //                 ___
    // Expecting:      |   |
    //          t1->->|Add|->->t4
    //          t3->->|___|

    Gaudi2Graph g;
    bool                   ret;
    std::list<std::string> keepNodePtrNames;
    const TSize            sizes[] = {1, 2, 2, 1};
    TensorPtr              t1      = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t4->setName("t4", true);
    vector<string>         expected_in_names_vec {"t1", "t3"};
    string                 expected_output = "t4";
    ns_PadKernelEx::Params params;
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));
    params.pads[0] = 0;
    params.pads[1] = 0;
    params.pads[2] = 0;
    params.pads[3] = 0;
    NodePtr pad    = NodeFactory::createNode({t1}, {t2}, &params, "pad_fwd_i32", "pad1");
    NodePtr add    = NodeFactory::createNode({t2, t3}, {t4}, nullptr, "add_fwd_i32", "add1");
    GraphEditor::addNode(g, pad);
    GraphEditor::addNode(g, add);
    keepNodePtrNames.push_back("add1");
    // run the pass
    ret = removeZeroSizedPad(g);
    ASSERT_TRUE(ret) << "failed to run handle identity cast nodes pass";

    const NodeVector& sortedNodes = g.getExeSortedNodes();
    ASSERT_EQ(sortedNodes.size(), 1);

    for (const NodePtr& node : sortedNodes)
    {
        if (node->getNodeName() == "add1")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_names_vec.begin(),
                                      expected_in_names_vec.end(),
                                      in_tensor->getName()) != expected_in_names_vec.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output);
        }
        keepNodePtrNames.remove(node->getNodeName());
    }
    ASSERT_EQ(keepNodePtrNames.size(), 0);
}

TEST_F(RemoveZeroSizedPadTest, simple_two_zero_pad)
{
    //                 ________            ________            ___
    // Creating:      |        |          |        |          |   |
    //          t1->->|Zero_Pad|->->t2->->|Zero_Pad|->->t3->->|Add|->->t5
    //                |________|          |________|    t4->->|___|
    //
    //                 ___
    // Expecting:     |   |
    //          t1->->|Add|->->t5
    //          t4->->|___|

    Gaudi2Graph g;
    bool                   ret;
    std::list<std::string> keepNodePtrNames;
    const TSize            sizes[] = {1, 2, 2, 1};
    TensorPtr              t1      = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t4->setName("t4", true);
    TensorPtr t5 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t5->setName("t5", true);
    vector<string>         expected_in_names_vec {"t1", "t4"};
    string                 expected_output = "t5";
    ns_PadKernelEx::Params params;
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));
    params.pads[0] = 0;
    params.pads[1] = 0;
    params.pads[2] = 0;
    params.pads[3] = 0;
    NodePtr pad1   = NodeFactory::createNode({t1}, {t2}, &params, "pad_fwd_i32", "pad1");
    NodePtr pad2   = NodeFactory::createNode({t2}, {t3}, &params, "pad_fwd_i32", "pad2");
    NodePtr add1   = NodeFactory::createNode({t3, t4}, {t5}, nullptr, "add_fwd_i32", "add1");
    GraphEditor::addNode(g, pad1);
    GraphEditor::addNode(g, pad2);
    GraphEditor::addNode(g, add1);
    keepNodePtrNames.push_back("add1");

    // run the pass
    ret = removeZeroSizedPad(g);
    ASSERT_TRUE(ret) << "failed to run handle identity cast nodes pass";

    const NodeVector& sortedNodes = g.getExeSortedNodes();
    ASSERT_EQ(sortedNodes.size(), 1);

    for (const NodePtr& node : sortedNodes)
    {
        if (node->getNodeName() == "add1")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_names_vec.begin(),
                                      expected_in_names_vec.end(),
                                      in_tensor->getName()) != expected_in_names_vec.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output);
        }
        keepNodePtrNames.remove(node->getNodeName());
    }
    ASSERT_EQ(keepNodePtrNames.size(), 0);
}

TEST_F(RemoveZeroSizedPadTest, graph_output_zero_pad)
{
    //                                                      ___
    //                               |-->-->-->-->-->-->-->|   |
    //                 ___           |    ________         |   |
    // Creating:      |   |          |   |        |        |Add|-->t5
    //          t1->->|Add|->->t3->->|-->|Zero_Pad|-->t4-->|___|
    //          t2->->|___|          |   |________|
    //                               |       ________
    //                               |      |        |
    //                               |-->-->|Zero_Pad|-->t6 (t6 is an output tensor of the graph)
    //                                      |________|
    //
    //                 ___                      ___
    // Expecting:     |   |          |-->-->-->|Add|-->t5
    //          t1->->|Add|->->t6->->|-->-->-->|___|
    //          t2->->|___|
    //

    Gaudi2Graph g;
    bool                   ret;
    std::list<std::string> keepNodePtrNames;
    const TSize            sizes[] = {1, 2, 2, 1};
    TensorPtr              t1      = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t4->setName("t4", true);
    TensorPtr t5 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t5->setName("t5", true);
    TensorPtr t6 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t6->setName("t6", true);
    t6->setMemoryDescriptor(std::move(synMemoryDescriptor(true)));
    vector<string>         expected_in_add1 {"t1", "t2"};
    vector<string>         expected_in_add2 {"t6", "t6"};
    string                 expected_output_add1 = "t6";
    string                 expected_output_add2 = "t5";
    ns_PadKernelEx::Params params;
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));
    params.pads[0] = 0;
    params.pads[1] = 0;
    params.pads[2] = 0;
    params.pads[3] = 0;
    NodePtr pad1   = NodeFactory::createNode({t3}, {t4}, &params, "pad_fwd_i32", "pad1");
    NodePtr pad2   = NodeFactory::createNode({t3}, {t6}, &params, "pad_fwd_i32", "pad2");
    NodePtr add1   = NodeFactory::createNode({t1, t2}, {t3}, nullptr, "add_fwd_i32", "add1");
    NodePtr add2   = NodeFactory::createNode({t3, t4}, {t5}, nullptr, "add_fwd_i32", "add2");
    GraphEditor::addNode(g, pad1);
    GraphEditor::addNode(g, pad2);
    GraphEditor::addNode(g, add1);
    GraphEditor::addNode(g, add2);
    keepNodePtrNames.push_back("add1");
    keepNodePtrNames.push_back("add2");

    // run the pass
    ret = removeZeroSizedPad(g);
    ASSERT_TRUE(ret) << "failed to run handle identity cast nodes pass";

    const NodeVector& sortedNodes = g.getExeSortedNodes();
    ASSERT_EQ(sortedNodes.size(), 2);

    for (const NodePtr& node : sortedNodes)
    {
        if (node->getNodeName() == "add1")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_add1.begin(), expected_in_add1.end(), in_tensor->getName()) !=
                            expected_in_add1.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output_add1);
        }
        if (node->getNodeName() == "add2")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_add2.begin(), expected_in_add2.end(), in_tensor->getName()) !=
                            expected_in_add2.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output_add2);
        }
        keepNodePtrNames.remove(node->getNodeName());
    }
    ASSERT_EQ(keepNodePtrNames.size(), 0);
}

TEST_F(RemoveZeroSizedPadTest, multiplie_consumers_zero_pad)
{
    // Creating:
    //               ________              ___
    //              |        |     t2---->|   |        ___
    //      t1-->-->|zero_pad|-->t3-->|-->|Add|->t4-->|   |
    //              |________|        |   |___|       |Add|-->t5
    //                                |-->-->-->-->-->|___|
    //                                |    ___
    //                                |   |   |
    //                                |-->|pad|-->t6
    //                                    |___|

    // Expecting:
    //                           ___
    //             t2-->-->---->|   |        ___
    //                 t1-->|-->|Add|->t4-->|   |
    //                      |   |___|       |Add|-->t5
    //                      |-->-->-->-->-->|___|
    //                      |    ___
    //                      |   |   |
    //                      |-->|pad|-->t6
    //                          |___|

    Gaudi2Graph g;
    bool                   ret;
    std::list<std::string> keepNodePtrNames;
    const TSize            sizes[] = {1, 2, 2, 1};
    TensorPtr              t1      = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t4->setName("t4", true);
    TensorPtr t5 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t5->setName("t5", true);
    TensorPtr t6 = TensorPtr(new Tensor(4U, sizes, syn_type_int32));
    t6->setName("t6", true);
    vector<string> expected_in_add1 {"t1", "t2"};
    vector<string> expected_in_add2 {"t1", "t4"};
    vector<string> expected_in_pad2 {"t1"};
    string         expected_output_add1 = "t4";
    string         expected_output_add2 = "t5";
    string         expected_output_pad2 = "t6";

    ns_PadKernelEx::Params params1;
    memset(&params1, 0, sizeof(ns_PadKernelEx::Params));
    params1.pads[0] = 0;
    params1.pads[1] = 0;
    params1.pads[2] = 0;
    params1.pads[3] = 0;
    NodePtr pad1    = NodeFactory::createNode({t1}, {t3}, &params1, "pad_fwd_i32", "pad1");

    ns_PadKernelEx::Params params2;
    memset(&params2, 0, sizeof(ns_PadKernelEx::Params));
    params2.pads[0] = 1;
    params2.pads[1] = 1;
    params2.pads[2] = 1;
    params2.pads[3] = 1;
    NodePtr pad2    = NodeFactory::createNode({t3}, {t6}, &params2, "pad_fwd_i32", "pad2");

    NodePtr add1 = NodeFactory::createNode({t3, t2}, {t4}, nullptr, "add_fwd_i32", "add1");
    NodePtr add2 = NodeFactory::createNode({t3, t4}, {t5}, nullptr, "add_fwd_i32", "add2");

    GraphEditor::addNode(g, pad1);
    GraphEditor::addNode(g, pad2);
    GraphEditor::addNode(g, add1);
    GraphEditor::addNode(g, add2);

    keepNodePtrNames.push_back("add1");
    keepNodePtrNames.push_back("add2");
    keepNodePtrNames.push_back("pad2");

    // run the pass
    ret = removeZeroSizedPad(g);
    ASSERT_TRUE(ret) << "failed to run handle identity cast nodes pass";

    const NodeVector& sortedNodes = g.getExeSortedNodes();
    ASSERT_EQ(sortedNodes.size(), 3);

    for (const NodePtr& node : sortedNodes)
    {
        if (node->getNodeName() == "add1")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_add1.begin(), expected_in_add1.end(), in_tensor->getName()) !=
                            expected_in_add1.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output_add1);
        }
        if (node->getNodeName() == "add2")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_add2.begin(), expected_in_add2.end(), in_tensor->getName()) !=
                            expected_in_add2.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output_add2);
        }
        if (node->getNodeName() == "pad2")
        {
            // check tensors input output changes
            for (auto in_tensor : node->getInputs())
            {
                ASSERT_TRUE(std::find(expected_in_pad2.begin(), expected_in_pad2.end(), in_tensor->getName()) !=
                            expected_in_pad2.end());
            }
            ASSERT_TRUE(node->getOutput(TENSOR_OFM)->getName() == expected_output_pad2);
        }
        keepNodePtrNames.remove(node->getNodeName());
    }
    ASSERT_EQ(keepNodePtrNames.size(), 0);
}