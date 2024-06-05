#include "perf_lib_layer_params.h"
#include "node_factory.h"
#include "test_utils.h"
#include "graph_optimizer_test.h"
#include "gaudi2_graph.h"
#include "data_type_utils.h"
using namespace std;

class FP8CastInjectionTest : public GraphOptimizerTest
{
protected:
    std::pair<std::vector<synDataType>, std::vector<synDataType>> find_exact_or_prefix(
        const std::string&                                                                                    input,
        const std::unordered_map<std::string, std::pair<std::vector<synDataType>, std::vector<synDataType>>>& map)
    {
        // Search for an exact match
        auto exact_match = map.find(input);
        if (exact_match != map.end())
        {
            return exact_match->second;
        }

        // Search for a key that is a prefix of the input string
        for (const auto& entry : map)
        {
            if (input.substr(0, entry.first.size()) == entry.first)
            {
                return entry.second;
            }
        }

        // No matching key found
        return std::make_pair(std::vector<synDataType>(), std::vector<synDataType>());
    }
};

TEST_F(FP8CastInjectionTest, simple_fp8_injection_test)
{
    //                 ________            ___
    // Creating:      |        |          |   |
    //          t1->->|Zero_Pad|->->t2->->|Add|->->t4
    //                |________|    t3->->|___|
    //
    //

    Gaudi2Graph g;
    g.setInferenceMode(true);
    const TSize sizes[] = {1, 2, 2, 1};
    TensorPtr   t1      = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16));
    t4->setName("t4", true);

    ns_PadKernelEx::Params params;
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));
    NodePtr pad = NodeFactory::createNode({t1}, {t2}, &params, "pad_fwd_bf16", "pad1");
    NodePtr add = NodeFactory::createNode({t2, t3}, {t4}, nullptr, "add_fwd_bf16", "add1");
    GraphEditor::addNode(g, pad);
    GraphEditor::addNode(g, add);

    ASSERT_TRUE(castForTPCNodes(g));

    std::unordered_map<std::string, std::pair<std::vector<synDataType>, std::vector<synDataType>>> expectedTypes = {
        {"pad1", {{syn_type_bf16}, {syn_type_bf16}}},
        {"add1", {{syn_type_bf16, syn_type_bf16}, {syn_type_bf16}}},
        {"pad1_cast_input0", {{syn_type_fp8_152}, {syn_type_bf16}}},
        {"pad1_cast_output0", {{syn_type_bf16}, {syn_type_fp8_152}}},
        {"add1_cast_input0", {{syn_type_fp8_152}, {syn_type_bf16}}},
    };

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 5);
    for (const NodePtr& node : nodes)
    {
        auto                res_pair = find_exact_or_prefix(node->getNodeName(), expectedTypes);
        const TensorVector& inputs   = node->getInputs();
        for (int i = 0; i < inputs.size(); ++i)
        {
            ASSERT_EQ(inputs.at(i)->getElementType(), res_pair.first.at(i % res_pair.second.size()));
        }
        const TensorVector& outputs = node->getOutputs();
        for (int i = 0; i < outputs.size(); ++i)
        {
            ASSERT_EQ(outputs.at(i)->getElementType(), res_pair.second.at(i % res_pair.second.size()));
        }
        if (node->isCast())
        {
            ASSERT_EQ(getCastGUID(inputs[TENSOR_IFM]->getElementType(), outputs[TENSOR_OFM]->getElementType()),
                      node->getGUID());
        }
    }
}

TEST_F(FP8CastInjectionTest, multiple_consumers_fp8_injection_test)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize sizes[] = {1, 2, 2, 1};
    TensorPtr   t1      = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_bf16));
    t4->setName("t4", true);

    ns_PadKernelEx::Params params;
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));
    NodePtr pad1 = NodeFactory::createNode({t1}, {t2}, &params, "pad_fwd_bf16", "pad");
    NodePtr pad2 = NodeFactory::createNode({t1}, {t3}, &params, "pad_fwd_bf16", "pad");
    NodePtr pad3 = NodeFactory::createNode({t1}, {t4}, &params, "pad_fwd_bf16", "pad");
    GraphEditor::addNode(g, pad1);
    GraphEditor::addNode(g, pad2);
    GraphEditor::addNode(g, pad3);

    ASSERT_TRUE(castForTPCNodes(g));

    std::unordered_map<std::string, std::pair<std::vector<synDataType>, std::vector<synDataType>>> expectedTypes = {
        {"pad", {{syn_type_bf16}, {syn_type_bf16}}},
        {"pad_cast_input0", {{syn_type_fp8_152}, {syn_type_bf16}}}};

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 4);
    for (const NodePtr& node : nodes)
    {
        auto                res_pair = find_exact_or_prefix(node->getNodeName(), expectedTypes);
        const TensorVector& inputs   = node->getInputs();
        for (int i = 0; i < inputs.size(); ++i)
        {
            ASSERT_EQ(inputs.at(i)->getElementType(), res_pair.first.at(i % res_pair.second.size()));
        }
        const TensorVector& outputs = node->getOutputs();
        for (int i = 0; i < outputs.size(); ++i)
        {
            ASSERT_EQ(outputs.at(i)->getElementType(), res_pair.second.at(i % res_pair.second.size()));
        }
        if (node->isCast())
        {
            ASSERT_EQ(getCastGUID(inputs[TENSOR_IFM]->getElementType(), outputs[TENSOR_OFM]->getElementType()),
                      node->getGUID());
        }
    }
}

TEST_F(FP8CastInjectionTest, multiple_output_fp8_injection_test)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize sizes[] = {1, 2, 2, 1};
    TensorPtr   t1      = TensorPtr(new Tensor(4U, sizes, syn_type_float));
    t1->setName("t1", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t4->setName("t4", true);

    NodePtr equal_bwd = NodeFactory::createNode({t1}, {t3, t4}, nullptr, "equal_bwd_f32", "equal_bwd");
    GraphEditor::addNode(g, equal_bwd);

    ASSERT_TRUE(castForTPCNodes(g));

    std::unordered_map<std::string, std::pair<std::vector<synDataType>, std::vector<synDataType>>> expectedTypes = {
        {"equal_bwd", {{syn_type_float}, {syn_type_float}}},
        {"equal_bwd_cast_output0", {{syn_type_float}, {syn_type_fp8_152}}},
        {"equal_bwd_cast_output1", {{syn_type_float}, {syn_type_fp8_152}}}};

    const NodeVector& nodes = g.getExeSortedNodes();

    ASSERT_EQ(nodes.size(), 3);
    for (const NodePtr& node : nodes)
    {
        auto                res_pair = find_exact_or_prefix(node->getNodeName(), expectedTypes);
        const TensorVector& inputs   = node->getInputs();
        for (int i = 0; i < inputs.size(); ++i)
        {
            ASSERT_EQ(inputs.at(i)->getElementType(), res_pair.first.at(i % res_pair.second.size()));
        }
        const TensorVector& outputs = node->getOutputs();
        for (int i = 0; i < outputs.size(); ++i)
        {
            ASSERT_EQ(outputs.at(i)->getElementType(), res_pair.second.at(i % res_pair.second.size()));
        }
        if (node->isCast())
        {
            ASSERT_EQ(getCastGUID(inputs[TENSOR_IFM]->getElementType(), outputs[TENSOR_OFM]->getElementType()),
                      node->getGUID());
        }
    }
}

TEST_F(FP8CastInjectionTest, multiple_cast_output_consumers_fp8_injection_test)
{
    Gaudi2Graph g;
    g.setInferenceMode(true);

    const TSize sizes[] = {1, 2, 2, 1};
    TensorPtr   t1      = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t1->setName("t1", true);
    TensorPtr t2 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t2->setName("t2", true);
    TensorPtr t3 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t3->setName("t3", true);
    TensorPtr t4 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t4->setName("t4", true);
    TensorPtr t5 = TensorPtr(new Tensor(4U, sizes, syn_type_fp8_152));
    t5->setName("t5", true);

    ns_PadKernelEx::Params params;
    memset(&params, 0, sizeof(ns_PadKernelEx::Params));
    NodePtr pad1 = NodeFactory::createNode({t1}, {t2}, &params, "pad_fwd_bf16", "pad1");
    NodePtr pad2 = NodeFactory::createNode({t2}, {t3}, &params, "pad_fwd_bf16", "pad2");
    NodePtr pad3 = NodeFactory::createNode({t2}, {t4}, &params, "pad_fwd_bf16", "pad3");
    NodePtr pad4 = NodeFactory::createNode({t2}, {t5}, &params, "pad_fwd_bf16", "pad4");
    GraphEditor::addNode(g, pad1);
    GraphEditor::addNode(g, pad2);
    GraphEditor::addNode(g, pad3);
    GraphEditor::addNode(g, pad4);

    ASSERT_TRUE(castForTPCNodes(g));

    std::unordered_map<std::string, std::pair<std::vector<synDataType>, std::vector<synDataType>>> expectedTypes = {
        {"pad1", {{syn_type_bf16}, {syn_type_bf16}}},
        {"pad2", {{syn_type_bf16}, {syn_type_bf16}}},
        {"pad3", {{syn_type_bf16}, {syn_type_bf16}}},
        {"pad4", {{syn_type_bf16}, {syn_type_bf16}}},
        {"pad1_cast_input0", {{syn_type_fp8_152}, {syn_type_bf16}}},
        {"pad1_cast_output0", {{syn_type_bf16}, {syn_type_fp8_152}}},
        {"pad2_cast_input0", {{syn_type_fp8_152}, {syn_type_bf16}}},
        {"pad2_cast_output0", {{syn_type_bf16}, {syn_type_fp8_152}}},
        {"pad3_cast_output0", {{syn_type_bf16}, {syn_type_fp8_152}}},
        {"pad4_cast_output0", {{syn_type_bf16}, {syn_type_fp8_152}}}};

    const NodeVector& nodes = g.getExeSortedNodes();

    ASSERT_EQ(nodes.size(), expectedTypes.size());
    for (const NodePtr& node : nodes)
    {
        auto                res_pair = find_exact_or_prefix(node->getNodeName(), expectedTypes);
        const TensorVector& inputs   = node->getInputs();
        for (int i = 0; i < inputs.size(); ++i)
        {
            ASSERT_EQ(inputs.at(i)->getElementType(), res_pair.first.at(i % res_pair.second.size()));
        }
        const TensorVector& outputs = node->getOutputs();
        for (int i = 0; i < outputs.size(); ++i)
        {
            ASSERT_EQ(outputs.at(i)->getElementType(), res_pair.second.at(i % res_pair.second.size()));
        }
        if (node->isCast())
        {
            ASSERT_EQ(getCastGUID(inputs[TENSOR_IFM]->getElementType(), outputs[TENSOR_OFM]->getElementType()),
                      node->getGUID());
        }
    }
}