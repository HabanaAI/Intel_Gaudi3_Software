#include "brain/bundler/bundle_break_rules.h"
#include "graph_optimizer_test.h"
#include "gaudi3_graph.h"
#include "node_factory.h"
#include "platform/gaudi3/graph_compiler/passes.h"

using namespace gc::layered_brain::bundler;

class BundlerBreakRulesTest : public GraphOptimizerTest
{
public:
    Gaudi3Graph m_graph;

    TensorPtr newTensor(const std::vector<TSize>& shape) const
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_float);
    }

    NodePtr getDbgNode() const
    {
        auto in  = newTensor({100, 100});
        auto out = newTensor({100, 100});
        return NodeFactory::createDebugNode(in, out);
    }
};

class BundlerAccessPatternBreakRuleTest : public BundlerBreakRulesTest
{
public:
    NodePtr getNode() const
    {
        auto output = newTensor({128, 128});

        ns_ConstantKernel::Params_v2 param {};
        return NodeFactory::createNode({}, {output}, &param, "constant_f32", "Const");
    }

    bool applyHasAccessPatternRule(const NodePtr& node) const
    {
        RuleContext ctx(m_graph, node);
        auto        rule = RuleLibrary::instance().getRule(RuleType::HAS_ACCESS_PATTERN);
        return rule->apply(ctx);
    }
};

TEST_F(BundlerAccessPatternBreakRuleTest, rule_should_allow_tpc_nodes_with_valid_access_pattern)
{
    auto node = getNode();
    GraphEditor::addNode(m_graph, node);
    gaudi3::loadTpcKernels(m_graph);

    EXPECT_TRUE(applyHasAccessPatternRule(node));
}

TEST_F(BundlerAccessPatternBreakRuleTest, rule_should_disallow_tpc_nodes_without_access_pattern)
{
    auto node = getDbgNode();
    GraphEditor::addNode(m_graph, node);

    EXPECT_FALSE(applyHasAccessPatternRule(node));
}

TEST_F(BundlerAccessPatternBreakRuleTest, rule_should_disallow_tpc_nodes_with_invalid_access_pattern)
{
    auto node = getNode();
    GraphEditor::addNode(m_graph, node);
    gaudi3::loadTpcKernels(m_graph);
    node->getNodeAccessPattern();  // Force access pattern generation

    // adding an input after access pattern generation, without graph editor and
    // re-instantiation will cause this tensor not to have access pattern mapping.
    node->addInput(0, newTensor({100, 100}));

    EXPECT_FALSE(applyHasAccessPatternRule(node));
}