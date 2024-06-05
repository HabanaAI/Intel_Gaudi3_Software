#include <memory>

#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "graph_pattern_finder.h"
#include "node_factory.h"
#include "tensor.h"
#include "graph_optimizer_test.h"
#include "types.h"

TEST_F(GoNodeTest, graph_pattern_finder)
{
    TSize sizes[] = {1,1,1};
    TSize concatSizes[] = {2,1,1};
    // Create original graph

    // Tensor creation
    TensorPtr expandDimInput1(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr expandDimOutput1(new Tensor(3, sizes, syn_type_fixed));
    TensorPtr expandDimInput2(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr expandDimOutput2(new Tensor(3, sizes, syn_type_fixed));
    TensorPtr expandDimInput3(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr expandDimOutput3(new Tensor(3, sizes, syn_type_fixed));

    TensorVector concatinput;
    concatinput.push_back(expandDimOutput2);
    concatinput.push_back(expandDimOutput3);
    TensorPtr concatOutput(new Tensor(3, concatSizes, syn_type_fixed));

    TensorPtr memcpyOutput(new Tensor(3, sizes, syn_type_fixed));

    GaudiGraph origGraph;
    // Nodes creation
    unsigned expandDim = 0;
    auto expandDimNode1 = NodeFactory::createNode({expandDimInput1}, {expandDimOutput1}, &expandDim, NodeFactory::expandDimsNodeTypeName, "");
    auto expandDimNode2 = NodeFactory::createNode({expandDimInput2}, {expandDimOutput2}, &expandDim, NodeFactory::expandDimsNodeTypeName, "");
    auto expandDimNode3 = NodeFactory::createNode({expandDimInput3}, {expandDimOutput3}, &expandDim, NodeFactory::expandDimsNodeTypeName, "");
    auto concatNode = NodeFactory::createNode(concatinput, {concatOutput}, nullptr, NodeFactory::concatenateNodeTypeName, "");
    auto memcpyNode = NodeFactory::createNode({expandDimOutput1}, {memcpyOutput}, nullptr, NodeFactory::memcpyNodeTypeName, "");

    GraphEditor::addNode(origGraph, expandDimNode1);
    GraphEditor::addNode(origGraph, expandDimNode2);
    GraphEditor::addNode(origGraph, expandDimNode3);
    GraphEditor::addNode(origGraph, concatNode);
    GraphEditor::addNode(origGraph, memcpyNode);

    // Create pattern graph

    // Create pattern tensors
    TensorPtr    patternExpandDimInput(new Tensor(2, sizes, syn_type_fixed));
    TensorPtr    patternExpandDimOutput(new Tensor(3, sizes, syn_type_fixed));
    TensorVector patternConcatinput(1, patternExpandDimOutput);
    TensorPtr    patternconcatOutput(new Tensor(3, sizes, syn_type_fixed));

    // Create pattern nodes
    auto patternExpandDimNode = NodeFactory::createNode({patternExpandDimInput}, {patternExpandDimOutput}, &expandDim, NodeFactory::expandDimsNodeTypeName, "");
    auto patternConcatNode = NodeFactory::createNode(patternConcatinput, {patternconcatOutput}, nullptr, NodeFactory::concatenateNodeTypeName, "");

    GaudiGraph pattern;
    GraphEditor::addNode(pattern, patternExpandDimNode);
    GraphEditor::addNode(pattern, patternConcatNode);

    const auto& matchPatterns = GraphPatternFinder::matchPattern(origGraph, pattern);
    ASSERT_EQ(matchPatterns.size(), 2);
    ASSERT_NE(matchPatterns.front(), matchPatterns.back());
}
