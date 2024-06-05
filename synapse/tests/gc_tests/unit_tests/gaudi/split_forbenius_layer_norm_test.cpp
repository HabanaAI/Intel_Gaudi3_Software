
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "perf_lib_layer_params.h"
#include <string>

namespace
{
// Testing Frobenius Norm Node Output pass
class GaudifrobeniusNormNode : public GraphOptimizerTest
{
};
}  // namespace

TEST_F(GaudifrobeniusNormNode, basicTest)
{
    TSize inSize[]  = {128, 8, 4, 1};
    TSize outSize[] = {1};
    synMemoryDescriptor persistentMemoryDesc(true);

    pTensor inTensor(new Tensor(4, inSize, syn_type_bf16));
    inTensor->setName("inputTensor");
    inTensor->setDramOffset(0x1000);
    inTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    inTensor->setMemoryDescriptor(persistentMemoryDesc);

    pTensor outTensor(new Tensor(1, outSize, syn_type_bf16));
    outTensor->setName("outputTensor");
    outTensor->setMemoryDescriptor(persistentMemoryDesc);
    outTensor->setDramOffset(0x5000);
    outTensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    pNode frobeniusNorm = NodeFactory::createNode({inTensor},
                                                  {outTensor},
                                                  nullptr,
                                                  NodeFactory::FrobeniusNormTypeName,
                                                  "lpnorm_frobenius_fwd");

    ASSERT_TRUE(frobeniusNorm != nullptr);

    GaudiGraph g;
    GraphEditor::addNode(g, frobeniusNorm);

    bool retVal = g.compile();
    ASSERT_TRUE(retVal) << "Failed to compile graph";

    const NodeVector& nodes = g.getExeSortedNodes();
    ASSERT_GE(nodes.size(), 2) << "Got " << nodes.size() << ", Expected >= 2";
}
