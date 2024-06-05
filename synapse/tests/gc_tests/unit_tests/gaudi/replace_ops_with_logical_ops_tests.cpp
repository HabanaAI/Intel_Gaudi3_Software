#include "gc_tests/unit_tests/graph_optimizer_test.h"
#include "gaudi_graph.h"
#include "node_factory.h"

class GaudiReplaceOpsWithLogicOpsTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::pair<unsigned, unsigned>>
{
public:
    TensorPtr createPersistentTensor(unsigned dim, const SizeArray& sizes, synDataType type, std::string name)
    {
        synMemoryDescriptor persistentMemoryDesc(true);
        TensorPtr           t = TensorPtr(new Tensor(dim, sizes.data(), type));
        t->setName(name);
        t->setDramOffset(m_nextTensorOffset);
        t->setMemorySectionID(m_tId++);
        t->setMemoryDescriptor(persistentMemoryDesc);
        m_nextTensorOffset += t->getTotalSizeInBytes();
        return t;
    }

private:
    unsigned         m_tId              = 1;
    deviceAddrOffset m_nextTensorOffset = 0x1000;
};

TEST_P(GaudiReplaceOpsWithLogicOpsTest, optimization_for_broadcast_sequence_creation)
{
    GaudiGraph     g;
    const auto&    testParams       = GetParam();
    const unsigned sizeOfConcatDim  = testParams.first;
    const unsigned nodeSetFinalSize = testParams.second;

    const unsigned c = 1;
    const unsigned w = 3;
    const unsigned h = 3;
    const unsigned n = sizeOfConcatDim;

    const unsigned  dim         = 4u;
    const SizeArray sizes       = {c, w, h, n};
    const unsigned  concatNum   = 10;
    const SizeArray sizesConcat = {c, w, h, n * concatNum};
    const unsigned  concatAxis  = 3;

    const auto inputTensor  = createPersistentTensor(dim, sizes, syn_type_bf16, "concatInput");
    const auto outputTensor = createPersistentTensor(dim, sizesConcat, syn_type_bf16, "concatOutputTensor");

    TensorVector         concatInputs(concatNum, inputTensor);
    synConcatenateParams params = {.axis = concatAxis};
    const auto           concat =
        NodeFactory::createNode(concatInputs, {outputTensor}, &params, NodeFactory::concatenateNodeTypeName, "concat");

    GraphEditor::addNode(g, concat);

    ASSERT_TRUE(replaceOpsWithLogicalOps(g)) << "replaceOpsWithLogicalOps pass failed";

    auto nodeSet = g.getNodes();

    ASSERT_TRUE(nodeSet.size() == nodeSetFinalSize) << "The amount of nodes is not as expected";
}

// Test params:
// 1) Size of concat axis.
// 2) Expected number of nodes after the pass.
INSTANTIATE_TEST_SUITE_P(,
                         GaudiReplaceOpsWithLogicOpsTest,
                         ::testing::Values(std::make_pair(1, 1), std::make_pair(2, 3)));
