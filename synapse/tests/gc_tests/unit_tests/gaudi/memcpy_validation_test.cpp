#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"

namespace gaudi
{

class MemcpyValidationDeathTest : public GraphOptimizerTest
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

// Create memcopy node, and set it's input and output in SRAM.
// graph: [t0]-->(memcpyNode)--[t1]
void sram_to_sram_death(bool cast)
{
    GaudiGraph g;
    TSize sizes[] = {1, 1, 1};
    pTensor t0(new Tensor(3, sizes, syn_type_float));
    synDataType type = cast? syn_type_bf16 : syn_type_float;
    pTensor t1(new Tensor(3, sizes, type));
    // Set both tensors in SRAM, which is invalid in Gaudi
    t0->setTensorInSram();
    t1->setTensorInSram();
    auto n = NodeFactory::createNode({t0}, {t1}, nullptr, NodeFactory::memcpyNodeTypeName, "node_memcpy");
    GraphEditor::addNode(g, n);
    // This function is expected to assert and fail for this graph in debug
    selectMemcpyEngine(g);
}

// The test calls directly the selectMemcpyEngine pass with invalid graphs, and validates that assert occurs in debug.
// ASSERT_DEBUG_DEATH triggers the function in a separate thread, and validates it terminates.
// It then looks for a matching string in stderr.
// Notice the function is called on a different thread, so breakpoints won't catch.
// see https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/docs/advanced.md#death-tests
TEST_F(MemcpyValidationDeathTest, sram_to_sram)
    {
        ASSERT_DEBUG_DEATH({ sram_to_sram_death(false /*cast*/); }, ".* Assertion .*validateMemCopy.*failed.*");
        // Create memcopy node with different input and output data types, so a tpc node
        // of cast is inserted to the graph instead of the original (semantic) memcpy node.
        ASSERT_DEBUG_DEATH({ sram_to_sram_death(true /*cast*/); }, ".* Assertion .*validateMemCopy.*failed.*");
    }

} // namespace gaudi