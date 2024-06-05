#include <stdint.h>
#include <memory>

#include "node_factory.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"

#define INPUT_OFFSET  0x1000000
#define OUTPUT_OFFSET 0x2000000

class SliceFcdTest : public GraphOptimizerTest
{
public:
    synSliceParams createDefaultSliceParams(const TSize sizes[], int dimCount)
    {
        synSliceParams params;
        memset(params.axes, 0, sizeof(params.axes));
        memset(params.ends, 0, sizeof(params.ends));
        memset(params.starts, 0, sizeof(params.starts));
        memset(params.steps, 0, sizeof(params.steps));

        for (auto i = 0; i < dimCount; i++)
        {
            params.axes[i]   = i;
            params.starts[i] = 0;
            params.ends[i]   = sizes[i];
            params.steps[i]  = 1;
        }

        return params;
    }

    TensorPtr createPersistentTensor(const unsigned dims, const TSize* shape, synDataType dataType, unsigned offset)
    {
        auto tensor = std::make_shared<Tensor>(dims, shape, dataType);

        synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(offset);
        tensor->setMemorySectionID(++m_sectionId);
        tensor->map();
        return tensor;
    }

private:
    unsigned m_sectionId = 0;
};

TEST_F(SliceFcdTest, high_utilization_slice_fcd)
{
    GaudiGraph     g;
    const unsigned dims        = 5;
    const TSize    in_sizes[]  = {256, 5, 2, 1, 1};
    const TSize    out_sizes[] = {128, 5, 2, 1, 1};

    TensorPtr tensorIn  = createPersistentTensor(dims, in_sizes, syn_type_float, INPUT_OFFSET);
    TensorPtr tensorOut = createPersistentTensor(dims, out_sizes, syn_type_float, OUTPUT_OFFSET);

    synSliceParams params = createDefaultSliceParams(in_sizes, dims);
    params.ends[0]        = out_sizes[0];

    NodePtr n = NodeFactory::createNode({tensorIn}, {tensorOut}, &params, NodeFactory::sliceNodeTypeName, "slice");
    GraphEditor::addNode(g, n);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    ASSERT_LE(g.getExeSortedNodes().size(), 2) << "Unexpected number of nodes - expecting 2 nodes: slice and memcpy";
}

TEST_F(SliceFcdTest, low_utilization_slice_fcd)
{
    GaudiGraph     g;
    const unsigned dims        = 5;
    const TSize    in_sizes[]  = {10, 32, 128, 1, 1};
    const TSize    out_sizes[] = {1, 32, 128, 1, 1};

    TensorPtr tensorIn  = createPersistentTensor(dims, in_sizes, syn_type_float, INPUT_OFFSET);
    TensorPtr tensorOut = createPersistentTensor(dims, out_sizes, syn_type_float, OUTPUT_OFFSET);

    synSliceParams params = createDefaultSliceParams(in_sizes, dims);
    params.ends[0]        = out_sizes[0];

    NodePtr n = NodeFactory::createNode({tensorIn}, {tensorOut}, &params, NodeFactory::sliceNodeTypeName, "slice");
    GraphEditor::addNode(g, n);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    ASSERT_GT(g.getExeSortedNodes().size(), 2)
        << "Unexpected number of nodes - expecting slice fcd optimization to add nodes";
}

TEST_F(SliceFcdTest, strided_slice_fcd)
{
    GaudiGraph     g;
    const unsigned dims        = 5;
    const TSize    in_sizes[]  = {12, 128, 2, 1, 1};
    const TSize    out_sizes[] = {6, 128, 2, 1, 1};

    TensorPtr tensorIn  = createPersistentTensor(dims, in_sizes, syn_type_float, INPUT_OFFSET);
    TensorPtr tensorOut = createPersistentTensor(dims, out_sizes, syn_type_float, OUTPUT_OFFSET);

    synSliceParams params = createDefaultSliceParams(in_sizes, dims);
    params.steps[0]       = 2;

    NodePtr n = NodeFactory::createNode({tensorIn}, {tensorOut}, &params, NodeFactory::sliceNodeTypeName, "slice");
    GraphEditor::addNode(g, n);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    ASSERT_GT(g.getExeSortedNodes().size(), 2)
        << "Unexpected number of nodes - expecting slice fcd optimization to add nodes";
}