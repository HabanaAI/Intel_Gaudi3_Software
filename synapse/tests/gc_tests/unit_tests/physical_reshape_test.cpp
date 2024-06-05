#include "graph_optimizer_test.h"
#include "physical_reshape_node.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include <memory>

using namespace gc;

class PhysicalReshapeTest : public GraphOptimizerTest {};


TEST_F(PhysicalReshapeTest, simple)
{
    TSize inMaxSizes[] = {5, 5, 5, 5};
    TSize inMinSizes[] = {5, 2, 5, 5};
    TSize outMaxSizes[] = {5, 5, 25};
    TSize outMinSizes[] = {5, 2, 25};
    Tensor in(4, inMaxSizes, syn_type_single, inMinSizes);
    Tensor out(3, outMaxSizes, syn_type_single, outMinSizes);
    EXPECT_FALSE(PhysicalReshapeNode::requiresPhysicalReshapeToHandleDynamicity(in, out));
    outMaxSizes[1] *= 5;
    outMaxSizes[2] /= 5;
    outMinSizes[1] *= 5;
    outMinSizes[2] /= 5;
    Tensor in2(4, inMaxSizes, syn_type_single, inMinSizes);
    Tensor out2(3, outMaxSizes, syn_type_single, outMinSizes);
    EXPECT_TRUE(PhysicalReshapeNode::requiresPhysicalReshapeToHandleDynamicity(in2, out2));
}
