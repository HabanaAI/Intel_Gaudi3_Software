#include "habana_graph.h"
#include "reduction_node.h"

// TODO: [SW-26372] Optimize to run only when adding internal reduction node
bool linkReductionMemsetShapes(HabanaGraph& graph)
{
    for (auto& node : graph.getNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_REDUCTION)
        {
            auto reductionNode = std::dynamic_pointer_cast<ReductionNode>(node);
            HB_ASSERT_PTR(reductionNode);

            bool res = reductionNode->linkConsumedMemsetShape(graph);
            CHECK_RET_FALSE(res, "Failed to link memset inputs of reduction node {} to a non internal memset producer.",
                            node->getNodeName());
        }
    }
    return true;
}