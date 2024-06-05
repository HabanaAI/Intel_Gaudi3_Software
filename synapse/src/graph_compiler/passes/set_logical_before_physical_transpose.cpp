#include "habana_graph.h"
#include "habana_pass.h"
#include "transpose_node.h"

/* When transpose node split to physical and logical transpose the sequence is: "physical -> logical".
The transpose node may have small FCD, and since the transpose output is strided,
when the conv node sliced, each DMA copy strided tensor with small FCD.
When all the consumers are MME node and the producer isn't MME node,
it may be good to give hint to the transpose solver to split the transpose to "logical -> physical" sequence,
and then the MME input will be dense. */

bool setLogicalBeforePhysicalTranspose(HabanaGraph& g)
{
    for (const auto& node : g.getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE)
        {
            std::shared_ptr<TransposeNode> transposeNode = std::dynamic_pointer_cast<TransposeNode>(node);
            HB_ASSERT_PTR(transposeNode);
            auto NodeRealProducers = g.getNodeRealProducers(transposeNode, Node::TENSOR_TYPE_DATA);
            auto NodeRealConsumers = g.getNodeRealConsumers(transposeNode, Node::TENSOR_TYPE_DATA);

            bool producerMME    = std::any_of(NodeRealProducers.begin(), NodeRealProducers.end(), [](NodePtr node) {
                return HabanaGraph::runsOnMME(node);
            });
            bool allConsumerMME = std::all_of(NodeRealConsumers.begin(),
                                              NodeRealConsumers.end(),
                                              [](NodePtr node) { return HabanaGraph::runsOnMME(node); }) &&
                                  NodeRealConsumers.size() > 0;  // We don't want this to be true for an empty set.

            if (allConsumerMME && !producerMME)
            {
                transposeNode->setPreferLogicalBeforePhysical(true);
            }
        }
    }
    return true;
}
