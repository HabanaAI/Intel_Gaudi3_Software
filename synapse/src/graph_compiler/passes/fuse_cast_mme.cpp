#include <memory>
#include <graph_editor.h>
#include "habana_graph.h"
#include "habana_pass.h"

#include "node.h"
#include "passes.h"
#include "quantization_data.h"

static bool canFuseCastToMme(HabanaGraph& g, synDataType toType, const NodePtr& mmeNode, const NodePtr& castNode)
{
    // validate that the MME engine support the cast output data type
    if (!g.getHALReader()->isSupportedMmeDataType(toType)) return false;

    // validate that the tensor between the cast and the MME node is not managed by user (i.e. persistent tensor)
    if (g.isUserManagedDram(castNode->getInput(0))) return false;

    // remove only if the blocking nodes of cast is subset of the mme blocking nodes
    const auto& castBlocking = g.getBlockingNodes(castNode);
    const auto& mmeBlocking  = g.getBlockingNodes(mmeNode);
    return std::includes(mmeBlocking.begin(), mmeBlocking.end(), castBlocking.begin(), castBlocking.end());
}

static void fuseCastIntoMmeNode(HabanaGraph& g, const NodePtr& castNode)
{
    const TensorPtr& output  = castNode->getOutput(0);
    const TensorPtr& input   = castNode->getInput(0);
    const NodePtr&   mmeNode = g.getTensorProducer(input);

    // Remove the node to disable the relationship with the graph output
    LOG_DEBUG(GC, "fuse Cast Into Mme Node '{}'", castNode->getNodeName());
    GraphEditor::removeNode(g, castNode);

    // Switch the mme node output
    GraphEditor::replaceTensor(g, mmeNode, input, output);

    // Set the mme rounding mode as the cast node's rounding mode
    mmeNode->setRoundingMode(castNode->getRoundingMode());

    MMENodePtr mmeNodePtr = std::dynamic_pointer_cast<MmeNode>(mmeNode);
    HB_ASSERT(mmeNodePtr, "could not downcast Node to MME Node");
    MmeExpBias MmeExpBias = mmeNodePtr->getMmeExpBias();
    MmeExpBias.fp8BiasOut = QuantizationData::getDefaultExpBias(output->getElementType());
    mmeNodePtr->setMmeExpBias(MmeExpBias);
}

bool fuseCastMme(HabanaGraph& g)
{
    if (!GCFG_FUSE_CAST_TO_MME.value()) return true;

    unsigned int removedNodes = 0;
    // We make a copy of the sortedNodes since during the loop we alter
    // the graph and thus invalidate the graph's sorted nodes cache.
    NodeVector sortedNodes = g.getExeSortedNodes();
    for (const auto& node : sortedNodes)
    {
        if (HabanaGraph::runsOnMME(node) && !MmeNode::isDmaOperation(node))
        {
            const pTensor output    = node->getOutput(0);
            const auto    consumers = g.getTensorConsumers(output);
            if (consumers.size() == 1)
            {
                auto castNode = consumers.front();
                if (castNode->isCast())
                {
                    synDataType castToType = castNode->getOutput(0)->getElementType();
                    if (canFuseCastToMme(g, castToType, node, castNode))
                    {
                        // Maintain origin_nodes tracking for debugging purposes
                        node->addOriginNodes(castNode->getOriginNodes());

                        fuseCastIntoMmeNode(g, castNode);
                        removedNodes++;
                    }
                }
            }
        }
    }
    if (removedNodes > 0)
    {
        g.turnOnPredicate(PREDICATE_ID_FUSED_NODE_TO_MME);
    }
    LOG_DEBUG(GC, "Removed {} fuse Cast Mme", removedNodes);
    return true;
}
