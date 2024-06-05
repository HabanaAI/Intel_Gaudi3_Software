#include "habana_pass.h"
#include "habana_graph.h"
#include "utils.h"
#include "types.h"
#include <map>

class ancestorsMinMax
{
public:
    ancestorsMinMax() : min(0), max(0) {}
    unsigned min;
    unsigned max;
};

typedef std::map<NodePtr , ancestorsMinMax> NodeDistanceMap;
typedef std::map<NodePtr , NodeDistanceMap> DescendantsMap;

void calculateDescendants(HabanaGraph& g, DescendantsMap& descendantsMap)
{
    // Map for each node, the minimum and maximum distance for ALL of its descendants.
    const NodeVector& graphNodes = g.getExeSortedNodes();
    for (auto nodeIter = graphNodes.rbegin(); nodeIter != graphNodes.rend(); ++nodeIter)
    {
        const NodePtr& node = *nodeIter;
        for (const TensorPtr& inputTensor : node->getInputs())
        {
            NodePtr inputNode = g.getTensorProducer(inputTensor);
            if (inputNode == nullptr)
            {
                continue;
            }

            // the minimum distance is 0 as this is the input to "node"
            descendantsMap[inputNode][node].min = 0;

            // for each descendant of "node", calculate the distance between it and the input node
            for (const std::pair<const NodePtr, ancestorsMinMax>& p : descendantsMap[node])
            {
                if (descendantsMap[inputNode].find(p.first) == descendantsMap[inputNode].end())
                {
                    // the descendant of "node" is not connected to the input node,
                    // so the distance from the input node is the descendant distance + 1
                    descendantsMap[inputNode][p.first].min = p.second.min + 1;
                    descendantsMap[inputNode][p.first].max = p.second.max + 1;
                }
                else
                {
                    // the descendant of "node" is connected to the input node, so increase its max value by 1
                    descendantsMap[inputNode][p.first].max += 1;
                }
            }
        }
    }
}

bool isIndirectAncestor(const NodePtr& ancestorNode, const NodePtr& ofNode, DescendantsMap& descendantsMap)
{
    // a node is an indirect ancestor of another node if the minimum distance is 0 but the maximum distance is not 0.
    // it means that there is a direct path to the node (min=0) and an indirect path to the node (max!=0)
    return descendantsMap[ancestorNode].find(ofNode) != descendantsMap[ancestorNode].end() &&
           descendantsMap[ancestorNode][ofNode].min == 0 &&
           descendantsMap[ancestorNode][ofNode].max != 0;
}

bool lockAncestorsForRequant(HabanaGraph& g)
{
    if (!GCFG_ENABLE_SYNAPSE_QUANTIZATION.value())
    {
        LOG_DEBUG(QUANT, "Quantization is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    // lock indirect ancestors.
    // if a node has a direct input node, that one of its outputs is connected indirectly to the node,
    // then lock the related input tensor.
    // the reason is to force requant on such conflicts.
    // Start by calculating the maximum and minimum distance from every node to all of its descendants and saving it in a map.
    DescendantsMap descendantsMap;
    LOG_DEBUG(QUANT, "{}: calculating decendants for all nodes in the graph", HLLOG_FUNC);
    calculateDescendants(g, descendantsMap);

    // Get all nodes in topological order
    const NodeVector& graphNodes = g.getExeSortedNodes();
    // lock common ancestors backwards
    LOG_DEBUG(QUANT, "{}: locking indirect ancestors for requant", HLLOG_FUNC);
    for (auto nodeIter = graphNodes.rbegin(); nodeIter != graphNodes.rend(); ++nodeIter)
    {
        const NodePtr& node = *nodeIter;
        for (const TensorPtr& inputTensor : node->getInputs())
        {
            NodePtr inputNode = g.getTensorProducer(inputTensor);
            if (inputNode == nullptr)
            {
                continue;
            }

            if (isIndirectAncestor(inputNode, node, descendantsMap))
            {
                // lock the indirect ancestor tensor
                LOG_DEBUG(QUANT,
                          "{}: node {} is an indirect ancestor of {}",
                          HLLOG_FUNC,
                          inputNode->getNodeName(),
                          node->getNodeName());
                inputTensor->lockQuantization(inputNode);
            }
        }
    }

    return true;
}
