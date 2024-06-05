#include "passes.h"
#include "node_utils.h"

static bool isConvertValidToRemove(const HabanaGraph& g, const NodePtr& n)
{
    // making sure convert has 2 inputs and 1 output
    auto nodeConsumers = g.getTensorConsumers(n->getOutput(0));
    if (nodeConsumers.size() != 1)
    {
        LOG_DEBUG(DATA_TYPES, "Convert node {} have {} consumers", n->getNodeName(), nodeConsumers.size());
        return false;
    }
    if (n->getInputs().size() != 2)
    {
        LOG_DEBUG(DATA_TYPES, "Convert node {} have {} inputs", n->getNodeName(), nodeConsumers.size());
        return false;
    }
    // checking scale is a single element.
    // [SW-169025] Support multi values scale in remove contiguous converts pass
    TensorPtr scaleTensor = n->getInput(CONVERT_INV_SCALE_IDX);
    if (!scaleTensor || scaleTensor->getTotalElements() != 1 || !scaleTensor->isStaticParam())
    {
        return false;
    }
    if (scaleTensor->getElementType() != syn_type_float && scaleTensor->getElementType() != syn_type_bf16)
    {
        LOG_DEBUG(DATA_TYPES, "Convert node {} scale dtype is different from f32", n->getNodeName());
        return false;
    }
    LOG_TRACE(DATA_TYPES, "Convert node {} is candidate for removal", n->getNodeName());
    return true;
}

static bool isContiguousConvertsValidToRemove(const NodePtr& n1, const NodePtr& n2)
{
    // check that n1 convert node have opposite direction to n2 convert node
    if(isConvertToFp8Node(n1) != isConvertFromFp8Node(n2))
    {
        return false;
    }
    if (n2->getOutput(0)->isPersistent() || n1->getOutput(0)->isPersistent() ||
        n2->getInput(0)->isPersistent())
    {
        return false;
    }

    synDataType castToTypeN1   = n1->getOutput(0)->getElementType();
    synDataType castFromTypeN2 = n2->getInput(0)->getElementType();

    float scaleN1 = getConvertNodeScale(n1);
    float scaleN2 = getConvertNodeScale(n2);

    return castToTypeN1 == castFromTypeN2 && scaleN1 == 1/scaleN2;
}

bool removeContinguousConverts(HabanaGraph& g)
{
    const NodeSet& nodes = g.getNodes();
    NodeVector convertNodesToRemove;
    NodeVector convertNodesProducers;

    for (const NodePtr& node : nodes)
    {
        if (!isConvertFp8Node(node) || !isConvertValidToRemove(g, node)) continue;

        auto nodeConsumers = g.getTensorConsumers(node->getOutput(0));
        auto prevNode = g.getTensorProducer(node->getInput(0));
        auto nextNode = nodeConsumers.front();

        if (!isConvertFp8Node(nextNode) || !isConvertValidToRemove(g, nextNode)) continue;

        if (!isContiguousConvertsValidToRemove(node, nextNode)) continue;

        LOG_TRACE(DATA_TYPES, "Nodes {}, {} are contiguous convert nodes",
                  node->getNodeName(), nextNode->getNodeName());

        convertNodesToRemove.push_back(node);
        convertNodesProducers.push_back(prevNode);
        convertNodesToRemove.push_back(nextNode);
        convertNodesProducers.push_back(prevNode);
    }

    int removedNodesAmount = convertNodesToRemove.size();
    GraphEditor::removeNodes(g, convertNodesToRemove, convertNodesProducers);
    LOG_TRACE(DATA_TYPES, "{} contiguous converts nodes were removed", removedNodesAmount);

    return true;
}
