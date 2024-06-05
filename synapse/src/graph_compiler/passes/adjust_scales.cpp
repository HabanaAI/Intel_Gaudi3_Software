#include "habana_pass.h"
#include "quantizer_factory.h"
#include <map>

void calcNumSuccessorsPerInput(HabanaGraph& g, NodePtr n, std::vector<uint32_t>& numSuccessorsPerInput)
{
    // fill numSuccessorsPerInput with the number of consumers of each input of n, in the order of the inputs.
    for (const TensorPtr& input : n->getInputs())
    {
        numSuccessorsPerInput.push_back(g.getNumberOfTensorConsumers(input));
    }
}

bool adjustScales(HabanaGraph& g)
{
    if (GCFG_QUANTIZATION_PARAMS_PATH.getValueStr() != std::string()) return true;

    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(DATA_TYPES, "Quantization is enabled in synapse only for Inference Mode. "
                              "Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    if (!GCFG_ENABLE_SYNAPSE_QUANTIZATION.value())
    {
        LOG_DEBUG(QUANT, "Quantization is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    //Get all nodes in topological order
    const NodeVector&                         graphNodes = g.getTopoSortedNodes();
    std::map<NodePtr , std::vector<uint32_t>> successorsMap;

    LOG_DEBUG(QUANT, "{}: calculating num of successors per input for each node", HLLOG_FUNC);

    // calculate successors per input for each node and save it in a map
    for (const NodePtr& n : graphNodes)
    {
        std::vector<uint32_t> numSuccessorsPerInput;
        calcNumSuccessorsPerInput(g, n, numSuccessorsPerInput);
        successorsMap[n] = numSuccessorsPerInput;
    }

    LOG_DEBUG(QUANT, "{}: adjusting scales backwards", HLLOG_FUNC);

    // run adjust scales backwards
    for (auto nodeIter = graphNodes.rbegin(); nodeIter != graphNodes.rend(); ++nodeIter)
    {
        const NodePtr& n = *nodeIter;
        QuantizerPtr q = n->getQuantizer();
        HB_ASSERT_PTR(q);
        q->adjustScales(g, n, false, successorsMap[n]);
    }

    LOG_DEBUG(QUANT, "{}: adjusting scales forwards", HLLOG_FUNC);

    // run adjust scales forwards
    for (const NodePtr& n : graphNodes)
    {
        QuantizerPtr q = n->getQuantizer();
        HB_ASSERT_PTR(q);
        q->adjustScales(g, n, true, successorsMap[n]);
    }

    return true;
}
