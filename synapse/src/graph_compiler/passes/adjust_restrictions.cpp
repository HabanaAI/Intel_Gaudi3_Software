#include "habana_pass.h"
#include "quantizer_factory.h"

bool skipAdjustRestrictionsPass(HabanaGraph& g)
{
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

    if (GCFG_DISABLE_ADJUST_RESTRICTIONS.value())
    {
        LOG_DEBUG(QUANT, "{} Pass is disabled", HLLOG_FUNC);
        return true;
    }

    return false;
}

bool adjustRestrictions(HabanaGraph& g)
{
    if (skipAdjustRestrictionsPass(g))
    {
        return true;
    }

    //Get all nodes in topological order
    const NodeVector& graphNodes = g.getTopoSortedNodes();

    LOG_DEBUG(QUANT, "{}: adjusting restrictions backwards", HLLOG_FUNC);

    // run adjust restrictions backwards
    for (auto nodeIter = graphNodes.rbegin(); nodeIter != graphNodes.rend(); ++nodeIter)
    {
        const NodePtr& n = *nodeIter;
        QuantizerPtr q = n->getQuantizer();
        HB_ASSERT_PTR(q);
        q->adjustRestrictions(n, false);
    }

    LOG_DEBUG(QUANT, "{}: adjusting restrictions forwards", HLLOG_FUNC);

    // run adjust restrictions forwards
    for (const NodePtr& n : graphNodes)
    {
        QuantizerPtr q = n->getQuantizer();
        HB_ASSERT_PTR(q);
        q->adjustRestrictions(n, true);
    }

    return true;
}

