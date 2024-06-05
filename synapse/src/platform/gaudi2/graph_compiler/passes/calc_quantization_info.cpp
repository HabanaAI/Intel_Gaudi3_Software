#include "habana_pass.h"
#include "habana_graph.h"
#include "quantization_utils.h"
#include "graph_compiler/quant_info_calculator.h"

using namespace QuantizationUtils;


class Gaudi2QuantInfoCalculator : public QuantInfoCalculator
{
protected:
    virtual bool canApplyPerChannelQuant(HabanaGraph& g, const NodePtr& n)
    {
        bool perChannel = GCFG_PER_CHANNEL_SCALING.value();
        LOG_TRACE(QUANT,
                  "Per channel scaling is {}abled, (PER_CHANNEL_SCALING={})",
                  (perChannel ? "en" : "dis"),
                  (perChannel ? "true" : "false"));
        return perChannel;
    }
};

bool calcQuantizationInfo(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(QUANT, "Graph is not in inference mode, calc quant info won't run.");
        return true;
    }
    Gaudi2QuantInfoCalculator quantCalculator;
    return quantCalculator.runCalcQuantInfo(g);
}

