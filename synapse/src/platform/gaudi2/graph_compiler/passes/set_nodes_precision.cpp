#include "nodes_precision_selection.h"
#include "gaudi2_graph.h"

class Gaudi2NodesPrecisionSelection : public NodesPrecisionSelection
{
public:
    Gaudi2NodesPrecisionSelection() { m_minNodePrecision = syn_type_fp8_143; }

    virtual synDataType getPrecisionToRaise()
    {
        if (m_precisionToRaise == syn_type_na)
        {
            return syn_type_bf16;
        }
        return m_precisionToRaise;
    }
};

namespace gaudi2
{
bool setGraphNodesPrecision(Gaudi2Graph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(DATA_TYPES,
                  "Data type selection is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    Gaudi2NodesPrecisionSelection nodesPrecisionRunner;
    return nodesPrecisionRunner.runSetNodesPrecision(g);
}
}  // namespace gaudi2