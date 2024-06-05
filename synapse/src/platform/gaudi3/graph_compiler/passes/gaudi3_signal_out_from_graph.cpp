#include "gaudi3_signal_out_from_graph.h"

#include "platform/gaudi3/graph_compiler/passes.h"
#include "types.h"

bool Gaudi3SignalOutFromGraph::executePass(HabanaGraph& g)
{
    m_outTensorsSorted = getSignalOutTensors(g);

    // Pass is currently not supported for gaudi3. If user set tensors to be
    // external - we will fail the compilation
    if (m_outTensorsSorted.size())
    {
        LOG_CRITICAL(GC, "Signaling from graph for Gaudi3 is currently not supported - failing compilation");
        return false;
    }

    return true;
}

namespace gaudi3
{
bool signalOutFromGraph(Gaudi3Graph& g)
{
    Gaudi3SignalOutFromGraph sigOut;

    return sigOut.executePass(g);
}
}  // namespace gaudi3
