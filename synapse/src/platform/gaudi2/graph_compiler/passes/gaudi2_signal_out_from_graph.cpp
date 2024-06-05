#include "gaudi2_signal_out_from_graph.h"

#include "graph_compiler/compilation_hal_reader.h"
#include "habana_device_types.h"
#include "habana_global_conf.h"
#include "habana_pass.h"
#include "node.h"
#include "node_annotation.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "types.h"

#include <algorithm>
#include <memory>
#include <vector>

void Gaudi2SignalOutFromGraph::setInitialSyncValues(HabanaGraph& g)
{
    if (m_outTensorsSorted.size() == 0) return;  // no signal-out tensors

    auto supportedDevices = g.getHALReader()->getSupportedDeviceTypes();

    for (auto& deviceType : supportedDevices)
    {
        if (m_deviceToSignals.find(deviceType) == m_deviceToSignals.end())
        {
            // device type is not signaling out - set its sync object to the max value
            g.getCodeGenerator()->getDeviceSfgInitValue()[deviceType] = m_outTensorsSorted.size();
        }
        else
        {
            unsigned startSignal = m_deviceToSignals[deviceType][0];
            if (startSignal == 0)
            {
                // Init value for SFG is 0 - no need to set it up
                continue;
            }
            g.getCodeGenerator()->getDeviceSfgInitValue()[deviceType] = startSignal;
        }
    }
}

void Gaudi2SignalOutFromGraph::init(HabanaGraph& g)
{
    commonInit(g);

    setInitialSyncValues(g);
}

void Gaudi2SignalOutFromGraph::addSigOutGroupMonitors(HabanaGraph&     g,
                                                      const NodeSet&   producers,
                                                      const TensorPtr& t,
                                                      int              index)
{
    for (auto& producer : producers)
    {
        auto device = g.getNodeUtility().getNodeDeviceType(producer);

        producer->getNodeAnnotation().sfgSyncObjValue.set(m_tensorsToSignals[index][device]);
    }
}

namespace gaudi2
{
bool signalOutFromGraph(Gaudi2Graph& g)
{
    Gaudi2SignalOutFromGraph sigOut;

    return sigOut.executePass(g);
}
}  // namespace gaudi2
