#include "signal_out_from_graph.h"

#include "compilation_hal_reader.h"
#include "habana_device_types.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "node.h"
#include "passes.h"

#include "sync_types.h"
#include "types_exception.h"
#include "types.h"

#include <algorithm>
#include <memory>
#include <vector>

bool SignalOutFromGraph::isSignalOutTensor(const TensorPtr& t)
{
    return t->getTensorIsExternal();
}

std::set<HabanaDeviceType> SignalOutFromGraph::getProducers(HabanaGraph& g, const TensorPtr& t)
{
    std::set<HabanaDeviceType> s;
    for (auto& node : g.getRealProducers(t))
    {
        s.insert(g.getNodeUtility().getNodeDeviceType(node));
    }
    return s;
}

NodeSet SignalOutFromGraph::getLastProducers(HabanaGraph& g, const TensorPtr& t)
{
    NodeSet s;
    for (auto& node : g.getRealProducers(t))
    {
        auto existing = std::find_if(s.begin(), s.end(), [&](const NodePtr& existing) {
            return g.getNodeUtility().getNodeDeviceType(existing) == g.getNodeUtility().getNodeDeviceType(node);
        });
        // getRealProducers does not guarantees that we process the nodes in ascending order when
        // dealing with logical nodes
        if (existing != s.end())
        {
            if (existing->get()->getExecutionOrderedIndex() < node->getExecutionOrderedIndex())
            {
                s.erase(existing);
                s.insert(node);
            }
        }
        else
        {
            s.insert(node);
        }
    }

    return s;
}

TensorVector SignalOutFromGraph::getSignalOutTensors(HabanaGraph& g)
{
    TensorVector outTensors;
    for (auto& tensor : g.getTensors())
    {
        if (isSignalOutTensor(tensor))
        {
            outTensors.push_back(tensor);
        }
    }
    std::sort(outTensors.begin(), outTensors.end(), [&](const TensorPtr& a, const TensorPtr& b) {
        return g.getTensorProducer(a)->getExecutionOrderedIndex() < g.getTensorProducer(b)->getExecutionOrderedIndex();
    });
    return outTensors;
}

void SignalOutFromGraph::setDeviceToSignals(HabanaGraph& g)
{
    for (int tensorIndex = 0; tensorIndex < m_outTensorsSorted.size(); tensorIndex++)
    {
        auto& t         = m_outTensorsSorted[tensorIndex];
        auto  producers = getProducers(g, t);
        for (auto& producer : producers)
        {
            auto& vec = m_deviceToSignals[producer];
            vec.push_back(tensorIndex);
        }
    }
    for (auto& item : m_deviceToSignals)
    {
        item.second.push_back(m_outTensorsSorted.size());
    }
}

void SignalOutFromGraph::setTensorToSignals(HabanaGraph& g)
{
    m_tensorsToSignals.resize(m_outTensorsSorted.size());
    for (auto& device : m_supportedDevices)
    {
        auto& deviceSignals = m_deviceToSignals.at(device);
        HB_ASSERT(deviceSignals.size() >= 2, "Expected at least 2 values in this array");
        auto index = 0;
        for (int tensorIndex = 0; tensorIndex < m_outTensorsSorted.size(); tensorIndex++)
        {
            auto& t         = m_outTensorsSorted[tensorIndex];
            auto  producers = getProducers(g, t);
            for (auto& producer : producers)
            {
                if (producer != device) continue;
                HB_ASSERT(index + 1 < deviceSignals.size(), "Out of bounds");
                m_tensorsToSignals[tensorIndex][producer] = deviceSignals[index + 1] - deviceSignals[index];
                index++;
            }
        }
    }
}

void SignalOutFromGraph::commonInit(HabanaGraph& g)
{
    m_outTensorsSorted = getSignalOutTensors(g);

    setDeviceToSignals(g);

    m_supportedDevices = g.getHALReader()->getSupportedDeviceTypes();
    // Removing unused devices
    m_supportedDevices.erase(
        std::remove_if(m_supportedDevices.begin(),
                       m_supportedDevices.end(),
                       [&](HabanaDeviceType device) { return m_deviceToSignals.count(device) == 0; }),
        m_supportedDevices.end());

    setTensorToSignals(g);
}

bool SignalOutFromGraph::executePass(HabanaGraph& g)
{
    try
    {
        init(g);

        for (int i = 0; i < m_outTensorsSorted.size(); i++)
        {
            auto& t         = m_outTensorsSorted[i];
            auto  producers = getLastProducers(g, t);
            HB_ASSERT(!producers.empty(), "Tensor must be created by at least one node");
            auto lastProducer =
                *std::max_element(producers.begin(), producers.end(), [](const NodePtr& lhs, const NodePtr& rhs) {
                    return lhs->getExecutionOrderedIndex() < rhs->getExecutionOrderedIndex();
                });

            addSigOutGroupMonitors(g, producers, t, i);
        }

        return true;
    }
    catch (const SynapseException& e)
    {
        LOG_CRITICAL(GC, "Signaling from graph pass failed: {}", e.what());
        return false;
    }
}
