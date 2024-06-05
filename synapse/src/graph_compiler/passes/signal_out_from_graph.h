#pragma once

#include "habana_device_types.h"
#include "node.h"
#include "types.h"
class HabanaGraph;

class SignalOutFromGraph
{
public:
    SignalOutFromGraph() {}
    virtual bool executePass(HabanaGraph& g);

protected:
    TensorVector           getSignalOutTensors(HabanaGraph& g);
    void                   setDeviceToSignals(HabanaGraph& g);
    void                   setTensorToSignals(HabanaGraph& g);
    bool                   isSignalOutTensor(const TensorPtr& t);
    void                   commonInit(HabanaGraph& g);

    std::set<HabanaDeviceType> getProducers(HabanaGraph& g, const TensorPtr& t);
    NodeSet                    getLastProducers(HabanaGraph& g, const TensorPtr& t);

    TensorVector                                      m_outTensorsSorted;
    std::map<HabanaDeviceType, std::vector<uint32_t>> m_deviceToSignals;
    std::vector<std::map<HabanaDeviceType, uint32_t>> m_tensorsToSignals;
    std::vector<HabanaDeviceType>                     m_supportedDevices;

private:
    virtual void init(HabanaGraph& g) {};
    virtual void setInitialSyncValues(HabanaGraph& g) {};
    virtual void addSigOutGroupMonitors(HabanaGraph& g, const NodeSet& producers, const TensorPtr& t, int index) {}
};
