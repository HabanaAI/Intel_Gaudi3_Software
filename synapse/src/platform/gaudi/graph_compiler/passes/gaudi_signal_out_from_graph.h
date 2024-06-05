#pragma once

#include "code_generator.h"
#include "signal_out_from_graph.h"
#include "sync_object_manager.h"
#include <queue>

class HabanaGraph;

class SigOutSyncGroup
{
public:
    SigOutSyncGroup() {}

    void init(const HabanaGraph&            g,
              SyncObjectManager::SyncId     firstSyncId,
              uint32_t                      groupSize,
              std::vector<HabanaDeviceType> supportedDevices,
              const HalReader&              halReader);

    std::vector<SyncObjectManager::SyncId> getSyncsForDevice(HabanaDeviceType device) const
    {
        return m_habanaDeviceToSyncId.at(device);
    }

    uint32_t getMask() const { return (1 << (m_total - 1)) - 1; }

    SyncObjectManager::SyncId getFirstSyncId() const { return m_firstSyncId; }

private:
    SyncObjectManager::SyncId                                          m_firstSyncId;
    uint32_t                                                           m_total;
    std::map<HabanaDeviceType, std::vector<SyncObjectManager::SyncId>> m_habanaDeviceToSyncId;
};

class GaudiSignalOutFromGraph : public SignalOutFromGraph
{
public:
    GaudiSignalOutFromGraph(CodeGenerator& codeGen) : SignalOutFromGraph(), m_codeGenerator(codeGen) {}
    bool executePass(HabanaGraph& g) override;

private:
    void     init(HabanaGraph& g) override;
    void     setInitialSyncValues(HabanaGraph& g) override;
    unsigned getAvailableMonitor();
    void     addBucketsMonitors(HabanaGraph& g);
    void     generateBucketForEngine(HabanaDeviceType engType, unsigned& leftMonitors, unsigned numSigOutTensors);
    void     allocateBuckets(unsigned availableMonitors);
    void     splitMonitors(std::map<HabanaDeviceType, unsigned>& engineMonitors,
                           std::map<HabanaDeviceType, unsigned>& engineLeftMonitorsNeeded,
                           unsigned                              totalMonitorsNeeded,
                           unsigned                              availableMonitors);

    PatchableMonitor getPatchableMonitor(HabanaGraph& g, const TensorPtr& t, int armValue, int setupValue = 1);
    SyncOrMonitor
    getMonitorsToUpdateTheSigOutGroup(HabanaGraph& g, const NodePtr& producer, uint32_t setupValue, const TensorPtr& t);

    void dumpLegacySFGMonitors(HabanaGraph& g);
    void dumpPatchableMonitor(HabanaGraph& g, const TensorPtr& t, int armValue);
    void dumpSigOutGroupMonitor(HabanaGraph& g, const NodePtr& producer, uint32_t setupValue, const TensorPtr& t);
    void dumpSigOutGroupMonitors(HabanaGraph& g, const NodeSet& producers, const TensorPtr& t, int index);

    std::queue<unsigned>          m_monitorObjects;
    SigOutSyncGroup               m_sigOutGroup;
    CodeGenerator&                m_codeGenerator;

    // For buckets handling
    std::map<HabanaDeviceType, std::vector<unsigned>> m_engineToBuckets;
    std::map<HabanaDeviceType, unsigned>              m_currentBucketIndex;
    std::map<HabanaDeviceType, unsigned>              m_currentIndexInBucket;
    std::map<HabanaDeviceType, unsigned>              m_bucketSetupValue;
    // For multi-produced tensors
    std::set<unsigned>                   m_multiProducedTensorIndex;
    std::map<HabanaDeviceType, unsigned> m_bucketMultiProdTensorDec;
};
