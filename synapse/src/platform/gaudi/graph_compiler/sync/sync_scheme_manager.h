#pragma once

#include <memory>
#include <list>
#include <map>
#include <set>

#include "gaudi_types.h"
#include "node_annotation.h"
#include "sync_object_manager.h"
#include "graph_compiler/sync/sync_scheme_manager.h"

class GaudiGraph;

class SyncSchemeManagerGaudi: public SyncSchemeManager
{
public:
    explicit SyncSchemeManagerGaudi(GaudiGraph* graph);

    ~SyncSchemeManagerGaudi() override;

protected:
    void _monitorRoiPipelineSyncs(NodePtr                        monitorNode,
                                  const std::list<NodeROI*>&     roi,
                                  std::list<MonObject>&          monitors,
                                  std::pair<unsigned, unsigned>* pForcedDependency = nullptr) override;

    logical_engine_id  _getLogicalEngineID(const NodePtr& node, unsigned int engineIdx) const override;
    unsigned           numEnginesByLogicalEngine(unsigned engineId) const override;
    unsigned           numSignalingEngines(unsigned engineId) const override;
    virtual unsigned   _getSyncIdIncrement(unsigned physEngineIdx, unsigned logicalEngine) const override;
    virtual bool       shouldWaitForLogicalQueue(uint32_t logicalQue) const override;
    virtual uint32_t   getCompletionLogicalQueue() const override;
    virtual uint32_t   getFinalSyncsQueueId() const override;
    virtual unsigned   getOverlapNumEngines() const override;

    virtual const std::vector<HabanaDeviceType>& getPlatformDeviceTypes() const override;

private:
    GaudiOverlap                                m_dependencyCalc;
    static const std::list<gaudi::LogicalQueue> s_gaudiLogicalEngines;
};
