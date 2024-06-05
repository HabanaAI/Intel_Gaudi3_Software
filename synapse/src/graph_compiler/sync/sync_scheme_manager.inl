#pragma once

#include "sync_scheme_manager.h"

template <typename T>
void SyncSchemeManager::addForcedDependency(T& dependency, std::pair<unsigned, unsigned>* pForcedDependency) const
{
    if (pForcedDependency)
    {
        unsigned engLogicalId = pForcedDependency->first;
        unsigned sigIdx = pForcedDependency->second;
        if (!dependency.valid[engLogicalId])
        {
            dependency.valid[engLogicalId] = true;
            dependency.signalIdx[engLogicalId] = sigIdx;
        }
        else
        {
            // Pick the most restrictive dependency
            dependency.signalIdx[engLogicalId] = std::max(dependency.signalIdx[engLogicalId], sigIdx);
        }
    }
}

template <typename T, typename LQ>
void SyncSchemeManager::addMonitorsByOverlap(const std::list<LQ>&                 logicalEngines,
                                             MonObject&                           mon,
                                             std::list<MonObject>&                monitors,
                                             T&                                   dependency) const
{
    for (LQ logicalEngineId : logicalEngines)
    {
        if (!dependency.valid[logicalEngineId])
        {
            continue;
        }

        Sync sync = _findSyncForValueByLogicalEngine((queue_id)logicalEngineId, dependency.signalIdx[logicalEngineId]);

        size_t i = 0;
        const int groupSize = (int)m_syncConventions.getGroupSize();
        int numEngines = numSignalingEngines(logicalEngineId);

        do {
            HB_ASSERT(i < sync.getNumIds(), "missing sync IDs");
            unsigned maskable = (numEngines - groupSize <= 0)? numEngines : groupSize;
            mon.mask.set(_numEnginesToMask(maskable));
            mon.syncId = sync.getId(i++);
            mon.armValue = sync.getValue();
            monitors.push_back(mon);
            numEngines -= groupSize;
        } while(numEngines > 0);
    }
}

// This used by gaudi and goya2 and is not rellevant (and incorrect) for goya1
template <typename T>
void SyncSchemeManager::removeNodeInternalDependencies(const NodePtr& node,
                                                       unsigned       pipelineLevel,
                                                       T&             ctx) const
{
    if (pipelineLevel == 0)
    {
        // The first can't depend on pipelines before himself on the same engine
        return;
    }
    auto engineId = _getLogicalEngineID(node, 0);
    if (!ctx.valid[engineId])
    {
        return; // no signaling anyway
    }

    Sync sync = _findSyncForValueByLogicalEngine(engineId, ctx.signalIdx[engineId]);
    HB_ASSERT(sync.isValueValid(), "missing valid sync for logical engine {}", engineId);
    SyncId currentSyncId = sync.getId(0); // it's enough to consider the first ID
    currentSyncId *= m_syncConventions.getGroupSize(); // unroll group to the first physical engine
    int16_t value = sync.getValue();
    auto prevValue0 = node->getNodeAnnotation().prevSyncVal[0][0];
    const Settable<unsigned>& prevSyncId0 = node->getNodeAnnotation().prevSyncId[0][0];

    if (prevSyncId0.is_set() && (currentSyncId > prevSyncId0.value() || value > prevValue0))
    {
        ctx.valid[engineId] = false;
    }
}
