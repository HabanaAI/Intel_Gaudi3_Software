#include "utils.h"
#include "sync_utils.h"
#include "gaudi_graph.h"
#include "monitor_setup_manager.h"
#include "habana_nodes.h"
#include "syn_logging.h"
#include "platform/gaudi/graph_compiler/hal_conventions.h"
#include "platform/gaudi/graph_compiler/command_queue.h"
#include "sync_scheme_manager.h"
#include "sync_conventions.h"

using namespace gaudi;


const std::list<LogicalQueue> SyncSchemeManagerGaudi::s_gaudiLogicalEngines = {
    DEVICE_MME_LOGICAL_QUEUE,
    DEVICE_TPC_LOGICAL_QUEUE,
    DEVICE_DMA_1_1_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_1_2_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_1_3_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_1_4_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_1_5_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_1_6_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_2_1_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_2_2_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_2_3_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_3_1_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_3_2_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_4_1_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_5_1_DRAM_SRAM_LOGICAL_QUEUE,
    DEVICE_DMA_6_1_DRAM_SRAM_LOGICAL_QUEUE
};

///////////////////////
// SyncSchemeManager //
///////////////////////

SyncSchemeManagerGaudi::SyncSchemeManagerGaudi(GaudiGraph *graph):
                    SyncSchemeManager(graph, gaudi::SyncConventions::instance())
{
    m_graph->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs = false;
}

SyncSchemeManagerGaudi::~SyncSchemeManagerGaudi()
{
}

unsigned SyncSchemeManagerGaudi::numEnginesByLogicalEngine(unsigned engineId) const
{
    switch (engineId)
    {
        case DEVICE_MME_LOGICAL_QUEUE:
            return 2;
        case DEVICE_TPC_LOGICAL_QUEUE:
            return m_graph->getNumTpcEng();
        case DEVICE_DMA_1_1_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_1_2_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_1_3_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_1_4_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_1_5_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_1_6_DRAM_SRAM_LOGICAL_QUEUE:
            return 1;
        case DEVICE_DMA_2_1_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_2_2_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_2_3_DRAM_SRAM_LOGICAL_QUEUE:
            return 2;
        case DEVICE_DMA_3_1_DRAM_SRAM_LOGICAL_QUEUE:
        case DEVICE_DMA_3_2_DRAM_SRAM_LOGICAL_QUEUE:
            return 3;
        case DEVICE_DMA_4_1_DRAM_SRAM_LOGICAL_QUEUE:
            return 4;
        case DEVICE_DMA_5_1_DRAM_SRAM_LOGICAL_QUEUE:
            return 5;
        case DEVICE_DMA_6_1_DRAM_SRAM_LOGICAL_QUEUE:
            return 6;
        case DEVICE_COMPLETION_LOGICAL_QUEUE:
            return 1;
        default:
            return 1;
    }
}

unsigned SyncSchemeManagerGaudi::numSignalingEngines(unsigned engineId) const
{
    switch (engineId)
    {
        case DEVICE_MME_LOGICAL_QUEUE:
            return 2 * numEnginesByLogicalEngine(engineId);
        default:
            return numEnginesByLogicalEngine(engineId);
    }
}

SyncSchemeManager::queue_id SyncSchemeManagerGaudi::_getLogicalEngineID(const NodePtr& node, unsigned int engineIdx) const
{
    UNUSED(engineIdx);
    return deviceTypeToLogicalQueue(node, m_graph->getNodeUtility().getNodeDeviceType(node));
}

const std::vector<HabanaDeviceType>& SyncSchemeManagerGaudi::getPlatformDeviceTypes() const
{
    return m_graph->getHALReader()->getSupportedDeviceTypes();
}

bool SyncSchemeManagerGaudi::shouldWaitForLogicalQueue(uint32_t logicalQue) const
{
    return logicalQue != DEVICE_COMPLETION_LOGICAL_QUEUE       &&
           logicalQue != DEVICE_DMA_DEVICE_HOST_LOGICAL_QUEUE  &&
           logicalQue != DEVICE_DMA_HOST_DEVICE_LOGICAL_QUEUE;
}

uint32_t SyncSchemeManagerGaudi::getCompletionLogicalQueue() const
{
    return DEVICE_COMPLETION_LOGICAL_QUEUE;
}

uint32_t SyncSchemeManagerGaudi::getFinalSyncsQueueId() const
{
    return gaudi::getQueueID(DEVICE_COMPLETION_QUEUE, 0);
}

unsigned SyncSchemeManagerGaudi::getOverlapNumEngines() const
{
    return GaudiOverlap::c_engines_nr;
}

unsigned SyncSchemeManagerGaudi::_getSyncIdIncrement(unsigned physEngineIdx, unsigned logicalEngine) const
{
    return (physEngineIdx % m_syncConventions.getGroupSize()) *
           (logicalEngine == DEVICE_MME_LOGICAL_QUEUE ? 2 : 1);
}

// overlap depandant
void SyncSchemeManagerGaudi::_monitorRoiPipelineSyncs(NodePtr                        monitorNode,
                                                      const std::list<NodeROI*>&     rois,
                                                      std::list<MonObject>&          monitors,
                                                      std::pair<unsigned, unsigned>* pForcedDependency)
{
    MonObject mon;
    mon.operation = MONITOR_SO_OP_GREQ;

    OverlapDescriptor desc;

    desc.numSignals        = 0;
    desc.engineID          = _getLogicalEngineID(monitorNode, (*rois.begin())->engineIndex);
    desc.engineIDForDepCtx = desc.engineID;

    for (const auto& roiPtr : rois)
    {
        auto& roi = *roiPtr;
        desc.numSignals += roi.numSignals;
        HB_ASSERT(desc.engineID == _getLogicalEngineID(monitorNode, roi.engineIndex),
                  "Recieved ROIs from different engine ids");
        generateOverlapRois(roi.inputRois, desc.inputRois);
        generateOverlapRois(roi.outputRois, desc.outputRois);
    }
    GaudiOverlap::DependencyCtx dependency;
    uint32_t                    maxEngineSignalId = getMaxSignalIdForEngineToBeDependentOn(monitorNode);

    m_dependencyCalc.addDescriptor(desc, dependency, maxEngineSignalId);
    addForcedDependency(dependency, pForcedDependency);
    addMonitorsByOverlap(s_gaudiLogicalEngines, mon, monitors, dependency);
}
