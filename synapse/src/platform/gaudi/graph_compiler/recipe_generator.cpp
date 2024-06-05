#include "recipe_generator.h"
#include "hal_conventions.h"
#include "command_queue.h"
#include "platform/gaudi/utils.hpp"
#include "recipe_allocator.h"

namespace gaudi
{
GaudiRecipeGenerator::GaudiRecipeGenerator(const HabanaGraph* g) : RecipeGenerator(g) {}

GaudiRecipeGenerator::~GaudiRecipeGenerator()
{
}

std::string GaudiRecipeGenerator::getEngineStr(unsigned id) const
{
    return getEngineName(static_cast<gaudi_queue_id>(id));
}

void GaudiRecipeGenerator::validateQueue(ConstCommandQueuePtr queue, bool isSetup) const
{
    if (isSetup)
    {
        HB_ASSERT(queue->getCommands(true).size() == 0, "We expect the Activate part to be empty in Gaudi");
    }
}

void GaudiRecipeGenerator::inspectRecipePackets(const void*      buffer,
                                                unsigned         bufferSize,
                                                std::string_view bufferName) const
{
    if (!gaudi::checkForUndefinedOpcode(buffer, bufferSize))
    {
        LOG_ERR(GC, "Invalid OPCODE found in {} blobs buffer", bufferName);
    }
}

void GaudiRecipeGenerator::serializeSyncSchemeDebugInfo(debug_sync_scheme_t* syncSchemeInfo) const
{
    std::vector<NodeSyncInfo> allNodesSyncInfo;

    collectNodeSyncInfo(allNodesSyncInfo);

    syncSchemeInfo->node_sync_info_nr       = allNodesSyncInfo.size();
    syncSchemeInfo->sync_scheme_legacy_mode = true;

    LOG_DEBUG(GC, "Node sync info size: {}", syncSchemeInfo->node_sync_info_nr);

    if (syncSchemeInfo->node_sync_info_nr > 0)
    {
        syncSchemeInfo->node_sync_info_legacy = (node_sync_info_legacy_t*)m_recipeAllocator->allocate(
            syncSchemeInfo->node_sync_info_nr * sizeof(node_sync_info_legacy_t));

        node_sync_info_legacy_t* pFiller = syncSchemeInfo->node_sync_info_legacy;

        for (auto nodeInfo : allNodesSyncInfo)
        {
            pFiller->node_exe_index = nodeInfo.node_exe_index;
            pFiller->engine_type    = nodeInfo.engine_type;
            pFiller->pipe_level     = nodeInfo.pipe_level;
            pFiller->emitted_signal = nodeInfo.emitted_signal;
            pFiller->sob_id         = nodeInfo.sob_id;
            pFiller->num_engines    = nodeInfo.num_engines;
            pFiller++;
        }
    }
    else
    {
        syncSchemeInfo->node_sync_info_legacy = nullptr;
    }
}

void GaudiRecipeGenerator::collectNodeSyncInfo(std::vector<NodeSyncInfo>& allNodesSyncInfo) const
{
    for (pNode node : m_sortedNodes)
    {
        if (node->isLogicalOperation()) continue;

        NodeSyncInfo syncInfo;

        syncInfo.node_exe_index = node->getExecutionOrderedIndex();
        syncInfo.node_name      = node->getNodeName();
        syncInfo.engine_type    = engineName2logical(node->getEngineTypeStr());

        if (node->getNodeAnnotation().syncScheme.size() > 0)
        {
            bool     bSinglePipeLevel = false;
            uint16_t pipe_level = 0;
            for (const auto& pipelineSync : node->getNodeAnnotation().syncScheme[0].pipelineSyncs)
            {
                // In MME signal-once mode - not all pipeline levels are signaling
                if (pipelineSync.sync != nullptr)
                {
                    if (syncInfo.engine_type == Recipe::EngineType::MME && pipelineSync.numSignalsForDbg > 1)
                    {
                        // If MME generated additional signaling - we will set the pipline level to 1 for TD
                        bSinglePipeLevel = true;
                    }
                    pipe_level++;
                }
            }
            syncInfo.pipe_level = pipe_level;

            if (bSinglePipeLevel)
            {
                syncInfo.pipe_level = 1;
            }
            syncInfo.emitted_signal = node->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().syncTotalValue;
            syncInfo.sob_id         = node->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().sync->id;
            syncInfo.num_engines    = node->getNodeAnnotation().syncScheme.size();
            allNodesSyncInfo.push_back(syncInfo);
        }
    }
}

}