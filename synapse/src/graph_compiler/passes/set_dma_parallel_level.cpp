#include "habana_graph.h"
#include "habana_nodes.h"
#include "habana_pass.h"
#include "hal_reader/hal_reader.h"
#include "queue_dispatcher.h"

#include <bitset>
#include <unordered_map>

std::unordered_map<uint32_t, QueueDispatcherParams> getParallelLevelMappings(HabanaGraph& g)
{
    uint32_t dispatcherIndex = 0;
    uint32_t numEngines      = std::bitset<32>(g.getHALReader()->getInternalDmaEnginesMask()).count();
    HB_ASSERT(numEngines >= 1, "Not enough engines to execute dma operations");
    uint32_t memsetEngines = numEngines;

    if (GCFG_MEMSET_PARALLEL_LEVEL.value())
    {
        auto memsetParallelLevel = GCFG_MEMSET_PARALLEL_LEVEL.value();
        numEngines -= memsetParallelLevel;
        memsetEngines = memsetParallelLevel;
        if (numEngines == memsetEngines)
        {
            dispatcherIndex = 1;
        }
        HB_ASSERT(numEngines >=1, "Not enough engines to separate memset and memcpy to multiple engines");
    }
    return std::unordered_map<uint32_t, QueueDispatcherParams> {
        {DMA_OP_TYPE::DMA_OP_COPY, QueueDispatcherParams(numEngines, dispatcherIndex)},
        {DMA_OP_TYPE::DMA_OP_BROADCAST, QueueDispatcherParams(numEngines, dispatcherIndex)},
        {DMA_OP_TYPE::DMA_OP_TRANSPOSE, QueueDispatcherParams(numEngines, dispatcherIndex)},
        {DMA_OP_TYPE::DMA_OP_MEMSET, QueueDispatcherParams(memsetEngines, 0)}};
}

bool setDmaParallelLevel(HabanaGraph& g)
{
    auto parallelLevel = getParallelLevelMappings(g);
    for (const auto& x : parallelLevel)
    {
        LOG_INFO(GC, "Dma Op Type: {}, parallel level: {}, dispatcher index: {}",
                 x.first, x.second.parallelLevel, x.second.index);
    }
    for (const auto& n : g.getExeSortedNodes())
    {
        auto* dmaNode = dynamic_cast<DMANode*>(n.get());
        if (dmaNode == nullptr)
        {
            continue;
        }
        HB_ASSERT(parallelLevel.count(dmaNode->getOpType()),
                  "parallel level isn't set for Dma Op Type: {}", dmaNode->getOpType());
        if (dmaNode->isMemset() && GCFG_GAUDI_MEMSET_HW_WA.value())
        {
            dmaNode->setParallelLevel(parallelLevel[DMA_OP_TYPE::DMA_OP_MEMSET].parallelLevel);
            dmaNode->setDispatcherIndex(parallelLevel[DMA_OP_TYPE::DMA_OP_MEMSET].index);
            continue;
        }
        dmaNode->setParallelLevel(parallelLevel[dmaNode->getOpType()].parallelLevel);
        dmaNode->setDispatcherIndex(parallelLevel[dmaNode->getOpType()].index);
    }
    return true;
}