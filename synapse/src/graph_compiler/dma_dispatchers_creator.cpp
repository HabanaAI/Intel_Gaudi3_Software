#include "dma_dispatchers_creator.h"

#include "habana_graph.h"
#include "habana_nodes.h"
#include "hal_reader/hal_reader.h"

#include <bitset>
#include <cstdint>

static std::set<QueueDispatcherParams> getAllUniqueDmaParallelLevels(const HabanaGraph &g)
{
    unsigned numInternalDmaEngines = std::bitset<32>(g.getHALReader()->getInternalDmaEnginesMask()).count();
    std::set<QueueDispatcherParams> allParallelLevels;

    for (const NodePtr& node : g.getExeSortedNodes())
    {
        HB_ASSERT_PTR(node);
        if (!node->isDma()) continue;
        DMANode* dmaNode = static_cast<DMANode*>(node.get());
        if (dmaNode->getDmaType() != DMA_TYPE_INTERNAL) continue;
        // only unique ones are inserted
        allParallelLevels.emplace(dmaNode->parallelLevel(), dmaNode->dispatcherIndex());
    }

    // add missing parallel levels if not all DMA engines are in use
    unsigned totalUsedDMAEngines = 0;
    for (QueueDispatcherParams params : allParallelLevels)
    {
        totalUsedDMAEngines += params.parallelLevel;
    }
    HB_ASSERT(totalUsedDMAEngines <= numInternalDmaEngines,
              "Cannot use more ({}) DMA engines than {}", totalUsedDMAEngines, numInternalDmaEngines);
    // add dummy dispatcher
    if (totalUsedDMAEngines < numInternalDmaEngines)
    {
        unsigned remainingDMAEngines = numInternalDmaEngines - totalUsedDMAEngines;
        unsigned dispatcherIndex = 0;
        // single dispatcher can handle all remaining engines
        while (allParallelLevels.find(QueueDispatcherParams(remainingDMAEngines, dispatcherIndex)) != allParallelLevels.end())
        {
            ++dispatcherIndex;
        }
        allParallelLevels.emplace(remainingDMAEngines, dispatcherIndex);
    }
    return allParallelLevels;
}

uint8_t DmaDispatcherCreator::getAvailableEnginesMask(HabanaGraph& g, unsigned parallelLevel) const
{
    HB_ASSERT(parallelLevel, "parallel level 0 is invalid");
    return (UINT64_MAX >> (NUM_BITS(uint64_t) - parallelLevel)) &
           g.getAvailableEnginesMask(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL);
}

bool DmaDispatcherCreator::go(HabanaGraph &g)
{
    unsigned                        currentIdx = 0;
    QueueDispatcherMap              dmaDispatchers;
    std::set<QueueDispatcherParams> parallelLevels = getAllUniqueDmaParallelLevels(g);

    if (!isParallelLevelsValid(parallelLevels)) return false;

    for (const QueueDispatcherParams& params : parallelLevels)
    {
        unsigned level = params.parallelLevel;
        HB_ASSERT(level != 0, "parallel level 0 is invalid");

        // Construct the mask for this parallel level
        uint8_t   mask  = 0;
        unsigned  count = 0;
        while ((count < level) && (currentIdx < NUM_BITS(mask)))
        {
            if (g.isEngineDisabled(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL, currentIdx))
            {
                currentIdx++;
            }
            else
            {
                mask |= (1 << currentIdx++);
                count++;
            }
        }

        HB_ASSERT(std::bitset<8>(mask).count() == level, "unsupported parallel level {}", level);

        // make a dispather for this parallel level and index
        dmaDispatchers[params] = makeDmaDispatcher(mask, params.index, &g);
    }

    g.getCodeGenerator()->setDmaDispatchers(std::move(dmaDispatchers));

    return true;
}
