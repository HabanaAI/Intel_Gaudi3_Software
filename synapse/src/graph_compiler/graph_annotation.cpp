#include "graph_annotation.h"

#include "infra/defs.h"
#include "passes.h"
#include "utils.h"

void SramRegionsInfo::print(AllocationMode mode) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;
    LOG_DEBUG(GC, "SRAM regions information is as follows:");
    if (persistentIOs)
    {
        LOG_DEBUG(GC, "    - Inputs/Outputs are persistent, their buffer size is {} ({}MB)",
                  IOsSramBufferSize, bToMb(IOsSramBufferSize));
    }
    else
    {
        LOG_DEBUG(GC, "    - Inputs/Outputs are not persistent , total IO size is {} ({}MB)",
                  IOsSramBufferSize, bToMb(IOsSramBufferSize));
    }
    if ((STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM == mode) || (STATIC_TENSORS_PREFETCHED_DRAM_ENABLED == mode))
    {
        LOG_DEBUG(GC, "    - Activations buffer size is {} ({}MB)",
                  activationsSramBufferSize, bToMb(activationsSramBufferSize));
        LOG_DEBUG(GC, "    - Pinning buffer size is {} ({}MB)",
                  pinningSramBufferSize, bToMb(pinningSramBufferSize));
    }
    if (STATIC_TENSORS_PREFETCHED_SEP_BUFF == mode)
    {
        LOG_DEBUG(GC, "    - prefetched data buffer size is {} ({}MB)",
                  prefetchedDataBufferSize, bToMb(prefetchedDataBufferSize));
    }
}

void CapacityInfo::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;
    LOG_DEBUG(GC, "Capacity information is as follows:");
    LOG_DEBUG(GC, "    - Max total SRAM Capacity is {} ({}MB)",
              maxTotalSramCap, bToMb(maxTotalSramCap));
    LOG_DEBUG(GC, "    - Max activations SRAM Capacity is {} ({}MB)",
              maxActivationSramCap, bToMb(maxActivationSramCap));
    LOG_DEBUG(GC, "    - Max static tensor Capacity is {} ({}MB)",
              maxStaticTensorCap, bToMb(maxStaticTensorCap));
    LOG_DEBUG(GC, "    - Used fragmentation is {} ({}MB)",
              fragmentation, bToMb(fragmentation));
}

void DramInfo::print() const
{
    if (!LOG_LEVEL_AT_LEAST_INFO(GC)) return;
    LOG_INFO(GC, "SRAM/DRAM information is as follows:");
    LOG_INFO(GC, "    - Inputs/Outputs are allocated in {}", IOsInDram ? "DRAM" : "SRAM");
    LOG_INFO(GC, "    - Allocating activations and static tensors from DRAM is {}",
             enableDramAlloc ? "enabled" : "disabled");
}

void MemoryStrategyParams::printAllocMode() const
{
    if (!LOG_LEVEL_AT_LEAST_INFO(GC)) return;
    LOG_INFO(GC, " Allocation mode is:");
    switch(getAllocMode())
    {
        case ALL_IN_SRAM_PERSISTENT:
            LOG_INFO(GC, "     ALL_IN_SRAM_PERSISTENT");
            break;
        case STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM:
            LOG_INFO(GC, "     STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM");
            break;
        case STATIC_TENSORS_PREFETCHED_DRAM_ENABLED:
            LOG_INFO(GC, "     STATIC_TENSORS_PREFETCHED_DRAM_ENABLED");
            break;
        case STATIC_TENSORS_PREFETCHED_SEP_BUFF:
            LOG_INFO(GC, "     STATIC_TENSORS_PREFETCHED_SEP_BUFF");
            break;
        case ALL_IN_DRAM:
            LOG_INFO(GC, "     ALL_IN_DRAM");
            break;
        default:
            break;
    }
}

void MemoryStrategyParams::print() const
{
    if (!LOG_LEVEL_AT_LEAST_INFO(GC)) return;
    AllocationMode mode = getAllocMode();
    LOG_INFO(GC, "Memory strategy params are:");
    LOG_INFO(GC, "===========================");
    printAllocMode();
    sramRegionsInfo.print(mode);
    capacityInfo.print();
    dramInfo.print();
}

void MemoryStrategyParams::setAllocModeAllInDRAM()
{
    setAllocMode(ALL_IN_DRAM);
    dramInfo.enableDramAlloc = true;
    dramInfo.IOsInDram = true;
}

void MemoryStrategyParams::setAllocModeAllPersistentInSRAM()
{
    setAllocMode(ALL_IN_SRAM_PERSISTENT);
    dramInfo.enableDramAlloc = false;
    dramInfo.IOsInDram = false;
}

void MemoryStrategyParams::setAllocModePrefetchStaticTensors(unsigned   minActBuffer,
                                                             bool       IOsInDram /* = false */,
                                                             bool       enableDramAlloc /* = false */)
{
    setAllocMode(STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM);
    sramRegionsInfo.minActivationsBufferSize = minActBuffer;
    dramInfo.enableDramAlloc = enableDramAlloc;
    dramInfo.IOsInDram = IOsInDram;
}

void MemoryStrategyParams::setAllocModePrefetchSTEnableDRAM(bool     IOsInDram /* = false */)
{
    setAllocMode(STATIC_TENSORS_PREFETCHED_DRAM_ENABLED);
    dramInfo.enableDramAlloc = true;
    dramInfo.IOsInDram = IOsInDram;
}

void MemoryStrategyParams::setAllocModePrefetchSTtoSepBuff(unsigned STbufferSizeInBytes)
{
    setAllocMode(STATIC_TENSORS_PREFETCHED_SEP_BUFF);
    dramInfo.enableDramAlloc = true;
    sramRegionsInfo.prefetchedDataBufferSize = STbufferSizeInBytes;
}


bool MemoryStrategyParams::allocationModeShouldEnableDram()
{
    switch(getAllocMode())
    {
        case ALL_IN_SRAM_PERSISTENT:
        case STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM:
            return false;
        case STATIC_TENSORS_PREFETCHED_DRAM_ENABLED:
        case ALL_IN_DRAM:
        case STATIC_TENSORS_PREFETCHED_SEP_BUFF:
            return true;
        default:
            HB_ASSERT(false, "Unhandled allocation mode");
            break;
    }
    return true;
}

uint32_t GraphAnnotation::getNumRanges() const
{
    return splitPerRange.size();
}

uint32_t GraphAnnotation::getNumSlicesInRange(unsigned rangeIdx) const
{
    return multiplyElements(splitPerRange[rangeIdx].begin(), splitPerRange[rangeIdx].end());
}

void GraphAnnotation::addAtomicNodesPair(const NodePtr& producer, const NodePtr& consumer)
{
    atomicNodes.emplace_back(producer, consumer);
}

void GraphAnnotation::replaceAtomicNode(const Node* oldNode, const NodePtr& newNode)
{
    std::for_each(atomicNodes.begin(), atomicNodes.end(), [&oldNode, &newNode](auto& atomicPair) {
        if (atomicPair.first.get() == oldNode)
        {
            atomicPair.first = newNode;
        }
        if (atomicPair.second.get() == oldNode)
        {
            atomicPair.second = newNode;
        }
    });
}

void GraphAnnotation::removeAtomicNode(const NodePtr& removedNode)
{
    atomicNodes.erase(std::remove_if(atomicNodes.begin(),
                                     atomicNodes.end(),
                                     [&removedNode](auto& atomicPair) {
                                         if (atomicPair.first == removedNode)
                                         {
                                             return true;
                                         }
                                         if (atomicPair.second == removedNode)
                                         {
                                             return true;
                                         }
                                         return false;
                                     }),
                      atomicNodes.end());
}