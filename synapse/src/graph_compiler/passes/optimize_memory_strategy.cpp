#include <math.h>
#include "passes.h"
#include "allocators_utils.h"
#include "liveness_analysis.h"

#include "optimize_memory_strategy.h"

#define ALLOCATOR_FRAGMENTATION             (float)0.1
#define ADDITIONAL_BUFFER_FOR_PREFETCHING   5 * 1024 * 1024 // 5MB to insure static tensors will be in SRAM and not in DRAM

unsigned MemoryStrategyManager::computeIosBufferSizeOfNodes(HabanaGraph& g, const NodeVector& nodes)
{
    unsigned totalSize = 0;

    for (pNode node : nodes)
    {
        for (pTensor tensor : node->getOutputs())
        {
            GET_REAL_TENSOR_IF_NULL_CONTINUE(tensor);

            if (!validateTensorComp(g, tensor, ALLOC_TYPE_PERSISTENT_IO)) continue;

            totalSize += getWriteSpaceForTensor(tensor);
        }
    }
    return totalSize;
}

unsigned MemoryStrategyManager::computeIosBufferSizeOfNodes(HabanaGraph &g, const NodeSet& nodes)
{
    return computeIosBufferSizeOfNodes(g, NodeVector(nodes.begin(), nodes.end()));
}

unsigned MemoryStrategyManager::computeIOsBufferSize(HabanaGraph &g)
{
    return computeIosBufferSizeOfNodes(g, g.getExeSortedNodes()) + computeIosBufferSizeOfNodes(g, g.getSetupNodes());
}


void MemoryStrategyManager::computeCapacityAndIOSize(HabanaGraph &g, MemoryStrategyParams &memParams)
{
    SramRegionsInfo &sramRegionsInfo = memParams.sramRegionsInfo;

    // Compute needed buffer size for persistent IO and activations (no fragmentation included)
    memParams.capacityInfo.PersistentIOsTotalSize = computeIOsBufferSize(g);
    sramRegionsInfo.IOsSramBufferSize = memParams.capacityInfo.PersistentIOsTotalSize;

    memParams.capacityInfo.maxTotalSramCap =
                        LivenessAnalysis(&g, NON_PERSISTENT_TENSORS).getGraphMaxCapacity();
    memParams.capacityInfo.maxActivationSramCap =
                        LivenessAnalysis(&g, NON_PERSISTENT_ACTIVATIONS_TENSORS).getGraphMaxCapacity();
    memParams.capacityInfo.maxStaticTensorCap =
                        LivenessAnalysis(&g, STATIC_TENSORS_COMPATIBILTY).getGraphMaxCapacity();

    sramRegionsInfo.activationsSramBufferSize = memParams.capacityInfo.maxActivationSramCap;

    LOG_TRACE(GC, "computeCapacityAndIOSize: Maximum activation capacity is {}MB",
              bToMb(memParams.capacityInfo.maxActivationSramCap));
    LOG_TRACE(GC, "computeCapacityAndIOSize: Total SRAM cap is {}MB (no fragmentation included)",
              bToMb(memParams.capacityInfo.maxTotalSramCap));
    LOG_TRACE(GC, "computeCapacityAndIOSize: Maximum static tensors capacity {}MB",
              bToMb(memParams.capacityInfo.maxStaticTensorCap));
    LOG_TRACE(GC, "computeCapacityAndIOSize: Total persistent IO size {}MB",
              bToMb(memParams.capacityInfo.PersistentIOsTotalSize));
}

bool MemoryStrategyManager::setMemoryRegions(HabanaGraph &g, MemoryStrategyParams &memParams)
{
    unsigned remainingSram = m_remainingSramSize;
    SramRegionsInfo &sramRegionsInfo = memParams.sramRegionsInfo;

    if (sramRegionsInfo.IOsSramBufferSize > remainingSram)
    {
        LOG_TRACE(GC, "setMemoryRegions: Needed IOs SRAM buffer {}MB is larger than remaining SRAM {}MB,"
                  "allocate IOs in DRAM",
                  bToMb(sramRegionsInfo.IOsSramBufferSize), bToMb(remainingSram));
        sramRegionsInfo.IOsSramBufferSize = 0;
        memParams.dramInfo.IOsInDram = true;
    }
    else
    {
        LOG_TRACE(GC, "setMemoryRegions: Needed IOs SRAM buffer {}MB is smaller than remaining SRAM {}MB,"
                  "allocate IOs in SRAM",
                  bToMb(sramRegionsInfo.IOsSramBufferSize), bToMb(remainingSram));
    }
    sramRegionsInfo.activationsSramBufferSize = std::max<unsigned>(sramRegionsInfo.activationsSramBufferSize,
                                                                   sramRegionsInfo.minActivationsBufferSize);

    remainingSram -= sramRegionsInfo.IOsSramBufferSize;

    if (g.pinningBufferSizeIsSet())
    {
        sramRegionsInfo.pinningSramBufferSize = std::min<uint32_t>(g.getPinningBufferSize(), remainingSram);
        sramRegionsInfo.activationsSramBufferSize = std::max<uint64_t>(remainingSram - sramRegionsInfo.pinningSramBufferSize, 0);
        return false;
    }

    if (sramRegionsInfo.activationsSramBufferSize == remainingSram)
    {
        return false;
    }

    if (g.tensorsPinningDisabled())
    {
        sramRegionsInfo.activationsSramBufferSize = remainingSram;
        sramRegionsInfo.pinningSramBufferSize = 0;
        return true;
    }

    // Set activations and pinned tensors regions.
    memParams.capacityInfo.fragmentation       += ALLOCATOR_FRAGMENTATION;
    sramRegionsInfo.activationsSramBufferSize   =
            std::min<unsigned>(sramRegionsInfo.activationsSramBufferSize * (1 + memParams.capacityInfo.fragmentation),remainingSram);
    sramRegionsInfo.pinningSramBufferSize       = std::max<unsigned>(0, remainingSram - sramRegionsInfo.activationsSramBufferSize);
    LOG_INFO(GC, "setMemoryRegions: Memory strategy params for trial were set as follows:");
    memParams.print();
    return true;
}

bool MemoryStrategyManager::allocAllPersistentInSramTrial(HabanaGraph &g,
                                          MemoryStrategyParams &memParams,
                                          GraphAnnotation &outAnnotations)
{
    memParams.setAllocModeAllPersistentInSRAM();
    return trial(g, memParams, outAnnotations);

}

bool MemoryStrategyManager::allocPrefetchStaticTensorsTrial(HabanaGraph &g,
                                            MemoryStrategyParams &memParams,
                                            GraphAnnotation &outAnnotations)
{
    memParams.setAllocModePrefetchStaticTensors(memParams.sramRegionsInfo.minActivationsBufferSize);
    unsigned round = 0;
    bool trialResult = false;

    while (!trialResult && outAnnotations.errors.memoryAllocationError)
    {
        LOG_DEBUG(GC, "OptimizeMemoryStrategy: round #{}", round++);
        if (!setMemoryRegions(g, memParams))
        {
            break;
        }
        trialResult = trial(g, memParams, outAnnotations);
    }

    if (trialResult)
    {
        // Add an additional buffer to the prefetching region to assure weights are in SRAM.
        unsigned additionalBuffer = g.prefetchingBufferSizeIsSet() ?
                                    g.getPrefetchingBufferSize() : ADDITIONAL_BUFFER_FOR_PREFETCHING;
        additionalBuffer = std::min<unsigned>(memParams.sramRegionsInfo.pinningSramBufferSize, additionalBuffer);
        LOG_TRACE(GC, "{}: add an additional {} to the prefetching region", HLLOG_FUNC, bToMb(additionalBuffer));
        memParams.sramRegionsInfo.pinningSramBufferSize -= additionalBuffer;
        memParams.sramRegionsInfo.activationsSramBufferSize += additionalBuffer;
        // Always enable allocating from DRAM for real graph compilation.
        memParams.dramInfo.enableDramAlloc = true;
        outAnnotations.memoryStrategyParams = memParams;
    }

    return trialResult;

}

/* check for three cases where the memory allocation scheme was already set.
   in these cases set configuration based on value set (no optimization attempted):
   1. allocation mode already set
   2. slicing and/or ranges are used in the graph (e.g. Large images)
   3. graph already has allocateAllInDram set
*/
bool MemoryStrategyManager::checkAndConfigurePresetMemoryStrategy(HabanaGraph& g)
{
    bool PresetValueFound = false;
    MemoryStrategyParams memParams = g.getGraphAnnotation().memoryStrategyParams;

    if (memParams.isAllocModeSet())  // allocation mode already set
    {
        PresetValueFound = true;
        LOG_DEBUG(GC,"OptimizeMemoryStrategy: Allocation mode is already set.");
    }
    // slicing and/or ranges are used in the graph (e.g. Large images)
    else if ( (  g.getGraphAnnotation().splitPerRange.size() > 1) ||            // Number of graph ranges > 1 OR
              ( (g.getGraphAnnotation().splitPerRange.size() == 1) &&           // Number of ranges equals 1 AND
                (std::any_of(g.getGraphAnnotation().splitPerRange[0].begin(),   // Number of slices > 1
                 g.getGraphAnnotation().splitPerRange[0].end(),
                 [](unsigned i){ return i > 1; }) ) ) )
    {
        PresetValueFound = true;
        LOG_DEBUG(GC,
                  "OptimizeMemoryStrategy: Set allocation mode to separate buffers for activations and static tensors."
                  "Size of static tensors buffer is {}MB", bToMb(memParams.sramRegionsInfo.prefetchedDataBufferSize));
        g.getGraphAnnotation().memoryStrategyParams.dramInfo.IOsInDram = true;
        g.getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc = true;
        g.getGraphAnnotation().memoryStrategyParams.setAllocModePrefetchSTtoSepBuff(memParams.sramRegionsInfo.prefetchedDataBufferSize);
    }
    else if (g.allocateAllInDramEnabled()) // graph already has allocateAllInDram set
    {
        PresetValueFound = true;
        LOG_DEBUG(GC,"OptimizeMemoryStrategy: Set allocation mode to All in DRAM.");
        g.getGraphAnnotation().memoryStrategyParams.setAllocModeAllInDRAM();
    }
    return PresetValueFound;
}


// run a series of trials to find the best memory allocation strategy for the current graph
// and configure this allocation strategy in the given graph annotation.
// return true if an optimization was found false otherwise.
bool MemoryStrategyManager::runOptimizationTrials(HabanaGraph& g, MemoryStrategyParams& memParams, GraphAnnotation& ann)
{
    bool                 optimizationFound = false;
    bool                 SramPersistentAllocSuccess = false;
    // optimization check 1: if persistent allocation in SRAM is possible */
    LOG_TRACE(GC,
              "OptimizeMemoryStrategy: try optimizing graph"
              "while fitting all tensors in SRAM and prefetching is disabled");
    SramPersistentAllocSuccess = allocAllPersistentInSramTrial(g, memParams, ann);

    if (g.tensorsPinningDisabled())
    {
        // Pinning is disabled so ALL_IN_SRAM_PERSISTENT mode cannot be used.
        // We still need to run check 1 to set the value of minActivationsBufferSize
        setMemoryAllocError(ann);
    }
    else if (SramPersistentAllocSuccess)
    {
        optimizationFound = true; //  ALL_IN_SRAM_PERSISTENT alloc mode already set by allocAllPersistentInSramTrial
    }

    LOG_TRACE(GC, "OptimizeMemoryStrategy: trial to fit all tensors in SRAM without prefetching {}",
              SramPersistentAllocSuccess ? "Succeeded" : "Failed");
    if (!g.pinningBufferSizeIsSet())
    {
        if (!optimizationFound && isMemoryAllocErrorSet(ann))
        {
            // optimization check 2: If failed to allocate everything in SRAM without using prefetching or pinning, divide SRAM to Regions.
            optimizationFound = allocPrefetchStaticTensorsTrial(g, memParams, ann);
            //  STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM alloc mode already set by allocPrefetchStaticTensorsTrial
        }
    }

    // optimization check 3: Failed to fit all tensors in SRAM, allow working from DRAM
    if ((!optimizationFound && isMemoryAllocErrorSet(ann)) || g.pinningBufferSizeIsSet())
    {
        LOG_TRACE(GC, "OptimizeMemoryStrategy: Failed to fit all tensors in SRAM, allow working from DRAM");
        memParams.setAllocModePrefetchSTEnableDRAM(memParams.dramInfo.IOsInDram);
        ann.memoryStrategyParams = memParams;
        resetMemoryAllocError(ann);
        optimizationFound = true;
    }
    return optimizationFound;
}

bool MemoryStrategyManager::runOptimizeMemoryStrategy(HabanaGraph& g)
{
    bool                 optimizationFound = false;
    GraphAnnotation      ann;
    MemoryStrategyParams memParams = g.getGraphAnnotation().memoryStrategyParams;

    // compute Capacity and IO size
    computeCapacityAndIOSize(g, memParams);

    if (checkAndConfigurePresetMemoryStrategy(g))
    {
        LOG_DEBUG(GC,"OptimizeMemoryStrategy: Preset memory strategy found - skipping optimization");
        optimizationFound = true; // no optimization can be done.
    }
    else
    {
        optimizationFound = runOptimizationTrials(g, memParams, ann);
        g.getGraphAnnotation() = ann;
    }

    if (!optimizationFound)
    {
        LOG_ERR(GC, "Could not optimize memory strategy. No optimization found.");
    }

    LOG_DEBUG(GC, "OptimizeMemoryStrategy: Original graph chosen memory strategy params are:");
    g.getGraphAnnotation().memoryStrategyParams.print();

    return optimizationFound;
}
