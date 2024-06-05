#include "types.h"
#include "utils.h"
#include "passes.h"
#include "habana_graph.h"

#include "tensors_epoch_allocator.h"
#include "alloc_utils.h"
#include "define_synapse_common.hpp"
#include "habana_global_conf.h"
#include "non_bundle_sram_tensor_comp.h"
#include "tensors_allocator.h"
#include "infra/defs.h"
#include "memory_management/range_epoch_allocator.h"
#include <cstdint>
#include "memory_management/heap_allocator.h"
#include "register_memory_coherence.h"

static bool allocateTensor(const TensorPtr& t, MemoryAllocator& allocator)
{
    if (!allocateTensor(t,
                        false /*inSram*/,
                        true /*allocateRealTensor*/,
                        false /*allowFailure*/,
                        allocator,
                        /*allocTracker*/ nullptr))
    {
        return false;
    }
    LOG_TRACE(GC, "    Allocated virtual address 0x{:x} for tensor {}", t->getDramOffset(), t->getName());
    return true;
}

static bool allocateTensorsUsingDefaultDRAMAllocator(HabanaGraph&          g,
                                                     LivenessAnalysis&     livenessAnalysis,
                                                     size_t&               workspaceSize,
                                                     bool                  dryRun,
                                                     HeapAllocatorWrapper& heapAllocatorWrapper)
{
    bool allocationResult;
    if (dryRun)
    {
        auto                 workspaceAllocatorClone = heapAllocatorWrapper.Clone();
        auto                 workspaceAllocatorPtr = static_cast<HeapAllocatorWrapper*>(workspaceAllocatorClone.get());
        DramTensorsAllocator allocator(&g, livenessAnalysis, *workspaceAllocatorPtr);
        allocationResult = allocator.allocateTensorsMemorySpace();
        workspaceSize    = allocator.getWorkspaceSize();
    }
    else
    {
        DramTensorsAllocator allocator(&g, livenessAnalysis, heapAllocatorWrapper);
        allocationResult = allocator.allocateTensorsMemorySpace();
        workspaceSize    = allocator.getWorkspaceSize();
    }

    return allocationResult;
}

static bool allocateTensorsUsingEpochDRAMAllocator(HabanaGraph&          g,
                                                   LivenessAnalysis&     livenessAnalysis,
                                                   size_t                epochSize,
                                                   size_t                workspaceSize,
                                                   bool                  dryRun,
                                                   HeapAllocatorWrapper& heapAllocatorWrapper)
{
    auto  heapAllocatorWrapperClonePtr = heapAllocatorWrapper.Clone();
    auto& heapAllocatorWrapperClone    = *static_cast<HeapAllocatorWrapper*>(heapAllocatorWrapperClonePtr.get());
    RangeEpochAllocator allocator(g, livenessAnalysis, heapAllocatorWrapperClone);
    bool                allocationResult;
    if (dryRun)
    {
        allocationResult = allocator.allocateMemoryForEpochs(heapAllocatorWrapperClone, epochSize, workspaceSize);
    }
    else
    {
        allocationResult = allocator.allocateMemoryForEpochs(heapAllocatorWrapper, epochSize, workspaceSize);
    }
    return allocationResult ? allocator.allocateTensors() : false;
}

static void clearAllocationStateBeforeRecalculation(HabanaGraph& g, const TensorVector& tensors)
{
    std::for_each(tensors.begin(), tensors.end(), [](auto& tensor) { tensor->unsetDramOffset(); });
}

static TensorVector gatherOutputTensors(HabanaGraph& g)
{
    TensorVector outputTensors;
    if (!GCFG_ENABLE_PERSISTENT_OUTPUT_REUSE.value()) return outputTensors;
    const auto& memoryCoherence = g.getGraphAnnotation().memoryCoherence;
    for (const TensorPtr& t : g.getTensors())
    {
        if (t && t->isPersistent())
        {
            if (g.getNumberOfTensorProducers(t) != 0)
            {
                if (memoryCoherence->previousOverlapsWithOthersInSection(t)) continue;
                outputTensors.push_back(t);
            }
        }
    }
    return outputTensors;
}

static std::pair<uint64_t, uint64_t> calculateSizesForEpochBinarySearch(const TensorVector&   outputTensors,
                                                                        LivenessAnalysis      livenessAnalysis,
                                                                        std::vector<uint64_t> usedMem)
{
    uint64_t outputSizes             = 0;
    uint64_t adjustedMaxLiveCapacity = 0;
    if (usedMem.size() == 0) return {};
    for (const TensorPtr& t : outputTensors)
    {
        uint64_t    tensor_sizes = t->getMinimalSizeInBytes();
        const auto& tensorLife   = livenessAnalysis.getTensorLifeTime(t);
        outputSizes += tensor_sizes;
        for (uint32_t i = 0; i < tensorLife.m_start; ++i)
        {
            usedMem[i] = std::max<int64_t>(1, usedMem[i] - tensor_sizes);
        }
    }
    adjustedMaxLiveCapacity = *std::max_element(begin(usedMem), end(usedMem));
    adjustedMaxLiveCapacity = std::max<int64_t>(1, adjustedMaxLiveCapacity);
    return std::make_pair(outputSizes, adjustedMaxLiveCapacity);
}

static bool allocateRemainingTensors(HabanaGraph& g, const TensorVector& tensors)
{
    std::shared_ptr<HeapAllocator> workSpaceAllocator =
        std::static_pointer_cast<HeapAllocator>(g.getCodeGenerator()->getWorkspaceAllocatorPtr());
    ForceDramAllocLivenessAnalysis livenessAnalysis(&g);
    AllLivenessAnalysis            allLivenessAnalysis(&g);
    auto                           maxLiveCapacity = livenessAnalysis.getGraphMaxCapacity();
    const std::vector<uint64_t>&   usedMem         = livenessAnalysis.getUsedMem();
    TensorVector                   outputTensors   = gatherOutputTensors(g);
    uint64_t                       outputSize, adjustedMaxLiveCapacity;
    std::tie(outputSize, adjustedMaxLiveCapacity) =
        calculateSizesForEpochBinarySearch(outputTensors, allLivenessAnalysis, usedMem);
    HeapAllocatorWrapper heapAllocatorWrapper("HeapAllocatorWrapper",
                                              outputTensors,
                                              allLivenessAnalysis,
                                              workSpaceAllocator,
                                              0);
    auto                 initialWorkspaceSize = maxLiveCapacity;
    bool                 defaultAllocationSuccess =
        allocateTensorsUsingDefaultDRAMAllocator(g, livenessAnalysis, initialWorkspaceSize, true, heapAllocatorWrapper);
    double workspaceSizeFactor = static_cast<double>(initialWorkspaceSize) / adjustedMaxLiveCapacity;
    if (defaultAllocationSuccess && (initialWorkspaceSize <= GCFG_WORKSPACE_MIN_SIZE_LIMIT.value() ||
                                     workspaceSizeFactor <= 1 + GCFG_WORKSPACE_EPOCH_SIZE_PERCISION.value()))
    {
        // actual allocation
        LOG_INFO(TENSORS_ALLOC,
                 "DRAM allocation using default allocator -- Tensors workspace size: {}, max live capacity measured: "
                 "{}, max live capacity with output reuse: {}",
                 initialWorkspaceSize,
                 maxLiveCapacity,
                 adjustedMaxLiveCapacity);
        clearAllocationStateBeforeRecalculation(g, tensors);
        return allocateTensorsUsingDefaultDRAMAllocator(g,
                                                        livenessAnalysis,
                                                        initialWorkspaceSize,
                                                        false,
                                                        heapAllocatorWrapper);
    }
    auto     minWorkspaceSize = adjustedMaxLiveCapacity;
    auto     maxWorkspaceSize = defaultAllocationSuccess ? initialWorkspaceSize : (2 * adjustedMaxLiveCapacity);
    uint64_t currentWorkspaceSize;
    uint64_t currentEpochSize;
    uint64_t bestEpochSize          = maxWorkspaceSize;
    uint64_t bestWorkspaceSize      = maxWorkspaceSize;
    bool     epochAllocationSuccess = false;
    int      learningIteration      = 0;
    LOG_DEBUG(TENSORS_ALLOC,
              "DRAM allocation starting epoch allocator binary search lower bound {} upper bound {} ",
              minWorkspaceSize,
              maxWorkspaceSize);
    while (minWorkspaceSize < maxWorkspaceSize)
    {
        // learning of the workspace size, we use a local allocator in here
        currentWorkspaceSize            = (minWorkspaceSize + maxWorkspaceSize) / 2;
        currentEpochSize                = currentWorkspaceSize + outputSize;
        double workspaceSizeRangeFactor = static_cast<double>(maxWorkspaceSize) / minWorkspaceSize;
        if (workspaceSizeRangeFactor <= 1 + GCFG_WORKSPACE_EPOCH_SIZE_PERCISION.value()) break;
        clearAllocationStateBeforeRecalculation(g, tensors);
        bool allocationSuccess = allocateTensorsUsingEpochDRAMAllocator(g,
                                                                        livenessAnalysis,
                                                                        currentEpochSize,
                                                                        currentWorkspaceSize,
                                                                        true,
                                                                        heapAllocatorWrapper);
        if (allocationSuccess)
        {
            epochAllocationSuccess = true;
            bestEpochSize          = currentEpochSize;
            bestWorkspaceSize      = currentWorkspaceSize;
            maxWorkspaceSize       = (1 - GCFG_WORKSPACE_EPOCH_SIZE_PERCISION.value()) * currentWorkspaceSize;
            LOG_DEBUG(
                TENSORS_ALLOC,
                "DRAM allocation using epoch allocator succeeded for epoch size {} workspace size {} iteration {}",
                currentEpochSize,
                currentWorkspaceSize,
                learningIteration++);
        }
        else
        {
            minWorkspaceSize = (1 + GCFG_WORKSPACE_EPOCH_SIZE_PERCISION.value()) * currentWorkspaceSize;
            LOG_DEBUG(TENSORS_ALLOC,
                      "DRAM allocation using epoch allocator failed for epoch size {} workspace size {} iteration {}",
                      currentEpochSize,
                      currentWorkspaceSize,
                      learningIteration++);
        }
    }
    if (!defaultAllocationSuccess && !epochAllocationSuccess)
    {
        LOG_ERR(TENSORS_ALLOC, "DRAM allocation failed using both epoch and default allocators");
        return false;
    }
    // actual allocation
    clearAllocationStateBeforeRecalculation(g, tensors);
    if (!defaultAllocationSuccess || bestWorkspaceSize < initialWorkspaceSize)
    {
        LOG_INFO(TENSORS_ALLOC,
                 "DRAM allocation using epoch allocator -- Epoch size {} Tensors workspace size: {}, default allocator "
                 "workspace "
                 "size {}, max live capacity measured: {}, max live capacity with output reuse: {}",
                 bestEpochSize,
                 bestWorkspaceSize,
                 defaultAllocationSuccess ? initialWorkspaceSize : UINT64_MAX,
                 maxLiveCapacity,
                 adjustedMaxLiveCapacity);
        return allocateTensorsUsingEpochDRAMAllocator(g,
                                                      livenessAnalysis,
                                                      bestEpochSize,
                                                      bestWorkspaceSize,
                                                      false,
                                                      heapAllocatorWrapper);
    }
    LOG_INFO(TENSORS_ALLOC,
             "DRAM allocation using default allocator -- Tensors workspace size: {}, max live capacity measured: {}, "
             "max live capacity with output reuse: {}",
             initialWorkspaceSize,
             maxLiveCapacity,
             adjustedMaxLiveCapacity);
    return allocateTensorsUsingDefaultDRAMAllocator(g,
                                                    livenessAnalysis,
                                                    initialWorkspaceSize,
                                                    false,
                                                    heapAllocatorWrapper);
}

bool allocateTensors(HabanaGraph& g)
{
    bool result = true;

    LOG_TRACE(GC, "{}: Allocating Tensors:", HLLOG_FUNC);

    const uint64_t sramSizeForNonBundleTensors = NonBundleSramTensorComp::getGraphNonBundleTensorSramSize(&g);
    if (sramSizeForNonBundleTensors > 0)
    {
        // Allocate non-bundle tensors
        LOG_INFO(GC, "Reserved SRAM size for non bundle tensors: {}", sramSizeForNonBundleTensors);
        if (!SramTensorsEpochAllocator(&g, sramSizeForNonBundleTensors, NonBundleSramTensorComp(&g))
                 .allocateTensorsMemorySpace())
        {
            LOG_ERR(GC, "Failed to allocate SRAM tensors");
            return false;
        }
    }

    // Allocate the rest of the SRAM tensors
    if (!SramTensorsEpochAllocator(&g).allocateTensorsMemorySpace())
    {
        LOG_ERR(GC, "Failed to allocate SRAM tensors");
        return false;
    }

    TensorList                                        scratchpadTensorsList;
    TensorVector                                      tensorsPendingAllocation;
    std::unordered_map<synDataType, deviceAddrOffset> unitTensors;

    for (const NodePtr& node : g.getExeSortedNodes())
    {
        for (TensorPtr tensor : node->getOperands())
        {
            if (tensor == nullptr) continue;

            // Do not allocate shape tensors
            if (tensor->isShapeTensor()) continue;

            // Allocate only real tensors (not tensors that aliased to other)
            tensor = tensor->getRealTensor(tensor);
            if (tensor->tensorIsAllocated()) continue;

            if (tensor->isZeroSizedDataTensor())
            {
                tensor->setDramOffset(0);
                continue;
            }

            if (!tensor->isPersistent() && tensor->isHost2DeviceTensor())
            {
                tensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
                // TODO Find a way to assign these tensors to program data section
                // at creation, and unify this branch with one below
                result = allocateTensor(tensor, g.getCodeGenerator()->getAllocatorForProgramData());
                g.getCodeGenerator()->registerProgramDataBlobForDownload(tensor->getData(),
                                                                         tensor->getDramOffset(),
                                                                         tensor->getTotalSizeInBytes());
            }
            else if (tensor->getMemorySectionID() == MEMORY_ID_RESERVED_FOR_WORKSPACE)
            {
                // Workspace tensors are allocated below using DRAM allocator.
                tensor->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;
                if (tensor->isAuxTensor())
                {
                    scratchpadTensorsList.push_back(tensor);
                }
            }
            else if (tensor->getMemorySectionID() == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
            {
                // If we are dealing with the unit matrix tensor (static param) and we already allocated program-data
                // memory for the same unit matrix data-type, reuse its address; otherwise, allocate new memory space.
                if (tensor->isUnitMatrix() && unitTensors.find(tensor->getElementType()) != unitTensors.end())
                {
                    tensor->setDramOffset(unitTensors[tensor->getElementType()]);
                }
                else
                {
                    result = allocateTensor(tensor, g.getCodeGenerator()->getAllocatorForProgramData());
                    g.getCodeGenerator()->registerProgramDataBlobForDownload(tensor->getData(),
                                                                             tensor->getDramOffset(),
                                                                             tensor->getTotalSizeInBytes());
                    // Save address of unit matrix tensor for future reuse
                    if (tensor->isUnitMatrix()) unitTensors[tensor->getElementType()] = tensor->getDramOffset();
                }
            }
            else if (tensor->getMemorySectionID() == getSramMemoryID())
            {
                // SRAM tensors is allocated above using epoch allocator, so nothing to do
            }
            else
            {
                // All unreserved memory IDs are assigned to user persistent tensors. We "allocate" them virtual
                // address which is simply their memory ID shifted to high bits; therefore, they never overlap.
                result = assignVirtualAddressToUserPersistentTensor(tensor);
            }

            if (result == false)
            {
                setMemAllocationError(g);
                return false;
            }
            if (!tensor->tensorIsAllocated())
            {
                tensorsPendingAllocation.push_back(tensor);
            }
        }
    }

    if (!allocateRemainingTensors(g, tensorsPendingAllocation))
    {
        LOG_ERR(GC, "Failed to allocate DRAM tensors");
        return false;
    }

    for (TensorPtr tensor : scratchpadTensorsList)
    {
        result = allocateTensor(tensor, g.getCodeGenerator()->getWorkspaceAllocator());
        if (result == false)
        {
            setMemAllocationError(g);
            return false;
        }
    }

    // Workspace tensors that reuse persistent output memory space are effectively moved from the workspace section
    // to persistent tensor section; thus, we need to update their memory section ID after the allocation.
    if (GCFG_ENABLE_PERSISTENT_OUTPUT_REUSE.value())
    {
        for (const TensorPtr& t : g.getTensors())
        {
            if (t && !t->isPersistent() && t->isDramOffsetSet())
            {
                t->setMemorySectionID(getMemoryIDFromVirtualAddress(t->getDramOffset()));
            }
        }
    }

    return true;
}
