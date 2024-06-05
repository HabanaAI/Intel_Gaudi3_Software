#pragma once

#include <list>
#include <map>
#include "flash_attention_nodes_db.h"
#include "tensor_shape.h"
#include "habana_device_types.h"
#include "settable.h"

using AuxiliaryTensorsKey = std::tuple<synDataType,uint32_t,std::string>; //dataType, crc32, sizes
// Keeping the data shared ptr
// to avoid deletion of the data if a node is removed from graph
using TensorAndData    = std::pair<TensorPtr, std::shared_ptr<char>>;
using AuxiliaryTensors = std::map<AuxiliaryTensorsKey, TensorAndData>;

struct SyncOrMonitor;

struct ErrorStatus
{
    bool memoryAllocationError = false;
    bool IOsMemAllocationError = false;
};

/* Allocation Modes are as follows:
 *
 * ALL_IN_SRAM_PERSISTENT:
 * ======================
 * Input, outputs, activations and static tensors will be allocated in SRAM only.
 * Prefetching of static tensors will not be used.
 * There will be reuse in the activations buffer (epoch allocator is used)
 *
 * STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM:
 * ======================================
 * Inputs and outputs will be allocated in SRAM if possible.
 * Activations will be allocated in SRAM only using Sliding window allocator.
 * Static tensors will be: Prefetched to the activations buffer OR pinned OR allocated in DRAM.
 *
 * STATIC_TENSORS_PREFETCHED_DRAM_ENABLED:
 * ======================================
 * Same as STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM mode except enabling allocating activations from DRAM when needed.
 *
 * STATIC_TENSORS_PREFETCHED_SEP_BUFF:
 * ======================================
 * Same as STATIC_TENSORS_PREFETCHED_DRAM_ENABLED mode except static tensors are prefetched into a separate buffer
 * and not into the activations buffer.
 *
 * */
enum AllocationMode
{
    ALL_IN_SRAM_PERSISTENT = 0,
    ALL_IN_DRAM,
    STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM,
    STATIC_TENSORS_PREFETCHED_DRAM_ENABLED,
    STATIC_TENSORS_PREFETCHED_SEP_BUFF
};

struct CapacityInfo
{
    float    fragmentation          = 0;    // % Fragmentation to use for activations.
    unsigned maxTotalSramCap        = 0;    // Maximum total SRAM capacity of activations and static tensors (no fragmentation included)
    unsigned maxActivationSramCap   = 0;    // Maximum total SRAM capacity of activations (no fragmentation included)
    unsigned maxStaticTensorCap     = 0;    // Maximum total capacity of static tensors
    unsigned PersistentIOsTotalSize = 0;    // total size of IO data
    void     print() const;
};

struct SramRegionsInfo
{
    // Regions Sizes
    unsigned IOsSramBufferSize          = 0;    // Buffer size allocated for graph inputs and outputs.
    unsigned activationsSramBufferSize  = 0;    // Buffer size allocated for activations and prefetched static tensors.
    unsigned pinningSramBufferSize      = 0;    // Buffer size allocated for pinned static tensors.
    unsigned prefetchedDataBufferSize   = 0;    // Buffer size allocated for prefetched data.
    unsigned minActivationsBufferSize   = 0;    // Minimal size to be used for activation buffer.

    bool     persistentIOs              = true;    // Specifies whether IOs are persistent or not

    void print(AllocationMode mode) const;
};

struct DramInfo
{
    bool    IOsInDram           = false;    // Indicates whether to allocate Inputs and outputs in DRAM or in SRAM.
    bool    unitMatricesInDram  = false;    // Indicates whether unit matrices are allocated only in SRAM or not.
    bool    enableDramAlloc     = false;    // Indicates whether activations and static tensors can be allocated from DRAM or not.

    void print() const;
};

struct MemoryStrategyParams
{
    Settable<AllocationMode>        allocatinMode;
    SramRegionsInfo                 sramRegionsInfo;
    DramInfo                        dramInfo;
    CapacityInfo                    capacityInfo;

    void            setAllocModeAllInDRAM();
    void            setAllocModeAllPersistentInSRAM();
    void            setAllocModePrefetchStaticTensors(unsigned minActBuffer,
                                                      bool IOsInDram = false,
                                                      bool enableDramAlloc = false);
    void            setAllocModePrefetchSTEnableDRAM(bool IOsInDram = false);
    void            setAllocModePrefetchSTtoSepBuff(unsigned STbufferSizeInBytes);

    bool            isAllocModeSet(){return allocatinMode.is_set();}

    bool            allocationModeShouldEnableDram();
    AllocationMode  getAllocMode() const { return allocatinMode.value(); }

    void print() const;
    void printAllocMode() const;

private:
    void setAllocMode(AllocationMode newAllocMode){allocatinMode.set(newAllocMode);}

};

class TensorCoherenceMapping;

typedef std::shared_ptr<TensorCoherenceMapping> TensorCoherencePtr;
struct GraphAnnotation
{
    GraphAnnotation() {}

    ErrorStatus                  errors;
    std::vector<SizeArray>       splitPerRange; //Per range, how to partition that range to slices
    std::map<unsigned, uint64_t> streamGroupToSize; // memory footprint for each streaming group
    bool                         logicalOperationsHandled = false;

    MemoryStrategyParams         memoryStrategyParams;
    mutable AuxiliaryTensors     cachedAuxiliaryTensors;

    TensorCoherencePtr memoryCoherence;

    uint32_t getNumRanges() const;
    uint32_t getNumSlicesInRange(unsigned rangeIdx) const;

    // Auxiliary container for optimizing tpc fuser clustering of nodes extracted from complex guids.
    // Key - original complex guid node Id. Value - list of clusterable nodes that were extracted from it.
    // All nodes in same list should be clustered together during tpc fuser pass.
    std::unordered_map<uint64_t, NodeList> complexGuidExtractedClusters;

    // This map stores rollover IDs that happen before the very first node of a given engine type (mme/tpc/rot/...)
    std::unordered_map<HabanaDeviceType, std::set<unsigned>> devicePreNodesRolloverIds;

    // Atomic nodes are pairs of marked nodes that are required to be adjacent to each other when compilation is
    // finished. The following struct members contain these nodes.
    using AtomicNodesContainer = std::vector<std::pair<NodePtr, NodePtr>>;
    AtomicNodesContainer atomicNodes;
    void                 addAtomicNodesPair(const NodePtr& producer, const NodePtr& consumer);
    void                 removeAtomicNode(const NodePtr& removedNode);
    void                 replaceAtomicNode(const Node*    oldNode,
                                           const NodePtr& newNode);  // Replace all occurrences of oldNode in the container with newNode

    FlashAttentionDb flashAttentionDb;

    bool                 partialGraph = false;
};
