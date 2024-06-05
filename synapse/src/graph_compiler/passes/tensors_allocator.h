#pragma once

#include "liveness_analysis.h"
#include "non_persistent_section_util.h"
#include "node.h"
#include "habana_global_conf.h"
#include <unordered_set>

class HabanaGraph;

/* Allocates memory for graph's tensors */
class TensorsAllocator
{
public:
    TensorsAllocator(HabanaGraph*                                      graph,
                     allocationType                                    allocType,
                     std::unique_ptr<NonPersistentSectionAllocTracker> allocTracker = nullptr)
    : m_graph(graph),
      m_allocType(allocType),
      m_nonPersistentSectionAllocTracker(std::move(allocTracker)),
      m_keepAllocatedKnob(GCFG_TENSORS_KEEP_ALLOCATED.value())
    {
    }

    TensorsAllocator(HabanaGraph* graph) : TensorsAllocator(graph, ALLOC_TYPE_ALL) {}

    virtual ~TensorsAllocator() = default;

    virtual bool allocateTensorsMemorySpace();

protected:
    virtual bool setTensorsAddresses(const TensorVector& tensors);
    virtual bool allocateInDram() const;
    template<typename NodeContainer = NodeVector>
    bool allocateOutputTensorsOfNodes(const NodeContainer& nodes);
    bool _validateTensorComp(pTensor t) const;

    std::map<unsigned, Settable<deviceAddrOffset>>    m_streamGroupToAddr;
    HabanaGraph*                                      m_graph;
    allocationType                                    m_allocType;
    std::unique_ptr<NonPersistentSectionAllocTracker> m_nonPersistentSectionAllocTracker;
    const unsigned                                    m_keepAllocatedKnob;
};

class HeapAllocatorWrapper;
class DramTensorsAllocator : public TensorsAllocator
{
public:
    DramTensorsAllocator(HabanaGraph* graph, LivenessAnalysis& livenessAnalysis, HeapAllocatorWrapper& allocator);
    virtual bool allocateTensorsMemorySpace() override;
    bool         allocateTensor(pTensor tensor);
    size_t       getWorkspaceSize() { return m_workspaceSize; }

protected:
    virtual bool allocateInDram() const override { return true; }

private:
    void reduceToBeFreed(size_t numTensorsToKeep);
    bool _allocateTensorInDram(pTensor tensor);

    deviceAddrOffset      m_minDRAMAllocation;
    deviceAddrOffset      m_maxDRAMAllocation;
    LivenessAnalysis&     m_livenessAnalysis;
    std::list<pTensor>    m_toBeFreed;  // List of dead tensors that needs to be freed
    HeapAllocatorWrapper& m_allocator;
    size_t                m_workspaceSize = 0;
};
