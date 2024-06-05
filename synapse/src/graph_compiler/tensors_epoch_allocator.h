#pragma once

#include "liveness_analysis.h"
#include "memory_management/heap_allocator.h"
#include "non_persistent_section_util.h"

#define FRIEND_TEST(test_case_name, test_name)\
    friend class test_case_name##_##test_name##_Test

class HabanaGraph;

/**
 * Allocate memory in epochs, assigning addresses to all tensors in
 * an epoch once SRAM capacity is exceeded for that epoch
 * Tensors that are alive at the end of an epoch are allocated
 * first assuming they were not carried over from the previous epoch
 */

class TensorsEpochAllocator
{
public:
    TensorsEpochAllocator(HabanaGraph*         graph,
                          TensorsCompatibility tensorsCompatibility,
                          uint64_t             maxEpochSize,
                          const std::string&   name,
                          std::unique_ptr<NonPersistentSectionAllocTracker> allocTracker = nullptr);
    virtual ~TensorsEpochAllocator();

    bool allocateTensorsMemorySpace();

protected:
    virtual bool shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor) = 0;
    virtual void allocateTensorInDram(const pTensor& tensor);

    void handleNodeTensors(const pNode& node);
    bool handleFallback(bool allowDramAllocation);
    // Returns <status, writeSpaceForTensor> - the return status will be false when the tensor's write
    // space is bigger than available SRAM size.
    std::pair<bool, uint64_t> handleNodeTensor(const pNode& node, pTensor tensor, bool allowFallback);
    void                      handleTensorDoesNotFitInSram(const pTensor& tensor, uint64_t writeSpaceForTensor);
    void handleEpochMemory(const pNode& node = nullptr);
    void allocateTensorsInSram();
    void pushBackBornAndDiedTensor(const pTensor& tensor);
    bool                      shouldHandleFallbackForNode(const pNode& node);

    class Epoch
    {
      public:
        Epoch(uint32_t slice = 0);

        void addTensor(const pNode& node,
                       const pTensor& tensor,
                       uint64_t writeSpaceForTensor);


        uint64_t            m_currentSlice  = 0;
        uint64_t            m_epochSize     = 0;
        TensorList          m_bornAtThisEpoch;
    };
    using TensorAndNode = std::pair<pTensor, pNode>;

    HabanaGraph*                        m_graph;
    TensorsCompatibility                m_tensorsCompatibility;
    pLivenessAnalysis                   m_livenessAnalysis;
    std::unique_ptr<HeapAllocator>      m_sramAllocator;

    std::vector<Epoch>                  m_epochs;
    Epoch                               m_currentEpoch;
    TensorList                          m_handledTensors;
    std::list<TensorAndNode>            m_lastBundleTensors;
    uint64_t                            m_currBundleIndex;
    std::string                         m_name;
    pNode                               m_lastNode = nullptr;
    NodeList                            m_epochLastNode;

    std::unique_ptr<NonPersistentSectionAllocTracker> m_nonPersistentSectionAllocTracker;

    FRIEND_TEST (AllocatorTest, epoch_allocator_test);
};

class ActivationTensorsEpochAllocator : public TensorsEpochAllocator
{
public:
    ActivationTensorsEpochAllocator(HabanaGraph* graph, uint64_t maxEpochSize = UINT64_MAX)
    : TensorsEpochAllocator(graph, ACTIVATIONS_TENSORS_COMPATIBILTY, maxEpochSize, "ActivationTensorsEpochAllocator")
    {
    }

private:
    bool shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor) override;
};

class StaticTensorsEpochAllocator : public TensorsEpochAllocator
{
public:
    StaticTensorsEpochAllocator(HabanaGraph* graph, uint64_t maxEpochSize = UINT64_MAX)
    : TensorsEpochAllocator(graph, STATIC_TENSORS_COMPATIBILTY, maxEpochSize, "StaticTensorsEpochAllocator")
    {
    }

private:
    bool shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor) override;
};

/**
 * Allocate tensors which their location (TensorAnnotation->memory->location) indicate on SRAM
 */
class SramTensorsEpochAllocator : public TensorsEpochAllocator
{
public:
    SramTensorsEpochAllocator(HabanaGraph*         graph,
                              uint64_t             maxEpochSize = UINT64_MAX,
                              TensorsCompatibility tComp        = SRAM_TENSORS_COMPATIBILTY)
    : TensorsEpochAllocator(graph,
                            tComp,
                            maxEpochSize,
                            "SramTensorsEpochAllocator",
                            std::make_unique<NonPersistentSectionAllocTracker>(*graph, /*sram*/ true)) {};

private:
    bool shouldCreateNewEpoch(const pNode& node, uint64_t writeSpaceForTensor) override;
    void allocateTensorInDram(const pTensor& tensor) override;
};
