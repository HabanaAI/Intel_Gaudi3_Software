/**
 * @brief Allocate tensors using livness analysis live and die vector
 * Use a pre known epoch size with given allocator
 */
#pragma once

#include <memory>
#include "allocators_utils.h"
#include "heap_allocator.h"
#include "liveness_analysis.h"

class RangeEpochAllocator
{
public:
    RangeEpochAllocator(HabanaGraph& graph, LivenessAnalysis& livenessAnalysis, HeapAllocatorWrapper& mainAllocator);
    bool allocateMemoryForEpochs(HeapAllocatorWrapper& mainAllocator, uint64_t maxEpochSize, uint64_t maxWSSize);
    bool allocateTensors();

private:
    // Using different type in case we would like to change it
    using AllocTensorVector = TensorVector;

    void handleEpochMemory();
    void deallocateMemory();
    void allocateMemory();
    void sortCandidatesForAllocation(AllocTensorVector& toAllocate);
    bool attemptCandidateAllocation(uint32_t candidateSize);
    bool adjustCandidatesForAllocation();
    void buildPotentialCandidatesForAllocation();
    void ProgressTensorIterators();

    LivenessAnalysis& m_livenessAnalysis;
    HeapAllocatorWrapper m_allocator;

    AllocTensorVector m_toAllocate;
    AllocTensorVector m_toBeFree;
    uint64_t          m_totalCapcity                 = 0;
    uint32_t          m_currentTimestamp             = 0;
    uint32_t          m_currentLiveTensorInTimestamp = 0;
    uint32_t          m_toAllocateLowerBound         = 0;
};