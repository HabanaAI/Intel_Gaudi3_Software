#pragma once

#include <exception>
#include "liveness_analysis.h"

#include "memory_management/heap_allocator.h"

#define MAX_NUM_SRAM_IMAGES         1000
#define MAX_ALLOCATED_DEAD_TENSORS  5       // Maximum number of dead tensors to remain allocated to avoid data race caused byreuse

class HabanaGraph;

/* The sliding window allocator allocates activations and model tensors that were not previously allocated.
 * For each graph node, the allocator first allocates its activations then its static tensors.
 * Allocating a static tensor is done by looking back up to MAX_NUM_SRAM_IMAGES nodes to find the best node
 * for prefetching this static tensor.
 * The chosen node must have a free space of size larger or equal to the size of the static tensor and
 * that remains relevant as long as the tensor is alive.
 * */
class SlidingWindowAllocator
{
public:
    SlidingWindowAllocator(HabanaGraph* graph) : m_graph(graph) {};
    SlidingWindowAllocator(HabanaGraph* graph, bool enableWorkingFromDram) : m_graph(graph), m_enableWorkingFromDram(enableWorkingFromDram) {};

    bool allocateTensorsMemorySpace();
    typedef struct
    {
        std::list<Range>                        freeRanges;
        pNode                                   barrierNode = nullptr;
        pNode                                   node; // for debug
        std::map<deviceAddrOffset, Range>       occupiedRanges; // for debug
    } SramImage;
private:
    bool allocateActivationTensors(pNode node);
    bool allocateStaticTensors(pNode node);
    bool allocateStaticTensorInSram(pTensor tensor);
    bool getPrefetchPoint(pTensor tensor, std::list<SramImage>::iterator &prefetchPointIter, Range &chosenRange);
    bool _allocateTensorInSram(pTensor tensor);
    bool updateSramImages(Range chosenRange, const std::list<SramImage>::iterator &prefetchPointIter);
    void freeTensorsMemorySpace(pNode node);
    void reduceToBeFreed(int maxSize);
    void saveSramImage(const pNode &barrierNode, const pNode &node); // Saves new SRAM image

    // Debug helpers
    void printSramImage(SramImage image);
    void printSavedSramImages(std::string_view msg = "");
    void printCurrentSramImage(pNode barrierNode, pNode node);


    HabanaGraph*                                m_graph;
    std::list<SramImage>                        m_sramImages;                    // A list of the last MAX_NUM_SRAM_IMAGES sramImages
    bool                                        m_enableWorkingFromDram = false; // Specifies whether to enable working from DRAM
                                                                                 // in case allocating activation in SRAM didn't succeed
    pLivenessAnalysis                           m_livenessAnalysis;
    std::list<pTensor>                          m_toBeFreed;                     // List of dead tensors that needs to be freed
    TensorSet                                   m_allocatedTensors;              // List of tensors that allocted in the this allocator
};
