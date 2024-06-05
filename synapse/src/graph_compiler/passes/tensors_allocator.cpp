#include "tensors_allocator.h"

#include "alloc_utils.h"
#include "allocators_utils.h"
#include "habana_graph.h"
#include "memory_management/heap_allocator.h"
#include "memory_management/memory_allocator.h"


bool TensorsAllocator::_validateTensorComp(pTensor t) const
{
    return validateTensorComp(m_graph, t, m_allocType);
}

bool TensorsAllocator::allocateInDram() const
{
    return (m_graph->getGraphAnnotation().memoryStrategyParams.dramInfo.enableDramAlloc);
}

bool TensorsAllocator::setTensorsAddresses(const TensorVector& tensors)
{
    for (const pTensor& t : tensors)
    {
        if (_validateTensorComp(t))
        {
            if (t->tensorIsAllocated() || t->getRealTensor(t)->inConstSection()) continue;

            LOG_TRACE(TENSORS_ALLOC, "    - {}: Allocating {}", HLLOG_FUNC, t->getName());

            if (!allocateTensorInDram(*m_graph, t))
            {
                LOG_ERR(TENSORS_ALLOC, "{}: Failed to allocate memory in DRAM for tensor {}", HLLOG_FUNC, t->getName());
                return false;
            }

            if (!allocateInDram())
            {
                // Working from SRAM
                if (!allocateTensorInSram(*m_graph, t))
                {
                    LOG_ERR(TENSORS_ALLOC, "{}: Failed to allocate memory in SRAM for tensor {}", HLLOG_FUNC, t->getName());
                    freeTensorFromDram(*m_graph,
                                       t,
                                       /*freeRealTensor*/ true,
                                       /*alloc*/ nullptr,
                                       m_nonPersistentSectionAllocTracker.get());
                    return false;
                }
            }
        }
    }
    return true;
}

template<typename NodeContainer>
bool TensorsAllocator::allocateOutputTensorsOfNodes(const NodeContainer& nodes)
{
    for (const NodePtr& n : nodes)
    {
        LOG_DEBUG(TENSORS_ALLOC, "    {}: Allocating output tensors of node {}", HLLOG_FUNC, n->getNodeName());
        if (!setTensorsAddresses(n->getOutputs()))
        {
            setMemAllocationError(*m_graph);
            LOG_ERR(TENSORS_ALLOC, " {}: Allocating output tensors of setup node {} failed", HLLOG_FUNC, n->getNodeName());
            return false;
        }
    }
    return true;
}

bool TensorsAllocator::allocateTensorsMemorySpace()
{
    if (!allocateOutputTensorsOfNodes(m_graph->getSetupNodes()))
    {
        LOG_ERR(TENSORS_ALLOC, "{}: allocateOutputTensorsOfNodes failed for setup nodes", HLLOG_FUNC);
        return false;
    }

    if (!allocateOutputTensorsOfNodes(m_graph->getExeSortedNodes()))
    {
        LOG_ERR(TENSORS_ALLOC, "{}: allocateOutputTensorsOfNodes failed for execution nodes", HLLOG_FUNC);
        return false;
    }

    return true;
}

DramTensorsAllocator::DramTensorsAllocator(HabanaGraph*          graph,
                                           LivenessAnalysis&     livenessAnalysis,
                                           HeapAllocatorWrapper& allocator)
: TensorsAllocator(graph,
                   ALLOC_TYPE_FORCED_DRAM,
                   std::make_unique<NonPersistentSectionAllocTracker>(*graph, /*sram*/ false)),
  m_minDRAMAllocation(0xFFFFFFFFFFFFFFFF),
  m_maxDRAMAllocation(0),
  m_livenessAnalysis(livenessAnalysis),
  m_allocator(allocator)
{
}

bool DramTensorsAllocator::_allocateTensorInDram(pTensor tensor)
{
    while (true)
    {
        /* Allocation failure due to a lack of space caused by 'dead' tensors */
        bool allowAllocationFailure = !m_toBeFreed.empty();
        Lifetime tensorLife             = m_livenessAnalysis.getTensorLifeTime(tensor);
        bool     allocated              = allocateTensorInDram(*m_graph,
                                              tensor,
                                              /*allacteRealTensor*/ false,
                                              allowAllocationFailure,
                                              &m_allocator,
                                              m_nonPersistentSectionAllocTracker.get(),
                                              tensorLife);

        if (allocated || !allowAllocationFailure)
        {
            return allocated;
        }

        reduceToBeFreed(m_toBeFreed.size() - 1);
    }
}

bool DramTensorsAllocator::allocateTensor(pTensor tensor)
{
    HB_ASSERT(tensor->getTensorAllocatedLocation() == UNDEFINED_LOCATION,
              "Tensor \"{}\" already allocated",
              tensor->getName());
    if (tensor->isZeroSizedDataTensor())
    {
        tensor->setDramOffset(0);
        return true;
    }
    else if (!_allocateTensorInDram(tensor))
    {
        LOG_ERR(TENSORS_ALLOC, "{}: Failed to allocate memory in DRAM for tensor {}", HLLOG_FUNC, tensor->getName());
        return false;
    }

    const auto dramOffset = tensor->getDramOffset();
    uint64_t   sectionidx = getMemoryIDFromVirtualAddress(dramOffset);
    if (sectionidx == MEMORY_ID_RESERVED_FOR_WORKSPACE)
    {
        m_maxDRAMAllocation = std::max(m_maxDRAMAllocation, dramOffset + getWriteSpaceForTensor(tensor));
        m_minDRAMAllocation = std::min(m_minDRAMAllocation, dramOffset);
    }

    LOG_TRACE(TENSORS_ALLOC,
              "{}: tensor {} was successfully allocated in DRAM 0x{:x}",
              HLLOG_FUNC,
              tensor->getName(),
              dramOffset);

    return true;
}

void DramTensorsAllocator::reduceToBeFreed(size_t numTensorsToKeep)
{
    while (m_toBeFreed.size() > numTensorsToKeep)
    {
        pTensor realTensor = m_toBeFreed.front();
        LOG_TRACE(TENSORS_ALLOC, "Free tensor {} (0x{:x}) and remove from m_toBeFreed", realTensor->getName(),
                  realTensor->getDramOffset());
        if (realTensor->isZeroSizedDataTensor())
        {
            m_toBeFreed.pop_front();
            continue;
        }
        freeTensorFromDram(*m_graph,
                           realTensor,
                           /*freeRealTensor*/ true,
                           &m_allocator,
                           m_nonPersistentSectionAllocTracker.get());
        m_toBeFreed.pop_front();
    }
}

bool DramTensorsAllocator::allocateTensorsMemorySpace()
{
    //TODO:improve DRAM reuse mechanism (SW-5951)
    const auto& liveAndDieTensors = m_livenessAnalysis.liveAndDieTensors();

    for (const auto& liveAndDie : liveAndDieTensors)
    {
        for (const auto& die : liveAndDie.m_die)
        {
            m_toBeFreed.push_back(die);
        }
        /* Avoid the deallocation of last k tensors that pushed to - m_toBeFreed
         * Why is that beneficial? As far as tensor not being deallocated its memory
         * will not be reused. This heuristic makes the memory allocation more pipeline friendly
         * as less 'close' tensors will share the same memory
         * */
        reduceToBeFreed(m_keepAllocatedKnob);
        for (const auto& live : liveAndDie.m_live)
        {
            if (!allocateTensor(live))
            {
                return false;
            }
        }
    }
    reduceToBeFreed(0);

    // reserve range of memory used by tensors reuse to make sure later allocations don't overlap released tensors.
    if (m_maxDRAMAllocation > 0) // if Dram allocator allocated memory.
    {
        Range rangeUsedByTensorAllocations;
        // get actual start of unallocated range including passing to prevent holes.
        Settable<deviceAddrOffset> realRangeStart = m_allocator.GetStartOfFreeRangeContaningOffset(m_minDRAMAllocation);
        if (!realRangeStart.is_set()) // minimal DRAM allocation is not part of free range (meaning it wasn't freed)
        {
            LOG_ERR(TENSORS_ALLOC, "{}: minimal allocated DRAM addres not freed as expected", HLLOG_FUNC);
            return false;
        }

        rangeUsedByTensorAllocations.base = realRangeStart.value();
        rangeUsedByTensorAllocations.size = m_maxDRAMAllocation - realRangeStart.value();
        bool allocated                    = m_allocator.allocateReqRange(rangeUsedByTensorAllocations, 0);
        if (!allocated)
        {
            LOG_ERR(TENSORS_ALLOC, "{}: failed to reserve memory range used by tensors to avoid conflicts", HLLOG_FUNC);
            return false;
        }
        m_workspaceSize = rangeUsedByTensorAllocations.size;
    }

    if (m_nonPersistentSectionAllocTracker)
    {
        m_nonPersistentSectionAllocTracker->verifyAllDone();
    }
    return true;
}

// instantiate tempalte functions for used containers
template bool TensorsAllocator::allocateOutputTensorsOfNodes<NodeList>(const NodeList& nodes);
template bool TensorsAllocator::allocateOutputTensorsOfNodes<NodeSet>(const NodeSet& nodes);
template bool TensorsAllocator::allocateOutputTensorsOfNodes<NodeVector>(const NodeVector& nodes);
