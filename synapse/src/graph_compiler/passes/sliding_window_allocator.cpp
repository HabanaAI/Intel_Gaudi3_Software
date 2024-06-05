#include "sliding_window_allocator.h"
#include <deque>
#include "habana_graph.h"
#include "allocators_utils.h"


static uint64_t getPadToAlignment(uint64_t base, uint64_t alignment)
{
    return (alignment - (base % alignment));
}

bool SlidingWindowAllocator::_allocateTensorInSram(pTensor tensor)
{
    while (true)
    {
        bool allocated = allocateTensorInSram(*m_graph, tensor, false, (m_enableWorkingFromDram || !m_toBeFreed.empty()));

        if (allocated || m_toBeFreed.empty())
        {
            return allocated;
        }

        reduceToBeFreed(m_toBeFreed.size() - 1);
    }
}

bool SlidingWindowAllocator::allocateActivationTensors(pNode node)
{
    // Go over all activation tensors of the given node
    for (pTensor tensor : node->getOutputs())
    {
        GET_REAL_TENSOR_IF_NULL_CONTINUE(tensor);

        if (tensor->inConstSection()) continue; // Const tensors are allocated by the user
        if (tensor->tensorIsAllocated()) continue;
        if (tensor->isShapeTensor()) continue; // shape tensors have no allocated data.
        if (tensor->isPersistent()) continue; // Persistent tensors are allocated by the user

        // Tensor is an activation that was not allocated before, try allocating it in SRAM
        if (_allocateTensorInSram(tensor))
        {
             m_allocatedTensors.insert(tensor);

             LOG_TRACE(GC,
                       "{}: Activation tensor {} was successfully allocated in SRAM 0x{:x}.",
                       HLLOG_FUNC,
                       tensor->getName(),
                       tensor->getSramOffset());

             continue;
        }

        if (!m_enableWorkingFromDram)
        {
            // Allocating activations in DRAM is not enabled, return an error
            LOG_ERR(GC,
                    "{}: Failed to allocate tensor {} in SRAM while allocating from DRAM is disabled",
                    HLLOG_FUNC,
                    tensor->getName());

            m_graph->getGraphAnnotation().errors.memoryAllocationError = true;

            return false;
        }
        else // instead of allocating the tensor in DRAM here, mark it and deffer allocator for later.
        {
            if((!isNonPersistentActivationTensor(m_graph, tensor)) && !isAllocInDramForced(tensor))
            {
                if (!allocateTensorInDram(*m_graph, tensor, false))
                {
                    LOG_ERR(GC, "{}: Failed to allocate memory in DRAM for tensor {}", HLLOG_FUNC, tensor->getName());

                    return false;
                }
                LOG_TRACE(GC,
                          "{}: Activation tensor {} was successfully allocated in DRAM 0x{:x}.",
                          HLLOG_FUNC,
                          tensor->getName(),
                          tensor->getDramOffset());
            }
            else
            {
                tensor->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;
                LOG_TRACE(GC, "- {}: Mark Tensor {} for allocation in DRAM", HLLOG_FUNC, tensor->getName());
            }
        }
    }

    return true;
}

bool SlidingWindowAllocator::getPrefetchPoint(pTensor tensor,
                                              std::list<SramImage>::iterator &prefetchPointIter,
                                              Range &chosenRange)
{

    uint64_t alignment = tensor->getTensorAnnotation().memory.alignment;
    uint64_t allocSize = getWriteSpaceForTensor(tensor);

    // Fill candRanges with the last sramImage ranges.
    std::list<Range> candRanges;
    std::list<Range> interRanges;

    for (Range range : m_sramImages.back().freeRanges)
    {
        if (range.size < (allocSize + getPadToAlignment(range.base, alignment))) continue;
        candRanges.push_back(range);
    }

    if (candRanges.empty())
    {
        LOG_TRACE(GC,
                  "{}: there is no enough free SRAM space to allocate tensor {} of size {}.",
                  HLLOG_FUNC,
                  tensor->getName(),
                  allocSize);
        return false;
    }

    // Reverse-Iterate the list of images to find the first possible point to start prefetching the tensor
    // This point must have contiguous SRAM space >= needed space and it should be free until it's used
    // (up until the last sramImage).

    prefetchPointIter = m_sramImages.end();
    while (prefetchPointIter != m_sramImages.begin())
    {
        prefetchPointIter--;
        // go over all free ranges at this point, add to candList ranges that are large enough and will still
        // be relevant until the end of the list (this is done by computing the intersection with the ranges in the list)
        for (Range range : prefetchPointIter->freeRanges)
        {
            if (range.size < allocSize) continue; // Range isn't large enough, continue to the next one.
            for (Range candRange : candRanges)
            {
                Range interRange;
                if (rangeIntersectsWith(range, candRange, interRange))
                {
                    // align interRange
                    if (interRange.size < (allocSize + getPadToAlignment(interRange.base, alignment)))
                    {
                        continue;
                    }
                    interRanges.push_back(interRange);
                }
            }
        }
        if (interRanges.empty())
        {
            prefetchPointIter++;
            break;
        }
        candRanges.clear();
        candRanges.splice(candRanges.end(), interRanges);
    }

    chosenRange.base = candRanges.front().base;
    chosenRange.size = allocSize +  getPadToAlignment(chosenRange.base, alignment);

    LOG_TRACE(GC,
              "{} Tensor {} will be prefetched {}, chosen range base is 0x{:x}, size is {}",
              HLLOG_FUNC,
              tensor->getName(),
              prefetchPointIter->node == nullptr ? "at the beginning of the graph"
                                                 : "at node {}" + prefetchPointIter->node->getNodeName(),
              chosenRange.base,
              chosenRange.size);
    return true;
}

bool SlidingWindowAllocator::updateSramImages(Range chosenRange, const std::list<SramImage>::iterator &prefetchPointIter)
{
    for (std::list<SramImage>::iterator imgIter=prefetchPointIter; imgIter != m_sramImages.end(); imgIter++)
    {
        bool allocated = false;
        for (std::list<Range>::iterator rangesIter = imgIter->freeRanges.begin();
                rangesIter!=imgIter->freeRanges.end(); ++rangesIter)
        {
            Range& range = *rangesIter;
            if (rangeContainedIn(chosenRange, range))
            {
                allocateRange(chosenRange, rangesIter, imgIter->freeRanges);
                allocated = true;
                break;
            }
        }
        if (!allocated)
        {
            LOG_ERR(GC, "updateSramImages: range (base = 0x{:x} size = {}) was not found in free ranges of node {}",
                    chosenRange.base, chosenRange.size, imgIter->node->getNodeName());
            return false;
        }
    }
    return true;
}

bool SlidingWindowAllocator::allocateStaticTensorInSram(pTensor tensor)
{
    Range chosenRange;
    std::list<SramImage>::iterator prefetchPointIter;
    HB_ASSERT((!m_sramImages.empty()), "No saved SRAM images.");

    bool foundPrefetchPoint = false;
    // Step 1: iterate the list of SramImages while the intersection of the free ranges has a contiguous space of
    // size larger than needed size - to get the first possible point to start the prefetching.
    foundPrefetchPoint = getPrefetchPoint(tensor, prefetchPointIter, chosenRange);
    while (!foundPrefetchPoint && !m_toBeFreed.empty())
    {
        // Didnt find a prefetch point, free a dead tensor and retry.
        reduceToBeFreed(m_toBeFreed.size() - 1);
        SramImage oldImg = m_sramImages.back();
        m_sramImages.pop_back();
        saveSramImage(oldImg.barrierNode, oldImg.node);

        foundPrefetchPoint = getPrefetchPoint(tensor, prefetchPointIter, chosenRange);
    }

    if (foundPrefetchPoint) // such a point was found, static tensor can be allocated in SRAM.
    {
        // Step 2: Set tensor address
        unsigned alignment = tensor->getTensorAnnotation().memory.alignment;
        unsigned pad = getPadToAlignment(chosenRange.base, alignment);
        tensor->setSramOffset(chosenRange.base + pad);
        LOG_TRACE(GC,
                  "{}: Set SRAM offset of static tensor {} to 0x{:x}",
                  HLLOG_FUNC,
                  tensor->getName(),
                  chosenRange.base + pad);

        // Step 3: add barrier and mark tensor as a prefetched tensor
        tensor->getTensorAnnotation().memorySpaceInfo.prefetchInfo.prefetch = true;
        if ((*prefetchPointIter).barrierNode != nullptr)
        {
            tensor->getTensorAnnotation().memorySpaceInfo.barriers.push_back((*prefetchPointIter).barrierNode);
        }

        // Step 4: Update SramImages starting from prefetchPoint up to the last image by allocating the chosenRange in all of them.
        if (!updateSramImages(chosenRange, prefetchPointIter)) return false;

        m_allocatedTensors.insert(tensor);

        // Step 5: Update allocator
        MemoryAllocator& sramAlloc  = m_graph->getCodeGenerator()->getSramAllocator();
        HB_ASSERT((sramAlloc.getMemAllocatorType() == MEMORY_HEAP_ALLOCATOR), "Allocator type not supported.");
        return ((HeapAllocator&)sramAlloc).allocateReqRange(chosenRange, pad);
    }
    return true;
}


bool SlidingWindowAllocator::allocateStaticTensors(pNode node)
{
    // Go over all static tensors of the given node
    for (pTensor tensor : node->getInputs())
    {
         GET_REAL_TENSOR_IF_NULL_CONTINUE(tensor);

         if (!tensor->isStaticParam()) continue; // Activations are allocated separately
         if (tensor->tensorIsAllocated()) continue;
         if (tensor->isShapeTensor()) continue; // shape tensors require no memory space

         if (!tensor->getTensorAnnotation().sparseAccess)
         {
             // Tensor is static tensor that was not allocated before, try allocating it in SRAM
             if (!allocateStaticTensorInSram(tensor))
             {
                 LOG_ERR(GC, "{}: Failed to allocate memory in SRAM for tensor {}", HLLOG_FUNC, tensor->getName());
                 return false;
             }
         }
        else
        {
            LOG_TRACE(GC,
                      "{}: tensor {} is marked as sparse accessed tensor and will not be prefetched",
                      HLLOG_FUNC,
                      tensor->getName());
        }

        if (tensor->inConstSection()) continue;  // const tensors are allocated by the user

        // Allocate in DRAM
        if (!allocateTensorInDram(*m_graph, tensor, false, false, &m_graph->getCodeGenerator()->getAllocatorForProgramData()))
        {
            LOG_ERR(GC, "{}: Failed to allocate memory in DRAM for tensor {}", HLLOG_FUNC, tensor->getName());
            return false;
        }

        LOG_TRACE(GC,
                  "{}: Static tensor {} was successfully allocated in DRAM 0x{:x}",
                  HLLOG_FUNC,
                  tensor->getName(),
                  tensor->getDramOffset());
    }
    return true;
}

void SlidingWindowAllocator::saveSramImage(const pNode &barrierNode, const pNode &node)
{
    MemoryAllocator& sramAlloc  = m_graph->getCodeGenerator()->getSramAllocator();
    SramImage img;
    HB_ASSERT((sramAlloc.getMemAllocatorType() == MEMORY_HEAP_ALLOCATOR), "Allocator type not supported.");

    img.freeRanges      = ((HeapAllocator&)sramAlloc).getFreeRanges(); // Current free ranges
    img.occupiedRanges  = ((HeapAllocator&)sramAlloc).getOccupiedRanges(); // Current occupied ranges
    img.barrierNode     = barrierNode;
    img.node            = node;

    if (m_sramImages.size() == MAX_NUM_SRAM_IMAGES)
    {
        // List is full, remove the oldest SRAM image
        m_sramImages.front().freeRanges.clear();
        m_sramImages.front().occupiedRanges.clear();
        m_sramImages.pop_front();
    }
    // [CID: 36962] False positive - coverity ignores std::map default c'tor
    m_sramImages.push_back(img);
}

void SlidingWindowAllocator::reduceToBeFreed(int maxSize)
{
    while (m_toBeFreed.size() > maxSize)
    {
        pTensor realTensor = m_toBeFreed.front();
        LOG_TRACE(GC,"Free tensor {} (0x{:x}) and remove from m_toBeFreed", realTensor->getName(), realTensor->getSramOffset());
        freeTensorFromSram(*m_graph,
                           realTensor,
                           /*freeRealTensor*/ true,
                           /*alloc*/ nullptr,
                           /*rollback*/ false,
                           /*allocTracker*/ nullptr);
        m_toBeFreed.pop_front();
    }
}

void SlidingWindowAllocator::freeTensorsMemorySpace(pNode node)
{
    for (pTensor t : node->getInputs())
    {
        GET_REAL_TENSOR_IF_NULL_CONTINUE(t);

        if (t->isShapeTensor()) continue;
        // tensor not found in m_allocatedTensors
        if(m_allocatedTensors.count(t) == 0) continue;

        // If tensor was not allocated by this allocator, continue
        if (!m_livenessAnalysis->isRealTensorAliveAfterNode(node, t))
        {
            if (std::find(m_toBeFreed.begin(), m_toBeFreed.end(), t) == m_toBeFreed.end())
            {
                // Free tensor
                LOG_TRACE(GC, "added tensor {} (0x{:x}) to m_toBeFreed list", t->getName(), t->getSramOffset());
                m_toBeFreed.push_back(t);
            }
        }
    }
    reduceToBeFreed(MAX_ALLOCATED_DEAD_TENSORS);
}

bool SlidingWindowAllocator::allocateTensorsMemorySpace()
{
    m_livenessAnalysis.reset(new AllLivenessAnalysis(m_graph));

    pNode prevNode = nullptr;

    for (pNode node : m_graph->getExeSortedNodes())
    {
        if (node->isLogicalOperation()) continue;

        if (!allocateActivationTensors(node))
        {
            LOG_ERR(GC, "{}: Failed to allocate activations for node {}", HLLOG_FUNC, node->getNodeName());
            return false;
        }
        LOG_TRACE(GC, "{}: Allocate activations of node {}", HLLOG_FUNC, node->getNodeName());

        saveSramImage(prevNode, node);

        if (!allocateStaticTensors(node))
        {
            LOG_ERR(GC, "{}: Failed to allocate static tensors for node {}", HLLOG_FUNC, node->getNodeName());
            return false;
        }
        LOG_TRACE(GC, "{}: Allocate static tensors of node {}", HLLOG_FUNC, node->getNodeName());

        if (!node->isDma())
        {
            prevNode = node;
        }

        LOG_TRACE(GC, "{}: Free dead tensors", HLLOG_FUNC);
        freeTensorsMemorySpace(node);
    }
    // Free all remaining tensors.
    reduceToBeFreed(0);

    return true;
}


/* ***************************** DEBUG HELPERS ***************************** */

#define LOG_SRAM_IMG LOG_TRACE
#define ONE_INDENT_STR "   "
unsigned indent_level = 0;

void incIndent() {++indent_level;}
void decIndent()
{
    HB_ASSERT(indent_level != 0, "indent_level can't be 0");
    --indent_level;
}
void updIndent(int indent)
{
    HB_ASSERT(indent != 0, "indent_level can't be 0");
    indent_level = indent;
}

static std::string getIndent()
{
    std::string indent = "";
    for (unsigned i = 0; i < indent_level; i++)
    {
        indent.append(ONE_INDENT_STR);
    }
    return indent;
}

#define LOG_ALLOC_SRAM(msg, ...)\
        LOG_DEBUG(GC, getIndent() + msg, ##__VA_ARGS__);

void SlidingWindowAllocator::printSramImage(SramImage img)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC)) return;

    LOG_SRAM_IMG(GC, "{}Sram image for node {}, barrierNode = {}:", getIndent(), img.node->getNodeName(),
                 img.barrierNode == nullptr ? "NULL" : img.barrierNode->getNodeName());
    incIndent();
    LOG_SRAM_IMG(GC, "Free Ranges:");
    for (auto rangesIter  = img.freeRanges.begin();
              rangesIter != img.freeRanges.end();
              ++rangesIter)
    {
        Range range = *rangesIter;
        LOG_SRAM_IMG(GC, "{} [0x{:x}, 0x{:x}] ([{}, {}]) size = {}KB ({}MB)",
                     getIndent(),
                     range.base, range.base + range.size,
                     range.base, range.base + range.size,
                     bToKb(range.size), bToMb(range.size));
    }

    LOG_SRAM_IMG(GC, "Occupied Ranges:");
    for (auto it = img.occupiedRanges.begin(); it != img.occupiedRanges.end(); it++)
    {
        Range range = it->second;
        LOG_SRAM_IMG(GC, "{} [0x{:x}, 0x{:x}] ([{}, {}]) size = {}KB ({}MB). Offset in HEAP alloc = 0x{:x} ({})",
                     getIndent(),
                     range.base, range.base + range.size,
                     range.base, range.base + range.size,
                     bToKb(range.size), bToMb(range.size),
                     it->first, it->first);
    }
    decIndent();
}

void SlidingWindowAllocator::printSavedSramImages(std::string_view msg)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC)) return;

    unsigned count = 0;
    if (m_sramImages.empty())
    {
        LOG_SRAM_IMG(GC, "{}There are no saved SRAM images", getIndent());
        return;
    }

    LOG_SRAM_IMG(GC, "{}Print #{} saved SRAM images {}", getIndent(), m_sramImages.size(), msg);
    incIndent();
    for (std::list<SramImage>::iterator rangesIter = m_sramImages.begin();
         rangesIter!=m_sramImages.end(); ++rangesIter )
    {
        LOG_SRAM_IMG(GC, "{}IMG #{}:", getIndent(), count);
        LOG_SRAM_IMG(GC, "{}======", getIndent());
        printSramImage(*rangesIter);
        count++;
    }
    decIndent();
}

void SlidingWindowAllocator::printCurrentSramImage(pNode barrierNode, pNode node)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC)) return;

    LOG_SRAM_IMG(GC, "{}Current SRAM image:", getIndent());
    MemoryAllocator& sramAlloc  = m_graph->getCodeGenerator()->getSramAllocator();
    HB_ASSERT((sramAlloc.getMemAllocatorType() == MEMORY_HEAP_ALLOCATOR), "Allocator type not supported.");

    SramImage img;
    img.freeRanges      = ((HeapAllocator&)sramAlloc).getFreeRanges();
    img.occupiedRanges  = ((HeapAllocator&)sramAlloc).getOccupiedRanges(); // Current occupied ranges
    img.barrierNode     = barrierNode;
    img.node            = node;

    incIndent();
    // [CID: 36891] False positive - coverity ignores std::map default c'tor
    printSramImage(img);
    decIndent();
}
