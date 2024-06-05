#include "heap_allocator.h"
#include <cassert>
#include <iterator>

#include <sstream>
#include "define_synapse_common.hpp"

#include "utils.h"
#include "timer.h"

#define LOG_PRINT_STATUS LOG_DEBUG
// #define LOG_PRINT_STATUS        LOG_ERR
//
// #define PRINT_STATUS_MINIMAL_INFO
// #define PRINT_STATUS_LIMITED_INFO
//  Otherwise - PRINT_STATUS_FULL_INFO (this is not a define!)

#define HEAP_ALLOCATOR_LOCK()                                                                                          \
    std::unique_lock lock(m_heapAllocatorMutex, std::defer_lock);                                                      \
    if (m_threadSafe) lock.lock()

template<class RangeT>
HeapAllocatorBase<RangeT>::HeapAllocatorBase(const std::string& name, uint32_t statFreq, bool threadSafe, bool logStats)
: MemoryAllocatorBase(MEMORY_HEAP_ALLOCATOR, name), m_threadSafe(threadSafe)
{
    if (logStats)
    {
        m_stat.emplace(name, statFreq, logStats);
    }
}

template<class RangeT>
HeapAllocatorBase<RangeT>::HeapAllocatorBase(const HeapAllocatorBase<RangeT>& other)
: MemoryAllocatorBase(other),
  m_freeRanges(other.m_freeRanges),
  m_occupiedRanges(other.m_occupiedRanges),
  m_currentlyUsed(other.m_currentlyUsed),
  m_stat(other.m_stat),
  m_threadSafe(other.m_threadSafe)
{
}

template<class RangeT>
HeapAllocatorBase<RangeT>::~HeapAllocatorBase() = default;

template<class RangeT>
void HeapAllocatorBase<RangeT>::Init(uint64_t memorySize, deviceAddrOffset base)
{
    MemoryAllocatorBase::Init(memorySize, base);
    RangeT allRanges;
    allRanges.size = m_memorySize;
    allRanges.base = m_base;

    m_freeRanges.push_back(allRanges);
    insertRange(m_freeRanges.begin());
}

template<class RangeT>
void HeapAllocatorBase<RangeT>::collectStatistics(StatPoints point, uint64_t sum)
{
    if (m_stat) m_stat->collect(point, sum);
}

template<class RangeT>
bool HeapAllocatorBase<RangeT>::allocateReqRange(const Range& reqRange, unsigned pad)
{
    HEAP_ALLOCATOR_LOCK();

    for (auto rangesIter = m_freeRanges.begin(); rangesIter != m_freeRanges.end(); ++rangesIter)
    {
        Range& range = *rangesIter;
        if (rangeContainedIn(reqRange, range))
        {
            setCandidateOneAfter(rangesIter);
            allocateRangeI(reqRange, rangesIter);
            m_occupiedRanges[reqRange.base + pad] = reqRange;
            m_currentlyUsed += reqRange.size;
            _PrintStatus();
            // Reset the current free ranges iterator if needed
            wrapCandidateIfEnd();
            return true;
        }
    }

    LOG_ERR(HEAP_ALLOC,
            "{}: requested range (base = 0x{:x} size = {}) was not found in current free ranges",
            HLLOG_FUNC,
            reqRange.base,
            reqRange.size);
    return false;
}

template<class RangeT>
void HeapAllocatorBase<RangeT>::allocateRangeI(const Range&                                reqRange,
                                               const typename std::list<RangeT>::iterator& rangesIter)
{
    {
        Range& fromRange = *rangesIter;
        assert(rangesIter->size >= reqRange.size);
        assert(reqRange.base >= fromRange.base);

        // Allocate a range inside a free range. 3 options
        // 1) Allocated range same as free range
        // 2) Allocated range start at the same address as the free but is smaller -> leftover at the end
        // 3) Allocated range ends  at the same address as the free but is smaller -> leftover at the beginning
        // 4) Allocated range in the middile of the free range -> leftovers at the begining and end

        if (fromRange.size == reqRange.size)
        {
            // 1) Allocated range same as free range
            removeRange(rangesIter);
            m_freeRanges.erase(rangesIter);
        }
        else
        {
            if (fromRange.base == reqRange.base)
            {
                // 2) Allocated range start at the same address as the free but is smaller -> leftover at the end
                fromRange.base += reqRange.size;
                fromRange.size -= reqRange.size;
                resizeRange(rangesIter);
            }
            else if ((fromRange.base + fromRange.size) == (reqRange.base + reqRange.size))
            {
                // 3) Allocated range ends  at the same address as the free but is smaller
                fromRange.size -= reqRange.size;
                resizeRange(rangesIter);
            }
            else
            {
                // 4) Allocated range in the middile of the free range
                Range newRange;
                newRange.base  = reqRange.base + reqRange.size;
                newRange.size  = fromRange.base + fromRange.size - newRange.base;
                fromRange.size = reqRange.base - fromRange.base;
                auto newIter   = m_freeRanges.insert(std::next(rangesIter), newRange);

                resizeRange(rangesIter);
                insertRange(newIter);
            }
        }
    }
}

template<class RangeT>
Settable<deviceAddrOffset> HeapAllocatorBase<RangeT>::Allocate(uint64_t size,
                                                               uint64_t alignment /* = 128*/,
                                                               uint64_t offset /* = 0*/,
                                                               bool     allowFailure /* = false*/,
                                                               uint64_t requestedAddress /* = 0 */)
{
    auto x = AllocateReturnInfo(size, alignment, offset, allowFailure, requestedAddress);
    return x.first;
}

template<class RangeT>
std::pair<Settable<deviceAddrOffset>, uint64_t>
HeapAllocatorBase<RangeT>::AllocateReturnInfo(uint64_t size,
                                              uint64_t alignment /* = 128*/,
                                              uint64_t offset /* = 0*/,
                                              bool     allowFailure /* = false*/,
                                              uint64_t requestedAddress /* = 0 */)
{
    HEAP_ALLOCATOR_LOCK();
    TimeTools::StdTime allocStart;
    if (m_stat)
    {
        allocStart = TimeTools::timeNow();
    }

    if (m_memorySize == m_currentlyUsed)
    {
        LOG_DEBUG(HEAP_ALLOC, "All memory is already been allocated");
        return {Settable<deviceAddrOffset>(), 0};
    }

    assert(!m_freeRanges.empty() && "m_freeRanges should have atleast 1 element");

    uint64_t allocationSize = size + offset;
    uint64_t pad            = 0;

    typename std::list<RangeT>::iterator freeRangeIter;

    setNextCandidate();

    do
    {
        if (requestedAddress == 0)
        {
            break;
        }

        // Validate padding - no padding allowed on a requested-address
        pad = _CalculatePadding(requestedAddress, alignment);
        // Not sure that there is sense for allowing offset either
        if (pad != 0)
        {
            LOG_WARN(HEAP_ALLOC, "Requested address must be aligned, but requires padding {}", pad);
            break;
        }

        freeRangeIter = m_freeRanges.begin();

        while (freeRangeIter != m_freeRanges.end())
        {
            if (freeRangeIter->base > requestedAddress)
            {
                break;
            }
            ++freeRangeIter;
        }

        typename std::list<RangeT>::iterator candidateFreeRangeIter = freeRangeIter;
        typename std::list<RangeT>::iterator nextFreeRangeIter      = m_freeRanges.end();

        if (freeRangeIter == m_freeRanges.begin())
        {
            if (freeRangeIter->base > requestedAddress)
            {
                LOG_DEBUG(HEAP_ALLOC,
                          "Failed to allocate requested adress. Minimum free address is 0x{:x}",
                          freeRangeIter->base);
                break;
            }

            if (freeRangeIter != m_freeRanges.end())
            {
                nextFreeRangeIter = ++freeRangeIter;
            }
        }
        else
        {
            nextFreeRangeIter      = freeRangeIter;
            candidateFreeRangeIter = std::prev(freeRangeIter);
        }

        uint64_t candidateRageStartAddress = candidateFreeRangeIter->base;
        uint64_t candidateRageEndAddress   = candidateRageStartAddress + candidateFreeRangeIter->size;
        uint64_t requestedRangeBaseAddress = requestedAddress - offset;
        uint64_t requestedRangeEndAddress  = requestedRangeBaseAddress + allocationSize;

        if (candidateRageEndAddress < requestedRangeEndAddress)
        {
            LOG_DEBUG(HEAP_ALLOC,
                      "Failed to allocate requested adress due to out-of-range (required 0x{:x} available 0x{:x})",
                      requestedRangeEndAddress,
                      candidateRageEndAddress);
            break;
        }

        // For simplicity (in general, when dealing with requested addresses, we probably just don't care about the
        //  other mechanism... at the moment)
        incCandidateIfSame(candidateFreeRangeIter);

        // create new range (leftovers from candidate range) - post requested range
        if (requestedRangeEndAddress < candidateRageEndAddress)
        {
            RangeT postReuqestedRangeLeftover;

            postReuqestedRangeLeftover.base = requestedRangeEndAddress;
            postReuqestedRangeLeftover.size = candidateRageEndAddress - postReuqestedRangeLeftover.base;

            if (nextFreeRangeIter != m_freeRanges.end())
            {
                nextFreeRangeIter = m_freeRanges.insert(nextFreeRangeIter, postReuqestedRangeLeftover);
                insertRange(nextFreeRangeIter);
            }
            else
            {
                m_freeRanges.push_back(postReuqestedRangeLeftover);
                nextFreeRangeIter = std::prev(m_freeRanges.end());
                insertRange(nextFreeRangeIter);
            }
        }

        // create new range (leftovers from candidate range) - pre requested range
        if (candidateRageStartAddress < requestedRangeBaseAddress)
        {
            RangeT preReuqestedRangeLeftover;

            preReuqestedRangeLeftover.base = candidateRageStartAddress;
            preReuqestedRangeLeftover.size = requestedRangeBaseAddress - preReuqestedRangeLeftover.base;

            if (nextFreeRangeIter != m_freeRanges.end())
            {
                auto iter = m_freeRanges.insert(nextFreeRangeIter, preReuqestedRangeLeftover);
                insertRange(iter);
            }
            else
            {
                m_freeRanges.push_back(preReuqestedRangeLeftover);
                auto iter = std::prev(m_freeRanges.end());
                insertRange(iter);
            }
        }

        RangeT requestedRange;
        requestedRange.base = requestedRangeBaseAddress;
        requestedRange.size = allocationSize;

        Settable<deviceAddrOffset> addr;
        addr = requestedRangeBaseAddress + offset;
        removeRange(candidateFreeRangeIter);
        m_freeRanges.erase(candidateFreeRangeIter);

        LOG_DEBUG(HEAP_ALLOC, "Allocate HEAP at addr 0x{:x}, size 0x{:x}", addr.value(), allocationSize);
        HB_ASSERT(requestedRangeBaseAddress + allocationSize <= m_base + m_memorySize,
                  "Allocation outside of memory range");

        m_occupiedRanges[addr.value()] = requestedRange;
        m_currentlyUsed += requestedRange.size;
        _PrintStatus();

        return {addr, requestedRange.size};
    } while (0);

    // use aligned size allocations to be consistent with outside code tensor size calculations.
    // should almost not increase memory consumption, but ease interaction with outside code.
    allocationSize = round_to_multiple(allocationSize, alignment);

    // Start allocating from the last used SRAM range
    uint64_t paddedAllocationSize = 0;
    auto     itr                  = findRange(allocationSize, alignment);
    if (itr != m_freeRanges.end())
    {
        pad                  = _CalculatePadding(itr->base, alignment);
        paddedAllocationSize = allocationSize + pad;

        Settable<deviceAddrOffset> addr;

        _AllocateRange(itr, paddedAllocationSize, pad + offset, addr);
        collectStatistics(StatPoints::alloc, TimeTools::timeFromNs(allocStart));
        return {addr, paddedAllocationSize};
    }

    if (!allowFailure)
    {
        LOG_ERR(HEAP_ALLOC, "Failed to allocate size 0x{:x} in {} memory", allocationSize, m_name);
    }
    else
    {
        LOG_TRACE(HEAP_ALLOC, "Failed to allocate size 0x{:x} in {} memory", allocationSize, m_name);
    }

    return {Settable<deviceAddrOffset>(), 0};
}

template<class RangeT>
void HeapAllocatorBase<RangeT>::Free(deviceAddrOffset ptr)
{
    FreeReturnInfo(ptr);
}

template<class RangeT>
uint64_t HeapAllocatorBase<RangeT>::FreeReturnInfo(deviceAddrOffset ptr)
{
    HEAP_ALLOCATOR_LOCK();
    TimeTools::StdTime freeStart;

    if (m_stat)
    {
        freeStart = TimeTools::timeNow();
    }

    auto               it        = m_occupiedRanges.find(ptr);
    if (it == m_occupiedRanges.end())
    {
        LOG_ERR(HEAP_ALLOC, "Attempt to free non-existent allocation at offset {}", ptr);
        assert(0);
        return 0;
    }

    // Found the allocation. Remove it
    RangeT allocation = it->second;
    m_currentlyUsed -= allocation.size;
    m_occupiedRanges.erase(it);

    LOG_DEBUG(HEAP_ALLOC, "Free HEAP at address 0x{:x}, size 0x{:x}", ptr, allocation.size);

    // Attempt to coalesce the freed allocation
    // Find the two adjacent allocations
    deviceAddrOffset allocationEnd = allocation.base + allocation.size;
    auto             next          = m_freeRanges.begin();
    while (next != m_freeRanges.end())
    {
        if (next->base >= allocationEnd)
        {
            break;
        }
        ++next;
    }
    // Corner case: no previous allocation
    if (next == m_freeRanges.begin())
    {
        if (next != m_freeRanges.end() && allocationEnd == next->base)
        {
            // can coalesce, increase the first range by <allocation size>
            next->base -= allocation.size;
            next->size += allocation.size;
            resizeRange(next);
        }
        else
        {
            m_freeRanges.push_front(allocation);
            insertRange(m_freeRanges.begin());
        }
        _PrintStatus();
        collectStatistics(StatPoints::free, TimeTools::timeFromNs(freeStart));
        return -allocation.size;
    }

    auto prev = std::prev(next);
    // attempt to fuse with prev
    if (prev->base + prev->size == allocation.base)
    {
        prev->size += allocation.size;
        resizeRange(prev);
        // Potentially add next
        if ((next != m_freeRanges.end()) && (allocationEnd == next->base))
        {
            // Double fuse
            // m_freeCont.rezise(range);
            prev->size += next->size;
            resizeRange(prev);
            incCandidateIfSame(next);
            removeRange(next);
            m_freeRanges.erase(next);
        }
    }
    // attempt to fuse with next
    else if ((next != m_freeRanges.end()) && (allocationEnd == next->base))
    {
        next->base -= allocation.size;
        next->size += allocation.size;
        resizeRange(next);
    }
    // No possible coalescing
    else
    {
        auto itr = m_freeRanges.insert(next, allocation);
        insertRange(itr);
    }
    collectStatistics(StatPoints::free, TimeTools::timeFromNs(freeStart));
    _PrintStatus();
    return -allocation.size;
}

template<class RangeT>
uint64_t HeapAllocatorBase<RangeT>::GetCurrentlyUsed() const
{
    return m_currentlyUsed;
}

template<class RangeT>
uint64_t HeapAllocatorBase<RangeT>::getMaxFreeContiguous() const
{
    Range range = getMaxFreeRange();
    return range.size;
}

template<class RangeT>
Range HeapAllocatorBase<RangeT>::getMaxFreeRange() const
{
    HEAP_ALLOCATOR_LOCK();
    // Initialize maxRange
    Range maxRange;
    maxRange.base = 0;
    maxRange.size = 0;

    uint64_t maxFree = 0;
    for (auto range : m_freeRanges)
    {
        if (range.size > maxFree)
        {
            maxFree  = range.size;
            maxRange = range;
        }
    }
    return maxRange;
}

std::unique_ptr<MemoryAllocator> HeapAllocator::Clone()
{
    HEAP_ALLOCATOR_LOCK();
    return std::unique_ptr<MemoryAllocator> {new HeapAllocator(*this)};
}

template<class RangeT>
bool HeapAllocatorBase<RangeT>::IsAllocated(deviceAddrOffset ptr) const
{
    HEAP_ALLOCATOR_LOCK();
    auto it = m_occupiedRanges.find(ptr);
    return (it != m_occupiedRanges.end());
}

template<class RangeT>
void HeapAllocatorBase<RangeT>::_PrintStatus() const
{
#if defined(PRINT_STATUS_MINIMAL_INFO)
    static constexpr std::string_view OCCUPIED_AREA_MARK = "+";
    static constexpr std::string_view FREE_AREA_MARK     = "-";
#elif defined(PRINT_STATUS_LIMITED_INFO)
    static constexpr std::string_view OCCUPIED_AREA_MARK = "[+++]";
    static constexpr std::string_view FREE_AREA_MARK     = "[---]";
#else
    static constexpr std::string_view OCCUPIED_AREA_MARK = "[++++++]";
    static constexpr std::string_view FREE_AREA_MARK     = "[------]";
#endif
    if (!m_isPrintStatusAllowed)
    {
        return;
    }

    std::stringstream rangeStatusStream;
#ifndef PRINT_STATUS_MINIMAL_INFO
    std::stringstream rangeSizeStream;
#endif

    deviceAddrOffset previousOccupiedRangeSuffix = m_base;
    auto             next                        = m_occupiedRanges.begin();
    const Range*     pNextOccupiedRange;

    while (next != m_occupiedRanges.end())
    {
        pNextOccupiedRange = &(next->second);

        if (pNextOccupiedRange->base != previousOccupiedRangeSuffix)
        {
            // PrintInfo of previous (free) range
            rangeStatusStream << FREE_AREA_MARK;
#ifndef PRINT_STATUS_MINIMAL_INFO
            rangeSizeStream << _GetRangeSizeDescription(pNextOccupiedRange->base - previousOccupiedRangeSuffix);
#endif
        }

        // PrintInfo of current (occupied) range
        rangeStatusStream << OCCUPIED_AREA_MARK;
#ifndef PRINT_STATUS_MINIMAL_INFO
        rangeSizeStream << _GetRangeSizeDescription(pNextOccupiedRange->size);
#endif

        // Update parameters
        previousOccupiedRangeSuffix = pNextOccupiedRange->base + pNextOccupiedRange->size;
        next++;
    }

    // Print the last region if not occupied
    if (previousOccupiedRangeSuffix != m_base + m_memorySize)
    {
        // PrintInfo of previous (free) range
        rangeStatusStream << FREE_AREA_MARK;
#ifndef PRINT_STATUS_MINIMAL_INFO
        rangeSizeStream << _GetRangeSizeDescription(m_base + m_memorySize - previousOccupiedRangeSuffix);
#endif
    }

    LOG_PRINT_STATUS(HEAP_ALLOC, "\nMemory status:");
    LOG_PRINT_STATUS(HEAP_ALLOC, "{}", rangeStatusStream.str());
#ifndef PRINT_STATUS_MINIMAL_INFO
    LOG_PRINT_STATUS(HEAP_ALLOC, "{}", rangeSizeStream.str());
#endif
}

template<class RangeT>
std::string HeapAllocatorBase<RangeT>::_GetRangeSizeDescription(uint64_t size) const
{
    static const uint32_t ONE_KILO_BYTE = 1024;
    static const uint32_t ONE_MEGA_BYTE = 1024 * ONE_KILO_BYTE;
    static const uint32_t ONE_GIGA_BYTE = 1024 * ONE_MEGA_BYTE;

    enum SizeType
    {
        SIZE_TYPE_KILO_BYTE,
        SIZE_TYPE_MEGA_BYTE,
        SIZE_TYPE_GIGA_BYTE
    };

    SizeType          rangeSizeType(SIZE_TYPE_KILO_BYTE);
    std::stringstream rangeSizeStream;
#ifndef PRINT_STATUS_LIMITED_INFO
    uint32_t numInSizeType;
#endif

    if (size < ONE_MEGA_BYTE)
    {
        rangeSizeType = SIZE_TYPE_KILO_BYTE;
#ifndef PRINT_STATUS_LIMITED_INFO
        numInSizeType = size / ONE_KILO_BYTE;
#endif
    }
    else if (size < ONE_GIGA_BYTE)
    {
        rangeSizeType = SIZE_TYPE_MEGA_BYTE;
#ifndef PRINT_STATUS_LIMITED_INFO
        numInSizeType = size / ONE_MEGA_BYTE;
#endif
    }
    else
    {
        rangeSizeType = SIZE_TYPE_GIGA_BYTE;
#ifndef PRINT_STATUS_LIMITED_INFO
        numInSizeType = size / ONE_GIGA_BYTE;
#endif
    }

    rangeSizeStream << "[ ";
#ifndef PRINT_STATUS_LIMITED_INFO
    if (numInSizeType == 0)
    {
        rangeSizeStream << "0.5";
    }
    else
    {
        rangeSizeStream << fmt::format("{:<3}", numInSizeType);
    }
#endif

    switch (rangeSizeType)
    {
        case SIZE_TYPE_KILO_BYTE:
            rangeSizeStream << "K";
            break;

        case SIZE_TYPE_MEGA_BYTE:
            rangeSizeStream << "M";
            break;

        case SIZE_TYPE_GIGA_BYTE:
            rangeSizeStream << "G";
            break;
    }
    rangeSizeStream << " ]";

    return rangeSizeStream.str();
}

template<class RangeT>
Settable<deviceAddrOffset>
HeapAllocatorBase<RangeT>::GetStartOfFreeRangeContaningOffset(const deviceAddrOffset offset) const
{
    HEAP_ALLOCATOR_LOCK();
    Settable<deviceAddrOffset> result;
    Range                      tmpRange;

    result.unset();
    tmpRange.base = offset;
    tmpRange.size = 0;

    for (auto rangesIter = m_freeRanges.begin(); rangesIter != m_freeRanges.end(); ++rangesIter)
    {
        Range range = *rangesIter;

        if (rangeContainedIn(tmpRange, range))
        {
            result.set(range.base);
            break;
        }
    }

    return result;
}

void HeapAllocator::_AdvanceFreeRangeIterator()
{
    ++m_freeRangesIterator;
    // Reset the current free ranges iterator if needed
    if (m_freeRangesIterator == m_freeRanges.end())
    {
        m_freeRangesIterator = m_freeRanges.begin();
    }
}

template<class RangeT>
void HeapAllocatorBase<RangeT>::_AllocateRange(typename std::list<RangeT>::iterator itr,
                                               uint64_t                    allocationSize,  // size + pad + offset
                                               uint64_t                    addressOffsetFromBase,  // pad + offset
                                               Settable<deviceAddrOffset>& addr)
{
    // Found a fit
    Range newAllocation;
    addr               = itr->base + addressOffsetFromBase;
    newAllocation.base = itr->base;
    newAllocation.size = allocationSize;

    LOG_DEBUG(HEAP_ALLOC, "Allocate HEAP at addr 0x{:x}, size 0x{:x}", addr.value(), newAllocation.size);
    HB_ASSERT(newAllocation.base + allocationSize <= m_base + m_memorySize, "Allocation outside of memory range");

    m_occupiedRanges[addr.value()] = newAllocation;

    if (itr->size == newAllocation.size)
    {
        // Update free ranges iterator before deleting in order to preserve valid iterator
        auto toBeErasedIterator = itr;
        incCandidateIfSame(itr);
        removeRange(toBeErasedIterator);
        m_freeRanges.erase(toBeErasedIterator);
    }
    else
    {
        itr->base += newAllocation.size;
        itr->size -= newAllocation.size;
        resizeRange(itr);
    }
    m_currentlyUsed += newAllocation.size;
    _PrintStatus();
}

template<class RangeT>
uint64_t HeapAllocatorBase<RangeT>::_CalculatePadding(uint64_t baseAddress, uint64_t alignment)
{
    uint64_t alignmentOffset = baseAddress % alignment;

    if (alignmentOffset == 0)
    {
        return 0;
    }

    return (alignment - alignmentOffset);
}

/*********************************************************************/
/****                  HeapAllocator                             *****/
/*********************************************************************/
void HeapAllocator::setNextCandidate()
{
    if (!m_cyclicAllocation || (m_freeRangesIterator == m_freeRanges.end()))
    {
        m_freeRangesIterator = m_freeRanges.begin();
    }
}

void HeapAllocator::incCandidateIfSame(typename std::list<Range>::iterator itr)
{
    if (itr == m_freeRangesIterator) m_freeRangesIterator++;
}

std::list<Range>::iterator HeapAllocator::findRange(uint64_t allocationSize, uint64_t alignment)
{
    auto                                 lastUsedRangeIterator = m_freeRangesIterator;
    uint64_t                             paddedAllocationSize  = 0;
    int                                  cnt                   = 0;
    Settable<std::list<Range>::iterator> bestFitRange;
    do
    {
        cnt++;
        Range& range = *m_freeRangesIterator;

        // calculate size including alignment
        uint64_t pad         = _CalculatePadding(range.base, alignment);
        paddedAllocationSize = allocationSize + pad;

        if (range.size >= paddedAllocationSize)
        {
            collectStatistics(StatPoints::triesToFind, cnt);
            if (!m_bestFitAllocation || (range.size == paddedAllocationSize))
            {
                return m_freeRangesIterator;
            }
            if (!bestFitRange.is_set() || (range.size < bestFitRange.value()->size))
            {
                bestFitRange = m_freeRangesIterator;
            }
        }
        _AdvanceFreeRangeIterator();

    } while (m_freeRangesIterator != lastUsedRangeIterator);

    return (!m_bestFitAllocation || !bestFitRange.is_set()) ? m_freeRanges.end() : bestFitRange.value();
}

void HeapAllocator::wrapCandidateIfEnd()
{
    if (m_freeRangesIterator == m_freeRanges.end())
    {
        m_freeRangesIterator = m_freeRanges.begin();
    }
}

/*********************************************************************/
/****                  HeapAllocatorBestFit                      *****/
/*********************************************************************/
void HeapAllocatorBestFit::removeRange(freeListIterT iter)
{
    auto setIter  = iter->setIter;
    iter->setIter = m_setBySize.end();  // Put "bad" value - Range not pointing to the set anymore.
    m_setBySize.erase(setIter);
}

void HeapAllocatorBestFit::resizeRange(freeListIterT iter)
{
    auto setIter = iter->setIter;
    m_setBySize.erase(setIter);
    setIter       = m_setBySize.insert(iter);
    iter->setIter = setIter;
}

void HeapAllocatorBestFit::insertRange(freeListIterT iter)
{
    auto setIter  = m_setBySize.insert(iter);
    iter->setIter = setIter;
}

std::list<RangeE>::iterator HeapAllocatorBestFit::findRange(uint64_t allocationSize, uint64_t alignment)
{
    // In some cases it might be better to search for a bigger size (because of alignment), it depends on the actual
    // sizes and alignment requested. You can change the +0 below to +alignment for testing
    RangeE rangeE {{0.0}};
    rangeE.size = allocationSize + 0;
    std::list<RangeE> list {rangeE};

    auto     iter                 = list.begin();
    auto     candidateIter        = m_setBySize.lower_bound(iter);
    uint64_t paddedAllocationSize = 0;
    int      cnt                  = 0;

    while (candidateIter != m_setBySize.end())
    {
        cnt++;
        uint64_t pad         = _CalculatePadding((*candidateIter)->base, alignment);
        paddedAllocationSize = allocationSize + pad;

        if ((*candidateIter)->size >= paddedAllocationSize)
        {
            collectStatistics(StatPoints::triesToFind, cnt);
            return *candidateIter;
        }
        candidateIter++;
    }
    return m_freeRanges.end();
}

std::unique_ptr<MemoryAllocator> HeapAllocatorBestFit::Clone()
{
    HEAP_ALLOCATOR_LOCK();
    return std::unique_ptr<MemoryAllocator> {new HeapAllocatorBestFit(*this)};
}

Range HeapAllocatorBestFit::getMaxFreeRange() const
{
    HEAP_ALLOCATOR_LOCK();
    auto lastItr = m_setBySize.rbegin();
    if (lastItr != m_setBySize.rend())
    {
        return *(*lastItr);
    }
    return Range {0, 0};
};

/*********************************************************************/
/****                  HeapAllocatorWrapper                     *****/
/*********************************************************************/
HeapAllocatorWrapper::HeapAllocatorWrapper(const std::string&                    name,
                                           const TensorVector&                   outputTensors,
                                           const LivenessAnalysis&               livenessAnalysis,
                                           const std::shared_ptr<HeapAllocator>& workspaceHeapAllocator,
                                           uint32_t                              statFreq,
                                           bool                                  threadSafe,
                                           bool                                  logStats)
: MemoryAllocatorBase(MEMORY_HEAP_ALLOCATOR, name),
  m_statFreq(statFreq),
  m_threadSafe(threadSafe),
  m_logStats(logStats),
  m_WSAllocator(workspaceHeapAllocator)
{
    for (const TensorPtr& t : outputTensors)
    {
        uint64_t      tensor_sizes     = t->getMinimalSizeInBytes();
        Lifetime      tensorLife       = livenessAnalysis.getTensorLifeTime(t);
        uint64_t      tensorSectionIdx = t->getMemorySectionID();
        std::string   allocatorName    = fmt::format("{}_output_buffer_{}", name, tensorSectionIdx);
        HeapAllocator outputAllocator(allocatorName, statFreq, false, threadSafe, logStats);

        outputAllocator.Init(tensor_sizes, t->getDramOffset());
        LOG_DEBUG(HEAP_ALLOC,
                  "creating new output buffer allocator with size {} and lifetime start {} end {} tensor sectionIdx {}",
                  tensor_sizes,
                  tensorLife.m_start,
                  tensorLife.m_end,
                  tensorSectionIdx);

        m_sectionIdxToAllocatorLivenessMap.emplace(tensorSectionIdx, std::make_pair(outputAllocator, tensorLife));
        m_sortedAllocatorLivenessMap.emplace_back(std::make_pair(tensor_sizes, tensorSectionIdx));
    }
    std::sort(m_sortedAllocatorLivenessMap.begin(), m_sortedAllocatorLivenessMap.end());
}

HeapAllocatorWrapper::HeapAllocatorWrapper(const HeapAllocatorWrapper& other)
: MemoryAllocatorBase(other),
  m_statFreq(other.m_statFreq),
  m_threadSafe(other.m_threadSafe),
  m_logStats(other.m_logStats),
  m_sectionIdxToAllocatorLivenessMap(other.m_sectionIdxToAllocatorLivenessMap),
  m_sortedAllocatorLivenessMap(other.m_sortedAllocatorLivenessMap)
{
    std::shared_ptr<MemoryAllocator> workspaceAllocatorClone = other.m_WSAllocator->Clone();
    m_WSAllocator = std::static_pointer_cast<HeapAllocator>(workspaceAllocatorClone);
}

HeapAllocatorWrapper::~HeapAllocatorWrapper() = default;

void HeapAllocatorWrapper::InitWSAllocator(uint64_t memorySize, deviceAddrOffset base)
{
    m_WSAllocator =
        std::make_shared<HeapAllocator>(fmt::format("{}_ws", m_name), m_statFreq, false, m_threadSafe, m_logStats);
    m_WSAllocator->Init(memorySize, base);
}

Settable<deviceAddrOffset> HeapAllocatorWrapper::Allocate(uint64_t size,
                                                          uint64_t alignment /* = 128*/,
                                                          Lifetime tensorLifeTime,
                                                          uint64_t offset /* = 0*/,
                                                          bool     allowFailure /* = false*/,
                                                          uint64_t requestedAddress /* = 0 */)
{
    for (auto& [sectionSize, sectionIdx] : m_sortedAllocatorLivenessMap)
    {
        if (sectionSize < size) continue;
        auto& [heapAllocator, allocatorLifeTime] = m_sectionIdxToAllocatorLivenessMap.at(sectionIdx);
        if (lifetimeIntersectsWith(allocatorLifeTime, tensorLifeTime)) continue;
        Settable<deviceAddrOffset> addr = heapAllocator.Allocate(size, alignment, offset, true, requestedAddress);
        if (addr.is_set())
        {
            LOG_DEBUG(HEAP_ALLOC, "allocate tensor in one of persitent output size 0x{:x}", size);
            return addr;
        }
    }
    LOG_DEBUG(HEAP_ALLOC, "allocate tensor in WS allocator tensors size 0x{:x}", size);
    return m_WSAllocator->Allocate(size, alignment, offset, allowFailure, requestedAddress);
}

void HeapAllocatorWrapper::Free(deviceAddrOffset ptr)
{
    uint64_t sectionidx = getMemoryIDFromVirtualAddress(ptr);
    if (sectionidx == MEMORY_ID_RESERVED_FOR_WORKSPACE)
    {
        m_WSAllocator->Free(ptr);
    }
    else
    {
        m_sectionIdxToAllocatorLivenessMap.at(sectionidx).first.Free(ptr);
    }
}

bool HeapAllocatorWrapper::allocateReqRange(const Range& reqRange, unsigned pad)
{
    HB_ASSERT(getMemoryIDFromVirtualAddress(reqRange.base) == MEMORY_ID_RESERVED_FOR_WORKSPACE,
              "allocateReqRange expecting ranges only from workspace section");
    return m_WSAllocator->allocateReqRange(reqRange, pad);
}

uint64_t HeapAllocatorWrapper::GetCurrentlyUsed() const
{
    uint64_t currentlyUsed = 0;
    for (auto& [sectionIdx, allocatorLiveness] : m_sectionIdxToAllocatorLivenessMap)
    {
        currentlyUsed += allocatorLiveness.first.GetCurrentlyUsed();
    }
    currentlyUsed += m_WSAllocator->GetCurrentlyUsed();
    return currentlyUsed;
}

uint64_t HeapAllocatorWrapper::getMaxFreeContiguous() const
{
    return m_WSAllocator->getMaxFreeContiguous();
}

bool HeapAllocatorWrapper::IsAllocated(deviceAddrOffset ptr) const
{
    uint64_t sectionidx = getMemoryIDFromVirtualAddress(ptr);
    if (sectionidx == MEMORY_ID_RESERVED_FOR_WORKSPACE)
    {
        return m_WSAllocator->IsAllocated(ptr);
    }
    else
    {
        return m_sectionIdxToAllocatorLivenessMap.at(sectionidx).first.IsAllocated(ptr);
    }
}

std::unique_ptr<MemoryAllocator> HeapAllocatorWrapper::Clone()
{
    return std::unique_ptr<MemoryAllocator> {new HeapAllocatorWrapper(*this)};
}

Range HeapAllocatorWrapper::getMaxFreeRange() const
{
    return m_WSAllocator->getMaxFreeRange();
}

Settable<deviceAddrOffset> HeapAllocatorWrapper::GetStartOfFreeRangeContaningOffset(const deviceAddrOffset offset) const
{
    HB_ASSERT(getMemoryIDFromVirtualAddress(offset) == MEMORY_ID_RESERVED_FOR_WORKSPACE,
              "GetStartOfFreeRangeContaningOffset expecting ranges only from workspace section");
    return m_WSAllocator->GetStartOfFreeRangeContaningOffset(offset);
}
