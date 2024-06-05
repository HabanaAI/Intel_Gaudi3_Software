#pragma once
#include <list>
#include <map>
#include "memory_allocator.h"
#include "range.h"
#include <mutex>
#include <statistics.hpp>
#include <optional>
#include "liveness_analysis.h"

template<class RangeT>
class HeapAllocatorBase : public MemoryAllocatorBase
{
protected:
    enum class StatPoints
    {
        alloc,
        free,
        triesToFind,
        LAST
    };

    static constexpr auto enumNamePoints = toStatArray<StatPoints>({{StatPoints::alloc, "Allocation time(ns)"},
                                                                    {StatPoints::free, "Free time(ns)"},
                                                                    {StatPoints::triesToFind, "Tries to find free"}});

public:
    using freeListT     = typename std::list<RangeT>;
    using freeListIterT = typename freeListT::iterator;

    explicit HeapAllocatorBase(const std::string& name,
                               uint32_t           statFreq,
                               bool               threadSafe = true,
                               bool               logStats   = false);
    virtual ~HeapAllocatorBase();

    HeapAllocatorBase(const HeapAllocatorBase& other);

    virtual void                       Init(uint64_t memorySize, deviceAddrOffset base = 0) override;
    virtual Settable<deviceAddrOffset> Allocate(uint64_t size,
                                                uint64_t alignment,
                                                uint64_t offset           = 0,
                                                bool     allowFailure     = false,
                                                uint64_t requestedAddress = 0) override;
    virtual void                       Free(deviceAddrOffset ptr) override;
    virtual uint64_t                   GetCurrentlyUsed() const override;
    virtual uint64_t                   getMaxFreeContiguous() const override;
    virtual bool                       IsAllocated(deviceAddrOffset ptr) const override;
    freeListT                          getFreeRanges() const { return m_freeRanges; }
    std::map<deviceAddrOffset, Range>  getOccupiedRanges() const
    {
        return m_occupiedRanges;
    }  // for debugging purposes only

    virtual Range getMaxFreeRange() const;
    bool          allocateReqRange(const Range& reqRange, unsigned pad);

    Settable<deviceAddrOffset> GetStartOfFreeRangeContaningOffset(const deviceAddrOffset offset) const;

protected:
    virtual std::pair<Settable<deviceAddrOffset>, uint64_t> AllocateReturnInfo(uint64_t size,
                                                                               uint64_t alignment,
                                                                               uint64_t offset           = 0,
                                                                               bool     allowFailure     = false,
                                                                               uint64_t requestedAddress = 0);
    virtual uint64_t                                        FreeReturnInfo(deviceAddrOffset ptr);
    void                                                    collectStatistics(StatPoints point, uint64_t sum);

private:
    void allocateRangeI(const Range& reqRange, const freeListIterT& rangesIter);

    // overridden by best-fit
    virtual void removeRange(freeListIterT iter) {}
    virtual void resizeRange(freeListIterT iter) {}
    virtual void insertRange(freeListIterT iter) {}

    // overridden by first-fit
    virtual void          setNextCandidate() {}
    virtual void          wrapCandidateIfEnd() {}
    virtual void          incCandidateIfSame(freeListIterT itr) {}
    virtual void          setCandidateOneAfter(freeListIterT itr) {};
    virtual freeListIterT findRange(uint64_t allocationSize, uint64_t alignment) = 0;

protected:
    void        _PrintStatus() const;
    std::string _GetRangeSizeDescription(uint64_t size) const;

    void _AllocateRange(freeListIterT               itr,
                        uint64_t                    allocationSize,
                        uint64_t                    addressOffsetFromBase,
                        Settable<deviceAddrOffset>& addr);

    uint64_t _CalculatePadding(uint64_t baseAddress, uint64_t alignment);

    // A Sorted (by start address) list of free ranges
    freeListT m_freeRanges;
    // A container for existing allocations
    std::map<deviceAddrOffset, Range> m_occupiedRanges;

    // For statistics
    uint64_t                                  m_currentlyUsed = 0;
    mutable std::mutex                        m_heapAllocatorMutex;
    std::optional<Statistics<enumNamePoints>> m_stat;
    bool                                      m_threadSafe = true;
};

struct RangeE : public Range
{
    RangeE() = default;
    RangeE(Range r) : Range(r) {}

    using freeSetIterT = std::set<HeapAllocatorBase<RangeE>::freeListIterT>::iterator;
    freeSetIterT setIter;
};

template class HeapAllocatorBase<RangeE>;
template class HeapAllocatorBase<Range>;

class HeapAllocator : public HeapAllocatorBase<Range>
{
public:
    HeapAllocator(const std::string& name,
                  uint32_t           statFreq         = 0,
                  bool               cyclicAllocation = true,
                  bool               threadSafe       = true,
                  bool               logStats         = false)
    : HeapAllocatorBase(name, statFreq, threadSafe, logStats),
      m_freeRangesIterator(m_freeRanges.end()),
      m_cyclicAllocation(cyclicAllocation)
    {
    }
    virtual std::unique_ptr<MemoryAllocator> Clone() override;
    virtual void setCyclicAllocation(bool cyclicAllocation = true) { m_cyclicAllocation = cyclicAllocation; }
    virtual void setBestFitAllocation(bool bestFitAllocation) { m_bestFitAllocation = bestFitAllocation; }

private:
    virtual void          setNextCandidate() override;
    virtual void          incCandidateIfSame(freeListIterT itr) override;
    virtual freeListIterT findRange(uint64_t allocationSize, uint64_t alignment) override;
    virtual void          wrapCandidateIfEnd() override;
    virtual void          setCandidateOneAfter(freeListIterT itr) override { m_freeRangesIterator = std::next(itr); }
    void                  _AdvanceFreeRangeIterator();

private:
    // Iterator on the free ranges, to be able to consume all the memory-space before
    // reusing freed ranges.
    freeListIterT m_freeRangesIterator;

    // Select between cyclic allocation (allocate sequential chunks until memory's end and only then start over),
    // to always pick the first available chunk of memory with enough space.
    bool m_cyclicAllocation = true;

    // Select between best-fit allocation policy (pick the smallest free range possible)
    // to first-fit (pick the first suitable free range).
    bool m_bestFitAllocation = false;
};

class HeapAllocatorBestFit : public HeapAllocatorBase<RangeE>
{
public:
    HeapAllocatorBestFit(const std::string& name, uint32_t statFreq = 0) : HeapAllocatorBase(name, statFreq) {}
    virtual std::unique_ptr<MemoryAllocator> Clone() override;

    virtual Range getMaxFreeRange() const override;

private:
    virtual void          removeRange(freeListIterT iter) override;
    virtual void          resizeRange(freeListIterT iter) override;
    virtual void          insertRange(freeListIterT iter) override;
    virtual freeListIterT findRange(uint64_t allocationSize, uint64_t alignment) override;

private:
    struct setCompare
    {
        bool operator()(freeListIterT lhs, freeListIterT rhs) const { return (*lhs).size < (*rhs).size; }
    };

    std::multiset<freeListIterT, setCompare> m_setBySize;
};

class HeapAllocatorWrapper : public MemoryAllocatorBase
{
public:
    explicit HeapAllocatorWrapper(const std::string&                    name,
                                  const TensorVector&                   outputTensors,
                                  const LivenessAnalysis&               livenessAnalysis,
                                  const std::shared_ptr<HeapAllocator>& workspaceHeapAllocator,
                                  uint32_t                              statFreq,
                                  bool                                  threadSafe = true,
                                  bool                                  logStats   = false);
    virtual ~HeapAllocatorWrapper();

    HeapAllocatorWrapper(const HeapAllocatorWrapper& other);

    virtual void                       InitWSAllocator(uint64_t memorySize, deviceAddrOffset base = 0);
    virtual Settable<deviceAddrOffset> Allocate(uint64_t size,
                                                uint64_t alignment,
                                                Lifetime tensorLifetime,
                                                uint64_t offset           = 0,
                                                bool     allowFailure     = false,
                                                uint64_t requestedAddress = 0) override;
    virtual Settable<deviceAddrOffset> Allocate(uint64_t size,
                                                uint64_t alignment,
                                                uint64_t offset           = 0,
                                                bool     allowFailure     = false,
                                                uint64_t requestedAddress = 0) override
    {
        return Allocate(size, alignment, {}, offset, allowFailure, requestedAddress);
    };
    virtual void                             Free(deviceAddrOffset ptr) override;
    virtual uint64_t                         GetCurrentlyUsed() const override;
    virtual uint64_t                         getMaxFreeContiguous() const override;
    bool                                     allocateReqRange(const Range& reqRange, unsigned pad);
    virtual bool                             IsAllocated(deviceAddrOffset ptr) const override;
    virtual std::unique_ptr<MemoryAllocator> Clone() override;

    Settable<deviceAddrOffset> GetStartOfFreeRangeContaningOffset(const deviceAddrOffset offset) const;
    virtual Range              getMaxFreeRange() const;

protected:
private:
    uint32_t                                                         m_statFreq;
    bool                                                             m_threadSafe;
    bool                                                             m_logStats;
    std::unordered_map<uint64_t, std::pair<HeapAllocator, Lifetime>> m_sectionIdxToAllocatorLivenessMap;
    std::vector<std::pair<uint64_t, uint64_t>>                       m_sortedAllocatorLivenessMap;
    std::shared_ptr<HeapAllocator>                                   m_WSAllocator;
};
