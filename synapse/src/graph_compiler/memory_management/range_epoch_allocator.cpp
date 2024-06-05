#include "range_epoch_allocator.h"
#include "alloc_utils.h"
#include "defs.h"
#include "compilation_hal_reader.h"

RangeEpochAllocator::RangeEpochAllocator(HabanaGraph&          graph,
                                         LivenessAnalysis&     livenessAnalysis,
                                         HeapAllocatorWrapper& mainAllocator)
: m_livenessAnalysis(livenessAnalysis), m_allocator(mainAllocator)
{
}

bool RangeEpochAllocator::allocateMemoryForEpochs(HeapAllocatorWrapper& mainAllocator,
                                                  uint64_t              maxEpochSize,
                                                  uint64_t              maxWSSize)
{
    Range maxRange = mainAllocator.getMaxFreeRange();
    maxRange.size  = std::min(maxWSSize, maxRange.size);

    bool res = mainAllocator.allocateReqRange(maxRange, 0);
    if (!res)
    {
        LOG_ERR(TENSORS_ALLOC,
                "Failed to allocate memory for epoch allocator {}, requested {}",
                maxRange.size,
                maxWSSize);
        return false;
    }

    // align the start and end in case the returned range was unaligned to cache line
    unsigned alignment = 1;
    if (CompilationHalReader::isHalReaderSet())
    {
        alignment = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    }
    auto alignedStart = round_to_multiple(maxRange.base, alignment);
    auto alignedEnd   = round_down(maxRange.base + maxRange.size, alignment);
    auto realWsSize   = alignedEnd - alignedStart;
    m_totalCapcity    = maxEpochSize - (maxWSSize - realWsSize);
    m_allocator.InitWSAllocator(realWsSize, alignedStart);
    return true;
}

void RangeEpochAllocator::sortCandidatesForAllocation(AllocTensorVector& toAllocate)
{
    auto EpochSort = [&](const TensorPtr& lhs, const TensorPtr& rhs) {
        const auto& lhsLife = m_livenessAnalysis.getTensorLifeTime(lhs);
        const auto& rhsLife = m_livenessAnalysis.getTensorLifeTime(rhs);

        if (lhsLife.m_end != rhsLife.m_end) return lhsLife.m_end > rhsLife.m_end;

        uint64_t lhsCap = getWriteSpaceForTensor(lhs);
        uint64_t rhsCap = getWriteSpaceForTensor(rhs);
        return (lhsCap != rhsCap ? lhsCap > rhsCap : lhs->getId() < rhs->getId());
    };

    std::sort(toAllocate.begin(), toAllocate.end(), EpochSort);
}

bool RangeEpochAllocator::attemptCandidateAllocation(uint32_t candidateTensorsCount)
{
    AllocTensorVector toAllocate(m_toAllocate.begin(), m_toAllocate.begin() + candidateTensorsCount);
    sortCandidatesForAllocation(toAllocate);
    auto heapAllocatorWrapperClone = m_allocator.Clone();
    auto heapAllocatorWrapperrPtr  = static_cast<HeapAllocatorWrapper*>(heapAllocatorWrapperClone.get());

    for (const auto& t : toAllocate)
    {
        uint64_t          tensorCap  = getWriteSpaceForTensor(t);
        Lifetime          tensorLife = m_livenessAnalysis.getTensorLifeTime(t);
        TensorAnnotation& tensorAnn  = t->getTensorAnnotation();
        uint64_t          alignment  = tensorAnn.memory.alignment;
        uint64_t          offset     = tensorAnn.memory.offset;
        auto              addr = heapAllocatorWrapperrPtr->Allocate(tensorCap, alignment, tensorLife, offset, true, 0);
        if (!addr.is_set()) return false;
    }
    return true;
}

bool RangeEpochAllocator::adjustCandidatesForAllocation()
{
    uint32_t lowerBound = m_toAllocateLowerBound;
    uint32_t upperBound = m_toAllocate.size();
    // Early exit in case no adjustment is needed
    if (lowerBound == upperBound)
    {
        LOG_DEBUG(TENSORS_ALLOC, "adjustCandidatesForAllocation tensors in Epoch {}", m_toAllocate.size());
        return m_toAllocate.size() != 0;
    }
    uint32_t bestCandidateTensorsCount = lowerBound;
    while (lowerBound <= upperBound)
    {
        auto currentCandidateTensorsCount = lowerBound + (upperBound - lowerBound) / 2;
        bool success                      = attemptCandidateAllocation(currentCandidateTensorsCount);
        if (success)
        {
            bestCandidateTensorsCount = std::max(bestCandidateTensorsCount, currentCandidateTensorsCount);
            lowerBound                = currentCandidateTensorsCount + 1;
        }
        else
        {
            upperBound = currentCandidateTensorsCount - 1;
        }
    }
    m_toAllocate.resize(bestCandidateTensorsCount);
    LOG_DEBUG(TENSORS_ALLOC, "adjustCandidatesForAllocation tensors in Epoch {}", m_toAllocate.size());
    return m_toAllocate.size() != 0;
}

void RangeEpochAllocator::buildPotentialCandidatesForAllocation()
{
    const auto& liveAndDieTensors            = m_livenessAnalysis.liveAndDieTensors();
    uint64_t    freeSpace                    = m_allocator.getMaxFreeContiguous();  // based only on WS size
    uint64_t    totalFreeSpace               = m_totalCapcity - m_allocator.GetCurrentlyUsed();
    uint32_t    currentLiveTensorInTimestamp = m_currentLiveTensorInTimestamp;
    bool        lowerBoundSet                = false;
    bool        finishedPickingCandidates    = false;
    for (int i = m_currentTimestamp; i < liveAndDieTensors.size() && !finishedPickingCandidates; i++)
    {
        const auto& liveTensors = liveAndDieTensors[i].m_live;
        for (int j = currentLiveTensorInTimestamp; j < liveTensors.size(); j++)
        {
            uint64_t tensorCap = getWriteSpaceForTensor(liveTensors[j]);
            if (tensorCap > freeSpace)
            {
                if (!lowerBoundSet)
                {
                    m_toAllocateLowerBound = m_toAllocate.size();
                    lowerBoundSet          = true;
                }
                if (tensorCap > totalFreeSpace)
                {
                    finishedPickingCandidates = true;
                    break;
                }
            }
            else
            {
                freeSpace -= tensorCap;
            }
            totalFreeSpace -= tensorCap;
            m_toAllocate.emplace_back(liveTensors[j]);
        }
        currentLiveTensorInTimestamp = 0;
    }
    if (!lowerBoundSet)
    {
        m_toAllocateLowerBound = m_toAllocate.size();
    }
    LOG_DEBUG(TENSORS_ALLOC,
              "buildPotentialCandidatesForAllocation lower bound {} upper bound {}",
              m_toAllocateLowerBound,
              m_toAllocate.size());
}

void RangeEpochAllocator::ProgressTensorIterators()
{
    const auto& liveAndDieTensors            = m_livenessAnalysis.liveAndDieTensors();
    auto        candidatesForAllocationCount = m_toAllocate.size();
    // Cope with cases where we start from the previous timestamp
    // and already accounted for tensors dying in the timestamp.
    if (m_currentTimestamp > 0 || m_currentLiveTensorInTimestamp > 0)
    {
        auto remainingInTimestamp =
            liveAndDieTensors[m_currentTimestamp].m_live.size() - m_currentLiveTensorInTimestamp;
        if (remainingInTimestamp > candidatesForAllocationCount)
        {
            m_currentLiveTensorInTimestamp += candidatesForAllocationCount;
            return;
        }
        m_currentTimestamp++;
        m_currentLiveTensorInTimestamp = 0;
        candidatesForAllocationCount -= remainingInTimestamp;
    }

    while (m_currentTimestamp < liveAndDieTensors.size())
    {
        const auto& liveAndDie = liveAndDieTensors[m_currentTimestamp];
        m_toBeFree.insert(m_toBeFree.end(), liveAndDie.m_die.begin(), liveAndDie.m_die.end());
        auto createdInTimestamp = liveAndDieTensors[m_currentTimestamp].m_live.size();
        if (createdInTimestamp > candidatesForAllocationCount)
        {
            m_currentLiveTensorInTimestamp = candidatesForAllocationCount;
            return;
        }
        m_currentTimestamp++;
        candidatesForAllocationCount -= createdInTimestamp;
    }
}

bool RangeEpochAllocator::allocateTensors()
{
    const auto& liveAndDieTensors = m_livenessAnalysis.liveAndDieTensors();
    unsigned    epochIndex        = 0;
    while (m_currentTimestamp < liveAndDieTensors.size())
    {
        LOG_DEBUG(TENSORS_ALLOC,
                  "Starting epoch {} start timestamp {} biggest hole {} free memory {}",
                  epochIndex,
                  m_currentTimestamp,
                  m_allocator.getMaxFreeContiguous(),
                  m_totalCapcity - m_allocator.GetCurrentlyUsed());
        buildPotentialCandidatesForAllocation();
        bool success = adjustCandidatesForAllocation();
        if (!success) return false;
        ProgressTensorIterators();
        handleEpochMemory();
        epochIndex++;
    }
    return true;
}

void RangeEpochAllocator::handleEpochMemory()
{
    allocateMemory();
    deallocateMemory();
}

void RangeEpochAllocator::deallocateMemory()
{
    for (const auto& die : m_toBeFree)
    {
        LOG_TRACE(TENSORS_ALLOC, "free {}", die->getName());
        if (!freeTensor(die, die->location() == TENSOR_IN_SRAM, die->isAliasedTensor(), m_allocator, false, nullptr))
        {
            HB_ASSERT(0, "Failed to free memory of tensor {}", die->getName());
        }
    }
    m_toBeFree.clear();
}

void RangeEpochAllocator::allocateMemory()
{
    sortCandidatesForAllocation(m_toAllocate);
    for (const auto& live : m_toAllocate)
    {
        Lifetime tensorLife = m_livenessAnalysis.getTensorLifeTime(live);
        LOG_TRACE(TENSORS_ALLOC, "Allocate {}", live->getName());
        if (!allocateTensor(live,
                            live->location() == TENSOR_IN_SRAM,
                            live->isAliasedTensor(),
                            false,  // allowFailure
                            m_allocator,
                            nullptr,
                            tensorLife))
        {
            HB_ASSERT(0, "Failed to allocate memory of tensor {}", live->getName());
        }
    }
    m_toAllocate.clear();
}