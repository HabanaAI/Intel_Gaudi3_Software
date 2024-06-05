#include "alloc_utils.h"

#include "habana_graph.h"
#include "memory_management/memory_allocator.h"
#include "non_persistent_section_util.h"


bool allocateTensor(pTensor                           tensor,
                    bool                              inSram,
                    bool                              allocateRealTensor,
                    bool                              allowFailure,
                    MemoryAllocator&                  alloc,
                    NonPersistentSectionAllocTracker* allocTracker,
                    Lifetime                          tensorLifetime)
{
    if (allocateRealTensor)
    {
        tensor = Tensor::getRealTensor(tensor);
    }

    const auto& si = tensor->getTensorAnnotation().nonPersistentSectionInfo;
    if (si.offsetFromBase.is_set() && allocTracker != nullptr)
    {
        size_t newOffset = -1;
        if (!allocTracker->markAsAlloc(tensor, newOffset))
        {
            LOG_TRACE(HEAP_ALLOC,
                      "{}: tensor \"{}\" allocation skipped; Instead setting tensor offset to 0x{:x}.",
                      HLLOG_FUNC,
                      tensor->getName(),
                      newOffset);
            if (inSram)
            {
                tensor->setSramOffset(newOffset);
            }
            else
            {
                tensor->setDramOffset(newOffset);
            }
            return true;
        }
    }

    TensorAnnotation &tensorAnn = tensor->getTensorAnnotation();
    uint64_t sizeInBytes        = getSizeToAllocate(tensor);
    uint64_t alignment          = tensorAnn.memory.alignment;
    uint64_t offset             = tensorAnn.memory.offset;

    Settable<deviceAddrOffset> addr = alloc.Allocate(sizeInBytes, alignment, tensorLifetime, offset, allowFailure);
    if (addr.is_set())
    {
        if (si.offsetFromBase.is_set() && allocTracker != nullptr)
        {
            allocTracker->setSectionBaseAddr(si.sectionId.value(), addr.value());
        }
        auto offset = addr.value() + (si.offsetFromBase.is_set() ? si.offsetFromBase.value() : 0);
        if (inSram)
        {
            tensor->setSramOffset(offset);
        }
        else
        {
            tensor->setDramOffset(offset);
        }
        return true;
    }

    if (!allowFailure)
    {
        LOG_ERR(GC, "{}: Failed to allocate memory in {} for tensor {}", HLLOG_FUNC, inSram ? "SRAM" : "DRAM", tensor->getName());
    }
    else
    {
        LOG_TRACE(GC, "{}: Failed to allocate memory in {} for tensor {}", HLLOG_FUNC, inSram ? "SRAM" : "DRAM", tensor->getName());
    }

    return false;
}

bool freeTensor(pTensor                  tensor,
                bool                     fromSram,
                bool                     freeRealTensor,
                MemoryAllocator&         alloc,
                bool                     rollback,
                NonPersistentSectionAllocTracker* allocTracker)
{
    if (freeRealTensor)
    {
        tensor = Tensor::getRealTensor(tensor);
    }

    const auto& si = tensor->getTensorAnnotation().nonPersistentSectionInfo;
    if (si.offsetFromBase.is_set() && allocTracker != nullptr)
    {
        if (!allocTracker->markAsFree(tensor, rollback))
        {
            LOG_TRACE(HEAP_ALLOC,
                      "{}: tensor \"{}\" (from non-persistent section {}) deallocation skipped.",
                      HLLOG_FUNC,
                      tensor->getName(),
                      si.sectionId.value());
            return true;
        }
        if (rollback)
        {
            // finishing rollback -> mark section as unallocated
            allocTracker->setSectionBaseAddr(si.sectionId.value(), (uint64_t)-1);
        }
    }

    deviceAddrOffset addr = fromSram ? tensor->getSramOffset() : tensor->getDramOffset();

    // if we free a tensor from a non-persistent section, the actual allocated address is minus
    // the offset within the non-persistent section.
    if (si.offsetFromBase.is_set())
    {
        addr -= si.offsetFromBase.value();
    }

    alloc.Free(addr);
    return true;
}

void setMemAllocationError(HabanaGraph& graph)
{
    graph.getGraphAnnotation().errors.memoryAllocationError = true;
}

uint64_t getSizeToAllocate(const pTensor& tensor)
{
    if (tensor->getTensorAnnotation().sizeToAllocate.is_set())
    {
        uint64_t sizeToAllocate = tensor->getTensorAnnotation().sizeToAllocate.value();
        HB_ASSERT(sizeToAllocate >= tensor->getTotalSizeInBytes(), "Size to allocate in bytes is smaller than tensor size");
        return sizeToAllocate;
    }
    return tensor->getTotalSizeInBytes();
}

std::tuple<uint64_t, uint64_t> getTensorMemoryParams(const TensorPtr& tensor)
{
    TensorAnnotation &tensorAnn = tensor->getTensorAnnotation();
    return std::make_tuple(tensorAnn.memory.alignment, tensorAnn.memory.offset);
}

template<> void setTensorAddress<true>(const TensorPtr& tensor, deviceAddrOffset address)
{
    tensor->setSramOffset(address);
}

template<> void setTensorAddress<false>(const TensorPtr& tensor, deviceAddrOffset address)
{
    tensor->setDramOffset(address);
}