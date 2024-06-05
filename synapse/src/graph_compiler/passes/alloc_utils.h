#pragma once

#include "types.h"
#include <tuple>
#include "memory_management/memory_allocator.h"

class HabanaGraph;
class NonPersistentSectionAllocTracker;

namespace SynapseInternals
{
class MemoryAllocator;
}  // namespace SynapseInternals

bool allocateTensor(pTensor                           tensor,
                    bool                              inSram,
                    bool                              allocateRealTensor,
                    bool                              allowFailure,
                    MemoryAllocator&                  alloc,
                    NonPersistentSectionAllocTracker* allocTracker,
                    Lifetime                          tensorLifetime = {});

bool freeTensor(pTensor                           tensor,
                bool                              fromSram,
                bool                              freeRealTensor,
                MemoryAllocator&                  alloc,
                bool                              rollback,
                NonPersistentSectionAllocTracker* allocTracker);

void setMemAllocationError(HabanaGraph& graph);

uint64_t getSizeToAllocate(const pTensor& tensor);

// return  aligment  offset
std::tuple<uint64_t, uint64_t> getTensorMemoryParams(const TensorPtr& tensor);

template<bool InSram> void setTensorAddress(const TensorPtr& tensor, deviceAddrOffset address);

template<> void setTensorAddress<true>(const TensorPtr& tensor, deviceAddrOffset address);
template<> void setTensorAddress<false>(const TensorPtr& tensor, deviceAddrOffset address);

template <bool InSram>
bool allocateTensor(const TensorPtr& tensor, MemoryAllocator& alloc, bool allowFailure)
{
    uint64_t sizeInBytes    = getSizeToAllocate(tensor);
    auto [alignment, offset] = getTensorMemoryParams(tensor);

    Settable<deviceAddrOffset> addr = alloc.Allocate(sizeInBytes, alignment, offset, allowFailure);
    if (addr.is_set())
    {
        setTensorAddress<InSram>(tensor, addr.value());
        return true;
    }
    return false;
}