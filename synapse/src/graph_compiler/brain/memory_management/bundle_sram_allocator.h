#pragma once

#include "bundle_memory_manager_interfaces.h"
#include "slicing_brain.h"
#include "brain_conf.h"

namespace gc::layered_brain
{
// Simple budget based sram allocator
class POCBundleSRAMAllocator : public BundleSRAMAllocator
{
public:
    POCBundleSRAMAllocator(uint64_t inputBudget)
    {
        m_budget = inputBudget * GCFG_FRAGMENTATION_COMPENSATION_FACTOR.value();
    }
    bool allocate(const TensorPtr& slice) override
    {
        uint64_t sizeToAllocate = slice->getDenseSizeInBytes();
        if (m_budget < sizeToAllocate) return false;

        m_budget -= sizeToAllocate;
        return true;
    }
    void free(const TensorPtr& slice) override
    {
        uint64_t allocatedSize = slice->getDenseSizeInBytes();
        m_budget += allocatedSize;
    }

private:
    uint64_t m_budget;
};
}  // namespace gc::layered_brain