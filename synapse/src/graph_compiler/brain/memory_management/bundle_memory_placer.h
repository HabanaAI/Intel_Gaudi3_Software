#pragma once

#include "memory_usage_db.h"
#include "bundle_memory_manager.h"

namespace gc::layered_brain
{
class POCBundleMemoryPlacer : public BundleMemoryPlacer
{
public:
    explicit POCBundleMemoryPlacer(MemoryUsageDB& db) : m_db(db) {}
    void placeStepSlices(MemoryUsageDB::BundleStepEntry& stepEntry, BundleSRAMAllocator* allocator) override;

private:
    MemoryUsageDB& m_db;
};
}  // namespace gc::layered_brain