#pragma once
#include "layered_brain.h"
#include "memory_usage_db.h"

namespace gc::layered_brain
{
// Preprocessors build the bundle memory usage database, used to decide on the placement of each slice.
struct BundleMemoryPreProcessor
{
    virtual MemoryUsageDB buildMemUsageDB() = 0;
    virtual ~BundleMemoryPreProcessor() {}
};

// Insulators make sure that slices are produced and consumed only by bundled nodes. External nodes interact with BPTs,
// which are separated from the slices by join and fork nodes (concat, split, tensor-view, etc.).
struct BundleMemoryInsulator
{
    // If a slice is a BPT, the insulator adds fork/join node to separate it. The new node will appear in order in
    // the returned BundleNodes.
    virtual BundleNodes getInsulatedBundle() = 0;
    virtual ~BundleMemoryInsulator() {}
};

// SRAM allocator manages SRAM available for the bundle w.r.t the slices that live and die as the bundle processing
// progresses.
struct BundleSRAMAllocator
{
    // Return true if allocation is successful
    virtual bool allocate(const TensorPtr& slice) = 0;
    virtual void free(const TensorPtr& slice)     = 0;
    virtual ~BundleSRAMAllocator() {}
};

// Bundle memory placer encapsulates the decision logic for each slice that is used by the provided step, regarding
// which memory these slices will be placed in.
struct BundleMemoryPlacer
{
    virtual void placeStepSlices(MemoryUsageDB::BundleStepEntry& stepEntry, BundleSRAMAllocator* allocator) = 0;
    virtual ~BundleMemoryPlacer() {}
};

// The directive executor performs the graph changes required to implement the decisions taken by the slicer - set
// slices in SRAM, plant spill/fill nodes, etc.
struct BundleMemoryDirectiveExecutor
{
    virtual bool executeDirectivesFor(TensorPtr slice) = 0;
    virtual ~BundleMemoryDirectiveExecutor() {}
};
}  // namespace gc::layered_brain