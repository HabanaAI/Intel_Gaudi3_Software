#include "bundle_memory_placer.h"

using namespace gc::layered_brain;

// Base class for the placer and evacuator
class BundleSlicePlacementHandler
{
public:
    BundleSlicePlacementHandler(MemoryUsageDB& db, BundleSRAMAllocator* allocator) : m_db(db), m_allocator(allocator) {}

    MemoryUsageDB::SliceEntry&       entry(const TensorPtr& slice) { return m_db.slices.at(slice); }
    const MemoryUsageDB::SliceEntry& entry(const TensorPtr& slice) const { return m_db.slices.at(slice); }

    MemoryUsageDB::SliceEntry::Properties&       properties(const TensorPtr& slice) { return entry(slice).properties; }
    const MemoryUsageDB::SliceEntry::Properties& properties(const TensorPtr& slice) const
    {
        return entry(slice).properties;
    }

    MemoryUsageDB::SliceEntry::Directives&       directives(const TensorPtr& slice) { return entry(slice).directives; }
    const MemoryUsageDB::SliceEntry::Directives& directives(const TensorPtr& slice) const
    {
        return entry(slice).directives;
    }

protected:
    MemoryUsageDB&       m_db;
    BundleSRAMAllocator* m_allocator;

    // If the slice is an alias to another slice, get the end of the chain (within the boundaries of the join and
    // forks - i.e stop following the chain before the join output or the fork input)
    TensorPtr getSliceToHandle(const TensorPtr& origSlice) const
    {
        if (!origSlice->isAliasedTensor()) return origSlice;

        if (properties(origSlice).joinedBy || properties(origSlice).forkedBy)
        {
            return origSlice;
        }

        if (!properties(origSlice).producingStep || properties(origSlice).consumingSteps.empty())
        {
            // If this is already outside the join/fork insulation, no need to follow the alias chain
            // either.
            return origSlice;
        }

        validateAliasInternally(origSlice);

        // The slice is an alias to another slice. Follow the alias chain.
        return getSliceToHandle(origSlice->getAliasTensor());
    }

    // Make sure that if the slice is aliased, it is aliased to another slice and not a tensor that is external to the
    // bundle.
    void validateAliasInternally(const TensorPtr& origTensor) const
    {
        TensorPtr realTensor = origTensor->getAliasTensor();
        HB_ASSERT(m_db.slices.find(realTensor) != m_db.slices.end(),
                  "Expected slice to be an alias to another slice, but the real tensor of slice {} has no entry in the "
                  "slices DB. (real tensor: {})",
                  origTensor->getName(),
                  realTensor->getName());
    }
};

//
// Implementation classes for the POC placer
//

class BundleSlicePlacer : public BundleSlicePlacementHandler
{
public:
    BundleSlicePlacer(MemoryUsageDB& db, BundleSRAMAllocator* allocator) : BundleSlicePlacementHandler(db, allocator) {}

    // Set slice in SRAM if needed and possible. Only intermediate slices are relevant since all BPTs are assumed to be
    // in HBM always.
    void place(const TensorPtr& output)
    {
        if (!output || output->isShapeTensor()) return;
        TensorPtr handledOutput = getSliceToHandle(output);
        if (requiresPlacement(handledOutput))
        {
            tryToPlaceInSRAM(handledOutput);
        }
    }

private:
    TensorPtr getOutputToHandle(const TensorPtr& origOutput) { return getSliceToHandle(origOutput); }

    bool requiresPlacement(const TensorPtr& output)
    {
        return directives(output).placement == MemoryUsageDB::SliceEntry::Directives::Placement::UNSET &&
               isPureIntermediate(output);
    }

    bool isPureIntermediate(const TensorPtr& slice)
    {
        return properties(slice).producingStep &&              // Produced in the bundle
               !properties(slice).consumingSteps.empty() &&    // Consumed in the bundle
               properties(slice).consumedExternally == false;  // Not consumed outside the bundle
    }

    void tryToPlaceInSRAM(const TensorPtr& slice)
    {
        if (m_allocator->allocate(slice))
        {
            LOG_DEBUG(LB_CACHE_MNGR,
                      "Placing slice {} in SRAM ({:.2f}MB)",
                      slice->getName(),
                      bToMb(slice->getDenseSizeInBytes()));
            directives(slice).placement = MemoryUsageDB::SliceEntry::Directives::Placement::SRAM;
        }
        else
        {
            LOG_DEBUG(LB_CACHE_MNGR,
                      "Placing slice {} in HBM (allocating {:.2f}MB in SRAM failed)",
                      slice->getName(),
                      slice->getDenseSizeInBytes() / (1024.0 * 1024.0));
            directives(slice).placement = MemoryUsageDB::SliceEntry::Directives::Placement::HBM;
        }
    }
};

class BundleSliceEvacuator : public BundleSlicePlacementHandler
{
public:
    BundleSliceEvacuator(MemoryUsageDB& db, BundleSRAMAllocator* allocator) : BundleSlicePlacementHandler(db, allocator)
    {
    }

    // Free the slice SRAM if this step is the last consumer of it.
    void evacuate(const TensorPtr& input, size_t stepIdx)
    {
        if (!input) return;
        TensorPtr handledInput = getSliceToHandle(input);
        if (requiresEvacuation(handledInput) && isLastConsumer(handledInput, stepIdx))
        {
            freeInput(handledInput, m_allocator);
        }
    }

private:
    bool requiresEvacuation(const TensorPtr& input) const
    {
        return directives(input).placement == MemoryUsageDB::SliceEntry::Directives::Placement::SRAM;
    }

    bool isLastConsumer(const TensorPtr& input, size_t stepIdx)
    {
        auto& consumingSteps = properties(input).consumingSteps;
        return stepIdx == *std::max_element(consumingSteps.begin(), consumingSteps.end());
    }

    void freeInput(const TensorPtr& input, BundleSRAMAllocator* allocator) { allocator->free(input); }
};

//
// "main"
//
void POCBundleMemoryPlacer::placeStepSlices(MemoryUsageDB::BundleStepEntry& stepEntry, BundleSRAMAllocator* allocator)
{
    BundleSlicePlacer    placer {m_db, allocator};
    BundleSliceEvacuator evacuator {m_db, allocator};

    for (const TensorPtr& output : stepEntry.sliceNode->getOutputs())
    {
        placer.place(output);
    }
    for (const TensorPtr& input : stepEntry.sliceNode->getInputs())
    {
        evacuator.evacuate(input, stepEntry.index);
    }
}
