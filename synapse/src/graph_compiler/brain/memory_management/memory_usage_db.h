#pragma once

#include "types.h"
#include <optional>

namespace gc::layered_brain
{
struct MemoryUsageDB
{
    // A step is an operation in the bundle execution schedule (an operation slice.)
    // A step index is the index in the bundle execution schedule of the step.
    // Currently entry.index and the index in the 'steps' vector are the same, but this doesn't have to stay true.
    struct BundleStepEntry
    {
        size_t  index;
        NodePtr sliceNode;
    };
    std::vector<BundleStepEntry> steps;

    struct SliceEntry
    {
        struct Properties
        {
            std::optional<int>      producingStep  = std::nullopt;
            std::unordered_set<int> consumingSteps = {};

            // No aliases aggregation - only bundled immediate consumers
            std::unordered_set<int> immediateConsumingSteps = {};

            // nullptr unless the slice is consumed by a join node
            NodePtr joinedBy = nullptr;  // Not counted in the consuming steps

            // nullptr unless the slice is produced by a fork node
            NodePtr forkedBy = nullptr;  // Not counted in the producing steps

            // true if the slice is read by a node from a different bundle or unbundled
            bool consumedExternally = false;

            // If this is an entry of a slice which is an alias to another intermediate slice, this field points to the
            // last slice in this slice's alias chain that is still an intermediate slice, otherwise - nullptr.
            TensorPtr realSlice = nullptr;

            // If the slice is not an alias to another intermediate slice (i.e. this is an entry of a 'real' slice),
            // this contains the list of direct and indirect aliases to it, otherwise, empty.
            std::list<TensorPtr> aliases;

        } properties;

        struct Directives
        {
            enum class Placement : uint8_t
            {
                UNSET,  // Not set by the manager
                HBM,
                SRAM,
            } placement = Placement::UNSET;
        } directives;
    };
    std::unordered_map<TensorPtr, SliceEntry> slices;

    void clear()
    {
        steps.clear();
        slices.clear();
    }
};

}  // namespace gc::layered_brain