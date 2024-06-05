#pragma once

#include "bundle_memory_manager_interfaces.h"

#include "habana_graph.h"

namespace gc::layered_brain
{
// Common bundle insulation data for all insulator components to work on.
struct BundleInsulation
{
    const BundleNodes& origBundle;
    BundleNodes        updatedBundle;  // bundle nodes with additional joins if required
    TensorSet          slices;
    BundleIndex        bundleIdx;

    explicit BundleInsulation(const BundleNodes& originalBundle) : origBundle(originalBundle)
    {
        HB_ASSERT(!originalBundle.empty(), "Unexpected empty bundle to insulate");
        bundleIdx = *getBundleIndex(originalBundle.front());
        updatedBundle.reserve(originalBundle.size());
        for (const NodePtr& n : originalBundle)
        {
            for (TensorPtr tensor : n->getOperands())
            {
                if (!tensor) continue;
                slices.insert(tensor);
            }
        }
    }
};

struct POCBundleMemorySliceInsulator
{
    virtual ~POCBundleMemorySliceInsulator() = default;

    virtual void insulate(NodePtr node, TensorPtr slice, BundleInsulation* insulation) = 0;
};

// Given a bundle with slices that are consumed both internally and externally like:
// [in]->n0->[t0]->n1->[out]
//             |
//             +-->extCons
// where n0 and n1 are bundled,
// This method adds a bundled insulation node between the slices and the BPT:
// [in]->n0->[t0_tile]->n1->[out]
//             |
//             +-->insulation->[t0]->extCons
// n0, n1 and insulation are bundled.
// Returned bundle order is n0, insulation, n1.
// The insulation can be a 'fork' or a 'join' node, depending on the scenario.
class POCBundleMemoryInsulator : public BundleMemoryInsulator
{
public:
    // inputs and outputs insulator can be supplied by the user to ease testing of one or both of them.
    // Null slice insulator will result in using the default insulator for the slices it's in charge of (inputs or
    // outputs).
    POCBundleMemoryInsulator(HabanaGraph&                   graph,
                             const BundleNodes&             bundle,
                             POCBundleMemorySliceInsulator* inputsInsulator  = nullptr,
                             POCBundleMemorySliceInsulator* outputsInsulator = nullptr);
    BundleNodes getInsulatedBundle() override;

private:
    HabanaGraph&                   m_graph;
    BundleInsulation               m_insulation;
    POCBundleMemorySliceInsulator* m_inputsInsulator;
    POCBundleMemorySliceInsulator* m_outputsInsulator;

    std::unique_ptr<POCBundleMemorySliceInsulator> m_defaultInputInsulator;
    std::unique_ptr<POCBundleMemorySliceInsulator> m_defaultOutputInsulator;

    void insulate(const NodePtr& node);
    void insulateOutputs(const NodePtr& node);
    void insulateInputs(const NodePtr& node);
};

}  // namespace gc::layered_brain