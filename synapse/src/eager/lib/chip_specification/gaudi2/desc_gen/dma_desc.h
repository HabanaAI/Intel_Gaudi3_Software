#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "utils/sequential_iter_tracker.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/node_roi.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"

// std includes
#include <array>

class DMANode;

namespace eager_mode
{
class EagerGraph;
class EagerNode;

namespace gaudi2_spec_info
{
class DmaDescGenerator final : public DescGeneratorBase
{
public:
    DmaDescGenerator(EagerGraph& graph, const EagerNode& node);

    bool generateDesc() override;
    void generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant) override;
    deviceAddrOffset getTensorVirtualAddress(unsigned tensorIdx) const override;
    const Byte*      getDescRaw(unsigned descIdx) const override;
    const Byte*      getWorkDistributionContextRaw(unsigned descIdx) const override;

    DMANode&                  getNode() const { return *m_node.get<DMANode>(); }
    const std::list<NodeROI>& getPhysicalRois() const { return m_physicalRois; }
    bool                      isDmaNopDescNeeded() const override { return m_isNopDescNeeded; }

private:
    gaudi2::DescriptorGenerator::DmaDescriptorsList m_descs;
    std::list<NodeROI>                              m_physicalRois;
    std::array<edma_wd_ctxt_t, 3>                   m_wdCtxs;
    bool m_isNopDescNeeded = false;  // Determine the need of additional NOP descriptor for the idle engines
    using IterType         = gaudi2::DescriptorGenerator::DmaDescriptorsList::const_iterator;
    SequentialIterTracker<IterType> m_sequentialIterTracker;
};

}  // namespace gaudi2_spec_info

}  // namespace eager_mode