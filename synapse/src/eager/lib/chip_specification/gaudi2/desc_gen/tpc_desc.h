#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/tpc_desc_base.h"
#include "node_info/eager_node.h"

// synapse-internal gaudi2-specific includes (relative to src/)
#include "platform/gaudi2/graph_compiler/tpc_descriptor_generator.h"

class TPCNode;

namespace eager_mode
{
class EagerGraph;

namespace gaudi2_spec_info
{
class TpcDescGenerator final : public TpcDescGeneratorBase
{
public:
    TpcDescGenerator(EagerGraph& graph, const EagerNode& node);
    void generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant) override;
    deviceAddrOffset getTensorVirtualAddress(unsigned tensorIdx) const override;
    const Byte*      getDescRaw(unsigned descIdx) const override;
    const Byte*      getWorkDistributionContextRaw(unsigned descIdx) const override;

private:
    void calcTensorsNr();
    bool generateTpcDesc() override;

private:
    gaudi2::TpcDescriptorGenerator::DescriptorsVector m_descs;
};

}  // namespace gaudi2_spec_info

}  // namespace eager_mode
