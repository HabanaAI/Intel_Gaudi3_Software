#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/mme_desc_base.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

// synapse-internal gaudi3-specific includes (relative to src/)
#include "platform/gaudi3/graph_compiler/gaudi3_types.h"

// relative to <mme>/
#include "include/mme_common/mme_common_enum.h"

// std includes
#include <array>
#include <memory>

namespace eager_mode
{
class EagerGraph;
class EagerNode;

namespace gaudi3_spec_info
{
// Generate MME descriptor
class MmeDescGenerator final : public MmeDescGeneratorBase
{
public:
    MmeDescGenerator(EagerGraph& graph, const EagerNode& node);

    bool generateDesc() override;
    void generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant) override;
    deviceAddrOffset getTensorVirtualAddress(unsigned tensorIdx) const override;
    const Byte*      getDescRaw(unsigned descIdx) const override;
    const Byte*      getWorkDistributionContextRaw(unsigned descIdx) const override;
    void             copyDescToBlob(Byte*          out,
                                    unsigned       descIdx,
                                    StructSizeType offsetInDescriptor,
                                    BlobSizeType   sizeToCopy) const override;

private:
    void copyPerfDescInfoToBlob(Byte*          out,
                                unsigned       activationIdx,
                                StructSizeType offsetInDescriptor,
                                BlobSizeType   sizeToCopy) const;

    std::shared_ptr<gaudi3::MmeDescriptorGenerator>               m_descGenerator;
    std::shared_ptr<const std::vector<gaudi3::MmeActivation>>     m_cachedActivations;
    // single interface regardless if we grabbed activations from cache or generated
    // them from scratch.
    const gaudi3::ActivationVec*                                  m_activationsPtr = nullptr;
    std::array<mme_wd_ctxt_t, 3>                                  m_wdCtxs;
};

}  // namespace gaudi3_spec_info

}  // namespace eager_mode
