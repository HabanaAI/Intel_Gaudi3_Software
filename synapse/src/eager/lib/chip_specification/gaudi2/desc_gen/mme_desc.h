#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/mme_desc_base.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

// relative to <mme>/
#include "include/gaudi2/mme_descriptor_generator.h"
#include "include/mme_common/mme_common_enum.h"

// relative to <qman_fw>/engines-arc/include/
#include "gaudi2_arc_eng_packets.h"

// std includes
#include <array>
#include <memory>

class MMENode;

namespace eager_mode
{
class EagerGraph;
class EagerNode;

namespace gaudi2_spec_info
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

    std::shared_ptr<Gaudi2::MmeDescriptorGenerator>               m_descGenerator;
    std::shared_ptr<const std::vector<Gaudi2::MmeActivation>>     m_cachedActivations;
    // single interface regardless if we grabbed activations from cache or generated
    // them from scratch.
    const Gaudi2::ActivationVec*                                  m_activationsPtr = nullptr;
    std::array<mme_wd_ctxt_t, 3>                                  m_wdCtxs;
};

}  // namespace gaudi2_spec_info

}  // namespace eager_mode
