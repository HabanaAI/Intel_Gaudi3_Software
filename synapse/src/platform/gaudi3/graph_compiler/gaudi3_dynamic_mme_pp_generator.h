#pragma once

#include "gaudi3_types.h"
#include "mme_node.h"
#include "descriptor_generator.h"
#include "platform/common/graph_compiler/arc_dynamic_mme_pp_generator.h"

namespace gaudi3
{
    struct MmeTypes
    {
        using MmeDescriptorGenerator = gaudi3::MmeDescriptorGenerator;
        using MmeDesc = gaudi3::Mme::Desc;
        using MmeTensorDesc = gaudi3::Mme::MmeTensorDesc;
        using MmeCmd = gaudi3::Mme::MmeCmd;
        using mme_wd_ctxt_t = ::mme_wd_ctxt_t;
    };

    class DynamicMMEPatchPointGenerator : public arc_platforms::DynamicMMEPatchPointGenerator<MmeTypes>
    {
        public:
            DynamicMMEPatchPointGenerator() = default;

        protected:

            virtual uint32_t getNullDescriptorControlWord() override
            {
                return 0x51ff;
            }

            virtual TensorTile getTensorTileFromEngine(const MmeNode&   node,
                                                       const TensorPtr& tensor,
                                                       unsigned         engineIdx,
                                                       bool&            outHaveTile) override;

            virtual void generateDynamicExecutionPatchPoint(const MmeNode& node,
                                                            DescriptorWrapper<MmeDesc>& descWrapper) override;
    };


}  // namespace gaudi3
