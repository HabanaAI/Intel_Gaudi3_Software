#pragma once

#include "gaudi2_types.h"
#include "mme_node.h"

#include "platform/common/graph_compiler/arc_dynamic_mme_pp_generator.h"
#include "platform/common/graph_compiler/arc_mme_field_infos.h"

namespace gaudi2
{
    struct MmeTypes
    {
        using MmeDescriptorGenerator = Gaudi2::MmeDescriptorGenerator;
        using MmeDesc = Gaudi2::Mme::Desc;
        using MmeTensorDesc = Gaudi2::Mme::MmeTensorDesc;
        using MmeCmd = Gaudi2::Mme::MmeCmd;
        using mme_wd_ctxt_t = ::mme_wd_ctxt_t;
    };

    class DynamicMMEPatchPointGenerator : public arc_platforms::DynamicMMEPatchPointGenerator<MmeTypes>
    {
        public:
            DynamicMMEPatchPointGenerator() = default;

            virtual uint32_t getNullDescriptorControlWord() override
            {
                return 0x83ff;
            }

        protected:

            virtual TensorTile getTensorTileFromEngine(const MmeNode&   node,
                                                       const TensorPtr& tensor,
                                                       unsigned engineIdx,
                                                       bool& outHaveTile) override;

            virtual void generateDynamicExecutionPatchPoint(const MmeNode& node,
                                                            DescriptorWrapper<MmeDesc>& descWrapper) override;
};

}  // namespace gaudi2
