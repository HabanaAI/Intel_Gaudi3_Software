#include "mme/mme_brain_ifc.h"
#include "mme/mme_desc_gen_utils.h"
#include "gaudi2_dynamic_mme_pp_generator.h"
#include "recipe_metadata.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "platform/gaudi/graph_compiler/smf/smf.h"
#include "platform/common/graph_compiler/arc_mme_field_infos.h"
#include "platform/common/graph_compiler/arc_dynamic_mme_pp_generator.inl"

namespace gaudi2
{
    using arc_platforms::DynamicMmeFieldInfo;
    using arc_platforms::MmeCommitFieldInfo;
    using arc_platforms::MmeSyncFieldInfo;

    void DynamicMMEPatchPointGenerator::generateDynamicExecutionPatchPoint(const MmeNode& node, DescriptorWrapper<MmeDesc>& descWrapper)
    {
        BasicFieldsContainerInfo& bfciCtx = descWrapper.getBasicFieldsContainerInfoForCtx();
        auto                      origin  = const_cast<MmeNode&>(node).shared_from_this();

        // null desc pp
        uint32_t                       fieldIndexOffset = offsetof(mme_wd_ctxt_t, mme_commit_reg) / sizeof(uint32_t);
        DynamicShapeFieldInfoSharedPtr fieldInfo =
            std::make_shared<MmeCommitFieldInfo>(fieldIndexOffset, origin, bfciCtx.getRoi());

        MmeCmd cmd;
        cmd.dw = getNullDescriptorControlWord();

        std::vector<uint8_t> serializedMetadata(sizeof(cmd));
        memcpy(serializedMetadata.data(), &cmd, sizeof(cmd));
        fieldInfo->setMetadata(serializedMetadata);

        bfciCtx.add({fieldIndexOffset, fieldInfo});

        // sync object pp
        BasicFieldsContainerInfo& bfci = descWrapper.getBasicFieldsContainerInfo();
        NodeROI*                  roi  = bfci.getRoi();

        fieldIndexOffset = offsetof(MmeDesc, syncObject.so0Val) / sizeof(uint32_t);
        fieldInfo        = std::make_shared<MmeSyncFieldInfo>(fieldIndexOffset, origin, roi);

        serializedMetadata.resize(sizeof(roi->numSignals));
        memcpy(serializedMetadata.data(), &roi->numSignals, sizeof(roi->numSignals));
        fieldInfo->setMetadata(serializedMetadata);

        bfci.add({fieldIndexOffset, fieldInfo});
    }

    DynamicMMEPatchPointGenerator::TensorTile DynamicMMEPatchPointGenerator::getTensorTileFromEngine(const MmeNode&                node,
                                                                                                     const TensorPtr&              tensor,
                                                                                                     unsigned                      engineIdx,
                                                                                                     bool&                         outHaveTile)
    {
        // Gaudi2 nodes work on the old-fashioned way, the entire tensor is the tile
        outHaveTile = false;

        static SizeArray zeros{0};
        TensorTile ret(tensor->getDim(), tensor->getShape().getMaxSizes(), zeros);
        return ret;
    }

}

namespace arc_platforms
{

// explicit instantiation for gaudi2
template class DynamicMMEPatchPointGenerator<gaudi2::MmeTypes>;

};


